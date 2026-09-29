# M4 路线 A 原型验证报告：模板/标签压缩 + 归一化卷积/FFT

> 依据：`Hex_model_recon.md` §4「路线 A 设计」、§5 M4。数据：`results/m4_routeA_prototype.json`；
> 命令：`python repro/m4_routeA_prototype.py`（约 1.4 min）。风险分类：局部（复现脚本 + 数据），未改产品代码。
> 目的：在动手改引擎前，先证明「模板+标签表示」与「归一化卷积/FFT」**与本仓库现有 CSR 迁移算子等价**，
> 并量化存储与每步成本。

## 1. 方法
在 `HexGrid(rows,cols,wrap=False)` + `build_gaussian_kernel(size=k,sigma=1.5)` 上：
1. 用 natal `fold_migration_csr`（`include_center=False, adjust_on_edge=False`）生成**参考 CSR**；
2. 独立地从 topology+kernel 构造**模板+标签**模型（按“有效邻居集合签名”分组 → 模板表 + `type-id`），
   比较 `(src,dest,weight)` 三元组；
3. 归一化卷积：`K'`=去中心核、`m`=域内指示、`Z=K'⊛m`（零填充线性卷积）；`g=rate·f/Z`，
   `f_new=f·(1−rate)+K'⊛g`（直接 stencil 与 FFT 两版），与 CSR 逐步对拍。

## 2. 等价性（与 CSR 完全一致 / 机器精度）

| 场地 | k | #模板 | 三元组完全相等 | 权重最大差 | 直接卷积 vs CSR | FFT vs CSR |
|---|---|---|---|---|---|---|
| 30×30 | 3 | 9 | **True** | 5.6e-17 | 2.2e-16 | 2.2e-16 |
| 30×30 | 5 | 25 | **True** | 8.3e-17 | 2.2e-16 | 2.2e-16 |
| 40×40 | 11 | 121 | **True** | 5.6e-17 | 3.3e-16 | 3.9e-16 |
| 60×60 | 21 | 441 | **True** | 1.3e-16 | 4.4e-16 | 1.0e-15 |

- **模板+标签与 CSR 的 `(src,dest,weight)` 三元组逐条相同**（模板权重独立计算，最大差 ~1e-16，仅浮点结合顺序）。
- **归一化卷积（Z 校正）与 CSR 逐格差 ~1e-16**，即边界「丢界 + 邻域重归一化」被 `Z=K'⊛m` **精确复现**。

## 3. 存储（模板/标签 vs CSR）

| 场地 | k | CSR MB | 模板 MB | 压缩比 |
|---|---|---|---|---|
| 30×30 | 3 | 0.111 | 0.0075 | 14.9× |
| 30×30 | 5 | 0.310 | 0.0120 | 25.8× |
| 40×40 | 11 | 2.553 | 0.137 | 18.7× |
| 60×60 | 21 | 20.15 | 1.69 | 11.9× |

模板存储 `O(#templates·k² + n)`，CSR 为 `O(n·k²)`；`#templates = O(k²)`，故大域下压缩比 ~`n/k²`
（300²×51² 预计 ~35×，即 3.4 GB → ~0.1 GB 量级）。**该模板/标签表正是“可持久化缓存”的对象。**

## 4. 每步迁移成本（300×300，一次前后向）

| k | CSR（numpy 散播代理）ms | 直接 stencil ms | FFT ms | FFT/直接 |
|---|---|---|---|---|
| 3 | 1.38 | **0.40** | 0.84 | 2.11 |
| 5 | 5.91 | **0.73** | 1.05 | 1.45 |
| 11 | 48.07 | 3.22 | **0.94** | 0.29 |
| 21 | 134.82 | 12.84 | **0.94** | 0.07 |
| 51 | 789.43 | 26.92 | **3.29** | 0.12 |

- **交叉点**：`k≲5` 直接 stencil 最快；`k≳11` FFT 胜且近乎与 `k` 无关（`O(n log n)`）。
- k=51 时 FFT ≈ 3.3 ms vs 直接 26.9 ms vs CSR 代理 789 ms。
- 注：CSR 列为 numpy `bincount` 散播代理，natal 的 Rust 实现更快，但**随 nnz 的标度一致**（M3：natal
  CPU 300²×21² 全类 tick ≈ 11 s）。

## 5. 结论

1. **表示层可行**：模板/标签与 CSR 完全等价（三元组相同），存储小 12–26×（大域更高），且天然可缓存持久化；
   地形/掩膜不变即可复用，σ/环境变只需原地更新权重（`Z` 与类型映射仅依赖形状）。
2. **计算层可行**：归一化卷积（`Z=K'⊛m`）与 CSR 机器精度一致；每步成本从 `O(n·k²)`（CSR）降到直接 stencil
   的 `O(n·k²)`（更小常数）或大核 FFT 的 `O(n log n)`。
3. **与论文/ M2 自洽**：论文用直接 `imfilter`（stencil + `'replicate'`）；FFT 是其上的额外优化；
   M2 的卷积路径已与 MATLAB 参考达 4.8e-12（端到端佐证）。
4. **必须默认关闭**：FFT/stencil 改变求和顺序，不能与 CPU CSR golden 逐位一致；`phase0`/CSR 保持 golden。
5. **实现决策（2026-09-29）**：当前聚焦**论文的大核**情形 → **FFT 优先实现**（CPU FFT / GPU cuFFT，按类 batched，GPU 为首选落点）；小核（`k≲5`）直接 stencil 更优，但**暂不实现**（本原型中的直接 stencil 仅用于验证等价性）。

## 6. 残余与边界（原型未覆盖）
- 本原型用**朴素签名构造**（O(n·k²)），未演示“构建时间”从 `O(n·k²)` 降到 `O(#templates·k²)`；产品化需
  解析/向量化的边界层模板构造（签名为到四边距离的查找表）。
- CSR 计时为 numpy 代理；未与 natal Rust 迁移直接对拍每步时间。
- 未测：多类（性别/年龄/ztype）批处理、GPU（cuFFT）、overlap-add 大域、随机语义下的行宽问题。
- 不规则地形（land mask）的签名分组未实测。

## 7. 复现
```bash
cd hexagon_spatial_test && ../.venv/bin/python repro/m4_routeA_prototype.py
# -> results/m4_routeA_prototype.json
```

## 8. 独立审查状态
以上为**主 agent 自测**（局部脚本）。如需，可在 `EVALUATE.md` 追加交接请 evaluator 复核本原型。
