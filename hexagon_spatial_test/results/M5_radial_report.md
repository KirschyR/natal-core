# M5 报告：径向（圆形）释放——natal 年龄结构模型 vs MATLAB 参考

> 依据：`Hex_model_recon.md` §5（M5）、prompt 「五、M5」；口径经用户确认（年龄结构蚊虫模型；500×500 主；
> 双阈值 0.5/0.9；先 CSR 小规模验证）。
> 风险分类：**局部（复现脚本 + 数据）**，未改任何 natal-core 产品代码。
> 产物：`repro/m5_radial_reference.py`、`repro/gen_golden_m5_radial.py`、`repro/verify_m5.py`、
> `golden/m5_radial_homing_arena500_shard*.json`、`results/m5_verify.json`。

## 1. 口径冻结（用户确认，2026-09-30）

| 项 | 取值 |
|---|---|
| 模型 | **蚊虫年龄结构 hex 模型**（2 幼 + 5 雄成虫 + 8 雌成虫，逐周） |
| 参考 | `Hex-model-main/radial_release_optimizer/main.m`（逐字复制到 `repro/ref/radial/`，仅加可选 `m,n` 参数） |
| 场地 | **500×500**（论文正文；代码硬编码 800×800 仅作参考） |
| 释放 | 中心圆释放 `(t, h, r)`：半径 `r` 内每 hex、持续 `t` 周、每周向 `male0`/`female0` 各加 `h/2` |
| 驱动 | homing **理想参数**（`drive_params.mat`：`dc=1, gcr=0, herr=0, ddfitness=1`；论文 Table 1「最优参数」） |
| 迁移核 | `mig.mat`：59×59 六边形高斯 `σ=10.437`（avd=13.08）、半径 25 盘截断、全核归一化 |
| 指标 | 覆盖面积（携带率 ≥ 0.9，及交叉口径 ≥ 0.5）、release_sum、efficiency = area / released |
| 判定时点 | 第 104 周（2 年）与第 261 周（5 年，`collect_week`） |

## 2. 方法

论文的径向释放评估用**年龄结构蚊虫模型**（recon §2.6、仿照 `radial_release_optimizer/main.m`）。
natal 的 `age_structured` 引擎（reproduction → survival → aging → migration）与参考 MATLAB 的
**转移矩阵 + `imfilter` 循环**在实现结构上不同（见 §6），因此 M5 采用与 M2 相同的方法学：
**忠实翻译参考模型，空间原语取自 natal-core**（`natal.frontend.spatial.topology.build_gaussian_kernel`）。

- **迁移核**：`build_gaussian_kernel("hex", size=59, mean_dispersal=avd)` 提供六边形高斯；natal 度量
  `dr²+dc²+dr·dc`（`COS_OPPOSITE_ANGLE=-0.5`）与参考 `dx²+dy²−dx·dy` 差一个轴翻折，故对核做
  `kernel[::-1, :]`，再施加半径 25 盘掩膜并重归一化 → 与 `mig.mat` **逐点差 ≤ 4.4e-19**。
- **迁移执行**：对每个成虫（基因型 × 年龄）平面做 **replicate 边界**卷积。参考用 MATLAB
  `imfilter(...,'replicate','same')`；翻译用「edge padding + FFT valid 卷积」，与
  `scipy.ndimage.convolve(mode="nearest")` 逐点差 ≤ 3e-15（500²×65 平面约 0.33 s/周）。
- **生命周期**：忠实翻译密度依赖（`s1 = λ/((λ-1)r+1)·2/(sf·eggs)`，`r=(j0+5j1)/(ages1+5ages2)`）、
  `renew_function` 后代算子、年龄转移矩阵、逐周释放、NaN→0、carrier < 1e-5 早停。
- **golden**：`repro/ref/radial/`（参考逐字副本，唯一改动为 arena 可参数化）在 MATLAB R2026a 上
  跑出 `output/<t>,<h>,<r>.csv`（12 个配置，500×500）。

## 3. 结果：翻译 vs MATLAB golden

12/12 配置的 104/261 周 **覆盖直径、覆盖面积、release_sum 与 MATLAB CSV 逐位相同**
（`max_abs_diff = 0.0`；命令 `python repro/verify_m5.py --jobs 8`，EXIT=0）。摘要（500×500 = 250000 hexes）：

| t | h | r | area(≥0.9)@2y | area(≥0.5)@2y | released@2y | area(≥0.9)@5y | eff(≥0.9)@2y |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 87 953 | 156 521 | 1 | 250 000 | 87 953 |
| 1 | 10 | 0 | 166 434 | 200 738 | 10 | 250 000 | 16 643 |
| 1 | 50 | 0 | 194 134 | 215 632 | 50 | 250 000 | 3 883 |
| 1 | 100 | 0 | 201 160 | 220 638 | 100 | 250 000 | 2 012 |
| 1 | 10 | 20 | 231 049 | 242 021 | 14 570 | 250 000 | 15.9 |
| 1 | 10 | 40 | 239 527 | 247 596 | 58 130 | 250 000 | 4.1 |
| 1 | 100 | 40 | 244 048 | 249 970 | 581 300 | 250 000 | 0.4 |
| 4 | 10 | 0 | 188 250 | 211 444 | 40 | 250 000 | 4 706 |
| 4 | 10 | 20 | 233 406 | 243 604 | 58 280 | 250 000 | 4.0 |
| 4 | 50 | 20 | 237 080 | 246 052 | 291 400 | 250 000 | 0.8 |
| 4 | 100 | 40 | 245 379 | 250 000 | 2 325 200 | 250 000 | 0.1 |

- 完整表见 `results/m5_verify.json`（每个配置含 0.5/0.9 两阈值、两周时刻）。
- 阈值效应：≥0.5 覆盖恒 ≥ ≥0.9 覆盖；小释放（`h=1..10, r=0`）2 年即可覆盖 35%–67% 场地，
  5 年全覆盖；增大 `h`/`r` 加快但效率骤降。

## 4. 与论文 Table 1/2 的定性对照（M5 验收口径）

- 论文：「归巢驱动效率远高于其他系统；**释放量越小反而效率越高**」（Table 1 正文，p.11–12）。
  本表：`h=1` 的 eff≈8.8e4，`h=100` 仅≈2.0e3；`r` 增大使 released 剧增而面积增益饱和 →
  **效率随释放量单调下降**，与论文一致。
- 论文 homing modification（Table 1）2 年即达 160 直径覆盖率、效率 2.34e5 量级；本模型同一释放
  （极小的 `h`）2 年覆盖直径已达 312（`h=1`）并在 5 年铺满 500×500，量级与排序一致。
- 论文强调「释放半径小的圆释放即可扩散」，本表 `r=0`（仅中心 1 hex）的小释放即可成波，一致。

## 5. 复现命令

```bash
cd hexagon_spatial_test
# 1) 生成 MATLAB golden（500×500，12 配置；分片并行）
.venv/bin/python repro/gen_golden_m5_radial.py --arena 500 --shard 0 --nshards 4   # 其余 shard 同理
# 2) 翻译校验（精确匹配，EXIT=0）
.venv/bin/python repro/verify_m5.py --jobs 8
```

## 6. 残余与边界

1. **自写翻译 ≠ natal 引擎生命周期**：本报告验证的是「natal 空间原语 + 忠实翻译的年龄结构生命周期」。
   natal `age_structured` 引擎在**迁移边界**（drop+邻域重归一化 vs 参考 `replicate`）、**繁殖/密度依赖的
   时序**（natal 在同一 tick 对 newborn 施加密度依赖；参考对上一周 j0→j1 施加）、**核截断**（引擎默认不做
   `d≤25` 盘截断）上语义不同，故未用于本轮的定量 golden 对照。
2. **场地口径**：论文正文 500×500，代码 `radial_release_optimizer/main.m` 硬编码 800×800；本轮按用户
   确认取 500×500。**800×800 未跑**。
3. **预计算 golden 不可直接用于本模型**：`parameter_sensitive/radial/<drive>/*.mat` 是**通用 hex 模型**
   （非年龄结构）在 500×500、阈值 0.5 的 `output`（横轴为**释放密度 h 0–2**、纵轴**释放半径 r 0–30**，
   **avd 固定 2.5**）——与年龄结构模型不同源，故未作 golden（prompt 中「avd 0–2」为笔误）。
4. **线性释放（Table 2）未做**：`linear_release_optimizer` 属同族但独立模块，留待 M5 扩展。
5. **驱动**：仅 homing 理想参数；TARE/TADE/Wolbachia/2-locus 未跑。
6. **route A FFT/GPU 未用于 M5**：本翻译内部用 FFT 卷积加速迁移（非 natal `migration_execution="fft"`），
   未在设备上运行。

## 7. 独立审查状态

以上为**主 agent 自测**（12/12 对 MATLAB golden 逐位相同）。已在 `EVALUATE.md` 追加 §26 交接，等待 evaluator 回执。
