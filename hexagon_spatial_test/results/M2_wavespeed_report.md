# M2 报告：均匀 hex 波速/波形（natal 空间原语 vs MATLAB 参考）

> 依据：`Hex_model_recon.md` §5（M2）「M2 前置口径」、§6；`hexagon_spatial_test/EVALUATE.md` §1/§2。
> 本阶段**未改动任何 natal-core 产品代码**；验证口径：针对性数值对照（natal 原语实现 vs MATLAB 参考）。
> 风险分类：局部（复现脚本 + 数据）。

## 1. 口径冻结（采纳 M0 evaluator §2.8）

| 项 | 主口径（paper） | 交叉口径（code） |
|---|---|---|
| 场地 | flat 300×300 | flat 200×200；junction L=600 |
| 检查点 | 50% / 60% | 40% / 70%（flat 发布 launcher） |
| junction | L=300，cp `L/(2√3)+0.45L` / `+0.65L` | L=600，同上 |

- avd ∈ {0.5, 1.0, 2.0}（对齐 Fig 3 的 0–2 段）。
- 驱动：homing（gcr=0, dc=1, herr=0）；左 20% 释放；λ=5；核 51×51。
- 单位：波速 = cells(或 domain units)/generation。PDE 参考 `retlist` 为 domain-units/time
  （reaction 每单位时间施加一次），据此 1 时间单位 ↔ 1 代，可直接对照。
- 方向命名：本报告沿用发布目录的 `flat`（沿列轴）/`junction`（沿行轴）；√3/2 归属见 §3.3。

## 2. 方法

论文波速实验用的「通用 hex 模型」是**离散代、连续计数的基因型频率更新**
（`data += renew(...)` 后逐基因型 `imfilter(...,'replicate')`），**不是 natal 的分阶段年龄结构生命周期**
（recon §2.6）。因此 M2 的决断为「**自写参考实现，空间原语取自 natal-core**」：

- `natal.frontend.spatial.topology.build_gaussian_kernel("hex", size=51, mean_dispersal=avd)`；
- 梯度/边界用 scipy 直接卷积 `mode='nearest'`（=MATLAB `imfilter` `'replicate'`；直接卷积保持非负，
  FFT 卷积会引入 ~1e-15 负振铃并被 logistic 项放大，故不用）。
- 坐标映射（M0 evaluator 已证）：natal `(dr,dc)=(Δrow,−Δcol)`，故 MATLAB `(row,col)` 框架下取
  natal 核的**列镜像** `K[:, ::-1]`。flat 波沿列轴对该镜像不变；junction 波沿行轴则敏感。

## 3. 结果

### 3.1 核（natal vs MATLAB `get_mig_matrix25`）
natal 核与 MATLAB 参考核的**列镜像逐点相等**：`max|K_natal − K_matlab[:, ::-1]| ≤ 1.1e-16`（avd 0.5/1/2 全部成立）。
另外，`HexGrid` 的 pointy-top 笛卡尔嵌入与核所用斜交度量一致：`max|dist²_embed − dist²_metric| = 1.14e-13`。

### 3.2 波速（忠实性：natal 原语 vs MATLAB golden）

> 修订（响应 evaluator §4）：首版 Python 翻译误把 MATLAB 的 **1-based** 下标当作 0-based
> （初值条件、检查点/`mid` 取样），且 `round` 用了 banker's rounding，造成 1e-4–2e-3 的残余。
> 已按 `main.m` 的 1-based 语义忠实翻译（初值 `(j+1) >= 2(i+1) − 0.4L`；取样 `cp−1`、`mid−1`；
> `round` 改为 MATLAB 的 half-away-from-zero）。修正后**逐例匹配到机器精度**（下表相对误差以 `verify_m2.py` 计）。

| 口径 | avd | MATLAB / natal | |rel| |
|---|---|---|---|
| flat_paper | 0.5 / 1.0 / 2.0 | 0.490371 / 1.073656 / 2.125093 | 2.5e-13 / 1.8e-13 / 4.8e-12 |
| flat_code | 0.5 / 1.0 / 2.0 | 0.488931 / 1.068142 / 2.105842 | 3.4e-13 / 1.4e-13 / 3.4e-12 |
| junction_paper | 0.5 / 1.0 / 2.0 | 0.427633 / 0.931870 / 1.845530 | 1.4e-13 / 2.3e-13 / 3.3e-12 |
| junction_code | 0.5 / 1.0 / 2.0 | 0.428500 / 0.935455 / 1.863452 | 1.6e-13 / 1.2e-13 / 3.8e-12 |

**12/12 与 golden 在机器精度内一致**（`verify_m2.py` EXIT=0，最大 |rel| = 4.8e-12），
远优于 5% 验收带（`compare_m2.py` 12/12，EXIT=0）。

### 3.3 方向交叉核对（flat×√3/2 vs junction）

| 口径 | avd=0.5 | avd=1.0 | avd=2.0 |
|---|---|---|---|
| paper | 0.9931 | 0.9978 | 0.9972 |
| code | 0.9882 | 0.9889 | 0.9787 |

paper 规模下两方向在 <1% 内一致（M0 小规模时偏差达 7%，已随规模收敛）。这支持发布
`wavespeed_analyse.m` 对 flat 乘 √3/2 的换算，也说明 ZIP 目录名「junction和flatside是反的」的
歧义主要影响**标签**而非数值。

### 3.4 核有效扩散（M0 evaluator §2.8-d 的量化）

| 标称 avd | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| 有效核均值 `Σ(w·d)/Σw` | 0.2065 | 0.950 | 1.988 |

低 avd 的离散核有效扩散显著低于标称值（avd=0.5 时约 41%），这是 Fig 3 低扩散段 hex 与 PDE
差异的主要来源，而非收敛问题；M2 已在对照中显式记录该量。

### 3.5 波形

- natal 复现的波形已导出：`results/m2_hex_waveshape.npz`（12 例的波前剖面，已按忠实索引重生成）。
- 归一化波前宽度（0.1→0.9）/域长：flat_paper 0.0067 / 0.0167 / 0.0301（avd 0.5/1/2）；flat_code
  0.0151 / 0.0251 / 0.0503；junction_paper 0.0042 / 0.0085 / 0.0169；junction_code 0.0021 / 0.0042 / 0.0095，
  随扩散单调变宽。
- 与 PDE 参考（`golden/pde_waveshape_reference.npz`）在 avd=0.5 的**绝对**波前宽度同量级：
  hex flat_paper ≈ 2.0 cells vs PDE homing ≈ 2.5 domain units。
- 说明：发布仓库**没有** hex 波形 golden（其 `launcher_waveshape` 依赖 `main` 未返回的 `ret`），
  故 hex 波形只有本实现的输出；定量叠加受「连续 vs 离散代、单位/域长不同」限制，留待 M8 图形对照。

## 4. 验证结论与命令

```bash
# MATLAB 参考 golden（约 2 分钟）
/opt/matlab/bin/matlab -batch "addpath('.../hexagon_spatial_test/repro'); gen_golden_m2_wavespeed"
# natal 原语复现（约 6 分钟）
cd hexagon_spatial_test && ../.venv/bin/python repro/natal_hex_wavespeed.py
# 验收带对照（EXIT=0）
cd hexagon_spatial_test && ../.venv/bin/python repro/compare_m2.py
# 忠实性回归（evaluator §4.6；EXIT=0）
cd hexagon_spatial_test && ../.venv/bin/python repro/verify_m2.py
```

- 波速 12/12 与 golden 机器精度内一致（§3.2）；核列镜像 1e-16（§3.1）；方向 <1%（paper，§3.3）。
- 未改产品代码；`phase0` 不涉及。

## 5. 残余风险 / 差异

1. **索引约定（已修复）**：首版残留 ~0.01–0.2% 源于 1-based/0-based 翻译缺陷（非插值敏感性）；
   修正后 12/12 达机器精度（§3.2）。仍存的仅浮点噪声（≤4.8e-12）。
2. **自写实现 vs 引擎**：M2 验证的是 natal **空间原语**（HexGrid + 高斯核）而非引擎生命周期；
   将通用 hex 模型接入引擎属 M4 决断范围。
3. **波形**：hex 无独立 MATLAB golden；PDE 参考仅 avd=0.5/10。定性同量级，定量留 M8。
4. **低 avd 离散化**：需以有效扩散为横轴重做 Fig 3 读数（M8）。

## 6. 独立审查状态

以上均为**主 agent 自测**。已在 `EVALUATE.md` 提交 §3 交接，等待 evaluator §4；获 `APPROVED` 前不宣称 M2 完成。
