# M3 报告（CPU 侧）：宽核/大规模迁移 CSR 折叠实测

> 依据：`Hex_model_recon.md` §5（M3）、§4（硬瓶颈表）；`EVALUATE.md` §6/§7。
> 范围：**仅 CPU 侧**（GPU 按用户决定待确认后另做）。风险分类：局部（复现脚本 + 数据），未改产品代码。
> 数据：`results/m3_scale_data.json`；命令：`python repro/m3_scale_probe.py`（约 6.5 min，峰值 RSS 6.72 GB）。

## 1. 方法

- 场地 `rows=cols ∈ {30,60,120,300}`；核尺寸 `k ∈ {3,5,11,21,51}`；`sigma=1.5`（最坏情形，全部
  `k²` 项非零）。
- 对每个组合测：核支持数 `support`、CSR 折叠耗时 `fold_s`、`nnz`、CSR 字节数、总体
  `build_s`（`SpatialPopulation.builder(...).build()`）、**deterministic** CPU `tick_s`（`run(3)` 平均，
  `stochastic=False`）。
- **口径一致**：`fold()` 与 builder 均取 `kernel_include_center=False`（builder 默认；evaluator §7.4-F1 修复）。
- CSR 字节 = `indptr(int64) + dest_idx(int64) + weights(float64)` = `16·nnz + 8·(n_demes+1)`。
- 预算 `1.2 GB`：CSR 超过则**显式标记**为 skipped（不静默），仍记录 fold/nnz/内存。
- 另测论文核（`size=51`、`mean_dispersal=avd∈{0.5,1,2}`）在 300² 下的支持/nnz/内存。

## 2. 结果（sigma=1.5）

| 场地 | k | support | nnz | nnz/deme | CSR GB | fold s | build s | tick s |
|---|---|---|---|---|---|---|---|---|
| 30² | 3 | 9 | 6.84e3 | 7.6 | 0.000 | 0.00 | 0.03 | 0.016 |
| 30² | 5 | 25 | 1.98e4 | 22.0 | 0.000 | 0.01 | 0.03 | 0.019 |
| 30² | 11 | 121 | 8.91e4 | 99.0 | 0.001 | 0.03 | 0.05 | 0.037 |
| 30² | 21 | 441 | 2.70e5 | 299.4 | 0.004 | 0.09 | 0.12 | 0.084 |
| 30² | 51 | 2601 | 7.74e5 | 859.4 | 0.012 | 0.43 | 0.46 | 0.218 |
| 60² | 3 | 9 | 2.81e4 | 7.8 | 0.000 | 0.01 | 0.09 | 0.060 |
| 60² | 5 | 25 | 8.28e4 | 23.0 | 0.001 | 0.03 | 0.11 | 0.074 |
| 60² | 11 | 121 | 3.93e5 | 109.2 | 0.006 | 0.12 | 0.20 | 0.154 |
| 60² | 21 | 441 | 1.32e6 | 366.4 | 0.021 | 0.41 | 0.50 | 0.399 |
| 60² | 51 | 2601 | 5.80e6 | 1612.4 | 0.093 | 2.13 | 2.22 | 1.571 |
| 120² | 3 | 9 | 1.14e5 | 7.9 | 0.002 | 0.05 | 0.40 | 0.241 |
| 120² | 5 | 25 | 3.38e5 | 23.5 | 0.006 | 0.11 | 0.46 | 0.300 |
| 120² | 11 | 121 | 1.65e6 | 114.6 | 0.027 | 0.48 | 0.84 | 0.635 |
| 120² | 21 | 441 | 5.79e6 | 402.3 | 0.093 | 1.71 | 2.10 | 1.718 |
| 120² | 51 | 2601 | 2.99e7 | 2076.8 | 0.479 | 9.43 | 9.81 | 7.997 |
| 300² | 3 | 9 | 7.16e5 | 8.0 | 0.012 | 0.31 | 3.06 | 1.510 |
| 300² | 5 | 25 | 2.14e6 | 23.8 | 0.035 | 0.70 | 3.52 | 1.869 |
| 300² | 11 | 121 | 1.06e7 | 117.8 | 0.170 | 3.05 | 6.03 | 4.059 |
| 300² | 21 | 441 | 3.82e7 | 424.7 | 0.612 | 10.96 | 14.08 | 11.292 |
| 300² | 51 | 2601 | **2.15e8** | 2383.7 | **3.433** | **63.29** | — | — *(超预算跳过)* |

## 3. 结果（论文核，300²）

| avd | sigma | support | nnz | CSR GB | fold s |
|---|---|---|---|---|---|
| 0.5 | 0.3989 | 859 | 7.34e7 | 1.175 | 30.04 |
| 1.0 | 0.7979 | 2349 | 1.95e8 | 3.118 | 58.23 |
| 2.0 | 1.5958 | 2601 | 2.15e8 | 3.433 | 62.90 |

论文 `avd=1.0` 核的 `support=2349 < 2601` 来自 **natal 浮点 underflow**（σ=0.798 时 `exp(-d²/2σ²)` 在
`d≳30.8` 下溢为 0），**不是**论文的 `d≤25` 六边形盘掩膜——natal `build_gaussian_kernel` 不做该截断
（recon §3/§2.8；evaluator §7.4-F3）。

## 4. 关键结论（M4 决断输入）

1. **CSR 折叠是纯 Python、随 nnz 线性**：吞吐约 **3.3–3.6e6 entries/s**；300²×51² 需 **63 s**，
   论文核 avd=1.0 需 **58 s**。
2. **内存墙**：`CSR ≈ 16·nnz B`。300²×51² = 3.43 GB；论文核 avd=1.0 = 3.12 GB。外推海南
   `2599×2601`（6.76e6 demes、论文 r25 盘 ~1951 支持）→ `nnz ≈ 1.32e10`，`CSR ≈ 211 GB`，
   折叠 ≈ 65 min；若按 natal 全核（support 2601）则 ~281 GB——**CPU 侧不可行**（recon §4 预判被实测确认）。
3. **确定性 CPU tick 亦随 nnz 线性且昂贵**：大网格约 **2.7–3.0e-7 s / CSR entry / tick**（该带对应
   `k≳21` 的大核渐近；300² k=11 实测 3.8e-7、更小核更高，含固定 per-tick 开销）。
   300²×21²（nnz 3.8e7）已 **11.3 s/tick**；外推论文核 avd=1.0（nnz 1.95e8）约 **58 s/tick**，
   海南规模约 **66 min/tick**——论文量级（数十–上百代）在 CPU 上不现实。
4. **显式预算**：probe 对超 1.2 GB 的 CSR 显式标记 skipped；但 natal **CPU 路径本身无内存预算守卫**
   （`ensure_migration_budget`/`migration_cache_bytes` 仅 GPU，evaluator §7.2 已核）——建议 M4 一并考虑。
5. 与 recon §4 表对比：本实现口径（16 B/entry）下实测 300²×51²=3.43 GB、论文核 avd=1.0=3.12 GB，
   与预判量级一致。

## 5. 复现命令

```bash
cd hexagon_spatial_test && ../.venv/bin/python repro/m3_scale_probe.py
# -> results/m3_scale_data.json （约 6.5 min，峰值 RSS ~6.7 GB）
```

## 6. 残余与下一步

- **GPU 侧已完成**：见 `results/M3_gpu_report.md` + `m3_gpu_data.json`（deterministic enable/缓存足迹、
  随机行宽 ≤32 拒绝、预算守卫、`phase0` bit-identical）。
- CPU 侧尚未测：migrate 在 age_structured（含年龄/性别类）下的 tick 成本放大；海南逐月 K 的额外开销。
- 结论边界：本报告结论针对 CPU CSR 路径与确定性模式；`gpu` feature 默认关闭，未触及。外推为线性外推，
  未实测。

## 7. 独立审查状态

- 主 agent 自测：见 §2–§4。
- 独立审查（evaluator）：§7 **NOT APPROVED**（F1 fold/builder 口径、F2 标签、F3 措辞）；修复后 §8 =
  **APPROVED**（M3 CPU 侧闭环）。修复：fold 与 builder 均 `kernel_include_center=False`、tick 改
  `stochastic=False`、区分 underflow 与 `d≤25`，并重跑全部数据。GPU 侧 **NOT CHECKED**（按用户指示待确认）。
