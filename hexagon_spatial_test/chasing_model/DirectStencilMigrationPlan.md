# 直接 stencil 随机迁移（方案 A）执行计划

> 状态：**方案 A 已选定**（用户决策）；尚未实现/产品化。
> 关联：`RandomnessRealization.md` 附录 E（问题分析与 A/B 对比）、`PLAN.md`（chasing 总体计划）、M4 模板压缩设计。
> 范围：`recon/hex-model`。原型先在复现脚本；若产品化，属高风险（根 `EVALUATE.md` + 全量门禁 + 独立复核）。

## 0. 决策摘要

- **问题**：完整 CSR（以及随机路径的 forward scratch + reverse CSR）内存为 $O(n\kappa^2)$；当平面 deme 数 $n$ 和核边长 $\kappa$ 都很大时不可承受。
- **选择**：**直接 stencil + 方案 A**（逐边 counter-based RNG + 两遍重算）。
- **放弃方案 B**（分块 stencil + 有界 scratch）：抽样约一遍、内存 $O(n+\text{tile}\cdot\kappa^2)$，但 halo/块边界复杂，可复现性更难保证。
- **目标语义**：与 CPU CSR **相同的随机模型**（源端 Beta 出境 + 目的地 Dirichlet 分配）；内存 $O(n+\kappa^2)$；无原子加；固定种子可复现。
- **代价**：约两遍 Gamma 抽样；Gamma 采样器必须支持逐边独立 seed。

## 1. 背景：为什么不用完整 CSR，也不用 FFT

- 一个平面有 $n$ 个 deme（如 $N\times N$ 网格时 $n=N^2$）。源 $s$ 到目的地 $d$ 的有效位移集合为 $\mathrm{valid}(s)$；核边长记作 $\kappa$（奇数），去中心核 $K'(u)$ 只依赖位移 $u=d-s$。
- 归一化权重 $w_{s,d}=K'(d-s)/Z_s$，其中 $Z_s=\sum_{u\in\mathrm{valid}(s)}K'(u)$ 是有效权重和。内部源有 $r_s=\kappa^2-1$ 个有效目的地，边界更少；CSR 非零边总数 $\mathrm{nnz}=\sum_s r_s=O(n\kappa^2)$。
- FFT（route A）把确定性迁移降到 $O(n\log n)$，但它**无法承载逐边随机**（见 `RandomnessRealization.md` §3）。直接 stencil 保留精确随机语义，只用 $O(\kappa^2)$ 的核模板，不保存完整 CSR。

## 2. 随机语义（必须与 CPU CSR 完全一致）

沿用 `RandomnessRealization.md` §2 的 natal 连续采样：

**出境量**：对源 $s$（计数 $N_s$，迁移率 $\rho$），

$$
O_s=N_s B_s,\qquad
B_s\sim\mathrm{Beta}\big(\rho(N_s-1),(1-\rho)(N_s-1)\big),
$$

即与 $\mathrm{Binomial}(N_s,\rho)$ 同均值、同方差的连续版本；$N_s\le 1$ 时取 $O_s=\rho N_s$。保留量 $R_s=N_s-O_s$。

**目的地分配**：对源 $s$ 的每个有效目的地 $d$，

$$
\alpha_{s,d}=w_{s,d}(O_s-1),\qquad
G_{s,d}\sim\mathrm{Gamma}(\alpha_{s,d},1),\qquad
S_s=\sum_d G_{s,d},\qquad
X_{s,d}=O_s\frac{G_{s,d}}{S_s}.
$$

若 $O_s\le 1$，退化为确定性 $X_{s,d}=O_s w_{s,d}$。

**合成**：

$$
Y_d=(N_d-O_d)+\sum_{s\ne d}X_{s,d}.
$$

逐源守恒 $\sum_d X_{s,d}=O_s$，全局守恒 $\sum_d Y_d=\sum_d N_d$。

## 3. 方案 A：逐边 RNG + 两遍重算

核心：**不保存逐边量**，用可重放的逐边随机数在两遍中重新生成同一个 $G_{s,d}$。

### 3.1 Pass 1（逐源，算 $S_s$）

对每个源 $s$：

1. 抽 $O_s$（用键 `(seed, tick, s, outbound)`）。
2. 按模板遍历有效位移 $u$，目的地 $d=s+u$；抽 $G_{s,d}$（键 `(seed, tick, s, d, gamma)`），累加得到 $S_s$。
3. **只保存** $O_s$、$S_s$、$Z_s$（各 $O(n)$）；不保存 $G_{s,d}$。

### 3.2 Pass 2（逐目的地 gather，重算并求和）

对每个目的地 $d$：

1. 取保留量 $N_d-O_d$。
2. 按模板（反向）遍历偏移 $u$，源 $s=d-u$；若该位移有效，则用**同一个逐边键** `(seed, tick, s, d, gamma)` 重新抽 $G_{s,d}$。
3. 计算 $X_{s,d}=O_sG_{s,d}/S_s$（若 $O_s\le 1$，用确定性 $O_s w_{s,d}$），按**固定偏移顺序**累加。
4. 写出 $Y_d$。

固定求和顺序 ⇒ 无原子加、固定种子可复现。

### 3.3 RNG 设计

- 需要 **counter-based RNG**（计数器随机数发生器）：输出是键 `(seed, tick, s, d, role)` 的哈希，给定键即可 $O(1)$ 重放同一次抽样；不依赖顺序消耗的随机流。
- 每条边构造一个独立的 RNG（由键 seed），再用它抽 Gamma；这样 Pass 1 与 Pass 2 得到相同 $G_{s,d}$。
- 不同用途使用不同 `role`（如 `outbound`、`gamma`），避免相关性。
- **风险**：常用 Gamma 采样器（如 Marsaglia–Tsang）会消耗不定长随机数；必须用“每条边独立 seed 的 RNG”或实现可重放的 counter-based Gamma。这是本方案最关键实现点。

### 3.4 边界与模板

- 内部模板：$\kappa^2-1$ 个偏移与权重；边界源只需少量边界模板（数量约 $O(R^2)$，$R=(\kappa-1)/2$ 为核半径）。
- 有效性与 $Z_s$：Pass 2 由源位置 $s=d-u$ 判断位移是否有效，并用 $Z_s$（$O(n)$ 预存或按位置算）归一化 $w_{s,d}=K'(u)/Z_s$。
- **不需要 reverse CSR**：gather 时枚举模板偏移即可。

## 4. 数据布局与复杂度

- 每平面状态：$O(n)$；$O_s,S_s,Z_s$：各 $O(n)$；核/模板：$O(\kappa^2 + R^2)$；输出 $O(n)$。
- 总计 **$O(n+\kappa^2)$ 每平面**；无 CSR、无 forward scratch、无 reverse CSR、无原子加。
- 时间：每平面每 tick $O(n\kappa^2)$（Pass 1 抽 $n\kappa^2$ 个 Gamma，Pass 2 再抽一遍并求和）；约 **2× Gamma 抽样**。
- 对比：CSR 随机也是 $O(n\kappa^2)$ 时间，但内存 $O(n\kappa^2)$；FFT 是 $O(n\log n)$ 时间但仅确定性。

## 5. GPU 实现计划

- **Pass 1**：一个线程/线程块处理一个源（或一个 tile 的多个源）；模板放常量/共享内存；逐偏移抽 $G$，累加 $S_s$；写 $O_s,S_s$。
- **Pass 2**：一个线程处理一个目的地；逐偏移重算 $G$ 并累加；写 $Y_d$。**无写冲突、无 atomics**。
- 合并访问：模板偏移按行主序；输出按目的地连续写；gather 读 $O_s,S_s$ 可能跨源，注意缓存。
- 可选：把 Pass 2 按 tile 分块以提升局部性（但仍是 gather，不是方案 B 的 scratch）。
- 精度：CPU 参考用 f64；现有 GPU 随机 CSR 用 f32。原型先用 f64 验证分布；产品化需明确 f32/f64 与容差。

## 6. 与现有实现/文档的关系

- CPU 精确参考：`rust/src/kernels/spatial.rs` 的 `migrate_csr_stochastic_rngs`（`sample_outbound` + `distribute_csr_outbound`）。
- 现有 GPU 随机 CSR：`rust/src/gpu/executor.rs` 的 `migrate_tick_stochastic`（forward scratch + reverse CSR，无 atomics），受 `MAX_CSR_ROW` 与内存限制。
- route A FFT：`rust/src/gpu/fft_migrate.rs`，确定性；随机近似见 `RandomnessRealization.md` §4 的 M1–M4。
- 模板压缩：M4 的 `m4_template_build.py`（确定性）可复用。

## 7. 风险与未决

1. **2× Gamma 抽样**：Gamma 采样较贵；需要基准测试确认可接受。
2. **counter-based Gamma**：能否为每条边独立 seed 并保持统计正确；可能需要自定义采样器。
3. **精度与可复现**：f32 与 f64 的差异；固定求和顺序；跨平台 RNG 一致性。
4. **性能常数**：gather 访存不连续；小 $\kappa$ 时可能不如 CSR；大 $\kappa$ 时优于 CSR 内存但时间可能仍远高于 FFT。
5. **与 CPU CSR 的关系**：RNG 键与求和顺序不同 ⇒ 目标是**分布一致**，不保证逐位等于 CPU CSR；若要求逐位，需要额外对齐 RNG 与求和顺序。
6. **产品化**：高风险，需默认关闭、文档、双构建门禁、`phase0` bit-identical、独立复核。

## 8. 验证计划

1. **算子级原型**（最小）：固定随机场，CPU CSR vs 方案 A；比较 per-deme 均值、方差比、lag 平均协方差、总种群 mean/std、负值比例；R 次重复，报告 MC 误差。
2. **可复现性**：固定种子两次运行逐位一致（固定求和顺序）。
3. **内存/时间阶梯**：$n$ 与 $\kappa$ 的若干组合，测显存与每 tick 时间，对比 CSR 与 FFT。
4. **整模型小规模**：CPU CSR vs 方案 A 的覆盖曲线、前沿、共存时长（分布对照）。
5. **验收草案**：均值在 MC 误差内；方差比中位 0.9–1.1；lag 协方差接近 CPU；总量守恒；可复现。

## 9. 里程碑

- [x] 方案 A 选定（放弃 B）。
- [ ] 算子级原型（`direct_stencil_operator.py` 或 Rust 原型）。
- [ ] 内存/时间基准。
- [ ] 整模型小规模分布对照。
- [ ] （可选，高风险）产品化。

## 10. 开放问题（需确认）

- 原型语言：先用 NumPy/torch（快、易验证），还是直接 Rust/CUDA？
- 精度：f32 还是 f64？（影响与 CPU CSR 的分布对照容差与产品化一致性。）
- RNG：选用哪种 counter-based 生成器（Philox/Threefry/自研哈希）与 Gamma 采样器？
- 是否要求与 CPU CSR **逐位一致**，还是只要求**分布一致**？
