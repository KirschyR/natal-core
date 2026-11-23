# 直接 stencil 随机迁移（方案 A）执行计划

> 状态：**方案 A 已选定**（用户决策）；**Stage 1 算子级 NumPy 原型已完成**（exact counter RNG + fast serial replay）；
> **NumPy CPU 全模型 demo 已产出**（small/large，HTML）；Rust/GPU/产品化未做。
> 本轮冻结：迁移率 `rate=0.5`；对照目标是 **CPU CSR**；GPU FFT 近似线（M0/M1/M2）暂停。
> 原型顺序：**NumPy 语义原型 → 独立 GPU 原型（3A，不接产品）→ 全模型小规模 → 最后接入 natal 产品**。
> 关联：`RandomnessRealization.md` 附录 E（问题分析与 A/B 对比）、`PLAN.md`（chasing 总体计划）、M4 模板压缩设计。
> 范围：`recon/hex-model`。原型在复现脚本/独立 harness；产品化属高风险（根 `EVALUATE.md` + 全量门禁 + 独立复核）。

## 0. 决策摘要（2026-11 更新）

- **问题**：完整 CSR（以及随机路径的 forward scratch + reverse CSR）内存为 $O(n\kappa^2)$；当平面 deme 数
  $n$ 和核边长 $\kappa$ 都很大时不可承受。
- **选择**：**直接 stencil + 方案 A**（逐边 counter-based RNG + 两遍重算）；放弃方案 B（分块 stencil +
  有界 scratch）。
- **目标语义**：与 CPU CSR **相同的随机模型**（源端 Beta 出境 + 目的地 Dirichlet 分配）；内存
  $O(n+\kappa^2)$；无原子加；固定种子可复现；代价约 2× Gamma 抽样。
- **rate**：本轮起测试与后续 chasing 一律使用 `migration_rate=0.5`；`rate=1.0` 只保留为历史对照或显式指定。
- **精度/对照**：与 **CPU CSR** 比较分布；GPU FFT 近似路径（M0/M1/M2）不再是本计划的对照物。
- **可复现性目标**：
  - 同一实现、同一设备、同一 seed：**逐位一致**（依靠 counter RNG、固定求和顺序、无原子加）；
  - CPU CSR vs 直接 stencil（RNG 流/求和顺序不同）、CPU vs GPU（f64 vs f32、设备/编译差异）：**分布一致**，
    不要求逐位。
- **原型顺序**：先做 NumPy 语义原型（使用与后续 GPU 一致的 RNG 设计），再独立 GPU 原型；**接入 natal
  产品是最后一步**，之前不改产品公开 API。

## 1. 背景：为什么不用完整 CSR，也不用 FFT

- 一个平面有 $n$ 个 deme（如 $N\times N$ 网格时 $n=N^2$）。源 $s$ 到目的地 $d$ 的有效位移集合为
  $\mathrm{valid}(s)$；核边长记作 $\kappa$（奇数），去中心核 $K'(u)$ 只依赖位移 $u=d-s$。
- 归一化权重 $w_{s,d}=K'(d-s)/Z_s$，其中 $Z_s=\sum_{u\in\mathrm{valid}(s)}K'(u)$ 是有效权重和。内部源有
  $r_s=\kappa^2-1$ 个有效目的地，边界更少；CSR 非零边总数 $\mathrm{nnz}=\sum_s r_s=O(n\kappa^2)$。
- FFT（route A）把确定性迁移降到 $O(n\log n)$，但它**无法承载逐边随机**（见
  `RandomnessRealization.md` §3）。直接 stencil 保留精确随机语义，只用 $O(\kappa^2)$ 的核模板，不保存
  完整 CSR。

## 2. 随机语义（与 CPU CSR 一致）

沿用 `RandomnessRealization.md` §2 的 natal 连续采样；本轮固定 `rate = rho = 0.5`。

**出境量**：对源 $s$（计数 $N_s$，迁移率 $\rho$），

$$
O_s=N_s B_s,\qquad
B_s\sim\mathrm{Beta}\big(\rho(N_s-1),(1-\rho)(N_s-1)\big),
$$

即与 $\mathrm{Binomial}(N_s,\rho)$ 同均值、同方差的连续版本；$N_s\le 1$ 时取 $O_s=\rho N_s$。保留量
$R_s=N_s-O_s$。

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
2. 按模板遍历有效位移 $u$，目的地 $d=s+u$；抽 $G_{s,d}$（键 `(seed, tick, s, d, gamma)`），累加得到
   $S_s$。
3. **只保存** $O_s$、$S_s$、$Z_s$（各 $O(n)$）；不保存 $G_{s,d}$。

### 3.2 Pass 2（逐目的地 gather，重算并求和）

对每个目的地 $d$：

1. 取保留量 $N_d-O_d$。
2. 按模板（反向）遍历偏移 $u$，源 $s=d-u$；若该位移有效，则用**同一个逐边键**
   `(seed, tick, s, d, gamma)` 重新抽 $G_{s,d}$。
3. 计算 $X_{s,d}=O_sG_{s,d}/S_s$（若 $O_s\le 1$，用确定性 $O_s w_{s,d}$），按**固定偏移顺序**累加。
4. 写出 $Y_d$。

固定求和顺序 ⇒ 无原子加、固定种子可复现。

### 3.3 RNG 设计（Philox + Marsaglia–Tsang）

- **Philox4x32-10 是 counter-based RNG（计数器随机数发生器）**：给定 key（种子）与 counter（计数器），
  直接哈希出 4 个 32-bit 均匀数；它不依赖“顺序消耗”的内部状态。因此同一 `(seed, tick, s, d, role)`
  在任何线程、任何遍次都能重放出同一随机数。现有 natal GPU 代码
  `rust/src/gpu/kernels.rs` 已有该实现（`RNG_SOURCE`/`RngState`）。
- **Marsaglia–Tsang 是 Gamma 采样算法**：shape ≥ 1 时用正态/拒绝采样，shape < 1 时做 boost；它会消耗
  **不定长**的均匀数。因为每条边有自己独立的 RNG 状态，Pass 1 与 Pass 2 从同一 `(key, counter)` 初始化，
  变长消耗不影响重放。
- **key/site 约定**：
  - `outbound` 与 `gamma` 使用不同 role/site，避免相关；
  - Pass 1 与 Pass 2 的 `G_{s,d}` 必须使用**完全相同**的 key/site/tick/edge；
  - 同一 tick 内不同源的 O_s 与 G 不能共享同一 RNG 流。
- **edge id / counter 位宽**：edge id 是每个 `(tick, plane, s, d, role)` 的唯一键。现有设备 `RngState`
  用 32-bit `cell` + 32-bit draw counter（最多约 4.29e9 条边）；当前 playground 远低于此，但大网格产品化
  会碰撞。**建议原型就用 64-bit edge id + 64-bit draw counter**，接入 Philox 的 128-bit counter
  （c0,c1=edge id，c2,c3=draw counter）；32-bit 只作小网格调试。这样不需要后续重新设计 key。
- **同 seed 逐位一致的条件**：同一实现、同一设备、同一编译/内核、固定求和顺序、无原子加、无
  `atomicAdd`/不确定 reduction。跨 CPU/GPU、跨设备或不同编译选项时只要求分布一致。
- **无原子加的保证**：Pass 2 每个目的地由一个线程（或固定顺序的 reduction）写出；Pass 1 每个源一个
  线程写自己的 `O_s/S_s`。不使用 scatter 原子加，因此相同 seed 的 GPU 运行可以做到逐位一致（在上述
  条件下）。

### 3.4 边界与模板

- 内部模板：$\kappa^2-1$ 个偏移与权重；边界源只需少量边界模板（数量约 $O(R^2)$，$R=(\kappa-1)/2$ 为
  核半径）。
- 有效性与 $Z_s$：Pass 2 由源位置 $s=d-u$ 判断位移是否有效，并用 $Z_s$（$O(n)$ 预存或按位置算）归一化
  $w_{s,d}=K'(u)/Z_s$。
- **不需要 reverse CSR**：gather 时枚举模板偏移即可。
- 可复用 `hexagon_spatial_test/repro/m4_template_build.py` 的 `analytic_templates`；先验证其 offset/权重
  与 CSR 一致，再用于 Pass 1。

## 4. 数据布局与复杂度

- 每平面状态：$O(n)$；$O_s,S_s,Z_s$：各 $O(n)$；核/模板：$O(\kappa^2 + R^2)$；输出 $O(n)$。
- 总计 **$O(n+\kappa^2)$ 每平面**；无 CSR、无 forward scratch、无 reverse CSR、无原子加。
- 时间：每平面每 tick $O(n\kappa^2)$（Pass 1 抽 $n\kappa^2$ 个 Gamma，Pass 2 再抽一遍并求和）；约
  **2× Gamma 抽样**。
- 对比：CSR 随机也是 $O(n\kappa^2)$ 时间，但内存 $O(n\kappa^2)$；FFT 是 $O(n\log n)$ 时间但仅确定性。

## 5. 原型与验证顺序

### 5.0 Playground（与既有 demo 一致）

- 沿用此前 M0/M1/M2 测试的同一 playground：
  - `small`: 15² / kernel 7 / avd 0.8；
  - `large`: 31² / kernel 11 / avd 1.5；
  - `rate=0.5`；100 ticks；同一初始态/中心释放；无 chasing。
- **三型最小模型**：沿用现有 v3 demo 的 3 个合子类型 `WT|WT, WT|Drive, Drive|Drive`
  （两个等位基因 WT/Drive，无 resistance），不是新模型；5 型/chasing 留到后续。
- **demo 策略**：CPU CSR vs **stencil**。把此前 demo 中的非 CSR 策略（M0/M1/M2）替换为 stencil；
  CPU CSR 继续作为参考；M1 等不再展示。
- **展示要求不变**：整 grid 时间滑块热图、沿线 4 deme 的 drive/non-egg/juvenile mean±SD、
  coverage/front/T50、计时表；HTML 放 `results/html/`。
- 样本量：demo 用 R=100（small）/ R=50（large）；若 stencil 原型过慢，可先降低并记录，但不改变
  playground 配置。

### 5.1 Stage 1 — NumPy 语义原型（先做）

- 新脚本 `direct_stencil_operator.py`（复现目录；暂不接产品）。
- 先做**完整 kernel + 边界判断**版本（不做模板优化），实现 Pass 1/Pass 2；逐边 RNG 使用与后续 GPU
  一致的 **Philox4x32-10 key 结构**，Gamma 用 **Marsaglia–Tsang**（与后续 GPU 原型一致）。
- 场景：`rate=0.5`；15²/k7/avd0.8、31²/k11/avd1.5；单平面；R=100–300。
- 对照：CPU CSR 随机迁移（`scripts/compare_migration_models.py` / v3 的 `cpu_migrate`）。
- 指标：per-deme 均值、方差比、lag 平均协方差、总种群 mean/std、负值比例、逐源守恒；
  - 同 seed 两次运行逐位一致；
  - Pass 1/Pass 2 的 `G_{s,d}` 重放一致（可在原型中直接断言）；
  - 与 CPU CSR 只比分布，不比逐位。
- 产物：JSON + 简短对照表；`EVALUATE.md` 交接。

### 5.2 Stage 2 — 模板、边界与内存/时间

- 接入 `analytic_templates`，验证：
  - 每个源的 offset 集合/权重与 CSR 一致；
  - `Z_s` 与模板归一化一致；
  - Pass 2 结果与“完整 kernel 版本”逐位一致。
- 内存：`O_s/S_s/Z_s/type_of`、模板 bytes；对比 CSR 的 `nnz` 内存与随机 CSR 的 forward scratch。
- 时间：Pass 1/Pass 2/total、Gamma 次数；对比 CPU CSR；给出 n×κ 阶梯（15²/k7、31²/k11、60²/k11、
  120²/k21、300²/k21/51 等）。
- 目标不是预设“通过/失败”，而是量化直接 stencil 与全 CSR 的差异与优化空间。

### 5.3 Stage 3 — 独立 GPU 原型（3A，不接产品）

- **承载方式：与 M2 GPU 原型一致**——在 `chasing_model/` 下写独立 Python/torch 脚本，用
  `/opt/conda/bin/python` 运行；不接 natal 公共 API/会话，不改默认路径。
- 在 torch 中实现 Philox4x32-10 + Marsaglia–Tsang；先在 f64 小网格验证语义，再用 f32 测性能/分布。
- 精度：GPU 硬件支持 f64；现有 natal 随机 GPU 内核是 f32。原型以 **f32 为主**（与未来产品一致），
  f64 用于语义/同实现逐位检查；CPU CSR（f64）与 GPU f32 只比分布。
- 验证：
  - 同 seed 两次 GPU 运行逐位一致（f32 同实现/同设备；hash 输出）；
  - 无原子加、固定 offset 顺序；
  - GPU vs CPU CSR 的分布差异（均值/方差/lag 协方差/总量/负值）；
  - f64 原型与 f32 原型的差异，用于估计 f32 容差。
- 产物：独立 GPU 原型结果 JSON + 内存/时间表；不改产品代码。

### 5.4 Stage 4 — 全模型 playground 对照

- 在 5.0 的同一 playground（small/large、3 型、rate=0.5、100 ticks）上，用 NumPy stencil 原型
  （或独立 GPU 原型的全模型入口）与 CPU CSR 比较：
  - 覆盖曲线、前沿半径、T50、总量；
  - 沿线 4 deme 的 drive/non-egg/juvenile mean±SD；
  - C2ST（R 足够大时；小 R 的单次 split 不作结论）；
  - 内存/时间。
- 本阶段不追求 chasing S1/S2；先看 stencil 与 full CSR 的差异。
- 只有确认差异与收益后，才讨论产品化。

### 5.5 Stage 5 — 接入 natal 产品（最后一步，高风险）

- 默认关闭、显式 opt-in；不改变 CPU CSR 语义；`phase0` bit-identical。
- 根 `EVALUATE.md`、中英文档、stub、双构建门禁、`cargo test`（含 gpu）、`pytest/ruff/pyright`、独立复核。

## 6. 与现有实现/文档的关系

- CPU 精确参考：`rust/src/kernels/spatial.rs` 的 `migrate_csr_stochastic_rngs`
  （`sample_outbound` + `distribute_csr_outbound`）。
- 现有 GPU 随机 CSR：`rust/src/gpu/executor.rs` 的 `migrate_tick_stochastic`
  （forward scratch + reverse CSR，无 atomics），受 `MAX_CSR_ROW` 与内存限制。
- route A FFT：`rust/src/gpu/fft_migrate.rs`，确定性；随机近似 M1–M4 见
  `RandomnessRealization.md` §4（**本计划暂停该近似线，除历史对照外不再扩展**）。
- 模板压缩：`hexagon_spatial_test/repro/m4_template_build.py`（确定性），可复用。
- 现有直接 stencil 分析：`RandomnessRealization.md` 附录 E。

## 7. 风险与未决

1. **2× Gamma 抽样**：Gamma 较贵；需要基准测试确认可接受；直接 stencil 的收益是内存，不保证时间。
2. **counter-based Gamma / key 空间**：逐边独立 RNG 可重放，但 64-bit/128-bit edge id 的产品化设计需确认。
3. **逐位 vs 分布**：同实现同设备可逐位；CPU CSR vs stencil、CPU vs GPU、跨设备只比分布。
4. **f32 精度**：GPU 随机路径 f32 与 CPU f64 的 Gamma 差异；需明确容差。
5. **性能常数**：gather 访存不连续；小 κ 可能不如 CSR；大 κ/大 n 时主要看内存是否可行。
6. **产品化**：最后一步，需默认关闭、文档、双构建门禁、`phase0` bit-identical、独立复核。
7. **不设“成功阈值”**：本轮以观察差异、量化内存/时间/分布为目标；是否产品化在 Stage 4 后决定。

## 8. 观察与验证清单（不预设通过/失败）

1. **语义**：CPU CSR vs 直接 stencil（rate=0.5）的 per-deme 均值、方差比、lag 平均协方差、总种群
   mean/std、负值比例；报告 MC 误差与样本量。
2. **复现**：同 seed 两次直接 stencil（同实现/设备）逐位一致；Pass 1/Pass 2 的 `G_{s,d}` 重放一致；
   输出 hash/逐位比较。
3. **模板/边界**：offset/权重与 CSR 一致；边界源、`O_s<=1`、`S_s<=EPS`、空行等分支。
4. **内存**：每平面实际 bytes（状态、O/S/Z、模板、输出）与 CSR/forward scratch 对比。
5. **时间**：Pass 1/Pass 2/总时间、Gamma 次数；对比 CPU CSR；给出 n×κ 阶梯与外推。
6. **全模型**：覆盖/前沿/T50/共存/固定概率的差异；R 足够大时再用 C2ST/等价检验。
7. **GPU 原型**：f32/f64、同 seed 逐位、CPU vs GPU 分布；无原子加；显存与每 tick 时间。

## 9. 里程碑

- [x] 方案 A 选定（放弃 B）。
- [x] Stage 1：NumPy 语义原型（Philox + Gamma，完整 kernel 版本）。
- [~] Stage 2：模板/边界验证 + 内存/时间阶梯（NumPy 算子级已测；全阶梯/Rust 公平计时待补）。
- [ ] Stage 3：独立 GPU 原型（3A，不接产品）。
- [~] Stage 4：全模型小规模差异评估（NumPy CPU demo 已产出 small/large；R 暂为 small=10、large=3）。
- [ ] Stage 5（可选，最后）：接入 natal 产品（高风险）。

### 9.1 本轮产物（2026-11，NumPy CPU）

- `scripts/direct_stencil_migration.py`：Philox4x32-10 + Marsaglia–Tsang、两遍 offset-loop；
  `stencil_migrate_planes` 为逐边 counter RNG 的 exact 路径，`stencil_migrate_planes_fast`
  为 NumPy 双 Generator 串行重放的 fast 路径。
- `scripts/direct_stencil_operator.py`：NumPy CPU CSR vs stencil 算子级对拍（分布指标、逐源守恒、
  同 seed 逐位、Gamma 重放）；`results/direct_stencil_operator_exact.json`、
  `results/direct_stencil_operator_fast.json`。
- `scripts/compare_stencil_demo.py`：复用 v3 生命周期与三型最小模型，仅替换 migration backend；
  `results/stencil_demo/{small,large}/` 与 `results/html/compare_stencil_demo_*.html`。
- 当前结论（自测，待独立复核）：算子级 small/large 的均值、方差比、lag 协方差在 MC 误差内；
  同实现同 seed 逐位一致；fast 与 exact 分布指标接近。全模型 small/large 在 100 tick 均固定，
  CPU 与 stencil 的覆盖/前沿/总量差异很小；C2ST（small R=10）未见可分。
- 残余：CPU golden 目前是 NumPy 镜像，尚未与 natal Rust CPU CSR 做一次分布交叉核对；Rust/GPU
  公平计时、模板压缩全阶梯、内存账本、chasing S1/S2 未做。

## 10. 已确认与待确认

已确认（本轮）：

- `rate=0.5`；对照 CPU CSR；GPU FFT 近似线暂停。
- 先 NumPy 语义原型；再独立 GPU 原型；产品接入最后。
- 目标为分布一致；同实现同种子应逐位一致；直接 stencil **无原子加**，不存在原子加导致的同 seed
  不可复现问题。
- RNG 方向：Philox4x32-10 counter-based + Marsaglia–Tsang Gamma；逐边 key。

已确认（补充）：

- Playground 与此前 M0/M1/M2 demo 完全一致：small 15²/k7/avd0.8、large 31²/k11/avd1.5、3 型最小模型、
  rate=0.5、100 ticks、同一初始/释放；demo 只把非 CSR 策略替换为 stencil。
- 独立 GPU 原型承载方式与 M2 原型一致：`chasing_model/` 下独立 Python/torch 脚本，用
  `/opt/conda/bin/python`，不接 natal 产品。
- Gamma 用 Marsaglia–Tsang。
- edge id/counter：原型即用 64-bit edge id + 64-bit draw counter；32-bit 只作小网格调试。
- 精度：f32 为主（与现有 GPU 随机路径一致），f64 用于语义与同实现检查；CPU CSR(f64) vs GPU(f32)
  只比分布。
- 先 3 型最小模型完成 playground 对照；不追求 chasing，5 型留到后续。

待确认（实现前只剩）：

1. 独立 GPU 原型的 torch 实现细节：Philox + Marsaglia–Tsang（含拒绝采样）在 torch 中的可行性与
   性能；若过慢，只用于语义/小网格，性能另行优化。
2. demo 的 R：计划 small R=100、large R=50；若 stencil 原型耗时过高，可先降低并记录。
3. f64 GPU 原型是否必须（用于语义检查）；若只做 f32，需在报告中声明。
