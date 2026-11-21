# Chasing Model 复现计划（natal 引擎 · RSA · 全随机 · route A 加速）

> 状态：计划 v3（按 evaluator §27 修订 + 用户 2026-11-09 新要求：随机也走 GPU/route A）
> 依据：用户决策；`AGENTS.md` / `quality_checks_spec*.md` / `docstring_spec.md` 为最高依据。
> 范围：`recon/hex-model`。复现脚本默认不改产品代码；**随机 route A 属高风险产品改动**，须用户批准 + 根 `EVALUATE.md` + 全量门禁 + 独立复核。

## 0. 目标与已冻结口径

- 在 natal 引擎的 `HexGrid` 上，用**完全随机**的年龄结构模型复现 **chasing**：驱动与 WT 空间共存、前沿持续移动、不必然全局固定。
- 事件顺序固定为 natal **R-S-A → migration（tick 末尾）**；不再逐位还原 MATLAB hex model。
- 首个驱动 **homing**；随机采样先用**连续采样**；小范围优先。
- **加速目标**：随机空间模型也要能用 GPU（route A / cuFFT）加速，这是 hex-recon 项目的核心目的之一。
- 结果需有**带时间滑块的 HTML 热图**（整 grid，热度随时间）。
- `event-seq/`、`hexagon_spatial_test/hex-model-new/` 与本轮无关，已 gitignore。

## 1. 加速路线与两阶段验证

### 1.1 route A 是什么
natal 原迁移把高斯核在构建期折叠成**显式 CSR**（每个 deme 一行邻居/权重），运行期按行发送。route A 改为**平移不变的归一化卷积**：

```
g = rate · f / Z，  f_new = f · (1 − rate) + K' ⊛ g，  Z = K' ⊛ m
```

其中 `K'` 是去中心核、`m` 是域内指示；`⊛` 用 **cuFFT** 在 GPU 上逐 `(sex, age, ztype)` 平面做线性卷积。优点是避开 CSR 的内存/构建墙、大核更快；代价是非逐位、当前**只支持确定性**。

### 1.2 V1 — 确定性 CPU CSR vs GPU FFT 一致性（低风险，先做）
- 一个**简单小例子**（小 HexGrid + 年龄结构 + 中心释放），分别跑：
  - 传统 `migration_execution="csr"`（CPU）；
  - route A `migration_execution="fft"` + `enable_gpu()`。
- 绘制两种方案的**种群折线图**（总种群、各基因型/携带率、覆盖随时间的曲线）。
- 判据：确定性下二者**不可能逐位相同**（f32 cuFFT），但应在机器精度档内一致（相对误差目标 `≤1e-5`，以实测为准）；同时检查重复运行可复现。
- 产物：`chasing_model/verify_routeA_deterministic.py` + `results/routeA_deterministic.json` + 折线图（HTML/PNG，按全局 ignore 不入库）。

### 1.3 V2 — route A 随机迁移：**先在复现脚本用 GPU 原型验证**（语义已定）
- 用户决策（2026-11-09）：随机语义 = **卷积期望 + 目的地噪声**；**先在 `chasing_model/` 的 GPU 原型验证，不改产品代码**。
- 原型：用 FFT（natal route A 或 conda torch/cuFFT）得到每目的地期望 `m_d`，再按连续采样（Poisson/正态/Beta 近似）逐目的地抽样；与 CPU CSR 随机做 V3 分布对照。
- 通过后再决定是否产品化。产品化属**高风险**：默认关闭、zh/en 文档、双构建门禁、`phase0` bit-identical、走**根 `EVALUATE.md`** + 独立复核。

### 1.4 V3 — 随机 CPU vs GPU 的**分布**一致性（统计检验）
- 不要求逐位一致；比较大量重复下同一配置的**分布**：总种群/携带率/覆盖的均值、方差、分位数、直方图。
- 需说明：样本量、容差依据、种子策略、偶发失败控制；给出 CPU 与 GPU 的均值差与方差比、置信区间。
- 判据（草案，待定）：均值相对差在若干 CI 内；方差比接近 1（大计数极限）；不能仅凭单次运行声称等价。

## 2. 模型定义（natal `SpatialPopulation`, `age_structured`, `HexGrid`）

- **年龄结构**：2 幼 + 5 雄 + 8 雌 → `n_ages=10`、`new_adult_age=2`，雄龄 7–9 零填充。
- **初始态**：每 deme WT 平衡年龄向量；S1 起先 **burn-in**；所有频率/总种群指标**排除 age0**。
- **遗传（5 型 homing）**：等位 `[WT, Drive, Res]`（6 ztype）；`HomingDrive` 用 `drive_conversion_rate`(=`dc`) 与 `late_germline_resistance_formation_rate`(=`gcr`)；carrier=`{Drive|Drive, WT|Drive, Drive|Res}`；复用 M1 已验证的 5×5 映射，不用 3 型。
- **存活/老化**：雌 `[1,1,7/8,6/7,5/6,4/5,3/4,2/3,1/2,0]`；雄 `[1,1,4/5,3/4,2/3,1/2,0,0,0,0]`。
- **繁殖**：`eggs_per_female=40`、`sex_ratio=0.5`、`fixed_egg_count=True`、`sperm_displacement_rate=1.0`。
- **密度**：`beverton_holt`；`λ=5`、`competition_strength=5`、`D=40+5·(4/9)`；K/期望产卵按平衡态推导。
- **迁移**：`build_gaussian_kernel("hex", size=7, mean_dispersal=avd)`；小范围 CPU CSR 归一化边界；route A FFT 见 §1。
- **释放**：中心圆 `r=3`、双性 `Drive|Drive` 投首个成虫龄、`Op.add` 钩子。
- **随机（F2）**：`stochastic=True` + `continuous_sampling=True`；`scaling_factor s` **同时乘初始年龄向量与 `age_1_carrying_capacity`**（`expected_comp=s·42.222`、`expected_survival_rate` 不变）；JSON 记录 `s/K/expected_comp`。

## 3. chasing 判据与指标

- **空间图**：每 k tick 记录 driver / WT(`ww`) / carrier 的 per-deme 频率（排除 age0）。
- **成功判据（S1/S2 决策门）**：① burn-in 后 WT 与 driver 频率均 > ε 且持续 ≥ `N_coexist`(默认 100) tick；② carrier 前沿半径随时间严格增长；③ 未全局固定。
- **失败判据**：前沿连续 `N_stall`(默认 100) tick 不增；或 WT 只在驱动后方均匀残留无滞后带；或固定/灭绝。
- **fallback**：homing + 连续采样在 S2 明确失败 → 缩小目标并如实报负；切换机制（TARE/阈值/Allee/WT 源）需用户批准。
- **结局分类**：重复 → drive 固定 / WT 固定 / 持续共存；固定概率 + 95% 区间。
- **辅助**：T50/T90、WT 避难所分布、总种群（排除 age0）。

## 4. 实验阶梯（小范围优先）

| 步 | 内容 | 规模 | 风险 |
|---|---|---|---|
| **V1** | 确定性 CPU CSR vs GPU FFT，种群折线图 + 误差 | 小 HexGrid（如 15×15） | 低 |
| **V2** | route A 随机迁移 **GPU 脚本原型**（卷积期望+目的地噪声） | 小 HexGrid | 低（脚本；产品化后高） |
| **V3** | CPU vs GPU 随机**分布**统计检验 | 小/中 | 中 |
| **S0** | 引擎随机 hex 冒烟（CPU CSR） | 15×15 | 低 |
| **S1** | homing 确定 vs 随机基线 + burn-in + release 小扫；判据 | 41×41 | 中 |
| **S2** | 扫 `dc`/`gcr`/fitness/`avd`/`s`/释放；决策门/fallback | 41×41 | 中 |
| **S3** | 重复统计（固定概率/共存时长/CI） | 41×41 | 中 |
| **S4** | 时间轴 HTML 热图 | S2/S3 | 低 |
| **S5** | 放大 101×101（CPU CSR 与 route A 各自） | 放大 | 中 |

## 5. 产物布局（`hexagon_spatial_test/chasing_model/`）

```
PLAN.md
verify_routeA_deterministic.py   # V1
model.py                         # 构建 natal 模型
run_chasing.py                   # 实验与重复统计
metrics.py                       # 共存/前沿/结局/分布统计
viz_html.py                      # 时间轴滑块 HTML 热图
results/                         # JSON（HTML/PNG 受全局 ignore）
```

- **记录（F7）**：逐 tick `pop.run(1, record_every=0)` 只存 per-deme 频率（`~41×41×T` float），不存全量 state。
- **结果格式（F8）**：统一 JSON；不使用 CSV；HTML/PNG 不入库。

## 6. 风险与坑

1. `record_every=1` OOM → `0` + 逐 tick 存频率。
2. 连续采样 `n≤1` 无噪声；`scaling_factor` 须按 F2 同步 K。
3. 迁移边界：natal 丢界+重归一化（非 replicate），已接受。
4. route A 非逐位（f32）；确定性对照用相对误差，随机对照用**分布**。
5. **route A 随机：先脚本原型**（M0/M1 等，不改产品代码）；若产品化，才属产品改动（默认关闭、文档同步、双构建门禁、`phase0` bit-identical、独立复核）。
6. 随机可复现：固定种子、重复数、区间。

## 7. 审查流程

- 复现脚本/数据（含 V1/V2 原型）：`hexagon_spatial_test/EVALUATE.md`（§N ↔ §N+1）。
- **若后续把随机 route A 产品化**：走**根 `EVALUATE.md`** + 全量门禁 + 独立复核；不得用本计划的复现通道代替。
- 当前：§26 被 §27 判 NOT APPROVED（计划文本）；v3 已并入 §28 追加，待 §29 复核。

## 8. 里程碑

- [x] V1：确定性 CPU CSR vs GPU FFT 多配置对照（2026-11-12，全通过）
- [x] V2a：算子级 M0/M1 vs CPU CSR（含 lag 平均协方差）
- [x] V2b：整模型 CPU CSR/M0/M1 数据比较（NumPy 原型，small/large，mean±SD + HTML，2026-11-12）
- [~] 算子拆模块 / M1 边界精修：用户暂缓（当前不做）
- [ ] V3：随机 CPU vs GPU 分布统计检验
- [ ]（可选，另审批）随机 route A 产品化
- [ ] S0：随机 hex 冒烟（CPU CSR）
- [ ] S1：homing 基线 + 判据
- [ ] S2：chasing 参数搜索
- [ ] S3：重复统计
- [ ] S4：时间轴 HTML
- [ ] S5：放大


## 9. 随机模型对比协议与 M0/M1 中期结果

### 9.1 对比协议（先算子级，后整模型）

- 同一初始场；各模型独立重复 R 次；固定种子与 R，报告 MC 误差（方差比 SE ≈ √(2/(R−1))）。
- **算子级指标**：
  1. per-deme 均值：中位/最大相对差（并与 MC 标准误比较）；
  2. per-deme 方差比（模型/CPU）：中位、5–95 分位；
  3. **按 lag 平均的空间协方差**（lag `(0,0),(0,1),(1,0),(1,1),(0,2),(2,0)`）；
  4. 总种群 mean/std（守恒：CPU std≈0）；
  5. 负值比例。
- **关键教训**：R~300 下不要用**全协方差矩阵 Frobenius 误差**——非对角真值约 0.1–0.3，而 MC 标准误约 0.6，噪声主导。应使用 lag 平均协方差（跨平移等价对平均）或解析目标协方差。
- **整模型指标**：覆盖曲线、前沿半径、共存时长、固定概率的 `mean±95% CI`；lifecycle 与 migration 用**独立 RNG 流**，让各模型共享 lifecycle 噪声以配对降方差。
- **验收草案**：M1 方差比≈1 且 lag 协方差接近 CPU；M0 作基线；总量守恒；均值在 MC 误差内一致。

### 9.2 当前 M0/M1（算子级：21×21，R=300，rate=0.5，avd=1，kernel 7）

| lag | CPU | M0 | M1 |
|---|---|---|---|
| (0,0) 方差 | 11.190 | 4.310 | **11.222** |
| (0,1) | −1.242 | −0.845 | −1.074 |
| (1,0) | −1.320 | −0.860 | −1.133 |
| (1,1) | −0.375 | −0.035 | −0.396 |
| (0,2) | −0.280 | +0.022 | −0.185 |
| (2,0) | −0.242 | −0.001 | −0.250 |

- 方差比中位：**M0 0.38**，**M1 1.001**；总量：CPU 严格守恒，M0/M1 std≈0。
- lag 协方差：**M1 明显优于 M0**（M0 丢失 (1,1)/(0,2) 的负相关）。
- 均值：M0 中位 0.9%/最大 4.6%；M1 1.2%/5.9%（R=300 的 MC 噪声内）。
- 残余：M1 的 per-deme 方差比 p05–p95 为 0.46–1.50（边界/非均匀 N 的局部调制）；部分 lag 协方差低 15–35%（circulant/uniform 近似）。下一步：改善局部缩放/边界处理，或用精确有限域协方差校验。
- 产物：`compare_migration_models.py`、`results/compare_migration_models.json`、`results/compare_migration_models.png`。

## 10. 下一步执行计划（文件与对比清单）

### 10.1 要创建的模块

| 文件 | 职责 |
|---|---|
| `migration_models.py` | 迁移算子：`build_hex_csr` / `cpu_csr_continuous` / `m0_source_deterministic` / `m1_spectral_gaussian`（numpy + torch 版）；支持批量平面 `(n_planes, rows, cols)` |
| `model.py` | 全模型 R-S-A 生命周期（NumPy 原型，RSA、连续采样），迁移 backend 可切换；`lifecycle_rng` 与 `migration_rng` 独立 |
| `metrics.py` | 算子级与整模型指标 + CI（bootstrap/百分位） |
| `run_chasing.py` | 子命令：`verify-deterministic` / `compare-operators` / `compare-full` / `html` / `chase` |
| `viz_html.py` | 时间轴滑块 HTML 热图（drive/WT/carrier） |

### 10.2 对比清单（明确“比什么”）

1. **确定性 CPU vs GPU**（natal 引擎）：CPU `csr` vs GPU `fft`；总种群/携带率/逐 deme 场；`max rel`（f32，目标 ≤1e-5）+ FFT 重复可复现；折线图。
2. **算子级随机**：CPU CSR vs M0 vs M1；per-deme 均值（中位/最大）、方差比（中位+5–95%）、**lag 平均协方差**、总量 mean/std、负值比例。
3. **跨引擎校验**：NumPy CPU CSR backend vs natal CPU CSR——确定性轨迹（小网格）与随机矩（均值/方差，R 次）。
4. **整模型随机**：CPU CSR vs M0/M1；覆盖曲线（≥阈值 deme 数，mean±95%CI）、前沿半径随时间、共存时长、固定概率、总种群；lifecycle 噪声共享（配对）。
5. **chasing**：homing 基线；成功/失败判据（§3）。

### 10.3 验收门槛

- **G1 确定性**：`max rel ≤ 1e-5`；FFT 重复逐位一致。
- **G2 M1 算子**：方差比中位 0.9–1.1；lag 协方差与 CPU 相对误差 20–30% 内；总量守恒；均值在 MC 误差内。
- **G3 整模型**：M0/M1 与 CPU 的覆盖曲线 CI 重叠、前沿速度无系统偏差；差异如实记录。
- **G4 chasing**：晚期共存 ≥ `N_coexist`、前沿严格增长、不全局固定。

### 10.4 执行顺序（2026-11-12 更新）

- **已完成**：三策略全模型数据比较（`compare_strategies.py`；small 15²/k7/avd0.8/R20 + large 31²/k11/avd1.5/R10；`scale=100`；mean±SD + 三策略 HTML）。
- **用户暂缓/当前不做**：① 算子单独抽模块（不必要）；② M1 边界/M4 精修（不在当前计划）。
- **下一步**：
  1. **NumPy 全模型 CPU CSR vs natal 引擎跨引擎校验**（确定性轨迹 + 随机矩），确认 CPU 参考的可信度。
  2. 若校验通过：用 **M0**（空间形态更接近 CPU）或 CPU CSR 做 chasing 初探（S1/S2）。
  3. 通用 `viz_html`/`run_chasing`（当前 HTML 已内联在 `compare_strategies.py`）。
  4. `chase` 实验（小范围 → 放大）。

## 11. 直接 stencil 随机迁移（方案 A，已选定）

- **背景**：完整 CSR（以及随机路径的 forward scratch + reverse CSR）内存为 $O(n\kappa^2)$，大 $n,\kappa$ 不可承受。
- **决策**：采用**直接 stencil + 方案 A**（逐边 counter-based RNG + 两遍重算），放弃方案 B（分块 stencil + 有界 scratch）。
- **目标语义**：与 CPU CSR 相同的源端 Beta + 目的地 Dirichlet；内存 $O(n+\kappa^2)$；无原子加、固定种子可复现；代价约 2× Gamma 抽样。
- **本轮口径**：`migration_rate=0.5`；对照目标是 **CPU CSR**；M0/M1/M2 的 GPU FFT 近似线暂停，不再扩展（仅保留历史结果）。
- **原型顺序**：NumPy 语义原型（Philox4x32-10 + Gamma）→ 独立 GPU 原型（3A，不接产品）→ 全模型小规模 → 最后接入 natal 产品。
- **执行计划**：见 `DirectStencilMigrationPlan.md`（策略、两遍算法、RNG、模板、GPU 实现、验证与开放问题）。
- **关系**：直接 stencil 是“精确随机 + 省内存”的另一条线；M0/M1/M2 是 FFT 近似线，当前不作为本计划对照。

