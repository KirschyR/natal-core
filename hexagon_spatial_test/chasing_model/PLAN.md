# Chasing Model 复现计划（natal 引擎 · RSA · 全随机）

> 状态：计划 v2（已按 evaluator §27 的 F1–F8 修订；待 §29 复核）
> 依据：用户 2026-11-09 决策；`AGENTS.md` / `quality_checks_spec*.md` / `docstring_spec.md` 为最高依据。
> 范围：`recon/hex-model`，本计划默认**不改 natal 产品代码**。

## 0. 目标与已冻结口径

- 在 natal 引擎的 `HexGrid` 上，用**完全随机**的年龄结构模型复现 **chasing**：驱动与野生型（WT）在空间上共存、前沿持续移动、不必然全局固定。
- 世代内事件顺序固定为 natal 的 **R-S-A → migration（tick 末尾）**（`rust/src/kernels/age_structured.rs` 的 reproduction→survival→aging，`rust/src/sessions/spatial.rs` 的 tick 末迁移）；**不再**逐位还原 MATLAB hex model。
- 首个驱动：**homing**。其他驱动（TARE 等）本轮不做；是否切换机制需用户批准（见 §2 fallback）。
- 随机采样：先使用 **连续采样**（`continuous_sampling=True`）。
- 规模：**小范围优先**；结果需有**带时间滑块的 HTML 热图**，可拖拽观察整个 grid 的热度随时间变化。
- `event-seq/` 仅为已完成的探索支线（结论：~40% 差异来自事件顺序，RSA 定案），不再使用，已 gitignore。
- `hexagon_spatial_test/hex-model-new/` 为重建 hex model 的参考，与 chasing 无关，已 gitignore。
- M5 径向翻译版保留为历史 golden；因转向 RSA 全随机 chasing，不再追求 MATLAB 逐位定量。

## 1. 模型定义（natal `SpatialPopulation`, `age_structured`, `HexGrid`）

- **年龄结构**：2 幼 + 5 雄 + 8 雌 → 共享年龄轴 `n_ages=10`、`new_adult_age=2`，雄龄 7–9 零填充。
- **初始态**：每 deme WT 平衡年龄向量（与既有 M5/event-seq 口径一致）。S1 起先做 **burn-in**，让 R-S-A 下 tick 边界的 age0 瞬态消失；**所有频率/总种群指标排除 age0**（卵/新生瞬态），避免把首 tick 瞬态当作 chasing 信号。
- **遗传（F4 钉死）**：**5 型论文 homing 映射**——物种等位 `[WT, Drive, Res]`（6 个 ztype），`HomingDrive` 用 `drive_conversion_rate`（=paper `dc`）与 `late_germline_resistance_formation_rate`（=paper `gcr`）；**carrier = `{Drive|Drive, WT|Drive, Drive|Res}`**（对应 M1 的 dd/dw/dr）。复用 M1 已验证的 5×5 子张量映射；**不采用** 3 型（`[WT, Drive]`）口径。若某实验令 `gcr=0`，`Res` 不可达但物种仍保留 3 等位（Z=6 ≤ 32）。
- **存活/老化**：
  - 雌 `[1, 1, 7/8, 6/7, 5/6, 4/5, 3/4, 2/3, 1/2, 0]`
  - 雄 `[1, 1, 4/5, 3/4, 2/3, 1/2, 0, 0, 0, 0]`
- **繁殖**：`eggs_per_female=40`、`sex_ratio=0.5`、`fixed_egg_count=True`、`sperm_displacement_rate=1.0`。
- **密度**：natal `beverton_holt` 校准；沿用 event-seq 的 RSA 口径（`λ=5`、`competition_strength=5`、`D = 40 + 5·(4/9)`；K/期望产卵按平衡态推导）。
- **迁移（F1）**：`build_gaussian_kernel("hex", size=7, mean_dispersal=avd)`；**所有随机实验一律走 CPU CSR**（丢界+行重归一化）。route A `migration_execution="fft"` **只支持确定性模型**；stochastic + route A 属引擎改动 → 须另行批准 + 根 `EVALUATE.md` + 全量门禁，**不作为本计划默认步骤**。
- **释放**：中心圆（`r=3`）内、双性 `Drive|Drive` 投入首个成虫龄；用声明式钩子（`Op.add`，可按周持续）。
- **随机（F2）**：`stochastic=True` + `continuous_sampling=True`；**`scaling_factor`（记 s）同时乘初始年龄向量与 `age_1_carrying_capacity`**（等价按 s 重推 K），使 `expected_comp = s·42.222`、`expected_survival_rate` 保持不变，动态按 s 等比缩放。结果 JSON 必须记录 `s`、`K`、`expected_comp`。**只放大初始态而不放大 K 会压低 recruitment，不能当作「只调信噪」**。

## 2. chasing 判据与指标

- **空间图**：每 k tick 记录 driver / WT(`ww`) / carrier 的 per-deme 频率（**排除 age0**）。
- **成功判据（F3，S1/S2 的决策门）**：
  1. 晚期（burn-in 后）WT 与 driver 的全局/局部频率均 > ε，且持续 ≥ `N_coexist`（默认 100）tick；
  2. carrier 前沿半径随时间**严格增长**（移动前沿）；
  3. 未全局固定（drive 与 WT 均未被清除）。
- **失败判据（F3）**：前沿半径连续 `N_stall`（默认 100）tick 不增；或 WT 只在驱动后方均匀残留、无滞后带；或全局固定/灭绝。
- **fallback（F3）**：若 homing + 连续采样在 S2 明确失败，则**缩小目标为「空间共存 + 移动前沿」并如实报告负结果**；是否切换机制（TARE/阈值/Allee/WT 源）**需用户批准**——本轮限定 homing，不擅自切换。
- **结局分类**：重复 N 次 → drive 固定 / WT 固定 / 持续共存；给固定概率与 95% 区间。
- **辅助**：T50/T90、WT 避难所空间分布、总种群（排除 age0）。

## 3. 实验阶梯（小范围优先）

| 步 | 内容 | 规模 |
|---|---|---|
| **S0 冒烟** | 引擎 API/语义/随机/钩子/HTML 最小跑通；确认 `stochastic + continuous_sampling` 与 CPU CSR | 15×15，几十 tick |
| **S1 基线** | homing 确定性 vs 随机基线，burn-in + release 小扫；应用 §2 成功/失败判据 | 41×41，~200–600 tick |
| **S2 chasing 搜索** | 扫 `dc`、`gcr`、fitness cost、`avd`、`scaling_factor`（按 F2 同步 K）、释放量；决策门 + fallback | 41×41 |
| **S3 重复统计** | 40–100 次，固定概率/共存时长/95% CI | 41×41 |
| **S4 可视化** | 带时间轴滑块的 HTML 热图（整 grid） | S2/S3 结果 |
| **S5 放大（可选）** | **101×101 CPU CSR**；route A `fft` 仅作**确定性对照**（不作 stochastic 默认路径） | 放大 |

## 4. 产物布局（均在 `hexagon_spatial_test/chasing_model/`）

```
hexagon_spatial_test/chasing_model/
  PLAN.md
  model.py          # 构建 natal 模型（结构/参数/钩子/迁移）
  run_chasing.py    # 实验阶梯与重复统计
  metrics.py        # 共存/前沿/结局指标
  viz_html.py       # 时间轴滑块 HTML 热图
  results/          # JSON（HTML/PNG 受全局 gitignore）
```

- **记录机制（F7）**：逐 tick `pop.run(1, record_every=0)` 步进，**只存 per-deme 频率数组**（约 `41×41×T` float，T=600 时 ~4 MB），不存全量 state；HTML 由这些数组生成。
- **结果格式（F8）**：统一 JSON，不使用 CSV（全局 `*.csv` 被忽略）；HTML/PNG 按全局规则不入库。

## 5. 风险与坑

1. `pop.run()` 默认 `record_every=1` 每 tick 存全量快照 → 长跑 OOM；必须 `record_every=0` 并逐 tick 只存频率。
2. 连续采样在 `n<=1` 时无噪声；要观察随机 chasing 需足够 `scaling_factor`，且须按 F2 **同步放大 K**。
3. 迁移边界：natal 为丢界+重归一化，非 MATLAB `replicate`；本路线已接受。
4. 随机性可复现：固定种子、记录重复数与区间；不得只报单次。
5. route A `fft` 只支持确定性；stochastic + route A 属引擎改动，须另行审批（F1）。
6. 若后续确需改 natal 产品代码（新增能力/科学语义），必须**另行获批**、默认关闭、走根 `EVALUATE.md` 与全量门禁 + 独立复核。

## 6. 审查流程

- 每个可复核增量在 `hexagon_spatial_test/EVALUATE.md`「主 agent 交接区」追加 §N；evaluator 在「evaluator 回执区」追加 §N+1。
- 本轮：§26 被 §27 判 NOT APPROVED（计划文本）；已按 F1–F8 修订并交 **§28**，等 **§29** 复核。复核通过后再进入 S0/S1。
- 默认不改产品代码；纯复现脚本按「局部修改」处理，仍提供自测证据、命令与残余风险。

## 7. 里程碑

- [ ] P0：`record_every=0` 修复、引擎随机 hex 冒烟、基线模型与参数固化
- [ ] S1：homing 确定性/随机基线对照（含 burn-in 与成功/失败判据）
- [ ] S2：chasing 参数搜索（寻找共存+移动前沿；含决策门/fallback）
- [ ] S3：重复统计与固定概率
- [ ] S4：时间轴 HTML 热图
- [ ] S5（可选）：101×101 CPU CSR（route A 仅确定性对照）
