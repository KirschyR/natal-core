# Chasing Model 复现计划（natal 引擎 · RSA · 全随机）

> 状态：计划草案（待独立 evaluator 评审）
> 依据：用户 2026-11-09 决策；`AGENTS.md` / `quality_checks_spec*.md` / `docstring_spec.md` 为最高依据。
> 范围：`recon/hex-model` @ `0906a5f`，本计划默认**不改 natal 产品代码**。

## 0. 目标与已冻结口径

- 在 natal 引擎的 `HexGrid` 上，用**完全随机**的年龄结构模型复现 **chasing**：驱动与野生型（WT）在空间上共存、前沿持续移动、不必然全局固定。
- 世代内事件顺序固定为 natal 的 **R-S-A**；**不再**逐位还原 MATLAB hex model。
- 首个驱动：**homing**。其他驱动（TARE 等）本轮不做。
- 随机采样：先使用 **连续采样**（`continuous_sampling=True`）。
- 规模：**小范围优先**；结果需有**带时间滑块的 HTML 热图**，可拖拽观察整个 grid 的热度随时间变化。
- `event-seq/` 仅为已完成的探索支线（结论：~40% 差异来自事件顺序，RSA 定案），不再使用，已 gitignore。
- `hexagon_spatial_test/hex-model-new/` 为重建 hex model 的参考，与 chasing 无关，已 gitignore。
- M5 径向翻译版保留为历史 golden；因转向 RSA 全随机 chasing，不再追求 MATLAB 逐位定量。

## 1. 模型定义（natal `SpatialPopulation`, `age_structured`, `HexGrid`）

- **年龄结构**：2 幼 + 5 雄 + 8 雌 → 共享年龄轴 `n_ages=10`、`new_adult_age=2`，雄龄 7–9 零填充。
- **初始态**：每 deme WT 平衡年龄向量（与既有 M5/event-seq 口径一致）。
- **遗传**：homing 5 型。直接复用 M1 已验证的 `offspring_tensor` 映射（`HomingDrive`，无抗性、无隐含代价；需要代价时显式加 `viability`/`fecundity`）。
- **存活/老化**：参考矩阵
  - 雌 `[1, 1, 7/8, 6/7, 5/6, 4/5, 3/4, 2/3, 1/2, 0]`
  - 雄 `[1, 1, 4/5, 3/4, 2/3, 1/2, 0, 0, 0, 0]`
- **繁殖**：`eggs_per_female=40`、`sex_ratio=0.5`、`fixed_egg_count=True`、`sperm_displacement_rate=1.0`。
- **密度**：natol `beverton_holt` 校准；沿用 event-seq 的 RSA 口径（`λ=5`、`competition_strength=5`、`D = 40 + 5·(4/9)`；K/期望产卵按平衡态推导）。
- **迁移**：`build_gaussian_kernel("hex", size=7, mean_dispersal=avd)`；小范围用 CSR 归一化边界（丢界+行重归一化），大范围再评估 route A `fft`。
- **释放**：中心圆（`r=3`）内、双性 `Drive|Drive` 投入首个成虫龄；用声明式钩子（`Op.add`，可按周持续）。
- **随机**：`stochastic=True` + `continuous_sampling=True`；用 `scaling_factor` 调节信噪（注意 `n<=1` 时连续采样退化为确定性均值）。

## 2. chasing 判据与指标

- **空间图**：每 tick（或每 k tick）记录 driver / WT(`ww`) / carrier 的 per-deme 频率。
- **共存**：晚期 WT 与 driver 均 > ε 的 deme 比例及其持续时间。
- **前沿移动**：`carrier > 阈值` 的最远半径随时间的变化。
- **结局分类**：重复 N 次 → drive 固定 / WT 固定 / 持续共存；给固定概率与 95% 区间。
- **辅助**：T50/T90、WT 避难所空间分布、总种群。

## 3. 实验阶梯（小范围优先）

| 步 | 内容 | 规模 |
|---|---|---|
| **S0 冒烟** | 引擎 API/语义/随机/钩子/HTML 最小跑通 | 15×15，几十 tick |
| **S1 基线** | homing 确定性 vs 随机，release 小扫描 | 41×41，~200–600 tick |
| **S2 chasing 搜索** | 扫 `dc`、fitness cost、`avd`、`scaling_factor`、释放量 | 41×41 |
| **S3 重复统计** | 40–100 次，固定概率/共存时长/95% CI | 41×41 |
| **S4 可视化** | 带时间轴滑块的 HTML 热图（整 grid） | S2/S3 结果 |
| **S5 放大（可选）** | 101×101 / route A GPU 稳健性 | 放大 |

## 4. 产物布局（均在 `hexagon_spatial_test/chasing_model/`）

```
hexagon_spatial_test/chasing_model/
  PLAN.md
  model.py          # 构建 natal 模型（结构/参数/钩子/迁移）
  run_chasing.py    # 实验阶梯与重复统计
  metrics.py        # 共存/前沿/结局指标
  viz_html.py       # 时间轴滑块 HTML 热图
  results/          # JSON / HTML（HTML/PNG 受全局 gitignore）
```

> 注意：仓库 `.gitignore` 全局忽略 `*.html`、`*.png`，故可视化结果默认**不入库**（生成在本地）。如需入库另行确认。

## 5. 风险与坑

1. `pop.run()` 默认 `record_every=1` 每 tick 存全量快照 → 长跑 OOM；必须 `record_every=0` 并按需另存。
2. 连续采样在 `n<=1` 时无噪声；要观察随机 chasing 需足够 `scaling_factor` 与计数。
3. 迁移边界：natal 为丢界+重归一化，非 MATLAB `replicate`；本路线已接受。
4. 随机性可复现：固定种子、记录重复数与区间；不得只报单次。
5. route A `fft` 与 hooks 组合少测；小范围先用 CSR。
6. 若后续确需改 natal 产品代码（新增能力/科学语义），必须**另行获批**、默认关闭、走根 `EVALUATE.md` 与全量门禁 + 独立复核。

## 6. 审查流程

- 每个可复核增量在 `hexagon_spatial_test/EVALUATE.md`「主 agent 交接区」追加 §N；evaluator 在「evaluator 回执区」追加 §N+1。
- 本轮先提交 **§26 = M5 收尾 + chasing 计划**，请 evaluator 做**计划评审**；实现后的结果另立交接。
- 默认不改产品代码；纯复现脚本按「局部修改」处理，仍提供自测证据、命令与残余风险。

## 7. 里程碑

- [ ] P0：`record_every=0` 修复、引擎随机 hex 冒烟、基线模型与参数固化
- [ ] S1：homing 确定性/随机基线对照
- [ ] S2：chasing 参数搜索（寻找共存+移动前沿）
- [ ] S3：重复统计与固定概率
- [ ] S4：时间轴 HTML 热图
- [ ] S5（可选）：放大/route A
