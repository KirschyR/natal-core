# EVALUATE.md — Hex 模型复现项目：agent ↔ evaluator 沟通通道

> 本文件是 `recon/hex-model` 分支上、`hexagon_spatial_test/` **复现项目**独立的交接/回执通道。
> 与仓库根目录的 `EVALUATE.md`（GPU 旁路主通道）**并行**：
> - **对 natal-core 产品代码的任何改动**仍走根 `EVALUATE.md` 与全量门禁；
> - 本文件只覆盖**复现脚本 / 数据 / 报告**（位于 `hexagon_spatial_test/`）。
>
> 双方都**只追加、不改写对方已写入的内容**。evaluator 职责见 `AGENTS.md`「验证时机与角色」。

---

## 规范化声明（2026-09-28，用户确认）

- **流程与 GPU 旁路一致**：主 agent 实现 → 基本自测 → 文档/stub 同步 → 高风险由独立 evaluator 复核（`adversarial-review`，数值按需 `numerical`）→ 门禁 → 交付说明（区分「自测」与「独立审查」）。
- **风险分级**按 `AGENTS.md`。复现脚本（不改 natal）属**局部代码修改**；涉及 natal 引擎的新增/改动属**高风险**，走根 `EVALUATE.md` 并全量门禁。
- **硬约束**：不得改 CPU 数值语义（`rust/src/kernels`、`rust/src/model`、`src/natal/contracts`、`rust/src/lib.rs`）；`phase0_baseline.py --check` 始终 bit-identical；`gpu` feature 默认关闭。
- **natal 改动边界**：仅「确有缺失且通用」的新增，默认关闭、独立复核；不得为单个复现项目特化。
- **产物位置**：复现脚本与产出的测试结果**都放在 `hexagon_spatial_test/` 下**（见 §1）。
- 未经用户明确要求，不 commit / push / 改 `.gitignore` / 新建无关 Markdown。

---

## 当前状态

| 项 | 值 |
|---|---|
| 分支 | `recon/hex-model` |
| 项目 | 复现 bioRxiv 2026 hex 基因驱动模型（首期：模块 1–3 波速/径向/线性） |
| 最近回执 | — |
| 待回执 | §1（M0 首份 golden）→ 期望 §2 |
| 主 agent 处理 | M0 首份 golden 已产出，交接区已追加 §1 |
| 待 evaluator 动作 | 独立核对 §1 的 golden、自测证据与结论边界 |

---

## 0. 一句话目标

独立核对：复现脚本是否正确调用 natal-core 已有计算方法、结果是否与 MATLAB 参考 / golden 一致、结论是否可复现；对 natal-core 的任何改动按高风险另行审查。

## 1. 审查范围与产物布局

- 分支 `recon/hex-model`；首期模块 1–3（波速/波形、径向、线性）。
- 目录（均在 `hexagon_spatial_test/` 下）：
  - `repro/` —— 复现脚本（调 natal-core；MATLAB wrapper）。
  - `golden/` —— 外部参考（MATLAB / ZIP 预计算 PDE）。
  - `results/` —— 脚本产出的结果、图表、报告。
- 计划与依据：`Hex_model_recon.md`、`hex_model_2026_paper/*`（只读来源，见 `.gitignore`）。
- 允许 evaluator 修改**测试/校验脚本**；不得改产品实现。发现产品缺陷时优先留下已运行且失败的回归测试。

## 2. 需求与约束来源

- `AGENTS.md`、`quality_checks_spec*.md`（最高依据）。
- `Hex_model_recon.md`（计划、冻结决策 §0.1、里程碑 M0–M8）。
- `hex_model_2026_paper/REPRODUCTION_NOTES.md`（缺失件与可行流水线）。
- 论文全文与图表索引：`hex_model_2026_paper/paper_fulltext.md`（p.n 引用）、`figures_and_tables.md`。
- 参考实现：`Hex-model-main/`（只读）。

## 3. 环境

- MATLAB：`/opt/matlab/bin/matlab`（R2026a Update 3；`matlab -batch "..."`）。
- Python：仓库 `.venv`（natal-core editable）；GPU：RTX 5090（共享，确定性为主）。
- 门禁（涉及 natal 改动时）：等价根 `EVALUATE.md` §4；纯脚本阶段用针对性数值对照。

## 4. 风险分类与验证时机

- **复现脚本（不改 natal）**：局部修改；针对性测试（vs MATLAB golden / 解析式）+ 数值验证；用户要求或存在不确定性时委派 evaluator。
- **涉及 natal 引擎**：高风险；必须独立 evaluator 复核，并走**根 `EVALUATE.md`** 与全量门禁；不得自审冒充批准。
- **文档/格式**：文档检查（准确性、链接、中英同步如涉及）。

## 5. 验证口径（按里程碑）

| 里程碑 | 验证口径 |
|---|---|
| M0 golden | MATLAB 自复现；`HexGrid` 度量与 MATLAB 距离公式坐标等价（手推偏移距离相等） |
| M1 遗传一步 | 与 `renew_function` 一步后代分布对照（相对误差档） |
| M2 波速/波形 | 与 MATLAB / PDE 参考对照；flat/junction 两方向 |
| M3 规模 | nnz / 内存 / 时间的可复现测量；超预算显式报错 |
| M5 径向/线性 | 与 Table 1/2 定性/量级对照 |
| 每轮 | 交付含：命令、期望、实测、残余风险；区分自测/独立审查 |

## 6. 交接格式

主 agent 在下方「主 agent 交接区」追加 **§N**：改动清单、行为与原因、自测证据（命令+结果）、请 evaluator 核对点、残余风险。
evaluator 在「evaluator 回执区」追加 **§N+1**：裁定（APPROVED / NOT APPROVED）、逐条发现、独立运行证据、残余风险。
结论冲突时以 `AGENTS.md` 与英文规范为准。

---

# 主 agent 交接区（追加式；主 agent 写，evaluator 据此审查）

> 首个交接将在 M0 完成后追加（编号从 §1 起）。

## §1 — M0：范围冻结与首份外部 golden（波速/波形）

### 1.1 改动清单（均在 `hexagon_spatial_test/`，未触碰 natal-core）
- 新增 `repro/`：`ref/{flat,junction}/`（自发布 ZIP 逐字复制的参考实现）、`gen_golden_hex_wavespeed.m`、
  `gen_golden_pde_reference.py`、`compare_goldens.py`、`verify_goldens.py`。
- 新增 `golden/`：`hex_homing_wavespeed_smallcase.{json,mat}`、`pde_wavespeed_reference.npz`、
  `pde_waveshape_reference.npz`、`pde_reference_manifest.json`。
- 新增 `results/`：`M0_golden_report.md`、`m0_fig3_wavespeed_compare.json`、`m0_figS5_waveshape_summary.json`。
- 未改任何 natal 源文件；`gpu` feature 默认关闭；未 commit。

### 1.2 行为与原因
- 目标 M0：产出可复跑的外部 golden 并建立 Fig 3/S5 数据基座，为 M2 波速/波形与 M1 遗传对照提供参照。
- 参考实现选用 ZIP 的 `wave justification/{flat,junction direction}`（论文 Fig S3/S5 实际实现、参数化、
  与预计算 PDE 同源），而非 prompt 提及的 `test_code/{hs,hv}`（后者 `m`/`n`/`return_list` 有缺陷且不可参数化）。
  已逐字复制并记录 SHA-256（见 golden JSON `reference_files`）。
- hex golden 为确定性：两次运行 6 个速度值逐位相同。
- PDE golden 直接来自 ZIP 预计算 `.mat`（v7.3 波形 / v5 波速），转换为 `.npz` + manifest。

### 1.3 自测证据（主 agent）
命令与结果见 `results/M0_golden_report.md` §5。摘要：
- `gen_golden_hex_wavespeed`：flat avd{0.25,0.5,1.0}→{0.082145,0.488931,1.068142}；
  junction→{0.076423,0.423412,0.916279}（cells/generation）。
- `verify_goldens.py`：硬检查全部 OK（有限/正/单调、PDE 波速单调、波形∈[0,1]）。
- 确定性：重跑 harness 速度值逐位一致。
- 方向交叉核对（观察）：`flat·√3/2 / junction` = 0.9309 / 1.0000 / 1.0096（avd=0.25/0.5/1.0）。

### 1.4 请 evaluator 核对点
1. `repro/ref/` 与 ZIP 源是否逐字一致（比对 golden JSON 的 SHA-256；ZIP 路径见 manifest `source_sha256`）。
2. PDE golden 的提取是否正确（字段、形状、单调性；可对照 ZIP 原始 `.mat`）。
3. 小规模 hex golden 的数值与确定性是否可独立复现（命令见报告 §5）。
4. 结论边界：报告已声明**小规模未收敛、不作定量复现声明**，请核对是否仍有过度解读。
5. 单位/方向换算（flat×√3/2）的表述是否恰当（已按“近似观察”而非“恒等”陈述）。

### 1.5 残余风险
- 小规模域 + 固定 51×51 核 → 低 avd 波速明显偏低，需 M2 以论文规模收敛。
- 核版本二义（51×51 无截断 vs 59×59+`d≤25`）；hex 波形未导出（参考 `main` 触发即 `return`）。
- `HexGrid` 与 MATLAB 六边形度量/旋转坐标的映射尚未核对（M0 计划项，顺延到 M2）。

---

# evaluator 回执区（追加式；evaluator 写，主 agent 据此行动）

> 首个回执将对应 §1 交接（编号 §2）。
