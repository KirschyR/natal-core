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
| 待回执 | — |
| 主 agent 处理 | 已建分支；规范声明与通道就绪；等待 M0 启动 |
| 待 evaluator 动作 | — |

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

---

# evaluator 回执区（追加式；evaluator 写，主 agent 据此行动）

> 首个回执将对应 §1 交接（编号 §2）。
