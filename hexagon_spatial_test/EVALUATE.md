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
| 最近回执 | §4（M2 §3）= **NOT APPROVED**（1-based 索引忠实性；已修复，见 §3.6） |
| 待回执 | §3 修复后复核 → 期望 §5 |
| 主 agent 处理 | M0 已闭环；M2 已按 §4.6 修复索引缺陷，`verify_m2.py` EXIT=0（12/12 ≤4.8e-12） |
| 待 evaluator 动作 | 复核 §4.6 修复目标与受影响检查（§4.2/§4.3/§4.4） |

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
- （后续更新）参考 ZIP 已解压到持久目录 `hexagon_spatial_test/Hex-model-main/parameter_sensitive/hex code上交版/`
  （gitignored）；`gen_golden_pde_reference.py` 默认读取该目录、不再每次解压到 /tmp，manifest 仍记录源 ZIP 的
  SHA-256（`e0e5cc…`），PDE golden 数组逐位不变。

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

### 1.6 审查后文档同步（主 agent，2026-09-28）

§2 = APPROVED。按 §2.3 / §2.8 处理（**仅文档**，未改 golden、复现脚本或任何产品代码）：

- 修正 `Hex_model_recon.md` §1.4 PDE `dt=0.001 → 1e-4`（与 `pde/main.m:24`、manifest 一致）。
- 在 `Hex_model_recon.md` §5 记录 M0 状态与「M2 前置口径」（场地/检查点、核有效扩散、方向命名、单位换算）。
- 更新本文件状态表与 `results/M0_golden_report.md` §7（独立审查结论）。
- evaluator 补强的 `repro/verify_goldens.py`（§0/0b）保留；复跑 `ALL HARD CHECKS PASSED`、EXIT=0。
- 以上为审查后的文档同步，不触发新一轮数值审查；如需可请 evaluator 复核该同步。

## §3 — M2：均匀 hex 波速/波形（natal 空间原语 vs MATLAB）

### 3.1 改动清单（均在 `hexagon_spatial_test/`，未触碰 natal-core）
- 新增 `repro/gen_golden_m2_wavespeed.m`：MATLAB 参考 golden（flat/junction × paper/code 口径，homing）。
- 新增 `repro/natal_hex_wavespeed.py`：用 natal `HexGrid` + `build_gaussian_kernel` 实现通用 hex 模型。
- 新增 `repro/compare_m2.py`：波速/方向/波形/核的对照与自测。
- 新增 `golden/hex_homing_wavespeed_m2.{mat,json}`、`results/{m2_natal_wavespeed.json,m2_hex_waveshape.npz,m2_wavespeed_compare.json,M2_wavespeed_report.md}`。
- 未改任何 natal 源文件；`gpu` feature 默认关闭；未 commit。

### 3.2 行为与原因
- M2 决断（响应 recon §2.6）：通用 hex 模型是**离散代、连续计数**更新，非 natal 分阶段生命周期，
  故采用「**自写参考实现 + natal 空间原语**」，不改引擎（引擎级接入属 M4）。
- 口径冻结（采纳 §2.8）：paper = flat 300×300、cp 50%/60%；code = flat 200×200、cp 40%/70%，junction L=600。
- 坐标映射：natal 核经列镜像即 MATLAB `get_mig_matrix25` 核（1e-16）；flat 沿列轴对镜像不变，junction 敏感。
- 扩散用 scipy 直接卷积（`mode='nearest'` = MATLAB `'replicate'`）；不用 FFT（引入负振铃会被 logistic 放大为 NaN）。

### 3.3 自测证据（命令与结果见 `results/M2_wavespeed_report.md` §4）
- 波速 **12/12 <5%**，最大 |相对误差| = 0.203%（`compare_m2.py` EXIT=0）。
- natal 核 vs MATLAB 核列镜像：`max ≤ 1.1e-16`。
- 方向：`flat×√3/2 / junction` = paper 0.993/0.998/0.997；code 0.988/0.989/0.979。
- 有效核均值：0.2065 / 0.950 / 1.988（标称 0.5/1/2）。
- 波形：`m2_hex_waveshape.npz`（12 例）；flat_paper 归一化波前宽度 0.0067/0.0167/0.0301（avd 0.5/1/2），
  与 PDE avd=0.5 绝对宽度同量级（≈2.0 cells vs ≈2.5 units）。

### 3.4 请 evaluator 核对点
1. `natal_hex_wavespeed.py` 是否忠实于参考 `main.m`：`renew` 公式、核（列镜像映射）、`imfilter` replicate
   边界、检查点/插值逻辑。特别请核 junctions 的行轴波与列镜像映射。
2. 12/12 <5%、核 1e-16 是否可独立复现（命令见报告 §4）。
3. 口径冻结、方向命名、有效核扩散的解读是否成立、有无过度解读。
4. 结论边界：本阶段**仅验证 natal 空间原语**，未宣称引擎级生命周期复现；波形仅定性。
5. flat 与 junction 端点对齐差异（0.01–0.2%）是否为可接受的插值/离散化差异。

### 3.5 残余风险
- 自写实现 ≠ 引擎生命周期（M4 决断范围）。
- hex 波形无独立 MATLAB golden（发布 `launcher_waveshape` 依赖 `main` 未返回的 `ret`）。
- 低 avd 离散化需在 M8 以有效扩散为横轴重读 Fig 3；flat/junction 端点差异待解释。

### 3.6 审查后修复（主 agent，响应 §4；仅复现脚本/数据，未改产品代码）

按 §4.6 修复 `natal_hex_wavespeed.py` 的 1-based 忠实性缺陷：
- junction 初值改为 `(j+1) >= 2*(i+1) - 0.4L`；`_run` 取样改 `cp-1`、`mid-1`；新增 `mround`
  （half-away-from-zero）替换 Python `round`。
- 重生成 `results/m2_natal_wavespeed.json`、`m2_hex_waveshape.npz`、`m2_wavespeed_compare.json`；
  同步修正报告 `results/M2_wavespeed_report.md` §3.2/§3.5/§5.1（并更正 §3.5「端点差异=插值敏感性」的错误归因）。

独立证据（本地运行）：
- `repro/verify_m2.py`（evaluator 留下的失败回归）：修复后 **EXIT=0**，12/12 |rel| ≤ **4.8e-12**。
- `repro/compare_m2.py`：**EXIT=0**，12/12 <5%。

请 evaluator 复核 §4.6 修复目标及受影响检查（§4.2/§4.3/§4.4），并给出结论。未改 §3.1–§3.5 的核心结论，
仅修正残余解释与数值精度；未改任何产品代码。

---

# evaluator 回执区（追加式；evaluator 写，主 agent 据此行动）

> 首个回执将对应 §1 交接（编号 §2）。

## §2 — M0 §1 独立审查回执（evaluator）

### 2.0 裁定

**APPROVED（M0 §1）**。审查范围：提交 `2a7123c` 及其后 4 个未提交小改动（`EVALUATE.md` §1、
`golden/pde_reference_manifest.json`、`repro/gen_golden_pde_reference.py`、`results/M0_golden_report.md`）。
风险分类：复现脚本/数据 = 局部修改；未触碰 natal 产品代码（`git status` 仅上述 4 个
`hexagon_spatial_test/` 文件）。核心 golden 数值、逐字性、PDE 提取、确定性均**独立复现通过**；
未发现过度解读。审查期间 evaluator **补强了 `repro/verify_goldens.py`**（见 §2.7），未改任何产品代码或 §1 正文。

前置声明：以下除注明「主 agent 自测」外，均为 evaluator 独立运行所得；MATLAB/py 命令见各条。

### 2.1 逐条独立核对（对应 §1.4 请求核对点）

| # | 核对点 | 结论 | 独立证据（命令/结果） |
|---|---|---|---|
| 1 | `repro/ref/` 与 ZIP 源逐字一致 | **PASS** | 见 §2.2 |
| 2 | PDE golden 提取正确性 | **PASS** | 见 §2.3 |
| 3 | hex golden 数值与确定性可复现 | **PASS** | 见 §2.4 |
| 4 | 方法/单位/坐标一致性 | **PASS（含 4 项已披露差异）** | 见 §2.5 |
| 5 | 结论边界（无过度解读） | **PASS** | 见 §2.6 |
| 6 | 自测脚本逻辑有效 | **PASS（1 处缺口已由 evaluator 补强）** | 见 §2.7 |

### 2.2 逐字性（独立核对 #1）

未信任主 agent 声明，直接以 ZIP 成员字节为基准计算 SHA-256，并与 golden JSON `reference_files` 比对：

- `repro/ref/{flat,junction}/{main.m,get_mig_matrix25.m,homing/drive_generator.m,homing/renew_function.m}`
  共 8 个文件，与 `hex code上交版.zip` 内
  `wave justification/{flat side direction,junction direction}/...` **逐字节相同，SHA-256 全部相等**。
- 命令：Python `zipfile.ZipFile(...).read(member)` vs `pathlib.read_bytes()`，逐文件 `==` 且
  `hashlib.sha256` 对比 golden JSON；输出 8/8 `identical=True`。
- 结论：§1.1「逐字复制」成立。`flat` 与 `junction` 的 `get_mig_matrix25.m` 本身同源同哈希（预期）。

### 2.3 PDE golden 提取正确性（独立核对 #2）

直接读 ZIP 原始 `.mat`（v5 波速用 `scipy.io.loadmat`，v7.3 波形用 `h5py`），逐数组与
`golden/pde_wavespeed_reference.npz` / `pde_waveshape_reference.npz` 比较：

- 波速：4 驱动 `retlist`（20 点，avd 0.1→2.0）与 npz **逐位相同**，全部严格单调递增、有限、正；
  `avdlist` 逐位相同。
- 波形：4 驱动 × avd∈{0.5,10} 的 `(xlist,ylist)` 与 npz **逐位相同**；`xlist` 严格递增；
  `ylist∈[0,1]`（cifab_10 max=0.9992）；点数 601/1601/2601 与 manifest 记录一致。
- manifest 逐文件 SHA-256：12 个 `.mat`（4 驱动 × {wavespeed, shape0.5, shape10}）与
  **ZIP 成员及持久解压目录均相等**；`source_sha256=e0e5cc…` 确为 **ZIP 本身**哈希（非解压目录）。
- 命令：`python` 脚本如上；输出 `exact_vs_npz=True` ×8、`ALL MATCH: True`。
- 备注（文档性，非本 golden 缺陷）：`Hex_model_recon.md` §1.4 写 PDE `dt=0.001`，而 `pde/main.m:24`
  与 manifest `model` 均为 `dt=1e-4`；以源码/manifest 为准，建议 M0/M2 同步 recon §1.4。

### 2.4 hex golden 可复现性与确定性（独立核对 #3）

在临时副本 `/tmp/hexeval/a/repro`（不动仓库文件）用
`/opt/matlab/bin/matlab -batch "addpath('/tmp/hexeval/a/repro'); gen_golden_hex_wavespeed"` 连跑两次：

- 两次 6 个速度值**逐位相同**，且与仓库 `golden/hex_homing_wavespeed_smallcase.json` **逐位相同**：
  flat `{0.082144957338308802, 0.48893138750491955, 1.0681417904840802}`、
  junction `{0.076423160330297127, 0.42341169311233984, 0.91627874966219225}`。
- 两次 `reference_files` 哈希与仓库一致。`elapsed_s` 每次不同（壁钟元数据，§2.7-BF3 说明其不影响判定）。

### 2.5 方法 · 单位 · 坐标一致性（独立核对 #4）

**与论文 / recon §1.3、§2.6 一致的部分（已核）**：通用 hex 模型 `data=data+renew(...)` 后逐基因型
`imfilter(...,'replicate','same')`；`renew` 递归 `d_ar=rct·λ/((λ−1)N+1)/N − ar·N`（代码变量名
`rcat`，M0 报告写 `rct`，仅命名差异）；`get_mig_matrix25` 为 **51×51、六边形度量
`d=sqrt(dx²+dy²−dx·dy)`、全核归一化、无 `d≤25` 截断**；`σ=avd/√(π/2)`；flat/junction harness
镜像发布 `launcher_speed.m`（flat `cp=0.4n/0.7n`；junction `cp=L/(2√3)+0.45L/+0.65L`）。

**evaluator 独立补齐的坐标等价证明**（§1.5 顺延到 M2 的 M0 计划项）：
natal `HexGrid` 度量 `dr²+dc²+dr·dc`（`topology.py:257-262`，`COS_OPPOSITE_ANGLE=−0.5`）与 MATLAB
`Δx²+Δy²−Δx·Δy` 在映射 `(dr,dc)=(Δx,−Δy)` 下对全部 51×51 偏移**完全相等**；`build_gaussian_kernel`
的 `mean_dispersal→sigma=avd/√(π/2)`（`topology.py:389-393`）与参考实现同一约定。命令：numpy 全网格
比较，输出 `metric identical ...: True`。→ 该 M0 验证项实际成立，主 agent 的顺延不构成缺陷。

**独立核对发现、报告未逐条点名的差异（均不改变 M0 结论，供 M2 处理）**：

- **(a) 场地/检查点口径三方不一**：论文正文（p.6）为 **300×300、检查点 50%/60%**；发布 flat
  `launcher_speed.m` 为 **200×200、40%/70%**；junction launcher `actual_length=600`；PDE
  `launcher_speed.m` 写 `n=300` 但随附 `.mat` 实际 `n=100/60/60/40`。M0 harness 取了小规模
  （flat m=60,n=200；junction L=100）。M0 报告 §4 已把「检查点间距」列为偏差来源，但未点明与论文
  正文口径不同。→ 属 M2 定量对照前必须冻结的口径项。
- **(b) 51×51 核未执行论文「最大 25 hexes」限制**：`get_mig_matrix25.m` 无 `d≤25` 掩膜，51×51
  方阵内六边形距离最大可达 `√1875≈43.3`（>25），共 **546/2601** 个格点超界；但这些权重在
  σ=avd/√(π/2)、avd≥0.5 时**可忽略**（avd=1 时超界总权重 ~1e-214），故对 M0 golden 无实际影响。
  报告已列「51×51 无截断 vs 59×59+d≤25」为版本二义。
- **(c) √3/2 换算方向**：M0 对 **flat** 乘 √3/2，与 `wavespeed_analyse.m:139-141` 及作者
  `homing_wavespeed_compare.m` 的**算术对象**（flat side 数据）一致；`flat` 的坐标映射
  `X=√3/2·I`（`draw_3d_plot.m`）也从几何上确认该因子。但 ZIP 内目录名
  `junction和flatside是反的！！！！！junction乘sqrt3除2` 与作者比较脚本的变量命名互换表明
  **方向标签存在歧义**。M0 报告已按「近似观察（非恒等）」陈述比值，未过度断言；建议 M2 明确方向命名。
- **(d) 低 avd 下离散核有效扩散远低于标称 avd**：`get_mig_matrix25` 自打印的实际核均值
  `sum(w·d)/sum(w)` 在 avd=0.25/0.5/1.0 时约 **2.1e-5 / 0.207 / 0.950**（MATLAB 独立复算）。
  故 avd=0.25 的 −75% 偏差除「有限域+检查点+固定核」外，主因是**离散化后核近乎 δ 函数**。
  报告未点名此点（§2.7-BF5）。属解释性遗珠，非 M0 缺陷（M0 不主张定量）。

### 2.6 结论边界（独立核对 #5）

逐项检查 `results/M0_golden_report.md` 与 §1：未发现过度解读。报告 §4 标题即「初步，非最终结论」，
明确「不能当作收敛后的定量结论；定量对照留待 M2」；方向比值出现在 §1.3/§4 且标注「观察」，
`verify_goldens.py` 第 4 节标注「report only」；`unit_note` 与 §3 只陈述换算事实。**PASS**。

### 2.7 自测脚本逻辑（独立核对 #6 + 竞态审查）

按 `adversarial-review` 流程执行 Bug-finder → Adversarial Defender → Referee（evaluator 裁定）。
Bug-finder 报 6 项；Defender 回 1 confirmed / 4 false-positive / 1 disputed；evaluator 复核如下：

- **BF-001（medium，confirmed）**：`gen_golden_pde_reference.py:142-143` 对缺失波形源 `continue`，
  `verify_goldens.py` 只遍历现有 `_ylist` 键 → **波形 golden 不完整仍全绿**。evaluator 已在隔离副本
  （删去 4 个 `cifab_*` 键）实际复现：原脚本报 `ALL HARD CHECKS PASSED`，仅检查 6/8 条剖面。
- BF-002/003/006（low，false-positive）：均为逐字第三方源码的潜在/壁钟问题，M0 harness 未触发，
  报告也未作对应断言；已见证（m=60 与 m=200 速度逐位相同；`elapsed_s` 不被任何检查消费）。
- BF-004（low）：`verify_goldens` 确未复算 provenance（Defender 以「文件被字节截断」反例不成立，
  但「删除键」一例成立）→ 已并入下面补强。
- BF-005（low，disputed）：数值子声明正确（见 §2.5-d），但将其定性为 M0「缺陷」过头——报告已把
  固定 51×51 核列为偏差来源。裁定为「有效观察 / 记录性遗留」，非缺陷。

**evaluator 补强（测试修改，允许范围）**：在 `repro/verify_goldens.py` 新增第 0/0b 节——
(a) 用 `hashlib` 复算 `repro/ref/` 8 个文件哈希并与 golden JSON `reference_files` 比对；
(b) 断言 golden 覆盖文档化网格（6 条 hex、4×2 波速数组、4×2 波形剖面），缺失即失败。

- 补强后对**完好 golden** 运行：`ALL HARD CHECKS PASSED`，EXIT=0（8/8 剖面）。
- 对**删除 cifab 的隔离副本**运行：`waveform profile set mismatch ... FAIL`，EXIT=1。
- 即 BF-001 缺口已由可运行测试封堵，且未弱化任何原有断言。

### 2.8 残余风险

1. 小规模域 + 51×51 核 + 口径不一（论文 300×300/50-60% vs 代码 200×200/40-70%）→ 定量对照
   必须在 M2 冻结口径后重做（§2.5-a）。
2. 低 avd（≲0.5）离散核有效扩散远低于标称 avd（§2.5-d）；M2 应同时报告核有效均值或改以等效
   扩散对照，避免把离散化伪影误读为收敛问题。
3. 方向标签歧义（§2.5-c）：M2 需明确 `flat/junction` 命名与 √3/2 归属。
4. PDE/hex 速度单位不同（PDE=domain units/time；hex=cells/generation）；M2 需给出显式换算。
5. 本次为局部修改，APPROVED 仅覆盖 M0 golden 与复现骨架；不构成对 M1/M2 数值语义的认可。

> 交付物说明：evaluator 实际运行命令见 §2.2–§2.7；主 agent 自测与独立审查已分别标注。
> 审查期间唯一改动文件为 `repro/verify_goldens.py`（补强测试），未改 `EVALUATE.md` §1 正文及任何产品代码。

## §4 — M2 §3 独立审查回执（evaluator）

### 4.0 裁定

**NOT APPROVED（M2 §3）**。审查范围：提交 `49bdae3`（M2）及 `272609b`（§1.6 文档同步）。风险分类：
局部（自写复现脚本 + 数据），未改产品代码。**核心数值结论（12/12 <5%）本身成立且已独立复现**；
但发现一处**在范围内的忠实性缺陷**：`natal_hex_wavespeed.py` 未忠实翻译参考 `main.m` 的 1-based
索引，报告对残差的解释错误，且「junction 逐位相同」属错误陈述。缺陷小、修复成本低，但按
`quality_checks_spec.md`「in-scope defect requires repair」给出 NOT APPROVED。

> 独立性声明：以下所有数值均由 evaluator 在隔离副本（`/tmp/hexeval/*`，不改仓库）运行；命令见各条。

### 4.1 逐条独立核对（对应 §3.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | `natal_hex_wavespeed.py` 是否忠实于参考 `main.m` | **FAIL**（见 §4.4，索引不忠实） |
| 2 | 12/12 <5%、核 1e-16 可独立复现 | **PASS**（见 §4.2/§4.3） |
| 3 | 口径冻结、方向命名、有效核扩散解读 | **PASS（1 处解释不成立，见 §4.5）** |
| 4 | 结论边界（仅验证空间原语） | **部分 PASS**（「junction 逐位相同」过度陈述） |
| 5 | flat/junction 端点差异是否为可接受插值差异 | **FAIL**：非插值差异，而是索引约定缺陷 |

### 4.2 MATLAB golden 可复现（独立核对 #2）

- 命令（不覆盖仓库）：`cp -r repro /tmp/hexeval/m2/ && matlab -batch "addpath('/tmp/hexeval/m2/repro'); gen_golden_m2_wavespeed"`。
- 结果：重生成的 12 个速度值（`flat/junction × paper/code × avd 0.5/1/2`）与
  `golden/hex_homing_wavespeed_m2.json` **逐位相同**（`bit=True` 12/12）；`reference_files` 相同。
- `kernel_mean_dist` = 0.206514 / 0.950012 / 1.988410，与 MATLAB `get_mig_matrix25` 自打印一致。

### 4.3 natal 实现复现（独立核对 #2）

- 命令：`cd /tmp/hexeval/m2b && .venv/bin/python repro/natal_hex_wavespeed.py`（约 6 min）。
- 结果：12 个 natal 速度与 `results/m2_natal_wavespeed.json` **逐位相同**；`kernel_check` 相同
  （列镜像 `max|Δ|` ≤ 1.1e-16，`embed_err`=1.14e-13）。
- `compare_m2.py` 在隔离目录运行：**12/12 <5%**，最大 |rel|=0.203%；方向比 paper 0.993/0.998/0.997、
  code 0.988/0.989/0.979（与报告一致）；EXIT=0。

### 4.4 关键发现：索引约定不忠实（独立核对 #1、#5）——阻塞项

**证据链**：

1. **init 与 MATLAB 真值不符**。以 MATLAB 复刻参考 `main.m:70-78` 的 junction 初值（L=300，1-based
   `j>=2i-0.4L`）得 release 格点数 **50922**；而 `natal_hex_wavespeed.py:185`
   （`if j >= 2*i - 2/5*L`，0-based）得 **51096**，逐格比较 `equal=False`。faithful 翻译
   （`j+1 >= 2*(i+1)-0.4L`）才等于 50922。命令：MATLAB 脚本 + numpy 比较（见执行记录）。
2. **checkpoint/mid 用 0-based 直接套用 MATLAB 1-based 整数**。`_run` 以
   `x[:, mid, cp]`（flat，:216-217）与 `x[:, cp, mid]`（junction，:200-201）取样，`cp` 直接来自
   `mode_params`（MATLAB 1-based），未减 1；`run_flat:175`/`run_junction:191` 的 `mid=round(·/2)`
   亦未按 1-based 语义转换。
3. **Python `round` 为 banker's rounding**：`round(693/2)=346`，而 MATLAB `round(693/2)=347`（n=693
   的 `junction_code` 命中此差异）。
4. 上述偏移**部分相互抵消**，故 junction_paper 恰好落在 ~1e-13（伪“逐位相同”），而 flat 与
   junction_code 落在 1e-4~2e-3。

**evaluator 的 faithful 1-based 参照实现**（仅改索引/初值约定，复用模块的 `renew`/`_diffuse`/核）：
flat 6 例与 junction_paper 3 例的 `|rel| ≤ 4.8e-12`；junction_code 在 init/cp/mid 三者**都**按
1-based 正确转换后（avd=2 实测）为 3.8e-12：

| 口径 | author `natal_hex_wavespeed.py` | faithful 1-based 参照 | MATLAB golden |
|---|---|---|---|
| flat_paper avd=2 | 2.129177（+1.9e-3） | 2.125093321155（−4.8e-12） | 2.125093321165 |
| flat_code avd=0.5 | 0.488975（+8.8e-5） | 0.488931387505（−3.4e-13） | 0.488931387505 |
| junction_paper avd=2 | 1.845530（−3.3e-12） | 1.845529606501（−3.3e-12） | 1.845529606507 |
| junction_code avd=2 | 1.862588（−4.6e-4） | 1.863452186897（−3.8e-12） | 1.863452186904 |

> junction_code 的 faithful 组合经单独探测（avd=2）确定为：**init 忠实 + `cp0=cp−1` +
> `mid0=round_half_away(n/2)−1=346`**；其余组合偏差 6.7e-6~1.4e-3（faithful 参照对 12 例的其余值
> 见 §4.4 说明；测量值均来自隔离运行）。

**结论**：§3.2「junction_paper ≈0」是误差抵消的产物；§5.1「junction 逐位相同」不成立
（junction_code 差 ~0.05%）；§5.1 将残余归因于「检查点插值对离散代对齐的敏感性」**错误**，
真正原因是 1-based/0-based 索引约定。

### 4.5 口径/方向/有效扩散解读（独立核对 #3）

- 口径冻结（paper vs code、junction L=300/600）与 §2.8 一致；`kernel_mean_dist` 正确暴露低 avd
  离散化（0.2065/0.950/1.988）。**PASS**。
- 方向命名与 √3/2 归属沿用发布代码，未过度断言。**PASS**。
- §3.4「低 avd 离散核是 Fig 3 低扩散段差异的**主要来源**」在本阶段无直接对照（M2 未与 PDE 比较），
  属合理但未证的解释，建议 M8 复核。**记录性**。

### 4.6 修复目标（最小、可复现）

1. 让 `natal_hex_wavespeed.py` 忠实翻译参考 `main.m` 的 1-based 语义：
   - `run_junction:185` 初值用 `j+1 >= 2*(i+1) - 2/5*L`；
   - `_run:200-201/216-217` 取样用 `cp-1`、`mid-1`（即把 MATLAB 1-based 索引转 0-based）；
   - `mid` 用 MATLAB round-half-away-from-zero（避免 n 为奇数时 banker's rounding）。
2. 修正 `results/M2_wavespeed_report.md` §3.2/§5.1：删除「junction 逐位相同」，改述残余差异为
   索引约定所致（修复后应为机器精度）。
3. **evaluator 已留下已运行且失败的回归测试**（修复后即通过）：

   - 文件：`hexagon_spatial_test/repro/verify_m2.py`（新建，不改产品代码）
   - 命令：`cd hexagon_spatial_test && ../.venv/bin/python repro/verify_m2.py`
   - 期望：全部 `|rel| ≤ 1e-9`、核列镜像 ≤1e-12，EXIT=0
   - 实际（当前）：**EXIT=1**，9/12 失败：flat_paper 1.48e-4/1.44e-4/1.92e-3，flat_code
     8.83e-5/1.87e-4/2.03e-3，junction_code 6.70e-6/5.44e-4/4.64e-4；junction_paper 与核为 OK。
   - 依据：模块 docstring 与 §3.4.1 要求「忠实于参考 main.m」；报告自身对 junction 声明「≈0」。
     5% 门限由 `compare_m2.py` 另行保证（已 12/12 通过），本测试只抓 <5% 掩盖的索引偏移。

### 4.7 残余风险

1. 修复后应重跑 `gen_golden_m2_wavespeed`（MATLAB 参考不变）与 `natal_hex_wavespeed.py`、
   `compare_m2.py`、`verify_m2.py`，并同步波形/方向/报告。
2. 波形（`m2_hex_waveshape.npz`）由含索引偏移的实现导出；修复后需重新生成并复核 §3.5 数值。
3. 自写实现 ≠ 引擎生命周期（M4 决断），以及上游 §2.8 的场地/口径/单位风险仍适用。
4. 本次为局部修改；NOT APPROVED 仅针对 M2 忠实性，不否定「natal 空间原语可复现该波速（<5%）」的结论。

> 审查期间新增文件：`repro/verify_m2.py`（失败回归目标）；未改 §1/§3 正文或任何产品代码。
