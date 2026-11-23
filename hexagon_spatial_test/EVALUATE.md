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
| 最近回执 | §41（A1 GPU M2 算子）= **APPROVED（含非阻断 findings）**；此前 §39/§29/§31/§33/§35/§37 均 APPROVED |
| 待回执 | §42（A1b：GPU 出境可行性）+ §42.6（rate=0.5 完整 M2）→ §43；§44（A2a：rate=0.5 全模型）→ §45 |
| 主 agent 处理 | §42/§42.6 完成 GPU 出境 + rate=0.5 完整 M2 一致性；§44 完成 rate=0.5 全模型 CPU/M0/M1/M2 一致性；下一步 A2b 生命周期等价、A3 块敏感性；产品化另走根 `EVALUATE.md` + 批准 |
| 待 evaluator 动作 | 复核 §42/§42.6/§44 |

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

### 3.7 M2 闭环（主 agent）

§5 = **APPROVED**，M2 完成。按 §5.4.1 更新 `repro/verify_m2.py` 的过时 docstring（仅措辞，断言未改）。
其余残余为上游/M4/M8 事项：§5.4.2（低 avd 解释留 M8 直接对照）、§5.4.3（自写实现 ≠ 引擎生命周期，M4）、
§5.4.4（本批准仅覆盖 M2 忠实性）。状态已同步：`Hex_model_recon.md` §5 M2、`results/M2_wavespeed_report.md` §6。

## §6 — M3（CPU 侧）：宽核/大规模 CSR 折叠实测

### 6.1 改动清单（均在 `hexagon_spatial_test/`，未触碰 natal-core）
- 新增 `repro/m3_scale_probe.py`（CPU 侧规模探针）。
- 新增 `results/m3_scale_data.json`、`results/M3_scale_report.md`。
- 未改任何 natal 源文件；`gpu` feature 默认关闭；未 commit。GPU 侧按用户指示待确认后另做。

### 6.2 行为与原因
- M3 目标（recon §5）：量化 CSR 折叠在宽核/大格点下的时间/内存，为 M4 决断。
- 测 `k∈{3,5,11,21,51}` × 场地 `{30²,60²,120²,300²}`（`sigma=1.5` 最坏情形）的 support/nnz/
  CSR 字节/fold 时间/build 时间/CPU tick；另测论文核（size 51, `mean_dispersal=avd`）@300²。
- 避免 O(n²) 邻接分配：kernel 模式下 `adjacency_dense=(1,1)`（读 `population.py:819-824` 确认）。
- 显式预算：CSR > 1.2 GB 的组合标记 skipped（不静默），仍报 fold/nnz/内存。

### 6.3 自测证据
- 命令：`cd hexagon_spatial_test && ../.venv/bin/python repro/m3_scale_probe.py`（~8 min，峰值 RSS 6.69 GB）。
- 关键实测：`CSR ≈ 16·nnz B`；fold ≈ 3.3–3.6e6 entries/s；300²×51² = 2.15e8 nnz / 3.44 GB / 60 s；
  论文核 avd=1.0 = 1.95e8 nnz / 3.12 GB / 56 s；CPU tick ≈ 6e-7 s/entry（300²×21² = 24.9 s/tick）。
- 完整表见 `results/M3_scale_report.md` §2/§3。

### 6.4 请 evaluator 核对点
1. `m3_scale_probe.py` 是否忠实调用 natal API（`fold_migration_csr`、`build_gaussian_kernel`、builder）；
   CSR 字节口径是否完整（indptr/dest/weights）。
2. 关键数值（fold 吞吐、nnz、CSR 字节、tick）可独立复现。
3. 结论边界：外推海南不可行为**线性外推**（非实测），请核对该推断的合理性与表述。
4. CPU 路径无内存预算守卫的观察是否正确（`migration_cache_bytes` 仅 GPU）。

### 6.5 残余风险
- 外推（海南 ~200 GB / ~64 min）未实测；age_structured 类放大未测。
- GPU 侧未做（待用户确认）：deterministic enable/显存、随机行宽 ≤32 拒绝。

### 6.6 审查后修复（主 agent，响应 §7；仅复现脚本/数据/报告，未改产品代码）

按 §7.5 修复：
- **F1**：`fold()` 与 builder 统一为 `kernel_include_center=False`（后者为 builder 默认）。核验：
  size=30,k=3 fold nnz 7744 → **6844**，与 builder 实际 nnz **一致**（`= 报告 nnz − n_demes`）。
- **F2**：`build_population` 改 `stochastic=False`（deterministic），重测 tick；报告标签随之成立。
- **F3**：报告区分 natal 浮点 underflow（σ=0.798 时 d≳30.8）与论文 `d≤25` 盘掩膜。
- 重跑 probe（~6.5 min）重生成 `results/m3_scale_data.json` 与 `M3_scale_report.md` §2/§3/§4。

修复后关键值：300²×51² = 2.15e8 nnz / 3.43 GB / 63.3 s；论文核 avd=1.0 = 1.95e8 nnz / 3.12 GB / 58.2 s；
确定性 tick 系数 ~2.7–3.0e-7 s/entry（300²×21² = 11.3 s/tick）；海南外推 ≈ 211 GB / ~65 min 折叠 /
~66 min/tick。结论方向（CPU 不可行、无预算守卫）不变。

请 evaluator 复核 §7.5 修复目标与受影响项（§7.2/§7.4）；未改产品代码。

### 6.7 M3 CPU 侧闭环（主 agent）

§8 = **APPROVED**，M3 **CPU 侧完成**。按 §8.4 在 `results/M3_scale_report.md` §4 注明「2.7–3.0e-7 s/entry」
为大核（`k≳21`）渐近带。M3 **整体仍为部分完成**：GPU 侧（deterministic `enable_gpu`/显存、随机行宽 ≤32 拒绝、
`phase0`）按用户指示 NOT CHECKED，待确认后另立。状态已同步 `Hex_model_recon.md` §5 M3。

## §9 — M3（GPU 侧）：空间 CUDA enable / 预算与行宽守卫

### 9.1 改动清单（均在 `hexagon_spatial_test/`，未触碰 natal-core 源码）
- 新增 `repro/m3_gpu_probe.py`；新增 `results/m3_gpu_data.json`、`results/M3_gpu_report.md`。
- 更新 `results/M3_scale_report.md` §6（GPU 侧指向新报告）。
- 构建带 GPU 的扩展：`.venv/bin/maturin develop --features "gpu,extension-module"`（写入 gitignored 的
  `_engine_rs`；未改源码）。运行需 `LD_LIBRARY_PATH` 含 `/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib`。
- 未 commit。

### 9.2 行为与原因
- 经**现有** session API `pop._rust_spatial_backend.enable_gpu()/gpu_status()/run_tick()`（与现有 tests 用法一致），
  不改引擎。
- 测：① deterministic enable 成功率与迁移缓存足迹；② stochastic `MAX_CSR_ROW=32` 行宽守卫；③ enable 期显存
  预算守卫（过预算须显式报错）；④ `phase0` bit-identical。

### 9.3 自测证据（命令见 `results/M3_gpu_report.md` §6；数据 `m3_gpu_data.json`）
- deterministic（30²–300²，k=5/11/21）：5/5 `disabled→enabled`；缓存 2.7–5105 MiB；GPU tick 0.0011–1.442 s
  （相对 CPU ~8–17×）。
- stochastic：`max_row` 24 → `tick_ok`；120、440 → **rejected**（`... exceeds the device scratch limit of 32`）。
- 预算：7 等位（Z=28）×200²×11 的 `enable_gpu` 抛 `RuntimeError: GPU memory budget exceeded: ... needs 30011 MiB,
  but only 19418 MiB is free`（显式、无回退）。
- `phase0_baseline --check`：all bit-identical，EXIT=0。
- 退出后显存回落（23961→23959 MiB）。

### 9.4 请 evaluator 核对点
1. probe 是否忠实调用现有 session API；`migration_cache_bytes` 的 Python 镜像是否与 Rust 公式一致
   （`4·nnz·A·Z² + 8·nnz·A·Z + 20·nnz + ...`）。
2. 三项守卫（enable 成功、行宽拒绝、预算报错）与 `phase0` 是否可独立复现。
3. 结论边界：GPU **正确性对照属 M7**，本阶段仅 enable/预算/守卫/时间。
4. 预算报错时 `free=19418` 低于 `nvidia-smi` 的 23961 的差额解释（state/sperm/上下文）是否合理。

### 9.5 残余风险
- GPU 正确性、随机统计等价、age_structured（年龄/性别放大）与 history/hook 未测；结论限本机 RTX 5090。
- 未测 MIG/多租户争用下的显存；预算守卫按实时空闲显存判定。

## §11 — M4 路线 A：原型验证与产品化设计（请复核）

> 说明：§10 是 M3 GPU 侧 §9 的回执，**未覆盖**本批 M4 路线 A 成果（此前尚未编号）。故此处补交 §11。

### 11.1 改动清单（均在 `hexagon_spatial_test/`，未触碰 natal-core）
- `repro/m4_routeA_prototype.py` + `results/M4_routeA_prototype_report.md`、`results/m4_routeA_prototype.json`。
- `repro/m4_template_build.py` + `results/m4_template_build.json`。
- `repro/m4_gpu_fft_prototype.py` + `results/m4_gpu_fft_prototype.json`（用 conda torch/cuFFT，非 natal；`.venv` 无 torch）。
- `results/M4_routeA_productization_design.md`（产品化设计提案）。
- `Hex_model_recon.md` §4「路线 A 设计」+ §5 M4 更新。
- 未改产品代码；未 commit。

### 11.2 行为与原因
- M4 路线 A =「模板/标签压缩（表示层）+ 归一化卷积 `Z=K'⊛m`（边界层）+ FFT（加速层）」的**可选、默认关闭**迁移路径；当前聚焦论文大核 → **FFT 优先**，小核直接 stencil 暂不实现。
- 原型仅验证「与现有 CSR 迁移算子等价 + 存储/构建/每步收益 + GPU cuFFT 可行性」，为 M4 选型提供依据。

### 11.3 自测证据（摘要；详情见各报告）
- **等价性**：模板/标签三元组与 CSR **逐条相同**（权重差 ≤1.3e-16）；归一化卷积（直接/FFT）vs CSR 逐格 ≤1.0e-15。
- **构建**：矩形解析模板 300²×51² **0.61 s vs fold 63.5 s（104×）**；存储 **58.7 MB vs CSR 3274 MB（~56×）**；行/目的地/权重一致（≤1.4e-16）。
- **GPU cuFFT**：归一化卷积 vs CPU 直接卷积，f32 ~2e-7、f64 ~1e-15；相对 CPU 直接卷积 **5–70×**。
- **每步成本**（300²，numpy 代理）：k≤5 直接 stencil 最优；k≥11 FFT 最优。

### 11.4 请 evaluator 核对点
1. **等价性推导**：`Z=K'⊛m`（`K'`=去中心核、`m`=域内指示）是否严格对应本仓库 CSR 的「丢界 + 邻域重归一化」（中心排除、每行权重和=1、源保留 `value−outbound`）；`apply_csr`（numpy `bincount` 代理）是否忠实于 Rust `migrate_csr_deterministic`（`rust/src/kernels/spatial.rs:461` 起，含 `stay_after` 分支）。
2. **模板构造**：解析签名 `min(r,R)/min(rows-1-r,R)/min(c,R)/min(cols-1-c,R)` 与 `normalize_coord`（`topology.py:103-120`）的边界规则一致；corners/edges 的行三元组与 CSR 逐条一致。
3. **GPU FFT**：线性卷积 padding/crop 无 circular wrap（原型曾有此 bug 已修）；f32/f64 误差与核对齐正确；CPU 参考与 GPU 同口径。
4. **结论边界**：CSR 计时为 numpy 代理（非 natal Rust；natal 实测见 M3）；构建时间结论仅覆盖**矩形域**；是否存在夸大。
5. （可选，非本批）§10.5 提到 M3「迁移缓存 delta 实测」可另补——与本 §11 无关。

### 11.5 残余风险
- 原型未覆盖：不规则 land mask 的签名分组、多类（性别/年龄/ztype）批处理、与 natal Rust 逐 tick 对拍、overlap-add 大域。
- 产品化实施属高风险，需用户批准 + 独立复核（设计提案见 `results/M4_routeA_productization_design.md`）。

### 11.6 审查后更正（主 agent，响应 §12；仅文档/措辞，未改产品代码或数据）

§12 = **APPROVED**（仅覆盖 §11 的原型/设计；引擎实现不在范围）。按 §12.6 处理：
- **F1（措辞）**：`results/M4_routeA_prototype_report.md` §5.3 已更正——M2 的 4.8e-12 只**佐证 natal 卷积核实现正确**；
  M2 用 `'replicate'`（Neumann），与本路线默认「丢界 + 重归一化」**边界语义不同**，不作直接互证。
- **F2（核验范围）**：`Hex_model_recon.md` §5 M4(1c) 与 `M4_routeA_productization_design.md` §3.1 已标注校验口径：
  小规模**全量**、300² 两例**抽样**（解析推导保证全体一致）。
- **F3（流程/文档）**：`.gitignore` 新增 `hexagon_spatial_test/research/` 系**用户**改动（非本批路线 A 改动）；
  §11.1 的「未 commit」为当时状态，其后路线 A 文件已由用户提交（`00bc4b3`/`9286570`/`c734f90`）。
- **F4（边界）**：等价性边界（`wrap`/`'replicate'`/不规则掩膜/多类）未验，已在 §11.5 与设计 §6 列出，落地时补。

范围：本批准**不含引擎实现**；路线 A 产品化仍需用户批准 + 独立复核 + 全量门禁。

## §13 — M4 路线 A 引擎实现：S0（cuFFT 绑定探针）+ S1（前端 FFT 计划）

> 用户已批准按设计 §9 的 GPU cuFFT 路径实施；本次交 S0+S1 复核。**高风险**（产品代码），后续 S2/S3 未做。

### 13.1 改动清单（产品代码）
- **S0（Rust，`gpu` feature 内）**：`rust/Cargo.toml` 给 `cudarc` 加 `cufft` feature（**不新增 crate**）；`rust/src/gpu/mod.rs` 导出 `pub mod cufft;`；新增 `rust/src/gpu/cufft.rs`（cuFFT r2c/c2r 线性卷积探针 + 单元测试）。**未接任何运行路径/契约。**
- **S1（Python 前端，默认关闭）**：`src/natal/frontend/spatial/migration.py` 新增 `MigrationFFTPlan` 与 `build_fft_migration_plan`（依赖无关：解析 `Z`、去中心核 `K'`）；新增 `tests/test_spatial_fft_plan.py`。
- 文档：`results/M4_routeA_productization_design.md` §9（S0–S4 计划 + S0 状态）、`Hex_model_recon.md` §5 M4。

### 13.2 行为与原因
- S0：启用 cuFFT 绑定（cudarc `cufft` feature，容器 `libcufft.so.12` 已备），隔离验证「cuFFT 线性卷积 = CPU 参照」；默认关闭（仅 `gpu` 构建）。
- S1：为路线 A 提供**符合 CSR 语义**的前端计划（`g=rate·f/Z; f_new=f(1-rate)+K'⊛g`）；`Z` 用向量化偏移扫描（解析），与本仓库 CSR 的「丢界+邻域重归一化」一致。**未改公开 API、未接运行路径**（S2 才接线）。

### 13.3 自测证据
- `cargo test`（默认）= **67 passed**；`cargo test --features gpu` = **261 passed**（含 `gpu::cufft::tests::cufft_linear_conv_matches_cpu`）。
- `scripts/phase0_baseline.py --check` = **all bit-identical，EXIT=0**。
- `ruff check .` = **All checks passed**；`pytest tests/test_spatial_fft_plan.py` = **21 passed**。
- `pyright` 对改动文件仅报**未改动**的 `normalize_migration_rate` 既有错误（新代码行无新增）；基线既有 1165 errors，本阶段未改任何既有 Python 源文件（除新增函数/文件）。
- S1 测试内容：`Z` 与逐 deme `normalize_coord` 暴力参考一致（wrap False/True）；`K'` 中心置零；`K'/Z` 与 `fold_migration_csr` 的 dest/权重逐条一致（wrap False/True）；偶核报错。

### 13.4 请 evaluator 核对点
1. **S1 等价性**：`build_fft_migration_plan` 的 `Z` 与 `K'/Z` 是否严格等价 CSR（含 `wrap`、中心排除、边界）；测试是否足以发现偏差（是否需补 `adjust_on_edge`、非矩形掩膜）。
2. **S0 特性启用**：给 `gpu` feature 加 `cudarc/cufft` 是否合规（无新 crate）、默认关闭、且**未影响非 gpu 构建与 `phase0`**。
3. **阶段边界**：S1 仅前端构建块、未接运行路径——是否符合「默认关闭、不改 CSR 行为」。
4. **pyright 基线判定**：改动文件仅报既有错误、无新增；该判定是否成立。

### 13.5 残余（未做）
- **S2**：Rust session/executor 接线（cuFFT 迁移内核 + 预算 + 参数传递）；**S3**：会话/`SpatialPopulation` 开关、中英文档、全量门禁。均未做、默认关闭。
- `wrap=True` 下「重复目的地」与 conv 的严格处理、`include_center=True`、不规则掩膜、多类批处理未验。

### 13.6 S2a 追加（前端开关；2026-09-29）

在 §13 基础上追加 S2a（仍在默认关闭、未接运行路径）：
- `src/natal/frontend/spatial/builder.py` 与 `population.py`：新增 `migration_execution: "csr"|"fft"`（默认 `"csr"`），并把它纳入 `SpatialInputs` 的归一化 migration 关键字，随 `build()` 传递。
- **`"fft"` 目前显式 `raise NotImplementedError`**（路线 A 的 executor/session 接线在 S2b），**不静默回退 CSR**。
- `tests/test_spatial_fft_plan.py` 增测：非法值报 `ValueError`；`"fft"` 报 `NotImplementedError`；默认 `"csr"` 正常构建。

验证：`pytest -q` = **3658 passed**（含 GPU 测试，需 `LD_LIBRARY_PATH` 含 nvidia cu13 lib，否则 GPU 用例因缺 `libnvrtc` 失败——环境项，与本改动无关）；`ruff` 通过；`pyright` 改动行无新增错误。S2b（executor cuFFT 内核 + 会话接线 + 预算）未做。

### 13.7 审查后修复（主 agent，响应 §14/§15；仅复现脚本/产品代码格式化与公开面收敛）

§14/§15 = **NOT APPROVED**（阻断：`cargo fmt --check`）。逐条修复：
- **F1（阻断，已修）**：`cd rust && cargo fmt` 仅格式化 `rust/src/gpu/cufft.rs`（10+5 行）。复跑
  `cd rust && cargo fmt -- --check` → **EXIT=0**；`scripts/check_rust.py` → **EXIT=0**
  （fmt/clippy/check/test --lib 67 全过；需 `PYO3_PYTHON/PYTHONHOME/PYTHONPATH` + `LD_LIBRARY_PATH` 含
  `/opt/conda/lib` 以加载 `libpython3.11`）。
- **F2（非阻断，已收敛）**：`migration.__all__` 不再导出 `MigrationFFTPlan`/`build_fft_migration_plan`
  （保留模块内可导入供测试/S2b；公开文档与 `migration_execution` kwarg 文档随 S3 补）。
- **F3（非阻断，已修）**：`test_fft_plan_weights_match_csr` 期望侧改为按 `plan.kernel>0` 过滤，与 CSR「跳过 0」一致。
- **F4（探针，预期）**：`cufft.rs` 为 S0 正确性探针（host 侧谱乘、f32 容差 1e-4），非缺陷。

复验：`pytest tests/test_spatial_fft_plan.py` = **24 passed**；`ruff` 通过；`pyright` 新行无报错（仅既有）；
`cargo fmt --check` = EXIT=0；`scripts/check_rust.py` = EXIT=0。请 evaluator 复跑 `scripts/check_rust.py` 与
受影响检查（预期转 APPROVED）。

### 13.8 S0+S1+S2a 闭环（主 agent）

§16 = **APPROVED**（M4 路线 A S0+S1+S2a 全部闭环；S2b/S3 不在范围）。后续进入 **S2b**（executor cuFFT
内核 + 会话接线 + 预算），再提交接送评。

## §17 — M4 路线 A：S2b 设备级 FFT 迁移内核（请复核）

### 17.1 改动清单（产品代码，`gpu` feature 内、默认不启用）
- 新增 `rust/src/gpu/fft_migrate.rs`：`gpu_fft_migrate(...)` —— 确定性空间迁移的**逐平面归一化卷积**（cuFFT）：
  NVRTC `scatter_g`（`g = rate·plane/Z`，零填充到 `(fr,fc)`）、`complex_mul`（乘预算核谱并折叠 `1/(fr·fc)`）、
  `combine`（`plane·(1−rate) + 裁剪后的卷积`）；外加单测 `fft_migrate_matches_csr`。
- `rust/src/gpu/mod.rs`：导出 `pub mod fft_migrate;`。
- 设计文档 `M4_routeA_productization_design.md` §10（S2b 方案 + 状态）、§9。
- **未接任何运行路径/会话/契约**；`migration_execution="fft"` 仍显式 `NotImplementedError`。

### 17.2 行为与依据
- 依据 §3.2c：确定性迁移 = 逐平面归一化卷积（ind0/ind1/sperm 各按性别·年龄 rate；virgin/stored 记账对 ind 平面抵消）。
- 布局：输入 batch-minor `(2,A,Z,B)`/`(A,Z,Z,B)`；`rate` deme-major `(n,2,A)`；`Z` 为 `K'⊛m`。
- 正确性对齐：与 `crate::kernels::spatial::migrate_csr_deterministic`（既有 CPU CSR 算子）逐元素比较。

### 17.3 自测证据（本机 RTX 5090）
- `cargo test --features gpu gpu::fft_migrate` → **1 passed**（`fft vs csr max relative error < 1e-4`）。
  调试中修正两处：信号不该被 `R` 偏移（否则与裁剪相消）、上传核须去中心（`K'`，中心由 `(1−rate)` 承担）。
- `cargo test`（默认）= **67 passed**；`cargo test --features gpu` = **262 passed**。
- `cargo clippy --features gpu -- -D warnings` = **PASS**；`cargo fmt -- --check` = **EXIT=0**；
  `scripts/check_rust.py` = **EXIT=0**；`scripts/phase0_baseline.py --check` = **bit-identical，EXIT=0**。

### 17.4 请 evaluator 核对点
1. **等价性**：`gpu_fft_migrate` 是否严格等价 `migrate_csr_deterministic`（确定性、含边界；布局/索引：ind 平面
   `(sex·A+age)·Z+z`、sperm 平面 `(age·Z+fz)·Z+mz`、`rate` 索引 `sex·A+age`）。
2. **数值对齐**：信号置左上 + 裁剪取 `R` 是否正确 'same'；核去中心、`Z` 口径是否与 CSR 一致；
   `1/(fr·fc)` 折叠是否恰当。
3. **边界与默认关闭**：该模块编译于 `gpu` feature、未被任何运行路径引用（grep 应无调用者）。
4. **未覆盖面**：随机路径、`wrap=True` 重复目的地、`include_center=True`、不规则掩膜、多类=更大 P、性能。

### 17.5 残余（未做）
- **S2c**：`GpuExecutor`/`SpatialSession`/`schema` 接线（复用 executor 的 stream/缓冲、预算、`enable_gpu` 选择）、
  `SpatialPopulation` 把 `MigrationFFTPlan` 传到 backend；**S3**：中英文档 + 全量门禁。均未做，默认关闭。
- `gpu_fft_migrate` 当前自建 context/stream（便于隔离测试），S2c 重构为复用执行器上下文。

### 17.6 审查后测试/文档补强（主 agent，响应 §18.4；仅测试/文档）
- **F1**：`fft_migrate_matches_csr` 现对每个 fixture 覆盖 `stay_after` **true 与 false**（运行时为 `true`）。
- **F2**：参数化多 fixture：`4×3 k=3`、`7×5 k=5`、`3×3 k=3`（边界主导）。
- **F3**：设计 §10 已标注「S2b 隔离版（逐平面 host 迭代 + 自建 context） vs S2c 目标版（`plan_many` batch=P + 设备谱乘 + 复用 executor）」。
- 复验：`cargo test --features gpu gpu::fft_migrate` = **1 passed**（多 fixture × 两种 `stay_after`）；
  `cargo fmt --check`、`clippy --features gpu -D warnings`、`scripts/check_rust.py` = EXIT=0；`phase0` bit-identical。

## §19 — M4 路线 A：S2c 会话/执行器/前端接线（请复核）

### 19.1 改动清单（产品代码；默认关闭，`"fft"` 需显式选择）
- `rust/src/gpu/fft_migrate.rs`：新增 `FftMigrator`（复用调用方 stream、双缓冲、`plane_base` 偏移）与 `MigrationPlanHost`；kernels 加 `plane_base`；`gpu_fft_migrate` 改为其 host 包装（测试用）。
- `rust/src/gpu/executor.rs`：新增 `fft: Option<FftMigrator>` 字段、`enable_fft_migration`（含显存预算）、`has_fft_migration`、`migrate_tick_fft`；`new`/`reconfigure` 置 `None`。
- `rust/src/sessions/spatial.rs`：`SpatialSession` 增 `migration_plan` 字段与 `from_parts` 可选参数；`enable_gpu` 有计划时构建 FFT 迁移器（否则原 CSR 预算）；`run_gpu_tick` 在「确定性且 FFT 就绪」时走 `migrate_tick_fft`。
- `src/natal/backends/rust/rust_backend.py`：`migration_plan` 透传；`src/natal/_engine_rs.pyi`：构造签名补该参数。
- `src/natal/frontend/spatial/population.py`：`migration_execution="fft"` 时构建 `MigrationFFTPlan` 并传 backend（**移除占位 `NotImplementedError`**）。
- 测试：`rust/tests/unit/gpu/executor.rs` 增 `fft_migration_tick_matches_host_reference`；`rust/tests/unit/gpu/spatial_session.rs` fixture 补 `migration_plan: None`；`tests/test_spatial_fft_plan.py` 增端到端用例。

### 19.2 行为与依据
- 依据 §3.2c（确定性迁移 = 逐平面归一化卷积）。`"fft"` 时 `GpuExecutor` 用 cuFFT 迁移器；`"csr"` 路径一行未改。
- 预算：有 FFT 计划时按 `O(P·(rows+2R)(cols+2R))` 量级计（`FftMigrator::footprint_bytes`），跳过 CSR 迁移缓存预算。

### 19.3 自测证据（本机 RTX 5090）
- `cargo test` 67；`cargo test --features gpu` **263 passed**（含 executor 级 `fft_migration_tick_matches_host_reference`）。
- Python 端到端（5×5,k=5,deterministic,3 ticks）GPU-FFT vs CPU-CSR：**max_rel 1.5e-7**；`tests/test_spatial_fft_plan.py` = **25 passed**。
- `cargo fmt --check`、`clippy --features gpu -D warnings`、`scripts/check_rust.py` = EXIT=0；`phase0` bit-identical。
- `pytest -q` = **3659 passed**；`ruff check src demos tests` = All checks passed；`pyright` = 基线 1165（**无新增**；`_engine_rs.pyi` 已同步签名）。

### 19.4 请 evaluator 核对点
1. **接线正确性**：`enable_gpu` 有计划时建 FFT 迁移器、`run_gpu_tick` 选择逻辑；`"csr"` 路径未变。
2. **数据传递**：Python `MigrationFFTPlan` → `(rows, cols, kernel, z)` → backend → `from_parts` → `MigrationPlanHost` → `FftMigrator`；类型/stub 一致。
3. **默认关闭**：不选 `"fft"` 时行为不变；`phase0` bit-identical。
4. **残余**：`"fft"` 仍折叠 CSR（构建墙仍在）；`fft`+`stochastic` 计划被忽略（未守卫）；hooks/history、`restore→run`、wrap/`include_center`/不规则/大域性能未测。

### 19.5 残余（未做，S3）
- 中英文档（`docs/{zh,en}`）；`"fft"` 跳过 CSR 折叠；`fft`+`stochastic` 守卫；hooks/history/restore 覆盖；大域与性能。

### 19.7 审查后修复（主 agent，响应 §20.5）
- **F1（中，首要残余，已修）**：`"fft"` + `stochastic` 原先被静默忽略。现两处显式守卫：
  - 前端 `population.py` 在 `"fft"` 构建时若任一 deme `stochastic` → `NotImplementedError`（fail-fast）；
  - 会话 `spatial.rs::enable_gpu` 若计划存在且 `blueprint.stochastic` → `PyValueError`（设备侧兜底）。
  - `builder.migration` 的 `migration_execution` docstring 注明「**deterministic-GPU-only**」。
  - 新增测试 `test_migration_execution_fft_rejects_stochastic`。
- **F3（低，已文档化）**：CPU / 未 `enable_gpu` 时 `"fft"` 仍走 CSR（与设计一致），已写入 `migration_execution` docstring。
- **F2（低，残余，留 S3）**：`"fft"` 仍先折叠 CSR；hooks/history、`restore→run`、wrap/`include_center`/不规则/大域性能未测。
- 复验：`pytest tests/test_spatial_fft_plan.py` = **26 passed**；`cargo fmt --check`、`clippy --features gpu -D warnings`、
  `cargo test --features gpu` = 263、`scripts/check_rust.py` = EXIT=0、`phase0` bit-identical、`ruff` 通过、`pyright` 基线 1165。

## §21 — M4 路线 A：S3 文档 / 消除构建墙 / CPU 守卫（请复核）

### 21.1 改动清单
- **文档（zh+en 同步）**：`docs/zh/4_simulation_engine.md` 与 `docs/en/4_simulation_engine.md` 新增 §11.3b「路线 A：FFT 迁移」（确定性 GPU、`enable_gpu` 必需、跳过 CSR 折叠、f32 容差、默认 `csr`）。
- **消除构建墙**：`src/natal/frontend/spatial/population.py` 在 `migration_execution="fft"` 时**不折叠 CSR**，改传空 CSR（满足冻结契约）。
- **CPU 守卫**：`rust/src/sessions/spatial.rs::run_inner` 在「有计划但未 `enable_gpu`」时显式 `PyValueError`（避免静默跳过迁移）。
- **测试**：`tests/test_spatial_fft_plan.py` 增「`fft` 构建后 `_migration_csr.dest_idx.size==0`」与「`fft` CPU 运行需 GPU」。

### 21.2 自测证据（本机 RTX 5090）
- e2e（5×5,k=5,deterministic,3 ticks）：GPU-FFT vs CPU-CSR **max_rel 1.5e-7**；`"fft"` 种群 `_migration_csr.dest_idx.size == 0`（CSR 已跳过）。
- `"fft"` 未 `enable_gpu()` 运行 → **显式报错**（`ValueError`）。
- `cargo test` 67 / `--features gpu` **263**；`cargo fmt --check`、`clippy --features gpu -D warnings`、`scripts/check_rust.py` = **EXIT=0**；`phase0` **bit-identical**。
- `pytest -q` = **3661 passed**；`tests/test_spatial_fft_plan.py` = **27 passed**；`ruff` 通过；`pyright` 基线 1165（无新增）。

### 21.3 请 evaluator 核对点
1. **跳过折叠**：空 CSR 是否满足冻结契约、`csr` 默认路径未变、`phase0` bit-identical；`"fft"` e2e 仍与 CSR 一致。
2. **CPU 守卫**：`run_inner` 在计划存在且无设备执行器时报错；不误伤 `"csr"`。
3. **文档**：zh/en §11.3b 是否同步、准确（确定性/`enable_gpu`/跳过折叠/容差/默认）。
4. **默认关闭**：不选 `"fft"` 行为不变。

### 21.4 残余
- `"fft"` 下 hooks/history、`restore→run`、`wrap=True`、`include_center=True`、不规则掩膜、大域/大核性能未测。
- `MigrationFFTPlan`/`build_fft_migration_plan` 仍非 `__all__` 导出，公开文档仅 `migration_execution`。

### 21.8 审查后修复（主 agent，响应 §22；默认构建显式拒绝）
- **根因**：CPU 守卫被 `#[cfg(feature="gpu")]` 门控，而默认构建无 `gpu` → `"fft"` 以空 CSR 在 CPU 静默 no-op。
- **修复 1**：`rust/src/sessions/spatial.rs::from_parts` 的 `#[cfg(not(feature="gpu"))]` 分支现当 `migration_plan.is_some()`
  → `PyValueError("migration_execution='fft' requires a gpu-enabled build")`。空 CSR 不再可能落到 CPU 静默运行。
- **修复 2（测试与能力对齐）**：`tests/test_spatial_fft_plan.py` 增 `_gpu_unavailable()`；`fft` 构建/e2e 用例在无 gpu
  扩展时 `skip`；`test_migration_execution_fft_requires_gpu_path` 接受「构建或运行」显式报错。
- **独立复现（默认构建）**：`unset PYTHONHOME PYO3_PYTHON PYTHONPATH && .venv/bin/maturin develop`（无 gpu）后：
  `migration_execution="fft"` 构建 → `ValueError: migration_execution='fft' requires a gpu-enabled build`；
  `pytest tests/test_spatial_fft_plan.py` = **25 passed, 2 skipped**（非 failed）。随后**已还原 gpu 构建**。
- **gpu 构建**：`pytest tests/test_spatial_fft_plan.py` = **27 passed**；`pytest -q` = 3661；`cargo test` 67 /
  `--features gpu` 263；`check_rust.py` EXIT=0；`phase0` bit-identical；`ruff` 通过；`pyright` 基线 1165。
- 文档 §11.3b「未启用即运行会显式报错」现于**默认构建**亦成立。

### 21.9 S3 / M4 路线 A 闭环（主 agent）

§23 = **APPROVED**。M4 路线 A（S0–S3）全部闭环：默认关闭的 `migration_execution="fft"` GPU cuFFT 迁移路径可用，
端到端与 CPU CSR 在 f32 容差内一致（~1.5e-7），消除 CSR 构建墙/内存墙，**默认（非 gpu）与 `--features gpu` 两种构建**
门禁均通过。残余（非阻断）见 §21.4/§23.3：`fft` 下 hooks/history/`restore→run`/`wrap`/`include_center`/不规则掩膜/
大域性能未测；`MigrationFFTPlan` 未公开导出。evaluator 建议 CI/交付同时覆盖两种构建。后续里程碑（M5/M7/M8）另立复核。

## §24 — M1：遗传系统一步后代对照（natal `offspring_tensor` vs 论文 `mats`）

### 24.1 改动清单（复现脚本/数据，未改产品代码）
- `repro/m1_offspring_check.py`；`results/M1_genetics_report.md`；`results/m1_offspring_check.json`。

### 24.2 行为与依据
- 论文 `drive_generator.m`（homing）把驱动烘焙成单张 `mats (gn², gn)`：`M[k,i,j]=mats[k*5+i,j]`。
- natal 编译后的 `offspring_tensor P[mother,father,offspring]`（驱动已入减数分裂/融合表，实测应用 `HomingDrive` 后
  表变化）。映射：`dd→Drive|Drive, dw→WT|Drive, ww→WT|WT, rw→WT|Res, dr→Drive|Res`（natal 另有 `Res|Res`，论文未用）。
- 比较 natal `P` 的 5 型子张量 `S[i,j,k]` 与 `M[k,i,j]`。

### 24.3 自测证据
- `python repro/m1_offspring_check.py`：dc=1.0 / 0.9 / 0.5 → **max_abs 差 = 0.0**（逐元素完全相等）；natal 全表行和=1。
- 即 natal 的分层遗传表示（meiosis + 修饰器 + `(Z,Z,Z)`）**精确表达论文 homing 驱动的一步后代算子**。

### 24.4 请 evaluator 核对点
1. `paper_homing_mats` 是否忠实翻译 `drive_generator.m`（含转置与 fitness cost）；基因型映射与 natal ztype 序是否正确。
2. 「精确相等」是否可信：natal `offspring_tensor` 是否确已烘焙 dc；5 型子集外（`Res|Res`）与列和不为 1 的说明是否恰当。
3. natal 应用 presets 的构建路径是否与论文同参（`dc`，无 resistance 形成、无 fitness cost）。

### 24.5 残余
- 仅 homing（一基因座 5 型）；TARE/TADE/CifAB/2-locus(25)/Wolbachia(2) 未逐一映射；`ddfitness`/性别特异未扫。
- 未做 natal `run` 一步新生儿分布 vs 论文 `renew_function` 的端到端对照（本阶段在**表级**验证等价）。

### 24.6 M1 闭环（主 agent）

§25 = **APPROVED**。M1（homing 一步算子）闭环。非阻断建议：F1 补 `herr≠0`（或不对称性别率）用例以钉住母/父轴序；
F2 报告统一标注 natal `(M,F,O)` 行和 vs 论文 `mats[(k,i),j]` 轴口径。后续进入 **M5**。

## §26 — M5 收尾 + Chasing 计划与骨架（请评审**计划**）

### 26.1 方向变更（用户 2026-11-09 决策）
- 世代内事件顺序固定为 natal 的 **R-S-A**；**不再**逐位还原 MATLAB hex model。
- `event-seq` 支线已验证：~40% 差异几乎全部来自 tick 内事件顺序；RSA 定案后该支线不再使用（已 gitignore）。
- chasing 采用**完全随机**模型，首个驱动 **homing**，先用**连续采样**，小范围优先；结果需带时间轴的 HTML 热图。
- `hexagon_spatial_test/hex-model-new/`（重建参考）与 chasing 无关，已 gitignore。
- M5 径向翻译版保留为历史 golden；因转向，**M5 不再送 MATLAB 定量复核**。

### 26.2 本次改动（均未触碰产品代码）
- `.gitignore`：按用户要求忽略 `event-seq/`、`geographic-score/`、`hexagon_spatial_test/hex-model-new/`。
- 新增 `hexagon_spatial_test/chasing_model/PLAN.md`（执行计划：模型定义、chasing 指标、S0–S5 阶梯、产物布局、风险、审查流程）。
- 更正记录：`results/M5_radial_report.md` §7 曾称「已在本文件追加 §26」，实际此前未写；本节即对 M5 的正式收尾。
- 待实现（P0）：恢复 `m5_engine_probe.py` 的 `record_every=0`（防长跑 OOM）；chasing 脚本/结果将置于 `hexagon_spatial_test/chasing_model/`。

### 26.3 请 evaluator 核对点（本轮=计划评审）
1. `PLAN.md` 是否覆盖：RSA、连续随机、homing、小范围优先、HTML 时间轴、指标/重复/种子策略。
2. 模型映射（年龄/存活/密度/homing `offspring_tensor`/迁移边界/释放）是否与既有 M1/M5/event-seq 结论自洽、有无过度承诺。
3. 产物布局与 `.gitignore` 改动是否合规；是否误伤应跟踪文件。
4. 高风险边界：本计划默认不改 natal 产品代码；若后续确需改引擎，是否明确要求另行批准 + 根 `EVALUATE.md` + 全量门禁。

### 26.4 残余
- homing + 连续采样能否产生 chasing（共存 + 移动前沿）尚未验证（S1/S2 待跑）。
- M6/M7/M8 与 500×500 规模化后置。

## §28 — 响应 §27：PLAN v2（F1–F8 修订，请复核）

### 28.1 改动清单（仅计划/文档，未触产品代码）
- `hexagon_spatial_test/chasing_model/PLAN.md` → **v2**，逐条闭环 §27：
  - **F1**：随机实验一律 **CPU CSR**；route A `fft` 仅确定性，stochastic+GPU 需另行批准；S5 改为 **101×101 CPU CSR**（route A 仅确定性对照）。
  - **F2**：`scaling_factor s` **同时乘初始年龄向量与 `age_1_carrying_capacity`**，使 `expected_comp=s·42.222`、`expected_survival_rate` 不变；结果 JSON 记录 `s/K/expected_comp`。
  - **F3**：S1/S2 增加成功/失败判据（共存 ≥ `N_coexist`、前沿严格增长、不全局固定）与 fallback（homing 失败则缩小目标并如实报告；切换机制需用户批准）。
  - **F4**：钉死 **5 型论文 homing**（等位 `[WT,Drive,Res]`，carrier=`{dd,dw,dr}`，复用 M1 映射），**不用** 3 型。
  - **F5**：事件顺序写「R-S-A → migration（tick 末尾）」。
  - **F6**：S1 起 **burn-in**；频率/总种群指标**排除 age0**。
  - **F7**：逐 tick `record_every=0` 只存 per-deme 频率（`~41×41×T` float），不存全量 state。
  - **F8**：结果统一 **JSON**，不用 CSV（全局忽略）。
- 本文件状态表同步。

### 28.2 自测/证据
- 本轮仅改 `PLAN.md` 与 `EVALUATE.md`（状态表 + §28）；`git diff --stat` 应只含二者，未触产品代码。
- 复核命令建议：
  `rg -n "CPU CSR|scaling_factor|N_coexist|burn-in|age0|R-S-A|JSON|F1|F2|F3|F4" hexagon_spatial_test/chasing_model/PLAN.md`
- 引擎事实沿用 §27.1 独立证据：`fft` 拒 stochastic（该测试 1 passed）；stochastic CSR 行宽上限 32 < size7 实测 48；15×15 stochastic+continuous CSR 可跑；`expected_comp=42.222`、`expected_survival_rate=1/90`。

### 28.3 请 evaluator 核对点
1. F1–F4 的修订是否闭环；F5–F8 是否补齐（对照 §27.2/§27.5）。
2. PLAN v2 是否仍与 M1/M5/event-seq 口径自洽、无过度承诺。
3. 是否可在批准后直接进入 P0/S0（小范围 CPU CSR）。

### 28.4 残余
- homing + 连续采样能否产生 chasing 仍未验证（S1/S2 待跑）；F3 已设决策门与 fallback。
- M6/M7/M8 与规模化后置。

### 28.5 追加（v3，用户新要求）：随机 route A 纳入范围

- 用户要求：随机空间模型也要能用 **GPU / route A** 加速（hex-recon 的核心目的）；并先验证**确定性**下 CPU CSR 与 GPU FFT 是否一致（简单例子 + 种群折线图）。
- `chasing_model/PLAN.md` 升 **v3**，新增：
  - **V1（低风险）**：确定性 CPU `migration_execution="csr"` vs GPU `"fft"`+`enable_gpu()` 小例对照 + 种群折线图；判据为 f32 相对误差（非逐位），并检查重复可复现。
  - **V2（先脚本原型，低风险）**：随机语义已定 = **卷积期望 + 目的地噪声**；先在 `chasing_model/` 用 GPU 脚本原型验证（不改产品代码），通过后再决定是否产品化。产品化属高风险：默认关闭、zh/en 文档、双构建门禁、`phase0` bit-identical、走**根 `EVALUATE.md`** + 独立复核。
  - **V3（中）**：CPU CSR 随机 vs GPU route A 随机的**分布**统计检验（均值/方差/分位数/CI），不要求逐位一致。
- 状态表已同步；V2 的批准与评审在根 `EVALUATE.md` 另立，本文件只记录复现计划。

## §30 — 确定性 route A 对照（CPU CSR vs GPU FFT，多配置）

### 30.1 执行内容
- 脚本：`hexagon_spatial_test/chasing_model/verify_routeA_deterministic.py`（由单例重写为**多配置矩阵**）。
- 模型：3 型 homing（等位 `WT/Drive`）、年龄结构（`n_ages=10, adult=2`）、`deterministic`、`record_every=0`。
- 初始/释放：uniform WT（**age0=0**；age1 与成虫用设计好的稳定数，**无 burn-in**）；中心圆（**radius 3**）内**仅雄性、纯合 `Drive|Drive`、首个成虫龄 age2**，每 deme `0.5 × age2 WT 雄数`，初始化时投放。与 event-seq 的空间释放模式一致（仅雄/中心圆/成虫龄/纯合 dd）。
- 每个配置用**同一初始态**分别跑：CPU `migration_execution="csr"`；GPU `"fft"` + `enable_gpu()`（route A cuFFT）；FFT 再跑一次验可复现。
- 配置与 tick：
  `a15_k7_avd0.8`(40)、`a21_k7_avd1.0`(60)、`a31_k7_avd1.0`(80)、`a21_k11_avd1.5`(60)。
- 环境：`.venv`（gpu 构建，RTX 5090）；`LD_LIBRARY_PATH` 含 cu13；运行约 30 s，**EXIT=0**。

### 30.1b 记录对象（RSA 口径）
- RSA 下 age0 只是**暂存槽**，本轮初始 **age0=0**，不再有参考卵池的 tick-0 瞬态；初始 `non_egg`（age≥1）≈ 2.1/deme、`adult`（age≥2）≈ 1.67/deme，与设计稳定态一致。
- 主对比指标仍用 `non_egg` 与 `adult`；`all_ages` 仅写入 JSON。此前「tick0 总数 40/deme 骤降」的问题已通过 age0=0 消除。

### 30.2 结果（同一初始态；CPU vs GPU）

| config | non-egg max rel | adult max rel | carrier max abs | final per-deme carrier abs | final state max abs | FFT repeat bit-identical |
|---|---|---|---|---|---|---|
| a15_k7_avd0.8 | 3.95e-7 | 4.72e-7 | 2.36e-7 | 3.08e-6 | 7.94e-7 | true |
| a21_k7_avd1.0 | 1.57e-7 | 1.94e-7 | 7.27e-8 | 6.05e-7 | 6.87e-7 | true |
| a31_k7_avd1.0 | 7.34e-8 | 9.84e-8 | 2.13e-6 | 3.29e-5 | 2.88e-5 | true |
| a21_k11_avd1.5 | 1.86e-7 | 2.13e-7 | 1.43e-7 | 1.62e-9 | 1.31e-7 | true |

- **全部通过**：`non-egg rel ≤ 4.0e-7`、`adult rel ≤ 4.8e-7`、`carrier abs ≤ 2.2e-6`、`final per-deme carrier abs ≤ 3.3e-5`、`final state abs ≤ 2.9e-5`；FFT 重复**逐位一致**。
- 近零元素的相对误差会被放大（a31 的 `final_state_max_rel_diff ≈ 1.99`，但绝对差仅 `5.1e-5`），故验收以**绝对差**为主，相对误差仅参考。

### 30.3 产物
- `hexagon_spatial_test/chasing_model/results/deterministic_compare/summary.json`
- `hexagon_spatial_test/chasing_model/results/deterministic_compare/{a15_k7_avd0.8,a21_k7_avd1.0,a31_k7_avd1.0,a21_k11_avd1.5}.png`
- `hexagon_spatial_test/chasing_model/results/html/deterministic_compare.html`（自包含：表格 + 轨迹/差异图；HTML 受全局 `*.html` ignore，不入库）

### 30.4 说明了什么
- **确定性下 route A（GPU cuFFT）与 CPU CSR 在 f32 精度内一致**（非逐位）；非卵/成虫轨迹相对误差 ≤4.8e-7，末态绝对 ≤2.9e-5；FFT 重复运行逐位可复现。
- 初始/释放已按 RSA 口径：age0=0、uniform WT、仅雄**纯合** `Drive|Drive`（中心圆 radius 3、0.5×age2 WT 雄）、无 burn-in；与 event-seq 的空间释放模式一致。
- 这为后续随机方案（M0/M1 的 GPU 卷积版）提供了可信的确定性底座；CPU CSR 继续作为随机参考。

### 30.5 请 evaluator 核对点
1. 配置/命令可独立复现，数值与 `summary.json` 一致。
2. 验收口径（绝对差为主、相对误差参考、FFT 重复逐位）是否恰当。
3. tick-0 age0 瞬态的解释与 non-egg/adult 口径是否成立。
4. 初始/释放口径（age0=0、仅雄纯合 `Drive|Drive`、中心圆 radius 3、无 burn-in）与 event-seq 的一致性是否记录清楚。
5. 结论边界：仅确定性 + 3 型最小模型；未覆盖 stochastic、hooks+fft、5 型/抗性、长跑/大规模；不构成对 chasing 随机结论的认可。


## §32 — 三策略随机模拟数据比较（CPU CSR / M0 / M1）

### 32.1 执行内容
- 脚本：`hexagon_spatial_test/chasing_model/compare_strategies.py`（**不拆算子模块**；三策略内联）。
- 全模型：R-S-A 年龄结构 hex（NumPy 原型，连续采样 stochastic）、homing 3 型、**age0=0**、uniform WT、仅雄纯合 `Drive|Drive`（中心圆 r=3，0.5×age2 WT 雄）、无 burn-in。
- 生命周期与迁移使用**独立 RNG 流**：三种策略共享同一 lifecycle 噪声，只替换迁移算子（配对比较）。
- 场景（类似确定性对照的小/大）：
  - `small`：15×15、kernel 7、avd 0.8、**100 tick**、R=20、scale=100；
  - `large`：31×31、kernel 11、avd 1.5、**100 tick**、R=10、scale=100。
- `scale=100` 同时放大初始态与 `DENSITY_REF`（否则每平面计数 ≤1，连续采样退化为均值，CPU≈M0 且无随机性）。
- **M1 可运行性 sanity**：非负截断 + 按平面总量重归一化（保持守恒；**不是** `RandomnessRealization.md` 的 M4 边界精修）。
- 运行约 **4m46s**，**EXIT=0**。
- 旧 V2a/M0-M1 算子级比较项归档到 `results/operator_baselines.json`；旧 `routeA_deterministic.*`、`stochastic_fft_operator.*`、`compare_migration_models.*` 已删除。

### 32.2 结果：终态饱和 + 瞬态差异

- **100 tick 时三策略都固定**：drive=1.0、cov≥0.1/0.5/0.9 = 全场、front = 最大（small 9.9 / large 21.2）；总量 small ≈47.1k、large ≈201–202k。终态指标已饱和，不能区分策略。
- 差异在**瞬态/到达时间**（`series.npz` 与 HTML 折线可见）：

**small（15×15, k7, avd0.8, R=20）**

| 策略 | drive@t40 | T50_selected | T50_cov(half) |
|---|---|---|---|
| cpu | 0.737±0.024 | 36.0±0.5 | 37.6±0.5 |
| m0 | 0.721±0.024 | 36.2±0.6 | 37.6±0.6 |
| m1 | 0.743±0.024 | 36.0±0.5 | 36.7±0.5 |

**large（31×31, k11, avd1.5, R=10）**

| 策略 | drive@t40 | T50_selected | T50_cov(half) |
|---|---|---|---|
| cpu | 0.342±0.021 | 44.2±0.4 | 47.4±0.5 |
| m0 | 0.333±0.028 | 44.4±0.7 | 47.5±0.5 |
| m1 | **0.238±0.033** | 45.0±0.7 | **45.3±0.7** |

### 32.3 说明了什么 / M1 差异原因

- **small**：三策略差异很小；**large**：M1 的 `drive@t40` 明显低（0.238 vs cpu 0.342），`T50_selected` 稍晚（45.0 vs 44.2），但 `T50_cov(half)` 反而更早（45.3 vs 47.4）——因为 M1 的场更**均匀**：很多 deme 较早跨过 0.5，但前沿/选定 deme 的峰值更低。
- **CPU 与 M0 接近**：两者共享源端 Beta 出境与确定性均值；M0 仅缺行内 Dirichlet 竞争，主要影响**方差**（算子级方差比 0.38），对均值/波前的改变较小。
- **M1 差异的可能原因**（按重要性）：
  1. **二阶矩近似**：M1 是协方差匹配的 Gaussian；CPU 是逐源 Dirichlet/多项，边际偏斜、逐源负相关；Gaussian 无法复现这些高阶/逐源结构。
  2. **均匀 N / circulant 核近似**：谱合成的协方差目标假设均匀 `N` 与平移不变核；全模型有非均匀 N、波前与边界，目标协方差失配（未做 M4 精修）。
  3. **截断 + 重归一化**：非负截断 + 按平面总量缩放会平滑/均匀化场；在低密度 deme 提前抬高覆盖率，同时压低前沿/选定 deme 的 drive。
  4. 迁移率=1 且噪声按平面独立加入，缺少 CPU 的「逐源守恒 + 行内竞争」相关结构。
- **结论**：100 tick 时三策略都固定，差异是**瞬态/到达时间**；M1 不能直接替代 CPU CSR；M0 更接近空间形态但方差不足。M4 精修按用户要求暂缓。

### 32.4 产物（v1，已归档）
- `results/strategies_v1/{small,large}/summary.json`（含 `timing` 字段）
- `results/strategies_v1/{small,large}/series.npz`
- `results/strategies_v1/{small,large}/selected_demes.png`
- `results/html/compare_strategies_v1_{small,large}.html`
- `results/operator_baselines.json`（归档的算子级比较项）
- 注：v1 使用旧的 5 个选定 deme；v2（M2 + 中心→边界 4 点）见 §34，v1 保留供参考。

### 32.5 请 evaluator 核对点
1. 脚本/配置可复现，`summary.json` 数值一致。
2. RNG 分离是否正确（lifecycle 共享、migration 独立）；配对比较是否成立。
3. M1 的「非负截断 + 重归一化」是否被清楚标注为 sanity（非 M4），是否影响结论解释。
4. **100 tick 终态饱和、差异在瞬态**的解读是否成立（T50/drive@t40 口径）。
5. 结论边界：NumPy 全模型生命周期**尚未**与 natal 引擎交叉校验；CPU 参考是 NumPy 复刻的 natal 连续 CSR，不是 natal 引擎本身；不构成对 natal/GPU 的认可。
6. 统计：R=20/10 较小，仅作描述性；未做显著性检验。

### 32.6 残余
- NumPy 全模型 vs natal 跨引擎校验未做（建议下一步）。
- M1 边界/非均匀未精修（用户暂缓）；截断+重归一化是权宜。
- `scale=100` 未做敏感性扫描；未测 chasing 指标（共存/前沿持续性）。
- 100 tick 下终态饱和；后续报告应使用 T50/瞬态指标而非终态。

### 32.7 计时（全部 tick × R 次重复；NumPy/CPU）

| config | strategy | wall s | s/replicate | migration-only s |
|---|---|---|---|---|
| small (R=20) | cpu | 46.06 | 2.303 | 45.02 |
| small (R=20) | m0 | 4.83 | 0.241 | 3.93 |
| small (R=20) | m1 | 8.62 | 0.431 | 7.67 |
| large (R=10) | cpu | 197.27 | 19.727 | 195.73 |
| large (R=10) | m0 | 5.90 | 0.590 | 4.59 |
| large (R=10) | m1 | 9.75 | 0.975 | 8.37 |

- 环境：本机 CPU（NumPy 实现）；**M0/M1 用 NumPy FFT，不是 GPU**。确定性 route A 的 GPU 计时是另一条线。
- CPU CSR 主导耗时（逐行循环）；M0/M1 的 FFT 实现快约 8–35×（large 约 20–33×）。
- HTML 已加入该计时表；`summary.json` 的 `timing` 字段含同数据。


## §34 — M2（局部块多项）+ 新 deme 选择；v1 归档

### 34.1 背景与变更
- 按用户要求：**不再考虑 GPU CSR**（内存/行宽在大核/上百万 deme 时不可行）。
- 新目标：在 FFT/卷积框架内同时保留**扩散波前**与**方差结构**。M0 只保均值/波前（方差扁），M1 只保二阶矩（场均匀、无波前），故尝试 **M2 局部块多项**。
- **展示 deme 调整**：从投放中心到边界沿一条线，均匀取 4 点（0、1/3、2/3、1）；小网格 `(7,7),(5,7),(2,7),(0,7)`，大网格 `(15,15),(10,15),(5,15),(0,15)`。
- **结果归档**：v1（CPU/M0/M1，旧 5 deme）已移到 `results/strategies_v1/` 与 `html/compare_strategies_v1_*`；本轮写入 `results/strategies_v2/` 与 `html/compare_strategies_v2_*`，不覆盖 v1。

### 34.2 M2 设计（脚本原型，未改产品代码）
每个成年平面：
1. 每源抽 Beta 出境 `O_s`，保留 `field − O`；
2. FFT 得 `μ = K' ⊛ (O/Z)`（给定抽样 O 的期望流入，带波前）；
3. 以核宽为块大小划分局部块；每块抽**独立连续 Poisson（Gamma）块总量** `T_b ~ Gamma(μ_b)`；
4. 块内按 `μ_d/μ_b` 做**连续多项**分配；
5. 将块总量按 `ΣO / ΣT_b` 缩放，使总量守恒（≈ 抽样出境总量）。
- 与 `RandomnessRealization.md` 的 M2 原文略有差异：原文先对全局 T 做块级多项再块内多项，组合后等价于**全局多项（M3）**；这里用**独立块总量 + 全局重归一化**，目的是保留**局域**负相关，同时守恒。
- 块大小 = 核宽（7 / 11）；块边界/大小仍是近似；未做 M4 边界精修。

### 34.3 结果（100 tick；选 4 个沿线 deme；drive@t40 均值 / SD 与 T50）

**large（31×31, k11, avd1.5, R=10）**

| 策略 | drive@t40（中心→边缘） | SD@t40 | T50 |
|---|---|---|---|
| cpu | 0.540, 0.415, 0.176, 0.060 | 0.033, 0.050, 0.035, 0.017 | 45.2±0.4 |
| m0 | 0.540, 0.413, 0.169, 0.056 | 0.044, 0.041, 0.022, 0.017 | 45.4±0.8 |
| m1 | 0.261, 0.241, 0.206, 0.225 | 0.047, 0.052, 0.034, 0.055 | 45.1±0.9 |
| **m2** | **0.549, 0.412, 0.151, 0.062** | **0.029, 0.049, 0.035, 0.021** | 45.7±0.8 |

**small（15×15, k7, avd0.8, R=20）**

| 策略 | drive@t40（中心→边缘） | SD@t40 | T50 |
|---|---|---|---|
| cpu | 0.818, 0.788, 0.663, 0.602 | 0.027, 0.027, 0.053, 0.051 | 36.5±0.6 |
| m0 | 0.803, 0.775, 0.646, 0.572 | 0.023, 0.032, 0.040, 0.046 | 36.7±0.7 |
| m1 | 0.751, 0.757, 0.727, 0.719 | 0.038, 0.031, 0.042, 0.046 | 36.4±0.6 |
| **m2** | **0.808, 0.763, 0.651, 0.566** | **0.029, 0.032, 0.032, 0.062** | 36.8±0.6 |

- **M2 同时保住了波前梯度与方差**：large 中心→边缘 `0.549→0.062`（CPU `0.540→0.060`），SD `0.029/0.049/0.035/0.021`（CPU `0.033/0.050/0.035/0.017`）。
- **M0**：梯度接近但 SD 多处偏离（如中心 0.044 vs CPU 0.033、中段 0.022 vs 0.035）。
- **M1**：几乎均匀（`0.261→0.225`），无波前；边缘 SD 明显偏高（0.055 vs 0.017）。
- 结论：**M2 是目前 FFT 框架下最有希望同时复现波前与方差的方法**；仍是近似（块边界/块大小）。

### 34.4 产物（v2，R=20/10，已归档）
- `results/strategies_v2_r20_r10/{small,large}/{summary.json,series.npz,selected_demes.png}`
- `results/html/compare_strategies_v2_r20r10_{small,large}.html`
- v1 归档：`results/strategies_v1/`、`results/html/compare_strategies_v1_*`
- v3（R=100 + C2ST）见 §36。

### 34.5 计时（全部 tick × R；NumPy/CPU）
| config | cpu | m0 | m1 | m2 |
|---|---|---|---|---|
| small (R20) | 45.56 s | 4.83 s | 8.63 s | 9.39 s |
| large (R10) | 195.15 s | 5.89 s | 9.63 s | 9.15 s |

### 34.6 请 evaluator 核对点
1. M2 的块总量方案（独立 Gamma + 全局重归一化）定义是否清晰、是否守恒；与文档 M2 的差异是否说明。
2. 新 deme 选择（中心→边界 4 点）与 v1 归档是否正确、未覆盖旧结果。
3. M2「同时保波前与方差」的结论是否被数据支持（均值梯度 + SD 列）。
4. 边界：NumPy/CPU、R 小、块大小未扫、未做跨引擎 natal 校验；M2 仍是近似。

### 34.7 残余
- 块大小/边界未调；未做 M4 精修。
- 未做 GPU 版（M2 的 FFT/块多项可后续上 GPU）。
- 未与 natal 引擎交叉校验。


## §36 — C2ST 相似性检验（R=100）+ 下三角热图

### 36.1 变更
- **v2（R=20/10）归档**为 `results/strategies_v2_r20_r10/`、`html/compare_strategies_v2_r20r10_*`。
- **v3**：`small`/`large` 均为 **R=100**；输出 `results/strategies_v3/`、`html/compare_strategies_v3_*`。
- 展示策略 **CPU / M0 / M2**（M1 仍计算并保存在 `summary.json`，仅不展示）；仍用中心→边界沿线的 4 个 deme。
- 计时表与 **C2ST 下三角 3×3 热图 + 表格**均写入 HTML（v3）。

### 36.2 C2ST 设计
- **样本单位**：每个 replicate 一个特征向量。
- **特征**：tick 25/50/75/100 的逐 deme drive、覆盖（3 阈值）、front、total；drive 的 time-mean/SD；T50。
- **分类器**：L2 逻辑回归（标准化），**5-fold × 5 repeats**；**置换零分布 20 次**；报告 accuracy / AUC / permutation p。
- **判读**：accuracy/AUC 越接近 **0.5** 越不可分；p 为置换检验（20 次 → 最小 0.048）。**效应量（accuracy/AUC）优先于 p**；需预设等价边界（如 acc ≤0.6）才谈“可合并”。

### 36.3 结果（R=100）

| pair | small acc / AUC / p | large acc / AUC / p |
|---|---|---|
| m0 vs cpu | 0.773 / 0.848 / 0.048 | 0.920 / 0.974 / 0.048 |
| **m2 vs cpu** | **0.581 / 0.618 / 0.048** | **0.591 / 0.604 / 0.048** |
| m2 vs m0 | 0.838 / 0.910 / 0.048 | 0.960 / 0.996 / 0.048 |

瞬态（drive@t40 沿线中心→边缘；SD；T50）：
- small：cpu `0.808,0.784,0.664,0.587`；SD `0.031,0.039,0.043,0.052`；T50 36.6。
  m0 `0.809,0.782,0.655,0.581`；SD `0.020,0.025,0.035,0.043`。
  m2 `0.811,0.780,0.648,0.570`；SD `0.035,0.037,0.053,0.063`。
- large：cpu `0.534,0.411,0.172,0.064`；SD `0.043,0.046,0.036,0.024`；T50 45.3。
  m0 `0.528,0.411,0.165,0.061`；SD `0.031,0.030,0.024,0.017`。
  m2 `0.530,0.401,0.172,0.060`；SD `0.050,0.046,0.038,0.026`。

计时（R=100）：small cpu 31.27 s / m0 3.39 / m1 6.04 / m2 6.60；large cpu 273.91 s / m0 8.56 / m1 13.77 / m2 13.12。

### 36.4 解释
- **M2 与 CPU 最接近**：C2ST accuracy ~0.58–0.59（AUC ~0.60–0.62），接近“不可分”边界。
- **M0 与 CPU 明显可分**：acc 0.77–0.92（AUC 0.85–0.97）。尽管均值接近，C2ST 主要抓到 **方差结构偏低**（M0 的 SD 明显小于 CPU）。
- **M2 与 M0 也可分**（0.84–0.96），说明两者分布差异大。
- **p 值注意**：20 次置换的分辨率下限为 0.048；即使 acc~0.59 也 p≈0.048。不能只看 p，应结合 accuracy/AUC 与预设等价边界。C2ST 检测的是“任何差异”，**可分 ≠ 实际不可合并**；最终仍需下游 chasing 指标的等价检验。

### 36.5 产物
- `results/strategies_v3/{small,large}/{summary.json, series.npz, c2st.json, c2st.png, selected_demes.png}`
- `results/html/compare_strategies_v3_{small,large}.html`（3 策略热图 + 滑块 + 沿线 4 点 mean±SD + 计时表 + C2ST 下三角热图/表；HTML 不入库）
- 归档：`results/strategies_v1/`、`results/strategies_v2_r20_r10/`

### 36.6 请 evaluator 核对点
1. C2ST 特征/分类器/CV/置换实现是否正确；R=100 是否确为两场景。
2. 下三角热图与表格数值是否一致；对角线/上三角是否正确屏蔽。
3. 结论边界：可分 ≠ 不可合并；p 分辨率（20 次）；特征选择/分类器敏感性；仅 NumPy/CPU；未做下游 chasing 等价检验。
4. v2 归档路径与 §34.4 是否一致。

### 36.7 残余
- 置换次数 20 → p 下限 0.048；可增至 100+。
- 特征集/分类器未做敏感性；可加 MMD/energy distance 作为无模型对照。
- 未做下游 chasing 指标（共存/前沿/固定概率）的等价检验。


### 36.8 补充说明（对角线 / p 值 / 计时口径，2026-11-16）
- **对角线**：此前错误地把 C2ST 矩阵对角线设为 1.0（显示约定，不是结果）。现改为**同种 self-split C2ST**：把同一方法的 R 次重复随机分成两半再比较，准确率应在 0.5 附近。实测对角线：small `0.458/0.432/0.538`，large `0.426/0.568/0.522`，与零假设一致。pairwise accuracy/AUC/p 不变。
- **p 值**：置换检验。零假设是两组方法标签可交换（同一分布）；打乱标签后重算 CV 准确率得到零分布，`p=(1+#{null≥obs})/(1+n_perm)`。`n_perm=20` → 最小 `1/21≈0.048`；全为 0.048 表示观测准确率高于**所有**零值，p 已到分辨率下限；要分辨更小的 p 需更多置换（如 999 次 → 0.001）。p 只表示“可检测”，效应量看 accuracy/AUC。
- **计时口径**：`wall_seconds` 是 `jobs=8` 的**并行墙钟**；`migration_seconds` 是**所有 replicate 的 migration 调用时间之和（CPU 时间）**。并行时墙钟 ≈ 总 CPU 时间 / workers，因此 **wall < migration 总和是正常的**。HTML 计时表已改为「wall s (parallel) / s per replicate / migration CPU-s (sum) / migration s per replicate」并加注。
- 热图深色格子的数字改为自适应浅色/深色，避免深底黑字看不清。


### 36.9 再澄清：计时口径与 p 的生物学含义

**计时（R=100，jobs=8；最后两列为 3 个种子的顺序探针）**

| config | strategy | par wall | par wall/reps | mig CPU sum | mig/reps | seq wall/rep | seq mig/rep |
|---|---|---|---|---|---|---|---|
| small | cpu | 31.27 | 0.313 | 235.61 | 2.356 | 2.304 | 2.254 |
| small | m0 | 3.39 | 0.034 | 21.17 | 0.212 | 0.241 | 0.196 |
| small | m1 | 6.04 | 0.060 | 41.34 | 0.413 | 0.426 | 0.379 |
| small | m2 | 6.60 | 0.066 | 45.46 | 0.455 | 0.464 | 0.416 |
| large | cpu | 273.91 | 2.739 | 2093.40 | 20.934 | 19.604 | 19.457 |
| large | m0 | 8.56 | 0.086 | 50.27 | 0.503 | 0.583 | 0.455 |
| large | m1 | 13.77 | 0.138 | 90.11 | 0.901 | 0.953 | 0.819 |
| large | m2 | 13.12 | 0.131 | 84.74 | 0.847 | 0.909 | 0.773 |

- 正确关系：`wall × jobs ≥ migration CPU-s (sum)`；`par wall/reps × jobs` 应和 `migration/reps` 比（small cpu 2.50 vs 2.36；large cpu 21.9 vs 20.9）。
- **不要把 `s/replicate × jobs` 与 `migration CPU-s (sum)` 比**：后者是 100 次重复之和，量级差 R 倍。之前的“wall < migration”正是这个口径混用。
- **顺序探针（同一时钟）确认 `seq wall ≥ seq mig`**：small cpu 2.304 ≥ 2.254；large cpu 19.604 ≥ 19.457；M0/M1/M2 亦然（见上表）。
- 顺序探针只取 3 个种子，`seq mig` 与并行跑的 `mig/reps` 有约 7% 的种子/运行差异，属正常。

**p 值的生物学含义**
- p 是置换检验的**统计可检测性**，不是效应量或生物学重要性。20 次置换的 p 下限是 0.048；全为 0.048 只说明观测准确率高于所有零值（p<~0.05），**无法**区分 p=0.01 与 p=0.049。
- 生物学判断应看：
  1. **效应量**：accuracy/AUC（M2 0.58–0.59 / AUC 0.60–0.62；M0 0.77–0.92 / AUC 0.85–0.97）；
  2. **下游 chasing 指标**（波前速度、覆盖率、共存时长、固定概率）与**预设生物学容差**的等价检验；
  3. **样本量**：R=100 时很小的差异也可被检测；p 随 R 变，不是生物常量。
- 结论：M2 在统计上仍“可检测”，但效应量小；是否“生物学可合并”取决于 chasing 指标是否在容差内。M0 的方差差异更大，可能具有生物学相关性。**非显著 p 也不证明等价**，需 TOST/等价检验。


### 36.10 HTML 计时/C2ST 展示精简（2026-11-16）
- HTML 计时表只保留**每 replicate 的实际时间**（jobs=8）：
  - `wall s / replicate = parallel wall / R`；
  - `migration s / replicate = migration CPU sum / (R × jobs)`；
  - 不再展示 sum/顺序探针列（仍保留在 `summary.json`）。
- 值（wall / migration，秒/replicate）：small cpu `0.313 / 0.295`，m0 `0.034 / 0.026`，m1 `0.060 / 0.052`，m2 `0.066 / 0.057`；large cpu `2.739 / 2.617`，m0 `0.086 / 0.063`，m1 `0.138 / 0.113`，m2 `0.131 / 0.106`。均满足 wall ≥ migration。
- C2ST 表**移除 permutation p**（p 仍是置换下限 0.048，非生物学量；仍保存在 `c2st.json`/`summary.json`）；HTML 只显示 accuracy/AUC + 下三角热图（对角线 self-split ≈0.5）。


### 36.11 HTML C2ST 展示：AUC 并入热图（2026-11-16）
- 下三角热图每格改为 **`accuracy (AUC)`**；颜色仍表示 accuracy；对角线为 self-split。
- 移除单独的 C2ST 表格；permutation p 仍保存在 `c2st.json` / `summary.json`。
- 示例：large `m2 vs cpu = 0.591 (0.604)`、`m0 vs cpu = 0.920 (0.974)`；small `m2 vs cpu = 0.581 (0.618)`。
- 代码：`compare_strategies_v3.py::_plot_c2st` 现在同时接收 `acc` 与 `auc`；HTML 只嵌入热图。


## §38 — A0：GPU M2 基线（CUDA 环境 + CPU FFT 基线，请复核）

### 38.1 改动清单（均未触碰 natal 产品代码）
- 新增 `hexagon_spatial_test/chasing_model/a0_baseline.py`：A0 基线脚本，`--mode cpu` 用仓库
  `.venv`，`--mode cuda` 用 `/opt/conda/bin/python`（两个环境分离，避免把 torch 装进项目环境）。
- 新增 `results/gpu_fft/a0_cpu.json`、`results/gpu_fft/a0_cuda.json`。
- 未覆盖 `strategies_v1`/`strategies_v2_r20_r10`/`strategies_v3`；未 commit/push。

### 38.2 执行内容
- CPU 基线：`a0_baseline.py --mode cpu`；M2 单 replicate，seeds `0/1/2`；配置
  `small = 15²/k7/avd0.8`、`large = 31²/k11/avd1.5`，`T=100`、`scale=100`（与 v3 一致）。
- CUDA 环境：`a0_baseline.py --mode cuda`；记录 torch/CUDA/cuFFT，做 f64/f32 roundtrip 精度检查，
  并测 batch 4/48 的 `rfft2 → irfft2` roundtrip wall（含 Python/launch，50 次平均）。
- 环境：`git_head = b4904e3`；conda Python 3.11.6、torch 2.12.0+cu130、CUDA 13.0；
  `.venv` Python 3.13.11。

### 38.3 结果

**CUDA / cuFFT**

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 D v2，cap (12, 0)，23975 MiB |
| torch / CUDA | 2.12.0+cu130 / 13.0 |
| roundtrip max abs（f64 / f32） | `1.33e-15` / `7.15e-7` |
| roundtrip wall（含 launch；batch 4 f32 / batch 48 f32 / batch 48 f64） | `0.016 / 0.016 / 0.036 ms` |

**CPU FFT M2 单 replicate（3 seeds 均值）**

| config | wall (s) | migration (s) | migration 占比 | final non-egg |
|---|---|---|---|---|
| small | 0.467（0.460–0.473） | 0.419（0.413–0.424） | 89.7% | ≈47.1k |
| large | 0.927（0.922–0.934） | 0.778（0.775–0.782） | 83.9% | ≈202.0k |

- CPU M2 的 wall 由迁移主导（84–90%）；cuFFT roundtrip wall（含 launch）只有 ~0.02 ms，
  真实 kernel 时间更小。注意这是 FFT 这一步，不是整个 M2 迁移；其余步骤仍占主导。
- 但 A0 尚未测 **CPU↔GPU 传输** 与 GPU 端 M2 块抽样；能否真正加速取决于这两项。A1 会单独计时，
  再决定是否需要更激进的批处理/驻留策略。
- 以上计时为当前共享机器上的单次测量，只用于量级对照，不作为验收硬阈值。

### 38.4 请 evaluator 核对点
1. 用两个环境分别执行 `--mode cpu` 与 `--mode cuda`，JSON 字段/量级可复现（计时允许机器差异）。
2. 确认 CPU 基线使用 v3 的 seeds/配置，且没有写入/覆盖 v1/v2/v3 归档。
3. 确认环境边界：`.venv` 未安装 torch，GPU 检查仅用 conda；未新增生产依赖。
4. `a0_baseline.py` 通过 `py_compile` 与 `ruff check`；无产品代码改动。
5. 结论边界：A0 只测环境和 CPU 基线，未实现 GPU M2、未测传输、未做分布等价；不构成对 GPU 加速的结论。

### 38.5 残余
- A1：实现 `gpu_migration.py` 的 GPU M2，并分别记录 FFT、块抽样、CPU↔GPU 传输耗时。
- A2：CPU/GPU M2 确定性均值场与随机矩等价性；A3：块大小/边界敏感性。
- 未做产品化；natal 的随机 GPU FFT 仍需后续单独设计、批准与根 `EVALUATE.md`。

### 38.6 响应 §39（主 agent）
- **F1**：已把 §38.3 的“纯变换耗时”改为“roundtrip wall（含 launch，50 次平均）”；结论仍成立且更保守。
- **F2**：`a0_baseline.py` 已增加 `git_dirty` 与 `script` 元数据（供后续运行使用）；A0 归档 JSON 保留原
  `git_head=b4904e3`，不复跑改写已复核的计时。
- **F3–F5**：接受；A0 计时仅作量级参考，不作为 CI/验收；A1 会分项计时（传输/FFT/块抽样）并进入 A2 等价性。

## §40 — A1：GPU M2 迁移算子（FFT + 块抽样，请复核）

### 40.1 改动清单（未触碰 natal 产品代码）
- 新增 `hexagon_spatial_test/chasing_model/gpu_migration.py`：纯 NumPy/torch 的 M2 迁移算子。
  - `M2GPU`：源端出境可选 CPU/GPU；FFT 均值用 cuFFT；块总量 Gamma + 块内连续多项可在 GPU 或 CPU；
  - `m2_reference_cpu`：与 GPU CPU-sampling 路径同 RNG 顺序的 NumPy 对照；
  - `--out` 默认写 `results/gpu_fft/a1_gpu_m2.json`。
- 新增 `results/gpu_fft/a1_gpu_m2.json`。
- 未覆盖 v1/v2/v3；未改产品代码/依赖；未 commit/push。
- 已响应 §39：§38.3 改为 roundtrip wall；`a0_baseline.py` 增加 `git_dirty`/`script` 元数据（见 §38.6）。

### 40.2 执行内容
- 环境：`/opt/conda/bin/python`；torch 2.12.0+cu130、CUDA 13.0、RTX 5090 D v2。
- 命令：`/opt/conda/bin/python chasing_model/gpu_migration.py`。
- 输入：合成成虫平面（`Z=3, S=2, A=10`，age 0/1 为 0，age 2–9 随机正数）；**不含生命周期**，
  只测迁移算子本身。
- 对照/路径：
  1. `outbound=cpu, sample=cpu`：与 `m2_reference_cpu` 同 seed/同 RNG 顺序，f64 与 f32；
  2. `outbound=gpu, sample=gpu`：完整 GPU 路径（出境+FFT+块抽样），f32，冷/热计时；
  3. `outbound=cpu, sample=gpu`：隔离 CPU 出境瓶颈。

### 40.3 结果

**等价性（CPU-sampling 路径）**

| case | f64 max abs | f32 max abs |
|---|---|---|
| small 15²/k7/avd0.8 | `1.07e-14` | `4.20e-6` |
| large 31²/k11/avd1.5 | `1.95e-14` | `6.37e-6` |

**完整 GPU 路径（warm）与 CPU M2 对照**

| case | full GPU ms/tick | CPU M2 ms/tick（A0） | 约加速 |
|---|---|---|---|
| small | `0.606` | `4.19` | `~6.9×` |
| large | `0.462` | `7.78` | `~16.8×` |

- full GPU warm 分解（ms/tick）：
  - small：transfer-in `0.078`、FFT `0.142`、sampling `0.341`、transfer-out `0.045`；
  - large：transfer-in `0.055`、FFT `0.070`、sampling `0.292`、transfer-out `0.044`。
- `outbound=cpu, sample=gpu`：small `2.01` ms/tick（其中 CPU 出境 `1.57`），large `3.89` ms/tick
  （CPU 出境 `3.40`）。→ **CPU 出境是剩余瓶颈；完整 GPU 出境把总时间降到 ~0.46–0.61 ms/tick**。
- A1 的 `_outbound_gpu` 对 `rate>=1` 走快速路径（直接返回源场），与 natal `sample_outbound` 的
  `rate>=1` 语义一致；因此 `rate=1` 时 GPU 出境不再产生 Beta 抽样开销。
- 守恒/有效性：adult in/out 差 small `5.7e-3`（相对 `1.1e-7`）、large `7.7e-3`（相对 `3.3e-8`）；
  最小值为 `0`（无负值）、全部有限；同 seed 完整 GPU 重复运行 **逐位一致**。
- 冷启动 transfer-in 可到 ~39 ms（CUDA 初始化），计时采用 warm 值；A0 的 CPU 值包含 100 tick，
  这里是单 tick 算子调用，两者相除得到上表加速比。

### 40.4 请 evaluator 核对点
1. 用 conda 复跑 `gpu_migration.py`；核对 f64/f32 CPU-sampling 最大差与 `a1_gpu_m2.json` 一致。
2. 核对 `m2_reference_cpu` 与 GPU CPU-sampling 路径的 RNG 顺序/块顺序一致；该对照是 v3 M2 数学的
   同序重写，不是直接调用 `v3.m2_migrate`（后者 RNG 交错顺序不同）。完整模型分布等价留 A2。
3. 核对 warm/cold 计时口径、分解项（transfer/FFT/sampling）和 CPU 出境瓶颈结论。
4. 确认未改产品代码/归档，未新增依赖；`gpu_migration.py` 通过 `py_compile`/`ruff`。
5. 结论边界：合成数据、单次算子调用；未含生命周期、未做 CPU/GPU 随机分布等价、未做块大小/边界扫描；
   不能据此宣称完整 chasing 已加速。

### 40.5 残余
- A2：接回生命周期，CPU M2 vs 完整 GPU M2 的确定性轨迹 + 随机矩（总非卵、逐 deme drive、方差比、
  lag 协方差），R=100–300。
- A3：块大小 `{5,7,9,11,15}` 与边界模式；默认块大小。
- natal 产品化（随机 GPU FFT 迁移）仍须单独设计、用户批准、根 `EVALUATE.md` 与全量门禁。



## §42 — A1b：GPU 源端出境可行性（方法切换影响 + 性能）

### 42.1 改动/产物（未触碰 natal 产品代码）
- 新增 `chasing_model/validate_gpu_outbound.py` 与 `results/gpu_fft/a2_outbound_validation.json`。
- `gpu_migration.py`：`_outbound_gpu` 增加 `rate >= 1` 快速路径（直接返回源场），与 natal
  `sample_outbound` 的 `rate >= 1` 语义一致；未改产品代码/依赖，未 commit/push。
- 响应 §41 findings：F1/F2/F3 在本验证中分别用**独立出境计时/50 次中位数/仅成人平面非负检查**处理；
  F4 的 `rate=1` 短路已补到 GPU 出境；F6 的本产物 JSON 记录 `git_head/git_dirty/script`；
  F5（3 型最小模型）、F7（A1 JSON 的 sample=cpu 计时字段）、F8（cold timing 措辞）留到 A2 统一。

### 42.2 方法
- 合成成人平面（`Z=3, S=2, A=10`），**不含生命周期**。
- outbound-only：rate `1.0`（当前模型）与 `0.5`（一般随机）；CPU `_continuous_binomial_vec` vs
  GPU `_outbound_gpu`；`R_outbound=400`。
- full M2：`cpu_cpu`（参考）、`gpu_gpu`（GPU 出境+FFT+块抽样）、`gpu_cpu`（GPU 出境 + CPU 块抽样）；
  `R_output=200`；比较逐 deme drive 代理、总 adult、成人最小值/有限性。
- 计时：warmup 后 `R_timing=50`，报告 median；CPU/GPU 出境单独计时。

### 42.3 结果

**出境分布**

| 项 | small | large |
|---|---|---|
| rate=1 同 seed max abs | `0` | `0` |
| rate=1 mean rel / var ratio / KS | `0 / 1.000 / 0` | `0 / 1.000 / 0` |
| rate=0.5 mean rel median（max） | `1.85e-2`（`0.21`） | `1.86e-2`（`0.25`） |
| rate=0.5 var ratio median（p05–p95） | `1.0003`（0.884–1.134） | `1.0001`（0.884–1.129） |
| rate=0.5 KS（p） | `0.0019`（0.879） | `0.0023`（0.651） |

- **rate=1 时 CPU/GPU 出境逐位相同**：现有连续二项在 `rate=1` 下 Beta 分量的 `Gamma(EPS)` 分量下溢，
  等价于“全部迁出”；这与 natal `sample_outbound` 的 `rate>=1` 直接返回一致。切换 GPU **不改变结果**。
- **rate=0.5 时分布一致**：均值差在 MC 噪声内，方差比 ~1，KS ~0.002 且 p 不显著；同 seed 数值不同
  只因 NumPy/torch RNG 流不同（同 seed max abs ~8.3 是独立的随机实现，不是偏差）。

**完整 M2 输出（rate=1）**

| 项 | small | large |
|---|---|---|
| gpu_gpu vs cpu_cpu drive mean diff median | `2.06e-3` | `1.94e-3` |
| gpu_gpu vs cpu_cpu drive var ratio median | `0.998` | `1.002` |
| gpu_cpu vs cpu_cpu drive mean diff median | `2.06e-3` | `2.07e-3` |
| total mean rel（gpu_gpu/gpu_cpu） | ~`0` | ~`0` |
| 成人最小值 / 有限 | >0 / true | >0 / true |

**计时（median，ms/call）**

| 路径 | small | large |
|---|---|---|
| CPU outbound-only（rate=1） | `1.43` | `3.19` |
| GPU outbound-only（rate=1） | `0.002` | `0.002` |
| CPU outbound-only（rate=0.5） | `1.32` | `2.79` |
| GPU outbound-only（rate=0.5） | `0.057` | `0.057` |
| full cpu_cpu | `6.99` | `10.20` |
| full gpu_gpu | `0.358` | `0.424` |
| full cpu-outbound / gpu-sampling | `1.90` | `3.82` |

- 结论：**源端出境可以放 GPU**。当前 `rate=1` 时结果无差异、出境阶段 ~700×（小）/ ~1600×（大）快，
  完整 M2 约 19×/24×；`rate<1` 时分布等价。A1 的 full GPU 数值在加入 `rate>=1` 快速路径后已更新，
  以本节的 50 次中位数为准。
- 边界：合成场、算子级；未接生命周期；不能据此宣称 chasing 结果已等价（A2）。

### 42.4 请 evaluator 核对点
1. 复跑 `validate_gpu_outbound.py`；确认 rate=1 逐位相同、rate=0.5 矩/KS、full M2 输出与计时。
2. 核对 rate=1 “全部迁出”解释与 natal `sample_outbound` 一致性；确认不是去掉本应存在的随机性。
3. 核对 adult-only 非负/有限检查、50 次 median 口径、CPU/GPU 出境独立计时。
4. 确认未改产品代码/归档；脚本通过 `py_compile`/`ruff`；JSON 元数据完整。

### 42.5 残余
- A2：接回生命周期，CPU M2 vs GPU M2 的确定性轨迹 + 随机矩（R=100–300）；同时统一 A1 JSON 的
  sample=cpu 计时字段和 cold/warm 标签。
- A3：块大小/边界；产品化随机 GPU FFT 仍须根 `EVALUATE.md` + 用户批准。


### 42.6 补充：rate=0.5 的完整 M2 一致性（GPU/CPU）

- `validate_gpu_outbound.py` 已加入 rate=0.5 的 CPU-sampling 等价与完整 M2 输出/计时。
- **CPU-sampling 等价**：f64 max abs small `1.07e-14`、large `1.24e-14`；f32 `2.99e-6`、`3.62e-6`。
- **完整 M2（full gpu_gpu vs cpu_cpu）**：
  - drive 均值差 median：small `1.77e-3`、large `1.81e-3`；
  - 方差比 median：small `1.010`、large `1.003`；
  - total 均值相对差：~`0`；`gpu_cpu` 同样接近（median `1.67e-3/1.80e-3`）。
- **完整 M2 计时（median）**：CPU/CPU rate=0.5 small `6.61 ms`、large `10.10 ms`；GPU/GPU rate=0.5
  small `0.415 ms`、large `0.559 ms`。
- 结论：rate=0.5 时 GPU 出境/块抽样与 CPU M2 的分布一致；出境不再是确定性退化，GPU 路径仍保持加速。
- 成人平面最小值 >0、有限；rate=0.5 的 outbound 分布见 §42.3。

## §44 — A2a：完整模型 migration_rate=0.5 一致性（CPU/M0/M1/M2）

### 44.1 背景与改动
- 确认：此前完整模型脚本 `compare_strategies_v3.py` 的 `MIGRATION_RATE=1.0`，CPU/M0/M1/M2 四条策略
  均以 rate=1 运行；归档 v1/v2/v3 不受影响。
- 改动：`run_replicate` 参数化为 `cfg["migration_rate"]`（缺省仍为 1.0，保证 v3 可复现）；新增
  `compare_rate05.py`，在 `results/strategies_rate05/` 用 `rate=0.5` 重跑：
  - small：15²/k7/avd0.8，R=20；
  - large：31²/k11/avd1.5，R=10。
- 未覆盖 v1/v2/v3；HTML 写在 `results/html/compare_strategies_v3_*_rate05.html`；未改产品代码。

### 44.2 结果

**drive@t40（沿线中心→边缘）**

| case | CPU | M0 | M1 | M2 |
|---|---|---|---|---|
| small | `0.904, 0.863, 0.575, 0.396` | `0.904, 0.856, 0.549, 0.372` | `0.848, 0.821, 0.694, 0.626` | `0.904, 0.846, 0.559, 0.380` |
| large | `0.754, 0.539, 0.095, 0.0055` | `0.735, 0.502, 0.094, 0.0045` | `0.382, 0.312, 0.197, 0.204` | `0.759, 0.529, 0.103, 0.0050` |

**与 CPU 的一致性**

| case | M2 median/max abs diff | M0 | M1 | M2 总 adult 相对差 |
|---|---|---|---|---|
| small | `1.62e-2 / 1.74e-2` | `1.56e-2 / 2.62e-2` | `8.74e-2 / 0.229` | `+0.09%` |
| large | `6.18e-3 / 9.91e-3` | `1.01e-2 / 3.62e-2` | `0.213 / 0.372` | `-0.01%` |

- T50（均值）：small CPU `36.3`、M0 `36.3`、M1 `35.65`、M2 `36.1`；
  large CPU `46.1`、M0 `46.4`、M1 `44.4`、M2 `46.5`。
- final front 与 coverage（0.1/0.5/0.9）：四策略相同（small 全场 225，large 全场 961）。
- **排序与 rate=1 一致**：M2 最接近 CPU，M0 次之，M1 明显偏离；M2/M0 的 T50 和总量仍在 CPU 邻域。

**rate=0.5 vs rate=1（CPU 参考）**

| case | drive@t40 变化（中心→边缘） | median/max abs diff | T50 | final total rel |
|---|---|---|---|---|
| small | `0.904,0.863,0.575,0.396` vs `0.808,0.784,0.664,0.587` | `9.26e-2 / 0.191` | `36.3` vs `36.6` | `+0.14%` |
| large | `0.754,0.539,0.095,0.0055` vs `0.534,0.411,0.172,0.064` | `0.103 / 0.220` | `46.1` vs `45.3` | `+0.14%` |

- 下调 rate 使波前**更陡**（中心更高、边缘更低），T50 仅小幅变化；总量/front/coverage 基本不变。
- 这与“rate 控制迁移量、但不改变最终占满格局”的预期一致；策略间一致性结论不因 rate=0.5 改变。
- C2ST（R=20/10）仅作探索，样本量不足以支撑等价/可区分结论，未用于本结论。

### 44.3 请 evaluator 核对点
1. 用 `.venv` 复跑 `compare_rate05.py`，核对 `results/strategies_rate05/consistency.json`、`series.npz` 与表值；
   确认 v1/v2/v3 未被覆盖。
2. 核对 `compare_strategies_v3.py` 的参数化缺省值仍为 1.0（v3 可复现），rate=0.5 只由新 config 触发。
3. 核对 drive@t40/T50/total/front/coverage 的口径；R=20/10 为描述性，不当作显著性或等价检验。
4. 结论边界：仍是 3 型、Poisson 生命周期的 NumPy 原型；未做 engine-matched 生命周期，也未把 GPU M2
   接进完整生命周期；不能据此宣称 natal/chasing 已等价。

### 44.4 残余
- A2b：接回生命周期（engine-matched `fixed_egg_count` + 分层抽样），CPU M2 vs GPU M2 的确定性轨迹/随机矩；
- A3：块大小/边界；产品化随机 GPU FFT 仍须根 `EVALUATE.md` + 用户批准。


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

## §5 — M2 §3.6 修复复核回执（evaluator）

### 5.0 裁定

**APPROVED（M2 §4.6 修复目标达成）**。修复提交 `f205d21` 仅动复现脚本/数据/报告，未触产品代码。
§4 的三项修复目标全部完成，且已由 evaluator 在隔离副本独立复现；受影响检查（§4.2/§4.3/§4.4）
复核通过。§3.6 声明的「12/12 |rel| ≤ 4.8e-12」「verify_m2 EXIT=0」属实。

### 5.1 §4.6 修复目标核对

| 目标 | 结果 | 独立证据 |
|---|---|---|
| 1. 1-based 索引忠实（init/`cp-1`/`mid-1`/half-away 取整） | **DONE** | 见 §5.2；`mround`、`(j+1)>=2(i+1)-0.4L`、`cp-1`/`mid-1` 已在代码中 |
| 2. 报告 §3.2/§3.5/§5.1 更正、删除错误归因 | **DONE** | §3.5/§5.1 已改述为索引约定（非插值敏感性）；旧「junction 逐位相同」已删 |
| 3. `verify_m2.py` 由失败转通过 | **DONE** | 独立运行 EXIT=0，12/12 |rel| ≤ 4.8e-12 |

### 5.2 独立运行证据（隔离副本 `/tmp/hexeval/m2c`，不改仓库）

- `matlab -batch "...gen_golden_m2_wavespeed"`：未受本次修复影响（`golden/hex_homing_wavespeed_m2.json`
  不在 `f205d21` 变更内）；§4.2 已见证其重跑与仓库逐位相同。
- `python repro/natal_hex_wavespeed.py`：修复后 12 个速度与提交的 `results/m2_natal_wavespeed.json`
  **逐位相同**；对 MATLAB golden 的最大 |rel| = **4.767e-12**（机器精度）。
- `python repro/verify_m2.py`：**EXIT=0**，12/12；`flat_paper 2.5e-13/1.8e-13/4.8e-12`、
  `flat_code 3.4e-13/1.4e-13/3.4e-12`、`junction_paper 1.4e-13/2.3e-13/3.3e-12`、
  `junction_code 1.6e-13/1.2e-13/3.8e-12`；核列镜像 ≤1.11e-16。
- `python repro/compare_m2.py`：**EXIT=0**，12/12 <5%；方向比 paper 0.9931/0.9978/0.9972、
  code 0.9882/0.9889/0.9787（不变，符合预期）。
- 波形 `m2_hex_waveshape.npz` 重生成值：flat_code 0.0151/0.0251/0.0503、flat_paper 0.0067/0.0167/0.0301、
  junction_paper 0.0042/0.0085/0.0169、junction_code 0.0021/0.0042/0.0095，与报告 §3.5 一致。

### 5.3 受影响检查复核（§4.2/§4.3/§4.4）

- §4.2 MATLAB golden：未变更、仍可逐位重生成。**PASS**
- §4.3 natal 复现：修复后仍可复现（隔离运行逐位一致），且误差由 2e-3 级降到 5e-12 级。**PASS**
- §4.4 缺陷：junction 初值、checkpoint/`mid` 取样、banker's rounding 三处已全部修正；
  `mround(346.5)=347`，`run_junction` 初值与 MATLAB 真值（release=50922）一致。**CLOSED**

### 5.4 残余 / 非阻塞

1. `repro/verify_m2.py:20` docstring 仍写「expected to FAIL until those offsets are fixed」，修复后已过时；
   仅注释措辞，建议后续顺手更新（不影响判定）。
2. §3.4「低 avd 离散核是 Fig 3 低扩散段差异的**主要来源**」仍属未直接对照的解释（M2 未与 PDE 比），
   已在上游 §4.5 记录，留待 M8。
3. 上游 §2.8 的场地/口径/单位风险、以及「自写实现 ≠ 引擎生命周期（M4）」仍适用。
4. APPROVED 仅覆盖 M2 §3 的忠实性修复；不构成对 M4 引擎接入或 M8 图形定量对照的认可。

> 审查期间未改任何文件；仅新增本回执。§4 的 NOT APPROVED 已由本次修复解除。

## §7 — M3（CPU 侧）§6 独立审查回执（evaluator）

### 7.0 裁定

**NOT APPROVED（M3 CPU 侧 §6）**。风险分类：局部（复现脚本 + 数据），未改产品代码。**M3 的核心结论
（CSR 内存墙、海南外推不可行、CPU 无预算守卫）经独立复现成立**；但 probe 的 CSR 折叠配置与被实际
构建/计时的 population 不一致，且报告有一处配置误标，属可修复的在范围内缺陷。GPU 侧按用户指示
延后（§6.5），本次**未审**。

> 独立性声明：以下数值由 evaluator 在隔离副本 `/tmp/hexeval/m3` 运行（不改仓库）；命令见 §7.3。

### 7.1 逐条独立核对（对应 §6.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | 是否忠实调用 natal API、CSR 字节口径 | **部分 FAIL**：API 调用正确；但 `fold()` 与 builder 配置不一致（§7.4-F1） |
| 2 | fold 吞吐、nnz、CSR 字节、tick 可独立复现 | **PASS（结构量逐位；时间 ~±25%）** |
| 3 | 海南线性外推的合理性与表述 | **PASS**（明示外推、算术正确） |
| 4 | CPU 路径无内存预算守卫 | **PASS**（已核 `ensure_migration_budget` 仅 GPU） |

### 7.2 API/字节口径核对

- `fold_migration_csr(..., mode="kernel")` 签名与调用一致（`migration.py`）；kernel 模式不读
  `adjacency_dense`，probe 传 `(1,1)` 与 builder 自身的 kernel 分支一致（`population.py:812-819`）。
- `MigrationCSR` 字段 `indptr(dest int64)`、`dest_idx(int64)`、`weights(float64)`（`migration.py:69-71`）
  → 字节 = `16·nnz + 8·(n_demes+1)`，与报告一致（逐行核验通过）。
- CPU 预算守卫：`ensure_migration_budget`/`migration_cache_bytes` 仅存在于
  `rust/src/sessions/spatial.rs`（GPU enable 分支，创建 `GpuContext` 后）与 `gpu/executor.rs`，
  用 `context.memory_info()`（设备内存）。CPU 路径确无对应守卫。**报告结论正确**。

### 7.3 独立复现证据

- `cd /tmp/hexeval/m3 && python repro/m3_scale_probe.py`（8m03s）：**20/20 组合的 support/nnz/csr_bytes
  与提交 JSON 逐位相同**；`16·nnz+8(n+1)` 公式逐行成立；论文核 support=859/2349/2601 一致；
  `peak_rss`=6.693 GB（提交 6.695）。fold 吞吐 3.552e6 entries/s（提交 3.554e6，报告 3.3–3.6e6）。
  `fold_s` 复核/提交比 0.97–1.25（机器负载差异，合理）；`tick_s` 亦在 ~±3% 内。
- 外推核验：`2599×2601=6.76e6`、r25 盘 1951 → nnz=1.32e10、CSR=16·nnz≈211 GB（报告 ~200 GB）、
  fold≈3.7e3 s≈62 min（报告 64 min）；tick 1.95e8×6e-7≈117 s（报告 ~2 min）。**算术正确、已明示外推**。

### 7.4 发现

- **F1（中）probe 的 fold 配置 ≠ 被计时的 population 配置**：`m3_scale_probe.py:59` 的
  `fold()` 传 `kernel_include_center=True`，而 `build_population` 走 builder 默认
  `kernel_include_center=False`（`builder.py:1567`、`population.py:697`）。故同一行的 `nnz`/`csr_gb`
  与实际 `build_s`/`tick_s` 所测 CSR 不是同一对象：真实 builder nnz = 报告 nnz − `n_demes`。
  - 实测（隔离）：size=30，k=3 → fold nnz 7744 vs **builder 实际 6844**（高估 11.6%）；
    k=51 → 774400 vs 773500（高估 0.12%）。`kernel_include_center=False` 在 `_kernel_row_entries`
    中确实剔除中心项（`migration.py:395`）。
  - 影响：k=3/5 行的 nnz/CSR 偏大约 11%/4%，相应 tick/entry 偏低；**对 k=51 与论文核（大核）结论
    无实质影响**（<0.2%），M4 内存墙/海南外推结论不变。
- **F2（低）配置误标**：报告 §1/§4 称「deterministic CPU tick」，但 `m3_scale_probe.py:79`
  为 `.setup(name="hex_deme", stochastic=True)`（builder 默认），tick 含随机抽样成本。标签与配置不符。
- **F3（低）措辞**：`M3_scale_report.md:48`「`d≤25` 之外 underflow 截断后 support=2349」把
  natal 的**浮点 underflow**（avd=1、σ=0.798 时阈值约 d≳30.8，并非 d=25）与论文的 `d≤25` 六边形盘
  掩膜混为一谈；natal `build_gaussian_kernel` 不做 `d≤25` 截断（recon §3/§2.8 已述）。数值 support=2349
  本身正确，仅解释性措辞有误。

### 7.5 修复目标（最小）

1. 令 `fold()` 与被计时 population 的 CSR 口径一致：`m3_scale_probe.py:59`
   `kernel_include_center=False`（与 builder 默认一致），或反之在
   `build_population` 的 `.migration(..., kernel_include_center=True)` 显式打开；二者取一并重跑。
   预期：k=3/5 行 nnz 减少 `n_demes`，大核行基本不变；同步 `M3_scale_report.md` §2/§4 与 JSON。
2. 修正 F2：若确要「deterministic」，`build_population` 设 `stochastic=False` 并重测 tick；
   否则改报告措辞为「stochastic（默认）CPU tick」。二者择一，保持标签与配置一致。
3. 修正 F3 措辞（区分 underflow 与 `d≤25` 盘掩膜）。

> 上述为小改动；修复后请 evaluator 复核（预期结构与结论不变，仅小核行与标签）。

### 7.6 未完成的检查

- **GPU 侧全部 NOT CHECKED**（§6.5 明示按用户指示待确认）：deterministic `enable_gpu` 成功率/显存、
  随机迁移行宽 ≤32 拒绝、`phase0` bit-identical。recon §5 M3 含 GPU 项，故 M3 目前为**部分完成**。
- CPU 侧未测：age_structured 下 tick 放大、海南逐月 K 开销（§6.5 已列）。

### 7.7 残余风险

1. 外推（海南 ~200 GB / ~64 min）仍是线性外推，未实测；age/性别/月 K 的结构性放大未纳入。
2. r25 盘 1951 支持取自论文；natal `size=51` 无 `d≤25` 截断（avd=2 时 support=2601），
   若按 natal 全核外推海南则为 ~281 GB——报告取 1951 属较乐观口径，结论方向不变。
3. APPROVED/NOT APPROVED 仅针对 CPU M3 的测量口径；不构成对 M4 迁移执行模型选型的认可。

> 审查期间未改任何文件；仅新增本回执。

## §8 — M3 §6.6 修复复核回执（evaluator）

### 8.0 裁定

**APPROVED（M3 CPU 侧 §7.5 修复目标达成）**。修复提交 `3cb6d43` 仅动复现脚本/数据/报告，未触产品
代码；§7.5 三项修复全部完成并经独立复现。M3 CPU 侧的测量口径与结论现已自洽。GPU 侧仍按用户指示
延后，**NOT CHECKED**。

### 8.1 §7.5 修复目标核对

| 目标 | 结果 | 独立证据 |
|---|---|---|
| F1：fold/builder 统一 `kernel_include_center=False` | **DONE** | size=30：k=3 fold nnz 7744→**6844 = builder 实际 nnz**；k=51 774400→**773500 = builder 实际 nnz**（逐位相等） |
| F2：deterministic 标签与配置一致（`stochastic=False`） | **DONE** | probe `:79` 已改；tick 系数由 ~6e-7 降至 ~2.95e-7 s/entry（确定性无抽样成本），标签成立 |
| F3：区分 underflow 与 `d≤25` 盘掩膜 | **DONE** | `M3_scale_report.md:52-54` 已改述 |
| 重跑并同步数据/报告 | **DONE** | §8.2 |

### 8.2 独立运行证据（隔离副本 `/tmp/hexeval/m3b`）

- `python repro/m3_scale_probe.py`（6m32s，峰值 RSS 6.73 GB）：**20/20 组合 support/nnz/csr_bytes 与
  提交 JSON 逐位相同**；论文核结构量一致；`nnz` 现等于 builder 实际 CSR（见 §8.1）。
- 300²×51² = 2.145e8 nnz / 3.433 GB / fold 63.5 s（提交 63.3 s）；论文核 avd=1.0 = 1.948e8 / 3.118 GB /
  58.8 s（提交 58.2 s）；300²×21² tick 11.44 s（提交 11.29 s）。
- `tick_s` 复核/提交比 0.98–1.15、`fold_s` 0.85–1.09（机器负载差异）；时间量与 §2–§4 表一致。
- `16·nnz+8(n+1)` 逐行成立；CPU 无预算守卫（`ensure_migration_budget` 仅 GPU）复核不变。

### 8.3 结论复核

- 内存墙（`CSR≈16·nnz`；300²×51²=3.43 GB；论文核 avd=1.0=3.12 GB）与海南外推（~211 GB / ~65 min 折叠）
  **不受 F1 修复影响**（大核 n_demes 占比 <0.2%）。
- tick 结论因 F2 修正而变化：确定性系数由 ~6e-7 降至 **~2.95e-7 s/entry**（300²×21²=11.3 s；
  论文核外推 ~58 s/tick；海南 ~66 min/tick）。结论方向（CPU 不可行）不变，且现在标签与配置一致。

### 8.4 非阻塞

- `M3_scale_report.md:63`「大网格约 2.7–3.0e-7 s/entry」为**大核渐近带**；300² k=11 实测 3.8e-7、
  小 k 更高（小核有固定 per-tick 开销）。用于外推的是大核系数，恰当；如需可在报告中注明该范围
  对应 k≳21。仅表述，不影响结论。

### 8.5 未完成 / 残余

- **GPU 侧 NOT CHECKED**（§6.5）：deterministic `enable_gpu`/显存、随机行宽 ≤32 拒绝、`phase0`；
  recon §5 M3 含 GPU 项，故 M3 整体仍为部分完成。
- 外推为线性、未实测；age_structured 放大与月 K 开销未测（§7.7/§6.5）。
- 本批准仅覆盖 CPU M3 测量口径，不构成 M4 迁移执行模型选型的认可。

> 审查期间未改任何文件；仅新增本回执。§7 的 NOT APPROVED 已由本次修复解除。

## §10 — M3（GPU 侧）§9 独立审查回执（evaluator）

### 10.0 裁定

**APPROVED（M3 GPU 侧 §9）**。风险分类：局部（复现脚本 + 数据），未改产品源码；`git status` 干净
（GPU feature 只重建 gitignored 的 `_engine_rs`）。§9.4 四项核对点全部独立复现通过；M3 的 CPU+GPU
守候项均已完成（GPU 数值正确性属 M7，不在本阶段）。

> 独立性声明：evaluator 在隔离副本 `/tmp/hexeval/m3g` 重跑，并把结果与提交 JSON、Rust 源码逐项比对。

### 10.1 逐条独立核对（对应 §9.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | 忠实调用 session API；`migration_cache_bytes` Python 镜像与 Rust 一致 | **PASS**（逐字节相等，见 §10.2） |
| 2 | enable/行宽/预算三项守卫 + `phase0` 可复现 | **PASS**（见 §10.3/§10.4） |
| 3 | 结论边界（正确性属 M7，本阶段仅 enable/预算/守卫/时间） | **PASS**（§9.4.3、报告 §7 明示） |
| 4 | 预算报错时 free<nvidia-smi 的差额解释 | **合理**（非阻塞，见 §10.5） |

### 10.2 API 与缓存公式核对

- Rust `migration_cache_bytes`（`executor.rs:3065`）展开 = `8(n_batch+1) + 20·nnz + 4·n_batch
  + 8·nnz·A·Z + 4·nnz·A·Z²`；Python 镜像（`m3_gpu_probe.py:49-56`）逐项相同。代入实测值逐行相等
  （如 300²×21²：nnz=3.82261e7、A=2、Z=3 → 5,352,734,008 B = 5104.8 MiB，与报告/JSON 一致）。
- Rust 常量：`MAX_Z=32`、`MAX_CSR_ROW=MAX_Z=32`（`kernels.rs:60-63`）；行宽守卫在
  `migrate_tick_stochastic`（`executor.rs:2360,2389-2392`）；预算守卫 `ensure_memory_budget`
  （`executor.rs:3038-3047`）经 `spatial.rs:386` 在 enable 期调用。报告引用的错误文本与源码逐字一致。

### 10.3 独立运行证据（隔离副本）

环境：RTX 5090 D v2，24455 MiB 总 / 23961 MiB 空闲；GPU feature 构建已存在。
命令：`unset PYTHONHOME PYO3_PYTHON PYTHONPATH` + `LD_LIBRARY_PATH=.../nvidia/cu13/lib` 下
`python repro/m3_gpu_probe.py`（1m17s）：

- deterministic 5/5 `disabled→enabled`；cache 2.7 / 52.6 / 773.7 / 1416.7 / 5104.8 MiB（与提交逐位相同）；
  GPU tick 0.0011 / 0.0156 / 0.2151 / 0.4194 / 1.4270 s（报告 0.0011/0.0155/0.2148/0.4218/1.4422）。
- stochastic：30²×k=5 `max_row=24 → tick_ok`；30²×k=11（120）、60²×k=21（440）→ **rejected**，
  文本 `stochastic migration CSR row length {120,440} exceeds the device scratch limit of 32`。
- budget：7 等位（Z=28）×200²×k=11 → `RuntimeError: GPU memory budget exceeded: the state needs
  30011 MiB (31468866008 B), but only 19418 MiB is free`；公式复算 31468866008 B **完全一致**。
- 空闲显存 23961→23959 MiB（上下文释放正常）。

### 10.4 `phase0` 回归（GPU feature 构建）

`unset PYTHONHOME PYO3_PYTHON PYTHONPATH && .venv/bin/python scripts/phase0_baseline.py --check`
→ 全部 scenario `[OK]`、`all scenarios bit-identical`、**EXIT=0**。GPU feature 未改变 CPU 数值语义。

### 10.5 非阻塞

- 报告的「迁移缓存足迹」由 Python 镜像**公式**给出，而非 nvidia-smi 实测的 used-memory delta；
  但 enable 成功（真实分配）与预算守卫（同一公式对实时空闲显存判定）已从经验上相互印证。recon §5 M3
  的「显存实测」若按字面要求逐例 delta，可另补；当前实现足以支撑 M4 结论。
- 预算报错时 `free=19418 < nvidia-smi 23961`（差 4543 MiB）归因 state/sperm/上下文；方向合理，
  未逐项分解。仅解释性，不影响守卫正确性。
- 报告 §3 用「2-locus Z=25 → ~69×」举例；本机 7 等位实测 Z=28（(28/3)²≈87×），趋势一致。仅示例。

### 10.6 范围 / 残余

- 本批准仅覆盖 M3 GPU 侧的 enable/预算/行宽/时间；**GPU 数值正确性（GPU vs CPU）属 M7**。
- 未测 GPU 随机统计等价、hook/history、age_structured 放大、MIG/多租户争用；结论限本机 RTX 5090。
- M3 至此 CPU（§7/§8）与 GPU（§9/§10）守候项均已独立复核；后续 M4 选型不在本回执范围。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §12 — M4 路线 A §11 原型/设计独立审查回执（evaluator）

### 12.0 裁定

**APPROVED（仅 §11 的原型验证与设计提案）**。风险分类：局部（独立 repro 脚本 + 报告）；`git status` 仅
`EVALUATE.md` 由 evaluator 追加，未改产品代码。§11.4 四项核对点独立复现通过；核心等价性（模板/标签、
归一化卷积/FFT 与 CSR）经推导与实测确认。**注意**：本批准**不覆盖**任何引擎级实现——设计提案
（`M4_routeA_productization_design.md`）明确「待用户批准后再实施」，其落地属高风险，须另立独立复核。

> 独立性声明：evaluator 在隔离副本 `/tmp/hexeval/m4{,_t,_g}` 重跑三个脚本，并与提交 JSON、natal/Rust 源码逐项比对。

### 12.1 逐条独立核对（对应 §11.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | `Z=K'⊛m` 等价 CSR（丢界+重归一化+源保留）；`apply_csr` 忠实于 Rust | **PASS**（见 §12.2） |
| 2 | 解析模板签名与 `normalize_coord` 边界规则一致 | **PASS**（300² 为抽样核验，见 §12.3/§12.6） |
| 3 | GPU FFT 线性卷积无 circular wrap、f32/f64 误差、CPU 同口径 | **PASS**（见 §12.4） |
| 4 | 结论边界（CSR 为 numpy 代理；构建仅矩形域） | **PASS**（报告 §4 注、§11.4.4、设计 §4） |

### 12.2 等价性推导核对

- natal `fold_migration_csr` kernel 模式（`include_center=False, adjust_on_edge=False`）每行最终权重
  `= k'_o / Σ_{o': s+o'∈D} k'_{o'}`，行和=1（实测 `max|row_sum−1| ≤ 1.1e-15`），`stay_after_send=True`。
- Rust `migrate_csr_deterministic`（`kernels/spatial.rs:461`，`stay_after=true`）：
  `out[d] += outbound·w`、`out[s] += value − Σ outbound·w`，即 `v(1−rate)+Σ`。与 `apply_csr`
  （`v − rate·v + bincount(dest, rate·v·w)`）逐项一致（对单标量场；Rust 的 virgin/stored-sperm
  记账属生命周期细节，不在迁移算子内）。
- 归一化卷积：`g=rate·f/Z`、`Z=K'⊛m=Σ_{o:s+o∈D}k'_o`（即 CSR 每行分母）；
  目的格 `d` 收到 `Σ_{s=d−o} rate·k'_o·f(s)/Z(s)`，与 CSR `rate·f(s)·k'_o/Z(s)` 相同；源保留
  `f(1−rate)`。`ndimage.convolve`/`fftconvolve` 对中心对称的 `K'` 与 CSR 的 visit 顺序一致
  （非 wrap 下无重复目的地）。实测逐格差 `2.2e-16 ~ 1.0e-15`。

### 12.3 模板/标签与构建核对

- 独立复跑 `m4_template_build.py`：60²k11 / 120²k21 / 300²k21 / 300²k51 → `row_length_mismatch=0`、
  `dest_mismatch=0`、`max_weight_diff ≤ 1.4e-16`；`n_templates`、`csr_bytes`、`template_bytes` 与提交**完全相同**。
- 签名 `(min(r,R),min(rows−1−r,R),min(c,R),min(cols−1−c,R))` 与 `normalize_coord`（`topology.py:103-120`
  逐轴越界判定）推导一致：有效 `dr∈[−key0,key1]`、`dc∈[−key2,key3]`。
- 构建时间：300²×51² 解析 **0.606 s vs fold 61.7 s（~102×）**；存储 58.73 MB vs 3274 MB（~56×），与 §11.3 一致。

### 12.4 GPU cuFFT 核对

- 独立复跑 `/opt/conda/bin/python repro/m4_gpu_fft_prototype.py`（torch 2.12.0+cu130，RTX 5090）：
  5/5 规模 `f32_rel ≤ 2.6e-7`、`f64_rel ≤ 2.6e-15`，与提交逐位相同。
- 复核 `gpu_conv`：零填充到 `(rows+k−1,cols+k−1)` 后 FFT → **线性卷积无 circular wrap**；裁剪
  `[R:R+rows,R:R+cols]` 为 `'same'` 区；与 CPU `ndimage.convolve`（中心对称 `K'`）同口径。
- 相对 CPU 直接卷积加速实测 ~4.7–72×（报告 5–70×）。

### 12.5 独立运行证据（隔离副本）

- `python repro/m4_routeA_prototype.py`（1m22s）：等价性/存储/每步成本表复现（`conv/fft ≤ 1e-15`；
  300² k51 FFT 1.39 ms vs 直接 26.6 ms）。
- `python repro/m4_template_build.py`（1m16s）：见 §12.3。
- `/opt/conda/bin/python repro/m4_gpu_fft_prototype.py`（3s）：见 §12.4。
- 三个提交 JSON 的结构量/误差量与复跑一致（时间量随负载波动，属正常）。

### 12.6 非阻塞发现

- **F1（低，措辞）**：`M4_routeA_prototype_report.md:61`「M2 的卷积路径已与 MATLAB 参考达 4.8e-12（端到端
  佐证）」把 M2 的 **replicate** 边界卷积与路线 A 默认的 **丢界+重归一化**卷积混为直接佐证——二者边界语义
  不同（设计 §3.3 亦将 `'replicate'` 列为另需实现的变体）。建议改为「M2 佐证 natal 卷积核正确，边界语义
  两者不同」。
- **F2（低，核验范围）**：`m4_template_build.py` 对 300² 两例为**抽样核验**（`full=False`，`rows·cols·k²>1.5e7`），
  仅 ~56 行 + 角/边/中心；§11.3/§11.4.2 的「行/目的地/权重一致」宜注明抽样。解析推导已保证全体一致，
  但可在小规模做全量、大规模抽样并明示。
- **F3（低，流程/文档）**：`c734f90` 修改了 `.gitignore`（新增 `hexagon_spatial_test/research/`）。
  AGENTS 要求改 `.gitignore` 需用户明确要求；`research/` 含用户研究材料（DeepSeek 边界卷积、
  hex 坐标系报告），大概率属用户要求，但 §11.1 未记录，且其「未 commit」与现存提交（`00bc4b3`/`9286570`/
  `c734f90`）不符。建议补记。
- **F4（低，边界）**：等价性仅在 `wrap=False`、`adjust_on_edge=False`、矩形域、单标量场、f64 下验证；
  重复目的地（wrap 核）、不规则掩膜、replicate、多类批处理未验（§11.5 已列）。设计落地时须补。

### 12.7 范围 / 残余

- 本批准**仅覆盖原型与设计证据**；路线 A 引擎实现（新增迁移路径、CPU/GPU 内核、会话接线、缓存持久化）
  属 **M4 高风险**，须用户批准 + 独立 evaluator 复核 + 全量门禁，**不在本回执范围**。
- 设计提案中的数值/确定性策略（非逐位、容差分档、默认关闭）方向正确；`phase0`/CSR golden 不受影响
  的前提是默认关闭，实施后须实测。
- 未测 overlap-add 大域、不规则地形签名分组、路线 A 与 natal Rust 的逐 tick 对拍（§11.5）。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §14 — M4 路线 A 引擎实现（S0+S1）§13 独立审查回执（evaluator）

### 14.0 裁定

**NOT APPROVED（M4 §13，S0+S1）**。高风险（产品代码）。S0/S1 的隔离性、等价性、默认关闭、
`phase0` bit-identical 经独立复核**成立**；但**必需 Rust 硬门禁 `cargo fmt --check` 失败**（新增
`rust/src/gpu/cufft.rs` 未按 rustfmt 格式化），故按 `quality_checks_spec.md`「required standards
violation」给出 NOT APPROVED。修复为一次性格式化，代价极低。

> 独立性声明：evaluator 在仓库工作树（未改任何文件）运行全部门禁；命令与结果见下。

### 14.1 逐条独立核对（对应 §13.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | S1 `Z`/`K'/Z` 等价 CSR（含 wrap、中心排除、边界） | **PASS**（见 §14.2） |
| 2 | S0 给 `gpu` 加 `cudarc/cufft` 合规、默认关闭、不影响非 gpu/`phase0` | **PASS**（见 §14.3） |
| 3 | 阶段边界：S1 未接运行路径、默认关闭、不改 CSR 行为 | **PASS**（见 §14.3） |
| 4 | pyright 仅既有错误、无新增 | **PASS**（见 §14.4） |

### 14.2 S1 前端计划核对

- `build_fft_migration_plan`（`migration.py:462-545`）：`K'` 去中心（`include_center=False` 默认）、
  `z(s)=Σ_{o:s+o∈D}K'(o)`（非 wrap 向量化、wrap 取常量=核总和）——与 `fold_migration_csr` 的每行分母一致。
- 独立运行 `tests/test_spatial_fft_plan.py`：**21 passed**（`z` vs 逐 deme `normalize_coord` 暴力参考；
  `K'/z` vs CSR dest/权重逐条相等，wrap True/False；偶核报错）。
- 代码独立检查：`offsets` 仅取 `kp>0`，与 CSR「`weight<=0` 跳过」一致；`include_center=True` 时中心计入
  `z`，语义正确。非 wrap 下无重复目的地，conv 与 CSR visit 顺序一致（M4 §12 已证）。

### 14.3 S0/S1 隔离性与回归

- **未接运行路径**：`grep -rn "build_fft_migration_plan|MigrationFFTPlan" src/` 除定义处**无调用者**；
  公开 API 仅**新增**、无既有签名变更；CPU CSR 一行未改。
- `clippy`（默认）= **PASS**；`clippy --features gpu -- -D warnings` = **PASS**；
  `cargo check --all-targets` = **PASS**；`cargo test --lib` = **67 passed**；
  `cargo test --features gpu` = **261 passed**（含 `gpu::cufft::tests::cufft_linear_conv_matches_cpu`）。
- `scripts/phase0_baseline.py --check` = **all scenarios bit-identical，EXIT=0**。
- `pytest -q`（设 `LD_LIBRARY_PATH` 含 `/opt/conda/lib` 与 cu13）= **3655 passed**；本测试文件 21 passed。
  注：不设 `LD_LIBRARY_PATH` 时 `tests/test_gpu_*_frontend.py` 会因缺 `libnvrtc` 失败——**环境性**、
  与本改动无关（设 env 后全过）。
- `ruff check src demos` = **All checks passed**。

### 14.4 门禁失败与其它发现

- **F1（阻断，必须修复）：`cargo fmt --check` 失败。** 新增 `rust/src/gpu/cufft.rs` 有 3 处 rustfmt
  差异（`:6` import 排序、`:56`/`:70` 调用断行）。独立证据：
  - `cd rust && cargo fmt -- --check` → **FMT=1**，diff 于 `cufft.rs:6/56/70`；
  - `python scripts/check_rust.py` → **EXIT=1**，`Rust hard gates failed: cargo fmt -- --check`。
  - 根因：`cargo fmt` 无视 `cfg(feature)`，会格式化 gpu-gated 文件；主 agent §13.3 未运行
    `cargo fmt`/`scripts/check_rust.py`（仅列 cargo test），故遗漏。
  - 修复：`cd rust && cargo fmt`（或手工格式化 `cufft.rs`），复跑 `scripts/check_rust.py` → 期望 EXIT=0。
- **F2（低，公开面/文档）**：`migration.__all__` 新增导出 `MigrationFFTPlan`、`build_fft_migration_plan`
  （属**新增公开 API**），而 §13.2 称「未改公开 API」。项目对同类构件 `fold_migration_csr` 有
  `docs/{zh,en}/migration_kernel_impl.md` 记录，新构件暂未入文档（设计称 S3 同步）。建议：要么在 S3
  前**不导出**（保持模块内部），要么补 docs/zh+en。shims 一致性测试当前通过（新增名未提升到包顶层，
  不触发 `__init__.pyi` 变更）。
- **F3（低，测试强度）**：`test_fft_plan_weights_match_csr` 逐 `kr,kc` 期望时未过滤零权重项；对当前
  σ 的高斯核（全正）无影响，但若核含 0 项会与 CSR「跳过 0」不一致——建议期望侧同样按 `>0` 过滤。
- **F4（低，S0 探针）**：`cufft.rs` 为 S0 正确性探针（f32 容差 1e-4、host 侧谱乘、无设备谱乘/批处理），
  与 `cufft` feature 隔离；`clippy --features gpu` 通过。属预期，非缺陷。

### 14.5 范围 / 残余

- 本回执仅覆盖 S0+S1；S2（session/executor 接线）、S3（会话开关/中英文档/全量门禁）**未做**，不在范围。
- `wrap=True` 下重复目的地与 conv 的严格等价、`include_center=True`、不规则掩膜、多类批处理未测（§13.5）。
- 修复 F1 后请 evaluator 复跑 `scripts/check_rust.py` 与受影响检查即可转 APPROVED。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §15 — M4 §13（含 §13.6 S2a）再复核回执（evaluator）

### 15.0 裁定

**NOT APPROVED（M4 §13 仍）**。本次复核对象：§13.6 追加的 S2a（`migration_execution` 前端开关）与
其提交 `4194afa`（产品代码：`builder.py`/`population.py`/测试）。S2a 本身**合规**（默认关闭、显式失败、
测试充分）；但 §14 的**阻断项 F1（`cargo fmt --check` 失败）仍未修复**——`4194afa` 未触及 Rust，
`rust/src/gpu/cufft.rs` 依旧未格式化。故维持 NOT APPROVED。

> 独立性声明：evaluator 在当前提交（`4194afa`，工作树干净）运行全部门禁；命令见下。

### 15.1 §14 阻断项复核（F1：未闭环）

- `cd rust && cargo fmt -- --check` → **EXIT=1**，仍在 `cufft.rs:6/56/70` 报 3 处 rustfmt 差异；
  `python scripts/check_rust.py` 相应 `Rust hard gates failed: cargo fmt -- --check`。
- 本次提交范围（`git show --stat 4194afa`）仅 EVALUATE/设计文档 + `builder.py`/`population.py`/测试，
  **未改任何 Rust 文件**，故 F1 不会自行消失。修复仍同 §14.4：`cd rust && cargo fmt` 后复跑 `check_rust.py`。

### 15.2 S2a 独立核对（§13.6）

- **默认关闭/无行为变化**：`migration_execution: "csr"|"fft"` 默认 `"csr"`；`"csr"` 路径与既有 CSR 完全一致。
- **显式失败、无静默回退**：`"fft"` 在 `SpatialPopulation.__init__`（`population.py:786-797`）抛
  `NotImplementedError`；非法值在 builder（`builder.py:1593-1595`）与 population 均抛 `ValueError`。
- **未接运行路径**：`"fft"` 不构建任何 FFT executor；S2b 才接线（§13.5/设计 §9）。
- 测试：`tests/test_spatial_fft_plan.py` 新增 3 例（非法值 `ValueError`、`"fft"` `NotImplementedError`、
  默认 `"csr"` 可构建）；独立运行该文件 = **24 passed**。
- **stub/一致性**：运行 `scripts/generate_init_pyi.py` 后 `src/natal/__init__.pyi` **无 diff**（新 kwarg 属
  方法签名，不在顶层 stub 面）；`tests/test_phase0_shims.py` 通过。

### 15.3 门禁结果（当前提交）

- `pytest -q`（设 `LD_LIBRARY_PATH` 含 `/opt/conda/lib` 与 cu13）= **3658 passed**（与 §13.6 声明一致；
  不设该 env 时 `test_gpu_*_frontend` 因缺 `libnvrtc` 失败，属环境项）。
- `tests/test_spatial_fft_plan.py` = 24 passed。
- `ruff check src demos` = **All checks passed**。
- `scripts/phase0_baseline.py --check` = **all bit-identical，EXIT=0**。
- `pyright`（改动 4 文件）= 27 errors，全部位于**未改动的既有行**（`migration.py:45/46/78/79/113`、
  `population.py` 既有 `RateDeclaration`/`normalize_migration_rate` 等行）；`builder.py`、新测试、
  `migration.py:462+` 与 `population.py:700/786-797` **无报错** → 无新增，主 agent 判定成立。
- **唯一失败门禁**：`cargo fmt --check`（Rust，S0 引入，§14-F1）。

### 15.4 修复目标（不变）

1. `cd rust && cargo fmt`（格式化 `cufft.rs`），复跑 `python scripts/check_rust.py` → 期望 EXIT=0。
2. （§14-F2，非阻断）`migration.__all__` 新增导出的 `MigrationFFTPlan`/`build_fft_migration_plan`
   与新增 `migration_execution` 公开 kwarg：在 S3 前补 `docs/{zh,en}` 或暂不导出。
3. （§14-F3，非阻断）`test_fft_plan_weights_match_csr` 期望侧按 `>0` 过滤零权重项。

### 15.5 残余

- S2b（executor cuFFT 内核 + 会话接线 + 预算）未做；S3（文档/全量门禁）未做。
- S2a 的 `migration_execution` 未进入 clone/definition 往返语义的专门测试（当前仅 `"csr"` 可用，暂无影响）——
  S2b 接线时需补「restore→run」状态往返用例。

> 审查期间未改任何仓库源码或数据；仅新增本回执。§14 的 F1 修复后即可转 APPROVED。

## §16 — M4 §13.7 修复复核回执（evaluator）

### 16.0 裁定

**APPROVED（M4 路线 A S0+S1+S2a）**。修复提交 `24bbcc7` 全部闭环 §14/§15 阻断与非阻断项；
evaluator 复跑全部门禁 **全绿**。§14 的 NOT APPROVED 解除。S0/S1/S2a 均**默认关闭、未接运行路径**，
未改 CPU 数值语义（`phase0` bit-identical）。S2b/S3 仍未做，不在范围。

> 独立性声明：evaluator 在当前提交（`24bbcc7`，工作树干净）重跑全部硬门禁；命令见 §16.2。

### 16.1 修复目标核对

| 项 | 结果 | 独立证据 |
|---|---|---|
| F1 `cargo fmt --check`（阻断） | **DONE** | `cd rust && cargo fmt -- --check` → **EXIT=0**；`scripts/check_rust.py` → **EXIT=0** |
| F2 公开面收敛（不导出 FFT 构件） | **DONE** | `migration.__all__` 已移除 `MigrationFFTPlan`/`build_fft_migration_plan`，仍可按模块路径导入 |
| F3 测试期望按 `>0` 过滤零权重 | **DONE** | `test_fft_plan_weights_match_csr` 改为 `if plan.kernel[kr,kc] <= 0.0: continue` |

### 16.2 全门禁结果（当前提交）

- `cargo fmt -- --check` = **EXIT=0**；`scripts/check_rust.py` = **EXIT=0**（fmt/clippy/check/`test --lib`
  **67 passed**；rust-analyzer 可选门禁未安装，按规范跳过）。
- `cargo clippy --features gpu -- -D warnings` = **PASS**；`cargo test --features gpu` = **261 passed**。
- `pytest -q`（设 `LD_LIBRARY_PATH` 含 `/opt/conda/lib` 与 cu13）= **3658 passed**；
  `tests/test_spatial_fft_plan.py` = **24 passed**。
- `ruff check src demos` = **All checks passed**。
- `scripts/phase0_baseline.py --check` = **all scenarios bit-identical，EXIT=0**。
- `pyright`（改动文件）= 7 errors，全部落在**未改动的既有行**（`migration.py:45/46/78/79/114`），
  新代码（`:465+`、S2a `population.py:700/786-797`）**无报错**；`builder.py`、新测试无报错。
- `generate_init_pyi.py` 无 diff；`tests/test_phase0_shims.py` 通过。

### 16.3 残余（非阻断）

1. `migration_execution`（默认 `"csr"`）为**新增公开 kwarg**，`docs/{zh,en}` 与
   `MigrationFFTPlan`/`build_fft_migration_plan` 文档按 §13.7 随 **S3** 补齐——S3 前该 kwarg 仅 `"csr"`
   可用（`"fft"` 显式 `NotImplementedError`），属分阶段占位。
2. **S2b**（executor cuFFT 内核 + 会话接线 + 预算）、**S3**（会话开关文档/全量门禁）未做；
   `migration_execution` 的 clone/definition 往返语义未专门测试（当前仅 csr 可用，暂无影响）。
3. `wrap=True` 重复目的地、`include_center=True`、不规则掩膜、多类批处理未验（§13.5）。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §18 — M4 路线 A S2b 设备级 FFT 内核 §17 独立审查回执（evaluator）

### 18.0 裁定

**APPROVED（M4 路线 A S2b：隔离设备内核）**。高风险（产品代码，`gpu` feature 内）。`gpu_fft_migrate`
与既有 CPU `migrate_csr_deterministic` 的等价性经推导与单测确认；该模块**未接运行路径、默认不启用**，
`migration_execution="fft"` 仍显式 `NotImplementedError`，CPU 数值语义与 `phase0` 不受影响。S2c/S3 未做，不在范围。

> 独立性声明：evaluator 在当前提交（`16fcd6f`，工作树干净）复跑全部硬门禁并自行核验等价性推导。

### 18.1 逐条独立核对（对应 §17.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | `gpu_fft_migrate` 等价 `migrate_csr_deterministic`（布局/索引/边界） | **PASS**（§18.2） |
| 2 | 数值对齐：`'same'` 裁剪、核去中心、`Z` 口径、`1/(fr·fc)` | **PASS**（§18.2） |
| 3 | `gpu` feature 内、无运行路径引用 | **PASS**（§18.3） |
| 4 | 未覆盖面（随机/wrap/include_center/掩膜/多类/性能） | **已知残余**（§18.4） |

### 18.2 等价性推导核对（evaluator 独立推演）

- **布局/索引**：`scatter_g` 以 `rate[deme·(2A) + sex·A+age]` 索引 compact rate（deme-major `(n,2,A)`）；
  ind 平面 `(sex·A+age)·Z+z`、sperm 平面 `(age·Z+fz)·Z+mz`，与 CPU 函数一致；sperm 用 `rate_off=age`
  （sex 0 = female rate），与 CPU「stored sperm 随 female rate」一致。
- **`'same'` 裁剪**：信号与去中心核 `K'` 均零填充到 `(fr,fc)=(rows+k−1,cols+k−1)`，`r2c·c2r` 即**线性卷积**
  （无 circular wrap）；`out_full[r+R,c+R]=Σ_o K'[o]·g(d−o)`，与 CSR `weight(s→d)=K'[o]/Z(s)` 的
  `Σ_o rate·K'[o]·plane(d−o)/Z(d−o)` 逐项相同（推导确认，无需核对称性）。
- **`Z`/去中心**：`K'` 排除中心（`:160-161`），`Z=K'⊛m` 与 fold 每行分母一致；中心由 `plane·(1−rate)` 保留。
- **归一化**：`complex_mul` 折叠 `1/(fr·fc)`，补偿 cuFFT C2R 未归一化。
- **virgin/stored 记账**：CPR ind 平面的 virgin+stored 均按 female rate 迁移，净效果 = 对 `ind` 平面整体做
  归一化卷积，故设备内核逐平面作用即等价（与 §17.2 一致）。
- **独立运行新单测**：`cargo test --features gpu gpu::fft_migrate::tests::fft_migrate_matches_csr` → **1 passed**
  （`max_rel < 1e-4`）。

### 18.3 隔离性与门禁（当前提交）

- **未接线**：`grep -rn "gpu_fft_migrate" rust/src` 除定义/测试外**无调用者**；`migration_execution="fft"`
  仍抛 `NotImplementedError`（`population.py:796`）。
- `cargo fmt -- --check` = **EXIT=0**；`scripts/check_rust.py` = **EXIT=0**（fmt/clippy/check/`test --lib` 67）。
- `cargo clippy --features gpu -- -D warnings` = **PASS**；`cargo test --features gpu` = **262 passed**（含新单测）。
- `scripts/phase0_baseline.py --check` = **all bit-identical，EXIT=0**。
- `pytest -q` = **3658 passed**；`ruff check src demos` = All checks passed（本次仅 Rust+文档改动，Python 面未变）。

### 18.4 非阻塞发现

- **F1（低，测试口径）**：`fft_migrate_matches_csr` 给 CPU 参照传 `stay_after=false`，而 CSR fold 的实际
  运行时为 `stay_after_send=True`。二者在**行权重和=1** 时数值等价（本测试 `build_csr` 每行除以自身和），
  且 `false` 分支的保留量 `value−rate·value` 恰与 conv 的 `plane·(1−rate)` 同式；故不清除风险。建议补一例
  `stay_after=true` 以覆盖真实路径。
- **F2（低，覆盖广度）**：等价性仅 1 个 fixture（`4×3, k=3, A=2, Z=2, σ=1`）。建议加一例异形/异 `k`（如
  `7×5,k=5`）与一个边界主导的小网格，提升对索引/裁剪的回归力。§17.4.4/§17.5 已列其余未覆盖面。
- **F3（低，文档一致性）**：设计 §10 描述的目标形态是 `plan_many` batch=P + 设备侧谱乘；当前 S2b 为
  逐平面 host 迭代 + host 侧谱乘（§17.1/§17.5 已说明 S2c 重构）。属阶段差异，建议 §10 标注「S2b 隔离版 vs S2c 目标版」。

### 18.5 范围 / 残余

- 本批准仅覆盖 S2b 隔离设备内核；**S2c**（executor/session/schema 接线、复用 stream/缓冲、预算、
  `MigrationFFTPlan` 前端传递）与 **S3**（中英文档 + 全量门禁）未做。
- `wrap=True`（需 circular conv）、`include_center=True`、不规则掩膜、多类更大 P、随机路径、性能均未验。
- S2c 接线后需重跑全量门禁并复核「restore→run」状态往返与预算失败路径。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §20 — M4 路线 A S2c 接线 §19 独立审查回执（evaluator）

### 20.0 裁定

**APPROVED（M4 路线 A S2c：executor/session/前端接线）**。高风险（产品代码）。`"fft"` 现为可用、
默认关闭的可选路径；确定性 GPU-FFT 与 CPU-CSR 的**端到端**等价已在设备上验证；`"csr"` 路径未改、
`phase0` bit-identical；全部硬门禁通过。`"fft"`+`stochastic` 计划被静默忽略（**已披露**，S3 守卫）
列为首要残余，属阶段性遗留、不阻断本次接线。

> 独立性声明：evaluator 在当前提交（`06991fb`，工作树干净）复跑全门禁并在 RTX 5090 上运行端到端 FFT 用例。

### 20.1 逐条独立核对（对应 §19.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | 接线正确性（`enable_gpu`/`run_gpu_tick` 选择、`"csr"` 未变） | **PASS**（§20.2） |
| 2 | 数据传递 `MigrationFFTPlan → plan → backend → from_parts → FftMigrator`、stub 一致 | **PASS**（§20.3） |
| 3 | 默认关闭；不选 `"fft"` 行为不变、`phase0` bit-identical | **PASS**（§20.4） |
| 4 | 残余（`fft`+stochastic 等） | **已披露**（§20.5） |

### 20.2 接线核对

- `spatial.rs:397-410`：`enable_gpu` 有 `migration_plan` → `enable_fft_migration`（否则原 CSR 缓存预算）。
- `spatial.rs:1338-1347`：迁移步位置与 CSR 相同（生命周期 tick 后、`state_tick` 递增前；hook-stop 抑制迁移一致）；
  选择序 `stochastic → stochastic CSR`；`!stochastic && has_fft → fft`；否则 CSR。
- `executor.rs`：`fft: Option<FftMigrator>` 字段、`enable_fft_migration`（校验 `rows*cols==n_batch` + 显存预算
  `ensure_memory_budget`）、`has_fft_migration`、`migrate_tick_fft`（上传 rate、`migrate_inplace`、双缓冲 swap）；
  `new`/`reconfigure` 置 `None`。`"csr"` 路径（`migrate_tick`）一行未改。

### 20.3 数据传递与 stub

- `population.py:788-818`：`"fft"` 时校验 `topology`/`migration_kernel` 存在、拒绝 `kernel_include_center=True`，
  构建 `build_fft_migration_plan`，打包 `(rows, cols, kernel f32, z f32)` → `_fft_migration_plan`；
  `build_spatial_backend` 传 `migration_plan=self._fft_migration_plan`（`csr` 为 `None`）。
- `rust_backend.py` 透传；`spatial.rs::from_parts` 签名新增 `migration_plan=None`，映射为 `MigrationPlanHost`。
- `src/natal/_engine_rs.pyi`（**手工维护** stub，非 `generate_init_pyi` 生成）已同步：
  `migration_plan: tuple[int, int, list[float], list[float]] | None = None`，与 pyo3 `Option<(usize,usize,Vec<f32>,Vec<f32>)>` 一致。

### 20.4 门禁与端到端（当前提交）

- `cargo fmt -- --check`、`scripts/check_rust.py` = **EXIT=0**（fmt/clippy/check/`test --lib` 67 passed）。
- `cargo clippy --features gpu -- -D warnings` = **PASS**；`cargo test --features gpu` = **263 passed**
  （含 `fft_migration_tick_matches_host_reference`，`stay_after=true`，逐元素对齐 CPU CSR）。
- `scripts/phase0_baseline.py --check` = **all bit-identical，EXIT=0**。
- `pytest -q`（设 CUDA env）= **3659 passed**；`tests/test_spatial_fft_plan.py` = **25 passed**，其中
  `test_migration_execution_fft_matches_csr_on_device` **实际执行（非 skip）**：5×5 k=5 确定性 3 tick，
  GPU-FFT vs CPU-CSR `max_rel < 1e-4`（实测 ~1.5e-7）。
- `ruff check src demos tests` = **All checks passed**。

### 20.5 发现（非阻断；已披露）

- **F1（中，首要残余）**：`"fft"` + `stochastic=True`（`setup` 默认值）时，`run_gpu_tick` 走
  `migrate_tick_stochastic`，**静默忽略** FFT 计划；且 `enable_gpu` 仍构建 `FftMigrator` 并计入预算（浪费）。
  与项目「不得静默回退」原则相悖。§19.4.4/§19.5 已披露并列为 S3。建议 S3 尽早加显式守卫
  （`migration_execution=="fft" and stochastic` → `raise`），并在 `migration_execution` docstring 注明「仅确定性 GPU 路径」。
- **F2（低，残余）**：`"fft"` 仍先折叠 CSR（构建墙未消除）；hooks/history、`restore→run`、`wrap`/`include_center`/
  不规则掩膜/大域性能未测（§19.4.4/§19.5）。
- **F3（低，信息）**：CPU（未 `enable_gpu`）时 `"fft"` 走 CSR——与设计「CPU 保持 CSR golden，不做 CPU FFT」
  一致，但应在文档/`migration_execution` 说明中显式写明，避免误解。

### 20.6 范围 / 残余

- 本批准仅覆盖 S2c 确定性接线；**S3**（中英文档、`"fft"` 跳过 CSR 折叠、`fft`+stochastic 守卫、
  hooks/history/restore 覆盖、大域与性能）未做。
- 建议 S3 完成前不将 `migration_execution="fft"` 作为推荐用法或默认；默认 `"csr"` 不变。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §22 — M4 路线 A S3 §21 独立审查回执（evaluator）

### 22.0 裁定

**NOT APPROVED（M4 §21 / S3）**。高风险（产品代码）。S3 的文档同步、跳过 CSR 折叠、stochastic 拒绝在
`gpu` 构建下**均正确且门禁全绿**；但 **CPU 守卫被 `#[cfg(feature="gpu")]` 门控**，而**默认构建是
`gpu` 关闭**——故在默认（非 gpu）构建下：`migration_execution="fft"` **静默跳过迁移**（空 CSR，无报错），
且主 agent 新增的 `test_migration_execution_fft_cpu_run_requires_gpu` **在该构建下失败**（未 skip）。
这违反文档 §11.3b「未启用即运行会显式报错，不会静默跳过迁移」的承诺，并使规范 `pytest` 门禁在默认
配置失败。evaluator 已实际构建默认扩展复现。

> 独立性声明：evaluator 备份并临时替换 `_engine_rs` 扩展为**默认（非 gpu）构建**复现失败，随后**已还原**
> gpu 扩展（103647136 B）；`git status` 干净，未改任何 tracked 文件。

### 22.1 逐条独立核对（对应 §21.3）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | 跳过折叠：空 CSR 满足契约、`csr` 默认未变、`phase0` bit-identical、`fft` e2e 一致 | **PASS（gpu 构建）** |
| 2 | CPU 守卫：有计划且无设备执行器时报错；不误伤 `csr` | **FAIL（默认非 gpu 构建）**（§22.2） |
| 3 | 文档 zh/en §11.3b 同步、准确 | **部分 FAIL**：承诺在默认构建不成立（§22.3） |
| 4 | 默认关闭、`"csr"` 行为不变 | **PASS** |

### 22.2 独立复现证据

- **gpu 构建（当前安装）**：全门禁绿——`scripts/check_rust.py` EXIT=0；`cargo test --features gpu` 263 passed；
  `phase0` bit-identical；`pytest -q` 3661 passed；`tests/test_spatial_fft_plan.py` 27 passed（含
  `test_migration_execution_fft_matches_csr_on_device` 真实执行，GPU-FFT vs CPU-CSR `max_rel ~1.5e-7`）；
  `ruff` 通过。
- **默认（非 gpu）构建**（`unset PYTHONHOME PYO3_PYTHON PYTHONPATH; .venv/bin/maturin develop`，pyproject
  `features=["extension-module"]`，即无 `gpu`）：
  - `gpu_status() == "unavailable"`；
  - `migration_execution="fft"` 建出的种群 `_migration_csr.dest_idx.size == 0`；
  - `sp.run(1)` **正常返回、无异常**（对比 e2e：种群总量前后不变 → 迁移被静默跳过）；
  - `pytest tests/test_spatial_fft_plan.py` → **1 failed, 25 passed, 1 skipped**：
    `test_migration_execution_fft_cpu_run_requires_gpu`：`Failed: DID NOT RAISE Exception`。
  - 复核后已 `cp` 还原 gpu 扩展，重跑该文件 27 passed。

### 22.3 根因

- `rust/src/sessions/spatial.rs`：`migration_plan` 字段与 `run_inner` 的 CPU 守卫均为 `#[cfg(feature="gpu")]`；
  `from_parts` 在 `#[cfg(not(feature="gpu"))]` 分支 `let _ = migration_plan;` **静默丢弃**（`:331-332`）。
- `pyproject.toml:123 features=["extension-module"]` + `Cargo.toml` `[features]` 默认无 `gpu` →
  规范（默认）构建即非 gpu；此时前端 `population.py` 仍无条件构建并传递 `MigrationFFTPlan`，CPU 路径以
  空 CSR 运行 → **静默 no-op 迁移**。
- 新增测试 `test_migration_execution_fft_cpu_run_requires_gpu` 未按构建能力 skip，故在默认构建失败。

### 22.4 修复目标（最小、可复现）

1. **非 gpu 构建显式拒绝**：在 `spatial.rs::from_parts` 的 `#[cfg(not(feature="gpu"))]` 分支，若
   `migration_plan.is_some()` 返回 `PyValueError`（例如 "migration_execution='fft' requires a gpu-enabled
   build"）；或等效地在前端按 backend 能力（`gpu_status()=="unavailable"`）拒绝。**空 CSR 绝不能再落到
   CPU 路径静默运行。**
2. **测试与构建能力对齐**：`test_migration_execution_fft_cpu_run_requires_gpu` 在扩展无 `gpu` 时应断言
   「构建/运行显式报错」并在无 gpu 时同样期望异常（修复 1 后自然成立）；避免仅在 gpu 构建通过。
3. **文档核对**：§11.3b 的「未启用即运行会显式报错」需在默认构建亦成立（修复 1 后应重跑默认构建验证）。
   回归命令（默认构建）：
   `unset PYTHONHOME PYO3_PYTHON PYTHONPATH && .venv/bin/maturin develop && .venv/bin/python -m pytest tests/test_spatial_fft_plan.py -q`
   期望 EXIT=0（修复前实测 1 failed）。

### 22.5 残余

- §21.4 所列（`fft` 下 hooks/history、`restore→run`、`wrap`、`include_center`、不规则掩膜、大域/大核性能）不变。
- 修复 1 后需在**两种构建**（默认非 gpu 与 `--features gpu`）分别复跑 `pytest`/`phase0`/`check_rust`，
  并确认 gpu 构建下 e2e 仍一致。

> 审查期间未改任何仓库源码；evaluator 临时替换的 `_engine_rs` 扩展已还原为 gpu 构建。

## §23 — M4 §21.8 修复复核回执（evaluator）

### 23.0 裁定

**APPROVED（M4 路线 A S3）**。修复提交 `38f80b6` 闭环 §22 阻断项：默认（非 gpu）构建现显式拒绝
`migration_execution="fft"`，空 CSR 不再可能落到 CPU 静默运行；测试与构建能力对齐。evaluator 在
**默认非 gpu 与 `--features gpu` 两种构建**分别复跑，全部门禁通过。§22 的 NOT APPROVED 解除。

> 独立性声明：evaluator 先后 `maturin develop`（默认，无 gpu）与 `maturin develop --features
> "gpu,extension-module"` 重建扩展，分别复跑；结束状态为 gpu 构建（103647136 B），`git status` 干净
> （仅本回执改动）。

### 23.1 修复目标核对

| 项 | 结果 | 独立证据 |
|---|---|---|
| 非 gpu 构建显式拒绝 `fft` | **DONE** | `from_parts` 现 `return Err(PyValueError("... requires a gpu-enabled build"))`；实测 build → `ValueError` |
| 测试与能力对齐 | **DONE** | `_gpu_unavailable()` + skip；`test_migration_execution_fft_requires_gpu_path` 接受构建或运行报错 |
| 文档 §11.3b 承诺在默认构建亦成立 | **DONE** | 默认构建下 `fft` 构建即报错 |

### 23.2 两构建门禁证据

- **默认（非 gpu）构建** `maturin develop`：
  - `migration_execution="fft"` 构建 → `ValueError: migration_execution='fft' requires a gpu-enabled build`；
  - `pytest tests/test_spatial_fft_plan.py` = **25 passed, 2 skipped, 0 failed**（修复前为 1 failed）。
- **gpu 构建** `maturin develop --features "gpu,extension-module"`：
  - `pytest tests/test_spatial_fft_plan.py` = **27 passed**（含 `test_migration_execution_fft_matches_csr_on_device`
    真实执行，非 skip；GPU-FFT vs CPU-CSR `max_rel ~1.5e-7`）；
  - `pytest -q` = **3661 passed**；`ruff check src demos tests` = All passed；
  - `scripts/phase0_baseline.py --check` = **all bit-identical, EXIT=0**；
  - `scripts/check_rust.py` = **EXIT=0**（`test --lib` 67 passed）；`cargo test --features gpu` = **263 passed**。

### 23.3 残余（非阻断）

- §21.4 所列`fft` 下 hooks/history、`restore→run`、`wrap=True`、`include_center=True`、不规则掩膜、
  大域/大核性能仍未测。
- `MigrationFFTPlan`/`build_fft_migration_plan` 仍未入 `__all__`（公开面仅 `migration_execution`），
  与 §13.7 的收敛一致；如后续要公开再补 docs。

### 23.4 建议

- 在 CI/交付脚本中**同时**覆盖默认与 `--features gpu` 两种构建的 `pytest`，避免此类「构建能力相关」
  行为只在单构建被验证（本次缺口正是由此产生）。
- M4 路线 A 的剩余里程碑（M5 径向/M6 海南接入、M7 GPU 正确性/性能）按计划另立复核。

> 审查期间未改任何仓库源码或数据；仅新增本回执。§22 的 NOT APPROVED 已由本次修复解除。

## §25 — M1 遗传一步对照 §24 独立审查回执（evaluator）

### 25.0 裁定

**APPROVED（M1 §24）**。风险分类：局部（复现脚本 + 数据），未改产品代码。「natal `offspring_tensor`
与论文 homing `mats` 逐元素相等」经 evaluator 独立复现（含 MATLAB 参考交叉核对）确认；两处小限制见
§25.3，均不影响结论。

### 25.1 逐条独立核对（对应 §24.4）

| # | 核对点 | 结论 |
|---|---|---|
| 1 | `paper_homing_mats` 忠实翻译 `drive_generator.m`（转置 + fitness cost） | **PASS**（MATLAB 逐位相等） |
| 2 | 「精确相等」可信；基因型映射/`Res\|Res` 说明恰当 | **PASS**（映射正确；见 §25.3 小限制） |
| 3 | natal preset 与论文同参（dc、无 resistance、无 fitness cost） | **PASS**（由精确相等反证） |

### 25.2 独立证据

- `python repro/m1_offspring_check.py`：dc=1.0/0.9/0.5 → `max_abs=0.0`，全表行和=1（复现提交 JSON）。
- **MATLAB 交叉核对**（`addpath repro/ref/flat/homing; [~,dp]=drive_generator(0,dc,0,1)`）：
  对 dc=1.0/0.9/0.5，MATLAB `dp.mats(25×5)` 与 Python `paper_homing_mats` **max|diff|=0.0**；
  `carrier_index=[1 2 5]`。→ 翻译忠实。
- **独立张量比较**：natal `P[np.ix_(idx,idx,idx)]` 与 `transpose(mats.reshape(5,5,5),(1,2,0))` **max|diff|=0.0**；
  natal 全表 `P.sum(axis=2)==1`。
- natal 侧参数：`HomingDrive(drive_conversion_rate=dc, late_germline_resistance_formation_rate=0.0)`，
  其余默认 `embryo_resistance_formation_rate=0.0`、`functional_resistance_ratio=0.0`、无 fitness cost；
  与 `gcr=0, herr=0, ddfitness=1` 同参（由与 MATLAB 的精确相等佐证）。

### 25.3 小限制（非阻断）

- **F1（低，轴序不可辨）**：本组参数（herr=0）下，论文 5×5 块在 `(i,j)` 上**严格对称**
  （实测 `max|ref−refᵀ|ᵀ=0`）；故等价性测试**不能区分** `[mother,father]` 与 `[father,mother]` 轴序。
  取 herr=0.3 时参考块非对称（实测 0.51），可作判别用例。建议：若要钉住「母 i × 父 j」的方向，
  补一例 `herr≠0`（natal preset 用 `late_germline_resistance_formation_rate` 或不对称性别率）。
  注：对称情形下该轴序不影响算子等价结论。
- **F2（低，术语）**：报告 §3「论文 `mats` 列和不为 1（属构造如此）」与 §2「行和」并列易混；natal 是
  `offspring_tensor` 行和=1，论文 `mats` 作为「父×子」算子其列和/行和口径不同——建议在报告中统一标注
  其各自轴含义（natal `(M,F,O)` 对 P.sum(O)；论文 `mats[(k,i),j]`）。仅表述。

### 25.4 残余

- 仅 homing（一基因座 5 型）；TARE/TADE/CifAB/2-locus(25)/Wolbachia(2) 未逐一映射（§24.5）。
- 仅**表级**一步算子等价；未做 natal `run` 一步新生儿分布 vs 论文 `renew_function` 的端到端对照（§24.5）。
- `ddfitness`/性别特异未扫（§24.5）。

> 审查期间未改任何仓库源码或数据；仅新增本回执。

## §27 — §26 M5 收尾 + chasing 计划评审回执（evaluator）

### 27.0 裁定

**NOT APPROVED（计划文本需修订；本轮不需改产品代码，也不需改 `.gitignore`）**

- 审查对象：`recon/hex-model` @ `9cf0be0`（`0906a5f` 的下一提交）。工作树 clean；该提交只含
  `.gitignore`、`hexagon_spatial_test/EVALUATE.md`、`hexagon_spatial_test/chasing_model/PLAN.md`，未触碰产品代码
  （`git show --stat HEAD`）。§26 所称“本次未提交改动”实际已在 `9cf0be0` 提交；审查据此进行，不影响内容。
- 结论：核对点 1/3/4 基本 PASS；核对点 2 的年龄/存活/密度/迁移/释放映射与既有 M1/M5/event-seq 口径一致（PASS），
  但 **F1（S5/route A GPU 与引擎能力冲突）**、**F2（`scaling_factor` 与密度标定关系未定义）** 属于计划文本中的
  可复现性缺陷；**F3（homing chasing 可行性）** 是需要在 S1/S2 设决策门的高风险项。核心 S0–S4 方向可行，
  修订 PLAN 后即可进入实现。

### 27.1 独立运行证据（命令 + 结果）

1. 版本/提交范围：`git status --short` → 空；`git rev-parse HEAD` → `9cf0be0`；`git show --stat HEAD` → 仅上述 3 文件。
2. `.gitignore` 未误伤已跟踪文件：`git ls-files event-seq geographic-score hexagon_spatial_test/hex-model-new` → 空；
   `git check-ignore -v event-seq/img/spatial/tare_chasing.json geographic-score hexagon_spatial_test/hex-model-new`
   → `.gitignore:81/82/83` 命中。`chasing_model/{model.py,run_chasing.py,results/foo.json}` → NOT IGNORED；
   `results/foo.html`/`foo.csv` → 被既有全局 `*.html`/`*.csv` 规则忽略。
3. 引擎能力：
   - `pytest tests/test_spatial_fft_plan.py::test_migration_execution_fft_rejects_stochastic -q` → **1 passed**。
   - 自建 15×15 `HexGrid`、`n_ages=10`、`stochastic=True + continuous_sampling=True`、`size=7` 核、`Op.add`
     释放钩子、`run(n_steps=3, record_every=0)`：
     - `execution="csr"` → `OK total=1051.5771006921543 n_ztypes=3 shape=(225, 2, 10, 3)`。
     - `execution="fft"` → `ERROR NotImplementedError migration_execution='fft' supports deterministic models only`。
   - `fold_migration_csr`（15×15、`size=7`）→ `include_center=False`：`nnz=8424, max_row_width=48`；
     `include_center=True`：`nnz=8649, max_row_width=49`（builder 默认 `kernel_include_center=False`）。
4. 密度/放大口径（同一 15×15 模型，首 tick 后每 deme 总人口）：
   - `scale=1, K=4/9` → `expected_comp=42.2222, total1=2.111, adult1=1.667`。
   - `scale=10, K=4/9` → `expected_comp=42.2222, total1=17.209, adult1=16.667`（recruitment 被压低）。
   - `scale=10, K=10·4/9=4.4444` → `expected_comp=422.2222, total1=21.111, adult1=16.667`（正好 10× 基准 2.111）。
   - `pop._demes[0].params.expected_comp / expected_survival_rate` → `42.2222 / 0.0111111 (=1/90)`。
5. 事件顺序（源码）：`rust/src/kernels/age_structured.rs:1087-1088`（first hook → reproduction → early hook →
   survival → late hook → aging）；`rust/src/sessions/spatial.rs:755-757`（“lifecycle … then migration”）。
6. event-seq 证据（不在当前工作树，读自 `explo/event-seq` 分支）：
   `git show explo/event-seq:event-seq/HANDOFF.md` §2#9/#13；`git show explo/event-seq:event-seq/age_model.py`
   第 247-258、386-387 行（`scale` 同时乘初始态与 `density_ref`）。

### 27.2 逐条发现

**F1（高，S5 原描述不可执行）**：`PLAN.md:28` 的“大范围再评估 route A `fft`”与 `PLAN.md:49` 的
“S5 放大（可选）101×101 / route A GPU 稳健性”默认服务 stochastic chasing；但 `fft` 路径明确只支持确定性
（`src/natal/frontend/spatial/population.py:806-811`；`src/natal/frontend/spatial/builder.py:1589-1593`；
现有测试 `tests/test_spatial_fft_plan.py:161` 也断言 stochastic 被拒）。同时，stochastic CSR GPU 有行宽上限
`MAX_CSR_ROW = MAX_Z = 32`（`rust/src/gpu/kernels.rs:63`；`rust/src/gpu/executor.rs:2470-2473`），而计划
`size=7` 核的 CSR 最大行宽实测为 48（builder 默认 `kernel_include_center=False`；`include_center=True` 时为 49）。
→ **S1–S3 必须显式写 CPU CSR；S5 不能按现文做 stochastic route A GPU。**
修订方式二选一：(a) S5 改为 101×101 CPU CSR；(b) 保留 route A/FFT 仅作**确定性对照**，并明确说明
“stochastic + route A 需要引擎改动 → 另行批准 + 根 `EVALUATE.md` + 全量门禁”，不得作为默认计划步骤。

**F2（高，影响 S2/S3 可复现性）**：`PLAN.md:30`/`PLAN.md:68` 用 `scaling_factor` 调节信噪，但未定义它与
`age_1_carrying_capacity`（K）/`eggs_per_female` 的关系。event-seq 的 `scale` 同时乘初始态与 `density_ref`
（`explo/event-seq:event-seq/age_model.py` 247-258、386-387；HANDOFF §3.1 “D 随 scale 缩放”）。natal 的
`expected_comp` 由 K 与 eggs 派生；上述证据 4 表明只放大初始态而不放大 K（`scale=10, K=4/9`）时
`expected_comp` 仍为 42.222，recruitment 被压低（total1=17.209 vs 21.111），所谓“只调信噪”不成立。
→ PLAN 必须写明：`scaling_factor` 同时乘初始年龄向量与 `age_1_carrying_capacity`（或等价地按 scale 重推 K）；
结果 JSON 记录该值。否则 S2 的 `scaling_factor` 扫描与 S3 的重复统计不可复现。

**F3（中高，科学可行性风险）**：`PLAN.md:9`/`PLAN.md:11` 以 homing 为第一驱动追求“驱动与 WT 空间共存、
前沿持续移动、不必然全局固定”。但 event-seq 的既有结论（`explo/event-seq:event-seq/HANDOFF.md` §2#9/#13）
是：homing + 足够 fitness cost 在均匀模型里落到**均匀内部共存**，不是严格 chasing；严格 chasing 需要
阈值/Allee（如 TARE），而 TARE 在该模型里又没有真正 WT 长期共存。`PLAN.md:72`/§26.4 已承认“尚未验证”，
但 S1/S2 没有明确的失败判据与回退。→ 建议（不是断言 homing 一定不可行）：在 S1/S2 增加可执行的决策门
（例如连续 N tick 前沿半径不增，或 WT 只在驱动后方均匀残留而无滞后带），并预先定义 fallback
（TARE/阈值机制，或把目标明确缩小为“空间共存 + 移动前沿”）。否则可能在不可达目标上反复调参。

**F4（中，映射口径不一致）**：`PLAN.md:22` 写“homing 5 型，直接复用 M1 已验证的 `offspring_tensor` 映射”；
M1 的 5 型映射需要 `Res` 等位（`repro/m1_offspring_check.py:45-49`，species `["WT","Drive","Res"]`），
而 `repro/m5_engine_probe.py:100` 只建 `["WT","Drive"]`（3 型）；本次冒烟实测 `n_ztypes=3`。→ 明确二选一：
(a) 3 型理想 homing（无抗性形成，`rw/dr` 不可达），相应调整 carrier 定义（如 carrier=`WT|Drive`+`Drive|Drive`）；
(b) 5 型论文映射（加 `Res` 等位并验证 M1 子张量）。不要在同一计划里混用。

**F5（低，事件顺序表述）**：`PLAN.md:10` 只写“R-S-A”。natal 的实际顺序是
reproduction→survival→aging（`rust/src/kernels/age_structured.rs:1087-1088`），随后 migration
（`rust/src/sessions/spatial.rs:755-757`）。→ 写成“R-S-A → migration（tick 末尾）”，避免误解迁移位置。

**F6（低，初始态/指标）**：`PLAN.md:21` 的“WT 平衡年龄向量”来自 M5/reference，含 age0=40 eggs/deme；
在 natal R-S-A 下 tick 边界 age0=0。实测初始总人口 42.111/deme，第一 tick 后 2.111/deme（age0 被本 tick
繁殖覆盖并处理）。→ S1 明确 burn-in，或总种群/频率指标排除 age0，避免把首 tick 瞬态当成 chasing 信号。

**F7（低，HTML 记录路径）**：`PLAN.md:13`/`PLAN.md:35`/`PLAN.md:48` 要带时间轴 HTML，`PLAN.md:67` 又要求
`record_every=0` 防 OOM；需要明确记录机制。实测 `pop.run(1, record_every=0)` + `_native_stacked_state()`
可逐 tick 取频率（本次冒烟即此方式），但 `record_every=0` 单独不产生 history。→ PLAN 写明“逐 tick 步进、
只存 per-deme 频率（41×41×T float）”及内存上限。

**F8（低，产物跟踪）**：`.gitignore:54` 全局忽略 `*.csv`，`PLAN.md:60` 只列 HTML/PNG。若脚本输出 CSV 会被
静默忽略。→ 结果统一用 JSON（PLAN 已写 JSON），或明确需要入库的 CSV 例外；不要为此无说明地改全局 ignore。

### 27.3 核对点结论

- **核对点 1（覆盖 RSA/连续随机/homing/小范围/HTML/指标/重复/种子）**：**PASS（带 F1–F7 的精度要求）**。
  `PLAN.md:10-13`、`32-49`、`68-70`、`82-86` 均有对应项；但 `scaling_factor`、成功判据、seed/CI 细节需按
  F2/F3/F7 补实。
- **核对点 2（模型映射自洽/无过度承诺）**：**PASS（映射本身）+ 需修订 F1/F3/F4**。年龄/存活与
  `repro/m5_engine_probe.py:34-49`、`repro/m5_radial_reference.py:43-54` 一致；本次构建
  `expected_comp=42.2222`、`expected_survival_rate=1/90`，与 `D=40+5·(4/9)` 及 event-seq 平衡式自洽；
  迁移的丢界+行重归一化见 `src/natal/frontend/spatial/migration.py:424-435`；释放钩子（双性、首个成虫龄、
  `Op.add`）与 `m5_engine_probe.py:73-86` 一致。M5 仅作历史 golden，§26.1/`PLAN.md:16` 明确不再送 MATLAB
  定量复核——**M5 的 12/12 是自测，不是 evaluator 数值批准**。
- **核对点 3（产物布局与 `.gitignore`）**：**PASS**。忽略项均为未跟踪的探索/外部目录，已跟踪文件不受影响；
  `chasing_model/` 脚本与 `results/*.json` 不被忽略；HTML/PNG（及 CSV）按既有全局规则不入库。产物布局
  `PLAN.md:53-61` 合规。
- **核对点 4（高风险边界）**：**PASS**。`PLAN.md:72` 明确“若后续确需改 natal 产品代码，必须另行获批、
  默认关闭、走根 `EVALUATE.md` 与全量门禁 + 独立复核”；HEAD 未触碰产品代码。

### 27.4 残余风险 / 未检查项

- 未运行 M5 MATLAB golden、长跑 41×41 随机重复或 HTML 生成（本轮为计划评审，不冒充结果复核）。
- 未在共享 GPU 上运行 stochastic CSR/route A（F1 依据源码、现有测试与 CSR 行宽实测，非 GPU 运行）。
- `event-seq` 源码不在当前工作树（`event-seq/` 仅剩 `img/` 与 `__pycache__`，已 gitignore）；本次从
  `explo/event-seq` 分支读取。若该分支被删除，~40% 结论将失去脚本证据，只剩本地被忽略的图/JSON。
- F3 是科学风险而非已证明的不可能；若用户明确接受“先试 homing、失败再换机制”，可在 PLAN 中记录该决策，
  但仍建议保留可执行的失败判据，避免结果解释歧义。

### 27.5 批准前需要的最小修订

1. **F1**：S5 改为 CPU CSR，或把 route A/FFT 限定为确定性对照并注明“stochastic 需另行批准的产品改动”。
2. **F2**：写明 `scaling_factor` 与 K/初始态的同步关系。
3. **F3**：在 S1/S2 加 chasing 成功/失败判据与 fallback（或明确缩小目标）。
4. **F4**：钉死 3 型或 5 型及 carrier 定义。
5. **F5–F8**：按上述低风险项补 PLAN 文本。
完成后追加 §28 交接，evaluator 以 §29 复核；本轮不需要修改产品代码。

> 审查期间未改任何产品源码、`.gitignore` 或数据；仅追加本回执。工作树在追加前 clean，追加后仅
> `hexagon_spatial_test/EVALUATE.md` 有本回执改动，未 commit/push。

## §29 — §28 PLAN v2/v3 复核回执（evaluator）

### 29.0 裁定

**APPROVED（计划文本；附非阻断 notes）**

- 审查对象：`hexagon_spatial_test/chasing_model/PLAN.md` v3（§28 + §28.5）。§27 的 F1–F8 已闭环：
  - F1：`PLAN.md:5`/`:25`/`:78` 明确随机实验用 CPU CSR；route A/`fft` 仅确定性，stochastic 产品化另走根
    `EVALUATE.md`；S5 改为 101×101 CPU CSR（route A 仅确定性对照）。与 §27.1 的引擎事实一致（`fft` 拒
    stochastic；`size=7` CSR 行宽 48 > 32）。
  - F2：`PLAN.md:55` 明确 `scaling_factor s` 同时乘初始年龄向量与 `age_1_carrying_capacity`，并记录
    `s/K/expected_comp`；与 §27.1 的 `expected_comp=42.222`、`expected_survival_rate=1/90` 自洽。
  - F3：`PLAN.md:60-62` 增加成功/失败判据（`N_coexist`/`N_stall`）与 fallback。
  - F4：`PLAN.md:49` 钉死 5 型（等位 `[WT,Drive,Res]`，carrier=`{dd,dw,dr}`），复用 M1 映射。
  - F5：`PLAN.md:10` 写为 R-S-A → migration（tick 末尾）。
  - F6/F7/F8：`PLAN.md:48` burn-in + 排除 age0；`:92` 逐 tick `record_every=0` 只存 per-deme 频率；
    `:93` 统一 JSON。
- 非阻断 notes（建议在主 agent 下一轮文档修订中处理）：
  1. `PLAN.md:33` 仍写 `results/routeA_deterministic.json`；§30 实际产物是
     `results/deterministic_compare/summary.json`，请更新。
  2. `PLAN.md:78` 的 S5 “CPU CSR 与 route A 各自” 对 stochastic 仍有歧义：route A 随机尚未产品化，
     建议写成 “random = CPU CSR；route A = deterministic-only 对照 / 脚本原型”。
  3. `PLAN.md:60` “前沿半径随时间严格增长” 对随机模型过强；建议与 `:61` 的 `N_stall` 对齐，改为
     “窗口内趋势增长/不降”的统计判据。
  4. §2 的 5 型 homing 与 §30/§32/§34/§36 的 3 型最小原型不同；建议在 PLAN 明示 “V1/V2/V3 原型可用 3 型
     最小模型，S1+ chasing 用 5 型”，避免读者误以为已有结果覆盖 5 型。
- 残余：homing chasing 仍待 S1/S2 验证；F3 的决策门已写入计划，非阻断。

## §31 — §30 确定性 route A 复核回执（evaluator）

### 31.0 裁定

**APPROVED（数值结论可复现；附非阻断文档修订）**

### 31.1 独立证据

- 独立导入 `verify_routeA_deterministic.py`，只跑 `a15_k7_avd0.8`（40 tick，CPU + GPU + FFT repeat）：
  `non_egg_max_rel=3.9487902407915926e-07`、`adult_max_rel=4.720489622492812e-07`、
  `carrier_max_abs=2.3598730602980567e-07`、`final_per_deme_carrier_abs=3.0843980290962136e-06`、
  `final_state_max_abs=7.944858966268953e-07`、`fft_repeat_bit_identical=true`；与
  `results/deterministic_compare/summary.json` 的 a15 条目逐值一致。
- `summary.json` 与 §30.2 表格一致：4/4 配置 `all_passed=true`；`non_egg/adult rel ≤4.8e-7`、
  `carrier abs ≤2.2e-6`、`final state abs ≤2.9e-5`、FFT repeat 全 true。
- 初始/释放代码与 §30.1/§30.1b 一致：`age0=0`、uniform WT、仅雄 `Drive|Drive` age2、中心圆 r=3、
  `0.5×age2 WT 雄`；`verify_routeA_deterministic.py` 的 HTML 文本（:346）也写 homozygous。

### 31.2 非阻断 findings

- **F1（文档）**：`verify_routeA_deterministic.py:74/79` 的 docstring 写 “heterozygous `WT|Drive`”，但代码
  （`initial_states`，约 :105）与 §30.1/HTML 都是 **homozygous `Drive|Drive`**；修正 docstring。
- **F2（文档）**：`:175` 与 figure suptitle `:291` 仍说 “tick 0 all-age total includes the initial age-0 egg
  pool”，但本脚本初始 `age0=0`，§30.1b 已说明无此瞬态；删除或改写。
- **F3（数值文本）**：§30.2 `EVALUATE.md:638` 说 a31 的绝对差 “仅 5.1e-5”，但同表 `:634` 与
  `summary.json` 是 `2.88e-5`；统一为 `2.88e-5`。
- **F4（低）**：只检查了 FFT 重复逐位；CPU CSR 重复未做逐位检查。可补一次 CPU 重复，或在报告中说明
  确定性 CPU 单次结果即为基准。
- **F5（范围）**：3 型最小模型、无 hooks；§30.5#5 已披露，非阻断。V1 不覆盖 5 型/随机/hooks+fft。

## §33 — §32 三策略 v1 复核回执（evaluator）

### 33.0 裁定

**APPROVED（描述性比较数值可复现；附使用前需澄清项）**

### 33.1 独立证据

- 从 `results/strategies_v1/{small,large}/series.npz` 独立复算（selected deme 平均）：
  - small `drive@t40`：cpu `0.7374±0.0238`、m0 `0.7209±0.0242`、m1 `0.7425±0.0239`；
    `T50_selected` `35.95/36.25/36.05`；`T50_cov(0.5)` `37.65/37.60/36.70`。
  - large `drive@t40`：cpu `0.3423±0.0207`、m0 `0.3332±0.0279`、m1 `0.2377±0.0327`；
    `T50_selected` `44.2/44.4/45.0`；`T50_cov(0.5)` `47.4/47.5/45.3`。
  - 与 §32.2/§32.3 表一致；100 tick 终态三策略均固定（coverage=全场，front=角落距离）。
- `summary.json` 的 coverage/front/total/timing 与 §32.2/§32.7 一致。
- 独立调用当前 `compare_strategies.py::run_replicate`（small，seed=0，cpu/m0/m1）：`drive/non_egg/front/total`
  与 `series.npz` 对应 seed 的 **max abs=0.0**，说明归档数据可由当前脚本逐位复现。

### 33.2 非阻断 findings

- **F1（中，模型语义）**：`compare_strategies.py:192-206` 用 `_v_poisson` 生成总卵数，但 `PLAN.md:51`
  设定 `fixed_egg_count=True`；natal 在该开关下对每个交配对返回**确定**的 `total_lambda`
  （`rust/src/kernels/age_structured.rs:311-320`），不做 Poisson。原型还绕过 mating/sperm，用成虫基因型
  频率直接算后代。故 “CPU” 是 **NumPy 简化生命周期**，不是 natal 引擎；报告 §32.5#5 已说未做跨引擎校验，
  但应显式写出 “Poisson 卵数 vs fixed_egg_count=True” 的差异，并在跨引擎校验前不要把它当作 natal CPU
  参考。该差异对三策略共享，故不推翻“策略间瞬态差异”的比较结论。
- **F2（中，指标）**：`record()`（`:418-425`）用索引空间欧氏距离
  `sqrt((r-c)^2+(s-c)^2)` 计算 front，不是六边形度量；`event-seq` HANDOFF §3.2 已警告这会拉伸/扭曲波前。
  三策略共用同一 front，故相对比较仍成立，但 §32.2 的“front=9.9/21.2”与 C2ST 的 front 特征应标为
  “index-space”或改用 hex 度量。
- **F3（低，carrier 定义）**：`record()` 用 `drive`（`dd`）算 coverage/front，而 `PLAN.md:60` 写
  `carrier`；`compare_strategies.py:54` 的 `CARRIER=(1,2)` 未被 `record` 使用。请明确 “dd 作为保守代理”
  或改用 carrier。
- **F4（低，配对）**：`rng_life`/`rng_mig` 分离正确（`:442-443`），但“共享 lifecycle 噪声”只是同
  seed/同流；迁移差异导致状态分叉后，lifecycle 抽样序列也会分叉，配对降方差是部分的，不是严格 CRN。
  报告措辞宜改为 “same seed/stream”。
- **F5（低，路径）**：脚本写 `results/strategies` 与 `compare_strategies_{small,large}.html`（`:38-39`、
  `:644`），归档路径却是 `results/strategies_v1/` 与 `compare_strategies_v1_*`；请补 `--out` 或记录归档
  `mv` 步骤，否则复跑不会落到归档路径。

## §35 — §34 M2 复核回执（evaluator）

### 35.0 裁定

**APPROVED（M2 原型数值可复现；附需补充的全局误差证据）**

### 35.1 独立证据

- 从 `results/strategies_v2_r20_r10/{small,large}/series.npz` 复算 `drive@t40` 均值/SD，与 §34.3 表逐值
  一致（large m2 `0.549/0.412/0.151/0.062`；small m2 `0.808/0.763/0.651/0.566` 等）。
- 独立调用当前 `compare_strategies_m2.py::run_replicate`（small，seed=0，cpu/m0/m1/m2）：`drive/non_egg/front/total`
  与 v2 `series.npz` 对应 seed 的 **max abs=0.0**。
- 独立守恒检查（15×15，kernel 7，rate=1，单平面，CPU/M0/M1/M2 各一次，相同初始场）：
  `total before == after`，相对差 `0.0`（四策略全局总量均守恒）。
- 独立 M2 平均场检查（15×15，kernel 7，rate=1，R=200，单 tick）：M2 vs CPU 平均场
  `max rel = 5.82%`、`median rel = 1.42%`（总均值相同）。→ 全局局部均值并非无偏；报告未给该指标。

### 35.2 非阻断 findings

- **F1（中，守恒口径）**：M2 的 `tb ~ Gamma(mb)` + 全局 `total/Σtb` 缩放（`:422-467`）保证的是
  **全局总量**守恒，不是 per-source 守恒；`§34.6#1` 的“是否守恒”应写明“global only”，并说明块间可
  搬移质量。
- **F2（中，缺失指标）**：上述 `max rel 5.82% / median 1.42%` 的平均场误差未报告；§34.3 的 selected
  deme 均值会掩盖局部偏差。建议在 §34 增加 per-deme mean-field max/median rel diff（至少 1 tick），
  并报告块大小敏感性。
- **F3（低）**：块大小固定为 kernel 直径（7/11），未扫；§34.2/§34.7 已承认，非阻断。
- **F4（低，模型范围）**：仍为 3 型最小原型；PLAN v3 §2 的 chasing 模型是 5 型。建议在 §34/§36 明示。
- **F5（低，路径）**：脚本写 `results/strategies_v2`（`:38`），归档是 `results/strategies_v2_r20_r10/`；
  补 `--out` 或记录归档步骤。

## §37 — §36 C2ST 复核回执（evaluator）

### 37.0 裁定

**APPROVED（C2ST 数值/热图可复现；附统计口径说明）**

### 37.1 独立证据

- 从 `results/strategies_v3/{small,large}/series.npz` 重建 `_feature_matrix` 并调用
  `_c2st_matrices(seed=0)`：与 `c2st.json`/`summary.json` 的 accuracy/AUC/p **max abs diff = 0.0**
  （small/large 均如此）；对角线 self-split 复现 small `0.458/0.432/0.538`、large `0.426/0.568/0.522`。
- 特征敏感性：去掉 `front` 特征后，`m2 vs cpu` accuracy/AUC 不变
  （small `0.581/0.6184`；large `0.591/0.6042`），说明 §33 F2 的 index-space front 不驱动该结论。
- 热图下三角/对角线屏蔽与 `_plot_c2st`（`:747-760`）一致；`c2st.json` 与 HTML 的 `acc (AUC)` 一致。
- 独立调用当前 `compare_strategies_v3.py::run_replicate`（small，seed=0，cpu/m0/m1/m2）：`drive/non_egg/front/total`
  与 v3 `series.npz` 对应 seed 的 **max abs=0.0**。

### 37.2 非阻断 findings

- **F1（中，置换检验）**：`_c2st_pair`（`:706-718`）观测统计用 `_cv_scores(..., repeats=5)`（`:709`），
  但置换零分布用 `repeats=2`（`:714`）。严格置换检验应让零分布与观测统计使用同一 CV 管线；当前 p 值
  可比性略弱（p 分辨率 0.048 已在 §36.8 说明）。建议统一 repeats 或明确记录该差异。
- **F2（低）**：`front` 特征沿用 §33 F2 的 index-space 度量；本次 no-front 检查显示不影响结论，但仍应
  修正或标注。
- **F3（低，模型范围）**：仍为 3 型最小原型；PLAN v3 的 chasing 用 5 型。建议在 §36.1/§36.6 明示。
- **F4（低）**：对角线 self-split 用半样本（R=100 → 50/50），pairwise 用 100/100，不可直接比较；
  报告已说明对角线仅作 null 检查，非阻断。
- **F5（低）**：特征集/分类器敏感性未做；§36.7 已列为残余。R=100、置换 20 次；p 只作可检测性，
  effect size 优先的解读正确。

> 本轮仅复核复现计划/脚本/归档结果，未运行产品全量门禁（未触产品代码）；数值证据均来自当前工作树
> 的归档 `summary.json`/`series.npz` 与独立小范围复跑。未 commit/push。

## §39 — §38 A0（GPU M2 基线）复核回执（evaluator）

### 39.0 裁定

**APPROVED（A0 基线；附非阻断的计时/元数据说明）**

### 39.1 独立证据

- `py_compile`：`.venv/bin/python -m py_compile chasing_model/a0_baseline.py` → OK；
  `.venv/bin/ruff check chasing_model/a0_baseline.py` → `All checks passed!`。
- 环境边界：`.venv` `find_spec("torch") is None`；`/opt/conda/bin/python -c "import torch"` →
  `2.12.0+cu130`、CUDA `13.0`、`torch.cuda.is_available() == True`。HEAD `a98489a` 的 `git show --stat`
  只含 `EVALUATE.md`、`a0_baseline.py`、`results/gpu_fft/a0_cpu.json`、`a0_cuda.json`；无产品代码/依赖改动。
- CPU 模式独立复跑（把 `a0_baseline.RESULTS` 指向临时目录后调用 `run_cpu()`，不覆盖归档）：
  - small：wall `0.4776 s`（0.4650–0.4959），migration `0.4271 s`（89.4%），
    final non-egg `[47115.47518529081, 46940.4892922449, 47104.04617511389]`；
  - large：wall `0.9354 s`（0.9316–0.9413），migration `0.7820 s`（83.6%），
    final non-egg `[202030.62904631652, 202039.89675241604, 201647.0239656939]`。
  与 `a0_cpu.json`（small wall 0.467、large 0.927）在机器波动内一致；三 seed 的 final non-egg 与
  `results/strategies_v3/{small,large}/series.npz` 的 `m2_total[seed,-1]` **逐值完全一致**，确认 A0 CPU
  基线确实复用 v3 的 M2/seeds/configs，不是另一套实现。
- CUDA 模式独立复跑（`/opt/conda/bin/python`，临时 RESULTS）：RTX 5090 D v2、cap (12,0)、
  torch `2.12.0+cu130`；roundtrip f64 `1.33e-15`、f32 `7.15e-7`；timing
  `batch4 f32 0.0160 / batch48 f32 0.0158 / batch48 f64 0.0352 ms`；与 `a0_cuda.json` 一致。
- 归档未被覆盖：HEAD 只新增 `results/gpu_fft/a0_*.json`；`strategies_v1/`、`strategies_v2_r20_r10/`、
  `strategies_v3/` 无改动。

### 39.2 非阻断 findings

- **F1（低，计时口径）**：`a0_baseline.py::_timing`（约 :196-218）测的是 50 次
  `rfft2 → irfft2` Python 循环的**wall/iteration**，包含 kernel launch 与 Python 开销，不是 cuFFT 纯 kernel 时间。
  §38.3 的“纯变换耗时”建议改为“roundtrip wall（含 launch）”，或改用 CUDA events；当前结论
  “FFT 本身不是瓶颈”因此是保守的（真实 kernel 时间只会更小），不影响 A0 的方向判断。
- **F2（低，元数据）**：归档 `a0_cpu.json`/`a0_cuda.json` 的 `git_head=b4904e3`（父提交），而脚本/结果提交在
  `a98489a`；独立复跑时 `_git_head()` 会记录 `a98489a`。建议在 JSON 中同时记录 `git_dirty`/生成后提交说明，
  或提交后重跑生成，避免“结果对应哪个脚本版本”歧义。
- **F3（低，统计口径）**：CPU 基线是 3 个 seed 的**单 replicate**，min/max 不是 95% CI；用于计时量级合适，
  报告中已说明“不作为验收硬阈值”，不要把它当作种群统计的置信区间。
- **F4（低，可运行性）**：`run_cpu()` 通过 `import compare_strategies_v3 as v3` 依赖脚本目录在 `sys.path`；
  从仓库根以 `python chasing_model/a0_baseline.py --mode cpu` 可运行，但 `python -m` 方式可能失败。
  docstring 已给正确命令，非阻断。
- **F5（范围）**：A0 未测 CPU↔GPU 传输、GPU 端 M2 块抽样或分布等价；§38.3/§38.5 已明确，不能据此得出
  “GPU 加速已成立”的结论。

### 39.3 核对点结论

1. 两环境分别执行 + JSON 可复现：**PASS**（CPU/CUDA 均在临时目录独立复跑，量级/精度与归档一致）。
2. CPU 基线使用 v3 seeds/configs 且不覆盖 v1/v2/v3：**PASS**（final non-egg 与 v3 M2 逐值一致；HEAD 未改
   归档目录）。
3. 环境边界/无新增生产依赖：**PASS**（`.venv` 无 torch；conda torch 仅用于 `--mode cuda`；pyproject/uv.lock
   未变）。
4. `py_compile`/`ruff check`/无产品代码改动：**PASS**（见 39.1）。
5. 结论边界：**PASS**（A0 仅环境 + CPU 基线；未实现 GPU M2、未测传输、未做分布等价；§38.3/§38.5 已声明）。

### 39.4 残余

- A1：GPU M2 实现与 FFT/块抽样/传输分项计时；A2：CPU/GPU M2 确定性均值场与随机矩等价；A3：块大小/边界
  敏感性。随机 GPU FFT 的 natal 产品化仍须单独设计、用户批准、根 `EVALUATE.md` 与全量门禁。
- 本轮未运行产品全量门禁（未触产品代码）；未 commit/push。

## §41 — §40 A1（GPU M2 迁移算子）复核回执（evaluator）

### 41.0 裁定

**APPROVED（A1 原型；附非阻断的计时/校验说明）**

- 审查对象是**未提交工作树**：`gpu_migration.py`（新增）、`results/gpu_fft/a1_gpu_m2.json`（新增）、
  `a0_baseline.py`（仅新增 `git_dirty`/`script` 元数据）与 `EVALUATE.md`（§38.6/§40）。未发现 natal 产品代码
  或依赖改动；`a0_cpu.json`/`a0_cuda.json` 与 v1/v2/v3 归档未变。

### 41.1 独立证据

- `py_compile`：`.venv/bin/python -m py_compile chasing_model/gpu_migration.py chasing_model/a0_baseline.py`
  → OK；`.venv/bin/ruff check ...` → `All checks passed!`。
- conda 独立复跑（`/opt/conda/bin/python chasing_model/gpu_migration.py --out /tmp/...`，不覆盖归档）：
  - small：f64 CPU-sampling max abs `1.066e-14`、f32 `4.195e-6`、full GPU warm `0.69 ms/tick`；
  - large：f64 `1.954e-14`、f32 `6.373e-6`、full GPU warm `0.60 ms/tick`。
  与 `a1_gpu_m2.json`（f64 `1.066e-14/1.954e-14`、f32 `4.195e-6/6.373e-6`、warm `0.645/0.485 ms`）
  在机器波动内一致；f64 差异处于双精度舍入量级。
- 归档 JSON 复核：adult in/out 差 small `5.700e-3`（相对 `1.056e-7`）、large `7.655e-3`（相对 `3.314e-8`）；
  `finite=true`、`repeat_identical=true`；`outbound=cpu,sample=gpu` 的 CPU 出境 small `1.555 ms`、
  large `3.429 ms`，与 §40.3 的“CPU 出境瓶颈”结论一致。
- 代码核对：`m2_reference_cpu` 与 `M2GPU(sample_device="cpu", outbound_device="cpu")` 的
  outbound→FFT→block 抽样顺序一致（`gpu_migration.py:139-177`、`:302-390`）；f64 `1e-14` 的最大差
  支持“同序重写”的说法。完整 GPU 路径与 CPU 路径的 **RNG 不同**（torch vs NumPy），A2 再比分布。
- 环境：`.venv` 无 torch；conda torch `2.12.0+cu130`/CUDA `13.0`；未新增生产依赖。

### 41.2 非阻断 findings

- **F1（中，计时标签）**：`migrate` 在 `outbound_device="gpu"` 时把 H2D `as_tensor` 与
  `_outbound_gpu` 都计入 `transfer_in_s`（`:334-343`）。因此 §40.3 的 “transfer-in 0.174/0.131 ms”
  实际是 **H2D + GPU 出境采样**，不是纯传输；“sampling” 仅是块抽样。请拆分或改标签
  “transfer-in + GPU outbound”。这不影响总时间与“CPU 出境是瓶颈”的结论（后者由独立
  `outbound=cpu,sample=gpu` 路径支持）。
- **F2（中，计时稳健性）**：full GPU warm 值来自**一次** `migrate` 调用（`:436-441`），没有重复/CI；
  `~6.5×/16×` 是点估计。建议 warmup 后重复 50 次，报告 median/min/max（A0 的 `_timing` 已用 50 次，口径应一致）。
  另外，CPU M2 值来自 A0 的 100 tick 全模型，GPU 是合成成年平面的单 tick；输入分布/计数不同，加速比只能
  作量级参考（§40.4#5 已声明“不构成完整 chasing 加速结论”）。
- **F3（低，有效性检查）**：`negative_min = float(out_gpu.min())`（`:469`）因 age0/1 恒为 0 而必为 0，
  不能证明成人平面无负值；应改为 `out_gpu[:, :, NEW_ADULT_AGE:].min()` 并同样限定 `finite` 检查范围。
- **F4（低，语义）**：`_continuous_binomial_vec`（`:76-97`）与 `_outbound_gpu`（`:285-300`）没有 natal
  `continuous_binomial` 的 `p >= 1-EPS → n` 短路；`rate=1` 时仍会抽 Beta 并引入极小噪声。该近似沿用 v3，
  A1 的 CPU-sampling 对照使用同一近似，故等价性结论不受影响；但 A2/产品化对 natal 时应补短路或明确记录。
- **F5（低，范围）**：`gpu_migration.py` 硬编码 `Z=3, S=2, A=10`（3 型最小模型），PLAN v3 的 chasing
  是 5 型/6 ztype；A2 接回生命周期时需扩展或明确只验证 3 型。
- **F6（低，元数据）**：`a1_gpu_m2.json` 的 `git_head=a98489a`，而 `gpu_migration.py` 当前未跟踪/未提交；
  JSON 无 `git_dirty`/脚本哈希。建议提交后重跑或用与 `a0_baseline.py` 相同的 `git_dirty`/`script` 元数据。
- **F7（低，计时字段）**：`sample_device="cpu"` 路径把 `sampling_s`/`transfer_out_s` 置 0（`:364-370`），
  实际 CPU 抽样/下载仍发生；报告未用这些字段，但读 JSON 时易误解。可填实际值或标 N/A。
- **F8（低，cold 数值）**：§40.3 写“冷启动 transfer-in 可到 ~39 ms”，归档 JSON 的首次调用 small 为
  `33.7 ms`、large 首次已是 warm（`0.229 ms`）；建议统一为“small 首次调用 33.7 ms”或注明运行差异。

### 41.3 核对点结论

1. conda 复跑 + f64/f32 CPU-sampling 最大差与 JSON 一致：**PASS**（见 41.1）。
2. `m2_reference_cpu` 与 GPU CPU-sampling 的 RNG/块顺序一致：**PASS（同序重写；f64 1e-14）**；
   完整模型分布等价留 A2（报告已声明）。
3. warm/cold 计时/分解/CPU 出境瓶颈：**PASS（结论）+ F1/F2/F8 标签与稳健性修订**。
4. 未改产品代码/归档、未新增依赖、`py_compile`/`ruff`：**PASS**（见 41.1）。
5. 结论边界：**PASS**（合成数据、单次算子调用；未含生命周期、未做 CPU/GPU 随机分布等价、未做块大小/
   边界扫描；§40.4#5/§40.5 已声明）。

### 41.4 残余

- A2：接回生命周期，CPU M2 vs 完整 GPU M2 的确定性轨迹、随机矩（总非卵/逐 deme drive/方差比/lag 协方差），
  R=100–300；A3：块大小/边界敏感性。随机 GPU FFT 的 natal 产品化仍须单独设计、用户批准、根
  `EVALUATE.md` 与全量门禁。
- 本轮未运行产品全量门禁（未触产品代码）；未 commit/push；审查基于未提交工作树。

## §42 — 直接 stencil（方案 A）算子级 + NumPy CPU demo 交接（主 agent）

> 编号说明：本文件当前最新为 §41；本条按“接续当前最新”使用 §42。此前状态表预留的 §42（A1b）尚未写入本文件；
> 若需保留该编号，请将本条视为直接 stencil 子任务的 §42，并把后续 A1b 改为 §43+。

### 42.0 范围与风险分类

- 仅新增 `hexagon_spatial_test/chasing_model/scripts/` 下的复现脚本与 `results/` 产物；**未触碰 natal 产品代码**，
  未新增依赖，未改 `.gitignore`，未 commit/push。
- 风险分类：**局部代码修改**（复现脚本；不改 CPU 数值语义）。产品化仍未开始，后续接入 natal 属高风险。
- CPU 参考：目前是 `compare_migration_models.py` 的 **NumPy 镜像**（natal 连续 CSR 的算法重实现），尚未在本轮与
  natal Rust CPU CSR 做分布交叉核对。

### 42.1 改动清单

- 新增 `scripts/direct_stencil_migration.py`：
  - `Philox4x32-10`（与 `rust/src/gpu/kernels.rs` 常数/轮数一致）、`Marsaglia–Tsang` Gamma；
  - 逐边 64-bit edge id + 64-bit draw counter；`outbound` 与 `gamma` 用不同 role key；
  - `stencil_migrate_planes`：exact 路径，Pass1 求 `S_s`、Pass2 按固定 offset 顺序重放并 gather；
  - `stencil_migrate_planes_fast`：NumPy 双 Generator 串行重放，用于 NumPy CPU demo 提速；
    Pass1/Pass2 的 Gamma 形状序列相同，`replayed_gamma_sum` 与 `source_gamma_sum` 逐位一致。
- 新增 `scripts/direct_stencil_operator.py`：NumPy CSR vs stencil 算子级对拍，支持 `--mode exact|fast`。
- 新增 `scripts/compare_stencil_demo.py`：复用 `compare_strategies_v3.py` 的 R-S-A 生命周期、3 型最小模型、
  初始/释放/记录；只把 migration backend 换成 `cpu` / `stencil`，输出 HTML 时间滑块热图。
- 新增结果：
  - `results/direct_stencil_operator_exact.json`
  - `results/direct_stencil_operator_fast.json`
  - `results/stencil_demo/{small,large}/`（`summary.json`、`series.npz`、`selected_demes.png`、可选 `c2st.png`）
  - `results/html/compare_stencil_demo_{small,large}.html`
- 更新 `DirectStencilMigrationPlan.md` 的状态与 §9 里程碑/产物。

### 42.2 行为与原因

- exact 路径验证方案 A 的两遍语义：不保存逐边 `G`，Pass2 用同一 `(seed, role, edge)` 重放；固定 offset 顺序、
  无原子加。用于回答“语义是否成立”。
- fast 路径只用于 NumPy CPU demo：逐边 counter RNG 对 48 个平面、100 tick 的纯 Python/NumPy 运行偏慢；
  NumPy 双 Generator 保证 Pass1/Pass2 同序重放，且 `replayed_gamma_sum` 与 `source_gamma_sum` 的实测最大误差为 0。
  它不是未来 GPU 的逐边 counter 实现，报告中已明确标注。
- demo 不改生命周期、初始态、释放、记录与 3 型模型，只替换 migration，保持与既有 v3 demo 可比。

### 42.3 自测命令与结果（主 agent）

命令（仓库根目录）：

```bash
source .venv/bin/activate
unset PYTHONHOME PYO3_PYTHON PYTHONPATH
python hexagon_spatial_test/chasing_model/scripts/direct_stencil_operator.py --mode exact
python hexagon_spatial_test/chasing_model/scripts/direct_stencil_operator.py --mode fast
python hexagon_spatial_test/chasing_model/scripts/compare_stencil_demo.py --configs small --reps 10
python hexagon_spatial_test/chasing_model/scripts/compare_stencil_demo.py --configs large --reps 3
.venv/bin/ruff check hexagon_spatial_test/chasing_model/scripts/direct_stencil_migration.py \
  hexagon_spatial_test/chasing_model/scripts/direct_stencil_operator.py \
  hexagon_spatial_test/chasing_model/scripts/compare_stencil_demo.py
```

算子级（exact / fast）：

| 配置 | R | 均值相对差中位/最大 | 方差比中位 (p05–p95) | 总量 std | 负值比例 | 守恒误差 max | 同 seed 逐位 |
|---|---|---|---|---|---|---|---|
| small 15²/k7/avd0.8 | 100 | 2.17% / 10.47% (exact) | 0.971 (0.725–1.341) | ~1.0e-13 | 0 | ~1.1e-14 | true |
| small | 100 | 2.21% / 9.31% (fast) | 1.001 (0.692–1.334) | ~1.0e-13 | 0 | ~1.4e-14 | true |
| large 31²/k11/avd1.5 | 50 | 2.93% / 15.57% (exact) | 1.020 (0.607–1.661) | ~2.6e-13 | 0 | ~2.1e-14 | true |
| large | 50 | 2.89% / 16.11% (fast) | 1.020 (0.634–1.663) | ~2.6e-13 | 0 | ~1.4e-14 | true |

- Gamma 重放：exact 的 `gamma_draw` 同 key 两次逐位一致；fast 的 `replayed_gamma_sum` 误差 0（small/large 均 0）。
- lag 平均协方差（exact）：small 的差异约在 1–2 个 bootstrap SE 内；large 的 `(0,1)` 为 −0.147 ± 0.076（约 1.9σ），
  其余在 ~1–2σ 内，未发现系统性符号/量级错误；R 增大后应复核。

全模型 demo（NumPy CPU，rate=0.5，100 ticks）：

| 配置 | R | 最终覆盖 (0.5) | 最终总量 mean | 总量平均相对差中位 | drive 平均绝对差最大 | migration s/rep (cpu / stencil) |
|---|---|---|---|---|---|---|
| small 15²/k7/avd0.8 | 10 | 225 / 225 | 47196.8 / 47238.5 | 0.08% | 0.063 | 2.31 / 1.53 |
| large 31²/k11/avd1.5 | 3 | 961 / 961 | 202263.7 / 202050.1 | 0.07% | 0.108 | 19.75 / 14.90 |

- 两配置在 100 tick 均已固定；覆盖/前沿曲线在早期有可见时序差异（front 最大平均差 small 0.47、large 1.18），
  但最终覆盖一致。small 的 C2ST（R=10）下三角 accuracy≈0.51、AUC≈0.44、置换 p≈0.476，未见可分；
  large R=3 不足以做 C2ST，已跳过。
- 计时仅 NumPy-vs-NumPy，且 fast stencil 使用 NumPy Generator；**不构成 natal Rust CSR 的公平计时结论**。

### 42.4 请 evaluator 核对

1. `direct_stencil_migration.py` 的两遍语义：Pass1 `S_s`、Pass2 `X=O·G/S` 与 CPU CSR 的 Beta+Dirichlet 是否一致；
   边界（角/边/内部）、`O_s<=1`、`S_s<=EPS`、空行分支是否正确。
2. exact 路径的 Philox 常数/轮数、role 分离、64-bit edge/counter 是否与计划一致；同 key 重放是否逐位。
3. fast 路径是否被清楚标注为“NumPy demo 提速、非逐边 counter RNG”，且没有用它冒充 exact/GPU 结果。
4. 算子级分布指标与 MC 误差是否支持“分布一致”的结论；lag 协方差的 1–2σ 差异是否可接受。
5. demo 是否确实只替换 migration，生命周期/初始/释放/记录与 v3 一致；HTML 是否含两策略热图、时间滑块、
   4 deme 曲线与（small）C2ST。
6. `ruff`/`py_compile` 是否通过；`pyright` 不在仓库 include（仅 `src`），本轮未将复现脚本纳入 pyright 门禁。

### 42.5 残余风险 / 未检查

- **CPU golden 尚未与 natal Rust CPU CSR 交叉核对**：当前 golden 是 NumPy 镜像；下一轮应在小网格上做一次
  natal Rust vs NumPy mirror 的分布对照，再宣称 CPU 参考可信。
- large 的 R=3 仅用于视觉 demo，不能用于分布结论；small R=10 的 C2ST 也只作观察。
- fast 路径不是逐边 counter RNG；若用户要求 demo 也用 exact，可重跑（预计 small ~10×、large ~10× 迁移时间）。
- 模板压缩全阶梯、内存账本、Rust 公平计时、GPU 原型、chasing S1/S2、产品化均未做。
- 未运行产品全量门禁（未触产品代码）；未 commit/push。
