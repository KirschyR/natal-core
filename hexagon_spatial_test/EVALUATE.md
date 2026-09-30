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
| 最近回执 | §12（M4 路线 A §11）= **APPROVED**（仅原型/设计） |
| 最近回执 | §18（S2b 设备级 FFT 内核）= **APPROVED**（已吸收 §18.4 的 F1–F3，见 §17.6） |
| 待回执 | —（下一交接为 S2c 完成后的 §19） |
| 主 agent 处理 | S0+S1+S2a+S2b 闭环；进入 **S2c**（executor/session/前端接线 + 预算） |
| 待 evaluator 动作 | —（S2c 交接后） |

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
