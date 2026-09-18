# EVALUATE.md — 主 agent ↔ evaluator 的沟通与审查通道（GPU 旁路 P0–P3）

> **约定（自本版起）**：本文件是主 agent 与独立 evaluator 之间的**唯一信息通道**。
> 主 agent 所有要传达给 evaluator 的内容，都写在「主 agent 交接」区（追加式）。
> evaluator 的裁定与发现都写回「evaluator 回执」区（追加式，不覆盖历史轮次）。
> 用户只负责通知 evaluator「按 `EVALUATE.md` 审查并回执」。
> 双方都**只追加、不改写对方已写入的内容**。
>
> evaluator 的职责定义见 `AGENTS.md`「验证时机与角色」与 `quality_checks_spec.md`。
> 同一主题冲突时以 `AGENTS.md` 和英文规范为准。

## 当前状态

| 项 | 值 |
|---|---|
| 分支 | `feat/gpu-merge-test` |
| 最近回执 | 第 1 轮：**NOT APPROVED**（§9，2 个阻塞项） |
| 主 agent 处理 | 已修复并自测（§10），**待第 2 轮独立复核** |
| 待 evaluator 动作 | 按 §11 的交接请求复核，并把第 2 轮结论写入 §12 |

## 0. 一句话目标

独立核对：**在 `feat/gpu-merge-test` 上新增的 CUDA 旁路是否正确、是否真的没有动 CPU golden reference、测试是否足以支撑结论**，并给出 `APPROVED` 或具体阻塞项。

## 1. 审查范围

分支：`feat/gpu-merge-test`（当前 HEAD 见 `git log -1`）。

被审查的提交（P0–P3）：

| 提交 | 内容 |
|---|---|
| P0-a / P0-b | CUDA 环境探针（`probe.rs`）、`cudarc` 绑定级探针（`cuda.rs`，NVRTC 实编译） |
| P1 | `context.rs`（设备/流）、`buffers.rs`（显存 RAII）、`layout.rs`（`(B,d…)↔(d…,B)` 转置） |
| P2 | `executor.rs`（设备状态 + §D5 显存预算守卫）、aging kernel |
| P3 | `kernels.rs` 中 density/equilibrium、survival、reproduction 三个确定性 kernel；`executor::tick` 编排；age-structured session 接线 + `enable_gpu()`；Python 透传 |

允许 evaluator 修改**测试**；**不得**修改产品实现。发现产品缺陷时，优先留下**已实际运行且失败**的回归测试（含路径、命令、预期/实际），交回主 agent 修复。

## 2. 需求与约束来源（先读）

- `AGENTS.md`：授权边界、风险分级、完成标准。
- `quality_checks_spec.md` / `_cn`：门禁命令、覆盖率、基线失败证据要求。
- `GPU_insert_PLAN.md` §2（D1–D7 已冻结决定）、§5.2（**精度分档**）、§5.3（阶段与验收）。
- `HANDOFF.md`：当前任务、编辑禁忌、已知基线失败。

硬约束（违反即阻塞）：

1. **CPU 路径一行都不能改**：现有 `rust/src/kernels/*.rs` 数值逻辑必须原样；CPU 是唯一 golden reference。
   - 被审查的 CPU 侧改动只允许「新增」：`sessions/age_structured.rs` 增加 `gpu` 字段、`enable_gpu/gpu_status/run_gpu`、`run_inner` 开头提前返回；`rust_backend.py` 增加 `enable_gpu/gpu_status`。
2. **`gpu` feature 默认关闭**：不带的 `cargo build` / `cargo test` 不受影响，测试数应为 **67**。
3. **精度验收分档**（§5.2）：
   - 纯搬运 kernel + 整数计数（< 2²⁴）→ 逐位一致；
   - 含算术 kernel（密度/存活/繁殖）确定性模式 → 相对误差 < 10×f32 eps ≈ 1.2e-6；
   - 随机模式 → 统计等价（本阶段尚未涉及）。
4. **不得静默回退 CPU**：显存不足 / 设备失败 / 模型不合格必须显式报错。
5. **未经明确要求不得 commit/push/改 `.gitignore`/新建 Markdown**（`EVALUATE.md` 本身是用户明确要求的例外）。

## 3. 环境准备（GPU 服务器）

```bash
cd /home/jovyan/node2/natal-core
source .venv/bin/activate
export RUSTUP_HOME="$PWD/.venv/rustup" CARGO_HOME="$PWD/.venv/cargo"
export PATH="$PWD/.venv/cargo/bin:$PATH"
export CARGO_TARGET_DIR="$PWD/.venv/cargo-target"
export LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
```

注意：

- `NATAL_GPU_REQUIRE` **默认等同 `1`**（服务器为主要场景）：缺卡/不符会硬失败；只有 CPU-only 主机才设 `NATAL_GPU_REQUIRE=0` 跳过。
- NVRTC 来自 conda 的 `nvidia` pip 包，必须靠上面的 `LD_LIBRARY_PATH`；`probe` 与 `cudarc` 都依赖它。
- 重编扩展（带 GPU）时必须写成 `maturin develop --features "gpu,extension-module"`——maturin 的 CLI `--features` 会**覆盖** pyproject 的 `extension-module`，否则会去链接不存在的 `libpython3.13`。
- 该 GPU 多租户共享；benchmark 前后应记录 `nvidia-smi` 快照，测出的加速比偏低先怀疑邻居。

## 4. 门禁与预期结果

| 命令 | 预期 |
|---|---|
| `python scripts/check_rust.py` | EXIT=0 |
| `cargo test`（默认 feature） | **67 passed**，0 failed |
| `cargo test --features gpu` | **96 passed**，0 failed（硬件断言真执行，非跳过） |
| `NATAL_GPU_REQUIRE=0 cargo test --features gpu` | 通过但硬件用例跳过 |
| `cargo clippy --features gpu -- -D warnings` | 通过 |
| `ruff check src demos` | 通过 |
| `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pyright` | 0 errors |
| `PYTHONUTF8=1 .venv/bin/pytest -q` | 3606 passed |
| `PYTHONUTF8=1 .venv/bin/python scripts/phase0_baseline.py --check` | `all scenarios bit-identical` |

L3（设备整 run 对照 CPU，panmictic 确定性）：可参考 `demos/gpu_spatial/` 的模型构造，或复现如下流程——建同一确定性单群体两次，CPU 跑 N tick 与 GPU（`backend.enable_gpu()` 后）跑 N tick，比较 `backend.state_snapshot()` 的 `ind/sperm`。期望相对误差与 f32 eps 同量级（实测 mode 0/2/3 分别为 8.3e-7 / 5.5e-8 / 4.6e-7）。

## 5. 建议的对抗性检查（自行设计，不要只复跑上面的命令）

1. **CPU 不变性**：`git diff` 证明 `rust/src/kernels/` 数值函数零改动；对 `sessions/age_structured.rs` 的改动确认只有新增 + 提前返回。
2. **对照输入自造**：不要只跑仓库自带用例。自行构造非平凡 `ind/sperm/eco/genetics`，逐个 kernel（aging / density / survival / reproduction / tick）与主机参考比，覆盖：
   - 整数计数（含 > 2²⁴ 时 f32 失真的情况）、含非整数率；
   - `n_ages` 边界（1、2、`MAX_AGES=64`）、`n_ztypes` 边界（1、`MAX_Z=32`）；
   - 空行/零总和的 density、`growth_mode` 0–4、`external_expected_eggs` 负值、`equilibrium_declared` 真/假、`has_sex_chromosomes` 真/假。
3. **拒绝路径**：`enable_gpu()` 对 `stochastic=true`、含钩子、`n_demes>1`（空间）、自定义 growth mode(≥5) 是否**显式报错**而非静默 CPU；扩展未带 gpu 时 Python 侧是否给出可操作 `RuntimeError`。
4. **不静默回退**：构造设备失败（如 `NATAL_GPU_REQUIRE`/显存/驱动不可用）时确认是报错，不是悄悄走 CPU。
5. **布局转置**：`batch_to_inner`/`batch_to_outer` 往返逐位；已知小例子的手工置换；长度不符的报错。
6. **显存预算守卫**：核对 `GpuExecutor::new` 用**实测可用显存**判断并在超预算时报错（可读代码 + 设计用例）。
7. **会话接线正确性**：
   - `run_inner` 设备分支是否真的提前返回、是否更新 `state_tick`、是否把最终状态下载为 f64；
   - 已知限制请**确认并记录**（不算阻塞，但要写进结论）：设备路径**不记录 history**；`enable_gpu` 时状态只上传一次，之后 Python 改参数/状态不会同步到设备；仅支持 panmictic、无迁移、无钩子、确定性。
8. **精度证伪**：尝试找出使相对误差超过 10×f32 eps 的确定性输入；若找到，给最小复现。
9. **门禁自跑**：evaluator 必须**亲自运行**第 4 节命令，不得引用主 agent 的结果。不得把未运行的测试说成已运行。
10. **稳定性**：共享 GPU 上若出现偶发失败，记录完整输出、重跑次数与复现率（开发中观察到过一次无法复现的失败）。

## 6. 已知基线 / 不要误判

- 既有基线失败见 `HANDOFF.md` §5.4（例如 `pytest` 的 28 个失败、`clippy --all-targets` 的 `neg_multiply`）；**本分支当前 pytest 为全绿 3606**，若你看到这些失败请先确认是否环境差异。
- `rust/target/`、`src/natal/_engine_rs*.so` 已加入 `.gitignore`；属构建产物，不入库。
- 本环境**没有**独立 `evaluator`/`adversarial-review` 之外的审查流程；`adversarial-review` 技能已装（`subagent_type` 已适配为 opencode 的 `general`），`numerical` 技能用于数值验证。

## 7. 结论格式

给出：

- **裁定**：`APPROVED` 或 `NOT APPROVED`。
- **逐条发现**：`severity`（high/medium/low）、`location`（`file:line`）、`evidence`（命令/输出/代码）、`recommendation`。
- **阻塞项**：附**已实际运行且失败**的回归测试（路径 + 命令 + 预期 + 实际），或静态检查的最小复现；未执行的测试不得声称复现失败。
- **自审 vs 独立**：明确哪些结论来自你独立运行，哪些来自代码阅读。
- **残余风险**：例如设备路径无 history、状态不自动重传、迁移/空间与随机模式未接入。

---

## 8. 可直接启用的 evaluator prompt

> 本文件顶部有「当前状态」表；末尾「主 agent 交接区」是最新一轮请求（evaluator 据此审查），
> 「evaluator 回执区」是回执位置。请先读这两处，再按下述要求执行。


把下面整段贴给一个**新的独立 agent 会话**即可（也可由主 agent 用 Task `general` 启动，但独立性较弱）：

```text
你是本仓库（/home/jovyan/node2/natal-core，分支 feat/gpu-merge-test）的独立 evaluator，负责对
新增的 CUDA GPU 旁路（P0–P3）做独立、对抗性的审查与验证。请先完整阅读：
  - EVALUATE.md（审查范围、约束、环境、门禁、对抗性检查清单、结论格式）
  - AGENTS.md 与 quality_checks_spec.md（角色职责、完成标准）
  - GPU_insert_PLAN.md §2/§5.2/§5.3 与 HANDOFF.md（已冻结决定、精度分档、编辑禁忌）

要求：
1) 独立核对需求、代码、测试、文档；不要只复跑命令，要自行设计对照与证伪用例（见 EVALUATE.md §5）。
2) 亲自运行 EVALUATE.md §4 的全部门禁，并记录真实输出；未运行的测试不得声称已运行。
3) 你只能修改测试，不能修改产品实现。发现产品缺陷时，留下一个“已实际运行且失败”的回归测试
   （路径 + 命令 + 预期 + 实际），交回主 agent 修复后再复核。
4) 确认“CPU golden reference 未被改动”与“gpu feature 默认关闭（默认 67 个测试）”两条硬约束。
5) 按 EVALUATE.md §7 输出：APPROVED / NOT APPROVED、逐条发现（severity/location/evidence/
   recommendation）、阻塞项的失败测试、哪些来自独立运行、残余风险。
6) 以 EVALUATE.md 末尾「主 agent 交接区」的**最新一轮请求**为本次审查目标；结论**追加**写入
   「evaluator 回执区」（第 2 轮起用 §12/§13…），并更新文件顶部「当前状态」表。
   只追加，不改写历史轮次与主 agent 的内容。

 环境变量与重建扩展的注意事项见 EVALUATE.md §3（NATAL_GPU_REQUIRE 默认强制；
maturin 需要 --features "gpu,extension-module"；需要 LD_LIBRARY_PATH 指向 conda 的 libnvrtc）。
若需要数值测试方法，加载 numerical 技能；对抗式多 agent 流程可加载 adversarial-review 技能。
```

---

## 9. 审查结论（evaluator 独立执行，2026-09-18，HEAD=`373fcbf`）

### 9.1 裁定：**NOT APPROVED**

发现 2 个可由行为测试复现的 in-scope 缺陷，均违反「GPU 确定性路径必须复现 CPU golden reference」。
架构 / 布局 / 显存守卫 / 门禁 / CPU 不变性 / 默认关闭均无问题。

evaluator **只改了测试**（`rust/tests/unit/gpu/executor.rs`，纯新增），未改任何产品代码。
产品缺陷修复目标 = 下面 9.2 的两个回归测试；**主 agent 请勿修改这两个测试**。

### 9.2 阻塞项（已实际运行且失败）

环境（在 `rust/` 下执行）：
```bash
cd rust
export RUSTUP_HOME="$PWD/../.venv/rustup" CARGO_HOME="$PWD/../.venv/cargo"
export PATH="$PWD/../.venv/cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/../.venv/cargo-target"
export LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
export PYO3_PYTHON=/opt/conda/bin/python PYTHONHOME=/opt/conda
```

**BLOCKER-1 — reproduction 提前返回不清 age-0，CPU 会清（severity: high）**
- 测试：`evaluator_reproduction_clears_newborns_when_no_recruits`
- 命令：`cargo test --features gpu evaluator_reproduction_clears_newborns_when_no_recruits`
- 预期：GPU `ind[age0]` == CPU == `0`
- 实际：`device 3 vs host 0 (|diff|=3, tol=1.2e-6)` → **FAILED**
- 位置：GPU `rust/src/gpu/kernels.rs:434`（`if (!has_any) return;`）与 `:441`
  （`if (total <= 1e-12f) return;`）直接返回，跳过 `:450-474` 的 age-0 写入；
  CPU `rust/src/kernels/age_structured.rs:378-384` 的 `fertilize` 提前返回后，
  `:503-506` **无条件**把 age-0 写成 `n_female/n_male = 0`。
- 触发：`eff_sum > 0` 但受精无任何 sperm 对（雌性 mating 率 0 / 无成年雌体 /
  selection fitness 全 0），且进入 reproduction 时 age-0 非零（tick 0 初始状态，
  或直接调用公开的 `reproduction_tick`/`tick`）。GPU 保留本应清除的新生个体，
  随后 survival/aging 会把它推进到 age 1，轨迹永久分叉。
- **建议修复**：在 `!has_any` 与 `total <= eps` 两个早退分支里，同样把 age-0
  写 0（与 CPU 的 `:503-506` 一致），即早退前先清 `ind[(0*A+0)*Z+z]` 与
  `ind[(1*A+0)*Z+z]`（对每个 `z<b>`）。

**BLOCKER-2 — mating 行阈值 GPU `1e-12` vs CPU `EPS=1e-10`（severity: medium）**
- 测试：`evaluator_mating_row_epsilon_matches_cpu`
- 命令：`cargo test --features gpu evaluator_mating_row_epsilon_matches_cpu`
- 预期：GPU `ind[age0]` == CPU == `0`
- 实际：`device 262.43997 vs host 0` → **FAILED**
- 位置：GPU `rust/src/gpu/kernels.rs:351`（`row_sum > 1e-12f`）；CPU 权威
  `rust/src/kernels/rng.rs:21` `EPS=1e-10`，判定在 `age_structured.rs:96`。
  同类差异还在 `kernels.rs:380/391/424/441/454/463`（CPU 对应
  `:193/:204/:329/:382/:389/:400`）。
- 触发：加权和落在 `(1e-12, 1e-10]`（如 `sexual_selection_fitness` 极小时），
  GPU 归一化该行、CPU 清零该行，产生「有后代 / 无后代」的定性差异。
- **建议修复**：把 GPU 源码里的 `1e-12f` 统一改为与 CPU `EPS=1e-10` 等价的阈值
  （cuDNN/CUDA 侧写 `1e-10f`），逐处核对上表。CPU 不可改。

### 9.3 硬约束核对（evaluator 独立运行）

1. **CPU golden reference 未改：确认。**
   `git diff adda656..81c3a14 -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空；
   `sessions/age_structured.rs` +128/-0、`rust_backend.py` +31/-0 纯新增；
   `phase0_baseline.py --check` → `all scenarios bit-identical`。
2. **`gpu` 默认关闭、67 测试：确认。**
   `cargo test` → 67 passed；`cargo test --features gpu` 作者原始 → 96 passed；
   `NATAL_GPU_REQUIRE=0 cargo test --features gpu` → 96 passed（硬件跳过）。

### 9.4 门禁自跑结果

| 命令 | 结果 |
|---|---|
| `python scripts/check_rust.py` | EXIT=0（需 `PYO3_PYTHON=/opt/conda/bin/python`，见 9.7） |
| `cargo test` | 67 passed, 0 failed |
| `cargo test --features gpu` | 作者原始 96 全过；加入 evaluator 用例后 **103 passed, 2 failed**（=9.2） |
| `NATAL_GPU_REQUIRE=0 cargo test --features gpu` | 96 passed（硬件跳过） |
| `cargo clippy --features gpu -- -D warnings` | 通过 |
| `ruff check src demos` | `All checks passed!` |
| `PYTHONUTF8=1 .venv/bin/pytest -q` | 3606 passed |
| `PYTHONUTF8=1 .venv/bin/python scripts/phase0_baseline.py --check` | `all scenarios bit-identical` |
| pyright | 0 errors |

额外独立验证（非仅复跑）：
- **L3 CPU↔GPU 端到端**（Python 公共 API，Beverton–Holtz 确定性单群体，5 tick）：
  `max_rel=1.68e-7`，tick 一致 → PASS。
- **拒绝路径**：stochastic→ValueError；含钩子→ValueError；
  `CUDA_VISIBLE_DEVICES=""`→`RuntimeError(CUDA_ERROR_NO_DEVICE)`，
  失败后仍可 CPU 运行（无静默回退）→ PASS。
- evaluator 新增 18 个对抗用例覆盖作者未覆盖分支：declared equilibrium、
  mode 4(ricker)、external eggs、sex chromosomes、`new_adult_age=3`、`Z=1`、
  `MAX_AGES=64`/`MAX_Z=32` 含边界与超界拒绝；除 9.2 外全部通过
  （整 tick 容差已收紧到 1.2e-6）。

### 9.5 非阻塞发现

| severity | location | evidence | recommendation |
|---|---|---|---|
| medium | 原 `executor.rs` 测试 | 单核用 `1e-5`、整 tick 用 `1e-4`，远松于冻结标准 `1.2e-6`；evaluator 收紧到 `1.2e-6` 全过 | 收紧容差到 1.2e-6 并说明理由 |
| medium | Python 集成 | `enable_gpu()` 后 `pop.run(3, record_every=1)` 的 `pop.history` 长度为 0（CPU 为 4），无提示 | 暂不支持历史时显式报错或文档化 |
| medium | `rust/src/gpu/executor.rs:219-286` | `density_scaling` 漏校验 `carrying_capacity/eggs_per_female/sex_ratio/external_expected_eggs/low_density_growth_rate/growth_mode` 长度，kernel 按 `b` 索引（`kernels.rs:92-93,118,138,144,156,172`），可越界读未初始化显存 | 补齐 `expect_len` |
| low | `executor.rs:247-255`、`sessions/age_structured.rs:269` | 只拒 `growth_mode>=5`，未拒负值；负值 CPU→1.0，GPU→`1.0*expected_surv` | 拒绝 `<0` |
| low | `kernels.rs:455-457`、`executor.rs:519-528` | 性染色体标志长度未校验，依赖 `Blueprint::validate` | 加长度断言 |
| low | `kernels.rs:406-408` vs `age_structured.rs:277` | NaN `eggs_per_female`：CPU→0，GPU 保留 NaN | 统一 NaN 语义 |
| low | `.gitignore` | P0–P3 期间被改（加 `rust/target/`、`_engine_rs*.so`），违背「未经要求不得改 `.gitignore`」；内容合理，§6 已列为构建产物 | 流程偏差，记录即可 |

### 9.6 残余风险（含已知限制，不计阻塞）

设备路径不记录 history（静默丢失）；状态仅在 `enable_gpu` 时上传一次
（参数每次 `run_gpu` 会重新读取，故参数改动生效；不同步的是宿主直接改状态）；
仅支持 panmictic、无迁移、无钩子、确定性；`stop_if_*` 因含钩子被显式拒绝；
空间/离散后端类没有 `enable_gpu` 入口（无静默回退，也无显式错误）；
显存预算未计入 ecology 上传与模块开销；f32 在计数 >2²⁴ 或长链归约
（`offspring_acc`、`virgins = n_female - mated`）理论上可能超 1.2e-6（未找到具体越界输入）。

### 9.7 环境偏差（非本分支引入）

`.venv` 的 Python 3.13 缺共享库 `libpython3.13`，直接 `cargo test --lib`
链接失败（`unable to find library -lpython3.13`），默认 feature 同样复现，
与本 GPU 改动无关。Rust 门禁改用 `PYO3_PYTHON=/opt/conda/bin/python`
（Python 3.11，含 `libpython3.11.so`）完成；Python 门禁仍在 `.venv` 3.13 下执行。

### 9.8 主 agent 修复清单与复核流程

1. 只改产品代码：`rust/src/gpu/kernels.rs`（BLOCKER-1 清 age-0、BLOCKER-2 阈值），
   必要时 `rust/src/gpu/executor.rs` 补齐校验；**不要动 evaluator 的测试**。
2. 自测：`cargo test --features gpu`（目标 105 passed / 0 failed，含 evaluator 用例）、
   `cargo test`（67）、`cargo clippy --features gpu -- -D warnings`、`cargo fmt --check`。
3. 回交 evaluator，由 evaluator 独立复核 9.2 两个目标及受影响检查后出最终结论。
4. 复核通过前，本分支维持 **NOT APPROVED**；不得以削弱/删除 evaluator 测试的方式转绿。

---

## 10. 主 agent 修复记录（待 evaluator 复核）

修复提交目标：使 9.2 两个回归测试转绿；**未改动 evaluator 的任何测试**。

### 10.1 产品代码改动

| 目标 | 文件 | 改动 |
|---|---|---|
| BLOCKER-1 | `rust/src/gpu/kernels.rs` | `reproduction` kernel 删除 `!has_any` 与 `total <= eps` 两个提前返回；改为**无条件**写入 age-0（`n_g<=EPS` 时写 0），与 CPU `reproduction` 的 `:503-506` 一致。顺带修正 `eggs_per_female` 的 NaN 语义（`!(epf>0)` → 0，对齐 CPU `.max(0.0)`） |
| BLOCKER-2 | `rust/src/gpu/kernels.rs` | 把 6 处硬编码 `1e-12f` 阈值统一为 CPU `EPS=1e-10` 等价：`row_sum`、`removed`/`mated`、`n_new`、`n_total`、`n_g`、性染色体 `denom` |
| 非阻塞 medium | `rust/src/gpu/executor.rs` | `density_scaling` 补齐 `external_expected_eggs` / `carrying_capacity` / `eggs_per_female` / `sex_ratio` / `low_density_growth_rate` / `growth_mode` 的长度校验（消除越界读未初始化显存） |
| 非阻塞 low | `rust/src/gpu/executor.rs`、`rust/src/sessions/age_structured.rs` | 拒绝负 `growth_mode`（`!(0..=4).contains(mode)`），不再把负值当 1.0/自定义曲线 |
| 非阻塞 low | `rust/src/gpu/executor.rs` | `reproduction_tick` 增加性染色体标志长度校验（`female_only`/`male_only` 各 `Z`） |

**明确未改**（非阻塞，保留为已知限制）：设备路径不记录 history（静默为空 → 见 9.5，建议后续显式报错或文档化）；`enable_gpu` 后宿主直接改状态不自动重传；空间/离散后端无 `enable_gpu` 入口。

### 10.2 主 agent 自测证据（非独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **105 passed, 0 failed**（含 evaluator 9 个用例，`evaluator_` 全绿） |
| `cargo clippy --features gpu -- -D warnings` | 通过 |
| `cargo fmt -- --check` | 通过 |
| `scripts/check_rust.py` | EXIT=0 |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 重编扩展（`--features "gpu,extension-module"`）+ Python L3（mode 0/2/3，20 tick） | max_rel ≤ 8.3e-7，ind/sperm 一致 |

### 10.3 状态

以上为**主 agent 自测**。按 9.8 流程，需 evaluator **独立复核** 9.2 两个目标及受影响检查后再出最终结论。复核通过前，本分支仍为 **NOT APPROVED**。

---

# 主 agent 交接区（追加式；主 agent 写，evaluator 据此审查）

## 11. 第 2 轮交接 — 请求独立复核阻塞项修复

- 日期：2026-09-18
- 交接时 HEAD：见 `git log -1`（上一轮回执为 `373fcbf` 之后的工作树）
- 本轮工作树改动：
  - 产品代码（主 agent）：`rust/src/gpu/kernels.rs`、`rust/src/gpu/executor.rs`、`rust/src/sessions/age_structured.rs`
  - evaluator 测试（**主 agent 未改**）：`rust/tests/unit/gpu/executor.rs`
  - 文档：`EVALUATE.md`
- 修复内容与自测证据：见 §10（BLOCKER-1 清 age-0、BLOCKER-2 阈值对齐 `EPS=1e-10`；另补 density 长度校验、拒绝负 growth_mode、性染色体标志长度校验；NaN eggs 语义对齐）。

### 请复核的内容（按优先级）

1. **9.2 的两个目标**是否真的转绿，且修复是产品语义层面的、不是绕过测试：
   - `cargo test --features gpu evaluator_reproduction_clears_newborns_when_no_recruits`
   - `cargo test --features gpu evaluator_mating_row_epsilon_matches_cpu`
   （`EVALUATE.md` §9.8 要求不得改动 evaluator 测试来转绿。）
2. 修复**未引入回归**：`cargo test` → 67；`cargo test --features gpu` → 105（含 evaluator 9 个用例外全绿）。
3. 核对 BLOCKER-2 的**全部**阈值点已与 CPU `EPS=1e-10` 一致，且没有把 CPU 的随机分支语义误接到确定性分支（本阶段只支持 `stochastic=false`）。
4. 主 agent 新增的校验（density 列长、负 growth_mode、性染色体标志长度、NaN eggs）是否正确且无副作用。
5. 如仍有异议，请给出**已实际运行且失败**的回归测试（路径 + 命令 + 预期 + 实际），交回主 agent 修复。

### 环境（与 §3 一致）

```bash
cd /home/jovyan/node2/natal-core
export RUSTUP_HOME="$PWD/.venv/rustup" CARGO_HOME="$PWD/.venv/cargo"
export PATH="$PWD/.venv/cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/.venv/cargo-target"
export LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
# Rust 测试链接需要 conda 的 libpython（.venv 的 3.13 无共享库）：
export PYO3_PYTHON=/opt/conda/bin/python PYTHONHOME=/opt/conda
```

> 注意：`PYTHONHOME` 只用于 cargo/pyo3；运行 Python 门禁（pytest/phase0）时请 `unset PYTHONHOME PYO3_PYTHON`，否则会破坏 `.venv` 的 3.13 解释器。

### 回执方式

请在下方「evaluator 回执区」**追加**「第 2 轮」结论，格式沿用 §9：裁定（APPROVED / NOT APPROVED）、逐条发现、独立运行证据、残余风险；并更新文件顶部的「当前状态」表（只改状态，不改他人内容请改表格数值）。

---

# evaluator 回执区（追加式；evaluator 写，主 agent 据此行动）

> 第 1 轮结论见上方 **§9**（已有内容）。为保持时间顺序，**第 2 轮及以后请追加到本区末尾**，
> 使用 §12、§13… 编号；不要改写上面的历史轮次。

## 12. 第 2 轮结论（待 evaluator 填写）
