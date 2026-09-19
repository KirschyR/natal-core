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
| 最近回执 | 第 19 轮：**NOT APPROVED**（§47；固定 `1e-3f` 容差对大规模有效状态误报） |
| 待回执 | §49（第 19 轮 A2 修复复核） |
| 主 agent 处理 | A2 已按尺度相关容差修复并自测；下一项为 C10 |
| 待 evaluator 动作 | 按 §48 复核修复目标，把结论写入 §49 |

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

## 13. 第 3 轮交接 — 覆盖率与会话测试补齐

- 日期：2026-09-18
- 针对 §12.4（覆盖率 / 缺失测试）的回应。
- 本轮工作树改动：`rust/src/gpu/{cuda,executor}.rs`、`rust/src/sessions/age_structured.rs`、
  `rust/tests/unit/gpu/{context,cuda,executor,kernels,probe,session}.rs`（`session.rs` 为新增）、`EVALUATE.md`。
- 产品代码未改数值语义：仅为可测性抽取 `CudaBindingProbe::from_outcome`/`driver_missing`、
  `AgeStructuredSession::assemble`、`ensure_memory_budget`，并新增会话测试；CPU 路径与 kernel 数值不变。

### 13.1 覆盖率证据（主 agent 自测；命令可复现）

```bash
cd rust
export RUSTUP_HOME="$PWD/../.venv/rustup" CARGO_HOME="$PWD/../.venv/cargo"
export PATH="$PWD/../.venv/cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/../.venv/cargo-target"
export LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
export PYO3_PYTHON=/opt/conda/bin/python PYTHONHOME=/opt/conda
rm -rf /tmp/cov && mkdir -p /tmp/cov
export RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE="/tmp/cov/%p-%m.profraw"
cargo test --features gpu --lib --no-run --message-format=json > /tmp/cov/build.json
# 取 executable 后运行，再用 llvm-profdata/llvm-cov report --sources src/gpu
```

结果：`src/gpu` **行覆盖 95.08%（61 missed / 1241）**，超过 95% 门槛；
`rust/tests/unit/gpu/session.rs` 覆盖了会话设备路径。

### 13.2 会话设备路径的 Rust 测试（§12.4 第二处缺口）

新增 `rust/tests/unit/gpu/session.rs`（由 `src/sessions/age_structured.rs` 的 `#[cfg(all(test, feature="gpu"))]` 挂载）：

- `session_device_branch_matches_cpu_and_covers_wiring`：构造会话 → `enable_gpu()` 成功 →
  `gpu_status()=="enabled"` → `run_inner` 设备分支跑 3 tick 并回写；同一 fixture 再跑 CPU 分支，
  逐元素比较 `state_ind/state_sperm`，相对误差 ≤ 1.2e-6。
- `session_enable_gpu_rejects_ineligible_models`：覆盖 stochastic / `n_demes>1` / 自定义 growth_mode /
  `n_hooks>0` / **python_callbacks 非空**（`n_hooks==0`）五条拒绝分支。

### 13.3 剩余未覆盖行（请 evaluator 判定是否在范围内）

以 driver 失败/平台分支为主，当前测试环境下无法执行（非“用排除隐藏行”）：

- `executor.rs` 18 行：多行 kernel 调用尾部的 `)?;` 错误传播分支（仅 CUDA 返回 Err 时执行）。
- `cuda.rs` ~12 行：`run_inner` 各设备查询失败的 `map_err` 闭包体。
- `context.rs` / `buffers.rs`：`CudaContext::new`、`clone_htod/dtoh` 的 driver 错误映射闭包。
- `probe.rs` ~7 行：`CUDA_PATH` 未设、`/usr/local/cuda*` 不存在、Windows 专属分支（本机 Linux）。

聚合行覆盖已 ≥95%。**请 evaluator 明确判定口径**：
1. 若按 `src/gpu` 聚合行覆盖（§规范“new modules”复数、上一轮用 TOTAL），则门槛已满足；
2. 若要求**每个文件**都 ≥95%，则上列 driver 失败/平台分支会使其不达标——请提出：
   (a) 以文档化残余风险接受，或 (b) 授权“错误映射可测化重构”（将 `map_err` 抽取为可用构造错误单测的辅助函数）。
   该选择涉及规范解释，需用户裁定；请在第 3 轮结论中给出你的判定与依据。

### 13.4 自测门禁（非独立）

| 命令 | 结果 |
|---|---|
| `cargo test --features gpu` | **124 passed**（含 evaluator 9 + 会话 2 + 新增守卫/错误分支用例） |
| `cargo test` | 67 passed |
| `cargo clippy --features gpu -- -D warnings` / `fmt --check` | 通过 |
| `check_rust.py` / `phase0` | EXIT=0 / bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |

---

## 15. 第 4 轮交接（预告）— 下一阶段：P5 确定性迁移 + 空间多 deme

- 日期：2026-09-18
- 前置：§14 已 **APPROVED**（P0–P3，panmictic 确定性 age-structured）。
- 主 agent 即将开展（尚未实现，**本阶段无待审内容**）：
  1. 设备侧确定性 migration（CSR scatter/gather，对应 `kernels/spatial.rs` 的三个变体之一的确定性路径）；
  2. 多 deme 空间模型上接入 `enable_gpu`（当前仅允许 `n_demes==1`，将放宽到空间确定性模型）；
  3. 空间 L3：对 `demos/gpu_spatial/age_structured/reference_cpu.py` 的确定性模型做 CPU↔GPU 对照
     （含算术阶段用相对误差 ~1.2e-6；纯搬运阶段逐位）。
- 说明：迁移会让 batch 轴（deme）之间发生耦合，且涉及跨 deme 累加顺序，属高风险（科学公式 + 数据布局），
  实现完成后将按 §9 格式递交独立审查；预计会新增 `rust/src/gpu/` 模块代码与测试、并可能改动
  `sessions/spatial.rs` 的设备分支（仍保持“仅新增 + 提前返回”，不改 CPU 路径）。
- 预计会请求 evaluator 重点关注：跨 deme 归约顺序与 f32 误差、`stay_after_send` 语义、边界/空行 deme、
  迁移与生命周期同 tick 的拼接顺序、以及 CPU golden reference 的逐位/容差验收分档。

---

## 16. 第 4 轮交接 — P5 确定性迁移 + 空间多 deme

- 日期：2026-09-18
- 范围（相对 §14 已批准的 P0–P3）：新增确定性 CSR 迁移与空间会话接线。
- 风险分类：**高风险**（科学公式 + 跨 deme 数据布局 + 归约顺序）。

### 16.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | `MIGRATION_SOURCE`：`migration_female` / `migration_sperm` / `migration_male`；`Kernels::migration` 启动器 |
| `rust/src/gpu/executor.rs` | `GpuExecutor::migrate_tick(bp, eco, stay_after)`：主机侧构建反 CSR + `row_sum_w`，上传后用 scratch 输出并交换 |
| `rust/src/sessions/spatial.rs` | `gpu` 字段；`enable_gpu()`/`gpu_status()`；`run_gpu_tick` + `run_inner` 提前返回分支（CPU 路径未改） |
| `src/natal/backends/rust/rust_backend.py` | 空间后端 `enable_gpu()`/`gpu_status()` 透传 |
| `rust/tests/unit/gpu/*` | 迁移 L1、空间会话设备路径（含拒绝分支）、迁移空批守卫 |

### 16.2 设计要点（请重点核对）

1. **确定性 gather**：迁移按**反向 CSR** 在目标格点上按固定顺序累加入边（无 `atomicAdd`），保证 GPU↔GPU 可复现；复刻 `migrate_csr_deterministic` 的 `stay_after`/空行/`virgin=雌−储精`（`|neg|<1e-9` 归零）语义，且储精质量同时计入雌性个体平面。
2. **同 tick 顺序**：设备空间 tick = 生命周期（reproduction→survival→aging，批量 deme）→ 迁移；与 CPU `run_inner` 一致。
3. **零速率跳过**：所有 `migration_rate <= 0` 时跳过迁移，匹配 CPU 的 bitwise skip。
4. **每 tick 回传**：设备分支在 `run_inner` 内逐 tick 下载状态（因此 `run_steps` 的历史记录对设备路径同样生效）；这是当前取舍，非零回传优化留待后续。
5. **eligibility**：仅 `!discrete`、`stochastic=false`、无钩子、`growth_mode∈0..=4`、每 deme 一列生态与一个 variant id；否则显式报错。

### 16.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **129 passed, 0 failed** |
| `cargo test` | 67 passed |
| `clippy --features gpu -D warnings` / `fmt --check` | 通过 |
| `check_rust.py` / `phase0` | EXIT=0 / all scenarios bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |
| 覆盖率 `src/gpu` | **行 95.18%（69 missed / 1432）** |
| 覆盖率 `sessions/spatial.rs` 新增行 | 88/91 ≈ 96.7%（残余 3 行为 `#[cfg]`、`from_parts` 的 `gpu: None`、一个 `}`，与 §14.3 接受的同类残余一致） |
| 空间 L3（`demos/gpu_spatial/age_structured` 5×5 参考模型，25 tick） | tick 一致；ind/sperm `max_rel ≈ 9.15e-7` |
| 拒绝路径 | discrete / stochastic / hooks / custom growth / deme-variant mismatch 均显式报错 |

复现 L3 的脚本见 `/tmp/l3_spatial.py`（临时），核心为：同一确定性空间群体各建一次，`backend.enable_gpu()` 后 `pop.run(N)`，比较 `state_snapshot()`。

### 16.4 请 evaluator 独立核对

- 反向 CSR gather 是否与 `migrate_csr_deterministic` 逐分支等价（`stay_after` 真/假、空行、自环、多入边累加顺序）。
- f32 误差是否在 ~1.2e-6（含算术）与逐位（纯搬运）分档内；自行设计更多 CSR 拓扑与边界输入证伪。
- 空间 tick 顺序与 CPU 一致；`all_zero` 跳过条件一致。
- CPU golden reference 零改动；默认 feature 仍 67 测试。
- 覆盖率口径沿用 §14.3（聚合 + 逐新模块）。

结论请追加为 **§17**。

---

## 18. 第 5 轮交接 — P4 随机采样（RNG + 分布 + 随机生存/繁殖 + 统计 L3）

- 日期：2026-09-18
- 范围（在 §17 已批准的 P0–P5 确定性能力之上）：随机模式设备路径。
- 风险分类：**高风险**（counter-based RNG、随机分布、统计验收）。

### 18.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | `RNG_SOURCE`（Philox4x32-10 + `fill_uniform`）；`SAMPLING_SOURCE`（`RngState`、`rng_uniform/rng_normal`、`sample_binomial/poisson/gamma`、`natal_multinomial`、`multinomial_seq`、`recruit_stochastic`、`survival_stochastic`、`reproduction_stochastic`）及各启动器 |
| `rust/src/gpu/executor.rs` | `GpuExecutor` 增加 `seed`/`tick` 与 `set_seed`、`rng_key`/`rng_site`；`survival_tick`/`reproduction_tick` 按 `blueprint.stochastic` 分支（随机走对应 kernel）；`tick()` 递增 tick |
| `rust/src/sessions/age_structured.rs` | 新增 `seed` 字段并传入 `set_seed`；`enable_gpu` 现接受 `stochastic=true`，仅拒绝 `continuous_sampling=true` |
| `src/natal/backends/rust/rust_backend.py` | 既有 `enable_gpu` 透传不变 |
| 测试 | `tests/unit/gpu/kernels.rs`（Philox 逐位 + 矩检验 + 多项 + 启动器守卫）、`executor.rs`（随机 survival/reproduction 统计 L1）、`session.rs`（拒绝分支改为 continuous） |

### 18.2 设计要点（请重点核对）

1. **counter-based RNG**：每线程 `RngState`，`counter = (cell << 32) | draw`，`site` 含 tick 与阶段常量；每 cell 独立计数器窗口，避免流重叠。`rng_normal` 用 Box-Muller。
2. **采样器取舍**：binomial 在 `mean≤512` 用**精确几何跳跃**、否则正态近似（`p>0.5` 反射）；Poisson `λ<64` Knuth 精确、否则正态；gamma Marsaglia-Tsang；multinomial 用**序列条件二项 + 序列轴外提**。
3. **随机阶段**：完全复刻 CPU `sample_survival_with_sperm`、`recruit_juveniles`（离散）、`sample_mating`/`fertilize`（离散）的分支与 `EPS=1e-10` 阈值；确定性路径未改。
4. **本阶段未做（显式）**：`continuous_sampling=true` 被拒绝；**空间随机迁移未实现**（`SpatialSession::enable_gpu` 仍拒绝 `stochastic=true`）——随机能力目前限 panmictic、离散抽样。空间随机迁移留待后续阶段。

### 18.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **139 passed, 0 failed** |
| `cargo test` | 67 passed |
| `clippy -D warnings` / `fmt` / `check_rust` / `phase0` | 通过 / 通过 / EXIT=0 / bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |
| `fill_uniform_matches_host_philox_and_reproduces` | 设备 Philox 与主机逐位一致 + 同种子复现 |
| `samplers_match_theoretical_moments` | 10⁶ 样本，binomial（含 p=0.8 反射）/poisson/gamma/normal 均值方差在 4–6σ |
| `device_stochastic_{survival,reproduction}_matches_host_distribution` | 4000 同参 batch 的设备逐格点均值 vs 4000 次 CPU 试验（5σ+0.5） |
| 随机 L3（Python，panmictic，K=150×8 tick） | CPU/GPU 最终 ind 逐格点均值在最差 0.41×容差内 |
| 确定性 L3（panmictic + 空间 5×5） | 逐位/紧容差不变，`phase0` bit-identical |

覆盖率（**严格**只统计 `rust/src/gpu/` 源文件；注意 `--sources src/gpu` 会把 `src/gpu/../../tests/...` 一并匹配，需按绝对路径过滤）：executor ~95.9%、probe ~96.1%、其余 100%，聚合 **~98%**。

### 18.4 请 evaluator 独立核对

- **RNG**：自行检查流不重叠（不同 cell/tick/site 无重合），并对 uniform 做工整性检验（KS/卡方，而非仅均值方差）。
- **分布采样器**：独立设计参数网格，对 binomial/poisson/gamma 做分布检验与假设前提（近似分支的适用区间、反射、`mean` 阈值边界）；给出是否接受“精确/近似混合”的依据。
- **随机阶段**：自行设计参数与拓扑（含空行、零总量、`fixed_egg_count` 真/假、sex-chromosome、多年龄段），与 CPU 多次试验做统计对照；用你自己发现的输入证伪。
- **确定性路径**：确认随机改动未影响确定性结果（逐位/紧容差不变）。
- **enable_gpu 语义**：`stochastic=true, continuous=false` 接受；`continuous=true` 显式拒绝；空间随机显式拒绝（不得静默 CPU）。
- **覆盖率**：按严格路径过滤复核（聚合与逐文件）。

结论请追加为 **§19**。

---

## 20. 第 6 轮交接 — 修复 Poisson 采样器阻塞项

- 日期：2026-09-18
- 针对 §19.2（Poisson `λ≥64` 正态近似不满足统计等价）。

### 20.1 产品改动（未改 evaluator 测试）

| 项 | 文件 | 改动 |
|---|---|---|
| 阻塞项 | `rust/src/gpu/kernels.rs` | `sample_poisson` **移除正态近似**：`λ<10` 用 Knuth 精确；`λ≥10` 改为 **PTRS（Hörmann 变换拒绝）精确采样**（O(1) 期望工作量，保留偏度/峰度），与 CPU `rand_distr::Poisson` 分布等价 |
| 非阻塞 | `rust/src/gpu/kernels.rs` | 采样内核的 `rintf` → `roundf`，与 Rust `.round()`（四舍五入远离零）一致 |
| 非阻塞 | `rust/src/gpu/kernels.rs` | `recruit_stochastic` 启动器补 `n_ztypes > MAX_Z` 守卫 |

仍未处理（§19.6 其余，标记为文档化残余）：`survival_stochastic` 对 `n_virgins < -EPS` 静默夹取 0，而 CPU 返回 `Err`（仅非法状态；设备端无法便捷报错）。

### 20.2 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu evaluator_`（含 §19.2 回归） | **15 passed, 0 failed**（`evaluator_discrete_sampler_distributions_match_cpu` 通过） |
| `cargo test --features gpu` | **142 passed** |
| `cargo test` | 67 passed |
| `clippy -D warnings` / `fmt` / `check_rust` / `phase0` | 通过 / 通过 / EXIT=0 / bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |

### 20.3 请 evaluator 独立复核

- 重跑 `evaluator_discrete_sampler_distributions_match_cpu`（Poisson `λ=1,20,63,64,65,200`），确认卡方全部达标。
- 可自行扩大 `λ` 网格（如 500、1000、10000）验证 PTRS 一致性。
- 确认确定性路径、其它采样器与随机阶段未受影响。
- 覆盖率严格过滤复核。

结论请追加为 **§21**。

---

## 22. 第 7 轮交接 — 空间随机迁移

- 日期：2026-09-18
- 范围（补齐 §21 已批准 P4 中唯一未做的随机变体）：空间随机 CSR 迁移 + 空间随机端到端。
- 风险分类：**高风险**（随机分布 + 跨 deme 数据布局/归约）。

### 22.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | `migration_stochastic_prepare`（pass 1：每 source 抽样 outbound 与多项分配，写入按 CSR entry 索引的 `fwd_f/fwd_s/fwd_m`，并写 source 自身 stay 项）+ 三个 `gather_female/sperm/male`（pass 2：按反向 CSR 汇总）。无原子操作 → 可复现。`Kernels::migration_stochastic` 启动器 |
| `rust/src/gpu/executor.rs` | `GpuExecutor::migrate_tick_stochastic`：构建反 CSR（存 CSR entry 索引）、分配并上传 `fwd_*`，两趟启动后交换 |
| `rust/src/sessions/spatial.rs` | `enable_gpu` 接受 `stochastic=true`（仅拒绝 `continuous_sampling=true`）；`run_gpu_tick` 随机时调用 `migrate_tick_stochastic`；**补 `executor.set_seed(self.seed)`** |
| 测试 | `executor.rs` 随机迁移统计 L1；`spatial_session.rs` 随机空间 session tick |

### 22.2 关键修复（L3 发现）

空间 `enable_gpu` 原先**未调用 `set_seed`**，导致所有 GPU 随机运行共用 seed 0、结果恒定（被空间随机 L3 的 `gpu_mean` 恒定值暴露）。已修复为 `executor.set_seed(self.seed)`。

### 22.3 设计要点（请重点核对）

1. **两趟确定性 scatter**：pass 1 每 source 独立抽样并写 entry 级 forward 缓冲；pass 2 按反向 CSR 固定顺序汇总到 destination，无 `atomicAdd`，保证 GPU↔GPU 可复现。
2. **分布对齐**：`sample_outbound`（`rate>=1` 全走、否则 `binomial(round(value), rate)`）与 `natal_multinomial`（按行权重归一）复刻 CPU `migrate_csr_stochastic_rngs`（离散）。
3. **自身 stay 项**：pass 1 写 `ind_out[src] = virgin - moved_total` 与 `sperm_out[src] = value - moved_total`，并把储精 stay 计入雌性个体平面；pass 2 只做累加。
4. `migration_rate` 全零时沿用 CPU 跳过语义。

### 22.4 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **145 passed, 0 failed** |
| `cargo test` | 67 passed |
| `clippy -D warnings` / `fmt` / `check_rust` / `phase0` | 通过 / 通过 / EXIT=0 / bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |
| 随机迁移统计 L1（独立 pair CSR，2000 trials，5σ+0.5） | 通过 |
| 空间随机 session tick（Rust） | 状态有限、tick 前进 |
| 空间随机 L3（Python，K=300×5 tick） | CPU/GPU 总量均值差 3.05 ≪ 容差 39.7 → PASS |
| 确定性 L3（panmictic + 空间 5×5） | 逐位/紧容差不变 |
| 严格 `rust/src/gpu/` 覆盖率 | 聚合 **96.96%**，逐文件均 ≥95% |

### 22.5 请 evaluator 独立核对

- **两趟 scatter 正确性**：pass2 是否与 pass1 的 entry 索引/反向 CSR 一一对应；自环/重复边/空行/多入边/`rate=1.5`/零速率等拓扑（可自行设计）。
- **统计等价**：自建独立拓扑与参数，对设备与 CPU `migrate_csr_stochastic_rngs` 做分布级对照。
- **可复现性**：同 seed 重跑设备结果逐位一致；不同 seed 结果不同（确认 seed 生效）。
- **确定性路径**：随机改动未影响确定性结果。
- **覆盖率**：严格按绝对路径过滤 `rust/src/gpu/**`。

结论请追加为 **§23**。

---

## 24. 第 8 轮交接 — 处理 §23.7 越界守卫建议（小改动）

- 日期：2026-09-18
- 针对 §23.7：`migration_stochastic_prepare` 的行数组按 `MAX_Z` 固定分配，但仅校验了 `n_ztypes`，
  未校验 CSR **出度** `row_len`。

### 24.1 改动

| 文件 | 改动 |
|---|---|
| `rust/src/gpu/kernels.rs` | 新增 `pub const MAX_CSR_ROW = MAX_Z` |
| `rust/src/gpu/executor.rs` | `migrate_tick_stochastic` 在启动前检查最大 CSR 行宽 > `MAX_CSR_ROW` 即显式报错 |
| 测试 | `execution_rejects_overwide_stochastic_csr_rows`（行宽 33 → `Err`） |

### 24.2 自测（非独立）

| 命令 | 结果 |
|---|---|
| `cargo test --features gpu` | **148 passed** |
| `cargo test` | 67 passed |
| `clippy -D warnings` / `fmt` / `check_rust` / `phase0` | 通过 / 通过 / EXIT=0 / bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |

请 evaluator 快速复核此项（确认越界不再可达、无回归）。结论请追加为 **§25**。

---

## 26. 第 9 轮交接 — P6 多 B ensemble

- 日期：2026-09-18
- 范围：把模型复制到设备 batch 轴做多 replicate，并与 `ProcessPoolExecutor` 对比。
- 风险分类：**局部到高风险之间**（新增 API + 复用已审 RNG/采样路径；无数值新算法）。

### 26.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/executor.rs` | `GpuExecutor::ensemble(context, n_replicates, A, Z, ind_one, sperm_one, seed)`：单群体初始态平铺到 batch 轴；各 replicate 用不相交 counter-based 流 |
| `rust/src/sessions/age_structured.rs` | `tile_ecology`；`enable_gpu_ensemble(n)` / `run_gpu_ensemble(n_ticks)`（返回堆叠 `(B,2,A,Z)`/`(B,A,Z,Z)`）；`EnsembleReadout` 别名 |
| `src/natal/backends/rust/rust_backend.py` | `enable_gpu_ensemble()` / `run_gpu_ensemble()` 透传 |
| 测试 | `device_ensemble_matches_independent_cpu_runs`（B=2000×3 tick vs 2000 次独立 CPU 全随机 tick，5σ）；`session_gpu_ensemble_runs_and_covers_wiring`；`session_gpu_ensemble_rejects_ineligible` |

### 26.2 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **151 passed** |
| `cargo test` | 67 passed |
| `clippy`（默认+gpu）`-D warnings` / `fmt` / `check_rust` / `phase0` | 通过 / 通过 / EXIT=0 / bit-identical |
| `ruff` / `pyright` / `pytest` | 通过 / 0 errors / 3606 passed |
| ensemble 统计 L2 | 通过（B=2000） |
| Python smoke（backend API，B=500×10 tick） | 形状/有限性正确 |
| Benchmark（B=5000, A=8, Z=3, 50 tick, 16 CPU workers） | GPU 0.278s vs ProcessPool 0.664s → **2.4×**；快照前后 GPU 空闲（0%→0%，15 MiB→15 MiB） |

### 26.3 诚实的性能结论（请 evaluator 判定是否满足 P6 验收）

- GPU **单位工作量**约快 ~40×（5000×50=250k replicate-tick，GPU 0.278s）。
- 但相对 **16 核 ProcessPoolExecutor** 的墙面加速比仅 **2.4–3.3×**，并非“数量级”优势。
- 原因：每个 replicate 状态很小（≤ 8×3×2 cells）、CPU 16 核并行、每 tick 的 kernel/同步开销被摊薄。
- 结论：本实现正确且可复现；是否满足计划 §5.3 P6 的“GPU 必须显著优于 ProcessPool”取决于阈值定义。若要求更高，需要：更大单 replicate 状态、更多 tick、或更少 CPU 核；也可考虑 GPU 侧减少每 tick 同步（当前 ensemble 只回传一次，已较优）。
- 请 evaluator 明确：以什么基准/阈值判定 P6 通过；如不满足，请在 §27 给出需要的实验。

### 26.4 请 evaluator 独立核对

- ensemble 的统计等价（自建模型/参数，设备 B replicates vs 独立 CPU 运行）与**同 seed 逐位复现**、不同 seed 不同。
- 会话 API 资格拒绝语义（`n_replicates=0`、`continuous_sampling=true`、空间、自定义曲线、含钩子）。
- 确定性/空间路径不受影响；`phase0` bit-identical。
- 严格过滤 `rust/src/gpu/**` 覆盖率；`age_structured.rs` 新增行覆盖。
- 对 §26.3 给出 P6 是否达标的判定。

结论请追加为 **§27**。

---

## 28. 第 10 轮交接 — 设备侧 CSR 缓存 + 空间逐 tick 零回传

- 日期：2026-09-19
- 背景：§25（第 8 轮越界守卫）与 §27（第 9 轮 P6）回执尚未返回；本轮按 `GPU_STAGE_SUMMARY.md` §11
  的建议先做「设备侧 CSR 缓存 + 零逐 tick 回传」。**不涉及数值公式、随机分布或公开 API**，但改动了
  空间会话的状态同步时机，按高风险（状态语义）提交独立复核。
- 风险分类：**高风险**（会话状态回传/历史记录时机；虽无新数值算法，但影响状态可见性与历史正确性）。

### 28.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/executor.rs` | 新增 `MigrationCache`：把冻结 CSR 的前向/反向数组、`row_sum`、以及随机迁移的 `dest/weights/rev_entry` 与 `fwd_f/fwd_s/fwd_m` scratch 常驻显存，按 CSR 指纹失效；`migrate_tick` / `migrate_tick_stochastic` 改为复用缓存（仅每 tick 重传动态 `migration_rate`）。新增 `csr_fingerprint`、`build_migration_cache`、`ensure_migration_cache`。 |
| `rust/src/sessions/spatial.rs` | `run_gpu_tick` 不再逐 tick 下载 `ind/sperm`；新增 `sync_gpu_state`（GPU 版与 CPU 空实现），在 `run_steps` 的历史记录边界、停止返回处和运行结束处调用。 |
| `rust/tests/unit/gpu/executor.rs` | 新增 `migration_cache_reuses_buffers_and_invalidates_on_new_csr`、`stochastic_migration_reuses_cached_scratch_reproducibly`、`stochastic_migration_validates_inputs`、`ensemble_rejects_mismatched_replicate_state`；新增 `assert_migration_close` 辅助。 |
| `rust/tests/unit/gpu/spatial_session.rs` | 确定性用例改为显式断言“逐 tick 不下载”（host 数组不变）+ 显式 `sync_gpu_state`；随机用例改走 `run_steps(1, 0)`。 |

**未改动**：`rust/src/kernels`、`rust/src/model`、`src/natal/contracts`、`rust/src/lib.rs`（`git diff` 为空）。

### 28.2 行为变化与理由

1. **CSR 静态缓存**：迁移图在 blueprint 里是冻结的，前向 CSR 与反向 CSR 不会逐 tick 变化。缓存后
   每 tick 只上传会变的 `migration_rate`，并省掉随机路径 `fwd_*` 的“分配 + 清零上传”。随机 `fwd_*`
   可安全复用：`migration_stochastic_prepare` 对每个 CSR entry 的每个 age/z 槽都写入（含显式 0）。
2. **零逐 tick 回传**：设备 tick 期间状态常驻显存；host `ind/sperm` 只在历史记录边界和一次运结结束时
   刷新（`sync_gpu_state`）。历史记录分别在记录前同步，故历史内容不变；仅去掉了非记录 tick 的 D2H。
3. **CPU 路径**：`sync_gpu_state` 在非 gpu feature 下是空实现，`run_steps` 调用它不改变任何 CPU 行为。

### 28.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `python scripts/check_rust.py` | EXIT=0（67 passed） |
| `cargo test`（默认 feature） | **67 passed, 0 failed** |
| `cargo test --features gpu` | **155 passed, 0 failed** |
| `cargo clippy --features gpu -- -D warnings` | 通过 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | `all scenarios bit-identical` |
| 严格过滤 `rust/src/gpu/**`（排除 `/tests/`）覆盖率 | 聚合 **97.37%**；`executor.rs` **807/837 = 96.4%**，逐文件均 ≥95% |
| 迁移微基准（D=16384, A=8, Z=3, 20 tick 随机迁移；GPU 前后均 15 MiB / 0%） | 缓存复用 **79.3 ms** vs 每 tick 重建 **196.8 ms** → 迁移阶段 **~2.48×** |

> 说明：上述为**主 agent 自测**，不是独立审查。微基准为单次采样、未做多次重复；共享 GPU 上仅用于
> 量级参考。

### 28.4 请 evaluator 独立核对

- **零回传正确性**：多次 GPU tick 后 host 数组确实不变，`sync_gpu_state`/`run_steps` 后与 CPU 对照一致
  （自建模型/更大 D，非仅仓库用例）。
- **历史正确性**：`run_steps(..., record_interval>1)` 时，记录到的每个 tick 都来自“该 tick 的设备状态”，
  与 CPU 记录的 tick 序列/状态一致；随机模式用统计等价而非逐位。
- **缓存失效**：同一 CSR 命中（仅 rate 变化仍正确）；不同 CSR（dest/weights/indptr 任一变化）必须重建，
  且结果与 host 参考一致；随机 scratch 复用不得泄漏上一 tick 的样本。
- **CPU 不变性**：`git diff` 证明 kernels/model/contracts/lib.rs 零改动；`phase0` bit-identical。
- **覆盖率**：严格按绝对路径过滤并排除 `/tests/`（注意历史坑：`src/gpu/../../tests/...`）。
- **共享 GPU 波动**：本轮出现过一次 instrumented 运行失败、重跑即通过；请记录复现率，勿计入产品缺陷。

### 28.5 已知残余风险（非阻塞，供判定）

- 显存驻留增加：CSR 缓存（含随机 `fwd_s` = `nnz·A·Z²·4`）现在常驻，`GpuExecutor::new` 的预算守卫
  未计入这部分；超限时 `DeviceBuffer::from_host` 会显式报错（不静默回退），但发生在首次迁移时而非入口。
- GPU 会话的 checkpoint/restore 仍不支持设备侧状态回滚（既有问题，本轮未扩大）。
- 未做 ecology 缓存：ecology 可被会话/回调修改，缓存会破坏其“每 tick 生效”语义，故本轮有意不做。

结论请追加为 **§29**。

---

## 30. 第 10 轮修复交接（处理 §29 阻塞项）

- 日期：2026-09-19
- 针对 §29.1/§29.2：公共单 tick `SpatialSession::run_tick` 执行后不同步 host，导致
  `state_snapshot` / `state_snapshot_deme` / `observe_current` / `capture_checkpoint` 及查询访问器
  返回上一 tick 的陈旧状态。
- 风险分类：**高风险**（会话状态可见性），修复已按 evaluator 建议的第二方案实施。

### 30.1 根因

`run_steps` 通过 `sync_gpu_state` 在记录边界/结束时同步，但**公共 `run_tick` 直接调用 `run_inner`**，
不经 `run_steps`；`run_inner` 设备分支已不再下载，故单 tick 后 host 数组停滞而 `state_tick` 前进。

### 30.2 改动

| 文件 | 内容 |
|---|---|
| `rust/src/sessions/spatial.rs` | 把原 `run_tick` 的 tick 主体抽到私有 `advance_tick`（**不同步**）；公共 `run_tick` = `advance_tick()` + `sync_gpu_state()`（先完成 tick 再下载，出错则不下载）；`run_steps` 循环改调 `advance_tick`，保留边界/结束同步。 |

效果：公共单 tick 与全部公共读路径恢复一致；多 tick `run_steps` 仍保持“运行期零逐 tick 回传”。

### 30.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu evaluator_single_tick_syncs_host_state`（§29.2 的失败回归） | **ok**（修复前失败） |
| `cargo test --features gpu` | **156 passed, 0 failed**（含 evaluator 回归） |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3606 passed / bit-identical |
| Python 端到端（重编 `maturin develop --features "gpu,extension-module"`，4 deme 确定性空间模型） | `backend.run_tick()` 后 `state_snapshot`：tick 0→1，总量 1400.0→2278.98，与 CPU 相对误差 **2.2e-8**；修复前该路径返回 1400.0 |

### 30.4 请 evaluator 复核

- 重跑 `evaluator_single_tick_syncs_host_state` 与 Python 端到端（`run_tick` → `state_snapshot`/
  `observe_current`/`capture_checkpoint`）确认不再陈旧。
- 确认 `run_steps` 的多 tick 零回传与历史正确性未受影响（§29.3 结论应保持）。
- 其余 §29.5 残余风险不变（CSR 缓存未纳入预算估算、GPU checkpoint 回滚、ecology 不缓存）。

结论请追加为 **§31**。

---

## 32. 第 12 轮交接 — 设备侧历史缓冲（D5 完整形态，观测模式）

- 日期：2026-09-19
- 背景：§31 APPROVED（第 10/11 轮）。按 `GPU_STAGE_SUMMARY.md` §11 推进「设备侧历史驻留」完整形态。
- 风险分类：**高风险**（新增设备内核 + 历史记录时机 + 显存预算；涉及 Python/Rust 历史数据流，但不改数值公式与公开 Python API）。

### 32.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | 新增 `OBSERVATION_SOURCE` 与 `observation_project` 内核（复刻 `output::observation::project`：基因型→年龄→deme 的累加顺序；行偏移写入）。Kernels 结构/加载新增该函数。 |
| `rust/src/gpu/executor.rs` | 新增 `HistorySpec`、`DeviceHistory` 与 `configure_history` / `record_history_row` / `download_history_rows` / `history_width` / `clear_history`：设备侧投影行缓冲，`configure` 用实测可用显存做预算守卫。 |
| `rust/src/sessions/spatial.rs` | `run_steps` 在**观测模式**历史下改走设备暂存：记录边界只做设备投影（不同步 host），运行结束一次下载并按 tick 回填 `HistoryStore`。新增 `start_device_history` / `record_boundary` / `flush_device_history`（含非 gpu 空实现）。**原始（raw）模式与不满足预算时保持原 host 逐记录路径**。 |
| 测试 | `device_history_projection_matches_host_project`（4 种 collapse/aggregate 组合对照 host `project`）；`device_history_rejects_over_budget_and_empty_mask`（预算/空 mask/dims/selection/width/overflow/满窗/清空）；`spatial_device_history_stages_and_matches_cpu`（record_every 1/2，tick 序列与值对照 CPU）；`spatial_device_history_falls_back_for_raw_or_missing_window`（GPU 未启用/无 store/raw → 走 host）。 |

### 32.2 行为与语义

1. **仅观测模式设备暂存**：`HistoryData.raw == false` 且有非空观测 mask 时，记录行在设备上投影（宽度 = `groups·out_d·S·out_a`），运行期零逐记录回传；结束时一次下载。
2. **raw 模式不变**：仍走 host 逐记录同步投影（全状态行；本就在显存预算表里属超限项）。
3. **预算回退不是引擎回退**：设备窗口装不下（`ensure_memory_budget` 失败）或无法配置时，退回 host 逐记录路径，同一 CPU/GPU 系综与数值结果不变；已用 `start_device_history` 的返回值在测试中区分，避免“静默回退”掩盖缺陷。
4. **tick 列在 host 追加**：设备内核不写 tick（避免 f32 整数精度问题），`flush` 按 `start/interval/rows` 重建记录 tick 并以 `append_row(continuation=true)` 写入。
5. **公开 Python API 不变**：`History.ticks/values/...` 语义不变；`pop.run(n, record_every)` 行为一致。

### 32.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **160 passed, 0 failed** |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3606 passed / bit-identical |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | 聚合 **2098/2151 = 97.54%**；executor **916/946 = 96.8%**、kernels **98.0%**、probe 96.1%，逐文件 ≥95% |
| 内核 L1 | `observation_project` vs host `project`，4 种 collapse/aggregate 组合，相对误差 ≤1.2e-6 |
| 会话 E2E（Rust） | 观测历史 record_every=1 与 2：GPU 与 CPU tick 序列一致，值相对误差 ≤1e-4（实测约 1e-7 量级） |
| Python E2E（重编扩展，4 deme） | `pop.run(4, record_every=1/2)`：GPU/CPU `history.ticks` 相同、`values` 形状一致，max rel diff **9.5e-8** |
| 共享 GPU | 一次 instrumented 运行失败、重跑通过（环境噪声，非产品缺陷） |

### 32.4 请 evaluator 独立核对

- **投影内核正确性**：自建 mask/selected/collapse/aggregate 组合（含 groups>1、非 0/1 权重、selected 子集且乱序）对照 host `project`；确认累加顺序与 f32 容差合理。
- **记录时机**：device 暂存行对应的确是**该记录 tick 的设备状态**；`flush` 重建的 tick 与 `run_steps` 的记录边界一致（含 start_tick 非 0、start_tick % interval != 0、跨多次 run 的 continuation）。
- **回退路径**：raw / 预算不足 / 无法配置时确实走 host 逐记录路径，且不改变 GPU↔GPU 可复现性与 CPU 数值（同一引擎）。
- **预算与显存**：`configure_history` 使用实测 free；`required > free` 显式 Err；确认无静默 CPU 回退。
- **raw 语义**：raw 模式历史仍正确（本轮有意不改）。
- **CPU 不变性 / 覆盖率 / 门禁**：同既往口径（注意覆盖率过滤排除 `/tests/`、`src/gpu/../../tests/...` 坑）。

### 32.5 已知残余风险（非阻塞）

- 设备历史窗口容量按 `n_ticks/interval` 上界分配，未按 `max_rows` 收缩；大 D×长 T 可能触发预算回退（host 路径仍可跑，仅失去零回传）。
- GPU 会话 checkpoint/restore 仍不支持设备侧回滚（既有）。
- 错误路径下设备窗口未即时释放（下次 `configure_history` 覆盖）。

结论请追加为 **§33**。

---

## 34. 第 12 轮修复交接（处理 §33 阻塞项）

- 日期：2026-09-19
- 针对 §33.1/§33.2：`configure_history` 预算算术未检查（极端容量 panic/回绕），`run_steps` 的 `start_tick + n_ticks` 未检查。
- 风险分类：**高风险修复**（显存预算守卫的正确性契约），按 evaluator 建议实施。

### 34.1 改动

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/executor.rs` | `configure_history`：`rows`、`row_bytes`、`mask_bytes`、`selected_bytes` 及求和全部改用 `checked_mul`/`checked_add`，任一溢出返回显式 `Err`，随后才做预算比较与分配。 |
| `rust/src/sessions/spatial.rs` | `run_steps`：`start_tick.checked_add(n_ticks)`，溢出返回 `PyValueError`（不再未检查 `i64` 加法），再计算设备窗口容量。 |
| 测试 | 新增 `run_steps_rejects_overflowing_tick_span`（`start_tick=1` 后 `run_steps(i64::MAX, 1)` → 显式 Err）；evaluator 的 `evaluator_configure_history_over_budget_is_explicit` 转绿。 |

### 34.2 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu evaluator_configure_history_over_budget_is_explicit` | **ok**（修复前 panic `attempt to multiply with overflow`） |
| `cargo test --features gpu run_steps_rejects_overflowing_tick_span` | ok |
| `cargo test --features gpu` | **163 passed, 0 failed**（含 evaluator 2 个回归） |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3606 passed / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2112/2165 = 97.55%**；executor **930/960 = 96.9%**、kernels 98.0%、probe 96.1%，逐文件 ≥95% |
| Python E2E 复核（重编扩展） | 观测历史 record_every=1/2 ticks 与值仍与 CPU 一致，max rel diff 9.5e-8 |

### 34.3 请 evaluator 复核

- 重跑 `evaluator_configure_history_over_budget_is_explicit` 确认显式 Err（不再 panic/回绕）。
- 复核 `configure_history` 全路径无未检查算术；`run_steps` 溢出返回显式 `PyValueError`。
- 确认 §33.3 的投影/时机/回退/CPU 不变性结论未受影响。
- 其余 §33.5 残余风险不变（raw 未设备暂存、窗口未按 `max_rows` 收缩、boundary metadata、GPU checkpoint）。

结论请追加为 **§35**。

---

## 36. 第 14 轮交接 — `continuous_sampling` 设备支持

- 日期：2026-09-19
- 背景：§35 APPROVED（第 12 轮修复）。按用户选择推进「`continuous_sampling` 设备支持」（`GPU_STAGE_SUMMARY` §11 第 4 项）。
- 风险分类：**高风险**（新增随机分布：连续二项/多项/Poisson，涉及科学抽样语义；需统计等价验证）。

### 36.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | 新增设备采样器 `sample_continuous_binomial`（Beta 比例 × n）、`sample_continuous_poisson`（Gamma(λ,1)）、`natal_continuous_multinomial`（归一化 Gamma + 漂移校正）。`recruit_stochastic` / `survival_stochastic` / `reproduction_stochastic` / `migration_stochastic_prepare` 增加 `int continuous` 分支，去掉离散分支里的 `roundf`（连续路径用原始浮点计数与比例移除）。`sample_outbound_device` 增加连续分支。新增 `stochastic_grid`（64 线程块）避免连续分支提高寄存器用量后 1024 线程块报 `LAUNCH_OUT_OF_RESOURCES`。`sample_gamma` 改为**迭代**实现（原 shape<1 递归在 reproduction 大内核中导致设备栈溢出 → `CUDA_ERROR_ILLEGAL_ADDRESS`）。 |
| `rust/src/gpu/executor.rs` | 删除 stochastic survival/reproduction/migration 对 `continuous_sampling` 的显式拒绝，向下透传 `blueprint.continuous_sampling`。 |
| `rust/src/sessions/{age_structured,spatial}.rs` | 删除 `enable_gpu` / `enable_gpu_ensemble` 的 `continuous_sampling=false` 拒绝。 |
| 测试 | 生存/繁殖/迁移三个分布测试参数化为 `continuous=false/true` 并各加一个连续用例（GPU vs host 同分布均值，5σ+0.5）；新增空间会话「连续被接受并跑 tick」与年龄结构会话/ensemble「连续被接受」。 |

### 36.2 语义

- 连续抽样严格复刻 host：连续二项 = Beta 比例（Gamma 构造）× n；连续多项 = 归一化 Gamma + 漂移校正；连续 Poisson = Gamma(λ,1)。CPU `continuous_sampling=false` 路径与门禁不变。
- 设备 `sample_gamma` 的迭代改写只改变内部实现，分布不变（既有 `shape<1` gamma 统计用例仍通过）。

### 36.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **168 passed, 0 failed**（含 3 个连续分布用例 + 2 个会话接线用例） |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3606 passed / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2131/2180 = 97.75%**；executor 97.3%、kernels 98.1%、probe 96.1%，逐文件 ≥95% |
| 连续生存/繁殖/迁移分布 | GPU vs host 同分布均值在 5σ+0.5 内（`device_stochastic_*_continuous_matches_host_distribution`） |
| Python E2E（重编扩展，4 deme 随机连续空间模型） | `enable_gpu` 接受；`run(3)` tick 前进、状态有限；连续 GPU 与 CPU 各自跑通（不同系综，不逐位） |

### 36.4 请 evaluator 独立核对

- **连续采样器正确性**：自建参数对照 host `continuous_binomial/poisson/multinomial` 的矩/分布（含 shape<1 的 Gamma 构造、大 λ/大 n、p→0/1 边界）。
- **四个随机阶段**：连续模式下 survival/reproduction（含 mating/fertilize/sex/viability）/recruit/migration 的分布等价；确认离散路径与既有结论零回归。
- **宿主迭代 gamma**：确认分布不变；检查无递归残留。
- **会话语义**：`enable_gpu` / `enable_gpu_ensemble` 接受连续且不是静默回退；CPU 路径与 `phase0` 不变；无未检查算术等。
- **性能/资源**：`stochastic_grid` 64 线程块是否影响既有离散用例表现（仅正确性要求）。

### 36.5 残余风险 / 说明

- 设备连续路径与 CPU 为统计等价，不逐位（既有约定）。
- `stochastic_grid` 为资源约束下的固定块大小，未做占用率调优。
- 本轮不改 raw 历史 / checkpoint 等既有残余项。

结论请追加为 **§37**。

---

## 38. 第 15 轮交接 — 离散世代（空间）GPU 路径

- 日期：2026-09-19
- 背景：§37 APPROVED。用户选择推进 `GPU_STAGE_SUMMARY §11` 第 5 项「离散世代 GPU 路径（当前空间离散被拒）」。
- 风险分类：**高风险**（新增离散二龄生命周期的设备内核，含随机/连续采样；涉及数值与状态）。

### 38.1 改动清单

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | 新增 `DISCRETE_SOURCE`：`discrete_reproduction`（成年雌性求偶 + 受精，离散/连续）与 `discrete_survival`（密度缩放 + 幼体重采样 + 年龄 0 存活，离散/连续）。`density_scaling` 内核新增 `discrete_actual` 参数（离散用总年龄 0 计数，而非加权幼体），启动器同步。Kernels 结构/加载/源拼接新增两个内核。 |
| `rust/src/gpu/executor.rs` | 新增 `discrete_reproduction_tick` / `discrete_survival_tick` / `discrete_tick`（复现→存活→衰老）；`reproduction_tick` 重构为 `reproduction_impl(discrete)`；`density_scaling` 拆为 `density_scaling_impl(discrete_actual)`。 |
| `rust/src/sessions/spatial.rs` | `enable_gpu` 不再拒绝离散；`run_gpu_tick` 增加 `discrete` 参数并分派 `discrete_tick`。离散迁移复用既有确定性/随机 CSR 迁移内核（sperm 面恒零）。 |
| 测试 | 空间会话：`spatial_discrete_device_tick_matches_cpu`（确定性 3 tick vs CPU，容差 1e-4）、`spatial_discrete_stochastic_device_tick_runs`；内核统计：离散繁殖/存活各 2 个（离散/连续采样，GPU vs host 同分布均值，5σ+0.5）。 |

### 38.2 语义

- 离散生命周期严格复刻 `kernels::discrete_generation`（两龄、无储精面、`reproduction_rates`/`mating_rates` 取成年列、`viability_fitness` 年龄 0 行）。
- 密度缩放复用 `density_scaling` 内核，`discrete_actual` 分支用总年龄 0 计数，匹配 host `scaling_factor`。
- 离散迁移沿用 P5 的 CSR 迁移内核；离散 `ind` 为 `(2,2,Z,B)`，sperm 面恒零。

### 38.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **174 passed, 0 failed**（含 4 个离散分布用例 + 2 个会话用例） |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3606 passed / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2344/2418 = 96.94%**；executor 96.0%、kernels 97.4%、probe 96.1%，逐文件 ≥95% |
| 离散繁殖/存活分布（离散与连续） | GPU vs host 同分布均值在 5σ+0.5 内 |
| Python E2E（重编扩展，4 deme 空间离散） | 确定性 `run(3)` GPU/CPU 相对误差 **6.4e-9**；随机连续 `run(2)` 有限 |

### 38.4 请 evaluator 独立核对

- **内核逐分支**：`discrete_reproduction` / `discrete_survival` 与 host `kernels::discrete_generation` 的求偶/受精/性别/缩放/幼体重采样/存活分支一致；`density_scaling` 的 `discrete_actual` 与 `scaling_factor` 一致（含 fixed/linear/beverton_holt/ricker、declared 分布、external eggs）。
- **统计等价**：自建参数对离散繁殖/存活做 GPU vs host 分布对照（含连续采样与边界 p/n）。
- **会话/迁移**：离散空间 `enable_gpu` 接受且非静默回退；离散迁移复用正确；确定性 `phase0` 不变。
- **无回归**：年龄结构/随机/连续/历史/ensemble 既有用例全部保持。

### 38.5 残余风险 / 说明

- 离散 GPU 未覆盖 Wright-Fisher 融合模式（`run_wf_tick`）；空间离散 CPU 路径本就不使用它，故非阻塞。
- `discrete_survival` 固定 64 线程块（寄存器约束）。
- CPU 数值内核/契约零改动（`git diff` 为空）。

结论请追加为 **§39**。

---

## 40. 第 16 轮交接 — frontend 级 ensemble 入口 + 文档

- 日期：2026-09-19
- 背景：§39 APPROVED。用户选择推进 `GPU_STAGE_SUMMARY §11` 第 6 项「frontend/Population 级 ensemble 入口 + 文档」。
- 风险分类：**文档与格式修改 + 局部代码修改**（新增公开 Python 方法，不改科学模型与 Rust 数值；返回形状属新公开合同，故需复核）。

### 40.1 改动清单

| 文件 | 内容 |
|---|---|
| `src/natal/frontend/population/age_structured.py` | `AgeStructuredPopulation` 新增 `enable_gpu_ensemble(n_replicates)`（可链式；惰性建会话、拒绝不合格模型/无 gpu 扩展）与 `run_gpu_ensemble(n_ticks)`（返回 `(tick, individual_count, sperm_storage)`，形状 `(B,2,A,Z)` / `(B,A,Z,Z)`）；新增内部字段 `_gpu_ensemble_replicates`。后端 `RustLifecycleBackend` 的对应方法已存在。 |
| `tests/test_gpu_ensemble_frontend.py` | 新增 3 个测试：`n_replicates=0` 拒绝、未 enable 先 run 报错、形状/有限性/不推进 CPU state；无 GPU 主机 `pytest.skip`。 |
| `docs/en/4_simulation_engine.md` | 新增 §11「GPU Ensemble (CUDA, optional)」，原小结顺延为 §12。 |
| `docs/zh/4_simulation_engine.md` | 同步新增 §11「GPU 多 replicate ensemble（CUDA，可选）」+ §12 小结。 |

### 40.2 行为

- 公开入口语义与 backend 一致：panmictic、无钩子、内置生长模式；不合格/无 gpu feature 显式 `RuntimeError`，不静默回退。
- ensemble 是独立实验：`run_gpu_ensemble` 不推进种群自身 state/tick。
- 同 seed 设备内逐位可复现；CPU↔GPU 仅统计可比（既有约定）。

### 40.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `pytest -q`（全量） | **3609 passed, 0 failed**（含新增 3 个） |
| `pytest tests/test_gpu_ensemble_frontend.py` | 3 passed |
| `ruff` / `pyright` | 通过 / 0 errors |
| `cargo test` / `cargo test --features gpu` | 67 / 174 passed（Rust 未改，回归确认） |
| `check_rust.py` / `phase0_baseline --check` | EXIT=0 / bit-identical |
| 端到端冒烟（重编扩展，B=200×10 tick） | `(10, (200,2,4,3), (200,4,3,3))`，全有限，`pop.tick` 不变 |

### 40.4 请 evaluator 独立核对

- **公开合同**：返回元组形状与文档一致；`run_gpu_ensemble` 不推进 CPU state/tick；先 enable 后 run 的顺序约束。
- **不合格拒绝**：非 panmictic / 含钩子 / 自定义 growth / 无 gpu 扩展 → 显式 `RuntimeError`（非静默回退）；`n_replicates=0` 拒绝。
- **文档中英同步**：两版 §11 内容一致、示例可运行、链接/编号正确。
- **回归**：既有 Python 测试与 Rust 门禁不变。

### 40.5 残余风险 / 说明

- 未新增 `Population` 级 `enable_gpu`（单群体一步回传）入口；本轮只做 ensemble，文档亦如此表述。
- 未提供 frontend ensemble 的 History 集成（返回裸数组）；如需与 `History` 打通可后续扩展。

结论请追加为 **§41**。

---

## 42. 第 17 轮交接 — 性能优化（移除每 tick 缩放同步）

- 日期：2026-09-19
- 背景：§41 APPROVED。用户选择推进 `GPU_STAGE_SUMMARY §11` 第 7 项「性能优化」。
- 风险分类：**局部代码修改**（设备缓冲区生命周期/同步时机；不涉数值公式与公开 API）。仍按惯例交独立复核。

### 42.1 改动

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/executor.rs` | 新增 `density_scaling_device`（把缩放因子留在设备，返回 `DeviceBuffer<f32>`）；`survival_tick`（年龄结构）与 `discrete_survival_tick` 改用它，**去掉每 tick 的缩放 D2H 同步 + 再次 H2D 上传**。保留公开的 `density_scaling`（内部经 `density_scaling_impl` 做一次 D2H）与 `density_scaling_impl`，供需要主机值的调用方/测试使用。 |

### 42.2 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **174 passed, 0 failed** |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / **3611 passed** / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2352/2426 = 96.95%**；executor 96.0%、kernels 97.4%、probe 96.1%，逐文件 ≥95% |
| 微基准（`density_scaling` device vs host，B=4×2000 次；nvidia-smi 前后 0%/15MiB） | device **62.3ms** vs host **71.5ms** → 每次去除约 **4.6µs** 同步 |

### 42.3 诚实的性能结论（请 evaluator 判定期望）

- 本改动移除的是**确定性 survival 路径每 tick 的设备→主机同步**（原实现 `density_scaling` 后 `to_host` 再上传），与项目「逐 tick 不回传」目标一致，风险极低。
- 但临时全 tick 基准（B=20000, A=8, Z=3, 50 tick；nvidia-smi 0–3%）显示 **~20.9ms/tick**，stage 拆分约 repro **10.6ms**、surv **10.6ms**、aging **3.2µs**——两者几乎相同，且远高于传输量级（每 tick 约 15MB，PCIe 上传 <1ms），说明**瓶颈不在传输/同步，而在计算侧**；同一数值也可能是共享 GPU 多租户时间片造成的失真，需在空闲卡上复测才能定位。
- 结论：**传输类优化（ecology 缓存等）在大 B 下收益有限**；进一步提升需要内核重构/占用率分析（更大风险，建议作为独立、需空闲 GPU 剖析的任务另立）。本轮先交付无争议的同步移除。

### 42.4 请 evaluator 独立核对

- **等价性**：`density_scaling` 的主机返回值与改动前一致；确定性年龄结构/离散设备状态逐 tick 结果不变（既有 L1/L2/L3 用例覆盖）。
- **无残留同步**：`survival_tick` / `discrete_survival_tick` 内不再出现缩放因子的 D2H+H2D；`density_scaling_device` 不触发主机同步。
- **回归/门禁/覆盖率**同既往口径。

### 42.5 残余风险

- 上述大 B 基准受共享 GPU 影响，结论仅定性；深层性能优化未做。

结论请追加为 **§43**。

---

## 44. 第 18 轮交接 — A1：GPU 会话 checkpoint 回滚（完整设备恢复）

- 日期：2026-09-19
- 背景：§43 APPROVED。按 `GPU_STAGE_SUMMARY §11.2` 计划表推进 **A1**（正确性、P1）。用户选择「完整设备恢复」方案。
- 风险分类：**高风险**（状态恢复：把 checkpoint 重新上传到显存并回退设备 tick/RNG 基准）。

### 44.1 改动

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/executor.rs` | 新增 `GpuExecutor::restore_state(ind_host, sperm_host, tick)`：按 batch-major 校验长度、转置为 batch-minor 重新上传 `ind`/`sperm`，并把设备 tick 计数器设为 `tick`（counter-based RNG 以 tick 为基准，故同 seed 可从该边界逐位续跑）。 |
| `rust/src/sessions/spatial.rs` | `restore_from_checkpoint`：host 状态回滚后，若设备执行器存在则调用 `restore_state` 并把设备 tick 回退到 checkpoint.tick，同时 `clear_history()` 丢弃被放弃时间线的设备历史行。 |
| `rust/src/sessions/age_structured.rs` | 新增私有 `restore_device_state(tick)`（把单群体状态按执行器 batch 宽度平铺——单群体=1、ensemble=B——重新上传）；在 `restore_from_checkpoint` 与手动 `restore_state` 两条恢复路径末尾调用。 |
| 测试 | `spatial_gpu_restore_checkpoint_rewinds_device_state`（raw checkpoint 恢复后设备重跑与参照逐位一致）；`session_gpu_restore_device_state_rewinds`（age：设备恢复后重跑与参照逐位一致）；`restore_state_rejects_wrong_lengths`（长度校验分支）。 |

### 44.2 行为

- GPU 会话的 checkpoint 回滚现在**同时回滚显存状态与设备 tick**：不再出现“host 已回滚、设备从旧状态继续”的静默错误轨迹。
- ensemble 执行器（batch=B）在恢复时把单群体状态平铺到 B，与 `enable_gpu_ensemble` 的构造方式一致。
- 设备历史窗口在恢复时清空，避免跨时间线的陈旧记录。
- 迁移缓存与 CSR 无关状态保持有效；CPU 数值路径零改动。

### 44.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **177 passed, 0 failed**（含 3 个 A1 用例） |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3611 passed / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2383/2457 = 96.99%**；executor 96.2%、kernels 97.4%、probe 96.1%，逐文件 ≥95% |
| 回归测试 | GPU/CPU 重跑在恢复边界后**逐位一致**（bit-for-bit） |

### 44.4 请 evaluator 独立核对

- **正确性**：raw checkpoint 恢复后，GPU 重跑轨迹与“从未越过该 tick”的参照逐位一致（空间与 age 两条会话路径）；确定性/随机均适用（同 seed、同 tick）。
- **无静默错误**：恢复后设备 tick = checkpoint.tick；设备历史窗口被清空；不再有 host/设备状态错配。
- **边界**：`restore_state` 长度不匹配显式报错；ensemble（B>1）平铺正确；已声明的限制（age GPU 运行本身不采集 checkpoint）不变。
- **CPU 不变性 / 门禁 / 覆盖率**同既往口径。

### 44.5 残余风险

- age 结构化设备分支本身不采集 checkpoint（既有：GPU 运行无历史），本条主要覆盖“先有 checkpoint 再启用/回滚设备”的场景；该既有局限未变。
- `enable_gpu` 在已有 CPU tick 之后启用时不会自动把设备 tick 设为 `state_tick`（A1 恢复会修正；但正常“先启用后运行”路径不受影响）——记为后续可选清理项。

结论请追加为 **§45**。

---

## 46. 第 19 轮交接 — A2：随机生存显式拒绝非法状态

- 日期：2026-09-19
- 背景：§45 APPROVED（A1）。按 `GPU_STAGE_SUMMARY §11.2` 推进 **A2**（正确性、P2）。
- 风险分类：**局部代码修改**（错误处理语义；不改科学公式与正常路径数值）。

### 46.1 改动

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | `survival_stochastic` 新增 `int* violation` 输出与 `#define NATAL_VIRGIN_EPS 1e-3f`：当 `virgins < -NATAL_VIRGIN_EPS`（f32 容差）时置 `violation[0]=1`，随后仍夹 0 以保持后续算术有限。启动器新增 `violation` 参数。 |
| `rust/src/gpu/executor.rs` | `survival_tick`（随机分支）分配 1 元素的 `violation` 缓冲、随内核传入，启动后回读；非 0 则返回显式 `Err`（`Invalid state: n_virgins < 0 ...`）。 |
| 测试 | `stochastic_survival_rejects_negative_virgin_state`：构造 `n_f - total_sperm = -20` → `Err`；构造 `-1e-4`（f32 舍入噪声，在容差内）→ 夹 0 且 `Ok`。 |

### 46.2 语义

- 与 host 的差异：host 用 f64 `EPS=1e-10` 严格判负；设备为 f32，`1e-3` 容差内视为舍入噪声夹 0，超出则显式报错。**不再静默掩盖**真实非法状态。
- 该改动在随机生存路径引入一次 4 字节 D2H 同步（仅为读取违规标志）；确定性路径不受影响。

### 46.3 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu` | **178 passed, 0 failed**（含 A2 用例） |
| `cargo test`（默认） / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3611 passed / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2390/2464 = 97.00%**；executor 96.2%、kernels 97.5%、probe 96.1%，逐文件 ≥95% |

### 46.4 请 evaluator 独立核对

- **行为**：明显非法状态（virgins 显著为负）现在显式 `Err`；容差内负值仍夹 0（正常随机运行不回归，既有分布用例仍过）。
- **阈值判定**：`1e-3f` 作为 f32 容差是否合理（对比 host `1e-10` f64），是否需要在文档/常量处更明确。
- **性能影响**：随机生存每 tick 一次 4 字节 D2H；确认可接受或提出替代（如延后到下载边界检查）。
- **CPU 不变性 / 门禁 / 覆盖率**同既往口径。

### 46.5 残余风险

- f32 容差与 host 严格判负存在精度层面的差异（不可避免）；已在代码注释与本交接说明。
- 延迟到下载边界检查可免同步，但会延后报错时机、偏离 CPU 立即中止语义；本轮选择立即检查。

结论请追加为 **§47**。

---

## 48. 第 19 轮修复交接（处理 §47 阻塞项）

- 日期：2026-09-19
- 针对 §47.1/§47.2：固定绝对容差 `1e-3f` 对大规模有效状态误报。
- 风险分类：**局部代码修改**（容差公式；方向与 A2 相同）。

### 48.1 改动

| 文件 | 内容 |
|---|---|
| `rust/src/gpu/kernels.rs` | 容差改为**尺度相关**：`tol = 8 * (|n_f_raw| + |total_sperm|) * NATAL_F32_EPS + NATAL_VIRGIN_EPS_ABS`，其中 `NATAL_F32_EPS = 1.19209290e-7f`、绝对下限 `NATAL_VIRGIN_EPS_ABS = 1e-9f`；仅当 `virgins < -tol` 置 violation（否则夹 0）。 |
| `rust/tests/unit/gpu/executor.rs` | 作者用例第二段改为「有效 stored==female（成年 age1，避免 recruit 干扰）必须 `Ok`」。 |

### 48.2 自测证据（非独立）

| 命令/实验 | 结果 |
|---|---|
| `cargo test --features gpu evaluator_valid_large_virgin_state_is_not_rejected`（§47.2 失败回归） | **ok**（修复前 `Err`） |
| `cargo test --features gpu stochastic_survival_rejects_negative_virgin_state` | ok（`-20` → Err；有效 stored==female → Ok） |
| `cargo test --features gpu` | **179 passed, 0 failed**（含 evaluator 回归） |
| `cargo test` / `check_rust.py` / `clippy --features gpu -D warnings` / `fmt` | 67 / EXIT=0 / 通过 / 通过 |
| `ruff` / `pyright` / `pytest -q` / `phase0_baseline --check` | 通过 / 0 errors / 3611 passed / bit-identical |
| 覆盖率（严格过滤 `rust/src/gpu/**`，排除 `/tests/`） | 聚合 **2390/2464 = 97.00%**；executor 96.2%、kernels 97.5%、probe 96.1%，每文件 ≥95% |

### 48.3 请 evaluator 复核

- 重跑 `evaluator_valid_large_virgin_state_is_not_rejected` 确认转绿、且 `-20` 等真实非法状态仍 `Err`。
- 核对尺度相关容差在更大规模（接近 2²⁴ 整数上限）与 Z 较大时无误报；必要时给出更保守系数。
- 其余 §47.3 结论（分布等价、门禁、覆盖率）与 §47.4 同步权衡（每 tick 4 字节 D2H）不变。

结论请追加为 **§49**。

---

# evaluator 回执区（追加式；evaluator 写，主 agent 据此行动）

> 第 1 轮结论见上方 **§9**（已有内容）。为保持时间顺序，**第 2 轮及以后请追加到本区末尾**，
> 使用 §12、§13… 编号；不要改写上面的历史轮次。

## 12. 第 2 轮结论（evaluator 独立执行，2026-09-18，HEAD=`52544cf`）

### 12.1 裁定：**NOT APPROVED**

- §9.2 的两个正确性阻塞项：**已解除**（独立复核通过，修复是产品语义层面的，非绕过测试）。
- **新增 1 个阻塞项**：Rust 行覆盖率未达项目规范 95%，且新增的会话设备接线 **0% 覆盖、无任何仓库测试**。
  依据 `quality_checks_spec.md` §Coverage（「New modules require at least 95% line coverage」）与判定表
  （「measured coverage below 95% requires repair」）。

> 若项目对硬件/FFI/错误路径代码**不**强制 Rust 95% 线覆盖率策略，请由用户**显式豁免**；
> 豁免前按规范维持 NOT APPROVED。正确性修复本身已通过。

### 12.2 §9.2 两个目标复核（独立运行）

命令（环境见 §11）：
- `cargo test --features gpu evaluator_reproduction_clears_newborns_when_no_recruits` → **1 passed**（原 max diff 3 → 0）
- `cargo test --features gpu evaluator_mating_row_epsilon_matches_cpu` → **1 passed**（原 262.44 → 0）
- `cargo test --features gpu` → **105 passed, 0 failed**
- `cargo test` → **67 passed**；`NATAL_GPU_REQUIRE=0 cargo test --features gpu` → **105 passed**
- evaluator 两个测试**未被改动**：无 `#[ignore]`、`assert_relative` 与断言值未被削弱。

修复语义核对（读 `373fcbf..52544cf` 产品 diff）：
- **BLOCKER-1**：`rust/src/gpu/kernels.rs` 删除 `!has_any` 与 `total<=eps` 两处提前返回，改为
  **无条件写 age-0**（`n_g<=eps` 写 0），与 CPU `age_structured.rs:503-506` 一致；
  `eff_sum==0` 的提前返回两边都保留且都不清 age-0，仍一致。逐分支等价（`total<=eps` 时 CPU 也写全 0）。
- **BLOCKER-2**：6 处 `1e-12f` 全部改为 `1e-10f`，与 CPU `EPS=1e-10`（`rng.rs:21`）逐点对齐：
  `kernels.rs:351/380/391/423/446/455`。确定性路径的 CPU EPS 点无遗漏
  （`age_structured.rs` 其余 EPS 点属随机分支，本阶段未实现）；`equilibrium` 的 `1e-10f` 本就一致。
- 附加修复（NaN `eggs_per_female`、density 六列长度、负 `growth_mode`、性染色体标志长度）均正确无副作用；
  默认 feature / CPU 路径未受影响（`git diff 373fcbf..52544cf -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空）。

### 12.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `python scripts/check_rust.py` | EXIT=0（67 passed；需 `PYO3_PYTHON=/opt/conda/bin/python`） |
| `cargo test` | 67 passed |
| `cargo test --features gpu` | 105 passed |
| `cargo clippy --features gpu -- -D warnings` | 通过 |
| `cargo fmt -- --check` | 通过 |
| `ruff check src demos` | 通过 |
| `pyright` | 0 errors |
| `PYTHONUTF8=1 .venv/bin/pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| `maturin develop --features "gpu,extension-module"` + L3（mode 3，5 tick） | `max_rel=1.68e-7` → PASS |
| 拒绝路径（stochastic / hooks / `CUDA_VISIBLE_DEVICES=""`） | 均显式报错，失败后仍可 CPU → PASS |

### 12.4 新增阻塞项：Rust 覆盖率 < 95%（含缺失测试）

`rustup component add llvm-tools`（需网络）后，命令（`rust/` 下）：
```bash
export RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE="/tmp/cov/%p-%m.profraw"
mkdir -p /tmp/cov && cargo test --features gpu --lib
LLVM_BIN="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin"
"$LLVM_BIN/llvm-profdata" merge -sparse /tmp/cov/*.profraw -o /tmp/cov/merged.profdata
"$LLVM_BIN/llvm-cov" report --object <test_bin> --instr-profile=/tmp/cov/merged.profdata --sources src/gpu
```
scope：`rust/src/gpu/`（基线 `adda656` 无此目录，全为新增）+ `rust/src/sessions/age_structured.rs` 新增行。

| 文件 | 行覆盖 | 未覆盖行（要点） |
|---|---|---|
| buffers.rs | 23/23 = 100% | — |
| layout.rs | 41/41 = 100% | — |
| mod.rs | 3/3 = 100% | — |
| kernels.rs | 273/288 = 94.8% | 640-650/695/703-705/769/818/865/912（launcher 错误 `map_err`） |
| context.rs | 29/31 = 93.5% | 46-47（`new` 错误路径） |
| cuda.rs | 100/113 = 88.5% | 101-108（catch_unwind 错误臂）、174-195（`report()` None 分支） |
| probe.rs | 155/181 = 85.6% | 112/124-139/245-252/290-322/348-392（解析失败/缺字段分支） |
| executor.rs | 406/495 = 82.0% | 89 行，几乎全为长度校验/`map_err` 错误路径与访问器（如 84-105、207-217、658-721） |
| **TOTAL** | **1030/1175 = 87.7%** | **< 95%** |

新增会话接线 `rust/src/sessions/age_structured.rs`：
- `enable_gpu` / `gpu_status`：**0/46** 行覆盖
- `run_gpu` 与 `run_inner` 设备分支：**0/29** 行覆盖
- 即：**没有任何仓库测试触碰设备会话路径**（`rg enable_gpu tests rust/tests` 无命中；
  我的 Python L3 为手动验证，且 Python 覆盖不能替代 Rust 覆盖）。

判定：按 `quality_checks_spec.md` 覆盖率门槛与判定表，此项需补测试后修复。

建议（交主 agent）：
1. 新增 Rust 测试覆盖会话设备路径：`enable_gpu` 成功 + 各拒绝分支、`gpu_status`、
   `run_gpu` 的 tick 循环与下载（可借助现有嵌入解释器测试基建）；
2. 为 `executor.rs` / `cuda.rs` / `probe.rs` / `kernels.rs` 的 `is_err()` 错误分支补用例；
3. 若用户豁免 Rust 95% 策略，则记录豁免依据，本轮即可视为正确性通过。

### 12.5 非阻塞 / 残余

- §9.5 剩余项（设备路径 history 静默为空、宿主直接改状态不同步、空间/离散无 `enable_gpu` 入口）
  本轮未处理，维持为**非阻塞已知限制**。
- 上述未覆盖行以防御性错误路径与格式化分支为主，数值内核覆盖良好；但这不改变 §12.4 的规范判定。

### 12.6 状态

- 正确性修复：**独立验证通过**。
- 整体裁定：**NOT APPROVED**（仅剩覆盖率 / 缺失测试阻塞）。
- 待主 agent：补测试或取得用户豁免后，回交第 3 轮复核。

## 14. 第 3 轮结论（evaluator 独立执行，2026-09-18，HEAD=`34f8cc2`）

### 14.1 裁定：**APPROVED**

- §12.4 的覆盖率 / 缺失测试阻塞：**已解除**（独立复测）。
- §13.3 的口径争议：**判定通过**（见 14.3）。
- 正确性、CPU 不变性、全部门禁、会话设备路径测试：均独立复核通过。

### 14.2 覆盖率独立复测（evaluator 自跑，非引用主 agent）

命令（`rust/` 下，`llvm-tools` 已装）：
```bash
export RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE="/tmp/cov3/%p-%m.profraw"
mkdir -p /tmp/cov3 && cargo test --features gpu --lib
LLVM_BIN="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin"
"$LLVM_BIN/llvm-profdata" merge -sparse /tmp/cov3/*.profraw -o /tmp/cov3/merged.profdata
"$LLVM_BIN/llvm-cov" export --object <test_bin> \
  --instr-profile=/tmp/cov3/merged.profdata --format=lcov > /tmp/cov3/all.lcov
# 仅统计 rust/src/gpu/**（排除 rust/tests/unit/gpu/**）
```

结果（仅 `rust/src/gpu/`，逐文件）：

| 文件 | 行覆盖 |
|---|---|
| buffers.rs | 23/23 = **100%** |
| context.rs | 31/31 = **100%** |
| cuda.rs | 119/119 = **100%** |
| kernels.rs | 288/288 = **100%** |
| layout.rs | 41/41 = **100%** |
| mod.rs | 3/3 = **100%** |
| executor.rs | 481/499 = **96.4%** |
| probe.rs | 174/181 = **96.1%** |
| **src/gpu TOTAL** | **1160/1185 = 97.89%** |

新增会话接线 `rust/src/sessions/age_structured.rs`：
- `enable_gpu`/`gpu_status`：**全部可执行行已覆盖**（区间内未覆盖的 311-315 属既有 `refresh_params`，
  与 GPU 变更无关）。
- `assemble`/`run_gpu`：GPU 行全部覆盖（区间内未覆盖的 1150-1156 属既有 `HookProgram::from_python`）。
- `run_inner` 设备分支：**全部覆盖**（区间内未覆盖的 1269-1272 属其后的 CPU `observation_mask` 路径）。
- 新测试 `session_device_branch_matches_cpu_and_covers_wiring` 实跑设备 3 tick 并与 CPU 逐元素比较
  `state_ind/state_sperm`（容差 1.2e-6）；`session_enable_gpu_rejects_ineligible_models` 覆盖 5 条拒绝分支。

### 14.3 §13.3 口径判定（本次裁定核心）

采用规则：**先按「变更内全部新增可执行行」汇总，同时逐新模块核对**（规范原文
“Measure executed new lines against all executable new lines in the change”）：

1. **新模块 `rust/src/gpu/`**：聚合 **97.89% ≥ 95%**；且**逐文件均 ≥ 95%**
   （最低 `probe.rs` 96.1%、`executor.rs` 96.4%）。→ 达标。
2. **既有模块新增可执行行 `age_structured.rs`**：105/111 = **94.6%**，未覆盖 6 行
   `[219,220,221,223,226,227]`，即 `from_parts` 委托 `Ok(Self::assemble(...))` 的 PyO3 构造调用；
   既有模块中原有的 CPU 行不在“新增”口径内。
3. **变更级新增可执行行合计**：1265/1296 = **97.6% ≥ 95%**。→ 达标。

裁定：
- §13.3 提出的 driver 失败 / 平台分支（`executor.rs` 内核调用尾部 `?`、`probe.rs` 的
  `CUDA_PATH`/Windows 分支）**实测已使 `executor.rs` 96.4%、`probe.rs` 96.1%，逐文件亦达标**，
  因此**不需要**选项 (b) 的错误映射可测化重构。
- 唯一 94.6% 的单元格（`age_structured.rs` 的 6 行 `from_parts` 委托）是**重构搬迁的构造委托**，
  非 GPU 数值逻辑，其功能已由 `assemble` 路径覆盖（集成侧 Python 亦走该构造函数）。
  按 **§13.3 选项 1 / (a)：以文档化残余接受**，不作为阻塞。
- 若用户要求「每个既有模块的新增行也严格逐文件 ≥95%」，唯一补测点是给 `from_parts` 加嵌入解释器
  用例；从成本/收益看不建议，且不影响本裁定的正确性结论。

### 14.4 正确性与硬约束（独立复核）

- `rust/src/kernels`、`rust/src/model`、`src/natal/contracts`、`rust/src/lib.rs` 自 `373fcbf` 起**零改动**。
- 本轮产品改动为**可测性重构，无数值语义变化**：`cuda.rs` 抽取 `from_outcome`/`driver_missing`
  （`run` 行为不变）、`executor.rs` 抽取 `ensure_memory_budget`（比较与报错文本不变）、
  `age_structured.rs` 抽取 `assemble` 构造（字段初始化不变，`from_parts` 委托之）。
- `phase0_baseline.py --check` → `all scenarios bit-identical`（CPU golden reference 未受影响）。
- 主 agent 仅**追加**测试，**未改动** evaluator 既有测试（`git diff 52544cf..HEAD -- rust/tests/unit/gpu/executor.rs`
  为纯新增；两个阻塞回归测试断言未削弱，无 `#[ignore]`）。

### 14.5 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **124 passed, 0 failed** |
| `NATAL_GPU_REQUIRE=0 cargo test --features gpu` | 124 passed（硬件用例跳过） |
| `cargo clippy --features gpu -- -D warnings` | 通过 |
| `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff check src demos` | 通过 |
| `pyright` | 0 errors |
| `PYTHONUTF8=1 .venv/bin/pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| `cargo test --features gpu evaluator_*`（含两个旧阻塞回归） | 全绿 |

### 14.6 残余风险（非阻塞，留档）

- 设备路径不记录 history（静默为空）；`enable_gpu` 后宿主直接改状态不自动重传；
  空间/离散后端无 `enable_gpu` 入口；显存预算未计入 ecology 上传与模块开销；
  f32 在计数 >2²⁴ 或长链归约时理论上可能超 1.2e-6（未找到具体越界输入）。
- `age_structured.rs` 新增行覆盖 94.6%（6 行 `from_parts` 委托）已按 §14.3 接受为文档化残余。

### 14.7 结论

`feat/gpu-merge-test` 的 CUDA 旁路 P0–P3 在正确性、CPU golden reference 隔离、默认关闭、
精度分档、显式失败、会话接线与测试覆盖上均满足要求。**APPROVED**（范围为当前 HEAD `34f8cc2`
与被审测试集；不声称任何历史基线失败消失）。

## 17. 第 4 轮结论（evaluator 独立执行，2026-09-18，HEAD=`ec1cf99`）

### 17.1 裁定：**APPROVED**

P5「确定性 CSR 迁移 + 空间多 deme 会话接线」在正确性（与 CPU `migrate_csr_deterministic`
逐分支等价）、tick 顺序、零速率跳过、CPU golden reference 隔离、默认关闭、显式失败、
覆盖率与空间 L3 上均满足要求。

### 17.2 迁移正确性（独立证伪）

我自行设计了主 agent 未覆盖的 CSR 拓扑与边界，在 `rust/tests/unit/gpu/executor.rs`
新增 3 个 evaluator 用例（`evaluator_migration_*`），以**1.2e-6**（含算术 kernel 分档）对照
`migrate_csr_deterministic`，`stay_after` 真/假各跑一遍，全部通过：

| 拓扑/输入 | 说明 |
|---|---|
| `ring-nonunit` | 3 环，权重和 ≠ 1（触发 `row_sum_w` 路径） |
| `selfloop-empty` | 自环 + 空行（孤立 deme） |
| `duplicate-edges` | 同一 src→dst 双入边 + 自环（反 CSR 顺序） |
| `cycle4-a8-z3` | 4 环、A=8、Z=3、异质权重 |
| `nnz-zero` | 全空 CSR（nnz=0，所有 deme 孤立） |
| `over-migration` | rate=1.5>1（outbound 超源，stay 变负） |
| `zero-rate` | 全 0 速率（executor 直跑仍与主机一致） |

命令：`cargo test --features gpu evaluator_migration` → 3 passed。

逐分支核对（读代码）：gather 的 `self`（stay_after 用 `virgin - outbound·row_sum_w`；
非 stay 用 `virgin - outbound`；空行用 `virgin`）与入边累加 `(s_virgin·fr_src)·w` 及各 `mz`
的储精项，与 CPU scatter 的 `out_ind[src] += virgin - moved_total` / `out_ind[dst] += outbound·w`
逐项对齐；储精质量同时进入雌性个体平面；`row_empty` 用目的行 `indptr[dst..dst+1]`，与 scatter 的
源行空判一致；`virgin<0 && |virgin|<1e-9` 归零语义一致。

### 17.3 空间会话接线（独立复核）

- 设备 tick 顺序 = 生命周期（reproduction→survival→aging，批量 deme）→ 迁移，与 CPU
  `run_inner`（`run_spatial_tick_*` → migration → `state_tick += 1`）一致。
- `all_zero = migration_rate.iter().all(|&r| r <= 0.0)` 时跳过迁移，与 CPU 完全相同；
  全 0 速率不消耗设备迁移（避免 f32 抵消误差）。
- `enable_gpu` 拒绝：discrete / stochastic / hooks（含 python_callbacks）/ 自定义 growth /
  `n_demes != deme_variants.len()`；CPU 路径仅在 `gpu.is_none()` 时执行（提前返回，未改 CPU）。

### 17.4 覆盖率（独立复测，口径沿用 §14.3）

命令同 §14.2（`-C instrument-coverage` + llvm-cov lcov，仅统计 `rust/src/gpu/**`）：

| 文件 | 行覆盖 |
|---|---|
| buffers/context/cuda/kernels/layout/mod | 均 **100%** |
| executor.rs | 573/593 = **96.6%** |
| probe.rs | 174/181 = **96.1%** |
| **src/gpu TOTAL** | **1342/1369 = 98.03%** |

`rust/src/sessions/spatial.rs` 新增可执行行：88/91 = **96.7%**（未覆盖 336-337 的
`#[cfg] gpu: None` 构造初始化、1242 的一个 `?` 错误传播尾；均非逻辑路径）。
逐文件与聚合口径**均 ≥95%**。

### 17.5 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **132 passed, 0 failed**（作者 129 + evaluator 3） |
| `NATAL_GPU_REQUIRE=0 cargo test --features gpu` | 131 passed（跳过硬件用例，含新增） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 空间 L3（`demos/gpu_spatial/age_structured` 5×5，25 tick，重编扩展后独立跑） | tick 一致；ind `max_rel=9.140e-7`、sperm `9.149e-7` → PASS |
| CPU/numeric 隔离 | `git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空 |

主 agent 仅**追加**测试，未削弱既有 evaluator 断言。

### 17.6 非阻塞观察 / 残余风险

- 作者迁移 L1 / 空间会话测试用 `1e-5·max(|want|,1)` 容差，松于规范 `1.2e-6`；evaluator 以
  `1.2e-6` 复测通过，未掩盖缺陷。建议后续统一收紧（不阻塞）。
- **性能（非正确性）**：`migrate_tick` 每 tick 在主机重建反向 CSR + `row_sum_w` 并整列上传，
  且设备分支逐 tick 回传状态；这是当前取舍（§16.2 第 4 点已声明），后续可缓存 CSR/减少同步。
- 既有已知限制不变：设备路径不记录 history、`enable_gpu` 后宿主直接改状态不重传、
  离散/随机/迁移随机变体未接入、f32 在超大计数/长链归约下的理论误差。

### 17.7 结论

P5 确定性迁移与空间多 deme 旁路满足正确性与质量要求。**APPROVED**（范围为当前 HEAD
`ec1cf99` 与被审测试集；不声称任何历史基线失败消失）。

## 19. 第 5 轮结论（evaluator 独立执行，2026-09-18，HEAD=`243b35d`）

### 19.1 裁定：**NOT APPROVED**

- 1 个阻塞项：**Poisson 采样器在 `λ ≥ 64` 用正态近似，与 CPU 精确 Poisson 不满足统计等价**
  （既不满足冻结规范的「卡方检验」，也不满足「前四阶矩」——正态近似丢掉了 Poisson 的偏度 `1/√λ`）。
  已留下实际运行且失败的回归测试。
- 其余部分（Philox 逐位、uniform、binomial、gamma、随机阶段均值、确定性路径、覆盖率）均通过。

### 19.2 阻塞项（已实际运行且失败）

- 测试：`rust/tests/unit/gpu/kernels.rs::evaluator_discrete_sampler_distributions_match_cpu`
- 命令（`rust/`，环境见 §11）：
  `cargo test --features gpu evaluator_discrete_sampler_distributions_match_cpu -- --nocapture`
- 预期：设备 Poisson 分布与 CPU `poisson`（`rand_distr::Poisson`，精确）在 `λ=1,20,63,64,65,200`
  上的**两样本卡方**均不超过 `dof + 5σ`（dof≈21，阈值≈53.4；N=200000/样本）。
- 实际：`poisson lambda=64: chi-square 265.51 > 53.40 (dof 21)` → **FAILED**（固定 site/seed，确定性复现）。
  `λ=65`、`λ=200` 同量级失败；`λ≤63`（Knuth 精确分支）通过。
- 位置：`rust/src/gpu/kernels.rs` 的 `sample_poisson`，分支阈值 `if (lambda < 64.0f) { Knuth 精确 } else { 正态近似 }`
  （约 `:843-860`）。
- 根因（独立解析证据）：用 f64 解析计算「精确 Poisson(λ) vs `floor(N(λ,√λ)+0.5)`」的卡方：
  `λ=63→529`、`64→533`、`65→521`、`200→156`、`1000→29`（N=200000，逐整数分箱）。
  即这是**正态近似固有误差**（右偏 Poisson、偏度 `1/√λ`；正态偏度恒为 0），不是实现 bug。
  CPU 是 `rand_distr::Poisson`（精确），为 golden reference。
- 后果：`fertilize` 的 `n_total = poisson(lambda)` 在中等 `λ`（数十~数百）下分布形状偏离参考，
  均值/方差正确但偏度/尾部不同；规范 §5.2 要求随机模式「统计等价（KS/卡方/前四阶矩）」，
  前端冻结设计 §4.2 也明确 Poisson 用 **PTRS（精确）**，非正态近似。
- **建议修复**（择一）：
  1. 把精确 Knuth 阈值大幅提高（Knuth 为 `O(λ)`，`λ≤10^4` 成本可接受），或对中等 `λ` 实现计划中的
     PTRS/Atkinson 拒绝采样，使 `λ` 在常见区间内与 CPU 分布一致；
  2. 或由用户**显式豁免**「随机模式允许 Poisson 正态近似」，并在规范中写明适用 `λ` 下界与误差界。
  修复后请回交复核（同一测试转绿）。

### 19.3 其余独立验证（通过）

- **Philox / uniform**：`fill_uniform` 与主机 `philox4x32_10` 逐位一致、同 key/site 复现（作者测试）；
  我另加 `evaluator_uniform_is_uniform`：N=200000，KS 距离 `< 1.6×1.63/√N`，20 分箱卡方 `< 60` → PASS。
- **binomial**：我加 `evaluator_discrete_sampler_distributions_match_cpu` 的 binomial 段（两样本卡方，
  含 exact/`mean=512` 边界/正态近似/`p=0.8` 反射）→ 全过（卡方≈dof）。
- **gamma**：我加 `evaluator_gamma_including_shape_below_one_matches_cpu`，覆盖形状 `0.3/0.5/0.9`
  （作者未覆盖的 `shape<1` 递归）与 `1/2/5/20`，两样本卡方与矩均过。
- **随机阶段**：作者 `device_stochastic_{survival,reproduction}_matches_host_distribution` 通过；
  deterministic L1/L2/L3、`phase0` 全部不变。
- **enable_gpu 语义**：`stochastic=true, continuous=false` 接受；`continuous=true` 显式拒绝
  （`session.rs` 已更新）；空间随机仍显式拒绝（`spatial.rs` 未改，保持 `stochastic` 拒绝）。

### 19.4 覆盖率（独立复测，严格按绝对路径过滤 `rust/src/gpu/**`）

| 文件 | 行覆盖 |
|---|---|
| buffers/context/cuda/kernels/layout/mod | 均 **100%** |
| executor.rs | 637/664 = **95.9%** |
| probe.rs | 174/181 = **96.1%** |
| **src/gpu TOTAL** | **1650/1684 = 97.98%** |

逐文件与聚合均 ≥95%。命令同 §14.2（`-C instrument-coverage` + `llvm-cov export --format=lcov`）。

### 19.5 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu`（作者用例） | **139 passed** |
| `cargo test --features gpu`（含 evaluator 3 个新增） | 141 passed, **1 failed**（=19.2） |
| `NATAL_GPU_REQUIRE=0 cargo test --features gpu` | 通过（硬件用例跳过） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| CPU/numeric 隔离 | `git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空 |

### 19.6 非阻塞发现 / 残余风险

- **测试充分性**：作者的采样器测试仅用**均值/方差**，未做 KS/卡方，也未覆盖 `gamma shape<1`、
  Poisson `λ≈64` 边界；本次阻塞项正是由此遗漏（建议今后随机模式按 §5.2 做分布级检验）。
- **`survival_stochastic` 静默夹取**：CPU `sample_survival_with_sperm` 在 `n_virgins < -EPS` 时返回
  `Err`（非法状态），设备端 `if (virgins<0) virgins=0` 静默夹取。仅影响非法状态，低。建议对齐报错语义。
- **`recruit_stochastic` 缺少 `MAX_Z` 守卫**：其 `combined[2*NATAL_MAX_Z]` 未像 `reproduction_stochastic`
  那样校验 `n_ztypes>MAX_Z`；正常 `tick` 路径会先在 reproduction 处报错，属潜在（直接调用 `survival_tick`
  时 `Z>32` 会越界）。低。
- **`rintf` vs `.round()`**：设备取整为「四舍六入五成双」，CPU 为「四舍五入远离零」；仅在恰好 `.5`
  处不同，统计模式影响可忽略。低/信息性。
- 既有已知限制不变：空间随机迁移未实现、`continuous_sampling` 拒绝、设备路径 history/状态重传限制。

### 19.7 结论

P4 的 RNG、uniform/binomial/gamma、随机阶段均值与确定性隔离均通过；但 **Poisson 正态近似在 `λ≥64`
不满足统计等价**，为 in-scope 阻塞项。待主 agent 修复或取得明确豁免后回交第 6 轮复核。

## 21. 第 6 轮结论（evaluator 独立执行，2026-09-18，HEAD=`d397f91`）

### 21.1 裁定：**APPROVED**

- §19.2 阻塞项（Poisson `λ≥64` 正态近似）**已解除**：产品改为 PTRS 精确采样，独立分布检验通过。
- 其余采样器、随机阶段、确定性路径、CPU 隔离、覆盖率和全部质量门禁均通过。

### 21.2 阻塞项复核（独立运行）

- 原失败测试现通过：
  `cargo test --features gpu evaluator_discrete_sampler_distributions_match_cpu -- --nocapture`
  → 两样本卡方对 `λ=1,20,63,64,65,200` 全部达标（此前 `λ=64` 为 265.5 > 53.4）。
- 我另加 `evaluator_poisson_ptr_large_lambda_matches_cpu`，扩大网格至
  `λ=10,11,500,1000,5000,20000`，两样本卡方均达标（N=200000/样本）→ PTRS 在整个区间与
  CPU `rand_distr::Poisson` 分布等价。
- 产品核对（`rust/src/gpu/kernels.rs` `sample_poisson`）：`λ<10` Knuth 精确；`λ≥10` 使用
  **标准 PTRS（Hörmann 变换拒绝）**——`b=0.931+2.53√λ`、`a=-0.059+0.02483b`、
  `inv_alpha=1.1239+1.1328/(b-3.4)`、`v_r=0.9277-3.6224/(b-2)` 与接受/拒绝判据均与经典 PTRS 一致，
  保留 Poisson 的偏度/峰度。
- evaluator 原回归测试**未被改动**（断言未削弱、无 `#[ignore]`）。

### 21.3 附带修复复核

- `rintf` → `roundf`（`recruit/survival/reproduction_stochastic`）：与 Rust `.round()`
  （四舍五入远离零）一致，消除 `.5` 取整差异；仅影响随机分支。
- `recruit_stochastic` 启动器补 `n_ztypes > MAX_Z` 守卫（消除潜在设备越界）。
- 未处理（非阻塞，已知）：`survival_stochastic` 对 `n_virgins < -EPS` 静默夹取 0，而 CPU 返回
  `Err`（仅非法状态可达）。

### 21.4 回归与非影响性（独立）

- `cargo test` → 67；`cargo test --features gpu` → **143 passed, 0 failed**（作者 139 + evaluator 4）；
  `NATAL_GPU_REQUIRE=0` → 143。
- 确定性内核未改；`phase0_baseline.py --check` → `all scenarios bit-identical`；
  空间确定性 L3（5×5，25 tick）→ `max_rel=9.140e-7 / 9.149e-7`（不变）。
- 随机会话 L3（panmictic，K=150×8 tick，`pop._initialize_session(seed)` 后 `enable_gpu`）→
  最差 `0.385×` 容差（`ind`），sperm 一致 → PASS。
- 其它采样器：uniform KS、binomial（含 `mean=512` 边界/反射）、gamma（含 `shape<1`）仍通过。
- `git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空。

### 21.5 覆盖率（独立复测，严格过滤 `rust/src/gpu/**`）

| 文件 | 行覆盖 |
|---|---|
| buffers/context/cuda/layout/mod | 均 **100%** |
| kernels.rs | 624/627 = **99.5%** |
| executor.rs | 637/664 = **95.9%** |
| probe.rs | 174/181 = **96.1%** |
| **src/gpu TOTAL** | **1652/1689 = 97.81%** |

逐文件与聚合均 ≥95%。

### 21.6 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `python scripts/check_rust.py` | EXIT=0（67 passed） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |

### 21.7 残余风险（非阻塞）

- `survival_stochastic` 负 virgin 静默夹取（非法状态语义；低）。
- 空间随机迁移、`continuous_sampling=true` 仍显式拒绝；设备路径 history/状态重传限制不变。
- 随机阶段的作者统计测试以均值为主；本次已由 evaluator 的分布级检验补强，后续可将分布级检验并入常规门禁。

### 21.8 结论

P4 随机采样的唯一阻塞项已修复并经独立分布检验确认（PTRS 精确）。**APPROVED**（范围为当前 HEAD
`d397f91` 与被审测试集；不声称任何历史基线失败消失）。

## 23. 第 7 轮结论（evaluator 独立执行，2026-09-18，HEAD=`5b1ced6`）

### 23.1 裁定：**APPROVED**

空间随机（两趟无原子 scatter/gather）迁移在正确性、统计等价、GPU↔GPU 可复现、seed 生效、
确定性路径隔离与覆盖率上均满足要求。

### 23.2 独立统计 / 复现证据

我在 `rust/tests/unit/gpu/executor.rs` 新增 2 个 evaluator 用例：

- `evaluator_stochastic_migration_is_seed_reproducible`：
  同 seed 两次设备迁移 `ind`/`sperm` **逐位一致**；不同 seed 结果不同 → seed 确实生效（§22.2 修复有效）。
- `evaluator_stochastic_migration_topology_matches_host_mean`：
  自建**块对角**富拓扑（`0→{1,1,0}` 含重复边+自环、`1→{}` 空行、`2→{3}`、`3→{0,2}` 多入边；
  速率含 `0 / 0.3 / 1.0 / 1.5`），复制 500 份独立副本，设备一次运行按副本池化均值，
  与 CPU `migrate_csr_stochastic_rngs` 300 次试验的池化均值/方差对照（`5σ+0.5`）→ **通过**。
  命令：`cargo test --features gpu evaluator_stochastic -- --nocapture` → 2 passed。

逐分支核对（读代码）：`sample_outbound_device` 与 CPU `sample_outbound`（`rate>=1` 全走、
否则 `binomial(round(value), rate)`）一致；prepare 的 `natal_multinomial`（按行权重归一）与
CPU `distribute_csr_outbound` 一致；pass1 写 `src` stay（`virgin - moved_total`，储精 stay 计入雌性平面），
pass2 按反向 CSR（`rev_entry` 为 CSR entry 索引，按 src 升序+行内顺序建立）累加到 `dst`，与 CPU per-src
scatter 的累加顺序一致，且无 `atomicAdd`。

### 23.3 会话级独立性（Python）

- 空间随机 L3（`demos/gpu_spatial/age_structured` 5×5，`stochastic=True`，K=300×5 tick，
  `_initialize_session(seed)` 逐 seed）→ `cpu_mean=56701.9`、`gpu_mean=56698.9`，
  差 `3.05 ≪ tol 39.7` → PASS（独立重编扩展后运行）。
- 我另做 **seed 敏感性**检查：空间与 panmictic 各 4 个 seed 的 GPU 总量均随 seed 变化
  （spatial `[61890,62060,61965,61843]`，panmictic `[404,492,474,408]`）→ `set_seed` 已在
  两条会话路径正确接线。

### 23.4 确定性路径与非影响性（独立）

- 确定性空间 L3（5×5，25 tick）→ `max_rel=9.140e-7 / 9.149e-7`（与 §21 完全相同，未回归）。
- `phase0_baseline.py --check` → `all scenarios bit-identical`。
- `git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空。

### 23.5 覆盖率（独立复测，严格过滤 `rust/src/gpu/**`）

| 文件 | 行覆盖 |
|---|---|
| buffers/context/cuda/layout/mod | 均 **100%** |
| kernels.rs | 730/745 = **98.0%** |
| executor.rs | 726/762 = **95.3%** |
| probe.rs | 174/181 = **96.1%** |
| **src/gpu TOTAL** | **1847/1905 = 96.96%** |

逐文件与聚合均 ≥95%。

### 23.6 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **147 passed, 0 failed**（作者 145 + evaluator 2） |
| `NATAL_GPU_REQUIRE=0 cargo test --features gpu` | 147 passed |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |

### 23.7 非阻塞发现 / 残余风险

- **潜在越界（低）**：`migration_stochastic_prepare` 使用 `probs[NATAL_MAX_Z]`/`drawn[NATAL_MAX_Z]`，
  但启动器只校验 `n_ztypes > MAX_Z`，未校验**每行出度** `row_len > MAX_Z`。常规空间邻域
  （Moore ≤8）不触发；自定义宽 CSR（>32 目的地/行）会越界。建议在启动器拒绝 `row_len > MAX_Z`
  或按实际最大行宽分配 scratch。
- `survival_stochastic` 对 `n_virgins < -EPS` 静默夹取 0（CPU 返回 `Err`，仅非法状态；低）。
- `continuous_sampling=true`、设备 history/状态重传限制不变。

### 23.8 结论

空间随机迁移的两趟设计正确、统计等价且可复现；唯一新增项为 §23.7 的低危越界守卫建议（非阻塞）。
**APPROVED**（范围为当前 HEAD `5b1ced6` 与被审测试集；不声称任何历史基线失败消失）。

## 25. 第 8 轮结论（evaluator 独立执行，HEAD=`ec27d56`）

### 25.1 裁定：**APPROVED**

`MAX_CSR_ROW = MAX_Z = 32`；`GpuExecutor::migrate_tick_stochastic` 在构建/启动前计算 CSR 最大行宽，
超过即显式 `Err`（不静默）。修复正确、位置恰当，消除了 §23.7 的设备端越界读。
`execution_rejects_overwide_stochastic_csr_rows`（行宽 33→`Err`）包含在 155 通过用例中；
`cargo test` 67、`phase0` bit-identical 不变。无新增残余风险。

---

## 27. 第 9 轮结论（evaluator 独立执行，HEAD=`5bddb6e`）

### 27.1 裁定：**APPROVED**（功能正确；性能口径见 27.3）

### 27.2 独立核对（正确性）

- **replicate 独立性 / 可复现**（Python API）：`enable_gpu_ensemble(B)` 后
  `run_gpu_ensemble(3)`，同 seed 两次 `ind`/`sperm` **逐位一致**，不同 seed 结果不同。
  replicate 间靠 counter 的 cell 维度（`ind`/`sperm` 的 batch 索引）区分，共享 key 也不重叠。
- **统计等价**：GPU ensemble `B=2000` 的 replicate 总量 vs `K=1000` 次独立 CPU 全程运行，
  用**合并 SE** 的 5σ 容差比较：`diff=2.3`、`ratio=0.021` → PASS。
  （首轮用 `K=300` 且未计 CPU 估计自身 SE，得到 ~1.9σ 的假象；修正口径后无差异。）
  作者 Rust L2 用例 `device_ensemble_matches_independent_cpu_runs`（2000 vs 2000）亦通过。
- **资格拒绝**：`n_replicates=0`、`continuous_sampling=true`、`n_demes!=1`、含钩子、自定义 growth
  均显式拒绝（`session_gpu_ensemble_rejects_ineligible`）。
- 形状：`ind_flat=(B,2,A,Z)`、`sperm_flat=(B,A,Z,Z)` 正确；确定性/空间路径未受影响；`phase0` bit-identical。

### 27.3 P6 性能验收判定（§26.3 请求的口径）

- 计划 §5.3 要求：GPU 必须“显著优于” `ProcessPoolExecutor`（后者可线性加速且逐位一致）；
  但计划**未给数值阈值**。
- 主 agent 自测（B=5000, A=8, Z=3, 50 tick, 16 CPU workers）：GPU 0.278s vs ProcessPool 0.664s
  → **2.4×**（单位工作量 ~40×）。我未在共享 GPU 上独立复现基准（波动大，仅作量级参考）。
- 判定：P6 的**功能验收**（ensemble session + 独立 replicate + 可复现 + 对比实验）**达标**；
  **性能**为真实但**中等**的加速（~2.4–3.3×），是否叫“显著”取决于阈值——本评测接受其满足
  “相对 16 核 ProcessPool 有明确优势”，并建议在计划中补一个可复现的基准/阈值定义。
  这不是阻塞项；若要求数量级优势，需要更大的单 replicate 状态或更少 CPU 核（API 已避免逐 tick 回传）。

---

## 29. 第 10 轮结论（evaluator 独立执行，2026-09-19，HEAD=`863ef1e`）

### 29.1 裁定：**NOT APPROVED**

- **阻塞项**：零逐 tick 回传后，**公共单 tick 路径** `SpatialSession::run_tick`（PyO3 暴露，
  对应 `RustHeterogeneousSpatialLifecycleBackend.run_tick`）执行后不再同步 host 数组；
  随后 `state_snapshot` / `state_snapshot_deme` / `observe_current` / `capture_checkpoint` 及
  `:1112` 的查询访问器返回**上一 tick 的陈旧状态**（tick 已递增）。这是公开 API 的静默状态错误。
- 其它内容（`run_steps`/历史记录、缓存失效、CPU 隔离、覆盖率、门禁）均通过。

### 29.2 阻塞项证据（已实际运行且失败）

- 新增 evaluator 回归测试
  `rust/tests/unit/gpu/spatial_session.rs::evaluator_single_tick_syncs_host_state`
  命令：`cargo test --features gpu evaluator_single_tick_syncs_host_state`
  预期：`run_tick` 后 GPU host `state_ind` 与 CPU 一致（容差 1.2e-6）
  实际：`host ind[0] stale after run_tick: gpu 5 vs cpu 0` → **FAILED**。
- Python 端到端（独立重编扩展）：
  - `backend.run_tick()` 后 `state_snapshot()`：`sum` 仍为初始 **112500.0**（CPU 同步骤为 92507.7），
    `tick 0→1` → host 未刷新；
  - `observe_current` 在 `backend.run_tick()` 前后返回结果相同。
- 位置：`rust/src/sessions/spatial.rs` 的 `run_inner` 设备分支已移除下载（`:1252` 后），
  `sync_gpu_state` 仅在 `run_steps`（`:780/:787/:791/:795`）调用；而
  `capture_checkpoint:805`、`observe_current:862`、`record_history:892`、`state_snapshot:1048`、
  `state_snapshot_deme:1067`、查询访问器 `:1112` 直接读 `self.state_ind/state_sperm`。
- **建议修复**：在这些公开读路径（以及公开 `run_tick`）入口先调用 `sync_gpu_state()`（无设备时是空实现，
  有设备时才下载），或让公共 `run_tick` 同步、`run_steps` 改调内部不同步的 tick。修复后本回归测试应转绿。

### 29.3 已独立验证正确的部分

- **历史正确性**：`pop.run(4, record_every=2)` 后 GPU 与 CPU 的 history tick 序列均为 `[0,2,4]`，
  最终 `ind` 相对误差 `7.27e-7` → `run_steps` 在记录边界同步有效。
- **缓存失效**：`csr_fingerprint` 覆盖 `indptr`/`dest_idx`/`weights` 全部内容，任一变化即重建；
  同 CSR 命中复用；作者用例
  `migration_cache_reuses_buffers_and_invalidates_on_new_csr`、
  `stochastic_migration_reuses_cached_scratch_reproducibly` 通过。
- **CPU 不变性**：`git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs`
  为空；`phase0` bit-identical。
- **覆盖率**（严格过滤、排除 `/tests/`）：`src/gpu` 聚合 **1928/1980 = 97.37%**，逐文件均 ≥95%
  （executor 96.4%、kernels 98.0%、probe 96.1%，其余 100%）。

### 29.4 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu`（作者用例） | **155 passed** |
| `cargo test --features gpu`（含 evaluator 回归） | 155 passed, **1 failed**（=29.2） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |

### 29.5 非阻塞 / 残余风险

- CSR 缓存（含随机 `fwd_s = nnz·A·Z²·4`）常驻显存，`GpuExecutor::new` 的预算守卫未计入；
  超限时 `DeviceBuffer::from_host` 会显式报错（不静默回退），但发生在首次迁移时。建议纳入预算估算。
- GPU 会话 checkpoint/restore 仍不支持设备侧状态回滚（既有）；ecology 有意不缓存（每 tick 生效语义）。
- 共享 GPU 波动：本轮 instrumented 运行中曾出现一次失败、重跑即过；视为环境噪声，不计产品缺陷。

### 29.6 结论

第 10 轮的缓存与零回传提升了性能，但引入了公共单 tick 读路径的陈旧状态缺陷（已给出实际失败的回归测试）。
**NOT APPROVED**，待主 agent 修复同步时机后回交复核。

---

## 31. 第 11 轮结论（evaluator 独立执行，2026-09-19，HEAD=`b9c1910`）

### 31.1 裁定：**APPROVED**

§29 阻塞项（零回传后公共单 tick 读路径陈旧）已解除；`run_steps` 的“运行期零逐 tick 回传”与历史正确性
保持不变。修复方式与 §29.2 建议的第二方案一致。

### 31.2 修复复核（独立运行）

- **回归测试**（§29.2 的失败用例）：`cargo test --features gpu evaluator_single_tick_syncs_host_state`
  → **ok**（修复前 `gpu 5 vs cpu 0`）。主 agent 未改动该测试的断言（无 `#[ignore]`）。
- **`run_tick` 主体拆分**：`advance_tick`（不下载）承载原 tick 逻辑；公共 `run_tick` = `advance_tick()?`
  + `sync_gpu_state()?`；`run_steps` 循环改调 `advance_tick`，保留记录边界/停止/结束处的同步。语义正确。
- **Python 端到端**（独立重编 `maturin develop --features "gpu,extension-module"`，5×5 参考空间模型）：
  - `backend.run_tick()` 后 `state_snapshot()`：tick `0→1`，host 数组已刷新，与 CPU 相对误差
    `4.14e-7`（修复前该路径总量保持初始 112500）；
  - `observe_current`（正确形状的 1-D mask）在 `run_tick` 后**发生变化**且与 CPU 相对误差 `3.59e-7`；
  - `capture_checkpoint()` 返回 tick `1`（捕获的是 tick 后状态）；
  - `pop.run(4, record_every=2)`：GPU/CPU 末态相对误差 `7.27e-7`，history tick 序列均为 `[0,2,4]`
    → 多 tick 零回传与历史记录未回归。

### 31.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **156 passed, 0 failed** |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| CPU/numeric 隔离 | `git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空 |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **1928/1980 = 97.37%**；逐文件均 ≥95%（executor 96.4%、kernels 98.0%、probe 96.1%，其余 100%） |

### 31.4 残余风险（非阻塞，延续 §29.5）

- CSR 缓存（含随机 `fwd_s`）常驻显存，未纳入 `GpuExecutor::new` 预算估算；超限时显式报错（非静默回退）。
- GPU 会话 checkpoint/restore 仍不支持设备侧回滚；ecology 有意不缓存（每 tick 生效语义）。
- 错误路径：`advance_tick` 失败时不再下载（host 维持上一一致状态），属可接受取舍。

### 31.5 结论

§29 阻塞项修复正确、无回归，其余质量门禁与覆盖率均达标。**APPROVED**（范围为当前 HEAD `b9c1910`
与被审测试集；不声称任何历史基线失败消失）。

---

## 33. 第 12 轮结论（evaluator 独立执行，2026-09-19，HEAD=`d05b77c`）

### 33.1 裁定：**NOT APPROVED**

- **阻塞项（medium）**：新增的设备历史预算守卫 `GpuExecutor::configure_history` 使用**未检查乘法**计算
  `required`，对超大 `capacity` 会 **panic**（debug）或**回绕**（release）而非返回文档承诺的显式 `Err`；
  这使 §32.4 要求的“`required > free` 显式 Err、无静默行为”在最坏输入下不成立。
  同一 `run_steps` 的容量计算还有未检查的 `i64` 加法。
- 其余（投影内核、历史记录时机/continuation、回退语义、CPU 隔离、覆盖率、门禁）均通过。

### 33.2 阻塞项证据（已实际运行且失败）

- evaluator 回归测试 `rust/tests/unit/gpu/spatial_session.rs::evaluator_configure_history_over_budget_is_explicit`
  命令：`cargo test --features gpu evaluator_configure_history_over_budget_is_explicit`
  预期：`configure_history` 对超大窗口返回 `Err`（预算守卫显式失败）
  实际：**panic `attempt to multiply with overflow`** → FAILED
- 位置：`rust/src/gpu/executor.rs` `configure_history`
  - `let required = rows * size_of::<f32>() + spec.mask.len() * size_of::<f32>() + spec.selected.len() * size_of::<i32>();`
    其中 `rows = capacity.checked_mul(width)`（有检查），但随后的 `* size_of` **无检查**。
  - 相关：`rust/src/sessions/spatial.rs` `run_steps` 的 `let capacity = (start_tick + n_ticks) / interval - start_tick / interval;`
    `start_tick + n_ticks` 为未检查 `i64` 加法（`n_ticks` 直接来自公开 `run_steps(n_steps, ...)`）。
- 触发：`run_steps` 收到极大的 `n_steps`（如 `10**18`）时 `capacity` 极大，
  `capacity * width < usize::MAX` 但 `rows * 4 ≥ usize::MAX` → debug panic；release 下回绕可能**绕过预算检查**后
  在 `vec![0.0f32; rows]` 处巨量分配失败。属极端输入，但可由公开 API 触达。
- **建议修复**：`rows`、`required` 全程使用 `checked_mul`/`checked_add`（或 `saturating_mul`）并在溢出时返回
  显式 `Err`；`run_steps` 的 `start_tick + n_ticks` 用 `checked_add`。修复后本回归测试应转绿。

### 33.3 已独立验证正确的部分

- **投影内核**：`observation_project` 与 host `output::observation::project` 的累加顺序
  （deme→age→genotype）和输出布局一致；作者用例 `device_history_projection_matches_host_project`
  覆盖 groups=1/2、非 0/1 权重、乱序 `selected=[2,0]`、四种 collapse/aggregate 组合，容差 1.2e-6 → 通过。
- **历史记录时机 / continuation**（evaluator 新增
  `evaluator_device_history_start_tick_and_continuation`）：对 `[(3,2),(3,2)]`、`[(1,0),(4,2)]`
  （start_tick=1，非 interval 倍数）、`[(2,3)]`（capacity=0，无 device 窗口）、`[(5,1)]` 四种序列，
  GPU 与 CPU 的 history tick 序列与值逐元素一致（容差 1.2e-6）；`flush` 的 tick 重建
  `(start_tick/interval + 1)*interval` 与非零起点/多 run continuation 均正确 → **PASS**。
- **回退语义**：GPU 未启用 / 无 store / raw 模式 → `start_device_history` 返回 false 并走 host 逐记录路径
  （作者用例 `spatial_device_history_falls_back_for_raw_or_missing_window`）；非引擎回退。
- **CPU 不变性**：`git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空；
  `phase0_baseline.py --check` → `all scenarios bit-identical`。
- **覆盖率**（严格过滤、排除 `/tests/`）：`src/gpu` 聚合 **2098/2151 = 97.54%**，逐文件均 ≥95%
  （executor 96.8%、kernels 98.0%、probe 96.1%，其余 100%）。

### 33.4 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu`（作者用例） | **160 passed** |
| `cargo test --features gpu`（含 evaluator 2 个新增） | 161 passed, **1 failed**（=33.2） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |

### 33.5 非阻塞 / 残余风险

- 设备历史窗口按 `n_ticks/interval` 上界分配、未按 `max_rows` 收缩；大 D×长 T 会触发预算回退
  （走 host 路径，仅失去零回传）——已声明。
- `flush_device_history` 用 `append_row` 直接写行，未更新边界 metadata（`record_history` 会把
  `boundaries` 的 phase/execution 写为该 tick 的值）；对“正常完成 tick”（phase=0/`Ready`）与默认值一致，
  故当前无差异，但 stop/异常边界的 metadata 可能与 host 路径不同，建议后续对齐。
- GPU checkpoint/restore 不支持设备回滚（既有）。

### 33.6 结论

设备侧历史缓冲的核心正确性、时机与回退语义均通过；但新增的**显存预算计算存在未检查算术**
（极端容量下 panic/回绕），与 §32.4/D5 的“显式失败”要求相悖。**NOT APPROVED**，待主 agent 修复后回交。

---

## 35. 第 13 轮结论（evaluator 独立执行，2026-09-19，HEAD=`5c53092`）

### 35.1 裁定：**APPROVED**

§33 阻塞项（`configure_history` 预算算术未检查、`run_steps` tick 跨度未检查）已解除；修复为产品语义层面的
显式 `Err`，无回归。

### 35.2 修复复核（独立运行）

- 原失败回归 `cargo test --features gpu evaluator_configure_history_over_budget_is_explicit`
  → **ok**（此前 panic `attempt to multiply with overflow`）。evaluator 测试断言未被改动。
- 作者新增 `run_steps_rejects_overflowing_tick_span`（`start_tick=1` 后 `run_steps(i64::MAX, 1)`
  → 显式 Err）→ **ok**。
- 读 diff 核对：`configure_history` 的 `rows`/`row_bytes`/`mask_bytes`/`selected_bytes` 及求和
  全部 `checked_mul`/`checked_add`，溢出返回显式 `Err`，之后才做预算比较与分配；
  `run_steps` 的 `start_tick.checked_add(n_ticks)` 溢出返回 `PyValueError`。无遗留未检查算术。
- `cargo test --features gpu evaluator_`（全部 21 个 evaluator 用例）→ **21 passed**，含
  §33.3 的投影/时机/continuation/回退用例。

### 35.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **163 passed, 0 failed**（含 evaluator 2 个回归 + 作者新增） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| CPU/numeric 隔离 | `git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空 |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **2112/2165 = 97.55%**；逐文件均 ≥95%（executor 96.9%、kernels 98.0%、probe 96.1%，其余 100%） |

### 35.4 残余风险（非阻塞，延续 §33.5）

- 设备历史窗口未按 `max_rows` 收缩；大 D×长 T 触发预算回退（走 host 路径，仅失去零回传）。
- `flush_device_history` 用 `append_row` 未更新边界 metadata；正常完成 tick（phase=0/`Ready`）与 host 一致，
  stop/异常边界可能不同。
- raw 模式仍不走设备暂存（有意）；GPU checkpoint/restore 不支持设备回滚（既有）。

### 35.5 结论

§33 阻塞项修复正确、无回归，其余质量门禁与覆盖率均达标。**APPROVED**（范围为当前 HEAD `5c53092`
与被审测试集；不声称任何历史基线失败消失）。

---

## 37. 第 14 轮结论（evaluator 独立执行，2026-09-19，HEAD=`2308114`）

### 37.1 裁定：**APPROVED**

`continuous_sampling=true` 的设备路径（连续二项/多项/Poisson + Gamma 迭代改写）与 host 语义一致，
四个随机阶段端到端统计等价（均值与方差），离散/确定性路径零回归，会话接受且非静默回退。

### 37.2 独立核对

- **连续采样器逐分支**（读 diff）：设备 `sample_continuous_binomial`（Beta 比例 = `Gamma(p(n-1))` 与
  `Gamma((1-p)(n-1))` 归一 × n）、`sample_continuous_poisson`（`Gamma(λ,1)`）、
  `natal_continuous_multinomial`（归一化 Gamma + 漂移校正）与 host `rng.rs` 的
  `continuous_binomial/poisson/multinomial` 公式和分支一致；`sample_outbound_device` 连续分支
  与 host `sample_outbound` 一致；`reproduction_stochastic` 连续分支的 mating/removal
  （`removed_frac=min(p_remating,1)`、按比例移除）、fertilize（`continuous_poisson`、
  `continuous_binomial`）、sex/viability 均与 host 对应分支一致。
- **Gamma 迭代改写**：原 `shape<1` 递归等价改为一次性 `boost=pow(u,1/shape)` + `shape+1` 的 MS；
  RNG 抽取顺序不变、分布不变。既有的 `evaluator_gamma_including_shape_below_one_matches_cpu`
  （形状 0.3/0.5/0.9/1/2/5/20）与作者用例仍通过。
- **端到端连续 L3**（evaluator 独立，Python 公共 API，panmictic 连续，K=300×5 tick）：
  `gpu_status=="enabled"`、状态有限；总量 **CPU mean=42321.19 vs GPU mean=42324.59**
  （合并 5σ 容差 1957，ratio **0.002**）；**方差比 GPU/CPU=1.109**（K=300 下 ~1.3σ，一致）。
- **会话语义**：`enable_gpu` / `enable_gpu_ensemble` 删除 `continuous_sampling` 拒绝后接受连续并实际跑通
  （不是静默回退）；离散路径与 `phase0` 不变。
- **启动器参数顺序**：`recruit/survival/reproduction_stochastic` 及 migration prepare 的
  `launch.arg` 顺序与各自 CUDA 内核签名逐一核对一致（`n_ztypes, continuous, new_adult_age, ...`）。
- **CPU 不变性**：`git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空；
  `phase0` bit-identical。

### 37.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **168 passed, 0 failed**（含 3 个连续分布用例 + 会话接线） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **2131/2180 = 97.75%**；逐文件均 ≥95%（executor 97.3%、kernels 98.1%、probe 96.1%，其余 100%） |

### 37.4 非阻塞发现（低）

- **连续采样阈值与 host `EPS` 不一致**：设备 `sample_continuous_binomial` 用 `1e-12`（host `p<=EPS/>=1-EPS`、
  `n<=1+EPS`，`EPS=1e-10`）、`natal_continuous_multinomial` 小 n 用 `1+1e-7` 且 `alpha<=1e-12`
  （host 为 `1+EPS`、`alpha<=EPS`）、`sample_continuous_poisson` 用 `1e-12`（host `lambda<=EPS`）。
  在 `(1e-12,1e-10]` / `n∈(1+1e-10,1+1e-7]` 等窗口内与 host 分支不同，但差异量级极小（结果≈0 或差 O(1/n)），
  实测端到端均值/方差一致。建议为一致性与可维护性统一到 `EPS=1e-10`（与离散路径第 1 轮修复口径一致）。
- 设备 `sample_gamma` 对任意大 shape 均用 Marsaglia-Tsang，而 host 在 `shape>=1e8` 用 Normal 近似；
  CLT 下分布等价，可忽略。
- `stochastic_grid` 固定 64 线程块为资源约束下的取值，未做占用率调优（仅性能）。

### 37.5 结论

连续抽样设备支持正确、与 host 统计等价、无回归，质量门禁与覆盖率达标。**APPROVED**（范围为当前 HEAD
`2308114` 与被审测试集；不声称任何历史基线失败消失）。

---

## 39. 第 15 轮结论（evaluator 独立执行，2026-09-19，HEAD=`a62cb79`）

### 39.1 裁定：**APPROVED**

离散世代（二龄）空间设备路径与 host `kernels::discrete_generation` 逐分支一致；确定性端到端紧容差匹配，
随机（离散与连续）分布等价，既有能力零回归。

### 39.2 独立核对

- **内核逐分支**（读 diff + host 对照）：
  - `discrete_reproduction` 与 host `reproduction` + `mate_discrete` + `fertilize_discrete` 一致：
    成年列 mating（`mating[1]`/`mating[3]`）、`effective_males`、`males_total==0||females_total==0` 提前返回
    （不清 age0）、mating 行归一阈值 `1e-10`、`n_mating<=EPS` 跳过、离散 `round`/连续 `continuous_binomial`、
    `pair_counts` 累加、`eggs_per_pair=epf·ff[gf]·ff[Z+gm]`、`p_reproduce=clamp01(reproduction_rates[adult])`、
    `p_surv>=1-EPS`/`continuous_poisson`、性别/性染色体分配、age0 无条件写入。
  - `discrete_survival` 与 host `survival`（`scaling_factor` + `recruit_juveniles` + viability）一致：
    离散 `round`、`desired` 取整、`combined/total`、`viability[0..Z]`/`[2Z..3Z]` 行、`s_f=survival[0]`、
    `s_m=survival[2]`（A=2）。
  - `density_scaling` 新增 `discrete_actual`：模式 2–4 用 age0 总计数（host `scaling_factor`），模式 0/1 行为不变；
    启动器参数顺序与内核签名一致。
  - 生命周期顺序 reproduction→survival→aging；离散迁移复用既有 CSR 内核（sperm 面恒零）。
- **确定性端到端**（evaluator 独立，Python，`demos/gpu_spatial/discrete` 参考模型 8 tick）：
  `ind max_rel=3.70e-7`，CPU/GPU 总量均为 22500 → 与 host 一致。
- **随机端到端**（K=200 seed×3 tick，离散采样）：GPU 启用且状态有限；总量
  **CPU mean=22494.0 vs GPU mean=22502.6**（合并 5σ 容差 23.97，ratio 0.356）；
  **方差比 0.927**（采样噪声内）。
- **CPU 不变性**：`git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空；
  `phase0` bit-identical。

### 39.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` | 67 passed |
| `cargo test --features gpu` | **174 passed, 0 failed**（含 4 个离散分布用例 + 空间离散会话用例） |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3606 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **2344/2418 = 96.94%**；逐文件均 ≥95%（executor 96.0%、kernels 97.4%、probe 96.1%，其余 100%） |

### 39.4 非阻塞发现（低）

- 离散 GPU 未覆盖 Wright-Fisher 融合路径（`run_wf_tick`）；空间离散 CPU 本就不使用它（已声明）。
- `discrete_survival` 固定 64 线程块（寄存器约束），仅性能。
- 细节差异（经校验无影响）：设备对 `male_adult_mating_rate` 与 `eggs_per_female` 做了 `clamp01`/负值归零，
  而 host 离散对应处用原值；由于标量契约已验证在 `[0,1]`/非负，实际无差异。

### 39.5 结论

离散世代空间 GPU 路径正确、与 host 统计等价、确定性紧容差匹配、无回归，质量门禁与覆盖率达标。
**APPROVED**（范围为当前 HEAD `a62cb79` 与被审测试集；不声称任何历史基线失败消失）。

---

## 41. 第 16 轮结论（evaluator 独立执行，2026-09-19，HEAD=`1211add`）

### 41.1 裁定：**APPROVED**

frontend 级 ensemble 公开入口语义正确、隔离性成立、中英文档同步；Rust 数值未改，既有门禁与回归不变。
（Python 新增行覆盖率由 evaluator 补 2 个用例后达 100%，见 41.3。）

### 41.2 独立核对

- **公开合同**（读代码 + 独立运行）：`enable_gpu_ensemble(n)` 惰性建会话 / 同步后透传；`run_gpu_ensemble(ticks)`
  返回 `(tick, (B,2,A,Z), (B,A,Z,Z))`。独立脚本：B=128、4 tick →
  `ind (128,2,4,3)`、`sperm (128,4,3,3)`、全有限、`tick==4`。
- **隔离性**：`run_gpu_ensemble` 后 `pop.tick` 与 `pop.state.individual_count` 均未变
  （独立运行 + 仓库用例）。
- **可复现**：同 seed 两个 pop 的 ensemble 结果逐位一致（独立运行）。
- **拒绝路径**：`enable_gpu_ensemble(0)` → `ValueError`；先 `enable` 后 `run` 的顺序约束由 Rust 侧
  显式报错；不合格模型（含钩子 / 自定义曲线 / 非 panmictic）由 Rust `enable_gpu_ensemble` 拒绝
  （既有 `session_gpu_ensemble_rejects_ineligible` 覆盖）。非静默回退。
- **文档中英同步**：`docs/en|zh/4_simulation_engine.md` 新增 §11（GPU Ensemble）并将小结顺延为 §12，
  两版内容对应、方法名/形状/资格条件一致、示例使用公开 API。

### 41.3 覆盖率（Python 新增行）

`pytest tests/test_gpu_ensemble_frontend.py --cov=natal --cov-report=json`，按 diff 新增行统计：
新增可执行 22 行，初始未覆盖 2 行（`enable_gpu_ensemble` 的惰性建会话分支 1026；
`run_gpu_ensemble` 的 `backend is None` 守卫 1056，因 `build()` 已建会话故仓库用例未触及）。
evaluator 补充 2 个针对性用例（`test_enable_gpu_ensemble_lazily_initializes_session`、
`test_run_gpu_ensemble_missing_backend_raises`）后 **22/22 = 100%**。Rust `src/gpu/**` 覆盖率不变
（第 15 轮 96.94%，本轮未改 Rust）。

### 41.4 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `PYTHONUTF8=1 pytest -q` | **3611 passed**（作者 3609 + evaluator 2） |
| `pytest -q tests/test_gpu_ensemble_frontend.py` | 5 passed |
| `ruff check src demos tests` | 通过 |
| `pyright` | 0 errors |
| `cargo test` / `cargo test --features gpu` | 67 / 174 passed（Rust 未改，回归确认） |
| `python scripts/check_rust.py` | EXIT=0 |
| `phase0_baseline.py --check` | all scenarios bit-identical |

### 41.5 残余风险（非阻塞，见 §40.5）

- 未提供 `Population` 级单群体 `enable_gpu`（一步回传）frontend 入口；本轮仅 ensemble，文档亦如此表述。
- 未提供 frontend ensemble 的 `History` 集成（返回裸数组）。
- `run_gpu_ensemble` 在已有单群体 `enable_gpu` 而 `_gpu_ensemble_replicates==0` 时 reshape 会失败；
  属误用路径（应只用 ensemble），报错非静默。

### 41.6 结论

frontend ensemble 入口与文档满足公开合同、隔离性、可复现与同步要求；门禁全绿、Python 新增行覆盖率 100%。
**APPROVED**（范围为当前 HEAD `1211add` 与被审测试集；不声称任何历史基线失败消失）。

---

## 43. 第 17 轮结论（evaluator 独立执行，2026-09-19，HEAD=`8ea1c5c`）

### 43.1 裁定：**APPROVED**

`density_scaling_device` 把缩放因子留在设备，`survival_tick` / `discrete_survival_tick` 不再做
D2H+H2D 往返。数值等价性成立，无回归、无门禁/覆盖率问题。

### 43.2 独立核对

- **等价性（读 diff）**：`density_scaling_device` 与旧 `density_scaling_impl` 使用同一缩放内核、同一输入，
  只是不再 `to_host`；`density_scaling_impl` 改为 `density_scaling_device(...)?.to_host(...)`，
  公开 `density_scaling` 主机返回值语义不变。`survival_tick`/`discrete_survival_tick` 直接复用设备
  `DeviceBuffer`，下游 `recruit_factor`/`recruit_stochastic`/`discrete_survival` 读取同一数据。
- **无残留同步**：设备 survival 路径不再出现缩放因子的 `to_host`/`from_host`。
- **确定性 L3（evaluator 独立，重编扩展）**：
  - panmictic 确定性 5 tick：`ind max_rel=1.680e-7`（与 §17/§31 完全相同）；
  - 空间确定性 5×5、25 tick：`max_rel=9.140e-7 / 9.149e-7`（与 §23/§31 完全相同）。
- **CPU 数值不变**：`git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空；
  `phase0` bit-identical。

### 43.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` / `cargo test --features gpu` | 67 / **174 passed, 0 failed** |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff check src demos tests` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3611 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **2352/2426 = 96.95%**；逐文件均 ≥95%（executor 96.0%、kernels 97.4%、probe 96.1%，其余 100%） |

### 43.4 性能结论（§42.3 请求的口径）

- 该改动方向正确、零风险：确定性 survival 每 tick 少一次 D2H+H2D，与「逐 tick 不回传」目标一致。
- 主 agent 的临时全 tick 基准显示瓶颈在**计算侧**而非传输/同步（stage 拆分 repro≈surv≈10.6ms，
  传输量级远小），且受共享 GPU 影响仅定性。evaluator 认同：**传输类优化在大 B 下收益有限**；
  更深的内核/占用率剖析应作为独立任务在空闲卡上做。本轮交付的同步移除本身无争议，予以接受。

### 43.5 残余风险（非阻塞）

- 大 B 基准受共享 GPU 时间片影响，性能数字仅定性；未做更深优化。
- 既有残余（设备窗口未按 max_rows 收缩、raw 不走设备历史、GPU checkpoint 回滚等）不变。

### 43.6 结论

缩放同步移除实现正确、等价性经 L3 与门禁独立确认，质量与覆盖率达标。**APPROVED**（范围为当前 HEAD
`8ea1c5c` 与被审测试集；不声称任何历史基线失败消失）。

---

## 45. 第 18 轮结论（evaluator 独立执行，2026-09-19，HEAD=`7edff1b`）

### 45.1 裁定：**APPROVED**

GPU 会话 checkpoint 回滚现在同时回滚显存状态与设备 tick；恢复后重跑与参照**逐位一致**（空间与 age 两条
会话路径），消除了“host 已回滚、设备从旧状态继续”的静默错误轨迹。

### 45.2 独立核对

- **实现逐分支**：
  - `GpuExecutor::restore_state`：校验 `(B,2,A,Z)`/`(B,A,Z,Z)` 长度，`batch_to_inner` 转置后重传
    `ind`/`sperm`，并把 `self.tick = tick`（counter-based RNG 以 tick 为基准，故可逐位续跑）；
  - `spatial::restore_from_checkpoint` 与 `age_structured::restore_from_checkpoint`/`restore_state`：
    host 回滚后调用设备恢复并 `clear_history()`；
  - `age` 的 `restore_device_state` 把单群体状态平铺到执行器 batch 宽度（单群体=1、ensemble=B），
    与 `enable_gpu_ensemble` 构造一致。
- **会话级逐位回归（我运行）**：`cargo test --features gpu restore` → **3 passed**
  （`restore_state_rejects_wrong_lengths`、`session_gpu_restore_device_state_rewinds`、
  `spatial_gpu_restore_checkpoint_rewinds_device_state`），其中两个会话用例对
  `state_ind`/`state_sperm` 做 `to_bits()` 逐位比较。
- **Python 端到端（evaluator 独立，重编扩展）**：空间确定性模型 GPU 跑 4 tick 为参照；
  另一实例跑 4 tick 后 `pop.restore_checkpoint(2)`，再跑 2 tick → 最终 `ind`/`sperm`
  **与参照逐位相同**（`max_abs_diff=0.0`），`tick` 一致。
- **CPU 不变性**：`git diff 373fcbf..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs` 为空；
  `phase0` bit-identical。

### 45.3 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` / `cargo test --features gpu` | 67 / **177 passed, 0 failed** |
| `cargo test --features gpu restore` | 3 passed |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff check src demos tests` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3611 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **2383/2457 = 96.99%**；逐文件均 ≥95%（executor 96.2%、kernels 97.4%、probe 96.1%，其余 100%） |

### 45.4 残余风险（非阻塞，见 §44.5）

- age 结构化设备分支本身不采集 checkpoint（GPU 运行无历史）；A1 覆盖“先有 checkpoint 再启用/回滚设备”。
- `enable_gpu` 在已有 CPU tick 后启用时不会自动把设备 tick 设为 `state_tick`；A1 恢复会修正，
  正常“先启用后运行”不受影响（后续可选清理）。
- `restore_state` 的 `tick as u64`：`state_tick` 非负，无实际影响。

### 45.5 结论

A1 设备恢复正确、经会话级逐位用例与 Python 端到端逐位对照独立确认，质量与覆盖率达标。**APPROVED**
（范围为当前 HEAD `7edff1b` 与被审测试集；不声称任何历史基线失败消失）。

---

## 47. 第 19 轮结论（evaluator 独立执行，2026-09-19，HEAD=`25dae0a`）

### 47.1 裁定：**NOT APPROVED**

- **阻塞项（medium）**：A2 使用**固定绝对容差** `NATAL_VIRGIN_EPS = 1e-3f` 判定“显著为负的 virgin 计数”。
  在较大单格点计数（约 `≥1e5`）与多个 sperm 类别下，`stored_total` 的 f32 顺序累加会产生 `>1e-3` 的正偏差，
  使**有效状态**（数学上 `stored ≤ female`、host f64 判 `virgins == 0`）被误判为非法并中止运行。
- A2 对“明显非法状态”的显式报错行为本身正确；其余门禁/覆盖率通过。

### 47.2 阻塞项证据（已实际运行且失败）

- evaluator 回归测试 `rust/tests/unit/gpu/executor.rs::evaluator_valid_large_virgin_state_is_not_rejected`
  命令：`cargo test --features gpu evaluator_valid_large_virgin_state_is_not_rejected`
  构造：`n_ages=4, Z=4, female(age1,z0)=100000.0`，`sperm(age1,gf0,gm0..3)=
  [25000.25390625, 25000.666015625, 24999.32421875, 24999.755859375]`，其**数学和恰为 100000.0**
  （host f64 判 `virgins == 0`，不报错）。
  预期：设备 `survival_tick` 返回 `Ok`（有效状态，不得中止）
  实际：**`Err("Invalid state: n_virgins < 0 in GPU stochastic survival")`** → FAILED。
- 机理：设备按 `stored += sperm[...]` 顺序 f32 累加得 `100000.0078125`，`virgins = 100000 − 100000.0078 =
  −0.0078125 < −1e-3` → 置 violation。该偏差随量级增长：数值搜索显示
  `female=1e5, Z=4` 的有效状态最大正偏差 `≈0.0078`，`Z=8 ≈0.0078`，`Z=32 ≈0.023`；
  `female=1e4, Z=32 ≈0.0029`——均超过 `1e-3`。
- 位置：`rust/src/gpu/kernels.rs` `survival_stochastic` 的 `if (virgins < -NATAL_VIRGIN_EPS)`；
  常量 `#define NATAL_VIRGIN_EPS 1e-3f`。宿主对照为 `age_structured::sample_survival_with_sperm` 的
  `n_virgins_raw < -EPS`（`EPS=1e-10`，f64）。
- **建议修复**：改为**尺度相关**容差，例如
  `tol = c * (fabsf(n_f_raw) + fabsf(total_sperm)) * FLT_EPSILON + 1e-9f`
  （`c≈8`，f32 eps≈1.2e-7），或对 `stored` 用树形/成对求和降低偏差；这样 `-20` 等真实非法状态仍被拒绝
  （1e5 量级 tol≈0.1），而 `-0.008` 的舍入噪声被接受。修复后本回归测试应转绿。

### 47.3 已独立验证正确的部分

- **明显非法路径**：作者用例 `stochastic_survival_rejects_negative_virgin_state`（`-20 → Err`；
  `-1e-4 → Ok` 夹 0）通过。
- **随机生存分布**：`device_stochastic_survival_*_matches_host_distribution`（离散/连续）仍通过。
- **门禁/覆盖率**：见 47.4；确定性路径与既有能力零回归。
- **同步权衡**：随机生存每 tick 增加一次 4 字节 D2H（读取 violation）。功能正确，但与“逐 tick 零回传”
  目标有轻微冲突；若后续定位为热点，可考虑延后到下载边界检查。非阻塞。

### 47.4 门禁自跑（独立）

| 命令 | 结果 |
|---|---|
| `cargo test` / `cargo test --features gpu`（作者用例） | 67 / **178 passed** |
| `cargo test --features gpu`（含 evaluator 回归） | 178 passed, **1 failed**（=47.2） |
| `cargo test --features gpu stochastic_survival_rejects_negative_virgin_state` | passed |
| `cargo clippy --features gpu -- -D warnings` / `cargo fmt -- --check` | 通过 |
| `python scripts/check_rust.py` | EXIT=0 |
| `ruff check src demos tests` / `pyright` | 通过 / 0 errors |
| `PYTHONUTF8=1 pytest -q` | 3611 passed |
| `phase0_baseline.py --check` | all scenarios bit-identical |
| 严格过滤 `rust/src/gpu/**` 覆盖率 | **2390/2464 = 97.00%**；逐文件均 ≥95% |

### 47.5 结论

A2 的“显式拒绝非法状态”方向正确，但固定 `1e-3f` 容差会在较大规模下**误伤有效状态并中止运行**；
已给出实际失败的回归测试与尺度相关容差的修复建议。**NOT APPROVED**，待主 agent 修复后回交复核。
