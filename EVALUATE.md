# EVALUATE.md — 独立 evaluator 审查参考（GPU 旁路 P0–P3）

> 本文件供**独立 evaluator agent**使用，不是给主开发 agent 的任务书。
> evaluator 的职责定义见 `AGENTS.md`「验证时机与角色」与 `quality_checks_spec.md`。
> 同一主题冲突时以 `AGENTS.md` 和英文规范为准。

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

环境变量与重建扩展的注意事项见 EVALUATE.md §3（NATAL_GPU_REQUIRE 默认强制；
maturin 需要 --features "gpu,extension-module"；需要 LD_LIBRARY_PATH 指向 conda 的 libnvrtc）。
若需要数值测试方法，加载 numerical 技能；对抗式多 agent 流程可加载 adversarial-review 技能。
```
