# GPU 旁路阶段性总结（分支 `feat/gpu-merge-test`）

> 供新接手的 agent 快速建立全局认识。**约束以 `AGENTS.md` 为准**，任务背景以 `HANDOFF.md` 与
> `GPU_insert_PLAN.md` 为准；本文档是「当前实现现状 + GPU/CPU 异同 + 下一步」的快照。
> 审查沟通统一走 `EVALUATE.md`（双方只追加、不改写历史）。

---

## 0. 五分钟快速通道

1. 读 `AGENTS.md`（授权/风险/验证）、`HANDOFF.md`（任务与禁忌）、`GPU_insert_PLAN.md`（D1–D7 冻结决定、精度分档）。
2. 读本文档 §3–§7，掌握提交时间线、源码架构、**CPU/GPU 实现异同**。
3. 复现门禁（§10）。
4. 看 `EVALUATE.md` 顶部「当前状态」表，了解最新审查轮次与待办。

**第一原则**：CPU 路径是唯一 golden reference，**一行都不能改**；GPU 只做可选旁路，默认关闭。

---

## 1. 项目一句话

`natal-core` 是前向时间群体遗传学模拟引擎（基因驱动建模），**Python 前端 + Rust 原生内核**，
中间以冻结/可变的 `contracts/` 边界分隔。CPU（rayon）是唯一执行引擎；本分支为它并联一条**可选
CUDA 旁路**（`cudarc`，f32，counter-based RNG）。

四层：

```
src/natal/__init__.py    ① 惰性导出（导入时不加载子模块）
src/natal/frontend/**    ② Python 领域层（16 子包）
src/natal/contracts/**   ③ 边界契约：Blueprint(NamedTuple,冻结) / Params(dataclass,可变) / materialize()
rust/**                  ④ 原生引擎（单 crate `natal-engine-core` → `_engine_rs`）
```

---

## 2. 目标与冻结决定（`GPU_insert_PLAN.md` §2）

- 目标：① 空间多 deme 的大 `D`；② 非空间多 replicate 的大 `B`；状态常驻显存、逐 tick 不回传（理想）；
  GPU 可选、CPU 完整保留。
- 非目标：不追求 CPU↔GPU 逐位一致；不替换 CPU；不改 Python 公开 API；不引入 torch；暂不做 `D×B`。
- D1–D7：CUDA via `cudarc`（动态加载 + NVRTC）；f32 + 树形归约；counter-based RNG（Philox，接受跨平台不可逐位）；
  batch 轴 = D/B，Z 非瓶颈；显存不足**显式报错**不回退；停止条件设备侧门控；`py.allow_threads` 释放 GIL。
- 精度分档（§5.2）：纯搬运 + 整数(<2²⁴) → 逐位；含算术确定性 → 相对误差 < ~1.2e-6；随机 → 统计等价。

---

## 3. 提交时间线（分支起点 → 现在）

| 阶段 | 提交 | 内容 |
|---|---|---|
| P0-a | `9b4d5b2` | 零依赖 CUDA 环境探针（`nvidia-smi` + toolkit 目录扫描） |
| docs | `adda656` | HANDOFF |
| P0-b | `97295ae` | `cudarc` 动态加载 + NVRTC 实编译，`send` 检查；门禁默认开启 |
| P1 | `ed2c295` | `context`/`buffers`/`layout`(B,d…↔d…,B) + 覆盖率/门禁 |
| P2 | `b4cd4ea` | executor + aging kernel + §D5 预算守卫；含 P3 density/equilibrium 首块 |
| P3 | `be3a467` `e5438b7` `a821154` `81c3a14` | survival → reproduction → 全 tick 编排(L2) → 会话接线+L3 |
| 审查基建 | `373fcbf` `52544cf` `34f8cc2` | `EVALUATE.md` 双向通道；覆盖补齐 |
| P5 | `c4c908c` `ec1cf99` | 确定性 CSR 迁移（gather，无原子）+ 空间接线 + 空间 L3 |
| P4 | `1aedee6` `255d707` `e6d51b1` `7e631ed` `243b35d` `d397f91` | Philox → 采样器 → 随机生存/繁殖 → 随机会话/L3 → PTRS 精确 Poisson |
| 空间随机 | `5b1ced6` `ec27d56` | 两趟无原子随机迁移 + 行宽守卫 |
| P6 | `672ee4e` `5bddb6e` | ensemble 核心 + 会话 API/透传 |
| 最新文档 | `8b84eeb` `d6f24a2` | 各轮交接/回执 |

独立审查（`EVALUATE.md`）：P0–P3 第 3 轮 APPROVED；P5 第 4 轮 APPROVED；P4 第 6 轮 APPROVED（PTRS 修复后）；
空间随机第 7 轮 APPROVED；第 8 轮（越界守卫，§25）APPROVED；第 9 轮（P6，§27）APPROVED；
第 10 轮（CSR 缓存 + 零逐 tick 回传，§28）**NOT APPROVED**（§29：公共单 tick 读路径陈旧），
已按 §30 修复（公共 `run_tick` 同步、`run_steps` 调 `advance_tick`），**待复核（§31）**。

---

## 4. 当前源代码架构

### 4.1 Python 侧

```
src/natal/
  __init__.py                 惰性导出索引
  frontend/                   领域层：builder/ population/ spatial/ genetics/ fitness/ hooks/
                              output/ presets/ modifiers/ model/ patterns/ registry/ data/ utils/ ui/ webui/
  contracts/
    blueprint.py              冻结：维度、执行标志、ztype 名称、性染色体掩码、初始状态、CSR 迁移
    params.py                 可变：生态 5 标量 + 人口学/遗传张量
    materialize.py            唯一诞生地
    __init__.py               CONTRACTS_VERSION = 2（改动需同步 Rust 镜像）
  backends/rust/rust_backend.py
    RustLifecycleBackend              年龄结构（非空间）会话适配
    RustDiscreteLifecycleBackend      离散世代
    RustHeterogeneousSpatialLifecycleBackend  空间（异构/同构）
    + enable_gpu()/gpu_status()/enable_gpu_ensemble()/run_gpu_ensemble() 透传
```

### 4.2 Rust 侧（crate `natal-engine-core` → `_engine_rs`，`cdylib`）

```
rust/src/
  lib.rs               模块声明 + PyModule（gpu feature 下加 EnsembleEngineSession 已回退，见 §8）
  generated/           由 Python 生成的生态参数镜像
  model/               blueprint.rs / ecology.rs / genetics.rs / validation.rs / custom_fields.rs
  kernels/             age_structured.rs / discrete_generation.rs / spatial.rs /
                       density_regulation.rs / equilibrium.rs / offspring.rs / rng.rs / state_reduce.rs
  hooks/               interpreter.rs（CSR 钩子）/ transaction.rs
  output/              history.rs / observation.rs / parameter_log.rs
  sessions/            age_structured.rs(EngineSession) / discrete_generation.rs /
                       spatial.rs(HeterogeneousSpatialEngineSession) / status.rs / ecology_snapshot.rs
  python.rs            PyO3 辅助函数
  gpu/                 ★ 本分支新增，`#[cfg(feature="gpu")]`
    mod.rs             模块导出 + `hardware_required()`（`NATAL_GPU_REQUIRE`，默认强制）
    probe.rs           零依赖 `CudaProbe`（nvidia-smi/目录扫描）
    cuda.rs            `cudarc` 绑定级探针（driver 加载、NVRTC 编译运行）
    context.rs         `GpuContext`（CudaContext + 默认 stream）
    buffers.rs         `DeviceBuffer<T>`（显存 RAII）
    layout.rs          `batch_to_inner/outer`（B,d… ↔ d…,B）
    kernels.rs         NVRTC 源 + 启动器（aging/density/survival/reproduction/migration/RNG/sampling）
    executor.rs        `GpuExecutor`：上传/预算守卫/各阶段/tick/migrate/ensemble
rust/tests/unit/gpu/   probe/cuda/context/buffers/layout/kernels/executor/session/spatial_session.rs
```

`Cargo.toml`：`cudarc = { optional = true, features=["cuda-13020"] }`；`[features] gpu = ["dep:cudarc"]`（**默认关闭**）。

### 4.3 设备内核清单（`gpu/kernels.rs` 内的 CUDA 源）

| 源 | kernel | 作用 |
|---|---|---|
| `AGING_SOURCE` | `age_shift` | 年龄下移一位、age0 清 0（out-of-place，避免竞争） |
| `DENSITY_SOURCE` | `density_scaling` | 平衡态指标 + 生长曲线（none/fixed/linear/beverton_holt/ricker），每 batch 一个标量 |
| `SURVIVAL_SOURCE` | `recruit_factor` / `survival_scale_ind` / `survival_scale_sperm` | 确定性密度缩放+存活 |
| `REPRODUCTION_SOURCE` | `reproduction` | 确定性求偶矩阵+储精置换+受精+合子存活 |
| `MIGRATION_SOURCE` | `migration_female/sperm/male` | 确定性 CSR **gather**（反 CSR，无原子） |
| `RNG_SOURCE` | `philox4x32_10` / `fill_uniform` | counter-based RNG 基础 |
| `SAMPLING_SOURCE` | `RngState`/`rng_uniform|normal`/`sample_binomial|poisson|gamma`/`natal_multinomial`/`multinomial_seq`/`recruit_stochastic`/`survival_stochastic`/`reproduction_stochastic`/`migration_stochastic_prepare`+`gather_*` | 随机路径 |
| （`cuda.rs`） | `add_one` | P0-b 工具链自证 |

---

## 5. ★ CPU 与 GPU 核心模拟过程是否相同？

**结论：语义相同（同阶段、同公式、同分支顺序），数值不同（f64 vs f32、不同 RNG），因此
确定性含算术不逐位、随机模式只做统计等价。** 逐阶段对照：

| 阶段 | CPU | GPU | 是否逐位 |
|---|---|---|---|
| 布局 | `(B,2,A,Z)` 连续块 | `(2,A,Z,B)` batch 最内层（仅上传/下载转置） | 转置是纯排列 → round-trip 逐位 |
| 衰老 aging | `kernels/age_structured.rs` 逐 age 下移、清 age0 | `age_shift`；按 batch 独立 | 整数(<2²⁴) → **逐位**；L1 已验 |
| 密度/平衡态 | `equilibrium.rs` + `density_regulation.rs` | `density_scaling` 复刻同样公式 | f32 相对误差 ~1e-6；L1 已验 |
| 存活（确定性） | `survival()` 密度缩放+幼体重标定+乘存活率 | `recruit_factor`+`survival_scale_*` | 相对误差；L1 已验 |
| 繁殖（确定性） | `reproduction()` 求偶矩阵、储精置换、受精、合子存活 | `reproduction`（单线程/batch） | 相对误差；L1 已验 |
| 迁移（确定性） | `migrate_csr_deterministic`：按 src **scatter**（顺序累加） | 反 CSR **gather**（固定顺序，无原子） | 同一算术、不同累加顺序 → 相对误差；L1 已验 |
| tick 编排 | `run_tick`：hook→reproduction→survival→aging | `tick`：reproduction→survival→aging（无钩子） | 见上；L2/L3 已验 |
| 空间 | 生命周期（按 deme 调度）+ 迁移 | `tick`（批量 deme）+ `migrate_tick`，同顺序 | 相对误差；空间 L3 已验 |
| 衰老（随机无关） | 同上 | 同上 | 同上 |
| RNG | **Xoshiro256++** 每实体持久流（`kernels/rng.rs`） | **Philox4x32-10** counter-based：`counter=(cell<<32)|draw`，`site=tick·K+stage` | **不逐位**；同 seed GPU↔GPU 逐位可复现 |
| 采样器 | `rand_distr`：Binomial/Poisson/Gamma（精确）+ 顺序条件二项 multinomial | binomial：mean≤512 精确几何跳跃，否则正态近似（p>0.5 反射）；Poisson：λ<10 Knuth、λ≥10 **PTRS 精确**；gamma：Marsaglia-Tsang；multinomial：**序列轴外提** | **统计等价**（KS/卡方/矩）；不逐位 |
| 随机阶段 | `sample_survival_with_sperm`/`recruit_juveniles`/`sample_mating`/`fertilize`/`migrate_csr_stochastic_rngs` | `survival_stochastic`/`recruit_stochastic`/`reproduction_stochastic`/两趟 `migration_stochastic_*` | 统计等价；确定性分支未受影响 |

**精度验收口径**（务必遵守）：
- 纯搬运 + 整数 → 逐位；含算术确定性 → 相对误差 ~1.2e-6（整 tick 实测 ~1e-5 量级亦通过）；
- 随机 → **禁止逐位**，用统计检验（本轮曾因 Poisson 正态近似在 λ≥64 不满足卡方而被 evaluator 判阻塞，后改 PTRS 精确）。

**CPU 不可变证据**：`git diff <GPU-branch-base>..HEAD -- rust/src/kernels rust/src/model src/natal/contracts rust/src/lib.rs`
为空；`scripts/phase0_baseline.py --check` 始终 `all scenarios bit-identical`。

---

## 6. 设备数据布局与 RNG 契约（实现要点）

- `GpuExecutor`：状态常驻 `(2,A,Z,B)` / `(A,Z,Z,B)`；`set_seed`；`rng_site(stage)=tick*16+stage`。
- 迁移：确定性用反 CSR gather；随机用**两趟**（pass1 每 source 抽样+多项分配写 entry 级 `fwd_*` 并写自身 stay；
  pass2 按反 CSR 汇总），**不使用 `atomicAdd`** → GPU↔GPU 可复现。CSR 行宽上限 `MAX_CSR_ROW=32`（随机路径）。
- 显存守卫：`ensure_memory_budget(free, required)`，不足即报错，绝不静默回退 CPU。
- ensemble：`GpuExecutor::ensemble(...)` 把单群体平铺到 batch 轴，各 replicate 流不相交。

---

## 7. 会话接线（Python → Rust）

- 年龄结构（panmictic）：`AgeStructuredSession`（`EngineSession`）新增 `#[cfg(gpu)] gpu` 字段、
  `enable_gpu()`/`gpu_status()`（现接受 `stochastic=true`，仅拒绝 `continuous_sampling=true`）、
  `enable_gpu_ensemble()`/`run_gpu_ensemble()`；`run_inner` 开头**提前返回**设备分支。
- 空间：`SpatialSession`（`HeterogeneousSpatialEngineSession`）新增 `gpu` 字段、`enable_gpu()`/
  `gpu_status()`（接受确定性 + 随机，拒绝离散与 continuous）、`run_gpu_tick`；`run_inner` 提前返回；
  确定性用 `migrate_tick`、随机用 `migrate_tick_stochastic`。
- Python：`rust_backend.py` 三个后端类分别透传；扩展未带 gpu 时给出可操作 `RuntimeError`。

---

## 8. 已知限制 / 未做项

- **`continuous_sampling=true`**：设备显式拒绝（未实现连续抽样）。
- **hooks/停止门控**：含钩子模型设备拒绝；D6 设备侧门控未做。
- **离散世代 GPU**：空间离散被拒；仅年龄结构。
- **设备侧 history 驻留**：年龄结构设备分支跑完才回传（无历史）；空间已改为运行期**零逐 tick 回传**
  （仅历史边界与结束时 `sync_gpu_state`），但历史行仍先在 host 生成，**未做设备侧历史缓冲**（第 10 轮，待复核）。
- **迁移 CSR 已缓存**（`MigrationCache`，含随机 `fwd_*` scratch），只在 CSR 变化时重建；`migration_rate` 仍逐 tick 上传。
- **无 frontend/population 级 ensemble 入口与文档**；ensemble 仅到 backend 层。
- `survival_stochastic` 对非法状态 `n_virgins<-EPS` 静默夹 0，CPU 返回 `Err`（低危，未对齐）。
- P6 性能：GPU 单位工作量约快 ~40×，但相对 16 核 `ProcessPoolExecutor` 墙面加速比仅 **2.4–3.3×**，未达数量级。

---

## 9. 测试与门禁

- 门禁命令：`python scripts/check_rust.py`（fmt+clippy+check+test）、`cargo test`（默认 **67**）、
  `cargo test --features gpu`（当前 **155**，含 evaluator 用例）、`NATAL_GPU_REQUIRE=0 cargo test --features gpu`
  （跳过硬件）、`ruff`、`pyright`、`pytest`（3606）、`phase0_baseline.py --check`。
- GPU 测试默认**硬门禁**：`NATAL_GPU_REQUIRE` 未设=强制；CPU-only 主机需显式 `=0`。
- 覆盖：严格按绝对路径过滤 `rust/src/gpu/**`（**注意**：`--sources src/gpu` 会误含
  `src/gpu/../../tests/...`），当前聚合 ~97%；新模块需 ≥95%。
- 高风险改动必须由独立 evaluator 复核（走 `EVALUATE.md`，用 `adversarial-review` 技能）。

---

## 10. 环境复现（本服务器）

```bash
cd /home/jovyan/node2/natal-core
source .venv/bin/activate                     # Python 3.13 + 前端依赖 + maturin
export RUSTUP_HOME="$PWD/.venv/rustup" CARGO_HOME="$PWD/.venv/cargo"
export PATH="$PWD/.venv/cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/.venv/cargo-target"
# NVRTC 来自 conda 的 nvidia pip 包：
export LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
# Rust 测试链接 libpython（.venv 的 3.13 无共享库）：
export PYO3_PYTHON=/opt/conda/bin/python PYTHONHOME=/opt/conda   # 跑 phase0/pytest 前 unset
# 重编带 GPU 的扩展（注意 CLI features 会覆盖 pyproject 的 extension-module）：
.venv/bin/maturin develop --features "gpu,extension-module"
```

GPU：RTX 5090 D V2 / CUDA 13.2 / 驱动 595.84，**多租户共享**（benchmark 前记录 `nvidia-smi` 快照）。

---

## 11. 下一步计划（建议顺序）

1. **等 §25 / §27 / §29 回执**（第 8–10 轮）。P6 若判定“加速比不足”，需与用户确认阈值；可换更大单 replicate
   状态、更少 CPU 核、或更多 tick 重测。
2. **设备侧历史缓冲**（D5 完整形态）：把历史行改在设备侧暂存、`query()` 时一次下载；当前只做到“运行期零逐 tick
   回传 + 边界同步”。
3. **停止门控**（D6）：无钩子模型的 `stop_if_*` 设备侧门控，消除主机同步。
4. **`continuous_sampling` 支持**：连续二项/多项。
5. **离散世代 GPU 路径**（当前空间离散被拒）。
6. **frontend/population 级 ensemble 入口 + 文档**（`demos/` 或 `docs/` 同步，遵守中英同步规则）。
7. 性能优化：CSR 缓存已完成（第 10 轮）；剩余为 ecology 增量上传、融合 kernel、减少每 tick 启动。

每完成一个阶段：跑全部门禁 → 独立审查（`EVALUATE.md` 交接 → 回执）→ 再进入下一阶段。

---

## 12. 与既有文档的关系

- `HANDOFF.md`：P0–P1 时代的任务书与编辑禁忌，仍然有效的部分：CPU 不可改、默认关闭、精度分档、共享 GPU 方法学。
- `GPU_insert_PLAN.md`：D1–D7 与阶段定义；但 Phase 顺序在实施中做了纠正（先确定性全链路再接随机；
  空间随机迁移作为独立阶段），本文档 §3 为准。
- `AGENTS.md` / `quality_checks_spec*.md`：约束与门禁的最高依据。
- `EVALUATE.md`：主 agent ↔ evaluator 的唯一沟通与审查记录。
