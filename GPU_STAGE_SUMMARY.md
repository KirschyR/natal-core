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
第 10 轮（CSR 缓存 + 零逐 tick 回传，§28）→ §29 **NOT APPROVED**（公共单 tick 读路径陈旧），
已按 §30 修复（公共 `run_tick` 同步、`run_steps` 调 `advance_tick`），第 11 轮 **APPROVED**（§31）；
第 12 轮（观测模式设备侧历史缓冲，§32）→ §33 **NOT APPROVED**（预算算术溢出 panic），
已按 §34 修复（全程 checked、显式 Err），第 13 轮 **APPROVED**（§35）。
第 14 轮（`continuous_sampling` 设备支持，§36）第 14 轮 **APPROVED**（§37）；
第 15 轮（离散世代空间 GPU，§38）已实现并自测，**待复核（§39）**。

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
| `OBSERVATION_SOURCE` | `observation_project` | 观测历史行设备投影（复刻 host `project`） |
| `DISCRETE_SOURCE` | `discrete_reproduction` / `discrete_survival` | 离散二龄生命周期（繁殖/存活，离散或连续） |
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
  `enable_gpu()`/`gpu_status()`（现接受 `stochastic=true` 与 `continuous_sampling=true`；第 14 轮起）、
  `enable_gpu_ensemble()`/`run_gpu_ensemble()`；`run_inner` 开头**提前返回**设备分支。
- 空间：`SpatialSession`（`HeterogeneousSpatialEngineSession`）新增 `gpu` 字段、`enable_gpu()`/
  `gpu_status()`（接受确定性 + 随机 + 连续 + 离散）、`run_gpu_tick`；`run_inner` 提前返回；
  确定性用 `migrate_tick`、随机用 `migrate_tick_stochastic`。
- Python：`rust_backend.py` 三个后端类分别透传；扩展未带 gpu 时给出可操作 `RuntimeError`。

---

## 8. 已知限制 / 未做项

- **`continuous_sampling=true`**：**已支持**（第 14 轮）：设备连续二项/多项/Poisson 采样，生存/繁殖/迁移随机阶段均可用。
- **hooks/停止门控**：**P7.1–P7.3 已实现**——声明式钩子（SCALE/SET/ADD/SUBTRACT/KILL/SAMPLE、STOP_IF_*、
  SET_PARAM、CONVERT，含 RPN 条件与 selector）在**确定性及随机** panmictic 模型上按 `first → reproduction →
  early → survival → late → aging` 事件点执行。降采样/SAMPLE/CONVERT 使用设备 counter-based RNG（site 4..7，
  与生命周期 0..3 不冲突；同 seed GPU↔GPU 可复现、与 CPU 统计等价）；`STOP_IF_*` 设备侧归约并中止当 tick；
  `SET_PARAM` 同 tick 可见（设备 RPN 写 eco scratch，host 事件边界提交）。仅 **Python 回调**仍显式拒绝。
  **空间钩子已支持（P7.4a）**：按 deme selector 调度，stop 按 batch 掩码冻结（其他 deme 仍跑完），迁移在停止时跳过；
  **ensemble 钩子仍拒绝**（P7.4b，语义待定）。
- **离散世代 GPU**：空间离散已支持（第 15 轮）；未覆盖 Wright-Fisher 融合模式。
- **设备侧 history 驻留已实现（B6）**：空间观测历史在设备上投影成行、运行期零逐记录回传、结束时一次下载回填
  `HistoryStore`；**raw 模式同样设备暂存**（设备原生布局，flush 时转置回 batch-major），且设备窗口按
  `max_rows` 收缩为环形（`capacity=min(records, max_rows)`），避免大 D×长 T 超预算回退；flush 时按行重建
  raw checkpoint 并写入 boundary phase/execution。窗口确实超预算时仍走 host 逐记录路径（非引擎回退）。
- **迁移 CSR 已缓存**（`MigrationCache`，含随机 `fwd_*` scratch），只在 CSR 变化时重建；`migration_rate` 仍逐 tick 上传。
- **frontend/population 级 ensemble 入口已提供**（`AgeStructuredPopulation.enable_gpu_ensemble` / `run_gpu_ensemble`，第 16 轮）；中英文档见 `docs/{zh,en}/4_simulation_engine.md` §11。
- `survival_stochastic` 对非法状态 `n_virgins<-EPS` 静默夹 0，CPU 返回 `Err`（低危，未对齐）。
- P6 性能：GPU 单位工作量约快 ~40×，但相对 16 核 `ProcessPoolExecutor` 墙面加速比仅 **2.4–3.3×**，未达数量级。

> 上述未完成项的编号、优先级、建议修法与验收口径统一见 **§11.2**。

---

## 9. 测试与门禁

- 门禁命令：`python scripts/check_rust.py`（fmt+clippy+check+test）、`cargo test`（默认 **67**）、
  `cargo test --features gpu`（当前 **241**，含 evaluator 用例）、`NATAL_GPU_REQUIRE=0 cargo test --features gpu`
  （跳过硬件）、`ruff`、`pyright`、`pytest`（3628）、`phase0_baseline.py --check`。
- GPU 测试默认**硬门禁**：`NATAL_GPU_REQUIRE` 未设=强制；CPU-only 主机需显式 `=0`。
- 覆盖：严格按绝对路径过滤 `rust/src/gpu/**`（**注意**：`--sources src/gpu` 会误含
  `src/gpu/../../tests/...`），当前聚合 **96.99%**（executor 96.5%、kernels 97.35%、probe 96.13%、其余 100%）；新模块需 ≥95%。
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

## 11. 计划与推进路线

> 本节是接手者的**唯一执行清单**。每完成一项：跑全部门禁 → 独立审查（`EVALUATE.md` 交接 → 回执）
> → 再进入下一项。高风险项（改科学公式/随机分布/状态恢复/公开 API/Python-Rust 交换）必须独立复核。

### 11.1 已完成路线

| 项 | 内容 | 状态 |
|---|---|---|
| P0–P6 | 探针/布局/executor/确定性内核/随机内核/迁移/ensemble | 全部 APPROVED |
| 历史 | 观测模式设备侧历史投影 + 一次回传（§32） | APPROVED |
| 缓存 | CSR 静态缓存 + 空间逐 tick 零回传（§28） | APPROVED |
| 连续采样 | 设备连续二项/多项/Poisson（§36） | APPROVED |
| 离散世代 | 空间离散设备生命周期（§38） | APPROVED |
| frontend | `enable_gpu_ensemble`/`run_gpu_ensemble` + 中英文档（§40） | APPROVED |
| 性能 | 移除每 tick 缩放 D2H 同步（§42） | APPROVED |
| A1 | GPU checkpoint 设备恢复（§44） | APPROVED（§45） |
| A2 | 随机生存非法状态显式报错（§46） | APPROVED（§49） |
| C10 | 迁移缓存纳入启用时显存预算（§50） | APPROVED（§51） |
| B4 | frontend 单群体 `enable_gpu` + 文档（§52） | APPROVED（§53） |
| B5 | ensemble 结果接入 Observation（§54） | APPROVED（§55） |
| B6 | raw 历史设备驻留 + 窗口按 `max_rows` 环形收缩（§56） | APPROVED（§57） |
| C8 | 设备连续采样阈值统一到 `NATAL_EPS=1e-10`（§56） | APPROVED（§57） |
| C11 | `run_gpu_ensemble` 误用前置显式报错（§56） | APPROVED（§57） |
| A3 | 设备 flush 按 tick 写 boundary metadata（随 B6，§56） | APPROVED（§57） |
| P7.1 | 设备侧确定性声明式钩子解释器 + 按 opcode 放开资格（§58） | APPROVED（§59） |
| P7.1-fix | §59.4 两条 medium：启用后设备 tick 对齐 + 设钩子重传设备 CSR（§60） | APPROVED（§61） |
| P7.2 | 设备侧 `STOP_IF_*` 门控 + 按 opcode 放开（§62） | APPROVED（§63） |
| P7.3 | `SET_PARAM` 同 tick 可见性 + `SAMPLE`/随机模型钩子 + 设备 RNG（§64） | APPROVED（§67；§65 阻塞项已修复） |
| P7.4a | 空间钩子（per-deme selector + per-batch stop 掩码/恢复 + 迁移跳过）（§68） | APPROVED（§69） |
| P9 | GPU particle：per-particle 参数（`enable_gpu_particles`/`run_gpu_particles`）（§70） | APPROVED（§71） |
| P9-fix | 粒子批次钩子 selector 语义统一为 deme=0（空间仍 deme=b）（§72） | APPROVED（§73） |
| P9b | 粒子内 replicate 轴（`P×R` 拍平，面向 ABC-SMC/参数 combo 批）（§74） | APPROVED（§75） |
| P9c | 粒子 `state_tick` 语义 + ABC 迭代间复用执行器（§76） | APPROVED（§77） |
| P9d | 灭绝 particle/replicate 跳过后续 stage（per-batch 活跃掩码，无钩子程序）（§78） | APPROVED（§79） |
| P9e | GPU particle 逐粒子 genetics（每粒子生态+遗传表；相同表去重为变体库）（§82） | APPROVED（§83） |
| P9f | GPU particle 逐周观测历史（设备投影、环形窗口、单次回传、无逐 tick 同步）（§84） | **已实现，待 §85 复核** |

### 11.2 未完成项计划表（按建议优先级）

| ID | 类别 | 问题（简单说） | 影响 | 建议修法 | 风险 | 状态 |
|---|---|---|---|---|---|---|
| **A1** | 正确性 | ~~GPU 会话 restore 不回滚显存状态~~ | ~~静默错误轨迹~~ | 已按「完整设备恢复」实现：`GpuExecutor::restore_state` + 会话恢复时重传/回退设备 tick（第 18 轮，§44） | 中 | **APPROVED（§45）** |
| **A2** | 正确性 | ~~`survival_stochastic` 静默夹 0~~ | ~~非法状态被掩盖~~ | 已实现：`violation` 标志 + f32 容差 `1e-3`，超出显式 `Err`（第 19 轮，§46） | 低 | **APPROVED（§49）** |
| **A3** | 质量 | ~~设备历史 `flush` 的 boundary metadata（phase/execution）未按 tick 写~~ | ~~stop/异常边界的历史元数据可能与 CPU 不同~~ | 已实现：flush 回填时按当前 phase/execution 写入（GPU 无钩子/停止，故与逐 tick 相同；第 24 轮，§56） | 低 | **已实现，待 §57 复核** |
| **C10** | 质量 | ~~CSR 迁移缓存未纳入显存预算估算~~ | ~~首次迁移才报错~~ | 已实现：`migration_cache_bytes` + `ensure_migration_budget`，启用时校验（第 21 轮，§50） | 低 | **APPROVED（§51）** |
| **B4** | 完整性 | ~~frontend 无单群体 `enable_gpu` 入口~~ | ~~无法通过 Population API 启用单群体 GPU~~ | 已实现：`enable_gpu()`/`gpu_status()` + 中英文档 §11.1（第 22 轮，§52） | 低 | **APPROVED（§53）** |
| **B5** | 完整性 | ~~ensemble 返回裸数组，未接 Observation~~ | ~~不能复用 observation/history 选择器~~ | 已实现：`observe_gpu_ensemble` 逐 replicate 经 `Observation` 投影（第 23 轮，§54） | 低-中 | **APPROVED（§55）** |
| **B6** | 性能/内存 | ~~raw 历史不走设备驻留；设备窗口未按 `max_rows` 收缩~~ | ~~raw 逐记录同步；大 D×长 T 超预算回退 host~~ | 已实现：raw 行设备暂存（设备原生布局，flush 转置）+ 环形窗口 `capacity=min(records, max_rows)`（第 24 轮，§56） | 中 | **APPROVED（§57）** |
| **C8** | 一致性 | ~~设备连续采样阈值未统一到 host `EPS=1e-10`~~ | ~~极小窗口分支差异~~ | 已实现：新增 `NATAL_EPS 1e-10f`，连续二项/多项/Poisson 全部改用（第 24 轮，§56） | 低 | **APPROVED（§57）** |
| **C11** | 质量 | ~~`run_gpu_ensemble` 在 `_gpu_ensemble_replicates==0` 时 reshape 失败~~ | ~~误用路径产生困惑报错~~ | 已实现：前置 `replicates < 1` 显式 `RuntimeError`（第 24 轮，§56） | 低 | **APPROVED（§57）** |
| **C9** | 一致性 | `male_adult_mating_rate`/`eggs_per_female` clamp 差异 | 契约范围内无差异 | 可选对齐或注释说明 | 低 | 已记录，可不做 |
| **B7** | 完整性 | Wright-Fisher 融合模式未设备化 | 空间离散 CPU 不用它；非空间离散无 GPU 入口 | 按需实现 | 中 | 低优先 |
| **D6** | 设计 | ~~hooks/停止门控设备侧未做~~ | ~~含 `stop_if_*` 的模型不可用 GPU~~ | **由 P7 承接**（用户选定声明式钩子）：设备侧解释器 + 停止门控 | 高 | **APPROVED（§63）** |
| **P7** | 完整性 | ~~GPU 不支持声明式钩子插入点（`first/early/late/finish`）~~ | ~~含声明式钩子的模型无法上 GPU~~ | 设备侧声明式解释器，分 P7.1–P7.4（见 §11.5）；Python 回调仍显式拒绝 | 高 | **P7.1–P7.4a APPROVED（§59/§61/§63/§67/§69）**；P7.4b（ensemble）待用户决定 |
| **E12** | 性能 | ~~debug 基准受宿主机重建/上传主导~~ | ~~大 B 计算侧瓶颈~~ | 已实现：executor 侧 `ParamCache` 缓存生态列与分块遗传表（内容不变即不重传），并复用 scaling/factor scratch；release 实测 B=5000/50tick 0.055→0.021s、B=100k 112→55ms/tick、B=500k 626→283ms/tick | 高（核心路径） | **APPROVED（§81）** |
| **E13** | 验收 | P6 相对 16 核 `ProcessPool` 仅 2.4–3.3×，阈值未定 | “显著优于”是否达标无口径 | 用户定阈值或换更大模型/更少核重测 | 低 | **待用户口径** |

### 11.3 建议推进顺序

1. **A1**（正确性、静默错误）→ **A2** → **C10**：先消灭正确性/显式失败缺口。
2. **B4** → **B5**：补齐用户可见入口与结果集成。
3. **B6** → **C8** → **C11** → **A3**（APPROVED，§57）：性能/内存与一致性收尾已完成。
4. **P7**（设备侧声明式钩子，含 D6）：大特性，按 §11.5 分阶段（P7.1→P7.4）；**P7.1–P7.3 APPROVED（§59/§61/§63/§67）**；**B7** 低优先；**E12/E13** 需空闲 GPU 与用户口径。

### 11.4 验收口径（按类别）

- 正确性/完整性（A/B/C）：针对性测试 + `phase0` bit-identical + L3 对照；公开 API/状态类改动走独立复核。
- 文档类：`docs/zh` 与 `docs/en` 同步、示例可运行。
- 性能类：`maturin develop --release` + 空闲 GPU + 前后 `nvidia-smi` 快照，给出可复现基准。

### 11.5 P7 规划：设备侧声明式钩子（用户已选定）

**目标**：让**仅含声明式钩子**（`HookProgram`，无 Python 回调）的模型在 GPU 上按 CPU 相同的
`first → reproduction → early → survival → late → aging` 事件顺序与优先级执行，且尽量保持
“运行期间零主机同步”。**Python 回调模型仍显式拒绝**（CUDA 内核无法回调 Python；如需支持另立 P8 主机桥接）。

现状（依据）：
- `HookProgram`（`rust/src/hooks/interpreter.rs`）是完整 CSR 字节码：`op_types`（SCALE/SET/ADD/SUBTRACT/KILL/SAMPLE/STOP_IF_*/SET_PARAM/CONVERT）、selector（`zidx/age/sex/deme`）、RPN 条件、`OP_SET_PARAM` 的 RPN 栈机 + wire bounds。
- GPU 会话原对 `n_hooks != 0` 或任何 `python_callbacks` 一律拒绝；**P7.1 起改为按 opcode 放开**。

分阶段（每阶段：全门禁 → 独立复核）：

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P7.1** | 设备侧 opcode 解释器：上传 CSR 数据并执行 SCALE/SET/ADD/SUBTRACT/KILL/CONVERT（含 RPN 条件与 selector/wire bounds）；**同时按 opcode 放开资格**（仅无 Python 回调且只用已支持 opcode 的确定性模型；其余显式拒绝）。**APPROVED（§59）；§59.4 两条 medium 修复 APPROVED（§61）** | 确定性 L2/L3 与 CPU 一致（相对误差档）；事件顺序/优先级一致；未支持 opcode/回调解仍 `Err` |
| **P7.2** | 设备侧 `STOP_IF_*` 门控（D6）：设备侧归约出 stop 标志，host 按需读取；停止点与 CPU 一致；放开 STOP_IF_* 资格。**APPROVED（§63）** | 含 `stop_if_*` 模型的停止 tick 与 CPU 一致；零逐 tick 同步 |
| **P7.3** | `OP_SET_PARAM` 同 tick 可见性 + `SAMPLE` 的设备 RNG site 对齐；放开二者资格。**APPROVED（§67）** | later-stage 读取已更新参数；随机 op 可复现/统计等价 |
| **P7.4a** | 空间 per-deme selector + per-batch stop 掩码/恢复 + 迁移跳过。**APPROVED（§69）** | 空间声明式钩子与 CPU 一致（含停止语义） |
| **P7.4b** | ensemble 钩子集成（语义待定） | 多 B 的声明式钩子与 CPU/N 个独立单群体一致 |

> **不可先放开资格再补解释器**：若允许 `n_hooks>0` 而不执行钩子，会**静默丢钩子**。因此资格必须在
> 对应 opcode 的解释器落地时**按 opcode 逐步放开**（P7.1 起）。

**关键语义约束**：事件顺序与 priority 交错、`set_param` 边界可见性、stop 中断时机、RNG site
映射必须与 CPU 完全一致（CPU 仍是 golden reference）；精度分档不变。

**风险**：高（钩子会改写状态/参数、影响停止点与随机流）。必须独立 evaluator 复核并补强对照测试。
**前置**：P7 是大特性，建议在完成 B5/B6/C8/C11/A3 等小项后启动，或由用户指定优先级。

### 11.6 接手状态快照（2026-09-24，上下文切换）

- **分支/HEAD**：`feat/gpu-merge-test`；P7.1–P7.4a 及 §59.4/§65 修复已由用户提交并 APPROVED（§59/§61/§63/§67/§69）。
- **最近回执**：§79（P9d）**APPROVED**；P9 增强项（P9a–P9d）已全部完成。
- **待回执**：**§85**（P9f：GPU particle 逐周观测历史，高风险公开 API/数值边界）——已实现并自测。
- **后续未开始**：**E13** 待用户口径；**P7.4b**（ensemble 钩子）、**B7**、**C9** 已决定不做（除非用户改口）。
- **ABC-SMC 性能基准已跑（§84.3）**：组件微基准 + 粒子历史对比完成，结果见 `EVALUATE.md` §84。
- **门禁基线（§69 时）**：`check_rust.py` EXIT=0、`cargo test` 67、`cargo test --features gpu` **221**、
  `ruff`/`pyright` 通过、`pytest` **3620**、`phase0` bit-identical、`rust/src/gpu/**` 聚合覆盖率 **96.62%**。
- **P9 要点（GPU particle）**：`AgeStructuredPopulation.enable_gpu_particles(param_sets)` /
  `run_gpu_particles` → `RustLifecycleBackend` 透传 → `AgeStructuredSession::enable_gpu_particles`
  （对每个 particle `EcologyParams::from_python(_,1)` 后 `stack_ecologies` 成 `n_demes=B`）+ `run_gpu_particles`；
  初始状态/genetics 共享；声明式钩子按 particle 执行。**P9b**：`n_replicates=R` 使设备 batch = `P×R`（particle-major 拍平），返回 `(P,R,2,A,Z)`。Rust 对照测试 `session_gpu_particles_match_independent_cpu_runs`、`session_gpu_particles_with_replicates_match_cpu`；
  Python 测试 `tests/test_gpu_particles_frontend.py`。
- **P9 selector 语义已修并 APPROVED（§72/§73）**：钩子内核按 panmictic 模式对**每个 batch 用 deme=0**（单群体/ensemble/particle），仅**空间**路径用 `deme=b`。
- **P9 已知边界（非阻塞）**：`run_gpu_particles` 不更新会话 `state_tick`（与 ensemble 同）；
  已灭绝 particle 不跳阶段；初始状态/genetics 共享。
- **P7.3 要点**：`HOOK_SOURCE` 增设备端 `hook_sample_survivors`/`hook_apply_target_*`/`hook_convert_count`、
  `hook_eval_rpn`（set_param）；内核接收 `eco`/RNG 参数；`DeviceHooks` 增 `has_set_param` 与 sp/RPN/eco 缓冲；
  `GpuExecutor` 增 `sync_eco_scratch`/`take_eco_scratch`/`apply_eco_values`/`take_pending_eco` 与 `EcoView`
  （set_param 同 tick 可见）；`run_gpu` 以 `&mut params` 提交；资格放开全部 opcode 与随机模型（仅拒 Python 回调）。
- **已知低风险**：设备 `STOP_IF_*` 归约用 f32（CPU 为 f64）；整数计数 <2²⁴ 逐位，超大/近阈值可能分歧（§63.5）。
- **证据入口**：每轮交接/回执在 `EVALUATE.md`（主 agent 交接区在前、evaluator 回执区在后；只追加不改写）。
- **CPU 底线**：`rust/src/kernels`、`rust/src/model`、`src/natal/contracts`、`rust/src/lib.rs` 数值语义零改动；
  `phase0_baseline.py --check` 必须始终 bit-identical。

---

## 12. 与既有文档的关系

- `HANDOFF.md`：P0–P1 时代的任务书与编辑禁忌，仍然有效的部分：CPU 不可改、默认关闭、精度分档、共享 GPU 方法学。
- `GPU_insert_PLAN.md`：D1–D7 与阶段定义；但 Phase 顺序在实施中做了纠正（先确定性全链路再接随机；
  空间随机迁移作为独立阶段），本文档 §3 为准。
- `AGENTS.md` / `quality_checks_spec*.md`：约束与门禁的最高依据。
- `EVALUATE.md`：主 agent ↔ evaluator 的唯一沟通与审查记录。
