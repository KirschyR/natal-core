# HANDOFF — 给接手 GPU 旁路实现的 agent

> **先读 [`AGENTS.md`](./AGENTS.md)。** 那是项目的**约束性规范**（授权边界、风险分类、验证流程、完成标准）。
> 本文档描述**当前任务**；`AGENTS.md` 描述**工作方式**。两者都必须遵守，冲突时以 `AGENTS.md` 为准。

---

## 0. 三十秒版

| 项 | 内容 |
|---|---|
| **分支** | `feat/gpu-merge-test` |
| **已完成** | P0-a：feature-gated 的零依赖 CUDA 环境探针（可编译、可测试、本机已验证） |
| **你现在要做** | **P0-b**：接入 `cudarc`，跑绑定级断言（动态加载 + NVRTC 实编译 + `Send` 检查） |
| **最终目标** | 给 Rust 引擎加一条**可选的 GPU 旁路**，加速**大 Deme 空间模型**与**大 Batch 参数推断** |
| **硬件** | RTX 5090 D V2 / CUDA 13.2 / 驱动 595.84 —— **但显卡是多租户共享的** |
| **第一原则** | **CPU 路径一行都不能改** —— 它是唯一的 golden reference |
| **最大的坑** | CPU 是 f64、GPU 是 f32，**不要用"逐位一致"验收含算术的 kernel** |

---

## 1. 这个项目是什么

### 1.1 一句话

**natal-core 是一个前向时间（forward-time）群体遗传学模拟引擎，专为基因驱动（gene drive）建模设计。** 架构上是 **Python 前端 + Rust 原生内核**，中间夹一层显式的**数据契约**做边界。

### 1.2 四层架构

```
src/natal/__init__.py        ← ① 惰性导出索引（导入时不加载任何子模块）
src/natal/frontend/**        ← ② 全部 Python 领域层（16 个子包，112/120 个 .py）
src/natal/contracts/**       ← ③ 边界契约（Blueprint 冻结 / Params 可变）
rust/                        ← ④ 原生引擎，单一 crate → 扩展模块 _engine_rs
```

**③ 是理解本项目的钥匙 —— 一条绝对的「冻结 / 可变」分界线：**

- `Blueprint`（`src/natal/contracts/blueprint.py`）：`NamedTuple`，**冻结**。回答"**这是什么模型**" —— 维度、执行标志、符号名目录、性染色体掩码、初始状态、CSR 迁移矩阵。**改任何一个字段 = 换了一个模型**，必须重建。数组通过 `frozen()` 标记为只读。
- `Params`（`src/natal/contracts/params.py`）：`@dataclass`，**可变**。**所有**运行时可调值都在这里：生态节（5 个有界标量 + 人口学向量）+ 遗传节（4 个适配度张量 + 后代张量 + 减数分裂映射等）。
- `materialize()`（`src/natal/contracts/materialize.py:210`）：两者的**唯一诞生地**。
- `CONTRACTS_VERSION = 2`（`src/natal/contracts/__init__.py:41`）—— 契约一变必须同步更新 Rust 侧镜像。

### 1.3 关键数字

| 项 | 数值 |
|---|---|
| Python 源码 | 120 个 `.py`（其中 112 个在 `frontend/`） |
| Rust 源码 | 43 个 `.rs` |
| Python 测试文件 | 132 个 |
| Rust 单测 | 67 个（`rust/tests/unit/`） |
| 构建后端 | `maturin`（`pyproject.toml`），crate `natal-engine-core`，`cdylib` → `_engine_rs` |
| Rust 依赖 | `pyo3 0.23.5`、`numpy 0.23`、`rand 0.10.2`、`rand_distr 0.6.0`、`rayon 1.10` |
| Rust msrv | `rust-version = "1.86"` |

### 1.4 你不需要先搞懂的部分

不要为了理解全貌去读这些（会浪费大量上下文）：

- ❌ `frontend/ui/`、`frontend/webui/` —— NiceGUI / FastAPI 界面，与本任务无关
- ❌ `frontend/genetics/` 的遗传学细节 —— 只要知道它**编译成张量**
- ❌ `docs/` 下的文档 —— **多处已过时**（见 §5.5）

**你真正需要读的只有：**

| 文件 | 为什么 |
|---|---|
| `rust/src/kernels/spatial.rs` | **唯一的并行点**，也是 GPU 的插入位置参照 |
| `rust/src/sessions/spatial.rs` | **批处理状态布局** —— GPU 的数据布局模板 |
| `rust/src/gpu/probe.rs` | 上一阶段留下的探针，P0-b 要扩展它 |
| `src/natal/contracts/blueprint.py` + `params.py` | 边界契约 |

---

## 2. 当前任务：给引擎加一条可选的 GPU 旁路

### 2.1 目标与非目标

**目标**

1. 让**空间模型的多 deme** 计算跑在 GPU 上（大 `D`）。
2. 让**非空间模型的多样本批（replicate）** 计算跑在 GPU 上（大 `B`）。
3. 状态常驻显存，**逐 tick 不回传**，只在计算结束时批量回传（含中间记录）。
4. **GPU 是可选启用项**；CPU 路径完整保留，作为唯一 golden reference。

**非目标**

- ❌ 不追求 GPU 与 CPU 逐位一致（物理上不可达，见 §5.2）。
- ❌ **不替换、不删除 CPU 路径的任何代码。**
- ❌ 不改动 Python 侧公开 API（`PopulationBuilder` / `run()` / `History` 语义不变）。
- ❌ 不引入 `torch` 作为运行时依赖。
- ❌ **暂不做 `D × B`**（即"空间模型的参数推断"）。batch 轴严格一维：要么大 `D`，要么大 `B`。

### 2.2 设计决定（**已冻结，不要重新讨论**）

这些是已经拍板的决定。**如果你认为某条是错的，先提出，不要擅自改。**

| # | 决定 | 理由 |
|---|---|---|
| **D1** | **用 CUDA（`cudarc` crate），不用 OpenCL** | 目标硬件是 RTX 5090；NVIDIA 的 OpenCL 实现停滞在 3.0。`cudarc` 支持**动态加载**（构建期不需要 CUDA toolkit）+ **NVRTC 运行时编译**（kernel 以字符串嵌入） |
| **D2** | **f32，不是 f64** | RTX 5090 的 FP64 只有 FP32 的 **1/64**（约 1.6 TFLOPS）→ f64 上 GPU 等于没有加速。**且必须配树形归约**（见 §5.2） |
| **D3** | **counter-based RNG（Philox4x32-10）**；接受 CPU↔GPU 不可逐位复现 | 顺序随机流与数据并行结构上不相容。目标改为 **GPU↔GPU 可复现** |
| **D4** | batch 轴：**大 `D`（deme）与大 `B`（replicate）为主力**；`Z`（基因型）不是威胁 | 实测模型 ≤2 个基因座，`Z ≈ 9~16` → `Z³ ≤ 4096`，不是瓶颈。**热点是「每格点一次随机抽样」** |
| **D5** | 历史记录**全程驻留显存**；显存不足时**显式报错，不静默回退 CPU** | 用户裁定。运行中换引擎会产出**两个引擎都无法复现的混合轨迹** |
| **D6** | 停止条件用**设备侧门控**（零主机同步） | 把门控搬到设备侧即可完全复现 CPU 语义 |
| **D7** | 设备路径用 `py.allow_threads` 释放 GIL | 现状**全树零 `allow_threads`**，批量循环全程持有 GIL |

### 2.3 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0-a** | 零依赖 CUDA 环境探针 | ✅ **已完成** |
| **P0-b** | 接入 `cudarc` + 绑定级断言 | ⬜ **← 你现在在这里** |
| **P1** | `rust/src/gpu/` 骨架 + **布局转置** | ⬜ |
| **P2** | 单内核贯通端到端 + 预算守卫 | ⬜ |
| **P3** | 确定性数值内核 + 停止门控 | ⬜ |
| **P4** | 采样内核 + counter-based RNG | ⬜ |
| **P5** | 迁移 CSR + 大 D 场景 | ⬜ |
| **P6** | 多 `B`（ensemble session） | ⬜ |

**P0-a 已交付的文件（本次接手的基础）：**

| 文件 | 内容 |
|---|---|
| `rust/Cargo.toml` | 新增 `gpu = []` feature（**目前无依赖**） |
| `rust/src/lib.rs` | `#[cfg(feature = "gpu")] pub mod gpu;` |
| `rust/src/gpu/mod.rs` | feature-gated 模块声明 |
| `rust/src/gpu/probe.rs` | `CudaProbe::scan()` / `is_usable()` / `report()`；`GpuDevice::memory_free_mib()` |
| `rust/tests/unit/gpu/probe.rs` | 7 个测试，无 GPU 时优雅跳过 |

---

## 3. 你现在要做的：P0-b

### 3.1 环境事实（已实测，不要再假设）

```
NVIDIA-SMI 595.84        Driver Version: 595.84     CUDA Version: 13.2
GPU: NVIDIA GeForce RTX 5090 D V2
Memory-Usage: 19643MiB / 24455MiB      ← ⚠️ 已用 19.2 GB，仅约 4.7 GB 可用
GPU-Util: 52%   Pwr: 397W/600W   P1
Processes: No running processes found  ← 却看不到进程
```

**两条关键推论：**

1. **这张卡是多租户共享的。** "有 19.2 GB 占用但看不到进程"是容器/多租户环境的典型现象 —— 本容器之外有别的负载在用这张卡。**不要假设你能独占，也不要假设 24 GB 可用。**
2. **`cudarc` 要用 `cuda-13020` feature**（cudarc 支持 `cuda-13010` … `cuda-13030`，13.2 正中靶心）。sm_120 需要 CUDA ≥ 12.8，已满足。

### 3.2 具体步骤

#### 步骤 1：接入 `cudarc`

```bash
cd rust

# 让 cargo 自己解析最新版本，不要手写一个你没验证过的版本号
cargo add cudarc --optional --features cuda-13020
```

然后**手动把 feature 接上**（`cargo add` 不会自动改你已有的 `gpu` feature）：

```toml
[features]
extension-module = ["pyo3/extension-module"]
gpu = ["dep:cudarc"]      # ← 把 [] 改成这样
```

> ⚠️ **这一步需要网络。** 服务器上首次 `cargo add` / `cargo build` 会拉 crates.io 索引。
> ⚠️ 从此刻起，**默认构建（不带 `--features gpu`）不再需要 `cudarc`，但 Cargo 解析依赖图仍需要它可获取** —— 如果服务器也断网，这条路走不通，必须先解决网络。

#### 步骤 2：扩展探针，加绑定级断言

在 `rust/src/gpu/` 下建议新增 `cuda.rs`（保持 `probe.rs` 的零依赖性质不变，两者职责分离）。

**需要新增的 5 条断言：**

| # | 断言 | 怎么验 |
|---|---|---|
| 1 | `cudarc` 能动态加载 CUDA | 构造 `CudaContext` / 调用设备计数 API，不 panic |
| 2 | 设备 `compute_capability == 12.0` | 从设备属性读取并与 `probe.rs` 的 `nvidia-smi` 结果交叉核对 |
| 3 | **NVRTC 能编译并运行一个 trivial kernel** | ⚠️ **最关键的一条**，见下 |
| 4 | `cudaMemGetInfo` 返回可用显存 | 与 `probe.rs` 的 `memory_free_mib()` 交叉核对 |
| 5 | `GpuExecutor`（或 CUDA 上下文句柄）是 `Send` | **编译期**断言，见下 |

**断言 3 的代码骨架**（NVRTC 端到端最小验证）：

```rust
// 1) 用 NVRTC 编译一段 CUDA C
const SOURCE: &str = r#"
extern "C" __global__ void add_one(float* x) { x[0] += 1.0f; }
"#;
// 2) 编译 -> PTX/cubin
// 3) 加载为 module，取出 kernel
// 4) 分配设备缓冲、写入 41.0f、启动 kernel、拷回
// 5) 断言结果约等于 42.0f（浮点用容差，不要用 ==）
```

> **为什么这条最关键**：如果容器里没有 `libnvrtc.so`，就必须改走 **AOT 路线**（构建期 `nvcc` 生成 cubin 嵌入），那会**改变 `maturin` 的打包流程**，是唯一可能推翻 D1 的情形。

**断言 5 的写法**（编译期，不占运行时）：

```rust
#[test]
fn cuda_context_is_send() {
    fn assert_send<T: Send>() {}
    // 换成 cudarc 里实际的上下文/句柄类型
    assert_send::<cudarc::driver::CudaContext>();
}
```

> **为什么这条重要**：D7 要求用 `py.allow_threads(|| ...)` 释放 GIL，而它要求闭包捕获的类型满足 `Send`。**如果句柄不是 `Send`，就必须用 `Mutex` 包装或在工作线程内持有上下文** —— 这会改变 §3 的架构细节，所以要**尽早知道**。

#### 步骤 3：在服务器上跑

```bash
export NATAL_GPU_REQUIRE=1
cargo test --features gpu -- --nocapture
```

`NATAL_GPU_REQUIRE=1` 是 P0-a 留下的开关：它把"硬件缺失/不符"从**跳过**升级为**硬失败**。本机（无 GPU）不设它时会优雅跳过。

### 3.3 验收

- [ ] `cargo test --features gpu` 全绿，且 P0-b 的 5 条断言**真的执行了**（不是跳过）
- [ ] `cargo test`（**默认 feature**）仍全绿，且**数量与改动前一致**
- [ ] `python scripts/check_rust.py` **EXIT=0**
- [ ] 把实测的**设备名 / compute capability / 可用显存 / NVRTC 是否可用**记录到交付说明里

---

## 4. 之后的路（P1–P6）

只做方向性说明，细节到阶段时再展开。

| 阶段 | 核心内容 | 关键点 |
|---|---|---|
| **P1** | `rust/src/gpu/` 骨架：`context.rs` / `buffers.rs` / **`layout.rs`** | ⚠️ **布局转置 `(B,2,A,Z) → (2,A,Z,B)` 是必需的** —— 见下 |
| **P2** | 衰老 kernel 打通端到端 + `gpu` 字段 + `enable_gpu()` + **显存预算守卫** | 唯一的分支点是 `run_inner`/`run_steps` 开头的**提前返回** |
| **P3** | 存活 / 密度 / 求偶 / 观测投影 / 树形归约 + **停止门控** | 用 L1 对照测试逐个验证 |
| **P4** | Philox RNG + 采样器 | **P3 全部验收通过前，不要写任何采样内核** |
| **P5** | 迁移 CSR scatter/gather | 大 `D` 主场景 |
| **P6** | `EnsembleEngineSession`（多 `B`） | 必须显著优于 `ProcessPoolExecutor`（那是免改代码就能拿到的基线） |

### 4.1 为什么「布局转置」是必需的

现有 CPU 布局 `(B, 2, A, Z)` 是为**每 batch 元素一块连续内存**设计的（`rust/src/kernels/spatial.rs:100` 的 `chunks_mut`）。GPU 线程若按 batch 索引，访问步长 = `2·A·Z`。

以 `A=8, Z=9` 计，**步长 = 144 个 f32** → **完全无法合并访存**，是 GPU 上最经典的性能杀手。

**解法：设备侧改用 `(2, A, Z, B)`**，把 batch 轴放到最后一维（stride = 1）。
**转置只发生在上传/下载边界，CPU 侧布局与全部现有代码不受影响。**

### 4.2 为什么「首个目标锁定空间模型（大 D）」

因为 batch 轴**已经存在**：

- `rust/src/kernels/spatial.rs:100` 已按 deme 切分：`ind_all.chunks_mut(ind_stride)`
- `rust/src/sessions/spatial.rs:61-88` 的 `rngs: Vec<SessionRng>` 已**每 deme 一条流**
- `(D, 2, A, Z)` 堆叠布局 **就是** GPU 需要的 batched 形状

**不需要新建概念。**

### 4.3 采样方法：序列轴外提（重要）

`multinomial` 的"顺序条件二项"法，其依赖链长度是**类别数 `K-1`**，**不是个体数 `n`**。因为 `Z ≈ 9~16`，链长仅 14~15。

**正确做法：把序列轴从"线程内层"提到"所有线程共同推进的外层循环"。**

```cuda
int cell = blockIdx.x * blockDim.x + threadIdx.x;
for (int k = 0; k < K - 1; ++k) {      // ← 所有线程同步推进同一个 k
    out[cell*K + k] = binomial(&rng, remaining[cell], p[k] / tail(p, k));
    remaining[cell] -= out[cell*K + k];
}
```

**为什么最优**：① **零 warp 发散**（同一 warp 内所有线程处于同一个 `k`）；② 工作量 `O(K)` 而非逐个体分桶的 `O(n)`；③ 完全精确；④ 单个 kernel 启动。

**基础分布采样器**：BTPE（binomial）、PTRS（Poisson）、Marsaglia-Tsang（gamma）本身就是**每线程独立拒绝采样、无跨线程依赖** —— 它们是"条件性"的但**不是"串行"的**。**真正需要改造的只有 `multinomial`。**

---

## 5. ⚠️ 编辑注意事项

### 5.1 绝对不能做的事

| ❌ 禁止 | 理由 |
|---|---|
| 修改任何现有 `rust/src/kernels/*.rs` 的数值逻辑 | CPU 是**唯一** golden reference（`natal.backends.reference` 已被删除，没有第二个引擎了） |
| 修改 `contracts/`（Python 与 Rust 两侧） | 契约是既定边界，GPU **复用**而非改造 |
| 删除或替换 CPU 路径任何代码 | 同上 |
| 让 `gpu` feature 默认开启 | 开发机无 NVIDIA，必须能默认构建 |
| 在 P3 验收通过前写采样内核 | 先打通管道，再引入 RNG 复杂度 |
| **未经明确要求 commit / push** | `AGENTS.md` 硬性规定 |
| **在 `natal-core/` 下新建 Markdown 文档** | `AGENTS.md` 硬性规定（本文档是用户明确要求的例外） |
| 修改 `.gitignore` | `AGENTS.md` 硬性规定 |

### 5.2 精度验收标准（**最容易搞错，务必看**）

> **CPU 侧是 `Vec<f64>`，GPU 侧是 f32。所以"与 CPU 逐位一致"这个标准，对含算术的 kernel 永远不可能满足。**

用错标准会让你花几天追一个不存在的 bug。

| 场景 | 验收标准 |
|---|---|
| **纯搬运 kernel**（衰老/移位）+ 整数计数 < 2²⁴ | **逐位一致** ✅ |
| **含算术 kernel**（存活/密度/求偶）+ 确定性模式 | **相对误差 < 10 × f32 eps（≈1.2×10⁻⁶）** |
| **随机模式** | **统计等价**（KS / 卡方 / 前四阶矩） |

**两条硬规则：**

1. **禁止用"逐位比对"验收含算术的 kernel。** 必然失败，且失败信息没有诊断价值。
2. **随机模式禁止逐位比对**（CPU 与 GPU 是两个不同系综）。

**量级校准参照**：`demos/gpu_spatial/age_structured/outputs/xpu_comparison_summary.json` 里记录的 f32 实测 `max_rel_diff = 7.3×10⁻⁷`，与 f32 eps 同量级。

**另一条相关约束**：f32 能精确表示的整数上限是 **2²⁴ = 16,777,216**。`state_ind` 存的是个体计数，**单个 batch 元素的计数一旦超过约 1600 万，整数本身就失真**。且**归约必须是树形/成对，禁止顺序累加**（顺序累加误差是 `O(n)`，树形是 `O(log n)`）。

> 顺带一提：现有 `rust/src/kernels/state_reduce.rs:20` 复刻 NumPy 成对求和，**动机正是精度**。GPU 路径必须沿用同样的树形结构。

### 5.3 `allow_threads` 的陷阱（**编译器不会拦你**）

D7 要求释放 GIL。但 PyO3 0.23.5 在 stable Rust 上的实现是：

```rust
// pyo3-0.23.5/src/marker.rs:191-193
pub unsafe trait Ungil {}
unsafe impl<T: Send> Ungil for T {}     // ← 所有 Send 类型自动满足
```

而 `Py<PyAny>` **是 `Send`**。

> **所以编译器不会阻止你在 `allow_threads` 闭包里访问 Python 对象。** 这是一个**人为 Safety 契约**（见该文件 `:140-143`），**违反 = 未定义行为** —— 可能崩溃，也可能静默错误，且在 stable 上编译器不拦，可能几个月后才暴露。

**对策：用类型设计堵死，而非靠"记得别写错"。** 把设备路径需要的所有数据放进一个**不含任何 `Py<...>` 字段**的 struct：

```rust
/// 刻意不含任何 Py<...> 字段 —— 满足 allow_threads 的 Ungil 契约。
pub struct BatchView<'a> {
    pub n_batch: usize,
    pub ind: &'a mut [f32],
    pub sperm: &'a mut [f32],
    pub eco: &'a mut [f32],
}
```

**另一个相关事实**：`allow_threads` **只释放 GIL，不会自己启动线程**（PyO3 文档 `marker.rs:37` 明说）。并行必须自己用 CUDA stream / rayon 实现。

### 5.4 已知的基线失败 —— **不要"修"它们**

当前分支上**已经存在**这些失败，**与你无关**。不要试图修复（`AGENTS.md` 禁止扩大范围），也不要把它们算作你的问题。**但它们会一直红着，你要能区分。**

| # | 现象 | 根因 | 影响 |
|---|---|---|---|
| 1 | `pytest` **28 个失败** | 26 个在 `tests/test_phase0_shims.py`（见下），1 个 `test_slab_presets.py`，1 个 `test_mgdrive1_formal_benchmark.py` | 已用 `git stash` 验证为基线失败 |
| 2 | `pytest` 报 **60 个收集错误**（`UnicodeDecodeError: 'gbk' codec`） | `src/natal/frontend/utils/parameters.py:182` 的 `open(path)` **缺 `encoding="utf-8"`** | **在本机中文区域设置下必现**；服务器若为 UTF-8 locale 则不会出现。临时绕过：`PYTHONUTF8=1 pytest -q` |
| 3 | `cargo clippy --all-targets -- -D warnings` **1 个 error** | `rust/tests/unit/kernels/density_regulation.rs:136` 的 `clippy::neg_multiply` | 本地 clippy 1.98 比 crate msrv 1.86 新。**不影响真实门禁** —— `check_rust.py` 用的是 `cargo clippy -- -D warnings`（**不带 `--all-targets`**），不编译测试代码 |

**关于第 1 条（26 个 `test_phase0_shims` 失败）的根因：**

> `src/natal/` 下遗留了 **15 个空目录**（`genetics/`、`hooks/`、`population/`、`engine/`、`numba/` …），里面**只剩陈旧的 `__pycache__`，零个 `.py` 文件**。
>
> 因为 Python 3 的隐式命名空间包，`import natal.genetics` **会静默成功**但返回一个空模块（`__file__` 为 `None`）。相关的 `test_no_legacy_packages_left_on_disk`、`test_legacy_engine_paths_not_importable`、`test_no_numba_package_remaining` 因此全部失败。
>
> **这是一个已确认的真实缺陷**（重构时没删干净旧目录），**但不在本任务授权范围内**。如果你想顺手清理，**先向用户确认** —— 删掉那 15 个空目录理论上就能修好这 26 个测试。

### 5.5 文档可信度警告

**仓库根目录有多份过时文档。不要相信它们。**

| 文档 | 状态 |
|---|---|
| `README.md`、`CHANGELOG.md` | ✅ **与代码一致** |
| `AGENTS.md` / `AGENTS.en.md` | ✅ 现行规范 |
| `CONTEXT.md` | ⚠️ 部分过时（仍写 `configurator/`，实际是 `builder/`） |
| `RUST_BACKEND_PLAN.md`、`RUST_BACKEND_IMPLEMENTATION.md`、`DENSITY_CURVE_PLAN.md`、`lifecycle-tick-unification-design.md` | ❌ **已过时**（仍描述 numba 后端，而 numba 已被完全移除） |
| `ARCHITECTURE_SIMPLIFICATION_PLAN.md` | ⚠️ 状态标注过时（方案其实已落地） |
| `TODO.md` | ⚠️ 混杂 |

**判断当前架构，以代码为准，不要以文档为准。**

### 5.6 环境陷阱

| 陷阱 | 说明 |
|---|---|
| **`git stash` 会改行尾符** | 在本项目上执行 `git stash`/`pop` 后，`rust/src/generated/ecology_parameters.rs` 会被 LF→CRLF 转换，产生 53 行的"假 diff"。**用 `git diff --ignore-cr-at-eol` 确认，用 `git checkout -- <file>` 恢复。** |
| **`_engine_rs.pyd` 是预编译产物** | 修改 Rust 后，**Python 侧不会自动生效**，必须 `maturin develop` 重新构建扩展。纯 Rust 阶段（P0/P1）不需要重建。 |
| **`nvidia-smi` 的显存 ≠ 你能用的显存** | 多租户共享，见 §3.1。§D5 的预算守卫必须用**实测可用值**。 |
| **benchmark 在共享 GPU 上会失真** | 每次测速前后必须记录同租户快照：<br>`nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv`<br>**加速比低于预期时，先怀疑邻居在跑活，再怀疑实现。** |

---

## 6. 验证门禁

**每个阶段结束时，以下全部必须绿**（来源：`.github/workflows/ci.yml`、`scripts/check_rust.py`）：

```bash
python scripts/check_rust.py        # cargo fmt --check + clippy -D warnings + check --all-targets + test --lib
cd rust && cargo test && cd ..      # 67 个 Rust 单测
cargo test --features gpu           # 你的 GPU 测试
ruff check src demos
pyright
pytest -q                           # 注意：见 §5.4，本机需 PYTHONUTF8=1
python scripts/phase0_baseline.py --check   # 架构/数值基线，应输出 "all scenarios bit-identical"
```

**特别重要**：`python scripts/phase0_baseline.py --check` 是**数值回归基线** —— 它验证所有场景仍然**逐位一致**。**只要这个还绿，就说明你没碰到 CPU 路径。** 这是你最重要的安全带。

---

## 7. 文件地图

### 7.1 你会改的

```
rust/Cargo.toml                      ← 加 cudarc 依赖，把 gpu feature 接上
rust/src/gpu/mod.rs                  ← 加新子模块
rust/src/gpu/probe.rs                ← 已存在（零依赖），P0-b 建议新建 cuda.rs 与之并列
rust/src/gpu/cuda.rs                 ← P0-b 新增：cudarc 绑定级探针
rust/tests/unit/gpu/*.rs             ← 对应测试
```

### 7.2 你要读但**不要改**的

```
rust/src/kernels/spatial.rs:76,100-138    ← 唯一并行点（rayon）；GPU 插入位置参照
rust/src/kernels/spatial.rs:115-118       ← sequential 谓词（守卫来源）
rust/src/kernels/age_structured.rs        ← 五阶段 tick：钩子→繁殖→存活→衰老
rust/src/kernels/offspring.rs:27          ← 后代张量 Z³（Z 小 → 非瓶颈）
rust/src/kernels/rng.rs:201-385           ← 现有采样器（GPU 需重写）
rust/src/kernels/state_reduce.rs:20       ← NumPy 成对求和复刻（精度动机）
rust/src/sessions/spatial.rs:61-88        ← ⭐ 堆叠状态布局 = GPU 数据布局模板
rust/src/sessions/spatial.rs:692          ← run_steps（分支点之一；需加 py 参数）
rust/src/sessions/age_structured.rs:480   ← run（已带 py: Python<'py>，无需改签名）
rust/src/sessions/age_structured.rs:1092  ← run_inner（分支点之一）
rust/src/hooks/interpreter.rs:922-924     ← n_hooks == 0 提前返回（无钩子模型零同步）
rust/src/output/history.rs                ← HistoryData（历史驻留参照）
rust/src/model/genetics.rs:37-45          ← GeneticsTensors 的 8 个张量
src/natal/contracts/*.py                  ← 边界契约
src/natal/backends/rust/rust_backend.py:873,1104  ← Python 侧后端适配（enable_gpu 透传点）
```

### 7.3 算法参考（demos 里的 PyTorch 实现）

`demos/gpu_spatial/` 下有一份**已验证可行**的 PyTorch 重写。**它的价值是"算法骨架 + 可行性证据"，不是"可直接搬的代码"。**

```
demos/gpu_spatial/age_structured/gpu_model.py:480   顺序条件二项 multinomial（→ 改成 §4.3 的方案）
demos/gpu_spatial/age_structured/gpu_model.py:461   _sample_binomial
demos/gpu_spatial/age_structured/gpu_model.py:318   _shift_grid（周期边界）
demos/gpu_spatial/age_structured/gpu_model.py:338   _migration_stencil
demos/gpu_spatial/age_structured/gpu_model.py:429   _mating_probability
demos/gpu_spatial/*/reference_cpu.py                用真实 natal 构建同一模型（可作测试夹具）
demos/gpu_spatial/*/compare_cpu_xpu.py              对照实验框架
```

**⚠️ 注意：demo 用的是 `torch` + `float32` + torch 自己的 RNG。这三样都**不能**搬进 Rust**（不能引入 torch 依赖；f32 可以但要走我们自己的 RNG）。

---

## 8. 遇到问题时

| 症状 | 先查 |
|---|---|
| `cargo build` 失败，说找不到 cudarc | 网络 / crates.io 索引；§3.2 步骤 1 的警告 |
| `cargo test --features gpu` 里 GPU 测试被跳过 | 忘了 `export NATAL_GPU_REQUIRE=1` |
| NVRTC 相关失败 | 容器内是否有 `libnvrtc.so`；**这是唯一可能推翻 D1 的情形**，若确认不可用要立刻上报 |
| `allow_threads` 编译不过 | 闭包捕获了 `Py<...>` 或非 `Send` 类型；见 §5.3 |
| 测试数量不对 | 默认 feature 下应始终是 **67** 个 |
| `phase0_baseline.py --check` 变红 | **立刻停下** —— 说明你碰到了 CPU 路径的数值行为 |
| 性能远低于预期 | 先记录同租户快照（§5.6），排除邻居干扰 |
| 不确定某个决定 | 看 §2.2 —— 七项决定**已冻结**；认为有错就**先提出**，不要擅自改 |

---

## 9. 工作原则（来自 `AGENTS.md`，摘要）

1. **默认用中文回答**；代码和 docstring 用英文，美式拼写。
2. **首次工作说明中给出风险分类**（文档修改 / 局部代码修改 / 高风险修改）和简短理由。
3. **高风险修改**（涉及科学计算公式、随机分布、状态恢复、可变数据共享、公开 API 合同、Python/Rust 数据交换）**必须由独立 evaluator 审查**。本项目的 GPU 工作**后期会进入高风险区间**（采样、RNG、状态），届时需要独立审查。
4. **未经明确要求不得 commit / push / 改 `.gitignore` / 新建 Markdown 文档。**
5. **交付说明必须包含**：变更文件、行为变化及原因、**实际执行的验证命令与结果**、残余风险。
6. **不得把已确认的基线失败说成"全部通过"**，也不得通过削弱断言、跳过或删除测试来制造通过结果。
7. **保留用户已有修改**；不要扩大修复范围。

---

## 10. 服务器端 clone（供参考）

分支 `feat/gpu-merge-test` 在 fork `KirschyR/natal-core` 上。

### 10.1 克隆指定分支

```bash
# -b 会同时把它设为初始检出，并【自动建立 tracking】
git clone -b feat/gpu-merge-test --single-branch \
    https://github.com/KirschyR/natal-core.git natal-core

cd natal-core

# 确认 tracking 正确：应显示 [origin/feat/gpu-merge-test]
git branch -vv
# * feat/gpu-merge-test 9b4d5b2 [origin/feat/gpu-merge-test] feat(gpu): add feature-gated CUDA environment probe (P0-a)

git log --oneline -1
# 9b4d5b2 feat(gpu): add feature-gated CUDA environment probe (P0-a)   ← P0-a 必须在这里
```

**说明：**

- `-b <branch>` **已经自动完成 tracking** —— 克隆出来的本地分支直接跟踪 `origin/feat/gpu-merge-test`，**不需要再 `git branch --set-upstream-to`**。这是 `git clone -b` 的标准行为，已实测确认。
- `--single-branch` 只拉该分支的历史，克隆更快更小。若之后需要别的分支：
  ```bash
  git remote set-branches origin '*' && git fetch origin
  ```
- 私有仓库需用 PAT 或 SSH：
  ```bash
  git clone -b feat/gpu-merge-test git@github.com:KirschyR/natal-core.git natal-core
  ```
- 想同时保留上游做参考：
  ```bash
  git remote add upstream https://github.com/jyzhu-pointless/natal-core.git
  git fetch upstream
  ```

### 10.2 ⚠️ 两个必须一起带过去的东西

clone **只会带来已提交的内容**。以下两项需要单独处理：

| 文件 | 问题 | 处理 |
|---|---|---|
| **`HANDOFF.md`**（本文档） | 必须**已提交并推送**，否则 clone 里没有它 | 见 §10.3 |
| **`GPU_insert_PLAN.md`** | **不在本仓库内**（它在开发机的工作区根目录），clone 永远带不来 | 手动 `scp` 到仓库根目录，作为**未跟踪的参考文件** |

```bash
# 从开发机把设计文档拷到服务器
scp GPU_insert_PLAN.md <server>:~/natal-core/
```

> 两者冲突时**以 `GPU_insert_PLAN.md` 为准**（它更详细）；但本文档 §2.2 的决定表是同步过的，可以直接用。

### 10.3 服务器上配置远端（如果要回推）

clone 出来的 `origin` 指向 fork，直接 push 即可：

```bash
git push origin feat/gpu-merge-test
```

若用的是 HTTPS 且仓库私有，需要配置凭证（PAT）。用 SSH 则无需额外配置。
