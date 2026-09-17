# GPU 后端插入方案 v2

> 把 `demos/gpu_spatial/` 中已验证的 GPU 算法，以**旁路（bypass）**形式融合进 `rust/` 引擎核心。
>
> **基线 revision**：`cd43ff9`（natal-core）
> **目标硬件**：服务器 RTX 5090 D V2（Blackwell / sm_120 / 21760 CUDA cores / 标称 **24 GB** GDDR7）
> **服务器实测（2026-09-17）**：驱动 595.84，CUDA 13.2，`24455 MiB` 总显存，**已用 19643 MiB → 仅约 4.7 GB 可用**
> **⚠️ 该 GPU 为多租户共享**，见 §D5 与风险 R12
> **本机**：Intel 核显（无 FP64、无 CUDA）→ 只能跑 CPU 路径
> **文档状态**：v2，七项决策已冻结（见 §2）
> **约定**：`[VERIFIED]` = 已在本仓库源码或公开规格中核对；`[UNVERIFIED]` = 待实测确认

---

## 1. v2 相对 v1 的修订摘要

| 项 | v1 | v2 | 原因 |
|---|---|---|---|
| GPU API | OpenCL | **CUDA（cudarc）** | 目标硬件从 Intel 核显改为 RTX 5090 |
| 精度 | f32 + 上限约束 | f32 + **树形归约强制** | 上限只是问题之一，累加误差更隐蔽 |
| 热点判断 | `Z³` 张量可能是杀手场景 | **撤回**；热点是**逐格点采样** | 用户确认 Z 很小（≤2 位点） |
| batch 轴优先级 | D 与 B 并列 | **D 与 B 为绝对主力**，Z 降级 | 同上 |
| RNG | 放弃跨引擎一致 | 追加：**确定性模式应可强验证** | 确定性模式不消耗 RNG 流 |
| 停止条件 | 需要在 exact 与 =N 间权衡 | **设备侧门控，零同步且语义一致** | 无钩子模型天然无此问题；有钩子时门控可搬到设备侧 |
| 历史驻留 | 需要环形缓冲 + 淘汰 | 诉求范围内**显存充足**，仅需预算核算 | T ≤ ~100 数量级 |
| 前端钩子 | 必须移植 CSR 解释器 | **不再必需**（典型模型无钩子） | HomingDrive 走张量路径 |
| 新增 | — | **布局转置** `(B,2,A,Z) → (2,A,Z,B)` | Z 小导致原始布局无法合并访存 |
| 新增 | — | **采样序列轴外提** | 满足"避免串行抽样"要求且不增工作量 |
| 改造风格 | trait 替换调度器 | **纯旁路 + 单一提前返回** | 用户要求最小增量、可加不可换 |
| 失败策略 | 静默回退 CPU | **显式报错，绝不静默降级；运行中绝不换引擎** | 用户裁定（§D5）：静默降级比报错危险得多 |

---

## 2. 七项决策（已冻结）

### D1 — GPU API：CUDA via `cudarc` ✅

**结论：用 CUDA，不用 OpenCL。** v1 推荐 OpenCL 的前提是"目标是 Intel 核显"；目标切换为 RTX 5090 后该前提消失。

| 方案 | 评价 |
|---|---|
| **`cudarc`** | **选定。** 安全/极简 CUDA 绑定；**动态加载 CUDA**（构建时不需装 CUDA toolkit）；支持 **NVRTC 运行时编译**（kernel 以字符串嵌入，运行时编成 sm_120 cubin）；CUDA 版本覆盖 11.4–11.8 / 12.0–12.9 / 13.0；[docs.rs](https://docs.rs/cudarc) |
| `cubecl` | 备选。多后端（CUDA / ROCm / **WGPU**），[tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)。**唯一能一套代码同时吃服务器 5090 和本机 Intel 核显的方案**，但抽象层更厚、对 CUDA 特性控制更弱 |
| `cust` | 不推荐。较老，构建期需要 CUDA toolkit |
| rust-cuda / cuda-oxide | 生态年轻（但已支持 sm_120） |

**为什么不用 OpenCL（明确理由）：**
- NVIDIA 的 OpenCL 实现长期停留在 3.0，性能显著低于 CUDA，且不跟进 Blackwell 新特性；
- 本机 Intel 核显跑不了 CUDA，但**本机本来就只能跑 CPU 路径**，不构成损失。

**⚠️ 关键前提**：动态加载 + NVRTC 之所以重要，是因为**本机没有 NVIDIA GPU**。若构建期强制链接 CUDA，本机将无法编译。cudarc 的动态加载避免了这一点 —— **本机可正常开发与编译，GPU 路径只在服务器上被启用。**

**✅ sm_120 支持已确认（2026-09-17 实测）：**

| 项 | 实测值 | 要求 | 判定 |
|---|---|---|---|
| 驱动版本 | **595.84** | — | — |
| CUDA（驱动支持上限） | **13.2** | ≥ 12.8（sm_120 最低） | ✅ 满足 |
| 次要版本兼容性 | CUDA 13.x 应用在驱动 ≥580 上可运行（[CUDA Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)） | ≥ 580 | ✅ 满足 |
| cudarc 支持范围 | `cuda-13010` … `cuda-13030`（[crates.io/cudarc](https://crates.io/crates/cudarc)） | 需覆盖 13.2 | ✅ **`-F cuda-13020` 直接对应** |

> **CUDA 版本不构成任何障碍。** v1 中"sm_120 需 12.8+"的顾虑已被实测排除。

**⚠️ 但容器内仍需确认 NVRTC 是否可用**（驱动支持 ≠ 容器内有 toolkit）：

```bash
ls /usr/local/cuda*/lib64/libnvrtc*     # NVRTC 运行时编译所需
nvcc --version                           # 可选（cudarc 走 NVRTC，不强制需要 nvcc）
python -c "import torch; print(torch.version.cuda)"   # 快速探知容器 toolkit 版本
```

---

### D2 — 精度：f32，且必须配树形归约 ✅

**物理限制（已检索确认）：**

> **RTX 5090 的 FP64 = 1.637 TFLOPS，是 FP32 的 1/64**
> 来源：[TechPowerUp RTX 5090](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216)、[rtx50series.co.uk](https://rtx50series.co.uk/rtx-5090-specifications/)

**裁决：用 f32，且不需要再纠结。** 理由：

1. 5090 的 f64 吞吐 ≈ 1.6 TFLOPS，**大约等于一颗 32 核 CPU** → 用 f64 上 GPU 等于没有加速。
2. f64 会让显存占用翻倍，而 5090 D V2 只有 **24 GB**（比标准 5090 的 32 GB 少，见 [TechPowerUp 5090 D V2](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090-d-v2.c4310)）。
3. 大 D 场景下**每格点计数天然很小**，f32 绰绰有余（见下）。

**但要把约束说准确 —— "仅仅是作上限限制"需要修正为两条：**

| # | 约束 | 说明 |
|---|---|---|
| a | **单格点值 < 2²⁴ = 16,777,216** | f32 的 24 位有效位决定。超过则整数本身失真 |
| b | **归约必须是树形/成对，禁止顺序累加** | **这条更隐蔽也更危险** |

关于 (b)：f32 有效位 24 bit。若某 deme 有 10⁴ 个体、跨 144 个格点顺序累加，误差可达 10⁻³ 量级 —— **对"计数"而言这是致命的**（计数本应是整数）。树形求和的误差是 `O(log n)` 而非 `O(n)`。

> **注意：现有 `kernels/state_reduce.rs:20` 复刻 NumPy 成对求和，其动机正是精度。GPU 路径必须沿用同样的树形结构 —— 既保精度，又使 GPU 内部结果可复现。**

**实测校验（验证 f32 足够）：**
- 大 D 场景：`D=10⁴`、总种群 `10⁸` → 每 deme `10⁴` → 每格点 `10⁴/(2·A·Z) ≈ 70`。**远低于 2²⁴。**
- 跨 batch 全局归约（extinction 检查）总和可达 `10⁸ > 2²⁴` → 由树形归约兜住，误差 `O(log n)`。

**守卫**：设备侧监视 `max_cell_value`；超过 `2²⁴ × 0.9` 时**发出 WARNING 并写入结果元数据**（不中止、也不切换引擎）。

> ⚠️ **不提供"运行中超限就切回 CPU"的选项** —— 理由同 §D5：CPU 与 GPU 是两个不同的随机系综，**运行中换引擎会产出两个引擎都无法复现的混合轨迹**。若需避免越界，应在 `run()` 之前减小规模，或改用 CPU 路径。

---

### D3 — 随机数：接受跨平台不可复现，但**确定性模式可强验证** ✅

用户已确认：跨平台同种子复现不是强限制。**这大幅简化了 D3，但同时解锁了一个更强的验证手段。**

**采用 counter-based RNG（Philox4x32-10）：**

```
随机数 = F(key, counter)
  key     = hash(seed, batch_index)
  counter = encode(tick, stage_id, cell_index)
```

- 无共享状态 → 天然并行；天然可复现（同配置 + 同种子 → 同结果）
- **每个 draw site 必须占用互不重叠的 counter 空间**（否则碰撞 → 隐性相关，需单元测试覆盖）
- 参考实现：PyTorch 在 GPU 上正是用 [Philox 做并行 RNG](https://blog.codingconfessions.com/p/how-pytorch-generates-random-numbers)；学术实现见 [OpenRAND](https://www.sciencedirect.com/science/article/pii/S2352711024001444)

**🔑 v2 新增的关键结论：**

> **确定性模式（`stochastic=False`）下不消耗任何 RNG 流** —— 繁殖走 `fixed_egg_count` 确定性路径，存活直接乘 `survival_rates`，迁移走 `migrate_csr_deterministic`（`kernels/spatial.rs:461`）。
>
> **因此确定性模式下 CPU 与 GPU 的差异只来自 f32 舍入，不来自 RNG。**

这提供了一个**远超"统计等价"的强验证手段**：

| 模式 | 验收标准 |
|---|---|
| 确定性 | **相对误差 ~ f32 eps（≈1.2×10⁻⁷）量级** —— 这是强断言，可用于定位实现错误 |
| 随机 | **统计等价**（KS 检验 / 卡方检验 / 矩检验）—— 不可用逐位比对 |

**必须文档化的后果：** GPU 与 CPU 在随机模式下产出**两个统计等价、逐位不同的系综**。GPU 结果是"另一个有效样本"，不是"同一个样本算得更快"。禁止用逐位比对作为随机模式的验收标准。

---

### D4 — Batch 轴：D 与 B 为绝对主力，Z 降级 ✅

用户确认：**B（batch）和 D（deme）是容易调大的参数；Z 不构成威胁**（实际测试不超过 2 个基因座、无大量等位基因 → 估计 `Z ≈ 9~16`）。

| 轴 | 含义 | 跨 batch 耦合 | 优先级 |
|---|---|---|---|
| **D**（deme） | 空间网格更细 | **有**（迁移 scatter） | **主力** |
| **B**（replicate） | 参数推断的并行模拟 | **无** | **主力** |
| ~~Z~~（基因型） | 位点更多 | 无 | **降级** —— 撤回 v1 的判断 |

**修订后的热点分析（这是 v2 最重要的修正）：**

既然 `Z ≈ 9~16`，那么：
- 后代张量 `Z³ ≤ 4096` 项 → **不是瓶颈**，撤回 v1 的"杀手场景"判断
- 收缩运算 `Z²G²` → **不是瓶颈**
- **真正的热点是「每格点一次随机抽样」**

规模估算（`B=10⁵, A=8, Z=9`）：

| 量 | 数值 |
|---|---|
| 每阶段格点数 | `B × 2 × A × Z = 1.44 × 10⁷` |
| 每 tick 五阶段 | `≈ 7 × 10⁷` 次操作 |
| `T=100` tick | `≈ 7 × 10⁹` 次操作 + **同等量级的随机抽样** |

> **结论：这份工作的主体是「在海量独立格点上做随机抽样与逐元素算术」。这是 GPU 最擅长、CPU 最吃力的形状。**
>
> **⚠️ 另一个推论：Z 小意味着每线程工作量小 → 瓶颈会从「算力」转向「内存带宽」。GPU 的带宽优势大，但前提是访存必须合并（见 §3.5 布局转置）。**

**用户实际需求（原文归纳）：**
1. **空间模型模拟** → 大 `D`
2. **参数推断中的大量模拟并行** → 大 `B`

**✅ 已确认（用户裁定）：暂不考虑 `D × B` 情形。**

即**不做"空间模型的参数推断"**（`B` 个 replicate × 每个 `D` 个 deme）。batch 轴因此**严格一维**：

| 场景 | batch 轴 | 耦合 |
|---|---|---|
| 空间模型模拟 | `D`（deme） | 有迁移耦合 |
| 非空间模型的参数推断 | `B`（replicate） | 无耦合 |

> **收益：状态量、显存预算、显存带宽需求都只按单轴增长，不按乘积增长。** 下方 §D5 的预算表与 §6 的风险 R6 据此收敛。

---

### D5 — 历史驻留：显存充足，但需按公式核算 ✅

**用户诉求**：模拟完成前不同步到 CPU；tick 数 ≤ ~100 数量级（不作硬编码）。

**预算公式（f32）：**
```
状态     = B × 2 × A × Z × 4 bytes
每 tick  = groups × out_d × 2 × out_a × 4 bytes      ← out_d = D（preserve）或 1（aggregate）
总历史   = T × 每 tick
```

**⚠️ 实测修正（2026-09-17）：服务器 GPU 是共享的，可用显存远低于标称 24 GB。**

`nvidia-smi` 实测：

```
NVIDIA GeForce RTX 5090 ...   24455MiB / 24455MiB   ← 标称 24 GB
Memory-Usage: 19643MiB / 24455MiB                  ← 已用 19.2 GB
GPU-Util: 52%   Pwr: 397W/600W   P1
Processes: No running processes found               ← 却看不到进程
```

> **可用显存 ≈ 24455 − 19643 = 4812 MiB ≈ 4.7 GB**（且可能被其他租户进一步挤占）。
> "有显存占用但看不到进程"是容器/多租户环境的典型现象（[NVIDIA 论坛](https://forums.developer.nvidia.com/t/11-gb-of-gpu-ram-used-and-no-process-listed-by-nvidia-smi/44459)、[GPU Operator #664](https://github.com/NVIDIA/gpu-operator/issues/664)）—— 说明**本容器之外有别的负载在用这张卡**。

**按 4.7 GB 可用重算（`B=10⁵, A=8, Z=9, T=100`）：**

| 项 | 计算 | 结果 | 4.7 GB 够用？ |
|---|---|---|---|
| 状态 `state_ind` | `10⁵×2×8×9×4` | **57.6 MB** | ✅ 毫无压力 |
| 历史（aggregate，deme 压成 1） | `100 × 1×1×2×8×4` | **6.4 KB** | ✅ |
| 历史（preserve 全 deme，`groups=1`） | `100 × 10⁵×2×8×4` | **640 MB** | ✅ |
| 历史（preserve，`groups=3`） | 同上 ×3 | **1.9 GB** | ✅ 但占 40% |
| 历史（**raw 全状态**） | `100 × 10⁵×2×8×9×4` | **5.76 GB** | ❌ **超出** |

> **修正结论：状态本身永远是小头（57.6 MB）；压力全部在历史。`raw` 模式在大 `B` 下已不可行。**
> v1 设想的"环形缓冲 + 淘汰"因此**重新变成必需**，但触发条件明确：**仅当所需历史超过实测可用显存时启用**。

**🔴 新增设计项：显存预算守卫 + 显式失败（不静默降级）**

原设计（"全程驻留、结束才下载"）必须改为**先算预算、再决定策略**。

**核心原则（用户裁定）：GPU 是显式启用的，所以失败也必须是显式的 —— 不允许静默回退 CPU。**

```
① 启动时 cudaMemGetInfo() 查询实测可用显存      ← 不假设 24 GB
② 由 (T, B, A, groups, out_d) 算出所需字节
③ 决策（仅两档成功路径）：
   - 需求 ≤ 可用 × 0.75  → 全程驻留（用户诉求 ⑤，零传输）★ 首选
   - 需求 >  可用 × 0.75  → 分段驻留：每 K 个 tick 下载一批并释放（**发警告**，见下）
④ 硬失败条件：
   - 连「状态」都放不下   → **ERROR，拒绝运行，不执行任何 tick**
   - 运行期分配失败        → **ERROR，中止，保留已完成结果，绝不换引擎**
```

**硬失败时的错误信息必须可操作**（而不是一个裸的 OOM）：

```python
GpuInsufficientMemory: GPU memory insufficient for the requested run.
  required_state  = 57.6 MB
  required_history= 5.76 GB   (T=100, B=100000, A=8, groups=9, raw=True)
  available       = 4.70 GB   (cudaMemGetInfo, 24455 MiB total, 19643 MiB held by other tenants)
  → 可选的缓解方式：
    a) 关闭 raw 模式：pop.observe(..., aggregate=True)     → 历史降至 640 MB
    b) 减小 T：pop.run(50) 分两次调用
    c) 减小 B
    d) 释放 GPU 后重试（当前有 19.2 GB 被其他租户占用）
```

**⚠️ 为什么运行中绝不能"回退 CPU"（这是比"不要静默"更强的理由）：**

> 由 §D3，**CPU 与 GPU 产出的是两个不同的随机系综**。若某次运行前半段在 GPU、后半段在 CPU，得到的轨迹**既不是 CPU 系综的样本，也不是 GPU 系综的样本**，且**两个引擎都无法复现它**。
>
> 这直接摧毁 D3 承诺的"GPU↔GPU 可复现"，产出一个**不可复现、也不属于任何已知分布**的结果。
>
> **因此：引擎切换只允许发生在任何 tick 执行之前。运行中一旦失败，只能中止 —— 这不是性能选择，是正确性问题。**

**失败分类总表：**

| 时机 | 条件 | 行为 | 理由 |
|---|---|---|---|
| `run()` 入口 | 状态装不下 | **ERROR，不运行** | ★ 用户裁定 |
| `run()` 入口 | 历史装不下、状态放得下 | **分段驻留 + WARNING** | 语义不变，仅传输策略改变 |
| `run()` 入口 | 模型不合格（见 §3.3） | **ERROR，不运行** | ★ 用户裁定（同原则） |
| 运行中 | 分配失败 | **ERROR，中止 + 保留结果** | ★ 不得中途换引擎（见上） |
| 运行中 | 计数超 2²⁴ | **WARNING + 记入结果元数据** | 中止会丢失全部计算；警告即非静默 |

**关于第 2 行（分段驻留）**：它**不是**引擎切换，只是"分批下载 + 释放"，语义与全程驻留**完全相同**，因此允许。但因为它违背了用户诉求 ⑤（"模拟完成前不同步"），**必须发出警告并写入结果元数据**，让用户知道这次运行发生了分段。

**关于最后一行**：精度告警走 `warn` 而非中止 —— 中止一次大 `B` 长跑以换取一个"计数越界"的告警并不划算，且**警告本身就不是静默**，符合原则。

以 `B=10⁵, A=8, Z=9, T=100`、可用 4.7 GB 为例：**`groups ≤ 5` 走全程驻留；`raw` 模式（5.76 GB）直接 ERROR。**

**⚠️ 一条警戒线：**

1. ~~`B × D` 二维 batch 按乘积增长~~ → **已排除**（用户裁定不做空间模型的参数推断），batch 轴严格一维，故不适用。
2. **`T` 不要写死** —— 方案按"预算驱动 + 分段下载"设计，`T` 增大时通过 `Observation` 调节杆降维，而非改架构。

> 单轴下的显存上界：`T × B × 2 × A × groups × 4`。
> **务必按实测可用显存核算，而非按 24455 MiB 标称值。**

**落地形态**：设备侧维护 `(n_rows, width)` f32 缓冲，`HistoryStore::query()` 时才下载。`Observation` 的 `deme_mode` / `collapse_age` / `mask`（`src/natal/frontend/output/observation.py:157-161`）就是现成的调节杆，**不需要新机制**。

---

### D6 — 停止条件：机制解释 + 设备侧门控方案 ✅

#### 停止条件是什么？

**它是"自动提前结束模拟"的规则。**

基因驱动模拟的典型场景：释放一批带 drive 的个体，观察 drive 能否扩散或压垮种群。很多模拟会**提前结束**：

- 种群灭绝了 → 继续跑全是 0，无意义
- drive 等位基因消失了（计数 = 0）→ 继续跑无意义
- drive 固定了（频率 = 1）→ 后续是确定性的，没必要跑

若不处理，跑 `10⁵` 个 replicate × 100 tick 时，**大部分算力浪费在"已经结束"的模拟上**。

引擎因此提供 `stop_if_*` 规则（`src/natal/frontend/hooks/entry/declarative.py:199-272`）：

```python
nt.stop_if_zero(genotypes=..., when="early")     # 计数归零则停
nt.stop_if_extinction(when="late")               # 灭绝则停
nt.stop_if_above(threshold=0.99, ...)            # 固定则停
```

对应 Rust 侧 opcode（`hooks/interpreter.rs:29-32`）：

```rust
const OP_STOP_IF_ZERO: i64 = 6;
const OP_STOP_IF_BELOW: i64 = 7;
const OP_STOP_IF_ABOVE: i64 = 8;
const OP_STOP_IF_EXTINCTION: i64 = 9;
```

#### 为什么它曾经是个问题？

因为检查必须**读取状态**（`hooks/interpreter.rs:1189-1191`）：

```rust
} else if op_type == OP_STOP_IF_EXTINCTION
    && individual_count.iter().sum::<f64>() <= 0.0 {
    return Ok(RESULT_STOP);
}
```

在 GPU 上，状态在显存里。判断"总和是否 ≤ 0"需要：设备侧全局归约 → 传回 CPU → CPU 判断。**每次都是一次"设备→主机同步"，会打断 GPU 流水线**，与 §0 目标"逐 tick 不回传"直接冲突。

**两难**：每 tick 检查（语义一致但慢）vs 每 N tick 检查（快但**停止点延后 → 结果改变**）。

#### 情形一：模型无钩子 → 完全不存在这个问题 `[VERIFIED]`

**关键机制：`stop_if_*` 不是独立功能，它是钩子程序里的一个 opcode。**

```rust
// src/natal/frontend/hooks/types.py:75-78
STOP_IF_ZERO = 6;  STOP_IF_BELOW = 7;  STOP_IF_ABOVE = 8;  STOP_IF_EXTINCTION = 9
```

它与 `SCALE(0) / SET(1) / ADD(2) / KILL(4) / SAMPLE(5)` 同属一个 opcode 家族，**都活在 `HookProgram` 的 CSR 字节码里**。所以「用了 `stop_if_*`」必然意味着「模型有钩子」。

**而空钩子程序有提前返回：**

```rust
// rust/src/hooks/interpreter.rs:922-924
if event_id < 0 || event_id >= self.n_events || self.n_hooks == 0 {
    return Ok(RESULT_CONTINUE);      // ← 立即返回，一个状态字节都不读
}
```

**HomingDrive 编译成遗传张量，不是钩子：**

```
src/natal/frontend/presets/homing.py:252  gamete_modifier  → 编译成减数分裂/融合张量
src/natal/frontend/presets/homing.py:343  zygote_modifier  → 编译成合子转换张量
src/natal/frontend/presets/homing.py:180  fitness_patch    → 编译成适配度张量
```

**因果链：**

```
HomingDrive → 编译成张量（而非钩子）
            → HookProgram 为空（n_hooks == 0）
            → execute_event 在 4 行内立即返回
            → 零状态读取 → 零归约 → 零同步点
```

> **表述要精确：不是「HomingDrive 特殊」，而是「凡遗传机制编译成张量的模型都没有钩子」。** HomingDrive 恰好属于这类，且是主流用法，所以典型场景天然干净。

**全仓 grep 确认 `[VERIFIED]`：`stop_if_*` 只由用户 API 产生**（`hooks/entry/declarative.py:216,238,260,272`），其余出现处只有类型定义、UI 显示名与测试。**没有任何 preset 或 builder 会自动生成它。**

**补充一条容易混淆的区分：有钩子 ≠ 有同步。**

| 钩子 opcode | 行为 | 是否需要同步 |
|---|---|---|
| SCALE / SET / ADD / SUBTRACT / KILL | 就地修改选中格点 | ❌ 否 |
| SAMPLE / CONVERT | 就地抽样 / 格点间搬移 | ❌ 否 |
| SET_PARAM | 写生态列 + 记 journal | ❌ 否（journal 每次 `run()` 结束才 drain） |
| **STOP_IF_ZERO / BELOW / ABOVE / EXTINCTION** | **对状态做归约** | ✅ **是** |

**只有这四个需要归约。**

#### 情形二：模型含 `stop_if_*` → 需要归约，但**不需要每 tick 主机同步**

**使用场景**：用户显式声明，典型动机正是**大 `B` 系综/参数推断** —— 让已经灭绝或已经固定的 replicate 提前退出以省算力。也就是说：**典型模型没有它，但一旦做你们的主要场景，很可能就会加上。** 所以必须处理。

**为什么它曾经看着需要同步 —— 两个原因叠加：**

**(1) 条件本身是一次归约：**

```rust
// rust/src/hooks/interpreter.rs:1151-1191
if (OP_STOP_IF_ZERO..=OP_STOP_IF_ABOVE).contains(&op_type) {
    // 把选中的 (sex, age, ztype) 格点全部累加成 selected_total
    ...
    if op_type == OP_STOP_IF_ZERO && selected_total <= 0.0 { return Ok(RESULT_STOP); }
} else if op_type == OP_STOP_IF_EXTINCTION
    && individual_count.iter().sum::<f64>() <= 0.0 {      // 整个状态数组求和
    return Ok(RESULT_STOP);
}
```

CPU 上状态在内存里，求和几乎免费。GPU 上状态在显存里 —— **但设备内归约同样便宜**。真正贵的是下一步。

**(2) 循环控制权在 CPU 手里 —— 这才是同步的真正来源：**

```rust
// rust/src/sessions/age_structured.rs:1199-1220
if step == n_ticks.max(0) || stopped { break; }     // ← stopped 是 CPU 变量
let result = run_tick(...)?;
if result != 0 { stopped = true; } else { current_tick += 1; }
```

`stopped` 是 CPU 上的一个布尔量。要设置它，CPU 必须先拿到设备算出的那个标量：

```
cudaMemcpy(&host_val, device_ptr, ..., DeviceToHost)
```

**这个调用会强制整个设备队列排空** —— 所有已入队 kernel 必须先跑完，拷贝才能返回有效值。**代价不是那 4 个字节，而是 GPU 停等、CPU 空转、流水线重叠完全丧失。**

**空间路径还有一层全局语义**（`sessions/spatial.rs:699-707`）：用"**tick 有没有推进**"判定停止（`state_tick == previous` → 返回 `stopped=true`）。因此**任一 deme 停止 → 整个 session 停止**。GPU 要复现它，需要一次**全局 OR 归约**。

#### ✅ v2 修正：同步可以完全避免

**关键洞察：CPU 其实不需要知道。**

同步之所以必要，**只是因为循环控制被留在了 CPU 上**。把门控搬到设备侧，同步就消失了：

```
设备侧持有：stop_flags[B]
每个 tick：
  ① 设备内评估停止条件 → 写 stop_flags          （设备内归约，不涉及主机）
  ② 设备内全局 OR       → global_stop           （设备内归约，不涉及主机）
  ③ tick kernel 开头检查 global_stop，为真则直接 return（空操作）
主机侧：无条件跑满 n 次循环，结束时下载一次 stop_flags
```

**这完全复现 CPU 语义**（任一元素停止 → 全体停止 + 冻结 tick），**且零 per-tick 主机同步**。

**一个必须对齐的细节**：CPU 循环在停止时 `current_tick` **不递增**，下一轮先 record（记录同一个 tick）再 break。GPU 侧同样处理 —— **record 无条件执行，只门控 tick 本身** → 历史行完全一致。

**代价**：停止后仍继续启动剩余的空 kernel。以 `T ≤ 100` 计：`≤100 次空启动 × ~5µs ≈ 0.5 ms`，可忽略。

#### D6 最终裁决

| 实现方式 | 语义 | per-tick 主机同步 |
|---|---|---|
| 朴素（循环控制留在 CPU） | 一致 | ❌ 每 tick 一次 |
| **设备侧门控（选定）** | **完全一致** | ✅ **零** |
| 每 N tick 检查 | 停止点延后 → **结果改变** | 少量（**不需要**） |

> **裁决：不做 `stop_check` 的 exact / =N 权衡 —— 那个设计是多余的。采用设备侧门控，默认即零同步、且语义与 CPU 完全一致。**
>
> 保留周期检查的唯一价值：若停止后剩余 tick 极多（如 `T=10000` 而第 10 tick 就停），空启动的累积开销才值得优化。`T ≤ 100` 时不需要。
>
> **结论：D6 从"需要在正确性与性能间权衡"降级为"按设备侧门控实现，两边同时满足"。**

---

### D7 — GIL 释放：优化方向 ✅

**现状 `[VERIFIED]`：全树零 `Python::allow_threads`。** 批量循环全程持有 GIL。

#### 三层优化方向

**层 1：释放 GIL（基础，必做）**

```rust
py.allow_threads(|| { /* 整个 batch loop */ })
```

收益：其他 Python 线程可运行。**但对单次 `pop.run(n)` 本身没有加速** —— `allow_threads` 只是"不再独占解释器"，PyO3 文档明确：

> *"`Python::allow_threads` just lets other Python threads run - it does not itself launch a new thread"*（`pyo3-0.23.5/src/marker.rs:37`）

**层 2：CPU 与 GPU 真正重叠（关键收益）**

- kernel 提交做成异步（CUDA stream）
- CPU 在 GPU 算 tick *k* 时，准备 tick *k+1* 的元数据
- 双缓冲（double buffering）
- **收益：总时间 → `max(CPU, GPU)` 而非 `CPU + GPU`**

**层 3：多 stream 并行（最大收益）**

- `B` 很大时，用多个 CUDA stream 并发跑 batch 分片
- 5090 有足够 SM 与显存吃下并发 kernel
- 可由多个 Python 线程各驱动一个 stream —— **此时层 1 的 GIL 释放是前提条件**
- **收益：提升 occupancy，隐藏内存延迟**

#### 代码形态

```rust
fn run(&mut self, py: Python<'_>, n_ticks: i64) -> PyResult<...> {
    #[cfg(feature = "gpu")]
    if self.device_eligible() {
        // 状态已在显存，整个循环不碰 Python 对象
        let out = py.allow_threads(|| self.gpu.as_mut().unwrap().run_ticks(n_ticks))?;
        self.download_history();          // 结束后一次性下载
        return Ok(out);
    }
    // ↓ 原 CPU 路径，一字不改
    ...
}
```

#### ⚠️ 实现陷阱（已在 PyO3 0.23.5 源码核实）

```rust
// pyo3-0.23.5/src/marker.rs:191-193  (非 nightly)
pub unsafe trait Ungil {}
unsafe impl<T: Send> Ungil for T {}     // ← 所有 Send 类型自动满足
```

`Py<PyAny>` **是 `Send`** → **编译器不会阻止你在闭包里访问 Python 对象**。

PyO3 自己把它写成 Safety 契约而非编译期保证（`marker.rs:140-143`）：

> *"The type must not carry borrowed Python references or, if it does, not allow access to them if the GIL is not held."*

**违反 = 未定义行为**，可能崩溃也可能静默错误，且因为在 stable 上编译器不拦，可能数月后才暴露。

**对策：用类型设计堵死，而非靠"记得别写错"。** 设备路径所需的一切数据放入**不含任何 `Py<...>` 字段**的 struct（见 §3.4 `BatchView`），并加显式注释说明该契约。

#### 另一个实现约束

`py.allow_threads(|| &mut self.gpu...)` 要求闭包捕获类型满足 `Send`。因此 **`GpuExecutor` 必须 `Send`**。`[UNVERIFIED]` cudarc 的设备/上下文句柄是否满足 `Send`，需在 P0 确认；若不满足，需要用 `Mutex` 包装或在工作线程内持有上下文。

---

## 3. 架构改造：最小增量旁路

**设计原则（按用户要求）：CPU 路径一行不改；GPU 只做旁路（bypass）；可加不可换。**

### 3.1 改动清单（全部为「增加」）

| # | 文件 | 改动 | 性质 |
|---|---|---|---|
| 1 | `rust/Cargo.toml` | 新增 `[features] gpu = ["dep:cudarc"]` + optional 依赖 | 增加（**默认关闭**） |
| 2 | `rust/src/gpu/**` | **全新模块树**（见 §3.2） | 增加 |
| 3 | `rust/src/lib.rs` | `#[cfg(feature="gpu")] mod gpu;` → **1 行** | 增加 |
| 4 | `rust/src/sessions/{spatial,age_structured}.rs` | 新增字段 `gpu: Option<GpuExecutor>`（`#[cfg]` 包裹）；`#[new]` 中初始化为 `None` | 增加 |
| 5 | 同上 | 新增方法 `enable_gpu(...)` / `gpu_status()` | 增加 |
| 6 | `run_inner` / `run_steps` 开头 | **唯一的分支点**：`if device_eligible() { ...; return ...; }` | **提前返回，不改原路径** |
| 7 | `src/natal/backends/rust/rust_backend.py` | `enable_gpu()` 透传 | 增加 |

> **改动 6 是整个方案中唯一触碰现有逻辑的地方，且形式是"提前返回"，原有代码路径零修改。**
>
> **feature 默认关闭 → GPU 代码编译失败不影响 CPU 构建 → 本机（无 NVIDIA）可正常开发。**

### 3.2 新增模块树

```
rust/src/gpu/
  mod.rs          feature-gated 导出、能力探测、可用性查询
  context.rs      CUDA 上下文/stream 管理（动态加载 + NVRTC）
  buffers.rs      设备缓冲 RAII 封装
  layout.rs       ★ (B,2,A,Z) <-> (2,A,Z,B) 转置（见 §3.5）
  rng.rs          ★ Philox4x32-10 counter-based RNG
  sampling.rs     ★ BTPE/PTRS/Marsaglia-Tsang + 序列轴外提（见 §4.2）
  kernels/        .cu 源码字符串（NVRTC 运行时编译）
  tick.rs         ★ 设备侧 tick 编排（五阶段）
  history.rs      设备侧历史驻留缓冲
  executor.rs     ★ GpuExecutor，实现 §3.4 的旁路接口

  # ★ = 本方案的核心工作量
```

### 3.3 守卫条件（是否走 GPU）

**复用已有的谓词并扩展**（`rust/src/kernels/spatial.rs:115-118`）：

```rust
fn device_eligible(&self) -> bool {
    self.gpu.is_some()                                  // 用户已启用
    && !self.hooks.python_callbacks.iter().any(|c| !c.is_empty())  // 复用现有判定
    && self.max_cell_value() < 16_777_216.0 * 0.9       // §D2 精度守卫
    && self.params.growth_mode < 5                      // 无自定义密度曲线
    // 含 stop_if_* 时改用设备侧门控（§D6），仍满足零主机同步，无需排除
}
```

> **一个漂亮的巧合**：`spatial.rs:115-118` 的 `sequential` 谓词（"有 Python 回调就退回串行"）当初是为了回避数据竞争，现在**恰好同时决定了"能不能并行"和"能不能上 GPU"**。不需要新造机制。

**不满足 → 默认 ERROR，拒绝运行**（用户裁定：GPU 是显式启用的，失败也必须显式）。

```python
# 默认：严格要求 GPU，任一条件不满足即报错，不执行任何 tick
pop.enable_gpu()

# 显式选择允许回退（用户主动声明，不属于"静默"）
pop.enable_gpu(fallback="cpu")      # 回退时发出 WARNING，并写入结果元数据
```

理由与 §D5 完全一致：**用户显式要了 GPU，静默给 CPU 会让一次 `10⁵` 规模的扫描在用户不知情的情况下慢几十倍。** 「静默降级」比「报错」危险得多 —— 报错会立刻被发现，静默降级不会。

失败信息同样必须可操作：

```python
GpuNotEligible: model is not GPU-eligible.
  reason = python_callbacks_present   (3 hook(s) carry Python callbacks)
  → 可选：pop.enable_gpu(fallback="cpu") 明确允许回退，或移除回调钩子
```

若确实需要探测当前实际执行器，用 `gpu_status()` 查询。

### 3.4 旁路接口（刻意不含 `Py`，见 D7 陷阱）

```rust
/// 一批相互独立的模型实例（空间模型里是 deme，多样本批里是 replicate）。
/// 刻意不含任何 Py<...> 字段 —— 满足 allow_threads 的 Ungil 契约。
pub struct BatchView<'a> {
    pub n_batch: usize,
    pub ind: &'a mut [f32],      // 设备侧，已转置为 (2, A, Z, B)
    pub sperm: &'a mut [f32],
    pub eco: &'a mut [f32],
}

pub struct GpuExecutor { /* CUDA context, buffers, rng state */ }

impl GpuExecutor {
    pub fn upload_initial(&mut self, bp: &Blueprint, st: &SessionState) -> Result<(), String>;
    pub fn run_ticks(&mut self, n: i64) -> Result<(i64, bool), String>;   // 全程零回传
    pub fn download_history(&mut self) -> Result<Vec<f32>, String>;        // 仅一次
    pub fn download_state(&mut self) -> Result<SessionState, String>;
}
```

### 3.5 ★ 布局转置（v2 新增，必需）

**问题：CPU 侧布局对 GPU 访存是反优化的。**

CPU 布局 `(B, 2, A, Z)` 是为「每 batch 元素一块连续内存」设计的（`spatial.rs:100` 的 `chunks_mut`）。线程若按 batch 索引，线程 *t* 访问偏移 `t × 2 × A × Z`，**步长 = 2·A·Z**。

以 `A=8, Z=9` 计，步长 = **144 个 f32** → **完全无法合并访存**，是 GPU 上最经典的性能杀手。

**解法：设备侧改用 `(2, A, Z, B)`** —— 把 `B` 放到最后一维（stride = 1）：

```
CPU 布局  (B, 2, A, Z)   →  设备布局  (2, A, Z, B)
            ↑ batch 轴在最外                          ↑ batch 轴在最内
```

于是相邻线程（处理相邻 batch 元素）访问相邻内存 → **完美合并（coalesced）**。

> **关键：转置只发生在上传/下载边界，CPU 侧布局与全部现有代码不受影响。** 这是"最小增量"原则的直接体现。

### 3.6 数据流（设备模式）

```
Python: pop.run(n)
   │
   ├─ device_eligible()? ──否──► 原 CPU 路径（一字不改）
   │                              是
   ▼
py.allow_threads(|| {                      ← D7
    upload_initial()  + 布局转置           ← 仅一次
    for tick in 0..n {
        run_device_tick()                  ← 全程零回传
        device_record_history(tick)        ← 写设备侧缓冲（D5）
    }                                      ← 无钩子时零同步（D6）
    download_history()  + 转置回          ← 仅一次，含全部中间记录
})
```

---

## 4. 内核移植清单与采样方法

### 4.1 移植优先级（按 D4 修订）

| 优先级 | 内核 | 现有位置 | 规模 | 难度 |
|---|---|---|---|---|
| **P0** | **逐格点采样：binomial / poisson / multinomial** | `kernels/rng.rs:201-385` | `B×2AZ` 次/阶段 | **高**（§4.2） |
| **P0** | 存活 / 衰老 / 密度调节 | `kernels/age_structured.rs:869/927`、`density_regulation.rs` | `B×2AZ` | 低 |
| **P0** | 求偶概率矩阵 | `kernels/age_structured.rs` (~`:133`) | `B×Z×Z` | 低 |
| **P1** | 繁殖管线的确定性部分 | `kernels/age_structured.rs:133-412` | `B×2AZ` | 中 |
| **P1** | 迁移 CSR scatter/gather | `kernels/spatial.rs:461/630/791` | `B×nnz×2AZ` | 中 |
| **P2** | 状态归约（**树形**） | `kernels/state_reduce.rs` | — | 低 |
| **P2** | 观测投影 | `output/observation.rs:11` | — | 低 |
| **降级** | 后代张量 `Z³` | `kernels/offspring.rs:27` | `Z³ ≤ 4096` | 低（**非瓶颈**） |
| **不移植** | CSR 钩子解释器 | `hooks/interpreter.rs:896` | — | 典型模型无钩子（§D6） |
| **不移植** | 历史 / ParameterLog | `output/*.rs` | — | 留 CPU |

### 4.2 ★ 采样方法：序列轴外提（满足"避免串行抽样"）

用户要求"研究优化的随机抽样方法，避免条件性或序列性的随机方法"。**核验后得到一个比"全部改成无分支算法"更好的答案。**

#### 关键区分：串行依赖在哪个轴上？

**纠正一个常见误解**：`multinomial` 的"顺序条件二项"法，其依赖链长度是 **类别数 `K-1`**，**不是个体数 `n`**：

```rust
// kernels/rng.rs:345  multinomial 的本质结构
for k in 0..K-1 {
    draws[k] = binomial(remaining, p_k / p_tail);   // remaining 依赖前一步
    remaining -= draws[k];
}
```

因为 D4 确认 `Z ≈ 9~16`，所以**链长仅 14~15**，而 `n`（每格点计数）可能远大于此。

#### 三种方案的对比

| 方案 | 并行性 | 工作量 | 适用 |
|---|---|---|---|
| A. 顺序条件二项（现状） | ❌ 线程内 15 步串行链 + 拒绝采样 → **warp 发散** | `O(K)` | — |
| B. 逐个体逆变换分桶 | ✅ 完全并行 | **`O(n)` 次抽样** | `n` 小时优 |
| **C. 序列轴外提（推荐）** | ✅ **完全并行** | **`O(K)`，与 A 相同** | **本场景最优** |

**方案 C：把序列轴从"线程内层"提到"外层循环"，内层对全部格点并行。**

```cuda
// 一个 kernel，外层循环 k，内层对全部 batch 单元并行
__global__ void multinomial(const float* p, float* remaining, float* out, int K, RngState rng) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;   // 全部线程同时处理各自的格点
    for (int k = 0; k < K - 1; ++k) {                   // ← 所有线程同步推进同一个 k
        out[cell * K + k] = binomial(&rng, remaining[cell], p[k] / tail(p, k));
        remaining[cell] -= out[cell * K + k];
    }
}
```

**为什么这是最优解：**
1. **零 warp 发散** —— 同一 warp 内所有线程处于同一个 `k`，走相同的代码路径（对比方案 A：线程各自在 15 步链上进度不同 → 严重发散）
2. **工作量与现状相同**（`O(K)` 而非方案 B 的 `O(n)`）
3. **完全精确**，非近似
4. **单个 kernel 启动**（循环内联，无需 `K-1` 次启动）

> **这个"序列轴外提"的模式是通用的** —— 繁殖管线里所有"条件性"结构（求偶、储精置换、受精分配）都可以用同样手法改造：**把原本在线程内的串行链，提到所有线程共同推进的外层循环。**

#### 基础分布采样器的选择（各线程独立推进，无跨线程依赖）

| 分布 | 推荐算法 | 并行性 | 备注 |
|---|---|---|---|
| **Binomial** | **BTPE**（Kachitvichyanukul-Schmeiser） | ✅ 每线程独立拒绝采样 | 见 [BTPE 平均运行时间分析](https://arxiv.org/html/2403.11018v1) |
| **Poisson** | **PTRS**（Hörmann 变换拒绝） | ✅ 同上 | 大 λ 时用正态近似分支 |
| **Gamma** | **Marsaglia-Tsang** | ✅ 同上 | 无跨线程依赖 |
| **Multinomial** | **方案 C**（序列轴外提） | ✅ | 见上 |

**注意**：BTPE / PTRS / Marsaglia-Tsang 都是"条件性"的（拒绝采样循环），但**每个线程独立推进自己的循环，不存在跨线程依赖** —— 这才是 GPU 友好与不友好的真正分界。真正需要改造的只有 `multinomial`。

`[UNVERIFIED]` 需实测：拒绝采样的 warp 发散在真实分布参数下的实际代价，以及分支预测优化空间。

#### 验证要求

- 确定性模式：**CPU↔GPU 相对误差 ~ f32 eps**（D3 强断言）
- 随机模式：**逐分布统计检验**（KS / 卡方 / 前四阶矩），阈值需预先设定并写入测试
- **counter 空间碰撞测试**：确认各 draw site 的 counter 不重叠

---

## 5. 分阶段实施

### 5.0 前置：基线与不可破坏的约束

**开始编辑 natal-core 之前必须先完成：**

```bash
git switch -c feat/gpu-bypass

# 记录基线（全部必须通过）
python scripts/check_rust.py       # cargo fmt --check + clippy -D warnings + check --all-targets
cd rust && cargo test && cd ..     # rust/tests/unit/ 约 67 个测试
ruff check src demos
pyright
pytest -q
python scripts/phase0_baseline.py --check
```

| Gate | 命令 | 出处 |
|---|---|---|
| Rust 硬闸 | `python scripts/check_rust.py` | `scripts/check_rust.py:1-16` |
| Rust 单测 | `cargo test`（在 `rust/`） | `rust/tests/unit/` |
| Python lint | `ruff check src demos` | `.github/workflows/ci.yml:31` |
| Python 类型 | `pyright`（strict） | `ci.yml:33` |
| Python 测试 | `pytest -q` | `ci.yml:58` |
| 架构基线 | `python scripts/phase0_baseline.py --check` | `ci.yml:75` |

> **每个阶段结束时以上全部必须绿。**

**不可触碰清单（整个 GPU 工作期间）：**

| ❌ 禁止 | 理由 |
|---|---|
| 修改任何现有 `kernels/*.rs` 的数值逻辑 | CPU 是唯一 golden reference（风险 R10） |
| 修改 `contracts/`（Python 与 Rust 两侧） | 契约是既定边界，GPU 复用而非改造 |
| 删除或替换 CPU 路径任何代码 | 同上 |
| 让 `gpu` feature 默认开启 | 本机无 NVIDIA，必须能默认构建 |
| 在 P3 之前写采样内核 | 先打通管道，再引入 RNG 复杂度 |

---

### 5.1 验证层级 L0–L3

**分层的理由：不要等到"整条 tick 跑通"才发现某个 kernel 写错了。**

| 层级 | 比较对象 | 何时可用 | 定位能力 |
|---|---|---|---|
| **L0** | 无（工具链自证） | P0 | 证明 CUDA 能用 |
| **L1** | 设备 kernel 输出 vs CPU kernel 输出（**同一份输入**） | **每写完一个 kernel 即可** | 单个 kernel 的错误 |
| **L2** | 完整设备 tick vs 完整 CPU tick（确定性模式） | P3 之后 | kernel 之间的装配错误 |
| **L3** | `pop.run(n)` 端到端 vs CPU 端到端 | P3 之后 | 传输 / 生命周期错误 |

> **L1 是本项目最重要的开发手段** —— 它让你在只写完一个 kernel 时就能验证它，不必等待整条链路。
> **要求：每个新增 GPU kernel 都必须配一个 L1 对照测试。**

---

### 5.2 ⚠️ 精度验收标准（分档，必须精确执行）

**CPU 是 f64，GPU 是 f32。所以"逐位一致"只在特定条件下成立 —— 用错标准会永远失败，并掩盖真正的 bug。**

| 场景 | 验收标准 | 依据 |
|---|---|---|
| **纯搬运 kernel**（衰老/移位）+ 整数计数 < 2²⁴ | **逐位一致** | 整数在 f32 中精确可表示，且无算术 |
| **含算术 kernel** + 确定性模式 | **相对误差 < 10 × f32 eps（≈1.2×10⁻⁶）** | f64→f32 舍入 + 运算顺序差异 |
| **随机模式** | **统计等价**（KS / 卡方 / 前四阶矩） | 两个不同系综（§D3） |

**两条硬规则：**

1. **禁止用"逐位比对"验收含算术的 kernel。** 那必然失败，且失败信息没有诊断价值。改用"相对误差是否落在 f32 理论界内"。
2. **随机模式禁止逐位比对**，必须走统计检验。

> 注：实测参考 —— demo 在 f32 下 `max_rel_diff = 7.3×10⁻⁷`（`gpu_spatial/age_structured/outputs/xpu_comparison_summary.json`），与 f32 eps 同量级，可作为量级校准。

---

### 5.3 阶段表

**首个目标锁定「空间模型（大 D）」** —— 因为 batch 轴（deme）**已经存在**：
`rust/src/kernels/spatial.rs:100` 的 `ind_all.chunks_mut(ind_stride)` 已按 deme 切分，`rngs: Vec<SessionRng>` 已每 deme 一条流。
`(B,2,A,Z)` 堆叠布局**就是** GPU 需要的 batched 形状，无需新建概念。

| 阶段 | 目标 | 交付物 | 验收门 |
|---|---|---|---|
| **P0** | 证明工具链可用（**不碰 natal 任何代码**） | `rust/src/gpu/probe.rs` | 见 P0 细则 |
| **P1** | 骨架 + 布局转置（**仍不改任何 tick 逻辑**） | `rust/src/gpu/{mod,context,buffers,layout}.rs` + `Cargo.toml` feature | L1 round-trip + 两种构建都通过 |
| **P2** | 单内核贯通端到端 + 预算守卫 | 衰老 kernel + `gpu` 字段 + `enable_gpu()` + 分支点 + §D5 显式失败 | **L3 端到端** |
| **P3** | 确定性数值内核 + 停止门控 | 存活/密度/求偶/观测投影/树形归约 + §D6 设备侧停止门控 | L2 + L3 |
| **P4** | 采样内核 + RNG | Philox + BTPE/PTRS/Marsaglia-Tsang + **序列轴外提** | 统计检验 + counter 无碰撞 |
| **P5** | 迁移 + 大 D 场景 | CSR scatter/gather（`spatial.rs:461/630/791` 三变体） | 大 D benchmark |
| **P6** | 多 B（ensemble session） | `EnsembleEngineSession` | vs `ProcessPoolExecutor` |

#### P0 细则 —— 工具链探针

**拆成两步**（因为开发机无网络、registry 缓存无 `cudarc`，一旦写入 `Cargo.toml` 连默认构建都会失败）：

| 步骤 | 内容 | 状态 |
|---|---|---|
| **P0-a** | **零依赖**环境探针：`nvidia-smi` + toolkit 目录扫描 | ✅ **已完成**（分支 `feat/gpu-merge-test`） |
| **P0-b** | 接入 `cudarc`，跑绑定级断言（动态加载 + NVRTC 实编译） | ✅ **已完成** |

**P0-a 交付物**：

| 文件 | 行数 | 内容 |
|---|---|---|
| `rust/src/gpu/mod.rs` | 17 | feature-gated 模块声明 + 分阶段说明 |
| `rust/src/gpu/probe.rs` | 375 | `CudaProbe::scan()` / `is_usable()` / `report()`，`GpuDevice::memory_free_mib()` |
| `rust/tests/unit/gpu/probe.rs` | 104 | 7 个测试，硬件门禁默认开启，`NATAL_GPU_REQUIRE=0` 可关闭 |
| `rust/Cargo.toml` | +6 | `gpu = []` feature（**暂无依赖**） |
| `rust/src/lib.rs` | +2 | `#[cfg(feature = "gpu")] pub mod gpu;` |

**P0-a 已经能回答：**

| # | 问题 | 由谁回答 |
|---|---|---|
| 1 | 设备是否存在、名称是什么 | `nvidia-smi --query-gpu` |
| 2 | **compute capability 是否为 12.0（sm_120）** | 同上 |
| 4 | **可用显存是多少**（喂给 §D5 预算守卫） | `memory_free_mib()` |
| — | **容器内是否有 NVRTC** | toolkit 目录扫描 |

**P0-b 才能回答：**

| # | 问题 | 为什么 P0-a 答不了 |
|---|---|---|
| 1 | `cudarc` 能否动态加载 CUDA | 需要该 crate |
| 3 | **NVRTC 能否真的编译出一个 kernel** | 需要 NVRTC 绑定 |
| 5 | `GpuExecutor: Send` 是否成立 | 需要该 crate |

**运行方式**：

```bash
# 默认行为（本项目主要运行在 GPU 服务器上）：硬件断言全部执行，
# 缺卡 / compute capability 不符 / NVRTC 不可用 都是硬失败。
cargo test --features gpu

# 无 GPU 的主机：显式关闭硬件门禁，否则 GPU 测试会失败。
NATAL_GPU_REQUIRE=0 cargo test --features gpu
```

> `NATAL_GPU_REQUIRE` 未设置时**等同 `1`**：默认要求硬件，只有显式写 `0` 才跳过。
> 这样服务器上不会因为忘记 export 而出现“静默跳过”的假绿。

**⚠️ 若 NVRTC 不可用**（容器内无 `libnvrtc.so`）：改用 AOT 路线（构建期 `nvcc` 生成 cubin 嵌入），需重新评估 `maturin` 打包流程 —— 这是唯一可能推翻 D1 的情形。

#### P1 细则 —— 骨架与布局

**`Cargo.toml` 增量：**

```toml
[features]
default = []
gpu = ["dep:cudarc"]

[dependencies]
cudarc = { version = "...", optional = true, features = ["cuda-13020"] }
```

**`layout.rs` 是核心交付**：实现 `(B,2,A,Z) → (2,A,Z,B)` 双向转置（§3.5）。

**验收**：
- L1：整数状态 round-trip **逐位一致**
- `cargo build`（默认）与 `cargo build --features gpu` **都能通过**
- 既有 67 个 Rust 测试全绿（feature 关闭时）

#### P2 细则 —— 贯通与安全

**唯一的分支点**（见 §3.1 改动 6）：

| 位置 | 现状 | 改动 |
|---|---|---|
| `rust/src/sessions/age_structured.rs:480 fn run` | **已带 `py: Python<'py>`** ✅ | 无需改签名 |
| `rust/src/sessions/age_structured.rs:1092 fn run_inner` | 已带 `py` | 加提前返回分支 |
| `rust/src/sessions/spatial.rs:692 fn run_steps` | **不带 `py`** | **需加 `py: Python<'_>`**（PyO3 注入，Python 调用方无感） |

**Python 侧接线**（`src/natal/backends/rust/rust_backend.py`）：
`RustHeterogeneousSpatialLifecycleBackend`（`:873`）新增 `enable_gpu()` 方法，透传到 `self._session`。**纯新增，不改现有方法。**
调用链：`frontend/spatial/population.py:2603` → `rust_backend.py:1104 run_steps` → `rust_backend.py:1114` → `sessions/spatial.rs:692`。

**同时必须落地 §D5 的预算守卫与显式失败** —— 否则第一条大 `B` 运行就可能 OOM。

**验收**：L3 —— 用 `demos/gpu_spatial/age_structured/reference_cpu.py` 的模型构建函数作为测试夹具，确定性模式 `run(n)`，整数计数下**逐位一致**。

#### P3 细则 —— 确定性内核与停止门控

内核：存活（`age_structured.rs:869`）、密度调节（`density_regulation.rs`）、求偶矩阵（`age_structured.rs` ~`:133`）、观测投影（`output/observation.rs:11`）、树形归约（**替换** `state_reduce.rs` 的成对求和复刻，见 §D2）。

**停止门控（§D6）必须在本阶段完成** —— 之后就会开始跑长任务。

**验收**：L2 相对误差 < 10 × f32 eps；含 `stop_if_*` 的模型停止点与 CPU 一致。

#### P4 细则 —— 采样

**P3 全部验收通过之前，不要写任何采样内核。**

Philox4x32-10 + 四个采样器 + **multinomial 序列轴外提**（§4.2 方案 C）。

**验收**：统计检验通过 + GPU↔GPU 同种子逐位可复现 + counter 空间无碰撞。

#### P5 / P6

- **P5**：三个迁移变体的设备实现；大 D benchmark。
- **P6**：`EnsembleEngineSession`（B 轴）；与 `ProcessPoolExecutor` 对比。**注意**：P6 之前，`ProcessPoolExecutor` 就已经能给出线性加速且逐位一致 —— GPU 必须显著优于它才有意义。

---

### 5.4 ⚠️ Benchmark 方法学（因 GPU 共享而新增）

实测显示该卡 `GPU-Util 52%`、`397W/600W`、已被占用 19.2 GB —— **有其他负载在跑**。在这种环境下测出的加速比**会系统性偏低且不可复现**（我们的 kernel 在与别的 kernel 争抢 SM）。

**因此所有性能数据必须附带同租户快照：**

```bash
# 每次 benchmark 前后各跑一次，随结果一起落盘
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
           --format=csv
```

规则：
1. **加速比必须在同一同租户状态下对比**（要么都在空载，要么都记录占用率）；
2. **空载窗口优先**：与卡片所有者协调一段独占时间跑正式 benchmark；
3. **报告三组数**：CPU 基线、GPU（含同租户快照）、`ProcessPoolExecutor` 基线（§D4 要求）；
4. **不要用绝对耗时**做跨天比较 —— 共享环境下只比较同一批次内的相对值；
5. 若测得的加速比明显低于理论预期，**先检查是不是同租户干扰**，再怀疑实现。

---

### 5.5 回滚策略

因为整个方案是**旁路 + feature 默认关闭**，回滚成本极低：

| 层级 | 操作 | 影响 |
|---|---|---|
| 关闭 GPU | 不调用 `enable_gpu()`，或 `cargo build`（不带 feature） | 完全等价于改动前 |
| 移除 GPU 代码 | `rm -rf rust/src/gpu/` + 删掉 `Cargo.toml` 的 feature + 删掉 3 处增量改动 | 回到基线 |
| 紧急回滚 | `git revert` 该分支 | 无残留 |

**唯一无法"回滚"的是数据**：一旦用 GPU 路径产出了实验结果，就必须在结果里记录执行器（`gpu_status()`），否则无法区分"这份数据是 CPU 还是 GPU 跑的" —— 而按 §D3 它们来自**不同的随机系综**，混用会污染分析。

> **强制要求：所有输出/历史/元数据都必须记录本次运行的执行器与精度模式。**

---

## 6. 风险登记册

| ID | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | f32 计数失真（> 2²⁴） | 静默错误结果 | §D2 守卫 + 告警 + 文档约束 |
| R2 | **f32 顺序累加误差** | **隐蔽的数值错误** | §D2 强制树形归约 |
| R3 | GPU/CPU 随机模式不可互复现 | 误用为"加速版" | §D3 文档化 + 随机模式禁用逐位比对 |
| R4 | 采样器拒绝循环的 warp 发散 | 性能不达预期 | §4.2 序列轴外提；P3 实测调优 |
| R5 | counter 空间碰撞 | 隐性统计偏差 | 每 draw site 独立空间 + 专项测试 |
| R6 | 停止门控实现错误（停止点或历史行与 CPU 不一致） | 静默语义偏移 | §D6 设备侧门控 + **纳入确定性模式的对照测试** |
| R7 | **`allow_threads` 闭包误捕获 `Py`** | **UB（编译器不拦）** | §D7 契约 + `BatchView` 无 `Py` 字段 + 注释 |
| R8 | cudarc 句柄非 `Send` | 无法用 `allow_threads` | P0 确认；否则 Mutex 包装或工作线程持有 |
| R9 | 服务器 CUDA < 12.8 不支持 sm_120 | 阻塞 | P0 确认 |
| R10 | **无第二引擎可交叉验证** | 错误无法发现 | **CPU 路径完整保留**；每阶段 CPU↔GPU 对照 |
| R11 | 本机无 NVIDIA，GPU 代码无法本地验证 | 开发效率低 | feature 默认关闭；CI 需接入 GPU runner |
| **R12** | **GPU 为多租户共享（实测仅约 4.7 GB 可用，且 GPU-Util 52%）** | **分配失败 / 性能不可预测 / benchmark 失真** | **§D5 显存预算守卫 + 显式失败**；benchmark 必须记录同租户状态 |
| **R13** | 可用显存随其他租户波动，运行期被挤占 | 运行中途 `cudaMalloc` 失败 | **ERROR 中止 + 保留已完成结果；绝不中途换引擎**（§D5） |

---

## 7. 关键文件索引

**Rust 核心（改动点）**
```
rust/Cargo.toml                         ← 增加 optional cudarc + feature
rust/src/lib.rs                         ← 增加 1 行 mod 声明
rust/src/kernels/spatial.rs:76,100-138  ← 唯一并行点，也是布局与守卫的来源
rust/src/sessions/spatial.rs:61-88      ← 增加 gpu 字段
rust/src/sessions/age_structured.rs     ← 增加 gpu 字段
    run_inner / run_steps               ← 唯一分支点（提前返回）
```

**Rust 核心（参照，不改）**
```
rust/src/hooks/interpreter.rs:115-118   sequential 谓词（守卫来源）
rust/src/hooks/interpreter.rs:1180-1191 stop_if_* 实现（D6）
rust/src/kernels/rng.rs:201-385         现有采样器（GPU 需重写）
rust/src/kernels/rng.rs:38-40           stream_seed = seed ^ deme
rust/src/kernels/state_reduce.rs:20     成对求和复刻（精度动机，D2）
rust/src/kernels/offspring.rs:27        Z³ 收缩（Z 小 → 非瓶颈）
rust/src/output/history.rs              HistoryData 结构（D5 参照）
```

**Python 侧**
```
src/natal/contracts/blueprint.py        冻结模型（GPU 复用同一契约）
src/natal/contracts/params.py           可变参数
src/natal/frontend/output/observation.py:157-161   历史宽度调节杆（D5）
src/natal/frontend/presets/homing.py:180,252,343   HomingDrive 走张量非钩子（D6 依据）
src/natal/frontend/hooks/entry/declarative.py:199-272  stop_if_* API
src/natal/backends/rust/rust_backend.py ← 增加 enable_gpu 透传
```

**demos（算法来源）**
```
demos/gpu_spatial/age_structured/gpu_model.py:480   顺序条件二项 multinomial（→ §4.2 方案 C 改造）
demos/gpu_spatial/age_structured/gpu_model.py:461   _sample_binomial
demos/gpu_spatial/age_structured/gpu_model.py:318   _shift_grid
demos/gpu_spatial/age_structured/gpu_model.py:338   _migration_stencil
demos/gpu_spatial/age_structured/gpu_model.py:429   _mating_probability
demos/gpu_spatial/*/compare_cpu_xpu.py               对照实验框架
```

---

## 8. 一句话总结

> **GPU 以「旁路」形式插入：新增 `rust/src/gpu/` 模块树（feature 默认关闭），在 `run_inner`/`run_steps` 开头加一个提前返回分支，CPU 路径一行不改。**
>
> 关键技术点四个：
> 1. **CUDA（cudarc 动态加载 + NVRTC）** —— 目标 RTX 5090，本机无 GPU 也能编译
> 2. **布局转置 `(B,2,A,Z) → (2,A,Z,B)`** —— Z 小导致原布局无法合并访存
> 3. **采样序列轴外提** —— 把 `multinomial` 的 `K-1` 步依赖链从线程内层提到所有线程共同推进的外层，零 warp 发散且不增加工作量
> 4. **f32 + 树形归约** —— 5090 的 FP64 只有 1/64，f64 上 GPU 等于没加速
>
> 工作量因两个核验结果而显著低于 v1 预估：**`Z` 小 → 后代张量非瓶颈**；**`HomingDrive` 不走钩子 → 典型模型零同步点、无需移植 CSR 解释器。**
>
> **⚠️ 但服务器实测暴露一个 v1/v2 都未预见的前提问题：该 GPU 是多租户共享的，24 GB 标称显存中仅约 4.7 GB 可用（已用 19.2 GB，且 GPU-Util 52%）。**
> 因此 **§D5 的"全程驻留、结束才下载"必须改为"预算驱动 + 显式失败"**（**显存不足即报错，不静默回退 CPU**），且 **所有 benchmark 必须附带同租户快照**（§5.4）。
> 这一条不影响架构设计，但直接影响可用的 `B` / `T` / 采样模式组合。
