# M4 路线 A 产品化设计（提案，待批准）

> 依据：`Hex_model_recon.md` §4「路线 A 设计」、§5 M4；证据：`results/M4_routeA_prototype_report.md`、
> `results/m4_template_build.json`、`results/m4_gpu_fft_prototype.json`、M3（`M3_scale_report.md`/`M3_gpu_report.md`）。
> 状态：**设计提案**。实施属**高风险**（新增迁移执行路径），需用户批准 + 独立 evaluator 复核；本文不改产品代码。

## 1. 目标与硬约束

- 目标：为空间会话新增一条**可选**、设备无关的「卷积/模板迁移」路径，使论文规模（宽核 r25、300²–海南）
  在内存与时间上可行；解决 M3 暴露的 `16·nnz` 内存墙与纯 Python fold 构建墙。
- **硬约束**：
  1. **CPU CSR 是唯一 golden**，其数值语义与 `phase0` 不得改变（路线 A 新路径**默认关闭**）。
  2. 路线 A 因 FFT/模板求和顺序不同，**不追求与 CPU 逐位一致**；以 CPU CSR 为基准做**容差对照**。
  3. 不引入生产依赖（FFT 走自研/系统库；GPU 用现有 `cudarc`/`cuFFT` 方向，不引 torch）。

## 2. 决策摘要（2026-09-29）

- 当前聚焦**论文大核** → **FFT 为主**实现（CPU FFT + GPU cuFFT，按类 batched）；小核直接 stencil 更优但**暂不实现**。
- 表示层用**模板 + 标签**（内部 1 + 边界 `O(k²)` 个），替代逐 deme CSR 物化。
- 边界用**归一化卷积** `Z = K' ⊛ m` 精确复现「丢界 + 邻域重归一化」；论文 `'replicate'` 用边缘填充变体。

## 3. 架构

### 3.1 迁移策略解析（构建期）
- 在 `resolve_migration_mode` 旁新增策略值（例如 `convolution`/`stencil`），与 `adjacency`/`kernel` 并列；
  默认仍 `auto→kernel(CSR)`，**仅显式启用**才走路线 A。
- 启用时按地形构造模板/标签：
  - **矩形域**（论文 flat/junction）：解析式
    `key=(min(r,R), min(rows-1-r,R), min(c,R), min(cols-1-c,R))`，构建 `O(n)` + `O(#templates·k²)`；
    已验：300²×k=51 构建 **0.61 s vs fold 63.5 s（104×）**，存储 58.7 MB vs CSR 3274 MB（~56×）；
    校验口径：小规模全量、300² 两例抽样（解析推导保证全体一致）。
  - **不规则域**（海南 land mask）：按「有效邻居集合签名」哈希分组（位打包/字典），签名数 ~`O(周长·k)`。
- **缓存 key**：topology（形状/`wrap`）、kernel size/support、`include_center`、`adjust_on_edge`、域掩膜。
  **不含** σ 数值/迁移率/生态参数。σ/环境变只**原地更新权重**；地形/掩膜变才重建类型映射。
  可选**持久化**：`(key) → (templates, type_map, K'-FFT)` 存盘，重复实验直接加载（你提出的“第一次慢、之后复用”）。

### 3.2 预计算
- `K'` = 去中心核；`Z = K' ⊛ m`（仅形状相关；边界层外为常数 `Z∞`，可只存 `O(R²)` 查找表）。
- FFT 路径：预算 `FFT(K')`（或对每个域尺寸预算一次）；大域用 overlap-add 分块。

### 3.3 运行时内核
- 每类（性别×年龄×ztype）每 tick：`g = rate·f / Z`；`f_new = f·(1−rate) + IDST( DST(K')·DST(g) )`。
- **CPU**：FFT 主路径（系统 FFT 库，或自实现 radix-2/混合基）；线程可用 `rayon` 并行 batched。
- **GPU**：cuFFT（计划经 `cudarc` 绑定），batch 维度 = 类；状态常驻显存，逐 tick 不回传。
- **边界选项**：默认「丢界+重归一化」(`Z`)；提供「`'replicate'`（论文口径）」= 边缘填充的线性卷积。

### 3.4 会话接线
- `SpatialPopulation` 新增可选参数（如 `migration_execution="csr"|"convolution"`，默认 `csr`）。
- 会话持有路线 A 的状态（模板表/类型映射/`Z`/核 FFT），与现有 CSR 字段并存、互斥。
- 公共 API 不变；默认行为不变。

## 4. 数值与确定性策略
- **不以逐位一致为目标**；以 CPU CSR 输出为基准，按设备/精度分档：
  - 浮点容差：CPU f64 FFT vs CSR ~1e-15/步（原型实测）；GPU f32 FFT ~2e-7/步（原型实测，优于项目 f32 档 ~1.2e-6）。
  - 随机模型：统计等价（不作逐位）。
- 记录：路线 A 结果必须带「非 golden、容差路径」标记。

## 5. 验证与门禁（实施阶段）
1. **单元/等价**：模板三元组与 CSR **逐条相同**（已验）；归一化卷积/FFT 与 CSR 逐格 ≤1e-12（已验）。
2. **端到端**：与论文 MATLAB 参考/PDE 对照（M2 已达 4.8e-12）。
3. **回归**：`phase0` 必须仍 bit-identical（新路径默认关闭，理论上不受影响，需实测）；`cargo test`（含 `--features gpu`）、
   `ruff`、`pyright`、`pytest` 全量门禁。
4. **GPU**：GPU-FP vs CPU-CSR 容差分档；显存/预算（路线 A 不再需要 `4·nnz·A·Z²` 迁移缓存）。
5. **独立 evaluator**（高风险）复核设计与实现。

## 6. 风险与缓解
| 风险 | 缓解 |
|---|---|
| 破坏 CPU golden | 新路径默认关闭；CSR 路径一行不改；`phase0` 门禁 |
| FFT 精度/边界 | 归一化卷积与 CPU 基准对照；`'replicate'` 变体可选 |
| 不规则地形模板膨胀 | 签名哈希 + 上限告警；超限回退 CSR |
| 大域 FFT 显存/内存 | overlap-add 分块；预算守卫 |
| 生产依赖 | 不引入第三方 FFT/torch；GPU 走既有 CUDA 绑定 |
| 复杂度/维护 | 模板+标签与 Z 的单元测试；与 CSR 的差分测试常驻 |

## 7. 落地步骤（批准后）
1. CPU FFT 内核 + 会话接线 + 模板缓存（最小可用，默认关闭）。
2. 不规则域签名构造 + 缓存持久化。
3. GPU cuFFT 内核 + 预算守卫 + history/hook 语义对齐。
4. 文档（中英同步）+ 全量门禁 + 独立复核。

## 8. 待决
- FFT 库选型（CPU：自研 vs 系统）；GPU cuFFT 经 `cudarc` 的绑定方式。
- 路线 A 是否也提供「小核直接 stencil」（当前决定：不实现，仅记录）。
- 缓存持久化格式与失效策略。
- 是否将路线 A 纳入 M5/M6（径向/海南）的默认执行。

## 9. 实施决定与分阶段计划（2026-09-29，用户确认）

**决定**：路线 A **GPU FFT 路径 = cuFFT，经 `cudarc` 绑定**（cudarc 0.19.9 已有 `cufft` feature；容器内 `libcufft.so.12` 就位；**不新增 crate**）。CPU 保持 CSR golden，不做 CPU FFT。设备态为 batch-minor `(2,A,Z,B)`，每个（性别,年龄,ztype）平面在 B 上连续，适合逐平面 2D FFT。

分阶段（每阶段保持仓库可构建、默认关闭、可独立复核）：

- **S0（本阶段）**：启用 `cudarc` 的 `cufft` feature；新增 `rust/src/gpu/cufft.rs` 薄封装与隔离 **smoke test**（r2c/c2r 往返、与 CPU 参考对比），证明绑定在本机可编译可运行。**不改任何运行路径/契约。**
- **S1**：契约与前端——新增可选迁移执行模式（如 `migration_execution="csr"|"fft"`，默认 `csr`）+ 传递 stencil 数据（`K'`、`Z`、核 FFT 句柄/尺寸、域 rows/cols）；前端算 `Z` 与模板；**默认关闭，不改 CSR 行为**。
- **S2**：`GpuExecutor` 接线——新增 FFT 迁移方法（NVRTC 点乘核 + cuFFT R2C/C2R + 预算按 `O(n·classes + FFT 工作区)`）；`enable_gpu` 在选择 fft 模式时走新路径；保持 CSR 路径不变。
- **S3**：会话/`SpatialPopulation` 暴露开关 + 中英文档 + 测试（GPU-FFT vs CPU-CSR 容差分档）+ 全量门禁（`phase0` 必须 bit-identical）。
- **S4**：独立 evaluator 复核（高风险）。

**约束**：所有阶段默认关闭；`phase0`/CSR 数值语义不变；不引入第三方 crate（仅启用现有 cudarc feature）。

**S0 状态（2026-09-29，已验）**：`rust/Cargo.toml` 给 cudarc 加 `cufft` feature（不新增 crate）；新增 `rust/src/gpu/cufft.rs`（S0 探针 + 单元测试）。测试
`cargo test --features gpu gpu::cufft` → **1 passed**：cuFFT r2c/c2r 线性卷积与 CPU 直接卷积一致（f32，相对误差 <1e-4），
在本机 RTX 5090 上可编译、可运行。**未改任何运行路径/契约/公开 API。** 后续 S1（契约/前端）、S2（executor 接线）、S3（会话/文档/门禁）待做。
