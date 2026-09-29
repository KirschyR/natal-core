# Hex 模型复现计划（GPU 旁路空间支持）

> 交接文档。供不同 agent 接手「用 natal-core（及其 GPU 旁路）复现六边形基因驱动空间模型论文」这一长程任务。
> 约束以 `AGENTS.md` 为准；GPU 旁路现状以 `GPU_STAGE_SUMMARY.md` 为准；审查流程走 `EVALUATE.md`。

---

## 0. 定位、限制与材料位置

**目标**：以 [论文] 为对象，评估并复现其六边形（hex）空间基因驱动模型，借此**检验/扩展 natal-core 空间 GPU 旁路**；最终给出可复跑的实验与方法学对照。

**阅读论文（重要）**
- opencode **不能直接读 PDF**，但仓库内已有**面向 agent 的全文衍生素材包** `hexagon_spatial_test/hex_model_2026_paper/`，可视为原文等价（信息未删改，仅重排为可读文本）：
  - `README.md`：元数据、**页→章节映射**、公式速查、阅读协议；
  - `paper_fulltext.md`：**原文全文**，按 `<!-- ===== PDF PAGE n ===== -->` 分页，可引用 “p.n”；
  - `figures_and_tables.md`：15 幅图 + 4 张表的**逐项索引**（题注原文、轴/色条读数、图像路径、要点）；
  - `candidate_simulation_methods.md`：方法学结构化档案（与 `hexagon_sptial_test.md` 同源）；
  - `figures/`：插图原始 PNG（`figures/_preview/*.jpg` ≤1400px，供视觉模型）。
- 阅读协议：方法细节 → `candidate_simulation_methods.md`；原文精确表述/数值 → `paper_fulltext.md` 按 p.n 定位；看图 → `figures/_preview/`。
- 参考材料目录 `hexagon_spatial_test/` 已加入 `.gitignore`（用户操作），**不随仓库分发**。
- 论文模型是**确定性、连续计数、无随机性**的（其自述首要缺陷）。这与 natal-core 的随机/离散能力互补：复现默认走**确定性**路径。

**参考材料**
| 路径 | 内容 |
|---|---|
| `hexagon_spatial_test/hex_model_2026_paper/` | **原文全文 + 图表索引 + 方法档案 + 插图（首选）** |
| `hexagon_spatial_test/hexagon_sptial_test.md` | 早期方法学梳理（与 methods 档案同源，741 行） |
| `hexagon_spatial_test/Hex-model-main/wavespeed_and_waveshape/` | 波速/波形：hex vs 反应扩散 PDE |
| `hexagon_spatial_test/Hex-model-main/parameter_sensitive/` | 扩散/适合度/转化率敏感性（含预计算 `.mat`） |
| `hexagon_spatial_test/Hex-model-main/radial_release_optimizer/` | 圆形释放优化（代码均匀 800×800） |
| `hexagon_spatial_test/Hex-model-main/linear_release_optimizer/` | 线性/道路释放优化 |
| `hexagon_spatial_test/Hex-model-main/hainan_model/` | 海南岛地理模型（2599×2601，月 K、轮渡） |
| `hexagon_spatial_test/Hex-model-main/README.md` | 仓库工作流与指标定义 |
- MATLAB 可用：`/opt/matlab`（R2026a）。用法：`matlab -batch "cd('<模块目录>'); <函数调用>"`（如 `main(1,10000,50)`），用于生成**外部 golden**。
- 缺失件：`hainan_model/main.m` 需要 `drop.mat` 与 `release_m.png`（**仓库中不存在**），故海南“具体释放方案”不能原样重跑；但波浪/径向/线性模块的输入齐备，且 `parameter_sensitive/radial/*.mat` 有预计算结果。

---

## 0.1 已冻结的决策（2026-09-28，用户确认）

| # | 决策项 | 结论 |
|---|---|---|
| 1 | 首期范围 | **只做模块 1–3：波速/波形、径向释放、线性释放**（零缺失、今天可跑；覆盖 Fig 3/S3/S5/S6–S9 与 Table 1/2） |
| 2 | 海南岛（Fig 4/5/6、Table 3/4） | **暂不做**。缺 `drop.mat`（释放时间表）、`release_m.png`（释放空间掩膜）、`spare.mat`（抑制率基线）——即缺"何时/何量/何处投放"与抑制率分母。留待有数据（索取）或后续自建 |
| 3 | 迁移执行模型 | **先 M3 实测再决断**：量化 CSR 内存/构建墙与 GPU 可行性后，再决定是否立项通用宽核迁移路径；不预先改引擎 |
| 4 | natal-core 改动边界 | **仅"确有缺失且通用"的新增**，默认关闭、独立 evaluator 复核；新增须服务同类模型而非本项目特化 |
| 5 | 年龄结构 | **共享年龄轴 + 雄龄零填充**近似（雌 8/雄 5 → 共享轴，雄龄后段填 0） |
| 6 | 场地口径 | **以论文 500×500 为对照口径**（代码 800×800 仅作参考） |
| 7 | 外部 golden / 随机 | **MATLAB 跑参考**（ZIP 预计算 PDE 结果可直接核对 Fig 3/S5）+ **以确定性为主**（论文无随机） |

> 缺件说明：`drop.mat`=释放时间表；`release_m.png`=释放位置掩膜；`spare.mat`=抑制率基线。三者均只影响海南释放实验（模块 4），不影响模块 1–3。

---

## 1. 论文模型与实验（基于方法学梳理）

### 1.1 遗传系统（6 类，`drive_params.mat` 的 `gn`）
| 系统 | 基因型数 `gn` | 机制要点 | 阈值（理想） |
|---|---|---|---|
| Homing modification | 5 | 生殖细胞 HDR 拷贝；dd/wd/ww/rw/dr；无功能抗性 | 0 |
| Homing suppression | 5 | 同上，靶向雌性生育力、无 rescue；`fitness=111` 哨兵→按“抑制率”统计 | 0 |
| TARE | 5 | 切割全部产生失活抗性；靶向必需单倍足量基因 | 0（频率依赖） |
| 2-locus TARE | **25** | 两位点独立驱动，互为毒/解药 | 18% |
| TADE | 3 | 显性胚胎致死；GE（生殖细胞）与 embryo-activity 两版 | 0 / 33% |
| TADE suppression | 3 | 额外破坏雌性生育力基因 | 0 |
| Wolbachia | 2 | 母系传播 + 细胞质不亲和 | 理想 0；带代价 31% |
| CifAB | — | 整合 CI 基因，机制同 Wolbachia | 37% |
- 释放基因型约定：**修饰驱动释放纯合子；抑制驱动释放杂合子**。
- 理想参数见 `drive_params.mat`：`gn`、`mats`（形状 `(gn², gn)`，每个后代基因型对应一个 `gn×gn` 亲本组合矩阵）、`fitness`、`carrier_index`、`cord_pure_w`、`cord_release_genotype`。实测：homing `carrier_index=[1,2,5]`；wolbachia `gn=2, fitness=0.75`；2ltare `gn=25, mats=(625,25)`（1-based）。

### 1.2 生命周期
- **通用 hex 模型**：1 时间步 = 1 代；`Δt=1`（径向释放前 5 代用 `Δt=0.01` 以避免负密度）。繁殖方程同 PDE。
- **蚊子 hex 模型**：1 步 = 1 周；**2 幼期 + 8 雌成虫期 + 5 雄成虫期**（雌雄寿命不同）；成虫无密度依赖死亡，密度依赖全在幼期。
  - 每周存活：female_i `7/8 − i/8`；male_i `4/5 − i/5`。
  - `eggs=40`、内禀增长 `β(λ)=5`、`survival1=1`、`s_f=(F+1)/2=4.5`。
  - 世代换算：1 年 = 26 周；2 年 = 33 代；5 年方案 `total_time=261` 周。
- **密度依赖**（juvenile0 存活）：
  `r = (j1 + 5·j2)/(K·(eggs + s_f))`，`survival0 = 2β/(((β−1)·r + 1)·s_f·eggs)`（逐 hex 除以地形 K）。

### 1.3 迁移（核心机制）
- 高斯扩散核，**六边形轴向坐标**距离：`d = sqrt(Δx² + Δy² − Δx·Δy)`（两轴夹角 60° 的余弦定理）。
- 代码 `get_mig_matrix.m`：`sigma=10.437`（**硬编码，覆盖输入 avd**）、`max_dist=25`、`kernel_len = 2·ceil(2·25/√3)+1 = 59`、`normpdf(d,0,σ)`、`d ≤ 25` 截断成**六边形盘**、全核归一化（守恒）。论文正文举例 `avd=11.211`，与代码 `σ=10.437 ⇒ avd=σ√(π/2)=13.08` 不一致（版本差异，需标注）。
- 运行时：对**每个成虫年龄类**做 `imfilter(field, mig_matrix, 'same', 'replicate')`（零通量边界）。
- 扩散系数换算：`D = avd²/π`。

### 1.4 参考一维 PDE（金标准）
域 300、dx=0.1、dt=1e-4、K=1、λ=5；方程 `∂N/∂t = D·N_xx + Nλ/(N(λ−1)+1) − N²`。（`pde/main.m:24` 为 `dt=0.0001`）

### 1.5 实验矩阵
| 实验 | 场地 | 初始/释放 | 读数 |
|---|---|---|---|
| 波速/波形 | 300×300 | 驱动纯合子占**左侧 20%** | 波中心(50%)过 50%/60% 检查点的速度；两个方向(flat/junction) |
| 径向释放优化 | 500×500（代码 800×800） | 全野生型、密度 1/hex，中心圆释放 (t,h,r) | 2 年末/5 年末 ≥90% 覆盖面积、efficiency=area/released |
| 线性/道路释放 | 同上 | 1-hex 宽线，优化 (时长,密度,间距) | 同上 |
| 海南岛 | 2599×2601 | 月度 K 图；网格阵列/道路释放；第 25 周释放 | 面积/人口覆盖率与效率；轮渡通道跨海 |

### 1.6 指标
- 波速：50% 频率（或 50% 抑制）点通过两检查点的时间差。
- Coverage Area：携带频率（或抑制率）≥ 90% 的 hex 面积；效率 = 覆盖面积 / 释放个体数。
- 评估时点：归巢驱动 2 年末；其余 5 年末。

### 1.7 论文 vs 公开代码：差异与不确定项（复现前必须冻结）
- **场所不一致**：径向/线性优化器代码用 **800×800**；论文正文写 **500×500**。取哪个需明确（建议：论文 500×500 作为对照口径，代码 800×800 作为可运行复现口径）。
- **核尺寸**：论文正文写 **51×51**（轴向 ±25 的有效支撑）；代码 `get_mig_matrix.m` 实际生成 **59×59**（`2·ceil(2·25/√3)+1`，为半径 25 的六边形盘留外接余量），再按 `d≤25` 截成盘。
- **平均扩散**：论文举例 **avd = 11.211** hex；代码把 σ 硬编码为 **10.437**（且覆盖传入的 avd 参数），得 **avd = σ√(π/2) ≈ 13.08**。两者不一致（疑似代码版本演进）。
- **Figure 3 横轴**：图像读数上限为 **2**；但正文/Figure S5 提到 dispersal 到 **10**。复现时应同时覆盖低扩散（0–2）与高扩散（10）。
- **承载量 K 参数无取值**：原文列出 `a1..a5, p1..p12` 的**命名**但**未给具体数值**（`rainfunc/waterfunc/bloodmeal/altitudefunc/tempfunc`）。海南 K 图只能从 `terrain.mat` 反推或向作者索取。
- **缺失输入**：`drop.mat` 与 `release_m.png` 不在仓库 → 海南释放掩膜无法原样重跑，需自建或索取。
- **抑制驱动统计口径**：`fitness = 111` 是**哨兵值**，触发分析脚本改用“抑制率” `S = 1 − 当前种群/基线种群`（基线来自 `spare.mat`）而非携带频率。
- **代码补充细节（论文未写）**：优化器 `collect_week=[104,261]`、`total_time=261`；海南 `total_time=311`、释放时刻 `drop_time+50`、释放量 `drop_freq/1.6667`；面积/人口覆盖只统计 `border` 之外且矩阵**第 580 行之后**的区域；Figure 4/5/6 用仿射剪切 `[1 0 0; −tan(π/6) 1 0; 0 0 1]` 把方格渲染成六边形外观。
- **数据来源**：人口 WorldPop+CIESIN、气候 WorldClim 2（1 km 分辨率）。

---

## 2. 参考 MATLAB 代码结构

### 2.1 目录
- `wavespeed_and_waveshape/`：`prepare_tests.m`（批处理建 mig.mat）、`wavespeed_analyse.m`、`test_code/{hs,ps,hv,pv}/*`。
- `parameter_sensitive/{radial,speed,plots}/`：敏感性 + 预计算 `.mat`。
- `radial_release_optimizer/<drive>/`：`main.m`（模拟）、`renew_function.m`（基因型重组）、`drive_params.mat`；共享 `get_mig_matrix.m`、`mig.mat`。
- `linear_release_optimizer/<drive>/`：同上，另有 `result_collector.py`。
- `hainan_model/`：`main.m`、`analyse.m`、`get_mig_matrix.m`、`terrain.mat/pop.mat/border.mat/mig.mat`。

### 2.2 核心循环（以 `radial_release_optimizer/main.m` 为准）
状态：`data(gn, ages_length, m, n)`；`ages_length = 2 + M + F`。
每步（周/代）：
1. 季节 K：`month = day2date(day)`，取 `terrain(:,:,month)`（径向均匀场为全 1）。
2. 释放：若 `t ∈ drop_time`，`data += release_mat`（在释放基因型 × 两释放年龄类，逐 hex 加 `h/2`）。
3. 早期终止：携带着总量 `< 1e-5` 则停。
4. 密度依赖：`j1,j2` 幼期总量 → `r` → `s1`，写入 `largemat(2,1,:,:)`。
5. 繁殖：`newborns = eggs · Σ雌成虫 · renew_function(data,mats,fitness)`。
6. 老化/存活：`data = pagemtimes(data,'none',largemat,'transpose')`，再覆盖第 1 年龄类为 `newborns`。
7. 迁移：对每个成虫年龄类 `imfilter(..., mig_matrix, 'same','replicate')`。
8. `NaN→0`；在 `collect_week` 记录覆盖率与释放量到 CSV。

**`renew_function.m`**（每驱动一个）：把成虫按性别/基因型频率结合，用 `mats` 的对应 `gn×gn` 子矩阵计算每个后代基因型的出生比例，并施加 `fitness`（含雌性合适度、雄性求偶权重）。它等价于一个**按基因型的后代转移张量**。

### 2.3 海南模型差异（`hainan_model/main.m`）
- `m=2599, n=2601, F=8, M=5, total_time=311`，`startdate=1`，轮渡点 `(1168,515)-(1308,718)`，`bridge_mr=0.01`。
- 读取 `border/terrain/drive_params/drop/mig` 与 `release_m.png`（后两者缺失）。
- 输出 `result.mat` 与 `release_result.png`。

### 2.4 迁移核构造（`get_mig_matrix.m`）
见 §1.3；结果 `mig.mat`：`mig_matrix (59,59)`、`avd=13.08`；实测核最大值 ~1.34e-3、均值 ~2.87e-4。

### 2.5 输入数据实测
- `terrain.mat`：`terrain (2599,2601,12)`，0–66.54（均 2.95）；`border (2599,2601)` 0/1 掩膜。
- `pop.mat`：`output (2599,2601)`，0–6.27e4（均 121.5）。
- `mig.mat`：见上。

### 2.6 波速模块的补充发现（M0 勘察，2026-09-28）

- **MATLAB 可用**：`/opt/matlab/bin/matlab`（R2026a Update 3；`matlab -batch "..."` 正常；一条 `personal folder` 警告无害，可用 `-sd` 规避）。
- **波速入口不在顶层模板**：`test_code/{hs,ps,hv,pv}/main.m`（顶层模板）有 bug —— `data=zeros(gn,m,n)` 的 `m` **未定义**；真正入口是**场景子目录**（如 `test_code/hs/homing/main.m`，内容与顶层模板相同、仍含该 bug）。场景目录含 `main.m`/`renew_function.m`/`drive_generator.m`，需先跑 `drive_generator.m` 生成 `drive_params.mat`（`cifab` 用 `dp.mat`）；运行前需自行定义 `m`（建议 `m=original_m` 或 `m=round(original_m*avd)`）。
- **语义差异（重要，影响 M2）**：波速用「通用 hex 模型」，其递归是**基因型频率 + ODE 式**更新：
  `d_ar = rct · λ/((λ-1)N+1)/N − ar.*N`，其中 `rct` 由 `new_ar'·mats·new_ar` 得到（`new_ar = reshape(ar, gn,1,m,n)`），再逐基因型 `imfilter(...,'replicate','same')`，`data += d_ar`。**这不是 natal 的年龄结构生命周期**（natal 是 reproduction→survival→aging 分阶段）。
- **径向/线性优化器**用的是**蚊子年龄结构**（F=8/M=5/eggs=40），与 natal 的 `age_structured` 更接近。
- → 结论：M2（波速）可能**无法直接用 natal 现有生命周期精确表达**该更新，需在 M2 明确选择「自写参考实现 / natal 近似 + 标定 / 通用新增（基因型频率-逻辑斯蒂招募）」。

### 2.7 预计算结果（外部 golden 候选）
- `parameter_sensitive/radial/<drive>/*.mat`（如 `homing3_0_2_21_0_30_21.mat`）：含 `avd`、`checkpoint1/2`、`output`、`drive_name`、`mode_params`（内含 `mats/fitness/gn/...`）等，是**已跑出的波速/径向结果**，可直接作对照；注意其 `m/n` 与 `avd`/σ 的版本差异（§1.7）。
- `hainan_model/spare/`：抑制驱动的基线档案（“抑制率”统计的对照来源）。

---

## 3. natal-core 现状与能力映射

| 论文要素 | natal-core 能力 | 差距 / 需要的工作 |
|---|---|---|
| 六边形网格 | **已有 `HexGrid`**（`src/natal/frontend/spatial/topology.py:251`），度量 `dr²+dc²+dr·dc`（与论文度量同构，差一个轴向符号） | 需核对坐标映射与释放圆几何 |
| 高斯迁移核 | `build_gaussian_kernel`（`topology.py:343`）支持 hex 度量与 `mean_dispersal→σ` | **无 hex-disk 截断**（论文用 `d≤25`；natal 取整个方阵）；需补截断或等价处理 |
| 迁移执行 | 构建期把核折叠成**显式 CSR**（`migration.py:fold_migration_csr:242`）；运行时 CSR | **宽核不可行**：nnz = `n_demes × 支撑度`（见 §4） |
| 随机 GPU 迁移 | `migrate_tick_stochastic`，行宽硬上限 `MAX_CSR_ROW=32`（`kernels.rs:63`，`executor.rs:2389`） | 论文核支撑 ~1951 ≫ 32 → 随机 GPU 直接拒绝 |
| 确定性 GPU 迁移 | `migrate_tick`（反向 CSR gather）无行宽限制 | 受 `migration_cache_bytes ~ nnz·A·Z²·4` 显存墙限制 |
| 年龄结构 | 单一共享 `n_ages`；异性别寿命靠**零填充**（`builder/_params.py:163`） | 雌 8 / 雄 5 需映射到共享轴（如 A=11，雄 5–10 填 0） |
| 状态计数 | f64 连续，允许非整数；确定性模式无随机 | ✓ 与论文一致（默认走 deterministic） |
| 基因型 | 2 性 × `n_ztypes`，`n_ztypes` 由等位×位点枚举；`offspring_tensor (Z,Z,Z)` 可直接写 | 论文 `mats (gn²,gn)` 需映射为 `offspring_tensor`；TADE/2-locus/CifAB 无现成 preset |
| 驱动 preset | `HomingDrive`、`ToxinAntidoteDrive`（涵盖 TARE/TADE）、`Wolbachia`/`CytoplasmicPreset`、`TransgenicBackground` | 缺 2-locus TARE、CifAB、专用 TADE suppression；可组合或直接写张量 |
| 释放 | 初态 per-deme / 声明式钩子 `Op.add(...)`（`when="tick>=N"`、deme selector） | 无日历/季节调度；释放掩膜需脚本生成 |
| 承载量 K | per-deme 标量或 `batch_setting` 2D/callable 栅格 | 无月切换（可用钩子/`set_param`）；无 GeoTIFF 读入 |
| 边界 | `wrap` 或 drop | 无 `replicate`（Neumann）语义；需近似 |
| 空间 GPU 会话 | deterministic+stochastic+continuous+hooks+history（`sessions/spatial.rs`） | 拒自定义 growth curve；`MAX_Z=32`、`MAX_AGES=64` |
| 已有 hex demo | `demos/spatial_hex.py`（5×5）、`demos/spatial_hex_discrete.py`（**501×501**，size-11 核，**CPU**） | 说明 hex+大网格可行，但宽核/GPU 尚未验证 |

---

## 4. 硬瓶颈与核心决断：迁移执行模型（本项目的关键）

**硬瓶颈速览**
1. **迁移执行（最关键）**：natal 构建期把核折叠成显式 CSR，nnz = `demes × 核支撑`；论文 r25 盘 ~1951 邻居/源，规模一大就内存/构建不可行（见下表），GPU 迁移 cache 更甚。→ 这是唯一**可能必须**新增到 natal 的通用能力（宽核/长程迁移执行路径）。
2. **随机 GPU 迁移行宽 ≤32**：论文是确定性，暂不阻塞；随机对照才会受限。
3. **单一共享年龄轴**：雌 8 / 雄 5 需零填充到共享轴（近似）；原生雌雄异龄是可选通用扩展。
4. **核不截断 & 边界无 `replicate`**：natal 高斯核取整方阵（非六边形盘）、边界仅 wrap/drop；属精度近似。
5. **驱动系统 / 释放 / K**：可用现有 API 表达，**无需改引擎**——任意基因型转移写 `offspring_tensor (Z,Z,Z)`；释放用声明式钩子 `Op.add`（tick 条件 + deme selector）；逐月 K 用 `batch_setting`/`set_param` 注入。
6. **维度上限**：`MAX_Z=32`（2-locus gn=25 通过）、`MAX_AGES=64`（A=11/15 通过）。

论文本质是**大格点上的稠密卷积**（`imfilter`）；natal 是**稀疏 CSR**。二者在核半径 25 时不可调和：

| 场地 | demes | 核支撑 | nnz ≈ demes×支撑 | CSR 存储量级* | GPU 迁移 cache 量级** |
|---|---|---|---|---|---|
| 300×300 | 9.0e4 | 121（size11） | 1.1e7 | ~0.13 GB | ~11 GB |
| 300×300 | 9.0e4 | 1951（r25 盘） | 1.8e8 | ~2.1 GB | ~180 GB（不可行） |
| 500×500 | 2.5e5 | 121 | 3.0e7 | ~0.36 GB | ~30 GB（超 24GB） |
| 500×500 | 2.5e5 | 1951 | 4.9e8 | ~5.9 GB | —— |
| 2599×2601 | 6.8e6 | 121 | 8.2e8 | ~9.8 GB | —— |
| 2599×2601 | 6.8e6 | 1951 | 1.3e10 | ~160 GB | —— |

\* 粗糙估计：每边 `int32` 索引 + `f64` 权重 ≈ 12 B；\*\* 依 `migration_cache_bytes` 的 `nnz·A·Z²·4` 主导项（A、Z 为年龄/基因型数）粗估，**需实测核对**。

**三条路线**
- **A. 原生卷积/模板迁移执行路径（泛化）**：为空间会话新增「稠密/可分离卷积迁移」策略，CPU 与 GPU 各实现一套迁移 kernel，避免 CSR 展开。代价高、风险高，但这是把论文规模（尤其大半径核与 2599×2601）纳入 natal 的唯一正解，且对所有「宽核/长程扩散」模型通用。
- **B. 多步最近邻近似**：用现有 CSR 小核（如 size-3/5）多次迭代近似扩散，避免宽核展开。实现快，但**波形/波速会偏离**论文核，需重新标定，且不是同一模型。
- **C. 缩减规模/范围**：只在现有能力内做**小场地 + 中等核 + 确定性 GPU** 的波速/波形与径向定性复现；不追求海南规模，把差距写成报告。

**建议**：先按 §5 的 M0–M3 在路线 C 内拿到**可验证的波速/波形与遗传对照**；在 M3 结束时用实测数据决定是否投入路线 A（或接受 B 的近似）。**不要在未决断前动 CSR/内核大改**。

#### 路线 A 设计（候选，2026-09-29 讨论冻结）

目标：把「显式 CSR 展开」换成「平移不变的模板/卷积」，同时解决 M3 暴露的**内存墙**（`16·nnz`）与**构建墙**（纯 Python fold）；每 tick 成本再由 FFT 从 `O(n·k²)` 降到 `O(n log n)`。

- **表示层：模板 + 标签压缩**（取代逐 deme 物化 `dest_idx`）
  - 迁移算子在内部平移不变、仅边界不同：内部 1 个模板（`k²` 个 `(Δrow,Δcol,weight)`）；边界模板数 ≈ `(R+1)²`（`R`=核半径；矩形 4 重反射对称可再降）。
  - 存 `n_demes` 的 `type-id` 表；运行时按 `dest = src + Δrow·cols + Δcol`（行主序、不 wrap、不跨行；跨行/越界只发生在边界，由边界模板覆盖）用**索引算术展开**，既不存 `dest_idx`、也不重复存内部核。
  - 内存 `O(#templates·k² + n)`（300²×51² ≈ 14 MB vs CSR 3.4 GB）；构建 `O(#templates·k²)`（亚秒 vs 63 s）。
  - 不规则地形（海南 land mask）：按「有效邻居集合签名」哈希分组，签名数 ~`O(周长·k)`。
  - **缓存 key**：topology（形状、`wrap`）、kernel size/support、`include_center`、`adjust_on_edge`、域掩膜；**不含** σ 数值/迁移率/生态参数。σ/环境变只**原地更新权重**（形状与 `Z` 不变）；**地形/掩膜变才重算类型映射**。
- **计算层：归一化卷积（边界正确）**
  - `Z = K' ⊛ m`（`K'`=去中心核，`m`=域内指示；预计算一次，只依赖形状，边界层外为常数）；每 tick `g = rate·f / Z`，`f_new = f·(1−rate) + K' ⊛ g`。
  - 与本仓库 CSR 的「丢掉越界 + 邻域重归一化」**逐项等价**（我们已验证 CSR 每行权重和恒为 1）。论文口径 `imfilter 'replicate'` 则不用 `Z`，改为边缘填充的线性卷积。
- **加速层（决策 2026-09-29）**：当前聚焦**论文的大核**情形 → **FFT 为优先实现**（核 FFT 预计算，每 tick `O(n log n)`；GPU 侧 cuFFT 按类 batched）；大域用 overlap-add。小核（`k≲log n`）下直接 stencil 更快（可 SIMD/缓存友好）——**记录为已知更优选项，当前不实现**（原型 `m4_routeA_prototype.py` 中两版仅为验证等价性）。核在六边形度量下**不可分离**，故无「一维×2」捷径。
- **约束**：FFT/stencil 改变浮点求和顺序，**不能与 CPU golden 逐位一致**；路线 A 必须是**默认关闭的可选迁移路径**，CPU CSR 继续当 golden 与小核路径，`phase0` 不受影响。原论文用的是直接 `imfilter`（stencil + `'replicate'`），**不是 FFT**；FFT 是此事实上的额外优化。

---

## 5. 分阶段复现里程碑（M0–M8）

> 每阶段：目标 → 输入/模型 → 输出 → 验证 → 风险级别。高风险阶段需独立 evaluator 复核（走 `EVALUATE.md`）。
> 全程遵守：CPU 是 golden reference，不得改其数值语义；`phase0` 必须始终 bit-identical；gpu feature 默认关闭。

### M0 — 范围冻结与外部 golden（低风险）
- **目标**：确定优先实验与规模；生成外部 golden。
- **动作**：跑 MATLAB 参考（`wavespeed_and_waveshape`、`radial_release_optimizer`）导出波速/波形与覆盖率；核对 `HexGrid` 度量与 MATLAB 距离公式的坐标映射（符号/轴变换）；记录 `drive_params.mat` 字段语义。
- **输出**：`hex_recon/golden/`（CSV/NPZ）+ 映射说明。
- **验证**：MATLAB 复跑自身可复现；文档坐标等价性证明（手推一组偏移距离相等）。
- **交付物**：M0 报告 + golden 数据。
- **状态（2026-09-28）**：首份 golden 与复现骨架已产出（实际路径 `hexagon_spatial_test/{repro,golden,results}`；参考实现用 ZIP `wave justification/{flat,junction}`）；坐标等价证明由 evaluator 独立补齐（natal `dr²+dc²+dr·dc` 与 MATLAB `Δx²+Δy²−Δx·Δy` 在 `(dr,dc)=(Δx,−Δy)` 下逐点相等）。`hexagon_spatial_test/EVALUATE.md` §2 = **APPROVED**。

### M1 — 遗传系统单点对照（中风险：数值语义）
- **目标**：在 natal 中表达 5/2/25 基因型系统的**一步后代转移**，与 `renew_function` 对照。
- **动作**：把 `mats` 映射为 natal `offspring_tensor (Z,Z,Z)`（或组合 presets）；对随机亲本频率验证一步新生儿分布。
- **输出**：对照脚本 + 误差表。
- **验证**：确定性一步分布逐元素相对误差 < 1e-6（若逐位可达成则逐位）。
- **风险**：`gn=25`（2-locus）→ `Z=25 ≤ 32` ✓，但 `offspring_tensor (25,25,25)=15625` 可写；需确认 natal 的性别/合子语义与 `mats` 对齐。
- **交付物**：遗传映射子模块 + 测试。

### M2 — 均匀 hex 波速/波形（中风险）
- **目标**：小场地（如 60×60 / 120×120）复现「左 20% 释放 → 波前推进」，测波速/波形，并对比 PDE 参考。
- **动作**：`HexGrid` + `build_gaussian_kernel(size=11..15)`（先不截断，记录与论文核差异）；deterministic；natal CPU vs MATLAB（同场地/同核）。
- **输出**：wave speed（50%/60% 检查点）、波形剖面。
- **验证**：natal CPU 与 MATLAB 波速相对误差目标 < 5%（先定性，再收紧）；两方向（flat/junction）各测。
- **风险**：核截断差异、边界 `replicate` 缺失、坐标映射。
- **交付物**：`hex_recon/wavespeed/` + 报告。
- **M2 前置口径（M0 + evaluator §2.8 冻结项）**：① 场地/检查点口径三方不一——论文正文 300×300、50%/60%；发布 flat launcher 200×200、40%/70%；PDE launcher 写 300 而随附 `.mat` 为 n=100/60/60/40——需先冻结；② 低 avd（≲0.5）离散核有效扩散远低于标称 avd，应同时报告核有效均值 `Σ(w·d)/Σw` 或改用等效扩散对照；③ `flat/junction` 命名与 √3/2 归属存在作者自注歧义，需明确；④ PDE（domain units/time）与 hex（cells/generation）速度单位不同，需给显式换算。
- **状态（2026-09-28）**：**完成**（`EVALUATE.md` §5 = APPROVED）。口径冻结为 paper(300×300,50/60%) 主 + code(200×200,40/70%,junction L=600) 交叉；采用「自写实现 + natal 空间原语（`HexGrid`+`build_gaussian_kernel`）」，非引擎生命周期。natal 复现与 MATLAB golden **12/12 在机器精度内一致**（|rel| ≤ 4.8e-12）；核与 MATLAB 核列镜像 ≤1.1e-16；方向 `flat×√3/2/junction` paper 0.993–0.998；有效核均值 0.2065/0.950/1.988。产物：`repro/{gen_golden_m2_wavespeed.m,natal_hex_wavespeed.py,compare_m2.py,verify_m2.py}`、`golden/hex_homing_wavespeed_m2.*`、`results/M2_wavespeed_report.md` 等。注：首版因 1-based 索引不忠实被 §4 判 NOT APPROVED，修复后 §5 通过（见 EVALUATE §3.6/§3.7）。

### M3 — 宽核与规模实测（中风险，决断前置）
- **目标**：量化 CSR 折叠在宽核/大格点下的内存/时间；核定 GPU 可行性。
- **动作**：对核尺寸 `{3,5,11,21,51}` × 场地 `{30²,60²,120²,300²}` 测构建时间、nnz、CSR 内存、CPU tick 时间；deterministic GPU `enable_gpu` 成功率与显存（`migration_cache_bytes` 实测）；随机 GPU 行宽拒绝验证。
- **输出**：规模/内存曲线 + 决断建议。
- **验证**：超预算必须显式报错（不得静默回退）；`phase0` 不变。
- **交付物**：`hex_recon/scale_report.md`。
- **状态（2026-09-28，CPU 侧）**：`results/M3_scale_report.md` + `m3_scale_data.json`。实测：CSR 折叠纯 Python、约 3.3–3.6e6 entries/s；`CSR ≈ 16·nnz B`；300²×51² = 2.15e8 nnz / 3.43 GB / 63 s；论文核 avd=1.0 = 1.95e8 nnz / 3.12 GB / 58 s；**确定性** CPU tick ≈ 2.7–3.0e-7 s/entry（300²×21² 已 11.3 s/tick）；海南 `2599×2601` 外推 ≈ 211 GB CSR / ~65 min 折叠 / ~66 min/tick → CPU 不可行。首版因 fold/builder 口径（`kernel_include_center`）与 stochastic 标签被 §7 判 NOT APPROVED，已修复并重跑（见 EVALUATE §6.6），§8 = **APPROVED**（CPU 侧闭环）。
- **状态（2026-09-29，GPU 侧）**：`results/M3_gpu_report.md` + `m3_gpu_data.json`。实测（RTX 5090）：deterministic `enable_gpu` 5/5 成功，迁移缓存 `≈4·nnz·A·Z²+…`（300²×21²=5.1 GiB），GPU tick 相对 CPU ~8–17×；**随机行宽守卫 `MAX_CSR_ROW=32`**（24 通过，120/440 拒绝）；**enable 期显存预算守卫**过预算显式报错（30011 > 19418 MiB）无回退；`phase0` bit-identical。见 EVALUATE §9，待 §10 复核。

### M4 — 决断门：迁移执行模型（高，需用户批准）
- 依 M3 数据在 A/B/C 中选择；若选 A，按 §4「路线 A 设计（候选）」推进：
  1. **原型验证（先做，低风险，repro 脚本）**：模板/标签展开与现 CSR **逐位一致**；归一化卷积/FFT 与 CSR **数学一致（~1e-12）**并与 M2 参考一致；给出内存/构建/每 tick 的对比。产物 `results/M4_routeA_prototype_report.md`。
     - **状态（2026-09-29）**：`results/M4_routeA_prototype_report.md` + `m4_routeA_prototype.json`。已验：模板/标签三元组与 CSR **逐条相同**（权重差 ≤1.3e-16）；归一化卷积（直接/FFT）与 CSR 逐格差 ≤1.0e-15；存储压缩 12–26×；每步成本 k≤5 直接 stencil 最优、k≥11 FFT 最优（300²×51²：FFT 3.3 ms vs 直接 26.9 ms vs CSR 代理 789 ms）。
  1b. **GPU cuFFT 概念验证**：`repro/m4_gpu_fft_prototype.py` + `results/m4_gpu_fft_prototype.json`（conda torch/cuFFT，非 natal）。GPU FFT 归一化卷积 vs CPU 直接卷积：f32 相对误差 ~2e-7、f64 ~1e-15；相对 CPU 直接卷积加速 5–70×（1000²×51²：GPU f32 3.6 ms vs CPU 226 ms）。
  1c. **解析模板构造 + 构建时间**：`repro/m4_template_build.py` + `results/m4_template_build.json`。矩形域解析签名 → 构建 `O(n)+O(#templates·k²)`：300²×51² **0.61 s vs fold 63.5 s（104×）**，存储 58.7 MB vs CSR 3274 MB（~56×）；行数/目的地/权重与 CSR 一致（≤1.4e-16；60²k11 与 120²k21 为**全量**，300² 两例为**抽样核验**（~56 行 + 角/边/中心），解析推导保证全体一致）。
   2. **产品化（高风险，需用户批准 + 独立 evaluator）**：以 **FFT 迁移内核**为主（CPU FFT + GPU cuFFT，按类 batched；GPU 为首选落点）+ 会话接线 + 模板/标签缓存持久化；**默认关闭**；中英文档同步；全量门禁（含 `phase0` bit-identical）。小核直接 stencil 暂不实现。
     - **设计提案（2026-09-29）**：`results/M4_routeA_productization_design.md`（架构/缓存 key·失效/数值与确定性策略/验证门禁/风险/落地步骤/待决）。
     - **实施（2026-09-29，用户确认 GPU cuFFT 路径）**：分阶段 S0–S4（见设计 §9）。
       **S0 已验**：cudarc 加 `cufft` feature + `rust/src/gpu/cufft.rs` 探针；`cargo test --features gpu gpu::cufft` = 1 passed（cuFFT 线性卷积 vs CPU 一致）；未改运行路径/契约。**S1（契约/前端）、S2（executor 接线）、S3（会话/文档/门禁）待做**；属高风险，需独立复核。
   3. 纳入根 `EVALUATE.md` 轮次。

### M5 — 径向释放优化（中）
- **目标**：以声明式钩子 `Op.add` 实现逐周/逐代释放；计算 ≥90% 覆盖面积与 efficiency；对照论文 Table 1 的**定性排序**（归巢驱动效率远高）。
- **动作**：参数扫描 `(t,h,r)` 的小网格版本；记录覆盖半径/面积。
- **验证**：natal CPU 与 MATLAB 同配置覆盖率一致（容差）；单调性/排序一致。
- **风险**：释放几何（中心圆）与坐标；`Δt=0.01` 前 5 代在 natal 无原生支持（可用更小 tick 或初始状态近似）。
- **交付物**：`hex_recon/radial/` + 报告。

### M6 — 海南岛地形/季节/轮渡/道路（高）
- **目标**：把 `terrain(2599,2601,12)`→逐月 per-deme K 栅格；`border` 掩膜；轮渡通道（额外迁移边）；沿道路释放。
- **动作**：用 `batch_setting`/`set_param` 注入月 K；用钩子做季节切换；桥接两点的额外迁移。
- **验证**：K 统计量（均值/极值）与论文一致；人口覆盖用 `pop.mat` 加权。
- **风险**：`drop.mat`/`release_m.png` 缺失 → 需用户提供或自建释放方案；规模取决于 M4 决断。
- **交付物**：`hex_recon/hainan/` + 报告。

### M7 — GPU 旁路测试（高）
- **目标**：在 GPU 空间路径上跑 M2/M5/M6 的可运行配置，做**正确性 + 性能**评估。
- **动作**：deterministic GPU vs CPU（相对误差档）；tick 时间、迁移占比、enable/显存；`nvidia-smi` 快照；网格/核尺寸阶梯。
- **验证**：`phase0` bit-identical；GPU-vs-CPU 在精度分档内；超预算显式报错。
- **交付物**：GPU 性能/正确性报告（可复用 `demos/` 基准范式）。

### M8 — 论文图/表复现（中）
目标：把各实验对齐论文的图/表并给出对照评级。映射（图像在 `hex_model_2026_paper/figures/`）：
| 论文产物 | 内容 | 对应里程碑 | 对照口径 |
|---|---|---|---|
| **Figure 1** C/D | Culex 扩散核（热图/按距离曲线，到 25） | M0/M3 | 核与 `get_mig_matrix` 逐元素一致（相对误差） |
| **Figure 3** | 波速 vs 平均扩散（Homing/TARE/TADE GE/CifAB × PDE/flat/junction） | M2 | 曲线形状与量级；PDE 反超点 ~avd 1.2–1.4 |
| **Figure S3/S5** | 两方向各向异性、波形（dispersal 0.5/10） | M2 | flat vs junction 差异；高扩散三曲线重合 |
| **Figure S6/S7** | 适合度/参数 → 波速（阈值行为） | M2 扩展 | CifAB 阈值 ~0.85；TADE 胚胎切割越强越慢 |
| **Table 1 / Table 2** | 圆形/线性释放优化参数与效率 | M5 | 定性排序（归巢效率远高）；数值受场地差异影响 |
| **Figure S8/S9** | 释放半径/密度 → 扩散面积响应面 | M5 扩展 | 单调性与阈值差异 |
| **Figure 2/S1/S2** | 海南 K 图（年度/月度/输入图层） | M6 | K 统计量（均值/极值）与论文一致 |
| **Figure 4/S4/Table 3** | 海南网格阵列释放 | M6/M8 | 覆盖率、效率、释放量量级 |
| **Figure 5/Table 4** | 海南道路释放 | M6/M8 | 同上 |
| **Figure 6** | TARE 手工优化（90.94% 人口覆盖 / 699270） | M6（需释放掩膜） | 仅定性（缺 `drop.mat`） |
明确标注“可复现 / 部分可复现 / 受表达或数据限制”三档。

---

## 6. 验证方法论

- **金标准层级**：MATLAB 参考（外部）→ natal CPU（本项目 golden）→ GPU（旁路）。
- **确定性口径**：论文无随机，默认 `stochastic=False`；natal CPU 与 MATLAB 用相对误差；GPU 对 CPU 用 f32 分档（纯搬运/整数逐位；含算术 ~1.2e-6；本模型含除法/指数，按相对误差）。
- **遗传一步**：逐基因型新生儿比例对照（相对误差）。
- **波速**：50% 波前过两检查点的时间差（离散采样可能需插值）；两方向都测。
- **覆盖率**：≥90% 频次/抑制的 hex 计数；释放效率 = 面积/释放数；人口覆盖用 `pop.mat` 加权。
- **回归底线**：任何改动不得触发 `phase0` 变化；CPU 数值语义不可改；新特性默认关闭并独立复核。

---

## 7. GPU 旁路测试设计（要点）

- **正确性**：deterministic spatial GPU vs CPU，逐 deme/年龄/基因型对照；核尺寸/网格尺寸阶梯；边界与释放几何。
- **性能**：分点计时（构建 CSR、enable、每 tick、迁移占比、下载）；`nvidia-smi` 前后；与 CPU 多核对照。
- **预期瓶颈（须实测量化）**：
  1. 迁移 cache 显存 `~nnz·A·Z²·4`（宽核/大 Z 致命）；
  2. 随机迁移行宽 32 上限（若需随机）；
  3. 宽核 CSR 构建/内存。
- **结论用途**：为 M4 决断与用户优先级提供数据。

---

## 8. 交付物、仓库布局、流程与实现原则

- **分支**：`recon/hex-model`（复现工作在此分支进行）。
- **gitignore 策略（已调整）**：`hexagon_spatial_test/` 只屏蔽**信息来源**——`*.pdf`、`Hex-model-main/`、`hex_model_2026_paper/`、`hexagon_sptial_test.md`；**复现脚本与产物保留在跟踪树中**。
- **产物布局（均在 `hexagon_spatial_test/` 下）**：`repro/`（脚本）、`golden/`（MATLAB/ZIP 外部参考）、`results/`（产出与报告）。
- **流程与沟通（与 GPU 旁路一致）**：实现 → 自测 → 文档 → 独立 evaluator 复核（高风险）→ 门禁。复现项目**独立通道**为 `hexagon_spatial_test/EVALUATE.md`（主 agent 交接 §N ↔ evaluator 回执 §N+1）；**对 natal-core 产品代码的改动**仍走根 `EVALUATE.md` 与全量门禁。
- **实现原则（用户确认）**：
  1. 复现尽量是**另写脚本调用 natal-core 现有 API**，不激进改动引擎；
  2. 只有「natal 缺失且必须」的功能才新增，且新增必须具备**泛化能力**（服务同类模型，而非仅本项目）；
  3. 任何引擎改动走高风险流程：默认关闭、独立 evaluator 复核、`phase0` 始终 bit-identical。
- 不新增生产依赖；MATLAB 仅用于生成 golden，不进入门禁。
- 本文件是跨 agent 交接主入口；每个里程碑完成后追加状态，并在 `hexagon_spatial_test/EVALUATE.md` 记录交接/回执。

---

## 9. 决策状态与残余

已在 §0.1 冻结：首期范围（模块 1–3）、海南暂缓、迁移先 M3 实测、natal 改动边界、年龄近似、场地口径、MATLAB+确定性。

**残余（不阻塞首期，随进展确认）**
1. **遗传系统表达方式**：拟**直接写 `offspring_tensor (Z,Z,Z)`**（把 `mats` 映射过去）——最忠实且通用；若发现某些驱动更宜用 presets 组合再调整。
2. **目标规模**：波速按论文 300×300；径向/线性按论文 500×500（§0.1#6）。是否额外跑更小网格用于快速迭代，随 M2 决定。
3. **是否向作者索取缺失件**：暂不阻塞；若用户后续决定做海南再启动（`REPRODUCTION_NOTES.md` §4 方案 D）。
4. **`stochastic=True` 对照**：默认不做（论文确定性）；若要做需注意随机 GPU 迁移行宽 32。

---

## 10. 关键索引（快速定位）

**论文资料包（agent 可读）**
- 全文：`hexagon_spatial_test/hex_model_2026_paper/paper_fulltext.md`（按 `<!-- ===== PDF PAGE n ===== -->` 引用 p.n）。
- 阅读协议与页映射：`.../README.md`；图表索引：`.../figures_and_tables.md`；方法档案：`.../candidate_simulation_methods.md`；插图：`.../figures/`。
- ✅ 有全文派生包后，**不必依赖 PDF**；opencode 不能直接读 PDF 的限制已被绕开。

**MATLAB**
- 核心循环：`radial_release_optimizer/main.m`；海南：`hainan_model/main.m`；分析：`hainan_model/analyse.m`。
- 基因型重组：`<module>/<drive>/renew_function.m`；驱动参数：`<drive>/drive_params.mat`（`gn/mats/fitness/carrier_index/cord_*`）。
- 迁移核：`get_mig_matrix.m`；`mig.mat (59,59), avd=13.08`。
- 数据：`hainan_model/{terrain,pop,border}.mat`；预计算：`parameter_sensitive/radial/*.mat`。

**natal-core**
- `HexGrid`/`build_gaussian_kernel`：`src/natal/frontend/spatial/topology.py:251/343`。
- 迁移折叠：`src/natal/frontend/spatial/migration.py:242`（`fold_migration_csr`）。
- 空间 GPU 会话：`rust/src/sessions/spatial.rs`（`enable_gpu:351`、`run_gpu_tick:1289`、`run_steps:756`）。
- 行宽/维度上限：`rust/src/gpu/kernels.rs:56-63`（`MAX_AGES=64, MAX_Z=32, MAX_CSR_ROW=32`）；`executor.rs:2389`（stochastic 迁移守卫）。
- 迁移 cache 预算：`executor.rs:~3063`（`migration_cache_bytes`）。
- 声明式释放：`src/natal/frontend/hooks/entry/declarative.py:105`（`Op.add`）。
- hex 示例：`demos/spatial_hex.py`、`demos/spatial_hex_discrete.py`（501×501，CPU）。
- 空间基准范式：`demos/gpu_spatial/age_structured/reference_cpu.py`、`benchmark_scaling.py`。

**状态**：M0 起待启动；先补齐 §9 的决断。
