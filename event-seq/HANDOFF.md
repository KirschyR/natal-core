# event-seq / HANDOFF（tick 内事件顺序探究）

> 本文件只在本分支（`explo/event-seq`）内记录。**不要**写入外部的 `EVALUATE.md`、
> 根 `HANDOFF.md`、`GPU_insert_PLAN.md`、`GPU_STAGE_SUMMARY.md` 等交接文档。
> 本探究为专项，不一定会合并回 `recon/hex-model`；产出只在 `event-seq/`。
>
> 提交规范：一句话中文 commit message，scope 用 `seq-explo`（例如
> `feat(seq-explo): ...`）。**主 agent 不代为提交**，由用户执行。

---

## 1. 研究问题

参考 MATLAB 模型 `hexagon_spatial_test/Hex-model-main/radial_release_optimizer/main.m`
在一个 tick 内的状态改写顺序是：

```
① 密度调节+存活+老化（pagemtimes，消费上一批 age0）
② 繁殖写入新生（覆盖 age0）
```

即 **S-A-R**（密度/存活 → 老化 → 繁殖写入）。

natal-core 的 `AgeStructuredPopulation`（`rust/src/kernels/age_structured.rs`）是：

```
① 繁殖写入新生（age0）
② 密度调节+存活（作用于本 tick 新生 age0）
③ 老化（age0→age1，清空 age0）
```

即 **R-S-A**。

二者**每个阶段的公式几乎相同**，差别只在**一个 tick 内三个事件的先后**，由此产生：

- **age0 语义**：S-A-R 的 age0 是**跨 tick 常驻的卵池**（约 40）；R-S-A 的 age0 是**同 tick 瞬态槽**（边界恒为 0）。
- **密度反馈相位**：S-A-R 用**上一 tick** 的 age0 做密度反馈；R-S-A 用**本 tick** 新生。差一步。
- **幼期时长/世代时间**：R-S-A 的新生在同 tick 就进 age1，幼期少一步 → 世代更短。

目标：量化这种「事件顺序 / 延迟密度调节」对（非空间与空间）模拟结果的影响。

---

## 2. 关键结论（已完成）

1. **判别实验**：把参考按 natal 顺序重排（R→D→A）后，其覆盖 **≈ natal 引擎**（配置 A
   差 1.7%、配置 B 差 2.0%），而原生参考与两者差 **~40%**（空间/无空间皆然）。
   → **引擎与参考的差异几乎全部来自 tick 内事件顺序**，非 bug、非标定错误。
   产物：`repro/../hexagon_spatial_test`（M5 侧）与 `event-seq/` 的脚本。
2. **顺序影响的是「时机/速度」，不是最终结局**：长时间后 R-S-A 与 S-A-R 的最终固定率趋同；
   差异体现在 T50、给定 tick 的均值、以及空间波速/覆盖。
3. **R-S-A 传播更快**（非空间：T50 更小、同 tick 均值更高；空间：波前更早、覆盖曲线左移）。
4. **随机性**：natal 风格**连续采样**（Beta/Gamma），`n≤1` 时退化为确定性均值。因此
   - sf 较小时小类**无噪声**，只有大计数（卵）有噪声；
   - sf 增大后更多类进入采样 → 噪声来源增加；N 足够大时相对噪声 ∝ 1/√N 变小，**顺序效应显现**。
5. **适合度代价**：`viability_dd`（末幼期 age1）与 `fecundity_dd`（繁殖）都拖慢 drive。受控对比
   （`dc=0.8, sf=10, ticks≤200, 100 次`，`Δ=RSA−SAR 的最终固定率`）：无代价 `+0.01`；仅
   `viability=0.8` `+0.02`；仅 `fecundity=0.8` `+0.07`；两者皆 0.8 `+0.09`。→ **fecundity 代价更显著地
   放大顺序差，viability 单独较弱，二者叠加最大**。（机制：代价使 drive 更勉强、早期随机丢失增多，
   「R-SA 建立更快」的优势更凸显。注：此为**最终固定率**口径、对时机不敏感；顺序主效应是时机。）
6. **初始额外队列的线性清除**：只加雄性时，额外 dd 队列随 5 个雄成年龄线性老化至死，
   总量每步减 `X/M`（X=额外量，M=5），**恢复时间由释放年龄决定（投 male0 则 5 tick），与数量无关**；
   这段线性下降与 Beverton–Holt 无关（BH 只钉平衡高度）。
7. **age0 记账差**：若把 age0 计入「幼体」，两顺序的幼体数差约 `1/s1 ≈ 90×`。故本探究把
   - `E`（卵）定义为**本 tick 繁殖产生的新生队列**（相位无关）；
   - `J`（幼体）**只取 age1**；`A` 为成虫。
8. **只有 `R-S-A` 与 `S-A-R` 自洽**：其余排列会让 age0 队列绕过密度调节（S）直接老化，
   未调节卵池泄漏 → 数值爆炸（可达上万倍）。
9. **「不固定但存在」的共存区容易找，且顺序差主要体现在动态上**：homing + **足够大的适合度代价**
   会落到**内部共存平衡**（如 `dc=0.6` 时 `cost∈[0.3,0.6]` → dd 平衡 0.26–0.83，是连续带而非刀锋区）。
   该区**终值两顺序几乎相同**（如 dc=0.6/cost=0.4：0.354/0.354），但**趋近共存的动态差很大**
   （同配置 t=80 全局 dd/(非卵)：R-S-A 0.116 vs S-A-R 0.031）。这是**均匀共存**，不是严格 chasing；
   严格 chasing（波钉扎/Allee）需**阈值/抗性**机制（如 TARE），见 §2#12–14。
10. **「启动慢」与「速度慢」同源**：事件顺序只改**局域有效增长率 `r`**（迁移核/`D` 两顺序**相同**）；
    波速 ∝ `√r`、启动时间 ∝ `1/r`，故二者一起变（`dc=0.8, avd=1`：R-S-A 前沿拟合 t0≈36、速度 0.79；
    S-A-R t0≈48、速度 0.64 cells/tick）。要「分开」只能独立改 `D`（核）或种子大小。
11. **迁移核 = 中心点采样**（natal `build_gaussian_kernel` 与 paper 一致）：`mean_dispersal=avd → sigma=avd/√(π/2)`，
    `w[dr,dc]=exp(-dist²/2σ²)`（hex 度量 `dr²+dc²+dr·dc`）后归一化。与「对 deme 面积积分」的差在
    `σ≫格尺寸`很小、`σ≲格尺寸`较大（avd=0.5 有效均值 0.21 vs 0.42）。**已接受中心点方案**
    （想更接近连续就缩小 hex 尺寸），不再纠结该细节。
12. **TARE 已实现并与 paper 逐位对齐**：`age_model.tare_mats(gcr, eecr, ddfitness)` 逐字翻译
    `radial_release_optimizer/tare/drive_generator.m`，与仓库自带 `tare/drive_params.mat` 的 `mats`
    **逐位一致**（max|diff| = 0）。两个与 homing 不同的约定：① TARE 生成器**不转置**原始矩阵（homing 转置），
    即 `mats[k*5+i, j]`=母本 `i` × 父本 `j` 的后代 `k` 权重，直接以母/父次序使用；② 原始基因型序是
    `[ww, dw, dd, rw, dr]`，导入时按 `dd<->ww`（`_TARE_TO_CANON`）置换成规范序 `[dd, dw, ww, rw, dr]`
    （block 后代轴本就是规范序，无需重排）。副作用：TARE 的 `dd×ww` 与 `ww×dd` **不对称**（母本早期胚胎
    切割 `eecr` 造成，`dd(母)×ww(父)→dr`），是 paper 模型自身的性质，非翻译错误。开关
    `drive∈{"homing","tare"}` 已加入 `age_model.run`/`Simulation`/`drive_mats` 与 `spatial_test.run_replicate`；
    仅 TARE 用 `gcr/eecr`（默认 1/1），homing 用 `dc`。
13. **TARE 在本模型里是「固定 / 丢失」双态：没有真正的野生型共存**（**修正早前误判**）。注意 `classify`
    的 `W` = **非 dd**（含 `dw/rw/dr`），**不等于**野生型 `ww`；早前把「非 dd 平衡」当成了 WT 共存。
    以真正的 `ww`（规范序 index 2）复核：well-mixed（seed `0.3 dd`，`t=1500–2000`）扫描
    `gcr=eecr∈{1.0,0.8,0.6,0.5,0.4}`、`ddfitness∈{1.0,0.6,0.4}`（含逐基因型代价组合）后，**`ww` 一律
    趋于 0（固定）或回到 1（丢失），没有内部稳定共存**；`ddfitness` 只是把 `dd` 换成 `dr`（抗性载体）、
    代价过大时驱动崩溃，并不保住 `ww`。**给抗性 `rw` 加代价违反科学假设**（靶基因为必需**单倍足量**基因，
    `rw` 中性），故**不采用**该做法。注意 `rr` 隐性致死**已经隐含**在 paper 的 mats 中：母本为载体的组合
    （`dw/rw/dr` × `dw/rw/dr`）后代总和 <1（`0.5–0.75`，见 §5#14），即毒素以「减少后代总数」形式编码，
    而非显式 `rr` 基因型。但即便如此野生型仍无法在高频下反超——`r` 由切割产生、与驱动连锁，高切割率下载体
    不产 `ww`。→ 严格 chasing 所需的「驱动与 WT 空间共存、不固定」在本（确定性连续）模型里**没有对应参数区**，
    需另加机制（见 §6#8）。空间上 TARE 仍表现为**播种阈值 + 长潜伏**（§2#14）。
14. **事件顺序显著移动 TARE 阈值/潜伏期**（同样释放，`R-S-A` 比 `S-A-R` 更早成波）：
    释放 `0.9×`：R-S-A 终值 carrier `0.72` vs S-A-R `0.01`；释放 `1.0×`、`t=600`：R-S-A `0.985` vs
    S-A-R `0.022`；释放 `1.25×`、`t=600`：S-A-R 处在**驱动 `0.35` / WT `0.48` 的空间共存潜伏期**
    （前沿半径 `r≈17`，`t=800` 才扫过），而 R-S-A 已扫过。强 fecundity 代价
    （`viability=1.0, fecundity=0.4`，释放 `1.5×`，`t=2000`）下 R-S-A `1.00` vs S-A-R `0.03`，佐证 §2#5。
    产物：`event-seq/tare_chasing.py`、`img/spatial/tare_threshold.png`、`tare_latency.png`、`tare_chasing.json`。
15. **TARE 有强性别特异传递，释放性别极大影响效率**：TARE 矩阵本身不对称（母本 `eecr` 效应）：
    `dd(母)×ww(父)→dr`，而 `ww(母)×dd(父)→dw`；homing 基本对称。实测（TARE，rel=2、rad=3、t=600）：

    | 释放性别 | carrier | WT |
    |---|---|---|
    | 仅雄 | 0.628 | 0.259 |
    | 仅雌 | 0.965 | 0.020 |
    | 双性 | 1.000 | 0.000 |

    TARE **雌性释放效率远高于雄性**，解释了既有 M5/event-seq「仅雄释放」对 TARE 低效；也说明假设两性
    对称传递会误判 TARE 释放效率，后续研究 TARE 波/chasing 应扫性别比或采用双性释放。
16. **显式 `rr`（6 基因型）确定性测试：忠实实现 ≡ 5 基因型，不产生 chasing**。`event-seq/tare_rr.py` 把 `rr`
    显式作为第 6 个基因型：后代矩阵 = paper `tare_mats`（5 型）+ `rr` 取「缺失质量」`1-Σ_k`（即隐性致死类；
    已验证每对亲本的 6 个后代和为 1）。
    - **忠实实现**（`rr` 在繁殖/合子期移除，`drop_rr=True`）与 **5 基因型模型逐位一致**（`rtol<1e-9`，自测）：
      paper 的 `renew` 本就只返回存活后代，`rr` 从不进入个体、不参与密度反馈。故**显式 `rr` 本身不改变确定性结果**。
    - **若把 `rr` 计入密度反馈后再死**（延迟致死，`rr_age=0/1`）：等于额外、频率依赖的抑制，会**放慢驱动波**
      并放大顺序差（`t=300`、rel=2、rad=3、avd=0.5：R-S-A carrier 5-gen `0.899` → rr@age0 `0.709`；
      S-A-R `0.352` → `0.105`；总种群约低 2–8%）。但 `t=1000` 两顺序仍都扫过（`ww→0`），**仍无 WT 共存**。
    结论：**显式 `rr` 单独不能产生 chasing**；只有「延迟致死」这种抑制解释才改变动态，且只改时机、不改结局。
    产物：`tare_rr.py`、`img/spatial/tare_rr_carrier.html`（交互六边形，R-S-A|S-A-R）、`tare_rr_compare.png`、`tare_rr_report.json`。
17. **无空间 TARE 的最终状态（精确）**：well-mixed（单群体、年龄结构、BH 密度）里 TARE 成功时 **`ww`（真野生型）
    被完全清除**（`ww→0`），留下**全是携带驱动的个体**；`ddfitness<1` 时终态是**稳定的 `dd + dr` 多态**（不是纯 `dd`），
    `ddfit=1` 时退化为纯 `dd`（`dr` 以 ~1/t 极慢衰减）。实测（`seed=0.3 dd`，`t=3000`，非卵分数）：

    | `ddfitness` | `dd` | `dr` | `ww` | 非卵总量 |
    |---|---|---|---|---|
    | 1.0 | 0.997 | 0.003 | 0 | 2.11 |
    | 0.8 | 0.900 | 0.100 | 0 | 2.00 |
    | 0.6 | 0.800 | 0.200 | 0 | 1.74 |
    | 0.4 | 0.698 | 0.302 | 0 | 1.39 |
    | ≤0.3 | 0（驱动失败） | 0 | →1.0 | 2.1 |

    `dd:dr` 由平衡决定：`dr` 比 `dd` 更适（代价 `F` vs `F²`），但 `dr×dr` 有 25% 产致死 `rr`，二者抵消。**`dr` 是
    「驱动+功能破坏抗性」载体，不是 WT**。均匀小种子（0.1%）即可固定（well-mixed 阈值≈0，与 paper 一致）；
    此前的「播种阈值」是**空间成核**（§2#14）。代价越大群体越被压低（`ddfit=0.4` 非卵总量 −34%）。
    → **无空间 TARE 终态 = 全 drive（`dd+dr`，比例由 `ddfitness` 定）且 WT 灭绝，或驱动失败回到纯 WT；
    没有 WT 与驱动的稳定共存。** 这也解释为何空间波过后后方没有 WT 可追（无 chasing）。

---

## 3. 模型与参数

### 3.1 非空间（`age_model.py`）

- 单群体、无空间；10 年龄（0–1 幼，2–9 成虫；雌用 2–9，雄用 2–6，雄 7–9 零填充）。
- 基因型（paper 序）`[dd, dw, ww, rw, dr]`；理想 homing（`dc`、`gcr=0`、`herr=0`、`ddfitness=1`）。
- 参数：`eggs=40`、`λ=5`、`sf=4.5`、`sex_ratio=0.5`、`fixed_egg_count=True`、
  `sperm_displacement_rate=1.0`（使精子库每周刷新，等价参考的无状态重配）。
- 存活：雌 `[1,1,7/8,6/7,5/6,4/5,3/4,2/3,1/2,0]`；雄 `[1,1,4/5,3/4,2/3,1/2,0,0,0,0]`。
- 密度：`s1 = λ/((λ−1)r+1)·2/(sf·eggs)`，`r=(j0+5·j1)/D`，`D=40+5·(4/9)=42.222`（由初始态派生，随 `scale` 缩放）。
- 初始（WT）：雌 `[20, 0.2222, …]`、雄 `[20, 0.2222, …]`（= 空间 M5 每 deme 初值）。
- drive 投入（非空间）：`Drive|Drive` 雄性于 age2，量 = `0.5×INITIAL_MALE[age2]`（随 `scale`）。
- 适合度：`viability_dd`（末幼期 age1）、`fecundity_dd`（繁殖，作用于雌+雄 dd）。
- 开关：`stochastic`（natal 连续采样）、`scale`（种群放大，`D` 同步缩放）。
- **分类**：`{E,A,J}_{F,M}_{W,D}` → `E`=本 tick 新生队列、`J`=age1、`A`=age≥2；`Drive`=`dd`，其余野生。
- 事件顺序可配（`R-S-A`/`S-A-R`/任意排列），**繁殖一律读 tick 起始成虫**（与参考「先算后转」和 natal「R 最先」一致）。

### 3.2 空间（`spatial_test.py` + `spatial_html.py`）

- 拓扑：`41×41` 六边形度量网格（`HexGrid` 度量 `dr²+dc²+dr·dc`）。
- 迁移核：`build_gaussian_kernel("hex", size=7, mean_dispersal=1.0)`。
- **边界 = natal 归一化卷积**（不是 replicate）：`K'`=去中心核，`Z=K'⊛m`（`m`=域内指示），
  `g=rate·f/Z`，`f_new=f(1−rate)+K'⊛g`；对每 `(sex, age, genotype)` 成虫平面施加；迁移固定 tick 末尾。
- 其余同非空间；`DC=0.8`、`VIABILITY_DD=0.8`、`FECUNDITY_DD=0.8`。
- 初始：WT 平衡 + 中心圆（半径 3）内 `Drive|Drive` 雄性于 age2，量 = `0.5×INITIAL_MALE[age2]`。
- **六边形渲染**必须用 natal `HexGrid.to_xy`：`x=i+0.5j`、`y=(√3/2)j`（i=行、j=列，pointy-top）；
  这样 `di²+dj²+di·dj` 等于欧氏距离，扩散波各向同性（域为平行四边形，是 natal 坐标系的自然形状）。
  用 offset/odd-r 等其它布局会**拉伸**度量、使波前变形。

---

## 4. 脚本与产物

脚本（`event-seq/`，纳入版本跟踪）：

| 文件 | 作用 |
|---|---|
| `age_model.py` | 非空间模型：参数、确定性事件、连续采样随机模式、12 分类 |
| `run_orders.py` | 非空间确定性出图（每顺序一张 4×3=12 子图）+ 总数叠加图；默认 `R-S-A`/`S-A-R` |
| `stochastic_test.py` | 非空间随机：100 次均值 ± 区间；`SCALING_FACTOR` 默认 10，`--scaling-factor` 可调 |
| `spatial_test.py` | 空间模型（natal 归一化迁移）+ 覆盖曲线图（`COVERS={0.1..0.9}`） |
| `spatial_html.py` | 六边形热图交互式 HTML（确定性 + sf10/sf100 随机均值） |
| `tare_chasing.py` | TARE 播种阈值/chasing 实验：阈值扫描 + 近阈值潜伏期（R-S-A vs S-A-R），自测含 `tare_mats` 对齐 paper |
| `tare_rr.py` | 显式 `rr`（6 基因型）确定性测试：5-gen vs `rr@age0/1`，交互六边形 HTML；自测含「忠实 rr ≡ 5-gen」 |

产物（`event-seq/img/`，**已 gitignore**，不入库）：

- 非空间（`img/non-spatial/`）：`order_RSA.png`、`order_SAR.png`、`total_deterministic.png`、
  `order_RSA_stoch_sf10.png`、`order_SAR_stoch_sf10.png`、`total_stochastic_sf10.png`、
  `order_RSA_stoch_sf100.png`、`order_SAR_stoch_sf100.png`、`total_stochastic_sf100.png`。
- 空间（`img/spatial/`）：`spatial_coverage_curves.png`（确定性）、`..._sf10.png`、`..._sf100.png`；
  `spatial_event_order.html`（确定性）、`..._sf10.html`、`..._sf100.html`；`spatial_event_order.npz`。

当前参数（脚本内常量）：

| 脚本 | dc | viability_dd | fecundity_dd | ticks | 其它 |
|---|---|---|---|---|---|
| `run_orders.py` | 0.6 | 0.8 | 0.8 | 150 | 确定性 |
| `stochastic_test.py` | 0.6（默认 `--dc`） | 0.8 | 0.8 | 200 | runs=100，sf=10（默认） |
| `spatial_test.py` | 0.8 | 0.8 | 0.8 | 120 | grid 41²，kernel=7，avd=1，release 0.5×age2 |
| `spatial_html.py` | 0.8 | 0.8 | 0.8 | 120 | runs=100，sf∈{10,100} |
| `tare_chasing.py` | —（TARE gcr=eecr=1） | 0.8 | 0.8 | 600/1000 | grid 41²，avd=0.5，**双性**释放 dd，中央圆 r=3，release∈{0.5..2.0}；carrier={dd,dw,dr} |
| `tare_rr.py` | —（TARE gcr=eecr=1，6 基因型） | 0.8 | 0.8 | 600 | grid 41²，avd=0.5，双性释放 dd rel=2、r=3；`rr` 致死（v=0，age0/age1 两种） |

覆盖曲线口径：每 deme 的 `dd` 占**非卵个体**（age≥1）比例 `> cover` 的 deme 数；随机图给出
**mean + 95% 运行分布区间（2.5–97.5 分位）**。

### 4.1 计算后端（重要）
event-seq 脚本是**纯 NumPy 自实现**，**不调用 natal 的 Rust 引擎**（`_engine_rs`）。唯一用到 natal 的是
`from natal.frontend.spatial.topology import build_gaussian_kernel`（`topology.py` 里的**纯 Python** 函数，只算迁移核）。
原因：natal 引擎的 tick 内事件顺序**固定为 R-S-A**，无法重排；为支持任意 R/S/A 顺序，生命周期在 NumPy 里**逐阶段复刻**。
（真正调用 natal 引擎/route A 的是 `hexagon_spatial_test/` 下的 M5 探针，与本分支无关。）

---

## 5. 注意事项 / 坑

1. **不要动外部交接文档**；只在本分支 `event-seq/` 记录。
2. **不代提交**；commit scope 用 `seq-explo`。
3. **只有 R-S-A / S-A-R 自洽**；其它排列会导致 age0 绕过密度调节 → 发散。
4. **连续采样阈值**：`n≤1` 返回确定性均值——小种群/小类无随机性；要得到有意义的随机性需放大 `sf`。
5. **空间迁移率=1 + 弱 drive**：10% 释放无法克服扩散稀释；需更大释放（本探究用 `0.5×age2` 于中心圆 3）。
6. **六边形渲染**必须匹配 natal `to_xy`，否则度量被拉伸。
7. **`pop.run()` 默认 `record_every=1` 会每 tick 存全量快照**（在 M5 侧 `m5_engine_probe.py` 曾 OOM，须用 `record_every=0`）。
8. **img 目录已 gitignore**；若曾跟踪过其中文件，需 `git rm --cached`。
9. 环境：`source .venv/bin/activate`；跑 natal 相关时导出 `LD_LIBRARY_PATH`（含 `/opt/conda/lib`）；
   跑 pytest/phase0 前 `unset PYTHONHOME PYO3_PYTHON PYTHONPATH`。
10. Git/GitHub：远端 `explo/event-seq` 需在 `remote.origin.fetch` 加一条 refspec 才能被 VSCode 识别为跟踪分支
    （否则显示 “Publish Branch”）；本地需显式 refspec fetch 建立 remote-tracking ref 才能识别跟踪分支。
11. **驱动类型（TARE，已实现）**：`age_model.tare_mats` = paper
    `Hex-model-main/radial_release_optimizer/tare/drive_generator.m` 的 `(25,5)` 翻译，与 `homing_mats` 同结构
    （drop-in），并已对齐仓库自带 `tare/drive_params.mat`（见 §2#12）。**不要**用 natal 的 `ToxinAntidoteDrive`
    （6 ztype 含 `R|R`、有性别、传递与致死分离，接入我们 5 基因型模型代价大）。要点：① paper TARE 基因型序是
    `ww,dw,dd,rw,dr`（homing 是 `dd,dw,ww,rw,dr`），导入时按 `dd<->ww` 归一到规范序；② TARE 生成器**不转置**
    （homing 转置），照原样以母/父次序使用；③ TARE 的 `dd×ww`/`ww×dd` 不对称属模型自身，非 bug。
12. **避免适合度重复计**：paper 把代价折进 `mats`；natal 放在 `viability_fitness`；我们模型另有 `viability_dd`/`fecundity_dd`。
    **同一份代价只在一个地方施加**（用 paper `mats` 就别再加同项 viability）。natal 里 `viability_scaling`（预设）与
    `.fitness(viability=...)` 写**同一张表**，构建次序 presets→fitness，后写覆盖前写。
13. **`classify` 的 `W` = 非 dd，不是野生型 `ww`**：`classify` 把除 `DRIVE_GENOTYPE`（=dd）外的**所有**基因型
    都归为 `W`（含 `dw/rw/dr`）。统计「与驱动对立的野生型」时必须直接按基因型 index 2（`CORD_PURE_W`=ww）取，
    否则会把抗性/杂合载体也算成野生型（曾因此误判 TARE 有 WT 共存，见 §2#13）。`tare_chasing.py` 的 `wt` 用的是真 `ww`。
14. **靶基因单倍足量 → `rw` 中性；`rr` 致死只以「减少后代总数」隐含在 mats**：`age_model.tare_mats`（paper）
    中 `sum_k m3[k,i,j]`（母 `i` × 父 `j` 的后代总和）对载体近交组合 <1——例如 `dw×dw=0.75`、`dw×ww=0.50`、
    `dr×ww=0.50`、`rw×rw=0.75`（homing 除 `rw/dr` 外恒为 1）。即毒素（`rr` 不可存活）按**减少该组合的后代
    总数**编码，5 基因型模型**不含显式 `rr`**。因此：**不要**再给 `rw` 加 fitness cost（违背单倍足量假设）；
    若要显式刻画 `rr` 的种群抑制，需要扩基因型或加密度项——属改科学模型，须授权 + 独立复核。

---

## 6. 待办 / 可继续的方向

1. **统计显著性**：对「顺序差异」做 bootstrap / permutation（例如 T50、覆盖数），而非只看均值。
2. **时间维度指标系统化**：T50/T90、固定 tick 均值、AUC，而非最终固定率。
3. **参数扫描**：`dc`、`viability_dd`、`fecundity_dd`、`release` 量/年龄、`kernel` 大小、`mean_dispersal`。
4. **尺度扫描**：给出「顺序效应 Δ vs sf」曲线，量化信噪比。
5. **对接 M5 空间规模**：在完整 `500×500`（route A GPU）上验证顺序对波速的影响是否稳健。
6. **natal 引擎交叉验证**：用 natal `SpatialPopulation`（R-S-A）与自写空间实现对比，确认语义一致。
7. **线性化解析**：推一 tick 反馈滞后对早期增长率/波速的解析修正。
8. **加 TARE 驱动、找严格 chasing（已落地，见 §2#12–16）**：`drive∈{"homing","tare"}` 开关、`tare_mats`
   （对齐 paper）、`tare_chasing.py`、显式 `rr` 的 `tare_rr.py` 均已完成。结论：本（1 位点 5 基因型，及忠实
   6 基因型）TARE 在确定性连续模型里只有「固定 / 丢失」两态，**没有 WT 共存**；空间表现为**播种阈值 + 长潜伏**，
   事件顺序显著移动阈值/潜伏期。
   **要得到真正的 chasing（持续移动、与 WT 共存、不固定），需要新增机制**，候选（按推荐顺序）：
   ① **显式 `rr`：已测试（§2#16）**——忠实实现 ≡ 5 基因型（无变化）；只有「延迟致死、计入密度」的变体才加抑制，
      但仍只改时机、`t=1000` 仍扫过，**不产生 WT 共存**。故单纯显式 `rr` 不行，需要「抑制 + WT 再定殖」结构。
   ② **WT 源 / 边界补给**：给域外 WT 补给或非均匀 Allee，使前沿后方可持续恢复 WT（抑制型 chase 的关键）。
   ③ **小规模随机性**：仓库自己指出 chasing 本质是随机现象；用 natal 连续采样器在低 `sf` 下跑，让漂变
      制造 WT 避难所（当前 sf≥10 仍太大）。
   ④ **抗性 `r` 加 fitness cost：不采用**——靶基因单倍足量，`rw` 必须中性；给 `rw` 加代价违背科学假设，
      且自测显示它单独也只移动固定↔丢失阈值、造不出 WT 共存。
   另可继续：换 `mean_dispersal`/`migration_rate` 分离波速与阈值；对阈值/潜伏期做随机与 bootstrap 显著性
   （§6#1、#4）。若动到科学模型（如显式 `rr`/抑制项），按高风险流程先获授权并独立复核，默认关闭。

---

## 7. 关键代码位置（natal-core）

- 逐 deme 生命周期与事件顺序：`rust/src/kernels/age_structured.rs`（`run_tick`：first hook → reproduction → survival → aging）。
- 密度调节曲线：`docs/en/2_population_initialization.md`（`logistic`/`beverton_holt`/`ricker`）。
- 连续采样器：`rust/src/kernels/rng.rs`（`continuous_poisson`/`continuous_binomial`/`continuous_multinomial`）。
- 迁移折叠与归一化：`src/natal/frontend/spatial/migration.py`（`fold_migration_csr`）；路线 A 归一化卷积见 M4 文档。
- 六边形拓扑：`src/natal/frontend/spatial/topology.py`（`HexGrid.to_xy`、`build_gaussian_kernel`）。
