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
5. **适合度代价**：`viability_dd`（作用于末幼期 age1）与 `fecundity_dd`（作用于繁殖）都会拖慢
   drive；`fecundity_dd` 更直接削弱 dd 的繁殖输出、**放大顺序差异**。
6. **初始额外队列的线性清除**：只加雄性时，额外 dd 队列随 5 个雄成年龄线性老化至死，
   总量每步减 `X/M`（X=额外量，M=5），**恢复时间由释放年龄决定（投 male0 则 5 tick），与数量无关**；
   这段线性下降与 Beverton–Holt 无关（BH 只钉平衡高度）。
7. **age0 记账差**：若把 age0 计入「幼体」，两顺序的幼体数差约 `1/s1 ≈ 90×`。故本探究把
   - `E`（卵）定义为**本 tick 繁殖产生的新生队列**（相位无关）；
   - `J`（幼体）**只取 age1**；`A` 为成虫。
8. **只有 `R-S-A` 与 `S-A-R` 自洽**：其余排列会让 age0 队列绕过密度调节（S）直接老化，
   未调节卵池泄漏 → 数值爆炸（可达上万倍）。

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

覆盖曲线口径：每 deme 的 `dd` 占**非卵个体**（age≥1）比例 `> cover` 的 deme 数；随机图给出
**mean + 95% 运行分布区间（2.5–97.5 分位）**。

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

---

## 6. 待办 / 可继续的方向

1. **统计显著性**：对「顺序差异」做 bootstrap / permutation（例如 T50、覆盖数），而非只看均值。
2. **时间维度指标系统化**：T50/T90、固定 tick 均值、AUC，而非最终固定率。
3. **参数扫描**：`dc`、`viability_dd`、`fecundity_dd`、`release` 量/年龄、`kernel` 大小、`mean_dispersal`。
4. **尺度扫描**：给出「顺序效应 Δ vs sf」曲线，量化信噪比。
5. **对接 M5 空间规模**：在完整 `500×500`（route A GPU）上验证顺序对波速的影响是否稳健。
6. **natal 引擎交叉验证**：用 natal `SpatialPopulation`（R-S-A）与自写空间实现对比，确认语义一致。
7. **线性化解析**：推一 tick 反馈滞后对早期增长率/波速的解析修正。

---

## 7. 关键代码位置（natal-core）

- 逐 deme 生命周期与事件顺序：`rust/src/kernels/age_structured.rs`（`run_tick`：first hook → reproduction → survival → aging）。
- 密度调节曲线：`docs/en/2_population_initialization.md`（`logistic`/`beverton_holt`/`ricker`）。
- 连续采样器：`rust/src/kernels/rng.rs`（`continuous_poisson`/`continuous_binomial`/`continuous_multinomial`）。
- 迁移折叠与归一化：`src/natal/frontend/spatial/migration.py`（`fold_migration_csr`）；路线 A 归一化卷积见 M4 文档。
- 六边形拓扑：`src/natal/frontend/spatial/topology.py`（`HexGrid.to_xy`、`build_gaussian_kernel`）。
