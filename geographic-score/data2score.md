# data2score：环境 GIS → H/K 的数据源与执行计划

> 本文件只在本分支（`method/gis2score`）记录。目标：为 Hawaii（NEON **PUUM** 站点，已有物种计数标签）
> 备齐**同时满足 ASCENT 六特征与 hex 乘性因子**的环境栅格，并把「数据输入 → 中间层 → H/K 矩阵」的
> minimal demo 计划写清楚。
>
> **当前状态：只做规划，不执行下载/处理。** 执行前需确认 §6 的决策项。

---

## 1. 环境变量需求（两个模型的并集）

| 变量 | ASCENT（六特征） | hex model（乘性因子） | 说明 |
|---|---|---|---|
| 海拔 elevation | ✅ elevation | ✅ `altitudefunc`（800 m 归零） | DEM |
| 建筑密度 building density | ✅ | （无直接项，可用建成区） | 建筑面/建成区 |
| 人口密度 population density | ✅ | ✅ `bloodmeal/foodfunc` | 栅格人口 |
| 最高温度 max temperature | ✅ | ✅ `tempfunc`（tmax） | 逐月气候 |
| 农田 cropland | ✅ | —（土地覆盖） | 类别→比例 |
| 森林 forest | ✅ | —（土地覆盖） | 类别→比例 |
| 降雨 rainfall | — | ✅ `rainfunc` | 逐月降水 |
| 水体（永久/季节性） | — | ✅ `waterfunc` | 地表水/季节水 |
| 月均温/月最低温 | — | ✅ `tempfunc` | 逐月气候 |

**结论**：并集 = **海拔、建筑、人口、温度（max/mean/min）、降雨、农田、森林、水体（永久/季节）**。

---

## 2. 数据源清单（推荐）

| 变量 | 推荐源 | 分辨率/时相 | 覆盖 | 备注/许可 |
|---|---|---|---|---|
| 海拔 | **Copernicus DEM GLO-30**（或 SRTM 30 m） | 30 m，静态 | 全球（含 Hawaii） | 开放 |
| 建筑密度 | **GHSL built-up surface**（或 Microsoft/OSM 建筑面派生） | 10/100 m/1 km；多期 | 全球 | 开放；OSM 需聚合 |
| 人口密度 | **WorldPop**（或 CIESIN **GPW v4** / GHSL population） | 100 m / 1 km；逐年/5 年 | 全球 | 开放 |
| 温度（max/mean/min） | **WorldClim 2**（`tmax/tmin/tavg`） | ~1 km；逐月 | 全球 | 开放 |
| 降雨 | **WorldClim 2** `prec`（或 CHIRPS 5 km） | ~1 km；逐月 | 全球 | 开放 |
| 农田 | **ESA WorldCover**（class 40）或 **NLCD**（Cultivated Crops） | 10 m / 30 m | 全球 / 美国 | 类别→比例 |
| 森林 | **ESA WorldCover**（tree cover）或 **Hansen GFC** / **NLCD** | 10 m / 30 m | 全球 / 美国 | 类别/树冠比例 |
| 水体（永久/季节） | **JRC Global Surface Water**（seasonality） | 30 m；月/季节 | 全球 | 开放；季节水首选 |
| （可选）土地覆盖底图 | **NLCD**（美国，30 m）或 **WorldCover** | 30 m / 10 m | 美国 / 全球 | Hawaii 优先 NLCD（与 NEON `nlcdClass` 一致） |

### 2.1 对 Hawaii 的优先组合
- **DEM**：Copernicus GLO-30 / SRTM 30 m；
- **人口**：WorldPop 100 m（或 GPW 1 km）；
- **气候**：WorldClim 2（逐月 `tmax/tmin/tavg/prec`）；
- **土地覆盖**：**NLCD 30 m**（与 NEON `nlcdClass` 同源，含农田/森林/建成区/水体/湿地）；
- **建筑**：GHSL built-up（若 NLCD “Developed” 强度不够细）；
- **季节水**：**JRC Global Surface Water**（hex 的 `water_p/water_s`）。

> 最小“备齐两模型”集合 ≈ **5 个源**：DEM + 人口 + WorldClim 2 + NLCD/WorldCover + JRC 水体；
> 建筑可由 GHSL 或 NLCD Developed 派生。

---

## 3. 最小 demo vs 完整集

### 3.1 Minimal demo（先跑通管线，不追求完整）
- **标签**：NEON PUUM，选 1–3 年「有物种表」的月份，1 个目标 taxon（presence/absence）。
- **环境**：先用 **2–3 个核心源**即可：
  - DEM（海拔）+ WorldClim 2（温度/降雨）+ NLCD/WorldCover（土地覆盖）；
  - 人口/建筑/季节水可后补。
- **网格**：栅格原生 30 m 或聚合到 ~100 m，范围覆盖 PUUM（约 19.55–19.58 N，−155.32…−155.22 E）。
- **输出**：
  - H：SVM（DEM/土地覆盖派生特征）→ H 矩阵；
  - K：`altitudefunc`/土地覆盖 + 用 NEON 计数标定 `baseK` → K 矩阵。

### 3.2 完整集（满足两模型原生计划）
- 六特征（ASCENT）+ 降雨/水体/月温（hex）全部备齐（§2 的 5 源 + 建筑）。
- 逐月时间维（WorldClim/JRC 月度或季节），与 hex 的 12 月 K、NEON 的逐月标签对齐。

---

## 4. 管线阶段（执行顺序）

**Phase 0 · 决策（先确认，见 §6）**
- H 还是 K（或两者共用数据层）；目标物种；时间窗；分析网格；土地覆盖源（NLCD vs WorldCover）；建筑源。

**Phase 1 · 获取与登记**
- NEON 子集（`~/.neon_token`，`X-API-Token`）；
- 环境栅格（§2）；
- 记录来源 URL/DOI、版本、下载日期、SHA256、许可（manifest）。

**Phase 2 · adapter**
- NEON：`mos_trapping`/`mos_sorting`/`mos_expertTaxonomistIDProcessed`/`mos_subsampling` → `LabelTable`；
  - absence：`samplingImpractical=OK` 且 `sampleID` 空（0 捕获）；有样本时按 `targetTaxaPresent`/计数；
  - 丰度：`individualCount / proportionIdentified`（或只做 presence/absence）。
- 栅格：每层重投影/重采样到目标网格（连续量 bilinear，类别量 nearest/mode）。

**Phase 3 · 对齐与聚合**
- 选 `AnalysisGrid`（30 m 或 ~100 m）；
- 特征：面积加权或中心点采样到网格；
- 标签：显式参数化「标签→网格」规则（单元大小、presence 判定、absence、未采样=背景/排除）；
- 保留 `trapHours` 作为 effort。

**Phase 4 · 模型**
- H：SVM（线性/ RBF；Optuna 可选）+ **空间分块 CV** → H；
- K：hex 乘性响应函数（rain/water/pop/alt/temp）→ K；或用 H→`f(H)`/`baseK` 标定到 NEON 计数；
- 产出 `ScoreMap`（`kind=suitability|capacity`、`units`、`uncertainty`、`provenance`）。

**Phase 5 · 验证与输出**
- 空间 CV / 留出站点/月份；与 NEON `elevation`、`nlcdClass` 交叉校验；
- 输出矩阵 + 地图 + 元数据；自测（常数场还原、NaN/对齐、单位）。

**Phase 6 · 扩展决策**
- 是否补全 6 特征/完整 hex 因子；是否从 PUUM 扩展到其他 NEON 站点/全岛。

---

## 5. 关键风险与口径
1. **尺度不一致**：气候 ~1 km、土地覆盖 30/10 m、样点 ~10 m → 统一网格与重采样规则必须显式。
2. **时间对齐**：WorldClim 月气候 vs NEON 采样日期；用「同月」或插值，需记录。
3. **absence 语义**：0 捕获 vs 未采样；子采样（`proportionIdentified`）影响绝对丰度。
4. **空间自相关/固定样点**：随机 CV 高估；用空间分块 CV，注意外推。
5. **分类分辨率**：`taxonID` 可能到属/复合种；`mos_barcoding` 可细化。
6. **许可/溯源**：每层记录来源、版本、许可、哈希；大文件放 `data/`（已 ignore）或 `~/storage`。

---

## 6. 需要先确认的决策项
1. **H 还是 K**（或两者共用数据层）。
2. **目标物种**（先统计 presence/absence 数量，确保可训练）。
3. **时间窗**（如 2019–2025）。
4. **分析网格/分辨率**（30 m 原生 vs ~100 m 聚合）。
5. **土地覆盖源**：NLCD（美国、与 NEON `nlcdClass` 一致）还是 WorldCover（全球）。
6. **建筑密度源**：GHSL built-up vs OSM/Microsoft 建筑面。
7. 是否把**季节水（JRC）**纳入第一版。

---

## 7. 与已定/未定路线的关系
- 数据层（`LabelTable`/`FeatureStack`/`AnalysisGrid`）与 **H/K 的最终选择无关**；两条路线共用。
- **ASCENT/H**：需要 6 特征（尤其海拔/建筑/人口/温度/农田/森林）+ 标签；
- **hex/K**：需要 rain/water/pop/alt/temp + 校准目标（NEON 计数）。
- 因此先备齐 §2 的源即可同时支撑两条路线。
