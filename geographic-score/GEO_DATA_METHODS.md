# 地理信息的使用方式与数据对齐现状（ASCENT vs hex model）

> 本文件只在本分支（`method/gis2score`）记录。目的：说明两条「地理信息 → 空间模型输入」的路线，
> 并盘点仓库现有的数据结构与尺度对齐情况，作为 `gis2score` 工具设计的事实依据。
>
> 结论先行：**ASCENT 用 6 类 GIS 栅格 + presence/absence 标签训练 SVM 得到相对适合度 H；
> hex model 直接把地理图层做参数化乘性运算并标定总量得到承载量 K。** 二者预测的量不同（H≠K），
> 数据需求不同（标签 vs 参数/校准目标），而当前仓库**没有任何一套已对齐的「GIS 特征 + 标签」数据集**。

---

## 1. ASCENT：六类 GIS → 适合度 H

### 1.1 输入特征（6 个归一化 GIS 栅格）
| 特征 | 说明 |
|---|---|
| elevation | 海拔 |
| building density | 建筑密度 |
| population density | 人口密度 |
| maximum temperature | 最高温度 |
| cropland extent | 农田比例 |
| forest cover | 森林覆盖 |

### 1.2 标签
- GPS 定位的**幼虫 presence/absence** 观测（2019–2023），来自 Chamberland et al. 2025
  （Landscape Ecology，DOI `10.1007/s10980-025-02059-3`，CC BY 4.0）。
- 注意：论文承诺开源代码/数据，但截至整理时**未发现公开仓库**；`code/habitat_suitability_svm.py`
  是基于论文描述**独立重建**的参考实现，不是作者原码。
- 原始 GPS 标注点**不是**独立公开数据集；GBIF 在 Príncipe 的带坐标记录几乎为零
  （实测：*An. coluzzii* 在 ST 有 57 条、带坐标 0 条；Príncipe bbox 内 Culicidae 仅 3 条）。

### 1.3 模型与评分
- **SVM 二分类**；`kernel / C / γ` 用 **Optuna + 交叉验证**（ROC-AUC）搜索。
- 评分是决策函数的 sigmoid：
  `H(x) = 1 / (1 + exp(-(b + Σ_i w_i r_i))) ∈ [0,1]`。
- 逐像素 H → **100 m×100 m 窗口**聚合 → 作为 ASCENT 的**静态环境张量**；模拟运行时不再学习。

### 1.4 H 在模型中的用途：移动偏置（而非承载力）
- 可移动个体采样飞行距离 `d ~ U(961, 2512) m`；
- 取距离 ≤ d 的候选窗口；
- **硬过滤** `H(new) ≥ H(current)`；
- 在合格窗口中**按 H 加权**选目标窗口；
- 在目标窗口内**均匀**落点；无合格候选则原地不动。
- 即：H 是**相对吸引面**，不是每格能承载多少个体。

### 1.5 特征贡献（论文报告）
海拔 30.15%、建筑 28.99%、人口 19.54%、最高温 10.56%、农田 6.57%、森林 4.20%。
（这是重要性比例；线性核下可用归一化 `|w_i|` 近似。）

### 1.6 数据/代码可用性
- 原码未公开；参考实现依赖 `rasterio`、`geopandas`、`scikit-learn`、`optuna`。
- 要求 6 个栅格**预先对齐**（同 CRS/transform/shape）；未对齐会直接报错，需 `gdalwarp`/`rasterio.warp` 预处理。

---

## 2. hex model：地理图层 → 参数化承载量 K

以论文的海南岛模型为例（物种 *Culex quinquefasciatus/pipiens*，2599×2601 hex，
单 hex 直径 **109.55 m**，平均扩散 **1.1121 km**）。

### 2.1 输入数据源（公开、约 1 km 分辨率）
| 输入 | 来源/说明 |
|---|---|
| 人口密度 | **WorldPop + CIESIN**（people/km²） |
| 气候 | **WorldClim 2**：逐月降雨、月均温、月最低温、月最高温 |
| 水体 | 永久水体 / 季节性水体（Figure S1 水体层；具体数据集未在解读中写明） |
| 海拔 | Figure S1 海拔层（具体数据集未在解读中写明） |

### 2.2 分层乘性响应函数
1. 降雨：`rainfunc = a1·(1 − e^{p1·rain} − e^{p2·(320−rain)})`
2. 水体：`waterfunc = a2·(1 − e^{−p2·(water_p + water_s·(1−e^{−p3·rain}))})`
3. 人口/血餐：`bloodmeal(pop) = a3·(1 − e^{−p4·pop}) + p5`
4. 海拔：`altitudefunc = a4·alt + p6`（`alt < 800 m`；`=0` at 800 m）
5. 温度：双高斯；对 `temp / tmin / tmax` 三个双高斯值**取三次方根后相乘**（几何平均）
6. 汇总（乘性耦合）：
   ```
   K = base + (rainfunc + waterfunc) · foodfunc · altitudefunc · tempfunc
   ```
   其中 `base = 0`。乘法结构意味着**任一因子差 → 该处 K 很低**。

### 2.3 时间维与总量
- 用 WorldClim **逐月**气候生成 **12 张 K 图**，模拟中按月切换。
- 输出的量级（论文）：全岛平均 `2.54×10⁷`，季节最大/最小 `3.54×10⁷ / 8.05×10⁶`，
  每 hex 平均成蚊单位 `7.6165`。
- 即：先按响应函数算出**相对空间格局**，再通过参数/总量**对齐到合理的绝对量级**。

### 2.4 校准的可复现性（重要）
- 论文（现有解读）给出函数形式与**结果统计**，但**未给出正式拟合流程、目标函数与 `a/p` 数值**。
- 公开仓库 `Hex-model-main/hainan_model/` 只有 `main.m` **加载** `terrain.mat`
  （最终 12 个月 K），**没有从 GIS 生成/校准 K 的代码或参数文件**。
- 因此：该方法**结构合理、可解释、可迁移**，但严格复现与定量验证依赖论文补充材料/作者数据；
  目前我们只能看到**最终 K 产物**，看不到**参数化与校准过程**。

---

## 3. 仓库现有数据结构

| 路径 | 内容 | 形状/单位 | 是否入库 |
|---|---|---|---|
| `geographic-score/ASCENT_paper/` | 论文 PDF/全文/图；SVM 参考实现 | — | 是（`ASCENT_paper/` 被 ignore） |
| `geographic-score/simple_spatial_model_ref/` | 事件顺序研究用的年龄结构模型参考（`age_model.py`、`spatial_html.py`） | 代码 | 被 ignore |
| `geographic-score/gis2score_explore/` | Phase-0 脚手架：合成 H、迁移偏置/承载量两种插入、HTML | 代码；生成图在 `img/` | 代码入库，`img/` ignore |
| `geographic-score/data/gbif_hawaii_culicidae/` | Hawaii Culicidae occurrence（GBIF） | **1362 行 × 23 列**（CSV+JSON） | 被 ignore |
| `hexagon_spatial_test/hex-model-new/terrain/MM/K_MM_<region>.mat` | 像素级成蚊丰度（源） | `img`，单位 `adult_females_per_pixel`，像素面积 ≈63351.98 m²（≈251.7 m） | 本地/ignore |
| 同上 `..._hex_scale_S.mat` | hex 级 K（面积守恒转换） | `K`，单位 `adult_females_per_hex`；含 `scale_factor`、`pixel_area_m2`、`hex_area_m2`、`center_distance_m`、哈希/`report_json` | 本地/ignore |
| `hexagon_spatial_test/hex-model-new/terrain/MM/` | Hawaii 月度 K 图 | 12 个月；区域 `hawaii/islands/kauai/oahu`；scale 1 为主，月 01 覆盖 1…400 | 本地/ignore |
| `hexagon_spatial_test/Hex-model-main/hainan_model/` | 海南模型输入/输出 | `terrain.mat: terrain 2599×2601×12`、`pop.mat: output 2599×2601`、`border.mat: border 2599×2601 uint8` | 旧参考/ignore |

补充：
- `hexagon_spatial_test/hex-model-new/` 是 hex 模型的**重构统一版**（`model/`+`drives/`+`terrain/`）；
  它自带「栅格 → hex 面积守恒积分」的转换器 `terrain/validate_hex_area_mapping.py`
  （精确多边形相交、总量守恒、缓存/哈希校验），这是目前**唯一已验证的对齐工具**。
- `environment_at_week.m` 按月切换 `K_series(:,:,month)`；`setup_area_terrain.m` 负责载入并校验单位/守恒。

---

## 4. 尺度与对齐现状

### 4.1 三套尺度
| 数据 | 典型尺度/形式 |
|---|---|
| GIS 特征 | 原生分辨率不一：WorldPop/WorldClim ~1 km；DEM ~30 m；WorldCover ~10 m；建筑为矢量 |
| 标签 | GBIF：**点**（`coordinateUncertainty` 中位 ~10 m，少数到 80 km；强聚集）；NEON：**固定样点**（小面积、按月重复） |
| 模型 deme | hex（海南 109.55 m 直径；`hex-model-new` 的 scale 1 中心距 100 m；ASCENT 用 100 m 窗口） |

### 4.2 当前对齐情况：**没有统一的共同网格**
- ASCENT 参考实现要求 6 个栅格**预先对齐**（同 CRS/transform/shape），否则报错；它**不做重投影**。
- hex model 的 K 已有「像素 → hex」的**面积守恒**转换（`validate_hex_area_mapping.py`），
  但那是针对**单个丰度场**的，**不是**一套通用的「多图层 + 标签」对齐流程。
- GBIF 标签是**点/带不确定度**，尚未聚合到任何网格；NEON 原始数据（含 0 计数）**本环境被 403 拒绝**，未下载。
- 因此现状是：**特征、标签、模型网格三者尺度不一致，且没有任何一步被显式对齐**。

### 4.3 GBIF Hawaii 数据的性质（已下载核对）
- 1362 条带坐标 Culicidae；数据集：**iNaturalist Research-grade 947（≈70%）**、NEON ≈187、其余条形码等。
- 物种：*Ae. albopictus* 581、*Cx. quinquefasciatus* 445、*Ae. aegypti* 114、*Ae. japonicus* 84 等。
- 时间：1961–2026，集中在 2016–2017（各 313/321 条）。
- 坐标质量：中位不确定度 10.1 m；≤100 m 86.8%、≤1 km 93.4%；少数 >1 km（最大 80 km）。
- **presence-only**（全部 `occurrenceStatus=PRESENT`）；**无 absence**；重复/聚集严重（300 条样本仅 131 个唯一 0.001° 坐标）。

### 4.4 两种「地理信息 → 模型」的对齐含义
- **ASCENT（H）**：对齐发生在「**特征栅格 ↔ 标签点**」之间（同一 100 m 窗口），
  H 输出后再对齐到模型网格；H 是**相对**适合度，进模型前还需 `baseK`/`f(H)` 映射到量纲。
- **hex model（K）**：对齐发生在「**源栅格 → hex**」之间（面积守恒积分），
  输出即带量纲的 K；校准是对**总量/季节统计**的标定。

---

## 5. 关键差距（Gap）

1. **没有本地「已标注」数据**：GBIF 只有 presence（且 iNat 主导、聚集）；NEON 含 0 计数的原始数据在本环境被 403 拒绝。
2. **H ≠ K**：ASCENT 的 H 是出现适合度（用于移动偏置）；hex/natal 需要的是承载量 K（用于密度调节）。
   `K = H·baseK` 需要**独立标定**，且假设了线性比例；更稳妥是单调映射 `K = f(H)`。
3. **hex 参数化 K 的校准不透明**：`a/p` 参数与拟合/校准流程不在公开仓库，只有最终 `terrain.mat`。
4. **缺少统一的「标签 → 网格」聚合规则**：单元大小、presence 判定（任意点/≥k/最近）、
   absence 定义（仅已调查 0 计数；未采样当背景/排除）都会改变类别平衡与学到的边界，必须显式参数化并做尺度敏感性。
5. **空间验证不足**：空间自相关 + 重复样点 ⇒ 随机 CV 会高估；需空间分块 CV 与独立数据验证。

---

## 6. 建议的统一数据契约（供 `gis2score` 实现）

1. **选定分析网格**：例如 100 m（靠近 ASCENT/hex scale 1）或直接模型 deme；所有特征/标签映射到它。
2. **特征重采样规则**：连续量（海拔/温度/人口）→ bilinear/average；类别量（农田/森林/建筑）→ nearest/mode；
   统一 CRS/extent/shape。
3. **标签→网格聚合（显式参数）**：
   - 单元大小/半径 `r`；presence 规则（任意点/≥k/最近）；
   - absence 规则（仅“已调查且 0 计数”为 0；未采样 → background 或排除）；
   - 做 100 m vs 1 km、`r=0` vs `r=250 m`、`k=1` vs `k≥2` 的敏感性。
4. **模型框架**：优先 presence–background（MaxEnt 式）或真正的 pre/ab（需 NEON 0 计数）；
   空间分块 CV；target-group background/稀疏化做偏差校正。
5. **量纲**：H 只作相对分；单独标定 `baseK` 或 `K=f(H)`（用计数/文献/区域总量），不要默认 `H·baseK`。
6. **可复现**：下载脚本 + 固定版本/哈希 + manifest；大文件放 `~/storage` 或显式 ignore（当前 `geographic-score/data/` 已 ignore）。

---

## 7. 一句话总结

- **ASCENT**：监督式「GIS → H」，依赖**足量、空间代表、定义良好的 presence/absence 标签**；H 是相对适合度、用于移动。
- **hex model**：参数化「GIS → K」，**不需要标签**，但依赖**响应函数假设 + 参数/总量校准**；K 是承载力、可直接进密度调节。
- **当前仓库**：只有 hex 的**最终 K 产物**与 Hawaii 的 **GBIF presence-only** 数据；**没有**已对齐的「特征 + 标签」数据集，也**没有** H→K 的标定。
