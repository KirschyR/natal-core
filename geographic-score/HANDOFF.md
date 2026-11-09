# HANDOFF：gis2score（环境 GIS → capacity K 工作流）

> 本文件面向**新接手的 agent/session**。先读本文件，再读 `geographic-score/data2score.md`（定稿配置）。
> 本分支：`method/gis2score`。仓库：`/home/jovyan/node2/natal-core`。
> **不要** commit/push；**不要**改 `.gitignore`（除非明确要求）；交付由用户提交。

---

## 0. 一句话目标

把**环境 GIS 数据**按 **hex model 的分层乘性响应函数**转换成**每个 hex（deme）的承载力矩阵 K**，
覆盖**夏威夷六岛**（Kauai、Oahu、Molokai、Maui、Lanai、Hawaii）。
**参数用占位值即可**（不拟合、不追求科学正确），只要**工作流能跑通、输出 12 个月的 K 矩阵**。

---

## 1. 当前状态

- 分支 `method/gis2score`；工作树干净（截至交接时）。
- **K 工作流已实现并跑通**：脚本在 `geographic-score/data2score_pipeline/`；
  六岛 12 个月输出在 `geographic-score/data/k_hawaii6_albers100.npz` + `.meta.json`；
  自测见 `selftest.py`（全部通过，面积守恒相对误差 ~1e-9）。
- 环境：`.venv`（numpy/scipy/matplotlib/pandas/plotly/rasterio/shapely/pyproj 可用；`natal` 可导入）。
  运行前：`unset PYTHONHOME PYO3_PYTHON PYTHONPATH`，用 `.venv/bin/python`。
- 网络可用；NEON token 在 `~/.neon_token`（`X-API-Token` 头可用）。

---

## 2. 已定决策（详见 `data2score.md`）

| 项目 | 决定 |
|---|---|
| 方案 | hex 乘性响应函数 → K（不用 ASCENT/SVM） |
| 参数 | 占位值（可运算即可），标 `uncalibrated` |
| K 尺度 | `k_scale = 10` |
| 时间 | 气候态 12 个月 → 输出 12×K |
| 聚合 | **先高分辨率逐像素算 K，再面积积分到 hex** |
| 区域 | 六岛（Kauai/Oahu/Molokai/Maui/Lanai/Hawaii） |
| 投影 | **Albers Equal Area**（建议 `lat_1=19,lat_2=22,lat_0=20.5,lon_0=-157`，WGS84） |
| 网格 | 中心距 **100 m**、`area_scale=1`、**hex-model-new 取向** |
| 陆海 | `elevation>0` 为陆地；海域 K=0 |
| DEM | **Copernicus GLO-30** |
| 人口 | **WorldPop 2020** |
| 气候 | **WorldClim 2**（逐月 `tavg/tmin/tmax/prec`） |
| 水体 | **JRC Global Surface Water**；`seasonality=12` 永久水、`1–11` 季节水、`<1` 非水；每像素 **0/1** |
| 输出 | `K.npz`（密集矩阵）+ `K.meta.json`；**不导出 mat、不输出中间因子图** |
| 验证 | 合成/几何自测；**NEON 验证延后** |

---

## 3. 已有产物（可复用）

| 路径 | 内容 |
|---|---|
| `geographic-score/data2score.md` | **定稿工作流计划/配置**（先读） |
| `geographic-score/GEO_DATA_METHODS.md` | ASCENT vs hex 的地理信息用法与数据对齐分析 |
| `geographic-score/gis2score_explore/env_spatial.py`、`env_html.py` | Phase-0：合成 H、迁移偏置 vs 承载量对比（**K 工作流不需要**，可参考 FFT/归一化思路） |
| `geographic-score/data/gbif_hawaii_culicidae/` | GBIF Hawaii 1362 条 + 物种分布图（已 ignore） |
| `geographic-score/data/neon_mosquito/PUUM/` | NEON 样例（2018-09、2024-09；已 ignore） |
| `geographic-score/data/basemap/` | Natural Earth 海岸线（已 ignore） |
| `hexagon_spatial_test/hex-model-new/terrain/` | hex-model-new 的 K 图（Hawaii/Maui Nui/...）、源像素 `img`、`validate_hex_area_mapping.py` |
| `hexagon_spatial_test/hex-model-new/model/` | hex 模型引擎（`reproduction.m` 含乘性密度、`setup_area_terrain.m` 载 K） |

**关键坑**：`terrain/*_islands` 只覆盖 **Maui Nui**（Molokai/Maui/Lanai），**不含 Kauai/Oahu/Hawaii**。
→ 六岛必须**自建网格**（沿用 hex-model-new 的 100 m/exact-overlap/取向）。

---

## 4. 已完成任务（本次实现）

**已搭建并跑通「环境 GIS → K per hex」工作流**，输出六岛 12 个月的 `K.npz` + `K.meta.json`。

运行方式：
```bash
unset PYTHONHOME PYO3_PYTHON PYTHONPATH
.venv/bin/python geographic-score/data2score_pipeline/fetch_raw.py   # 下载/校验原始数据
.venv/bin/python geographic-score/data2score_pipeline/build_k.py     # 构建 K（约 10 分钟，含几何缓存）
.venv/bin/python geographic-score/data2score_pipeline/selftest.py    # 合成 + 输出自测
```

实现说明：
- 原始数据在 `geographic-score/data/raw/`（含 `manifest.json` 哈希）；
  几何权重缓存在 `geographic-score/data/cache/`；均为 gitignored。
- 计算格网为 **Albers 下 30 m**（逐像素算 K 后 exact-overlap 积分到 100 m hex）；
  输出格网 `rows=6961, cols=6249`，`land_mask` 为“与任一 `elevation>0` 的 30 m 像素相交的 hex”。
- WorldClim 使用官方 30s 分块 COG `tile_25`，已是 float32 `°C`/`mm`（**不是 ×10**）；
  WorldPop 使用 USA 2020 constrained counts，nodata（非建成区）按 0 处理。
- 参数仍为占位值，`status=uncalibrated`；不要据此解释生态量级。

原始建议步骤（保留供参考）：
1. **定网格**：Albers 投影；六岛 lon/lat bbox（约 `[-160.3,-154.7]×[18.9,22.3]`）投影后取 bbox+边距；
   `origin_xy_m=bbox.min`；`rows/cols` 由 100 m 中心距与 bbox 推导；取向按 hex-model-new
   （`x=(√3/2)j·d`，`y=(i−j/2)·d`）。
2. **下载 4 个源**（Copernicus GLO-30、WorldPop 2020、WorldClim 2、JRC GSW），存 `geographic-score/data/raw/`（已 ignore）。
3. **adapter**：统一 CRS/单位/nodata。注意：
   - **WorldClim 2.1 温度是 °C×10**（÷10）；降水 mm/月；
   - WorldPop 是 count 或 density，统一到 people/km²；
   - JRC `seasonality` → `water_p/water_s`（0/1）；DEM `elevation>0` → `land_mask`。
4. **逐像素算 K**（占位参数）：
   ```
   rain  = a1*(1 - exp(-p1*rain) - exp(-p2*(320-rain)))        # clamp >=0
   water = a2*(1 - exp(-p2*(water_p + water_s*(1-exp(-p3*rain)))))
   food  = a3*(1 - exp(-p4*pop)) + p5
   alt   = (a4*alt + p6) if alt<800 else 0                     # clamp >=0
   tg(x) = p7*exp(-((x-p8)/p9)^2) + p10*exp(-((x-p11)/p12)^2)
   temp  = a5 * tg(tavg)^(1/3) * tg(tmin)^(1/3) * tg(tmax)^(1/3)
   K     = base + (rain+water)*food*alt*temp                   # base=0
   K    *= k_scale;  K[land_mask==False]=0;  clamp >=0
   ```
   > 注：论文摘要里的降雨公式可能丢失了指数负号；用上面“饱和型、结果非负”的形式即可（参数不重要）。
5. **面积积分到 hex**：优先复用 `hexagon_spatial_test/hex-model-new/terrain/validate_hex_area_mapping.py`
   的 exact-overlap（需 `shapely/rasterio`，可 `uv pip install`）；若太重，退化为面积加权/中心采样（说明近似）。
6. **输出**：`geographic-score/data/k_hawaii6_albers100.npz`（`K (12,R,C) float32`、`land_mask (R,C) bool`、`months (12,)`）
   + `k_hawaii6_albers100.meta.json`（crs/网格/单位/k_scale/参数/来源哈希/版本）。
7. **自测**：常量层、`alt=800→0`、无雨/无水、NaN/Inf、**面积守恒**、land/sea；跑通后打印每月的 K 范围/总量。

**占位参数（可直接用）**：
```
a1=a2=a3=a5=1; a4=-1/800; p6=1
p1=p2=0.01; p3=0.05; p4=0.001; p5=0.1
p7=p10=1; p8=22; p9=6; p11=28; p12=6
base=0; k_scale=10
```

---

## 5. 数据获取提示（公开源）
- **Copernicus DEM GLO-30**：AWS Open Data `s3://copernicus-dem-30m/` 或 Copernicus Data Space；用 `rasterio`/`requests` 读取。
- **WorldPop 2020**：worldpop.org REST/GeoTIFF（100 m）。
- **WorldClim 2.1**：`geodata.ucdavis.edu/climate/worldclim/2_1/base/`（30s；`tavg/tmin/tmax/prec` 12 个月）。
- **JRC GSW**：global-surface-water.appspot.com / JRC 门户（`seasonality` 瓦片，Hawaii 在相应 10°×10° tile）。
- 只需 **Hawaii 六岛范围**，下载时按 bbox 裁剪以省空间。

---

## 6. 环境与工具
- `unset PYTHONHOME PYO3_PYTHON PYTHONPATH`；`.venv/bin/python`。
- 如需安装：优先 `uv pip install ...`（用户级/venv），不要 sudo。
- 大文件放 `geographic-score/data/`（已 gitignore）或 `~/storage`。
- 参考 hex-model-new 的转换器与元数据字段（`center_distance_m / hex_area_m2 / pixel_area_m2 / mapping_version`）。

---

## 7. 交付规范
- 交付物：脚本（建议 `geographic-score/data2score_pipeline/` 或单文件 `build_k.py`）、`K.npz`+`meta.json`（data/ 下）、自测。
- **不 commit/push**；给一句话中文 commit message（scope `gis2score`）。
- 不写外部交接文档；只在本目录记录。
- 风险分类：局部（复现脚本/数据）；若改 natal 产品代码需另走高风险流程。

---

## 8. 已知坑 / 注意
1. `terrain/*_islands` ≠ 六岛（只是 Maui Nui）→ 自建网格。
2. **非线性**：必须“先算 K 再积分”，不要先聚合输入再算 K。
3. WorldClim 温度 ×10；WorldPop count/density 混淆；JRC 季节字段。
4. 六岛 bbox 会包含 Niihau/Kahoolawe；默认 `elevation>0` 都算陆地（若不想要需额外掩膜）。
5. 12×~1200 万 hex、float32 ≈ 0.5–1 GB；密集存储、同文件原子更新。
6. NEON 现在不用于验证（占位参数下无意义）；保留数据待将来校准。

---

## 9. 启动提示（复制到新 session）

```
你是 natal-core 仓库（/home/jovyan/node2/natal-core）分支 method/gis2score 的新主 agent。
接手「环境 GIS → capacity K」工作流。请先读：
- geographic-score/HANDOFF.md（本文件）
- geographic-score/data2score.md（定稿配置）
按其中的决策实现并跑通工作流：六岛、Albers、100 m hex、hex-model 取向、
Copernicus DEM + WorldPop 2020 + WorldClim 2 + JRC GSW、逐像素算 K 再面积积分、
输出 geographic-score/data/k_hawaii6_albers100.npz + .meta.json（12×K）。
参数用占位值即可，不拟合、不追求科学正确；重点是**能跑通、输出有限非负的 K、面积守恒**。
先给执行计划再动手；不 commit/push；大文件放 geographic-score/data/（已 gitignore）。
```
