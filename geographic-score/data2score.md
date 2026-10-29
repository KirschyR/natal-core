# data2score：环境 GIS → capacity K 的工作流计划（定稿配置）

> 本文件只在本分支（`method/gis2score`）记录。
> 目标：走 **hex model 的分层乘性响应函数**，把环境 GIS 数据转换为 **每 deme 的 capacity K**；
> **不训练/不拟合**响应函数参数（用占位值），重点是**工作流框架**是否正确、可复现、可替换。
>
> **当前状态：规划已定稿，尚未执行任何下载/处理。**

---

## 1. 最终配置（已定）

| 项目 | 决定 |
|---|---|
| 方案 | hex model 乘性响应函数 → K（不使用 ASCENT/SVM；不保留 ASCENT 特征接口） |
| 参数 | **占位值**（可运算、无科学意义），输出标 `uncalibrated`；拟合后续外挂 |
| K 尺度 | `k_scale = 10`（占位） |
| 时间 | **气候态 12 个月**，输出 **12×K** |
| 聚合顺序 | **先在高分辨率逐像素算 K，再面积积分到 hex** |
| 区域 | 六个主岛：**Kauai、Oahu、Molokai、Maui、Lanai、Hawaii** |
| 投影 | **Albers Equal Area（Hawaii 参数）** |
| 网格 | 中心距 **100 m**、`area_scale=1`、**hex-model-new 的取向** |
| 陆海 | `elevation > 0` 为陆地；海域 `K=0` |
| DEM | **Copernicus DEM GLO-30** |
| 人口 | **WorldPop 2020** |
| 气候 | **WorldClim 2**（逐月气候态；`tmax/tmin/tavg/prec`） |
| 水体 | **JRC Global Surface Water**；`seasonality=12` → 永久水，`1–11` → 季节水，`<1` → 非水；每像素 **0/1** |
| 输出 | **`.npz`** 密集矩阵；**不输出中间因子图**；**不导出 `.mat`**；只存网格定义（不存 cell centers） |
| 验证 | 合成/几何自测；**NEON 验证移出当前阶段**（待参数校准后再做） |

---

## 2. 变量与数据源（hex 因子）

| 变量 | 用途（响应函数） | 源 | 分辨率/时相 | adapter 注意 |
|---|---|---|---|---|
| 海拔 | `altitudefunc`（800 m 归零） | **Copernicus GLO-30** | 30 m，静态 | 米；nodata/海面处理 |
| 降雨 | `rainfunc` | **WorldClim 2 `prec`** | ~1 km，逐月气候态 | **mm**；注意单位 |
| 温度 | `tempfunc`（mean/min/max） | **WorldClim 2 `tavg/tmin/tmax`** | ~1 km，逐月气候态 | **°C×10**（adapter ÷10） |
| 人口 | `bloodmeal/foodfunc` | **WorldPop 2020** | 100 m，年 | count vs density；统一 people/km² |
| 水体 | `waterfunc`（永久/季节） | **JRC GSW** | 30 m，季节统计 | `seasonality` 切分；0/1 |

> 注：**ASCENT 的 6 特征（建筑/农田/森林）不在本阶段**；若以后切回 H 路线再接。

---

## 3. 网格定义

### 3.1 投影（建议参数，执行前确认）
```
+proj=aea +lat_1=19 +lat_2=22 +lat_0=20.5 +lon_0=-157 +datum=WGS84 +units=m +no_defs
```
- Albers Equal Area，标准纬线 19°/22°，中央经线 -157°，原点纬度 20.5°；
- 覆盖六岛（约 lon −160.3…−154.7，lat 18.9…22.3），跨群岛变形小。

### 3.2 网格参数
- `center_distance_m = 100`（hex-model-new scale 1）；
- `hex_area_m2 = 8660.254…`；`area_scale = 1`；
- **取向 = hex-model-new**：`x = (√3/2)·j·d`，`y = (i − j/2)·d`（左上原点，x 东、y 南）；
- `origin_xy_m`：六岛在 Albers 下投影 bbox 的 `(min_x, min_y)`（+ 边距）；
- `rows/cols`：由 bbox 与 d 推导（约 4000×3000，~1200 万 hex）；
- `land_mask = elevation > 0`（海域 K=0）。

> 说明：现有 `hex-model-new/terrain/*_islands` 只覆盖 **Maui Nui**（Molokai/Maui/Lanai），不含 Kauai/Oahu/Hawaii，**不能直接作为六岛网格**；因此采用**自建六岛网格**，但沿用其 100 m/exact-overlap 转换器与取向。

### 3.3 面积积分
- 先在各源层原生分辨率上算 K（逐像素）；
- 再用 **exact-overlap 面积积分**（`validate_hex_area_mapping.py` 的算法）把 K 聚合到 hex；
- 避免“先把输入聚合到 hex 再算 K”的非线性误差。

---

## 4. 输出

### 4.1 `K.npz`（数组）
| 字段 | 形状/类型 | 内容 |
|---|---|---|
| `K` | `(12, R, C)` float32 | 每 hex 的 capacity（单位见 meta） |
| `land_mask` | `(R, C)` bool | `elevation>0` |
| `months` | `(12,)` int16 | 1–12 |

- **密集矩阵**存储；float32 以控制体积（~0.5–1 GB 量级）。
- **不存** `cell_center_x/y`；网格定义在 meta 中用公式描述。
- **不存**中间因子图。

### 4.2 `K.meta.json`（元数据）
`crs`、`convention`（hex-model）、`center_distance_m`、`area_scale`、`hex_area_m2`、
`pixel_area_m2`、`rows`、`cols`、`origin_xy_m`、`extent_m`、`units`、`k_scale`、
`parameters`（占位值 + `uncalibrated: true`）、`inputs`（各层源/版本/年份/哈希/许可）、
`mapping_version`、`months`（气候态说明）、`created_at`、`updated_at`、`version`。

### 4.3 更新策略
- 固定输出路径（如 `geographic-score/data/k_hawaii6_albers100.npz` + `.meta.json`）；
- 输入地理信息更新时：**重写同一文件**（临时文件 + 原子替换），不保留多份；
- 更新时同步 `meta.version / updated_at / inputs.*.sha256`，便于追溯但**不额外占空间**。

---

## 5. 管线阶段（执行顺序）

1. **获取**：Copernicus GLO-30、WorldClim 2、WorldPop 2020、JRC GSW；记录来源/哈希/许可。
2. **adapter**：
   - 每层统一单位/CRS/nodata（WorldClim 温度 ÷10）；
   - 水体：JRC `seasonality` → `water_p/water_s`（0/1）；
   - DEM：`elevation>0` → `land_mask`。
3. **高分辨率计算**：逐像素套 `rainfunc/waterfunc/bloodmeal/altitudefunc/tempfunc` → `S`；`K = base + k_scale·S`；海域 `K=0`。
4. **面积积分**：exact-overlap 聚合到六岛 hex 网格 → `K[month]`。
5. **输出**：写 `K.npz` + `K.meta.json`。
6. **自测**：见 §6。

---

## 6. 验证（当前阶段）

- **合成数据自测**：
  - `altitudefunc(800)=0`、altitude>800 → 0；
  - 无雨/无水/极端温度的边界行为；
  - NaN/Inf 处理；
  - **面积积分守恒**（`ΣK_hex ≈ ΣK_pixel`）；
  - 单位/CRS/网格对齐。
- **几何验证**：用六岛 bbox 生成网格，抽查 hex 覆盖/边界；
- **NEON：移出当前阶段**。理由：参数为占位值，K 不可能准确描述观测丰度；NEON 留待**参数校准后**做独立验证（秩相关/AUC/季节相关）。

---

## 7. 风险与注意
1. **六岛范围**：现有 `islands` 仅 Maui Nui；必须自建六岛网格。
2. **非线性聚合**：务必“先算 K 再积分”。
3. **单位**：WorldClim 温度 °C×10；WorldPop count vs density；JRC 季节统计口径。
4. **内存/IO**：12×~1200 万 hex 的 float32 约 0.5–1 GB；密集存储 + 同文件重写有 IO 开销。
5. **陆地掩膜**：`elevation>0` 会把 bbox 内其他陆地（如 Niihau/Kahoolawe）也纳入；若严格只要六岛需额外岛礁掩膜（待定，非当前默认）。
6. **占位参数**：K 仅用于验证工作流，不具科学解释力。

---

## 8. 执行前仅需核对的具体数值
1. Albers 参数（`lat_1/lat_2/lat_0/lon_0`）与六岛 bbox → `origin/extent/rows/cols`。
2. WorldClim 2 的变量名与单位（`tavg/tmin/tmax` 的 ×10、`prec` 的 mm/月）。
3. JRC `seasonality` 的读取字段与切分（12 / 1–11 / <1）。
4. 是否对 Niihau/Kahoolawe 做额外排除（默认：不排除，`elevation>0` 即陆地）。
