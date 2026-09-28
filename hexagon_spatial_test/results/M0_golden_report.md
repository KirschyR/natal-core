# M0 报告：范围冻结与外部 golden（波速/波形）

> 依据：`Hex_model_recon.md` §0.1、§5（M0）、§6；`hexagon_spatial_test/EVALUATE.md`。
> 本阶段**未改动任何 natal-core 产品代码**；仅新增 `hexagon_spatial_test/` 下的复现脚本与外部 golden。
> 风险分类：局部（复现脚本）。验证方式：针对性数值对照与确定性复核（见 §5）。

## 1. 产出物

| 路径 | 内容 |
|---|---|
| `repro/ref/{flat,junction}/` | 从发布 ZIP 逐字复制的参考实现（`main.m`、`get_mig_matrix25.m`、`homing/{drive_generator,renew_function}.m`）；SHA-256 记录在 golden JSON 中 |
| `repro/gen_golden_hex_wavespeed.m` | MATLAB harness：在临时目录运行参考实现，产出小规模 homing 波速 golden |
| `repro/gen_golden_pde_reference.py` | 解析 ZIP 内预计算 PDE 结果 → 可移植 `.npz` golden + manifest |
| `repro/compare_goldens.py` | 生成 Fig 3 / Fig S5 对照表（JSON） |
| `repro/verify_goldens.py` | golden 自测（有限性、单调性、波形范围、方向交叉核对） |
| `golden/hex_homing_wavespeed_smallcase.{json,mat}` | hex 小规模波速（flat + junction，avd=0.25/0.5/1.0） |
| `golden/pde_wavespeed_reference.npz` | PDE 波速：4 驱动 × 20 个 avd（0.1–2.0） |
| `golden/pde_waveshape_reference.npz` | PDE 波形：4 驱动 × avd∈{0.5,10} 的 `(xlist, ylist)` |
| `golden/pde_reference_manifest.json` | ZIP 源路径、逐文件 SHA-256、域长/检查点 |
| `Hex-model-main/parameter_sensitive/hex code上交版/` | 参考 ZIP 的持久解压（gitignored）；PDE 提取脚本默认读取该目录 |
| `results/m0_fig3_wavespeed_compare.json`、`results/m0_figS5_waveshape_summary.json` | 对照表 |

## 2. 选用的参考实现（与 prompt 的偏差说明）

prompt §8 提到 `test_code/{hs,hv}/homing/main.m`（修 `m`、先 `drive_generator`）。勘察后改用 ZIP 中
`wave justification/{flat side,junction direction}/`，理由：

- 该目录才是产出论文 Fig S3/S5 的**实际实现**，且 ZIP 内同目录带预计算 PDE 结果可自洽对照；
- 其 `main` 以参数接收 `(m,n,avd)`，无 `test_code/hv` 的 `n=original_n*avd` 非整数、`hs` 无 `return_list` 等问题；
- `test_code/get_mig_matrix.m` 把 σ 硬编码为 10.437、不可参数化，不适合小规模/多 avd 扫描。

`test_code` 入口保留为 M2 备选（可在需要时补 golden）。

## 3. 参考模型与单位

- **通用 hex 模型**（确定性、连续计数）：每代
  `data = data + renew_function(data,mats,λ)`，再对每个基因型做
  `imfilter(field, mig_matrix, 'replicate','same')`；`renew` 为
  `d_ar = rct·λ/((λ-1)N+1)/N − ar·N`，`rct` 由 `new_ar'·mats·new_ar` 得（非 natal 的分阶段年龄结构）。
- 迁移核：`get_mig_matrix25(σ)` 为 **51×51**、六边形度量 `d=sqrt(dx²+dy²−dx·dy)`、`normpdf`、全核归一化；
  `σ = avd/sqrt(π/2)`。与 `test_code` 的 59×59+`d≤25` 截断版本不同（版本差异，见 recon §1.7）。
- 初值：左 20% 为释放基因型，其余纯合野生型；`λ=5`；检查点为 40%/70%（flat）与
  `L/(2√3)+0.45L`、`+0.65L`（junction）。
- 单位：速度 = cells/generation。发布 `wavespeed_analyse.m` 对 flat 速度乘 `sqrt(3)/2` 换算为 1-D 等效。

## 4. Fig 3 / S5 对照（初步，非最终结论）

PDE 预计算（ZIP）端点为：

| 驱动 | speed(avd=0.1) | speed(avd=2.0) |
|---|---|---|
| homing | 0.1233 | 2.1598 |
| tare | 0.0338 | 0.6776 |
| tade11 | 0.0129 | 0.2685 |
| cifab | 0.0049 | 0.1008 |

homing 小规模 hex vs PDE（1-D 等效）：

| 方向 | avd | hex(1-D 等效) | PDE | 相对误差 |
|---|---|---|---|---|
| flat | 0.25 | 0.0711 | 0.2859 | −75% |
| flat | 0.50 | 0.4234 | 0.5606 | −24% |
| flat | 1.00 | 0.9250 | 1.1049 | −16% |
| junction | 0.25 | 0.0764 | 0.2859 | −73% |
| junction | 0.50 | 0.4234 | 0.5606 | −24% |
| junction | 1.00 | 0.9163 | 1.1049 | −17% |

**读法**：趋势与论文一致（PDE 在中等 avd 已高于 hex，随 avd 增大差距收窄），但小规模域下低 avd
偏差很大，且 flat/junction 仅近似一致（`flat·√3/2 / junction`：avd=0.25 为 0.93、0.5 为 1.000、1.0 为 1.01）。
这些是**有限域 + 检查点间距 + 固定 51×51 核**的共同产物，不能当作收敛后的定量结论；定量对照留待 M2
（论文规模 300×300、两方向、核差异界定）。

## 5. 自测证据（命令与结果）

```bash
# 1) 生成 hex golden（MATLAB R2026a）
matlab -batch "addpath('.../hexagon_spatial_test/repro'); gen_golden_hex_wavespeed"
# 2) 提取 PDE golden
.venv/bin/python repro/gen_golden_pde_reference.py
# 3) 对照表
.venv/bin/python repro/compare_goldens.py
# 4) 自测
.venv/bin/python repro/verify_goldens.py
```

- 确定性：连续两次运行 harness，6 个速度值**逐位相同**。
- `verify_goldens.py`：硬检查全部 `OK`（有限/正/单调；PDE 波速单调；波形 ∈[0,1]）。
- 方向交叉核对：见 §4，作为观察报告，不作硬断言。

## 6. 残余风险与 M2 事项

1. **小规模未收敛**：需 M2 用论文 300×300 与更大检查点间距复核波速；本报告不作定量复现声明。
2. **核版本二义**：本 golden 用 51×51 无截断版本；需与 `test_code` 59×59+`d≤25` 版本分清（recon §1.7）。
3. **hex 波形缺位**：参考 `main` 在触发后立即 `return`，未导出波形；hex 波形需在 M2 用可追踪版本的
   等价实现补齐（PDE 波形已具备）。
4. **junction 旋转几何**：`m=ceil(L(1+1/√3))`、`n=ceil(2L/√3)` 与 natal `HexGrid` 的坐标映射尚未核对（M0
   计划项，留给 M2 一并做）。

## 7. 独立审查状态

- 主 agent 自测：见 §5。
- 独立审查（evaluator，`hexagon_spatial_test/EVALUATE.md` §2）：**APPROVED（M0 §1）**。独立复现了逐字性、
  PDE 提取、hex 确定性与数值；补齐坐标等价证明；并补强 `verify_goldens.py`（新增 §0/0b：provenance 哈希
  与 golden 覆盖网格断言），未改产品代码或 §1 正文。
- 审查遗留（口径冻结、核有效扩散、方向命名、单位换算）已记入 `Hex_model_recon.md` §5「M2 前置口径」；
  recon §1.4 的 PDE `dt` 已按源码修正为 `1e-4`。
- 结论：M0 §1 通过独立审查；仅覆盖 M0 golden 与复现骨架，不构成对 M1/M2 数值语义的认可。
