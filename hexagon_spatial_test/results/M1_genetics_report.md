# M1 报告：遗传系统一步后代对照（natal `offspring_tensor` vs 论文 `mats`）

> 依据：`Hex_model_recon.md` §5（M1）。数据：`results/m1_offspring_check.json`；
> 脚本：`repro/m1_offspring_check.py`。风险分类：局部（复现脚本 + 数据），未改产品代码。

## 1. 方法

- **论文参考**：`drive_generator.m`（homing, `gcr/dc/herr/ddfitness`）产出 `mats (gn², gn)`，其中
  `M[k,i,j] = mats[k*gn + i, j]` = 母本基因型 `i` × 父本基因型 `j` 产生后代基因型 `k` 的速率/比例。
- **natal**：编译后的 `offspring_tensor P[mother, father, offspring]`（基因型顺序 WT|WT, WT|Drive, WT|Res,
  Drive|Drive, Drive|Res, Res|Res）；驱动转化已烘焙进减数分裂/融合表（实测应用 `HomingDrive` 后表发生变化）。
- **基因型映射**（论文 5 型 → natal）：`dd→Drive|Drive`、`dw→WT|Drive`、`ww→WT|WT`、`rw→WT|Res`、
  `dr→Drive|Res`（natal 另有 `Res|Res`，论文未用）。
- **比较**：取 natal `P` 在 5 型上的子张量 `S[i,j,k]` 与论文 `M[k,i,j]` 逐元素比较。

## 2. 结果

| dc | max_abs 差 | max_rel 差 | natal 全表行和=1 |
|---|---|---|---|
| 1.0 | **0.0** | 0.0 | ✅ |
| 0.9 | **0.0** | 0.0 | ✅ |
| 0.5 | **0.0** | 0.0 | ✅ |

**natal 的 `offspring_tensor` 与论文 homing `mats` 算子逐元素完全相等**（dc=1/0.9/0.5），即 natal 能在
其分层表示（meiosis + 修饰器 + `(Z,Z,Z)` 张量）中**精确表达论文的 homing 驱动一步后代算子**。

## 3. 口径与说明

- 论文 5 型子集**不含 `Res|Res`**，故其行和在子集上不必为 1；natal **全表** `P` 行和为 1（已核）。
- 论文 `mats` 是**速率/比例算子**，其**列和不为 1**（属构造如此，非缺陷）。
- 本阶段只覆盖 **homing**（一基因座，5 型）。论文的 TARE/TADE/CifAB（ToxinAntidote 类）、2-locus TARE（25 型）、
  Wolbachia（2 型）**未逐一映射**（natal 有对应 presets，但需各自的 `drive_generator` 口径对齐）。

## 4. 复现

```bash
cd hexagon_spatial_test && ../.venv/bin/python repro/m1_offspring_check.py
# -> results/m1_offspring_check.json
```

## 5. 残余

- 其余驱动系统（TARE/TADE/CifAB/2-locus/Wolbachia）的一步对照；性别特异性与 fitness cost（`ddfitness`）扫描未覆盖。
- 未测 natal 运行期（`run`）一步新生儿分布 vs 论文 `renew_function` 的 `rcat` 端到端（本阶段在表级验证等价）。

## 6. 独立审查状态

以上为**主 agent 自测**。已在 `EVALUATE.md` 交接（见 §24），等待 evaluator 回执。
