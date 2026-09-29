# M3 报告（GPU 侧）：空间 CUDA 路径 enable / 预算与行宽守卫实测

> 依据：`Hex_model_recon.md` §5（M3）、§4；`EVALUATE.md` §6/§8。数据：`results/m3_gpu_data.json`；
> 命令：`python repro/m3_gpu_probe.py`（约 1.3 min）。风险分类：局部（复现脚本 + 数据），未改产品代码。
> 环境：RTX 5090 24455 MiB、驱动 595.84 / CUDA 13.2、`gpu` feature 构建（`maturin develop --features "gpu,extension-module"`）。

## 1. deterministic 空间 GPU：enable 与迁移缓存足迹

| 场地 | k | Z | nnz | 迁移缓存 MiB | status | GPU tick s | CPU tick s（M3） | 加速比 |
|---|---|---|---|---|---|---|---|---|
| 30² | 5 | 3 | 1.98e4 | 2.7 | disabled→enabled | 0.0011 | 0.019 | ~17× |
| 60² | 11 | 3 | 3.93e5 | 52.6 | disabled→enabled | 0.0155 | 0.154 | ~10× |
| 120² | 21 | 3 | 5.79e6 | 773.7 | disabled→enabled | 0.2148 | 1.718 | ~8× |
| 300² | 11 | 3 | 1.06e7 | 1416.7 | disabled→enabled | 0.4218 | 4.059 | ~9.6× |
| 300² | 21 | 3 | 3.82e7 | 5104.8 | disabled→enabled | 1.4422 | 11.292 | ~7.8× |

- `enable_gpu` 在 5 个规模均成功（`disabled→enabled`），无静默回退。
- 迁移缓存足迹与 Rust `migration_cache_bytes` 公式一致：**`≈ 4·nnz·A·Z² + 8·nnz·A·Z + 20·nnz`**
  （本模型 A=2、Z=3）；大核/大场地时随 `nnz` 线性增长（300²×21² 已 5.1 GiB）。
- deterministic GPU tick 相对 CPU 约 **8–17×**（tick 单位为秒/代，含迁移与生命周期）。

## 2. stochastic 迁移行宽守卫（`MAX_CSR_ROW = 32`）

| 场地 | k | 最大 CSR 行宽 | 结果 |
|---|---|---|---|
| 30² | 5 | 24 | `tick_ok` |
| 30² | 11 | 120 | **rejected**：`stochastic migration CSR row length 120 exceeds the device scratch limit of 32` |
| 60² | 21 | 440 | **rejected**：`... row length 440 exceeds ... 32` |

结论：随机空间迁移的设备 scratch 硬上限为 **32**；宽核（`k≥7`，行宽 >32）在**首次迁移**即显式报错，
不静默回退。论文宽核（r25 盘 ~1951 邻居）在随机 GPU 上不可用（确定性 gather 路径无此限制，见 §1）。

## 3. enable 期显存预算守卫（显式报错，不回退）

- 构造过预算 CSR：7 等位（Z=28）× 200²×k=11。`enable_gpu` 的 `ensure_migration_budget` 立即抛出：
  `GPU memory budget exceeded: the state needs 30011 MiB (31468866008 B), but only 19418 MiB is free; reduce the batch size or free the device`。
- 说明：缓存足迹 `4·nnz·A·Z²` 对 **Z²** 极敏感——homing（Z=3）尚可，2-locus（Z=25）将放大 ~69×；
  这对 M4「是否/如何上 GPU」是关键约束。
- 报告观察到：报错时 free=19418 MiB `< nvidia-smi` 的 23961 MiB，差额为 executor 已分配的 state/sperm
  与 CUDA 上下文；守卫按 `context.memory_info()` 的**实时**空闲显存判定（正确）。

## 4. 回归：`phase0` bit-identical

`unset PYTHONHOME PYO3_PYTHON PYTHONPATH && .venv/bin/python scripts/phase0_baseline.py --check`
→ **all scenarios bit-identical，EXIT=0**（GPU feature 构建未改变 CPU 数值语义）。

## 5. 对 M4 的输入

1. **确定性空间 GPU 可用**且相对 CPU 有 ~8–17× 单 tick 加速；但迁移缓存 `∝ nnz·Z²`。
2. **随机空间 GPU 行宽 ≤32**：宽核不可用；确定性 gather 路径无行宽限制但受显存预算。
3. **守卫齐备且显式**：enable 期预算、随机行宽均在超限时报错，无静默回退（符合 recon §5 验证要求）。
4. 因此论文规模（宽核 + 2-locus + 大格点）在现有 GPU 旁路上仍受限，M4 需在 A/B/C 间决断。

## 6. 复现命令

```bash
# 需先构建带 GPU 的扩展：
unset PYTHONHOME PYO3_PYTHON PYTHONPATH
export RUSTUP_HOME="$PWD/.venv/rustup" CARGO_HOME="$PWD/.venv/cargo" \
       PATH="$PWD/.venv/cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/.venv/cargo-target" \
       LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
.venv/bin/maturin develop --features "gpu,extension-module"
cd hexagon_spatial_test && ../.venv/bin/python repro/m3_gpu_probe.py
# -> results/m3_gpu_data.json
```

## 7. 残余与边界

- GPU 侧仅测 enable/预算/行宽/tick 时间；**正确性对照**（GPU vs CPU）属 M7。
- 未测 GPU 随机确定性统计、hook、history；未测 age_structured 空间 GPU 的年龄/性别放大。
- 一次进程内多次 enable：退出后显存回落（23961→23959 MiB），说明上下文释放正常。
- 结论仅针对本机 RTX 5090 与环境。

## 8. 独立审查状态

以上为**主 agent 自测**。已在 `EVALUATE.md` 提交 §9 交接，等待 evaluator §10。
