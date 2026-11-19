"""Feasibility validation for moving M2 source-outbound sampling to the GPU.

The M2 migration has two stochastic stages: source outbound (continuous
binomial / Beta) and destination block sampling.  A1 moved the FFT and the
destination stage to the GPU but kept the source stage on the CPU; timing
showed that the CPU source stage dominates.  This script checks whether the
source stage can also run on the GPU by comparing:

* the outbound-only distribution (CPU vs GPU, many replicates);
* the full M2 output distribution (CPU/CPU, GPU/GPU, GPU-outbound/CPU-sampling);
* the per-call timing of each path.

It uses synthetic adult planes and no lifecycle, so it is an operator-level
feasibility check, not a chasing result.  Run with the conda torch environment:

    /opt/conda/bin/python chasing_model/validate_gpu_outbound.py
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import gpu_migration as gm
import numpy as np
import torch
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "gpu_fft"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "unknown"


CASES = [
    ("small", 15, 7, 0.8),
    ("large", 31, 11, 1.5),
]
R_OUTBOUND = 400
R_OUTPUT = 200
R_TIMING = 50
NEW_ADULT_AGE = gm.NEW_ADULT_AGE
Z = gm.Z
S = gm.S
A = gm.A


def _synthetic_data(rows: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((Z, S, A, rows, rows))
    data[:, :, NEW_ADULT_AGE:] = (
        rng.random((Z, S, A - NEW_ADULT_AGE, rows, rows)) * 10.0
    )
    return data


def _cpu_outbound(
    field: np.ndarray, kernel: np.ndarray, seed: int, rate: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = field.shape
    planes = field.reshape(-1, shape[-2] * shape[-1])
    out = np.empty_like(planes)
    for i in range(planes.shape[0]):
        out[i] = gm._continuous_binomial_vec(rng, planes[i], rate)
    return out.reshape(shape)


def _summarize_moments(
    sum_a: np.ndarray,
    sumsq_a: np.ndarray,
    sum_b: np.ndarray,
    sumsq_b: np.ndarray,
    reps: int,
) -> dict[str, float]:
    mean_a = sum_a / reps
    mean_b = sum_b / reps
    var_a = np.maximum(sumsq_a / reps - mean_a**2, 0.0)
    var_b = np.maximum(sumsq_b / reps - mean_b**2, 0.0)
    abs_diff = np.abs(mean_b - mean_a)
    floor = np.maximum(np.abs(mean_a), 1e-9)
    rel_diff = abs_diff / floor
    ratio = np.divide(
        var_b,
        var_a,
        out=np.full(var_a.shape, np.nan),
        where=var_a > 1e-12,
    )
    finite = ratio[np.isfinite(ratio)]
    return {
        "mean_abs_diff_median": float(np.median(abs_diff)),
        "mean_abs_diff_max": float(abs_diff.max()),
        "mean_rel_diff_median": float(np.median(rel_diff)),
        "mean_rel_diff_max": float(rel_diff.max()),
        "variance_ratio_median": float(np.median(finite)) if finite.size else float("nan"),
        "variance_ratio_p05": float(np.percentile(finite, 5)) if finite.size else float("nan"),
        "variance_ratio_p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "variance_ratio_mc_se": float(
            2.0 * (np.median(finite) if finite.size else 1.0) / math.sqrt(max(reps - 1, 1))
        ),
    }


def _drive_proxy(data: np.ndarray) -> np.ndarray:
    adult = data[:, :, NEW_ADULT_AGE:]
    drive = adult[2].sum(axis=(0, 1))
    total = adult.sum(axis=(0, 1, 2))
    return np.divide(drive, total, out=np.zeros_like(total), where=total > 0)


def _compare_outputs(
    samples: dict[str, np.ndarray],
    totals: dict[str, np.ndarray],
    reference: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ref_drive = samples[reference]
    ref_total = totals[reference]
    for method, drive in samples.items():
        if method == reference:
            continue
        diff = np.abs(drive.mean(axis=0) - ref_drive.mean(axis=0))
        var_ref = ref_drive.var(axis=0, ddof=1)
        var = drive.var(axis=0, ddof=1)
        ratio = np.divide(
            var,
            var_ref,
            out=np.full(var_ref.shape, np.nan),
            where=var_ref > 1e-12,
        )
        finite = ratio[np.isfinite(ratio)]
        reps = drive.shape[0]
        out[method] = {
            "drive_mean_abs_diff_median": float(np.median(diff)),
            "drive_mean_abs_diff_max": float(diff.max()),
            "drive_variance_ratio_median": float(np.median(finite)) if finite.size else float("nan"),
            "drive_variance_ratio_p05": float(np.percentile(finite, 5)) if finite.size else float("nan"),
            "drive_variance_ratio_p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
            "drive_variance_ratio_mc_se": float(
                2.0 * (np.median(finite) if finite.size else 1.0) / math.sqrt(max(reps - 1, 1))
            ),
            "total_mean_ref": float(ref_total.mean()),
            "total_mean_method": float(totals[method].mean()),
            "total_mean_rel_diff": float(
                (totals[method].mean() - ref_total.mean()) / ref_total.mean()
            ),
            "total_sd_ref": float(ref_total.std(ddof=1)),
            "total_sd_method": float(totals[method].std(ddof=1)),
            "total_sd_ratio": (
                float(totals[method].std(ddof=1) / ref_total.std(ddof=1))
                if ref_total.std(ddof=1) > 1e-12
                else None
            ),
        }
    return out


def _time_call(fn, repeats: int) -> dict[str, float]:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": float(np.mean(times) * 1e3),
        "median_ms": float(np.median(times) * 1e3),
        "min_ms": float(np.min(times) * 1e3),
        "max_ms": float(np.max(times) * 1e3),
    }


def run_case(name: str, rows: int, kernel_size: int, avd: float) -> dict[str, Any]:
    kernel = gm.hex_gaussian(kernel_size, avd)
    data = _synthetic_data(rows, seed=0)
    field = data[:, :, NEW_ADULT_AGE:].copy()
    flat_n = field.size
    rng_pick = np.random.default_rng(12345)
    qidx = rng_pick.choice(flat_n, size=min(500, flat_n), replace=False)

    # --- outbound-only distribution -------------------------------------
    field_t = torch.as_tensor(field, dtype=torch.float64, device="cuda")
    gpu = gm.M2GPU(rows, rows, kernel, dtype=torch.float64, seed=0)
    outbound = {}
    for rate in (1.0, 0.5):
        sum_cpu = np.zeros(field.shape, dtype=np.float64)
        sumsq_cpu = np.zeros(field.shape, dtype=np.float64)
        sum_gpu = np.zeros(field.shape, dtype=np.float64)
        sumsq_gpu = np.zeros(field.shape, dtype=np.float64)
        total_cpu = np.zeros(R_OUTBOUND)
        total_gpu = np.zeros(R_OUTBOUND)
        q_cpu = np.empty((R_OUTBOUND, qidx.size))
        q_gpu = np.empty((R_OUTBOUND, qidx.size))
        max_abs = 0.0
        for seed in range(R_OUTBOUND):
            c = _cpu_outbound(field, kernel, seed, rate)
            gpu.torch_gen.manual_seed(seed)
            g = gpu._outbound_gpu(field_t, rate).cpu().numpy()
            max_abs = max(max_abs, float(np.abs(c - g).max()))
            sum_cpu += c
            sumsq_cpu += c * c
            sum_gpu += g
            sumsq_gpu += g * g
            total_cpu[seed] = c.sum()
            total_gpu[seed] = g.sum()
            q_cpu[seed] = c.ravel()[qidx]
            q_gpu[seed] = g.ravel()[qidx]
        summary = _summarize_moments(
            sum_cpu, sumsq_cpu, sum_gpu, sumsq_gpu, R_OUTBOUND
        )
        sd_cpu = float(total_cpu.std(ddof=1))
        sd_gpu = float(total_gpu.std(ddof=1))
        ks = ks_2samp(q_cpu.ravel(), q_gpu.ravel())
        summary.update(
            {
                "rate": rate,
                "total_mean_cpu": float(total_cpu.mean()),
                "total_mean_gpu": float(total_gpu.mean()),
                "total_mean_rel_diff": float(
                    (total_gpu.mean() - total_cpu.mean()) / total_cpu.mean()
                ),
                "total_sd_cpu": sd_cpu,
                "total_sd_gpu": sd_gpu,
                "total_sd_ratio": (sd_gpu / sd_cpu if sd_cpu > 1e-12 else None),
                "same_seed_max_abs_diff": max_abs,
                "quantiles_cpu": np.percentile(
                    q_cpu.ravel(), [1, 5, 50, 95, 99]
                ).tolist(),
                "quantiles_gpu": np.percentile(
                    q_gpu.ravel(), [1, 5, 50, 95, 99]
                ).tolist(),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
        outbound[str(rate)] = summary

    # --- CPU-sampling equivalence at both rates --------------------------
    sampling_equivalence: dict[str, Any] = {}
    for rate in (1.0, 0.5):
        ref = gm.m2_reference_cpu(data, kernel, rate=rate, seed=0)
        entry: dict[str, Any] = {"rate": rate}
        for dtype, tag in ((torch.float64, "f64"), (torch.float32, "f32")):
            g = gm.M2GPU(rows, rows, kernel, dtype=dtype, seed=0)
            out = g.migrate(
                data, rate=rate, sample_device="cpu", outbound_device="cpu"
            )
            diff = np.abs(out - ref)
            entry[f"{tag}_max_abs"] = float(diff.max())
            entry[f"{tag}_mean_abs"] = float(diff.mean())
        sampling_equivalence[str(rate)] = entry

    # --- full M2 output distribution at both rates -----------------------
    m2 = gm.M2GPU(rows, rows, kernel, dtype=torch.float64, seed=0)
    methods = ("cpu_cpu", "gpu_gpu", "gpu_cpu")
    outputs: dict[str, Any] = {}
    for rate in (1.0, 0.5):
        samples: dict[str, np.ndarray] = {}
        totals: dict[str, np.ndarray] = {}
        adult_stats: dict[str, dict[str, Any]] = {}
        for method in methods:
            drive = np.zeros((R_OUTPUT, rows * rows))
            total = np.zeros(R_OUTPUT)
            adult_min = np.inf
            adult_finite = True
            for seed in range(R_OUTPUT):
                m2.np_rng = np.random.default_rng(seed)
                m2.torch_gen.manual_seed(seed)
                if method == "cpu_cpu":
                    out = m2.migrate(
                        data,
                        rate=rate,
                        sample_device="cpu",
                        outbound_device="cpu",
                    )
                elif method == "gpu_gpu":
                    out = m2.migrate(
                        data,
                        rate=rate,
                        sample_device="gpu",
                        outbound_device="gpu",
                    )
                else:
                    out = m2.migrate(
                        data,
                        rate=rate,
                        sample_device="cpu",
                        outbound_device="gpu",
                    )
                drive[seed] = _drive_proxy(out).ravel()
                adult = out[:, :, NEW_ADULT_AGE:]
                total[seed] = float(adult.sum())
                adult_min = min(adult_min, float(adult.min()))
                adult_finite = adult_finite and bool(np.isfinite(adult).all())
            samples[method] = drive
            totals[method] = total
            adult_stats[method] = {"min": adult_min, "finite": adult_finite}
        output = _compare_outputs(samples, totals, reference="cpu_cpu")
        output["cpu_cpu"] = {
            "reference": True,
            "adult_min": adult_stats["cpu_cpu"]["min"],
            "adult_finite": adult_stats["cpu_cpu"]["finite"],
        }
        for method in methods:
            if method == "cpu_cpu":
                continue
            output[method]["adult_min"] = adult_stats[method]["min"]
            output[method]["adult_finite"] = adult_stats[method]["finite"]
        outputs[str(rate)] = output

    # --- timing ----------------------------------------------------------
    m2_timing = gm.M2GPU(rows, rows, kernel, dtype=torch.float32, seed=0)
    timing = {}
    field_t32 = torch.as_tensor(field, dtype=torch.float32, device="cuda")
    for rate in (1.0, 0.5):
        tag = str(rate)

        def _cpu_call(rate=rate):
            return _cpu_outbound(field, kernel, 0, rate)

        def _gpu_call(rate=rate):
            m2_timing.torch_gen.manual_seed(0)
            m2_timing._outbound_gpu(field_t32, rate)

        timing[f"cpu_outbound_only_rate{tag}"] = _time_call(
            _cpu_call, R_TIMING
        )
        timing[f"gpu_outbound_only_rate{tag}"] = _time_call(
            _gpu_call, R_TIMING
        )
    for rate in (1.0, 0.5):
        tag = str(rate)

        def _cpu_full(rate=rate):
            return m2_timing.migrate(
                data, rate=rate, sample_device="cpu", outbound_device="cpu"
            )

        def _gpu_full(rate=rate):
            return m2_timing.migrate(
                data, rate=rate, sample_device="gpu", outbound_device="gpu"
            )

        def _cpu_out_full(rate=rate):
            return m2_timing.migrate(
                data, rate=rate, sample_device="gpu", outbound_device="cpu"
            )

        timing[f"cpu_cpu_full_rate{tag}"] = _time_call(_cpu_full, R_TIMING)
        timing[f"gpu_gpu_full_rate{tag}"] = _time_call(_gpu_full, R_TIMING)
        timing[f"cpu_outbound_gpu_full_rate{tag}"] = _time_call(
            _cpu_out_full, R_TIMING
        )
    return {
        "rows": rows,
        "kernel_size": kernel_size,
        "avd": avd,
        "R_outbound": R_OUTBOUND,
        "R_output": R_OUTPUT,
        "outbound": outbound,
        "sampling_equivalence": sampling_equivalence,
        "outputs": outputs,
        "timing": timing,
    }


def main() -> int:
    payload: dict[str, Any] = {
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "script": Path(__file__).name,
        "note": (
            "Operator-level feasibility of GPU outbound sampling; synthetic "
            "adult planes, no lifecycle. Distributions are compared across "
            "independent replicates (different RNG streams)."
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "cases": {},
    }
    for name, rows, kernel_size, avd in CASES:
        print(f"[{name}] outbound + output distributions ...", flush=True)
        payload["cases"][name] = run_case(name, rows, kernel_size, avd)
        case = payload["cases"][name]
        for rate, stats in case["outbound"].items():
            print(
                f"  outbound rate {rate}: mean rel "
                f"{stats['mean_rel_diff_median']:.3e} "
                f"(max {stats['mean_rel_diff_max']:.3e}); "
                f"var ratio {stats['variance_ratio_median']:.3f}; "
                f"KS {stats['ks_stat']:.3f}; "
                f"same-seed max abs {stats['same_seed_max_abs_diff']:.3e}",
                flush=True,
            )
        for rate, eq in case["sampling_equivalence"].items():
            print(
                f"  CPU-sampling equivalence rate {rate}: "
                f"f64 {eq['f64_max_abs']:.3e}, f32 {eq['f32_max_abs']:.3e}",
                flush=True,
            )
        for rate, output in case["outputs"].items():
            for method, stats in output.items():
                if stats.get("reference"):
                    continue
                print(
                    f"  output rate {rate} {method}: drive mean diff "
                    f"{stats['drive_mean_abs_diff_median']:.3e}, "
                    f"var ratio {stats['drive_variance_ratio_median']:.3f}, "
                    f"total mean rel {stats['total_mean_rel_diff']:+.3e}",
                    flush=True,
                )
        print(
            f"  timing: cpu outbound(rate1) "
            f"{case['timing']['cpu_outbound_only_rate1.0']['median_ms']:.3f} ms, "
            f"gpu outbound(rate1) "
            f"{case['timing']['gpu_outbound_only_rate1.0']['median_ms']:.3f} ms, "
            f"full cpu/cpu rate1 "
            f"{case['timing']['cpu_cpu_full_rate1.0']['median_ms']:.3f} ms, "
            f"full gpu/gpu rate1 "
            f"{case['timing']['gpu_gpu_full_rate1.0']['median_ms']:.3f} ms, "
            f"full cpu/cpu rate0.5 "
            f"{case['timing']['cpu_cpu_full_rate0.5']['median_ms']:.3f} ms, "
            f"full gpu/gpu rate0.5 "
            f"{case['timing']['gpu_gpu_full_rate0.5']['median_ms']:.3f} ms",
            flush=True,
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "a2_outbound_validation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
