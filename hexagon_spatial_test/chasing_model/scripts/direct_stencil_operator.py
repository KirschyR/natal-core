"""Operator-level check: NumPy CPU CSR vs the direct-stencil two-pass scheme.

The CPU reference is the NumPy mirror of natal's continuous CSR migration
(Beta outbound + per-source Dirichlet).  The candidate is
``direct_stencil_migration`` with independent Philox streams and a replayed
second pass.  The CPU and stencil RNG streams differ, so the comparison is
distributional; same-implementation repeatability is checked bit for bit.

Run from the repository root:

    source .venv/bin/activate
    unset PYTHONHOME PYO3_PYTHON PYTHONPATH
    python hexagon_spatial_test/chasing_model/scripts/direct_stencil_operator.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import compare_migration_models as reference
import direct_stencil_migration as stencil
import numpy as np

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"

LAGS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
CONFIGS = [
    {"name": "small", "arena": 15, "kernel_size": 7, "avd": 0.8, "reps": 100},
    {"name": "large", "arena": 31, "kernel_size": 11, "avd": 1.5, "reps": 50},
]


def _bootstrap_lag_covariance(
    samples: np.ndarray,
    rows: int,
    cols: int,
    rng: np.random.Generator,
    draws: int = 200,
) -> dict[str, tuple[float, float]]:
    """Return lag-covariance point estimates and bootstrap standard errors."""
    point = reference.lag_covariance(samples, rows, cols, LAGS)
    reps = samples.shape[0]
    boot = {lag: np.empty(draws) for lag in LAGS}
    for b in range(draws):
        index = rng.integers(0, reps, size=reps)
        values = reference.lag_covariance(samples[index], rows, cols, LAGS)
        for lag in LAGS:
            boot[lag][b] = values[lag]
    return {
        f"{dr},{dc}": (float(point[(dr, dc)]), float(boot[(dr, dc)].std(ddof=1)))
        for dr, dc in LAGS
    }


def _summary(
    cpu_samples: np.ndarray,
    stencil_samples: np.ndarray,
    rows: int,
    cols: int,
    seed: int,
) -> dict[str, object]:
    reps = cpu_samples.shape[0]
    cpu_mean = cpu_samples.mean(axis=0)
    candidate_mean = stencil_samples.mean(axis=0)
    cpu_var = cpu_samples.var(axis=0, ddof=1)
    candidate_var = stencil_samples.var(axis=0, ddof=1)
    mean_diff = candidate_mean - cpu_mean
    mean_scale = np.maximum(np.abs(cpu_mean), 1e-9)
    mean_rel = np.abs(mean_diff) / mean_scale
    mean_se = np.sqrt(cpu_var / reps + candidate_var / reps)
    outside = np.abs(mean_diff) > 3.0 * np.maximum(mean_se, 1e-12)
    ratio = np.divide(
        candidate_var,
        cpu_var,
        out=np.ones_like(candidate_var),
        where=cpu_var > 0.0,
    )
    lags_cpu = _bootstrap_lag_covariance(cpu_samples, rows, cols, np.random.default_rng(seed + 7))
    lags_candidate = _bootstrap_lag_covariance(
        stencil_samples, rows, cols, np.random.default_rng(seed + 8)
    )
    lag_compare = {}
    for dr, dc in LAGS:
        key = f"{dr},{dc}"
        cpu_value, cpu_se = lags_cpu[key]
        candidate_value, candidate_se = lags_candidate[key]
        diff = candidate_value - cpu_value
        lag_compare[key] = {
            "cpu": cpu_value,
            "cpu_se": cpu_se,
            "stencil": candidate_value,
            "stencil_se": candidate_se,
            "diff": diff,
            "diff_se": float(np.hypot(cpu_se, candidate_se)),
        }
    return {
        "reps": reps,
        "mean_abs_rel_median": float(np.median(mean_rel)),
        "mean_abs_rel_max": float(np.max(mean_rel)),
        "mean_diff_outside_3se_fraction": float(outside.mean()),
        "variance_ratio_median": float(np.median(ratio)),
        "variance_ratio_p05": float(np.percentile(ratio, 5)),
        "variance_ratio_p95": float(np.percentile(ratio, 95)),
        "lag_covariance": lag_compare,
        "cpu_total_mean": float(cpu_samples.sum(axis=1).mean()),
        "cpu_total_std": float(cpu_samples.sum(axis=1).std(ddof=1)),
        "stencil_total_mean": float(stencil_samples.sum(axis=1).mean()),
        "stencil_total_std": float(stencil_samples.sum(axis=1).std(ddof=1)),
        "cpu_negative_fraction": float((cpu_samples < 0.0).mean()),
        "stencil_negative_fraction": float((stencil_samples < 0.0).mean()),
    }


def _run_config(
    cfg: dict[str, object],
    reps: int,
    seed: int,
    low: int,
    high: int,
    mode: str,
) -> dict[str, object]:
    rows = cols = int(cfg["arena"])
    kernel_size = int(cfg["kernel_size"])
    avd = float(cfg["avd"])
    kernel = reference.hex_gaussian(kernel_size, avd)
    csr = reference.build_csr(rows, cols, kernel)
    field_rng = np.random.default_rng(seed)
    field = field_rng.integers(low, high, size=(rows, cols)).astype(np.float64)
    n = rows * cols

    cpu_samples = np.empty((reps, n), dtype=np.float64)
    stencil_samples = np.empty((reps, n), dtype=np.float64)
    conservation = np.empty(reps, dtype=np.float64)
    total_error = np.empty(reps, dtype=np.float64)
    stencil_seconds = 0.0
    cpu_seconds = 0.0
    gamma_replay_error = 0.0

    for rep in range(reps):
        rep_seed = seed + 1 + rep
        cpu_rng = np.random.default_rng(rep_seed)
        t0 = time.perf_counter()
        cpu_out = reference.cpu_stochastic(field.ravel(), csr, float(cfg.get("rate", 0.5)), cpu_rng)
        cpu_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        migrate = (
            stencil.stencil_migrate_field
            if mode == "exact"
            else stencil.stencil_migrate_field_fast
        )
        stencil_out, diagnostics = migrate(
            field.ravel(), rows, cols, kernel, float(cfg.get("rate", 0.5)), rep_seed
        )
        stencil_seconds += time.perf_counter() - t0
        cpu_samples[rep] = cpu_out
        stencil_samples[rep] = stencil_out
        assigned = diagnostics["assigned"]
        outbound = diagnostics["outbound"]
        conservation[rep] = float(np.max(np.abs(assigned - outbound)))
        total_error[rep] = float(abs(stencil_out.sum() - field.sum()))
        if "replayed_gamma_sum" in diagnostics:
            gamma_replay_error = max(
                gamma_replay_error,
                float(
                    np.max(
                        np.abs(
                            diagnostics["source_gamma_sum"]
                            - diagnostics["replayed_gamma_sum"]
                        )
                    )
                ),
            )

    repeat_seed = seed + 12345
    repeat_a, _ = migrate(
        field.ravel(), rows, cols, kernel, float(cfg.get("rate", 0.5)), repeat_seed
    )
    repeat_b, _ = migrate(
        field.ravel(), rows, cols, kernel, float(cfg.get("rate", 0.5)), repeat_seed
    )
    repeat_identical = bool(np.array_equal(repeat_a, repeat_b))

    replay_edges = np.arange(4096, dtype=np.uint64) + np.uint64(10_000)
    replay_shape = np.random.default_rng(seed + 99).uniform(0.05, 8.0, size=replay_edges.size)
    replay_a = stencil.gamma_draw(replay_shape, replay_edges, seed + 5, stencil.ROLE_GAMMA)
    replay_b = stencil.gamma_draw(replay_shape, replay_edges, seed + 5, stencil.ROLE_GAMMA)
    gamma_replay_identical = bool(np.array_equal(replay_a, replay_b))

    return {
        "config": cfg,
        "reps": reps,
        "rate": float(cfg.get("rate", 0.5)),
        "initial_total": float(field.sum()),
        "conservation_max_abs_per_rep": {
            "max": float(conservation.max()),
            "median": float(np.median(conservation)),
        },
        "total_error_max": float(total_error.max()),
        "gamma_replay_error_max": gamma_replay_error,
        "repeat_identical": repeat_identical,
        "gamma_replay_identical": gamma_replay_identical,
        "mode": mode,
        "distribution": _summary(cpu_samples, stencil_samples, rows, cols, seed),
        "timing_seconds": {
            "cpu_total": cpu_seconds,
            "stencil_total": stencil_seconds,
            "cpu_per_rep": cpu_seconds / reps,
            "stencil_per_rep": stencil_seconds / reps,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="small,large")
    parser.add_argument("--reps", type=int, default=0, help="override replicate count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low", type=int, default=5)
    parser.add_argument("--high", type=int, default=26)
    parser.add_argument("--mode", choices=("exact", "fast"), default="exact")
    parser.add_argument("--out", type=Path, default=RESULTS / "direct_stencil_operator.json")
    args = parser.parse_args()

    wanted = set(args.configs.split(","))
    payload: dict[str, object] = {
        "reference": "NumPy mirror of natal continuous CSR",
        "seed": args.seed,
        "low": args.low,
        "high": args.high,
        "mode": args.mode,
        "configs": {},
    }
    for cfg in CONFIGS:
        if cfg["name"] not in wanted:
            continue
        reps = args.reps if args.reps > 0 else int(cfg["reps"])
        print(f"[run] {cfg['name']} reps={reps} ...", flush=True)
        payload["configs"][cfg["name"]] = _run_config(
            cfg, reps, args.seed, args.low, args.high, args.mode
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
