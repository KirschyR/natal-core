"""Full-model strategy comparison at migration rate 0.5.

The archived v3 comparison used ``migration_rate=1.0``.  This script reruns the
same CPU/M0/M1/M2 NumPy models with ``migration_rate=0.5`` into a separate
``results/strategies_rate05/`` tree, leaving ``strategies_v1``/``v2``/``v3``
untouched.  It then reports consistency metrics (selected-deme drive, T50,
total, coverage/front) relative to the CPU reference and relative to the
archived rate-1 CPU series.

Run with the natal virtualenv:

    source .venv/bin/activate
    unset PYTHONHOME PYO3_PYTHON PYTHONPATH
    python chasing_model/compare_rate05.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import compare_strategies_v3 as v3
import numpy as np

# Scripts live in chasing_model/scripts/; outputs stay under chasing_model/.
HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results" / "strategies_rate05"
HTML_DIR = HERE / "results" / "html"
RATE = 0.5
CONFIGS = [
    {
        **v3.CONFIGS[0],
        "name": "small_rate05",
        "migration_rate": RATE,
        "reps": 20,
    },
    {
        **v3.CONFIGS[1],
        "name": "large_rate05",
        "migration_rate": RATE,
        "reps": 10,
    },
]
SOURCE_CONFIG = {"small_rate05": "small", "large_rate05": "large"}


def _series(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def _t50(drive: np.ndarray) -> np.ndarray:
    selected = drive.mean(axis=2)
    hit = selected >= 0.5
    index = np.argmax(hit, axis=1)
    return np.where(hit.any(axis=1), index, selected.shape[1])


def _method_metrics(series: dict[str, np.ndarray], method: str) -> dict[str, Any]:
    drive = series[f"{method}_drive"]
    total = series[f"{method}_total"]
    front = series[f"{method}_front"]
    coverage = series[f"{method}_coverage"]
    t40 = min(40, drive.shape[1] - 1)
    return {
        "drive_t40_mean": drive[:, t40].mean(axis=0).tolist(),
        "drive_t40_sd": drive[:, t40].std(axis=0, ddof=1).tolist(),
        "t50_mean": float(_t50(drive).mean()),
        "t50_sd": float(_t50(drive).std(ddof=1)),
        "final_total_mean": float(total[:, -1].mean()),
        "final_total_sd": float(total[:, -1].std(ddof=1)),
        "final_front_mean": float(front[:, -1].mean()),
        "final_coverage_mean": coverage[:, -1].mean(axis=0).tolist(),
        "reps": int(drive.shape[0]),
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_drive = np.asarray(reference["drive_t40_mean"])
    drive = np.asarray(candidate["drive_t40_mean"])
    return {
        "drive_t40_abs_diff": np.abs(drive - ref_drive).tolist(),
        "drive_t40_abs_diff_median": float(np.median(np.abs(drive - ref_drive))),
        "drive_t40_abs_diff_max": float(np.max(np.abs(drive - ref_drive))),
        "t50_mean_diff": candidate["t50_mean"] - reference["t50_mean"],
        "final_total_mean_rel_diff": (
            candidate["final_total_mean"] - reference["final_total_mean"]
        )
        / reference["final_total_mean"],
        "final_front_mean_diff": (
            candidate["final_front_mean"] - reference["final_front_mean"]
        ),
        "final_coverage_mean_diff": (
            np.asarray(candidate["final_coverage_mean"])
            - np.asarray(reference["final_coverage_mean"])
        ).tolist(),
    }


def summarize() -> dict[str, Any]:
    out: dict[str, Any] = {"rate": RATE, "configs": {}}
    for cfg in CONFIGS:
        name = cfg["name"]
        source = SOURCE_CONFIG[name]
        series = _series(RESULTS / name / "series.npz")
        methods = {m: _method_metrics(series, m) for m in v3.STRATEGIES}
        comparisons = {
            m: _compare(methods["cpu"], methods[m])
            for m in ("m0", "m1", "m2")
        }
        v3_path = HERE / "results" / "strategies_v3" / source / "series.npz"
        rate1 = None
        if v3_path.exists():
            v3_series = _series(v3_path)
            rate1 = {
                "cpu_rate1": _method_metrics(v3_series, "cpu"),
                "cpu_rate05": methods["cpu"],
            }
            rate1["rate05_vs_rate1_cpu"] = _compare(
                rate1["cpu_rate1"], rate1["cpu_rate05"]
            )
        out["configs"][name] = {
            "config": cfg,
            "methods": methods,
            "comparisons_vs_cpu": comparisons,
            "rate05_vs_rate1": rate1,
        }
    return out


def main() -> int:
    v3.RESULTS = RESULTS
    v3.HTML_DIR = HTML_DIR
    RESULTS.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        print(f"[run] {cfg['name']} rate={RATE} ...", flush=True)
        v3.run_config(cfg, jobs=8)
    summary = summarize()
    (RESULTS / "consistency.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
