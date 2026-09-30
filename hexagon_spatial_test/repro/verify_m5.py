"""Validate the M5 faithful reference translation against MATLAB golden.

Runs ``m5_radial_reference.run`` for every golden configuration at the
reference 500x500 arena and asserts the week-104 / week-261 coverage diameter,
coverage area and cumulative release count match the MATLAB CSV outputs
exactly.  The 0.5 carrier-frequency coverage (not emitted by the reference
CSV, which uses 0.9) is additionally computed and reported.

Usage:
    python repro/verify_m5.py [--jobs N]

Exit code 0 iff every configuration matches.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_radial_reference as ref  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
REF_DIR = HERE / "ref" / "radial"


def _load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = sio.loadmat(REF_DIR / "homing" / "drive_params.mat")
    mats = data["mats"]
    fitness = data["fitness"].ravel().astype(float)
    kernel = ref.natal_hex_kernel()
    return mats, fitness, kernel


def _run_one(args: tuple[tuple[int, int, int], int, int]) -> dict:
    (t, h, r), m, n = args
    mats, fitness, kernel = _load_inputs()
    res = ref.run(
        t,
        h,
        r,
        m,
        n,
        mats=mats,
        fitness=fitness,
        kernel=kernel,
        collect_weeks=(104, 261),
        capture_maps=True,
    )
    mid_row = ref._mround(m / 2.0) - 1
    thresholds: dict[str, dict[str, float]] = {}
    for week, s in res.s_maps.items():
        for thr in (0.5, 0.9):
            diameter, area = ref.coverage_counts(s, mid_row, thr)
            thresholds[f"{week}_{thr}"] = {"diameter": diameter, "area": area}
    return {"t": t, "h": h, "r": r, "rows": res.rows, "thresholds": thresholds}


def _load_golden() -> dict[tuple[int, int, int], dict]:
    golden: dict[tuple[int, int, int], dict] = {}
    for path in sorted(glob.glob(str(GOLDEN / "m5_radial_homing_arena*_shard*.json"))):
        payload = json.loads(Path(path).read_text())
        for rec in payload["results"]:
            rows = rec["rows"]
            flat = rows[0] if isinstance(rows[0], list) else rows
            week_rows = {int(flat[0]): flat[:6], int(flat[6]): flat[6:12]}
            golden[(rec["t"], rec["h"], rec["r"])] = {
                "arena": rec["arena"],
                "rows": week_rows,
            }
    return golden


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    golden = _load_golden()
    if not golden:
        print("no golden files found", file=sys.stderr)
        return 2

    tasks = [((t, h, r), rec["arena"], rec["arena"]) for (t, h, r), rec in golden.items()]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_run_one, tasks))

    ok = True
    report: list[dict] = []
    for res in results:
        key = (res["t"], res["h"], res["r"])
        g = golden[key]["rows"]
        repro = {int(row[0]): row for row in res["rows"]}
        diffs: dict[int, float] = {}
        for week, grow in g.items():
            if week not in repro:
                diffs[week] = float("inf")
                continue
            diffs[week] = float(np.max(np.abs(np.asarray(repro[week]) - np.asarray(grow))))
        match = all(d == 0.0 for d in diffs.values())
        ok = ok and match
        entry = {
            "t": res["t"],
            "h": res["h"],
            "r": res["r"],
            "arena": golden[key]["arena"],
            "week_rows": {str(k): v for k, v in repro.items()},
            "max_abs_diff_vs_golden": diffs,
            "thresholds": res["thresholds"],
            "match": match,
        }
        report.append(entry)
        flag = "OK " if match else "FAIL"
        print(
            f"[{flag}] (t={res['t']},h={res['h']},r={res['r']}) "
            f"maxdiff={diffs}  cover0.9@104={res['thresholds']['104_0.9']['area']:.0f} "
            f"cover0.5@104={res['thresholds']['104_0.5']['area']:.0f}"
        )

    out = Path(args.out) if args.out else (HERE.parent / "results" / "m5_verify.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"configs": report, "all_match": ok}, indent=2))
    print(f"wrote {out}; all_match={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
