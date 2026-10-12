"""Test sensitivity of the reference to its initial age-0 (egg-pool) value.

Hypothesis to check: the engine overwrites the initial age-0 with the newborns
it produces at tick 1, so ``reference`` and ``engine`` might start from a
different age-0.  This script

1. computes the reference's newborn total at the initial state (what natal
   writes into age 0 at tick 1), and
2. reruns the reference with ``initial_age0`` set to that value (and to 0),
   comparing week-104 coverage against the original reference.

If the original reference already uses ``ages[0] = eggs`` equal to the
computed newborn total, aligning the two is a no-op.

Usage:
    python repro/m5_init_experiment.py --arena 500 --t 1 --h 10 --r 0 --weeks 104
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_radial_reference as ref  # noqa: E402

REF_DIR = HERE / "ref" / "radial"
RESULTS = HERE.parent / "results"


def _load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = sio.loadmat(REF_DIR / "homing" / "drive_params.mat")
    return data["mats"], data["fitness"].ravel().astype(float), ref.natal_hex_kernel()


def initial_newborn_total(mats: np.ndarray, fitness: np.ndarray) -> float:
    """Reference newborn total per deme at the initial equilibrium state."""
    age = ref.model_ages()
    data = np.zeros((ref.N_Z, ref.AGES, 1, 1))
    data[ref.CORD_PURE_W, :, :, :] = age[:, None, None]
    total_female = data[:, ref.CORD_F1 : ref.AGES].sum()
    newborns = ref.EGGS * total_female * ref.renew(data, mats, fitness)
    return float(newborns.sum())


def _coverage(res: "ref.RadialResult", mid_row: int, thr: float) -> dict[str, float]:
    week = res.rows[0][0]
    s = res.s_maps[int(week)]
    diameter, area = ref.coverage_counts(s, mid_row, thr)
    return {"diameter": diameter, "area": area}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=int, default=500)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    mats, fitness, kernel = _load_inputs()
    m = n = args.arena
    mid_row = ref._mround(m / 2.0) - 1
    newborn = initial_newborn_total(mats, fitness)
    print(f"reference newborn total per deme at init = {newborn:.6f}  (ages[0]=eggs={ref.EGGS})")

    report: dict[str, object] = {
        "arena": args.arena,
        "t": args.t,
        "h": args.h,
        "r": args.r,
        "weeks": args.weeks,
        "engine_overwritten_age0_per_deme": newborn,
        "reference_ages0": float(ref.model_ages()[0]),
        "variants": {},
    }

    for label, init in [
        ("original(40)", None),
        ("age0=0", 0.0),
    ]:
        res = ref.run(
            args.t, args.h, args.r, m, n,
            mats=mats, fitness=fitness, kernel=kernel,
            collect_weeks=(args.weeks,), capture_maps=True, initial_age0=init,
        )
        entry = {str(thr): _coverage(res, mid_row, thr) for thr in (0.5, 0.9)}
        report["variants"][label] = entry
        print(f"{label:16s}: {entry}")

    out = Path(args.out) if args.out else (RESULTS / "m5_init_experiment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
