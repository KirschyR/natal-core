"""Diagnostic: does within-tick event ordering explain the M5 engine/reference gap?

The reference ``radial_release_optimizer/main.m`` runs one tick as
``density+survival+aging`` then ``newborn insertion`` (so age-0 persists across
the tick boundary).  natal's ``AgeStructuredPopulation`` runs
``reproduction`` (age-0 insertion) then ``density+survival`` then ``aging``
(age-0 cleared every tick).  This script isolates that ordering difference:

1. ``reference``  - original reference order (from ``m5_radial_reference``).
2. ``reordered``  - same reference math, reordered to natal's R -> D -> A order.
3. ``engine``     - natal ``age_structured`` engine (route A GPU).

If the gap is caused by ordering alone, ``reordered`` should move from the
original reference toward the engine.  Any residual ``reordered`` vs
``engine`` difference is due to the *other* structural differences (density
normalization, sexed juveniles, sperm storage, migration boundary).

Usage:
    python repro/m5_order_experiment.py --arena 500 --t 1 --h 10 --r 0 \
        --weeks 104 --components ref,reordered,engine
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
    mats = data["mats"]
    fitness = data["fitness"].ravel().astype(float)
    kernel = ref.natal_hex_kernel()
    return mats, fitness, kernel


def run_reordered(
    t: int,
    h: int,
    r: int,
    m: int,
    n: int,
    *,
    mats: np.ndarray,
    fitness: np.ndarray,
    kernel: np.ndarray,
    collect_weeks: tuple[int, ...] = (104, 261),
    max_ticks: int = 261,
    capture_maps: bool = False,
) -> "ref.RadialResult":
    """Run the reference math with natal's within-tick order (R -> D -> A).

    Same building blocks as ``m5_radial_reference.run`` (kernel, ``renew``,
    density formula, aging coefficients, migration); only the intra-tick order
    differs: reproduction writes age 0 first, then density regulation acts on
    *this tick's* newborns, then aging shifts and clears age 0.
    """
    age = ref.model_ages()
    data = np.zeros((ref.N_Z, ref.AGES, m, n))
    data[ref.CORD_PURE_W, :, :, :] = age[:, None, None]

    release_mat = np.zeros_like(data)
    mask = ref.release_geometry(m, n, r)
    release_mat[ref.CORD_RELEASE_GENOTYPE, ref.CORD_M1, :, :][mask] = h / 2.0
    release_mat[ref.CORD_RELEASE_GENOTYPE, ref.CORD_F1, :, :][mask] = h / 2.0
    release_num = release_mat.sum()
    release_sum = 0.0
    drop_time = set(range(1, t + 1))
    mid_row = ref._mround(m / 2.0) - 1

    rows: list[list[float]] = []
    s_maps: dict[int, np.ndarray] = {}

    for tick in range(1, max_ticks + 1):
        if tick in drop_time:
            data = data + release_mat
            release_sum += release_num
        if data[ref.CARRIER_INDEX][:, ref.ADULT_AGES].sum() < 1e-5:
            break

        # --- R: reproduction (natal order) -> overwrite age 0 with newborns ---
        total_female = data[:, ref.CORD_F1 : ref.AGES].sum(axis=(0, 1))
        newborns = ref.EGGS * total_female[None, :, :] * ref.renew(data, mats, fitness)
        data[:, 0, :, :] = newborns

        # --- D: density regulation on this tick's age-0 + age-1 ---
        j1 = data[:, 0].sum(axis=0)
        j2 = data[:, 1].sum(axis=0)
        rr = (j1 + 5.0 * j2) / (age[0] + 5.0 * age[1])
        s1 = ref.LAMBDA / ((ref.LAMBDA - 1.0) * rr + 1.0) * 2.0 / ref.SF / ref.EGGS
        data[:, 0, :, :] *= s1

        # --- A: aging (shift, split juvenile1, clear age 0) ---
        out = np.zeros_like(data)
        out[:, 1] = data[:, 0]
        out[:, ref.CORD_M1] = 0.5 * data[:, 1]
        out[:, ref.CORD_F1] = 0.5 * data[:, 1]
        for i in range(1, ref.F):
            out[:, ref.CORD_F1 + i] = (1.0 - 1.0 / (ref.F - i + 1)) * data[:, ref.CORD_F1 + i - 1]
        for i in range(1, ref.M):
            out[:, ref.CORD_M1 + i] = (1.0 - 1.0 / (ref.M - i + 1)) * data[:, ref.CORD_M1 + i - 1]
        data = out

        ref._migrate(data, kernel)
        data[np.isnan(data)] = 0.0

        if tick in collect_weeks:
            s = ref._carrier_frequency(data)
            rows.append(ref._coverage_row(tick, h, r, s, mid_row, release_sum))
            if capture_maps:
                s_maps[tick] = s.copy()

    return ref.RadialResult(
        t=t, h=h, r=r, arena=m, release_sum=release_sum, rows=rows, s_maps=s_maps
    )


def _coverage_from_maps(
    s_maps: dict[int, np.ndarray], mid_row: int
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for week, s in s_maps.items():
        for thr in (0.5, 0.9):
            diameter, area = ref.coverage_counts(s, mid_row, thr)
            out[f"{week}_{thr}"] = {"diameter": diameter, "area": area}
    return out


def run_engine(arena: int, t: int, h: float, r: float, weeks: int, kernel: np.ndarray) -> dict:
    import m5_engine_probe as eng

    m = n = arena
    pop = eng.build(m, n, t=t, h=h, r=r, kernel=kernel, execution="fft")
    backend = pop._rust_spatial_backend
    backend.enable_gpu()
    pop.run(n_steps=weeks, record_every=0)
    carrier_idx = [1, 2]  # WT|Drive, Drive|Drive
    adult = slice(eng.NEW_ADULT_AGE, eng.N_AGES)
    s = eng.carrier_frequency_grid(pop, m, n, carrier_idx, adult)
    return eng.coverage(s, ref._mround(m / 2.0) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=int, default=500)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--components", type=str, default="ref,reordered,engine")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    mats, fitness, kernel = _load_inputs()
    m = n = args.arena
    mid_row = ref._mround(m / 2.0) - 1
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    report: dict[str, object] = {
        "arena": args.arena,
        "t": args.t,
        "h": args.h,
        "r": args.r,
        "weeks": args.weeks,
        "coverage": {},
    }

    if "ref" in wanted:
        res = ref.run(
            args.t, args.h, args.r, m, n,
            mats=mats, fitness=fitness, kernel=kernel,
            collect_weeks=(args.weeks,), capture_maps=True,
        )
        report["coverage"]["ref"] = _coverage_from_maps(res.s_maps, mid_row)
        print("ref      :", report["coverage"]["ref"])

    if "reordered" in wanted:
        res = run_reordered(
            args.t, args.h, args.r, m, n,
            mats=mats, fitness=fitness, kernel=kernel,
            collect_weeks=(args.weeks,), capture_maps=True,
        )
        report["coverage"]["reordered"] = _coverage_from_maps(res.s_maps, mid_row)
        print("reordered:", report["coverage"]["reordered"])

    if "engine" in wanted:
        cov = run_engine(args.arena, args.t, args.h, args.r, args.weeks, kernel)
        report["coverage"]["engine"] = cov
        print("engine   :", cov)

    out = Path(args.out) if args.out else (RESULTS / "m5_order_experiment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
