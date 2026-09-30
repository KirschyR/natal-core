"""Bounded natal-engine probe for the M5 age-structured radial-release model.

Builds the reference mosquito lifecycle on natal's ``SpatialPopulation``
``age_structured`` engine (shared age axis, male zero-padding), releases drive
homozygotes in a central circle through declarative ``Op.add`` hooks, and
compares the resulting carrier coverage against the validated faithful
translation (``m5_radial_reference``).

This is a *feasibility probe*: natal's engine differs structurally from the
reference MATLAB loop (drop+renormalize boundary migration vs ``replicate``;
density dependence applied in the reproduction tick vs the next tick's
juvenile transition; no ``d<=25`` kernel truncation by default), so exact
agreement is not expected. It exists to demonstrate that the mapped model
builds, equilibrates, spreads, and that the CSR and route-A FFT migration
paths agree on the device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import natal as nt
from natal.frontend.spatial import HexGrid, batch_setting

import m5_radial_reference as ref

HERE = Path(__file__).resolve().parent
F = ref.F
M = ref.M
N_AGES = 10  # 2 juvenile + 8 female-adult stages; males occupy ages 2..6
NEW_ADULT_AGE = 2
EGGS = ref.EGGS
SEX_RATIO = 0.5
K_AGE1 = float(ref.model_ages()[1])  # reference age-1 total (0.4444)
COMPETITION_STRENGTH = 5.0
LOW_DENSITY_GROWTH_RATE = ref.LAMBDA

_FEMALE_AGING = [1.0 - 1.0 / (F - i + 1) for i in range(1, F)]  # 7 values
_MALE_AGING = [1.0 - 1.0 / (M - i + 1) for i in range(1, M)]  # 4 values


def _survival() -> tuple[list[float], list[float]]:
    female = [1.0, 1.0, *_FEMALE_AGING, 0.0]  # ages 0..9
    male = [1.0, 1.0, *_MALE_AGING] + [0.0] * (N_AGES - 2 - len(_MALE_AGING))
    return female, male


def _initial_state(m: int, n: int) -> list[dict[str, dict[str, list[float]]]]:
    ages = ref.model_ages()
    female = [0.0] * N_AGES
    male = [0.0] * N_AGES
    # Split the reference's unsexed juveniles by the sex ratio.
    female[0] = ages[0] * SEX_RATIO
    male[0] = ages[0] * (1.0 - SEX_RATIO)
    female[1] = ages[1] * SEX_RATIO
    male[1] = ages[1] * (1.0 - SEX_RATIO)
    # Reference female adults live in ages 2..9; male adults in ages 2..6.
    for i in range(F):
        female[NEW_ADULT_AGE + i] = ages[NEW_ADULT_AGE + M + i]
    for i in range(M):
        male[NEW_ADULT_AGE + i] = ages[NEW_ADULT_AGE + i]
    state = {
        "female": {"WT|WT": female},
        "male": {"WT|WT": male},
    }
    return [state for _ in range(m * n)]


def _release_hook(t: int, h: float, demes: list[int]):
    @nt.hook(event="first", priority=0)
    def _release():
        return [
            nt.Op.add(
                genotypes="Drive|Drive",
                ages=NEW_ADULT_AGE,
                sex="both",
                delta=h / 2.0,
                when=f"tick >= 1 and tick <= {t}",
            )
        ]

    return _release, demes


def build(
    m: int,
    n: int,
    *,
    t: int,
    h: float,
    r: float,
    kernel: np.ndarray,
    execution: str = "csr",
):
    """Build the mapped age-structured spatial model with a central release."""
    species = nt.Species.from_dict(name="HexM5Mosquito", structure={"chr1": {"loc": ["WT", "Drive"]}})
    drive = nt.HomingDrive(
        name="M5Homing",
        drive_allele="Drive",
        target_allele="WT",
        cas9_allele="Drive",
        drive_conversion_rate=1.0,
        late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0,
        functional_resistance_ratio=0.0,
        fecundity_scaling=1.0,
        viability_scaling=1.0,
    )
    female_survival, male_survival = _survival()
    mask = ref.release_geometry(m, n, r)
    release_demes = np.flatnonzero(mask.reshape(-1)).tolist()
    hook, demes = _release_hook(t, h, release_demes)

    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=m * n,
            topology=HexGrid(rows=m, cols=n),
            pop_type="age_structured",
        )
        .setup(name="m5_engine", stochastic=False, continuous_sampling=False)
        .age_structure(n_ages=N_AGES, new_adult_age=NEW_ADULT_AGE)
        .initial_state(individual_count=batch_setting(_initial_state(m, n)))
        .survival(female_age_based_survival=female_survival, male_age_based_survival=male_survival)
        .reproduction(
            female_age_based_mating_rate=[0.0, 0.0] + [1.0] * (N_AGES - 2),
            male_age_based_mating_rate=[0.0, 0.0] + [1.0] * (N_AGES - 2),
            eggs_per_female=EGGS,
            sex_ratio=SEX_RATIO,
            fixed_egg_count=True,
            sperm_displacement_rate=1.0,
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            low_density_growth_rate=LOW_DENSITY_GROWTH_RATE,
            age_1_carrying_capacity=K_AGE1,
            competition_strength=COMPETITION_STRENGTH,
        )
        .presets(drive)
        .migration(
            kernel=kernel,
            migration_rate=1.0,
            strategy="kernel",
            migration_execution=execution,
        )
        .hooks(hook, deme=demes)
        .build()
    )


def carrier_frequency_grid(pop, m: int, n: int, carrier_idx: list[int], adult: slice) -> np.ndarray:
    """Return the per-deme adult carrier frequency reshaped to (m, n).

    Uses the native stacked state for a vectorized read (the per-deme Python
    objects are not iterated, so 500x500 remains practical).
    """
    ind = np.asarray(pop._native_stacked_state()[1])  # (n_demes, 2, n_ages, Z)
    carrier = ind[:, :, adult, :][:, :, :, carrier_idx].sum(axis=(1, 2, 3))
    total = ind[:, :, adult, :].sum(axis=(1, 2, 3))
    freq = np.divide(carrier, total, out=np.zeros_like(carrier), where=total > 0)
    return freq.reshape(m, n)


def coverage(s: np.ndarray, mid_row: int) -> dict[str, dict[str, float]]:
    """Coverage counts at 0.5 and 0.9 for a carrier-frequency grid."""
    result: dict[str, dict[str, float]] = {}
    for thr in (0.5, 0.9):
        result[str(thr)] = {
            "diameter": float((s[mid_row, :] >= thr).sum()),
            "area": float((s >= thr).sum()),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=int, default=100)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--execution", choices=["csr", "fft"], default="csr")
    args = ap.parse_args()

    m = n = args.arena
    mats = None  # engine uses its own genetics
    kernel = ref.natal_hex_kernel()
    pop = build(m, n, t=args.t, h=args.h, r=args.r, kernel=kernel, execution=args.execution)
    # Biallelic ordering: 0 = WT|WT, 1 = WT|Drive, 2 = Drive|Drive.
    carrier_idx = [1, 2]
    adult = slice(NEW_ADULT_AGE, N_AGES)

    if args.execution == "fft":
        backend = pop._rust_spatial_backend
        backend.enable_gpu()
        print("gpu status:", backend.gpu_status())

    pop.run(n_steps=args.weeks)
    s = carrier_frequency_grid(pop, m, n, carrier_idx, adult)
    cov = coverage(s, ref._mround(m / 2.0) - 1)
    print(f"engine {m}x{n} (t={args.t},h={args.h},r={args.r}) exec={args.execution}")
    print("  coverage:", cov)
    print("  total pop:", float(np.asarray(pop._native_stacked_state()[1]).sum()))

    res = ref.run(
        args.t,
        args.h,
        args.r,
        m,
        n,
        mats=_homing_mats(),
        fitness=np.ones(5),
        kernel=kernel,
        collect_weeks=(args.weeks,),
        capture_maps=True,
    )
    s_ref = res.s_maps[args.weeks]
    print("  reference coverage:", coverage(s_ref, ref._mround(m / 2.0) - 1))
    return 0


def _homing_mats() -> np.ndarray:
    import scipy.io as sio

    return sio.loadmat(HERE / "ref" / "radial" / "homing" / "drive_params.mat")["mats"]


if __name__ == "__main__":
    raise SystemExit(main())
