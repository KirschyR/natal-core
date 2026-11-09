"""V1: verify deterministic CPU-CSR vs GPU-FFT (route A) migration agreement.

Builds one small age-structured hex population with a central homing-drive
release, clones it, and runs two deterministic trajectories:

* ``migration_execution="csr"`` on the CPU (the golden path);
* ``migration_execution="fft"`` with ``enable_gpu()`` (route A cuFFT).

Route A convolves in f32, so the two are not bit-identical.  This script
records per-tick total population and adult carrier fraction, reports the
relative/absolute differences, checks that repeated FFT runs are reproducible,
and writes line plots plus a JSON summary.

Usage:
    python chasing_model/verify_routeA_deterministic.py [--arena 15] [--ticks 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import natal as nt  # noqa: E402
from natal.frontend.spatial import HexGrid, batch_setting  # noqa: E402
from natal.frontend.spatial.topology import build_gaussian_kernel  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

F = 8
M = 5
N_AGES = 10
NEW_ADULT_AGE = 2
EGGS = 40.0
SF = (F + 1) / 2.0
K_AGE1 = 4.0 / (F + 1)
COMPETITION_STRENGTH = 5.0
LOW_DENSITY_GROWTH_RATE = 5.0
CARRIER_ZTYPES = (1, 2)  # WT|Drive, Drive|Drive for alleles [WT, Drive]

_FEMALE_AGING = [1.0 - 1.0 / (F - i + 1) for i in range(1, F)]
_MALE_AGING = [1.0 - 1.0 / (M - i + 1) for i in range(1, M)]
FEMALE_SURVIVAL = [1.0, 1.0, *_FEMALE_AGING, 0.0]
MALE_SURVIVAL = [1.0, 1.0, *_MALE_AGING] + [0.0] * (N_AGES - 2 - len(_MALE_AGING))


def age_vector() -> np.ndarray:
    """Return the reference 15-class age density vector."""
    ages_m = (1.0 / SF) * np.array([1.0 - i / M for i in range(M)])
    ages_f = (1.0 / SF) * np.array([1.0 - i / F for i in range(F)])
    return np.concatenate([[EGGS, 4.0 / (F + 1)], ages_m, ages_f])


def initial_states(m: int, n: int, radius: float, release_multiple: float) -> list[dict]:
    """WT equilibrium everywhere plus a both-sex Drive|Drive disc at age 2."""
    ages = age_vector()
    base_f = np.zeros(N_AGES)
    base_m = np.zeros(N_AGES)
    base_f[0] = ages[0] * 0.5
    base_m[0] = ages[0] * 0.5
    base_f[1] = ages[1] * 0.5
    base_m[1] = ages[1] * 0.5
    base_f[NEW_ADULT_AGE:] = ages[2 + M :]
    base_m[NEW_ADULT_AGE : NEW_ADULT_AGE + M] = ages[2 : 2 + M]

    rr, cc = np.mgrid[0:m, 0:n]
    disc = (rr - (m - 1) / 2.0) ** 2 + (cc - (n - 1) / 2.0) ** 2 <= radius**2

    states: list[dict] = []
    for r in range(m):
        for c in range(n):
            f = base_f.copy()
            mm = base_m.copy()
            drive_f = np.zeros(N_AGES)
            drive_m = np.zeros(N_AGES)
            if disc[r, c]:
                drive_f[NEW_ADULT_AGE] = release_multiple * base_f[NEW_ADULT_AGE]
                drive_m[NEW_ADULT_AGE] = release_multiple * base_m[NEW_ADULT_AGE]
            states.append(
                {
                    "female": {"WT|WT": f.tolist(), "Drive|Drive": drive_f.tolist()},
                    "male": {"WT|WT": mm.tolist(), "Drive|Drive": drive_m.tolist()},
                }
            )
    return states


def build(
    m: int, n: int, execution: str, kernel: np.ndarray, tag: str = ""
) -> nt.SpatialPopulation:
    """Build the deterministic hex age-structured population."""
    species = nt.Species.from_dict(
        name=f"RouteAVerify_{execution}{tag}", structure={"chr1": {"loc": ["WT", "Drive"]}}
    )
    drive = nt.HomingDrive(
        name="RouteAHoming",
        drive_allele="Drive",
        target_allele="WT",
        cas9_allele="Drive",
        drive_conversion_rate=0.8,
        late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0,
        functional_resistance_ratio=0.0,
        fecundity_scaling=1.0,
        viability_scaling=1.0,
    )
    states = initial_states(m, n, radius=2.0, release_multiple=0.5)
    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=m * n,
            topology=HexGrid(rows=m, cols=n),
            pop_type="age_structured",
        )
        .setup(name=f"routeA_{execution}", stochastic=False, continuous_sampling=False)
        .age_structure(n_ages=N_AGES, new_adult_age=NEW_ADULT_AGE)
        .initial_state(individual_count=batch_setting(states))
        .survival(
            female_age_based_survival=FEMALE_SURVIVAL,
            male_age_based_survival=MALE_SURVIVAL,
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 0.0] + [1.0] * (N_AGES - 2),
            male_age_based_mating_rate=[0.0, 0.0] + [1.0] * (N_AGES - 2),
            eggs_per_female=EGGS,
            sex_ratio=0.5,
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
        .build()
    )


def _state(pop: nt.SpatialPopulation) -> np.ndarray:
    """Return the native ``(n_demes, 2, n_ages, Z)`` individual-count array."""
    return np.asarray(pop._native_stacked_state()[1], dtype=np.float64)


def metrics(ind: np.ndarray) -> tuple[float, float]:
    """Return ``(total population, adult carrier fraction)``."""
    total = float(ind.sum())
    adult = ind[:, :, NEW_ADULT_AGE :, :].sum()
    carrier = ind[:, :, NEW_ADULT_AGE :, list(CARRIER_ZTYPES)].sum()
    return total, float(carrier / adult) if adult > 0 else 0.0


def run_trajectory(pop: nt.SpatialPopulation, ticks: int) -> dict[str, np.ndarray]:
    """Step one population tick by tick, recording scalar metrics."""
    total = np.zeros(ticks + 1)
    carrier = np.zeros(ticks + 1)
    total[0], carrier[0] = metrics(_state(pop))
    for t in range(1, ticks + 1):
        pop.run(1, record_every=0)
        total[t], carrier[t] = metrics(_state(pop))
    final_carrier = _per_deme_carrier(_state(pop))
    return {"total": total, "carrier": carrier, "final_carrier": final_carrier}


def _per_deme_carrier(ind: np.ndarray) -> np.ndarray:
    """Return per-deme adult carrier fraction."""
    adult = ind[:, :, NEW_ADULT_AGE :, :].sum(axis=(1, 2, 3))
    carrier = ind[:, :, NEW_ADULT_AGE :, list(CARRIER_ZTYPES)].sum(axis=(1, 2, 3))
    return np.divide(carrier, adult, out=np.zeros_like(adult), where=adult > 0)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.divide(np.abs(a - b), denom, out=np.zeros_like(a), where=denom > 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=int, default=15)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--avd", type=float, default=0.8)
    ap.add_argument("--kernel-size", type=int, default=7)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    m = n = args.arena
    kernel = build_gaussian_kernel("hex", size=args.kernel_size, mean_dispersal=args.avd)

    cpu = build(m, n, "csr", kernel, "cpu")
    cpu_res = run_trajectory(cpu, args.ticks)

    gpu = build(m, n, "fft", kernel, "gpu1")
    backend = gpu._rust_spatial_backend
    if backend is None or backend.gpu_status() == "unavailable":
        print("GPU unavailable; skipping FFT trajectory")
        return 1
    backend.enable_gpu()
    gpu_res = run_trajectory(gpu, args.ticks)

    # Reproducibility: a second FFT run must match the first.
    gpu2 = build(m, n, "fft", kernel, "gpu2")
    gpu2._rust_spatial_backend.enable_gpu()
    gpu2_res = run_trajectory(gpu2, args.ticks)

    total_rel = _max_rel(cpu_res["total"], gpu_res["total"])
    carrier_abs = float(np.max(np.abs(cpu_res["carrier"] - gpu_res["carrier"])))
    final_carrier_abs = float(np.max(np.abs(cpu_res["final_carrier"] - gpu_res["final_carrier"])))
    fft_repeat_equal = bool(
        np.array_equal(gpu_res["total"], gpu2_res["total"])
        and np.array_equal(gpu_res["carrier"], gpu2_res["carrier"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.arange(args.ticks + 1)
    axes[0].plot(t, cpu_res["total"], label="CPU CSR", lw=2)
    axes[0].plot(t, gpu_res["total"], "--", label="GPU FFT", lw=2)
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("total population")
    axes[0].set_title(f"Total population (max rel diff {total_rel:.2e})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, cpu_res["carrier"], label="CPU CSR", lw=2)
    axes[1].plot(t, gpu_res["carrier"], "--", label="GPU FFT", lw=2)
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("adult carrier fraction")
    axes[1].set_title(f"Carrier fraction (max abs diff {carrier_abs:.2e})")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(
        f"Deterministic route A check: {m}x{n} hex, kernel {args.kernel_size}, avd={args.avd:g}"
    )
    fig.tight_layout()
    png = RESULTS / "routeA_deterministic.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    summary = {
        "arena": m,
        "ticks": args.ticks,
        "kernel_size": args.kernel_size,
        "avd": args.avd,
        "total_max_rel_diff": total_rel,
        "carrier_max_abs_diff": carrier_abs,
        "final_per_deme_carrier_max_abs_diff": final_carrier_abs,
        "fft_repeat_bit_identical": fft_repeat_equal,
        "cpu_final_total": float(cpu_res["total"][-1]),
        "gpu_final_total": float(gpu_res["total"][-1]),
        "passed": bool(total_rel < 1e-5 and carrier_abs < 1e-6 and fft_repeat_equal),
        "plot": png.name,
    }
    out = RESULTS / "routeA_deterministic.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
