"""Deterministic comparison: CPU CSR vs GPU route-A FFT on hex grids.

Runs a small matrix of deterministic age-structured hex populations (homing
drive, central release) twice with identical initial states:

* ``migration_execution="csr"`` on the CPU (reference);
* ``migration_execution="fft"`` with ``enable_gpu()`` (route A cuFFT).

Route A convolves in f32, so agreement is expected at the floating-point
tolerance (~1e-6 relative), not bitwise.  The script records per-tick total
population and adult carrier fraction, the final stacked-state difference, and
checks that repeated FFT runs are bitwise reproducible.  Outputs go to
``results/deterministic_compare/`` plus a self-contained
``results/html/deterministic_compare.html``.

Usage:
    python chasing_model/verify_routeA_deterministic.py
"""

from __future__ import annotations

import base64
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
RESULTS = HERE / "results" / "deterministic_compare"
HTML_DIR = HERE / "results" / "html"

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
RELEASE_RADIUS = 3.0
RELEASE_MULTIPLE = 0.5  # per central deme: 0.5 x the age-2 WT male count

_FEMALE_AGING = [1.0 - 1.0 / (F - i + 1) for i in range(1, F)]
_MALE_AGING = [1.0 - 1.0 / (M - i + 1) for i in range(1, M)]
FEMALE_SURVIVAL = [1.0, 1.0, *_FEMALE_AGING, 0.0]
MALE_SURVIVAL = [1.0, 1.0, *_MALE_AGING] + [0.0] * (N_AGES - 2 - len(_MALE_AGING))

DEFAULT_CONFIGS = [
    {"name": "a15_k7_avd0.8", "arena": 15, "kernel_size": 7, "avd": 0.8, "ticks": 40},
    {"name": "a21_k7_avd1.0", "arena": 21, "kernel_size": 7, "avd": 1.0, "ticks": 60},
    {"name": "a31_k7_avd1.0", "arena": 31, "kernel_size": 7, "avd": 1.0, "ticks": 80},
    {"name": "a21_k11_avd1.5", "arena": 21, "kernel_size": 11, "avd": 1.5, "ticks": 60},
]


def age_vector() -> np.ndarray:
    """Return the reference 15-class age density vector."""
    ages_m = (1.0 / SF) * np.array([1.0 - i / M for i in range(M)])
    ages_f = (1.0 / SF) * np.array([1.0 - i / F for i in range(F)])
    return np.concatenate([[EGGS, 4.0 / (F + 1)], ages_m, ages_f])


def initial_states(m: int, n: int, radius: float, release_multiple: float) -> list[dict]:
    """Uniform WT community plus a male heterozygous-drive release in a central disc.

    RSA note: age 0 is only a transient newborn/egg staging slot, so its initial
    count is left at zero; the designed age-1 and adult counts are seeded
    directly (no burn-in).  Release follows the event-seq spatial pattern
    (male-only, first adult age, central disc) but uses a heterozygous
    ``WT|Drive`` carrier rather than the event-seq ``Drive|Drive`` release.
    """
    ages = age_vector()
    base_f = np.zeros(N_AGES)
    base_m = np.zeros(N_AGES)
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
            drive_m = np.zeros(N_AGES)
            if disc[r, c]:
                drive_m[NEW_ADULT_AGE] = release_multiple * base_m[NEW_ADULT_AGE]
            states.append(
                {
                    "female": {"WT|WT": f.tolist()},
                    "male": {"WT|WT": mm.tolist(), "Drive|Drive": drive_m.tolist()},
                }
            )
    return states


def build(m: int, n: int, execution: str, kernel: np.ndarray, tag: str) -> nt.SpatialPopulation:
    """Build the deterministic hex age-structured population."""
    species = nt.Species.from_dict(
        name=f"RouteAVerify_{execution}_{tag}", structure={"chr1": {"loc": ["WT", "Drive"]}}
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
    states = initial_states(m, n, radius=RELEASE_RADIUS, release_multiple=RELEASE_MULTIPLE)
    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=m * n,
            topology=HexGrid(rows=m, cols=n),
            pop_type="age_structured",
        )
        .setup(name=f"routeA_{execution}_{tag}", stochastic=False, continuous_sampling=False)
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


def metrics(ind: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(all ages, non-egg ages>=1, adult ages>=2, carrier fraction)``.

    The all-age total includes the reference initial age-0 egg pool
    (``eggs=40``), which natal R-S-A overwrites at the first reproduction.  The
    non-egg and adult totals are the meaningful standing-population series.
    """
    all_ages = float(ind.sum())
    non_egg = float(ind[:, :, 1:, :].sum())
    adult = float(ind[:, :, NEW_ADULT_AGE :, :].sum())
    carrier = float(ind[:, :, NEW_ADULT_AGE :, list(CARRIER_ZTYPES)].sum())
    return all_ages, non_egg, adult, (carrier / adult) if adult > 0 else 0.0


def _per_deme_carrier(ind: np.ndarray) -> np.ndarray:
    adult = ind[:, :, NEW_ADULT_AGE :, :].sum(axis=(1, 2, 3))
    carrier = ind[:, :, NEW_ADULT_AGE :, list(CARRIER_ZTYPES)].sum(axis=(1, 2, 3))
    return np.divide(carrier, adult, out=np.zeros_like(adult), where=adult > 0)


def run_trajectory(pop: nt.SpatialPopulation, ticks: int) -> dict[str, np.ndarray]:
    """Step one population tick by tick, recording scalar metrics."""
    all_ages = np.zeros(ticks + 1)
    non_egg = np.zeros(ticks + 1)
    adult = np.zeros(ticks + 1)
    carrier = np.zeros(ticks + 1)
    all_ages[0], non_egg[0], adult[0], carrier[0] = metrics(_state(pop))
    for t in range(1, ticks + 1):
        pop.run(1, record_every=0)
        all_ages[t], non_egg[t], adult[t], carrier[t] = metrics(_state(pop))
    final_state = _state(pop)
    return {
        "all_ages": all_ages,
        "non_egg": non_egg,
        "adult": adult,
        "carrier": carrier,
        "final_state": final_state,
        "final_carrier": _per_deme_carrier(final_state),
    }


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.divide(np.abs(a - b), denom, out=np.zeros_like(a), where=denom > 0)))


def _max_rel_floored(a: np.ndarray, b: np.ndarray, floor: float = 1e-3) -> float:
    """Max relative error with a floor, so near-zero entries do not dominate."""
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return float(np.max(np.abs(a - b) / denom))


def _img_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def run_config(cfg: dict, index: int) -> dict:
    """Run one configuration and return its metrics and plot path."""
    m = n = cfg["arena"]
    kernel = build_gaussian_kernel(
        "hex", size=cfg["kernel_size"], mean_dispersal=cfg["avd"]
    )
    cpu = build(m, n, "csr", kernel, f"{index}_cpu")
    cpu_res = run_trajectory(cpu, cfg["ticks"])

    gpu = build(m, n, "fft", kernel, f"{index}_gpu1")
    backend = gpu._rust_spatial_backend
    if backend is None or backend.gpu_status() == "unavailable":
        raise RuntimeError("GPU unavailable")
    backend.enable_gpu()
    gpu_res = run_trajectory(gpu, cfg["ticks"])

    gpu2 = build(m, n, "fft", kernel, f"{index}_gpu2")
    gpu2._rust_spatial_backend.enable_gpu()
    gpu2_res = run_trajectory(gpu2, cfg["ticks"])

    total_rel = _max_rel(cpu_res["all_ages"], gpu_res["all_ages"])
    non_egg_rel = _max_rel(cpu_res["non_egg"], gpu_res["non_egg"])
    adult_rel = _max_rel(cpu_res["adult"], gpu_res["adult"])
    carrier_abs = float(np.max(np.abs(cpu_res["carrier"] - gpu_res["carrier"])))
    final_carrier_abs = float(
        np.max(np.abs(cpu_res["final_carrier"] - gpu_res["final_carrier"]))
    )
    state_rel = _max_rel(cpu_res["final_state"], gpu_res["final_state"])
    state_rel_floored = _max_rel_floored(cpu_res["final_state"], gpu_res["final_state"])
    state_abs = float(np.max(np.abs(cpu_res["final_state"] - gpu_res["final_state"])))
    fft_repeat = bool(
        np.array_equal(gpu_res["final_state"], gpu2_res["final_state"])
        and np.array_equal(gpu_res["non_egg"], gpu2_res["non_egg"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    t = np.arange(cfg["ticks"] + 1)
    axes[0].plot(t, cpu_res["non_egg"], label="CPU non-egg", lw=2)
    axes[0].plot(t, gpu_res["non_egg"], "--", label="GPU non-egg", lw=2)
    axes[0].plot(t, cpu_res["adult"], label="CPU adult", lw=1.2, ls="-.")
    axes[0].plot(t, gpu_res["adult"], label="GPU adult", lw=1.2, ls=":")
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("population")
    axes[0].set_title(f"Standing pop. (non-egg rel {non_egg_rel:.2e})")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(t, cpu_res["carrier"], label="CPU CSR", lw=2)
    axes[1].plot(t, gpu_res["carrier"], "--", label="GPU FFT", lw=2)
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("adult carrier fraction")
    axes[1].set_title(f"Carrier (max abs {carrier_abs:.2e})")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    diff = (cpu_res["final_carrier"] - gpu_res["final_carrier"]).reshape(m, n)
    limit = max(float(np.abs(diff).max()), 1e-12)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title(f"Final carrier diff (max |CPU-GPU| {final_carrier_abs:.2e})")
    axes[2].set_xlabel("column")
    axes[2].set_ylabel("row")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04,
                 label="CPU - GPU carrier fraction")
    fig.suptitle(
        f"{cfg['name']}: {m}x{n} hex, kernel {cfg['kernel_size']}, avd={cfg['avd']:g}\n"
        "tick 0 all-age total includes the initial age-0 egg pool"
    )
    fig.tight_layout()
    plot = RESULTS / f"{cfg['name']}.png"
    fig.savefig(plot, dpi=150)
    plt.close(fig)

    return {
        **cfg,
        "all_age_max_rel_diff": total_rel,
        "non_egg_max_rel_diff": non_egg_rel,
        "adult_max_rel_diff": adult_rel,
        "carrier_max_abs_diff": carrier_abs,
        "final_per_deme_carrier_max_abs_diff": final_carrier_abs,
        "final_state_max_rel_diff": state_rel,
        "final_state_max_rel_diff_floored": state_rel_floored,
        "final_state_max_abs_diff": state_abs,
        "fft_repeat_bit_identical": fft_repeat,
        "cpu_final_non_egg": float(cpu_res["non_egg"][-1]),
        "gpu_final_non_egg": float(gpu_res["non_egg"][-1]),
        "cpu_final_all_ages": float(cpu_res["all_ages"][-1]),
        "gpu_final_all_ages": float(gpu_res["all_ages"][-1]),
        "plot": plot.name,
    }


def _write_html(summary: dict, html_dir: Path) -> Path:
    rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['arena']}²</td><td>{r['kernel_size']}</td>"
        f"<td>{r['avd']:g}</td><td>{r['non_egg_max_rel_diff']:.2e}</td>"
        f"<td>{r['adult_max_rel_diff']:.2e}</td>"
        f"<td>{r['carrier_max_abs_diff']:.2e}</td>"
        f"<td>{r['final_state_max_rel_diff_floored']:.2e}</td>"
        f"<td>{r['final_state_max_abs_diff']:.2e}</td>"
        f"<td>{r['fft_repeat_bit_identical']}</td></tr>"
        for r in summary["configs"]
    )
    images = "".join(
        f"<h3>{r['name']}</h3><img src='{_img_uri(RESULTS / r['plot'])}'>"
        for r in summary["configs"]
    )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Deterministic CPU CSR vs GPU FFT</title>
<style>body {{ font-family: sans-serif; margin: 16px; }} img {{ max-width: 98vw; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #999; padding: 4px 8px; }}</style>
</head><body>
<h2>Deterministic route A check (CPU CSR vs GPU FFT)</h2>
<p>Same initial state, deterministic model. FFT is f32, so agreement is at the
floating-point tolerance, not bitwise. Trajectory errors are relative; the final
stacked-state check uses absolute error because near-zero entries make relative
error meaningless. FFT repeat bit-identity is required.</p>
<p><b>Initial state / release:</b> under RSA the age-0 egg pool is only a
transient staging slot, so it is initialized to zero; the designed age-1 and
adult WT counts are seeded directly (no burn-in). Drive is released at
initialization as <b>homozygous Drive|Drive adult males only</b> in a central
disc (radius 3), each at 0.5x the age-2 WT male count (matching event-seq). The standing population
is reported as non-egg (age >= 1) and adult (age >= 2).</p>
<table><tr><th>config</th><th>arena</th><th>kernel</th><th>avd</th>
<th>non-egg max rel</th><th>adult max rel</th><th>carrier max abs</th>
<th>final state rel (floor 1e-3)</th><th>final state max abs</th>
<th>FFT repeat bit-identical</th></tr>{rows}</table>
{images}
</body></html>"""
    out = html_dir / "deterministic_compare.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, cfg in enumerate(DEFAULT_CONFIGS):
        print(f"[run] {cfg['name']}")
        results.append(run_config(cfg, index))

    summary = {
        "description": "Deterministic CPU CSR vs GPU route-A FFT (same initial state)",
        "configs": results,
        "all_passed": all(
            r["non_egg_max_rel_diff"] < 1e-5
            and r["adult_max_rel_diff"] < 1e-5
            and r["carrier_max_abs_diff"] < 1e-5
            and r["final_per_deme_carrier_max_abs_diff"] < 1e-3
            and r["final_state_max_abs_diff"] < 1e-3
            and r["fft_repeat_bit_identical"]
            for r in results
        ),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html = _write_html(summary, HTML_DIR)
    print(json.dumps(summary, indent=2))
    print(f"wrote {html}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
