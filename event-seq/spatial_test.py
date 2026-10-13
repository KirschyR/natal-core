"""Spatial age-structured event-order test with an interactive HTML output.

A small hex-metric spatial model built on the same per-deme lifecycle as
``age_model`` (natal semantics):

* per-deme reproduction / density regulation + survival / aging, ordered as the
  chosen event order (``R-S-A`` for natal, ``S-A-R`` for the reference);
* migration last, as an **natal normalized convolution** -- the kernel's centre
  is removed and each source renormalizes by the valid-neighbor weight sum
  ``Z = K' (*) m`` (drop + renormalize), so no ``replicate`` edge padding:

      g = rate * f / Z ;  f_new = f * (1 - rate) + K' (*) g

Outputs under ``event-seq/img/spatial/``: an interactive Plotly HTML (two
facets, shared time slider / play; adult ``Drive|Drive`` fraction) and
coverage-curve figures (covered-deme count over time for each threshold in
``COVERS``, using the ``Drive|Drive`` fraction of non-egg individuals).

Stochastic mode mirrors ``age_model``'s natal-style *continuous* samplers
(Beta/Gamma based, mean when ``n <= 1``), vectorized over the grid.

Usage:
    python event-seq/spatial_test.py [--ticks 120] [--runs 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import plotly.express as px
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import age_model as m  # noqa: E402

from natal.frontend.spatial.topology import build_gaussian_kernel  # noqa: E402

IMG = HERE / "img" / "spatial"
COVERS = [0.1, 0.3, 0.5, 0.7, 0.9]

ROWS, COLS = 41, 41
KERNEL_SIZE = 7
MEAN_DISPERSAL = 1.0
MIGRATION_RATE = 1.0
DC = 0.8
VIABILITY_DD = 0.8
FECUNDITY_DD = 0.8
RELEASE_RADIUS = 3.0
RELEASE_MULTIPLE = 0.5  # dd males added at age 2 = RELEASE_MULTIPLE * equilibrium male count
SCALING_FACTORS = [10.0, 100.0]
ORDERS = ["R-S-A", "S-A-R"]

_EPS = 1e-12


def kernel_off_center(size: int, mean_dispersal: float) -> np.ndarray:
    """Hex-metric Gaussian kernel with the centre removed (natal ``K'``)."""
    kernel = build_gaussian_kernel("hex", size=size, mean_dispersal=mean_dispersal)
    kernel = kernel.astype(float).copy()
    kernel[size // 2, size // 2] = 0.0
    return kernel


def initial_state(
    rows: int, cols: int, radius: float, release_multiple: float, scale: float
) -> np.ndarray:
    """``(N_Z, N_SEX, N_AGES, rows, cols)`` WT equilibrium (+ central drive) * scale."""
    data = np.zeros((m.N_Z, m.N_SEX, m.N_AGES, rows, cols))
    data[m.CORD_PURE_W, 0] = m.INITIAL_FEMALE[:, None, None]
    data[m.CORD_PURE_W, 1] = m.INITIAL_MALE[:, None, None]
    rr, cc = np.mgrid[0:rows, 0:cols]
    mask = (rr - (rows - 1) / 2.0) ** 2 + (cc - (cols - 1) / 2.0) ** 2 <= radius**2
    data[m.CORD_RELEASE_GENOTYPE, 1, m.NEW_ADULT_AGE, mask] += (
        release_multiple * m.INITIAL_MALE[m.NEW_ADULT_AGE]
    )
    return data * scale


# --- natal-style continuous samplers (vectorized) ---------------------------


def _v_poisson(rng: np.random.Generator, lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    out = np.zeros_like(lam)
    mask = lam > _EPS
    if mask.any():
        out[mask] = rng.gamma(shape=lam[mask], scale=1.0)
    return out


def _v_binomial(rng: np.random.Generator, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    n, p = (np.asarray(a, dtype=float) for a in np.broadcast_arrays(n, p))
    out = np.zeros(n.shape)
    large_p = p >= 1.0 - _EPS
    out[large_p] = n[large_p]
    middle = (p > _EPS) & ~large_p
    small_n = middle & (n <= 1.0 + _EPS)
    out[small_n] = n[small_n] * p[small_n]
    big_n = middle & (n > 1.0 + _EPS)
    if big_n.any():
        concentration = n[big_n] - 1.0
        alpha = np.maximum(p[big_n] * concentration, _EPS)
        beta = np.maximum((1.0 - p[big_n]) * concentration, _EPS)
        num = rng.gamma(shape=alpha, scale=1.0)
        den = rng.gamma(shape=beta, scale=1.0)
        denom = num + den
        proportion = np.zeros_like(num)
        np.divide(num, denom, out=proportion, where=denom > _EPS)
        out[big_n] = proportion * n[big_n]
    return out


def _v_multinomial(rng: np.random.Generator, nvec: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Continuous multinomial over axis 0; ``probs`` is ``(K, M)``, ``nvec`` ``(M,)``."""
    probs = np.asarray(probs, dtype=float)
    n = np.asarray(nvec, dtype=float)[None, :]
    out = n * probs
    big = (n > 1.0 + _EPS).ravel()
    if big.any():
        nb = n[:, big]
        pb = probs[:, big]
        alpha = np.maximum(pb * (nb - 1.0), 0.0)
        draws = np.zeros_like(alpha)
        mask = alpha > _EPS
        draws[mask] = rng.gamma(shape=alpha[mask], scale=1.0)
        total = draws.sum(axis=0, keepdims=True)
        good = total > _EPS
        out[:, big] = np.where(good, draws * (nb / np.where(good, total, 1.0)), pb * nb)
    return out


# --- lifecycle --------------------------------------------------------------


def _reproduction(data: np.ndarray, mats: np.ndarray, fecundity_dd: float) -> np.ndarray:
    fecundity = np.ones(m.N_Z)
    fecundity[m.DRIVE_GENOTYPE] = fecundity_dd
    female_w = data[:, 0, m.NEW_ADULT_AGE:].sum(axis=1) * fecundity[:, None, None]
    male_w = data[:, 1, m.NEW_ADULT_AGE:].sum(axis=1) * fecundity[:, None, None]
    total_female = female_w.sum(axis=0)
    total_male = male_w.sum(axis=0)
    female_freq = np.divide(female_w, total_female, out=np.zeros_like(female_w), where=total_female > 0)
    male_freq = np.divide(male_w, total_male, out=np.zeros_like(male_w), where=total_male > 0)
    m3 = mats.reshape(m.N_Z, m.N_Z, m.N_Z)
    prop = np.einsum("irc,jrc,kij->krc", female_freq, male_freq, m3)
    return m.EGGS * total_female[None, :, :] * prop


def _reproduction_stochastic(
    data: np.ndarray, mats: np.ndarray, fecundity_dd: float, rng: np.random.Generator
) -> np.ndarray:
    fecundity = np.ones(m.N_Z)
    fecundity[m.DRIVE_GENOTYPE] = fecundity_dd
    female_w = data[:, 0, m.NEW_ADULT_AGE:].sum(axis=1) * fecundity[:, None, None]
    male_w = data[:, 1, m.NEW_ADULT_AGE:].sum(axis=1) * fecundity[:, None, None]
    total_female = female_w.sum(axis=0)
    total_male = male_w.sum(axis=0)
    female_freq = np.divide(female_w, total_female, out=np.zeros_like(female_w), where=total_female > 0)
    male_freq = np.divide(male_w, total_male, out=np.zeros_like(male_w), where=total_male > 0)
    m3 = mats.reshape(m.N_Z, m.N_Z, m.N_Z)
    prop = np.einsum("irc,jrc,kij->krc", female_freq, male_freq, m3)
    total_eggs = _v_poisson(rng, (m.EGGS * total_female).ravel()).reshape(total_female.shape)
    rows, cols = total_female.shape
    offspring = _v_multinomial(rng, total_eggs.ravel(), prop.reshape(m.N_Z, rows * cols))
    return offspring.reshape(m.N_Z, rows, cols)


def _event_R(data: np.ndarray, newborns: np.ndarray) -> None:
    data[:, 0, 0] = m.SEX_RATIO * newborns
    data[:, 1, 0] = (1.0 - m.SEX_RATIO) * newborns


def _event_R_stochastic(data: np.ndarray, newborns: np.ndarray, rng: np.random.Generator) -> None:
    female = _v_binomial(rng, newborns, m.SEX_RATIO)
    data[:, 0, 0] = female
    data[:, 1, 0] = newborns - female


def _density_scaling(data: np.ndarray, density_ref: float) -> np.ndarray:
    j0 = data[:, :, 0].sum(axis=(0, 1))
    j1 = data[:, :, 1].sum(axis=(0, 1))
    r = (j0 + 5.0 * j1) / density_ref
    return m.LAMBDA / ((m.LAMBDA - 1.0) * r + 1.0) * 2.0 / m.SF / m.EGGS


def _event_S(data: np.ndarray, density_ref: float, viability_dd: float) -> None:
    data[:, :, 0] *= _density_scaling(data, density_ref)[None, None, :, :]
    data *= m.SURVIVAL[:, :, None, None]
    data[m.DRIVE_GENOTYPE, :, m.NEW_ADULT_AGE - 1] *= viability_dd


def _event_S_stochastic(
    data: np.ndarray, density_ref: float, viability_dd: float, rng: np.random.Generator
) -> None:
    shape = data.shape
    s1 = _density_scaling(data, density_ref)
    age0 = data[:, :, 0]
    total0 = age0.sum(axis=(0, 1))
    probs = np.divide(
        age0.reshape(m.N_Z * m.N_SEX, -1),
        total0.reshape(1, -1),
        out=np.zeros((m.N_Z * m.N_SEX, total0.size)),
        where=total0.reshape(1, -1) > 0,
    )
    draws = _v_multinomial(rng, (total0 * s1).ravel(), probs)
    data[:, :, 0] = draws.reshape(m.N_Z, m.N_SEX, *total0.shape)
    survival = np.broadcast_to(m.SURVIVAL[:, :, None, None], shape).astype(float).copy()
    survival[m.DRIVE_GENOTYPE, :, m.NEW_ADULT_AGE - 1] *= viability_dd
    data[:] = _v_binomial(rng, data, survival)


def _event_A(data: np.ndarray) -> None:
    data[:, :, 1:] = data[:, :, :-1].copy()
    data[:, :, 0] = 0.0


def _event_M(data: np.ndarray, kernel: np.ndarray, valid_sum: np.ndarray, rate: float) -> None:
    adults = data[:, :, m.NEW_ADULT_AGE:]
    shape = adults.shape
    planes = adults.reshape(-1, adults.shape[-2], adults.shape[-1])
    g = rate * planes / valid_sum[None, :, :]
    conv = fftconvolve(g, kernel[None, :, :], mode="same", axes=(1, 2))
    data[:, :, m.NEW_ADULT_AGE:] = (planes * (1.0 - rate) + conv).reshape(shape)


def adult_drive_fraction(data: np.ndarray) -> np.ndarray:
    """Per-deme adult ``Drive|Drive`` fraction, shape ``(rows, cols)``."""
    total = data[:, :, m.NEW_ADULT_AGE:].sum(axis=(0, 1, 2))
    drive = data[m.DRIVE_GENOTYPE, :, m.NEW_ADULT_AGE:].sum(axis=(0, 1))
    return np.divide(drive, total, out=np.zeros_like(drive), where=total > 0)


def non_egg_drive_fraction(data: np.ndarray) -> np.ndarray:
    """Per-deme ``Drive|Drive`` fraction of non-egg individuals (ages >= 1)."""
    total = data[:, :, 1:].sum(axis=(0, 1, 2))
    drive = data[m.DRIVE_GENOTYPE, :, 1:].sum(axis=(0, 1))
    return np.divide(drive, total, out=np.zeros_like(drive), where=total > 0)


def _coverage_counts(data: np.ndarray) -> np.ndarray:
    frac = non_egg_drive_fraction(data)
    return np.array([float((frac > cover).sum()) for cover in COVERS])


def run_replicate(
    order: str, rows: int, cols: int, ticks: int, kernel: np.ndarray, valid_sum: np.ndarray,
    *, scale: float, stochastic: bool, seed: int, keep_grids: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return ``(adult_grids or None, coverage_counts)`` for one replicate."""
    rng = np.random.default_rng(seed)
    mats = m.homing_mats(dc=DC)
    data = initial_state(rows, cols, RELEASE_RADIUS, RELEASE_MULTIPLE, scale)
    density_ref = scale * m.DENSITY_REF
    coverage = np.zeros((ticks + 1, len(COVERS)))
    coverage[0] = _coverage_counts(data)
    grids = np.zeros((ticks + 1, rows, cols)) if keep_grids else None
    if grids is not None:
        grids[0] = adult_drive_fraction(data)
    for tick in range(1, ticks + 1):
        if stochastic:
            newborns = _reproduction_stochastic(data, mats, FECUNDITY_DD, rng)
        else:
            newborns = _reproduction(data, mats, FECUNDITY_DD)
        for name in order.split("-"):
            if name == "R":
                _event_R_stochastic(data, newborns, rng) if stochastic else _event_R(data, newborns)
            elif name == "S":
                if stochastic:
                    _event_S_stochastic(data, density_ref, VIABILITY_DD, rng)
                else:
                    _event_S(data, density_ref, VIABILITY_DD)
            else:
                _event_A(data)
        _event_M(data, kernel, valid_sum, MIGRATION_RATE)
        coverage[tick] = _coverage_counts(data)
        if grids is not None:
            grids[tick] = adult_drive_fraction(data)
    return grids, coverage


# --- plotting ---------------------------------------------------------------


def _params_title(br: str, *, stochastic: bool, scale: float) -> str:
    mode = f"stochastic | sf={scale:g}" if stochastic else "deterministic"
    return (
        f"dc={DC:g} | viability_dd={VIABILITY_DD:g} | fecundity_dd={FECUNDITY_DD:g} | "
        f"grid={ROWS}x{COLS} | kernel_size={KERNEL_SIZE} | mean_dispersal={MEAN_DISPERSAL:g}{br}"
        f"migration_rate={MIGRATION_RATE:g} | release_radius={RELEASE_RADIUS:g} | "
        f"release_age2_mult={RELEASE_MULTIPLE:g} | boundary=natal-normalized | {mode}"
    )


def _coverage_figure(
    coverage_by_order: dict[str, np.ndarray], ticks: int, out: Path, title: str
) -> Path:
    """Dual-panel coverage figure; arrays are ``(runs, ticks+1, n_covers)``."""
    x = np.arange(ticks + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, order in zip(axes, ORDERS):
        arr = coverage_by_order[order]
        mean = arr.mean(axis=0)
        n_runs = arr.shape[0]
        ci = (1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else None
        for idx, cover in enumerate(COVERS):
            line = ax.plot(x, mean[:, idx], linewidth=1.6, label=f"cover>{cover:g}")[0]
            if ci is not None:
                ax.fill_between(x, mean[:, idx] - ci[:, idx], mean[:, idx] + ci[:, idx],
                                alpha=0.25, color=line.get_color())
        ax.set_title(order)
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"covered demes (of {ROWS * COLS})")
    axes[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    kernel = kernel_off_center(KERNEL_SIZE, MEAN_DISPERSAL)
    valid_sum = fftconvolve(np.ones((ROWS, COLS)), kernel, mode="same")

    # deterministic: interactive HTML + coverage figure
    adult_by_order: dict[str, np.ndarray] = {}
    det_coverage: dict[str, np.ndarray] = {}
    for order in ORDERS:
        grids, coverage = run_replicate(
            order, ROWS, COLS, args.ticks, kernel, valid_sum,
            scale=1.0, stochastic=False, seed=0, keep_grids=True,
        )
        adult_by_order[order] = grids if grids is not None else np.zeros(0)
        det_coverage[order] = coverage[None, :, :]
    grids = np.array([adult_by_order[order] for order in ORDERS])
    np.savez(IMG / "spatial_event_order.npz", grids=grids, orders=np.array(ORDERS))

    fig = px.imshow(
        grids, animation_frame=1, facet_col=0, origin="upper",
        color_continuous_scale="Viridis", zmin=0.0, zmax=1.0,
        labels={"animation_frame": "tick", "facet_col": "order", "color": "Drive|Drive frac"},
        aspect="equal",
    )
    for annotation in fig.layout.annotations:
        if annotation.text in ("0", "1"):
            annotation.text = ORDERS[int(annotation.text)]
    fig.update_layout(
        title={"text": "spatial event-order (adult Drive|Drive fraction): R-S-A vs S-A-R<br>"
               + _params_title("<br>", stochastic=False, scale=1.0),
               "x": 0.5, "xanchor": "center"},
        coloraxis_colorbar_title="frac",
    )
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 120
    html_out = IMG / "spatial_event_order.html"
    fig.write_html(html_out, include_plotlyjs="inline")
    print(f"wrote {html_out}")

    det_out = _coverage_figure(
        det_coverage, args.ticks, IMG / "spatial_coverage_curves.png",
        "covered demes over time (Drive|Drive fraction of non-egg individuals)\n"
        + _params_title("\n", stochastic=False, scale=1.0),
    )
    print(f"wrote {det_out}")

    # stochastic: one coverage figure per scaling factor
    for scale in SCALING_FACTORS:
        coverage_by_order = {}
        for order in ORDERS:
            runs = [
                run_replicate(order, ROWS, COLS, args.ticks, kernel, valid_sum,
                              scale=scale, stochastic=True, seed=r)[1]
                for r in range(args.runs)
            ]
            coverage_by_order[order] = np.array(runs)
        out = _coverage_figure(
            coverage_by_order, args.ticks, IMG / f"spatial_coverage_curves_sf{scale:g}.png",
            "covered demes over time (Drive|Drive frac of non-egg; mean +/-95% CI over "
            f"{args.runs} runs)\n" + _params_title("\n", stochastic=True, scale=scale),
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
