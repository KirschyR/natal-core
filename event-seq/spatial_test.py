"""Spatial age-structured event-order test with an interactive HTML output.

A small hex-metric spatial model built on the same per-deme lifecycle as
``age_model`` (natal semantics):

* per-deme reproduction / density regulation + survival / aging, ordered as the
  chosen event order (``R-S-A`` for natal, ``S-A-R`` for the reference);
* migration last, as an **natal normalized convolution** -- the kernel's centre
  is removed and each source renormalizes by the valid-neighbor weight sum
  ``Z = K' (*) m`` (drop + renormalize), so no ``replicate`` edge padding:

      g = rate * f / Z ;  f_new = f * (1 - rate) + K' (*) g

The output is an interactive Plotly HTML in ``event-seq/img/spatial/`` where the
two orders are shown side by side (one facet each) with a shared time slider and
a play button; the heatmap is the per-deme adult ``Drive|Drive`` fraction.

Usage:
    python event-seq/spatial_test.py [--ticks 60]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import age_model as m  # noqa: E402

from natal.frontend.spatial.topology import build_gaussian_kernel  # noqa: E402

IMG = HERE / "img" / "spatial"

ROWS, COLS = 41, 41
KERNEL_SIZE = 7
MEAN_DISPERSAL = 1.0
MIGRATION_RATE = 1.0
DC = 0.6
VIABILITY_DD = 0.8
FECUNDITY_DD = 0.8
RELEASE_RADIUS = 3.0
# dd males added per patch deme at each adult male age, as a multiple of the
# equilibrium male count of that age (1.0 => the patch's adult males are all dd).
DRIVE_FRACTION = 1.0
ORDERS = ["R-S-A", "S-A-R"]


def kernel_off_center(size: int, mean_dispersal: float) -> np.ndarray:
    """Hex-metric Gaussian kernel with the centre removed (natal ``K'``)."""
    kernel = build_gaussian_kernel("hex", size=size, mean_dispersal=mean_dispersal)
    kernel = kernel.astype(float).copy()
    kernel[size // 2, size // 2] = 0.0
    return kernel


def initial_state(rows: int, cols: int, radius: float, drive_fraction: float) -> np.ndarray:
    """``(N_Z, N_SEX, N_AGES, rows, cols)`` WT equilibrium + central drive males.

    Drive (``Drive|Drive``) males are added at every adult male age in the
    central disk, ``drive_fraction`` times that age's equilibrium male count.
    """
    data = np.zeros((m.N_Z, m.N_SEX, m.N_AGES, rows, cols))
    data[m.CORD_PURE_W, 0] = m.INITIAL_FEMALE[:, None, None]
    data[m.CORD_PURE_W, 1] = m.INITIAL_MALE[:, None, None]
    rr, cc = np.mgrid[0:rows, 0:cols]
    mask = (rr - (rows - 1) / 2.0) ** 2 + (cc - (cols - 1) / 2.0) ** 2 <= radius**2
    for age in range(m.NEW_ADULT_AGE, m.NEW_ADULT_AGE + m.M):
        data[m.CORD_RELEASE_GENOTYPE, 1, age, mask] += drive_fraction * m.INITIAL_MALE[age]
    return data


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


def _event_R(data: np.ndarray, newborns: np.ndarray) -> None:
    data[:, 0, 0] = m.SEX_RATIO * newborns
    data[:, 1, 0] = (1.0 - m.SEX_RATIO) * newborns


def _event_S(data: np.ndarray, density_ref: float, viability_dd: float) -> None:
    j0 = data[:, :, 0].sum(axis=(0, 1))
    j1 = data[:, :, 1].sum(axis=(0, 1))
    r = (j0 + 5.0 * j1) / density_ref
    s1 = m.LAMBDA / ((m.LAMBDA - 1.0) * r + 1.0) * 2.0 / m.SF / m.EGGS
    data[:, :, 0] *= s1[None, None, :, :]
    data *= m.SURVIVAL[:, :, None, None]
    data[m.DRIVE_GENOTYPE, :, m.NEW_ADULT_AGE - 1] *= viability_dd


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


def run_order(
    order: str, rows: int, cols: int, ticks: int, kernel: np.ndarray, valid_sum: np.ndarray
) -> np.ndarray:
    """Return ``(ticks+1, rows, cols)`` adult drive-fraction grids for one order."""
    mats = m.homing_mats(dc=DC)
    data = initial_state(rows, cols, RELEASE_RADIUS, DRIVE_FRACTION)
    frames = [adult_drive_fraction(data)]
    for _ in range(ticks):
        newborns = _reproduction(data, mats, FECUNDITY_DD)
        for name in order.split("-"):
            if name == "R":
                _event_R(data, newborns)
            elif name == "S":
                _event_S(data, m.DENSITY_REF, VIABILITY_DD)
            else:
                _event_A(data)
        _event_M(data, kernel, valid_sum, MIGRATION_RATE)
        frames.append(adult_drive_fraction(data))
    return np.array(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=90)
    args = ap.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    kernel = kernel_off_center(KERNEL_SIZE, MEAN_DISPERSAL)
    valid_sum = fftconvolve(np.ones((ROWS, COLS)), kernel, mode="same")

    grids = np.array(
        [run_order(order, ROWS, COLS, args.ticks, kernel, valid_sum) for order in ORDERS]
    )  # (n_orders, ticks+1, rows, cols)
    np.savez(IMG / "spatial_event_order.npz", grids=grids, orders=np.array(ORDERS))

    fig = px.imshow(
        grids,
        animation_frame=1,
        facet_col=0,
        origin="upper",
        color_continuous_scale="Viridis",
        zmin=0.0,
        zmax=1.0,
        labels={"animation_frame": "tick", "facet_col": "order", "color": "Drive|Drive frac"},
        aspect="equal",
    )
    for annotation in fig.layout.annotations:
        if annotation.text in ("0", "1"):
            annotation.text = ORDERS[int(annotation.text)]

    title = (
        "spatial age-structured event-order (adult Drive|Drive fraction): R-S-A vs S-A-R<br>"
        f"dc={DC:g} | viability_dd={VIABILITY_DD:g} | fecundity_dd={FECUNDITY_DD:g} | "
        f"grid={ROWS}x{COLS} | kernel_size={KERNEL_SIZE} | mean_dispersal={MEAN_DISPERSAL:g} | "
        f"migration_rate={MIGRATION_RATE:g} | release_radius={RELEASE_RADIUS:g} | "
        f"release_frac={DRIVE_FRACTION:g} | boundary=natal-normalized | deterministic"
    )
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"}, coloraxis_colorbar_title="frac")
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 120

    out = IMG / "spatial_event_order.html"
    fig.write_html(out, include_plotlyjs="inline")
    print(f"wrote {out}")
    for i, order in enumerate(ORDERS):
        final = grids[i, -1]
        print(f"  {order}: final covered (frac>=0.5) = {int((final >= 0.5).sum())}/{ROWS*COLS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
