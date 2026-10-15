"""Phase-0 scaffold: synthetic suitability field H and two ways to feed it into the deme model.

Companion to ``simple_spatial_model_ref/`` (the age-structured deme lifecycle from
the event-order study).  This module keeps the same per-deme lifecycle (density
regulation + survival + aging, S-A-R order) but strips the genotype machinery to
a single wild-type population, so that the only thing that varies is how a
geographic suitability field ``H`` enters the dynamics:

* ``mode="capacity"`` (idea 2, hex-style): migration is the plain normalized
  convolution; ``H`` scales the local density reference, ``D_ref_j = DENSITY_REF * H_j``,
  so a deme's equilibrium population tracks ``H``.
* ``mode="bias"`` (idea 1, ASCENT-style, soft): migration is biased toward high
  ``H`` with a *soft* target weight ``w = H`` (``beta = 1``), i.e. the ASCENT
  hard indicator ``1[H_t >= H_s]`` is dropped.  The operator is

      Z = G (*) w ;  g = f / Z ;  h = G (*) g ;  f_new = (1-r) f + r * w * h

  which is a *separable* operator (input scaling x one convolution x output
  scaling) and therefore keeps a single cached FFT kernel.

The baseline ``H == 1`` must reduce both modes to the same plain model (checked
in ``selftest``).

Usage:
    python geographic-score/gis2score_explore/env_spatial.py --ticks 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simple_spatial_model_ref"))
import age_model as am  # noqa: E402

from natal.frontend.spatial.topology import build_gaussian_kernel  # noqa: E402

ROWS, COLS = 61, 61
KERNEL_SIZE = 7
MEAN_DISPERSAL = 1.0
MIGRATION_RATE = 1.0
# Four Gaussian peaks on the hex grid (row, col); the hex metric is
# d^2 = di^2 + dj^2 + di*dj, so these are physically distinct bumps.
PEAKS = ((16, 16), (16, 44), (44, 16), (44, 44))
PEAK_SIGMA = 6.0
H_FLOOR = 0.10
EPS = 1e-12


def hex_metric(rows: int, cols: int, centers: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return ``(n_centers, rows, cols)`` hex-metric distance to each center."""
    rr, cc = np.mgrid[0:rows, 0:cols]
    out = np.empty((len(centers), rows, cols))
    for k, (ci, cj) in enumerate(centers):
        di, dj = rr - ci, cc - cj
        out[k] = np.sqrt(di * di + dj * dj + di * dj)
    return out


def make_H(
    rows: int = ROWS,
    cols: int = COLS,
    *,
    peaks: tuple[tuple[int, int], ...] = PEAKS,
    sigma: float = PEAK_SIGMA,
    floor: float = H_FLOOR,
) -> np.ndarray:
    """Synthetic suitability field in ``(0, 1]`` with four Gaussian peaks."""
    d = hex_metric(rows, cols, peaks)  # (P, R, C)
    h = np.sum(np.exp(-0.5 * (d / sigma) ** 2), axis=0)
    h = floor + (1.0 - floor) * h / h.max()
    return h


def kernel_off_center(size: int, mean_dispersal: float) -> np.ndarray:
    """Hex-metric Gaussian kernel with the centre removed (natal ``K'``)."""
    kernel = build_gaussian_kernel("hex", size=size, mean_dispersal=mean_dispersal).astype(float).copy()
    kernel[size // 2, size // 2] = 0.0
    return kernel


def hex_geometry(rows: int, cols: int, size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pointy-top hex vertices ``(rows*cols, 6, 2)`` and bounding box.

    Matches natal ``HexGrid.to_xy``: ``x = i + 0.5*j``, ``y = (sqrt(3)/2)*j``
    (``i`` = row, ``j`` = col), so the hex metric ``di^2 + dj^2 + di*dj`` maps to
    Euclidean distance.
    """
    sqrt3 = np.sqrt(3.0)
    vertices = np.empty((rows * cols, 6, 2))
    angles = np.deg2rad(np.arange(6) * 60.0 + 30.0)
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    k = 0
    for i in range(rows):
        for j in range(cols):
            cx = size * (i + 0.5 * j)
            cy = size * (sqrt3 / 2.0) * j
            vertices[k, :, 0] = cx + size * np.cos(angles)
            vertices[k, :, 1] = cy + size * np.sin(angles)
            xmin, xmax = min(xmin, cx - size), max(xmax, cx + size)
            ymin, ymax = min(ymin, cy - size), max(ymax, cy + size)
            k += 1
    return vertices, np.array([xmin, xmax, ymin, ymax])


# --- per-deme lifecycle (single wild-type population, S-A-R order) -----------------


def initial_state(rows: int, cols: int) -> np.ndarray:
    """``(N_SEX, N_AGES, rows, cols)`` WT equilibrium per deme."""
    data = np.zeros((am.N_SEX, am.N_AGES, rows, cols))
    data[0] = am.INITIAL_FEMALE[:, None, None]
    data[1] = am.INITIAL_MALE[:, None, None]
    return data


def reproduction(data: np.ndarray) -> np.ndarray:
    """Newborn count per deme from the tick-start adult females."""
    total_female = data[0, am.NEW_ADULT_AGE:, :, :].sum(axis=0)
    return am.EGGS * total_female


def event_R(data: np.ndarray, newborns: np.ndarray) -> None:
    data[0, 0] = am.SEX_RATIO * newborns
    data[1, 0] = (1.0 - am.SEX_RATIO) * newborns


def event_S(data: np.ndarray, density_ref: np.ndarray) -> None:
    """Beverton-Holt density regulation (per-deme ``density_ref``) + survival."""
    j0 = data[:, 0].sum(axis=0)
    j1 = data[:, 1].sum(axis=0)
    r = (j0 + 5.0 * j1) / np.maximum(density_ref, EPS)
    s1 = am.LAMBDA / ((am.LAMBDA - 1.0) * r + 1.0) * 2.0 / am.SF / am.EGGS
    data[:, 0] *= s1[None, :, :]
    data *= am.SURVIVAL[:, :, None, None]


def event_A(data: np.ndarray) -> None:
    data[:, 1:] = data[:, :-1].copy()
    data[:, 0] = 0.0


def _adult_planes(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    adults = data[:, am.NEW_ADULT_AGE:, :, :]
    shape = adults.shape
    return adults.reshape(-1, *adults.shape[-2:]), shape


def migration(
    data: np.ndarray, kernel: np.ndarray, weight: np.ndarray, rate: float, z_norm: np.ndarray
) -> None:
    """Separable H-biased migration of adults (in place).

    ``weight`` is the per-deme target weight ``w`` (``H`` for the bias mode,
    all-ones for the baseline/capacity mode).  ``z_norm = G (*) weight`` is the
    per-source normaliser; it doubles as boundary compensation.
    """
    planes, shape = _adult_planes(data)
    g = planes / np.maximum(z_norm, EPS)[None, :, :]
    h = fftconvolve(g, kernel[None, :, :], mode="same", axes=(1, 2))
    moved = weight[None, :, :] * h
    data[:, am.NEW_ADULT_AGE:] = (planes * (1.0 - rate) + rate * moved).reshape(shape)


def run(
    mode: str,
    H: np.ndarray,
    *,
    ticks: int,
    rows: int = ROWS,
    cols: int = COLS,
    rate: float = MIGRATION_RATE,
    kernel: np.ndarray | None = None,
    record_every: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run one deterministic replicate; return ``(frames, times)``.

    ``frames`` is ``(n_recorded, rows, cols)`` total population per deme.
    """
    kernel = kernel_off_center(KERNEL_SIZE, MEAN_DISPERSAL) if kernel is None else kernel
    weight = (H if mode == "bias" else np.ones_like(H))
    z_norm = fftconvolve(weight, kernel, mode="same")
    density_ref = am.DENSITY_REF * (H if mode == "capacity" else np.ones_like(H))

    data = initial_state(rows, cols)
    frames = [data.sum(axis=(0, 1)).copy()]
    times = [0]
    for t in range(1, ticks + 1):
        newborns = reproduction(data)
        # S-A-R
        event_S(data, density_ref)
        event_A(data)
        event_R(data, newborns)
        migration(data, kernel, weight, rate, z_norm)
        if t % record_every == 0:
            frames.append(data.sum(axis=(0, 1)).copy())
            times.append(t)
    return np.stack(frames), times


def selftest() -> int:
    """H == 1 must reduce both modes to the plain (unweighted) model."""
    H1 = np.ones((ROWS, COLS))
    kernel = kernel_off_center(KERNEL_SIZE, MEAN_DISPERSAL)
    base, _ = run("baseline", H1, ticks=40, kernel=kernel)
    bias, _ = run("bias", H1, ticks=40, kernel=kernel)
    cap, _ = run("capacity", H1, ticks=40, kernel=kernel)
    assert np.allclose(base, bias, rtol=1e-10, atol=1e-8), "bias != baseline at H=1"
    assert np.allclose(base, cap, rtol=1e-10, atol=1e-8), "capacity != baseline at H=1"
    print("[ok] H==1: bias and capacity both reduce to the plain model")

    H = make_H()
    assert H.min() > 0 and abs(H.max() - 1.0) < 1e-12
    # a local maximum near each requested peak
    for k, (ci, cj) in enumerate(PEAKS):
        nearest = np.unravel_index(np.argmin(np.abs(np.arange(ROWS)[:, None] - ci) + np.abs(np.arange(COLS)[None, :] - cj)), H.shape)
        assert H[nearest] > H.mean() + 0.2, (k, nearest, H[nearest])
    print(f"[ok] synthetic H: 4 peaks, range [{H.min():.3f}, {H.max():.3f}], mean {H.mean():.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    H = make_H()
    frames, times = run("bias", H, ticks=args.ticks)
    print(f"bias run: {frames.shape}, t0 total={frames[0].sum():.1f}, t{times[-1]} total={frames[-1].sum():.1f}")
    frames2, times2 = run("capacity", H, ticks=args.ticks)
    print(f"capacity run: {frames2.shape}, t{times2[-1]} total={frames2[-1].sum():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
