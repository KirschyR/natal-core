"""Faithful NumPy reproduction of the age-structured mosquito hex radial model.

This translates ``Hex-model-main/radial_release_optimizer/main.m`` (the
reference optimizer that underlies the paper's Table 1 circular-release
analysis) into a deterministic NumPy/natal-primitive implementation:

* the hexagonal Gaussian migration kernel is built with natal's
  ``build_gaussian_kernel`` and then truncated to the reference ``d <= 25``
  hex disk and renormalized (matching ``get_mig_matrix.m`` bit-for-bit);
* migration uses ``scipy.ndimage.convolve(..., mode="nearest")``, equivalent
  to MATLAB ``imfilter(..., 'replicate', 'same')``;
* the lifecycle (density-dependent juvenile survival, reproduction via the
  ``renew_function`` offspring operator, age-based survival/aging, adult-only
  migration) follows the reference loop operation-for-operation.

The state is ``(genotype, age, row, col)`` with 15 age classes:
``[juvenile0, juvenile1, male0..male4, female0..female7]``.

Outputs are coverage area and coverage diameter at the reference collect
weeks, at both the 0.9 and 0.5 carrier-frequency thresholds, plus the
cumulative released count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from natal.frontend.spatial.topology import build_gaussian_kernel

F = 8
M = 5
AGES = 2 + M + F
N_Z = 5
CORD_PURE_W = 2  # 0-based genotype index of WT|WT (drive_params cord_pure_w = 3)
CORD_RELEASE_GENOTYPE = 0
CORD_M1 = 2  # 0-based male0
CORD_F1 = 7  # 0-based female0
ADULT_AGES = slice(2, AGES)
CARRIER_INDEX = [0, 1, 4]
EGGS = 40.0
LAMBDA = 5.0
SF = (F + 1) / 2.0
MIG_MAX_DIST = 25.0
MIG_SIGMA = 10.437


def model_ages() -> np.ndarray:
    """Return the reference ``age`` density vector (15 entries)."""
    ages_m = (1.0 / SF) * np.arange(1, 0, -1.0 / M)  # 1, 1-1/M, ..., 1/M
    ages_f = (1.0 / SF) * np.arange(1, 0, -1.0 / F)
    return np.concatenate([[EGGS, 4.0 / (F + 1)], ages_m, ages_f])


def natal_hex_kernel(size: int = 59, sigma: float = MIG_SIGMA) -> np.ndarray:
    """Build the reference migration kernel via natal + the ``d <= 25`` disk.

    ``build_gaussian_kernel("hex", ...)`` uses the hexagonal metric
    ``dr^2 + dc^2 + dr*dc`` (``COS_OPPOSITE_ANGLE = -0.5``), whereas the
    reference ``get_mig_matrix.m`` uses ``dx^2 + dy^2 - dx*dy``; flipping one
    axis maps between them. The reference additionally truncates to the
    radius-25 hex disk and renormalizes.
    """
    avd = sigma * np.sqrt(np.pi / 2.0)
    kernel = build_gaussian_kernel("hex", size=size, mean_dispersal=avd)
    kernel = kernel[::-1, :]  # natal metric (+dr*dc) -> reference metric (-dx*dy)
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dx = xx - centre
    dy = yy - centre
    dist = np.sqrt(dx**2 + dy**2 - dx * dy)
    kernel = np.where(dist <= MIG_MAX_DIST, kernel, 0.0)
    return kernel / kernel.sum()


def transfer_matrix() -> np.ndarray:
    """Return the static 15x15 age transfer matrix (density term left at 0).

    ``mat[a, b]`` is the transfer from age ``b`` to age ``a`` (matching
    MATLAB ``pagemtimes(data, 'none', largemat, 'transpose')``).
    """
    mat = np.zeros((AGES, AGES))
    mat[CORD_M1, 1] = 0.5  # juvenile1 -> male0
    mat[CORD_F1, 1] = 0.5  # juvenile1 -> female0
    for i in range(1, F):
        mat[CORD_F1 + i, CORD_F1 + i - 1] = 1.0 - 1.0 / (F - i + 1)
    for i in range(1, M):
        mat[CORD_M1 + i, CORD_M1 + i - 1] = 1.0 - 1.0 / (M - i + 1)
    return mat


def release_geometry(m: int, n: int, radius: float) -> np.ndarray:
    """Return a boolean mask of the central release disk in hex coordinates.

    Mirrors the reference loop ``x = sqrt(3)/2*i``, ``y = j - i/2`` around
    ``(midx, midy)``; ``i`` runs over rows, ``j`` over columns, 1-based.
    """
    midi = _mround((n + 1) / 2.0)
    midj = _mround((m + 1) / 2.0)
    midy = midj - 0.5 * midi
    midx = (np.sqrt(3) / 2.0) * midi
    i = np.arange(1, m + 1)[:, None]
    j = np.arange(1, n + 1)[None, :]
    x = (np.sqrt(3) / 2.0) * i
    y = j - 0.5 * i
    return (midx - x) ** 2 + (midy - y) ** 2 <= radius**2


def renew(data: np.ndarray, mats: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """Reference ``renew_function``: per-deme offspring proportions (N_Z, m, n)."""
    female = data[:, CORD_F1:AGES]  # (N_Z, F, m, n)
    sum_female = female.sum(axis=(0, 1))  # (m, n)
    male = data * fitness[:, None, None, None]
    sum_male = male[:, CORD_M1:CORD_F1].sum(axis=(0, 1))
    female_freq = _safe_ratio(female.sum(axis=1), sum_female)
    male_freq = _safe_ratio(male[:, CORD_M1:CORD_F1].sum(axis=1), sum_male)
    out = np.empty((N_Z, data.shape[2], data.shape[3]))
    for k in range(N_Z):
        mat_k = mats[k * N_Z : (k + 1) * N_Z, :]  # (N_Z, N_Z) female x male
        out[k] = np.einsum("imn,jmn,ij->mn", female_freq, male_freq, mat_k)
    return out


@dataclass
class RadialResult:
    """Summary of one radial-release simulation."""

    t: int
    h: int
    r: int
    arena: int
    release_sum: float
    rows: list[list[float]]
    s_maps: dict[int, np.ndarray]


def run(
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
) -> RadialResult:
    """Run one radial-release configuration (reference semantics)."""
    age = model_ages()
    data = np.zeros((N_Z, AGES, m, n))
    data[CORD_PURE_W, :, :, :] = age[:, None, None]

    release_mat = np.zeros_like(data)
    mask = release_geometry(m, n, r)
    release_mat[CORD_RELEASE_GENOTYPE, CORD_M1, :, :][mask] = h / 2.0
    release_mat[CORD_RELEASE_GENOTYPE, CORD_F1, :, :][mask] = h / 2.0
    release_num = release_mat.sum()
    release_sum = 0.0
    drop_time = set(range(1, t + 1))
    mid_row = _mround(m / 2.0) - 1  # MATLAB round(m/2), 1-based -> 0-based

    rows: list[list[float]] = []
    s_maps: dict[int, np.ndarray] = {}

    for tick in range(1, max_ticks + 1):
        if tick in drop_time:
            data = data + release_mat
            release_sum += release_num
        if data[CARRIER_INDEX][:, ADULT_AGES].sum() < 1e-5:
            break

        j1 = data[:, 0].sum(axis=0)
        j2 = data[:, 1].sum(axis=0)
        rr = (j1 + 5.0 * j2) / (age[0] + 5.0 * age[1])
        s1 = LAMBDA / ((LAMBDA - 1.0) * rr + 1.0) * 2.0 / SF / EGGS

        total_female = data[:, CORD_F1:AGES].sum(axis=(0, 1))
        newborns = EGGS * total_female[None, :, :] * renew(data, mats, fitness)

        data = _advance(data, s1)
        data[:, 0, :, :] = newborns
        _migrate(data, kernel)
        data[np.isnan(data)] = 0.0

        if tick in collect_weeks:
            s = _carrier_frequency(data)
            rows.append(_coverage_row(tick, h, r, s, mid_row, release_sum))
            if capture_maps:
                s_maps[tick] = s.copy()

    return RadialResult(t=t, h=h, r=r, arena=m, release_sum=release_sum, rows=rows, s_maps=s_maps)


def _advance(data: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """Apply the age transfer matrix (with per-deme density term) in place-ish."""
    out = np.zeros_like(data)
    out[:, 1] = data[:, 0] * s1  # juvenile0 -> juvenile1 (density dependent)
    out[:, CORD_M1] = 0.5 * data[:, 1]
    out[:, CORD_F1] = 0.5 * data[:, 1]
    for i in range(1, F):
        out[:, CORD_F1 + i] = (1.0 - 1.0 / (F - i + 1)) * data[:, CORD_F1 + i - 1]
    for i in range(1, M):
        out[:, CORD_M1 + i] = (1.0 - 1.0 / (M - i + 1)) * data[:, CORD_M1 + i - 1]
    return out


def _migrate(data: np.ndarray, kernel: np.ndarray) -> None:
    """Apply replicate-boundary migration to every adult (genotype, age) plane.

    Uses batched FFT linear convolution with edge padding, which reproduces
    ``scipy.ndimage.convolve(..., mode="nearest")`` (MATLAB
    ``imfilter(..., 'replicate', 'same')``) to floating-point round-off without
    the prohibitive direct-convolution cost at the 500x500 paper arena.
    """
    radius = kernel.shape[0] // 2
    n_z, n_ages, m, n = data.shape
    planes = data[:, 2:n_ages, :, :].reshape(-1, m, n)
    padded = np.pad(planes, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    out = fftconvolve(padded, kernel[None, :, :], mode="valid", axes=(1, 2))
    data[:, 2:n_ages, :, :] = out.reshape(n_z, n_ages - 2, m, n)


def _carrier_frequency(data: np.ndarray) -> np.ndarray:
    carrier = data[CARRIER_INDEX][:, ADULT_AGES].sum(axis=(0, 1))
    total = data[:, ADULT_AGES].sum(axis=(0, 1))
    return _safe_ratio(carrier, total)


def _coverage_row(
    tick: int,
    h: int,
    r: int,
    s: np.ndarray,
    mid_row: int,
    release_sum: float,
) -> list[float]:
    """Match the reference CSV row for a modification (non-suppression) drive."""
    cover_diameter = float((s[mid_row, :] >= 0.9).sum())
    cover_area = float((s >= 0.9).sum())
    return [float(tick), float(h), float(r), cover_diameter, cover_area, float(release_sum)]


def coverage_counts(s: np.ndarray, mid_row: int, threshold: float) -> tuple[float, float]:
    """Return ``(diameter, area)`` for an arbitrary carrier-frequency threshold."""
    return float((s[mid_row, :] >= threshold).sum()), float((s >= threshold).sum())


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros(np.broadcast(numerator, denominator).shape, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _mround(value: float) -> int:
    """MATLAB ``round``: half away from zero (avoids Python banker's rounding)."""
    return int(np.floor(value + 0.5)) if value >= 0 else -int(np.floor(-value + 0.5))
