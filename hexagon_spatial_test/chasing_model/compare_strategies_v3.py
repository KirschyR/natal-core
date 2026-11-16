"""Full-model comparison of three stochastic migration strategies on hex grids.

Runs a stochastic RSA age-structured hex population (homing drive, male
homozygous central release) under three migration strategies:

* ``cpu`` -- natal's continuous CSR (Beta outbound + per-source Dirichlet);
* ``m0``  -- source Beta outbound + deterministic kernel spread;
* ``m1``  -- covariance-matching Gaussian field + deterministic route-A mean.

Lifecycle and migration use separate RNG streams, so all strategies share the
same lifecycle randomness (paired comparison).  The script records, per selected
deme over time, the drive fraction, non-egg population and juvenile fraction
(mean +/- 1 SD over replicates), plus aggregate coverage/front, and writes a
self-contained HTML with three synchronized hex heatmaps and a time slider.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PolyCollection
from scipy.optimize import minimize
from scipy.stats import rankdata  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

import natal as nt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "strategies_v3"
HTML_DIR = HERE / "results" / "html"

F = 8
M = 5
Z = 3  # WT|WT, WT|Drive, Drive|Drive
S = 2
A = 10
NEW_ADULT_AGE = 2
EGGS = 40.0
LAMBDA = 5.0
SF = (F + 1) / 2.0
SEX_RATIO = 0.5
DENSITY_REF = EGGS + 5.0 * (4.0 / (F + 1))
PURE_W = 0
DRIVE = 2
CARRIER = (1, 2)
RELEASE_RADIUS = 3.0
RELEASE_MULTIPLE = 0.5
MIGRATION_RATE = 1.0
VIABILITY_DD = 1.0
FECUNDITY_DD = 1.0
EPS = 1e-12
COVERS = (0.1, 0.5, 0.9)

_FEMALE_AGING = [1.0 - 1.0 / (F - i + 1) for i in range(1, F)]
_MALE_AGING = [1.0 - 1.0 / (M - i + 1) for i in range(1, M)]
SURVIVAL = np.array(
    [
        [1.0, 1.0, *_FEMALE_AGING, 0.0],
        [1.0, 1.0, *_MALE_AGING] + [0.0] * (A - 2 - len(_MALE_AGING)),
    ]
)

CONFIGS = [
    {"name": "small", "arena": 15, "kernel_size": 7, "avd": 0.8, "ticks": 100, "reps": 100, "scale": 100.0},
    {"name": "large", "arena": 31, "kernel_size": 11, "avd": 1.5, "ticks": 100, "reps": 100, "scale": 100.0},
]
STRATEGIES = ("cpu", "m0", "m1", "m2")
DISPLAY = ("cpu", "m0", "m2")  # M1 is computed but omitted from the display
COLORS = {"cpu": "tab:blue", "m0": "tab:orange", "m1": "tab:green", "m2": "tab:red"}


# --- genetics / initial state ----------------------------------------------


def homing_tensor(dc: float = 0.8) -> np.ndarray:
    """Return natal's ``(Z, Z, Z)`` homing offspring tensor for [WT, Drive]."""
    species = nt.Species.from_dict(
        "ChasingStrategies", {"chr1": {"loc": ["WT", "Drive"]}}
    )
    drive = nt.HomingDrive(
        name="ChasingHoming",
        drive_allele="Drive",
        target_allele="WT",
        cas9_allele="Drive",
        drive_conversion_rate=dc,
        late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0,
        functional_resistance_ratio=0.0,
        fecundity_scaling=1.0,
        viability_scaling=1.0,
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
        .setup(name="chasing_tensor")
        .initial_state({"female": {"WT|WT": 10.0}, "male": {"WT|WT": 10.0}})
        .presets(drive)
        .build()
    )
    return np.asarray(pop.config.offspring_tensor, dtype=np.float64)


def age_vector() -> np.ndarray:
    ages_m = (1.0 / SF) * np.array([1.0 - i / M for i in range(M)])
    ages_f = (1.0 / SF) * np.array([1.0 - i / F for i in range(F)])
    return np.concatenate([[EGGS, 4.0 / (F + 1)], ages_m, ages_f])


def initial_state(rows: int, cols: int, radius: float, release_multiple: float,
                  scale: float = 1.0) -> np.ndarray:
    """Uniform WT (age0=0) plus male homozygous-drive release in a central disc."""
    ages = age_vector()
    base_f = np.zeros(A)
    base_m = np.zeros(A)
    base_f[1] = ages[1] * 0.5
    base_m[1] = ages[1] * 0.5
    base_f[NEW_ADULT_AGE:] = ages[2 + M :]
    base_m[NEW_ADULT_AGE : NEW_ADULT_AGE + M] = ages[2 : 2 + M]

    data = np.zeros((Z, S, A, rows, cols))
    data[PURE_W, 0] = base_f[:, None, None]
    data[PURE_W, 1] = base_m[:, None, None]
    rr, cc = np.mgrid[0:rows, 0:cols]
    mask = (rr - (rows - 1) / 2.0) ** 2 + (cc - (cols - 1) / 2.0) ** 2 <= radius**2
    data[DRIVE, 1, NEW_ADULT_AGE, mask] += release_multiple * base_m[NEW_ADULT_AGE]
    return data * scale


# --- continuous samplers (natal-style) -------------------------------------


def _v_poisson(rng: np.random.Generator, lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    out = np.zeros_like(lam)
    mask = lam > EPS
    if mask.any():
        out[mask] = rng.gamma(shape=lam[mask], scale=1.0)
    return out


def _v_binomial(rng: np.random.Generator, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    n, p = (np.asarray(a, dtype=float) for a in np.broadcast_arrays(n, p))
    out = np.zeros(n.shape)
    large_p = p >= 1.0 - EPS
    out[large_p] = n[large_p]
    middle = (p > EPS) & ~large_p
    small_n = middle & (n <= 1.0 + EPS)
    out[small_n] = n[small_n] * p[small_n]
    big_n = middle & (n > 1.0 + EPS)
    if big_n.any():
        conc = n[big_n] - 1.0
        alpha = np.maximum(p[big_n] * conc, EPS)
        beta = np.maximum((1.0 - p[big_n]) * conc, EPS)
        num = rng.gamma(shape=alpha, scale=1.0)
        den = rng.gamma(shape=beta, scale=1.0)
        denom = num + den
        prop = np.full_like(num, 0.5)
        np.divide(num, denom, out=prop, where=denom > EPS)
        prop = np.where(denom > EPS, prop, p[big_n])
        out[big_n] = prop * n[big_n]
    return out


def _v_multinomial(rng: np.random.Generator, nvec: np.ndarray, probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    n = np.asarray(nvec, dtype=float)[None, :]
    out = n * probs
    big = (n > 1.0 + EPS).ravel()
    if big.any():
        nb = n[:, big]
        pb = probs[:, big]
        alpha = np.maximum(pb * (nb - 1.0), 0.0)
        draws = np.zeros_like(alpha)
        mask = alpha > EPS
        draws[mask] = rng.gamma(shape=alpha[mask], scale=1.0)
        total = draws.sum(axis=0, keepdims=True)
        good = total > EPS
        out[:, big] = np.where(good, draws * (nb / np.where(good, total, 1.0)), pb * nb)
    return out


# --- lifecycle --------------------------------------------------------------


def reproduction(data: np.ndarray, tensor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fec = np.ones(Z)
    fec[DRIVE] = FECUNDITY_DD
    female_w = data[:, 0, NEW_ADULT_AGE:].sum(axis=1) * fec[:, None, None]
    male_w = data[:, 1, NEW_ADULT_AGE:].sum(axis=1) * fec[:, None, None]
    total_female = female_w.sum(axis=0)
    total_male = male_w.sum(axis=0)
    female_freq = np.divide(female_w, total_female, out=np.zeros_like(female_w), where=total_female > 0)
    male_freq = np.divide(male_w, total_male, out=np.zeros_like(male_w), where=total_male > 0)
    prop = np.einsum("irc,jrc,ijk->krc", female_freq, male_freq, tensor)
    total_eggs = _v_poisson(rng, (EGGS * total_female).ravel()).reshape(total_female.shape)
    rows, cols = total_female.shape
    offspring = _v_multinomial(rng, total_eggs.ravel(), prop.reshape(Z, rows * cols))
    return offspring.reshape(Z, rows, cols)


def event_r(data: np.ndarray, newborns: np.ndarray, rng: np.random.Generator) -> None:
    female = _v_binomial(rng, newborns, SEX_RATIO)
    data[:, 0, 0] = female
    data[:, 1, 0] = newborns - female


def _density_scaling(data: np.ndarray, density_ref: float) -> np.ndarray:
    j0 = data[:, :, 0].sum(axis=(0, 1))
    j1 = data[:, :, 1].sum(axis=(0, 1))
    r = (j0 + 5.0 * j1) / density_ref
    return LAMBDA / ((LAMBDA - 1.0) * r + 1.0) * 2.0 / SF / EGGS


def event_s(data: np.ndarray, density_ref: float, rng: np.random.Generator) -> None:
    s1 = _density_scaling(data, density_ref)
    age0 = data[:, :, 0]
    total0 = age0.sum(axis=(0, 1))
    probs = np.divide(
        age0.reshape(Z * S, -1), total0.reshape(1, -1),
        out=np.zeros((Z * S, total0.size)), where=total0.reshape(1, -1) > 0,
    )
    draws = _v_multinomial(rng, (total0 * s1).ravel(), probs)
    data[:, :, 0] = draws.reshape(Z, S, *total0.shape)
    survival = np.broadcast_to(SURVIVAL[:, :, None, None], data.shape).astype(float).copy()
    survival[DRIVE, :, NEW_ADULT_AGE - 1] *= VIABILITY_DD
    data[:] = _v_binomial(rng, data, survival)


def event_a(data: np.ndarray) -> None:
    data[:, :, 1:] = data[:, :, :-1].copy()
    data[:, :, 0] = 0.0


# --- migration strategies ---------------------------------------------------


def hex_gaussian(size: int, mean_dispersal: float) -> np.ndarray:
    radius = size // 2
    dr, dc = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = mean_dispersal / np.sqrt(np.pi / 2.0)
    kernel = np.exp(-0.5 * (dr**2 + dc**2 + dr * dc) / sigma**2)
    return kernel / kernel.sum()


def build_csr(rows: int, cols: int, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = kernel.shape[0]
    radius = k // 2
    off = kernel.copy()
    off[radius, radius] = 0.0
    indptr = [0]
    dest: list[int] = []
    weights: list[float] = []
    for src in range(rows * cols):
        r, c = divmod(src, cols)
        row_dest: list[int] = []
        row_w: list[float] = []
        for kr in range(k):
            for kc in range(k):
                w = float(off[kr, kc])
                if w <= 0.0:
                    continue
                rr, cc = r + kr - radius, c + kc - radius
                if 0 <= rr < rows and 0 <= cc < cols:
                    row_dest.append(rr * cols + cc)
                    row_w.append(w)
        wsum = sum(row_w)
        weights.extend([w / wsum for w in row_w])
        dest.extend(row_dest)
        indptr.append(len(dest))
    return (np.asarray(indptr, np.int64), np.asarray(dest, np.int64), np.asarray(weights, np.float64))


def _conv(field: np.ndarray, off: np.ndarray) -> np.ndarray:
    rows, cols = field.shape
    k = off.shape[0]
    radius = k // 2
    fr, fc = rows + k - 1, cols + k - 1
    fs = np.zeros((fr, fc))
    fs[:rows, :cols] = field
    ks = np.zeros((fr, fc))
    ks[:k, :k] = off
    full = np.fft.irfft2(np.fft.rfft2(fs) * np.fft.rfft2(ks), s=(fr, fc))
    return full[radius : radius + rows, radius : radius + cols]


def _off_z(rows: int, cols: int, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = kernel.shape[0] // 2
    off = kernel.copy()
    off[radius, radius] = 0.0
    z = _conv(np.ones((rows, cols)), off)
    return off, np.where(z > 0.0, z, 1.0)


def _continuous_binomial_vec(rng: np.random.Generator, n: np.ndarray, p: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    out = n * p
    big = n > 1.0 + EPS
    if big.any():
        conc = n[big] - 1.0
        ga = rng.gamma(shape=np.maximum(p * conc, EPS))
        gb = rng.gamma(shape=np.maximum((1.0 - p) * conc, EPS))
        denom = ga + gb
        prop = np.full_like(ga, p)
        np.divide(ga, denom, out=prop, where=denom > EPS)
        prop = np.where(denom > EPS, prop, p)
        out[big] = n[big] * prop
    return out


def cpu_migrate(data: np.ndarray, csr: tuple[np.ndarray, np.ndarray, np.ndarray],
                rate: float, rng: np.random.Generator) -> None:
    _, _, _, rows, cols = data.shape
    n = rows * cols
    indptr, dest, weights = csr
    adult_ages = A - NEW_ADULT_AGE
    planes = data[:, :, NEW_ADULT_AGE:].reshape(Z * S * adult_ages, n).copy()
    out = planes.copy()
    for src in range(n):
        vals = planes[:, src]
        outbound = _continuous_binomial_vec(rng, vals, rate)
        out[:, src] -= outbound
        lo, hi = indptr[src], indptr[src + 1]
        row_len = hi - lo
        if row_len == 0:
            continue
        probs = weights[lo:hi]
        conc = np.maximum(outbound - 1.0, 0.0)
        alpha = probs[None, :] * conc[:, None]
        draws = np.where(alpha > 0.0, rng.gamma(shape=np.maximum(alpha, EPS)), 0.0)
        total = draws.sum(axis=1, keepdims=True)
        distributed = np.where(
            total > EPS, draws * (outbound[:, None] / np.where(total > EPS, total, 1.0)),
            probs[None, :] * outbound[:, None],
        )
        small = (outbound <= 1.0)[:, None]
        distributed = np.where(small, probs[None, :] * outbound[:, None], distributed)
        out[:, dest[lo:hi]] += distributed
    data[:, :, NEW_ADULT_AGE:] = out.reshape(Z, S, adult_ages, rows, cols)


def m0_migrate(data: np.ndarray, kernel: np.ndarray, rate: float,
               rng: np.random.Generator) -> None:
    rows, cols = data.shape[-2:]
    off, z = _off_z(rows, cols, kernel)
    for zz in range(Z):
        for ss in range(S):
            for aa in range(NEW_ADULT_AGE, A):
                field = data[zz, ss, aa]
                outbound = _continuous_binomial_vec(rng, field.ravel(), rate).reshape(rows, cols)
                data[zz, ss, aa] = field - outbound + _conv(outbound / z, off)


def _spectral_noise(rows: int, cols: int, kernel: np.ndarray, rate: float,
                    field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    radius = kernel.shape[0] // 2
    off = kernel.copy()
    off[radius, radius] = 0.0
    ksum = off.sum()
    w_grid = np.zeros((rows, cols))
    for kr in range(kernel.shape[0]):
        for kc in range(kernel.shape[1]):
            value = off[kr, kc]
            if value > 0.0:
                w_grid[(kr - radius) % rows, (kc - radius) % cols] += value / ksum
    w_hat = np.fft.fft2(w_grid).real
    n_bar = float(field.mean())
    a = n_bar * rate * (1.0 - rate) + rate * n_bar
    b = n_bar * rate * (1.0 - rate)
    c = (rate**2) * n_bar
    s = np.maximum(a - 2.0 * b * w_hat - c * np.abs(w_hat) ** 2, 0.0)
    white = rng.normal(size=(rows, cols))
    eta = np.fft.ifft2(np.sqrt(s) * np.fft.fft2(white)).real
    scale = np.zeros_like(field)
    np.divide(field, max(n_bar, EPS), out=scale, where=field > 0.0)
    eta *= np.sqrt(scale)
    eta -= eta.mean()
    return eta


def m1_migrate(data: np.ndarray, kernel: np.ndarray, rate: float,
               rng: np.random.Generator) -> None:
    rows, cols = data.shape[-2:]
    off, z = _off_z(rows, cols, kernel)
    for zz in range(Z):
        for ss in range(S):
            for aa in range(NEW_ADULT_AGE, A):
                field = data[zz, ss, aa]
                mean = (1.0 - rate) * field + _conv(rate * field / z, off)
                value = np.maximum(mean + _spectral_noise(rows, cols, kernel, rate, field, rng), 0.0)
                target = float(field.sum())
                current = float(value.sum())
                if current > EPS and target > 0.0:
                    value *= target / current
                data[zz, ss, aa] = value


def _blocks(rows: int, cols: int, block: int):
    for r0 in range(0, rows, block):
        r1 = min(r0 + block, rows)
        for c0 in range(0, cols, block):
            c1 = min(c0 + block, cols)
            yield r0, r1, c0, c1


def _continuous_multinomial_block(rng: np.random.Generator, n: float, probs: np.ndarray) -> np.ndarray:
    if n <= 1.0 + EPS:
        return n * probs
    alpha = probs * (n - 1.0)
    draws = np.where(alpha > 0.0, rng.gamma(shape=np.maximum(alpha, EPS)), 0.0)
    total = draws.sum()
    return n * draws / total if total > EPS else n * probs


def m2_migrate(data: np.ndarray, kernel: np.ndarray, rate: float,
               rng: np.random.Generator) -> None:
    """FFT mean + local block multinomial (independent block totals, total conserved).

    Each source samples its outbound (Beta); the FFT gives the expected inbound
    field ``mu`` from those sampled outbounds.  ``mu`` is split into blocks of
    size ~= the kernel support; each block draws an independent continuous-
    Poisson total and then a within-block continuous multinomial.  Block totals
    are scaled to conserve the sampled outbound total.  This keeps the spatial
    wavefront (mu) and local negative correlations (within-block multinomial).
    """
    rows, cols = data.shape[-2:]
    off, z = _off_z(rows, cols, kernel)
    block = kernel.shape[0]
    for zz in range(Z):
        for ss in range(S):
            for aa in range(NEW_ADULT_AGE, A):
                field = data[zz, ss, aa]
                outbound = _continuous_binomial_vec(rng, field.ravel(), rate).reshape(rows, cols)
                keep = field - outbound
                mu = _conv(outbound / z, off)
                total = float(mu.sum())
                if total <= EPS:
                    data[zz, ss, aa] = keep
                    continue
                parts = []
                block_total = 0.0
                for r0, r1, c0, c1 in _blocks(rows, cols, block):
                    blk = mu[r0:r1, c0:c1]
                    mb = float(blk.sum())
                    if mb <= EPS:
                        continue
                    tb = float(rng.gamma(shape=mb))
                    parts.append((r0, r1, c0, c1, blk, mb, tb))
                    block_total += tb
                scale = total / block_total if block_total > EPS else 1.0
                counts = np.zeros_like(mu)
                for r0, r1, c0, c1, blk, mb, tb in parts:
                    probs = (blk / mb).ravel()
                    cnt = _continuous_multinomial_block(rng, tb * scale, probs)
                    counts[r0:r1, c0:c1] = cnt.reshape(r1 - r0, c1 - c0)
                data[zz, ss, aa] = keep + counts


# --- metrics / experiment ---------------------------------------------------


def select_demes(rows: int, cols: int) -> list[tuple[int, int]]:
    """Four demes on the centre-to-boundary line (0, 1/3, 2/3, 1)."""
    cr, cc = (rows - 1) // 2, (cols - 1) // 2
    points: list[tuple[int, int]] = []
    for fraction in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
        r = int(round((1.0 - fraction) * cr))
        points.append((min(max(r, 0), rows - 1), cc))
    unique: list[tuple[int, int]] = []
    for point in points:
        if point not in unique:
            unique.append(point)
    return unique


def record(data: np.ndarray, selected: list[tuple[int, int]]) -> dict:
    non_egg = data[:, :, 1:].sum(axis=(0, 1, 2))
    dd = data[DRIVE, :, 1:].sum(axis=(0, 1))
    juvenile = data[:, :, 1].sum(axis=(0, 1))
    drive_frac = np.divide(dd, non_egg, out=np.zeros_like(non_egg), where=non_egg > 0)
    juv_frac = np.divide(juvenile, non_egg, out=np.zeros_like(non_egg), where=non_egg > 0)
    rr, cc = np.mgrid[0 : data.shape[-2], 0 : data.shape[-1]]
    dist = np.sqrt((rr - (data.shape[-2] - 1) / 2.0) ** 2 + (cc - (data.shape[-1] - 1) / 2.0) ** 2)
    covered = drive_frac > 0.5
    front = float(dist[covered].max()) if covered.any() else 0.0
    sel = [(int(r), int(c)) for r, c in selected]
    return {
        "drive": np.array([drive_frac[r, c] for r, c in sel]),
        "non_egg": np.array([non_egg[r, c] for r, c in sel]),
        "juvenile": np.array([juv_frac[r, c] for r, c in sel]),
        "coverage": np.array([float((drive_frac > thr).sum()) for thr in COVERS]),
        "front": front,
        "total": float(non_egg.sum()),
        "drive_field": drive_frac,
    }


def run_replicate(strategy: str, cfg: dict, seed: int, selected, tensor, csr, kernel) -> dict:
    rows = cols = cfg["arena"]
    rng_life = np.random.default_rng(seed)
    rng_mig = np.random.default_rng(seed + 100000)
    scale = float(cfg.get("scale", 1.0))
    density_ref = DENSITY_REF * scale
    data = initial_state(rows, cols, RELEASE_RADIUS, RELEASE_MULTIPLE, scale)
    ticks = cfg["ticks"]
    drive = np.zeros((ticks + 1, len(selected)))
    non_egg = np.zeros_like(drive)
    juvenile = np.zeros_like(drive)
    coverage = np.zeros((ticks + 1, len(COVERS)))
    front = np.zeros(ticks + 1)
    total = np.zeros(ticks + 1)
    fields = np.zeros((ticks + 1, rows, cols))
    migration_seconds = 0.0

    def capture(t: int) -> None:
        rec = record(data, selected)
        drive[t] = rec["drive"]
        non_egg[t] = rec["non_egg"]
        juvenile[t] = rec["juvenile"]
        coverage[t] = rec["coverage"]
        front[t] = rec["front"]
        total[t] = rec["total"]
        fields[t] = rec["drive_field"]

    capture(0)
    for t in range(1, ticks + 1):
        newborns = reproduction(data, tensor, rng_life)
        event_r(data, newborns, rng_life)
        event_s(data, density_ref, rng_life)
        event_a(data)
        t0 = time.perf_counter()
        if strategy == "cpu":
            cpu_migrate(data, csr, MIGRATION_RATE, rng_mig)
        elif strategy == "m0":
            m0_migrate(data, kernel, MIGRATION_RATE, rng_mig)
        elif strategy == "m1":
            m1_migrate(data, kernel, MIGRATION_RATE, rng_mig)
        else:
            m2_migrate(data, kernel, MIGRATION_RATE, rng_mig)
        migration_seconds += time.perf_counter() - t0
        capture(t)
    return {
        "drive": drive,
        "non_egg": non_egg,
        "juvenile": juvenile,
        "coverage": coverage,
        "front": front,
        "total": total,
        "fields": fields,
        "migration_seconds": migration_seconds,
    }


def _hex_geometry(rows: int, cols: int, size: float = 1.0):
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


def render_frames(mean_fields: dict[str, np.ndarray], rows: int, cols: int, title: str) -> list[str]:
    vertices, bounds = _hex_geometry(rows, cols)
    xmin, xmax, ymin, ymax = bounds
    frames: list[str] = []
    for t in range(next(iter(mean_fields.values())).shape[0]):
        fig, axes = plt.subplots(1, len(DISPLAY), figsize=(5.0 * len(DISPLAY), 4.8), dpi=90)
        for ax, strategy in zip(axes, DISPLAY):
            coll = PolyCollection(
                vertices, array=mean_fields[strategy][t].ravel(), cmap="viridis",
                clim=(0.0, 1.0), rasterized=True,
            )
            ax.add_collection(coll)
            ax.set_xlim(xmin - 0.8, xmax + 0.8)
            ax.set_ylim(ymin - 0.8, ymax + 0.8)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(strategy)
        fig.colorbar(
            ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis"),
            ax=list(axes), fraction=0.03, pad=0.03, label="mean dd fraction",
        )
        fig.suptitle(f"{title}\ntick = {t}", fontsize=9)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return frames


_WORKER: dict = {}


def _worker_init(cfg, selected, tensor, csr, kernel):
    _WORKER.update({"cfg": cfg, "selected": selected, "tensor": tensor, "csr": csr, "kernel": kernel})


def _replicate_task(args):
    strategy, seed = args
    w = _WORKER
    return run_replicate(strategy, w["cfg"], seed, w["selected"], w["tensor"], w["csr"], w["kernel"])


def _feature_matrix(per_strategy, ticks, n_deme):
    feats = {}
    for s in DISPLAY:
        d = per_strategy[s]
        drive, cov = d["drive"], d["coverage"]
        front, total = d["front"], d["total"]
        cols = []
        for t in ticks:
            cols.append(drive[:, t, :])
            cols.append(cov[:, t, :])
            cols.append(front[:, t][:, None])
            cols.append(total[:, t][:, None])
        cols.append(drive.mean(axis=1))
        cols.append(drive.std(axis=1, ddof=1))
        sel = drive.mean(axis=2)
        t50 = []
        for r in range(sel.shape[0]):
            idx = np.where(sel[r] >= 0.5)[0]
            t50.append(idx[0] if len(idx) else sel.shape[1])
        cols.append(np.asarray(t50, dtype=float)[:, None])
        feats[s] = np.concatenate(cols, axis=1)
    return feats


def _fit_logreg(X, y, l2=1e-2):
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])

    def obj(w):
        z = np.clip(Xb @ w, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        ll = np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        f = -ll / n + 0.5 * l2 * np.sum(w[1:] ** 2)
        g = Xb.T @ (p - y) / n
        g[1:] += l2 * w[1:]
        return f, g

    return minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B").x


def _predict_logreg(w, X):
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    z = np.clip(Xb @ w, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _auc_score(y, p):
    r = rankdata(p)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 <= 0 or n0 <= 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _cv_scores(X, y, folds=5, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    accs, aucs = [], []
    for _ in range(repeats):
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        groups = [[] for _ in range(folds)]
        for arr in (idx0, idx1):
            for i, v in enumerate(arr):
                groups[i % folds].append(int(v))
        for k in range(folds):
            va = np.asarray(groups[k], dtype=int)
            tr = np.setdiff1d(np.arange(len(y)), va)
            mu = X[tr].mean(axis=0)
            sd = X[tr].std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            w = _fit_logreg((X[tr] - mu) / sd, y[tr])
            p = _predict_logreg(w, (X[va] - mu) / sd)
            accs.append(float(((p >= 0.5) == y[va]).mean()))
            aucs.append(_auc_score(y[va], p))
    return float(np.mean(accs)), float(np.mean(aucs))


def _c2st_pair(Xa, Xb, seed=0, perms=20):
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    acc, auc = _cv_scores(X, y, folds=5, repeats=5, seed=seed)
    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(perms):
        yp = rng.permutation(y)
        accp, _ = _cv_scores(X, yp, folds=5, repeats=2, seed=int(rng.integers(1_000_000_000)))
        null.append(accp)
    p = (1.0 + float(np.sum(np.asarray(null) >= acc))) / (1.0 + perms)
    return acc, auc, p


def _c2st_matrices(feats, methods, seed=0):
    n = len(methods)
    acc = np.full((n, n), np.nan)
    auc = np.full((n, n), np.nan)
    pval = np.ones((n, n))
    for i in range(n):
        # self-split: split the same method's replicates into two halves; C2ST ~0.5 under the null.
        Xm = feats[methods[i]]
        rng = np.random.default_rng(seed + 101 * (i + 1))
        order = rng.permutation(len(Xm))
        half = len(Xm) // 2
        a, b, p = _c2st_pair(Xm[order[:half]], Xm[order[half : 2 * half]], seed=seed + 7 * i + 3)
        acc[i, i], auc[i, i], pval[i, i] = a, b, p
        for j in range(i):
            a, b, p = _c2st_pair(feats[methods[i]], feats[methods[j]], seed=seed + 17 * (i + 1) + j)
            acc[i, j] = acc[j, i] = a
            auc[i, j] = auc[j, i] = b
            pval[i, j] = pval[j, i] = p
    return acc, auc, pval


def _plot_c2st(acc, auc, methods, out):
    n = len(methods)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    data = np.ma.masked_where(mask, acc)
    norm = TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=1.0)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(data, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n)); ax.set_xticklabels(methods)
    ax.set_yticks(range(n)); ax.set_yticklabels(methods)
    for i in range(n):
        for j in range(i + 1):
            rgba = im.cmap(norm(acc[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{acc[i, j]:.2f}\n({auc[i, j]:.2f})", ha="center", va="center",
                    fontsize=8.5, color=("white" if lum < 0.5 else "black"))
    fig.colorbar(im, ax=ax, label="accuracy (0.5 = indistinguishable); cell = acc (AUC)")
    ax.set_title("C2ST: accuracy (AUC), lower triangle")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _html(frames: list[str], line_png: Path, title: str, timing: dict,
          c2st_png: Path) -> str:
    line_uri = base64.b64encode(line_png.read_bytes()).decode("ascii")
    c2st_uri = base64.b64encode(c2st_png.read_bytes()).decode("ascii")
    timing_rows = "".join(
        f"<tr><td>{s}</td>"
        f"<td>{timing[s]['wall_seconds'] / max(timing[s].get('reps', 1), 1):.3f}</td>"
        f"<td>{timing[s]['migration_seconds'] / (max(timing[s].get('reps', 1), 1) * max(timing[s].get('jobs', 1), 1)):.3f}</td></tr>"
        for s in DISPLAY if s in timing
    )
    meta = next(iter(timing.values()))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;text-align:center;margin:8px}}img{{max-width:96vw}}#bar{{margin:6px}}input[type=range]{{width:60%}}</style>
</head><body>
<h3>{title}</h3>
<h4>Timing per replicate (actual, with {meta.get('jobs', 1)} workers; NumPy/CPU): wall = parallel wall / R; migration = migration CPU sum / (R x workers). Sums and the sequential probe are in summary.json. M1 is computed but omitted from this display.</h4>
<table border="1" cellpadding="4" style="margin:auto;border-collapse:collapse">
<tr><th>strategy</th><th>wall s / replicate (actual)</th><th>migration s / replicate (actual)</th></tr>
{timing_rows}
</table>
<img id="frame" src="data:image/png;base64,{frames[0]}">
<div id="bar"><button id="play">Play</button>
<input id="slider" type="range" min="0" max="{len(frames)-1}" value="0">
<span id="lbl">tick 0</span></div>
<h3>Selected demes: drive fraction / non-egg population / juvenile fraction (mean +/- 1 SD)</h3>
<img src="data:image/png;base64,{line_uri}">
<h3>C2ST: are the two methods' replicate distributions separable? (lower triangle; 0.5 = indistinguishable)</h3>
<p>Each cell = accuracy (AUC); color = accuracy; diagonal = self-split. Permutation p is in <code>c2st.json</code>/<code>summary.json</code>.</p>
<img src="data:image/png;base64,{c2st_uri}">
<script>
const frames=[{','.join(f'"{f}"' for f in frames)}];let i=0,timer=null;
const img=document.getElementById('frame'),sl=document.getElementById('slider'),lbl=document.getElementById('lbl');
function show(k){{i=k;img.src='data:image/png;base64,'+frames[i];sl.value=i;lbl.textContent='tick '+i;}}
sl.oninput=()=>show(+sl.value);
document.getElementById('play').onclick=function(){{if(timer){{clearInterval(timer);timer=null;this.textContent='Play';}}else{{this.textContent='Pause';timer=setInterval(()=>show((i+1)%frames.length),300);}}}};
</script></body></html>"""


def line_plot(mean: dict, std: dict, ticks: int, selected, out: Path) -> None:
    n_deme = len(selected)
    fig, axes = plt.subplots(n_deme, 3, figsize=(15, 3.2 * n_deme), sharex=True)
    t = np.arange(ticks + 1)
    names = ("drive fraction", "non-egg population", "juvenile fraction")
    keys = ("drive", "non_egg", "juvenile")
    for d in range(n_deme):
        for c, (key, name) in enumerate(zip(keys, names)):
            ax = axes[d, c]
            for strategy in DISPLAY:
                m = mean[strategy][key][:, d]
                s = std[strategy][key][:, d]
                ax.plot(t, m, color=COLORS[strategy], label=strategy, lw=1.0)
                ax.fill_between(t, m - s, m + s, color=COLORS[strategy], alpha=0.22)
            ax.grid(alpha=0.3)
            if d == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(f"deme {selected[d]}")
            if d == n_deme - 1:
                ax.set_xlabel("tick")
            if d == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_config(cfg: dict, jobs: int = 1) -> dict:
    rows = cols = cfg["arena"]
    kernel = hex_gaussian(cfg["kernel_size"], cfg["avd"])
    csr = build_csr(rows, cols, kernel)
    tensor = homing_tensor()
    selected = select_demes(rows, cols)
    per_strategy = {}
    timing = {}
    for strategy in STRATEGIES:
        t0 = time.perf_counter()
        if jobs > 1:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_worker_init,
                initargs=(cfg, selected, tensor, csr, kernel),
            ) as pool:
                reps = list(pool.map(_replicate_task, [(strategy, seed) for seed in range(cfg["reps"])]))
        else:
            reps = [run_replicate(strategy, cfg, seed, selected, tensor, csr, kernel)
                    for seed in range(cfg["reps"])]
        wall = time.perf_counter() - t0
        migration = sum(r["migration_seconds"] for r in reps)
        timing[strategy] = {
            "wall_seconds": wall,
            "per_replicate_seconds": wall / cfg["reps"],
            "migration_seconds": migration,
            "reps": cfg["reps"],
            "ticks": cfg["ticks"],
            "jobs": jobs,
        }
        probe_wall, probe_mig = [], []
        for seed in (0, 1, 2):
            t0 = time.perf_counter()
            r = run_replicate(strategy, cfg, seed, selected, tensor, csr, kernel)
            probe_wall.append(time.perf_counter() - t0)
            probe_mig.append(r["migration_seconds"])
        timing[strategy]["sequential"] = {
            "wall_seconds": float(np.mean(probe_wall)),
            "migration_seconds": float(np.mean(probe_mig)),
            "seeds": [0, 1, 2],
        }
        per_strategy[strategy] = {
            "drive": np.stack([r["drive"] for r in reps]),
            "non_egg": np.stack([r["non_egg"] for r in reps]),
            "juvenile": np.stack([r["juvenile"] for r in reps]),
            "coverage": np.stack([r["coverage"] for r in reps]),
            "front": np.stack([r["front"] for r in reps]),
            "total": np.stack([r["total"] for r in reps]),
            "fields": np.stack([r["fields"] for r in reps]),
        }
    mean = {s: {k: v.mean(axis=0) for k, v in per_strategy[s].items()} for s in STRATEGIES}
    std = {s: {k: v.std(axis=0, ddof=1) for k, v in per_strategy[s].items()} for s in STRATEGIES}

    out_dir = RESULTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    line_png = out_dir / "selected_demes.png"
    line_plot(mean, std, cfg["ticks"], selected, line_png)

    mean_fields = {
        s: per_strategy[s]["fields"][:5].mean(axis=0) for s in STRATEGIES
    }
    frames = render_frames(mean_fields, rows, cols, cfg["name"])

    ticks = [max(1, cfg["ticks"] // 4), cfg["ticks"] // 2, 3 * cfg["ticks"] // 4, cfg["ticks"]]
    feats = _feature_matrix(per_strategy, ticks, len(selected))
    methods = list(DISPLAY)
    c2st_acc, c2st_auc, c2st_p = _c2st_matrices(feats, methods)
    c2st_png = out_dir / "c2st.png"
    _plot_c2st(c2st_acc, c2st_auc, methods, c2st_png)
    (out_dir / "c2st.json").write_text(
        json.dumps({"methods": methods, "accuracy": c2st_acc.tolist(),
                    "auc": c2st_auc.tolist(), "p": c2st_p.tolist()}, indent=2),
        encoding="utf-8",
    )

    html = HTML_DIR / f"compare_strategies_v3_{cfg['name']}.html"
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    html.write_text(_html(frames, line_png, f"strategies v3: {cfg['name']} ({rows}x{cols}, k{cfg['kernel_size']}, avd={cfg['avd']:g})", timing, c2st_png), encoding="utf-8")

    def summarize(arr: np.ndarray) -> dict:
        return {
            "final_mean": float(arr[..., -1].mean()),
            "final_std": float(arr[..., -1].std(ddof=1)),
            "mean_over_time": float(arr.mean()),
        }

    summary = {
        "config": cfg,
        "selected_demes": selected,
        "coverage": {
            s: {
                f"thr_{thr:g}": {
                    "final_count_mean": float(per_strategy[s]["coverage"][:, -1, i].mean()),
                    "final_count_std": float(per_strategy[s]["coverage"][:, -1, i].std(ddof=1)),
                }
                for i, thr in enumerate(COVERS)
            }
            for s in STRATEGIES
        },
        "front": {s: summarize(per_strategy[s]["front"]) for s in STRATEGIES},
        "total": {s: summarize(per_strategy[s]["total"]) for s in STRATEGIES},
        "drive_final_selected": {
            s: [float(mean[s]["drive"][-1, d]) for d in range(len(selected))] for s in STRATEGIES
        },
        "timing": timing,
        "c2st": {"methods": methods, "accuracy": c2st_acc.tolist(),
                 "auc": c2st_auc.tolist(), "p": c2st_p.tolist()},
        "html": html.name,
        "line_plot": line_png.name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "series.npz",
        **{f"{s}_{k}": per_strategy[s][k] for s in STRATEGIES
           for k in ("drive", "non_egg", "juvenile", "coverage", "front", "total")},
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="small,large")
    ap.add_argument("--reps", type=int, default=0, help="override replicate count")
    ap.add_argument("--jobs", type=int, default=8, help="parallel replicate workers")
    args = ap.parse_args()
    wanted = set(args.configs.split(","))
    RESULTS.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cfg in CONFIGS:
        if cfg["name"] not in wanted:
            continue
        if args.reps > 0:
            cfg = {**cfg, "reps": args.reps}
        print(f"[run] {cfg['name']} ...")
        summaries.append(run_config(cfg, args.jobs))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
