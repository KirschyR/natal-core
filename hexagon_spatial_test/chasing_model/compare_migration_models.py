"""Compare stochastic migration models against the CPU CSR reference (operator level).

One scalar field, one migration tick, many replicates.  Models:

* ``cpu``  -- natal's continuous CSR algorithm: per source, Beta outbound
  (continuous binomial) + per-source Dirichlet (continuous multinomial).  Exact
  per-source and total conservation, and the reference covariance.
* ``m0``   -- source Beta outbound, then deterministic spread by the normalized
  kernel (route-A-like).  Conserves total, keeps source variance, drops the
  within-row Dirichlet competition.
* ``m1``   -- covariance-matching Gaussian field (spectral synthesis) added to
  the deterministic route-A mean.  Targets the full CPU covariance (including
  the negative within-row terms); ``S(0) = 0`` keeps the total conserved.

Metrics: per-deme mean error, variance ratio, full spatial-covariance Frobenius
error, total population distribution, and the negative-value fraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def hex_gaussian(size: int, mean_dispersal: float) -> np.ndarray:
    """Return a normalized hex-metric Gaussian kernel (centre kept)."""
    radius = size // 2
    dr, dc = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = mean_dispersal / np.sqrt(np.pi / 2.0)
    kernel = np.exp(-0.5 * (dr**2 + dc**2 + dr * dc) / sigma**2)
    return kernel / kernel.sum()


def build_csr(rows: int, cols: int, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold the kernel into drop-and-renormalize CSR (row sums equal 1)."""
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
    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(dest, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def continuous_binomial(rng: np.random.Generator, n: float, p: float) -> float:
    """natal ``continuous_binomial``: Beta proportion times ``n``."""
    if p <= 0.0 or n <= 0.0:
        return 0.0
    if p >= 1.0:
        return n
    if n <= 1.0:
        return n * p
    conc = n - 1.0
    ga = rng.gamma(shape=max(p * conc, 1e-12))
    gb = rng.gamma(shape=max((1.0 - p) * conc, 1e-12))
    return n * ga / (ga + gb)


def continuous_multinomial(rng: np.random.Generator, n: float, probs: np.ndarray) -> np.ndarray:
    """natal ``continuous_multinomial``: Gamma draws normalized to sum ``n``."""
    if n <= 0.0:
        return np.zeros_like(probs)
    if n <= 1.0:
        return n * probs
    alpha = probs * (n - 1.0)
    draws = np.where(alpha > 0.0, rng.gamma(shape=np.maximum(alpha, 1e-12)), 0.0)
    total = draws.sum()
    if total <= 0.0:
        return n * probs
    return n * draws / total


def cpu_stochastic(field: np.ndarray, csr: tuple[np.ndarray, np.ndarray, np.ndarray],
                   rate: float, rng: np.random.Generator) -> np.ndarray:
    """Continuous CSR stochastic migration (binomial outbound + Dirichlet row)."""
    indptr, dest, weights = csr
    out = field.astype(np.float64).copy()
    for src in range(field.shape[0]):
        n = field[src]
        if n <= 0.0:
            continue
        outbound = continuous_binomial(rng, n, rate)
        out[src] -= outbound
        if outbound <= 0.0:
            continue
        lo, hi = indptr[src], indptr[src + 1]
        probs = weights[lo:hi]
        probs = probs / probs.sum()
        out[dest[lo:hi]] += continuous_multinomial(rng, outbound, probs)
    return out


def _conv(field: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Linear convolution ``off (*) field`` cropped to the field size."""
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


def _off_and_z(field: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = kernel.shape[0] // 2
    off = kernel.copy()
    off[radius, radius] = 0.0
    z = _conv(np.ones_like(field), off)
    return off, np.where(z > 0.0, z, 1.0)


def m0_stochastic(field2d: np.ndarray, kernel: np.ndarray, rate: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Source Beta outbound + deterministic kernel spread (route-A-like)."""
    off, z = _off_and_z(field2d, kernel)
    outbound = np.array(
        [continuous_binomial(rng, float(v), rate) for v in field2d.ravel()]
    ).reshape(field2d.shape)
    keep = field2d - outbound
    inbound = _conv(outbound / z, off)
    return keep + inbound


def m1_stochastic(field2d: np.ndarray, kernel: np.ndarray, rate: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Covariance-matching Gaussian field added to the deterministic mean."""
    rows, cols = field2d.shape
    off, z = _off_and_z(field2d, kernel)
    mean = (1.0 - rate) * field2d + _conv(rate * field2d / z, off)

    radius = kernel.shape[0] // 2
    ksum = off.sum()
    w_grid = np.zeros((rows, cols))
    for kr in range(kernel.shape[0]):
        for kc in range(kernel.shape[1]):
            value = off[kr, kc]
            if value > 0.0:
                dr, dc = kr - radius, kc - radius
                w_grid[dr % rows, dc % cols] += value / ksum
    w_hat = np.fft.fft2(w_grid).real

    n_bar = float(field2d.mean())
    a = n_bar * rate * (1.0 - rate) + rate * n_bar
    b = n_bar * rate * (1.0 - rate)
    c = (rate**2) * n_bar
    s = a - 2.0 * b * w_hat - c * np.abs(w_hat) ** 2
    s = np.maximum(s, 0.0)

    white = rng.normal(size=(rows, cols))
    eta = np.fft.ifft2(np.sqrt(s) * np.fft.fft2(white)).real
    scale = np.zeros_like(field2d)
    np.divide(field2d, max(n_bar, 1e-12), out=scale, where=field2d > 0.0)
    eta *= np.sqrt(scale)
    eta -= eta.mean()  # zero total so the field stays conserved
    return mean + eta


def lag_covariance(values: np.ndarray, rows: int, cols: int,
                   lags: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Lag-averaged spatial covariance (averages over replicate and grid pairs).

    Averaging over translation-equivalent pairs reduces the Monte Carlo noise
    that dominates a raw full-covariance comparison at a few hundred replicates.
    """
    field = values.reshape(values.shape[0], rows, cols)
    mean = field.mean(axis=0)
    out: dict[tuple[int, int], float] = {}
    for dr, dc in lags:
        r0, r1 = max(0, -dr), min(rows, rows - dr)
        c0, c1 = max(0, -dc), min(cols, cols - dc)
        a = field[:, r0:r1, c0:c1]
        b = field[:, r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        ma = mean[r0:r1, c0:c1]
        mb = mean[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        out[(dr, dc)] = float(((a - ma) * (b - mb)).mean())
    return out


def sample_summary(name: str, samples: np.ndarray, cpu_samples: np.ndarray) -> dict:
    """Summarize one model's replicate fields against the CPU reference."""
    cpu_mean = cpu_samples.mean(axis=0)
    mean = samples.mean(axis=0)
    cpu_var = cpu_samples.var(axis=0, ddof=1)
    var = samples.var(axis=0, ddof=1)
    rel = np.abs(mean - cpu_mean) / np.maximum(np.abs(cpu_mean), 1e-9)
    ratio = np.divide(var, cpu_var, out=np.ones_like(var), where=cpu_var > 0)
    cpu_cov = np.cov(cpu_samples, rowvar=False)
    cov = np.cov(samples, rowvar=False)
    cov_fro = float(np.linalg.norm(cov - cpu_cov) / max(np.linalg.norm(cpu_cov), 1e-12))
    return {
        "model": name,
        "mean_rel_median": float(np.median(rel)),
        "mean_rel_max": float(rel.max()),
        "variance_ratio_median": float(np.median(ratio)),
        "variance_ratio_p05": float(np.percentile(ratio, 5)),
        "variance_ratio_p95": float(np.percentile(ratio, 95)),
        "cov_fro_rel_error": cov_fro,
        "total_mean": float(samples.sum(axis=1).mean()),
        "total_std": float(samples.sum(axis=1).std(ddof=1)),
        "negative_fraction": float((samples < 0.0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=int, default=21)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--avd", type=float, default=1.0)
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--low", type=int, default=5)
    ap.add_argument("--high", type=int, default=26)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = cols = args.arena
    n = rows * cols
    kernel = hex_gaussian(args.kernel_size, args.avd)
    csr = build_csr(rows, cols, kernel)
    field = np.random.default_rng(args.seed).integers(args.low, args.high, size=(rows, cols)).astype(float)

    models = {
        "cpu": None,
        "m0": m0_stochastic,
        "m1": m1_stochastic,
    }
    samples: dict[str, np.ndarray] = {}
    for name in models:
        arr = np.empty((args.reps, n))
        for rep in range(args.reps):
            rng = np.random.default_rng(args.seed + 1 + rep)
            if name == "cpu":
                out = cpu_stochastic(field.ravel(), csr, args.rate, rng)
            else:
                out = models[name](field, kernel, args.rate, rng).ravel()
            arr[rep] = out
        samples[name] = arr

    summaries = [sample_summary(name, samples[name], samples["cpu"]) for name in models]
    lags = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
    lagcov = {
        name: lag_covariance(samples[name], rows, cols, lags) for name in models
    }
    payload = {
        "arena": args.arena,
        "reps": args.reps,
        "rate": args.rate,
        "avd": args.avd,
        "kernel_size": args.kernel_size,
        "initial_total": float(field.sum()),
        "summaries": summaries,
        "lag_covariance": {name: {f"{dr},{dc}": v for (dr, dc), v in cov.items()}
                            for name, cov in lagcov.items()},
    }
    (RESULTS / "compare_migration_models.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Plot variance-ratio distributions and total-population histograms.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for name in ("m0", "m1"):
        cpu_var = samples["cpu"].var(axis=0, ddof=1)
        ratio = np.divide(
            samples[name].var(axis=0, ddof=1), cpu_var,
            out=np.ones_like(cpu_var), where=cpu_var > 0,
        )
        axes[0].hist(ratio, bins=40, alpha=0.5, label=name)
    axes[0].axvline(1.0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("variance ratio vs CPU")
    axes[0].set_ylabel("deme count")
    axes[0].set_title("Per-deme variance ratio")
    axes[0].legend()
    for name in ("cpu", "m0", "m1"):
        totals = samples[name].sum(axis=1)
        if float(totals.std()) > 1e-9:
            axes[1].hist(totals, bins=30, alpha=0.5, label=name)
        else:
            axes[1].axvline(float(totals[0]), lw=2, label=name)
    axes[1].set_xlabel("total population")
    axes[1].set_ylabel("replicates")
    axes[1].set_title("Total distribution")
    axes[1].legend()
    lag_names = [f"{dr},{dc}" for (dr, dc) in lags]
    width = 0.25
    for idx, name in enumerate(("cpu", "m0", "m1")):
        axes[2].bar(
            np.arange(len(lags)) + (idx - 1) * width,
            [lagcov[name][lag] for lag in lags],
            width=width, label=name,
        )
    axes[2].set_xticks(np.arange(len(lags)))
    axes[2].set_xticklabels(lag_names)
    axes[2].set_xlabel("lag (dr,dc)")
    axes[2].set_ylabel("lag-averaged covariance")
    axes[2].set_title("Spatial covariance vs lag")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].legend()
    fig.suptitle(f"Stochastic migration models: {args.arena}x{args.arena}, reps={args.reps}")
    fig.tight_layout()
    fig.savefig(RESULTS / "compare_migration_models.png", dpi=150)
    plt.close(fig)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
