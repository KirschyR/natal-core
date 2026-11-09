"""V2a: operator-level check of stochastic route-A migration.

Isolates the migration operator (one scalar field, one tick) and compares the
two stochastic semantics over many replicates:

* **CPU CSR** (natal algorithm): per source, draw the outbound count with a
  binomial, then distribute it multinomially over the CSR destinations.  The
  total population is conserved exactly.
* **GPU FFT route A** (prototype semantics): compute the expected inbound field
  with a normalized convolution ``g = rate * f / Z`` followed by ``K' (*) g``,
  retain ``f * (1 - rate)``, and draw each destination's inbound count from a
  continuous Poisson / Gamma (``shape = mean``).  Means match the CPU operands;
  destination noise is independent, so the total is not conserved exactly.

The script compares per-deme means and variances and the total-population
distribution.  It is a *semantics prototype*, not a natal-engine change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # optional GPU path
    import torch

    _TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    _TORCH = False

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
                dr = kr - radius
                dc = kc - radius
                w = float(off[kr, kc])
                if w <= 0.0:
                    continue
                rr, cc = r + dr, c + dc
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


def cpu_stochastic(field: np.ndarray, csr: tuple[np.ndarray, np.ndarray, np.ndarray],
                   rate: float, rng: np.random.Generator) -> np.ndarray:
    """natal-style CSR stochastic migration (binomial outbound + multinomial)."""
    indptr, dest, weights = csr
    out = field.astype(np.float64).copy()
    for src in range(field.shape[0]):
        n = int(field[src])
        if n <= 0:
            continue
        outbound = int(rng.binomial(n, rate))
        out[src] -= outbound
        if outbound <= 0:
            continue
        lo, hi = indptr[src], indptr[src + 1]
        probs = weights[lo:hi]
        probs = probs / probs.sum()
        counts = rng.multinomial(outbound, probs)
        np.add.at(out, dest[lo:hi], counts)
    return out


def _fft_conv(field: np.ndarray, off_kernel: np.ndarray, device: str) -> np.ndarray:
    """Linear convolution ``off_kernel (*) field`` cropped to the field size."""
    rows, cols = field.shape
    k = off_kernel.shape[0]
    radius = k // 2
    if _TORCH and device == "cuda":
        f = torch.as_tensor(field, dtype=torch.float32, device="cuda")
        ker = torch.as_tensor(off_kernel, dtype=torch.float32, device="cuda")
        fr, fc = rows + k - 1, cols + k - 1
        fs = torch.zeros((fr, fc), dtype=torch.float32, device="cuda")
        fs[:rows, :cols] = f
        ks = torch.zeros((fr, fc), dtype=torch.float32, device="cuda")
        ks[:k, :k] = ker
        spec = torch.fft.rfft2(fs) * torch.fft.rfft2(ks)
        full = torch.fft.irfft2(spec, s=(fr, fc))
        return full[radius : radius + rows, radius : radius + cols].detach().cpu().numpy()
    f = field.astype(np.float64)
    fr, fc = rows + k - 1, cols + k - 1
    fs = np.zeros((fr, fc))
    fs[:rows, :cols] = f
    ks = np.zeros((fr, fc))
    ks[:k, :k] = off_kernel
    full = np.fft.irfft2(np.fft.rfft2(fs) * np.fft.rfft2(ks), s=(fr, fc))
    return full[radius : radius + rows, radius : radius + cols]


def gpu_fft_stochastic(field: np.ndarray, kernel: np.ndarray, rate: float,
                       rng: np.random.Generator, device: str,
                       generator: "torch.Generator | None") -> np.ndarray:
    """route-A stochastic migration: convolved mean + Gamma destination noise."""
    k = kernel.shape[0]
    radius = k // 2
    off = kernel.copy()
    off[radius, radius] = 0.0
    ones = np.ones_like(field)
    z = _fft_conv(ones, off, device)  # per-source valid weight sum
    safe_z = np.where(z > 0, z, 1.0)
    g = rate * field / safe_z
    mean_inbound = _fft_conv(g, off, device)
    keep = field * (1.0 - rate)
    m = np.maximum(mean_inbound, 0.0)
    if _TORCH and device == "cuda":
        mt = torch.as_tensor(m, dtype=torch.float32, device="cuda").clamp(min=1e-12)
        noise = (
            torch.distributions.Gamma(mt, torch.ones_like(mt))
            .sample()
            .detach()
            .cpu()
            .numpy()
        )
        noise = np.where(m > 0, noise, 0.0)
    else:
        noise = np.where(m > 0, rng.gamma(shape=np.maximum(m, 1e-12)), 0.0)
    return keep + noise


def compare(args: argparse.Namespace) -> dict:
    rows = cols = args.arena
    n = rows * cols
    kernel = hex_gaussian(args.kernel_size, args.avd)
    csr = build_csr(rows, cols, kernel)

    # One fixed initial field; repeat only the migration sampling.
    init_rng = np.random.default_rng(args.seed)
    field = init_rng.integers(args.low, args.high, size=n).astype(np.float64)

    device = "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"
    generator = None
    if device == "cuda":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)

    cpu_samples = np.empty((args.reps, n))
    gpu_samples = np.empty((args.reps, n))
    for rep in range(args.reps):
        cpu_rng = np.random.default_rng(args.seed + 1 + rep)
        gpu_rng = np.random.default_rng(args.seed + 1 + rep)
        cpu_samples[rep] = cpu_stochastic(field, csr, args.rate, cpu_rng)
        if device == "cuda":
            torch.manual_seed(args.seed + 1 + rep)
        gpu_samples[rep] = gpu_fft_stochastic(
            field.reshape(rows, cols), kernel, args.rate, gpu_rng, device, generator
        ).reshape(-1)

    cpu_mean = cpu_samples.mean(axis=0)
    gpu_mean = gpu_samples.mean(axis=0)
    cpu_var = cpu_samples.var(axis=0, ddof=1)
    gpu_var = gpu_samples.var(axis=0, ddof=1)
    mean_rel_deme = np.abs(cpu_mean - gpu_mean) / np.maximum(np.abs(cpu_mean), 1e-9)
    large = cpu_mean >= 5.0
    mean_rel = float(mean_rel_deme.max())
    mean_rel_large = float(mean_rel_deme[large].max()) if large.any() else float("nan")
    var_ratio = np.divide(gpu_var, cpu_var, out=np.ones_like(cpu_var), where=cpu_var > 0)
    cpu_total = cpu_samples.sum(axis=1)
    gpu_total = gpu_samples.sum(axis=1)

    return {
        "device": device,
        "arena": rows,
        "reps": args.reps,
        "rate": args.rate,
        "avd": args.avd,
        "kernel_size": args.kernel_size,
        "initial_total": float(field.sum()),
        "mean_field_max_rel_diff": mean_rel,
        "mean_field_median_rel_diff": float(np.median(mean_rel_deme)),
        "mean_field_max_rel_diff_demes_ge5": mean_rel_large,
        "variance_ratio_median": float(np.median(var_ratio)),
        "variance_ratio_p05": float(np.percentile(var_ratio, 5)),
        "variance_ratio_p95": float(np.percentile(var_ratio, 95)),
        "cpu_total_mean": float(cpu_total.mean()),
        "cpu_total_std": float(cpu_total.std(ddof=1)),
        "gpu_total_mean": float(gpu_total.mean()),
        "gpu_total_std": float(gpu_total.std(ddof=1)),
        "cpu_samples": cpu_samples,
        "gpu_samples": gpu_samples,
    }


def plot(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cpu_mean = res["cpu_samples"].mean(axis=0)
    gpu_mean = res["gpu_samples"].mean(axis=0)
    axes[0].scatter(cpu_mean, gpu_mean, s=8)
    lim = [0, max(cpu_mean.max(), gpu_mean.max()) * 1.05]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("CPU CSR mean")
    axes[0].set_ylabel("GPU FFT mean")
    axes[0].set_title(f"Per-deme mean (max rel {res['mean_field_max_rel_diff']:.2e})")

    cpu_var = res["cpu_samples"].var(axis=0, ddof=1)
    gpu_var = res["gpu_samples"].var(axis=0, ddof=1)
    ratio = np.divide(gpu_var, cpu_var, out=np.ones_like(cpu_var), where=cpu_var > 0)
    axes[1].hist(ratio, bins=40)
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("GPU var / CPU var")
    axes[1].set_ylabel("deme count")
    axes[1].set_title(f"Variance ratio (median {res['variance_ratio_median']:.2f})")

    axes[2].hist(res["cpu_samples"].sum(axis=1), bins=30, alpha=0.6, label="CPU CSR")
    axes[2].hist(res["gpu_samples"].sum(axis=1), bins=30, alpha=0.6, label="GPU FFT")
    axes[2].set_xlabel("total population")
    axes[2].set_ylabel("replicates")
    axes[2].set_title("Total distribution (CPU conserved, GPU noisy)")
    axes[2].legend()
    fig.suptitle(f"V2a stochastic operator: {res['arena']}x{res['arena']} hex, device={res['device']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    res = compare(args)
    png = RESULTS / "stochastic_fft_operator.png"
    plot(res, png)
    summary = {k: v for k, v in res.items() if k not in ("cpu_samples", "gpu_samples")}
    summary["plot"] = png.name
    summary["mean_passed"] = bool(
        res["mean_field_max_rel_diff"] < 5e-2
        and 0.9 <= res["variance_ratio_median"] <= 1.1
    )
    (RESULTS / "stochastic_fft_operator.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["mean_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
