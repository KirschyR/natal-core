"""GPU M2 migration operator (torch/cuFFT) for the chasing prototype.

The CPU M2 operator is split into two stages:

1. per-plane source outbound sampling (NumPy, identical to the v3 model);
2. the FFT convolution mean plus local block multinomial sampling.

This module keeps stage 1 on the CPU and moves stage 2 to the GPU.  It does not
use natal product code and does not batch replicates: one call handles the
adult planes of one spatial state.  The class supports a CPU block-sampling
path for exact FFT validation and a fully vectorised GPU block-sampling path
for performance measurements.

The module needs torch (conda environment), not the natal virtualenv.  Run the
self-test with:

    /opt/conda/bin/python chasing_model/gpu_migration.py --rows 15 31
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Scripts live in chasing_model/scripts/; outputs stay under chasing_model/.
HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results" / "gpu_fft"
Z = 3
S = 2
A = 10
NEW_ADULT_AGE = 2
EPS = 1e-12


def hex_gaussian(size: int, mean_dispersal: float) -> np.ndarray:
    """Return a normalized hex Gaussian kernel (mirrors the v3 model)."""
    radius = size // 2
    dr, dc = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = mean_dispersal / np.sqrt(np.pi / 2.0)
    kernel = np.exp(-0.5 * (dr**2 + dc**2 + dr * dc) / sigma**2)
    return kernel / kernel.sum()


def _off_center(kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    off = kernel.copy()
    off[radius, radius] = 0.0
    return off


def _conv_cpu(field: np.ndarray, off: np.ndarray) -> np.ndarray:
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


def _off_z_cpu(rows: int, cols: int, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    off = _off_center(kernel)
    z = _conv_cpu(np.ones((rows, cols)), off)
    return off, np.where(z > 0.0, z, 1.0)


def _continuous_binomial_vec(
    rng: np.random.Generator, n: np.ndarray, p: float
) -> np.ndarray:
    """Continuous binomial used by the v3 CPU migration."""
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


def _blocks(rows: int, cols: int, block: int):
    for r0 in range(0, rows, block):
        r1 = min(r0 + block, rows)
        for c0 in range(0, cols, block):
            c1 = min(c0 + block, cols)
            yield r0, r1, c0, c1


def _continuous_multinomial_block(
    rng: np.random.Generator, n: float, probs: np.ndarray
) -> np.ndarray:
    if n <= 1.0 + EPS:
        return n * probs
    alpha = probs * (n - 1.0)
    draws = np.where(alpha > 0.0, rng.gamma(shape=np.maximum(alpha, EPS)), 0.0)
    total = draws.sum()
    return n * draws / total if total > EPS else n * probs


def _sample_blocks_cpu(
    mu: np.ndarray, total: float, block: int, rng: np.random.Generator
) -> np.ndarray:
    rows, cols = mu.shape
    counts = np.zeros_like(mu)
    if total <= EPS:
        return counts
    parts: list[tuple[int, int, int, int, np.ndarray, float, float]] = []
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
    for r0, r1, c0, c1, blk, mb, tb in parts:
        counts[r0:r1, c0:c1] = _continuous_multinomial_block(
            rng, tb * scale, (blk / mb).ravel()
        ).reshape(r1 - r0, c1 - c0)
    return counts


def m2_reference_cpu(
    data: np.ndarray,
    kernel: np.ndarray,
    rate: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """CPU M2 reference with the same global ordering as the GPU CPU path.

    All source outbound draws happen first, then the FFT means, then all block
    samples.  This ordering lets the GPU CPU-sampling path consume the same
    NumPy RNG stream and be compared field by field.
    """
    rng = np.random.default_rng(seed)
    rows, cols = data.shape[-2:]
    planes = data[:, :, NEW_ADULT_AGE:].reshape(Z * S * (A - NEW_ADULT_AGE), rows, cols)
    outbound = np.zeros_like(planes)
    for i in range(planes.shape[0]):
        outbound[i] = _continuous_binomial_vec(
            rng, planes[i].ravel(), rate
        ).reshape(rows, cols)
    keep = planes - outbound
    off, z = _off_z_cpu(rows, cols, kernel)
    mu = np.stack([_conv_cpu(outbound[i] / z, off) for i in range(planes.shape[0])])
    counts = np.zeros_like(planes)
    block = kernel.shape[0]
    for i in range(planes.shape[0]):
        counts[i] = _sample_blocks_cpu(mu[i], float(mu[i].sum()), block, rng)
    out = data.copy()
    out[:, :, NEW_ADULT_AGE:] = (keep + counts).reshape(
        Z, S, A - NEW_ADULT_AGE, rows, cols
    )
    return out


class M2GPU:
    """GPU FFT + local block multinomial M2 migration operator."""

    def __init__(
        self,
        rows: int,
        cols: int,
        kernel: np.ndarray,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        block: int | None = None,
        seed: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.rows = rows
        self.cols = cols
        self.device = torch.device(device)
        self.dtype = dtype
        self.kernel = np.asarray(kernel, dtype=np.float64)
        k = self.kernel.shape[0]
        self.radius = k // 2
        self.fr, self.fc = rows + k - 1, cols + k - 1
        self.block = block or k
        self.off, z_np = _off_z_cpu(rows, cols, self.kernel)
        self.z = torch.as_tensor(z_np, dtype=self.dtype, device=self.device)
        ks = np.zeros((self.fr, self.fc))
        ks[:k, :k] = self.off
        self.kernel_fft = torch.fft.rfft2(
            torch.as_tensor(ks, dtype=self.dtype, device=self.device)
        )
        self.np_rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator(device=self.device)
        self.torch_gen.manual_seed(seed)
        self.pad_r = ((rows + self.block - 1) // self.block) * self.block
        self.pad_c = ((cols + self.block - 1) // self.block) * self.block

    def fft_conv(self, field: torch.Tensor) -> torch.Tensor:
        """Convolve ``(planes, rows, cols)`` with the de-centered kernel."""
        fs = field.new_zeros((field.shape[0], self.fr, self.fc))
        fs[:, : self.rows, : self.cols] = field
        full = torch.fft.irfft2(
            torch.fft.rfft2(fs) * self.kernel_fft, s=(self.fr, self.fc)
        )
        return full[
            :,
            self.radius : self.radius + self.rows,
            self.radius : self.radius + self.cols,
        ]

    def _block_probabilities(self, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        planes = mu.shape[0]
        mu_pad = mu.new_zeros((planes, self.pad_r, self.pad_c))
        mu_pad[:, : self.rows, : self.cols] = mu
        view = mu_pad.reshape(
            planes,
            self.pad_r // self.block,
            self.block,
            self.pad_c // self.block,
            self.block,
        )
        mb = view.sum(dim=(2, 4)).reshape(planes, -1)
        blk = view.permute(0, 1, 3, 2, 4).reshape(
            planes, -1, self.block * self.block
        )
        safe = torch.where(mb > EPS, mb, torch.ones_like(mb))
        probs = torch.where(
            (mb > EPS)[..., None], blk / safe[..., None], torch.zeros_like(blk)
        )
        return mb, probs

    def _sample_blocks_gpu(self, mu: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        planes = mu.shape[0]
        mb, probs = self._block_probabilities(mu)
        valid = mb > EPS
        tb = torch.zeros_like(mb)
        if valid.any():
            tb[valid] = torch._standard_gamma(  # noqa: SLF001 - torch private API
                mb[valid], generator=self.torch_gen
            )
        block_total = tb.sum(dim=1, keepdim=True)
        scale = torch.where(
            block_total > EPS, total[:, None] / block_total, torch.ones_like(block_total)
        )
        n = tb * scale
        n_exp = n[..., None]
        small = n_exp <= 1.0 + EPS
        counts_small = n_exp * probs
        alpha = probs * (n_exp - 1.0)
        draws = torch._standard_gamma(  # noqa: SLF001 - torch private API
            alpha.clamp(min=EPS), generator=self.torch_gen
        )
        draws = torch.where(alpha > EPS, draws, torch.zeros_like(draws))
        sum_draws = draws.sum(dim=-1, keepdim=True)
        counts_big = torch.where(
            sum_draws > EPS,
            draws * (n_exp / sum_draws.clamp(min=EPS)),
            counts_small,
        )
        counts = torch.where(small, counts_small, counts_big)
        counts = torch.where(valid[..., None], counts, torch.zeros_like(counts))
        view = counts.reshape(
            planes,
            self.pad_r // self.block,
            self.pad_c // self.block,
            self.block,
            self.block,
        ).permute(0, 1, 3, 2, 4)
        counts = view.reshape(planes, self.pad_r, self.pad_c)
        return counts[:, : self.rows, : self.cols]

    def _outbound_gpu(self, field: torch.Tensor, rate: float) -> torch.Tensor:
        """Vectorised continuous-binomial outbound sampling on the GPU."""
        if rate >= 1.0:
            # Matches natal's ``sample_outbound``: a full-rate source sends all
            # of its mass; no Beta draw is needed.
            return field
        out = field * rate
        big = field > 1.0 + EPS
        conc = field - 1.0
        ga = torch._standard_gamma(  # noqa: SLF001 - torch private API
            (rate * conc).clamp(min=EPS), generator=self.torch_gen
        )
        gb = torch._standard_gamma(  # noqa: SLF001 - torch private API
            ((1.0 - rate) * conc).clamp(min=EPS), generator=self.torch_gen
        )
        denom = ga + gb
        prop = torch.where(
            denom > EPS, ga / denom.clamp(min=EPS), torch.full_like(field, rate)
        )
        return torch.where(big, field * prop, out)

    def migrate(
        self,
        data: np.ndarray,
        *,
        rate: float = 1.0,
        sample_device: str = "gpu",
        outbound_device: str = "cpu",
        return_timing: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        """Migrate the adult ages of ``data`` and return the new state."""
        rows, cols = self.rows, self.cols
        planes_np = data[:, :, NEW_ADULT_AGE:].reshape(
            Z * S * (A - NEW_ADULT_AGE), rows, cols
        )
        timing: dict[str, float] = {}
        keep_gpu: torch.Tensor | None = None
        keep_np: np.ndarray | None = None
        if outbound_device == "cpu":
            t0 = time.perf_counter()
            outbound = np.zeros_like(planes_np)
            for i in range(planes_np.shape[0]):
                outbound[i] = _continuous_binomial_vec(
                    self.np_rng, planes_np[i].ravel(), rate
                ).reshape(rows, cols)
            keep_np = planes_np - outbound
            timing["outbound_s"] = time.perf_counter() - t0

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_gpu = torch.as_tensor(outbound, dtype=self.dtype, device=self.device)
            torch.cuda.synchronize()
            timing["transfer_in_s"] = time.perf_counter() - t0
        elif outbound_device == "gpu":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            field_gpu = torch.as_tensor(
                planes_np, dtype=self.dtype, device=self.device
            )
            out_gpu = self._outbound_gpu(field_gpu, rate)
            keep_gpu = field_gpu - out_gpu
            torch.cuda.synchronize()
            timing["transfer_in_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            outbound = None
            timing["outbound_s"] = 0.0
        else:
            raise ValueError("outbound_device must be 'cpu' or 'gpu'")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mu = self.fft_conv(out_gpu / self.z)
        torch.cuda.synchronize()
        timing["fft_s"] = time.perf_counter() - t0
        total = mu.sum(dim=(1, 2))

        if sample_device == "cpu":
            mu_cpu = mu.cpu().numpy()
            counts = np.zeros_like(planes_np)
            for i in range(planes_np.shape[0]):
                counts[i] = _sample_blocks_cpu(
                    mu_cpu[i], float(mu_cpu[i].sum()), self.block, self.np_rng
                )
            timing["sampling_s"] = 0.0
            timing["transfer_out_s"] = 0.0
            if keep_gpu is not None:
                t0 = time.perf_counter()
                keep_np = keep_gpu.cpu().numpy()
                timing["transfer_out_s"] += time.perf_counter() - t0
        elif sample_device == "gpu":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            counts_gpu = self._sample_blocks_gpu(mu, total)
            torch.cuda.synchronize()
            timing["sampling_s"] = time.perf_counter() - t0
            if keep_gpu is not None:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                planes_out = keep_gpu + counts_gpu
                out_planes = planes_out.cpu().numpy()
                torch.cuda.synchronize()
                timing["transfer_out_s"] = time.perf_counter() - t0
                keep_np = None
                counts = None
            else:
                t0 = time.perf_counter()
                counts = counts_gpu.cpu().numpy()
                timing["transfer_out_s"] = time.perf_counter() - t0
        else:
            raise ValueError("sample_device must be 'cpu' or 'gpu'")

        out = data.copy()
        if keep_np is None:
            out[:, :, NEW_ADULT_AGE:] = out_planes.reshape(
                Z, S, A - NEW_ADULT_AGE, rows, cols
            )
        else:
            assert counts is not None
            out[:, :, NEW_ADULT_AGE:] = (keep_np + counts).reshape(
                Z, S, A - NEW_ADULT_AGE, rows, cols
            )
        timing["total_s"] = sum(timing.values())
        if return_timing:
            return out, timing
        return out


def _synthetic_data(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((Z, S, A, rows, cols))
    data[:, :, NEW_ADULT_AGE:] = rng.random((Z, S, A - NEW_ADULT_AGE, rows, cols)) * 10.0
    return data


def _run_benchmark(cases: list[tuple[str, int, int, float]]) -> dict[str, Any]:
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        git_head = "unknown"
    payload: dict[str, Any] = {
        "git_head": git_head,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "note": "synthetic adult planes; no lifecycle; warm timings after one run",
        "cases": {},
    }
    for name, rows, kernel_size, avd in cases:
        kernel = hex_gaussian(kernel_size, avd)
        data = _synthetic_data(rows, rows, seed=0)
        cpu = m2_reference_cpu(data, kernel, rate=1.0, seed=0)
        case: dict[str, Any] = {
            "rows": rows,
            "kernel_size": kernel_size,
            "avd": avd,
        }
        for dtype, tag in ((torch.float64, "f64"), (torch.float32, "f32")):
            gpu = M2GPU(rows, rows, kernel, dtype=dtype, seed=0)
            out_cpu_sample = gpu.migrate(
                data, sample_device="cpu", outbound_device="cpu"
            )
            diff = np.abs(out_cpu_sample - cpu)
            case[f"{tag}_cpu_sampling_max_abs"] = float(diff.max())
            case[f"{tag}_cpu_sampling_mean_abs"] = float(diff.mean())

        # Full GPU path: outbound + FFT + block sampling on the device.
        gpu = M2GPU(rows, rows, kernel, dtype=torch.float32, seed=0)
        out_gpu, timing = gpu.migrate(
            data,
            sample_device="gpu",
            outbound_device="gpu",
            return_timing=True,
        )
        _, timing_warm = gpu.migrate(
            data,
            sample_device="gpu",
            outbound_device="gpu",
            return_timing=True,
        )
        repeat = M2GPU(rows, rows, kernel, dtype=torch.float32, seed=0)
        out_repeat = repeat.migrate(
            data, sample_device="gpu", outbound_device="gpu"
        )
        case["gpu_full"] = {
            "adult_in": float(data[:, :, NEW_ADULT_AGE:].sum()),
            "adult_out": float(out_gpu[:, :, NEW_ADULT_AGE:].sum()),
            "negative_min": float(out_gpu.min()),
            "finite": bool(np.isfinite(out_gpu).all()),
            "repeat_identical": bool(np.array_equal(out_gpu, out_repeat)),
            "timing": timing,
            "timing_warm": timing_warm,
        }

        # CPU outbound + GPU FFT/sampling: isolates the remaining CPU stage.
        gpu_cpu_out = M2GPU(rows, rows, kernel, dtype=torch.float32, seed=0)
        _, timing_cpu_out = gpu_cpu_out.migrate(
            data,
            sample_device="gpu",
            outbound_device="cpu",
            return_timing=True,
        )
        case["cpu_outbound_gpu_sampling_timing"] = timing_cpu_out
        payload["cases"][name] = case
        print(
            f"[{name} {rows}x{rows} k{kernel_size} avd{avd:g}] "
            f"f64 cpu-sampling max abs {case['f64_cpu_sampling_max_abs']:.3e}; "
            f"f32 {case['f32_cpu_sampling_max_abs']:.3e}; "
            f"full GPU {case['gpu_full']['timing_warm']['total_s']*1e3:.2f} ms/tick",
            flush=True,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=RESULTS / "a1_gpu_m2.json"
    )
    args = parser.parse_args()
    cases = [
        ("small", 15, 7, 0.8),
        ("large", 31, 11, 1.5),
    ]
    payload = _run_benchmark(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
