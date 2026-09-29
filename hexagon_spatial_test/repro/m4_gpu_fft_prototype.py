#!/usr/bin/env python
"""M4 route-A prototype, step 1: GPU cuFFT proof of concept.

Runs the normalized-convolution migration step (``Z = K' (*) m``;
``f_new = f*(1-rate) + K' (*) (rate*f/Z)``) with cuFFT via PyTorch and compares
it to the CPU direct-stencil reference.  This does not use natal (the GPU
extension is not an FFT engine); it is a standalone feasibility check for using
cuFFT as the route-A acceleration.

Run with the CUDA-enabled interpreter (PyTorch is not in the repo venv)::

    /opt/conda/bin/python repro/m4_gpu_fft_prototype.py

Outputs ``results/m4_gpu_fft_prototype.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
RESULTS = HEXROOT / "results"
SIGMA = 1.5
RATE = 0.5


def hex_kernel(size: int) -> np.ndarray:
    c = (size - 1) / 2.0
    ii, jj = np.indices((size, size), dtype=float)
    dr, dc = ii - c, jj - c
    d2 = dr**2 + dc**2 + dr * dc
    k = np.exp(-d2 / (2 * SIGMA**2))
    return k / k.sum()


def setup(rows: int, cols: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = hex_kernel(k)
    c = k // 2
    Kp = kernel.copy()
    Kp[c, c] = 0.0
    Z = ndimage.convolve(np.ones((rows, cols)), Kp, mode="constant", cval=0.0)
    return kernel, Kp, Z


def cpu_conv(f: np.ndarray, Kp: np.ndarray, Z: np.ndarray) -> np.ndarray:
    g = RATE * f / Z
    return f - RATE * f + ndimage.convolve(g, Kp, mode="constant", cval=0.0)


def gpu_conv(f, Kp, Z, dtype):
    rows, cols = f.shape
    k = Kp.shape[0]
    R = k // 2
    shape = (rows + k - 1, cols + k - 1)  # avoids circular wrap
    dev = torch.device("cuda")
    fpad = np.zeros(shape, dtype=np.float64)
    fpad[:rows, :cols] = RATE * f / Z
    Kpad = np.zeros(shape, dtype=np.float64)
    Kpad[:k, :k] = Kp
    ft = torch.as_tensor(fpad, dtype=dtype, device=dev)
    kt = torch.as_tensor(Kpad, dtype=dtype, device=dev)
    out_full = torch.fft.irfft2(torch.fft.rfft2(ft) * torch.fft.rfft2(kt), s=shape)
    out = out_full[R:R + rows, R:R + cols]
    res = torch.as_tensor(f, dtype=dtype, device=dev) * (1 - RATE) + out
    torch.cuda.synchronize()
    return res.cpu().numpy().astype(np.float64)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 2
    print(f"torch {torch.__version__} device={torch.cuda.get_device_name(0)}")
    rng = np.random.default_rng(0)
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "rows": []}

    print(f"{'grid':>8} {'k':>3} {'f32_rel':>10} {'f64_rel':>10} "
          f"{'cpu_ms':>8} {'gpu32_ms':>9} {'gpu64_ms':>9}")
    for rows, cols, k in [(120, 120, 11), (300, 300, 21), (300, 300, 51),
                          (600, 600, 51), (1000, 1000, 51)]:
        _, Kp, Z = setup(rows, cols, k)
        f = rng.random((rows, cols))
        ref = cpu_conv(f, Kp, Z)
        g32 = gpu_conv(f, Kp, Z, torch.float32)
        g64 = gpu_conv(f, Kp, Z, torch.float64)
        scale = max(np.abs(ref).max(), 1e-30)
        rel32 = float(np.abs(g32 - ref).max() / scale)
        rel64 = float(np.abs(g64 - ref).max() / scale)

        def bench_cpu():
            ndimage.convolve(RATE * f / Z, Kp, mode="constant", cval=0.0)

        t0 = time.perf_counter()
        for _ in range(3):
            bench_cpu()
        cpu_ms = (time.perf_counter() - t0) / 3 * 1e3

        def bench_gpu(dtype):
            t0 = time.perf_counter()
            for _ in range(10):
                gpu_conv(f, Kp, Z, dtype)
            return (time.perf_counter() - t0) / 10 * 1e3

        gpu32_ms = bench_gpu(torch.float32)
        gpu64_ms = bench_gpu(torch.float64)
        out["rows"].append({"rows": rows, "cols": cols, "k": k,
                            "f32_rel": rel32, "f64_rel": rel64,
                            "cpu_ms": cpu_ms, "gpu32_ms": gpu32_ms, "gpu64_ms": gpu64_ms})
        print(f"{rows}x{cols:<4} {k:>3} {rel32:>10.2e} {rel64:>10.2e} "
              f"{cpu_ms:>8.2f} {gpu32_ms:>9.2f} {gpu64_ms:>9.2f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "m4_gpu_fft_prototype.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote results/m4_gpu_fft_prototype.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
