#!/usr/bin/env python
"""M4 route-A prototype: template/label migration vs normalized convolution/FFT.

Validates the route-A design (``Hex_model_recon.md`` §4) against the current
natal CSR migration operator, for the *general hex model* migration semantics
(source outbound = rate*value; out-of-grid neighbours dropped; valid neighbours
renormalized to sum 1; source keeps value - outbound).

Sections:
1. reference CSR via ``fold_migration_csr``;
2. template+label compression reproduces the CSR triples *exactly* (same
   (src,dest,weight) set) and the independently derived weights agree to fp;
3. normalized (masked) convolution ``Z = K' (*) m`` reproduces the CSR operator
   to ~1e-15 (both direct stencil and FFT);
4. storage and per-step cost comparison (CSR vs template vs direct vs FFT).

Outputs ``results/m4_routeA_prototype.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from natal.frontend.spatial.migration import fold_migration_csr
from natal.frontend.spatial.topology import HexGrid, build_gaussian_kernel

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
RESULTS = HEXROOT / "results"
SIGMA = 1.5


def build_csr(rows, cols, size):
    topo = HexGrid(rows=rows, cols=cols, wrap=False)
    kernel = build_gaussian_kernel(HexGrid, size=size, sigma=SIGMA)
    csr = fold_migration_csr(
        n_demes=rows * cols, topology=topo, adjacency_dense=np.zeros((1, 1)),
        migration_kernel=kernel, kernel_bank=None, deme_kernel_ids=None,
        kernel_include_center=False, adjust_on_edge=False, mode="kernel",
    )
    return topo, kernel, csr


def csr_triples(csr, rows, cols):
    n = rows * cols
    src = np.repeat(np.arange(n), np.diff(csr.indptr))
    triples = {(int(s), int(d)): float(w)
               for s, d, w in zip(src, csr.dest_idx, csr.weights)}
    return triples


def template_model(rows, cols, kernel):
    """Independent template+label model: (type_of_deme, list of templates).

    A template is ``(linear_offsets, weights)`` in kernel row-major order.
    """
    topo = HexGrid(rows=rows, cols=cols, wrap=False)
    n = rows * cols
    k = kernel.shape[0]
    c = k // 2
    offs = [(r - c, col - c) for r in range(k) for col in range(k)
            if not (r == c and col == c)]
    templates: list[tuple[tuple[int, ...], np.ndarray]] = []
    index: dict[frozenset, int] = {}
    type_of = np.empty(n, dtype=np.int64)
    for s in range(n):
        r, col = divmod(s, cols)
        valid = [(dr, dc) for dr, dc in offs
                 if topo.normalize_coord(r + dr, col + dc) is not None]
        sig = frozenset(valid)
        if sig not in index:
            w = np.array([kernel[dr + c, dc + c] for dr, dc in valid])
            w = w / w.sum()
            lin = tuple(dr * cols + dc for dr, dc in valid)
            index[sig] = len(templates)
            templates.append((lin, w))
        type_of[s] = index[sig]
    return type_of, templates


def template_triples(rows, cols, type_of, templates):
    linear_of = np.arange(rows * cols)
    triples = {}
    for s in range(rows * cols):
        lin, w = templates[type_of[s]]
        for off, ww in zip(lin, w):
            triples[(s, s + off)] = float(ww)
    return triples


def template_weights_diff(csr, rows, cols, type_of, templates):
    """Max |weight - CSR weight| for identically keyed triples."""
    ref = csr_triples(csr, rows, cols)
    ind = template_triples(rows, cols, type_of, templates)
    if set(ref) != set(ind):
        return float("inf"), len(set(ref) ^ set(ind))
    return max(abs(ref[key] - ind[key]) for key in ref), 0


def apply_csr(v, csr, rate):
    n = v.size
    src = np.repeat(np.arange(n), np.diff(csr.indptr))
    outbound = rate * v
    dest_sum = np.bincount(csr.dest_idx, weights=outbound[src] * csr.weights, minlength=n)
    return v - outbound + dest_sum


def apply_template(v, type_of, templates, rate):
    n = v.size
    outbound = rate * v
    dest_sum = np.zeros(n)
    for s in range(n):
        lin, w = templates[type_of[s]]
        ob = outbound[s]
        for off, ww in zip(lin, w):
            dest_sum[s + off] += ob * ww
    return v - outbound + dest_sum


def normalization_field(rows, cols, kernel):
    kernel = kernel.copy()
    c = kernel.shape[0] // 2
    kernel[c, c] = 0.0
    m = np.ones((rows, cols))
    Z = ndimage.convolve(m, kernel, mode="constant", cval=0.0)
    return kernel, Z


def apply_conv(v, Kp, Z, rate):
    g = rate * v / Z
    return v - rate * v + ndimage.convolve(g, Kp, mode="constant", cval=0.0)


def apply_conv_fft(v, Kp, Z, rate):
    g = rate * v / Z
    return v - rate * v + fftconvolve(g, Kp, mode="same")


def memory_bytes(csr, rows, cols, type_of, templates, Kp):
    csr_b = int(csr.indptr.nbytes + csr.dest_idx.nbytes + csr.weights.nbytes)
    tmpl_b = int(type_of.nbytes)
    for lin, w in templates:
        tmpl_b += len(lin) * 8 + w.nbytes
    fft_extra = 0  # kernel FFT is n_complex*16, same order as Z
    return {"csr_bytes": csr_b, "template_bytes": tmpl_b,
            "csr_MB": csr_b / 2**20, "template_MB": tmpl_b / 2**20}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict = {"section": [], "timing": [], "crossover": []}
    rng = np.random.default_rng(0)

    print("1-3) operator equivalence (CSR vs template vs conv vs FFT)")
    print(f"{'grid':>6} {'k':>3} {'#tmpl':>6} {'triples_equal':>13} {'w_diff':>9} "
          f"{'conv_max':>10} {'fft_max':>10}")
    for rows, cols, size in [(30, 30, 3), (30, 30, 5), (40, 40, 11), (60, 60, 21)]:
        topo, kernel, csr = build_csr(rows, cols, size)
        type_of, templates = template_model(rows, cols, kernel)
        ref = csr_triples(csr, rows, cols)
        ind = template_triples(rows, cols, type_of, templates)
        triples_equal = set(ref) == set(ind)
        n_sym = len(set(ref) ^ set(ind))
        w_diff = (max(abs(ref[k] - ind[k]) for k in ref) if triples_equal
                  else float("inf"))
        Kp, Z = normalization_field(rows, cols, kernel)
        v = rng.random(rows * cols)
        rate = np.full(rows * cols, 0.5)
        v2 = v.reshape(rows, cols)
        rate2 = rate.reshape(rows, cols)
        base = apply_csr(v, csr, rate)
        conv = apply_conv(v2, Kp, Z, rate2).ravel()
        fftv = apply_conv_fft(v2, Kp, Z, rate2).ravel()
        conv_err = float(np.abs(conv - base).max())
        fft_err = float(np.abs(fftv - base).max())
        row = {"rows": rows, "cols": cols, "k": size, "n_templates": len(templates),
               "triples_equal": bool(triples_equal), "weight_diff": w_diff,
               "n_symmetric_diff": int(n_sym), "conv_max_abs_err": conv_err,
               "fft_max_abs_err": fft_err, **memory_bytes(csr, rows, cols, type_of, templates, Kp)}
        out["section"].append(row)
        print(f"{rows}x{cols:>3} {size:>3} {len(templates):>6} {str(triples_equal):>13} "
              f"{w_diff:>9.2e} {conv_err:>10.2e} {fft_err:>10.2e}")

    print("\n4) per-step cost (300x300; CSR-scatter vs direct vs FFT)")
    print(f"{'k':>3} {'csr_ms':>9} {'direct_ms':>10} {'fft_ms':>9} {'fft/direct':>10}")
    rows = cols = 300
    v = rng.random(rows * cols)
    rate = np.full(rows * cols, 0.5)
    v2 = v.reshape(rows, cols)
    rate2 = rate.reshape(rows, cols)
    for size in [3, 5, 11, 21, 51]:
        topo, kernel, csr = build_csr(rows, cols, size)
        Kp, Z = normalization_field(rows, cols, kernel)

        def bench(fn, reps=2):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps * 1e3

        csr_ms = bench(lambda: apply_csr(v, csr, rate))
        dir_ms = bench(lambda: apply_conv(v2, Kp, Z, rate2))
        fft_ms = bench(lambda: apply_conv_fft(v2, Kp, Z, rate2))
        out["timing"].append({"k": size, "csr_ms": csr_ms, "direct_ms": dir_ms,
                              "fft_ms": fft_ms, "fft_over_direct": fft_ms / dir_ms,
                              "csr_nnz": int(csr.dest_idx.size)})
        print(f"{size:>3} {csr_ms:>9.2f} {dir_ms:>10.2f} {fft_ms:>9.2f} {fft_ms / dir_ms:>10.2f}")

    with open(RESULTS / "m4_routeA_prototype.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("\nwrote results/m4_routeA_prototype.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
