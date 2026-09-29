#!/usr/bin/env python
"""M4 route-A prototype, step 2: analytic/vectorized template construction.

The naive template builder loops over every deme and scans the kernel, i.e.
O(n*k^2) — the same wall as the CSR fold.  For a rectangular domain the valid
offset set of a deme depends only on its clipped distance to the four edges:

    key = (min(r,R), min(rows-1-r,R), min(c,R), min(cols-1-c,R))

so a deme's template is fully determined by that 4-tuple.  Template construction
is then O(n) + O(#templates*k^2), independent of n*k^2.

This script builds templates analytically, verifies them against the natal CSR
(row lengths exactly; destinations exactly; weights to fp), and compares build
time and storage with ``fold_migration_csr``.

Outputs ``results/m4_template_build.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

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


def analytic_templates(rows, cols, kernel):
    """Return (type_of, offsets[list of (dr,dc)], weights[list of arrays])."""
    k = kernel.shape[0]
    R = k // 2
    offs = [(r - R, c - R) for r in range(k) for c in range(k)
            if not (r == R and c == R)]
    n = rows * cols
    type_of = np.empty(n, dtype=np.int64)
    index: dict[tuple, int] = {}
    off_templates: list[tuple[tuple[int, ...], ...]] = []
    w_templates: list[np.ndarray] = []
    rr = np.arange(n) // cols
    cc = np.arange(n) % cols
    key0 = np.minimum(rr, R)
    key1 = np.minimum(rows - 1 - rr, R)
    key2 = np.minimum(cc, R)
    key3 = np.minimum(cols - 1 - cc, R)
    for s in range(n):
        key = (int(key0[s]), int(key1[s]), int(key2[s]), int(key3[s]))
        t = index.get(key)
        if t is None:
            valid = [(dr, dc) for dr, dc in offs
                     if -key[0] <= dr <= key[1] and -key[2] <= dc <= key[3]]
            w = np.array([kernel[dr + R, dc + R] for dr, dc in valid])
            t = len(off_templates)
            index[key] = t
            off_templates.append(tuple(valid))
            w_templates.append(w / w.sum())
        type_of[s] = t
    return type_of, off_templates, w_templates


def _check_deme(csr, cols, s, offs, w, state):
    r, c = divmod(s, cols)
    dest = np.array([(r + dr) * cols + (c + dc) for dr, dc in offs], dtype=np.int64)
    sl = slice(csr.indptr[s], csr.indptr[s + 1])
    if not np.array_equal(dest, csr.dest_idx[sl]):
        state["dest_mismatch"] += 1
    else:
        state["max_weight_diff"] = max(
            state["max_weight_diff"], float(np.abs(w - csr.weights[sl]).max()))


def verify(csr, rows, cols, type_of, off_templates, w_templates, full: bool):
    lengths = np.array([len(o) for o in off_templates])
    row_len_mismatch = int(np.sum(lengths[type_of] != np.diff(csr.indptr)))
    state = {"dest_mismatch": 0, "max_weight_diff": 0.0}
    n = rows * cols
    samples = list({0, cols - 1, n - 1, n - cols, (rows // 2) * cols + cols // 2,
                    min(rows - 1, rows // 3) * cols + min(cols - 1, cols // 3)})
    samples += list(np.linspace(0, n - 1, 50, dtype=int))
    for s in sorted(set(int(x) for x in samples)):
        _check_deme(csr, cols, s, off_templates[type_of[s]], w_templates[type_of[s]], state)
    if full:
        for s in range(n):
            _check_deme(csr, cols, s, off_templates[type_of[s]], w_templates[type_of[s]], state)
    return {"row_length_mismatch": row_len_mismatch,
            "dest_mismatch": state["dest_mismatch"],
            "max_weight_diff": state["max_weight_diff"]}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"rows": []}
    print(f"{'grid':>8} {'k':>3} {'#tmpl':>7} {'fold_s':>8} {'analytic_s':>11} "
          f"{'speedup':>8} {'csr_MB':>8} {'tmpl_MB':>8} {'rowlen_bad':>10} "
          f"{'dest_bad':>9} {'wdiff':>9} {'full':>5}")
    for rows, cols, k in [(60, 60, 11), (120, 120, 21), (300, 300, 21), (300, 300, 51)]:
        full = rows * cols * k * k <= 1.5e7
        t0 = time.perf_counter()
        topo, kernel, csr = build_csr(rows, cols, k)
        fold_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        type_of, off_t, w_t = analytic_templates(rows, cols, kernel)
        analytic_s = time.perf_counter() - t0

        v = verify(csr, rows, cols, type_of, off_t, w_t, full=full)
        csr_b = int(csr.indptr.nbytes + csr.dest_idx.nbytes + csr.weights.nbytes)
        tmpl_b = int(type_of.nbytes) + sum(len(o) * 8 + w.nbytes for o, w in zip(off_t, w_t))
        row = {"rows": rows, "cols": cols, "k": k, "n_templates": len(off_t),
               "fold_s": fold_s, "analytic_s": analytic_s,
               "speedup": fold_s / analytic_s, "csr_bytes": csr_b, "template_bytes": tmpl_b,
               "csr_MB": csr_b / 2**20, "template_MB": tmpl_b / 2**20,
               "verified_full": full, **v}
        out["rows"].append(row)
        print(f"{rows}x{cols:<4} {k:>3} {len(off_t):>7} {fold_s:>8.2f} {analytic_s:>11.3f} "
              f"{fold_s / analytic_s:>8.0f} {csr_b / 2**20:>8.1f} {tmpl_b / 2**20:>8.2f} "
              f"{v['row_length_mismatch']:>10} {v['dest_mismatch']:>9} "
              f"{v['max_weight_diff']:>9.1e} {str(full):>5}")

    with open(RESULTS / "m4_template_build.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote results/m4_template_build.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
