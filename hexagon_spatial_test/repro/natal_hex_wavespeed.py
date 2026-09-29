#!/usr/bin/env python
"""Reproduce the general hex wave model with natal-core spatial primitives.

This is the M2 reproduction: the *general hex model* used for the paper's
wave-speed experiments is a discrete-generation, continuous-count recursion
with Gaussian dispersal on a hexagonal lattice.  It is NOT natal-core's
age-structured lifecycle, so instead of abusing the engine we implement the
recursion in numpy while sourcing the two spatial primitives from natal-core:

* ``natal.frontend.spatial.topology.HexGrid`` -- hexagonal metric
  ``dr^2 + dc^2 + dr*dc``;
* ``natal.frontend.spatial.topology.build_gaussian_kernel`` -- the same
  Gaussian in that metric, ``sigma = mean_dispersal / sqrt(pi/2)``.

We compare the natal kernel against the reference MATLAB
``get_mig_matrix25`` kernel and the resulting wave speeds against the MATLAB
golden (``golden/hex_homing_wavespeed_m2.json``).

Usage::

    python repro/natal_hex_wavespeed.py [label ...]
    # labels default to: flat_paper flat_code junction_paper junction_code
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from natal.frontend.spatial.topology import HexGrid, build_gaussian_kernel

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
GOLDEN = HEXROOT / "golden"
RESULTS = HEXROOT / "results"

LAMBDA = 5.0
GN = 5
CARRIER = np.array([0, 1, 4])  # 1-based [1,2,5]
BOUNDARY1 = 0.2
KERNEL_SIZE = 51
SPEED_CRITERION = 0.8


def homing_mats() -> np.ndarray:
    """Reproduce the homing ``drive_generator.m`` (gcr=0, dc=1, herr=0)."""
    gcr, dc, herr, ddfitness = 0.0, 1.0, 0.0, 1.0
    mat_dd = np.array([
        [1, dc / 2 + 1 / 2, 0, 0, 1 / 2],
        [dc / 2 + 1 / 2, dc * (dc / 2 + 1 / 2) - (dc / 4 + 1 / 4) * (dc + gcr - 1) + gcr * (dc / 4 + 1 / 4), 0, 0, dc / 4 + 1 / 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1 / 2, dc / 4 + 1 / 4, 0, 0, 1 / 4],
    ])
    mat_dw = np.array([
        [0, -((herr - 1) ** 2 * (dc + gcr - 1)) / 2, (herr - 1) ** 2, (herr - 1) ** 2 / 2, 0],
        [(herr / 2 - 1 / 2) * (dc + gcr - 1), dc * (herr / 2 - 1 / 2) * (dc + gcr - 1) - (dc + gcr - 1) * ((herr / 2 - 1 / 2) * (dc + gcr - 1) + (dc * (herr - 1) ** 2) / 2 - gcr * (herr / 4 - 1 / 4)) + gcr * (herr / 4 - 1 / 4) * (dc + gcr - 1), (herr / 2 - 1 / 2) * (dc + gcr - 1) + dc * (herr - 1) ** 2 - gcr * (herr / 2 - 1 / 2), (herr / 4 - 1 / 4) * (dc + gcr - 1) + (dc * (herr - 1) ** 2) / 2 - gcr * (herr / 4 - 1 / 4), (herr / 4 - 1 / 4) * (dc + gcr - 1)],
        [1, dc / 2 + 1 / 2, 0, 0, 1 / 2],
        [1 / 2, dc / 4 + 1 / 4, 0, 0, 1 / 4],
        [0, (herr / 4 - 1 / 4) * (dc + gcr - 1), 1 / 2 - herr / 2, 1 / 4 - herr / 4, 0],
    ])
    mat_ww = np.array([
        [0, 0, 0, 0, 0],
        [0, ((herr - 1) ** 2 * (dc + gcr - 1) ** 2) / 4, -((herr - 1) ** 2 * (dc + gcr - 1)) / 2, -((herr - 1) ** 2 * (dc + gcr - 1)) / 4, 0],
        [0, 1 / 2 - gcr / 2 - dc / 2, 1, 1 / 2, 0],
        [0, 1 / 4 - gcr / 4 - dc / 4, 1 / 2, 1 / 4, 0],
        [0, 0, 0, 0, 0],
    ])
    mat_rw = np.array([
        [0, 0, 0, 0, 0],
        [0, (gcr * (herr / 4 - 1 / 4) - (herr * (herr - 1) * (dc + gcr - 1)) / 2) * (dc + gcr - 1) + gcr * (herr / 4 - 1 / 4) * (dc + gcr - 1), herr * (herr - 1) * (dc + gcr - 1) - gcr * (herr / 2 - 1 / 2), (dc + gcr - 1) * (herr / 4 + (herr * (herr - 1)) / 2 - 1 / 4) - gcr * (herr / 4 - 1 / 4), (herr / 4 - 1 / 4) * (dc + gcr - 1)],
        [0, gcr / 2, 0, 1 / 2, 1 / 2],
        [0, 1 / 4 - dc / 4, 1 / 2, 1 / 2, 1 / 4],
        [0, (herr / 4 - 1 / 4) * (dc + gcr - 1), 1 / 2 - herr / 2, 1 / 4 - herr / 4, 0],
    ])
    mat_dr = np.array([
        [0, gcr / 2 + ((herr - 1) ** 2 / 2 - 1 / 2) * (dc + gcr - 1), 1 - (herr - 1) ** 2, 1 - (herr - 1) ** 2 / 2, 1 / 2],
        [gcr / 2 - (herr * (dc + gcr - 1)) / 2, gcr * (dc / 2 + gcr / 2 - (herr / 4 + 1 / 4) * (dc + gcr - 1)) + (dc + gcr - 1) * (dc * ((herr - 1) ** 2 / 2 - 1 / 2) - gcr * (herr / 4 + 1 / 4) + (herr * (dc + gcr - 1)) / 2) + dc * (gcr / 2 - (herr * (dc + gcr - 1)) / 2), (gcr * herr) / 2 - dc * ((herr - 1) ** 2 - 1) - (herr * (dc + gcr - 1)) / 2, gcr * (herr / 4 + 1 / 4) - (herr / 4 + 1 / 4) * (dc + gcr - 1) - dc * ((herr - 1) ** 2 / 2 - 1), dc / 2 + gcr / 2 - (herr / 4 + 1 / 4) * (dc + gcr - 1)],
        [0, 0, 0, 0, 0],
        [1 / 2, dc / 4 + 1 / 4, 0, 0, 1 / 4],
        [1 / 2, dc / 2 + gcr / 2 - (herr / 4 + 1 / 4) * (dc + gcr - 1), herr / 2, herr / 4 + 1 / 4, 1 / 2],
    ])
    F = np.sqrt(ddfitness)
    fit = np.array([F**2, F, 1, 1, F])
    cost = np.tile(fit, (5, 1))
    blocks = [m.T * cost for m in (mat_dd, mat_dw, mat_ww, mat_rw, mat_dr)]
    return np.vstack(blocks)  # (25, 5)


def renew(data: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """One discrete-generation reaction step (reference renew_function.m)."""
    gn, m, n = data.shape
    N = data.sum(axis=0)  # (m,n)
    a = data  # (gn,m,n)
    rcat = np.empty_like(data)
    for b in range(gn):
        block = mats[b * gn:(b + 1) * gn, :]  # (gn,gn)
        rcat[b] = np.einsum("imp,ij,jmp->mp", a, block, a)
    return rcat * LAMBDA / ((LAMBDA - 1) * N + 1) / N - data * N


def matlab_kernel(avd: float, size: int = KERNEL_SIZE) -> np.ndarray:
    """Reference get_mig_matrix25 kernel (no d<=25 mask)."""
    sigma = avd / np.sqrt(np.pi / 2)
    c = (size - 1) / 2.0
    ii, jj = np.indices((size, size), dtype=float)
    dx = ii - c
    dy = jj - c
    d2 = dx**2 + dy**2 - dx * dy
    k = np.exp(-d2 / (2 * sigma**2))
    return k / k.sum()


def hexgrid_embedding_check(size: int = KERNEL_SIZE) -> float:
    """Max |HexGrid Euclidean embedding distance^2 - kernel metric distance^2|.

    Confirms that natal HexGrid's pointy-top Cartesian embedding reproduces the
    oblique metric used by ``build_gaussian_kernel`` over the full offset grid.
    """
    c = size // 2
    grid = HexGrid(rows=size, cols=size, wrap=False)
    worst = 0.0
    for i in range(size):
        for j in range(size):
            x, y = grid.to_xy((i, j))
            x0, y0 = grid.to_xy((c, c))
            embed_sq = (x - x0) ** 2 + (y - y0) ** 2
            di, dj = i - c, j - c
            metric_sq = di**2 + dj**2 + di * dj
            worst = max(worst, abs(embed_sq - metric_sq))
    return worst


def natal_kernel(avd: float, size: int = KERNEL_SIZE) -> np.ndarray:
    """natal hex kernel, indexed by (dr, dc)."""
    return build_gaussian_kernel("hex", size=size, mean_dispersal=avd)


def natal_kernel_rowcol(avd: float, size: int = KERNEL_SIZE) -> np.ndarray:
    """natal hex kernel expressed in MATLAB's (row, col) frame.

    The coordinate equivalence established for M0 is
    ``(dr, dc) = (row_offset, -col_offset)``; mirroring the natal kernel along
    the column axis therefore reproduces the reference kernel (verified to
    1e-16).  Flat-side waves are invariant to this mirror; junction-side waves
    are not, which is why the mapping matters.
    """
    return build_gaussian_kernel("hex", size=size, mean_dispersal=avd)[:, ::-1]


def _diffuse(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Replicate-boundary diffusion, equivalent to MATLAB imfilter 'replicate'.

    Direct (spatial) convolution is used deliberately: it is a sum of
    non-negative products, so it preserves non-negativity exactly.  FFT-based
    convolution introduces tiny negative ringing that the logistic reaction term
    amplifies for narrow kernels (small dispersal), leading to NaNs.
    """
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = ndimage.convolve(data[i], kernel, mode="nearest")
    return out


def mround(x: float) -> int:
    """MATLAB ``round``: half away from zero (Python's round is half-to-even)."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def run_flat(m, n, cp1, cp2, avd, kernel, mats, max_iters=20000, capture=True):
    """Faithful port of the flat reference ``main.m``.

    ``cp1``/``cp2`` are the 1-based checkpoint columns from ``mode_params``.
    """
    x = np.zeros((GN, m, n))
    # borderx = 1 + round((n-1)*0.2) (1-based); cols 1..borderx -> [:borderx]
    border = 1 + mround((n - 1) * BOUNDARY1)
    x[0, :, :border] = 1.0
    x[2, :, border:] = 1.0
    mid0 = mround(m / 2) - 1  # MATLAB round(m/2) is 1-based
    return _run(x, cp1, cp2, avd, kernel, mats, max_iters, axis=2, mid=mid0, capture=capture)


def run_junction(L, avd, kernel, mats, max_iters=20000, capture=True):
    """Faithful port of the junction reference ``main.m`` (1-based)."""
    m = int(np.ceil(L * (1 + 1 / np.sqrt(3))))
    n = int(np.ceil(L * 2 / np.sqrt(3)))
    x = np.zeros((GN, m, n))
    for i in range(m):  # 0-based i -> MATLAB i+1
        for j in range(n):  # 0-based j -> MATLAB j+1
            if (j + 1) >= 2 * (i + 1) - 2 / 5 * L:
                x[0, i, j] = 1.0
            else:
                x[2, i, j] = 1.0
    cp1 = mround(L / (2 * np.sqrt(3)) + 0.45 * L)
    cp2 = mround(L / (2 * np.sqrt(3)) + 0.65 * L)
    mid0 = mround(n / 2) - 1
    return _run(x, cp1, cp2, avd, kernel, mats, max_iters, axis=1, mid=mid0, capture=capture)


def _profile(x, axis, mid):
    if axis == 2:
        num = x[CARRIER, mid, :].sum(axis=0)
        den = x[:, mid, :].sum(axis=0)
    else:
        num = x[CARRIER, :, mid].sum(axis=0)
        den = x[:, :, mid].sum(axis=0)
    return num / den


def _run(x, cp1, cp2, avd, kernel, mats, max_iters, axis, mid, capture=False):
    # cp1/cp2 are 1-based MATLAB indices; mid is already 0-based.
    distance = cp2 - cp1
    flag1 = flag2 = 0.0
    last = [np.nan, np.nan]
    for t in range(max_iters):
        x = x + renew(x, mats)
        x = _diffuse(x, kernel)

        def freq(cp):
            if axis == 2:
                col = x[:, mid, cp - 1]
            else:
                col = x[:, cp - 1, mid]
            return float(col[CARRIER].sum() / col.sum())

        s1 = freq(cp1)
        if flag1 == 0 and s1 > SPEED_CRITERION:
            flag1 = t - 1 + (SPEED_CRITERION - last[0]) / (s1 - last[0])
        last[0] = s1

        s2 = freq(cp2)
        if s2 > SPEED_CRITERION and flag2 == 0:
            flag2 = t - 1 + (SPEED_CRITERION - last[1]) / (s2 - last[1])
            res = {"speed": distance / (flag2 - flag1), "t": t + 1, "axis": axis}
            if capture:
                res["profile"] = _profile(x, axis, mid)
            return res
        last[1] = s2

        if t % 20 == 0 and float(x[CARRIER].sum() / x.sum()) < 0.05:
            return {"speed": 0.0, "t": t + 1, "axis": axis}
    return {"speed": float("nan"), "t": max_iters, "axis": axis}


@dataclass
class Case:
    label: str
    kind: str
    m: int = 0
    n: int = 0
    cp: tuple = ()
    L: int = 0


CASES = {
    "flat_paper": Case("flat_paper", "flat", m=300, n=300, cp=(150, 180)),
    "flat_code": Case("flat_code", "flat", m=200, n=200, cp=(80, 140)),
    "junction_paper": Case("junction_paper", "junction", L=300),
    "junction_code": Case("junction_code", "junction", L=600),
}
AVDS = [0.5, 1.0, 2.0]


def main(argv: list[str]) -> int:
    labels = argv[1:] or list(CASES)
    mats = homing_mats()

    # kernel cross-check
    kernel_check = {}
    for avd in AVDS:
        kn = natal_kernel(avd)
        km = matlab_kernel(avd)
        diff = float(np.abs(kn - km).max())
        diff_mirror = float(np.abs(kn - km[:, ::-1]).max())
        kernel_check[f"{avd}"] = {
            "max_abs_diff": diff,
            "max_abs_diff_col_mirror": diff_mirror,
            "equals_matlab_col_mirror": diff_mirror < 1e-12,
            "kernel_sum": float(kn.sum()),
        }

    results = []
    shapes = {}
    for label in labels:
        c = CASES[label]
        for avd in AVDS:
            kn = natal_kernel_rowcol(avd)
            if c.kind == "flat":
                r = run_flat(c.m, c.n, c.cp[0], c.cp[1], avd, kn, mats)
                info = {"m": c.m, "n": c.n, "checkpoint1": c.cp[0], "checkpoint2": c.cp[1]}
            else:
                r = run_junction(c.L, avd, kn, mats)
                m = int(np.ceil(c.L * (1 + 1 / np.sqrt(3))))
                n = int(np.ceil(c.L * 2 / np.sqrt(3)))
                info = {"L": c.L, "m": m, "n": n}
            row = {"label": label, "kind": c.kind, "avd": avd, "speed": r["speed"], **info}
            results.append(row)
            if "profile" in r:
                tag = f"{label}_{avd:g}"
                shapes[f"{tag}_x"] = np.arange(r["profile"].size)
                shapes[f"{tag}_y"] = r["profile"]
            print(f"{label:15s} avd={avd:<4.2f} speed={r['speed']:.6f} (iters={r['t']})")

    embed_worst = hexgrid_embedding_check()
    out = {"kernel_check": kernel_check, "avds": AVDS, "runs": results,
           "hexgrid_embedding_max_abs_err": embed_worst}
    print(f"HexGrid embedding vs kernel metric max|err| = {embed_worst:.2e}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "m2_natal_wavespeed.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(RESULTS / "m2_hex_waveshape.npz", **shapes)
    print("kernel max|natal - matlab_col_mirror|:",
          {k: f"{v['max_abs_diff_col_mirror']:.2e}" for k, v in kernel_check.items()})
    print("kernel equals matlab col-mirror:",
          {k: v["equals_matlab_col_mirror"] for k, v in kernel_check.items()})
    print("wrote results/m2_natal_wavespeed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
