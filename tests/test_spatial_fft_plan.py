"""Route-A FFT migration plan: boundary normalization matches the CSR fold.

``build_fft_migration_plan`` is the frontend building block for the optional
GPU cuFFT migration path (M4 route A).  These tests pin it to the existing CSR
semantics: the normalization field ``z`` must equal the per-row denominator the
CSR fold divides out, and ``kernel'/z`` must reproduce the CSR weights, for
both rectangular boundaries (``wrap=False``) and periodic ones (``wrap=True``).
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.frontend.spatial.migration import (
    build_fft_migration_plan,
    fold_migration_csr,
)
from natal.frontend.spatial.topology import HexGrid, build_gaussian_kernel

CASES = [
    (5, 7, 3, 1.1),
    (6, 6, 5, 1.5),
    (9, 4, 5, 0.9),
    (8, 11, 11, 1.7),
]


def _brute_force_z(rows: int, cols: int, kp: np.ndarray, wrap: bool) -> np.ndarray:
    """Reference z via per-deme ``normalize_coord`` (independent of the plan)."""
    topo = HexGrid(rows=rows, cols=cols, wrap=wrap)
    k = kp.shape[0]
    center = k // 2
    z = np.zeros(rows * cols, dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            total = 0.0
            for kr in range(k):
                for kc in range(k):
                    w = kp[kr, kc]
                    if w <= 0.0:
                        continue
                    if topo.normalize_coord(r + kr - center, c + kc - center) is not None:
                        total += w
            z[r * cols + c] = total
    return z


@pytest.mark.parametrize("rows,cols,k,sigma", CASES)
@pytest.mark.parametrize("wrap", [False, True])
def test_fft_plan_z_matches_brute_force(rows, cols, k, sigma, wrap):
    kernel = build_gaussian_kernel(HexGrid, size=k, sigma=sigma)
    plan = build_fft_migration_plan(HexGrid(rows=rows, cols=cols, wrap=wrap), kernel)
    expected = _brute_force_z(rows, cols, plan.kernel, wrap)
    assert plan.rows == rows and plan.cols == cols
    assert plan.z.shape == (rows, cols)
    np.testing.assert_allclose(plan.z.reshape(-1), expected, rtol=0.0, atol=1e-14)
    if wrap:
        assert np.allclose(plan.z, plan.z.flat[0])


@pytest.mark.parametrize("rows,cols,k,sigma", CASES)
def test_fft_plan_kernel_center_removed(rows, cols, k, sigma):
    kernel = build_gaussian_kernel(HexGrid, size=k, sigma=sigma)
    plan = build_fft_migration_plan(HexGrid(rows=rows, cols=cols, wrap=False), kernel)
    assert plan.kernel.shape == (k, k)
    assert plan.kernel[k // 2, k // 2] == 0.0
    kernel[k // 2, k // 2] = 0.0
    np.testing.assert_array_equal(plan.kernel, kernel)


@pytest.mark.parametrize("rows,cols,k,sigma", CASES)
@pytest.mark.parametrize("wrap", [False, True])
def test_fft_plan_weights_match_csr(rows, cols, k, sigma, wrap):
    topo = HexGrid(rows=rows, cols=cols, wrap=wrap)
    kernel = build_gaussian_kernel(HexGrid, size=k, sigma=sigma)
    csr = fold_migration_csr(
        n_demes=rows * cols,
        topology=topo,
        adjacency_dense=np.zeros((1, 1)),
        migration_kernel=kernel,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=False,
        adjust_on_edge=False,
        mode="kernel",
    )
    plan = build_fft_migration_plan(topo, kernel)
    center = k // 2
    for s in range(rows * cols):
        r, c = divmod(s, cols)
        start, end = int(csr.indptr[s]), int(csr.indptr[s + 1])
        dest = csr.dest_idx[start:end]
        weights = csr.weights[start:end]
        expected_dest = []
        expected_w = []
        for kr in range(k):
            for kc in range(k):
                if kr == center and kc == center:
                    continue
                mapped = topo.normalize_coord(r + kr - center, c + kc - center)
                if mapped is None:
                    continue
                expected_dest.append(topo.to_index(mapped))
                expected_w.append(plan.kernel[kr, kc] / plan.z[r, c])
        np.testing.assert_array_equal(dest, np.array(expected_dest, dtype=np.int64))
        np.testing.assert_allclose(weights, np.array(expected_w), rtol=1e-12, atol=1e-15)


def test_fft_plan_rejects_even_kernel():
    with pytest.raises(ValueError):
        build_fft_migration_plan(
            HexGrid(rows=4, cols=4, wrap=False), np.ones((4, 4))
        )
