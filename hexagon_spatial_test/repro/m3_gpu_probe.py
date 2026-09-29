#!/usr/bin/env python
"""M3 GPU-side probe: spatial CUDA enable, memory budget and row-width guards.

Exercises the optional spatial CUDA path via the existing session API
(``pop._rust_spatial_backend.enable_gpu()`` / ``gpu_status()`` / ``run_tick()``):

1. deterministic models: ``enable_gpu`` success and the migration-cache footprint
   (computed with the same formula as the Rust ``migration_cache_bytes``);
2. stochastic models: the ``MAX_CSR_ROW = 32`` device scratch guard (k=5 passes,
   wider kernel rows are rejected at the first migration);
3. the enable-time memory budget guard: a deliberately over-budget CSR must raise
   an explicit error (never silently fall back to CPU).

Requires the GPU extension build and ``LD_LIBRARY_PATH`` to include the CUDA
runtime/NVRTC libraries. Outputs ``results/m3_gpu_data.json``.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import natal as nt
from natal.frontend.spatial.migration import fold_migration_csr
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import HexGrid, build_gaussian_kernel

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
RESULTS = HEXROOT / "results"
SIGMA = 1.5
MAX_CSR_ROW = 32


def free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return int(out)


def migration_cache_bytes(n_batch: int, n_ages: int, n_ztypes: int, nnz: int) -> int:
    """Python mirror of the Rust ``migration_cache_bytes`` (executor.rs)."""
    total = 2 * (n_batch + 1) * 4            # indptr + rev_indptr (i32)
    total += nnz * 4 * 5                     # rev_src/dest/rev_entry (i32) + rev_weight/weights (f32)
    total += n_batch * 4                     # row_sum (f32)
    total += nnz * n_ages * n_ztypes * 4 * 2  # fwd_f + fwd_m (f32)
    total += nnz * n_ages * n_ztypes * n_ztypes * 4  # fwd_s (f32)
    return total


def csr_nnz(size: int, k: int) -> int:
    K = build_gaussian_kernel(HexGrid, size=k, sigma=SIGMA)
    _, nnz, _ = _fold(size, K)
    return nnz


def _fold(size: int, kernel: np.ndarray):
    csr = fold_migration_csr(
        n_demes=size * size, topology=HexGrid(rows=size, cols=size, wrap=False),
        adjacency_dense=np.zeros((1, 1)), migration_kernel=kernel,
        kernel_bank=None, deme_kernel_ids=None,
        kernel_include_center=False, adjust_on_edge=False, mode="kernel",
    )
    nbytes = int(csr.indptr.nbytes + csr.dest_idx.nbytes + csr.weights.nbytes)
    return 0.0, int(csr.dest_idx.size), nbytes


def build(size: int, k: int, stochastic: bool, n_alleles: int = 2) -> SpatialPopulation:
    locs = [chr(ord("A") + i) for i in range(n_alleles)]
    name = f"M3Gpu_{size}_{k}_{int(stochastic)}_{n_alleles}"
    return (
        SpatialPopulation.builder(
            nt.Species.from_dict(name=name, structure={"chr1": {"loc": locs}}),
            n_demes=size * size,
            topology=HexGrid(rows=size, cols=size, wrap=False),
            pop_type="discrete_generation",
        )
        .setup(name="d", stochastic=stochastic)
        .initial_state(
            individual_count={
                "female": {f"{locs[0]}|{locs[0]}": 500.0},
                "male": {f"{locs[0]}|{locs[1]}": 500.0},
            }
        )
        .reproduction(eggs_per_female=50.0)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=1000,
            low_density_growth_rate=6,
        )
        .migration(kernel=build_gaussian_kernel(HexGrid, size=k, sigma=SIGMA), migration_rate=0.5)
        .build()
    )


def dims(pop: SpatialPopulation) -> tuple[int, int]:
    shape = np.asarray(pop.deme(0).state.individual_count).shape
    return int(shape[1]), int(shape[2])  # (n_ages, n_ztypes)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict = {"free_mib_before": free_mib(), "max_csr_row": MAX_CSR_ROW,
                 "deterministic": [], "stochastic": [], "budget": None}

    print("1) deterministic enable (status before, success, cache footprint)")
    for size, k in [(30, 5), (60, 11), (120, 21), (300, 11), (300, 21)]:
        pop = build(size, k, stochastic=False)
        n_ages, n_z = dims(pop)
        nnz = csr_nnz(size, k)
        required = migration_cache_bytes(size * size, n_ages, n_z, nnz)
        backend = pop._rust_spatial_backend  # noqa: SLF001 - existing session API
        before = backend.gpu_status()
        t0 = time.perf_counter()
        err = None
        gpu_tick = None
        try:
            backend.enable_gpu()
            for _ in range(3):
                backend.run_tick()
            t0 = time.perf_counter()
            for _ in range(7):
                backend.run_tick()
            gpu_tick = (time.perf_counter() - t0) / 7.0
            status = backend.gpu_status()
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
        row = {"size": size, "k": k, "n_ages": n_ages, "n_ztypes": n_z,
               "nnz": nnz, "cache_MiB": required / 2**20, "status_before": before,
               "status_after": status, "gpu_tick_s": gpu_tick, "error": err}
        out["deterministic"].append(row)
        print(f"   {size}^2 k={k:<2} Z={n_z} nnz={nnz:<10} cache={required/2**20:8.1f}MiB "
              f"{before}->{status} tick={'-' if gpu_tick is None else f'{gpu_tick:.4f}s'}"
              + (f"  ERR {err[:80]}" if err else ""))
        del pop, backend
        gc.collect()

    print("2) stochastic row-width guard (MAX_CSR_ROW=32)")
    for size, k, expect_ok in [(30, 5, True), (30, 11, False), (60, 21, False)]:
        pop = build(size, k, stochastic=True)
        backend = pop._rust_spatial_backend  # noqa: SLF001
        err = None
        try:
            backend.enable_gpu()
            backend.run_tick()
            result = "tick_ok"
        except Exception as exc:  # noqa: BLE001
            result = "rejected"
            err = f"{type(exc).__name__}: {exc}"
        max_row = max(int(b - a) for a, b in zip(_indptr(size, k)[:-1], _indptr(size, k)[1:]))
        row = {"size": size, "k": k, "max_csr_row": max_row, "result": result,
               "expected_ok": expect_ok, "error": err}
        out["stochastic"].append(row)
        print(f"   {size}^2 k={k:<2} max_row={max_row:<4} -> {result}"
              + (f"  {err[:90]}" if err else ""))
        del pop, backend
        gc.collect()

    print("3) enable-time memory budget guard (over-budget CSR must raise explicitly)")
    pop = build(200, 11, stochastic=False, n_alleles=7)  # Z=28 inflates the cache
    n_ages, n_z = dims(pop)
    nnz = csr_nnz(200, 11)
    required = migration_cache_bytes(200 * 200, n_ages, n_z, nnz)
    backend = pop._rust_spatial_backend  # noqa: SLF001
    try:
        backend.enable_gpu()
        out["budget"] = {"raised": False, "cache_MiB": required / 2**20,
                         "n_ztypes": n_z, "nnz": nnz}
        print("   ERROR: enable_gpu unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        out["budget"] = {"raised": True, "cache_MiB": required / 2**20, "n_ztypes": n_z,
                         "nnz": nnz, "error": f"{type(exc).__name__}: {exc}"}
        print(f"   raised: {type(exc).__name__}: {str(exc)[:150]}")
    del pop, backend
    gc.collect()

    out["free_mib_after"] = free_mib()
    with open(RESULTS / "m3_gpu_data.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nfree GPU MiB: {out['free_mib_before']} -> {out['free_mib_after']}")
    print("wrote results/m3_gpu_data.json")
    return 0


def _indptr(size: int, k: int) -> np.ndarray:
    csr = fold_migration_csr(
        n_demes=size * size, topology=HexGrid(rows=size, cols=size, wrap=False),
        adjacency_dense=np.zeros((1, 1)),
        migration_kernel=build_gaussian_kernel(HexGrid, size=k, sigma=SIGMA),
        kernel_bank=None, deme_kernel_ids=None,
        kernel_include_center=False, adjust_on_edge=False, mode="kernel",
    )
    return csr.indptr


if __name__ == "__main__":
    sys.exit(main())
