#!/usr/bin/env python
"""M3 CPU-side scale probe for spatial migration CSR folding.

Measures, for kernel sizes {3,5,11,21,51} x domains {30^2,60^2,120^2,300^2}:

* kernel support (non-zero entries) and its build time;
* CSR fold time, nnz and CSR byte size (``fold_migration_csr``, Python);
* total population build time and deterministic CPU tick time where the CSR
  fits a memory budget (larger CSRs are reported as "exceeds budget" rather
  than silently skipped).

Also records the realistic paper kernel (size 51, ``mean_dispersal = avd``)
support/nnz to connect the square-kernel worst case to the paper's r25 kernel.

Outputs ``results/m3_scale_data.json``.
"""

from __future__ import annotations

import gc
import json
import resource
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

DOMAINS = [30, 60, 120, 300]
KERNELS = [3, 5, 11, 21, 51]
SIGMA = 1.5
CSR_BUDGET_BYTES = 1_200_000_000  # build+tick only when CSR fits this
PAPER_AVDS = [0.5, 1.0, 2.0]


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def fold(rows: int, cols: int, kernel: np.ndarray):
    n = rows * cols
    topo = HexGrid(rows=rows, cols=cols, wrap=False)
    t0 = time.perf_counter()
    csr = fold_migration_csr(
        n_demes=n,
        topology=topo,
        adjacency_dense=np.zeros((1, 1)),
        migration_kernel=kernel,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=True,
        adjust_on_edge=False,
        mode="kernel",
    )
    dt = time.perf_counter() - t0
    nbytes = int(csr.indptr.nbytes + csr.dest_idx.nbytes + csr.weights.nbytes)
    return dt, int(csr.dest_idx.size), nbytes


def build_population(size: int, kernel: np.ndarray) -> SpatialPopulation:
    species = nt.Species.from_dict(
        name="M3ProbeSpecies", structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    return (
        SpatialPopulation.builder(
            species,
            n_demes=size * size,
            topology=HexGrid(rows=size, cols=size, wrap=False),
            pop_type="discrete_generation",
        )
        .setup(name="hex_deme", stochastic=True)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 500.0, "Dr|WT": 0.0},
                "male": {"WT|WT": 0.0, "Dr|WT": 500.0},
            }
        )
        .reproduction(eggs_per_female=50.0)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=1000,
            low_density_growth_rate=6,
        )
        .migration(kernel=kernel, migration_rate=0.5)
        .build()
    )


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"{'domain':>7} {'k':>3} {'support':>8} {'nnz':>14} {'nnz/deme':>9} "
          f"{'csr_GB':>8} {'fold_s':>8} {'build_s':>8} {'tick_s':>8}")
    for size in DOMAINS:
        for k in KERNELS:
            t0 = time.perf_counter()
            kernel = build_gaussian_kernel(HexGrid, size=k, sigma=SIGMA)
            kernel_s = time.perf_counter() - t0
            support = int((kernel > 0.0).sum())
            fold_s, nnz, nbytes = fold(size, size, kernel)

            build_s = tick_s = None
            note = ""
            if nbytes <= CSR_BUDGET_BYTES:
                gc.collect()
                t0 = time.perf_counter()
                sp = build_population(size, kernel)
                build_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                sp.run(3, record_every=0)
                tick_s = (time.perf_counter() - t0) / 3.0
                del sp
                gc.collect()
            else:
                note = "exceeds budget: population build/tick skipped"

            row = {
                "domain": size, "kernel": k, "sigma": SIGMA,
                "support": support, "nnz": nnz, "nnz_per_deme": nnz / (size * size),
                "csr_bytes": nbytes, "csr_gb": nbytes / 1e9,
                "kernel_build_s": kernel_s, "fold_s": fold_s,
                "build_s": build_s, "tick_s": tick_s, "note": note,
            }
            rows.append(row)
            b = "-" if build_s is None else f"{build_s:.2f}"
            t = "-" if tick_s is None else f"{tick_s:.3f}"
            print(f"{size:>7} {k:>3} {support:>8} {nnz:>14} {nnz / (size * size):>9.1f} "
                  f"{nbytes / 1e9:>8.3f} {fold_s:>8.2f} {b:>8} {t:>8}{'  ' + note if note else ''}")

    print("\nrealistic paper kernel (size 51, mean_dispersal=avd) at 300x300:")
    paper = []
    for avd in PAPER_AVDS:
        kernel = build_gaussian_kernel(HexGrid, size=51, mean_dispersal=avd)
        support = int((kernel > 0.0).sum())
        fold_s, nnz, nbytes = fold(300, 300, kernel)
        paper.append({"avd": avd, "sigma": avd / np.sqrt(np.pi / 2), "support": support,
                      "nnz": nnz, "csr_bytes": nbytes, "csr_gb": nbytes / 1e9,
                      "fold_s": fold_s})
        print(f"  avd={avd:<4.1f} sigma={avd / np.sqrt(np.pi / 2):.4f} support={support:<5d} "
              f"nnz={nnz:<12d} csr={nbytes / 1e9:.3f}GB fold={fold_s:.2f}s")

    out = {
        "generator": "hexagon_spatial_test/repro/m3_scale_probe.py",
        "sigma_sweep": SIGMA,
        "csr_budget_bytes": CSR_BUDGET_BYTES,
        "peak_rss_gb": peak_rss_gb(),
        "sweep": rows,
        "paper_kernel_300": paper,
    }
    with open(RESULTS / "m3_scale_data.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\npeak RSS: {peak_rss_gb():.2f} GB")
    print("wrote results/m3_scale_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
