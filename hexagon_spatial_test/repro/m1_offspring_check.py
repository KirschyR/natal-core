#!/usr/bin/env python
"""M1: verify natal's offspring tensor against the paper's ``mats`` operator.

The paper's homing ``drive_generator.m`` folds the drive into a single
``mats (gn^2, gn)`` operator: offspring ``k`` from female genotype ``i`` and
male genotype ``j`` has rate ``mats[k*gn + i, j]``.  natal compiles the same
physics into its ``offspring_tensor`` ``P[mother, father, offspring]`` (drive
conversion is baked into the meiosis/fusion tables), so the two should agree
on the paper's 5 genotypes (dd, dw, ww, rw, dr).

Genotype mapping (paper -> natal ztype index for alleles WT/Drive/Res):
``dd->Drive|Drive, dw->WT|Drive, ww->WT|WT, rw->WT|Res, dr->Drive|Res``.

Usage: python repro/m1_offspring_check.py
Outputs ``results/m1_offspring_check.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import natal as nt
from natal.frontend.presets.homing import HomingDrive

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
RESULTS = HEXROOT / "results"

PAPER_ORDER = ["dd", "dw", "ww", "rw", "dr"]
# natal ztype order for alleles [WT, Drive, Res] is:
#   0 WT|WT, 1 WT|Drive, 2 WT|Res, 3 Drive|Drive, 4 Drive|Res, 5 Res|Res
NAT_INDEX = {"ww": 0, "dw": 1, "rw": 2, "dd": 3, "dr": 4}


def paper_homing_mats(gcr: float, dc: float, herr: float, ddfitness: float) -> np.ndarray:
    """Reproduce the homing ``drive_generator.m`` (row-major as in MATLAB)."""
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
    # MATLAB: matrices are transposed, then scaled by the (female) fitness cost.
    F = np.sqrt(ddfitness)
    fit = np.array([F**2, F, 1, 1, F])
    cost = np.tile(fit, (5, 1))
    blocks = [m.T * cost for m in (mat_dd, mat_dw, mat_ww, mat_rw, mat_dr)]
    return np.vstack(blocks)  # (25, 5): mats[k*5 + i, j]


def natal_offspring_tensor(dc: float) -> np.ndarray:
    species = nt.Species.from_dict(
        f"M1Homing_{dc}", {"Chr1": {"L1": ["WT", "Drive", "Res"]}}
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
        .setup(name=f"m1_{dc}")
        .initial_state({"female": {"WT|WT": 20}, "male": {"WT|WT": 20}})
        .presets(
            HomingDrive(
                name=f"hd1_{dc}",
                drive_allele="Drive",
                target_allele="WT",
                resistance_allele="Res",
                drive_conversion_rate=dc,
                late_germline_resistance_formation_rate=0.0,
            )
        )
        .build()
    )
    return np.asarray(pop.config.offspring_tensor, dtype=np.float64)


def compare(dc: float, gcr: float = 0.0, herr: float = 0.0, ddfitness: float = 1.0):
    mats = paper_homing_mats(gcr, dc, herr, ddfitness)  # (25,5)
    P = natal_offspring_tensor(dc)  # (6,6,6): P[mother, father, offspring]
    # natal sub-tensor on the paper's 5 genotypes.
    idx = [NAT_INDEX[g] for g in PAPER_ORDER]
    S = P[np.ix_(idx, idx, idx)]  # S[pi,pj,pk]
    # paper M[k,i,j] = mats[k*5+i, j]; compare S[i,j,k] to M[k,i,j].
    ref = np.transpose(mats.reshape(5, 5, 5), (1, 2, 0))  # ref[i,j,k]
    diff = np.abs(S - ref)
    max_abs = float(diff.max())
    scale = float(np.abs(ref).max())
    # The paper subset omits Res|Res, so its row sums need not be 1; check the
    # *full* natal tensor instead.
    full_row_sums_ok = bool(np.allclose(P.sum(axis=2), 1.0, atol=1e-9))
    return {
        "dc": dc,
        "gcr": gcr,
        "herr": herr,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_abs / scale if scale else max_abs,
        "natal_full_row_sums_ok": full_row_sums_ok,
        "paper_col_sums_are_1": bool(np.allclose(mats.sum(axis=0), 1.0, atol=1e-9)),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"paper_order": PAPER_ORDER, "natal_index": NAT_INDEX, "cases": []}
    for dc in [1.0, 0.9, 0.5]:
        r = compare(dc)
        out["cases"].append(r)
        print(
            f"dc={dc:<4} max_abs={r['max_abs_diff']:.3e} max_rel={r['max_rel_diff']:.3e} "
            f"natal_full_rowsum_ok={r['natal_full_row_sums_ok']}"
        )
    with open(RESULTS / "m1_offspring_check.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote results/m1_offspring_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
