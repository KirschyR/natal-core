#!/usr/bin/env python
"""Build the M0 Fig-3 / Fig-S5 comparison tables from the tracked goldens.

Reads ``golden/hex_homing_wavespeed_smallcase.json`` (small-scale hex runs
produced by ``gen_golden_hex_wavespeed.m``) and the pre-computed PDE reference
``golden/pde_wavespeed_reference.npz`` / ``pde_waveshape_reference.npz``
(extracted by ``gen_golden_pde_reference.py``), and writes comparison tables to
``results/``.

The published ``wavespeed_analyse.m`` normalises flat-side hex speeds by
``sqrt(3)/2``; the same factor is applied here so hex and PDE speeds share the
1-D-equivalent unit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
GOLDEN = HEXROOT / "golden"
RESULTS = HEXROOT / "results"
FLAT_TO_1D = np.sqrt(3.0) / 2.0
DRIVES = ("homing", "tare", "tade11", "cifab")


def _pde_speed(npz: dict, drive: str, avd: float) -> float:
    xs = npz[f"{drive}_avdlist"]
    ys = npz[f"{drive}_speed"]
    return float(np.interp(avd, xs, ys))


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    hexdat = json.load(open(GOLDEN / "hex_homing_wavespeed_smallcase.json"))
    spd = dict(np.load(GOLDEN / "pde_wavespeed_reference.npz"))
    shp = dict(np.load(GOLDEN / "pde_waveshape_reference.npz"))

    fig3 = {"flat_to_1d_factor": FLAT_TO_1D, "drives": {}, "hex_small_case": {}}
    for drive in DRIVES:
        xs = spd[f"{drive}_avdlist"]
        ys = spd[f"{drive}_speed"]
        fig3["drives"][drive] = {
            "avd": xs.tolist(),
            "pde_speed": ys.tolist(),
            "pde_speed_endpoints": [float(ys[0]), float(ys[-1])],
        }

    # hex small-case points vs PDE at the same dispersal values
    for direction in ("flat", "junction"):
        rows = []
        for r in hexdat[direction]:
            avd = r["avd"]
            pde_homing = _pde_speed(spd, "homing", avd)
            raw = r["speed"]
            scaled = raw * FLAT_TO_1D if direction == "flat" else raw
            rows.append(
                {
                    "avd": avd,
                    "hex_raw_speed": raw,
                    "hex_1d_equivalent": scaled,
                    "pde_homing_speed": pde_homing,
                    "rel_err_raw_vs_pde": (raw - pde_homing) / pde_homing,
                    "rel_err_1d_vs_pde": (scaled - pde_homing) / pde_homing,
                }
            )
        fig3["hex_small_case"][direction] = rows

    # Fig S5: summarize the reference PDE waveform profiles
    figS5 = {"profiles": {}}
    for drive in DRIVES:
        for avd in (0.5, 10):
            tag = f"{drive}_{avd:g}"
            if f"{tag}_xlist" not in shp:
                continue
            x = shp[f"{tag}_xlist"]
            y = shp[f"{tag}_ylist"]
            above = np.where(y >= 0.5)[0]
            figS5["profiles"][tag] = {
                "n_points": int(x.size),
                "x_range": [float(x[0]), float(x[-1])],
                "front_x_50pct": float(x[above[-1]]) if above.size else None,
                "ymax": float(y.max()),
            }

    with open(RESULTS / "m0_fig3_wavespeed_compare.json", "w") as fh:
        json.dump(fig3, fh, indent=2, sort_keys=True)
    with open(RESULTS / "m0_figS5_waveshape_summary.json", "w") as fh:
        json.dump(figS5, fh, indent=2, sort_keys=True)

    print("Fig 3 (PDE wave speed, first/last of avd grid):")
    for drive in DRIVES:
        e = fig3["drives"][drive]["pde_speed_endpoints"]
        print(f"  {drive:7s} speed(avd=0.1)={e[0]:.4f}  speed(avd=2.0)={e[1]:.4f}")
    print("\nHoming small-case hex vs PDE:")
    for direction in ("flat", "junction"):
        for row in fig3["hex_small_case"][direction]:
            print(f"  {direction:8s} avd={row['avd']:<4.2f} hex_raw={row['hex_raw_speed']:.4f} "
                  f"hex_1d={row['hex_1d_equivalent']:.4f} pde={row['pde_homing_speed']:.4f} "
                  f"rel_err_1d={row['rel_err_1d_vs_pde']:+.2%}")
    print("\nwrote results/m0_fig3_wavespeed_compare.json")
    print("wrote results/m0_figS5_waveshape_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
