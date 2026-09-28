#!/usr/bin/env python
"""Compare M2 natal-primitive wave speeds against the MATLAB reference golden.

Inputs:
  golden/hex_homing_wavespeed_m2.json   (MATLAB reference, M2 frozen settings)
  results/m2_natal_wavespeed.json       (natal HexGrid + build_gaussian_kernel)

Outputs:
  results/m2_wavespeed_compare.json

Also reports the flat->junction unit factor sqrt(3)/2 and the effective kernel
mean distance (which exposes the low-avd lattice discretisation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
GOLDEN = HEXROOT / "golden"
RESULTS = HEXROOT / "results"
F2 = np.sqrt(3.0) / 2.0
TOL = 0.05


def main() -> int:
    mat = json.load(open(GOLDEN / "hex_homing_wavespeed_m2.json"))["runs"]
    nat = json.load(open(RESULTS / "m2_natal_wavespeed.json"))
    natruns = {(r["label"], round(r["avd"], 6)): r for r in nat["runs"]}
    effective = {(round(r["avd"], 6)): r["kernel_mean_dist"] for r in mat}

    rows = []
    for r in mat:
        key = (r["label"], round(r["avd"], 6))
        n = natruns.get(key)
        ref = r["speed"]
        got = n["speed"] if n else float("nan")
        rel = (got - ref) / ref if ref else float("nan")
        rows.append({
            "label": r["label"], "kind": r["kind"], "avd": r["avd"],
            "kernel_mean_dist": r["kernel_mean_dist"],
            "matlab_speed": ref, "natal_speed": got, "rel_err": rel,
            "pass": bool(abs(rel) < TOL),
        })
        print(f"{r['label']:15s} avd={r['avd']:<4.2f} mean_d={r['kernel_mean_dist']:.4g} "
              f"matlab={ref:.6f} natal={got:.6f} rel={rel:+.3%} {'OK' if abs(rel) < TOL else 'FAIL'}")

    print("\nflat*sqrt(3)/2 vs junction (per convention):")
    byc = {}
    for x in rows:
        byc.setdefault(x["label"], {})[round(x["avd"], 6)] = x["matlab_speed"]
    direction = {}
    for conv in ("paper", "code"):
        fk, jk = f"flat_{conv}", f"junction_{conv}"
        if fk in byc and jk in byc:
            for avd in sorted(byc[fk]):
                ratio = byc[fk][avd] * F2 / byc[jk][avd]
                direction[f"{conv}_{avd}"] = ratio
                print(f"  {conv}: avd={avd:<4.2f} flat*F2/junction={ratio:.4f}")

    # waveform front width (0.1 -> 0.9) normalised by domain length
    def front_width(x, y):
        # carrier frequency is high behind the front and low ahead; measure the
        # descending 0.9 -> 0.1 edge using the last index above each threshold.
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        above10 = np.where(ys >= 0.1)[0]
        above90 = np.where(ys >= 0.9)[0]
        if above10.size == 0 or above90.size == 0:
            return float("nan"), float(xs[-1] - xs[0])
        return float(xs[above10[-1]] - xs[above90[-1]]), float(xs[-1] - xs[0])

    shapes = {}
    hex_npz = RESULTS / "m2_hex_waveshape.npz"
    pde_npz = GOLDEN / "pde_waveshape_reference.npz"
    hexshp = dict(np.load(hex_npz)) if hex_npz.exists() else {}
    pdeshp = dict(np.load(pde_npz))
    print("\nwaveform front width (0.1->0.9), normalised by domain:")
    for tag, arr in sorted(hexshp.items()):
        if not tag.endswith("_y"):
            continue
        base = tag[:-2]
        w, span = front_width(hexshp[base + "_x"], arr)
        shapes[base] = {"front_width": w, "span": span, "front_width_frac": w / span}
        print(f"  hex {base:22s} width/span={w / span:.4f}")
    for tag in ("homing_0.5", "homing_10"):
        base = tag
        w, span = front_width(pdeshp[base + "_xlist"], pdeshp[base + "_ylist"])
        shapes["pde_" + base] = {"front_width": w, "span": span, "front_width_frac": w / span}
        print(f"  pde {base:22s} width/span={w / span:.4f}")

    out = {
        "tol": TOL,
        "flat_to_1d_factor": F2,
        "effective_kernel_mean": effective,
        "runs": rows,
        "direction_ratio": direction,
        "kernel_check": nat["kernel_check"],
        "waveform_front_width": shapes,
    }
    with open(RESULTS / "m2_wavespeed_compare.json", "w") as fh:
        json.dump(out, fh, indent=2)
    n_pass = sum(r["pass"] for r in rows)
    print(f"\n{n_pass}/{len(rows)} within {TOL:.0%}; wrote results/m2_wavespeed_compare.json")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
