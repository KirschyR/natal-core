#!/usr/bin/env python
"""Self-tests for the M0 goldens (no natal-core code involved).

Hard invariants (fail the script):
* hex small-case speeds are finite and strictly increasing with dispersal;
* extracted PDE speed arrays are finite, positive and monotone;
* pre-computed PDE waveform profiles are finite and within [0, 1].

Report-only observation:
* the two hex directions (flat vs junction, after the sqrt(3)/2 factor used by
  the published ``wavespeed_analyse.m``) agree only approximately on the small
  domains used here; the ratios are printed for the M2 convergence work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
GOLDEN = HEXROOT / "golden"
F2 = np.sqrt(3.0) / 2.0
DRIVES = ("homing", "tare", "tade11", "cifab")


def main() -> int:
    hexdat = json.load(open(GOLDEN / "hex_homing_wavespeed_smallcase.json"))
    failures = []

    print("1) hex small-case speeds (finite, positive, increasing)")
    for direction in ("flat", "junction"):
        rows = hexdat[direction]
        avds = [r["avd"] for r in rows]
        speeds = [r["speed"] for r in rows]
        ok = (all(np.isfinite(s) and s > 0 for s in speeds)
              and all(np.diff(speeds) > 0))
        print(f"   {direction:8s} speeds={['%.6f' % s for s in speeds]} "
              f"avd={avds} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{direction}: non-monotone/non-finite speed")

    print("2) PDE speed arrays (finite, positive, monotone)")
    spd = dict(np.load(GOLDEN / "pde_wavespeed_reference.npz"))
    for drive in DRIVES:
        ys = spd[f"{drive}_speed"]
        ok = np.all(np.isfinite(ys)) and np.all(ys > 0) and np.all(np.diff(ys) > 0)
        print(f"   {drive:7s} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"PDE speed array invalid for {drive}")

    print("3) PDE waveform profiles (finite, within [0,1])")
    shp = dict(np.load(GOLDEN / "pde_waveshape_reference.npz"))
    for key, y in shp.items():
        if not key.endswith("_ylist"):
            continue
        ok = np.all(np.isfinite(y)) and y.min() >= -1e-9 and y.max() <= 1 + 1e-6
        print(f"   {key:22s} range=[{y.min():.4f},{y.max():.4f}] {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"waveform out of range: {key}")

    print("4) direction cross-check (report only): flat*sqrt(3)/2 / junction")
    flat = {r["avd"]: r["speed"] for r in hexdat["flat"]}
    junc = {r["avd"]: r["speed"] for r in hexdat["junction"]}
    for avd in sorted(flat):
        ratio = flat[avd] * F2 / junc[avd]
        print(f"   avd={avd:<4.2f} ratio={ratio:.4f}")

    if failures:
        print("\nFAILURES:\n  " + "\n  ".join(failures))
        return 1
    print("\nALL HARD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
