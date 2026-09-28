#!/usr/bin/env python
"""Self-tests for the M0 goldens (no natal-core code involved).

Hard invariants (fail the script):
* the vendored repro/ref files still match the SHA-256 recorded in the golden;
* the golden covers exactly the documented case grid (6 hex rows, 4x2 PDE
  speed arrays and 4x2 PDE waveform profiles) -- no silently missing series;
* hex small-case speeds are finite and strictly increasing with dispersal;
* extracted PDE speed arrays are finite, positive and monotone;
* pre-computed PDE waveform profiles are finite and within [0, 1].

Report-only observation:
* the two hex directions (flat vs junction, after the sqrt(3)/2 factor used by
  the published ``wavespeed_analyse.m``) agree only approximately on the small
  domains used here; the ratios are printed for the M2 convergence work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
GOLDEN = HEXROOT / "golden"
REF = HERE / "ref"
F2 = np.sqrt(3.0) / 2.0
DRIVES = ("homing", "tare", "tade11", "cifab")
SHAPE_AVDS = (0.5, 10.0)
HEX_AVDS = (0.25, 0.5, 1.0)
REF_FILES = {
    "flat": ("main.m", "get_mig_matrix25.m",
             "homing/drive_generator.m", "homing/renew_function.m"),
    "junction": ("main.m", "get_mig_matrix25.m",
                 "homing/drive_generator.m", "homing/renew_function.m"),
}


def _ref_key(direction: str, rel: str) -> str:
    return f"{direction}_{rel.replace('/', '_').replace('.', '_')}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    hexdat = json.load(open(GOLDEN / "hex_homing_wavespeed_smallcase.json"))
    failures = []

    print("0) vendored repro/ref SHA-256 vs golden reference_files")
    for direction, files in REF_FILES.items():
        for rel in files:
            expected = hexdat["reference_files"][_ref_key(direction, rel)]
            actual = _sha256(REF / direction / rel)
            ok = actual == expected
            print(f"   {direction:8s} {rel:28s} {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"reference hash mismatch: {direction}/{rel}")

    print("0b) golden covers the documented case grid")
    got_hex = {(r["direction"], round(float(r["avd"]), 6)) for r in
               hexdat["flat"] + hexdat["junction"]}
    want_hex = {(d, round(a, 6)) for d in ("flat", "junction") for a in HEX_AVDS}
    if got_hex != want_hex:
        failures.append(f"hex case grid mismatch: got {sorted(got_hex)}")
    print(f"   hex rows: got {len(got_hex)} expected {len(want_hex)} "
          f"{'OK' if got_hex == want_hex else 'FAIL'}")

    spd_keys = set(np.load(GOLDEN / "pde_wavespeed_reference.npz").files)
    for drive in DRIVES:
        need = {f"{drive}_avdlist", f"{drive}_speed"}
        if not need <= spd_keys:
            failures.append(f"missing PDE speed arrays for {drive}: {need - spd_keys}")
    print(f"   PDE speed arrays: {len(spd_keys)} keys "
          f"{'OK' if not failures else 'FAIL'}")

    shp = dict(np.load(GOLDEN / "pde_waveshape_reference.npz"))
    got_shape = {k[: -len("_ylist")] for k in shp if k.endswith("_ylist")}
    want_shape = {f"{d}_{a:g}" for d in DRIVES for a in SHAPE_AVDS}
    if got_shape != want_shape:
        failures.append(f"waveform profile set mismatch: missing {want_shape - got_shape}, "
                        f"extra {got_shape - want_shape}")
    print(f"   PDE waveform profiles: got {len(got_shape)} expected {len(want_shape)} "
          f"{'OK' if got_shape == want_shape else 'FAIL'}")

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
