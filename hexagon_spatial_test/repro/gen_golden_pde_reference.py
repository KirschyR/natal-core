#!/usr/bin/env python
"""Extract the published PDE reference results embedded in the Hex-model ZIP.

The repository ships ``parameter_sensitive/hex code上交版.zip``.  Inside
``wave justification/pde/<drive>/`` the authors stored the pre-computed
reaction-diffusion (1-D PDE) outputs used for Figures 3 and S5:

* ``<drive>_pde_wavespeed.mat``     -- 20 dispersal values and the fitted PDE
  wave speed for each (HDF5 for the shapes, MAT v5 for the speeds).
* ``<drive>_pde_waveshape0.5.mat``  -- carrier-frequency profile at avd = 0.5.
* ``<drive>_pde_waveshape10.mat``   -- carrier-frequency profile at avd = 10.

This script converts them into portable ``.npz`` goldens under
``hexagon_spatial_test/golden/`` and writes a manifest with source hashes.

Usage::

    python repro/gen_golden_pde_reference.py [path-to-zip-or-extracted-dir]

Defaults to ``../Hex-model-main/parameter_sensitive/hex code上交版.zip``
relative to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

DRIVES = ("homing", "tare", "tade11", "cifab")
SHAPE_AVDS = (0.5, 10)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_hdf5_mat(path: Path) -> dict:
    import h5py

    out = {}
    with h5py.File(path, "r") as h:
        for key in ("xlist", "ylist", "avd", "D", "n", "checkpoint1", "checkpoint2", "ret"):
            if key in h:
                out[key] = np.asarray(h[key][()]).ravel()
    return out


def _load_speed_mat(path: Path) -> dict:
    d = sio.loadmat(path)
    out = {}
    for key in ("avdlist", "retlist", "speed", "checkpoint1", "checkpoint2", "n", "D", "avd"):
        if key in d:
            out[key] = np.asarray(d[key]).ravel()
    # drive_name is a MATLAB string, sometimes stored as an unnamed struct
    return out


def _pde_dir(root: Path) -> Path:
    return root / "wave justification" / "pde"


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    hexroot = here.parent
    golden = hexroot / "golden"
    golden.mkdir(parents=True, exist_ok=True)

    if len(argv) > 1:
        src = Path(argv[1]).resolve()
    else:
        src = (hexroot / "Hex-model-main" / "parameter_sensitive"
               / "hex code上交版.zip").resolve()

    tmp = None
    if src.is_dir():
        root = src
    elif zipfile.is_zipfile(src):
        tmp = tempfile.mkdtemp(prefix="hexzip_")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        root = Path(tmp) / "hex code上交版"
    else:
        print(f"error: source not found: {src}", file=sys.stderr)
        return 2

    pde = _pde_dir(root)
    if not pde.is_dir():
        print(f"error: no '{pde}' in source", file=sys.stderr)
        return 2

    manifest = {
        "source": str(src),
        "source_sha256": _sha256(src) if src.is_file() else None,
        "generator": "hexagon_spatial_test/repro/gen_golden_pde_reference.py",
        "model": "reaction-diffusion PDE main.m (D=avd^2/pi, dx=0.1, dt=1e-4)",
        "speed_units": "domain units per unit time (reaction applied once per unit time)",
        "drives": {},
    }

    speed_arrays = {}
    for drive in DRIVES:
        f = pde / drive / f"{drive}_pde_wavespeed.mat"
        d = _load_speed_mat(f)
        speed_arrays[f"{drive}_avdlist"] = d["avdlist"]
        speed_arrays[f"{drive}_speed"] = d["retlist"]
        manifest["drives"].setdefault(drive, {})
        manifest["drives"][drive]["wavespeed"] = {
            "file": os.path.relpath(f, root),
            "sha256": _sha256(f),
            "n": float(d["n"][0]),
            "checkpoint1": float(d["checkpoint1"][0]),
            "checkpoint2": float(d["checkpoint2"][0]),
            "n_dispersal": int(d["avdlist"].size),
        }
        print(f"speed  {drive:7s} n={int(d['n'][0]):<4d} "
              f"avd[{d['avdlist'][0]:.2g}..{d['avdlist'][-1]:.2g}] "
              f"speed[{d['retlist'][0]:.4f}..{d['retlist'][-1]:.4f}]")

    shape_arrays = {}
    for drive in DRIVES:
        for avd in SHAPE_AVDS:
            f = pde / drive / f"{drive}_pde_waveshape{avd:g}.mat"
            if not f.is_file():
                continue
            d = _load_hdf5_mat(f)
            tag = f"{drive}_{avd:g}"
            shape_arrays[f"{tag}_xlist"] = d["xlist"]
            shape_arrays[f"{tag}_ylist"] = d["ylist"]
            manifest["drives"][drive][f"waveshape_{avd:g}"] = {
                "file": os.path.relpath(f, root),
                "sha256": _sha256(f),
                "avd": float(d["avd"][0]),
                "n": int(d["n"][0]),
                "n_points": int(d["xlist"].size),
            }
            print(f"shape  {drive:7s} avd={avd:<4g} points={d['xlist'].size}")

    np.savez_compressed(golden / "pde_wavespeed_reference.npz", **speed_arrays)
    np.savez_compressed(golden / "pde_waveshape_reference.npz", **shape_arrays)
    with open(golden / "pde_reference_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    if tmp is not None:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print(f"wrote golden/pde_wavespeed_reference.npz "
          f"({len(speed_arrays)} arrays)")
    print(f"wrote golden/pde_waveshape_reference.npz "
          f"({len(shape_arrays)} arrays)")
    print("wrote golden/pde_reference_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
