"""Generate M5 radial-release golden with the reference MATLAB optimizer.

Runs the (patched-arena) copy of ``radial_release_optimizer/main.m``
under ``repro/ref/radial/`` for each ``(t, h, r)`` release configuration and
collects the week-104 / week-261 coverage rows written to ``output/<t>,<h>,<r>.csv``.

The reference copy differs from the upstream file only in that ``m``/``n`` are
optional trailing arguments (the upstream hardcodes ``m=800;n=800``); every
other line is byte-identical.  See ``golden/m5_radial_homing.json`` for the
recorded provenance hashes.

Usage (one shard per process):
    python repro/gen_golden_m5_radial.py --arena 500 --shard 0 --nshards 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "ref" / "radial"
GOLDEN = HERE.parent / "golden"
MATLAB = "/opt/matlab/bin/matlab"

DRIVE = "homing"
CONFIGS: list[tuple[int, int, int]] = [
    (1, 1, 0),
    (1, 10, 0),
    (1, 50, 0),
    (1, 100, 0),
    (1, 10, 20),
    (1, 50, 20),
    (1, 10, 40),
    (1, 100, 40),
    (4, 10, 0),
    (4, 10, 20),
    (4, 50, 20),
    (4, 100, 40),
]


def _run_configs(arena: int, configs: list[tuple[int, int, int]], work: Path) -> list[dict]:
    """Run a list of configs in one MATLAB process and parse their CSVs."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "homing").mkdir(exist_ok=True)
    (work / "output").mkdir(exist_ok=True)
    shutil.copy(REF / "mig.mat", work / "mig.mat")
    shutil.copy(REF / DRIVE / "drive_params.mat", work / DRIVE / "drive_params.mat")

    cfg_lines = "\n".join(
        f"main({t}, {h}, {r}, {arena}, {arena});" for (t, h, r) in configs
    )
    driver = work / "driver.m"
    driver.write_text(
        "addpath('" + str(REF) + "');\n"
        "addpath('" + str(REF / DRIVE) + "');\n"
        f"cd('{work / DRIVE}');\n"
        + cfg_lines
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [MATLAB, "-batch", f"run('{driver}')"],
        cwd=str(work),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    results: list[dict] = []
    for (t, h, r) in configs:
        csv = work / DRIVE / "output" / f"{t},{h},{r}.csv"
        rows = [line for line in csv.read_text().strip().splitlines() if line.strip()]
        parsed = [[float(v) for v in line.rstrip(",").split(",")] for line in rows]
        record = {"t": t, "h": h, "r": r, "arena": arena, "rows": parsed}
        results.append(record)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    shard = CONFIGS[args.shard :: args.nshards]
    with tempfile.TemporaryDirectory(prefix="m5gold_") as td:
        results = _run_configs(args.arena, shard, Path(td))

    out = Path(args.out) if args.out else (GOLDEN / f"m5_radial_homing_arena{args.arena}_shard{args.shard}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "drive": DRIVE,
        "arena": args.arena,
        "reference_main_sha256": _sha256(REF / "main.m"),
        "reference_mig_sha256": _sha256(REF / "mig.mat"),
        "reference_drive_params_sha256": _sha256(REF / DRIVE / "drive_params.mat"),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(results)} configs)")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
