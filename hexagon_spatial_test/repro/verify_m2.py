#!/usr/bin/env python
"""Faithfulness regression for the M2 natal-primitive port.

The M2 contract has two layers:

* ``compare_m2.py`` enforces the published acceptance band (wave-speed relative
  error < 5% vs the MATLAB golden);
* this script enforces the stronger *faithfulness* property stated in the M2
  report and requested for review: ``natal_hex_wavespeed.py`` should reproduce
  the verbatim MATLAB ``main.m`` logic, so a correct port matches the golden to
  machine precision.

The reference is written in 1-based MATLAB indexing.  A faithful port must
translate that convention exactly; the common failure is to reuse MATLAB's
1-based integers as 0-based indices (initial-condition condition, checkpoint and
``mid`` sampling) and to use Python's round-half-to-even instead of MATLAB's
round-half-away-from-zero.  Such offsets partially cancel, so the residual can
stay inside the 5% band while still being far above floating-point noise.

This test is expected to FAIL until those offsets are fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEXROOT = HERE.parent
GOLDEN = HEXROOT / "golden" / "hex_homing_wavespeed_m2.json"
NATAL = HEXROOT / "results" / "m2_natal_wavespeed.json"
SPEED_TOL = 1e-9
KERNEL_TOL = 1e-12


def main() -> int:
    golden = json.load(open(GOLDEN))["runs"]
    natal = json.load(open(NATAL))
    natruns = {(r["label"], round(r["avd"], 6)): r for r in natal["runs"]}
    failures = []

    print(f"1) natal port vs MATLAB golden (faithfulness, |rel| <= {SPEED_TOL:g})")
    for r in golden:
        key = (r["label"], round(r["avd"], 6))
        got = natruns[key]["speed"]
        rel = abs((got - r["speed"]) / r["speed"])
        ok = rel <= SPEED_TOL
        print(f"   {r['label']:15s} avd={r['avd']:<4.2f} "
              f"rel={rel:.3e} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{key}: |rel|={rel:.3e} > {SPEED_TOL:g}")

    print("2) natal kernel equals MATLAB kernel after documented column mirror")
    for avd, chk in natal["kernel_check"].items():
        ok = chk["max_abs_diff_col_mirror"] <= KERNEL_TOL
        print(f"   avd={avd:>4s} max|diff|={chk['max_abs_diff_col_mirror']:.2e} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"kernel col-mirror mismatch at avd={avd}")

    if failures:
        print("\nFAILURES:\n  " + "\n  ".join(failures))
        return 1
    print("\nALL FAITHFULNESS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
