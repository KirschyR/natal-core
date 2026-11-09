"""Synthetic and output-level tests for the GIS -> K pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hexgrid import build_overlap  # noqa: E402
from response import (  # noqa: E402
    altitude_func,
    carrying_capacity,
    rain_func,
    water_func,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_NPZ = DATA_DIR / "k_hawaii6_albers100.npz"
OUT_META = DATA_DIR / "k_hawaii6_albers100.meta.json"


def _constant_overlap(size=120):
    d = 100.0
    pixel = 30.0
    radius = d / np.sqrt(3.0)
    dx = np.sqrt(3.0) / 2.0 * d
    width = height = size * pixel
    nj = int(np.ceil((width + 2 * radius) / dx)) + 1
    ni = int(np.ceil(height / d + (nj - 1) / 2.0 + 2))
    u0 = -radius - 1.0
    v0 = -radius - 1.0
    matrix = build_overlap(pixel, size, size, d, u0, v0, ni, nj)
    return matrix, ni, nj, pixel, d


def test_altitude():
    alt = np.array([0.0, 400.0, 799.0, 800.0, 801.0, 2000.0])
    factor = altitude_func(alt)
    expected = np.array([1.0, 0.5, 1.0 - 799.0 / 800.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(factor, expected, atol=1e-12)
    assert factor[3] == 0.0 and factor[4] == 0.0
    print("PASS altitude: alt=800 -> 0, alt>800 -> 0")


def test_no_rain_no_water():
    shape = (8, 8)
    zeros = np.zeros(shape)
    land = np.ones(shape, dtype=bool)
    temp = np.full(shape, 22.0)
    value = carrying_capacity(
        np.full(shape, 100.0), temp, temp, temp, zeros,
        np.full(shape, 1000.0), zeros, zeros, land,
    )
    assert float(value.sum()) == 0.0
    assert rain_func(zeros).sum() == 0.0
    assert water_func(zeros, zeros, zeros).sum() == 0.0
    print("PASS no rain / no water -> K = 0")


def test_nan_inf():
    shape = (6, 6)
    bad = np.full(shape, np.nan)
    inf = np.full(shape, np.inf)
    land = np.ones(shape, dtype=bool)
    value = carrying_capacity(
        bad, inf, np.full(shape, -np.inf), bad, np.full(shape, 300.0),
        inf, bad, inf, land,
    )
    assert np.isfinite(value).all()
    assert (value >= 0).all()
    print("PASS NaN/Inf inputs -> finite non-negative K")


def test_constant_layer():
    matrix, ni, nj, pixel, d = _constant_overlap()
    pixel_count = 120 * 120
    coverage = np.asarray(matrix.sum(axis=0)).ravel()
    np.testing.assert_allclose(coverage, 1.0, atol=1e-6)
    constant = np.ones(pixel_count)
    hex_values = matrix.dot(constant)
    np.testing.assert_allclose(hex_values.sum(), pixel_count, rtol=1e-6)
    expected_hex = (np.sqrt(3.0) / 2.0 * d * d) / (pixel * pixel)
    full = np.asarray(matrix.sum(axis=1)).ravel() > 0.999 * expected_hex
    assert full.any()
    np.testing.assert_allclose(hex_values[full], expected_hex, rtol=1e-6)
    print(f"PASS constant layer: total conserved, interior hex value={expected_hex:.6f}")


def test_land_sea():
    matrix, ni, nj, pixel, d = _constant_overlap()
    size = 120
    land = np.zeros((size, size), dtype=bool)
    land[20:-20, 20:-20] = True
    keep = land.ravel()
    land_values = matrix.dot(keep.astype(float))
    land_hex = land_values > 0.0
    sea_hex = ~land_hex
    assert sea_hex.any() and land_hex.any()
    np.testing.assert_allclose(land_values[sea_hex], 0.0, atol=0.0)
    np.testing.assert_allclose(land_values.sum(), keep.sum(), rtol=1e-6)
    print("PASS land/sea: sea hexes = 0, land mass conserved")


def test_real_output():
    if not OUT_NPZ.exists() or not OUT_META.exists():
        raise FileNotFoundError(f"missing {OUT_NPZ} or {OUT_META}")
    with np.load(OUT_NPZ) as data:
        K = data["K"]
        land = data["land_mask"]
        months = data["months"]
    meta = json.loads(OUT_META.read_text(encoding="utf-8"))
    assert K.shape == (12, meta["rows"], meta["cols"])
    assert K.dtype == np.float32
    assert land.shape == K.shape[1:]
    assert land.dtype == bool
    assert months.dtype == np.int16
    np.testing.assert_array_equal(months, np.arange(1, 13))
    assert np.isfinite(K).all()
    assert (K >= 0).all()
    assert float(np.abs(K[:, ~land]).max()) == 0.0
    assert int(land.sum()) > 1_000_000
    for row in meta["diagnostics"]["months"]:
        assert row["conservation_relative_error"] < 1e-6, row
    print(f"PASS real output: shape={K.shape}, land_hex={int(land.sum())}, "
          f"max conservation error="
          f"{max(r['conservation_relative_error'] for r in meta['diagnostics']['months']):.2e}")
    print("monthly K range / total:")
    for month in range(12):
        values = K[month][land]
        print(f"  {month + 1:02d}: min={values.min():.6g} max={values.max():.6g} "
              f"sum={values.sum(dtype=np.float64):.6g}")


def main():
    test_altitude()
    test_no_rain_no_water()
    test_nan_inf()
    test_constant_layer()
    test_land_sea()
    test_real_output()
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
