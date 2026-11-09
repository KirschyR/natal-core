"""Build the monthly Hawaii six-island capacity matrix K on a 100 m hex grid.

Pipeline: raw GIS -> Albers 30 m pixel grid -> per-pixel response functions ->
exact area integration onto the hex-model-new hex lattice.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hexgrid import (  # noqa: E402
    CENTER_DISTANCE_M,
    CRS_AEA,
    ISLAND_BBOX,
    ISLAND_ORDER,
    PIXEL_SIZE_M,
    build_overlap,
    compute_grid,
    local_hex_offsets,
    window_lonlat_bounds,
)
from response import PARAMETERS, carrying_capacity  # noqa: E402
from sources import load_climate, load_dem, load_population, load_water, read_manifest  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_NPZ = DATA_DIR / "k_hawaii6_albers100.npz"
OUT_META = DATA_DIR / "k_hawaii6_albers100.meta.json"
MAPPING_VERSION = "exact_overlap_v1_center_distance_100_pixel_30"
BUILD_VERSION = "gis2score_hawaii6_v1"


def _packed_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(array, dtype=bool)).tobytes()).hexdigest()[:16]


def _cache_path(island: str, signature: str) -> Path:
    return CACHE_DIR / f"overlap_{island}_{signature}.npz"


def _load_or_build_overlap(island, window, grid, ilo, ihi, jlo, jhi, ni, nj,
                           u0l, v0l, keep, rebuild=False):
    signature = hashlib.sha256(
        json.dumps(
            [
                MAPPING_VERSION,
                window.r0, window.r1, window.c0, window.c1,
                ilo, ihi, jlo, jhi, ni, nj,
                round(u0l, 6), round(v0l, 6),
                grid.center_distance_m, grid.pixel_size_m,
                _packed_hash(keep),
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]
    path = _cache_path(island, signature)
    if path.exists() and not rebuild:
        matrix = sparse.load_npz(path)
        print(f"  overlap cache hit {path.name} nnz={matrix.nnz}", flush=True)
        return matrix, signature
    print(f"  building overlap {island}: pixels={window.rows}x{window.cols} "
          f"hexes={ni}x{nj} ...", flush=True)
    start = time.time()
    matrix = build_overlap(
        grid.pixel_size_m, window.rows, window.cols, grid.center_distance_m,
        u0l, v0l, ni, nj, keep_flat=keep,
    )
    print(f"  overlap built in {time.time() - start:.1f}s nnz={matrix.nnz}", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".part.npz")
    sparse.save_npz(temp, matrix)
    temp.replace(path)
    return matrix, signature


def _check_overlap(matrix, keep, label):
    coverage = np.asarray(matrix.sum(axis=0)).ravel()
    bad = keep & (np.abs(coverage - 1.0) > 1e-4)
    if bad.any():
        worst = float(np.abs(coverage[keep] - 1.0).max())
        raise AssertionError(f"{label}: {int(bad.sum())} kept pixels not fully covered (max {worst:g})")
    fill = np.asarray(matrix.sum(axis=1)).ravel()
    return coverage, fill


def _hex_values(matrix, pixel_values):
    return matrix.dot(pixel_values)


def _scatter(target, month_index, ilo, ihi, jlo, jhi, ni, nj, values):
    block = values.reshape(nj, ni).T
    target[month_index, ilo:ihi + 1, jlo:jhi + 1] += block


def build(islands=ISLAND_ORDER, rebuild_overlap=False, out_npz=OUT_NPZ, out_meta=OUT_META):
    started = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grid = compute_grid()
    print(f"grid rows={grid.rows} cols={grid.cols} cells={grid.rows * grid.cols / 1e6:.2f}M", flush=True)
    print(f"pixel grid rows={grid.pixel_rows} cols={grid.pixel_cols}", flush=True)

    K = np.zeros((12, grid.rows, grid.cols), dtype=np.float32)
    land_mask = np.zeros((grid.rows, grid.cols), dtype=bool)
    claimed = np.zeros((grid.pixel_rows, grid.pixel_cols), dtype=bool)
    pixel_sum = np.zeros(12, dtype=np.float64)
    hex_sum = np.zeros(12, dtype=np.float64)
    island_reports = []

    for island in islands:
        island_start = time.time()
        bbox = ISLAND_BBOX[island]
        window = grid.local_window(*bbox, buffer_m=4000.0)
        print(f"\n[{island}] window rows={window.rows} cols={window.cols} "
              f"({window.rows * window.cols / 1e6:.2f}M px)", flush=True)
        elevation, land = load_dem(grid, window, bbox)
        claimed_slice = claimed[window.r0:window.r1, window.c0:window.c1]
        land = land & ~claimed_slice
        population = load_population(grid, window)
        water_p, water_s = load_water(grid, window)
        print(f"  land pixels={int(land.sum())} "
              f"({land.sum() * grid.pixel_area_m2 / 1e6:.1f} km2); "
              f"pop max={float(population.max()):.1f}/km2; "
              f"perm={int(water_p.sum())} seas={int(water_s.sum())}", flush=True)

        ilo, ihi, jlo, jhi, ni, nj, u0l, v0l = local_hex_offsets(grid, window)
        keep = land.ravel()
        matrix, signature = _load_or_build_overlap(
            island, window, grid, ilo, ihi, jlo, jhi, ni, nj, u0l, v0l,
            keep, rebuild=rebuild_overlap,
        )
        coverage, fill = _check_overlap(matrix, keep, island)
        hex_land = (fill > 0.0).reshape(nj, ni).T
        max_fill = float(fill.max()) / (grid.hex_area_m2 / grid.pixel_area_m2)

        island_month_totals = []
        for month in range(1, 13):
            climate = load_climate(grid, window, month)
            pixel_k = carrying_capacity(
                elevation,
                climate["tavg"], climate["tmin"], climate["tmax"], climate["prec"],
                population, water_p, water_s, land,
            )
            flat_k = pixel_k.ravel()
            local_pixel_sum = float(flat_k.sum(dtype=np.float64))
            local_hex = _hex_values(matrix, flat_k)
            local_hex_sum = float(local_hex.sum(dtype=np.float64))
            _scatter(K, month - 1, ilo, ihi, jlo, jhi, ni, nj, local_hex)
            pixel_sum[month - 1] += local_pixel_sum
            hex_sum[month - 1] += local_hex_sum
            island_month_totals.append(
                {"month": month, "pixel_sum": local_pixel_sum,
                 "hex_sum": local_hex_sum,
                 "relative_error": abs(local_hex_sum - local_pixel_sum) / max(1.0, local_pixel_sum)}
            )
            del climate, pixel_k, flat_k, local_hex
            gc.collect()

        land_mask[ilo:ihi + 1, jlo:jhi + 1] |= hex_land
        claimed_slice |= land
        report = {
            "island": island,
            "window": {"r0": window.r0, "r1": window.r1, "c0": window.c0, "c1": window.c1},
            "hex_rect": {"ilo": ilo, "ihi": ihi, "jlo": jlo, "jhi": jhi, "ni": ni, "nj": nj},
            "land_pixels": int(land.sum()),
            "land_area_km2": float(land.sum() * grid.pixel_area_m2 / 1e6),
            "overlap_nnz": int(matrix.nnz),
            "overlap_signature": signature,
            "coverage_min": float(coverage[keep].min()),
            "coverage_max": float(coverage[keep].max()),
            "max_hex_fill_ratio": max_fill,
            "months": island_month_totals,
            "elapsed_seconds": time.time() - island_start,
        }
        island_reports.append(report)
        print(f"  done in {report['elapsed_seconds']:.1f}s; "
              f"max hex fill ratio={max_fill:.6f}", flush=True)
        del elevation, land, population, water_p, water_s, matrix, keep, hex_land
        gc.collect()

    K[:, ~land_mask] = 0.0
    K = np.maximum(np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    months = np.arange(1, 13, dtype=np.int16)

    stats = []
    for month in range(12):
        land_values = K[month][land_mask]
        total_hex = float(hex_sum[month])
        rel = abs(total_hex - float(pixel_sum[month])) / max(1.0, float(pixel_sum[month]))
        stats.append(
            {
                "month": month + 1,
                "min": float(land_values.min()) if land_values.size else 0.0,
                "max": float(land_values.max()) if land_values.size else 0.0,
                "mean_land_hex": float(land_values.mean()) if land_values.size else 0.0,
                "total_hex": total_hex,
                "total_pixel": float(pixel_sum[month]),
                "conservation_relative_error": rel,
                "n_land_hex_nonzero": int((land_values > 0).sum()),
            }
        )

    _write_outputs(K, land_mask, months, grid, stats, island_reports, started, out_npz, out_meta)
    for row in stats:
        print(f"month {row['month']:02d}: K min={row['min']:.6g} max={row['max']:.6g} "
              f"sum={row['total_hex']:.6g} rel_err={row['conservation_relative_error']:.2e}",
              flush=True)
    print(f"\nwrote {out_npz} ({out_npz.stat().st_size / 1e9:.2f} GB) and {out_meta}", flush=True)
    return stats


def _crs_wkt():
    from pyproj import CRS

    return CRS.from_user_input(CRS_AEA).to_wkt()


def _write_outputs(K, land_mask, months, grid, stats, island_reports, started, out_npz, out_meta):
    now = datetime.now(timezone.utc).isoformat()
    temp_npz = out_npz.with_name(out_npz.stem + ".part.npz")
    np.savez_compressed(temp_npz, K=K, land_mask=land_mask, months=months)
    temp_npz.replace(out_npz)

    manifest = read_manifest()
    inputs = {}
    for rel, record in sorted(manifest.get("files", {}).items()):
        entry = {k: record[k] for k in ("url", "bytes", "sha256", "fetched_at") if k in record}
        if record.get("missing"):
            entry = {"url": record.get("url"), "missing": True}
        inputs[rel] = entry
    meta = {
        "version": BUILD_VERSION,
        "mapping_version": MAPPING_VERSION,
        "created_at": now,
        "updated_at": now,
        "crs": CRS_AEA,
        "crs_wkt": _crs_wkt(),
        "convention": "hex-model-new: x=(sqrt(3)/2)*j*d, y_south=(i-j/2)*d; origin at NW corner",
        "center_distance_m": grid.center_distance_m,
        "area_scale": 1.0,
        "hex_area_m2": grid.hex_area_m2,
        "pixel_size_m": grid.pixel_size_m,
        "pixel_area_m2": grid.pixel_area_m2,
        "rows": grid.rows,
        "cols": grid.cols,
        "origin_xy_m": [grid.xmin, grid.ymax],
        "extent_m": [grid.xmax - grid.xmin, grid.ymax - grid.ymin],
        "pixel_grid": {"rows": grid.pixel_rows, "cols": grid.pixel_cols,
                       "origin_xy_m": [grid.xmin, grid.ymax], "pixel_size_m": grid.pixel_size_m},
        "months": "monthly climatology 1..12 from WorldClim 2.1 30s; no single year",
        "units": {
            "K": "uncalibrated capacity per hex (placeholder; k_scale applied)",
            "elevation": "m",
            "population": "people/km^2",
            "temperature": "degC",
            "precipitation": "mm/month",
            "water": "0/1 per 30 m pixel",
        },
        "k_scale": PARAMETERS["k_scale"],
        "parameters": PARAMETERS,
        "land_mask_definition": "hex overlaps at least one 30 m pixel with elevation > 0",
        "region": {name: list(box) for name, box in ISLAND_BBOX.items()},
        "inputs": inputs,
        "source_notes": {
            "dem": "Copernicus DEM GLO-30, 1-degree COG tiles (AWS Open Data)",
            "population": "WorldPop 2020 USA UN-adjusted constrained counts; nodata treated as 0",
            "worldclim": "WorldClim 2.1 30s tiled COG tile_25, float32 degC and mm/month (not x10)",
            "water": "JRC Global Surface Water v1.4 seasonality; 12 permanent, 1-11 seasonal, 0/255 none",
        },
        "diagnostics": {
            "months": stats,
            "islands": island_reports,
            "total_elapsed_seconds": time.time() - started,
        },
    }
    temp_meta = out_meta.with_suffix(".json.part")
    temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    temp_meta.replace(out_meta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--islands", nargs="+", choices=ISLAND_ORDER, default=list(ISLAND_ORDER))
    parser.add_argument("--rebuild-overlap", action="store_true")
    parser.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    parser.add_argument("--out-meta", type=Path, default=OUT_META)
    args = parser.parse_args(argv)
    stats = build(tuple(args.islands), rebuild_overlap=args.rebuild_overlap,
                  out_npz=args.out_npz, out_meta=args.out_meta)
    return 0 if all(s["conservation_relative_error"] < 1e-3 for s in stats) else 1


if __name__ == "__main__":
    sys.exit(main())
