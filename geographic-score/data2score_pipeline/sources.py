"""Read raw GIS layers and resample them onto the Albers 30 m build grid."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

from hexgrid import CRS_AEA, Grid, Window, window_lonlat_bounds

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEM_DIR = RAW_DIR / "copernicus_dem"
WORLDCLIM_DIR = RAW_DIR / "worldclim"
JRC_DIR = RAW_DIR / "jrc"
WORLDPOP_PATH = RAW_DIR / "worldpop" / "usa_ppp_2020_UNadj_constrained.tif"

WORLDCLIM_VARS = ("tavg", "tmin", "tmax", "prec")


def read_manifest() -> dict:
    path = RAW_DIR / "manifest.json"
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=256)
def _bounds(path: str):
    with rasterio.open(path) as src:
        return tuple(src.bounds)


def _expanded(lon0, lon1, lat0, lat1, pad):
    return lon0 - pad, lon1 + pad, lat0 - pad, lat1 + pad


def _intersects(bounds, lon0, lon1, lat0, lat1) -> bool:
    left, bottom, right, top = bounds
    return not (right <= lon0 or left >= lon1 or top <= lat0 or bottom >= lat1)


def dem_tiles(lon0, lon1, lat0, lat1):
    paths = sorted(str(p) for p in DEM_DIR.glob("*/*.tif"))
    return [p for p in paths if _intersects(_bounds(p), lon0, lon1, lat0, lat1)]


def jrc_tiles(lon0, lon1, lat0, lat1):
    paths = sorted(str(p) for p in JRC_DIR.glob("*.tif"))
    return [p for p in paths if _intersects(_bounds(p), lon0, lon1, lat0, lat1)]


def _target_transform(grid: Grid, window: Window):
    return grid.pixel_transform(window.r0, window.c0)


def _warp(source, src_transform, src_crs, window: Window, grid: Grid,
          src_nodata, dst_nodata, resampling) -> np.ndarray:
    destination = np.full(window.shape, dst_nodata, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=_target_transform(grid, window),
        dst_crs=CRS_AEA,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return destination


def _merge_window(paths, lon0, lon1, lat0, lat1, nodata, dtype, resampling=None):
    if not paths:
        return None, None
    destination, transform = merge(
        paths,
        bounds=(lon0, lat0, lon1, lat1),
        nodata=nodata,
        dtype=dtype,
        indexes=[1],
    )
    return destination[0], transform


def load_dem(grid: Grid, window: Window, bbox) -> tuple[np.ndarray, np.ndarray]:
    """Return elevation (m) and land mask (elevation > 0) on the build window."""
    tlon0, tlon1, tlat0, tlat1 = window_lonlat_bounds(grid, window)
    lon0, lon1, lat0, lat1 = _expanded(tlon0, tlon1, tlat0, tlat1, 0.10)
    paths = dem_tiles(lon0, lon1, lat0, lat1)
    merged, transform = _merge_window(paths, lon0, lon1, lat0, lat1,
                                      nodata=0.0, dtype="float32")
    if merged is None:
        raise RuntimeError("no DEM tiles for window")
    elevation = _warp(merged, transform, "EPSG:4326", window, grid,
                      src_nodata=0.0, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
    elevation = np.where(np.isfinite(elevation), elevation, 0.0).astype(np.float32)
    return elevation, elevation > 0.0


def load_population(grid: Grid, window: Window) -> np.ndarray:
    """Return population density in people/km^2 on the build window."""
    tlon0, tlon1, tlat0, tlat1 = window_lonlat_bounds(grid, window)
    lon0, lon1, lat0, lat1 = _expanded(tlon0, tlon1, tlat0, tlat1, 0.05)
    with rasterio.open(WORLDPOP_PATH) as src:
        read_window = from_bounds(lon0, lat0, lon1, lat1, src.transform)
        counts = src.read(1, window=read_window).astype(np.float64)
        transform = src.window_transform(read_window)
        src_crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        counts[counts == nodata] = 0.0
    counts[~np.isfinite(counts)] = 0.0
    counts = np.maximum(counts, 0.0)
    rows = np.arange(counts.shape[0])
    latitude = transform.f + (rows + 0.5) * transform.e
    width_m = abs(transform.a) * 111320.0 * np.cos(np.deg2rad(latitude))
    height_m = abs(transform.e) * 110574.0
    area_m2 = np.maximum(width_m * height_m, 1.0)
    density = counts / area_m2[:, None] * 1e6
    result = _warp(density, transform, src_crs, window, grid,
                   src_nodata=0.0, dst_nodata=0.0,
                   resampling=Resampling.bilinear)
    result[~np.isfinite(result)] = 0.0
    return np.maximum(result, 0.0).astype(np.float32)


def load_water(grid: Grid, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Return (permanent, seasonal) 0/1 water masks.

    JRC seasonality: 12 -> permanent, 1..11 -> seasonal, 0 or 255 -> no water.
    """
    tlon0, tlon1, tlat0, tlat1 = window_lonlat_bounds(grid, window)
    lon0, lon1, lat0, lat1 = _expanded(tlon0, tlon1, tlat0, tlat1, 0.05)
    paths = jrc_tiles(lon0, lon1, lat0, lat1)
    merged, transform = _merge_window(paths, lon0, lon1, lat0, lat1,
                                      nodata=255, dtype="uint8")
    if merged is None:
        raise RuntimeError("no JRC tiles for window")
    seasonality = _warp(merged, transform, "EPSG:4326", window, grid,
                        src_nodata=255, dst_nodata=255,
                        resampling=Resampling.nearest)
    seasonality = np.rint(seasonality).astype(np.int16)
    permanent = seasonality == 12
    seasonal = (seasonality >= 1) & (seasonality <= 11)
    return permanent, seasonal


def _fill_nearest_valid(data: np.ndarray) -> np.ndarray:
    """Fill NaN/Inf cells with the value of the nearest finite cell."""
    from scipy import ndimage

    valid = np.isfinite(data)
    if valid.all():
        return data
    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)
    indices = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = data[tuple(indices)].astype(np.float32)
    return filled


def _warp_climate(source, transform, window, grid):
    # WorldClim marks ocean as NaN.  On small islands the 1 km cells touching
    # the coast can be ocean-only, so fill the source from nearest land before
    # interpolation instead of letting zeros leak onto land.
    source = _fill_nearest_valid(source)
    warped = _warp(source, transform, "EPSG:4326", window, grid,
                   src_nodata=None, dst_nodata=np.nan,
                   resampling=Resampling.bilinear)
    if np.isnan(warped).any():
        nearest = _warp(source, transform, "EPSG:4326", window, grid,
                        src_nodata=None, dst_nodata=np.nan,
                        resampling=Resampling.nearest)
        missing = ~np.isfinite(warped)
        warped[missing] = nearest[missing]
    warped[~np.isfinite(warped)] = 0.0
    return warped


def load_climate(grid: Grid, window: Window, month: int) -> dict[str, np.ndarray]:
    """Return monthly tavg/tmin/tmax (degC) and prec (mm/month)."""
    tlon0, tlon1, tlat0, tlat1 = window_lonlat_bounds(grid, window)
    lon0, lon1, lat0, lat1 = _expanded(tlon0, tlon1, tlat0, tlat1, 0.05)
    output = {}
    for var in WORLDCLIM_VARS:
        path = WORLDCLIM_DIR / f"tile_25_wc2.1_30s_{var}.tif"
        with rasterio.open(path) as src:
            read_window = from_bounds(lon0, lat0, lon1, lat1, src.transform)
            data = src.read(month, window=read_window).astype(np.float32)
            transform = src.window_transform(read_window)
        output[var] = _warp_climate(data, transform, window, grid)
    return output
