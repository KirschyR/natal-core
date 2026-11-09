"""Albers hex grid definition and exact pixel-to-hex area integration.

The hex lattice follows the hex-model-new convention with a north-west
origin, ``x = (sqrt(3)/2) * j * d`` and ``y_south = (i - j/2) * d``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from pyproj import Transformer
from scipy import sparse

CRS_AEA = (
    "+proj=aea +lat_1=19 +lat_2=22 +lat_0=20.5 +lon_0=-157 "
    "+datum=WGS84 +units=m +no_defs"
)
CENTER_DISTANCE_M = 100.0
PIXEL_SIZE_M = 30.0
GLOBAL_MARGIN_M = 5000.0

# lon_min, lon_max, lat_min, lat_max for the six main islands.
ISLAND_BBOX = {
    "kauai": (-159.79, -159.25, 21.85, 22.25),
    "oahu": (-158.28, -157.65, 21.22, 21.72),
    "molokai": (-157.33, -156.69, 21.03, 21.24),
    "lanai": (-157.06, -156.80, 20.70, 20.93),
    "maui": (-156.72, -155.97, 20.50, 21.05),
    "hawaii": (-156.10, -154.75, 18.90, 20.30),
}
ISLAND_ORDER = ("kauai", "oahu", "molokai", "lanai", "maui", "hawaii")


def _boundary_lonlat(lon0: float, lon1: float, lat0: float, lat1: float, n: int = 600):
    lon = np.linspace(lon0, lon1, n)
    lat = np.linspace(lat0, lat1, n)
    return (
        np.r_[lon, lon, np.full(n, lon0), np.full(n, lon1)],
        np.r_[np.full(n, lat0), np.full(n, lat1), lat, lat],
    )


def lonlat_extent_to_aea(lon0, lon1, lat0, lat1):
    """Return (xmin, xmax, ymin, ymax) after projecting the lon/lat boundary."""
    transformer = Transformer.from_crs("EPSG:4326", CRS_AEA, always_xy=True)
    lon, lat = _boundary_lonlat(lon0, lon1, lat0, lat1)
    x, y = transformer.transform(lon, lat)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


@dataclass(frozen=True)
class Grid:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    rows: int
    cols: int
    pixel_rows: int
    pixel_cols: int
    center_distance_m: float = CENTER_DISTANCE_M
    pixel_size_m: float = PIXEL_SIZE_M
    crs: str = CRS_AEA

    @property
    def dx(self) -> float:
        return np.sqrt(3.0) / 2.0 * self.center_distance_m

    @property
    def radius(self) -> float:
        return self.center_distance_m / np.sqrt(3.0)

    @property
    def hex_area_m2(self) -> float:
        return np.sqrt(3.0) / 2.0 * self.center_distance_m ** 2

    @property
    def pixel_area_m2(self) -> float:
        return self.pixel_size_m ** 2

    def hex_center(self, i: int, j: int) -> tuple[float, float]:
        return (
            self.xmin + self.dx * j,
            self.ymax - (i - j / 2.0) * self.center_distance_m,
        )

    def pixel_transform(self, r0: int, c0: int):
        from affine import Affine

        return Affine(
            self.pixel_size_m,
            0.0,
            self.xmin + c0 * self.pixel_size_m,
            0.0,
            -self.pixel_size_m,
            self.ymax - r0 * self.pixel_size_m,
        )

    def subset(self, lon0, lon1, lat0, lat1, buffer_m: float = 3000.0):
        return self.local_window(lon0, lon1, lat0, lat1, buffer_m)

    def local_window(self, lon0, lon1, lat0, lat1, buffer_m: float = 3000.0):
        x0, x1, y0, y1 = lonlat_extent_to_aea(lon0, lon1, lat0, lat1)
        x0 -= buffer_m
        x1 += buffer_m
        y0 -= buffer_m
        y1 += buffer_m
        p = self.pixel_size_m
        c0 = max(0, int(np.floor((x0 - self.xmin) / p)) - 1)
        c1 = min(self.pixel_cols, int(np.ceil((x1 - self.xmin) / p)) + 1)
        r0 = max(0, int(np.floor((self.ymax - y1) / p)) - 1)
        r1 = min(self.pixel_rows, int(np.ceil((self.ymax - y0) / p)) + 1)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("empty local window")
        return Window(r0=r0, r1=r1, c0=c0, c1=c1)

    def hex_window_bounds(self, window: "Window"):
        """Return the global hex index rectangle covering a pixel window."""
        ilo, ihi, jlo, jhi, _, _, _, _ = local_hex_offsets(self, window)
        return ilo, ihi, jlo, jhi


@dataclass(frozen=True)
class Window:
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def rows(self) -> int:
        return self.r1 - self.r0

    @property
    def cols(self) -> int:
        return self.c1 - self.c0

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols


def compute_grid(margin_m: float = GLOBAL_MARGIN_M, pixel_size_m: float = PIXEL_SIZE_M,
                 center_distance_m: float = CENTER_DISTANCE_M) -> Grid:
    lon0 = min(v[0] for v in ISLAND_BBOX.values())
    lon1 = max(v[1] for v in ISLAND_BBOX.values())
    lat0 = min(v[2] for v in ISLAND_BBOX.values())
    lat1 = max(v[3] for v in ISLAND_BBOX.values())
    xmin, xmax, ymin, ymax = lonlat_extent_to_aea(lon0, lon1, lat0, lat1)
    xmin -= margin_m
    xmax += margin_m
    ymin -= margin_m
    ymax += margin_m
    d = center_distance_m
    radius = d / np.sqrt(3.0)
    dx = np.sqrt(3.0) / 2.0 * d
    width, height = xmax - xmin, ymax - ymin
    cols = int(np.ceil((width + radius) / dx)) + 1
    rows = int(np.ceil(height / d + cols / 2.0 + 1.0)) + 1
    pixel_cols = int(np.ceil(width / pixel_size_m))
    pixel_rows = int(np.ceil(height / pixel_size_m))
    return Grid(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        rows=rows,
        cols=cols,
        pixel_rows=pixel_rows,
        pixel_cols=pixel_cols,
        center_distance_m=center_distance_m,
        pixel_size_m=pixel_size_m,
    )


def local_hex_offsets(grid: Grid, window: Window):
    """Translate global hex indices into a local rectangle with offsets."""
    d = grid.center_distance_m
    dx = grid.dx
    radius = grid.radius
    u0 = -window.c0 * grid.pixel_size_m
    v0 = -window.r0 * grid.pixel_size_m
    width = window.cols * grid.pixel_size_m
    height = window.rows * grid.pixel_size_m
    # j is chosen so that local u starts just left of the window.
    jlo = max(0, int(np.floor((-radius - u0) / dx)) - 1)
    jhi = min(grid.cols - 1, int(np.ceil((width + radius - u0) / dx)) + 1)
    nj = jhi - jlo + 1
    u0l = u0 + dx * jlo
    # i is chosen so that local v starts just above the window.  Because
    # v = (i - j/2) * d + v0, i must grow with jlo/2 for a north-west origin.
    ilo = max(0, int(np.floor((-radius - v0) / d + jlo / 2.0)) - 1)
    v0l = v0 + (ilo - jlo / 2.0) * d
    extra_rows = int(np.ceil((height + radius - v0l) / d + (nj - 1) / 2.0)) + 2
    ihi = min(grid.rows - 1, ilo + extra_rows - 1)
    ni = ihi - ilo + 1
    return ilo, ihi, jlo, jhi, ni, nj, u0l, v0l


def build_overlap(pixel_size: float, rows: int, cols: int, distance: float,
                  u0: float, v0: float, ni: int, nj: int,
                  keep_flat: np.ndarray | None = None,
                  progress_every: int = 400) -> sparse.csr_matrix:
    """Exact area-overlap matrix from a rectangular pixel window to a hex grid.

    Row index is ``j * ni + i`` (local hex indices), column index is the
    C-order flat pixel index ``r * cols + c``.  Values are intersection area
    divided by pixel area.  When ``keep_flat`` is given only those source
    pixels receive entries.
    """
    radius = distance / np.sqrt(3.0)
    dx = np.sqrt(3.0) / 2.0 * distance
    width = cols * pixel_size
    height = rows * pixel_size
    pixel_area = pixel_size * pixel_size
    vertices_x = radius * np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    vertices_y = (distance / 2.0) * np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0])
    rr: list[np.ndarray] = []
    cc: list[np.ndarray] = []
    vv: list[np.ndarray] = []
    for j in range(nj):
        i = np.arange(ni)
        x = np.full(ni, dx * j + u0)
        y = (i - j / 2.0) * distance + v0
        keep_h = (
            (y + distance / 2.0 > 0)
            & (y - distance / 2.0 < height)
            & (x - radius < width)
            & (x + radius > 0)
        )
        i, x, y = i[keep_h], x[keep_h], y[keep_h]
        if not len(i):
            continue
        polygons = shapely.polygons(
            np.stack([x[:, None] + vertices_x, y[:, None] + vertices_y], axis=-1)
        )
        r0 = np.floor((y - distance / 2.0) / pixel_size).astype(np.int64)
        r1 = np.floor((y + distance / 2.0) / pixel_size).astype(np.int64)
        c0 = np.floor((x - radius) / pixel_size).astype(np.int64)
        c1 = np.floor((x + radius) / pixel_size).astype(np.int64)
        for dr in range(int(np.max(r1 - r0)) + 1):
            for dc in range(int(np.max(c1 - c0)) + 1):
                r = r0 + dr
                c = c0 + dc
                ok = (
                    (r <= r1)
                    & (c <= c1)
                    & (r >= 0)
                    & (c >= 0)
                    & (r < rows)
                    & (c < cols)
                )
                if keep_flat is not None:
                    flat = np.clip(r * cols + c, 0, rows * cols - 1)
                    ok &= keep_flat[flat]
                if not np.any(ok):
                    continue
                rectangles = shapely.box(
                    c[ok] * pixel_size,
                    r[ok] * pixel_size,
                    (c[ok] + 1) * pixel_size,
                    (r[ok] + 1) * pixel_size,
                )
                areas = shapely.area(shapely.intersection(polygons[ok], rectangles))
                nonzero = areas > pixel_area * 1e-14
                if not np.any(nonzero):
                    continue
                rr.append(((i[ok] + j * ni)[nonzero]).astype(np.int32))
                cc.append(((r[ok] * cols + c[ok])[nonzero]).astype(np.int32))
                vv.append((areas[nonzero] / pixel_area).astype(np.float32))
        if progress_every and (j + 1) % progress_every == 0:
            print(f"    overlap column {j + 1}/{nj}", flush=True)
    matrix = sparse.coo_matrix(
        (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
        shape=(ni * nj, rows * cols),
    ).tocsr()
    return matrix


def window_lonlat_bounds(grid: Grid, window: Window, n: int = 100):
    transformer = Transformer.from_crs(CRS_AEA, "EPSG:4326", always_xy=True)
    x0 = grid.xmin + window.c0 * grid.pixel_size_m
    x1 = x0 + window.cols * grid.pixel_size_m
    y1 = grid.ymax - window.r0 * grid.pixel_size_m
    y0 = y1 - window.rows * grid.pixel_size_m
    xs = np.r_[np.linspace(x0, x1, n), np.linspace(x0, x1, n),
               np.full(n, x0), np.full(n, x1)]
    ys = np.r_[np.full(n, y0), np.full(n, y1),
               np.linspace(y0, y1, n), np.linspace(y0, y1, n)]
    lon, lat = transformer.transform(xs, ys)
    return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
