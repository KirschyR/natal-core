"""Plot monthly capacity K on the hex grid as overview heatmaps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, PowerNorm  # noqa: E402
from pyproj import Transformer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_NPZ = DATA_DIR / "k_hawaii6_albers100.npz"
DEFAULT_META = DATA_DIR / "k_hawaii6_albers100.meta.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "imgs_output"

ISLAND_LABELS = {
    "Kauai": (-159.50, 22.05),
    "Oahu": (-157.98, 21.47),
    "Molokai": (-157.00, 21.13),
    "Lanai": (-156.92, 20.82),
    "Maui": (-156.35, 20.80),
    "Hawaii": (-155.50, 19.60),
}
ISLAND_ZOOM_BBOX = {
    "Kauai": (-159.85, -159.20, 21.80, 22.30),
    "Oahu": (-158.35, -157.60, 21.15, 21.78),
    "Molokai": (-157.40, -156.65, 20.98, 21.30),
    "Lanai": (-157.12, -156.75, 20.65, 20.98),
    "Maui": (-156.78, -155.92, 20.45, 21.10),
    "Hawaii": (-156.15, -154.70, 18.85, 20.35),
}


def load_grid(npz_path: Path, meta_path: Path):
    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    K = data["K"]
    land = data["land_mask"]
    months = data["months"]
    ii, jj = np.nonzero(land)
    d = float(meta["center_distance_m"])
    dx = np.sqrt(3.0) / 2.0 * d
    xmin, ymax = map(float, meta["origin_xy_m"])
    ymin = ymax - float(meta["extent_m"][1])
    X = xmin + dx * jj
    Y = ymax - (ii - jj / 2.0) * d
    return K, land, months, meta, ii, jj, X, Y, xmin, ymin


def _to_km(meta, lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    x, y = transformer.transform(lon, lat)
    xmin, ymax = map(float, meta["origin_xy_m"])
    ymin = ymax - float(meta["extent_m"][1])
    return (x - xmin) / 1000.0, (y - ymin) / 1000.0


def make_norm(values, kind: str):
    """Build a shared color normalization from the non-negative values."""
    flat = np.asarray(values, dtype=np.float64)
    flat = flat[np.isfinite(flat)]
    positive = flat[flat > 0]
    if kind == "linear":
        vmax = max(float(flat.max()), 1e-6)
        return Normalize(vmin=0.0, vmax=vmax)
    if kind == "log":
        vmin = max(float(np.percentile(positive, 1)), 1e-3) if positive.size else 1e-3
        vmax = max(float(np.percentile(positive, 99.5)), vmin * 10.0) if positive.size else 1.0
        return LogNorm(vmin=vmin, vmax=vmax)
    vmax = max(float(np.percentile(positive, 99)), 1e-6) if positive.size else 1e-6
    return PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)


def _plot_values(values, norm):
    if isinstance(norm, LogNorm):
        return np.where(values > 0, values, np.nan)
    return values


def _norm_label(kind: str, norm) -> str:
    if kind == "power":
        return f"power norm (gamma=0.35, vmax={norm.vmax:.3g})"
    if kind == "log":
        return f"log norm ({norm.vmin:.3g}..{norm.vmax:.3g})"
    return f"linear (0..{norm.vmax:.3g})"


def _style(ax, meta):
    ax.set_xlim(0.0, float(meta["extent_m"][0]) / 1000.0)
    ax.set_ylim(0.0, float(meta["extent_m"][1]) / 1000.0)
    ax.set_aspect("equal")
    ax.set_xlabel("East from grid origin (km)")
    ax.set_ylabel("North from grid origin (km)")
    ax.grid(color="white", linewidth=0.3, alpha=0.25)
    ax.set_facecolor("#f4f4f4")


def plot_overview(ax, x, y, values, norm, title, meta, kind, annotate=False, s=0.7):
    scatter = ax.scatter(
        x, y, c=_plot_values(values, norm), s=s, cmap="viridis", norm=norm,
        linewidths=0, rasterized=True,
    )
    _style(ax, meta)
    ax.set_title(title, fontsize=10)
    if annotate:
        for name, (lon, lat) in ISLAND_LABELS.items():
            lx, ly = _to_km(meta, lon, lat)
            ax.text(lx, ly, name, fontsize=8, color="black", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    return scatter


def make_monthly(output: Path, x, y, land_values, norm, kind, meta, months):
    for index, month in enumerate(months):
        fig, ax = plt.subplots(figsize=(13, 10), layout="constrained")
        scatter = plot_overview(
            ax, x, y, land_values[index], norm,
            f"Hawaii six-island K (placeholder, uncalibrated) - month {int(month):02d}",
            meta, kind, annotate=True,
        )
        fig.colorbar(scatter, ax=ax, label=f"K per hex, {_norm_label(kind, norm)}", shrink=0.85)
        fig.suptitle("Albers Equal Area; 100 m hex-model-new lattice; exact area integration")
        fig.savefig(output / f"k_hawaii6_month{int(month):02d}_{kind}.png", dpi=170)
        plt.close(fig)
        print(f"  wrote month {int(month):02d}", flush=True)


def make_monthly_panels(output: Path, x, y, land_values, norm, kind, meta, months):
    fig, axes = plt.subplots(3, 4, figsize=(22, 15), layout="constrained")
    for index, (ax, month) in enumerate(zip(axes.ravel(), months)):
        scatter = plot_overview(
            ax, x, y, land_values[index], norm,
            f"month {int(month):02d}", meta, kind, annotate=False, s=0.35,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.colorbar(scatter, ax=axes, label=f"K per hex, {_norm_label(kind, norm)}", shrink=0.7)
    fig.suptitle(
        "Hawaii six-island K, monthly climatology "
        f"(Albers, 100 m hex, {_norm_label(kind, norm)})",
        fontsize=16,
    )
    fig.savefig(output / f"k_hawaii6_monthly_panels_{kind}.png", dpi=150)
    plt.close(fig)
    print("  wrote monthly panels", flush=True)


def make_annual_mean(output: Path, x, y, mean, norm, kind, meta):
    fig, ax = plt.subplots(figsize=(13, 10), layout="constrained")
    scatter = plot_overview(
        ax, x, y, mean, norm,
        "Hawaii six-island K, annual mean (placeholder, uncalibrated)",
        meta, kind, annotate=True,
    )
    fig.colorbar(scatter, ax=ax, label=f"Mean K per hex, {_norm_label(kind, norm)}", shrink=0.85)
    fig.savefig(output / f"k_hawaii6_annual_mean_{kind}.png", dpi=170)
    plt.close(fig)
    print("  wrote annual mean", flush=True)


def make_island_zooms(output: Path, x, y, mean, norm, kind, meta):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    for ax, (name, (lon0, lon1, lat0, lat1)) in zip(axes.ravel(), ISLAND_ZOOM_BBOX.items()):
        corners = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]
        projected = [_to_km(meta, lon, lat) for lon, lat in corners]
        zx = [p[0] for p in projected]
        zy = [p[1] for p in projected]
        xlim = (min(zx) - 2.0, max(zx) + 2.0)
        ylim = (min(zy) - 2.0, max(zy) + 2.0)
        inside = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        values = _plot_values(mean[inside], norm)
        scatter = ax.scatter(
            x[inside], y[inside], c=values, s=4.0, cmap="viridis",
            norm=norm, linewidths=0, rasterized=True,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.grid(color="white", linewidth=0.3, alpha=0.25)
        ax.set_facecolor("#f4f4f4")
    fig.colorbar(scatter, ax=axes, label=f"Mean K per hex, {_norm_label(kind, norm)}", shrink=0.7)
    fig.suptitle(
        f"Island zoom: annual mean K (placeholder, uncalibrated; {_norm_label(kind, norm)})",
        fontsize=16,
    )
    fig.savefig(output / f"k_hawaii6_islands_annual_mean_{kind}.png", dpi=160)
    plt.close(fig)
    print("  wrote island zooms", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--norm", choices=("power", "linear", "log"), default="power")
    args = parser.parse_args(argv)

    if not args.npz.exists() or not args.meta.exists():
        raise FileNotFoundError(f"missing {args.npz} or {args.meta}")
    (K, land, months, meta, ii, jj, X, Y, xmin, ymin) = load_grid(args.npz, args.meta)
    print(f"loaded K={K.shape}, land_hex={int(land.sum())}, norm={args.norm}", flush=True)
    x = (X - xmin) / 1000.0
    y = (Y - ymin) / 1000.0
    land_values = K[:, ii, jj]
    mean = land_values.mean(axis=0)
    monthly_norm = make_norm(land_values, args.norm)
    annual_norm = make_norm(mean, args.norm)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing heatmaps to {args.out_dir}", flush=True)
    make_monthly(args.out_dir, x, y, land_values, monthly_norm, args.norm, meta, months)
    make_monthly_panels(args.out_dir, x, y, land_values, monthly_norm, args.norm, meta, months)
    make_annual_mean(args.out_dir, x, y, mean, annual_norm, args.norm, meta)
    make_island_zooms(args.out_dir, x, y, mean, annual_norm, args.norm, meta)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
