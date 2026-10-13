"""Hexagonal-grid interactive HTML for the spatial event-order model.

Renders each tick as a hexagonal-grid heatmap (two panels: R-S-A | S-A-R) with
matplotlib, embeds the frames as base64 PNGs in a single self-contained HTML
with a time slider and a play/pause button, and writes:

* ``spatial_event_order.html``        -- deterministic
* ``spatial_event_order_sf10.html``   -- stochastic, mean over ``--runs``
* ``spatial_event_order_sf100.html``  -- stochastic, mean over ``--runs``

The heatmap is the per-deme adult ``Drive|Drive`` fraction (fixed 0-1 color
scale).  For stochastic runs the displayed value at each tick is the mean over
the replicates.

Usage:
    python event-seq/spatial_html.py [--ticks 120] [--runs 100]
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_test as sp  # noqa: E402


def hex_geometry(rows: int, cols: int, size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return pointy-top hexagon vertices ``(rows*cols, 6, 2)`` and bounding box.

    Matches natal ``HexGrid.to_xy``: ``x = i + 0.5*j``, ``y = (sqrt(3)/2)*j``
    (``i`` = row, ``j`` = col), so the hex distance
    ``di^2 + dj^2 + di*dj`` maps to Euclidean distance.  Row-major order matches
    ``grid.ravel()``.
    """
    sqrt3 = np.sqrt(3.0)
    vertices = np.empty((rows * cols, 6, 2))
    angles = np.deg2rad(np.arange(6) * 60.0 + 30.0)  # pointy-top
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    k = 0
    for i in range(rows):
        for j in range(cols):
            cx = size * (i + 0.5 * j)
            cy = size * (sqrt3 / 2.0) * j
            vertices[k, :, 0] = cx + size * np.cos(angles)
            vertices[k, :, 1] = cy + size * np.sin(angles)
            xmin, xmax = min(xmin, cx - size), max(xmax, cx + size)
            ymin, ymax = min(ymin, cy - size), max(ymax, cy + size)
            k += 1
    return vertices, np.array([xmin, xmax, ymin, ymax])


def render_frame(
    grids: list[np.ndarray], tick: int, vertices: np.ndarray, bounds: np.ndarray,
    title: str, cmap: str = "viridis",
) -> str:
    """Render one tick (two hex panels) to a base64 PNG data URI."""
    xmin, xmax, ymin, ymax = bounds
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=90)
    for ax, grid, order in zip(axes, grids, sp.ORDERS):
        collection = PolyCollection(
            vertices, array=grid.ravel(), cmap=cmap, clim=(0.0, 1.0), rasterized=True
        )
        ax.add_collection(collection)
        ax.set_xlim(xmin - 0.8, xmax + 0.8)
        ax.set_ylim(ymin - 0.8, ymax + 0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(order)
    fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap),
        ax=list(axes), fraction=0.03, pad=0.03, label="Drive|Drive frac",
    )
    fig.suptitle(f"{title}\ntick = {tick}", fontsize=8.5)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_html(frames: list[str], title: str, out: Path, frame_ms: int) -> Path:
    """Write a self-contained HTML with a slider and play/pause."""
    encoded = ",\n".join(f'"data:image/png;base64,{frame}"' for frame in frames)
    safe_title = title.replace("\n", "<br>")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>spatial event-order</title>
<style>
  body {{ font-family: sans-serif; text-align: center; margin: 8px; }}
  img {{ max-width: 96vw; height: auto; }}
  #bar {{ margin-top: 6px; }}
  input[type=range] {{ width: 60%; vertical-align: middle; }}
</style>
</head>
<body>
<h3>{safe_title}</h3>
<img id="frame" alt="frame" src="data:image/png;base64,{frames[0]}">
<div id="bar">
  <button id="play">Play</button>
  <input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0">
  <span id="lbl">tick 0</span>
</div>
<script>
const frames = [
{encoded}
];
const img = document.getElementById("frame");
const slider = document.getElementById("slider");
const label = document.getElementById("lbl");
const playBtn = document.getElementById("play");
const FRAME_MS = {frame_ms};
let timer = null;
function show(i) {{
  i = Math.max(0, Math.min(frames.length - 1, i));
  img.src = frames[i];
  slider.value = i;
  label.textContent = "tick " + i;
}}
function stop() {{
  if (timer) {{ clearInterval(timer); timer = null; playBtn.textContent = "Play"; }}
}}
slider.addEventListener("input", () => {{ stop(); show(Number(slider.value)); }});
playBtn.addEventListener("click", () => {{
  if (timer) {{ stop(); return; }}
  playBtn.textContent = "Pause";
  timer = setInterval(() => show((Number(slider.value) + 1) % frames.length), FRAME_MS);
}});
</script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def _write_case(
    grids_by_order: dict[str, np.ndarray], vertices: np.ndarray, bounds: np.ndarray,
    ticks: int, title: str, out: Path, frame_ms: int,
) -> Path:
    frames = [
        render_frame([grids_by_order[order][tick] for order in sp.ORDERS],
                     tick, vertices, bounds, title)
        for tick in range(ticks + 1)
    ]
    return build_html(frames, title, out, frame_ms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--frame-ms", type=int, default=120)
    args = ap.parse_args()

    sp.IMG.mkdir(parents=True, exist_ok=True)
    kernel = sp.kernel_off_center(sp.KERNEL_SIZE, sp.MEAN_DISPERSAL)
    valid_sum = fftconvolve(np.ones((sp.ROWS, sp.COLS)), kernel, mode="same")
    vertices, bounds = hex_geometry(sp.ROWS, sp.COLS)

    def run(order: str, scale: float, seed: int, stochastic: bool) -> np.ndarray:
        grids, _ = sp.run_replicate(
            order, sp.ROWS, sp.COLS, args.ticks, kernel, valid_sum,
            scale=scale, stochastic=stochastic, seed=seed, keep_grids=True,
        )
        return grids if grids is not None else np.zeros(0)

    # deterministic
    det = {order: run(order, 1.0, 0, False) for order in sp.ORDERS}
    out = _write_case(
        det, vertices, bounds, args.ticks,
        "spatial event-order (adult Drive|Drive fraction): R-S-A vs S-A-R\n"
        + sp._params_title("\n", stochastic=False, scale=1.0),
        sp.IMG / "spatial_event_order.html", args.frame_ms,
    )
    print(f"wrote {out}")

    # stochastic: mean over runs
    for scale in sp.SCALING_FACTORS:
        means: dict[str, np.ndarray] = {}
        for order in sp.ORDERS:
            accumulator = None
            for r in range(args.runs):
                grids = run(order, scale, r, True)
                accumulator = grids if accumulator is None else accumulator + grids
            means[order] = accumulator / args.runs
        out = _write_case(
            means, vertices, bounds, args.ticks,
            "spatial event-order (adult Drive|Drive fraction; mean over "
            f"{args.runs} runs): R-S-A vs S-A-R\n"
            + sp._params_title("\n", stochastic=True, scale=scale),
            sp.IMG / f"spatial_event_order_sf{scale:g}.html", args.frame_ms,
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
