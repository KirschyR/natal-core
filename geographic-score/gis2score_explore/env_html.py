"""Interactive hex HTML for the Phase-0 H-insertion comparison.

Renders, per recorded tick, three hex panels sharing the time slider:

    [ synthetic suitability H ] [ idea 1: migration bias ] [ idea 2: capacity scaling ]

Each panel is a per-deme total-population heatmap; panel 1 is the static field H
so the spatial pattern of the two simulation modes can be compared against it by
eye.  Populated via a self-contained base64-PNG HTML (no external assets).

Usage:
    python geographic-score/gis2score_explore/env_html.py --ticks 200 --stride 1
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "simple_spatial_model_ref"))
import env_spatial as es  # noqa: E402

IMG = HERE / "img"


def _panel(ax, vertices, values, bounds, cmap, clim, title):
    collection = PolyCollection(vertices, array=values.ravel(), cmap=cmap, clim=clim, rasterized=True)
    ax.add_collection(collection)
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin - 0.8, xmax + 0.8)
    ax.set_ylim(ymin - 0.8, ymax + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    return collection


def build_frames(H, bias, capacity, times, stride, out):
    vertices, bounds = es.hex_geometry(es.ROWS, es.COLS)
    vmax = float(np.percentile(np.concatenate([bias, capacity]), 99.5))
    vmax = max(vmax, 1.0)
    idx = list(range(0, len(times), stride))
    if idx[-1] != len(times) - 1:
        idx.append(len(times) - 1)
    frames = []
    for k in idx:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=80)
        _panel(axes[0], vertices, H, bounds, "magma", (0.0, 1.0), "suitability H (4 peaks)")
        _panel(axes[1], vertices, bias[k], bounds, "viridis", (0.0, vmax), "idea 1: migration bias (H soft)")
        _panel(axes[2], vertices, capacity[k], bounds, "viridis", (0.0, vmax), "idea 2: capacity scaling (H*K)")
        fig.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma"), ax=axes[0], fraction=0.03, pad=0.02)
        fig.colorbar(ScalarMappable(norm=Normalize(0.0, vmax), cmap="viridis"), ax=axes[2], fraction=0.03, pad=0.02)
        fig.suptitle(
            f"per-deme total population  |  tick = {times[k]}  |  grid {es.ROWS}x{es.COLS}, "
            f"avd={es.MEAN_DISPERSAL}, rate={es.MIGRATION_RATE}, beta=1",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    encoded = ",\n".join(f'"data:image/png;base64,{f}"' for f in frames)
    labels = ",".join(str(times[k]) for k in idx)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>gis2score: H insertion comparison</title>
<style>body{{font-family:sans-serif;text-align:center;margin:8px}}img{{max-width:98vw}}
input[type=range]{{width:60%}}</style></head><body>
<h3>H as migration bias vs H as capacity scaling (per-deme total population)</h3>
<img id="frame" src="data:image/png;base64,{frames[0]}">
<div><button id="play">Play</button>
<input id="slider" type="range" min="0" max="{len(frames)-1}" value="0">
<span id="lbl">tick {times[idx[0]]}</span></div>
<script>
const frames=[{encoded}];const ticks=[{labels}];const img=document.getElementById("frame");
const slider=document.getElementById("slider");const label=document.getElementById("lbl");
const play=document.getElementById("play");let timer=null;
function show(i){{i=Math.max(0,Math.min(frames.length-1,i));img.src=frames[i];slider.value=i;label.textContent="tick "+ticks[i];}}
function stop(){{if(timer){{clearInterval(timer);timer=null;play.textContent="Play";}}}}
slider.addEventListener("input",()=>{{stop();show(Number(slider.value));}});
play.addEventListener("click",()=>{{if(timer){{stop();return;}}play.textContent="Pause";
timer=setInterval(()=>show((Number(slider.value)+1)%frames.length),200);}});
</script></body></html>"""
    out.write_text(html, encoding="utf-8")
    return out, vmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    H = es.make_H()
    kernel = es.kernel_off_center(es.KERNEL_SIZE, es.MEAN_DISPERSAL)
    bias, times = es.run("bias", H, ticks=args.ticks, kernel=kernel, record_every=1)
    capacity, _ = es.run("capacity", H, ticks=args.ticks, kernel=kernel, record_every=1)
    out, vmax = build_frames(H, bias, capacity, times, args.stride, IMG / "h_insertion_compare.html")
    np.savez(IMG / "h_insertion_compare.npz", H=H, bias=bias, capacity=capacity, times=np.array(times))
    print(f"wrote {out}  (density vmax={vmax:.1f})")
    print(f"t=0 totals: bias={bias[0].sum():.1f} capacity={capacity[0].sum():.1f}")
    print(f"t={times[-1]} totals: bias={bias[-1].sum():.1f} capacity={capacity[-1].sum():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
