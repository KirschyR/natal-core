"""Run the single-population event-order test and plot population counts.

For every tick order this produces one figure with twelve subplots (4 rows x 3
columns) -- ``{A,J,E}_{F,M}_{W,D}`` (rows = sex/type, columns = age) -- written
to ``event-seq/img/order_<ORDER>.png``, plus one total-population figure
overlaying the orders (eggs excluded), ``img/total_deterministic.png``.  The
figure title records the event order.

Usage:
    python event-seq/run_orders.py [--ticks 261]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import age_model as m  # noqa: E402

IMG = HERE / "img"

# Only these two orders keep the density-regulation gate (S) between writing
# age-0 (R) and aging (A); the other four permutations let the unregulated
# age-0 pool leak into age 1 and diverge.
_CANONICAL = ["R-S-A", "S-A-R"]


def _all_orders() -> list[str]:
    return list(_CANONICAL)


def _final_adult_drive_fraction(history: list[dict[str, float]]) -> float:
    last = history[-1]
    drive = last["A_F_D"] + last["A_M_D"]
    total = drive + last["A_F_W"] + last["A_M_W"]
    return drive / total if total > 0 else 0.0


def plot_order(order: str, history: list[dict[str, float]], ticks: int) -> Path:
    """Write one 4x3 figure for a single event order and return its path."""
    series = {key: np.array([h[key] for h in history]) for key in m.CATEGORIES}
    x = np.arange(len(history))

    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
    for ax, key in zip(axes.ravel(), m.CATEGORIES):
        ax.plot(x, series[key], linewidth=1.6)
        ax.set_title(key)
        ax.set_ylabel("count")
        ax.set_xlim(0, ticks)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("tick")
    fig.suptitle(f"event order: {order}", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    out = IMG / f"order_{order.replace('-', '')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def total_series(history: list[dict[str, float]]) -> np.ndarray:
    """Total population over time, eggs (age 0) excluded."""
    return np.array([sum(h[key] for key in m.TOTAL_CATEGORIES) for h in history])


def plot_total_deterministic(
    histories: dict[str, list[dict[str, float]]], ticks: int
) -> Path:
    """Overlay the deterministic total population of each order (eggs excluded)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for order, history in histories.items():
        ax.plot(np.arange(len(history)), total_series(history), linewidth=1.6, label=order)
    ax.set_xlabel("tick")
    ax.set_ylabel("total population (eggs excluded)")
    ax.set_title("deterministic total population (eggs excluded)")
    ax.set_xlim(0, ticks)
    ax.grid(alpha=0.3)
    ax.legend()
    out = IMG / "total_deterministic.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=261)
    ap.add_argument("--orders", type=str, default="", help="comma-separated subset, e.g. R-S-A,S-A-R")
    args = ap.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    orders = [o.strip() for o in args.orders.split(",") if o.strip()] or _all_orders()
    histories: dict[str, list[dict[str, float]]] = {}
    for order in orders:
        history = m.run(order, args.ticks)
        histories[order] = history
        out = plot_order(order, history, args.ticks)
        frac = _final_adult_drive_fraction(history)
        print(f"{order}: wrote {out.name}  final adult drive fraction = {frac:.4f}")
    total_out = plot_total_deterministic(histories, args.ticks)
    print(f"wrote {total_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
