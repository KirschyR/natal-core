"""Stochastic event-order experiment: mean trajectory with 95% CI.

Runs ``--runs`` independent stochastic replicates of the single-population model
for R-S-A and S-A-R, with drive conversion ``--dc`` (default 0.8), and plots the
per-category mean with its 95% CI (mean +/- 1.96 SEM) in a 4x2 grid under
``event-seq/img/``.

Usage:
    python event-seq/stochastic_test.py [--runs 100] [--ticks 60] [--dc 0.8]
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
ORDERS = ["R-S-A", "S-A-R"]
# Initial-population scaling factor: every per-group count is multiplied by this
# (the density reference scales too, so normalized dynamics are unchanged).
SCALING_FACTOR = 10.0


def run_replicates(order: str, runs: int, ticks: int, dc: float,
                   scaling_factor: float) -> dict[str, np.ndarray]:
    """Return ``{category: (runs, ticks+1)}`` trajectories."""
    series = {key: np.zeros((runs, ticks + 1)) for key in m.CATEGORIES}
    for r in range(runs):
        history = m.Simulation(
            dc=dc, stochastic=True, seed=r, scale=scaling_factor
        ).run(order, ticks)
        for key in m.CATEGORIES:
            series[key][r] = [h[key] for h in history]
    return series


def plot_order(order: str, series: dict[str, np.ndarray], runs: int, dc: float,
               scaling_factor: float) -> Path:
    x = np.arange(series[m.CATEGORIES[0]].shape[1])
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True)
    for ax, key in zip(axes.ravel(), m.CATEGORIES):
        traj = series[key]
        mean = traj.mean(axis=0)
        ci = 1.96 * traj.std(axis=0, ddof=1) / np.sqrt(runs)
        ax.plot(x, mean, linewidth=1.6, label="mean")
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.3, label="95% CI")
        ax.set_title(key)
        ax.set_ylabel("count")
        ax.set_xlim(0, x[-1])
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tick")
    fig.suptitle(
        f"event order: {order} | stochastic (n={runs}) | dc={dc} | sf={scaling_factor:g}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    out = IMG / f"order_{order.replace('-', '')}_stoch_sf{scaling_factor:g}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--dc", type=float, default=0.8)
    ap.add_argument("--scaling-factor", type=float, default=SCALING_FACTOR)
    args = ap.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    for order in ORDERS:
        series = run_replicates(order, args.runs, args.ticks, args.dc, args.scaling_factor)
        out = plot_order(order, series, args.runs, args.dc, args.scaling_factor)
        final = series["A_F_D"] + series["A_M_D"]
        adult = final + series["A_F_W"] + series["A_M_W"]
        frac = final[:, -1] / np.where(adult[:, -1] > 0, adult[:, -1], np.nan)
        fixed = int(np.sum(frac > 0.99))
        lost = int(np.sum(frac < 0.01))
        print(
            f"{order}: wrote {out.name}  mean final adult drive frac={np.nanmean(frac):.4f} "
            f"fixed(>0.99)={fixed}/{args.runs} lost(<0.01)={lost}/{args.runs}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
