"""Explicit-``rr`` TARE test in the deterministic spatial event-order model.

The paper's 5-genotype TARE (``ww, dw, dd, rw, dr``) folds the recessive-lethal
homozygote ``rr`` away: the toxin shows up only as reduced offspring totals in
``tare_mats`` (see ``HANDOFF`` §5#14).  This module makes ``rr`` **explicit** as
a sixth genotype and tests, with the **deterministic** spatial model, whether the
explicit class changes the wave behaviour or enables a drive/wildtype coexistence
("chasing").

``rr`` transmission is recovered from the paper operator: for every parental
pair the surviving proportions over the five genotypes are taken verbatim, and
the **missing mass** ``1 - sum_k mats5[k, i, j]`` is assigned to ``rr`` (this is
exactly the recessive-lethal class, verified in ``HANDOFF`` §5#14).  ``rr`` is
then lethal (``viability_rr = 0``) applied at a configurable age; whether it is
counted in the density feedback before it dies is the knob that distinguishes
"pure embryo lethality" (equivalent to the 5-genotype collapse) from
"late-acting toxin" (suppression).

Outputs under ``event-seq/img/spatial/``: a comparison figure, an interactive
hex HTML of the carrier wave, and a JSON report.

Usage:
    python event-seq/tare_rr.py [--ticks 300] [--selftest]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import age_model as m  # noqa: E402
import spatial_test as sp  # noqa: E402

IMG = HERE / "img" / "spatial"

# 6-genotype canonical order.
GENOTYPES6 = ("dd", "dw", "ww", "rw", "dr", "rr")
DD, DW, WW, RW, DR, RR = range(6)
N_Z6 = 6

ROWS, COLS = 41, 41
KERNEL_SIZE = 7
MEAN_DISPERSAL = 0.5
MIGRATION_RATE = 1.0
RELEASE_RADIUS = 3.0
RELEASE_MULTIPLE = 2.0
CARRIER_GENOTYPES = (DD, DW, DR)
ORDERS = ("R-S-A", "S-A-R")
TICKS = 600


def tare_mats6(gcr: float = 1.0, eecr: float = 1.0, ddfitness: float = 1.0) -> np.ndarray:
    """Return the 6-genotype TARE operator, shape ``(36, 6)``.

    The five surviving genotypes reuse the paper's ``tare_mats``; the ``rr``
    block is the missing (recessive-lethal) mass so that each parental pair's
    offspring proportions sum to one.
    """
    mats5 = m.tare_mats(gcr=gcr, eecr=eecr, ddfitness=ddfitness)
    out = np.zeros((N_Z6 * N_Z6, N_Z6))
    for k in range(5):
        for i in range(5):
            for j in range(5):
                out[k * N_Z6 + i, j] = mats5[k * 5 + i, j]
    for i in range(5):
        for j in range(5):
            surviving = mats5[[i, 5 + i, 10 + i, 15 + i, 20 + i], j].sum()
            out[RR * N_Z6 + i, j] = 1.0 - surviving
    return out


def kernel_and_valid_sum() -> tuple[np.ndarray, np.ndarray]:
    """Return the off-centre migration kernel and its per-deme normalizer ``Z``."""
    kernel = sp.kernel_off_center(KERNEL_SIZE, MEAN_DISPERSAL)
    valid = fftconvolve(np.ones((ROWS, COLS)), kernel, mode="same")
    return kernel, valid


def initial_state(radius: float, release_multiple: float, scale: float = 1.0) -> np.ndarray:
    """WT equilibrium plus a central, both-sex release of drive homozygotes."""
    data = np.zeros((N_Z6, m.N_SEX, m.N_AGES, ROWS, COLS))
    data[WW, 0] = m.INITIAL_FEMALE[:, None, None]
    data[WW, 1] = m.INITIAL_MALE[:, None, None]
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    mask = (rr - (ROWS - 1) / 2.0) ** 2 + (cc - (COLS - 1) / 2.0) ** 2 <= radius**2
    data[DD, 0, m.NEW_ADULT_AGE, mask] += release_multiple * m.INITIAL_FEMALE[m.NEW_ADULT_AGE]
    data[DD, 1, m.NEW_ADULT_AGE, mask] += release_multiple * m.INITIAL_MALE[m.NEW_ADULT_AGE]
    return data * scale


def _reproduction(
    data: np.ndarray, mats: np.ndarray, fecundity_dd: float, *, drop_rr: bool
) -> np.ndarray:
    """Newborn genotype counts ``(N_Z6, rows, cols)`` from the adult state.

    ``drop_rr`` removes the recessive-lethal class at reproduction, which makes
    the explicit model reduce exactly to the paper's 5-genotype collapse.
    """
    fecundity = np.ones(N_Z6)
    fecundity[DD] = fecundity_dd
    female_w = data[:, 0, m.NEW_ADULT_AGE:].sum(axis=1) * fecundity[:, None, None]
    male_w = data[:, 1, m.NEW_ADULT_AGE:].sum(axis=1) * fecundity[:, None, None]
    total_female = female_w.sum(axis=0)
    total_male = male_w.sum(axis=0)
    female_freq = np.divide(female_w, total_female, out=np.zeros_like(female_w), where=total_female > 0)
    male_freq = np.divide(male_w, total_male, out=np.zeros_like(male_w), where=total_male > 0)
    m3 = mats.reshape(N_Z6, N_Z6, N_Z6)
    prop = np.einsum("irc,jrc,kij->krc", female_freq, male_freq, m3)
    if drop_rr:
        prop[RR] = 0.0
    return m.EGGS * total_female[None, :, :] * prop


def _event_R(data: np.ndarray, newborns: np.ndarray) -> None:
    data[:, 0, 0] = m.SEX_RATIO * newborns
    data[:, 1, 0] = (1.0 - m.SEX_RATIO) * newborns


def _event_S(
    data: np.ndarray, density_ref: float, viability_dd: float,
    viability_rr: float, rr_age: int,
) -> None:
    j0 = data[:, :, 0].sum(axis=(0, 1))
    j1 = data[:, :, 1].sum(axis=(0, 1))
    r = (j0 + 5.0 * j1) / density_ref
    s1 = m.LAMBDA / ((m.LAMBDA - 1.0) * r + 1.0) * 2.0 / m.SF / m.EGGS
    data[:, :, 0] = data[:, :, 0] * s1[None, None, :, :]
    data[:] = data * m.SURVIVAL[:, :, None, None]
    data[DD, :, m.NEW_ADULT_AGE - 1] *= viability_dd
    if viability_rr < 1.0:
        data[RR, rr_age] *= viability_rr


def _event_A(data: np.ndarray) -> None:
    data[:, :, 1:] = data[:, :, :-1].copy()
    data[:, :, 0] = 0.0


def _event_M(data: np.ndarray, kernel: np.ndarray, valid_sum: np.ndarray) -> None:
    adults = data[:, :, m.NEW_ADULT_AGE:]
    shape = adults.shape
    planes = adults.reshape(-1, adults.shape[-2], adults.shape[-1])
    conv = fftconvolve(planes / valid_sum[None, :, :], kernel[None, :, :], mode="same", axes=(1, 2))
    data[:, :, m.NEW_ADULT_AGE:] = conv.reshape(shape)


def _fraction(data: np.ndarray, genotypes: tuple[int, ...]) -> np.ndarray:
    total = data[:, :, 1:].sum(axis=(0, 1, 2))
    part = data[list(genotypes), :, :][:, :, 1:].sum(axis=(0, 1, 2))
    return np.divide(part, total, out=np.zeros_like(total), where=total > 0)


def run_replicate(
    order: str, ticks: int, kernel: np.ndarray, valid_sum: np.ndarray, *,
    viability_rr: float = 0.0, rr_age: int = 0, drop_rr: bool = False,
    viability_dd: float = sp.VIABILITY_DD, fecundity_dd: float = sp.FECUNDITY_DD,
    release_multiple: float = RELEASE_MULTIPLE, release_radius: float = RELEASE_RADIUS,
    keep_grids: bool = False,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Deterministic explicit-``rr`` spatial run; returns ``(carrier_grids, series)``."""
    mats = tare_mats6()
    data = initial_state(release_radius, release_multiple)
    density_ref = m.DENSITY_REF
    grids = np.zeros((ticks + 1, ROWS, COLS)) if keep_grids else None
    series = {k: np.zeros(ticks + 1) for k in ("carrier", "ww", "rr", "total")}

    def record(t: int) -> None:
        series["carrier"][t] = _fraction(data, CARRIER_GENOTYPES).mean()
        series["ww"][t] = _fraction(data, (WW,)).mean()
        series["rr"][t] = _fraction(data, (RR,)).mean()
        series["total"][t] = data[:, :, 1:].sum()
        if grids is not None:
            grids[t] = _fraction(data, CARRIER_GENOTYPES)

    record(0)
    for t in range(1, ticks + 1):
        newborns = _reproduction(data, mats, fecundity_dd, drop_rr=drop_rr)
        for name in order.split("-"):
            if name == "R":
                _event_R(data, newborns)
            elif name == "S":
                _event_S(data, density_ref, viability_dd, viability_rr, rr_age)
            else:
                _event_A(data)
        _event_M(data, kernel, valid_sum)
        record(t)
    return grids, series


def _run5(order: str, ticks: int, kernel: np.ndarray, valid_sum: np.ndarray) -> dict[str, np.ndarray]:
    """5-genotype TARE reference via ``spatial_test`` primitives (both-sex release)."""
    mats = m.tare_mats()
    data = np.zeros((m.N_Z, m.N_SEX, m.N_AGES, ROWS, COLS))
    data[m.CORD_PURE_W, 0] = m.INITIAL_FEMALE[:, None, None]
    data[m.CORD_PURE_W, 1] = m.INITIAL_MALE[:, None, None]
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    mask = (rr - (ROWS - 1) / 2.0) ** 2 + (cc - (COLS - 1) / 2.0) ** 2 <= RELEASE_RADIUS**2
    data[m.CORD_RELEASE_GENOTYPE, 0, m.NEW_ADULT_AGE, mask] += RELEASE_MULTIPLE * m.INITIAL_FEMALE[m.NEW_ADULT_AGE]
    data[m.CORD_RELEASE_GENOTYPE, 1, m.NEW_ADULT_AGE, mask] += RELEASE_MULTIPLE * m.INITIAL_MALE[m.NEW_ADULT_AGE]
    series = {k: np.zeros(ticks + 1) for k in ("carrier", "ww", "total")}

    def _frac(genotypes: tuple[int, ...]) -> float:
        total = data[:, :, 1:].sum(axis=(0, 1, 2))
        part = data[list(genotypes), :, :][:, :, 1:].sum(axis=(0, 1, 2))
        return float(np.divide(part, total, out=np.zeros_like(total), where=total > 0).mean())

    for t in range(ticks + 1):
        if t > 0:
            newborns = sp._reproduction(data, mats, sp.FECUNDITY_DD)
            for name in order.split("-"):
                if name == "R":
                    sp._event_R(data, newborns)
                elif name == "S":
                    sp._event_S(data, m.DENSITY_REF, sp.VIABILITY_DD)
                else:
                    sp._event_A(data)
            sp._event_M(data, kernel, valid_sum, MIGRATION_RATE)
        series["carrier"][t] = _frac((0, 1, 4))
        series["ww"][t] = _frac((m.CORD_PURE_W,))
        series["total"][t] = float(data[:, :, 1:].sum())
    return series


def _hex_geometry(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    from spatial_html import hex_geometry

    vertices, bounds = hex_geometry(rows, cols)
    return vertices, bounds


def write_carrier_html(
    grids_by_order: dict[str, np.ndarray], out: Path, title: str, stride: int = 5
) -> Path:
    """Interactive two-panel hex HTML of the carrier fraction (R-S-A | S-A-R)."""
    from matplotlib.cm import ScalarMappable  # noqa: E402
    from matplotlib.collections import PolyCollection  # noqa: E402
    from matplotlib.colors import Normalize  # noqa: E402

    vertices, bounds = _hex_geometry(ROWS, COLS)
    xmin, xmax, ymin, ymax = bounds
    ticks = next(iter(grids_by_order.values())).shape[0]
    frame_ticks = list(range(0, ticks, stride)) + [ticks - 1]
    frames = []
    for tick in frame_ticks:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=90)
        for ax, order in zip(axes, ORDERS):
            collection = PolyCollection(
                vertices, array=grids_by_order[order][tick].ravel(),
                cmap="viridis", clim=(0.0, 1.0), rasterized=True,
            )
            ax.add_collection(collection)
            ax.set_xlim(xmin - 0.8, xmax + 0.8)
            ax.set_ylim(ymin - 0.8, ymax + 0.8)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(order)
        fig.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis"),
                     ax=list(axes), fraction=0.03, pad=0.03, label="carrier frac")
        fig.suptitle(f"{title}\ntick = {tick}", fontsize=8.5)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    encoded = ",\n".join(f'"data:image/png;base64,{f}"' for f in frames)
    tick_list = ",".join(str(t) for t in frame_ticks)
    safe = title.replace("\n", "<br>")
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>explicit rr TARE</title>
<style>body{{font-family:sans-serif;text-align:center;margin:8px}}img{{max-width:96vw}}
input[type=range]{{width:60%}}</style></head><body>
<h3>{safe}</h3>
<img id="frame" src="data:image/png;base64,{frames[0]}">
<div><button id="play">Play</button>
<input id="slider" type="range" min="0" max="{len(frames)-1}" value="0">
<span id="lbl">tick {frame_ticks[0]}</span></div>
<script>
const frames=[{encoded}];const ticks=[{tick_list}];const img=document.getElementById("frame");
const slider=document.getElementById("slider");const label=document.getElementById("lbl");
const play=document.getElementById("play");let timer=null;
function show(i){{i=Math.max(0,Math.min(frames.length-1,i));img.src=frames[i];slider.value=i;label.textContent="tick "+ticks[i];}}
function stop(){{if(timer){{clearInterval(timer);timer=null;play.textContent="Play";}}}}
slider.addEventListener("input",()=>{{stop();show(Number(slider.value));}});
play.addEventListener("click",()=>{{if(timer){{stop();return;}}play.textContent="Pause";
timer=setInterval(()=>show((Number(slider.value)+1)%frames.length),150);}});
</script></body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


def _plot_compare(data: dict[str, dict[str, np.ndarray]], out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(next(iter(data.values()))["carrier"]))
    for label, series in data.items():
        axes[0].plot(x, series["carrier"], label=label)
        axes[1].plot(x, series["ww"], label=label)
        axes[2].plot(x, series["total"], label=label)
    for ax, title in zip(axes, ("carrier fraction", "wild-type ww fraction", "non-egg population")):
        ax.set_title(title)
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Explicit rr vs 5-genotype TARE (deterministic, avd=0.5, both-sex release)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def selftest() -> int:
    """Validate the 6-genotype operator and the explicit-rr construction."""
    mats5 = m.tare_mats()
    mats6 = tare_mats6()
    # rr block = missing mass; each parental pair's offspring sums to 1.
    for i in range(5):
        for j in range(5):
            col = mats6[[i, 6 + i, 12 + i, 18 + i, 24 + i, 30 + i], j]
            assert abs(col.sum() - 1.0) < 1e-12, (i, j, col.sum())
            assert abs(mats6[30 + i, j] - (1.0 - mats5[[i, 5 + i, 10 + i, 15 + i, 20 + i], j].sum())) < 1e-12
    print("[ok] 6-genotype operator: every parental pair sums to 1, rr = missing mass")

    # Dropping rr reproduces the 5-genotype reference exactly (deterministic).
    kernel, valid = kernel_and_valid_sum()
    ref = _run5("R-S-A", 120, kernel, valid)
    _, series = run_replicate("R-S-A", 120, kernel, valid, drop_rr=True)
    for key in ("carrier", "ww", "total"):
        assert np.allclose(series[key], ref[key], rtol=1e-9, atol=1e-7), key
    print("[ok] drop_rr=True matches the 5-genotype reference (rtol<1e-9)")

    # Explicit rr (lethal, age 0) is deterministic and finite.
    _, s0 = run_replicate("R-S-A", 120, kernel, valid, viability_rr=0.0, rr_age=0)
    assert np.isfinite(s0["carrier"]).all() and np.isfinite(s0["total"]).all()
    print("[ok] explicit rr (viability 0) run is finite and deterministic")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    IMG.mkdir(parents=True, exist_ok=True)
    kernel, valid = kernel_and_valid_sum()

    comparison: dict[str, dict[str, np.ndarray]] = {}
    grids_by_order: dict[str, np.ndarray] = {}
    for order in ORDERS:
        comparison[f"5-gen {order}"] = _run5(order, args.ticks, kernel, valid)
        for tag, rr_age in (("rr@age0", 0), ("rr@age1", 1)):
            comparison[f"{tag} {order}"] = run_replicate(
                order, args.ticks, kernel, valid, viability_rr=0.0, rr_age=rr_age
            )[1]
        grids, series = run_replicate(order, args.ticks, kernel, valid,
                                      viability_rr=0.0, rr_age=0, keep_grids=True)
        if grids is not None:
            grids_by_order[order] = grids

    png = _plot_compare(comparison, IMG / "tare_rr_compare.png")
    html = write_carrier_html(
        grids_by_order, IMG / "tare_rr_carrier.html",
        "Explicit rr TARE (carrier fraction; deterministic, avd=0.5)",
    )
    report = {
        "params": {"ticks": args.ticks, "avd": MEAN_DISPERSAL, "release_multiple": RELEASE_MULTIPLE,
                   "release": "both-sex dd", "viability_rr": 0.0, "rr_age": 0},
        "final": {k: {"carrier": float(v["carrier"][-1]), "ww": float(v["ww"][-1]),
                      "total": float(v["total"][-1])} for k, v in comparison.items()},
    }
    (IMG / "tare_rr_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {png}")
    print(f"wrote {html}")
    for k, v in report["final"].items():
        print(f"  {k}: carrier={v['carrier']:.3f} ww={v['ww']:.3f} total={v['total']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
