"""TARE chasing / nucleation-threshold experiment for the spatial event-order model.

Extends ``spatial_test`` with the paper's TARE drive (see ``age_model.tare_mats``)
and searches for a *chasing* regime: a drive wave that persists while wild-type
(WT) individuals remain spatially present, rather than a fast global sweep.

Findings this script reproduces (deterministic, both-sex central release):

* TARE at ``gcr = eecr = 1`` has a **critical release** (nucleation threshold):
  below it the drive is lost, above it a drive wave nucleates and eventually
  sweeps.  Homing has no such threshold over the tested releases.
* The event order shifts the threshold: ``R-S-A`` (natal) nucleates from a
  smaller release and with a shorter latency than ``S-A-R`` (reference).  Near
  the threshold the slow order spends hundreds of ticks with drive and WT
  spatially coexisting (a latent, slowly advancing front) -- the chasing regime.

Outputs under ``event-seq/img/spatial/``:

* ``tare_threshold.png``      -- final carrier frequency vs release size.
* ``tare_latency.png``        -- carrier/WT/front time series near threshold.
* ``tare_chasing.json``       -- numeric results.

Usage:
    python event-seq/tare_chasing.py [--quick] [--selftest]
"""

from __future__ import annotations

import argparse
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

ROWS, COLS = 41, 41
KERNEL_SIZE = 7
MEAN_DISPERSAL = 0.5
MIGRATION_RATE = 1.0
RELEASE_RADIUS = 3.0
# Paper TARE carriers ``carrier_index = [2, 3, 5]`` (1-based TARE order) =
# ``{dd, dw, dr}`` in the canonical order.
CARRIER_GENOTYPES = (0, 1, 4)
ORDERS = ("R-S-A", "S-A-R")
DRIVES = ("homing", "tare")

RELEASES = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0)
THRESHOLD_TICKS = 600
LATENCY_RELEASES = (1.0, 1.25)
LATENCY_TICKS = 1000
COVER_THRESHOLD = 0.5  # carrier fraction above which a deme is "covered"

_TARE_MAT = (
    Path(__file__).resolve().parent.parent
    / "hexagon_spatial_test" / "Hex-model-main" / "radial_release_optimizer"
    / "tare" / "drive_params.mat"
)


def kernel_and_valid_sum(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the off-centre migration kernel and its per-deme normalizer ``Z``."""
    kernel = sp.kernel_off_center(KERNEL_SIZE, MEAN_DISPERSAL)
    valid = fftconvolve(np.ones((rows, cols)), kernel, mode="same")
    return kernel, valid


def initial_state_both_sex(
    rows: int, cols: int, radius: float, release_multiple: float, scale: float = 1.0
) -> np.ndarray:
    """WT equilibrium everywhere, plus drive homozygotes released in a centre disc.

    Unlike ``spatial_test.initial_state`` (male-only, as in the M5 probe), this
    releases ``release_multiple`` drive individuals at the first adult age for
    **both sexes**, matching the paper ``main.m`` release (it seeds both
    ``cord_m1`` and ``cord_f1``).
    """
    data = np.zeros((m.N_Z, m.N_SEX, m.N_AGES, rows, cols))
    data[m.CORD_PURE_W, 0] = m.INITIAL_FEMALE[:, None, None]
    data[m.CORD_PURE_W, 1] = m.INITIAL_MALE[:, None, None]
    rr, cc = np.mgrid[0:rows, 0:cols]
    mask = (rr - (rows - 1) / 2.0) ** 2 + (cc - (cols - 1) / 2.0) ** 2 <= radius**2
    data[m.CORD_RELEASE_GENOTYPE, 0, m.NEW_ADULT_AGE, mask] += (
        release_multiple * m.INITIAL_FEMALE[m.NEW_ADULT_AGE]
    )
    data[m.CORD_RELEASE_GENOTYPE, 1, m.NEW_ADULT_AGE, mask] += (
        release_multiple * m.INITIAL_MALE[m.NEW_ADULT_AGE]
    )
    return data * scale


def _fraction(data: np.ndarray, genotypes: tuple[int, ...]) -> np.ndarray:
    """Per-deme fraction of non-egg (ages >= 1) individuals in ``genotypes``."""
    total = data[:, :, 1:].sum(axis=(0, 1, 2))
    part = data[list(genotypes), :, :][:, :, 1:].sum(axis=(0, 1, 2))
    return np.divide(part, total, out=np.zeros_like(total), where=total > 0)


def run_spatial(
    order: str,
    drive: str,
    *,
    release_multiple: float,
    ticks: int,
    kernel: np.ndarray,
    valid_sum: np.ndarray,
    radius: float = RELEASE_RADIUS,
    viability_dd: float = sp.VIABILITY_DD,
    fecundity_dd: float = sp.FECUNDITY_DD,
    scale: float = 1.0,
    stochastic: bool = False,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run one spatial replicate and return per-tick summary time series.

    Returns:
        Dict with ``t``, ``carrier`` (mean carrier fraction), ``wt`` (mean WT
        fraction), ``front`` (max radius where carrier > ``COVER_THRESHOLD``) and
        ``total`` (non-egg population).
    """
    kwargs: dict[str, float] = {"dc": sp.DC}
    if drive == "tare":
        kwargs.update(gcr=sp.GCR, eecr=sp.EECR)
    mats = m.drive_mats(drive, **kwargs)
    rng = np.random.default_rng(seed)
    data = initial_state_both_sex(ROWS, COLS, radius, release_multiple, scale)
    density_ref = scale * m.DENSITY_REF
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    dist = np.sqrt((rr - (ROWS - 1) / 2.0) ** 2 + (cc - (COLS - 1) / 2.0) ** 2)

    carrier = np.zeros(ticks + 1)
    wt = np.zeros(ticks + 1)
    front = np.zeros(ticks + 1)
    total = np.zeros(ticks + 1)

    def record(t: int) -> None:
        carrier[t] = _fraction(data, CARRIER_GENOTYPES).mean()
        wt[t] = _fraction(data, (m.CORD_PURE_W,)).mean()
        frac = _fraction(data, CARRIER_GENOTYPES)
        mask = frac > COVER_THRESHOLD
        front[t] = dist[mask].max() if mask.any() else 0.0
        total[t] = data[:, :, 1:].sum()

    record(0)
    for t in range(1, ticks + 1):
        if stochastic:
            newborns = sp._reproduction_stochastic(data, mats, fecundity_dd, rng)
        else:
            newborns = sp._reproduction(data, mats, fecundity_dd)
        for name in order.split("-"):
            if name == "R":
                if stochastic:
                    sp._event_R_stochastic(data, newborns, rng)
                else:
                    sp._event_R(data, newborns)
            elif name == "S":
                if stochastic:
                    sp._event_S_stochastic(data, density_ref, viability_dd, rng)
                else:
                    sp._event_S(data, density_ref, viability_dd)
            else:
                sp._event_A(data)
        sp._event_M(data, kernel, valid_sum, MIGRATION_RATE)
        record(t)
    return {"t": np.arange(ticks + 1), "carrier": carrier, "wt": wt,
            "front": front, "total": total}


def threshold_scan(
    kernel: np.ndarray, valid_sum: np.ndarray, *, ticks: int, drives: tuple[str, ...] = DRIVES,
    releases: tuple[float, ...] = RELEASES,
) -> dict[str, dict[str, list[float]]]:
    """Final carrier frequency vs release size for each (drive, order)."""
    out: dict[str, dict[str, list[float]]] = {}
    for drive in drives:
        for order in ORDERS:
            key = f"{drive}|{order}"
            vals = []
            for rel in releases:
                res = run_spatial(order, drive, release_multiple=rel, ticks=ticks,
                                  kernel=kernel, valid_sum=valid_sum)
                vals.append(float(res["carrier"][-1]))
            out[key] = {"release": list(releases), "final_carrier": vals}
    return out


def _plot_threshold(scan: dict[str, dict[str, list[float]]], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, drive in zip(axes, DRIVES):
        for order in ORDERS:
            d = scan[f"{drive}|{order}"]
            ax.plot(d["release"], d["final_carrier"], "o-", label=order)
        ax.axhline(COVER_THRESHOLD, color="gray", ls=":", lw=1)
        ax.set_title(drive)
        ax.set_xlabel("release multiple (x WT first-adult count, both sexes)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("final mean carrier fraction (non-egg)")
    fig.suptitle(
        "TARE nucleation threshold vs event order (avd=0.5, r=3, both-sex release)\n"
        "dotted line = 0.5 carrier; TARE has a critical release, homing does not"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_latency(latency: dict[str, dict[str, np.ndarray]], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(LATENCY_RELEASES), figsize=(13, 5), sharey=True)
    for ax, rel in zip(axes, LATENCY_RELEASES):
        for order in ORDERS:
            res = latency[f"{order}|{rel}"]
            ax.plot(res["t"], res["carrier"], label=f"{order} carrier")
            ax.plot(res["t"], res["wt"], ls="--", label=f"{order} WT")
        ax.set_title(f"TARE, release multiple = {rel:g}")
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean fraction (non-egg)")
    fig.suptitle(
        "TARE near-threshold latency: drive and WT coexist while the front is latent\n"
        "R-S-A nucleates earlier; S-A-R holds a long drive/WT coexistence phase"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _validate_tare_mats() -> dict[str, float]:
    """Check ``age_model.tare_mats`` against the paper's shipped ``drive_params.mat``."""
    if not _TARE_MAT.is_file():
        return {"checked": 0.0}
    import scipy.io as sio

    stored = sio.loadmat(_TARE_MAT)["mats"].astype(float)
    canon = m.tare_mats(gcr=1.0, eecr=1.0, ddfitness=1.0)
    inv = np.argsort(m._TARE_TO_CANON)  # TARE position -> canonical index
    recon = np.empty_like(canon)
    m3 = canon.reshape(5, 5, 5)
    for k in range(5):
        recon[k * 5:(k + 1) * 5, :] = m3[k][np.ix_(inv, inv)]
    return {"checked": 1.0, "max_abs_diff": float(np.abs(recon - stored).max())}


def selftest() -> int:
    """Run deterministic and stochastic sanity checks; return a process code."""
    kernel, valid_sum = kernel_and_valid_sum(ROWS, COLS)

    mats_check = _validate_tare_mats()
    assert mats_check.get("max_abs_diff", 1.0) < 1e-12, mats_check
    print(f"[ok] tare_mats matches paper drive_params.mat (max |diff| = "
          f"{mats_check['max_abs_diff']:.2e})")

    # Exact deterministic reproducibility.
    a = run_spatial("R-S-A", "tare", release_multiple=1.0, ticks=200,
                    kernel=kernel, valid_sum=valid_sum)
    b = run_spatial("R-S-A", "tare", release_multiple=1.0, ticks=200,
                    kernel=kernel, valid_sum=valid_sum)
    assert np.array_equal(a["carrier"], b["carrier"]), "deterministic run not reproducible"
    print("[ok] deterministic reproducibility (bit-identical)")

    # Event-order effect on the nucleation threshold at release = 1.0.
    rsa = run_spatial("R-S-A", "tare", release_multiple=1.0, ticks=THRESHOLD_TICKS,
                      kernel=kernel, valid_sum=valid_sum)["carrier"][-1]
    sar = run_spatial("S-A-R", "tare", release_multiple=1.0, ticks=THRESHOLD_TICKS,
                      kernel=kernel, valid_sum=valid_sum)["carrier"][-1]
    assert rsa > sar + 0.2, (rsa, sar)
    print(f"[ok] order shifts TARE threshold: carrier R-S-A={rsa:.3f} > S-A-R={sar:.3f}")

    # Stochastic mean with a 95% interval over replicates.
    finals = np.array([
        run_spatial("S-A-R", "tare", release_multiple=1.25, ticks=400,
                    kernel=kernel, valid_sum=valid_sum, stochastic=True, seed=s,
                    scale=10.0)["carrier"][-1]
        for s in range(40)
    ])
    mean = float(finals.mean())
    lo, hi = (float(x) for x in np.percentile(finals, [2.5, 97.5]))
    assert np.isfinite(finals).all() and lo <= mean <= hi, (lo, mean, hi)
    print(f"[ok] stochastic S-A-R rel=1.25 t=400: carrier mean={mean:.3f} "
          f"95% interval=[{lo:.3f}, {hi:.3f}] (n=40, sf=10)")

    # Homing sweeps even at the smallest tested release (no nucleation threshold).
    hom = run_spatial("S-A-R", "homing", release_multiple=0.5, ticks=400,
                      kernel=kernel, valid_sum=valid_sum)["carrier"][-1]
    assert hom > 0.9, hom
    print(f"[ok] homing sweeps at release=0.5 (carrier={hom:.3f})")
    print("all self-tests passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="shorter runs for a smoke test")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    IMG.mkdir(parents=True, exist_ok=True)
    kernel, valid_sum = kernel_and_valid_sum(ROWS, COLS)
    threshold_ticks = 200 if args.quick else THRESHOLD_TICKS
    latency_ticks = 300 if args.quick else LATENCY_TICKS

    scan = threshold_scan(kernel, valid_sum, ticks=threshold_ticks)
    _plot_threshold(scan, IMG / "tare_threshold.png")

    latency: dict[str, dict[str, np.ndarray]] = {}
    for rel in LATENCY_RELEASES:
        for order in ORDERS:
            latency[f"{order}|{rel}"] = run_spatial(
                order, "tare", release_multiple=rel, ticks=latency_ticks,
                kernel=kernel, valid_sum=valid_sum,
            )
    _plot_latency(latency, IMG / "tare_latency.png")

    report = {
        "params": {
            "grid": [ROWS, COLS], "mean_dispersal": MEAN_DISPERSAL,
            "migration_rate": MIGRATION_RATE, "release_radius": RELEASE_RADIUS,
            "release": "both-sex dd at first adult age", "carriers": list(CARRIER_GENOTYPES),
            "threshold_ticks": threshold_ticks, "latency_ticks": latency_ticks,
        },
        "tare_mats_vs_paper": _validate_tare_mats(),
        "threshold_scan": scan,
        "latency": {
            key: {"t": res["t"].tolist(), "carrier": res["carrier"].tolist(),
                  "wt": res["wt"].tolist(), "front": res["front"].tolist()}
            for key, res in latency.items()
        },
    }
    out = IMG / "tare_chasing.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {IMG / 'tare_threshold.png'}")
    print(f"wrote {IMG / 'tare_latency.png'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
