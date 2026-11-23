"""Full-model demo: CPU CSR, NumPy stencil, and PyTorch GPU stencil.

Runs the same three-type RSA hex population as ``compare_strategies_v3.py``
and swaps only the migration backend:

* ``cpu``         -- NumPy mirror of natal's continuous CSR;
* ``stencil``     -- NumPy two-pass counter RNG stencil;
* ``gpu_stencil`` -- the same two-pass stencil on CUDA with torch float32.

The GPU lifecycle stays on the CPU, so each tick copies adult planes to the GPU
and back.  Timing is per replicate; for ``gpu_stencil`` it includes H2D,
GPU compute, and D2H, reported separately.

The script does not require natal: the homing tensor is loaded from a cached
``.npy`` when natal is unavailable.  Run it with the conda Python that has torch:

    /opt/conda/bin/python hexagon_spatial_test/chasing_model/scripts/compare_stencil_demo.py \
        --configs small --reps 50 --jobs 8
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import compare_strategies_v3 as v3  # noqa: E402
import direct_stencil_migration as stencil  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results" / "stencil_demo"
HTML_DIR = HERE / "results" / "html"
STRATEGIES = ("cpu", "stencil", "gpu_stencil")
DISPLAY = ("cpu", "stencil", "gpu_stencil")
COLORS = {"cpu": "tab:blue", "stencil": "tab:orange", "gpu_stencil": "tab:green"}
MIN_C2ST_REPS = 30

CONFIGS = [
    {
        "name": "small",
        "arena": 15,
        "kernel_size": 7,
        "avd": 0.8,
        "ticks": 100,
        "reps": 50,
        "scale": 100.0,
        "migration_rate": 0.5,
    },
    {
        "name": "large",
        "arena": 31,
        "kernel_size": 11,
        "avd": 1.5,
        "ticks": 100,
        "reps": 3,
        "scale": 100.0,
        "migration_rate": 0.5,
    },
]


def stencil_migrate_data(
    data: np.ndarray, kernel: np.ndarray, rate: float, seed: int, tick: int
) -> None:
    """Migrate the adult planes of one full-model state with the stencil."""
    rows, cols = data.shape[-2:]
    planes = data[:, :, v3.NEW_ADULT_AGE :].reshape(-1, rows * cols)
    migrated, _ = stencil.stencil_migrate_planes(
        planes, rows, cols, kernel, rate, seed, tick
    )
    data[:, :, v3.NEW_ADULT_AGE :] = migrated.reshape(
        data.shape[0], data.shape[1], data.shape[2] - v3.NEW_ADULT_AGE, rows, cols
    )


def stencil_migrate_data_gpu(
    data: np.ndarray, kernel: np.ndarray, rate: float, seed: int, tick: int
) -> dict[str, float]:
    """Migrate adult planes with the torch GPU stencil and time the transfers."""
    import direct_stencil_migration_torch as gpu_stencil
    import torch

    rows, cols = data.shape[-2:]
    planes = data[:, :, v3.NEW_ADULT_AGE :].reshape(-1, rows * cols)
    device = torch.device("cuda")
    t0 = time.perf_counter()
    fields = torch.as_tensor(planes, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    migrated, _ = gpu_stencil.stencil_migrate_planes_torch(
        fields,
        rows,
        cols,
        kernel,
        rate,
        seed,
        tick,
        dtype=torch.float32,
        device=device,
    )
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    migrated_np = migrated.to(device="cpu", dtype=torch.float64).numpy()
    t3 = time.perf_counter()
    data[:, :, v3.NEW_ADULT_AGE :] = migrated_np.reshape(
        data.shape[0], data.shape[1], data.shape[2] - v3.NEW_ADULT_AGE, rows, cols
    )
    return {
        "transfer_in": t1 - t0,
        "compute": t2 - t1,
        "transfer_out": t3 - t2,
        "total": t3 - t0,
    }


def _gpu_warmup(rows: int, cols: int, kernel: np.ndarray) -> None:
    """Initialize the CUDA context and geometry cache outside the timing loop."""
    import direct_stencil_migration_torch as gpu_stencil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_stencil requires CUDA, but torch.cuda.is_available() is false")
    device = torch.device("cuda")
    fields = torch.ones((1, rows * cols), dtype=torch.float32, device=device)
    gpu_stencil.stencil_migrate_planes_torch(
        fields, rows, cols, kernel, 0.5, 0, 0, dtype=torch.float32, device=device
    )
    torch.cuda.synchronize()


def run_replicate(
    strategy: str,
    cfg: dict,
    seed: int,
    selected: list[tuple[int, int]],
    tensor: np.ndarray,
    csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    kernel: np.ndarray,
) -> dict:
    """Run one replicate for one migration strategy."""
    replicate_start = time.perf_counter()
    rows = cols = cfg["arena"]
    rng_life = np.random.default_rng(seed)
    rng_mig = np.random.default_rng(seed + 100000)
    scale = float(cfg.get("scale", 1.0))
    rate = float(cfg.get("migration_rate", 0.5))
    density_ref = v3.DENSITY_REF * scale
    data = v3.initial_state(rows, cols, v3.RELEASE_RADIUS, v3.RELEASE_MULTIPLE, scale)
    ticks = cfg["ticks"]
    drive = np.zeros((ticks + 1, len(selected)))
    non_egg = np.zeros_like(drive)
    juvenile = np.zeros_like(drive)
    coverage = np.zeros((ticks + 1, len(v3.COVERS)))
    front = np.zeros(ticks + 1)
    total = np.zeros(ticks + 1)
    fields = np.zeros((ticks + 1, rows, cols))
    migration_seconds = 0.0
    transfer_in_seconds = 0.0
    compute_seconds = 0.0
    transfer_out_seconds = 0.0

    def capture(t: int) -> None:
        rec = v3.record(data, selected)
        drive[t] = rec["drive"]
        non_egg[t] = rec["non_egg"]
        juvenile[t] = rec["juvenile"]
        coverage[t] = rec["coverage"]
        front[t] = rec["front"]
        total[t] = rec["total"]
        fields[t] = rec["drive_field"]

    capture(0)
    for t in range(1, ticks + 1):
        newborns = v3.reproduction(data, tensor, rng_life)
        v3.event_r(data, newborns, rng_life)
        v3.event_s(data, density_ref, rng_life)
        v3.event_a(data)
        t0 = time.perf_counter()
        if strategy == "cpu":
            v3.cpu_migrate(data, csr, rate, rng_mig)
            elapsed = time.perf_counter() - t0
            migration_seconds += elapsed
            compute_seconds += elapsed
        elif strategy == "stencil":
            stencil_migrate_data(data, kernel, rate, seed + 100000, t)
            elapsed = time.perf_counter() - t0
            migration_seconds += elapsed
            compute_seconds += elapsed
        else:
            parts = stencil_migrate_data_gpu(data, kernel, rate, seed + 100000, t)
            migration_seconds += parts["total"]
            transfer_in_seconds += parts["transfer_in"]
            compute_seconds += parts["compute"]
            transfer_out_seconds += parts["transfer_out"]
        capture(t)
    return {
        "drive": drive,
        "non_egg": non_egg,
        "juvenile": juvenile,
        "coverage": coverage,
        "front": front,
        "total": total,
        "fields": fields,
        "migration_seconds": migration_seconds,
        "migration_parts": {
            "transfer_in": transfer_in_seconds,
            "compute": compute_seconds,
            "transfer_out": transfer_out_seconds,
        },
        "wall_seconds": time.perf_counter() - replicate_start,
    }


_WORKER: dict = {}


def _worker_init(cfg, selected, tensor, csr, kernel) -> None:
    _WORKER.update(
        {"cfg": cfg, "selected": selected, "tensor": tensor, "csr": csr, "kernel": kernel}
    )


def _replicate_task(args):
    strategy, seed = args
    worker = _WORKER
    return run_replicate(
        strategy,
        worker["cfg"],
        seed,
        worker["selected"],
        worker["tensor"],
        worker["csr"],
        worker["kernel"],
    )


def _html(
    frames: list[str],
    line_png: Path,
    title: str,
    timing: dict,
    c2st_png: Path | None,
    c2st_note: str | None,
) -> str:
    line_uri = base64.b64encode(line_png.read_bytes()).decode("ascii")
    def _timing_cell(strategy: str, prefix: str) -> str:
        entry = timing[strategy]
        return (
            f"{entry[f'{prefix}_mean']:.3f} "
            f"({entry[f'{prefix}_min']:.3f}-{entry[f'{prefix}_max']:.3f})"
        )

    def _breakdown_cell(strategy: str) -> str:
        entry = timing[strategy]
        return (
            f"{entry['migration_transfer_in_seconds_per_replicate_mean']:.3f} / "
            f"{entry['migration_compute_seconds_per_replicate_mean']:.3f} / "
            f"{entry['migration_transfer_out_seconds_per_replicate_mean']:.3f}"
        )

    timing_rows = "".join(
        f"<tr><td>{s}</td>"
        f"<td>{_timing_cell(s, 'wall_seconds_per_replicate')}</td>"
        f"<td>{_timing_cell(s, 'migration_seconds_per_replicate')}</td>"
        f"<td>{_breakdown_cell(s)}</td></tr>"
        for s in DISPLAY
        if s in timing
    )
    c2st_section = ""
    if c2st_png is not None and c2st_png.exists():
        c2st_uri = base64.b64encode(c2st_png.read_bytes()).decode("ascii")
        c2st_section = (
            "<h3>C2ST: CPU CSR vs NumPy stencil vs GPU stencil (lower triangle; 0.5 = indistinguishable)</h3>"
            f'<img src="data:image/png;base64,{c2st_uri}">'
        )
    elif c2st_note:
        c2st_section = f"<p><b>C2ST skipped:</b> {c2st_note}</p>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;text-align:center;margin:8px}}img{{max-width:96vw}}#bar{{margin:6px}}input[type=range]{{width:60%}}</style>
</head><body>
<h3>{title}</h3>
<h4>Timing per replicate (mean with min-max over independent replicates).
Migration is a subset of wall. The breakdown is H2D / GPU-or-CPU compute / D2H;
CPU strategies have zero transfer. This is a prototype comparison, not natal's
compiled CSR speed.</h4>
<table border="1" cellpadding="4" style="margin:auto;border-collapse:collapse">
<tr><th>strategy</th><th>wall s / replicate</th><th>migration s / replicate</th><th>in / compute / out (s/rep)</th></tr>
{timing_rows}
</table>
<img id="frame" src="data:image/png;base64,{frames[0]}">
<div id="bar"><button id="play">Play</button>
<input id="slider" type="range" min="0" max="{len(frames)-1}" value="0">
<span id="lbl">tick 0</span></div>
<h3>Selected demes: drive fraction / non-egg population / juvenile fraction (mean +/- 1 SD)</h3>
<img src="data:image/png;base64,{line_uri}">
{c2st_section}
<script>
const frames=[{','.join(f'"{frame}"' for frame in frames)}];let i=0,timer=null;
const img=document.getElementById('frame'),sl=document.getElementById('slider'),lbl=document.getElementById('lbl');
function show(k){{i=k;img.src='data:image/png;base64,'+frames[i];sl.value=i;lbl.textContent='tick '+i;}}
sl.oninput=()=>show(+sl.value);
document.getElementById('play').onclick=function(){{if(timer){{clearInterval(timer);timer=null;this.textContent='Play';}}else{{this.textContent='Pause';timer=setInterval(()=>show((i+1)%frames.length),300);}}}};
</script></body></html>"""


def _c2st(
    per_strategy: dict[str, dict[str, np.ndarray]],
    cfg: dict,
    selected: list[tuple[int, int]],
    out_dir: Path,
) -> tuple[dict | None, Path | None]:
    reps = per_strategy["cpu"]["drive"].shape[0]
    ticks = [max(1, cfg["ticks"] // 4), cfg["ticks"] // 2, 3 * cfg["ticks"] // 4, cfg["ticks"]]
    if reps < MIN_C2ST_REPS:
        return None, None
    feats = v3._feature_matrix(per_strategy, ticks, len(selected))
    accuracy, auc, pvalue = v3._c2st_matrices(feats, list(DISPLAY))
    path = out_dir / "c2st.png"
    v3._plot_c2st(accuracy, auc, list(DISPLAY), path)
    return {
        "methods": list(DISPLAY),
        "accuracy": accuracy.tolist(),
        "auc": auc.tolist(),
        "p": pvalue.tolist(),
    }, path


def run_config(cfg: dict, jobs: int = 1) -> dict:
    """Run one playground configuration and write its artifacts."""
    rows = cols = cfg["arena"]
    kernel = v3.hex_gaussian(cfg["kernel_size"], cfg["avd"])
    csr = v3.build_csr(rows, cols, kernel)
    tensor = v3.homing_tensor()
    selected = v3.select_demes(rows, cols)
    per_strategy: dict[str, dict[str, np.ndarray]] = {}
    timing: dict[str, dict] = {}
    if "gpu_stencil" in STRATEGIES:
        _gpu_warmup(rows, cols, kernel)
    for strategy in STRATEGIES:
        strategy_jobs = 1 if strategy == "gpu_stencil" else jobs
        if strategy_jobs > 1:
            with ProcessPoolExecutor(
                max_workers=strategy_jobs,
                initializer=_worker_init,
                initargs=(cfg, selected, tensor, csr, kernel),
            ) as pool:
                reps = list(
                    pool.map(
                        _replicate_task,
                        [(strategy, seed) for seed in range(cfg["reps"])],
                    )
                )
        else:
            reps = [
                run_replicate(strategy, cfg, seed, selected, tensor, csr, kernel)
                for seed in range(cfg["reps"])
            ]
        walls = np.asarray([r["wall_seconds"] for r in reps], dtype=np.float64)
        migrations = np.asarray([r["migration_seconds"] for r in reps], dtype=np.float64)
        transfer_in = np.asarray(
            [r["migration_parts"]["transfer_in"] for r in reps], dtype=np.float64
        )
        compute = np.asarray(
            [r["migration_parts"]["compute"] for r in reps], dtype=np.float64
        )
        transfer_out = np.asarray(
            [r["migration_parts"]["transfer_out"] for r in reps], dtype=np.float64
        )
        timing[strategy] = {
            "wall_seconds_per_replicate_mean": float(walls.mean()),
            "wall_seconds_per_replicate_median": float(np.median(walls)),
            "wall_seconds_per_replicate_min": float(walls.min()),
            "wall_seconds_per_replicate_max": float(walls.max()),
            "migration_seconds_per_replicate_mean": float(migrations.mean()),
            "migration_seconds_per_replicate_median": float(np.median(migrations)),
            "migration_seconds_per_replicate_min": float(migrations.min()),
            "migration_seconds_per_replicate_max": float(migrations.max()),
            "migration_transfer_in_seconds_per_replicate_mean": float(transfer_in.mean()),
            "migration_compute_seconds_per_replicate_mean": float(compute.mean()),
            "migration_transfer_out_seconds_per_replicate_mean": float(transfer_out.mean()),
            "migration_fraction_of_wall_mean": float((migrations / walls).mean()),
            "reps": cfg["reps"],
            "ticks": cfg["ticks"],
            "jobs": strategy_jobs,
        }
        per_strategy[strategy] = {
            key: np.stack([r[key] for r in reps])
            for key in ("drive", "non_egg", "juvenile", "coverage", "front", "total", "fields")
        }

    mean = {
        strategy: {key: value.mean(axis=0) for key, value in per_strategy[strategy].items()}
        for strategy in STRATEGIES
    }
    std = {
        strategy: {
            key: value.std(axis=0, ddof=1) if cfg["reps"] > 1 else np.zeros_like(value.mean(axis=0))
            for key, value in per_strategy[strategy].items()
        }
        for strategy in STRATEGIES
    }

    out_dir = RESULTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    line_png = out_dir / "selected_demes.png"
    v3.DISPLAY = DISPLAY
    v3.COLORS = COLORS
    v3.line_plot(mean, std, cfg["ticks"], selected, line_png)

    mean_fields = {
        strategy: per_strategy[strategy]["fields"][: min(5, cfg["reps"])].mean(axis=0)
        for strategy in STRATEGIES
    }
    frames = v3.render_frames(mean_fields, rows, cols, cfg["name"])
    c2st, c2st_png = _c2st(per_strategy, cfg, selected, out_dir)
    c2st_note = None
    if c2st is None:
        c2st_note = (
            f"R={cfg['reps']} is below the {MIN_C2ST_REPS}-replicate threshold; "
            "self-split C2ST at small R is dominated by sampling noise."
        )

    html = HTML_DIR / f"compare_stencil_demo_{cfg['name']}.html"
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    html.write_text(
        _html(
            frames,
            line_png,
            f"stencil demo (CPU/GPU): {cfg['name']} "
            f"({rows}x{cols}, k{cfg['kernel_size']}, avd={cfg['avd']:g}, "
            f"rate={cfg['migration_rate']:g}, R={cfg['reps']})",
            timing,
            c2st_png,
            c2st_note,
        ),
        encoding="utf-8",
    )

    def summarize(array: np.ndarray) -> dict:
        return {
            "final_mean": float(array[..., -1].mean()),
            "final_std": float(array[..., -1].std(ddof=1)) if array.shape[0] > 1 else 0.0,
            "mean_over_time": float(array.mean()),
        }

    summary = {
        "config": cfg,
        "selected_demes": selected,
        "coverage": {
            strategy: {
                f"thr_{threshold:g}": {
                    "final_count_mean": float(per_strategy[strategy]["coverage"][:, -1, i].mean()),
                    "final_count_std": (
                        float(per_strategy[strategy]["coverage"][:, -1, i].std(ddof=1))
                        if cfg["reps"] > 1
                        else 0.0
                    ),
                }
                for i, threshold in enumerate(v3.COVERS)
            }
            for strategy in STRATEGIES
        },
        "front": {strategy: summarize(per_strategy[strategy]["front"]) for strategy in STRATEGIES},
        "total": {strategy: summarize(per_strategy[strategy]["total"]) for strategy in STRATEGIES},
        "drive_final_selected": {
            strategy: [float(mean[strategy]["drive"][-1, d]) for d in range(len(selected))]
            for strategy in STRATEGIES
        },
        "timing": timing,
        "c2st": c2st,
        "c2st_note": c2st_note,
        "html": html.name,
        "line_plot": line_png.name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "series.npz",
        **{
            f"{strategy}_{key}": per_strategy[strategy][key]
            for strategy in STRATEGIES
            for key in ("drive", "non_egg", "juvenile", "coverage", "front", "total")
        },
        **{
            f"{strategy}_field_mean5": per_strategy[strategy]["fields"][
                : min(5, cfg["reps"])
            ].mean(axis=0)
            for strategy in STRATEGIES
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="small,large")
    parser.add_argument("--reps", type=int, default=0, help="override replicate count")
    parser.add_argument("--ticks", type=int, default=0, help="override tick count")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    wanted = set(args.configs.split(","))
    RESULTS.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for original in CONFIGS:
        if original["name"] not in wanted:
            continue
        cfg = dict(original)
        if args.reps > 0:
            cfg["reps"] = args.reps
        if args.ticks > 0:
            cfg["ticks"] = args.ticks
        print(f"[run] {cfg['name']} reps={cfg['reps']} ticks={cfg['ticks']} ...", flush=True)
        summaries.append(run_config(cfg, args.jobs))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
