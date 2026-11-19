"""A0 baseline for the GPU M2 migration operator.

Records the two facts needed before writing the GPU operator:

* the CUDA / cuFFT environment (conda torch, RTX device, transform accuracy);
* the CPU FFT M2 single-replicate baseline (wall and migration time) for the
  small and large v3 configurations.

The script has two modes because the repository virtualenv and the conda
environment are separate:

* ``--mode cpu``  -- run with ``.venv/bin/python`` (needs natal);
* ``--mode cuda`` -- run with ``/opt/conda/bin/python`` (needs torch).

Each mode writes its own JSON under ``results/gpu_fft/``.  No natal product
code is modified.  Example:

    source .venv/bin/activate
    unset PYTHONHOME PYO3_PYTHON PYTHONPATH
    python chasing_model/a0_baseline.py --mode cpu
    /opt/conda/bin/python chasing_model/a0_baseline.py --mode cuda
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "gpu_fft"
SEEDS = (0, 1, 2)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except Exception:
        pass
    return value


def _write(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def run_cpu() -> Path:
    """Run the CPU FFT M2 baseline for the v3 small/large configs."""
    import compare_strategies_v3 as v3
    import numpy as np

    payload: dict[str, Any] = {
        "mode": "cpu",
        "git_head": _git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "operator": "m2",
        "seeds": list(SEEDS),
        "configs": {},
    }
    for cfg in v3.CONFIGS:
        name = cfg["name"]
        rows = cfg["arena"]
        kernel = v3.hex_gaussian(cfg["kernel_size"], cfg["avd"])
        csr = v3.build_csr(rows, rows, kernel)
        tensor = v3.homing_tensor()
        selected = v3.select_demes(rows, rows)
        runs = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            rep = v3.run_replicate("m2", cfg, seed, selected, tensor, csr, kernel)
            wall = time.perf_counter() - t0
            runs.append(
                {
                    "seed": seed,
                    "wall_seconds": wall,
                    "migration_seconds": rep["migration_seconds"],
                    "migration_fraction": (
                        rep["migration_seconds"] / wall if wall > 0 else float("nan")
                    ),
                    "final_total_non_egg": float(rep["total"][-1]),
                    "final_front": float(rep["front"][-1]),
                }
            )
        walls = [r["wall_seconds"] for r in runs]
        migrations = [r["migration_seconds"] for r in runs]
        payload["configs"][name] = {
            "config": {
                "arena": rows,
                "kernel_size": cfg["kernel_size"],
                "avd": cfg["avd"],
                "ticks": cfg["ticks"],
                "scale": cfg["scale"],
            },
            "runs": runs,
            "wall_seconds_mean": float(statistics.mean(walls)),
            "wall_seconds_min": float(min(walls)),
            "wall_seconds_max": float(max(walls)),
            "migration_seconds_mean": float(statistics.mean(migrations)),
            "migration_seconds_min": float(min(migrations)),
            "migration_seconds_max": float(max(migrations)),
            "migration_fraction_mean": float(
                statistics.mean(r["migration_fraction"] for r in runs)
            ),
            "initial_total_non_egg": float(
                np.asarray(
                    v3.initial_state(
                        rows, rows, v3.RELEASE_RADIUS, v3.RELEASE_MULTIPLE, cfg["scale"]
                    )[:, :, 1:]
                ).sum()
            ),
        }
        print(
            f"[cpu:{name}] wall {statistics.mean(walls):.3f}s "
            f"(migration {statistics.mean(migrations):.3f}s, "
            f"{statistics.mean(r['migration_fraction'] for r in runs):.1%})",
            flush=True,
        )
    return _write("a0_cpu.json", payload)


def run_cuda() -> Path:
    """Record the CUDA / cuFFT environment and transform accuracy."""
    import torch

    payload: dict[str, Any] = {
        "mode": "cuda",
        "git_head": _git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        payload["device"] = {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mib": props.total_memory / 1024**2,
        }
        try:
            smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            payload["nvidia_smi"] = smi
        except Exception as exc:
            payload["nvidia_smi_error"] = str(exc)

        def _roundtrip(dtype: torch.dtype) -> dict[str, float]:
            x = torch.randn(4, 31, 31, dtype=dtype, device=device)
            y = torch.fft.irfft2(
                torch.fft.rfft2(x), s=(x.shape[-2], x.shape[-1])
            )
            torch.cuda.synchronize()
            return {
                "max_abs_roundtrip": float((y - x).abs().max().item()),
                "mean_abs_roundtrip": float((y - x).abs().mean().item()),
            }

        payload["cufft_roundtrip_f64"] = _roundtrip(torch.float64)
        payload["cufft_roundtrip_f32"] = _roundtrip(torch.float32)

        def _timing(batch: int, dtype: torch.dtype) -> dict[str, float]:
            x = torch.randn(batch, 31, 31, dtype=dtype, device=device)
            for _ in range(3):
                torch.fft.irfft2(
                    torch.fft.rfft2(x), s=(x.shape[-2], x.shape[-1])
                )
            torch.cuda.synchronize()
            repeats = 50
            t0 = time.perf_counter()
            for _ in range(repeats):
                torch.fft.irfft2(
                    torch.fft.rfft2(x), s=(x.shape[-2], x.shape[-1])
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / repeats
            return {
                "batch": batch,
                "dtype": str(dtype).replace("torch.", ""),
                "milliseconds_per_roundtrip": elapsed * 1e3,
            }

        payload["cufft_timing"] = [
            _timing(4, torch.float32),
            _timing(48, torch.float32),
            _timing(48, torch.float64),
        ]
        print(
            f"[cuda] {props.name} cap {torch.cuda.get_device_capability(0)}, "
            f"torch {torch.__version__} cuda {torch.version.cuda}",
            flush=True,
        )
        for row in payload["cufft_timing"]:
            print(
                f"  batch {row['batch']} {row['dtype']}: "
                f"{row['milliseconds_per_roundtrip']:.3f} ms",
                flush=True,
            )
    else:
        payload["error"] = "torch.cuda.is_available() is False"
        print("[cuda] CUDA not available", flush=True)
    return _write("a0_cuda.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    path = run_cpu() if args.mode == "cpu" else run_cuda()
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
