"""Refresh v3 C2ST (self-split diagonal, adaptive text) and HTML without re-simulating."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import numpy as np

import compare_strategies_v3 as v3

HERE = Path(__file__).resolve().parent
HTML = HERE / "results" / "html"

for name in ("small", "large"):
    out = HERE / "results" / "strategies_v3" / name
    summary = json.loads((out / "summary.json").read_text())
    cfg = summary["config"]
    z = np.load(out / "series.npz")
    per = {
        s: {k: z[f"{s}_{k}"] for k in ("drive", "coverage", "front", "total")}
        for s in v3.DISPLAY
    }
    ticks = [max(1, cfg["ticks"] // 4), cfg["ticks"] // 2, 3 * cfg["ticks"] // 4, cfg["ticks"]]
    feats = v3._feature_matrix(per, ticks, len(summary["selected_demes"]))
    methods = list(v3.DISPLAY)
    acc, auc, pval = v3._c2st_matrices(feats, methods)
    c2st_png = out / "c2st.png"
    v3._plot_c2st(acc, auc, methods, c2st_png)
    c2st = {"methods": methods, "accuracy": acc.tolist(), "auc": auc.tolist(), "p": pval.tolist()}
    (out / "c2st.json").write_text(json.dumps(c2st, indent=2), encoding="utf-8")
    summary["c2st"] = c2st
    kernel = v3.hex_gaussian(cfg["kernel_size"], cfg["avd"])
    csr = v3.build_csr(cfg["arena"], cfg["arena"], kernel)
    tensor = v3.homing_tensor()
    for s in summary["timing"]:
        summary["timing"][s]["jobs"] = 8
        if "sequential" in summary["timing"][s]:
            continue
        walls, migs = [], []
        for seed in (0, 1, 2):
            t0 = time.perf_counter()
            rep = v3.run_replicate(s, cfg, seed, summary["selected_demes"], tensor, csr, kernel)
            walls.append(time.perf_counter() - t0)
            migs.append(rep["migration_seconds"])
        summary["timing"][s]["sequential"] = {
            "wall_seconds": float(sum(walls) / len(walls)),
            "migration_seconds": float(sum(migs) / len(migs)),
            "seeds": [0, 1, 2],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html_path = HTML / f"compare_strategies_v3_{name}.html"
    html = html_path.read_text(encoding="utf-8")
    m = re.search(r"const frames=\[(.*?)\];let i=0", html, re.S)
    frames = json.loads("[" + m.group(1) + "]")
    m2 = re.search(r'<h3>Selected demes.*?</h3>\s*<img src="data:image/png;base64,([^"]+)"', html, re.S)
    (out / "selected_demes.png").write_bytes(base64.b64decode(m2.group(1)))
    title = f"strategies v3: {name} ({cfg['arena']}x{cfg['arena']}, k{cfg['kernel_size']}, avd={cfg['avd']:g})"
    html_path.write_text(
        v3._html(frames, out / "selected_demes.png", title, summary["timing"], c2st_png),
        encoding="utf-8",
    )
    print(name, "diag", [round(acc[i, i], 3) for i in range(len(methods))],
          "pairs", [(methods[i], methods[j], round(acc[i, j], 3), round(auc[i, j], 3), round(pval[i, j], 3))
                    for i in range(len(methods)) for j in range(i)])
