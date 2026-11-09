"""Download the raw GIS layers used by the Hawaii six-island K build.

Files are stored under ``geographic-score/data/raw`` (git-ignored) and a
manifest with SHA-256 hashes is written next to them. Downloads are resumable
at file granularity: an existing file with a recorded hash is not downloaded
again.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"
SOURCE_VERSION = "raw_v1"

WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile"
JRC_BASE = "https://storage.googleapis.com/global-surface-water/downloads2021/seasonality"
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "2020/BSGM/USA/usa_ppp_2020_UNadj_constrained.tif"
)
DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# 1-degree Copernicus DEM tiles that intersect the six main islands.
DEM_TILES = [
    (18, -157), (18, -156), (18, -155),
    (19, -157), (19, -156), (19, -155),
    (20, -158), (20, -157), (20, -156), (20, -155),
    (21, -160), (21, -159), (21, -158), (21, -157), (21, -156),
    (22, -160), (22, -159),
]

WORLDCLIM_VARS = ("tavg", "tmin", "tmax", "prec")
JRC_TILES = (
    "seasonality_170W_30Nv1_4_2021.tif",
    "seasonality_160W_30Nv1_4_2021.tif",
    "seasonality_170W_20Nv1_4_2021.tif",
    "seasonality_160W_20Nv1_4_2021.tif",
)


def _dem_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def source_table() -> dict[str, str]:
    """Return ``relative_path -> URL`` for every required raw file."""
    table: dict[str, str] = {}
    for var in WORLDCLIM_VARS:
        name = f"tile_25_wc2.1_30s_{var}.tif"
        table[f"worldclim/{name}"] = f"{WORLDCLIM_BASE}/{name}"
    for name in JRC_TILES:
        table[f"jrc/{name}"] = f"{JRC_BASE}/{name}"
    table["worldpop/usa_ppp_2020_UNadj_constrained.tif"] = WORLDPOP_URL
    for lat, lon in DEM_TILES:
        name = _dem_name(lat, lon)
        table[f"copernicus_dem/{name}/{name}.tif"] = f"{DEM_BASE}/{name}/{name}.tif"
    return table


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"source_version": SOURCE_VERSION, "files": {}}


def save_manifest(manifest: dict) -> None:
    temp = MANIFEST.with_suffix(".json.part")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(MANIFEST)


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "natal-core-gis2score/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                total = int(response.headers.get("Content-Length", "0"))
                written = 0
                last = time.time()
                with part.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if time.time() - last > 15:
                            pct = 100.0 * written / total if total else -1.0
                            print(f"  {dest.name}: {written / 1e6:.1f} MB ({pct:.0f}%)", flush=True)
                            last = time.time()
            if total and written != total:
                raise IOError(f"size mismatch: {written} != {total}")
            part.replace(dest)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, TimeoutError) as exc:
            print(f"  attempt {attempt}/{retries} failed for {dest.name}: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    files = manifest.setdefault("files", {})
    failures = []
    for rel, url in sorted(source_table().items()):
        dest = RAW_DIR / rel
        record = files.get(rel)
        if record and record.get("missing"):
            print(f"skip   {rel} (known ocean-only DEM tile)", flush=True)
            continue
        if dest.exists() and record and record.get("bytes") == dest.stat().st_size:
            print(f"cached {rel}", flush=True)
            continue
        print(f"fetch  {rel}\n  {url}", flush=True)
        try:
            download(url, dest)
        except urllib.error.HTTPError as exc:
            if rel.startswith("copernicus_dem/") and exc.code == 404:
                # Copernicus publishes 1-degree tiles only where land exists.
                files[rel] = {"url": url, "missing": True, "reason": "404 ocean-only tile"}
                save_manifest(manifest)
                print("  skipped: ocean-only DEM tile (404)", flush=True)
                continue
            failures.append((rel, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((rel, str(exc)))
            continue
        size = dest.stat().st_size
        files[rel] = {
            "url": url,
            "bytes": size,
            "sha256": sha256_file(dest),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        save_manifest(manifest)
        print(f"  ok {size / 1e6:.1f} MB  sha256={files[rel]['sha256'][:16]}...", flush=True)
    manifest["source_version"] = SOURCE_VERSION
    save_manifest(manifest)
    if failures:
        print("FAILURES:", flush=True)
        for rel, msg in failures:
            print(f"  {rel}: {msg}", flush=True)
        return 1
    print(f"all {len(files)} raw files present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
