#!/usr/bin/env python3
"""
ALERTCalifornia camera baseline benchmark — California 2021 100-fire subset.

For each of the 100 benchmark fires (RANDOM_SEED=42, same subset used throughout the
paper), checks whether the fire ignition point falls within ``--radius`` km of any
ALERTCalifornia camera site.  A fire is counted as *detected* (at Δt = 0) if and only
if its ignition point is within the detection radius of at least one camera.

Output
------
- Detection percentage printed to stdout.
- Per-fire CSV with columns:
    fire_name, grid_row, grid_col, fire_lat, fire_lon,
    min_dist_to_camera_m, min_dist_to_camera_km, detected

Usage (run from project root):
    python code/benchmark_alertcalifornia.py --radius 20
    python code/benchmark_alertcalifornia.py --radius 20 --out results/alertca_20km.csv
    python code/benchmark_alertcalifornia.py --radius 20 --dataset-dir /path/to/California2021Dataset

Arguments
---------
--radius FLOAT        Detection radius in kilometres (required).
--out PATH            Output CSV path.  Default:
                        results/alertcalifornia_baseline_<radius>km_<YYYYMMDD_HHMMSS>.csv
--dataset-dir PATH    Path to California2021Dataset (default: <project_root>/California2021Dataset).
                        Falls back to paper/final_report/placement_data/ if the canonical
                        directory is absent (same fallback as generate_final_report.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent          # wildfire_drone_routing/
CAMERAS_JSON = (
    PROJECT_ROOT
    / "code/dataset_creation/nature_dataset_creation/data/cameras.json"
)
# WFPI raster archives are only used to recover the grid geo-referencing (CRS +
# affine transform). The grid is identical across WFPI forecast horizons, so the
# forecast-1 archive we ship works exactly like forecast-2. To avoid shipping any
# multi-GB raster, the derived georef is also cached in wfpi_georef.json and used
# preferentially (see get_wfpi_georef).
WFPI_ZIP_DIR = (
    PROJECT_ROOT
    / "code/dataset_creation/nature_dataset_creation/data"
    / "2021_Wind-enhanced_Fire_Potential_Index_Forecast_1_DATA"
)
WFPI_GEOREF_JSON = (
    PROJECT_ROOT
    / "code/dataset_creation/nature_dataset_creation"
    / "wfpi_georef.json"
)
CA_TRACTS_SHP = (
    PROJECT_ROOT
    / "code/dataset_creation/nature_dataset_creation/data"
    / "tl_2024_06_tract/tl_2024_06_tract.shp"
)
RESULTS_DIR = PROJECT_ROOT / "results"
BUNDLE_DIR = PROJECT_ROOT / "paper/final_report/placement_data"


# ---------------------------------------------------------------------------
# Dataset resolution (mirrors generate_final_report.py)
# ---------------------------------------------------------------------------

def resolve_dataset_root(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.is_dir():
            sys.exit(f"Error: --dataset-dir not found: {p}")
        return p
    canonical = PROJECT_ROOT / "California2021Dataset"
    if (
        (canonical / "config_california_2021.json").is_file()
        and (canonical / "scenarii").is_dir()
    ):
        return canonical
    fallback = BUNDLE_DIR
    if (
        (fallback / "config_california_2021.json").is_file()
        and (fallback / "scenarii").is_dir()
    ):
        print(
            f"[info] California2021Dataset not found at root; using bundled copy at\n"
            f"       {fallback}"
        )
        return fallback
    sys.exit(
        "Error: Could not find California2021Dataset.\n"
        f"  Looked at: {canonical}\n"
        f"  Fallback:  {fallback}\n"
        "  Use --dataset-dir to point to the dataset."
    )


# ---------------------------------------------------------------------------
# 100-fire benchmark subset (seed = 42)
# ---------------------------------------------------------------------------

def load_benchmark_fires(dataset_root: Path, all_fires: bool = False) -> list[tuple[str, int, int]]:
    """Return list of (fire_name, grid_row, grid_col).

    When *all_fires* is False (default) replicates the 100-fire subset used by
    generate_final_report.py (np.random.default_rng(42), sorted valid scenarii).
    When *all_fires* is True all valid scenarii are returned.
    """
    # Locate config dynamically so the function works for any year's dataset.
    config_candidates = sorted(dataset_root.glob("config_california_*.json"))
    if not config_candidates:
        sys.exit(f"Error: no config_california_*.json found in {dataset_root}")
    config_path = config_candidates[0]
    scenarii_dir = dataset_root / "scenarii"
    with config_path.open() as f:
        config = json.load(f)
    scenarii = sorted(scenarii_dir.glob("*.npy"))
    valid = [
        sf
        for sf in scenarii
        if all(
            f"{k}_{sf.stem.replace('_scenario1', '')}" in config
            for k in ("offset", "date", "time")
        )
    ]
    if all_fires:
        selected = valid
    else:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(valid), size=100, replace=False))
        selected = [valid[i] for i in idx]
    fires: list[tuple[str, int, int]] = []
    for sf in selected:
        pt = np.load(str(sf))
        fires.append((sf.stem.replace("_scenario1", ""), int(pt[0]), int(pt[1])))
    return fires


# ---------------------------------------------------------------------------
# WFPI geo-referencing (mirrors create_california_2021_dataset.py)
# ---------------------------------------------------------------------------

def get_wfpi_georef():
    """Return (cropped_affine, wfpi_crs) for the 1309×805 California crop.

    Prefers the cached ``wfpi_georef.json`` (tiny, shipped in the repo) so no WFPI
    raster is required. Falls back to deriving it from a WFPI forecast-1 archive if
    the cache is absent (the grid is identical across forecast horizons).
    """
    try:
        import rasterio
        import rasterio.transform
        import geopandas as gpd
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall with: pip install rasterio geopandas")

    # Fast path: use the cached georef (no raster needed).
    if WFPI_GEOREF_JSON.is_file():
        from rasterio.crs import CRS as _CRS
        meta = json.loads(WFPI_GEOREF_JSON.read_text())
        a, b, c, d, e, f = meta["cropped_transform"]
        return rasterio.transform.Affine(a, b, c, d, e, f), _CRS.from_wkt(meta["crs_wkt"])

    zips = sorted(WFPI_ZIP_DIR.glob("wfpi-forecast-1_data_*.zip"))
    if not zips:
        sys.exit(
            f"No cached georef at {WFPI_GEOREF_JSON} and no WFPI zips found in:\n"
            f"  {WFPI_ZIP_DIR}\n"
            "Either is needed to recover the grid geo-referencing."
        )

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(zips[0])) as zf:
            zf.extractall(tmp)
        tif_path = next(
            f
            for f in Path(tmp).rglob("*")
            if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml")
        )
        with rasterio.open(str(tif_path)) as src:
            raw_t = src.transform
            wfpi_crs = src.crs
            raw_h, raw_w = src.height, src.width

    ca_tracts = gpd.read_file(str(CA_TRACTS_SHP)).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()
    ca_wfpi = ca_boundary.to_crs(wfpi_crs)
    minx, miny, maxx, maxy = ca_wfpi.total_bounds
    buf = 50_000
    minx -= buf
    miny -= buf
    maxx += buf
    maxy += buf
    row_min, col_min = rasterio.transform.rowcol(raw_t, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(raw_t, maxx, miny)
    row_min = max(0, int(np.floor(row_min)))
    col_min = max(0, int(np.floor(col_min)))
    row_max = min(raw_h, int(np.ceil(row_max)) + 1)
    col_max = min(raw_w, int(np.ceil(col_max)) + 1)

    # Build cropped affine (same formula as create_california_2021_dataset.py)
    _Affine = type(raw_t)
    cropped_t = _Affine(
        raw_t.a,
        raw_t.b,
        raw_t.c + col_min * raw_t.a,
        raw_t.d,
        raw_t.e,
        raw_t.f + row_min * raw_t.e,
    )
    return cropped_t, wfpi_crs


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_unique_camera_positions(wfpi_crs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ALERTCalifornia camera sites, deduplicated by physical location.

    Returns
    -------
    cam_xy   : (N, 2) float64 — projected coordinates in wfpi_crs (metres)
    cam_lons : (N,)   float64 — WGS84 longitudes
    cam_lats : (N,)   float64 — WGS84 latitudes
    """
    try:
        from pyproj import Transformer
    except ImportError:
        sys.exit("Missing dependency: pyproj\nInstall with: pip install pyproj")

    with CAMERAS_JSON.open() as f:
        data = json.load(f)

    seen: set[tuple[float, float]] = set()
    lons: list[float] = []
    lats: list[float] = []
    for feat in data["features"]:
        lon = feat["properties"].get("longitude")
        lat = feat["properties"].get("latitude")
        if lon is None or lat is None:
            continue
        key = (float(lon), float(lat))
        if key in seen:
            continue
        seen.add(key)
        lons.append(float(lon))
        lats.append(float(lat))

    tf = Transformer.from_crs("EPSG:4326", wfpi_crs, always_xy=True)
    xs, ys = tf.transform(lons, lats)
    return np.column_stack([xs, ys]), np.array(lons), np.array(lats)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    radius_km: float,
    out_csv: Path,
    dataset_root: Path,
    all_fires: bool = False,
) -> float:
    """Run the ALERTCalifornia baseline benchmark.  Returns the detection percentage."""
    import rasterio.transform
    from pyproj import Transformer

    radius_m = radius_km * 1_000.0

    print(f"[1/4] Loading geo-referencing from WFPI zip …")
    cropped_t, wfpi_crs = get_wfpi_georef()

    print(f"[2/4] Loading camera positions ({CAMERAS_JSON.name}) …")
    cam_xy, cam_lons, cam_lats = load_unique_camera_positions(wfpi_crs)
    print(f"      {len(cam_xy)} unique ALERTCalifornia camera sites loaded.")

    fire_label = "all fires" if all_fires else "100-fire benchmark subset (seed=42)"
    print(f"[3/4] Loading {fire_label} from {dataset_root.name} …")
    fires = load_benchmark_fires(dataset_root, all_fires=all_fires)
    print(f"      {len(fires)} fires loaded.")

    # Transformer to convert fire projected coords → WGS84 for the CSV
    to_wgs84 = Transformer.from_crs(wfpi_crs, "EPSG:4326", always_xy=True)

    print(f"[4/4] Running detection check (radius = {radius_km} km) …")
    csv_rows: list[dict] = []
    n_detected = 0

    for fire_name, row, col in fires:
        # Fire projected coordinates (centre of the grid cell)
        x, y = rasterio.transform.xy(cropped_t, row, col)
        # Fire WGS84 lat/lon for human-readable CSV output
        fire_lon, fire_lat = to_wgs84.transform(x, y)
        # Euclidean distances to all cameras in the projected CRS (metres)
        dists = np.sqrt((cam_xy[:, 0] - x) ** 2 + (cam_xy[:, 1] - y) ** 2)
        min_dist_m = float(np.min(dists))
        detected = min_dist_m <= radius_m
        if detected:
            n_detected += 1
        csv_rows.append(
            {
                "fire_name": fire_name,
                "grid_row": row,
                "grid_col": col,
                "fire_lat": round(fire_lat, 6),
                "fire_lon": round(fire_lon, 6),
                "min_dist_to_camera_m": round(min_dist_m, 1),
                "min_dist_to_camera_km": round(min_dist_m / 1_000.0, 3),
                "detected": int(detected),
            }
        )

    detection_pct = 100.0 * n_detected / len(fires)

    print()
    print("=" * 50)
    print(f"  ALERTCalifornia baseline — radius = {radius_km} km")
    print(f"  Detected:       {n_detected} / {len(fires)}")
    print(f"  Detection rate: {detection_pct:.1f}%")
    print("=" * 50)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys())
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Per-fire CSV:   {out_csv}")

    return detection_pct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ALERTCalifornia camera baseline benchmark on the California 2021 "
            "100-fire benchmark subset."
        )
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Detection radius in kilometres (e.g. 20 for 20 km).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output CSV path.  Default: "
            "results/alertcalifornia_baseline_<radius>km_<YYYYMMDD_HHMMSS>.csv"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Path to CaliforniaYYYYDataset directory "
            "(default: <project_root>/California2021Dataset)."
        ),
    )
    parser.add_argument(
        "--all-fires",
        action="store_true",
        default=False,
        help="Run on all fires in the dataset instead of the 100-fire benchmark subset.",
    )
    args = parser.parse_args()

    if args.radius <= 0:
        sys.exit("Error: --radius must be a positive number.")

    dataset_root = resolve_dataset_root(args.dataset_dir)

    if args.out:
        out_csv = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        radius_tag = f"{args.radius:g}km"
        out_csv = RESULTS_DIR / f"alertcalifornia_baseline_{radius_tag}_{ts}.csv"

    run_benchmark(args.radius, out_csv, dataset_root, all_fires=args.all_fires)


if __name__ == "__main__":
    main()
