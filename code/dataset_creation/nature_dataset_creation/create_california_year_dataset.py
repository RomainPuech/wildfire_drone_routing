#!/usr/bin/env python3
"""
Create a California <YEAR> Wildfire Dataset (ignition-point-only).

Produces CaliforniaYDataset/ with:
  - mask.npy                  (copied byte-for-byte from California2021Dataset)
  - static_risk_pyrologix.npy (copied byte-for-byte from California2021Dataset)
  - scenarii/<FireName>_<UNIQFIREID>_scenario1.npy  (2-element (row, col) array)
  - config_california_<YEAR>.json
  - dataset_summary.json

Run from project root:
    python code/dataset_creation/nature_dataset_creation/create_california_year_dataset.py --year 2022
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import save_scenario_ignition_point  # noqa: E402

CA_TRACTS = DATA_DIR / "tl_2024_06_tract" / "tl_2024_06_tract.shp"
URBAN_SHP = DATA_DIR / "tl_2025_us_uac20" / "tl_2025_us_uac20.shp"
WFPI_2020_ZIP_DIR = DATA_DIR / "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA"
REF_DATASET = PROJECT_ROOT / "California2021Dataset"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build CaliforniaYDataset (ignition-point-only) for a given year."
    )
    p.add_argument("--year", type=int, required=True, help="Fire year (e.g. 2022)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo_root>/California<YEAR>Dataset)",
    )
    p.add_argument(
        "--usfs-csv",
        type=Path,
        default=None,
        help=(
            "Path to USFS ignition-points CSV. "
            "Default: most recently modified USFS_ignition_points*.csv in data/"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if non-empty.",
    )
    return p.parse_args()


def discover_usfs_csv(data_dir: Path) -> Path:
    """Return the most recently modified *USFS_ignition_points*.csv in data_dir."""
    candidates = sorted(
        data_dir.glob("*USFS_ignition_points*.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No USFS_ignition_points*.csv found in {data_dir}"
        )
    chosen = candidates[0]
    print(f"  Using CSV: {chosen}  (mtime {chosen.stat().st_mtime:.0f})")
    return chosen


def get_wfpi_grid(ca_tracts_path: Path, wfpi_2020_zip_dir: Path):
    """
    Recover California-cropped affine transform from any 2020 WFPI zip.
    Returns (mask, grid_h, grid_w, cropped_t, wfpi_crs, transformer, ca_boundary).
    mask is loaded from California2021Dataset/mask.npy.
    """
    sample_zip = next(Path(wfpi_2020_zip_dir).glob("wfpi-forecast-2_data_*.zip"), None)
    if not sample_zip:
        raise FileNotFoundError(f"No WFPI 2020 zip in {wfpi_2020_zip_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(sample_zip)) as z:
            z.extractall(tmp)
        tif_path = next(
            f for f in Path(tmp).rglob("*")
            if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml")
        )
        with rasterio.open(str(tif_path)) as src:
            raw_t = src.transform
            wfpi_crs = src.crs
            raw_H, raw_W = src.height, src.width

    ca_tracts = gpd.read_file(str(ca_tracts_path)).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()
    ca_wfpi = ca_boundary.to_crs(wfpi_crs)
    minx, miny, maxx, maxy = ca_wfpi.total_bounds
    buf = 50_000
    minx -= buf; miny -= buf; maxx += buf; maxy += buf
    row_min, col_min = rasterio.transform.rowcol(raw_t, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(raw_t, maxx, miny)
    row_min = max(0, int(np.floor(row_min)))
    col_min = max(0, int(np.floor(col_min)))
    row_max = min(raw_H, int(np.ceil(row_max)) + 1)
    col_max = min(raw_W, int(np.ceil(col_max)) + 1)

    cropped_t = Affine(
        raw_t.a, raw_t.b, raw_t.c + col_min * raw_t.a,
        raw_t.d, raw_t.e, raw_t.f + row_min * raw_t.e,
    )
    grid_h = row_max - row_min
    grid_w = col_max - col_min

    mask = np.load(str(REF_DATASET / "mask.npy"))
    transformer = Transformer.from_crs("EPSG:4326", wfpi_crs, always_xy=True)

    return mask, grid_h, grid_w, cropped_t, wfpi_crs, transformer, ca_boundary


def sanitize(s: str) -> str:
    """Filesystem-safe scenario base name."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(s))


def main():
    args = parse_args()
    year = args.year

    # Resolve output directory
    out_dir = args.out_dir or (PROJECT_ROOT / f"California{year}Dataset")
    scenarii_dir = out_dir / "scenarii"
    config_path = out_dir / f"config_california_{year}.json"
    summary_path = out_dir / "dataset_summary.json"

    print("=" * 60)
    print(f"California {year} Wildfire Dataset Creation (ignition-point-only)")
    print("=" * 60)

    # Idempotency check
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            print(
                f"  WARNING: {out_dir} already exists and is non-empty. "
                "Pass --force to overwrite. Exiting."
            )
            sys.exit(0)
        else:
            print(f"  --force: removing existing {out_dir}")
            shutil.rmtree(str(out_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    scenarii_dir.mkdir(parents=True, exist_ok=True)

    # Resolve USFS CSV
    usfs_csv = args.usfs_csv or discover_usfs_csv(DATA_DIR)
    if not usfs_csv.exists():
        print(f"ERROR: USFS CSV not found: {usfs_csv}", file=sys.stderr)
        sys.exit(1)

    # [1] Copy mask and static_risk_pyrologix from 2021 reference dataset
    print(f"\n[1] Copying mask and pyrologix from {REF_DATASET} …")
    for fname in ("mask.npy", "static_risk_pyrologix.npy"):
        src = REF_DATASET / fname
        dst = out_dir / fname
        if not src.exists():
            print(f"ERROR: Reference file missing: {src}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(str(src), str(dst))
        print(f"  Copied {fname}")

    # [2] Recover WFPI grid geometry
    print("\n[2] Recovering WFPI grid geometry (from 2020 zip) …")
    mask, grid_h, grid_w, cropped_t, wfpi_crs, transformer, ca_boundary = get_wfpi_grid(
        CA_TRACTS, WFPI_2020_ZIP_DIR
    )
    print(f"  Grid: {grid_h} x {grid_w}")

    def to_rowcol(lat, lon):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(cropped_t, x, y)
        return int(r), int(c)

    # [3] Stage-1 filter
    print(f"\n[3] Stage-1 filter: year={year}, CA, non-urban, valid coords …")
    df_all = pd.read_csv(str(usfs_csv), low_memory=False)
    n_total = len(df_all)

    df_year = df_all[
        (df_all["FIREYEAR"] == year)
        & (df_all["UNIQFIREID"].str.startswith(f"{year}-CA", na=False))
        & (df_all["FIRETYPECATEGORY"] == "WF")
        & (df_all["LATDD83"].notna())
        & (df_all["LONGDD83"].notna())
        & (df_all["DISCOVERYDATETIME"].notna())
    ].copy()
    df_year["discovery_dt"] = pd.to_datetime(
        df_year["DISCOVERYDATETIME"], errors="coerce", utc=True
    )
    df_year = df_year[df_year["discovery_dt"].notna()]
    print(f"  Raw year+CA+WF+coords filter: {len(df_year)} rows")
    n_total_ca_wf = len(df_year)

    if n_total_ca_wf == 0:
        print(
            f"ERROR: 0 fires found for year {year} in {usfs_csv}. "
            "Check that the CSV covers this year.",
            file=sys.stderr,
        )
        sys.exit(1)

    # California boundary filter
    fire_gdf = gpd.GeoDataFrame(
        df_year,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(df_year["LONGDD83"], df_year["LATDD83"])
        ],
        crs="EPSG:4326",
    )
    in_ca = gpd.sjoin(fire_gdf, ca_boundary[["geometry"]], how="inner", predicate="within")
    fire_gdf = fire_gdf.loc[in_ca.index]
    n_dropped_outside_ca = n_total_ca_wf - len(fire_gdf)
    print(f"  After CA boundary filter: {len(fire_gdf)} (dropped {n_dropped_outside_ca})")

    # Urban exclusion
    urban_gdf = gpd.read_file(str(URBAN_SHP)).to_crs("EPSG:4326")
    urban_gdf["geometry"] = urban_gdf.buffer(0)
    in_urban = gpd.sjoin(fire_gdf, urban_gdf[["geometry"]], how="inner", predicate="within")
    non_urban = fire_gdf[~fire_gdf.index.isin(in_urban.index)].copy()
    n_dropped_urban = len(fire_gdf) - len(non_urban)
    print(f"  After urban exclusion: {len(non_urban)} (dropped {n_dropped_urban})")

    # [4] Stage-2 filter: grid bounds + mask
    print("\n[4] Stage-2 filter: grid bounds + burnable mask …")
    rows, cols = [], []
    for lat, lon in zip(non_urban["LATDD83"], non_urban["LONGDD83"]):
        r, c = to_rowcol(lat, lon)
        rows.append(r)
        cols.append(c)
    non_urban = non_urban.copy()
    non_urban["_row"] = rows
    non_urban["_col"] = cols

    in_bounds = (
        (non_urban["_row"] >= 0) & (non_urban["_row"] < grid_h)
        & (non_urban["_col"] >= 0) & (non_urban["_col"] < grid_w)
    )
    in_bounds_gdf = non_urban[in_bounds].copy()
    n_dropped_bounds = len(non_urban) - len(in_bounds_gdf)

    in_mask = in_bounds_gdf.apply(
        lambda r: mask[int(r["_row"]), int(r["_col"])] == 1, axis=1
    )
    kept_gdf = in_bounds_gdf[in_mask].copy()
    n_dropped_mask = len(in_bounds_gdf) - len(kept_gdf)
    print(
        f"  After grid+mask filter: {len(kept_gdf)} kept "
        f"(dropped {n_dropped_bounds} out-of-bounds, {n_dropped_mask} non-burnable)"
    )

    if len(kept_gdf) == 0:
        print(
            f"ERROR: 0 fires kept for year {year} after all filters. "
            "Investigate CSV or filter logic.",
            file=sys.stderr,
        )
        sys.exit(1)

    # [5] Write scenarii + config
    print("\n[5] Writing scenarii and config …")
    random.seed(42)
    config = {}
    for _, row in kept_gdf.iterrows():
        fire_name = row.get("FIRENAME") or row.get("UNIQFIREID") or "Fire"
        uniq = row.get("UNIQFIREID", "")
        base = f"{sanitize(fire_name)}_{sanitize(uniq)}"
        scenario_name = f"{base}_scenario1"
        scenario_path = scenarii_dir / f"{scenario_name}.npy"
        r_idx, c_idx = int(row["_row"]), int(row["_col"])
        save_scenario_ignition_point(
            row=r_idx, col=c_idx, start_timestep=0, out_filename=str(scenario_path)
        )
        config[f"offset_{base}"] = random.randint(1, 12)
        dt = row["discovery_dt"]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        config[f"date_{base}"] = dt.strftime("%Y%m%d")
        config[f"time_{base}"] = dt.strftime("%H%M")

    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config written: {config_path}")

    # [6] dataset_summary.json
    print("\n[6] Writing dataset_summary.json …")
    discovery_dates = kept_gdf["discovery_dt"].apply(
        lambda x: x.to_pydatetime().strftime("%Y-%m-%d")
        if hasattr(x, "to_pydatetime") else str(x)[:10]
    )
    n_kept = len(kept_gdf)
    summary = {
        "year": year,
        "n_kept": n_kept,
        "n_total_in_csv": n_total,
        "n_total_year_ca_wf": n_total_ca_wf,
        "n_dropped_outside_ca": n_dropped_outside_ca,
        "n_dropped_urban": n_dropped_urban,
        "n_dropped_bounds": n_dropped_bounds,
        "n_dropped_mask": n_dropped_mask,
        "grid_dimensions": {"height": grid_h, "width": grid_w},
        "date_range": {
            "start": discovery_dates.min(),
            "end": discovery_dates.max(),
        },
        "usfs_csv": str(usfs_csv),
        "mask_source": str(REF_DATASET / "mask.npy"),
    }
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary written: {summary_path}")

    # [7] SHA256 verification of copied files
    print("\n[7] Verifying copied files (SHA256) …")
    import hashlib

    def sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(str(p), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    for fname in ("mask.npy", "static_risk_pyrologix.npy"):
        src_hash = sha256_file(REF_DATASET / fname)
        dst_hash = sha256_file(out_dir / fname)
        if src_hash != dst_hash:
            print(f"  ERROR: {fname} SHA256 mismatch! src={src_hash} dst={dst_hash}", file=sys.stderr)
            sys.exit(1)
        print(f"  {fname}: OK ({src_hash[:16]}…)")

    n_scenarios = len(list(scenarii_dir.glob("*.npy")))
    print(f"\nDone. California{year}Dataset ready.")
    print(f"  n_kept={n_kept}  scenarii={n_scenarios}")
    print(f"  n_dropped_outside_ca={n_dropped_outside_ca}  n_dropped_urban={n_dropped_urban}")
    print(f"  n_dropped_bounds={n_dropped_bounds}  n_dropped_mask={n_dropped_mask}")


if __name__ == "__main__":
    main()
