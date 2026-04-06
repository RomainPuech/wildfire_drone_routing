#!/usr/bin/env python3
"""
Create California 2021 Wildfire Dataset in the same format as California2020Dataset.

- **Mask:** 2020 D1 union-of-burnable (mask_union_burnable_no_snow_excluded_day1.npy)
  copied to California2021Dataset/mask.npy.
- **Burn map logic:** D2 day-before before 10 am, D1 from 10 am (see documentation/04_california_2020_dataset.md).
  Requires 2021 WFPI D2 and D1 daily files and builds static_risk_wfpi_yearly.npy (730 frames).
- **Scenarii:** One ignition-point scenario per USFS 2021 fire (stage-1 + stage-2 filtered).
- **Config:** offset_ (1–12), date_ (YYYYMMDD), time_ (HHMM) from discovery datetime.
- **Averaged maps:** static_risk_wfpi_avg.npy (mean excluding >=249), 
  static_risk_wfpi_burn_at_least_once.npy (P(burn at least once) rescaled 0–248).

Run from project root:
    python code/dataset_creation/nature_dataset_creation/create_california_2021_dataset.py
"""

import os
import sys
import re
import json
import random
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Point
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from dataset import save_scenario_ignition_point

# Paths
CSV_PATH = os.path.join(DATA_DIR, "USFS_ignition_points.csv")
CA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract/tl_2024_06_tract.shp")
URBAN_SHP = os.path.join(DATA_DIR, "tl_2025_us_uac20/tl_2025_us_uac20.shp")
MASK_SOURCE = os.path.join(PROJECT_ROOT, "California2020Dataset", "mask_union_burnable_no_snow_excluded_day1.npy")
WFPI_2020_ZIP_DIR = os.path.join(DATA_DIR, "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
WFPI_2021_D2_DIR = os.path.join(DATA_DIR, "2021_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
WFPI_2021_D1_DIR = os.path.join(DATA_DIR, "2021_Wind-enhanced_Fire_Potential_Index_Forecast_1_DATA")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "California2021Dataset")
OUTPUT_DAY1_DIR = os.path.join(PROJECT_ROOT, "California2021Dataset_Day1")
SCENARII_DIR = os.path.join(OUTPUT_DIR, "scenarii")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_california_2021.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "dataset_summary.json")

# 2021: 365 days (not leap year)
ALL_2021_DAYS = []
d = date(2021, 1, 1)
while d <= date(2021, 12, 31):
    ALL_2021_DAYS.append(d)
    d += timedelta(days=1)


def get_missing_2021_wfpi_dates():
    """Return sorted list of YYYYMMDD for which 2021 D2 zip is missing."""
    d2_dir = Path(WFPI_2021_D2_DIR)
    if not d2_dir.exists():
        return []
    have = set()
    for f in d2_dir.glob("wfpi-forecast-2_data_*_*.zip"):
        parts = f.stem.split("_")
        if len(parts) >= 4 and len(parts[3]) == 8:
            have.add(parts[3])
    all_2021 = set((date(2021, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(365))
    return sorted(all_2021 - have)


def get_wfpi_grid_and_mask():
    """Reuse 2020 grid geometry (crop) from any WFPI zip; load mask from 2020 D1."""
    sample_zip = next(Path(WFPI_2020_ZIP_DIR).glob("wfpi-forecast-2_data_*.zip"), None)
    if not sample_zip:
        raise FileNotFoundError(f"No WFPI zip in {WFPI_2020_ZIP_DIR}")
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

    ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()
    ca_wfpi = ca_boundary.to_crs(wfpi_crs)
    minx, miny, maxx, maxy = ca_wfpi.total_bounds
    buf = 50_000
    minx -= buf; miny -= buf; maxx += buf; maxy += buf
    row_min, col_min = rasterio.transform.rowcol(raw_t, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(raw_t, maxx, miny)
    row_min = max(0, int(np.floor(row_min))); col_min = max(0, int(np.floor(col_min)))
    row_max = min(raw_H, int(np.ceil(row_max)) + 1); col_max = min(raw_W, int(np.ceil(col_max)) + 1)
    cropped_t = Affine(
        raw_t.a, raw_t.b, raw_t.c + col_min * raw_t.a,
        raw_t.d, raw_t.e, raw_t.f + row_min * raw_t.e,
    )
    grid_h = row_max - row_min
    grid_w = col_max - col_min
    mask = np.load(MASK_SOURCE)
    transformer = Transformer.from_crs("EPSG:4326", wfpi_crs, always_xy=True)

    def latlon_to_rowcol(lat, lon):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(cropped_t, x, y)
        return int(r), int(c)

    return mask, grid_h, grid_w, cropped_t, wfpi_crs, transformer, ca_boundary


def load_wfpi_from_zip(zip_path: Path):
    """Extract WFPI zip and return (data, meta, transform, crs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(tmpdir)
        tif_files = [f for f in Path(tmpdir).rglob("*.tif") if not f.name.endswith(".xml")]
        if not tif_files:
            tif_files = [f for f in Path(tmpdir).rglob("*.tiff") if not f.name.endswith(".xml")]
        if not tif_files:
            return None
        with rasterio.open(str(tif_files[0])) as src:
            return src.read(1), src.meta.copy(), src.transform, src.crs


def crop_to_california(data, wfpi_transform, wfpi_crs, california_gdf):
    """Crop 2D WFPI to California bbox + 50 km buffer. Returns (cropped_2d, new_transform)."""
    california_proj = california_gdf.to_crs(wfpi_crs)
    minx, miny, maxx, maxy = california_proj.total_bounds
    buf = 50_000
    minx -= buf; miny -= buf; maxx += buf; maxy += buf
    row_min, col_min = rasterio.transform.rowcol(wfpi_transform, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(wfpi_transform, maxx, miny)
    row_min, col_min = int(np.floor(row_min)), int(np.floor(col_min))
    row_max, col_max = int(np.ceil(row_max)) + 1, int(np.ceil(col_max)) + 1
    H = data.shape[-2] if data.ndim == 3 else data.shape[0]
    W = data.shape[-1] if data.ndim == 3 else data.shape[1]
    row_min = max(0, row_min); row_max = min(H, row_max)
    col_min = max(0, col_min); col_max = min(W, col_max)
    cropped = data[row_min:row_max, col_min:col_max] if data.ndim == 2 else data[:, row_min:row_max, col_min:col_max]
    new_transform = Affine(
        wfpi_transform.a, wfpi_transform.b,
        wfpi_transform.c + col_min * wfpi_transform.a,
        wfpi_transform.d, wfpi_transform.e,
        wfpi_transform.f + row_min * wfpi_transform.e,
    )
    return (cropped[0] if data.ndim == 3 else cropped), new_transform


def process_zip_to_npy(zip_path: Path, out_path: Path, california_gdf) -> bool:
    """Load raw WFPI zip, crop to California, save (1, H, W) float32."""
    result = load_wfpi_from_zip(zip_path)
    if result is None:
        return False
    data, meta, transform, crs = result
    cropped, _ = crop_to_california(data, transform, crs, california_gdf)
    arr = np.maximum(cropped.astype(np.float32), 0)[np.newaxis, :, :]
    np.save(str(out_path), arr)
    return True


def nearest_neighbour_fallback(target_date, available_dates, date_to_path):
    best = min(available_dates, key=lambda d: abs((d - target_date).days))
    return date_to_path[best]


def complete_2021_wfpi(california_gdf):
    """Fill California2021Dataset (D2) and California2021Dataset_Day1 (D1) with all 365 days."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DAY1_DIR, exist_ok=True)

    for label, dataset_dir, raw_dir, npy_glob, zip_pattern, npy_name in [
        ("D2", OUTPUT_DIR, WFPI_2021_D2_DIR, "wfpi_????????.npy",
         lambda d: f"wfpi-forecast-2_data_{d}_{d}.zip", lambda d: f"wfpi_{d}.npy"),
        ("D1", OUTPUT_DAY1_DIR, WFPI_2021_D1_DIR, "wfpi_day1_????????.npy",
         lambda d: f"wfpi-forecast-1_data_{d}_{d}.zip", lambda d: f"wfpi_day1_{d}.npy"),
    ]:
        existing = {}
        for f in Path(dataset_dir).glob(npy_glob):
            m = re.search(r"(\d{8})", f.name)
            if m:
                d_str = m.group(1)
                d_obj = date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:]))
                existing[d_obj] = f
        missing = [d for d in ALL_2021_DAYS if d not in existing]
        if not missing:
            continue
        for day in tqdm(missing, desc=f"Fill 2021 {label}"):
            d_str = day.strftime("%Y%m%d")
            out_npy = Path(dataset_dir) / npy_name(d_str)
            zip_path = Path(raw_dir) / zip_pattern(d_str)
            if zip_path.exists():
                ok = process_zip_to_npy(zip_path, out_npy, california_gdf)
                if ok:
                    existing[day] = out_npy
                    continue
            if existing:
                src = nearest_neighbour_fallback(day, list(existing.keys()), existing)
                shutil.copy2(str(src), str(out_npy))
                existing[day] = out_npy


def build_yearly_map():
    """Build static_risk_wfpi_yearly.npy (730, H, W): before 10 am = D2 day-1, after 10 am = D1 same day."""
    def load_frame(path):
        arr = np.load(str(path))
        return arr[0] if arr.ndim == 3 else arr

    n_days = len(ALL_2021_DAYS)
    n_frames = 2 * n_days
    sample = load_frame(Path(OUTPUT_DIR) / f"wfpi_{ALL_2021_DAYS[1].strftime('%Y%m%d')}.npy")
    H, W = sample.shape
    yearly = np.empty((n_frames, H, W), dtype=np.float32)
    for i, day in enumerate(tqdm(ALL_2021_DAYS, desc="Yearly map")):
        d_str = day.strftime("%Y%m%d")
        prev_str = (day - timedelta(days=1)).strftime("%Y%m%d")
        d2_path = Path(OUTPUT_DIR) / f"wfpi_{prev_str}.npy"
        if not d2_path.exists():
            d2_path = Path(OUTPUT_DIR) / f"wfpi_{d_str}.npy"
        yearly[2 * i] = load_frame(d2_path)
        d1_path = Path(OUTPUT_DAY1_DIR) / f"wfpi_day1_{d_str}.npy"
        if not d1_path.exists():
            d1_path = Path(OUTPUT_DIR) / f"wfpi_{d_str}.npy"
        yearly[2 * i + 1] = load_frame(d1_path)
    out_path = os.path.join(OUTPUT_DIR, "static_risk_wfpi_yearly.npy")
    np.save(out_path, yearly)
    print(f"  Saved {out_path} shape {yearly.shape}")


def build_avg_maps():
    """Build static_risk_wfpi_avg.npy (mean, exclude >=249) and static_risk_wfpi_burn_at_least_once.npy (0–248)."""
    yearly_path = os.path.join(OUTPUT_DIR, "static_risk_wfpi_yearly.npy")
    if not os.path.exists(yearly_path):
        print("  Skipping avg maps: yearly map not found.")
        return
    yearly = np.load(yearly_path)  # (730, H, W)
    H, W = yearly.shape[1], yearly.shape[2]
    valid = yearly < 249
    # Mean: exclude >=249; if no valid value, use 0
    sum_valid = np.where(valid, yearly, 0).sum(axis=0)
    count_valid_raw = valid.sum(axis=0)
    count_valid = np.maximum(count_valid_raw, 1)
    mean_map = (sum_valid / count_valid).astype(np.float32)
    mean_map[count_valid_raw == 0] = 0
    np.save(os.path.join(OUTPUT_DIR, "static_risk_wfpi_avg.npy"), mean_map[np.newaxis, :, :])

    # P(burn at least once): p_i = min(1, v/248) for v<249 else 0; P = 1 - prod(1-p_i); scale to 0-248
    p = np.where(yearly < 249, np.minimum(1.0, yearly.astype(np.float64) / 248.0), 0.0)
    one_minus_p = np.clip(1.0 - p, 1e-10, 1.0)  # avoid 0 product
    prob_burn_once = 1.0 - np.exp(np.log(one_minus_p).sum(axis=0))
    burn_once_248 = (np.clip(prob_burn_once, 0, 1) * 248).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, "static_risk_wfpi_burn_at_least_once.npy"), burn_once_248[np.newaxis, :, :])
    print("  Saved static_risk_wfpi_avg.npy and static_risk_wfpi_burn_at_least_once.npy")


def sanitize(s):
    """Filesystem-safe scenario base name."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(s))


def main():
    print("=" * 60)
    print("California 2021 Wildfire Dataset Creation")
    print("=" * 60)

    # 1) Copy mask and get grid
    print("\n[1] Copy mask and recover grid …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCENARII_DIR, exist_ok=True)
    shutil.copy2(MASK_SOURCE, os.path.join(OUTPUT_DIR, "mask.npy"))
    mask, grid_h, grid_w, cropped_t, wfpi_crs, transformer, ca_boundary = get_wfpi_grid_and_mask()

    def to_rowcol(lat, lon):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(cropped_t, x, y)
        return int(r), int(c)
    print(f"  Grid: {grid_h} x {grid_w}")

    # 2) Stage-1 + Stage-2 filter 2021 fires
    print("\n[2] Filter USFS 2021 fires (stage-1 + stage-2) …")
    df_all = pd.read_csv(CSV_PATH, low_memory=False)
    urban_gdf = gpd.read_file(URBAN_SHP).to_crs("EPSG:4326")
    urban_gdf["geometry"] = urban_gdf.buffer(0)

    df_21 = df_all[
        (df_all["FIREYEAR"] == 2021)
        & (df_all["UNIQFIREID"].str.startswith("2021-CA", na=False))
        & (df_all["FIRETYPECATEGORY"] == "WF")
        & (df_all["LATDD83"].notna())
        & (df_all["LONGDD83"].notna())
        & (df_all["DISCOVERYDATETIME"].notna())
    ].copy()
    df_21["discovery_dt"] = pd.to_datetime(df_21["DISCOVERYDATETIME"], errors="coerce", utc=True)
    df_21 = df_21[df_21["discovery_dt"].notna()]

    fire_gdf = gpd.GeoDataFrame(
        df_21,
        geometry=[Point(lon, lat) for lon, lat in zip(df_21["LONGDD83"], df_21["LATDD83"])],
        crs="EPSG:4326",
    )
    in_ca = gpd.sjoin(fire_gdf, ca_boundary[["geometry"]], how="inner", predicate="within")
    fire_gdf = fire_gdf.loc[in_ca.index]
    in_urban = gpd.sjoin(fire_gdf, urban_gdf[["geometry"]], how="inner", predicate="within")
    non_urban = fire_gdf[~fire_gdf.index.isin(in_urban.index)].copy()

    rows, cols = [], []
    for lat, lon in zip(non_urban["LATDD83"], non_urban["LONGDD83"]):
        r, c = to_rowcol(lat, lon)
        rows.append(r); cols.append(c)
    non_urban["_row"] = rows
    non_urban["_col"] = cols
    in_bounds = (
        (non_urban["_row"] >= 0) & (non_urban["_row"] < grid_h)
        & (non_urban["_col"] >= 0) & (non_urban["_col"] < grid_w)
    )
    in_bounds_gdf = non_urban[in_bounds].copy()
    in_mask = in_bounds_gdf.apply(
        lambda r: mask[int(r["_row"]), int(r["_col"])] == 1, axis=1
    )
    kept_gdf = in_bounds_gdf[in_mask].copy()
    # Exclude fires whose discovery date has no 2021 WFPI zip
    missing_dates = set(get_missing_2021_wfpi_dates())
    if missing_dates:
        kept_gdf["_date_str"] = kept_gdf["discovery_dt"].apply(
            lambda x: x.strftime("%Y%m%d") if hasattr(x, "strftime") else str(x)[:10].replace("-", "")
        )
        kept_gdf = kept_gdf[~kept_gdf["_date_str"].isin(missing_dates)].copy()
        kept_gdf = kept_gdf.drop(columns=["_date_str"], errors="ignore")
    print(f"  Kept {len(kept_gdf)} fires (excl. missing WFPI zip dates)")

    # 3) Scenarii + config (offset, date, time)
    print("\n[3] Writing scenarii and config …")
    random.seed(42)
    config = {}
    for _, r in kept_gdf.iterrows():
        fire_name = r.get("FIRENAME") or r.get("UNIQFIREID") or "Fire"
        uniq = r.get("UNIQFIREID", "")
        base = f"{sanitize(fire_name)}_{sanitize(uniq)}"
        scenario_name = f"{base}_scenario1"
        scenario_path = os.path.join(SCENARII_DIR, f"{scenario_name}.npy")
        row, col = int(r["_row"]), int(r["_col"])
        save_scenario_ignition_point(row=row, col=col, start_timestep=0, out_filename=scenario_path)
        config[f"offset_{base}"] = random.randint(1, 12)
        dt = r["discovery_dt"]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        config[f"date_{base}"] = dt.strftime("%Y%m%d")
        config[f"time_{base}"] = dt.strftime("%H%M")
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {CONFIG_PATH}")

    # 4) dataset_summary.json
    discovery_dates = kept_gdf["discovery_dt"].apply(
        lambda x: x.to_pydatetime().strftime("%Y-%m-%d") if hasattr(x, "to_pydatetime") else str(x)[:10]
    )
    summary = {
        "dataset_name": "California2021",
        "total_fires_stage1": int(len(non_urban)),
        "successful_fires": len(kept_gdf),
        "grid_dimensions": {"height": grid_h, "width": grid_w},
        "crs": str(wfpi_crs),
        "date_range": {"start": discovery_dates.min(), "end": discovery_dates.max()},
        "mask_source": "California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy",
        "burn_map_logic": "D2 day-before before 10am, D1 from 10am (see documentation/04_california_2020_dataset.md)",
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {SUMMARY_PATH}")

    # 5) Complete 2021 WFPI D2 and D1
    print("\n[4] Completing 2021 WFPI daily files (D2 + D1) …")
    ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    california_gdf = gpd.GeoDataFrame([1], geometry=[ca_tracts.dissolve().geometry.iloc[0]], crs="EPSG:4326")
    complete_2021_wfpi(california_gdf)

    # 6) Yearly map (mask-independent; skip if file already exists)
    yearly_path = os.path.join(OUTPUT_DIR, "static_risk_wfpi_yearly.npy")
    if os.path.exists(yearly_path):
        print(f"\n[5] static_risk_wfpi_yearly.npy already exists — skipping rebuild.")
    else:
        print("\n[5] Building static_risk_wfpi_yearly.npy …")
        build_yearly_map()

    # 7) Averaged maps (mask-independent; skip if files already exist)
    avg_path = os.path.join(OUTPUT_DIR, "static_risk_wfpi_avg.npy")
    burn_path = os.path.join(OUTPUT_DIR, "static_risk_wfpi_burn_at_least_once.npy")
    if os.path.exists(avg_path) and os.path.exists(burn_path):
        print(f"\n[6] static_risk_wfpi_avg and burn_at_least_once already exist — skipping rebuild.")
    else:
        print("\n[6] Building static_risk_wfpi_avg and burn_at_least_once …")
        build_avg_maps()

    print("\nDone. California2021Dataset ready.")


if __name__ == "__main__":
    main()
