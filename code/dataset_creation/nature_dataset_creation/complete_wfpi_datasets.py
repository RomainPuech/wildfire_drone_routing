#!/usr/bin/env python3
"""
Complete California 2020 WFPI Datasets

Fills in every calendar day of 2020 for both:
  - California2020Dataset/         (Day-2 forecasts: wfpi_YYYYMMDD.npy)
  - California2020Dataset_Day1/    (Day-1 forecasts: wfpi_day1_YYYYMMDD.npy)

Strategy (in priority order):
  1. If a raw zip exists for the missing date → extract, crop to California, save.
  2. Otherwise → copy the nearest available existing .npy file (previous day first,
     then next day if no previous exists).

Run from the project root:
    python code/dataset_creation/nature_dataset_creation/complete_wfpi_datasets.py
"""

import os
import sys
import shutil
import zipfile
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from datetime import date, timedelta
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."
DATA_DIR     = SCRIPT_DIR / "data"

D2_DATASET = PROJECT_ROOT / "California2020Dataset"
D1_DATASET = PROJECT_ROOT / "California2020Dataset_Day1"

D2_RAW_DIR = DATA_DIR / "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA"
D1_RAW_DIR = DATA_DIR / "2020_Wind-enhanced_Fire_Potential_Index_Forecast_1_DATA"

CALIFORNIA_TRACTS = DATA_DIR / "tl_2024_06_tract/tl_2024_06_tract.shp"

# All 366 days of 2020 (leap year)
ALL_2020_DAYS = []
d = date(2020, 1, 1)
while d <= date(2020, 12, 31):
    ALL_2020_DAYS.append(d)
    d += timedelta(days=1)


# ── raster helpers (same logic as create_california_2020_dataset.py) ───────────

def load_wfpi_from_zip(zip_path: Path):
    """Extract a WFPI zip and return (data, meta, transform, crs)."""
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
    """
    Crop a 2-D or (1,H,W) WFPI array to the California bounding box (+ 50 km buffer).
    Returns (cropped_2d, new_transform).
    """
    from affine import Affine

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

    cropped = data[row_min:row_max, col_min:col_max] if data.ndim == 2 \
              else data[:, row_min:row_max, col_min:col_max]

    new_transform = Affine(
        wfpi_transform.a, wfpi_transform.b,
        wfpi_transform.c + col_min * wfpi_transform.a,
        wfpi_transform.d, wfpi_transform.e,
        wfpi_transform.f + row_min * wfpi_transform.e,
    )
    return (cropped[0] if data.ndim == 3 else cropped), new_transform


def process_zip_to_npy(zip_path: Path, out_path: Path, california_gdf) -> bool:
    """Load a raw WFPI zip, crop to California, save as (1, H, W) float32 .npy."""
    result = load_wfpi_from_zip(zip_path)
    if result is None:
        print(f"    WARNING: no .tif found in {zip_path.name}")
        return False
    data, meta, transform, crs = result
    cropped, _ = crop_to_california(data, transform, crs, california_gdf)
    arr = np.maximum(cropped.astype(np.float32), 0)[np.newaxis, :, :]  # (1, H, W)
    np.save(str(out_path), arr)
    return True


def nearest_neighbour_fallback(target_date: date, available_dates: list[date],
                                date_to_path: dict) -> Path:
    """Return the .npy path of the closest available date to target_date."""
    best = min(available_dates, key=lambda d: abs((d - target_date).days))
    return date_to_path[best]


# ── main ───────────────────────────────────────────────────────────────────────

def complete_dataset(dataset_dir: Path, raw_zip_dir: Path,
                     npy_glob: str, zip_pattern_fn,
                     npy_name_fn, forecast_label: str,
                     california_gdf):
    """
    Fill missing days in one dataset directory.

    Parameters
    ----------
    dataset_dir     : Path  – e.g. California2020Dataset/
    raw_zip_dir     : Path  – directory with raw WFPI zips
    npy_glob        : str   – glob to find existing .npy files, e.g. "wfpi_????????.npy"
    zip_pattern_fn  : callable(date_str) -> zip filename stem
    npy_name_fn     : callable(date_str) -> .npy filename
    forecast_label  : str   – "D2" or "D1" for logging
    california_gdf  : GeoDataFrame
    """
    print(f"\n{'='*60}")
    print(f"  {forecast_label}  →  {dataset_dir.name}")
    print(f"{'='*60}")

    # Existing .npy dates
    existing = {}
    for f in dataset_dir.glob(npy_glob):
        # extract the 8-digit date from the filename
        import re
        m = re.search(r"(\d{8})", f.name)
        if m:
            d_str = m.group(1)
            d_obj = date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:]))
            existing[d_obj] = f

    # Which 2020 days are missing?
    missing = [d for d in ALL_2020_DAYS if d not in existing]
    print(f"  Existing: {len(existing)}  |  Missing: {len(missing)}")

    if not missing:
        print("  Nothing to do.")
        return

    extracted = 0
    fallback  = 0

    for day in tqdm(missing, desc=f"  Filling {forecast_label}"):
        d_str   = day.strftime("%Y%m%d")
        out_npy = dataset_dir / npy_name_fn(d_str)

        # 1. Try raw zip
        zip_name = zip_pattern_fn(d_str)
        zip_path = raw_zip_dir / zip_name
        if zip_path.exists():
            ok = process_zip_to_npy(zip_path, out_npy, california_gdf)
            if ok:
                existing[day] = out_npy
                extracted += 1
                continue

        # 2. Nearest-neighbour fallback
        if existing:
            src = nearest_neighbour_fallback(day, list(existing.keys()), existing)
            shutil.copy2(str(src), str(out_npy))
            existing[day] = out_npy
            fallback += 1
        else:
            print(f"    ERROR: no existing files yet for fallback on {d_str}")

    print(f"\n  Done — extracted from zip: {extracted}, nearest-neighbour copies: {fallback}")


def main():
    # Load California boundary (needed for cropping)
    print("Loading California boundary...")
    ca_tracts = gpd.read_file(str(CALIFORNIA_TRACTS))
    ca_tracts = ca_tracts.to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    california_gdf = gpd.GeoDataFrame(
        [1], geometry=[ca_tracts.dissolve().geometry.iloc[0]], crs="EPSG:4326"
    )
    print("  California boundary loaded.")

    # ── Day-2 dataset ──────────────────────────────────────────────────────────
    complete_dataset(
        dataset_dir    = D2_DATASET,
        raw_zip_dir    = D2_RAW_DIR,
        npy_glob       = "wfpi_????????.npy",
        zip_pattern_fn = lambda d: f"wfpi-forecast-2_data_{d}_{d}.zip",
        npy_name_fn    = lambda d: f"wfpi_{d}.npy",
        forecast_label = "D2",
        california_gdf = california_gdf,
    )

    # ── Day-1 dataset ──────────────────────────────────────────────────────────
    complete_dataset(
        dataset_dir    = D1_DATASET,
        raw_zip_dir    = D1_RAW_DIR,
        npy_glob       = "wfpi_day1_????????.npy",
        zip_pattern_fn = lambda d: f"wfpi-forecast-1_data_{d}_{d}.zip",
        npy_name_fn    = lambda d: f"wfpi_day1_{d}.npy",
        forecast_label = "D1",
        california_gdf = california_gdf,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
