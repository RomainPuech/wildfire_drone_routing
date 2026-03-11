#!/usr/bin/env python3
"""
Build union-of-burnable mask from 2020 WFPI Day 1 daily maps (snow not excluded).

Output: California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy

Same logic as the Day 2 mask: a cell is valid if it has value < 249 or value == 250
on at least one day. Uses California2020Dataset_Day1/wfpi_day1_YYYYMMDD.npy (366 files).
"""

import os
import numpy as np
from pathlib import Path
from rasterio.features import rasterize
from scipy import ndimage
import geopandas as gpd
import rasterio
import zipfile
import tempfile
from affine import Affine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
D1_DIR = os.path.join(PROJECT_ROOT, "California2020Dataset_Day1")
OUT_DIR = os.path.join(PROJECT_ROOT, "California2020Dataset")
CA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract/tl_2024_06_tract.shp")
URBAN_SHP = os.path.join(DATA_DIR, "tl_2025_us_uac20/tl_2025_us_uac20.shp")

def main():
    # Load one Day 1 file to get shape
    sample = np.load(next(Path(D1_DIR).glob("wfpi_day1_*.npy")))[0]
    GRID_H, GRID_W = sample.shape

    # Ever burnable: value < 249 or value == 250 (snow not excluded)
    ever_burnable = np.zeros((GRID_H, GRID_W), dtype=bool)
    for f in sorted(Path(D1_DIR).glob("wfpi_day1_*.npy")):
        d = np.load(f)[0]
        ever_burnable |= (d < 249) | (d == 250)

    print(f"Ever burnable (Day 1, 366 days): {ever_burnable.sum():,}")

    mask = ever_burnable.astype(np.float32)

    # Get WFPI CRS and cropped transform for rasterize
    WFPI_ZIP_DIR = os.path.join(DATA_DIR, "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
    sample_zip = next(Path(WFPI_ZIP_DIR).glob("wfpi-forecast-2_data_*.zip"))
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(sample_zip)) as z:
            z.extractall(tmp)
        tif = next(f for f in Path(tmp).rglob("*") if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml"))
        with rasterio.open(str(tif)) as src:
            RAW_T = src.transform
            WFPI_CRS = src.crs
            raw_H, raw_W = src.height, src.width

    ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()
    ca_wfpi = ca_boundary.to_crs(WFPI_CRS)
    minx, miny, maxx, maxy = ca_wfpi.total_bounds
    buf = 50_000
    minx -= buf; miny -= buf; maxx += buf; maxy += buf
    row_min, col_min = rasterio.transform.rowcol(RAW_T, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(RAW_T, maxx, miny)
    row_min = max(0, int(row_min)); col_min = max(0, int(col_min))
    row_max = min(raw_H, int(row_max) + 1); col_max = min(raw_W, int(col_max) + 1)
    CROPPED_T = Affine(
        RAW_T.a, RAW_T.b, RAW_T.c + col_min * RAW_T.a,
        RAW_T.d, RAW_T.e, RAW_T.f + row_min * RAW_T.e,
    )

    ca_raster = rasterize(
        [(ca_wfpi.unary_union, 1)],
        out_shape=(GRID_H, GRID_W),
        transform=CROPPED_T,
        fill=0,
        dtype=np.float32,
        all_touched=True,
    )
    mask[ca_raster == 0] = 0

    urban_gdf = gpd.read_file(URBAN_SHP).to_crs(WFPI_CRS)
    urban_gdf["geometry"] = urban_gdf.buffer(0)
    urban_raster = rasterize(
        [(g, 1) for g in urban_gdf.geometry],
        out_shape=(GRID_H, GRID_W),
        transform=CROPPED_T,
        fill=0,
        dtype=np.float32,
        all_touched=True,
    )
    mask[urban_raster == 1] = 0

    labeled, n = ndimage.label(mask == 1)
    if n > 0:
        sizes = ndimage.sum(mask == 1, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1)).astype(np.float32)

    out_path = os.path.join(OUT_DIR, "mask_union_burnable_no_snow_excluded_day1.npy")
    np.save(out_path, mask)
    print(f"Valid cells: {(mask == 1).sum():,}")
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
