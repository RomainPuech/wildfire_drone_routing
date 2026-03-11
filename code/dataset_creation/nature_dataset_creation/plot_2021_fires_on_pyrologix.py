#!/usr/bin/env python3
"""
Plot 2021 final filtered fires on the Pyrologix burn map.

Masked zones (mask == 0) are shown as white. Uses the same grid and filtering
as filter_wfpi_and_plot.py (Day 1 mask, 2021 USFS ignition points).

Output: report/california_2021_ignition_points_pyrologix.png
"""

import os
import sys
import zipfile
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS as RioCRS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point
from affine import Affine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "USFS_ignition_points.csv")
CA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract/tl_2024_06_tract.shp")
URBAN_SHP = os.path.join(DATA_DIR, "tl_2025_us_uac20/tl_2025_us_uac20.shp")
WFPI_ZIP_DIR = os.path.join(DATA_DIR, "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
DATASET_DIR = os.path.join(PROJECT_ROOT, "California2020Dataset")
MASK_PATH = os.path.join(DATASET_DIR, "mask_union_burnable_no_snow_excluded_day1.npy")
PYROLOGIX_PATH = os.path.join(DATASET_DIR, "static_risk_pyrologix_resampled.npy")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
OUT_PNG = os.path.join(REPORT_DIR, "california_2021_ignition_points_pyrologix.png")

SIZE_CLASS_ORDER = list("ABCDEFGHIJK")
SIZE_CLASS_LABELS = {
    "A": "A  (≤0.25 ac)", "B": "B  (0.26–9.9 ac)", "C": "C  (10–99 ac)",
    "D": "D  (100–299 ac)", "E": "E  (300–999 ac)", "F": "F  (1,000–4,999 ac)",
    "G": "G  (5,000–9,999 ac)", "H": "H  (10,000–49,999 ac)",
    "I": "I  (50,000–99,999 ac)", "J": "J  (100,000–299,999 ac)", "K": "K  (≥300,000 ac)",
}


def main():
    print("Loading grid transform and mask …")
    sample_zip = next(Path(WFPI_ZIP_DIR).glob("wfpi-forecast-2_data_*.zip"), None)
    if not sample_zip:
        raise FileNotFoundError(f"No WFPI zip in {WFPI_ZIP_DIR}")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(sample_zip)) as z:
            z.extractall(tmp)
        tif_path = next(
            f for f in Path(tmp).rglob("*")
            if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml")
        )
        with rasterio.open(str(tif_path)) as src:
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
    row_min = max(0, int(np.floor(row_min))); col_min = max(0, int(np.floor(col_min)))
    row_max = min(raw_H, int(np.ceil(row_max)) + 1); col_max = min(raw_W, int(np.ceil(col_max)) + 1)
    GRID_H = row_max - row_min
    GRID_W = col_max - col_min
    CROPPED_T = Affine(
        RAW_T.a, RAW_T.b, RAW_T.c + col_min * RAW_T.a,
        RAW_T.d, RAW_T.e, RAW_T.f + row_min * RAW_T.e,
    )
    transformer = Transformer.from_crs("EPSG:4326", WFPI_CRS, always_xy=True)

    def latlon_to_rowcol(lat, lon):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(CROPPED_T, x, y)
        return int(r), int(c)

    mask = np.load(MASK_PATH)
    pyrologix = np.load(PYROLOGIX_PATH)[0]  # (H, W)
    # Masked zones → white (nan)
    pyro_display = np.where(mask == 1, pyrologix.astype(np.float32), np.nan)

    print("Filtering 2021 fires (stage-1 + stage-2) …")
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
    non_urban = fire_gdf[~fire_gdf.index.isin(in_urban.index)]

    # Stage-2: same as filter_wfpi_and_plot (WFPI grid + mask)
    rows, cols = [], []
    for lat, lon in zip(non_urban["LATDD83"], non_urban["LONGDD83"]):
        r, c = latlon_to_rowcol(lat, lon)
        rows.append(r)
        cols.append(c)
    non_urban = non_urban.copy()
    non_urban["_row"] = rows
    non_urban["_col"] = cols
    in_bounds = (
        (non_urban["_row"] >= 0) & (non_urban["_row"] < GRID_H)
        & (non_urban["_col"] >= 0) & (non_urban["_col"] < GRID_W)
    )
    in_bounds_gdf = non_urban[in_bounds].copy()
    in_mask = in_bounds_gdf.apply(
        lambda r: mask[int(r["_row"]), int(r["_col"])] == 1, axis=1
    )
    kept_gdf = in_bounds_gdf[in_mask].copy()
    n_kept = len(kept_gdf)
    print(f"  2021 final filtered fires: {n_kept}")

    # Reproject Pyrologix to EPSG:4326 for display
    dst_crs = RioCRS.from_epsg(4326)
    dst_transform, dst_W, dst_H = calculate_default_transform(
        WFPI_CRS, dst_crs, GRID_W, GRID_H,
        left=CROPPED_T.c,
        top=CROPPED_T.f,
        right=CROPPED_T.c + GRID_W * CROPPED_T.a,
        bottom=CROPPED_T.f + GRID_H * CROPPED_T.e,
    )
    pyro_geo = np.full((dst_H, dst_W), np.nan, dtype=np.float32)
    reproject(
        source=pyro_display,
        destination=pyro_geo,
        src_transform=CROPPED_T,
        src_crs=WFPI_CRS,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    lon_left = dst_transform.c
    lon_right = lon_left + dst_W * dst_transform.a
    lat_top = dst_transform.f
    lat_bottom = lat_top + dst_H * dst_transform.e

    # Plot
    fig, ax = plt.subplots(figsize=(10, 13))
    im = ax.imshow(
        pyro_geo,
        extent=[lon_left, lon_right, lat_bottom, lat_top],
        origin="upper",
        cmap="YlOrRd",
        alpha=0.85,
        zorder=1,
        aspect="auto",
        vmin=0,
        vmax=255,
    )
    ax.set_facecolor("white")  # masked zones show as white (nan)
    ca_boundary.plot(ax=ax, color="none", edgecolor="#555555", linewidth=0.8, zorder=2)

    CMAP = plt.cm.plasma_r
    pt_colors = {cls: CMAP(i / (len(SIZE_CLASS_ORDER) - 1)) for i, cls in enumerate(SIZE_CLASS_ORDER)}
    pt_sizes = {cls: max(8, 8 + 4 * i) for i, cls in enumerate(SIZE_CLASS_ORDER)}
    for cls in SIZE_CLASS_ORDER:
        subset = kept_gdf[kept_gdf["SIZECLASS"] == cls] if "SIZECLASS" in kept_gdf.columns else pd.DataFrame()
        if subset.empty:
            continue
        ax.scatter(
            subset["LONGDD83"], subset["LATDD83"],
            c=[pt_colors[cls]], s=pt_sizes[cls],
            alpha=0.85, linewidths=0.3, edgecolors="white",
            zorder=4,
            label=f"{SIZE_CLASS_LABELS.get(cls, cls)}  (n={len(subset):,})",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.4, anchor=(1.0, 0.85))
    cbar.set_label("Pyrologix ignition prob (resampled, 0–255)", fontsize=8)
    ax.legend(loc="lower right", fontsize=7, title="Fire size class", title_fontsize=8, framealpha=0.92, edgecolor="#cccccc", markerscale=1.2)
    ax.set_xlim(-124.8, -113.8)
    ax.set_ylim(32.2, 42.2)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(f"California 2021 — Final filtered fires on Pyrologix burn map\nn = {n_kept:,} fires  ·  masked zones in white", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    plt.tight_layout()
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
