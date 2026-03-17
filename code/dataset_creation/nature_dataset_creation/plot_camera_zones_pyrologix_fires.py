#!/usr/bin/env python3
"""
Overlay camera zones (cameras.json), masked Pyrologix map, and 2021 fires dataset.

Uses same grid/mask and 2021 fire filtering as plot_2021_fires_on_pyrologix.py.
Output: camera_zones_pyrologix_2021_fires.png
"""

import argparse
import os
import zipfile
import tempfile
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS as RioCRS
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pyproj import Transformer
from shapely.geometry import Point
from affine import Affine

CA_ALBERS = "EPSG:3310"  # California Albers (meters) for circle buffer
MILES_TO_METERS = 1609.344


def zones_to_full_circles(zones_gdf, radius_miles=20):
    """Replace each zone polygon (sector) with a full circle. Center from properties; radius = radius_miles (default 20)."""
    radius_m = radius_miles * MILES_TO_METERS
    zones_m = zones_gdf.to_crs(CA_ALBERS)
    circles = []
    for idx, row in zones_m.iterrows():
        lon = row.get("longitude", None)
        lat = row.get("latitude", None)
        if lon is None or lat is None or (hasattr(lon, "__float__") and (lon != lon or lat != lat)):
            circles.append(row.geometry)
            continue
        center_wgs = Point(float(lon), float(lat))
        center_m = gpd.GeoSeries([center_wgs], crs="EPSG:4326").to_crs(CA_ALBERS).iloc[0]
        circles.append(center_m.buffer(radius_m))
    out = gpd.GeoDataFrame(
        zones_gdf.drop(columns=["geometry"]),
        geometry=circles,
        crs=CA_ALBERS,
    )
    return out.to_crs("EPSG:4326")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CAMERAS_JSON = os.path.join(DATA_DIR, "cameras.json")
CSV_PATH = os.path.join(DATA_DIR, "USFS_ignition_points.csv")
CA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract", "tl_2024_06_tract.shp")
URBAN_SHP = os.path.join(DATA_DIR, "tl_2025_us_uac20", "tl_2025_us_uac20.shp")
WFPI_ZIP_DIR = os.path.join(DATA_DIR, "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
WFPI_2021_D2_DIR = os.path.join(DATA_DIR, "2021_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
DATASET_DIR = os.path.join(PROJECT_ROOT, "California2020Dataset")
MASK_PATH = os.path.join(DATASET_DIR, "mask_union_burnable_no_snow_excluded_day1.npy")
PYROLOGIX_PATH = os.path.join(DATASET_DIR, "static_risk_pyrologix_resampled.npy")

SIZE_CLASS_ORDER = list("ABCDEFGHIJK")
SIZE_CLASS_LABELS = {
    "A": "A  (≤0.25 ac)", "B": "B  (0.26–9.9 ac)", "C": "C  (10–99 ac)",
    "D": "D  (100–299 ac)", "E": "E  (300–999 ac)", "F": "F  (1,000–4,999 ac)",
    "G": "G  (5,000–9,999 ac)", "H": "H  (10,000–49,999 ac)",
    "I": "I  (50,000–99,999 ac)", "J": "J  (100,000–299,999 ac)", "K": "K  (≥300,000 ac)",
}


def get_missing_2021_wfpi_dates():
    """Return set of YYYYMMDD for which 2021 D2 zip is missing."""
    d2_dir = Path(WFPI_2021_D2_DIR)
    if not d2_dir.exists():
        return set()
    have = set()
    for f in d2_dir.glob("wfpi-forecast-2_data_*_*.zip"):
        parts = f.stem.split("_")
        if len(parts) >= 4 and len(parts[3]) == 8:
            have.add(parts[3])
    all_2021 = set((date(2021, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(365))
    return all_2021 - have


def main():
    parser = argparse.ArgumentParser(description="Overlay camera zones (circles), Pyrologix map, and 2021 fires.")
    parser.add_argument("--radius-miles", type=float, default=20, help="Camera zone circle radius in miles (default: 20)")
    args = parser.parse_args()
    radius_miles = args.radius_miles

    out_png = os.path.join(SCRIPT_DIR, "camera_zones_pyrologix_2021_fires.png")
    if radius_miles != 20:
        out_png = os.path.join(SCRIPT_DIR, f"camera_zones_pyrologix_2021_fires_{int(radius_miles)}mi.png")

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

    def latlon_to_rowcol(lon, lat):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(CROPPED_T, x, y)
        return int(r), int(c)

    mask = np.load(MASK_PATH)
    pyrologix = np.load(PYROLOGIX_PATH)[0]
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

    rows, cols = [], []
    for _, row in non_urban.iterrows():
        r, c = latlon_to_rowcol(row["LONGDD83"], row["LATDD83"])
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
    missing_dates = set(get_missing_2021_wfpi_dates())
    if missing_dates:
        kept_gdf["_date_str"] = kept_gdf["discovery_dt"].apply(
            lambda x: x.strftime("%Y%m%d") if hasattr(x, "strftime") else str(x)[:10].replace("-", "")
        )
        kept_gdf = kept_gdf[~kept_gdf["_date_str"].isin(missing_dates)].copy()
        kept_gdf = kept_gdf.drop(columns=["_date_str"], errors="ignore")
    n_kept = len(kept_gdf)
    print(f"  2021 final filtered fires: {n_kept:,}")

    # Reproject Pyrologix to EPSG:4326 for display
    dst_crs = RioCRS.from_epsg(4326)
    dst_transform, dst_W, dst_H = calculate_default_transform(
        WFPI_CRS, dst_crs, GRID_W, GRID_H,
        left=CROPPED_T.c, top=CROPPED_T.f,
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
    extent = [lon_left, lon_right, lat_bottom, lat_top]

    print("Loading camera zones …")
    zones = gpd.read_file(CAMERAS_JSON)
    if zones.crs is None:
        zones.set_crs("EPSG:4326", inplace=True)
    else:
        zones = zones.to_crs("EPSG:4326")
    print(f"Converting zone sectors to full circles (radius={radius_miles} mi) …")
    zones = zones_to_full_circles(zones, radius_miles=radius_miles)

    # Classify fires: discoverable = within at least one camera zone circle
    fire_gdf = kept_gdf[["geometry", "LONGDD83", "LATDD83", "SIZECLASS"]].copy()
    joined = gpd.sjoin(fire_gdf, zones[["geometry"]], how="left", predicate="within")
    discoverable_series = joined.groupby(level=0)["index_right"].apply(lambda x: x.notna().any())
    discoverable_mask = discoverable_series.reindex(kept_gdf.index, fill_value=False).values
    n_discoverable = int(discoverable_mask.sum())
    n_non_discoverable = len(kept_gdf) - n_discoverable
    print(f"  Discoverable (within ≥1 zone): {n_discoverable:,}  ·  Non-discoverable: {n_non_discoverable:,}")

    # Plot: Pyrologix -> CA outline -> camera zones -> fire points
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    im = ax.imshow(
        pyro_geo,
        extent=extent,
        origin="upper",
        cmap="YlOrRd",
        alpha=0.9,
        zorder=1,
        aspect="auto",
        vmin=0,
        vmax=255,
    )

    ca_boundary.plot(ax=ax, color="none", edgecolor="#333333", linewidth=0.9, zorder=2)

    zones.plot(
        ax=ax,
        facecolor=to_rgba("steelblue", 0.3),
        edgecolor=to_rgba("steelblue", 0.6),
        linewidth=0.35,
        zorder=3,
    )

    # Fire points: discoverable vs non-discoverable (by color), size by class
    CMAP = plt.cm.plasma_r
    pt_sizes = {cls: max(10, 10 + 4 * i) for i, cls in enumerate(SIZE_CLASS_ORDER)}
    disc_gdf = kept_gdf[discoverable_mask]
    non_disc_gdf = kept_gdf[~discoverable_mask]
    for cls in SIZE_CLASS_ORDER:
        subset_d = disc_gdf[disc_gdf["SIZECLASS"] == cls] if "SIZECLASS" in disc_gdf.columns else pd.DataFrame()
        subset_n = non_disc_gdf[non_disc_gdf["SIZECLASS"] == cls] if "SIZECLASS" in non_disc_gdf.columns else pd.DataFrame()
        sz = pt_sizes.get(cls, 12)
        if not subset_d.empty:
            ax.scatter(
                subset_d["LONGDD83"], subset_d["LATDD83"],
                c="forestgreen", s=sz, alpha=0.9, linewidths=0.4, edgecolors="white", zorder=5,
            )
        if not subset_n.empty:
            ax.scatter(
                subset_n["LONGDD83"], subset_n["LATDD83"],
                c="darkred", s=sz, alpha=0.9, linewidths=0.4, edgecolors="white", zorder=5,
            )
    # Legend: discoverable vs non-discoverable with counts
    ax.scatter([], [], c="forestgreen", s=50, alpha=0.9, linewidths=0.4, edgecolors="white", label=f"Discoverable (n={n_discoverable:,})")
    ax.scatter([], [], c="darkred", s=50, alpha=0.9, linewidths=0.4, edgecolors="white", label=f"Non-discoverable (n={n_non_discoverable:,})")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.35, anchor=(1.0, 0.85))
    cbar.set_label("Pyrologix ignition prob (masked, 0–255)", fontsize=9)
    ax.legend(
        loc="lower right", fontsize=9, title="2021 fires (by camera coverage)",
        framealpha=0.92, edgecolor="#cccccc", markerscale=1.1,
    )
    ax.set_xlim(-124.8, -113.8)
    ax.set_ylim(32.2, 42.2)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(
        f"California: Pyrologix (masked) + camera zones ({radius_miles:.0f} mi) + 2021 fires\n"
        f"Pyrologix risk · AlertCalifornia zones (blue) · {n_kept:,} fires (by size class)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
