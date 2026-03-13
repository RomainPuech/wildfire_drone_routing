#!/usr/bin/env python3
"""
Stage-2 filter: overlay USFS ignition points on the WFPI grid.

For a fire to survive this stage its lat/lon must:
  1. Map to a (row, col) that lies inside the cropped-California WFPI grid
  2. Fall in a valid mask cell (mask[row, col] == 1)

The same mask and grid geometry (1309 × 805, 1 km resolution,
Lambert Azimuthal Equal-Area) are shared by both the 2020 and 2021
datasets, derived from the existing California2020Dataset.

Outputs (per year):
  report/california_{year}_ignition_points_wfpi.png  – fires on WFPI avg map
  report/california_{year}_ignition_points.md        – updated with wfpi section
"""

import os, sys, zipfile, tempfile, json
from datetime import date, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS as RioCRS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from affine import Affine
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")

CSV_PATH     = os.path.join(DATA_DIR, "USFS_ignition_points.csv")
CA_TRACTS    = os.path.join(DATA_DIR, "tl_2024_06_tract/tl_2024_06_tract.shp")
URBAN_SHP    = os.path.join(DATA_DIR, "tl_2025_us_uac20/tl_2025_us_uac20.shp")
WFPI_ZIP_DIR = os.path.join(DATA_DIR, "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
WFPI_2021_D2_DIR = os.path.join(DATA_DIR, "2021_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA")
DATASET_DIR  = os.path.join(PROJECT_ROOT, "California2020Dataset")
MASK_PATH    = os.path.join(DATASET_DIR, "mask_union_burnable_no_snow_excluded_day1.npy")
WFPI_AVG     = os.path.join(DATASET_DIR, "static_risk_wfpi_avg.npy")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "report")


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


# ── Step 1: Recover WFPI grid transform ───────────────────────────────────────
print("[1/6] Recovering WFPI grid transform …")

# Pick any available zip to read the raw raster metadata
sample_zip = next(
    Path(WFPI_ZIP_DIR).glob("wfpi-forecast-2_data_*.zip"),
    None
)
assert sample_zip, f"No WFPI zip found in {WFPI_ZIP_DIR}"

with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(str(sample_zip)) as z:
        z.extractall(tmp)
    tif_path = next(
        f for f in Path(tmp).rglob("*")
        if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml")
    )
    with rasterio.open(str(tif_path)) as src:
        RAW_TRANSFORM = src.transform
        WFPI_CRS      = src.crs
        raw_H, raw_W  = src.height, src.width

print(f"  Raw grid: {raw_H} × {raw_W},  CRS: {WFPI_CRS.to_epsg() or 'custom'}")

# ── Step 2: Reproduce California crop to get the cropped transform ─────────────
print("[2/6] Computing cropped-California transform …")

ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
ca_tracts["geometry"] = ca_tracts.buffer(0)
ca_boundary = ca_tracts.dissolve()
ca_wfpi_crs = ca_boundary.to_crs(WFPI_CRS)

minx, miny, maxx, maxy = ca_wfpi_crs.total_bounds
buf = 50_000  # 50 km — same buffer used in create_california_2020_dataset.py
minx -= buf; miny -= buf; maxx += buf; maxy += buf

row_min, col_min = rasterio.transform.rowcol(RAW_TRANSFORM, minx, maxy)
row_max, col_max = rasterio.transform.rowcol(RAW_TRANSFORM, maxx, miny)
row_min = max(0, int(np.floor(row_min)));  col_min = max(0, int(np.floor(col_min)))
row_max = min(raw_H, int(np.ceil(row_max)) + 1)
col_max = min(raw_W,  int(np.ceil(col_max)) + 1)

CROPPED_TRANSFORM = Affine(
    RAW_TRANSFORM.a, RAW_TRANSFORM.b,
    RAW_TRANSFORM.c + col_min * RAW_TRANSFORM.a,
    RAW_TRANSFORM.d, RAW_TRANSFORM.e,
    RAW_TRANSFORM.f + row_min * RAW_TRANSFORM.e,
)
GRID_H = row_max - row_min
GRID_W = col_max - col_min

print(f"  Cropped grid: {GRID_H} × {GRID_W}  (expected 1309 × 805)")

# ── Step 3: Load mask & WFPI avg ──────────────────────────────────────────────
print("[3/6] Loading California mask and WFPI average map …")
mask     = np.load(MASK_PATH)        # (H, W)
wfpi_avg = np.load(WFPI_AVG)[0]     # (H, W)
print(f"  Mask shape: {mask.shape},  valid cells: {int((mask==1).sum()):,}")

# Pyproj transformer (EPSG:4326 → WFPI CRS)
TRANSFORMER = Transformer.from_crs("EPSG:4326", WFPI_CRS, always_xy=True)

def latlon_to_rowcol(lat, lon):
    x, y = TRANSFORMER.transform(lon, lat)
    row, col = rasterio.transform.rowcol(CROPPED_TRANSFORM, x, y)
    return int(row), int(col)

def wfpi_filter(gdf):
    """
    Keep only fires whose (row, col) is inside the grid and in a valid mask cell.
    Returns (kept_gdf, reasons) where reasons is a dict of drop counts.
    """
    rows, cols = [], []
    for lat, lon in zip(gdf["LATDD83"], gdf["LONGDD83"]):
        r, c = latlon_to_rowcol(lat, lon)
        rows.append(r); cols.append(c)

    gdf = gdf.copy()
    gdf["_row"] = rows
    gdf["_col"] = cols

    in_bounds = (
        (gdf["_row"] >= 0) & (gdf["_row"] < GRID_H) &
        (gdf["_col"] >= 0) & (gdf["_col"] < GRID_W)
    )
    n_oob = (~in_bounds).sum()

    in_bounds_gdf = gdf[in_bounds].copy()
    in_mask = in_bounds_gdf.apply(
        lambda r: mask[int(r["_row"]), int(r["_col"])] == 1, axis=1
    )
    n_masked = (~in_mask).sum()

    kept = in_bounds_gdf[in_mask].copy()
    return kept, {"out_of_bounds": int(n_oob), "in_masked_cell": int(n_masked)}

# ── Step 4: Run Stage-1 + Stage-2 filters for both years ─────────────────────
print("[4/6] Running filters for 2020 and 2021 …")

df_all   = pd.read_csv(CSV_PATH, low_memory=False)
urban_gdf = gpd.read_file(URBAN_SHP).to_crs("EPSG:4326")
urban_gdf["geometry"] = urban_gdf.buffer(0)

results = {}
for YEAR in (2020, 2021):
    print(f"\n  ── Year {YEAR} ──")

    # Stage-1a: year + state code + fire type
    df_y = df_all[
        (df_all["FIREYEAR"] == YEAR) &
        (df_all["UNIQFIREID"].str.startswith(f"{YEAR}-CA", na=False)) &
        (df_all["FIRETYPECATEGORY"] == "WF") &
        (df_all["LATDD83"].notna()) &
        (df_all["LONGDD83"].notna()) &
        (df_all["DISCOVERYDATETIME"].notna())
    ].copy()
    df_y["discovery_dt"] = pd.to_datetime(df_y["DISCOVERYDATETIME"], errors="coerce", utc=True)
    df_y = df_y[df_y["discovery_dt"].notna()].copy()
    n_raw = len(df_y)
    print(f"    Raw CA WF fires: {n_raw:,}")

    fire_gdf = gpd.GeoDataFrame(
        df_y,
        geometry=[Point(lon, lat) for lon, lat in zip(df_y["LONGDD83"], df_y["LATDD83"])],
        crs="EPSG:4326"
    )

    # Stage-1b: California boundary filter
    in_ca = gpd.sjoin(fire_gdf, ca_boundary[["geometry"]], how="inner", predicate="within")
    outside_ca_gdf = fire_gdf[~fire_gdf.index.isin(in_ca.index)].copy()
    n_out_ca = len(outside_ca_gdf)
    fire_gdf = fire_gdf.loc[in_ca.index].copy()
    print(f"    Outside CA boundary: {n_out_ca:,}")

    # Stage-1c: Urban filter
    in_urban = gpd.sjoin(fire_gdf, urban_gdf[["geometry"]], how="inner", predicate="within")
    urban_ids = set(in_urban.index.tolist())
    fire_gdf["is_urban"] = fire_gdf.index.isin(urban_ids)
    urban_fires_gdf = fire_gdf[fire_gdf["is_urban"]].copy()
    n_urban = len(urban_fires_gdf)
    non_urban_gdf = fire_gdf[~fire_gdf["is_urban"]].copy()
    print(f"    Urban fires removed: {n_urban:,}")
    print(f"    After stage-1: {len(non_urban_gdf):,}")

    # Stage-2: WFPI grid + mask filter
    kept_gdf, drop_reasons = wfpi_filter(non_urban_gdf)
    wfpi_dropped_gdf = non_urban_gdf[~non_urban_gdf.index.isin(kept_gdf.index)].copy()
    n_wfpi_dropped = sum(drop_reasons.values())
    print(f"    WFPI out-of-bounds: {drop_reasons['out_of_bounds']:,}")
    print(f"    WFPI masked cell:   {drop_reasons['in_masked_cell']:,}")
    print(f"    After stage-2: {len(kept_gdf):,}")

    # For 2021 only: exclude fires whose discovery date has no WFPI zip (use nearest-neighbour in dataset)
    excluded_missing_wfpi_gdf = pd.DataFrame()
    n_excluded_missing_wfpi = 0
    if YEAR == 2021:
        missing_dates = set(get_missing_2021_wfpi_dates())
        if missing_dates:
            kept_gdf["_date_str"] = kept_gdf["discovery_dt"].apply(
                lambda x: x.strftime("%Y%m%d") if hasattr(x, "strftime") else str(x)[:10].replace("-", "")
            )
            on_missing = kept_gdf["_date_str"].isin(missing_dates)
            excluded_missing_wfpi_gdf = kept_gdf[on_missing].copy()
            kept_gdf = kept_gdf[~on_missing].copy()
            kept_gdf = kept_gdf.drop(columns=["_date_str"], errors="ignore")
            n_excluded_missing_wfpi = len(excluded_missing_wfpi_gdf)
            excluded_missing_wfpi_gdf = excluded_missing_wfpi_gdf.drop(columns=["_date_str"], errors="ignore")
            print(f"    Excluded (missing WFPI zip date): {n_excluded_missing_wfpi:,}")
            print(f"    Kept (in dataset): {len(kept_gdf):,}")

    results[YEAR] = {
        "n_raw":            n_raw,
        "n_out_ca":         n_out_ca,
        "n_urban":          n_urban,
        "n_stage1":         len(non_urban_gdf),
        "drop_oob":         drop_reasons["out_of_bounds"],
        "drop_masked":      drop_reasons["in_masked_cell"],
        "n_final":          len(kept_gdf),
        "outside_ca_gdf":   outside_ca_gdf,
        "urban_fires_gdf":  urban_fires_gdf,
        "wfpi_dropped_gdf": wfpi_dropped_gdf,
        "non_urban_gdf":    non_urban_gdf,
        "kept_gdf":         kept_gdf,
        "excluded_missing_wfpi_gdf": excluded_missing_wfpi_gdf,
        "n_excluded_missing_wfpi": n_excluded_missing_wfpi,
    }

# ── Step 5: Generate WFPI-overlay plots ───────────────────────────────────────
print("\n[5/6] Generating WFPI-overlay plots …")

SIZE_CLASS_ORDER = list("ABCDEFGHIJK")
SIZE_CLASS_LABELS = {
    "A": "A  (≤0.25 ac)",   "B": "B  (0.26–9.9 ac)", "C": "C  (10–99 ac)",
    "D": "D  (100–299 ac)", "E": "E  (300–999 ac)",   "F": "F  (1,000–4,999 ac)",
    "G": "G  (5,000–9,999 ac)", "H": "H  (10,000–49,999 ac)",
    "I": "I  (50,000–99,999 ac)", "J": "J  (100,000–299,999 ac)", "K": "K  (≥300,000 ac)",
}
CMAP = plt.cm.plasma_r
N    = len(SIZE_CLASS_ORDER)
pt_colors = {cls: CMAP(i / (N - 1)) for i, cls in enumerate(SIZE_CLASS_ORDER)}
pt_sizes  = {cls: max(8, 8 + 4 * i) for i, cls in enumerate(SIZE_CLASS_ORDER)}

# WFPI avg: mask invalid cells, then reproject LAEA → EPSG:4326
wfpi_src = np.where(mask == 1, wfpi_avg, np.nan).astype(np.float32)

dst_crs = RioCRS.from_epsg(4326)
dst_transform, dst_W, dst_H = calculate_default_transform(
    WFPI_CRS, dst_crs, GRID_W, GRID_H,
    left   = CROPPED_TRANSFORM.c,
    top    = CROPPED_TRANSFORM.f,
    right  = CROPPED_TRANSFORM.c + GRID_W * CROPPED_TRANSFORM.a,
    bottom = CROPPED_TRANSFORM.f + GRID_H * CROPPED_TRANSFORM.e,
)
wfpi_geo = np.full((dst_H, dst_W), np.nan, dtype=np.float32)
reproject(
    source      = wfpi_src,
    destination = wfpi_geo,
    src_transform = CROPPED_TRANSFORM,
    src_crs       = WFPI_CRS,
    dst_transform = dst_transform,
    dst_crs       = dst_crs,
    resampling    = Resampling.bilinear,
    src_nodata    = np.nan,
    dst_nodata    = np.nan,
)

# Extent for imshow: [left, right, bottom, top] in lon/lat
lon_left   = dst_transform.c
lon_right  = lon_left  + dst_W * dst_transform.a
lat_top    = dst_transform.f
lat_bottom = lat_top   + dst_H * dst_transform.e  # e is negative

for YEAR in (2020, 2021):
    r  = results[YEAR]
    kept_gdf         = r["kept_gdf"]
    wfpi_dropped_gdf = r["wfpi_dropped_gdf"]
    outside_ca_gdf   = r["outside_ca_gdf"]
    urban_fires_gdf  = r["urban_fires_gdf"]
    n_final          = r["n_final"]
    n_wfpi           = r["drop_oob"] + r["drop_masked"]
    excluded_missing_wfpi_gdf = r.get("excluded_missing_wfpi_gdf", pd.DataFrame())
    n_excluded_missing = r.get("n_excluded_missing_wfpi", 0)

    fig, ax = plt.subplots(figsize=(10, 13))

    # WFPI avg as background (reprojected to EPSG:4326)
    ax.imshow(
        wfpi_geo,
        extent=[lon_left, lon_right, lat_bottom, lat_top],
        origin="upper", cmap="YlOrRd", alpha=0.65,
        zorder=1, aspect="auto", vmin=0, vmax=200,
    )
    ca_boundary.plot(ax=ax, color="none", edgecolor="#555555", linewidth=0.8, zorder=2)

    # ── Stage-1 drops (bottom layer) ──
    if not outside_ca_gdf.empty:
        ax.scatter(
            outside_ca_gdf["LONGDD83"], outside_ca_gdf["LATDD83"],
            c="#888888", s=20, marker="x", linewidths=1.2, alpha=0.80,
            zorder=3, label=f"Outside CA boundary (n={r['n_out_ca']:,})"
        )
    if not urban_fires_gdf.empty:
        ax.scatter(
            urban_fires_gdf["LONGDD83"], urban_fires_gdf["LATDD83"],
            c="#e07b39", s=18, marker="^", linewidths=0.3, alpha=0.75,
            edgecolors="white",
            zorder=3, label=f"Urban area / stage-1 (n={r['n_urban']:,})"
        )

    # ── Stage-2 WFPI drops ──
    if not wfpi_dropped_gdf.empty:
        ax.scatter(
            wfpi_dropped_gdf["LONGDD83"], wfpi_dropped_gdf["LATDD83"],
            c="#bbbbbb", s=7, marker="o", linewidths=0, alpha=0.55,
            zorder=4, label=f"WFPI masked / stage-2 (n={n_wfpi:,})"
        )

    # ── 2021 only: excluded (discovery on missing WFPI zip date) ──
    if YEAR == 2021 and not excluded_missing_wfpi_gdf.empty:
        ax.scatter(
            excluded_missing_wfpi_gdf["LONGDD83"], excluded_missing_wfpi_gdf["LATDD83"],
            c="#9b59b6", s=28, marker="D", linewidths=0.5, alpha=0.9,
            edgecolors="white", zorder=4.5,
            label=f"Excluded — missing WFPI zip date (n={n_excluded_missing:,})"
        )

    # ── Kept fires coloured by size class ──
    for cls in SIZE_CLASS_ORDER:
        subset = kept_gdf[kept_gdf["SIZECLASS"] == cls]
        if subset.empty:
            continue
        ax.scatter(
            subset["LONGDD83"], subset["LATDD83"],
            c=[pt_colors[cls]], s=pt_sizes[cls],
            alpha=0.85, linewidths=0.3, edgecolors="white",
            zorder=5,
            label=f"{SIZE_CLASS_LABELS.get(cls, cls)}  (n={len(subset):,})"
        )

    # Colorbar for WFPI
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(0, 200))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.4,
                        anchor=(1.0, 0.85))
    cbar.set_label("WFPI avg (0–200)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Legend",
              loc="lower right", fontsize=7, title_fontsize=8,
              framealpha=0.92, edgecolor="#cccccc", markerscale=1.2)

    ax.set_xlim(-124.8, -113.8)
    ax.set_ylim(32.2, 42.2)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    title_line2 = f"{n_final:,} kept  ·  {n_wfpi:,} WFPI-masked  ·  {r['n_out_ca'] + r['n_urban']:,} stage-1 removed"
    if YEAR == 2021 and n_excluded_missing > 0:
        title_line2 += f"  ·  {n_excluded_missing:,} excluded (missing WFPI date)"
    ax.set_title(
        f"California {YEAR} — All Filter Stages on WFPI Background\n{title_line2}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    plt.tight_layout()

    out_png = os.path.join(REPORT_DIR, f"california_{YEAR}_ignition_points_wfpi.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_png}")

# ── Step 6: Update markdown reports ───────────────────────────────────────────
print("\n[6/6] Updating markdown reports …")

for YEAR in (2020, 2021):
    r = results[YEAR]
    md_path = os.path.join(REPORT_DIR, f"california_{YEAR}_ignition_points.md")

    with open(md_path) as f:
        existing = f.read()

    excluded_row = (f"| Excluded — discovery on missing WFPI zip date | {r['n_excluded_missing_wfpi']:,} |\n" if r.get("n_excluded_missing_wfpi", 0) > 0 else "")
    wfpi_section = f"""
---

## Stage 2: WFPI Grid & Mask Filter

After the initial stage-1 filters (unit code, year, fire type, California
boundary, urban exclusion), ignition points are overlaid on the WFPI 1 km
grid (1309 × 805 cells, Lambert Azimuthal Equal-Area) that underlies the
California2020Dataset.  A fire is **kept** only if:

1. Its lat/lon maps to a `(row, col)` inside the cropped-California grid
2. `mask[row, col] == 1` — i.e. the cell is inside the California state
   boundary, is not always unburnable (see mask logic below), and is not an urban area

> **Mask logic:** a cell is valid if it has WFPI < 249 or WFPI = 250 (snow)
> on at least one day of 2020 (union-of-burnable, snow not excluded).
> Mask built from **2020 WFPI Day 1** daily maps (`mask_union_burnable_no_snow_excluded_day1.npy`). Cells always in 249 or 251-255 are excluded.

| Stage | Count |
|-------|-------|
| After stage-1 (boundary + urban) | {r['n_stage1']:,} |
| Removed — out of WFPI grid bounds | {r['drop_oob']:,} |
| Removed — WFPI masked cell (urban / nodata / outside CA) | {r['drop_masked']:,} |
{excluded_row}| **Kept after stage-2 (in dataset)** | **{r['n_final']:,}** |

**WFPI-overlay plot:** `california_{YEAR}_ignition_points_wfpi.png`
"""

    # Append only if not already present
    if "Stage 2: WFPI" not in existing:
        with open(md_path, "a") as f:
            f.write(wfpi_section)
        print(f"  Updated → {md_path}")
    else:
        # Replace existing section
        import re
        existing = re.sub(
            r"\n---\n\n## Stage 2: WFPI Grid.*",
            wfpi_section,
            existing,
            flags=re.DOTALL
        )
        with open(md_path, "w") as f:
            f.write(existing)
        print(f"  Re-wrote → {md_path}")

print("\n=== Done ===")
for YEAR in (2020, 2021):
    r = results[YEAR]
    print(f"  {YEAR}: {r['n_raw']:,} raw → {r['n_stage1']:,} stage-1 → {r['n_final']:,} final")
