#!/usr/bin/env python3
"""
Explore and filter California 2020 ignition points from USFS dataset.

Filters:
  - State: California only (UNIQFIREID starts with '2020-CA')
  - Year: 2020 (FIREYEAR == 2020)
  - Type: Wildfire only (FIRETYPECATEGORY == 'WF')
  - Boundary: Coordinates must fall within California state polygon
  - Urban: Excludes fires within urban areas (US Census Urban Area Criteria 2025)
  - Validity: Must have valid discovery date and non-null lat/lon

Outputs:
  - Plot saved to report/california_2020_ignition_points.png
  - Summary saved to report/california_2020_ignition_points.md
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from shapely.geometry import Point
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

CSV_PATH      = os.path.join(SCRIPT_DIR, "data/USFS_ignition_points.csv")
CA_TRACTS     = os.path.join(SCRIPT_DIR, "data/tl_2024_06_tract/tl_2024_06_tract.shp")
URBAN_SHP     = os.path.join(SCRIPT_DIR, "data/tl_2025_us_uac20/tl_2025_us_uac20.shp")
REPORT_DIR    = os.path.join(PROJECT_ROOT, "report")
PLOT_OUT      = os.path.join(REPORT_DIR, "california_2020_ignition_points.png")
MARKDOWN_OUT  = os.path.join(REPORT_DIR, "california_2020_ignition_points.md")

os.makedirs(REPORT_DIR, exist_ok=True)

# ── Step 1: Load & filter CSV ──────────────────────────────────────────────────
print("[1/5] Loading USFS ignition points CSV …")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Total records: {len(df):,}")

# California 2020 wildfires
df_ca_2020 = df[
    (df["FIREYEAR"] == 2020) &
    (df["UNIQFIREID"].str.startswith("2020-CA", na=False)) &
    (df["FIRETYPECATEGORY"] == "WF") &
    (df["LATDD83"].notna()) &
    (df["LONGDD83"].notna()) &
    (df["DISCOVERYDATETIME"].notna())
].copy()

print(f"  CA 2020 WF fires (pre-urban filter): {len(df_ca_2020):,}")

# Parse discovery datetime
df_ca_2020["discovery_dt"] = pd.to_datetime(
    df_ca_2020["DISCOVERYDATETIME"], errors="coerce", utc=True
)
df_ca_2020 = df_ca_2020[df_ca_2020["discovery_dt"].notna()].copy()
print(f"  CA 2020 WF fires with valid datetime: {len(df_ca_2020):,}")

# ── Step 2: Load California boundary (needed for spatial filter) ───────────────
print("[2/5] Loading California boundary …")
ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
ca_tracts["geometry"] = ca_tracts.buffer(0)
ca_boundary = ca_tracts.dissolve()

# ── Step 3: Create GeoDataFrame, filter to CA boundary, apply urban filter ─────
print("[3/5] Applying spatial & urban filters …")
all_fire_gdf = gpd.GeoDataFrame(
    df_ca_2020,
    geometry=[Point(lon, lat) for lon, lat in zip(df_ca_2020["LONGDD83"], df_ca_2020["LATDD83"])],
    crs="EPSG:4326"
)

# Spatial filter: keep only fires whose coordinates fall within California
fires_in_ca = gpd.sjoin(all_fire_gdf, ca_boundary[["geometry"]], how="inner", predicate="within")
outside_ca_gdf = all_fire_gdf[~all_fire_gdf.index.isin(fires_in_ca.index)].copy()
n_outside_ca = len(outside_ca_gdf)
fire_gdf = all_fire_gdf.loc[fires_in_ca.index].copy()
print(f"  Outside California boundary removed: {n_outside_ca:,}")
print(f"  After boundary filter: {len(fire_gdf):,}")

# Load urban areas
urban_gdf = gpd.read_file(URBAN_SHP).to_crs("EPSG:4326")
urban_gdf["geometry"] = urban_gdf.buffer(0)

# Spatial join – mark urban fires
fires_in_urban = gpd.sjoin(fire_gdf, urban_gdf[["geometry"]], how="inner", predicate="within")
urban_ids = set(fires_in_urban.index.tolist())

fire_gdf["is_urban"] = fire_gdf.index.isin(urban_ids)
urban_fires_gdf = fire_gdf[fire_gdf["is_urban"]].copy()
non_urban_gdf   = fire_gdf[~fire_gdf["is_urban"]].copy()

n_urban    = len(urban_fires_gdf)
n_nonurban = len(non_urban_gdf)
print(f"  Urban fires removed: {n_urban:,}")
print(f"  Non-urban fires kept: {n_nonurban:,}")

# ── Step 4: Build the plot ─────────────────────────────────────────────────────
print("[4/5] Generating plot …")

SIZE_CLASS_ORDER = list("ABCDEFGHIJK")
SIZE_CLASS_LABELS = {
    "A": "A  (≤0.25 ac)",
    "B": "B  (0.26–9.9 ac)",
    "C": "C  (10–99 ac)",
    "D": "D  (100–299 ac)",
    "E": "E  (300–999 ac)",
    "F": "F  (1,000–4,999 ac)",
    "G": "G  (5,000–9,999 ac)",
    "H": "H  (10,000–49,999 ac)",
    "I": "I  (50,000–99,999 ac)",
    "J": "J  (100,000–299,999 ac)",
    "K": "K  (≥300,000 ac)",
}
CMAP = plt.cm.plasma_r
N = len(SIZE_CLASS_ORDER)
colors = {cls: CMAP(i / (N - 1)) for i, cls in enumerate(SIZE_CLASS_ORDER)}
sizes  = {cls: max(8, 8 + 4 * i) for i, cls in enumerate(SIZE_CLASS_ORDER)}

fig, ax = plt.subplots(figsize=(10, 13))

# California boundary
ca_boundary.plot(ax=ax, color="#e8f4e8", edgecolor="#888888", linewidth=0.8, zorder=1)

# ── Filtered-out fires (shown first, underneath) ──
if not outside_ca_gdf.empty:
    ax.scatter(
        outside_ca_gdf["LONGDD83"], outside_ca_gdf["LATDD83"],
        c="#999999", s=22, marker="x", linewidths=1.2, alpha=0.85,
        zorder=2, label=f"Outside CA boundary (n={n_outside_ca:,})"
    )
if not urban_fires_gdf.empty:
    ax.scatter(
        urban_fires_gdf["LONGDD83"], urban_fires_gdf["LATDD83"],
        c="#e07b39", s=22, marker="^", linewidths=0.4, alpha=0.80,
        edgecolors="white",
        zorder=2, label=f"Urban area — filtered (n={n_urban:,})"
    )

# ── Kept fires coloured by size class ──
for cls in SIZE_CLASS_ORDER:
    subset = non_urban_gdf[non_urban_gdf["SIZECLASS"] == cls]
    if subset.empty:
        continue
    ax.scatter(
        subset["LONGDD83"], subset["LATDD83"],
        c=[colors[cls]], s=sizes[cls],
        alpha=0.75, linewidths=0.3, edgecolors="white",
        zorder=3, label=f"{SIZE_CLASS_LABELS.get(cls, cls)}  (n={len(subset):,})"
    )

# Month distribution as a small inset bar chart
months_counts = non_urban_gdf["discovery_dt"].dt.month.value_counts().sort_index()
ax_inset = fig.add_axes([0.13, 0.12, 0.28, 0.16])
ax_inset.bar(months_counts.index, months_counts.values, color="#4a90d9", edgecolor="white", linewidth=0.5)
ax_inset.set_xticks(range(1, 13))
ax_inset.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=7)
ax_inset.set_ylabel("# fires", fontsize=7)
ax_inset.set_title("Discovery month (kept)", fontsize=8, pad=3)
ax_inset.tick_params(axis="y", labelsize=7)
ax_inset.spines[["top","right"]].set_visible(False)

# Legend — split into two: filter reasons first, then size classes
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title="Legend",
    loc="lower right",
    fontsize=7.5, title_fontsize=9,
    framealpha=0.92, edgecolor="#cccccc",
    markerscale=1.2
)

ax.set_xlim(-124.8, -113.8)
ax.set_ylim(32.2, 42.2)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
n_filtered_s1 = n_outside_ca + n_urban
ax.set_title(
    f"California 2020 — Stage-1 Filter Results\n"
    f"{n_nonurban:,} kept  ·  {n_outside_ca:,} outside CA  ·  {n_urban:,} urban",
    fontsize=12, fontweight="bold", pad=12
)
ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved plot → {PLOT_OUT}")

# ── Step 5: Write Markdown summary ────────────────────────────────────────────
print("[5/5] Writing markdown report …")

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_str = "  \n".join(
    f"  - {month_names[m-1]}: {cnt}"
    for m, cnt in months_counts.items()
)

size_counts = non_urban_gdf["SIZECLASS"].value_counts().reindex(SIZE_CLASS_ORDER).dropna().astype(int)
size_table_rows = "\n".join(
    f"| {cls} | {SIZE_CLASS_LABELS.get(cls, cls).split('(')[1].rstrip(')')} | {cnt:,} |"
    for cls, cnt in size_counts.items()
)

cause_counts = non_urban_gdf["STATCAUSE"].value_counts()
cause_rows = "\n".join(
    f"| {cause} | {cnt:,} | {100*cnt/n_nonurban:.1f}% |"
    for cause, cnt in cause_counts.items()
)

markdown = f"""# California 2020 Non-Urban Wildfire Ignition Points

## Overview

This report summarises the filtering of the USFS ignition points dataset to extract
non-urban wildfires that occurred in California during 2020.

**Source file:** `USFS_ignition_points.csv`  
**Plot:** `california_2020_ignition_points.png`

---

## Filtering Criteria

| Criterion | Value |
|-----------|-------|
| State | California (`UNIQFIREID` starts with `2020-CA`) |
| Year | 2020 (`FIREYEAR == 2020`) |
| Fire Type | Wildfire only (`FIRETYPECATEGORY == 'WF'`) |
| Boundary filter | Coordinates must fall within California state polygon |
| Urban filter | Removed fires within US Census Urban Areas 2025 |
| Date validity | Must have non-null `DISCOVERYDATETIME` |
| Coordinate validity | Must have non-null `LATDD83` / `LONGDD83` |

---

## Filter Summary

| Stage | Count |
|-------|-------|
| Total records in CSV | {len(df):,} |
| CA 2020 WF fires (raw) | {len(df_ca_2020):,} |
| Outside CA boundary removed | {n_outside_ca:,} |
| Urban fires removed | {n_urban:,} |
| **Non-urban fires kept** | **{n_nonurban:,}** |

---

## Fire Size Class Distribution

| Class | Acreage range | Count |
|-------|---------------|-------|
{size_table_rows}

---

## Discovery Month Distribution

{month_str}

---

## Cause of Ignition

| Cause | Count | % |
|-------|-------|---|
{cause_rows}

---

## Spatial Coverage

| Metric | Value |
|--------|-------|
| Latitude range | {non_urban_gdf['LATDD83'].min():.4f}° – {non_urban_gdf['LATDD83'].max():.4f}° N |
| Longitude range | {non_urban_gdf['LONGDD83'].min():.4f}° – {non_urban_gdf['LONGDD83'].max():.4f}° W |
| Discovery date range | {non_urban_gdf['discovery_dt'].min().date()} – {non_urban_gdf['discovery_dt'].max().date()} |

---

## Next Steps

The filtered `non_urban_gdf` GeoDataFrame (1086 → {n_nonurban} fires) will be used as the
ignition point input for the **California 2020 dataset**, following the same pipeline as
`create_california_2020_dataset.py`:

1. Load corresponding WFPI Day-2 forecast for each fire's discovery date
2. Convert lat/lon to grid (row, col) using the cropped California WFPI transform
3. Validate against the California mask
4. Save each fire as an ignition-point scenario (`[row, col, start_timestep]`)
"""

with open(MARKDOWN_OUT, "w") as f:
    f.write(markdown)
print(f"  Saved report → {MARKDOWN_OUT}")

print("\nDone!")
print(f"  Non-urban CA 2020 fires: {n_nonurban}")
print(f"  Plot: {PLOT_OUT}")
print(f"  Report: {MARKDOWN_OUT}")
