#!/usr/bin/env python3
"""
Build figures and counts for report/paper_2021_dataset_creation_explainer/
(co-located paper_2021_dataset_creation_explainer.md and counts.md).

Uses Pyrologix on the 1 km grid, cumulative mask steps (CA → −urban → WFPI union
burnable → largest CC), and 2021 USFS fire categories aligned with
filter_wfpi_and_plot.py. Benchmark subset matches run_benchmark_california2021_yearly.py.

Run from project root:
  conda run -n wf python code/dataset_creation/nature_dataset_creation/generate_paper_2021_dataset_explainer.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
# nature_dataset_creation → dataset_creation → code → repo root
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from displays import (
    make_usfs_fire_legend_handles,
    plot_pyrologix_fire_categories,
    plot_pyrologix_valid_region,
)

DATA_DIR = SCRIPT_DIR / "data"
CA_TRACTS = DATA_DIR / "tl_2024_06_tract" / "tl_2024_06_tract.shp"
URBAN_SHP = DATA_DIR / "tl_2025_us_uac20" / "tl_2025_us_uac20.shp"
WFPI_ZIP_DIR = DATA_DIR / "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA"
D1_DIR = PROJECT_ROOT / "California2020Dataset_Day1"
CSV_PATH = DATA_DIR / "USFS_ignition_points.csv"
WFPI_2021_D2_DIR = DATA_DIR / "2021_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA"

PYROLOGIX_PATH = PROJECT_ROOT / "California2021Dataset" / "static_risk_pyrologix.npy"
MASK_FINAL_PATH = PROJECT_ROOT / "California2020Dataset" / "mask_union_burnable_no_snow_excluded_day1.npy"
CONFIG_PATH = PROJECT_ROOT / "California2021Dataset" / "config_california_2021.json"
SCENARII_DIR = PROJECT_ROOT / "California2021Dataset" / "scenarii"

REPORT_DIR = PROJECT_ROOT / "report"
OUT_DIR = REPORT_DIR / "paper_2021_dataset_creation_explainer"
BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED = 42


def get_cropped_transform_and_dims():
    sample_zip = next(Path(WFPI_ZIP_DIR).glob("wfpi-forecast-2_data_*.zip"))
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(sample_zip)) as z:
            z.extractall(tmp)
        tif = next(
            f
            for f in Path(tmp).rglob("*")
            if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml")
        )
        with rasterio.open(str(tif)) as src:
            raw_t = src.transform
            wfpi_crs = src.crs
            raw_h, raw_w = src.height, src.width

    ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
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
    cropped_t = Affine(
        raw_t.a,
        raw_t.b,
        raw_t.c + col_min * raw_t.a,
        raw_t.d,
        raw_t.e,
        raw_t.f + row_min * raw_t.e,
    )
    grid_h = row_max - row_min
    grid_w = col_max - col_min
    return cropped_t, wfpi_crs, grid_h, grid_w, ca_wfpi


def compute_ever_burnable_d1(grid_h: int, grid_w: int) -> np.ndarray:
    files = sorted(D1_DIR.glob("wfpi_day1_*.npy"))
    if not files:
        raise FileNotFoundError(f"No Day1 WFPI npy in {D1_DIR}")
    ever = np.zeros((grid_h, grid_w), dtype=bool)
    for f in files:
        d = np.load(f)[0]
        ever |= (d < 249) | (d == 250)
    return ever


def rasterize_ca_and_urban(ca_wfpi, cropped_t, grid_h, grid_w, wfpi_crs):
    ca_poly = ca_wfpi.geometry.iloc[0]
    ca_raster = rasterize(
        [(ca_poly, 1)],
        out_shape=(grid_h, grid_w),
        transform=cropped_t,
        fill=0,
        dtype=np.float32,
        all_touched=True,
    )
    urban_gdf = gpd.read_file(URBAN_SHP).to_crs(wfpi_crs)
    urban_gdf["geometry"] = urban_gdf.buffer(0)
    urban_raster = rasterize(
        [(g, 1) for g in urban_gdf.geometry],
        out_shape=(grid_h, grid_w),
        transform=cropped_t,
        fill=0,
        dtype=np.float32,
        all_touched=True,
    )
    return ca_raster, urban_raster


def largest_cc(mask_bool: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask_bool)
    if n == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    sizes = ndimage.sum(mask_bool, labeled, range(1, n + 1))
    keep = int(np.argmax(sizes) + 1)
    return labeled == keep


def mask_keep_components_min_area_km2(m_pre_lcc: np.ndarray, side_km: float = 9.0) -> np.ndarray:
    """
    From a boolean pre–connected-component mask (~1 km cells), keep every connected
    component whose area is at least side_km × side_km (in cell count: side_km**2).
    """
    min_cells = max(1, int(round(side_km * side_km)))
    labeled, n = ndimage.label(m_pre_lcc)
    if n == 0:
        return np.zeros_like(m_pre_lcc, dtype=bool)
    sizes = ndimage.sum(m_pre_lcc, labeled, range(1, n + 1))
    out = np.zeros_like(m_pre_lcc, dtype=bool)
    for i, sz in enumerate(sizes, start=1):
        if sz >= min_cells:
            out |= labeled == i
    return out


def get_missing_2021_wfpi_dates():
    d2_dir = Path(WFPI_2021_D2_DIR)
    if not d2_dir.exists():
        return set()
    have = set()
    for f in d2_dir.glob("wfpi-forecast-2_data_*_*.zip"):
        parts = f.stem.split("_")
        if len(parts) >= 4 and len(parts[3]) == 8:
            have.add(parts[3])
    all_2021 = set(
        (date(2021, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(365)
    )
    return all_2021 - have


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Grid, ever-burnable (2020 D1), CA / urban rasters …")
    cropped_t, wfpi_crs, gh, gw, ca_wfpi = get_cropped_transform_and_dims()
    ever_burnable = compute_ever_burnable_d1(gh, gw)
    ca_raster, urban_raster = rasterize_ca_and_urban(ca_wfpi, cropped_t, gh, gw, wfpi_crs)

    in_ca = ca_raster == 1
    urban_bool = urban_raster >= 0.5
    urban_eroded = ndimage.binary_erosion(urban_bool, iterations=1)
    non_urban = ~urban_eroded
    m_pre_cc = in_ca & non_urban & ever_burnable
    components_filtered = mask_keep_components_min_area_km2(m_pre_cc, side_km=9.0)
    mask_final = ndimage.binary_dilation(components_filtered, iterations=1)
    if MASK_FINAL_PATH.is_file():
        mask_saved = np.load(MASK_FINAL_PATH).astype(bool)
        if not np.array_equal(mask_final, mask_saved):
            print(
                "  WARNING: recomputed mask != saved mask file; using saved file for fires.",
            )
    else:
        print(
            f"  WARNING: saved final mask missing at {MASK_FINAL_PATH}; "
            "using recomputed mask for figures.",
        )
        mask_saved = mask_final

    mask_step1 = in_ca
    mask_step2 = in_ca & non_urban
    mask_step3 = m_pre_cc
    mask_step4 = mask_saved

    print("[2/4] Pyrologix …")
    pyrologix = np.load(PYROLOGIX_PATH)[0].astype(np.float32)

    print("[3/4] USFS 2021 fire coordinates …")
    transformer = Transformer.from_crs("EPSG:4326", wfpi_crs, always_xy=True)

    def latlon_to_rowcol(lat, lon):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(cropped_t, x, y)
        return int(r), int(c)

    df_all = pd.read_csv(CSV_PATH, low_memory=False)
    urban_gdf = gpd.read_file(URBAN_SHP).to_crs("EPSG:4326")
    urban_gdf["geometry"] = urban_gdf.buffer(0)

    df_y = df_all[
        (df_all["FIREYEAR"] == 2021)
        & (df_all["UNIQFIREID"].str.startswith("2021-CA", na=False))
        & (df_all["FIRETYPECATEGORY"] == "WF")
        & (df_all["LATDD83"].notna())
        & (df_all["LONGDD83"].notna())
        & (df_all["DISCOVERYDATETIME"].notna())
    ].copy()
    df_y["discovery_dt"] = pd.to_datetime(df_y["DISCOVERYDATETIME"], errors="coerce", utc=True)
    df_y = df_y[df_y["discovery_dt"].notna()].copy()

    fire_gdf = gpd.GeoDataFrame(
        df_y,
        geometry=[Point(lon, lat) for lon, lat in zip(df_y["LONGDD83"], df_y["LATDD83"])],
        crs="EPSG:4326",
    )
    ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()
    in_ca_idx = gpd.sjoin(fire_gdf, ca_boundary[["geometry"]], how="inner", predicate="within").index
    outside_ca = fire_gdf[~fire_gdf.index.isin(in_ca_idx)].copy()
    in_ca_gdf = fire_gdf.loc[in_ca_idx].copy()

    in_urban = gpd.sjoin(in_ca_gdf, urban_gdf[["geometry"]], how="inner", predicate="within")
    urban_ids = set(in_urban.index.tolist())
    urban_fires = in_ca_gdf[in_ca_gdf.index.isin(urban_ids)].copy()
    non_urban_fires = in_ca_gdf[~in_ca_gdf.index.isin(urban_ids)].copy()

    rows, cols = [], []
    for lat, lon in zip(non_urban_fires["LATDD83"], non_urban_fires["LONGDD83"]):
        r, c = latlon_to_rowcol(lat, lon)
        rows.append(r)
        cols.append(c)
    non_urban_fires = non_urban_fires.copy()
    non_urban_fires["_row"] = rows
    non_urban_fires["_col"] = cols

    in_bounds = (
        (non_urban_fires["_row"] >= 0)
        & (non_urban_fires["_row"] < gh)
        & (non_urban_fires["_col"] >= 0)
        & (non_urban_fires["_col"] < gw)
    )
    oob = non_urban_fires[~in_bounds].copy()
    bounded = non_urban_fires[in_bounds].copy()
    in_mask = bounded.apply(
        lambda r_: mask_saved[int(r_["_row"]), int(r_["_col"])], axis=1
    )
    masked_drop = bounded[~in_mask].copy()
    kept_stage2 = bounded[in_mask].copy()

    missing_dates = get_missing_2021_wfpi_dates()
    kept_stage2 = kept_stage2.copy()
    kept_stage2["_date_str"] = kept_stage2["discovery_dt"].apply(
        lambda x: x.strftime("%Y%m%d") if hasattr(x, "strftime") else str(x)[:10].replace("-", "")
    )
    on_missing = kept_stage2["_date_str"].isin(missing_dates)
    excluded_missing = kept_stage2[on_missing].copy()
    in_dataset = kept_stage2[~on_missing].copy()

    def rc(gdf):
        if gdf.empty:
            return np.array([], dtype=int), np.array([], dtype=int)
        return gdf["_row"].astype(int).values, gdf["_col"].astype(int).values

    # Outside CA / urban: lat/lon only — project to grid for plotting
    def rc_from_latlon(gdf):
        if gdf.empty:
            return np.array([], dtype=int), np.array([], dtype=int)
        rs, cs = [], []
        for lat, lon in zip(gdf["LATDD83"], gdf["LONGDD83"]):
            r, c = latlon_to_rowcol(lat, lon)
            rs.append(r)
            cs.append(c)
        return np.asarray(rs, dtype=int), np.asarray(cs, dtype=int)

    r_out, c_out = rc_from_latlon(outside_ca)
    r_urb, c_urb = rc_from_latlon(urban_fires)
    r_oob, c_oob = rc(oob)
    r_mask, c_mask = rc(masked_drop)
    n_excluded_missing_wfpi = len(excluded_missing)
    r_ok, c_ok = rc(in_dataset)

    wfpi_combined_r = np.concatenate([r_oob, r_mask]) if len(r_oob) or len(r_mask) else np.array([], int)
    wfpi_combined_c = np.concatenate([c_oob, c_mask]) if len(c_oob) or len(c_mask) else np.array([], int)

    n_urb, n_off, n_ds = len(r_urb), len(wfpi_combined_r), len(r_ok)
    URBAN_TEAL = "#0d9488"
    legend_before_wfpi = make_usfs_fire_legend_handles(
        n_urb, n_off, n_ds, include_off_mask=False, urban_color=URBAN_TEAL
    )
    _leg_full = make_usfs_fire_legend_handles(
        n_urb, n_off, n_ds, include_off_mask=True, urban_color=URBAN_TEAL
    )
    _leg_full[1].set_label("_nolegend_")
    _leg_full[2].set_label("_nolegend_")
    legend_full = [
        _leg_full[0],
        (
            _leg_full[1],
            _leg_full[2],
            f"Fire in unburnable area (n={n_off});\nfire in dataset (n={n_ds})",
        ),
    ]

    from matplotlib.lines import Line2D as _L2D

    # fig01: CA boundary only — urban + off-mask + in-dataset all uniform black dots (no outside-CA fires)
    r_all_in_ca = np.concatenate([r_urb, wfpi_combined_r, r_ok])
    c_all_in_ca = np.concatenate([c_urb, wfpi_combined_c, c_ok])
    legend_fig01 = [
        _L2D([0], [0], linestyle="none", marker="o", color="#0d0d0d",
             markersize=7, label=f"Fire inside California (n={len(r_all_in_ca)})"),
    ]
    fig01_fire_layers = [
        {"rows": r_all_in_ca, "cols": c_all_in_ca, "color": "#0d0d0d", "marker": "o", "s": 22, "zorder": 7, "include_in_legend": False},
    ]

    # fig02: urban removed, WFPI mask NOT yet applied → off-mask fires are black dots
    r_non_urban = np.concatenate([wfpi_combined_r, r_ok])
    c_non_urban = np.concatenate([wfpi_combined_c, c_ok])
    h_urb_f2 = _L2D(
        [0],
        [0],
        linestyle="none",
        marker="^",
        color=URBAN_TEAL,
        markersize=9,
        label="_nolegend_",
    )
    h_nu_f2 = _L2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        color="#0d0d0d",
        markersize=7,
        label="_nolegend_",
    )
    legend_fig02 = [
        (
            h_urb_f2,
            h_nu_f2,
            f"Urban fire (n={n_urb});\nnon-urban fire (n={len(r_non_urban)})",
        ),
    ]
    fig02_fire_layers = [
        {
            "rows": r_urb,
            "cols": c_urb,
            "color": URBAN_TEAL,
            "marker": "^",
            "s": 28,
            "alpha": 1.0,
            "edgecolors": URBAN_TEAL,
            "linewidths": 0.6,
            "zorder": 4,
            "include_in_legend": False,
        },
        {"rows": r_non_urban, "cols": c_non_urban, "color": "#0d0d0d", "marker": "o", "s": 22, "zorder": 7, "include_in_legend": False},
    ]

    # fig03+: urban as teal triangle, unburnable as grey disk, in-dataset as black disk
    all_fire_layers = [
        {
            "rows": r_urb,
            "cols": c_urb,
            "color": URBAN_TEAL,
            "marker": "^",
            "s": 28,
            "alpha": 1.0,
            "edgecolors": URBAN_TEAL,
            "linewidths": 0.6,
            "zorder": 4,
            "include_in_legend": False,
        },
        {
            "rows": wfpi_combined_r,
            "cols": wfpi_combined_c,
            "color": "#888888",
            "marker": "o",
            "s": 14,
            "alpha": 0.75,
            "linewidths": 0,
            "edgecolors": "none",
            "zorder": 5,
            "include_in_legend": False,
        },
        {
            "rows": r_ok,
            "cols": c_ok,
            "color": "#0d0d0d",
            "marker": "o",
            "s": 22,
            "zorder": 7,
            "include_in_legend": False,
        },
    ]

    print("[4/4] Mask progression + figures …")
    _border_kw = dict(ca_boundary_gdf_wfpi=ca_wfpi, cropped_affine=cropped_t)
    # Manuscript figure (Fig.~2) legend: previous 11 pt was doubled to 22; +1.3x total scale.
    _fig2_legend_fontsize = 11 * 2 * 1.3
    plot_pyrologix_valid_region(
        pyrologix,
        mask_step1,
        OUT_DIR / "fig01_pyrologix_california_boundary.png",
        fire_layers=fig01_fire_layers,
        legend_handles=legend_fig01,
        legend_fontsize=_fig2_legend_fontsize,
        show_colorbar=False,
        **_border_kw,
    )
    plot_pyrologix_valid_region(
        pyrologix,
        mask_step2,
        OUT_DIR / "fig02_pyrologix_exclude_urban.png",
        fire_layers=fig02_fire_layers,
        legend_handles=legend_fig02,
        legend_fontsize=_fig2_legend_fontsize,
        show_colorbar=True,
        **_border_kw,
    )
    plot_pyrologix_valid_region(
        pyrologix,
        mask_step3,
        OUT_DIR / "fig03_pyrologix_exclude_always_unburnable.png",
        fire_layers=all_fire_layers,
        legend_handles=legend_full,
        legend_fontsize=_fig2_legend_fontsize,
        show_colorbar=False,
        **_border_kw,
    )
    plot_pyrologix_valid_region(
        pyrologix,
        mask_step4,
        OUT_DIR / "fig04_pyrologix_components_ge_9km2.png",
        fire_layers=all_fire_layers,
        legend_handles=legend_full,
        legend_fontsize=_fig2_legend_fontsize,
        show_colorbar=False,
        **_border_kw,
    )

    plot_pyrologix_fire_categories(
        pyrologix,
        mask_step4.astype(bool),
        all_fire_layers,
        OUT_DIR / "fig05_fires_all_categories.png",
        legend_handles=legend_full,
    )

    plot_pyrologix_fire_categories(
        pyrologix,
        mask_step4.astype(bool),
        [
            {
                "rows": r_ok,
                "cols": c_ok,
                "color": "#0d0d0d",
                "marker": "o",
                "s": 26,
                "zorder": 7,
                "include_in_legend": False,
            },
        ],
        OUT_DIR / "fig06_fires_dataset_only.png",
        legend_handles=legend_full,
    )

    config = json.loads(CONFIG_PATH.read_text())
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    n_scenario_files = len(all_scenario_files)
    valid_scenarios = [
        sf
        for sf in all_scenario_files
        if all(
            f"{k}_{sf.stem.replace('_scenario1', '')}" in config
            for k in ("offset", "date", "time")
        )
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(
        rng.choice(len(valid_scenarios), size=BENCHMARK_SUBSET_SIZE, replace=False)
    )
    bench_files = [valid_scenarios[i] for i in subset_idx]

    br, bc = [], []
    for sf in bench_files:
        pt = np.load(sf)
        br.append(int(pt[0]))
        bc.append(int(pt[1]))
    br = np.asarray(br, dtype=int)
    bc = np.asarray(bc, dtype=int)

    _lb = make_usfs_fire_legend_handles(
        n_urb,
        n_off,
        n_ds,
        include_off_mask=True,
        urban_color=URBAN_TEAL,
        benchmark_label=f"Fire in benchmark subset (n={len(br)}, seed={RANDOM_SEED})",
    )
    _lb[1].set_label("_nolegend_")
    _lb[2].set_label("_nolegend_")
    legend_benchmark = [
        _lb[0],
        (
            _lb[1],
            _lb[2],
            f"Fire in unburnable area (n={n_off}); fire in dataset (n={n_ds})",
        ),
        _lb[3],
    ]
    plot_pyrologix_fire_categories(
        pyrologix,
        mask_step4.astype(bool),
        [
            {
                "rows": br,
                "cols": bc,
                "color": "#0d0d0d",
                "marker": "o",
                "s": 38,
                "zorder": 7,
                "include_in_legend": False,
            },
        ],
        OUT_DIR / "fig07_fires_benchmark_100_seed42.png",
        legend_handles=legend_benchmark,
    )

    n_valid_cells = int(mask_step4.sum())
    stats_lines = [
        "# Auto-generated counts (see paper_2021_dataset_creation_explainer.md in this folder)",
        "",
        "## Mask steps (cell counts)",
        "",
        "| Step | Valid cells |",
        "|------|-------------|",
        f"| (1) California boundary | {int(mask_step1.sum()):,} |",
        f"| (2) Excluding urban | {int(mask_step2.sum()):,} |",
        f"| (3) Excluding always WFPI-invalid | {int(mask_step3.sum()):,} |",
        f"| (4) Components ≥ 9×9 km² + 1 px dilation | {n_valid_cells:,} |",
        "",
        "## 2021 USFS wildfires (ignition points)",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Raw CA wildfires (CSV filters) | {len(df_y):,} |",
        f"| Outside CA boundary | {len(outside_ca):,} |",
        f"| Urban | {len(urban_fires):,} |",
        f"| Non-urban, off grid or mask | {len(wfpi_combined_r):,} |",
        f"| Excluded — missing WFPI zip date | {n_excluded_missing_wfpi:,} |",
        f"| **In dataset (filter pipeline)** | **{len(r_ok):,}** |",
        f"| Scenario files `*.npy` on disk | {n_scenario_files:,} |",
        "",
        "## Benchmark pool",
        "",
        f"| Scenarios with date/time in config | {len(valid_scenarios):,} |",
        f"| Random benchmark subset | {BENCHMARK_SUBSET_SIZE} (seed {RANDOM_SEED}) |",
        "",
    ]
    (OUT_DIR / "counts.md").write_text("\n".join(stats_lines))

    print(f"Figures and counts → {OUT_DIR}")


if __name__ == "__main__":
    main()
