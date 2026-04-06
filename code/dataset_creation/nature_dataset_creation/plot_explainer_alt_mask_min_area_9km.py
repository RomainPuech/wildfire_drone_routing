#!/usr/bin/env python3
"""
Single figure: Pyrologix + mask keeping every connected component of the
pre–LCC union-burnable mask whose area is at least (9 km)^2 (81 cells at ~1 km).

Does not regenerate other paper explainer figures.

Run from project root:
  conda run -n wf python code/dataset_creation/nature_dataset_creation/plot_explainer_alt_mask_min_area_9km.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.transform
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import dataset_creation.nature_dataset_creation.generate_paper_2021_dataset_explainer as gen
from displays import make_usfs_fire_legend_handles, plot_pyrologix_valid_region

OUT_NAME = "fig_alt_mask_components_ge_9km2.png"


def prepare_fires(cropped_t, wfpi_crs, gh, gw):
    transformer = Transformer.from_crs("EPSG:4326", wfpi_crs, always_xy=True)

    def latlon_to_rowcol(lat, lon):
        x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(cropped_t, x, y)
        return int(r), int(c)

    df_all = pd.read_csv(gen.CSV_PATH, low_memory=False)
    urban_gdf = gpd.read_file(gen.URBAN_SHP).to_crs("EPSG:4326")
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
    ca_tracts = gpd.read_file(gen.CA_TRACTS).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()
    in_ca_idx = gpd.sjoin(fire_gdf, ca_boundary[["geometry"]], how="inner", predicate="within").index
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

    def rc(df):
        if df.empty:
            return np.array([], dtype=int), np.array([], dtype=int)
        return df["_row"].astype(int).values, df["_col"].astype(int).values

    def rc_from_latlon(gdf):
        if gdf.empty:
            return np.array([], dtype=int), np.array([], dtype=int)
        rs, cs = [], []
        for lat, lon in zip(gdf["LATDD83"], gdf["LONGDD83"]):
            r, c = latlon_to_rowcol(lat, lon)
            rs.append(r)
            cs.append(c)
        return np.asarray(rs, dtype=int), np.asarray(cs, dtype=int)

    r_urb, c_urb = rc_from_latlon(urban_fires)
    r_oob, c_oob = rc(oob)
    return r_urb, c_urb, r_oob, c_oob, bounded


def layers_for_mask(mask_bool, bounded, r_urb, c_urb, r_oob, c_oob, urban_purple="#0d9488"):
    def rc(df):
        if df.empty:
            return np.array([], dtype=int), np.array([], dtype=int)
        return df["_row"].astype(int).values, df["_col"].astype(int).values

    in_m = bounded.apply(
        lambda row: bool(mask_bool[int(row["_row"]), int(row["_col"])]), axis=1
    )
    drop = bounded[~in_m]
    keep = bounded[in_m]
    r_drop, c_drop = rc(drop)
    r_keep, c_keep = rc(keep)

    if len(r_oob) or len(r_drop):
        wr = np.concatenate([r_oob, r_drop])
        wc = np.concatenate([c_oob, c_drop])
    else:
        wr = np.array([], dtype=int)
        wc = np.array([], dtype=int)

    fire_layers = [
        {
            "rows": r_urb,
            "cols": c_urb,
            "color": urban_purple,
            "marker": "^",
            "s": 28,
            "alpha": 1.0,
            "zorder": 4,
            "include_in_legend": False,
        },
        {
            "rows": wr,
            "cols": wc,
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
            "rows": r_keep,
            "cols": c_keep,
            "color": "#0d0d0d",
            "marker": "o",
            "s": 22,
            "zorder": 7,
            "include_in_legend": False,
        },
    ]
    n_urb, n_un, n_in = len(r_urb), len(wr), len(r_keep)
    leg = make_usfs_fire_legend_handles(
        n_urb, n_un, n_in, include_off_mask=True, urban_color=urban_purple
    )
    leg[-1] = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        color="#0d0d0d",
        markersize=7,
        label=f"Inside mask (n={n_in})",
    )
    return fire_layers, leg


def main():
    gen.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cropped_t, wfpi_crs, gh, gw, ca_wfpi = gen.get_cropped_transform_and_dims()
    ever = gen.compute_ever_burnable_d1(gh, gw)
    ca_r, ur = gen.rasterize_ca_and_urban(ca_wfpi, cropped_t, gh, gw, wfpi_crs)
    # Erode urban mask by 1 pixel so fires on urban boundaries are treated as non-urban.
    urban_bool = ur >= 0.5
    urban_eroded = ndimage.binary_erosion(urban_bool, iterations=1)
    m_pre = (ca_r == 1) & (~urban_eroded) & ever
    mask_alt_raw = gen.mask_keep_components_min_area_km2(m_pre, side_km=9.0)
    # Dilate the valid mask by 1 pixel so fires lying on unburnable boundaries
    # are absorbed into the mask (equivalently, shrink unburnable areas by 1 px).
    mask_alt = ndimage.binary_dilation(mask_alt_raw, iterations=1)

    r_urb, c_urb, r_oob, c_oob, bounded = prepare_fires(cropped_t, wfpi_crs, gh, gw)
    fire_layers, legend_handles = layers_for_mask(mask_alt, bounded, r_urb, c_urb, r_oob, c_oob)

    pyro = np.load(gen.PYROLOGIX_PATH)[0].astype(np.float32)
    out_path = gen.OUT_DIR / OUT_NAME
    plot_pyrologix_valid_region(
        pyro,
        mask_alt,
        fire_layers=fire_layers,
        legend_handles=legend_handles,
        out_path=out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
