#!/usr/bin/env python3
"""
Visualize Sensor Placement on California 2021 Pyrologix Map

Generates up to four PNG files depending on --scale:

  Data scale (1 km/cell, 1309×805):
    report/california_2021_sensor_placement_{tag}.png   — Pyrologix + sensors + stations
    report/california_2021_sensor_clusters_{tag}.png    — same + cluster union zones + fires

  Operational scale (5 km/cell, 261×161, pooled):
    report/california_2021_sensor_placement_opt_{tag}.png
    report/california_2021_sensor_clusters_opt_{tag}.png

Cluster zones are drawn as the UNION of each station's L∞ reachable square with
pixel-aligned borders (via LineCollection). All such areas use one green fill/border;
the legend uses a green rectangle labeled "Area reachable by drones".
Ground-sensed fires are drawn as discoverable (black crosses), not separately.

Pyrologix is shown on a **0–1** scale (raw 0–255 divided by 255); optional **colorbar** uses the
same inset as Figure 2 (``displays._colorbar_inset_top_right``). All text uses **serif** fonts,
preferring **Latin Modern Roman** /
**Computer Modern** (``mathtext.fontset = cm``); if Latin Modern is installed (e.g. MacTeX/TeX Live
``lmroman10-regular.otf``), it is auto-registered. California outline comes from Census tracts
(``tl_2024_06_tract.shp`` under ``code/dataset_creation/nature_dataset_creation/data/``),
dissolved and reprojected to the WFPI crop; requires **geopandas**, **rasterio**, and a
sample WFPI zip in ``.../2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA/``.

Usage:
    python visualize_sensor_placement_2021.py [sensor_cache_json] [--scale data|opt|both] [--tag _suffix] [--colorbar]
    python visualize_sensor_placement_2021.py path/to/sensor_alloc_*.json --scale opt --tag _foo \\
        --dataset-dir paper/final_report/placement_data --report-dir paper/final_report/images
    # Add ``--colorbar`` for a **right-side** colorbar (opt scale: height of the map axes, Fig.~5 style).
    #    Nature Figure~4: use on 50M panel only if matching prior layout.
    # For CA outline: use conda env ``wf`` (``conda run -n wf python …``) — see ``paper/final_report/docs/reproduce_placement_plots.sh``.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm

# Latin Modern / Computer Modern (LaTeX-style serif) for legend, colorbar, and annotations.
_LM_OTF_CANDIDATES = (
    "/Library/TeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    str(Path.home() / "texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
)


def _register_latin_modern_if_available() -> None:
    for path in _LM_OTF_CANDIDATES:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except (OSError, ValueError, RuntimeError):
            continue
        return


_register_latin_modern_if_available()
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Latin Modern Roman",
            "Latin Modern",
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.collections as mc
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import placement_map_style as pms

DATASET_DIR  = PROJECT_ROOT / "California2021Dataset"
LOG_DIR      = DATASET_DIR / "logs"
SCENARII_DIR = DATASET_DIR / "scenarii"
REPORT_DIR   = PROJECT_ROOT / "report"

COVERAGE_W           = 5
MAX_BATTERY_SUBSTEPS = 7
DRONE_REACH          = MAX_BATTERY_SUBSTEPS // 2   # = 3 opt-cells one-way

# Single fill + border colour for all drone-reachable unions (Figure 4 style).
DRONE_REACH_AREA_COLOR = "#2e7d32"
DRONE_REACH_AREA_LEGEND = "Area reachable by drones"

# California outline (Census tracts dissolved) — same source as `create_california_2021_dataset.py`.
NATURE_DATA_DIR = PROJECT_ROOT / "code/dataset_creation/nature_dataset_creation/data"
CA_TRACTS_SHP = NATURE_DATA_DIR / "tl_2024_06_tract/tl_2024_06_tract.shp"
WFPI_ZIP_DIR = NATURE_DATA_DIR / "2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA"

_PYROLOGIX_GEOREF_CACHE: tuple | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def pool_mean_2d(arr, block):
    H, W = arr.shape
    rH, rW = H // block, W // block
    return arr[:rH*block, :rW*block].reshape(rH, block, rW, block).mean(axis=(1, 3))


def pool_max_2d(arr, block):
    H, W = arr.shape
    rH, rW = H // block, W // block
    return arr[:rH*block, :rW*block].reshape(rH, block, rW, block).max(axis=(1, 3))


def get_pyrologix_georef():
    """Return ``(cropped_affine, wfpi_crs, grid_h, grid_w)`` for the 1309×805 Pyrologix/WFPI crop, or ``None``."""
    global _PYROLOGIX_GEOREF_CACHE
    if _PYROLOGIX_GEOREF_CACHE is not None:
        return _PYROLOGIX_GEOREF_CACHE
    try:
        import tempfile
        import zipfile

        import geopandas as gpd
        import rasterio
    except ImportError:
        return None

    if not CA_TRACTS_SHP.is_file():
        return None
    # The georef (grid transform + CRS) is identical across WFPI forecast products, so
    # any wfpi-forecast zip works. Prefer the canonical dir, then fall back to any
    # ``*_Wind-enhanced_Fire_Potential_Index_Forecast_*_DATA`` dir present locally.
    zips = sorted(WFPI_ZIP_DIR.glob("wfpi-forecast-2_data_*.zip"))
    if not zips:
        zips = sorted(WFPI_ZIP_DIR.glob("wfpi-forecast-*_data_*.zip"))
    if not zips:
        for d in sorted(NATURE_DATA_DIR.glob("*Wind-enhanced_Fire_Potential_Index_Forecast_*_DATA")):
            zips = sorted(d.glob("wfpi-forecast-*_data_*.zip"))
            if zips:
                break
    if not zips:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(zips[0])) as zf:
            zf.extractall(tmp)
        tif_path = next(
            f for f in Path(tmp).rglob("*")
            if f.suffix in (".tif", ".tiff") and not f.name.endswith(".xml")
        )
        with rasterio.open(str(tif_path)) as src:
            raw_t = src.transform
            wfpi_crs = src.crs
            raw_h, raw_w = src.height, src.width

    # Use the same Affine class as rasterio's transform (avoids requiring the standalone ``affine`` package).
    _Affine = type(raw_t)

    ca_tracts = gpd.read_file(CA_TRACTS_SHP).to_crs("EPSG:4326")
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
    grid_h = row_max - row_min
    grid_w = col_max - col_min
    cropped_t = _Affine(
        raw_t.a,
        raw_t.b,
        raw_t.c + col_min * raw_t.a,
        raw_t.d,
        raw_t.e,
        raw_t.f + row_min * raw_t.e,
    )
    _PYROLOGIX_GEOREF_CACHE = (cropped_t, wfpi_crs, grid_h, grid_w)
    return _PYROLOGIX_GEOREF_CACHE


def california_boundary_pixel_paths(
    cropped_t, wfpi_crs, grid_h: int, grid_w: int
) -> list[np.ndarray]:
    """Project dissolved CA boundary to **fractional** data raster (col, row) for ``imshow``.

    Uses ``~cropped_t * (x, y)`` so vertices are not snapped to integer pixels (which misaligns
    the outline relative to the Pyrologix / mask grid). The exterior is densified in projected
    metres so the polyline follows the coast when zoomed.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    ca_tracts = gpd.read_file(CA_TRACTS_SHP).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    geom = ca_tracts.dissolve().geometry.iloc[0]
    gser = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(wfpi_crs)
    poly = gser.iloc[0]
    polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    inv = ~cropped_t
    paths: list[np.ndarray] = []
    for g in polys:
        xs, ys = g.exterior.xy
        line = LineString(np.column_stack([np.asarray(xs), np.asarray(ys)]))
        # Max segment length in projected CRS units (~metres for CONUS Albers).
        line = line.segmentize(2500.0)
        xs2, ys2 = line.xy
        cols, rows = [], []
        for x, y in zip(xs2, ys2):
            col_f, row_f = inv * (float(x), float(y))
            cols.append(col_f)
            rows.append(row_f)
        cols = np.clip(np.asarray(cols, dtype=float), 0.0, float(grid_w - 1))
        rows = np.clip(np.asarray(rows, dtype=float), 0.0, float(grid_h - 1))
        paths.append(np.column_stack([cols, rows]))
    return paths


def data_pixel_paths_to_opt_plot(paths: list[np.ndarray], r_h: int, r_w: int) -> list[np.ndarray]:
    """Map fractional data (col, row) paths to operational ``imshow`` coordinates (÷ pooling)."""
    out: list[np.ndarray] = []
    for p in paths:
        if len(p) < 2:
            continue
        oc = np.clip(p[:, 0] / COVERAGE_W, 0.0, float(r_w))
        orow = np.clip(p[:, 1] / COVERAGE_W, 0.0, float(r_h))
        out.append(np.column_stack([oc, orow]))
    return out


def compute_clusters(charging_locs_opt, drones_per_station):
    n = len(charging_locs_opt)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = charging_locs_opt[i]
            xj, yj = charging_locs_opt[j]
            if max(abs(xi - xj), abs(yi - yj)) <= MAX_BATTERY_SUBSTEPS:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    for indices in groups.values():
        stations = [charging_locs_opt[i] for i in indices]
        n_drones = sum(drones_per_station[i] for i in indices)
        clusters.append({"stations_opt": stations, "n_drones": n_drones})
    return clusters


def classify_fires(fire_rows, fire_cols, clusters, ground_opt_set, scale):
    """Split fires into detected_by_ground, discoverable, non_disc in data-space."""
    detected_ground, discoverable, non_disc = [], [], []
    for r, c in zip(fire_rows, fire_cols):
        fopt = (r // scale, c // scale)
        if fopt in ground_opt_set:
            detected_ground.append((r, c))
            continue
        in_range = any(
            max(abs(fopt[0] - sx), abs(fopt[1] - sy)) <= DRONE_REACH
            for cl in clusters for sx, sy in cl["stations_opt"]
        )
        if in_range:
            discoverable.append((r, c))
        else:
            non_disc.append((r, c))
    return detected_ground, discoverable, non_disc


def classify_fires_opt(fire_rows_opt, fire_cols_opt, clusters, ground_opt_set):
    """Same but coordinates already in opt-space."""
    detected_ground, discoverable, non_disc = [], [], []
    for r, c in zip(fire_rows_opt, fire_cols_opt):
        fopt = (r, c)
        if fopt in ground_opt_set:
            detected_ground.append((r, c))
            continue
        in_range = any(
            max(abs(r - sx), abs(c - sy)) <= DRONE_REACH
            for cl in clusters for sx, sy in cl["stations_opt"]
        )
        if in_range:
            discoverable.append((r, c))
        else:
            non_disc.append((r, c))
    return detected_ground, discoverable, non_disc


# ── Base axes helper ────────────────────────────────────────────────────────────

def make_base_axes(
    bmap_masked,
    *,
    boundary_paths: list[np.ndarray] | None = None,
    cbar_label: str = "Ignition probability (0–1)",
    show_colorbar: bool = False,
):
    H, W = bmap_masked.shape
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    bmap01 = np.asarray(bmap_masked, dtype=float) / 255.0
    im = ax.imshow(
        bmap01,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        extent=[0, W, H, 0],
        zorder=1,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    if boundary_paths:
        for path in boundary_paths:
            if path is None or len(path) < 2:
                continue
            ax.plot(
                path[:, 0],
                path[:, 1],
                color="#7a7a7a",
                linewidth=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )
    if show_colorbar:
        # Same inset colorbar as Figure 2 (`displays._colorbar_inset_top_right`).
        from displays import _colorbar_inset_top_right

        _colorbar_inset_top_right(fig, ax, im, label=cbar_label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0, width=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax, im


def imshow_pooled_bmap_on_ax(
    ax,
    bmap_opt: np.ndarray,
    *,
    rH: int,
    rW: int,
):
    """
    Pooled 5~km risk map on an existing ``Axes`` (for montages). CA state outline is drawn
    in ``draw_operational_fig4_map_on_ax`` (after zone overlays) so it is not covered by
    the green drone wash (z=3) like an early z=2 stroke.

    Returns the ``imshow`` mappable.
    """
    W, H = rW, rH
    bmap01 = np.asarray(bmap_opt, dtype=float) / 255.0
    im = ax.imshow(
        bmap01,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        extent=[0, W, H, 0],
        zorder=1,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0, width=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_operational_fig4_map_on_ax(
    ax,
    bmap_opt: np.ndarray,
    boundary_paths_opt: list | None,
    rH: int,
    rW: int,
    ground_locs_opt: list,
    charging_locs_opt: list,
    drones_per_station: list,
    clusters: list,
    det_gnd_opt: list,
    disc_opt: list,
    ndisc_opt: list,
    *,
    marker_scale: float = 0.5,
    discoverable_marker: str = "x",
    show_drone_count_labels: bool = True,
) -> tuple[object, bool, int, int, list]:
    """
    One operational-scale map (cluster zones, ignitions, ground cells, stations + drone labels)
    in Fig.4/Fig.5 style; **no** figure-level colorbar/legend. Returns
    (``mappable``, ``any_zone``, ``n_discoverable``, ``n_unreachable``, ``fire_legend_items``).
    """
    im = imshow_pooled_bmap_on_ax(ax, bmap_opt, rH=rH, rW=rW)
    any_zone = add_cluster_unions(
        ax,
        clusters,
        rH,
        rW,
        DRONE_REACH,
        [],
        fill_alpha=pms.DEFAULT_FIG4_DRONE_ZONE_ALPHA,
        append_zone_legend=False,
        zone_legend="line2d",
    )
    fire_leg: list = []
    add_fire_markers(
        ax,
        det_gnd_opt,
        disc_opt,
        ndisc_opt,
        fire_leg,
        discoverable_marker=discoverable_marker,
    )
    pms.add_ground_sensor_cells(ax, ground_locs_opt)
    pms.add_charging_stations(
        ax,
        charging_locs_opt,
        drones_per_station,
        marker_scale=marker_scale,
        show_drone_count_labels=show_drone_count_labels,
    )
    # ``equal`` + ``adjustable='box'`` alone leaves horizontal gutters in subplot cells when
    # the cell aspect ratio differs from the pooled grid (rW×rH). ``set_box_aspect`` fixes
    # the axes box shape in figure space so the map fills the cell (composite montages).
    ax.set_box_aspect(float(rH) / float(rW))
    ax.set_aspect("equal", adjustable="box")
    pms.draw_california_state_outline(ax, boundary_paths_opt)
    n_disc = len(disc_opt) + len(det_gnd_opt)
    n_unreach = len(ndisc_opt)
    return im, any_zone, n_disc, n_unreach, fire_leg


# ── Placement-only overlay (no cluster zones) ──────────────────────────────────

def add_sensors_and_stations(ax, ground_locs, charging_locs,
                              drones_per_station, legend_items, marker_scale=0.5):
    if ground_locs:
        g_rows = [r for r, _ in ground_locs]
        g_cols = [c for _, c in ground_locs]
        ax.scatter(g_cols, g_rows, marker="*", s=int(200 * marker_scale),
                   color="white", edgecolors="black", linewidths=0.8, zorder=5)
        legend_items.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="None",
                color="white",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=max(8, int(11 * marker_scale)),
                label=f"Ground sensor (n={len(ground_locs)})",
            )
        )
    if charging_locs:
        for (r, c), nd in zip(charging_locs, drones_per_station):
            ax.scatter(c, r, marker="D", s=int(120 * marker_scale),
                       color="cyan", edgecolors="black", linewidths=0.8, zorder=5)
            off = max(1, int(4 * marker_scale))
            ax.text(c + off, r - off, str(nd), color="cyan",
                    fontsize=max(6, int(7 * marker_scale)), fontweight="bold", zorder=6)
        legend_items.append(
            Line2D(
                [],
                [],
                marker="D",
                linestyle="None",
                color="cyan",
                markerfacecolor="cyan",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=max(6, int(8 * marker_scale)),
                label=f"Charging station (n={len(charging_locs)}; # = drones/station)",
            )
        )


# ── Cluster union zones + fires ─

def add_cluster_unions(
    ax,
    clusters,
    H,
    W,
    zone_half,
    legend_items,
    *,
    fill_alpha: float = 0.25,
    append_zone_legend: bool = True,
    zone_legend: str = "patch",
):
    """
    Draw each cluster's zone as the UNION of that cluster's stations' reachable squares.
    All clusters use one green fill + border; legend: single green rectangle, "Area reachable by drones".
    zone_half : reachable half-size in the same units as H, W
                (data-cells for data scale; opt-cells for opt scale).
    fill_alpha : rgba alpha for the semi-transparent green wash (default 0.25).
    ``zone_legend`` "patch" (default) or "line2d" (Nature Fig.~4/5 / ``placement_map_style``).
    """
    colour = DRONE_REACH_AREA_COLOR
    rv, gv, bv = mcolors.to_rgb(colour)
    fill_overlay = np.zeros((H, W, 4), dtype=float)
    any_zone = False
    fa = float(np.clip(fill_alpha, 0.05, 0.85))

    for cl in clusters:
        cl_mask = np.zeros((H, W), dtype=bool)
        for r_s, c_s in cl["stations_opt"]:
            r0 = max(0, r_s - zone_half);  r1 = min(H, r_s + zone_half + 1)
            c0 = max(0, c_s - zone_half);  c1 = min(W, c_s + zone_half + 1)
            cl_mask[r0:r1, c0:c1] = True

        if cl_mask.any():
            any_zone = True

        fill_overlay[cl_mask, 0] = rv
        fill_overlay[cl_mask, 1] = gv
        fill_overlay[cl_mask, 2] = bv
        fill_overlay[cl_mask, 3] = fa

        m = cl_mask.astype(np.int8)
        # Horizontal edges
        m_v = np.zeros((H + 2, W), dtype=np.int8); m_v[1:H+1, :] = m
        diff_v = m_v[:-1, :] ^ m_v[1:, :]
        ry, cx = np.where(diff_v)
        if len(ry):
            segs = np.stack([np.column_stack([cx.astype(float), ry.astype(float)]),
                             np.column_stack([cx.astype(float)+1, ry.astype(float)])], axis=1)
            ax.add_collection(
                mc.LineCollection(
                    segs,
                    colors=colour,
                    linewidths=1.5,
                    linestyle="solid",
                    capstyle="projecting",
                    joinstyle="miter",
                    zorder=4,
                )
            )
        # Vertical edges
        m_h = np.zeros((H, W + 2), dtype=np.int8); m_h[:, 1:W+1] = m
        diff_h = m_h[:, :-1] ^ m_h[:, 1:]
        ry2, cx2 = np.where(diff_h)
        if len(ry2):
            segs = np.stack([np.column_stack([cx2.astype(float), ry2.astype(float)]),
                             np.column_stack([cx2.astype(float), ry2.astype(float)+1])], axis=1)
            ax.add_collection(
                mc.LineCollection(
                    segs,
                    colors=colour,
                    linewidths=1.5,
                    linestyle="solid",
                    capstyle="projecting",
                    joinstyle="miter",
                    zorder=4,
                )
            )

    if any_zone and append_zone_legend:
        if zone_legend == "line2d":
            r_, g_, b_ = mcolors.to_rgb(colour)
            fa = float(np.clip(fill_alpha, 0.05, 0.85))
            legend_items.append(
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="None",
                    markerfacecolor=(r_, g_, b_, fa),
                    markeredgecolor=(r_, g_, b_, 1.0),
                    markeredgewidth=1.0,
                    markersize=11.0,
                    label=DRONE_REACH_AREA_LEGEND,
                )
            )
        else:
            fa = float(np.clip(fill_alpha, 0.05, 0.85))
            legend_items.append(
                mpatches.Patch(
                    facecolor=colour,
                    edgecolor=colour,
                    linewidth=1.0,
                    alpha=fa,
                    label=DRONE_REACH_AREA_LEGEND,
                )
            )

    ax.imshow(
        fill_overlay,
        origin="upper",
        extent=[0, W, H, 0],
        zorder=3,
        interpolation="nearest",
    )
    return any_zone


def add_fire_markers(
    ax,
    detected_ground,
    discoverable,
    non_disc,
    legend_items,
    *,
    discoverable_marker: str = "x",
):
    """Plot fires; ground-sensed ignitions use the same marker as discoverable (no separate layer)."""
    all_discoverable = list(discoverable) + list(detected_ground)
    n_discoverable = len(all_discoverable)
    n_nd = len(non_disc)

    h_nd = h_dx = None
    if non_disc:
        r, c = zip(*non_disc)
        ax.scatter(c, r, marker=".", s=12, color="gray", alpha=0.6, zorder=5)
        h_nd = Line2D(
            [],
            [],
            marker=".",
            linestyle="None",
            color="gray",
            markerfacecolor="gray",
            markersize=5,
            alpha=0.75,
            label="_nolegend_",
        )
    if n_discoverable:
        r, c = zip(*all_discoverable)
        if discoverable_marker == ".":
            ax.scatter(c, r, marker=".", s=20, color="black", alpha=0.9, zorder=6)
            h_dx = Line2D(
                [],
                [],
                marker=".",
                linestyle="None",
                color="black",
                markerfacecolor="black",
                markersize=5,
                alpha=0.9,
                label="_nolegend_",
            )
        else:
            ax.scatter(
                c,
                r,
                marker="x",
                s=30,
                color="black",
                linewidths=1.2,
                alpha=0.9,
                zorder=6,
            )
            h_dx = Line2D(
                [],
                [],
                marker="x",
                linestyle="None",
                color="black",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=7,
                label="_nolegend_",
            )

    if h_nd is not None and h_dx is not None:
        legend_items.append(
            (
                h_nd,
                h_dx,
                f"Ignitions: not in range (n={n_nd}); discoverable (n={n_discoverable})",
            )
        )
    elif h_nd is not None:
        h_nd.set_label(f"Not in drone range (n={n_nd})")
        legend_items.append(h_nd)
    elif h_dx is not None:
        h_dx.set_label(f"Discoverable fires (n={n_discoverable})")
        legend_items.append(h_dx)


def render_and_save(fig, ax, H, W, out_path, legend_items):
    """Data-scale path: below-map legend (inset colorbar is added in ``make_base_axes`` if requested)."""
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    from displays import _pyrologix_legend_below_map

    _pyrologix_legend_below_map(
        fig, ax, legend_items, legend_fontsize=11, framed=False
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    try:
        rel = out_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path.resolve()
    print(f"  Saved → {rel}", flush=True)


def render_and_save_fig4_opt(
    fig,
    ax,
    im,
    H,
    W,
    out_path: Path,
    legend_items: list,
    *,
    show_colorbar: bool,
) -> None:
    """Operational (Fig.~4) path: side colorbar = map height, ``fig.legend`` (Fig.~5–aligned)."""
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    # One data unit x == one data unit y (imshow cell grid); matches Fig.5 maps.
    ax.set_aspect("equal", adjustable="box")
    pms.style_ca_outline_figure2(ax)

    if show_colorbar:
        pms.add_side_colorbar_single_row(fig, ax, im)

    n_items = len(legend_items)
    r_margin = 0.86 if show_colorbar else 0.98
    bottom = 0.16 if n_items else 0.10
    fig.subplots_adjust(
        left=0.04, right=r_margin, top=0.98, bottom=bottom,
    )

    handles, labels, handler_map = pms.legend_entries_to_handles_labels(legend_items)
    ncol = min(4, max(1, len(handles)))
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        frameon=False,
        fontsize=17.0,
        markerscale=1.0,
        columnspacing=1.35,
        handletextpad=0.4,
        handlelength=1.55,
        handleheight=0.92,
        alignment="center",
        borderaxespad=0.0,
        labelspacing=0.35,
        handler_map=handler_map,
    )
    pms.save_white_tight(out_path, fig, dpi=320, pad=0.12)
    try:
        rel = out_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path.resolve()
    print(f"  Saved (Fig.4 opt) → {rel}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", nargs="?", help="Path to sensor_alloc_*.json")
    parser.add_argument("--scale", choices=["data", "opt", "both"], default="both")
    parser.add_argument("--tag", default="", help="Extra suffix for output filenames")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory with static_risk_pyrologix.npy, mask.npy, config_california_2021.json, scenarii/ "
        "(default: California2021Dataset next to this script).",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for output PNGs (default: report/ next to this script).",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Draw a right-side 0–1 ignition-probability colorbar (opt: height matches map; e.g. 50M / Fig.4b).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else DATASET_DIR
    report_dir = Path(args.report_dir).resolve() if args.report_dir else REPORT_DIR
    log_dir = dataset_dir / "logs"
    scenarii_dir = dataset_dir / "scenarii"

    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sensor cache ───────────────────────────────────────────────────────
    if args.cache:
        cache_path = Path(args.cache)
    else:
        candidates = sorted(log_dir.glob("sensor_alloc_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No sensor_alloc_*.json in {log_dir}")
        cache_path = candidates[-1]

    print(f"Loading sensor cache: {cache_path.name}", flush=True)
    with open(cache_path) as f:
        d = json.load(f)

    ground_locs_opt    = [tuple(x) for x in d["ground_sensor_locations"]]
    charging_locs_opt  = [tuple(x) for x in d["charging_station_locations"]]
    drones_per_station = d["drones_per_charging_station"]
    clusters           = compute_clusters(charging_locs_opt, drones_per_station)
    ground_opt_set     = set(ground_locs_opt)

    n_gs = len(ground_locs_opt)
    n_cs = len(charging_locs_opt)
    n_dr = sum(drones_per_station)
    n_cl = len(clusters)
    print(f"  {n_gs} sensors, {n_cs} stations, {n_dr} drones, {n_cl} clusters", flush=True)

    # Data-space centres
    ground_locs_data   = [(r * COVERAGE_W + COVERAGE_W // 2,
                           c * COVERAGE_W + COVERAGE_W // 2)
                          for r, c in ground_locs_opt]
    charging_locs_data = [(r * COVERAGE_W + COVERAGE_W // 2,
                           c * COVERAGE_W + COVERAGE_W // 2)
                          for r, c in charging_locs_opt]

    # ── Load Pyrologix map and mask ─────────────────────────────────────────────
    print("Loading Pyrologix map and mask ...", flush=True)
    pyro_map = np.load(str(dataset_dir / "static_risk_pyrologix.npy"))
    mask     = np.load(str(dataset_dir / "mask.npy"))
    H, W     = mask.shape

    pyro_2d          = pyro_map[0].astype(float)
    bmap_data        = pyro_2d.copy(); bmap_data[mask == 0] = np.nan

    rH, rW           = H // COVERAGE_W, W // COVERAGE_W
    boundary_paths_data: list[np.ndarray] = []
    boundary_paths_opt: list[np.ndarray] = []
    try:
        gr = get_pyrologix_georef()
        if gr is not None:
            _ct, _crs, gh, gw = gr
            if (gh, gw) == (H, W):
                boundary_paths_data = california_boundary_pixel_paths(_ct, _crs, gh, gw)
                boundary_paths_opt = data_pixel_paths_to_opt_plot(boundary_paths_data, rH, rW)
            else:
                print(
                    f"[viz] CA outline skipped: WFPI crop {gw}×{gh} != mask {W}×{H}",
                    file=sys.stderr,
                )
    except Exception as exc:
        print(f"[viz] CA outline skipped: {exc}", file=sys.stderr)

    pyro_masked       = pyro_2d * mask.astype(float)
    mask_opt          = pool_max_2d(mask.astype(float), COVERAGE_W)
    bmap_opt_raw      = pool_mean_2d(pyro_masked, COVERAGE_W)
    bmap_opt          = bmap_opt_raw.copy(); bmap_opt[mask_opt == 0] = np.nan

    # ── Load fires (with date+time) ─────────────────────────────────────────────
    print("Loading fires ...", flush=True)
    config_path = dataset_dir / "config_california_2021.json"
    with open(config_path) as f:
        config = json.load(f)
    valid_names = {
        key[len("offset_"):]
        for key in config
        if key.startswith("offset_")
        and f"date_{key[len('offset_'):]}" in config
        and f"time_{key[len('offset_'):]}" in config
    }
    fire_rows_data, fire_cols_data = [], []
    for fp in sorted(scenarii_dir.glob("*.npy")):
        name = fp.stem.replace("_scenario1", "")
        if name not in valid_names:
            continue
        pt = np.load(str(fp))
        fire_rows_data.append(int(pt[0])); fire_cols_data.append(int(pt[1]))
    fire_rows_data = np.array(fire_rows_data)
    fire_cols_data = np.array(fire_cols_data)
    fire_rows_opt = fire_rows_data // COVERAGE_W
    fire_cols_opt = fire_cols_data // COVERAGE_W
    n_fires = len(fire_rows_data)
    print(f"  {n_fires} fires loaded", flush=True)

    # Classify fires at data scale
    det_gnd, discoverable, non_disc = classify_fires(
        fire_rows_data, fire_cols_data, clusters, ground_opt_set, COVERAGE_W)
    # Classify fires at opt scale
    # For opt-scale clusters, stations_opt coords are already in opt-space
    det_gnd_opt, disc_opt, ndisc_opt = classify_fires_opt(
        fire_rows_opt, fire_cols_opt, clusters, ground_opt_set)

    # zone_half for data scale: DRONE_REACH opt-cells × COVERAGE_W data-cells/opt-cell
    zone_half_data = DRONE_REACH * COVERAGE_W   # = 3 * 5 = 15 data-cells
    zone_half_opt  = DRONE_REACH                # = 3 opt-cells

    do_data = args.scale in ("data", "both")
    do_opt  = args.scale in ("opt",  "both")
    tag     = args.tag

    # ══════════════════════════════════════════════════════════════════════════
    # DATA SCALE
    # ══════════════════════════════════════════════════════════════════════════
    if do_data:
        print("Rendering [data] — cluster unions + fires ...", flush=True)
        # For the clusters+fires plot use stations in DATA space as cluster centres
        clusters_data = [
            {"stations_opt": [(r * COVERAGE_W + COVERAGE_W // 2,
                                c * COVERAGE_W + COVERAGE_W // 2)
                               for r, c in cl["stations_opt"]],
             "n_drones": cl["n_drones"]}
            for cl in clusters
        ]
        fig, ax, im = make_base_axes(
            bmap_data,
            boundary_paths=boundary_paths_data or None,
            show_colorbar=args.colorbar,
        )
        _ = im  # colorbar (if any) attached in make_base_axes
        leg: list = []
        add_cluster_unions(ax, clusters_data, H, W, zone_half_data, leg)
        add_fire_markers(ax, det_gnd, discoverable, non_disc, leg)
        add_sensors_and_stations(ax, ground_locs_data, charging_locs_data,
                                 drones_per_station, leg, marker_scale=0.25)
        render_and_save(fig, ax, H, W,
                        report_dir / f"california_2021_sensor_clusters{tag}.png", leg)

    # ══════════════════════════════════════════════════════════════════════════
    # OPERATIONAL SCALE
    # ══════════════════════════════════════════════════════════════════════════
    if do_opt:
        print("Rendering [opt] — cluster unions + fires (Fig.4 / Fig.5 style) ...", flush=True)
        aspect = rW / rH
        fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
        fig.patch.set_facecolor("white")
        im, any_zone, _, _, fire_leg = draw_operational_fig4_map_on_ax(
            ax,
            bmap_opt,
            boundary_paths_opt,
            rH,
            rW,
            ground_locs_opt,
            charging_locs_opt,
            drones_per_station,
            clusters,
            det_gnd_opt,
            disc_opt,
            ndisc_opt,
        )
        leg_fig4: list = []
        if n_gs:
            leg_fig4.append(pms.line2d_ground_legend())
        if n_cs:
            leg_fig4.append(pms.line2d_charging_legend())
        if any_zone:
            leg_fig4.append(
                pms.line2d_drone_zone_legend(pms.DEFAULT_FIG4_DRONE_ZONE_ALPHA)
            )
        leg_fig4.extend(fire_leg)
        render_and_save_fig4_opt(
            fig,
            ax,
            im,
            rH,
            rW,
            report_dir / f"california_2021_sensor_clusters_opt{tag}.png",
            leg_fig4,
            show_colorbar=args.colorbar,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
