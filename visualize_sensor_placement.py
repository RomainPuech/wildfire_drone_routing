#!/usr/bin/env python3
"""
Visualize Sensor Placement on California 2020 Averaged WFPI Map

Generates up to four PNG files depending on --scale:

  Data scale (1 km/cell, 1309×805):
    california_sensor_placement.png       — avg WFPI + sensors + stations
    california_sensor_clusters.png        — same + cluster zones + all fires

  Operational scale (5 km/cell, 261×161, pooled):
    california_sensor_placement_opt.png   — pooled avg WFPI + sensors + stations
    california_sensor_clusters_opt.png    — same + cluster zones + all fires

Usage:
    python visualize_sensor_placement.py [sensor_cache_json] [--scale data|opt|both]

Defaults: most-recent sensor_alloc_*.json, --scale both
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / "California2020Dataset"
LOG_DIR      = DATASET_DIR / "logs"
SCENARII_DIR = DATASET_DIR / "scenarii"

# ── Parameters (must match benchmark script) ───────────────────────────────────
COVERAGE_W           = 5   # opt-space cell width in data cells
MAX_BATTERY_SUBSTEPS = 7   # L∞ reachability radius in opt-space

# ── Colours ────────────────────────────────────────────────────────────────────
CLUSTER_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


# ── Map helpers ────────────────────────────────────────────────────────────────

def pool_mean_2d(arr, block):
    """Block-mean pool a 2-D float array."""
    H, W = arr.shape
    rH, rW = H // block, W // block
    return (arr[:rH * block, :rW * block]
            .reshape(rH, block, rW, block)
            .mean(axis=(1, 3)))


def pool_max_2d(arr, block):
    """Block-max pool a 2-D array (used for mask)."""
    H, W = arr.shape
    rH, rW = H // block, W // block
    return (arr[:rH * block, :rW * block]
            .reshape(rH, block, rW, block)
            .max(axis=(1, 3)))


# ── Cluster helpers ────────────────────────────────────────────────────────────

def compute_clusters(charging_locs_opt, drones_per_station):
    """Union-find on charging stations; L∞ distance ≤ MAX_BATTERY_SUBSTEPS."""
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


DRONE_REACH = MAX_BATTERY_SUBSTEPS // 2   # one-way reach in opt-cells (must reserve battery for return)


# ── Fire loader ────────────────────────────────────────────────────────────────

def load_fire_ignition_points(valid_names=None):
    """Return (rows, cols) arrays of fire ignition points in data-space.

    If valid_names is provided (a set of scenario name strings), only load
    scenarios whose stem (without '_scenario1') is in that set.
    """
    rows, cols = [], []
    for f in sorted(SCENARII_DIR.glob("*.npy")):
        name = f.stem.replace("_scenario1", "")
        if valid_names is not None and name not in valid_names:
            continue
        pt = np.load(str(f))
        rows.append(int(pt[0]))
        cols.append(int(pt[1]))
    return np.array(rows), np.array(cols)


# ── Plot primitives ────────────────────────────────────────────────────────────

def make_base_axes(bmap_masked, title, xlabel, ylabel):
    """Figure + axes with the burn map already rendered."""
    H, W  = bmap_masked.shape
    aspect = W / H
    fig_h  = 12
    fig, ax = plt.subplots(figsize=(fig_h * aspect + 1.8, fig_h))
    im = ax.imshow(
        bmap_masked,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0, vmax=255,
        extent=[0, W, H, 0],
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="WFPI (0–255)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def add_sensors_and_stations(ax, ground_locs, charging_locs,
                              drones_per_station, legend_items,
                              marker_scale=1.0):
    """Overlay ground sensors (stars) and charging stations (diamonds)."""
    if ground_locs:
        g_rows = [r for r, _ in ground_locs]
        g_cols = [c for _, c in ground_locs]
        ax.scatter(g_cols, g_rows, marker="*", s=int(200 * marker_scale),
                   color="white", edgecolors="black", linewidths=0.8, zorder=5)
        legend_items.append(
            mpatches.Patch(facecolor="white", edgecolor="black",
                           label=f"Ground sensor (n={len(ground_locs)})")
        )

    if charging_locs:
        for (r, c), nd in zip(charging_locs, drones_per_station):
            ax.scatter(c, r, marker="D", s=int(120 * marker_scale),
                       color="cyan", edgecolors="black", linewidths=0.8, zorder=5)
            offset = max(1, int(4 * marker_scale))
            ax.text(c + offset, r - offset, str(nd),
                    color="cyan", fontsize=max(6, int(7 * marker_scale)),
                    fontweight="bold", zorder=6)
        legend_items.append(
            mpatches.Patch(facecolor="cyan", edgecolor="black",
                           label=f"Charging station (n={len(charging_locs)}, label=drones)")
        )


def add_cluster_boxes(ax, clusters, H, W, scale, legend_items):
    """Overlay the true reachable zone of each cluster.

    For each charging station draw an L∞ square of radius DRONE_REACH
    (= MAX_BATTERY_SUBSTEPS // 2), which is the furthest a drone can go
    and still return on a single charge.  All stations in the same cluster
    share a colour; their squares together form the full cluster coverage.
    """
    for i, cluster in enumerate(clusters):
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        nd     = cluster["n_drones"]

        for r_opt, c_opt in cluster["stations_opt"]:
            row_lo = max(0, (r_opt - DRONE_REACH) * scale)
            row_hi = min(H, (r_opt + DRONE_REACH + 1) * scale)
            col_lo = max(0, (c_opt - DRONE_REACH) * scale)
            col_hi = min(W, (c_opt + DRONE_REACH + 1) * scale)
            w = col_hi - col_lo
            h = row_hi - row_lo

            ax.add_patch(Rectangle(
                (col_lo, row_lo), w, h,
                linewidth=1.5, edgecolor=colour, facecolor=colour,
                alpha=0.20, zorder=3,
            ))
            ax.add_patch(Rectangle(
                (col_lo, row_lo), w, h,
                linewidth=1.5, edgecolor=colour, facecolor="none", zorder=4,
            ))

        legend_items.append(
            mpatches.Patch(facecolor=colour, alpha=0.4, edgecolor=colour,
                           label=f"Cluster {i} ({nd} drone{'s' if nd != 1 else ''})")
        )


def add_fires(ax, fire_rows, fire_cols, legend_items, s=4, alpha=0.5, color="black"):
    """Overlay fire ignition points."""
    ax.scatter(fire_cols, fire_rows, s=s, color=color,
               alpha=alpha, zorder=2, linewidths=0)
    legend_items.append(
        mpatches.Patch(facecolor=color, alpha=min(1.0, alpha + 0.3),
                       label=f"Fire ignition points (n={len(fire_rows)})")
    )


def render_and_save(fig, ax, H, W, out_path, legend_items, legend_loc="upper right"):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.legend(handles=legend_items, loc=legend_loc, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", nargs="?", help="Path to sensor_alloc_*.json")
    parser.add_argument("--scale", choices=["data", "opt", "both"],
                        default="both",
                        help="Which coordinate scale to render (default: both)")
    args = parser.parse_args()

    # ── Resolve sensor cache ───────────────────────────────────────────────────
    if args.cache:
        cache_path = Path(args.cache)
    else:
        candidates = sorted(LOG_DIR.glob("sensor_alloc_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No sensor_alloc_*.json found in {LOG_DIR}")
        cache_path = candidates[-1]

    print(f"Loading sensor cache: {cache_path.name}", flush=True)
    with open(cache_path) as f:
        d = json.load(f)

    ground_locs_opt    = [tuple(x) for x in d["ground_sensor_locations"]]
    charging_locs_opt  = [tuple(x) for x in d["charging_station_locations"]]
    drones_per_station = d["drones_per_charging_station"]
    clusters           = compute_clusters(charging_locs_opt, drones_per_station)
    combo_name         = cache_path.stem.replace("sensor_alloc_", "")

    n_gs = len(ground_locs_opt)
    n_cs = len(charging_locs_opt)
    n_dr = sum(drones_per_station)
    n_cl = len(clusters)

    # Data-space station/sensor centres
    ground_locs_data   = [(r * COVERAGE_W + COVERAGE_W // 2,
                           c * COVERAGE_W + COVERAGE_W // 2)
                          for r, c in ground_locs_opt]
    charging_locs_data = [(r * COVERAGE_W + COVERAGE_W // 2,
                           c * COVERAGE_W + COVERAGE_W // 2)
                          for r, c in charging_locs_opt]

    # ── Load maps ──────────────────────────────────────────────────────────────
    print("Loading avg WFPI map and mask ...", flush=True)
    avg_map = np.load(str(DATASET_DIR / "static_risk_wfpi_avg.npy"))
    mask    = np.load(str(DATASET_DIR / "mask.npy"))
    H, W    = mask.shape

    bmap_data        = avg_map[0].astype(float).copy()
    bmap_data[mask == 0] = np.nan

    # Pooled versions for opt scale
    rH, rW           = H // COVERAGE_W, W // COVERAGE_W
    bmap_opt_raw      = pool_mean_2d(avg_map[0].astype(float), COVERAGE_W)
    mask_opt          = pool_max_2d(mask.astype(float),          COVERAGE_W)
    bmap_opt          = bmap_opt_raw.copy()
    bmap_opt[mask_opt == 0] = np.nan

    # ── Load config to find benchmarked fires (have date + time) ──────────────
    config_path = DATASET_DIR / "config_california_2020.json"
    print(f"Loading config to filter benchmarked fires ...", flush=True)
    with open(config_path) as f:
        config = json.load(f)
    valid_names = {
        key[len("offset_"):]
        for key in config
        if key.startswith("offset_")
        and f"date_{key[len('offset_'):]}" in config
        and f"time_{key[len('offset_'):]}" in config
    }
    print(f"  {len(valid_names)} scenarios with date+time in config", flush=True)

    # ── Load fires ─────────────────────────────────────────────────────────────
    print("Loading fire ignition points ...", flush=True)
    fire_rows_data, fire_cols_data = load_fire_ignition_points(valid_names)
    fire_rows_opt = fire_rows_data // COVERAGE_W
    fire_cols_opt = fire_cols_data // COVERAGE_W
    n_fires = len(fire_rows_data)
    print(f"  {n_fires} fires loaded", flush=True)

    do_data = args.scale in ("data", "both")
    do_opt  = args.scale in ("opt",  "both")

    # ══════════════════════════════════════════════════════════════════════════
    # DATA SCALE plots
    # ══════════════════════════════════════════════════════════════════════════
    if do_data:
        # Plot 1-D: placement only
        print("Rendering [data] Plot 1 — placement ...", flush=True)
        fig, ax = make_base_axes(
            bmap_data,
            f"Avg yearly WFPI + sensor/station placement  [{combo_name}]\n"
            f"{n_gs} ground sensors · {n_cs} charging stations · {n_dr} drones",
            "Column (1 km / cell)", "Row (1 km / cell)",
        )
        leg = []
        add_sensors_and_stations(ax, ground_locs_data, charging_locs_data,
                                  drones_per_station, leg)
        render_and_save(fig, ax, H, W,
                        PROJECT_ROOT / "california_sensor_placement.png", leg)

        # Plot 2-D: clusters + fires
        print("Rendering [data] Plot 2 — clusters + fires ...", flush=True)
        fig, ax = make_base_axes(
            bmap_data,
            f"Avg yearly WFPI + sensor placement + cluster zones + all fires  [{combo_name}]\n"
            f"{n_cl} clusters · {n_fires} fires",
            "Column (1 km / cell)", "Row (1 km / cell)",
        )
        leg = []
        add_cluster_boxes(ax, clusters, H, W, scale=COVERAGE_W, legend_items=leg)
        add_fires(ax, fire_rows_data, fire_cols_data, leg, s=10, alpha=0.5, color="black")
        add_sensors_and_stations(ax, ground_locs_data, charging_locs_data,
                                  drones_per_station, leg)
        render_and_save(fig, ax, H, W,
                        PROJECT_ROOT / "california_sensor_clusters.png", leg)

    # ══════════════════════════════════════════════════════════════════════════
    # OPERATIONAL SCALE plots
    # ══════════════════════════════════════════════════════════════════════════
    if do_opt:
        cell_label = f"{COVERAGE_W} km / cell"

        # Plot 1-O: placement only
        print("Rendering [opt] Plot 1 — placement ...", flush=True)
        fig, ax = make_base_axes(
            bmap_opt,
            f"Avg yearly WFPI (opt scale) + sensor/station placement  [{combo_name}]\n"
            f"{n_gs} ground sensors · {n_cs} charging stations · {n_dr} drones"
            f"  (grid {rH}×{rW}, {cell_label})",
            f"Column ({cell_label})", f"Row ({cell_label})",
        )
        leg = []
        add_sensors_and_stations(ax, ground_locs_opt, charging_locs_opt,
                                  drones_per_station, leg, marker_scale=2.0)
        render_and_save(fig, ax, rH, rW,
                        PROJECT_ROOT / "california_sensor_placement_opt.png", leg)

        # Plot 2-O: clusters + fires
        print("Rendering [opt] Plot 2 — clusters + fires ...", flush=True)
        fig, ax = make_base_axes(
            bmap_opt,
            f"Avg yearly WFPI (opt scale) + cluster zones + all fires  [{combo_name}]\n"
            f"{n_cl} clusters · {n_fires} fires  (grid {rH}×{rW}, {cell_label})",
            f"Column ({cell_label})", f"Row ({cell_label})",
        )
        leg = []
        add_cluster_boxes(ax, clusters, rH, rW, scale=1, legend_items=leg)
        add_fires(ax, fire_rows_opt, fire_cols_opt, leg, s=8, alpha=0.40)
        add_sensors_and_stations(ax, ground_locs_opt, charging_locs_opt,
                                  drones_per_station, leg, marker_scale=2.0)
        render_and_save(fig, ax, rH, rW,
                        PROJECT_ROOT / "california_sensor_clusters_opt.png", leg)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
