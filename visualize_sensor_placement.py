#!/usr/bin/env python3
"""
Visualize Sensor Placement on California 2020 Averaged WFPI Map

Generates two PNG files in the project root:

  1. california_sensor_placement.png
       Averaged yearly WFPI burn map (with California mask applied),
       overlaid with all ground sensor and charging station positions.

  2. california_sensor_clusters.png
       Same base map plus the reachable zone (L∞ bounding box in opt-space)
       of every drone cluster, and all fire ignition points from the dataset
       (so you can visually assess how many fires fall inside vs outside clusters).

Run from the project root:
    python visualize_sensor_placement.py [sensor_cache_json]

If no cache file is specified, the most recently modified
California2020Dataset/logs/sensor_alloc_*.json is used.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def opt_to_data_center(loc_opt):
    """Opt-space (row, col) → data-space centre pixel."""
    r, c = loc_opt
    return r * COVERAGE_W + COVERAGE_W // 2, c * COVERAGE_W + COVERAGE_W // 2


def compute_clusters(charging_locs_opt, drones_per_station):
    """Union-find on charging stations with L∞ distance ≤ MAX_BATTERY_SUBSTEPS."""
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


def cluster_bbox_data(cluster, H, W):
    """Return (col_min, row_min, width, height) in data-space for a cluster's
    reachable zone (union of L∞ balls of radius MAX_BATTERY_SUBSTEPS around
    each station, converted to data-space pixels)."""
    stations = cluster["stations_opt"]
    r_min = min(r for r, _ in stations) - MAX_BATTERY_SUBSTEPS
    r_max = max(r for r, _ in stations) + MAX_BATTERY_SUBSTEPS + 1
    c_min = min(c for _, c in stations) - MAX_BATTERY_SUBSTEPS
    c_max = max(c for _, c in stations) + MAX_BATTERY_SUBSTEPS + 1

    # Convert opt-space bounds to data-space pixels
    row_lo = max(0, r_min * COVERAGE_W)
    row_hi = min(H, r_max * COVERAGE_W)
    col_lo = max(0, c_min * COVERAGE_W)
    col_hi = min(W, c_max * COVERAGE_W)

    # matplotlib Rectangle: (x=col, y=row), width=Δcol, height=Δrow
    return col_lo, row_lo, col_hi - col_lo, row_hi - row_lo


def load_fire_ignition_points():
    """Return arrays (rows, cols) of every fire ignition point in data-space."""
    rows, cols = [], []
    for f in sorted(SCENARII_DIR.glob("*.npy")):
        pt = np.load(str(f))
        rows.append(int(pt[0]))
        cols.append(int(pt[1]))
    return np.array(rows), np.array(cols)


def make_base_axes(bmap_masked, title):
    """Create a figure/axes with the WFPI burn map already drawn."""
    H, W = bmap_masked.shape
    aspect = W / H
    fig_h = 12
    fig, ax = plt.subplots(figsize=(fig_h * aspect + 1.5, fig_h))
    im = ax.imshow(
        bmap_masked,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0, vmax=255,
        extent=[0, W, H, 0],   # (left, right, bottom, top) in data coords
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # row 0 at top, row H at bottom
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="WFPI (0–255)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Column (1 km / cell)")
    ax.set_ylabel("Row (1 km / cell)")
    return fig, ax


def add_sensors_and_stations(ax, ground_locs_data, charging_locs_data,
                              drones_per_station, legend_items):
    """Scatter ground sensors and charging stations onto ax."""
    # Ground sensors — white star
    if ground_locs_data:
        g_rows = [r for r, _ in ground_locs_data]
        g_cols = [c for _, c in ground_locs_data]
        ax.scatter(g_cols, g_rows, marker="*", s=200, color="white",
                   edgecolors="black", linewidths=0.8, zorder=5, label="_noleg")
        legend_items.append(
            mpatches.Patch(facecolor="white", edgecolor="black",
                           label=f"Ground sensor (n={len(ground_locs_data)})")
        )

    # Charging stations — colour-coded by drone allocation
    if charging_locs_data:
        c_rows = [r for r, _ in charging_locs_data]
        c_cols = [c for _, c in charging_locs_data]
        for i, ((r, c), nd) in enumerate(zip(charging_locs_data, drones_per_station)):
            ax.scatter(c, r, marker="D", s=120, color="cyan",
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.text(c + 4, r - 4, str(nd), color="cyan",
                    fontsize=7, fontweight="bold", zorder=6)
        legend_items.append(
            mpatches.Patch(facecolor="cyan", edgecolor="black",
                           label=f"Charging station (n={len(charging_locs_data)}, label=drones)")
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Resolve sensor cache file ──────────────────────────────────────────────
    if len(sys.argv) > 1:
        cache_path = Path(sys.argv[1])
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

    ground_locs_data   = [opt_to_data_center(loc) for loc in ground_locs_opt]
    charging_locs_data = [opt_to_data_center(loc) for loc in charging_locs_opt]

    clusters = compute_clusters(charging_locs_opt, drones_per_station)

    # ── Load map and mask ──────────────────────────────────────────────────────
    print("Loading avg WFPI map and mask ...", flush=True)
    avg_map = np.load(str(DATASET_DIR / "static_risk_wfpi_avg.npy"))
    mask    = np.load(str(DATASET_DIR / "mask.npy"))
    H, W    = mask.shape

    bmap_masked = avg_map[0].astype(float).copy()
    bmap_masked[mask == 0] = np.nan

    # ── Load fire ignition points ──────────────────────────────────────────────
    print("Loading fire ignition points ...", flush=True)
    fire_rows, fire_cols = load_fire_ignition_points()
    print(f"  {len(fire_rows)} fires loaded", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 1 — avg WFPI + sensors + charging stations
    # ══════════════════════════════════════════════════════════════════════════
    print("Rendering Plot 1 ...", flush=True)
    combo_name = cache_path.stem.replace("sensor_alloc_", "")
    fig1, ax1 = make_base_axes(
        bmap_masked,
        f"Averaged yearly WFPI + sensor/station placement\n({combo_name},"
        f"  {len(ground_locs_opt)} sensors, {len(charging_locs_opt)} charging stations,"
        f" {sum(drones_per_station)} drones)"
    )

    legend_items1 = []
    add_sensors_and_stations(ax1, ground_locs_data, charging_locs_data,
                              drones_per_station, legend_items1)
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
    ax1.legend(handles=legend_items1, loc="upper right", fontsize=9)
    fig1.tight_layout()

    out1 = PROJECT_ROOT / "california_sensor_placement.png"
    fig1.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved → {out1.name}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 2 — same + cluster bounding boxes + all fires
    # ══════════════════════════════════════════════════════════════════════════
    print("Rendering Plot 2 ...", flush=True)
    fig2, ax2 = make_base_axes(
        bmap_masked,
        f"Averaged yearly WFPI + sensor placement + cluster zones + all fires\n"
        f"({combo_name},  {len(clusters)} clusters,  {len(fire_rows)} fires)"
    )

    legend_items2 = []

    # Cluster bounding boxes
    for i, cluster in enumerate(clusters):
        col_lo, row_lo, w, h = cluster_bbox_data(cluster, H, W)
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        rect = Rectangle(
            (col_lo, row_lo), w, h,
            linewidth=1.5, edgecolor=colour, facecolor=colour, alpha=0.15,
            zorder=3
        )
        ax2.add_patch(rect)
        # Also draw a solid border
        rect_border = Rectangle(
            (col_lo, row_lo), w, h,
            linewidth=1.5, edgecolor=colour, facecolor="none",
            zorder=4
        )
        ax2.add_patch(rect_border)
        legend_items2.append(
            mpatches.Patch(facecolor=colour, alpha=0.4, edgecolor=colour,
                           label=f"Cluster {i} ({cluster['n_drones']} drone{'s' if cluster['n_drones'] != 1 else ''})")
        )

    # All fires
    ax2.scatter(fire_cols, fire_rows, s=4, color="white", alpha=0.35,
                zorder=2, linewidths=0)
    legend_items2.append(
        mpatches.Patch(facecolor="white", alpha=0.6,
                       label=f"Fire ignition points (n={len(fire_rows)})")
    )

    add_sensors_and_stations(ax2, ground_locs_data, charging_locs_data,
                              drones_per_station, legend_items2)

    ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
    ax2.legend(handles=legend_items2, loc="upper right", fontsize=8,
               framealpha=0.85)
    fig2.tight_layout()

    out2 = PROJECT_ROOT / "california_sensor_clusters.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {out2.name}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
