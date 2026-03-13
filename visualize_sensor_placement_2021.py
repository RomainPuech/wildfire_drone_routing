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
pixel-aligned borders (via LineCollection) — matching displays.py::plot_benchmark_overview.
Fires are classified as discoverable vs non-discoverable.

Usage:
    python visualize_sensor_placement_2021.py [sensor_cache_json] [--scale data|opt|both] [--tag _suffix]
"""

import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mc
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

DATASET_DIR  = PROJECT_ROOT / "California2021Dataset"
LOG_DIR      = DATASET_DIR / "logs"
SCENARII_DIR = DATASET_DIR / "scenarii"
REPORT_DIR   = PROJECT_ROOT / "report"

COVERAGE_W           = 5
MAX_BATTERY_SUBSTEPS = 7
DRONE_REACH          = MAX_BATTERY_SUBSTEPS // 2   # = 3 opt-cells one-way

CLUSTER_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def pool_mean_2d(arr, block):
    H, W = arr.shape
    rH, rW = H // block, W // block
    return arr[:rH*block, :rW*block].reshape(rH, block, rW, block).mean(axis=(1, 3))


def pool_max_2d(arr, block):
    H, W = arr.shape
    rH, rW = H // block, W // block
    return arr[:rH*block, :rW*block].reshape(rH, block, rW, block).max(axis=(1, 3))


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

def make_base_axes(bmap_masked, title, xlabel, ylabel, vmax=255,
                   cbar_label="Ignition Probability (Pyrologix, 0–255)"):
    H, W = bmap_masked.shape
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    im = ax.imshow(bmap_masked, cmap="YlOrRd", origin="upper",
                   interpolation="nearest", vmin=0, vmax=vmax, extent=[0, W, H, 0])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=cbar_label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return fig, ax


# ── Placement-only overlay (no cluster zones) ──────────────────────────────────

def add_sensors_and_stations(ax, ground_locs, charging_locs,
                              drones_per_station, legend_items, marker_scale=0.5):
    if ground_locs:
        g_rows = [r for r, _ in ground_locs]
        g_cols = [c for _, c in ground_locs]
        ax.scatter(g_cols, g_rows, marker="*", s=int(200 * marker_scale),
                   color="white", edgecolors="black", linewidths=0.8, zorder=5)
        legend_items.append(
            mpatches.Patch(facecolor="white", edgecolor="black",
                           label=f"Ground sensor (n={len(ground_locs)})"))
    if charging_locs:
        for (r, c), nd in zip(charging_locs, drones_per_station):
            ax.scatter(c, r, marker="D", s=int(120 * marker_scale),
                       color="cyan", edgecolors="black", linewidths=0.8, zorder=5)
            off = max(1, int(4 * marker_scale))
            ax.text(c + off, r - off, str(nd), color="cyan",
                    fontsize=max(6, int(7 * marker_scale)), fontweight="bold", zorder=6)
        legend_items.append(
            mpatches.Patch(facecolor="cyan", edgecolor="black",
                           label=f"Charging station (n={len(charging_locs)}, label=drones)"))


# ── Cluster union zones + fires (matches displays.py::plot_benchmark_overview) ─

def add_cluster_unions(ax, clusters, H, W, zone_half, legend_items):
    """
    Draw cluster zones as the UNION of each station's reachable square.
    Per-cluster fill (25% alpha) + pixel-aligned border via LineCollection.
    zone_half : reachable half-size in the same units as H, W
                (data-cells for data scale; opt-cells for opt scale).
    """
    fill_overlay = np.zeros((H, W, 4), dtype=float)

    for i, cl in enumerate(clusters):
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        rv, gv, bv = mcolors.to_rgb(colour)

        cl_mask = np.zeros((H, W), dtype=bool)
        for r_s, c_s in cl["stations_opt"]:
            r0 = max(0, r_s - zone_half);  r1 = min(H, r_s + zone_half + 1)
            c0 = max(0, c_s - zone_half);  c1 = min(W, c_s + zone_half + 1)
            cl_mask[r0:r1, c0:c1] = True

        fill_overlay[cl_mask, 0] = rv
        fill_overlay[cl_mask, 1] = gv
        fill_overlay[cl_mask, 2] = bv
        fill_overlay[cl_mask, 3] = 0.25

        m = cl_mask.astype(np.int8)
        # Horizontal edges
        m_v = np.zeros((H + 2, W), dtype=np.int8); m_v[1:H+1, :] = m
        diff_v = m_v[:-1, :] ^ m_v[1:, :]
        ry, cx = np.where(diff_v)
        if len(ry):
            segs = np.stack([np.column_stack([cx.astype(float), ry.astype(float)]),
                             np.column_stack([cx.astype(float)+1, ry.astype(float)])], axis=1)
            ax.add_collection(mc.LineCollection(segs, colors=colour, linewidths=1.5, zorder=4))
        # Vertical edges
        m_h = np.zeros((H, W + 2), dtype=np.int8); m_h[:, 1:W+1] = m
        diff_h = m_h[:, :-1] ^ m_h[:, 1:]
        ry2, cx2 = np.where(diff_h)
        if len(ry2):
            segs = np.stack([np.column_stack([cx2.astype(float), ry2.astype(float)]),
                             np.column_stack([cx2.astype(float), ry2.astype(float)+1])], axis=1)
            ax.add_collection(mc.LineCollection(segs, colors=colour, linewidths=1.5, zorder=4))

        legend_items.append(
            mpatches.Patch(facecolor=colour, alpha=0.4, edgecolor=colour,
                           label=f"Cluster {i} ({cl['n_drones']} drones)"))

    ax.imshow(fill_overlay, origin="upper", extent=[0, W, H, 0], zorder=3,
              interpolation="nearest")


def add_fire_markers(ax, detected_ground, discoverable, non_disc, legend_items):
    if non_disc:
        r, c = zip(*non_disc)
        ax.scatter(c, r, marker=".", s=12, color="gray", alpha=0.6, zorder=5)
        legend_items.append(mpatches.Patch(facecolor="gray", alpha=0.7,
            label=f"Non-discoverable (n={len(non_disc)})"))
    if discoverable:
        r, c = zip(*discoverable)
        ax.scatter(c, r, marker="x", s=30, color="black", linewidths=1.2, alpha=0.9, zorder=6)
        legend_items.append(mpatches.Patch(facecolor="black", alpha=0.8,
            label=f"Discoverable (n={len(discoverable)})"))
    if detected_ground:
        r, c = zip(*detected_ground)
        ax.scatter(c, r, marker="o", s=60, color="limegreen",
                   edgecolors="darkgreen", linewidths=0.8, zorder=8)
        legend_items.append(mpatches.Patch(facecolor="limegreen", edgecolor="darkgreen",
            label=f"At ground sensor (n={len(detected_ground)})"))


def render_and_save(fig, ax, H, W, out_path, legend_items, legend_loc="upper right"):
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.legend(handles=legend_items, loc=legend_loc, fontsize=7, framealpha=0.85,
              ncol=max(1, len(legend_items) // 20))
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", nargs="?", help="Path to sensor_alloc_*.json")
    parser.add_argument("--scale", choices=["data", "opt", "both"], default="both")
    parser.add_argument("--tag", default="", help="Extra suffix for output filenames")
    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)

    # ── Load sensor cache ───────────────────────────────────────────────────────
    if args.cache:
        cache_path = Path(args.cache)
    else:
        candidates = sorted(LOG_DIR.glob("sensor_alloc_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No sensor_alloc_*.json in {LOG_DIR}")
        cache_path = candidates[-1]

    print(f"Loading sensor cache: {cache_path.name}", flush=True)
    with open(cache_path) as f:
        d = json.load(f)

    ground_locs_opt    = [tuple(x) for x in d["ground_sensor_locations"]]
    charging_locs_opt  = [tuple(x) for x in d["charging_station_locations"]]
    drones_per_station = d["drones_per_charging_station"]
    clusters           = compute_clusters(charging_locs_opt, drones_per_station)
    ground_opt_set     = set(ground_locs_opt)
    combo_name         = cache_path.stem.replace("sensor_alloc_", "")

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
    pyro_map = np.load(str(DATASET_DIR / "static_risk_pyrologix.npy"))
    mask     = np.load(str(DATASET_DIR / "mask.npy"))
    H, W     = mask.shape

    pyro_2d          = pyro_map[0].astype(float)
    bmap_data        = pyro_2d.copy(); bmap_data[mask == 0] = np.nan

    rH, rW           = H // COVERAGE_W, W // COVERAGE_W
    pyro_masked       = pyro_2d * mask.astype(float)
    mask_opt          = pool_max_2d(mask.astype(float), COVERAGE_W)
    bmap_opt_raw      = pool_mean_2d(pyro_masked, COVERAGE_W)
    bmap_opt          = bmap_opt_raw.copy(); bmap_opt[mask_opt == 0] = np.nan

    # ── Load fires (with date+time) ─────────────────────────────────────────────
    print("Loading fires ...", flush=True)
    config_path = DATASET_DIR / "config_california_2021.json"
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
    for fp in sorted(SCENARII_DIR.glob("*.npy")):
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
        fig, ax = make_base_axes(
            bmap_data,
            f"Pyrologix + cluster zones + fires  [{combo_name}]\n"
            f"{n_cl} clusters · {n_fires} fires  "
            f"({len(discoverable)+len(det_gnd)} discoverable, {len(non_disc)} non-disc)",
            "Column (1 km / cell)", "Row (1 km / cell)",
        )
        leg = []
        add_cluster_unions(ax, clusters_data, H, W, zone_half_data, leg)
        add_fire_markers(ax, det_gnd, discoverable, non_disc, leg)
        add_sensors_and_stations(ax, ground_locs_data, charging_locs_data,
                                  drones_per_station, leg, marker_scale=0.25)
        render_and_save(fig, ax, H, W,
                        REPORT_DIR / f"california_2021_sensor_clusters{tag}.png", leg)

    # ══════════════════════════════════════════════════════════════════════════
    # OPERATIONAL SCALE
    # ══════════════════════════════════════════════════════════════════════════
    if do_opt:
        cell_label = f"{COVERAGE_W} km / cell"
        vmax_opt   = float(bmap_opt_raw[mask_opt > 0].max())

        print("Rendering [opt] — cluster unions + fires ...", flush=True)
        fig, ax = make_base_axes(
            bmap_opt,
            f"Pyrologix (opt scale) + cluster zones + fires  [{combo_name}]\n"
            f"{n_cl} clusters · {n_fires} fires  (grid {rH}×{rW}, {cell_label})",
            f"Column ({cell_label})", f"Row ({cell_label})",
            vmax=vmax_opt,
        )
        leg = []
        add_cluster_unions(ax, clusters, rH, rW, zone_half_opt, leg)
        add_fire_markers(ax, det_gnd_opt, disc_opt, ndisc_opt, leg)
        add_sensors_and_stations(ax, ground_locs_opt, charging_locs_opt,
                                  drones_per_station, leg, marker_scale=0.5)
        render_and_save(fig, ax, rH, rW,
                        REPORT_DIR / f"california_2021_sensor_clusters_opt{tag}.png", leg)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
