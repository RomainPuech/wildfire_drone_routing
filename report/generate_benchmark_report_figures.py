#!/usr/bin/env python3
"""
Generate figures for the benchmark report (20M budget-optimized strategy).

Run from project root:
  python report/generate_benchmark_report_figures.py

Before benchmark completes: produces fire locations map and overview map with
cluster zones and fire locations (discoverable vs non-discoverable; detection
outcomes TBD until benchmark CSV exists).

After benchmark completes: pass the results CSV to update the overview map
with detected / missed discoverable / non-discoverable.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))


def _plot_fire_locations(background_map, fire_rows, fire_cols, out_path, title="Fire ignition points"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = background_map.shape
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    im = ax.imshow(background_map, cmap="YlOrRd", origin="upper", interpolation="nearest", vmin=0, vmax=255, extent=[0, W, H, 0])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Avg WFPI (0–255)")
    ax.scatter(fire_cols, fire_rows, marker="o", s=15, color="black", edgecolors="none", alpha=0.8, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Column (~1 km/cell)")
    ax.set_ylabel("Row (~1 km/cell)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_overview(background_map, mask, clusters, sensors_data, stations_data, detected_fires, missed_disc_fires, non_disc_fires, one_way_reach_opt, coverage_w, out_path, title="Benchmark overview"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import matplotlib.colors as mcolors
    _CLUSTER_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
    H, W = background_map.shape
    zone_half = one_way_reach_opt * coverage_w
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    im = ax.imshow(background_map, cmap="YlOrRd", origin="upper", interpolation="nearest", vmin=0, vmax=255, extent=[0, W, H, 0])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Avg WFPI (0–255)")
    fill_overlay = np.zeros((H, W, 4), dtype=float)
    for i, cl in enumerate(clusters):
        colour = _CLUSTER_COLOURS[i % len(_CLUSTER_COLOURS)]
        r_val, g_val, b_val = mcolors.to_rgb(colour)
        cl_mask = np.zeros((H, W), dtype=bool)
        for r_opt, c_opt in cl["stations_opt"]:
            r_d = r_opt * coverage_w + coverage_w // 2
            c_d = c_opt * coverage_w + coverage_w // 2
            r0, r1 = max(0, r_d - zone_half), min(H, r_d + zone_half + 1)
            c0, c1 = max(0, c_d - zone_half), min(W, c_d + zone_half + 1)
            cl_mask[r0:r1, c0:c1] = True
        fill_overlay[cl_mask, 0], fill_overlay[cl_mask, 1] = r_val, g_val
        fill_overlay[cl_mask, 2], fill_overlay[cl_mask, 3] = b_val, 0.25
        m = cl_mask.astype(np.int8)
        m_v = np.zeros((H + 2, W), dtype=np.int8)
        m_v[1:H + 1, :] = m
        diff_v = m_v[:-1, :] ^ m_v[1:, :]
        ry, cx = np.where(diff_v)
        if len(ry):
            x0, y0 = cx.astype(float), ry.astype(float)
            segs = np.stack([np.column_stack([x0, y0]), np.column_stack([x0 + 1, y0])], axis=1)
            ax.add_collection(mc.LineCollection(segs, colors=colour, linewidths=1.5, zorder=4))
        m_h = np.zeros((H, W + 2), dtype=np.int8)
        m_h[:, 1:W + 1] = m
        diff_h = m_h[:, :-1] ^ m_h[:, 1:]
        ry2, cx2 = np.where(diff_h)
        if len(ry2):
            x0, y0 = cx2.astype(float), ry2.astype(float)
            segs = np.stack([np.column_stack([x0, y0]), np.column_stack([x0, y0 + 1])], axis=1)
            ax.add_collection(mc.LineCollection(segs, colors=colour, linewidths=1.5, zorder=4))
    ax.imshow(fill_overlay, origin="upper", extent=[0, W, H, 0], zorder=3, interpolation="nearest")

    def get_rc(obj):
        if hasattr(obj, "iterrows"):
            return obj["row"].tolist(), obj["col"].tolist()
        return [r for r, _ in obj], [c for _, c in obj]
    nd_r, nd_c = get_rc(non_disc_fires)
    md_r, md_c = get_rc(missed_disc_fires)
    dt_r, dt_c = get_rc(detected_fires)
    if nd_r:
        ax.scatter(nd_c, nd_r, marker=".", s=20, color="gray", alpha=0.7, zorder=5, label=f"Non-discoverable (n={len(nd_r)})")
    if md_r:
        ax.scatter(md_c, md_r, marker="x", s=50, color="black", linewidths=1.5, alpha=0.9, zorder=6, label=f"Missed discoverable (n={len(md_r)})")
    if dt_r:
        ax.scatter(dt_c, dt_r, marker="o", s=80, color="limegreen", edgecolors="darkgreen", linewidths=0.8, zorder=8, label=f"Detected (n={len(dt_r)})")
    if sensors_data:
        sr, sc = zip(*sensors_data)
        ax.scatter(sc, sr, marker="*", s=40, color="white", edgecolors="black", linewidths=0.5, zorder=7, label=f"Ground sensor (n={len(sensors_data)})")
    if stations_data:
        tr, tc = zip(*stations_data)
        ax.scatter(tc, tr, marker="^", s=50, color="blue", edgecolors="white", linewidths=0.8, zorder=7, label=f"Charging station (n={len(stations_data)})")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Column (~1 km/cell)")
    ax.set_ylabel("Row (~1 km/cell)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Match run_benchmark_california2020_yearly.py
DATASET_DIR = PROJECT_ROOT / "California2020Dataset"
AVG_MAP = DATASET_DIR / "static_risk_wfpi_avg.npy"
MASK_PATH = DATASET_DIR / "mask.npy"
CONFIG_PATH = DATASET_DIR / "config_california_2020.json"
SCENARII_DIR = DATASET_DIR / "scenarii"
LOG_DIR = DATASET_DIR / "logs"

BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED = 42
SENSOR_POOLING = "mean"
cell_size_m = 1000
speed = 600
coverage_r_m = 2900
COMBO_NAME = "GaussianBudget20M_TOP"


def compute_clusters(charging_locs_opt, drones_per_station, max_battery_substeps):
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
            if max(abs(xi - xj), abs(yi - yj)) <= max_battery_substeps:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    for indices in groups.values():
        stations = [charging_locs_opt[i] for i in indices]
        n_drones = sum(drones_per_station[i] for i in indices)
        fp = "_".join(f"{x}-{y}" for x, y in sorted(stations))
        clusters.append({"stations_opt": stations, "n_drones": n_drones, "fingerprint": fp})
    return clusters


def fire_cluster(fire_opt, clusters, max_battery_substeps):
    one_way_reach = max_battery_substeps // 2
    fr, fc = fire_opt
    for cluster in clusters:
        for sx, sy in cluster["stations_opt"]:
            if max(abs(fr - sx), abs(fc - sy)) <= one_way_reach:
                return cluster
    return None


def main():
    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(exist_ok=True)

    # Rescaling (match run_benchmark_california2020_yearly.py: coverage_w=5, substeps=7)
    coverage_w = 5
    rescaled_max_battery = 7

    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
    rescaled_N = H // coverage_w
    rescaled_M = W // coverage_w

    # Load config and scenario list
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf for sf in all_scenario_files
        if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config for k in ("offset", "date", "time"))
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(rng.choice(len(valid_scenarios), size=BENCHMARK_SUBSET_SIZE, replace=False))
    benchmark_scenarios = [valid_scenarios[i] for i in subset_idx]

    # Fire locations (data-space row, col) from each scenario .npy
    fire_rows, fire_cols = [], []
    for sf in benchmark_scenarios:
        pt = np.load(str(sf))
        fire_rows.append(int(pt[0]))
        fire_cols.append(int(pt[1]))

    # Background map at data scale (avg WFPI, masked)
    avg_map = np.load(str(AVG_MAP))
    if avg_map.ndim == 3:
        avg_map = avg_map[0]
    mask_arr = np.load(str(MASK_PATH))
    background_vis = np.where(mask_arr > 0, avg_map, 0.0)

    # 1) Fire locations only
    out_fire = report_dir / "benchmark_fire_locations_budget.png"
    _plot_fire_locations(
        background_vis,
        fire_rows,
        fire_cols,
        out_fire,
        title="Fire ignition points (100 random, seed=42)",
    )
    print(f"Saved: {out_fire}")

    # Load sensor placement (budget strategy)
    sensor_log = LOG_DIR / f"sensor_alloc_{COMBO_NAME}_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}.json"
    if not sensor_log.exists():
        print(f"Missing {sensor_log}; run benchmark once to create sensor cache.")
        return
    with open(sensor_log) as f:
        sensor_data = json.load(f)
    ground_locs_opt = [tuple(x) for x in sensor_data["ground_sensor_locations"]]
    charging_locs_opt = [tuple(x) for x in sensor_data["charging_station_locations"]]
    drones_per_station = sensor_data["drones_per_charging_station"]

    clusters = compute_clusters(charging_locs_opt, drones_per_station, rescaled_max_battery)
    one_way_reach_opt = rescaled_max_battery // 2  # 3

    # Convert to data-space for display
    sensors_data = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in ground_locs_opt]
    stations_data = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in charging_locs_opt]

    # Classify fires: at ground sensor -> detected; in drone reach -> discoverable; else non-discoverable
    fire_opts = [(r // coverage_w, c // coverage_w) for r, c in zip(fire_rows, fire_cols)]
    ground_opt_set = set(ground_locs_opt)
    detected_by_ground = [(fire_rows[i], fire_cols[i]) for i in range(len(fire_rows)) if fire_opts[i] in ground_opt_set]
    non_disc = []
    discoverable = []
    for i, fopt in enumerate(fire_opts):
        if fopt in ground_opt_set:
            continue  # already in detected_by_ground
        if fire_cluster(fopt, clusters, rescaled_max_battery) is None:
            non_disc.append((fire_rows[i], fire_cols[i]))
        else:
            discoverable.append((fire_rows[i], fire_cols[i]))

    # If results CSV exists, split discoverable into detected vs missed
    detected_fires = list(detected_by_ground)
    missed_disc_fires = list(discoverable)
    csv_candidates = list(PROJECT_ROOT.glob("benchmark_results_yearly_*.csv"))
    csv_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if csv_candidates:
        import pandas as pd
        df = pd.read_csv(csv_candidates[0])
        if "GaussianBudget20M_TOP" in df["strategy_combo"].values:
            df = df[df["strategy_combo"] == "GaussianBudget20M_TOP"].reset_index(drop=True)
            if len(df) == len(benchmark_scenarios):
                detected_fires = list(detected_by_ground)
                missed_disc_fires = []
                for i in range(len(benchmark_scenarios)):
                    r, c = fire_rows[i], fire_cols[i]
                    device = df.iloc[i]["device"]
                    if device in ("drone", "ground sensor", "charging station"):
                        if (r, c) not in set(detected_fires):
                            detected_fires.append((r, c))
                missed_disc_fires = [(r, c) for r, c in discoverable if (r, c) not in set(detected_fires)]

    out_overview = report_dir / "benchmark_fire_map_budget.png"
    _plot_benchmark_overview(
        background_vis,
        mask_arr,
        clusters,
        sensors_data,
        stations_data,
        detected_fires,
        missed_disc_fires,
        non_disc,
        one_way_reach_opt,
        coverage_w,
        out_overview,
        title="Benchmark overview (20M budget-optimized)",
    )
    print(f"Saved: {out_overview}")

    # 100M budget placement figure (if cache exists)
    sensor_log_100M = LOG_DIR / f"sensor_alloc_GaussianBudget100M_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}.json"
    if sensor_log_100M.exists():
        with open(sensor_log_100M) as f:
            sensor_data_100 = json.load(f)
        ground_locs_100 = [tuple(x) for x in sensor_data_100["ground_sensor_locations"]]
        charging_locs_100 = [tuple(x) for x in sensor_data_100["charging_station_locations"]]
        drones_100 = sensor_data_100["drones_per_charging_station"]
        clusters_100 = compute_clusters(charging_locs_100, drones_100, rescaled_max_battery)
        sensors_data_100 = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in ground_locs_100]
        stations_data_100 = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in charging_locs_100]
        ground_opt_set_100 = set(ground_locs_100)
        detected_100 = [(fire_rows[i], fire_cols[i]) for i in range(len(fire_rows)) if fire_opts[i] in ground_opt_set_100]
        non_disc_100 = []
        discoverable_100 = []
        for i, fopt in enumerate(fire_opts):
            if fopt in ground_opt_set_100:
                continue
            if fire_cluster(fopt, clusters_100, rescaled_max_battery) is None:
                non_disc_100.append((fire_rows[i], fire_cols[i]))
            else:
                discoverable_100.append((fire_rows[i], fire_cols[i]))
        out_100M = report_dir / "benchmark_fire_map_budget_100M.png"
        _plot_benchmark_overview(
            background_vis,
            mask_arr,
            clusters_100,
            sensors_data_100,
            stations_data_100,
            detected_100,  # fires at ground sensor cells = detected
            discoverable_100,  # show discoverable as missed
            non_disc_100,
            one_way_reach_opt,
            coverage_w,
            out_100M,
            title="Sensor placement (100M budget-optimized)",
        )
        print(f"Saved: {out_100M}")
    else:
        print(f"100M cache not found ({sensor_log_100M.name}); skip 100M figure.")

    # 500M budget placement figure (if cache exists)
    sensor_log_500M = LOG_DIR / f"sensor_alloc_GaussianBudget500M_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}.json"
    if sensor_log_500M.exists():
        with open(sensor_log_500M) as f:
            sensor_data_500 = json.load(f)
        ground_locs_500 = [tuple(x) for x in sensor_data_500["ground_sensor_locations"]]
        charging_locs_500 = [tuple(x) for x in sensor_data_500["charging_station_locations"]]
        drones_500 = sensor_data_500["drones_per_charging_station"]
        clusters_500 = compute_clusters(charging_locs_500, drones_500, rescaled_max_battery)
        sensors_data_500 = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in ground_locs_500]
        stations_data_500 = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in charging_locs_500]
        ground_opt_set_500 = set(ground_locs_500)
        detected_500 = [(fire_rows[i], fire_cols[i]) for i in range(len(fire_rows)) if fire_opts[i] in ground_opt_set_500]
        non_disc_500 = []
        discoverable_500 = []
        for i, fopt in enumerate(fire_opts):
            if fopt in ground_opt_set_500:
                continue
            if fire_cluster(fopt, clusters_500, rescaled_max_battery) is None:
                non_disc_500.append((fire_rows[i], fire_cols[i]))
            else:
                discoverable_500.append((fire_rows[i], fire_cols[i]))
        out_500M = report_dir / "benchmark_fire_map_budget_500M.png"
        _plot_benchmark_overview(
            background_vis,
            mask_arr,
            clusters_500,
            sensors_data_500,
            stations_data_500,
            detected_500,  # fires at ground sensor cells = detected
            discoverable_500,
            non_disc_500,
            one_way_reach_opt,
            coverage_w,
            out_500M,
            title="Sensor placement (500M budget-optimized)",
        )
        print(f"Saved: {out_500M}")
    else:
        print(f"500M cache not found ({sensor_log_500M.name}); skip 500M figure.")

    # ── Pyrologix placement figures (20M, 100M, 500M) ─────────────────────────────
    PYROLOGIX_RESAMPLED = DATASET_DIR / "static_risk_pyrologix_resampled.npy"
    if PYROLOGIX_RESAMPLED.exists():
        pyro_map = np.load(str(PYROLOGIX_RESAMPLED))
        if pyro_map.ndim == 3:
            pyro_map = pyro_map[0]
        background_pyro = np.where(mask_arr > 0, pyro_map, 0.0)
        for budget in (20, 100, 500):
            sensor_log_pyro = LOG_DIR / f"sensor_alloc_GaussianBudget{budget}M_{rescaled_N}x{rescaled_M}_pyrologix.json"
            if not sensor_log_pyro.exists():
                print(f"Pyrologix {budget}M cache not found ({sensor_log_pyro.name}); skip.")
                continue
            with open(sensor_log_pyro) as f:
                data_p = json.load(f)
            ground_p = [tuple(x) for x in data_p["ground_sensor_locations"]]
            charging_p = [tuple(x) for x in data_p["charging_station_locations"]]
            drones_p = data_p["drones_per_charging_station"]
            clusters_p = compute_clusters(charging_p, drones_p, rescaled_max_battery)
            sensors_p = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in ground_p]
            stations_p = [(r * coverage_w + coverage_w // 2, c * coverage_w + coverage_w // 2) for r, c in charging_p]
            ground_opt_set_p = set(ground_p)
            detected_p = [(fire_rows[i], fire_cols[i]) for i in range(len(fire_rows)) if fire_opts[i] in ground_opt_set_p]
            non_disc_p = []
            discoverable_p = []
            for i, fopt in enumerate(fire_opts):
                if fopt in ground_opt_set_p:
                    continue
                if fire_cluster(fopt, clusters_p, rescaled_max_battery) is None:
                    non_disc_p.append((fire_rows[i], fire_cols[i]))
                else:
                    discoverable_p.append((fire_rows[i], fire_cols[i]))
            out_pyro = report_dir / f"benchmark_fire_map_pyrologix_{budget}M.png"
            _plot_benchmark_overview(
                background_pyro,
                mask_arr,
                clusters_p,
                sensors_p,
                stations_p,
                detected_p,  # fires at ground sensor cells = detected
                discoverable_p,
                non_disc_p,
                one_way_reach_opt,
                coverage_w,
                out_pyro,
                title=f"Sensor placement (Pyrologix, {budget}M budget)",
            )
            print(f"Saved: {out_pyro}")
    else:
        print(f"Pyrologix resampled map not found ({PYROLOGIX_RESAMPLED.name}); skip Pyrologix figures.")

    print("Done.")


if __name__ == "__main__":
    main()
