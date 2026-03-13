#!/usr/bin/env python3
"""
Generate benchmark report figures for California 2021 (20M Pyrologix placement).

Produces:
  report/benchmark_fire_locations_budget_2021.png
      100 benchmark fires on Pyrologix background.

  report/benchmark_fire_map_budget_2021_{combo}.png
      Overview map per routing combo (TOP, MaxCov):
      cluster zones + detected (green) / missed discoverable (black ×) /
      non-discoverable (gray ·) fires.

Run from project root:
  python report/generate_benchmark_report_figures_2021.py

Works before benchmark completes (no CSV): shows discoverable vs non-discoverable
only.  Re-run after benchmark to add detected/missed split.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

DATASET_DIR  = PROJECT_ROOT / "California2021Dataset"
PLACEMENT_MAP = DATASET_DIR / "static_risk_pyrologix.npy"
MASK_PATH    = DATASET_DIR / "mask.npy"
CONFIG_PATH  = DATASET_DIR / "config_california_2021.json"
SCENARII_DIR = DATASET_DIR / "scenarii"
LOG_DIR      = DATASET_DIR / "logs"
REPORT_DIR   = PROJECT_ROOT / "report"

BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED           = 42
SENSOR_POOLING        = "mean"
COVERAGE_W            = 5
RESCALED_MAX_BATTERY  = 7
ONE_WAY_REACH         = RESCALED_MAX_BATTERY // 2   # 3 opt-cells


# ── Geometry helpers ──────────────────────────────────────────────────────────

def compute_clusters(charging_locs_opt, drones_per_station, max_battery_substeps=7):
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


def fire_cluster(fire_opt, clusters):
    fr, fc = fire_opt
    for cluster in clusters:
        for sx, sy in cluster["stations_opt"]:
            if max(abs(fr - sx), abs(fc - sy)) <= ONE_WAY_REACH:
                return cluster
    return None


# ── Plot functions ────────────────────────────────────────────────────────────

def _plot_fire_locations(background_map, fire_rows, fire_cols, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = background_map.shape
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    im = ax.imshow(background_map, cmap="YlOrRd", origin="upper",
                   interpolation="nearest", vmin=0, vmax=255, extent=[0, W, H, 0])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Ignition Probability (Pyrologix, 0–255)")
    ax.scatter(fire_cols, fire_rows, marker="o", s=15,
               color="black", edgecolors="none", alpha=0.8, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Column (~1 km/cell)")
    ax.set_ylabel("Row (~1 km/cell)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def _plot_benchmark_overview(background_map, clusters, sensors_data, stations_data,
                              detected_fires, missed_disc_fires, non_disc_fires,
                              out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import matplotlib.colors as mcolors
    _CC = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
           "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]

    H, W = background_map.shape
    zone_half = ONE_WAY_REACH * COVERAGE_W
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    im = ax.imshow(background_map, cmap="YlOrRd", origin="upper",
                   interpolation="nearest", vmin=0, vmax=255, extent=[0, W, H, 0])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Ignition Probability (Pyrologix, 0–255)")

    # ── Cluster zone fill + border ─────────────────────────────────────────
    fill_overlay = np.zeros((H, W, 4), dtype=float)
    for i, cl in enumerate(clusters):
        colour = _CC[i % len(_CC)]
        rv, gv, bv = mcolors.to_rgb(colour)
        cl_mask = np.zeros((H, W), dtype=bool)
        for r_opt, c_opt in cl["stations_opt"]:
            r_d = r_opt * COVERAGE_W + COVERAGE_W // 2
            c_d = c_opt * COVERAGE_W + COVERAGE_W // 2
            r0, r1 = max(0, r_d - zone_half), min(H, r_d + zone_half + 1)
            c0, c1 = max(0, c_d - zone_half), min(W, c_d + zone_half + 1)
            cl_mask[r0:r1, c0:c1] = True
        fill_overlay[cl_mask, 0] = rv
        fill_overlay[cl_mask, 1] = gv
        fill_overlay[cl_mask, 2] = bv
        fill_overlay[cl_mask, 3] = 0.25
        # Border edges
        m = cl_mask.astype(np.int8)
        m_v = np.zeros((H + 2, W), dtype=np.int8); m_v[1:H + 1, :] = m
        diff_v = m_v[:-1, :] ^ m_v[1:, :]
        ry, cx = np.where(diff_v)
        if len(ry):
            segs = np.stack([np.column_stack([cx.astype(float), ry.astype(float)]),
                             np.column_stack([cx.astype(float) + 1, ry.astype(float)])], axis=1)
            ax.add_collection(mc.LineCollection(segs, colors=colour, linewidths=1.5, zorder=4))
        m_h = np.zeros((H, W + 2), dtype=np.int8); m_h[:, 1:W + 1] = m
        diff_h = m_h[:, :-1] ^ m_h[:, 1:]
        ry2, cx2 = np.where(diff_h)
        if len(ry2):
            segs = np.stack([np.column_stack([cx2.astype(float), ry2.astype(float)]),
                             np.column_stack([cx2.astype(float), ry2.astype(float) + 1])], axis=1)
            ax.add_collection(mc.LineCollection(segs, colors=colour, linewidths=1.5, zorder=4))
    ax.imshow(fill_overlay, origin="upper", extent=[0, W, H, 0], zorder=3,
              interpolation="nearest")

    def _rc(obj):
        if hasattr(obj, "iterrows"):
            return obj["row"].tolist(), obj["col"].tolist()
        return [r for r, _ in obj], [c for _, c in obj]

    nd_r, nd_c = _rc(non_disc_fires)
    md_r, md_c = _rc(missed_disc_fires)
    dt_r, dt_c = _rc(detected_fires)
    if nd_r:
        ax.scatter(nd_c, nd_r, marker=".", s=20, color="gray", alpha=0.7,
                   zorder=5, label=f"Non-discoverable (n={len(nd_r)})")
    if md_r:
        ax.scatter(md_c, md_r, marker="x", s=50, color="black", linewidths=1.5,
                   alpha=0.9, zorder=6, label=f"Missed discoverable (n={len(md_r)})")
    if dt_r:
        ax.scatter(dt_c, dt_r, marker="o", s=80, color="limegreen",
                   edgecolors="darkgreen", linewidths=0.8, zorder=8,
                   label=f"Detected (n={len(dt_r)})")
    if sensors_data:
        sr, sc = zip(*sensors_data)
        ax.scatter(sc, sr, marker="*", s=40, color="white", edgecolors="black",
                   linewidths=0.5, zorder=7, label=f"Ground sensor (n={len(sensors_data)})")
    if stations_data:
        tr, tc = zip(*stations_data)
        ax.scatter(tc, tr, marker="^", s=50, color="blue", edgecolors="white",
                   linewidths=0.8, zorder=7, label=f"Charging station (n={len(stations_data)})")

    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Column (~1 km/cell)"); ax.set_ylabel("Row (~1 km/cell)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    REPORT_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    pyro_map  = np.load(str(PLACEMENT_MAP))
    if pyro_map.ndim == 3:
        pyro_map = pyro_map[0]
    mask_arr  = np.load(str(MASK_PATH))
    H, W      = mask_arr.shape
    background_vis = np.where(mask_arr > 0, pyro_map.astype(float), 0.0)

    rescaled_N = H // COVERAGE_W
    rescaled_M = W // COVERAGE_W

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf for sf in all_scenario_files
        if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config
               for k in ("offset", "date", "time"))
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(rng.choice(len(valid_scenarios),
                                    size=BENCHMARK_SUBSET_SIZE, replace=False))
    benchmark_scenarios = [valid_scenarios[i] for i in subset_idx]

    fire_rows, fire_cols = [], []
    for sf in benchmark_scenarios:
        pt = np.load(str(sf))
        fire_rows.append(int(pt[0])); fire_cols.append(int(pt[1]))

    # ── Fire locations only plot ───────────────────────────────────────────────
    out_fire = REPORT_DIR / "benchmark_fire_locations_budget_2021.png"
    _plot_fire_locations(
        background_vis, fire_rows, fire_cols, out_fire,
        title="Fire ignition points — California 2021 (100 random, seed=42)"
    )

    # ── Load sensor placement cache ────────────────────────────────────────────
    sensor_cache_name = f"GaussianBudget20M_TOP_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}"
    sensor_log = LOG_DIR / f"sensor_alloc_{sensor_cache_name}.json"
    if not sensor_log.exists():
        print(f"Missing sensor cache {sensor_log.name}; run --sensor-only first.")
        return

    with open(sensor_log) as f:
        sd = json.load(f)
    ground_locs_opt    = [tuple(x) for x in sd["ground_sensor_locations"]]
    charging_locs_opt  = [tuple(x) for x in sd["charging_station_locations"]]
    drones_per_station = sd["drones_per_charging_station"]
    clusters           = compute_clusters(charging_locs_opt, drones_per_station)

    sensors_data  = [(r * COVERAGE_W + COVERAGE_W // 2,
                      c * COVERAGE_W + COVERAGE_W // 2) for r, c in ground_locs_opt]
    stations_data = [(r * COVERAGE_W + COVERAGE_W // 2,
                      c * COVERAGE_W + COVERAGE_W // 2) for r, c in charging_locs_opt]

    # ── Classify fires ────────────────────────────────────────────────────────
    fire_opts       = [(r // COVERAGE_W, c // COVERAGE_W)
                       for r, c in zip(fire_rows, fire_cols)]
    ground_opt_set  = set(ground_locs_opt)
    detected_ground = [(fire_rows[i], fire_cols[i]) for i in range(len(fire_rows))
                       if fire_opts[i] in ground_opt_set]
    non_disc        = []
    discoverable    = []
    for i, fopt in enumerate(fire_opts):
        if fopt in ground_opt_set:
            continue
        if fire_cluster(fopt, clusters) is None:
            non_disc.append((fire_rows[i], fire_cols[i]))
        else:
            discoverable.append((fire_rows[i], fire_cols[i]))

    # ── Load results CSV (if available) and generate per-combo plots ───────────
    csv_candidates = sorted(
        PROJECT_ROOT.glob("benchmark_results_yearly_2021_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    # Fallback: any benchmark results CSV
    if not csv_candidates:
        csv_candidates = sorted(
            PROJECT_ROOT.glob("benchmark_results_yearly_*.csv"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )

    # Combos to plot (only those present in CSV or as placement-only)
    combos_to_plot = [
        ("GaussianBudget20M_TOP",    "TOP routing (Team Orienteering Problem)"),
        ("GaussianBudget20M_MaxCov", "Max Coverage routing (growing suppression)"),
    ]

    for combo_name, combo_label in combos_to_plot:
        detected   = list(detected_ground)
        missed_disc = list(discoverable)

        if csv_candidates:
            import pandas as pd
            df_all = pd.read_csv(csv_candidates[0])
            if combo_name in df_all["strategy_combo"].values:
                df = df_all[df_all["strategy_combo"] == combo_name].reset_index(drop=True)
                if len(df) == len(benchmark_scenarios):
                    detected = list(detected_ground)
                    missed_disc = []
                    for i in range(len(benchmark_scenarios)):
                        r, c   = fire_rows[i], fire_cols[i]
                        device = df.iloc[i]["device"]
                        if device in ("drone", "ground sensor", "charging station"):
                            if (r, c) not in set(detected):
                                detected.append((r, c))
                    missed_disc = [(r, c) for r, c in discoverable
                                   if (r, c) not in set(detected)]
                    print(f"{combo_name}: detected={len(detected)}, "
                          f"missed_disc={len(missed_disc)}, non_disc={len(non_disc)}")

        safe = combo_name.replace("GaussianBudget20M_", "").lower()
        out = REPORT_DIR / f"benchmark_fire_map_budget_2021_{safe}.png"
        _plot_benchmark_overview(
            background_vis, clusters, sensors_data, stations_data,
            detected, missed_disc, non_disc, out,
            title=(f"California 2021 — 20M budget — {combo_label}\n"
                   f"n={len(fire_rows)} fires · {len(clusters)} clusters · "
                   f"discovered={len(detected)+len(missed_disc)} · "
                   f"non-disc={len(non_disc)}"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
