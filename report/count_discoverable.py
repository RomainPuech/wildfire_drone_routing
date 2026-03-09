#!/usr/bin/env python3
"""Count discoverable fires for each placement.

A fire is discoverable if either:
  - within L∞ ≤ 3 of a charging station (drone can reach it), or
  - its ignition cell is at a ground sensor (detected by ground sensor).

Run from project root: python report/count_discoverable.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "California2020Dataset"
LOG_DIR = DATASET_DIR / "logs"
CONFIG_PATH = DATASET_DIR / "config_california_2020.json"
SCENARII_DIR = DATASET_DIR / "scenarii"
BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED = 42
coverage_w = 5
rescaled_max_battery = 7


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
            if max(abs(xi - xj), abs(yi - yj)) <= rescaled_max_battery:
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
    one_way = rescaled_max_battery // 2  # 3
    fr, fc = fire_opt
    for cl in clusters:
        for sx, sy in cl["stations_opt"]:
            if max(abs(fr - sx), abs(fc - sy)) <= one_way:
                return cl
    return None


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    all_sf = sorted(SCENARII_DIR.glob("*.npy"))
    valid = [sf for sf in all_sf if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config for k in ("offset", "date", "time"))]
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.sort(rng.choice(len(valid), size=BENCHMARK_SUBSET_SIZE, replace=False))
    benchmark_scenarios = [valid[i] for i in idx]

    fire_opts = []
    for sf in benchmark_scenarios:
        pt = np.load(str(sf))
        fire_opts.append((int(pt[0]) // coverage_w, int(pt[1]) // coverage_w))

    placements = [
        ("WFPI 20M", "sensor_alloc_GaussianBudget20M_TOP_261x161_mean.json"),
        ("WFPI 100M", "sensor_alloc_GaussianBudget100M_261x161_mean.json"),
        ("WFPI 500M", "sensor_alloc_GaussianBudget500M_261x161_mean.json"),
        ("Pyrologix 20M", "sensor_alloc_GaussianBudget20M_261x161_pyrologix.json"),
        ("Pyrologix 100M", "sensor_alloc_GaussianBudget100M_261x161_pyrologix.json"),
        ("Pyrologix 500M", "sensor_alloc_GaussianBudget500M_261x161_pyrologix.json"),
    ]
    results = {}
    for label, filename in placements:
        path = LOG_DIR / filename
        if not path.exists():
            results[label] = None
            continue
        with open(path) as f:
            data = json.load(f)
        ground_opt = set(tuple(x) for x in data["ground_sensor_locations"])
        charging = [tuple(x) for x in data["charging_station_locations"]]
        drones = data["drones_per_charging_station"]
        clusters = compute_clusters(charging, drones)
        n_disc = sum(
            1 for fopt in fire_opts
            if fopt in ground_opt or fire_cluster(fopt, clusters) is not None
        )
        results[label] = n_disc

    print("Discoverable fires (of 100) — drone reach L∞≤3 or at ground sensor")
    print()
    for budget in ("20M", "100M", "500M"):
        w = results.get(f"WFPI {budget}")
        p = results.get(f"Pyrologix {budget}")
        print(f"  {budget}:  WFPI={w}   Pyrologix={p}")
    print()
    # Output as dict for pasting
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
