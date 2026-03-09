#!/usr/bin/env python3
"""Count how many of the 100 benchmark scenarios are covered by current routing logs.
Run from project root: python report/count_scenarios_done.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "California2020Dataset"
LOG_DIR = DATASET_DIR / "logs"
CONFIG_PATH = DATASET_DIR / "config_california_2020.json"
SCENARII_DIR = DATASET_DIR / "scenarii"
BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED = 42
coverage_w = 5
rescaled_max_battery = 7
N_SCENARIO_DATA_STEPS = 6
operational_substeps = 7


def round_to_nearest_hour(dt):
    if dt.minute >= 30:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)


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
    one_way = rescaled_max_battery // 2
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

    sensor_path = LOG_DIR / "sensor_alloc_GaussianBudget20M_TOP_261x161_mean.json"
    with open(sensor_path) as f:
        sensor_data = json.load(f)
    ground_locs_opt = [tuple(x) for x in sensor_data["ground_sensor_locations"]]
    charging_locs_opt = [tuple(x) for x in sensor_data["charging_station_locations"]]
    drones_per_station = sensor_data["drones_per_charging_station"]
    clusters = compute_clusters(charging_locs_opt, drones_per_station)

    # Load TOP routing logs (one per cluster)
    top_logs = {}
    for cl in clusters:
        path = LOG_DIR / f"routing_yearly_DroneRoutingTOPMasked_10OH_5RS_cluster_{cl['fingerprint']}.json"
        if path.exists():
            with open(path) as f:
                top_logs[cl["fingerprint"]] = json.load(f)
        else:
            top_logs[cl["fingerprint"]] = {}

    # Load MaxCoverage routing logs if any
    maxcov_logs = {}
    for cl in clusters:
        path = LOG_DIR / f"routing_yearly_DroneRoutingMaxCoverageResetStaticMasked_10OH_5RS_cluster_{cl['fingerprint']}.json"
        if path.exists():
            with open(path) as f:
                maxcov_logs[cl["fingerprint"]] = json.load(f)
        else:
            maxcov_logs[cl["fingerprint"]] = {}

    def has_routing(data, log_key, min_steps):
        entry = data.get(log_key)
        return entry and len(entry.get("actions_history", [])) >= min_steps

    top_done = 0
    maxcov_done = 0
    no_cluster = 0
    for sf in benchmark_scenarios:
        name = sf.stem.replace("_scenario1", "")
        date_str = config[f"date_{name}"]
        time_str = config[f"time_{name}"]
        offset = config[f"offset_{name}"]
        discovery_dt = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
                                int(time_str[:2]), int(time_str[2:4]))
        sim_start = round_to_nearest_hour(discovery_dt - timedelta(minutes=30 * offset))
        log_key = f"{sim_start.strftime('%Y%m%d')}_{sim_start.hour:02d}"
        pt = np.load(str(sf))
        fire_opt = (int(pt[0]) // coverage_w, int(pt[1]) // coverage_w)
        cl = fire_cluster(fire_opt, clusters)
        total_substeps = (offset + N_SCENARIO_DATA_STEPS) * operational_substeps
        if cl is None:
            no_cluster += 1
            # These are "done" as soon as we process them (sensor-only path)
            top_done += 1
            maxcov_done += 1
        else:
            if has_routing(top_logs[cl["fingerprint"]], log_key, total_substeps):
                top_done += 1
            if has_routing(maxcov_logs[cl["fingerprint"]], log_key, total_substeps):
                maxcov_done += 1

    print("Benchmark: 100 scenarios (seed=42), same sensor placement (20M TOP).")
    print(f"  No cluster (sensor-only): {no_cluster}")
    print(f"  TOP:    {top_done}/100 scenarios covered by current routing logs.")
    print(f"  MaxCov: {maxcov_done}/100 scenarios covered by current routing logs.")
    n_top_routing = sum(len(v) for v in top_logs.values() if isinstance(v, dict))
    n_maxcov_routing = sum(len(v) for v in maxcov_logs.values() if isinstance(v, dict))
    print(f"  TOP routing log entries (cluster, log_key): {n_top_routing}")
    print(f"  MaxCov routing log entries: {n_maxcov_routing}")


if __name__ == "__main__":
    main()
