#!/usr/bin/env python3
"""Print placement-only fire counts for benchmark subset (n=100, seed=42).

Uses the same geometry as visualize_sensor_placement_2021.py. Run from repo root:
  python3 report/benchmark_2021_greedy_kernel/print_placement_detectability.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location(
    "viz2021", PROJECT_ROOT / "visualize_sensor_placement_2021.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATASET = PROJECT_ROOT / "California2021Dataset"
CONFIG_PATH = DATASET / "config_california_2021.json"
SCENARII_DIR = DATASET / "scenarii"
COVERAGE_W = 5
BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED = 42


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf for sf in all_scenario_files
        if all(
            f"{k}_{sf.stem.replace('_scenario1', '')}" in config
            for k in ("offset", "date", "time")
        )
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(
        rng.choice(len(valid_scenarios), size=BENCHMARK_SUBSET_SIZE, replace=False)
    )
    benchmark_scenarios = [valid_scenarios[i] for i in subset_idx]
    fire_rows: list[int] = []
    fire_cols: list[int] = []
    for sf in benchmark_scenarios:
        pt = np.load(str(sf))
        fire_rows.append(int(pt[0]))
        fire_cols.append(int(pt[1]))

    for budget in (20, 100, 500):
        path = (
            DATASET / "logs"
            / f"sensor_alloc_GaussianBudget{budget}M_StationMaxGreedyUniform_261x161_mean.json"
        )
        with open(path) as f:
            d = json.load(f)
        ground_opt = {tuple(x) for x in d["ground_sensor_locations"]}
        charging = [tuple(x) for x in d["charging_station_locations"]]
        drones = d["drones_per_charging_station"]
        clusters = mod.compute_clusters(charging, drones)
        dg, disc, nd = mod.classify_fires(
            fire_rows, fire_cols, clusters, ground_opt, COVERAGE_W
        )
        total = len(dg) + len(disc)
        print(
            f"GaussianBudget{budget}M: ground={len(dg)}, drone_reachable={len(disc)}, "
            f"not_discoverable={len(nd)}, placement_detectable={total}/100"
        )


if __name__ == "__main__":
    main()
