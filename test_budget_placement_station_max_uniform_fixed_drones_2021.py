#!/usr/bin/env python3
"""
California 2021: budget StationMax with exactly ``fixed_drones_per_station`` drones
per open charging station and binary per-station cell coverage (UniformFixedDrones).

Optional ``--warm-start`` accepts the same sensor_alloc JSON as greedy-uniform
(e.g. 100M StationMaxGreedyUniform) to seed Gurobi MIP starts for ground + stations.
"""

import sys
import os
import json
import argparse
import shutil
from typing import Optional

import numpy as np
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

print("Importing modules...", flush=True)
from Strategy import SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniformFixedDrones
from benchmark import compute_operational_substeps, pool_burnmap_mean, pool_mask
print("Imports done.\n", flush=True)

DATASET_DIR = PROJECT_ROOT / "California2021Dataset"
PLACEMENT_MAP = DATASET_DIR / "static_risk_pyrologix.npy"
MASK_PATH = DATASET_DIR / "mask.npy"
LOG_DIR = DATASET_DIR / "logs"

SENSOR_POOLING = "mean"


def main(
    budget_millions: float = 20.0,
    time_limit_seconds: float = 600.0,
    warm_start_file: str = "",
    candidate_percentile: Optional[float] = None,
    output_tag: Optional[str] = None,
    fixed_drones_per_station: int = 7,
    budget_regularization_epsilon: float = -1.0,
):
    cell_size_m = 1000
    speed = 600
    coverage_r_m = 2900

    operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
    coverage_w = round(coverage_r_m * 2 / cell_size_m)
    if coverage_w % 2 == 0:
        coverage_w -= 1

    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
    rescaled_N = H // coverage_w
    rescaled_M = W // coverage_w
    rescaled_max_battery = 1 * operational_substeps

    print(
        f"Grid: {H}x{W} -> opt {rescaled_N}x{rescaled_M}, substeps={operational_substeps}, "
        f"max_battery_substeps={rescaled_max_battery}",
        flush=True,
    )

    suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"

    rescaled_mask_path = str(MASK_PATH).replace(".npy", suffix)
    if not Path(rescaled_mask_path).exists():
        rescaled_mask = pool_mask(mask, coverage_w, mode="max")
        np.save(rescaled_mask_path, rescaled_mask)

    rescaled_avg_path = str(PLACEMENT_MAP).replace(".npy", f"_{SENSOR_POOLING}{suffix}")
    if not Path(rescaled_avg_path).exists():
        avg_map = np.load(str(PLACEMENT_MAP))
        avg_map_masked = avg_map * mask
        rescaled_avg = pool_burnmap_mean(avg_map_masked, coverage_w)
        rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
        np.save(rescaled_avg_path, rescaled_avg)

    if candidate_percentile is not None:
        cp = candidate_percentile
    else:
        cp = 0.5 if 100.0 <= budget_millions < 150.0 else 0.0 if budget_millions >= 150.0 else 0.8

    filt_suffix = f"_filt{int(cp * 100)}" if cp > 0 else ""

    LOG_DIR.mkdir(exist_ok=True)
    combo_name = f"GaussianBudget{int(budget_millions)}M_StationMaxUniformFixedDrones"
    tag_part = f"_{output_tag.strip()}" if (output_tag and output_tag.strip()) else ""
    canonical_path = LOG_DIR / (
        f"sensor_alloc_{combo_name}_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}{tag_part}.json"
    )
    sensor_log_path = str(canonical_path)

    auto_params = {
        "N": rescaled_N,
        "M": rescaled_M,
        "max_battery_distance": -1,
        "max_battery_time": rescaled_max_battery,
        "n_drones": 50,
        "n_ground_stations": 100,
        "n_charging_stations": 50,
    }

    custom_params = {
        "burnmap_filename": rescaled_avg_path,
        "mask_filename": rescaled_mask_path,
        "budget_millions": float(budget_millions),
        "cost_sensor": 0.1,
        "cost_station": 0.15,
        "cost_drone": 0.05,
        "candidate_percentile": cp,
        "time_limit_seconds": float(time_limit_seconds),
        "warm_start_file": warm_start_file,
        "fixed_drones_per_station": int(fixed_drones_per_station),
        "budget_regularization_epsilon": float(budget_regularization_epsilon),
    }

    print("\n" + "=" * 70, flush=True)
    print("  Running SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniformFixedDrones", flush=True)
    print("=" * 70 + "\n", flush=True)

    strat = SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniformFixedDrones(
        auto_params, custom_params
    )

    ground_locs, charging_locs = strat.get_locations()
    drones_per_station = strat.get_drone_allocation()

    log_data = {
        "ground_sensor_locations": [[int(v) for v in x] for x in ground_locs],
        "charging_station_locations": [[int(v) for v in x] for x in charging_locs],
        "drones_per_charging_station": [int(x) for x in drones_per_station],
        "device_counts": strat.get_device_counts(),
        "budget_millions": strat.budget_millions,
        "strategy": strat.strategy_name,
    }
    with open(sensor_log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    if filt_suffix:
        alt_path = LOG_DIR / (
            f"sensor_alloc_{combo_name}_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}{tag_part}{filt_suffix}.json"
        )
        shutil.copy2(sensor_log_path, alt_path)
        print(f"Also wrote filt copy: {alt_path}", flush=True)
    print(f"\nSaved placement cache to: {sensor_log_path}", flush=True)

    print("\n=== RESULTS ===", flush=True)
    print(f"Ground sensors:     {strat.n_ground_sensors}", flush=True)
    print(f"Charging stations:  {strat.n_charging_stations}", flush=True)
    print(f"Drones:             {strat.n_drones}", flush=True)
    print(f"Drone allocation:   {drones_per_station}", flush=True)
    budget_used = (
        strat.n_ground_sensors * 0.1
        + strat.n_charging_stations * 0.15
        + strat.n_drones * 0.05
    )
    print(f"Budget used:        {budget_used:.2f}M / {budget_millions:.1f}M", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument(
        "--warm-start",
        type=str,
        default="",
        help="sensor_alloc JSON (e.g. 100M StationMaxGreedyUniform) for MIP warm start",
    )
    parser.add_argument(
        "--candidate-percentile",
        type=float,
        default=None,
        help="Julia quantile p: keep top (1-p)*100%% cells (0.8→top 20%%, 0.2→top 80%%).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional tag in filename for parallel runs, e.g. ws100M_filt20",
    )
    parser.add_argument(
        "--fixed-drones-per-station",
        type=int,
        default=7,
        help="Drones per open charging station (default 7).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=-1.0,
        help="Budget regularization epsilon (default: auto, 0.1 for budget>300M else 0).",
    )
    args = parser.parse_args()
    main(
        budget_millions=args.budget,
        time_limit_seconds=args.time_limit,
        warm_start_file=args.warm_start,
        candidate_percentile=args.candidate_percentile,
        output_tag=(args.output_tag.strip() or None),
        fixed_drones_per_station=args.fixed_drones_per_station,
        budget_regularization_epsilon=args.epsilon,
    )
