#!/usr/bin/env python3
"""
Quick test: run the budget-based sensor placement on the California 2020 dataset.
Results are saved in the same cache format used by run_benchmark_california2020_yearly.py
so that a subsequent full benchmark run can reuse the placement without recomputing.
"""

import sys, os, json, numpy as np
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

print("Importing modules...", flush=True)
from Strategy import SensorPlacementMaxCoverageGaussianTimeMaskedBudget
from benchmark import compute_operational_substeps, pool_burnmap_mean, pool_mask
print("Imports done.\n", flush=True)

# ── Paths (must match run_benchmark_california2020_yearly.py) ─────────────────
DATASET_DIR = PROJECT_ROOT / "California2020Dataset"
AVG_MAP     = DATASET_DIR / "static_risk_wfpi_avg.npy"
MASK_PATH   = DATASET_DIR / "mask.npy"
LOG_DIR     = DATASET_DIR / "logs"

SENSOR_POOLING = "mean"
COMBO_NAME     = "GaussianBudget20M_TOP"

# ── Simulation parameters (must match the benchmark runner) ───────────────────
cell_size_m  = 1000
speed        = 600
coverage_r_m = 2900

operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
coverage_w = round(coverage_r_m * 2 / cell_size_m)
if coverage_w % 2 == 0:
    coverage_w -= 1

mask = np.load(str(MASK_PATH))
H, W = mask.shape
rescaled_N = H // coverage_w
rescaled_M = W // coverage_w
rescaled_max_battery = 1 * operational_substeps   # 1 hour * substeps

print(f"Grid: {H}x{W} -> opt {rescaled_N}x{rescaled_M}, substeps={operational_substeps}, "
      f"max_battery_substeps={rescaled_max_battery}", flush=True)

suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"

rescaled_mask = pool_mask(mask, coverage_w, mode="max")
rescaled_mask_path = str(MASK_PATH).replace(".npy", suffix)
np.save(rescaled_mask_path, rescaled_mask)

avg_map = np.load(str(AVG_MAP))
avg_map_masked = avg_map * mask
rescaled_avg = pool_burnmap_mean(avg_map_masked, coverage_w)
rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
rescaled_avg_path = str(AVG_MAP).replace(".npy", f"_{SENSOR_POOLING}{suffix}")
np.save(rescaled_avg_path, rescaled_avg)

# ── Cache path (same format as the benchmark runner) ──────────────────────────
LOG_DIR.mkdir(exist_ok=True)
sensor_log_path = str(
    LOG_DIR / f"sensor_alloc_{COMBO_NAME}_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}.json"
)

if Path(sensor_log_path).exists():
    print(f"Cache already exists at {sensor_log_path} — skipping computation.", flush=True)
    with open(sensor_log_path) as f:
        cached = json.load(f)
    print(json.dumps(cached, indent=2))
    sys.exit(0)

# ── Run placement ─────────────────────────────────────────────────────────────
auto_params = {
    "N": rescaled_N,
    "M": rescaled_M,
    "max_battery_time": rescaled_max_battery,
}

custom_params = {
    "burnmap_filename":  rescaled_avg_path,
    "mask_filename":     rescaled_mask_path,
    "recompute_kernel":  False,
    "budget_millions":   20.0,
    "cost_sensor":       0.1,
    "cost_station":      0.15,
    "cost_drone":        0.05,
    "time_limit_seconds": 600.0,
}

print("\n" + "="*70, flush=True)
print("  Running SensorPlacementMaxCoverageGaussianTimeMaskedBudget", flush=True)
print("="*70 + "\n", flush=True)

strat = SensorPlacementMaxCoverageGaussianTimeMaskedBudget(auto_params, custom_params)

ground_locs, charging_locs = strat.get_locations()
drones_per_station = strat.get_drone_allocation()

# ── Save in the benchmark-runner cache format ─────────────────────────────────
log_data = {
    "ground_sensor_locations":     [[int(v) for v in x] for x in ground_locs],
    "charging_station_locations":  [[int(v) for v in x] for x in charging_locs],
    "drones_per_charging_station": [int(x) for x in drones_per_station],
    "device_counts":               strat.get_device_counts(),
    "budget_millions":             strat.budget_millions,
}

with open(sensor_log_path, "w") as f:
    json.dump(log_data, f, indent=2)
print(f"\nSaved placement cache to: {sensor_log_path}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== RESULTS ===")
print(f"Ground sensors:     {strat.n_ground_sensors}")
print(f"Charging stations:  {strat.n_charging_stations}")
print(f"Drones:             {strat.n_drones}")
print(f"Drone allocation:   {drones_per_station}")
budget_used = (strat.n_ground_sensors * 0.1
               + strat.n_charging_stations * 0.15
               + strat.n_drones * 0.05)
print(f"Budget used:        {budget_used:.2f}M / 20.0M")
print("Done.", flush=True)
