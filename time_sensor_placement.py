#!/usr/bin/env python3
"""
Profile sensor placement ILP time across different hardware counts.

Tests:
  - JIT compilation (first call)
  - Pre-filtering (coverage potential computation)
  - Model creation
  - ILP solving

Usage:
    python -u time_sensor_placement.py
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

print("Importing modules...", flush=True)
from benchmark import (
    compute_operational_substeps,
    pool_burnmap_mean,
    pool_mask,
)
from Strategy import SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation
print("Imports done.\n", flush=True)

DATASET_DIR = PROJECT_ROOT / "California2020Dataset"
AVG_MAP     = DATASET_DIR / "static_risk_wfpi_avg.npy"
MASK_PATH   = DATASET_DIR / "mask.npy"

# Fixed simulation params
CELL_SIZE_M    = 1000
SPEED          = 600
COVERAGE_R_M   = 2900
MAX_BATTERY    = 1

operational_substeps = compute_operational_substeps(CELL_SIZE_M, SPEED, COVERAGE_R_M)
coverage_w = round(COVERAGE_R_M * 2 / CELL_SIZE_M)
if coverage_w % 2 == 0:
    coverage_w -= 1

mask        = np.load(str(MASK_PATH))
H, W        = mask.shape
rescaled_N  = H // coverage_w
rescaled_M  = W // coverage_w
rescaled_max_battery = MAX_BATTERY * operational_substeps

print(f"Grid: {H}×{W} → opt {rescaled_N}×{rescaled_M}, coverage_w={coverage_w}, substeps={operational_substeps}", flush=True)

# Build rescaled mask and burnmap (masked before pooling)
suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"
rescaled_mask_path = str(MASK_PATH).replace(".npy", suffix)

from benchmark import pool_mask
rescaled_mask = pool_mask(mask, coverage_w, mode="max")
np.save(rescaled_mask_path, rescaled_mask)

avg_map = np.load(str(AVG_MAP))
avg_map_masked = avg_map * mask
rescaled_avg = pool_burnmap_mean(avg_map_masked, coverage_w)
rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
rescaled_avg_path = str(AVG_MAP).replace(".npy", f"_mean{suffix}")
np.save(rescaled_avg_path, rescaled_avg)

base_auto = {
    "N":                   rescaled_N,
    "M":                   rescaled_M,
    "max_battery_distance":-1,
    "max_battery_time":    rescaled_max_battery,
    "speed_m_per_min":     SPEED,
    "coverage_radius_m":   COVERAGE_R_M,
    "cell_size_m":         CELL_SIZE_M,
    "transmission_range":  50000,
    "mask_filename":       rescaled_mask_path,
}
base_custom = {
    "mask_filename":        rescaled_mask_path,
    "burnmap_filename":     rescaled_avg_path,
    "recompute_logfile":    False,
    "recompute_kernel":     False,
    "use_linf_cost":        True,
    "regularization_param": 1e5,
    "reevaluation_step":    5,
    "optimization_horizon": 10,
}

# Hardware configs to test
CONFIGS = [
    (10,  10,  10),   # small: 10 sensors, 10 stations, 10 drones
    (20,  20,  20),
    (50,  25,  25),
    (100, 50,  50),   # target budget config
]

for (n_sensors, n_stations, n_drones) in CONFIGS:
    print(f"\n{'='*60}", flush=True)
    print(f"  n_sensors={n_sensors}  n_stations={n_stations}  n_drones={n_drones}", flush=True)
    print(f"{'='*60}", flush=True)

    # Delete stale sensor cache so it recomputes
    log_path = DATASET_DIR / "logs" / f"sensor_alloc_timing_{rescaled_N}x{rescaled_M}_{n_sensors}s_{n_stations}cs.json"
    if log_path.exists():
        log_path.unlink()

    auto = {**base_auto, "n_drones": n_drones, "n_ground_stations": n_sensors, "n_charging_stations": n_stations}
    custom = {**base_custom}

    t0 = time.perf_counter()
    try:
        strat = SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation(auto, custom)
        ground  = strat.get_locations()[0]
        stations = strat.get_locations()[1]
        alloc   = strat.get_drone_allocation()
        t1 = time.perf_counter()
        print(f"  Total wall time: {t1-t0:.1f}s", flush=True)
        print(f"  Sensors placed: {len(ground)}  Stations: {len(stations)}  Drones: {sum(alloc)}", flush=True)
    except Exception as e:
        t1 = time.perf_counter()
        print(f"  FAILED after {t1-t0:.1f}s: {e}", flush=True)

print("\nDone.", flush=True)
