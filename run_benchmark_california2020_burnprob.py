#!/usr/bin/env python3
"""
Sensor placement on California 2020 using the BP (FSim burn probability) risk map.

Source: California2020Dataset_BurnProb/static_risk_burn_prob.npy (native 4865×2834).
Resampled to WFPI grid (1309×805) and cached as California2020Dataset/static_risk_burn_prob_resampled.npy.
Runs Budget sensor placement for 20M, 100M, 500M; after each budget, generates the placement plot.

Run from project root:
  python-jl run_benchmark_california2020_burnprob.py --sensor-only
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

print("Importing modules...", flush=True)
from Strategy import SensorPlacementMaxCoverageGaussianTimeMaskedBudget
from benchmark import compute_operational_substeps, pool_burnmap_mean, pool_mask
print("Imports done.\n", flush=True)

WFPI_DIR = PROJECT_ROOT / "California2020Dataset"
BP_RAW = PROJECT_ROOT / "California2020Dataset_BurnProb" / "static_risk_burn_prob.npy"
BP_RESAMPLED = WFPI_DIR / "static_risk_burn_prob_resampled.npy"
MASK_PATH = WFPI_DIR / "mask.npy"
LOG_DIR = WFPI_DIR / "logs"
SENSOR_POOLING = "mean"
BUDGET_TOTAL, COST_SENSOR, COST_DRONE, COST_STATION = 20_000_000, 100_000, 50_000, 150_000
SIMULATION_PARAMETERS = {
    "max_battery_distance": -1, "max_battery_time": 1, "n_drones": 50,
    "n_ground_stations": 100, "n_charging_stations": 50,
    "drone_speed_m_per_min": 600, "coverage_radius_m": 2900,
    "cell_size_m": 1000, "transmission_range": 50000, "mask_pooling_mode": "max",
}


def resample_burnprob(raw_path: Path, target_shape: tuple, out_path: Path) -> np.ndarray:
    """Resample BP from native grid to (H, W). Scale to [0, 255]. Save as (1, H, W)."""
    if out_path.exists():
        print(f"  [burnprob] Loading cached resampled map from {out_path.name}", flush=True)
        return np.load(str(out_path))[0]
    print(f"  [burnprob] Resampling {raw_path.name} → {target_shape} ...", flush=True)
    from skimage.transform import resize
    raw = np.load(str(raw_path))
    if raw.ndim == 3:
        raw = raw[0]
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0)
    H_t, W_t = target_shape
    resampled = resize(raw, (H_t, W_t), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    valid = resampled[np.isfinite(resampled) & (resampled > 0)]
    if valid.size:
        resampled = resampled / valid.max() * 255.0
    resampled = np.nan_to_num(resampled, nan=0.0)
    out = resampled[np.newaxis]
    np.save(str(out_path), out)
    print(f"  [burnprob] Saved to {out_path.name}  range=[{resampled.min():.2f}, {resampled.max():.2f}]", flush=True)
    return resampled


def load_or_compute_sensor_placement(strategy_cls, rescaled_auto, rescaled_custom, log_path: str):
    if Path(log_path).exists():
        print(f"  [sensor] Loading cached placement from {Path(log_path).name}", flush=True)
        with open(log_path) as f:
            d = json.load(f)
        return (
            [tuple(x) for x in d["ground_sensor_locations"]],
            [tuple(x) for x in d["charging_station_locations"]],
            d["drones_per_charging_station"],
        )
    print("  [sensor] Computing placement (Julia) ...", flush=True)
    strat = strategy_cls(rescaled_auto, rescaled_custom)
    ground_locs, charging_locs = strat.get_locations()
    drones_per_station = strat.get_drone_allocation()
    log_data = {
        "ground_sensor_locations": [[int(v) for v in x] for x in ground_locs],
        "charging_station_locations": [[int(v) for v in x] for x in charging_locs],
        "drones_per_charging_station": [int(x) for x in drones_per_station],
    }
    if hasattr(strat, "get_device_counts"):
        log_data["device_counts"] = strat.get_device_counts()
        log_data["budget_millions"] = getattr(strat, "budget_millions", None)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  [sensor] Saved to {Path(log_path).name}", flush=True)
    return (
        [tuple(int(v) for v in x) for x in ground_locs],
        [tuple(int(v) for v in x) for x in charging_locs],
        [int(x) for x in drones_per_station],
    )


def run_sensor_placement_only_burnprob(budgets=(20, 100, 500), time_limit_seconds=600.0):
    LOG_DIR.mkdir(exist_ok=True)
    print("Loading WFPI mask ...", flush=True)
    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
    print("Preparing BP (burn probability) risk map ...", flush=True)
    bp_2d = resample_burnprob(BP_RAW, (H, W), BP_RESAMPLED)

    cell_size_m = SIMULATION_PARAMETERS["cell_size_m"]
    speed = SIMULATION_PARAMETERS["drone_speed_m_per_min"]
    coverage_r_m = SIMULATION_PARAMETERS["coverage_radius_m"]
    operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
    coverage_w = round(coverage_r_m * 2 / cell_size_m)
    if coverage_w % 2 == 0:
        coverage_w -= 1
    rescaled_N = H // coverage_w
    rescaled_M = W // coverage_w
    rescaled_max_battery = SIMULATION_PARAMETERS["max_battery_time"] * operational_substeps
    suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"
    rescaled_mask_path = str(MASK_PATH).replace(".npy", suffix)
    if not Path(rescaled_mask_path).exists():
        rescaled_mask = pool_mask(mask, coverage_w, mode=SIMULATION_PARAMETERS["mask_pooling_mode"])
        np.save(rescaled_mask_path, rescaled_mask)
    bp_masked = bp_2d * mask
    rescaled_avg = pool_burnmap_mean(bp_masked[np.newaxis], coverage_w)
    rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
    rescaled_avg_path = str(BP_RESAMPLED).replace(".npy", f"_{SENSOR_POOLING}{suffix}")
    np.save(rescaled_avg_path, rescaled_avg)

    base_rescaled_auto = {
        "N": H, "M": W,
        "max_battery_distance": SIMULATION_PARAMETERS["max_battery_distance"],
        "max_battery_time": rescaled_max_battery,
        "n_drones": SIMULATION_PARAMETERS["n_drones"],
        "n_ground_stations": SIMULATION_PARAMETERS["n_ground_stations"],
        "n_charging_stations": SIMULATION_PARAMETERS["n_charging_stations"],
        "speed_m_per_min": speed,
        "coverage_radius_m": coverage_r_m,
        "cell_size_m": cell_size_m,
        "transmission_range": SIMULATION_PARAMETERS["transmission_range"],
        "mask_filename": rescaled_mask_path,
        "N": rescaled_N, "M": rescaled_M,
    }
    base_custom = {
        "mask_filename": rescaled_mask_path,
        "recompute_logfile": True,
        "recompute_kernel": False,
        "use_linf_cost": True,
        "regularization_param": 1e5,
    }

    for budget in budgets:
        combo_name = f"GaussianBudget{int(budget)}M"
        sensor_log_path = str(LOG_DIR / f"sensor_alloc_{combo_name}_{rescaled_N}x{rescaled_M}_burnprob.json")
        sensor_custom = {
            **base_custom,
            "burnmap_filename": rescaled_avg_path,
            "budget_millions": float(budget),
            "cost_sensor": COST_SENSOR / 1_000_000,
            "cost_station": COST_STATION / 1_000_000,
            "cost_drone": COST_DRONE / 1_000_000,
            "time_limit_seconds": time_limit_seconds,
        }
        print(f"\n{'='*70}", flush=True)
        print(f"  BP (burn probability) sensor placement: budget={budget}M, time limit={time_limit_seconds}s", flush=True)
        print(f"{'='*70}\n", flush=True)
        load_or_compute_sensor_placement(
            SensorPlacementMaxCoverageGaussianTimeMaskedBudget,
            base_rescaled_auto,
            sensor_custom,
            sensor_log_path,
        )
        # Generate plot for this budget
        print(f"  Generating placement plot for {budget}M ...", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "report" / "generate_benchmark_report_figures.py")],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    print("\nBP sensor placement done for budgets:", budgets, flush=True)


if __name__ == "__main__":
    if "--sensor-only" not in sys.argv:
        print("Use: python-jl run_benchmark_california2020_burnprob.py --sensor-only", file=sys.stderr)
        sys.exit(1)
    budgets = [20, 100, 500]
    if "--budget" in sys.argv:
        try:
            i = sys.argv.index("--budget")
            budgets = [int(b) for b in sys.argv[i + 1].split(",")]
        except (ValueError, IndexError):
            pass
    time_limit = 600.0
    if "--time-limit" in sys.argv:
        try:
            j = sys.argv.index("--time-limit")
            time_limit = float(sys.argv[j + 1])
        except (ValueError, IndexError):
            pass
    run_sensor_placement_only_burnprob(budgets=budgets, time_limit_seconds=time_limit)
