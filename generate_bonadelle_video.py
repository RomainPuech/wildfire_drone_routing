#!/usr/bin/env python3
"""
Generate a video of drone movement for the missed discoverable fire:
BONADELLE_400569264 — ignition 2020-07-12 15:16, cluster 131-59_138-66

The fire is *discoverable* (within L∞ ≤ 7 of a charging station) but was
NOT detected by the drones in the benchmark.  This video shows the PSO-optimised
patrol routes alongside the fire ignition point so we can see why it was missed.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

from dataset import load_scenario_npy
from benchmark import (
    compute_operational_substeps,
    pool_burnmap_mean,
)
import displays   # must import so that LogNorm & imageio are in scope inside the function

# ── Constants (must match run_benchmark_california2020_yearly.py) ────────────
DATASET_DIR  = PROJECT_ROOT / "California2020Dataset"
YEARLY_MAP   = DATASET_DIR  / "static_risk_wfpi_yearly.npy"
MASK_PATH    = DATASET_DIR  / "mask.npy"
LOG_DIR      = DATASET_DIR  / "logs"

DRONE_SPEED_M_PER_MIN = 600
COVERAGE_RADIUS_M     = 2900
CELL_SIZE_M           = 1000
MAX_BATTERY_TIME_H    = 1          # hours → rescaled to substeps

MAX_ROUTING_DATA_STEPS = 24        # frames built for routing
N_SCENARIO_DATA_STEPS  = 12        # 12 × 30 min = 6 h

# ── Fire / routing specifics ─────────────────────────────────────────────────
FIRE_NAME    = "BONADELLE_400569264"
SCENARIO_NPY = DATASET_DIR / "scenarii" / f"{FIRE_NAME}_scenario1.npy"
CLUSTER_FP   = "131-59_138-66"
LOG_KEY      = "20200712_15"       # sim_start = 2020-07-12 15:00

ROUTING_LOG_PATH = LOG_DIR / (
    f"routing_yearly_DroneRoutingTOPMasked_10OH_5RS_cluster_{CLUSTER_FP}.json"
)
SENSOR_LOG_PATH  = LOG_DIR / "sensor_alloc_GaussianAlloc_TOP_261x161_mean.json"

OUTPUT_NAME = "bonadelle_missed_fire"

# ── Fire offset (from config) ─────────────────────────────────────────────────
# offset=4 means the fire ignites 4×30 min = 2 h after sim_start (at substep 4×7=28)
FIRE_OFFSET_DATA_STEPS = 4


def frame_index(dt: datetime) -> int:
    """Map a datetime to its index in the yearly WFPI map (2 frames per day)."""
    doy  = dt.timetuple().tm_yday
    half = 0 if dt.hour < 10 else 1
    return 2 * (doy - 1) + half


def build_burn_map(yearly_map: np.ndarray, sim_start: datetime, num_steps: int) -> np.ndarray:
    """Return (num_steps, H, W) WFPI frames starting at sim_start (30-min steps)."""
    frames = []
    for t in range(num_steps):
        dt = sim_start + timedelta(minutes=30 * t)
        frames.append(yearly_map[frame_index(dt)])
    return np.stack(frames)


def main():
    # ── Derived geometry ─────────────────────────────────────────────────────
    operational_substeps = compute_operational_substeps(
        CELL_SIZE_M, DRONE_SPEED_M_PER_MIN, COVERAGE_RADIUS_M
    )
    coverage_w = round(2 * COVERAGE_RADIUS_M / CELL_SIZE_M)
    if coverage_w % 2 == 0:
        coverage_w -= 1

    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
    rescaled_N = H // coverage_w   # 261
    rescaled_M = W // coverage_w   # 161
    rescaled_max_battery = MAX_BATTERY_TIME_H * operational_substeps  # 7

    print(f"coverage_w={coverage_w}, substeps={operational_substeps}")
    print(f"opt grid = {rescaled_N}×{rescaled_M}, max_battery_substeps={rescaled_max_battery}")

    # ── Build burn map (operational scale) ───────────────────────────────────
    print(f"\nLoading yearly WFPI map …")
    yearly_map = np.load(str(YEARLY_MAP), mmap_mode="r")

    sim_start = datetime(2020, 7, 12, 15, 0)
    print(f"Building burn map: sim_start={sim_start}, {MAX_ROUTING_DATA_STEPS} data steps …")
    bm_data = build_burn_map(yearly_map, sim_start, MAX_ROUTING_DATA_STEPS)  # (24, H, W)

    # Apply mask before pooling (to avoid contamination from non-CA cells)
    bm_data_masked = bm_data * mask

    rescaled_bm = pool_burnmap_mean(bm_data_masked, coverage_w)              # (24, 261, 161)
    rescaled_bm = np.repeat(rescaled_bm, operational_substeps, axis=0)       # (168, 261, 161)
    rescaled_bm = rescaled_bm / operational_substeps
    print(f"Burn map shape: {rescaled_bm.shape}")

    # ── Load routing log ──────────────────────────────────────────────────────
    print(f"\nLoading routing log for cluster {CLUSTER_FP} …")
    with open(str(ROUTING_LOG_PATH)) as f:
        rlog = json.load(f)

    if LOG_KEY not in rlog:
        raise KeyError(f"Log key '{LOG_KEY}' not found in routing log. "
                       f"Available: {list(rlog.keys())}")

    entry = rlog[LOG_KEY]
    initial_raw    = entry["initial_drone_locations"]   # [["charge", [x,y]], ...]
    actions_raw    = entry["actions_history"]            # [[["fly",[x,y]], ...], ...]

    print(f"  Initial locations: {initial_raw}")
    print(f"  Actions history: {len(actions_raw)} substeps, "
          f"{len(actions_raw[0])} drones")

    # Build drone_locations_history[t] = list of (x, y) in operational space
    # Frame 0: initial positions (before any action)
    initial_positions = [(int(loc[1][0]), int(loc[1][1])) for loc in initial_raw]

    drone_locations_history = [initial_positions]
    for step_actions in actions_raw:
        positions = [(int(act[1][0]), int(act[1][1])) for act in step_actions]
        drone_locations_history.append(positions)

    # Trim/pad to exactly match burn map length
    total_frames = rescaled_bm.shape[0]   # 168
    if len(drone_locations_history) < total_frames:
        # Pad with last known position
        last = drone_locations_history[-1]
        drone_locations_history += [last] * (total_frames - len(drone_locations_history))
    drone_locations_history = drone_locations_history[:total_frames]

    print(f"  drone_locations_history length: {len(drone_locations_history)}")

    # ── Load fire scenario ───────────────────────────────────────────────────
    print(f"\nLoading scenario: {SCENARIO_NPY.name} …")
    fire_scenario = load_scenario_npy(
        str(SCENARIO_NPY),
        grid_height=H,
        grid_width=W,
        num_timesteps=N_SCENARIO_DATA_STEPS,
    )
    print(f"  fire_scenario shape: {fire_scenario.shape}")

    # Print ignition details for reference
    raw_pt = np.load(str(SCENARIO_NPY), allow_pickle=True)
    fire_row, fire_col = int(raw_pt[0]), int(raw_pt[1])
    fire_start_step    = int(raw_pt[2])
    fire_opt_row, fire_opt_col = fire_row // coverage_w, fire_col // coverage_w
    print(f"  Ignition data-space: ({fire_row}, {fire_col}), "
          f"opt-space: ({fire_opt_row}, {fire_opt_col})")
    print(f"  Fire ignites at scenario step {fire_start_step} "
          f"→ video substep {fire_start_step * operational_substeps}")

    # Nearest charging station in the cluster
    print(f"  Cluster stations: (138,66) and (131,59)")
    print(f"  L∞ distance to (138,66): {max(abs(fire_opt_row-138), abs(fire_opt_col-66))}")
    print(f"  L∞ distance to (131,59): {max(abs(fire_opt_row-131), abs(fire_opt_col-59))}")

    # ── Load sensor placement ─────────────────────────────────────────────────
    print(f"\nLoading sensor placement …")
    with open(str(SENSOR_LOG_PATH)) as f:
        placement = json.load(f)

    ground_sensor_locs   = [tuple(x) for x in placement["ground_sensor_locations"]]
    charging_station_locs = [tuple(x) for x in placement["charging_station_locations"]]
    print(f"  Ground sensors: {len(ground_sensor_locs)}, "
          f"Charging stations: {len(charging_station_locs)}")

    # ── Generate video ────────────────────────────────────────────────────────
    os.chdir(str(PROJECT_ROOT))
    print(f"\nGenerating video '{OUTPUT_NAME}' …")
    displays.create_video_scenario_burnmap(
        burn_map=rescaled_bm,
        drone_locations_history=drone_locations_history,
        out_filename=OUTPUT_NAME,
        ground_sensor_locations=ground_sensor_locs,
        charging_stations_locations=charging_station_locs,
        frames_per_image=2,
        fire_scenario=fire_scenario,
        substeps_per_timestep=operational_substeps,
        mask=mask,
        coverage_width_cells=coverage_w,
        mask_pooling_mode="max",   # must match benchmark (SIMULATION_PARAMETERS)
        display_zones=True,
    )
    print(f"\nDone! Video saved to: display_{OUTPUT_NAME}/{OUTPUT_NAME}.mp4")


if __name__ == "__main__":
    main()
