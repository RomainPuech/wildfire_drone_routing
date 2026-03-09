#!/usr/bin/env python3
"""
Generate a zoomed video of drone movement for the missed discoverable fire:
THUMPER_DR__SHINGLETOWN_400597253 — cluster 53-43 (1 drone, 1 station)

Fire opt-space (53,44), station (53,43).  Round-trip cost = 2 ≤ battery(7).

Changes from v1:
- All inputs are in operational scale (no data-scale pooling inside the function).
- Burn map includes the fixed_reset effect: visited cells are zeroed for
  reset_time_periods × reevaluation_step = 2 × 7 = 14 substeps after each visit.
- Video zoomed to the cluster patrol area.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

from benchmark import compute_operational_substeps, pool_burnmap_mean, pool_mask
import displays

# ── Constants (must match benchmark) ────────────────────────────────────────
DATASET_DIR  = PROJECT_ROOT / "California2020Dataset"
YEARLY_MAP   = DATASET_DIR  / "static_risk_wfpi_yearly.npy"
MASK_PATH    = DATASET_DIR  / "mask.npy"
LOG_DIR      = DATASET_DIR  / "logs"

DRONE_SPEED_M_PER_MIN = 600
COVERAGE_RADIUS_M     = 2900
CELL_SIZE_M           = 1000
MAX_BATTERY_TIME_H    = 1
MAX_ROUTING_DATA_STEPS = 24
N_SCENARIO_DATA_STEPS  = 12

# ── Fire / routing specifics ─────────────────────────────────────────────────
FIRE_NAME    = "THUMPER_DR__SHINGLETOWN_400597253"
SCENARIO_NPY = DATASET_DIR / "scenarii" / f"{FIRE_NAME}_scenario1.npy"
CLUSTER_FP   = "53-43"
LOG_KEY      = "20200614_17"

ROUTING_LOG_PATH = LOG_DIR / f"routing_yearly_DroneRoutingTOPMasked_10OH_5RS_cluster_{CLUSTER_FP}.json"
SENSOR_LOG_PATH  = LOG_DIR / "sensor_alloc_GaussianAlloc_TOP_261x161_mean.json"

OUTPUT_NAME = "thumper_missed_fire"

ZOOM_PADDING = 2    # extra cells of visual margin around the round-trip reachable zone

# ── fixed_reset parameters (must match DroneRoutingTOPMasked) ────────────────
# Strategy.py line 3748: self.reevaluation_step = auto_params["max_battery_time"]
# NOT the combo param (5); it uses max_battery_substeps = 7.
# reset_time = 2 * max_battery_time = 14; reset_time_periods = 14 // 7 = 2.
REEVALUATION_STEP  = 7   # = max_battery_substeps (overrides combo reevaluation_step)
RESET_TIME_PERIODS = 2   # reset_time=14 / reevaluation_step=7


def frame_index(dt: datetime) -> int:
    doy  = dt.timetuple().tm_yday
    half = 0 if dt.hour < 10 else 1
    return 2 * (doy - 1) + half


def build_burn_map(yearly_map, sim_start, num_steps):
    frames = []
    for t in range(num_steps):
        dt = sim_start + timedelta(minutes=30 * t)
        frames.append(yearly_map[frame_index(dt)])
    return np.stack(frames)


def apply_fixed_reset(burn_map, drone_history, reevaluation_step, reset_time_periods):
    """
    Simulate fixed_reset: when a drone is at (ax, ay) at substep t, zero
    burn_map[t : t + time_left + (reset_time_periods-1)*reevaluation_step, ax, ay].

    This mirrors DroneRoutingTOPMasked's burnmap update logic exactly.
    """
    T, N, M = burn_map.shape
    bm = burn_map.copy().astype(float)
    for t, positions in enumerate(drone_history):
        for (ax, ay) in positions:
            if not (0 <= ax < N and 0 <= ay < M):
                continue
            time_left = reevaluation_step - t % reevaluation_step
            reset_end = min(
                t + time_left + (reset_time_periods - 1) * reevaluation_step,
                T,
            )
            bm[t:reset_end, ax, ay] = 0.0
    return bm


def main():
    # ── Derived geometry ─────────────────────────────────────────────────────
    operational_substeps = compute_operational_substeps(
        CELL_SIZE_M, DRONE_SPEED_M_PER_MIN, COVERAGE_RADIUS_M
    )
    coverage_w = round(2 * COVERAGE_RADIUS_M / CELL_SIZE_M)
    if coverage_w % 2 == 0:
        coverage_w -= 1

    mask_data = np.load(str(MASK_PATH))
    H, W = mask_data.shape
    rescaled_N = H // coverage_w   # 261
    rescaled_M = W // coverage_w   # 161
    rescaled_max_battery = MAX_BATTERY_TIME_H * operational_substeps  # 7

    print(f"coverage_w={coverage_w}, substeps={operational_substeps}")
    print(f"opt grid = {rescaled_N}×{rescaled_M}, max_battery={rescaled_max_battery}")

    # ── Build full operational burn map ──────────────────────────────────────
    print("Loading yearly WFPI map …")
    yearly_map = np.load(str(YEARLY_MAP), mmap_mode="r")
    sim_start  = datetime(2020, 6, 14, 17, 0)
    print(f"Building burn map: sim_start={sim_start} …")
    bm_data = build_burn_map(yearly_map, sim_start, MAX_ROUTING_DATA_STEPS)
    bm_data_masked = bm_data * mask_data
    rescaled_bm = pool_burnmap_mean(bm_data_masked, coverage_w)            # (24, 261, 161)
    rescaled_bm = np.repeat(rescaled_bm, operational_substeps, axis=0)     # (168, 261, 161)
    rescaled_bm = rescaled_bm / operational_substeps
    print(f"Burn map shape: {rescaled_bm.shape}")

    # ── Pre-compute operational-scale mask ────────────────────────────────────
    rescaled_mask = pool_mask(mask_data, coverage_w, mode="max")            # (261, 161)

    # ── Load routing log ──────────────────────────────────────────────────────
    print(f"Loading routing log …")
    with open(str(ROUTING_LOG_PATH)) as f:
        rlog = json.load(f)
    entry = rlog[LOG_KEY]
    initial_raw = entry["initial_drone_locations"]
    actions_raw = entry["actions_history"]

    initial_positions = [(int(loc[1][0]), int(loc[1][1])) for loc in initial_raw]
    drone_history_full = [initial_positions]
    for step in actions_raw:
        drone_history_full.append([(int(a[1][0]), int(a[1][1])) for a in step])
    total_frames = rescaled_bm.shape[0]
    drone_history_full = (drone_history_full + [drone_history_full[-1]] * total_frames)[:total_frames]

    # ── Apply fixed_reset to get dynamic burn map ─────────────────────────────
    print("Applying fixed_reset to burn map …")
    print(f"  reevaluation_step={REEVALUATION_STEP}, reset_time_periods={RESET_TIME_PERIODS}")
    rescaled_bm_reset = apply_fixed_reset(
        rescaled_bm, drone_history_full,
        reevaluation_step=REEVALUATION_STEP,
        reset_time_periods=RESET_TIME_PERIODS,
    )
    print("  Done.")

    # ── Fire scenario in operational scale ────────────────────────────────────
    raw_pt = np.load(str(SCENARIO_NPY), allow_pickle=True)
    fire_row_d, fire_col_d = int(raw_pt[0]), int(raw_pt[1])
    fire_start_step        = int(raw_pt[2])
    fire_opt = (fire_row_d // coverage_w, fire_col_d // coverage_w)
    print(f"Fire data=({fire_row_d},{fire_col_d}), opt={fire_opt}, start_step={fire_start_step}")

    # Build (N_SCENARIO_DATA_STEPS, rescaled_N, rescaled_M) operational-scale fire scenario
    fire_ops_full = np.zeros((N_SCENARIO_DATA_STEPS, rescaled_N, rescaled_M), dtype=np.float32)
    fire_ops_full[fire_start_step:, fire_opt[0], fire_opt[1]] = 1.0

    # ── Load sensor placement ─────────────────────────────────────────────────
    with open(str(SENSOR_LOG_PATH)) as f:
        placement = json.load(f)
    all_ground   = [tuple(x) for x in placement["ground_sensor_locations"]]
    all_charging = [tuple(x) for x in placement["charging_station_locations"]]

    # ── Compute zoom bounds ───────────────────────────────────────────────────
    # Round-trip reachable zone from a single station = L∞ ≤ battery//2 = 3.
    station_r, station_c = 53, 43
    zoom_radius = rescaled_max_battery // 2   # 3 cells (round-trip max range)

    row_min = max(0, station_r - zoom_radius - ZOOM_PADDING)
    row_max = min(rescaled_N,  station_r + zoom_radius + ZOOM_PADDING + 1)
    col_min = max(0, station_c - zoom_radius - ZOOM_PADDING)
    col_max = min(rescaled_M,  station_c + zoom_radius + ZOOM_PADDING + 1)
    print(f"Zoom: rows [{row_min},{row_max}), cols [{col_min},{col_max}) → "
          f"{row_max-row_min}×{col_max-col_min} cells")

    # ── Crop everything to zoom region ────────────────────────────────────────
    bm_crop       = rescaled_bm_reset[:, row_min:row_max, col_min:col_max]
    mask_crop     = rescaled_mask[row_min:row_max, col_min:col_max]    # already op-scale
    fire_crop     = fire_ops_full[:, row_min:row_max, col_min:col_max]  # op-scale

    def in_crop(r, c):
        return row_min <= r < row_max and col_min <= c < col_max

    def shift(r, c):
        return (r - row_min, c - col_min)

    drone_history_crop = [[shift(r, c) for (r, c) in frame] for frame in drone_history_full]
    ground_crop        = [shift(r, c) for (r, c) in all_ground   if in_crop(r, c)]
    charging_crop      = [shift(r, c) for (r, c) in all_charging if in_crop(r, c)]

    fire_in_crop = shift(*fire_opt)
    print(f"Fire in crop: {fire_in_crop}, station in crop: {shift(station_r, station_c)}")
    print(f"Ground sensors in crop: {len(ground_crop)}, "
          f"Charging stations in crop: {len(charging_crop)}")

    # ── Generate video ────────────────────────────────────────────────────────
    # Pass coverage_width_cells=1 so mask and fire_scenario are used as-is
    # (they are already in operational scale — no internal pooling needed).
    os.chdir(str(PROJECT_ROOT))
    print(f"\nGenerating video '{OUTPUT_NAME}' …")
    displays.create_video_scenario_burnmap(
        burn_map=bm_crop,
        drone_locations_history=drone_history_crop,
        out_filename=OUTPUT_NAME,
        ground_sensor_locations=ground_crop,
        charging_stations_locations=charging_crop,
        frames_per_image=2,
        fire_scenario=fire_crop,
        substeps_per_timestep=operational_substeps,
        mask=mask_crop,
        coverage_width_cells=1,     # already operational scale → identity pooling
        mask_pooling_mode="max",
        display_zones=True,
    )
    print(f"\nDone!  display_{OUTPUT_NAME}/{OUTPUT_NAME}.mp4")


if __name__ == "__main__":
    main()
