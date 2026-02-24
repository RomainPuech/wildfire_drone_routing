#!/usr/bin/env python3
"""
Benchmark on California 2020 Dataset — Time-Aware Yearly WFPI Burn Map

For each scenario:
  1. Discovery date+time and offset are read from config_california_2020.json
     (run augment_config_with_times.py first to populate those fields).
  2. sim_start = round(discovery_time - offset × 30 min, nearest hour)
  3. A 12-step burn map is built from static_risk_wfpi_yearly.npy by picking
     the correct frame for each 30-min timestep (before/after 10 am rule).
  4. Routing is cached per (date, start_hour) in a single JSON log file.
     Scenarios sharing the same (date, start_hour) reuse the cached routing.
  5. Simulation is replayed from the cached routing and fire-detection metrics
     are collected.

Run from the project root:
    python -u run_benchmark_california2020_yearly.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

print("Importing modules...", flush=True)
from Drone import Drone
from dataset import load_scenario_npy
from benchmark import (
    compute_operational_substeps,
    detect_fire_within_coverage,
    operational_space_to_dataspace_coordinates,
    pool_burnmap_mean,
    pool_mask,
)
import wrappers
print("Imports done.\n", flush=True)


# ── Dataset paths ──────────────────────────────────────────────────────────────
DATASET_DIR    = PROJECT_ROOT / "California2020Dataset"
YEARLY_MAP     = DATASET_DIR / "static_risk_wfpi_yearly.npy"
AVG_MAP        = DATASET_DIR / "static_risk_wfpi_avg.npy"
MASK_PATH      = DATASET_DIR / "mask.npy"
CONFIG_PATH    = DATASET_DIR / "config_california_2020.json"
SCENARII_DIR   = DATASET_DIR / "scenarii"
LOG_DIR        = DATASET_DIR / "logs"
TMP_DIR        = PROJECT_ROOT / "tmp_burnmaps"

# Maximum routing steps computed per (date, start_hour) entry:
# offset ≤ 12, scenario = 12 data steps → worst case = 24 data steps.
MAX_ROUTING_DATA_STEPS = 24
N_SCENARIO_DATA_STEPS  = 12   # each scenario has 12 × 30-min timesteps = 6 hours


# ── Simulation parameters ──────────────────────────────────────────────────────
# cell_size_m should match the WFPI data resolution (~1 km for the full
# California 2020 dataset at shape 1309×805).  Verify against the first
# WFPI file if unsure.
SIMULATION_PARAMETERS = {
    "max_battery_distance": -1,
    "max_battery_time":      1,        # hours
    "n_drones":              2,
    "n_ground_stations":     8,
    "n_charging_stations":   2,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m":     2900,
    "cell_size_m":           1000,     # ~1 km WFPI resolution
    "mask_pooling_mode":     "max",
}


# ── Strategy combinations ──────────────────────────────────────────────────────
STRATEGY_COMBINATIONS = [
    {
        "name":   "Gaussian_TOP",
        "sensor": wrappers.SensorPlacementMaxCoverageGaussianTimeMasked,
        "drone":  wrappers.DroneRoutingTOPMaskedLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10},
    },
    {
        "name":   "Random_TOP",
        "sensor": wrappers.RandomSensorPlacementStrategyLogged,
        "drone":  wrappers.DroneRoutingTOPMaskedLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10},
    },
]


# ── Frame-index helpers ────────────────────────────────────────────────────────

def round_to_nearest_hour(dt: datetime) -> datetime:
    """Round a datetime to the nearest whole hour."""
    if dt.minute >= 30:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)


def frame_index(dt: datetime) -> int:
    """Index into the (732, H, W) yearly WFPI map for a given datetime.

    Frame layout:
      2*(day_of_year-1) + 0  →  before 10 am  (D2 forecast)
      2*(day_of_year-1) + 1  →  10 am or later (D1 forecast)
    """
    doy  = dt.timetuple().tm_yday          # 1–366
    half = 0 if dt.hour < 10 else 1
    return 2 * (doy - 1) + half


def build_burn_map(yearly_map: np.ndarray, sim_start: datetime,
                   num_steps: int) -> np.ndarray:
    """Return a (num_steps, H, W) array of WFPI frames.

    For each 30-min timestep starting at sim_start, pick the correct
    before/after-10-am frame from the yearly map.
    """
    frames = []
    for t in range(num_steps):
        dt = sim_start + timedelta(minutes=30 * t)
        frames.append(yearly_map[frame_index(dt)])
    return np.stack(frames)   # (num_steps, H, W)


# ── Routing log ────────────────────────────────────────────────────────────────

class RoutingLog:
    """Single JSON file keyed by 'YYYYMMDD_HH' routing-solution entries.

    Each entry:
      {
        "initial_drone_locations": [["charge", [x, y]], ...],
        "actions_history":         [[[type, param], ...], ...]
      }
    actions_history length = MAX_ROUTING_DATA_STEPS * operational_substeps.
    """

    def __init__(self, path: str):
        self.path = path
        if Path(path).exists():
            with open(path) as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def has(self, key: str, min_steps: int) -> bool:
        entry = self.data.get(key)
        return (entry is not None
                and len(entry.get("actions_history", [])) >= min_steps)

    def get(self, key: str) -> dict:
        return self.data[key]

    def put(self, key: str, initial_locs: list, actions_history: list):
        self.data[key] = {
            "initial_drone_locations": initial_locs,
            "actions_history":         actions_history,
        }
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Action normalisation ───────────────────────────────────────────────────────

def _normalise_initial(raw) -> list:
    """Bring any return variant to  [(state, (x, y)), ...]."""
    if isinstance(raw, list):
        if raw and isinstance(raw[0], tuple) and isinstance(raw[0][0], str):
            return [(st, (int(x), int(y))) for st, (x, y) in raw]
        return [("charge", (int(x), int(y))) for x, y in raw]
    if isinstance(raw, tuple) and len(raw) == 2:
        positions, states = raw
        return [(st, (int(x), int(y))) for (x, y), st in zip(positions, states)]
    raise ValueError(f"Unexpected initial-location format: {type(raw)}")


def _normalise_actions(acts) -> list:
    return [[typ, None if param is None else list(param)] for typ, param in acts]


def _denormalise_actions(stored) -> list:
    return [(typ, None if param is None else tuple(param)) for typ, param in stored]


# ── Routing computation ────────────────────────────────────────────────────────

def compute_routing(routing_strategy_cls,
                    rescaled_auto_params: dict,
                    rescaled_custom_params: dict,
                    n_data_steps: int,
                    operational_substeps: int) -> tuple[list, list]:
    """Run routing strategy for n_data_steps × substeps and collect all actions.

    Returns (initial_locs_normalised, actions_history) where
    actions_history has n_data_steps * operational_substeps entries.
    """
    strategy = routing_strategy_cls(rescaled_auto_params, rescaled_custom_params)

    rescaled_N            = rescaled_auto_params["N"]
    rescaled_M            = rescaled_auto_params["M"]
    n_drones              = rescaled_auto_params["n_drones"]
    rescaled_max_battery  = rescaled_auto_params["max_battery_time"]

    initial_raw  = strategy.get_initial_drone_locations()
    initial_norm = _normalise_initial(initial_raw)

    drone_locs      = [pos for _, pos in initial_norm]
    drone_batteries = [rescaled_max_battery] * n_drones
    drone_states    = [st  for st, _  in initial_norm]

    actions_history = []
    t = 0

    for _ in range(n_data_steps):
        for _ in range(operational_substeps):
            step_params = {
                "drone_locations": drone_locs,
                "drone_batteries": drone_batteries,
                "drone_states":    drone_states,
                "t":               t,
            }
            acts = strategy.next_actions(step_params, {})
            actions_history.append(_normalise_actions(acts))

            # Update opt-scale drone state for the next call
            new_locs = []
            for i, (act_type, param) in enumerate(acts):
                if act_type in ("charge", "fly"):
                    new_locs.append(tuple(param))
                    if act_type == "charge":
                        drone_batteries[i] = rescaled_max_battery
                    else:
                        drone_batteries[i] = max(0, drone_batteries[i] - 1)
                elif act_type == "move":
                    ox, oy = drone_locs[i]
                    nx = max(0, min(rescaled_N - 1, ox + param[0]))
                    ny = max(0, min(rescaled_M - 1, oy + param[1]))
                    new_locs.append((nx, ny))
                    drone_batteries[i] = max(0, drone_batteries[i] - 1)
                else:
                    new_locs.append(drone_locs[i])
            drone_locs = new_locs

        t += 1

    return initial_norm, actions_history


# ── Simulation runner ──────────────────────────────────────────────────────────

def run_simulation(scenario: np.ndarray,
                   starting_time: int,
                   routing_entry: dict,
                   ground_locs_data: list,
                   charging_locs_data: list,
                   N: int, M: int,
                   coverage_width_cells: int,
                   operational_substeps: int,
                   max_battery_distance,
                   max_battery_time) -> dict:
    """Replay cached routing on a fire scenario and return detection metrics.

    Parameters
    ----------
    scenario        : (T, H, W) fire evolution array
    starting_time   : number of pre-fire steps (= offset)
    routing_entry   : dict with keys 'initial_drone_locations' and
                      'actions_history' (from RoutingLog)
    ground_locs_data, charging_locs_data : sensor positions at data scale
    N, M            : grid dimensions at data scale
    """
    initial_norm  = routing_entry["initial_drone_locations"]
    actions_log   = routing_entry["actions_history"]
    action_ptr    = 0

    rescaled_max_battery_time = max_battery_time * operational_substeps

    # Initialise drones at data scale
    drones = [
        Drone(
            x * coverage_width_cells + coverage_width_cells // 2,
            y * coverage_width_cells + coverage_width_cells // 2,
            state,
            charging_locs_data,
            N, M,
            max_battery_distance, max_battery_time,
            max_battery_distance - 1 * (state == "fly"),
            max_battery_time     - 1 * (state == "fly"),
        )
        for state, (x, y) in initial_norm
    ]

    drone_locs_data = [drone.get_position() for drone in drones]
    drone_locs_opt  = [(x, y) for _, (x, y) in initial_norm]
    drone_batteries_opt = [rescaled_max_battery_time] * len(drones)

    # Sensor index arrays for vectorised detection
    if ground_locs_data:
        rows_g = [x for x, _ in ground_locs_data]
        cols_g = [y for _, y in ground_locs_data]
    else:
        rows_g = cols_g = []

    if charging_locs_data:
        rows_c = [x for x, _ in charging_locs_data]
        cols_c = [y for _, y in charging_locs_data]
    else:
        rows_c = cols_c = []

    fire_detected        = False
    device               = "undetected"
    fire_size_cells      = 0
    fire_size_percentage = 0.0
    total_distance       = 0
    visited_cells        = set(drone_locs_data)
    t_found              = 0

    max_time_steps = N_SCENARIO_DATA_STEPS + starting_time

    for time_step in range(-starting_time, min(max_time_steps, len(scenario))):

        # ── Fixed-sensor fire detection ───────────────────────────────────────
        if time_step >= 0:
            grid = scenario[time_step]

            if rows_g and (grid[rows_g, cols_g] == 1).any():
                fire_detected        = True
                device               = "ground sensor"
                fire_size_cells      = int(np.sum(grid > 0.5))
                fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                break

            if rows_c and (grid[rows_c, cols_c] == 1).any():
                fire_detected        = True
                device               = "charging station"
                fire_size_cells      = int(np.sum(grid > 0.5))
                fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                break

        # ── Routing substeps ──────────────────────────────────────────────────
        for _ in range(operational_substeps):
            if action_ptr < len(actions_log):
                acts = _denormalise_actions(actions_log[action_ptr])
            else:
                # Log shorter than expected — keep drones in place
                acts = [("charge", drone_locs_opt[i]) for i in range(len(drones))]
            action_ptr += 1

            # Compute opt-scale new positions (needed for next call consistency)
            new_locs_opt = []
            for i, (act_type, param) in enumerate(acts):
                if act_type in ("charge", "fly"):
                    new_locs_opt.append(tuple(param))
                    if act_type == "charge":
                        drone_batteries_opt[i] = rescaled_max_battery_time
                    else:
                        drone_batteries_opt[i] = max(0, drone_batteries_opt[i] - 1)
                elif act_type == "move":
                    ox, oy = drone_locs_opt[i]
                    nx = max(0, min(N // coverage_width_cells - 1, ox + param[0]))
                    ny = max(0, min(M // coverage_width_cells - 1, oy + param[1]))
                    new_locs_opt.append((nx, ny))
                    drone_batteries_opt[i] = max(0, drone_batteries_opt[i] - 1)
                else:
                    new_locs_opt.append(drone_locs_opt[i])

            drone_locs_opt = new_locs_opt

            # Convert to data scale and move Drone objects
            actions_data = []
            for act_type, param in acts:
                if act_type == "fly":
                    converted = operational_space_to_dataspace_coordinates(
                        param,
                        coverage=SIMULATION_PARAMETERS["coverage_radius_m"],
                        datacell_size_m=SIMULATION_PARAMETERS["cell_size_m"],
                    )
                    actions_data.append((act_type, converted))
                elif act_type == "move":
                    actions_data.append(
                        (act_type,
                         (coverage_width_cells * param[0],
                          coverage_width_cells * param[1]))
                    )
                else:
                    converted = operational_space_to_dataspace_coordinates(
                        param,
                        coverage=SIMULATION_PARAMETERS["coverage_radius_m"],
                        datacell_size_m=SIMULATION_PARAMETERS["cell_size_m"],
                    )
                    actions_data.append((act_type, converted))

            for i, (drone, action) in enumerate(zip(drones, actions_data)):
                old_x, old_y             = drone_locs_data[i]
                new_x, new_y, *_         = drone.route(action)
                drone_locs_data[i]       = (new_x, new_y)
                total_distance          += abs(new_x - old_x) + abs(new_y - old_y)
                visited_cells.add((new_x, new_y))

            # ── Drone fire detection ──────────────────────────────────────────
            if time_step >= 0:
                for pos in drone_locs_data:
                    if detect_fire_within_coverage(grid, pos, coverage_width_cells):
                        fire_detected        = True
                        device               = "drone"
                        fire_size_cells      = int(np.sum(grid > 0.5))
                        fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                        break
            if fire_detected:
                break

        if fire_detected:
            break

        t_found += 1

    delta_t = t_found - starting_time
    if device == "undetected":
        delta_t     = -1
        final_grid  = scenario[-1]
        fire_size_cells      = int(np.sum(final_grid > 0.5))
        fire_size_percentage = fire_size_cells / (final_grid.shape[0] * final_grid.shape[1]) * 100

    return {
        "delta_t":                 delta_t,
        "device":                  device,
        "fire_size_cells":         fire_size_cells,
        "fire_size_percentage":    fire_size_percentage,
        "total_distance_traveled": total_distance,
        "percentage_map_explored": len(visited_cells) / (N * M) * 100,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    # ── Static resources (loaded once) ────────────────────────────────────────
    print("Loading yearly WFPI map (memory-mapped) ...", flush=True)
    yearly_map = np.load(str(YEARLY_MAP), mmap_mode="r")   # (732, H, W)

    print("Loading avg WFPI map ...", flush=True)
    avg_map = np.load(str(AVG_MAP))                        # (1, H, W)

    print("Loading mask ...", flush=True)
    mask = np.load(str(MASK_PATH))                         # (H, W)
    H, W = mask.shape

    print("Loading config ...", flush=True)
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # ── Rescaling parameters (fixed for all scenarios) ────────────────────────
    cell_size_m     = SIMULATION_PARAMETERS["cell_size_m"]
    speed           = SIMULATION_PARAMETERS["drone_speed_m_per_min"]
    coverage_r_m    = SIMULATION_PARAMETERS["coverage_radius_m"]

    operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
    coverage_w = round(coverage_r_m * 2 / cell_size_m)
    if coverage_w % 2 == 0:
        coverage_w -= 1                         # keep odd

    rescaled_N = H // coverage_w
    rescaled_M = W // coverage_w
    rescaled_max_battery = SIMULATION_PARAMETERS["max_battery_time"] * operational_substeps

    print(
        f"Rescaling: coverage_width={coverage_w} cells, "
        f"opt grid={rescaled_N}×{rescaled_M}, "
        f"substeps={operational_substeps}",
        flush=True,
    )

    # ── Save rescaled avg map and mask (used for sensor placement) ────────────
    suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"

    rescaled_avg = pool_burnmap_mean(avg_map, coverage_w)
    rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
    rescaled_avg_path = str(AVG_MAP).replace(".npy", suffix)
    np.save(rescaled_avg_path, rescaled_avg)

    rescaled_mask = pool_mask(
        mask, coverage_w, mode=SIMULATION_PARAMETERS["mask_pooling_mode"]
    )
    rescaled_mask_path = str(MASK_PATH).replace(".npy", suffix)
    np.save(rescaled_mask_path, rescaled_mask)

    # ── Base parameter dicts ──────────────────────────────────────────────────
    base_auto = {
        "N":                    H,
        "M":                    W,
        "max_battery_distance": SIMULATION_PARAMETERS["max_battery_distance"],
        "max_battery_time":     SIMULATION_PARAMETERS["max_battery_time"],
        "n_drones":             SIMULATION_PARAMETERS["n_drones"],
        "n_ground_stations":    SIMULATION_PARAMETERS["n_ground_stations"],
        "n_charging_stations":  SIMULATION_PARAMETERS["n_charging_stations"],
        "speed_m_per_min":      speed,
        "coverage_radius_m":    coverage_r_m,
        "cell_size_m":          cell_size_m,
        "mask_filename":        rescaled_mask_path,
    }

    base_rescaled_auto = {
        **base_auto,
        "N":                rescaled_N,
        "M":                rescaled_M,
        "max_battery_time": rescaled_max_battery,
    }

    base_custom = {
        "mask_filename":      rescaled_mask_path,
        "recompute_logfile":  False,
        "recompute_kernel":   False,
        "use_linf_cost":      True,
        "regularization_param": 1e5,
    }

    # ── Filter valid scenarios ────────────────────────────────────────────────
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf for sf in all_scenario_files
        if all(
            f"{key}_{sf.stem.replace('_scenario1', '')}" in config
            for key in ("date", "time", "offset")
        )
    ]
    print(
        f"\nScenarios with date+time info: {len(valid_scenarios)}/{len(all_scenario_files)}",
        flush=True,
    )

    all_results = []

    # ── Strategy loop ─────────────────────────────────────────────────────────
    for combo in STRATEGY_COMBINATIONS:
        combo_name = combo["name"]
        SensorCls  = combo["sensor"]
        RoutingCls = combo["drone"]
        combo_custom = {
            **base_custom,
            "reevaluation_step":    combo["params"]["reevaluation_step"],
            "optimization_horizon": combo["params"]["optimization_horizon"],
        }

        print(f"\n{'='*70}", flush=True)
        print(f"  STRATEGY: {combo_name}", flush=True)
        print(f"{'='*70}\n", flush=True)

        # ── Sensor placement (once per strategy) ──────────────────────────────
        print("  Computing sensor placement ...", flush=True)
        sensor_custom = {**combo_custom, "burnmap_filename": rescaled_avg_path}
        sensor_strat  = SensorCls(base_rescaled_auto, sensor_custom)
        ground_locs_opt, charging_locs_opt = sensor_strat.get_locations()

        ground_locs_data = [
            (x * coverage_w + coverage_w // 2, y * coverage_w + coverage_w // 2)
            for x, y in ground_locs_opt
        ]
        charging_locs_data = [
            (x * coverage_w + coverage_w // 2, y * coverage_w + coverage_w // 2)
            for x, y in charging_locs_opt
        ]
        print(
            f"  Sensor placement: {len(ground_locs_opt)} ground, "
            f"{len(charging_locs_opt)} charging",
            flush=True,
        )

        # ── Routing log (one file per strategy combination) ───────────────────
        base_routing_cls = wrappers._deep_unwrap(RoutingCls).__name__
        layout_fp  = "_".join(f"{x}-{y}" for x, y in sorted(charging_locs_opt))
        oh         = combo["params"]["optimization_horizon"]
        rs         = combo["params"]["reevaluation_step"]
        log_path   = str(
            LOG_DIR / f"routing_yearly_{base_routing_cls}_{oh}OH_{rs}RS_{layout_fp}.json"
        )
        routing_log = RoutingLog(log_path)
        print(f"  Routing log: {Path(log_path).name}", flush=True)

        rescaled_auto_with_locs = {
            **base_rescaled_auto,
            "ground_sensor_locations":   ground_locs_opt,
            "charging_stations_locations": charging_locs_opt,
        }

        # ── Scenario loop ─────────────────────────────────────────────────────
        n_cached  = 0
        n_computed = 0

        for sf in valid_scenarios:
            name     = sf.stem.replace("_scenario1", "")
            date_str = config[f"date_{name}"]   # "YYYYMMDD"
            time_str = config[f"time_{name}"]   # "HHMM"
            offset   = config[f"offset_{name}"]

            # Parse discovery datetime and compute sim_start
            discovery_dt = datetime(
                int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]),
                int(time_str[:2]), int(time_str[2:]),
            )
            sim_start = round_to_nearest_hour(
                discovery_dt - timedelta(minutes=30 * offset)
            )
            log_key   = f"{sim_start.strftime('%Y%m%d')}_{sim_start.hour:02d}"

            total_substeps_needed = (offset + N_SCENARIO_DATA_STEPS) * operational_substeps

            # ── Compute routing if not yet cached ─────────────────────────────
            if not routing_log.has(log_key, total_substeps_needed):
                # Build and rescale dynamic burn map
                bm = build_burn_map(yearly_map, sim_start, MAX_ROUTING_DATA_STEPS)
                # bm shape: (MAX_ROUTING_DATA_STEPS, H, W)
                rescaled_bm = pool_burnmap_mean(bm, coverage_w)
                rescaled_bm = (
                    np.repeat(rescaled_bm, operational_substeps, axis=0)
                    / operational_substeps
                )

                tmp_path = str(TMP_DIR / f"yearly_{log_key}.npy")
                np.save(tmp_path, rescaled_bm)

                routing_custom = {**combo_custom, "burnmap_filename": tmp_path}
                initial_norm, actions_hist = compute_routing(
                    RoutingCls,
                    rescaled_auto_with_locs,
                    routing_custom,
                    MAX_ROUTING_DATA_STEPS,
                    operational_substeps,
                )
                routing_log.put(log_key, initial_norm, actions_hist)
                n_computed += 1
                print(
                    f"  [{log_key}] routing computed "
                    f"({len(actions_hist)} substeps)",
                    flush=True,
                )
            else:
                n_cached += 1

            # ── Load fire scenario and run simulation ─────────────────────────
            scenario = load_scenario_npy(
                str(sf), grid_height=H, grid_width=W,
                num_timesteps=N_SCENARIO_DATA_STEPS,
            )

            results = run_simulation(
                scenario          = scenario,
                starting_time     = offset,
                routing_entry     = routing_log.get(log_key),
                ground_locs_data  = ground_locs_data,
                charging_locs_data= charging_locs_data,
                N=H, M=W,
                coverage_width_cells  = coverage_w,
                operational_substeps  = operational_substeps,
                max_battery_distance  = SIMULATION_PARAMETERS["max_battery_distance"],
                max_battery_time      = SIMULATION_PARAMETERS["max_battery_time"],
            )

            all_results.append({
                "strategy_combo": combo_name,
                "scenario_name":  name,
                "date":           date_str,
                "sim_start_hour": sim_start.hour,
                "log_key":        log_key,
                "offset":         offset,
                **results,
            })

        print(
            f"\n  Done: {n_computed} routing entries computed, "
            f"{n_cached} replayed from cache",
            flush=True,
        )

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("  RESULTS", flush=True)
    print("=" * 70, flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = f"benchmark_results_yearly_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}", flush=True)

        # Summary per strategy
        for combo_name, gdf in df.groupby("strategy_combo"):
            detected   = (gdf["delta_t"] != -1).sum()
            n          = len(gdf)
            det_rate   = detected / n * 100
            mean_dt    = gdf.loc[gdf["delta_t"] != -1, "delta_t"].mean()
            print(
                f"  {combo_name}: "
                f"detection rate={det_rate:.1f}%  "
                f"mean delta_t={mean_dt:.2f} (detected only)  "
                f"n={n}",
                flush=True,
            )
    else:
        print("No results collected.", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
