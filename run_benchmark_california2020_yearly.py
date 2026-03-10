#!/usr/bin/env python3
"""
Benchmark on California 2020 Dataset — Time-Aware Yearly WFPI Burn Map

Pipeline
--------
1. **Setup (single-threaded):** load static resources, precompute rescaled
   mask and avg burn map.  These are written once before any parallel work.
2. Sensor/charging placement with drone allocation (once per strategy, cached).
3. Drone clusters: connected components of charging stations whose inter-station
   L∞ distance in opt-space is ≤ max_battery_substeps.  Each cluster stores
   its stations and the number of drones allocated to it.
4. **Pre-scan (single-threaded):** for every benchmark scenario, resolve the
   target cluster and rounded sim-start hour (log_key).
5. **Wave assignment:** scenarios that share the same (cluster, log_key) are
   placed in different waves so they never run concurrently.  All scenarios
   within a wave can safely run in parallel.
6. **Parallel execution:** for each wave, a ``ProcessPoolExecutor`` (spawn)
   dispatches workers that compute routing (if not cached) and replay
   simulation.  Each worker process has its own Julia runtime.
   Process safety is ensured by:
   - Each (cluster, log_key) pair has its own routing-log JSON file, so
     no two workers ever write the same file.
   - Temp burn-map files include a UUID so no two workers write the same path.
   - Rescaled mask/avg files are precomputed in step 1 and only read by workers.
   - The yearly WFPI map is loaded independently (memory-mapped) per worker.
7. Routing is cached per (cluster_fingerprint, YYYYMMDD_HH) in one JSON file
   per pair.  Scenarios that share the same cluster AND the same rounded
   sim-start hour reuse the cached routing.
8. Simulation is replayed from the cached routing.

Prerequisites
-------------
  python code/dataset_creation/nature_dataset_creation/augment_config_with_times.py

Run from the project root:
  python -u run_benchmark_california2020_yearly.py --budget 20   # 20M (default)
  python -u run_benchmark_california2020_yearly.py --budget 100  # 100M
  python -u run_benchmark_california2020_yearly.py --budget 500  # 500M

  python -u run_benchmark_california2020_yearly.py --sensor-only --budget 100  # precompute sensor placement only
"""

import sys
import os
import json
import uuid
import multiprocessing
from multiprocessing import get_context
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

print("Importing modules...", flush=True)
from Drone import Drone
from dataset import load_scenario_npy
from Strategy import (
    SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation,
    SensorPlacementMaxCoverageGaussianTimeMaskedBudget,
)
from benchmark import (
    compute_operational_substeps,
    detect_fire_within_coverage,
    operational_space_to_dataspace_coordinates,
    pool_burnmap_mean,
    pool_burnmap_proba_at_least_one,
    pool_mask,
)
import wrappers
print("Imports done.\n", flush=True)


# ── Dataset paths ──────────────────────────────────────────────────────────────
DATASET_DIR  = PROJECT_ROOT / "California2020Dataset"
YEARLY_MAP   = DATASET_DIR / "static_risk_wfpi_yearly.npy"
AVG_MAP      = DATASET_DIR / "static_risk_wfpi_avg.npy"
MASK_PATH    = DATASET_DIR / "mask.npy"
CONFIG_PATH  = DATASET_DIR / "config_california_2020.json"
SCENARII_DIR = DATASET_DIR / "scenarii"
LOG_DIR      = DATASET_DIR / "logs"
TMP_DIR      = PROJECT_ROOT / "tmp_burnmaps"

MAX_ROUTING_DATA_STEPS = 24   # offset ≤ 12 + 12 scenario steps
N_SCENARIO_DATA_STEPS  = 6    # 6 × 30 min = 3 h


# ── Simulation parameters ──────────────────────────────────────────────────────
SIMULATION_PARAMETERS = {
    "max_battery_distance": -1,
    "max_battery_time":      1,        # hours
    "n_drones":              50,
    "n_ground_stations":     100,
    "n_charging_stations":   50,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m":     2900,
    "cell_size_m":           1000,     # ~1 km WFPI resolution
    "transmission_range":    50000,
    "mask_pooling_mode":     "max",
}

# Budget: 20M total, 50/50 split between sensors and stations+drones.
#   100 sensors × 100k = 10M
#   50 stations × 150k + 50 drones × 50k = 7.5M + 2.5M = 10M
BUDGET_TOTAL      = 20_000_000
COST_SENSOR       = 100_000
COST_DRONE        = 50_000
COST_STATION      = 150_000

# How to pool the avg WFPI map to the operational scale for sensor placement.
#   "mean"  — block mean (average WFPI over the 5×5 data cells)
#   "proba" — 1 – ∏(1–p_i) treating WFPI/255 as per-cell fire probability
SENSOR_POOLING = "mean"

# Random subset of fires to benchmark
BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED           = 42

# Parallelism
MAX_WORKERS = os.cpu_count() or 4


# ── Strategy combo from budget ──────────────────────────────────────────────────
# Budget is selected via CLI (--budget 20 | 100 | 500). One combo per run.

def build_strategy_combinations(budget_millions: float):
    """Single combo: Gaussian budget placement + TOP masked, routing capped at 1 min."""
    return [
        {
            "name":   f"GaussianBudget{int(budget_millions)}M_TOP",
            "sensor": SensorPlacementMaxCoverageGaussianTimeMaskedBudget,
            "drone":  "DroneRoutingTOPMaskedLogged",
            "params": {
                "reevaluation_step": 5,
                "optimization_horizon": 10,
                "time_limit_seconds": 60,
            },
            "sensor_params": {
                "budget_millions": float(budget_millions),
                "cost_sensor":     COST_SENSOR  / 1_000_000,
                "cost_station":    COST_STATION / 1_000_000,
                "cost_drone":      COST_DRONE   / 1_000_000,
            },
        },
    ]


# ── Frame-index helpers ────────────────────────────────────────────────────────

def round_to_nearest_hour(dt: datetime) -> datetime:
    if dt.minute >= 30:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)


def frame_index(dt: datetime) -> int:
    doy  = dt.timetuple().tm_yday
    half = 0 if dt.hour < 10 else 1
    return 2 * (doy - 1) + half


def build_burn_map(yearly_map: np.ndarray, sim_start: datetime,
                   num_steps: int) -> np.ndarray:
    """Return (num_steps, H, W) array of WFPI frames starting at sim_start."""
    frames = []
    for t in range(num_steps):
        dt = sim_start + timedelta(minutes=30 * t)
        frames.append(yearly_map[frame_index(dt)])
    return np.stack(frames)


# ── Cluster computation ────────────────────────────────────────────────────────

def compute_clusters(charging_locs_opt: list,
                     drones_per_station: list,
                     max_battery_substeps: int) -> list[dict]:
    """Build connected components of charging stations.

    Two stations are in the same cluster if their L∞ distance in opt-space is
    ≤ max_battery_substeps (a drone can fly directly between them on one charge).

    Returns a list of cluster dicts:
      {
        "stations_opt":  [(x, y), ...],   # charging station positions (opt)
        "n_drones":      int,             # total drones allocated to this cluster
        "fingerprint":   str,             # stable string key for log naming
      }
    """
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
            if max(abs(xi - xj), abs(yi - yj)) <= max_battery_substeps:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    for indices in groups.values():
        stations = [charging_locs_opt[i] for i in indices]
        n_drones = sum(drones_per_station[i] for i in indices)
        fp = "_".join(f"{x}-{y}" for x, y in sorted(stations))
        clusters.append({
            "stations_opt": stations,
            "n_drones":     n_drones,
            "fingerprint":  fp,
        })
    return clusters


def fire_cluster(fire_opt: tuple, clusters: list,
                 max_battery_substeps: int) -> dict | None:
    """Return the cluster whose reachable zone contains fire_opt, or None.

    A fire is considered reachable if it lies within L∞ distance
    floor(max_battery_substeps / 2) of at least one station in a cluster.
    This is the one-way reach: drones must fly to the fire *and* return to a
    station on a single charge.
    """
    one_way_reach = max_battery_substeps // 2
    fr, fc = fire_opt
    for cluster in clusters:
        for sx, sy in cluster["stations_opt"]:
            if max(abs(fr - sx), abs(fc - sy)) <= one_way_reach:
                return cluster
    return None


# ── Sensor placement cache (manual, since no loggable wrapper exists) ──────────

def load_or_compute_sensor_placement(strategy_cls, rescaled_auto_params: dict,
                                     rescaled_custom_params: dict,
                                     log_path: str) -> tuple[list, list, list]:
    """Return (ground_locs_opt, charging_locs_opt, drones_per_station).

    Loads from log_path if it exists, otherwise runs the strategy and saves.
    """
    if Path(log_path).exists():
        print(f"  [sensor] Loading cached placement from {Path(log_path).name}",
              flush=True)
        with open(log_path) as f:
            d = json.load(f)
        return (
            [tuple(x) for x in d["ground_sensor_locations"]],
            [tuple(x) for x in d["charging_station_locations"]],
            d["drones_per_charging_station"],
        )

    print("  [sensor] Computing placement (Julia) ...", flush=True)
    strat = strategy_cls(rescaled_auto_params, rescaled_custom_params)
    ground_locs, charging_locs = strat.get_locations()
    drones_per_station         = strat.get_drone_allocation()

    log_data = {
        "ground_sensor_locations":    [[int(v) for v in x] for x in ground_locs],
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


# ── Routing log (one file per (cluster, log_key)) ─────────────────────────────
# Each (cluster_fingerprint, log_key) pair gets its own JSON file so that
# parallel worker processes never write to the same file.

def _routing_log_path(log_dir, base_cls_name, oh, rs, cluster_fp, log_key):
    return os.path.join(
        log_dir,
        f"routing_yearly_{base_cls_name}_{oh}OH_{rs}RS_"
        f"cluster_{cluster_fp}_time_{log_key}.json",
    )


def _routing_log_has(path, min_steps):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    return len(data.get("actions_history", [])) >= min_steps


def _routing_log_read(path):
    with open(path) as f:
        return json.load(f)


def _routing_log_write(path, initial_locs, actions_history):
    with open(path, "w") as f:
        json.dump(
            {"initial_drone_locations": initial_locs,
             "actions_history":         actions_history},
            f, indent=2,
        )


# ── Worker process state ──────────────────────────────────────────────────────
_worker_ctx = {}


def _worker_init(yearly_map_path, shared_ctx):
    """Initialise per-worker state (runs once per spawned worker process).

    Loads the yearly WFPI map (memory-mapped) and stores the shared context
    so that ``process_scenario`` can access it without pickling large objects
    for every task.
    """
    global _worker_ctx
    _worker_ctx = dict(shared_ctx)
    _worker_ctx["yearly_map"] = np.load(yearly_map_path, mmap_mode="r")


def process_scenario(task):
    """Process a single fire scenario in a worker process.

    Reads shared state from the module-level ``_worker_ctx`` (set by
    ``_worker_init``) and per-scenario data from *task*.
    """
    ctx = _worker_ctx
    yearly_map         = ctx["yearly_map"]
    H                  = ctx["H"]
    W                  = ctx["W"]
    coverage_w         = ctx["coverage_w"]
    operational_substeps = ctx["operational_substeps"]
    base_rescaled_auto = ctx["base_rescaled_auto"]
    combo_custom       = ctx["combo_custom"]
    combo_name         = ctx["combo_name"]
    ground_locs_opt    = ctx["ground_locs_opt"]
    ground_locs_data   = ctx["ground_locs_data"]
    tmp_dir            = ctx["tmp_dir"]
    routing_cls_name   = ctx["routing_cls_name"]

    RoutingCls = getattr(wrappers, routing_cls_name)

    sf              = task["sf"]
    name            = task["name"]
    date_str        = task["date_str"]
    offset          = task["offset"]
    sim_start       = task["sim_start"]
    log_key         = task["log_key"]
    cluster         = task["cluster"]
    routing_log_file = task["routing_log_file"]

    status = None

    if cluster is None:
        scenario = load_scenario_npy(
            sf, grid_height=H, grid_width=W,
            num_timesteps=N_SCENARIO_DATA_STEPS,
        )
        results = check_ground_sensor_detection(
            scenario, ground_locs_data, offset
        )
        results["routed"] = False
        status = "sensor_only"

    else:
        total_substeps_needed = (
            (offset + N_SCENARIO_DATA_STEPS) * operational_substeps
        )

        if not _routing_log_has(routing_log_file, total_substeps_needed):
            bm = build_burn_map(
                yearly_map, sim_start, MAX_ROUTING_DATA_STEPS
            )
            rescaled_bm = pool_burnmap_mean(bm, coverage_w)
            rescaled_bm = (
                np.repeat(rescaled_bm, operational_substeps, axis=0)
                / operational_substeps
            )
            tmp_path = os.path.join(
                tmp_dir, f"yearly_{log_key}_{uuid.uuid4().hex[:8]}.npy"
            )
            np.save(tmp_path, rescaled_bm)

            cluster_auto = {
                **base_rescaled_auto,
                "n_drones":                    cluster["n_drones"],
                "n_charging_stations":         len(cluster["stations_opt"]),
                "ground_sensor_locations":     ground_locs_opt,
                "charging_stations_locations": cluster["stations_opt"],
            }
            routing_custom = {
                **combo_custom,
                "burnmap_filename": tmp_path,
            }

            initial_norm, actions_hist = compute_routing(
                RoutingCls, cluster_auto, routing_custom,
                MAX_ROUTING_DATA_STEPS, operational_substeps,
            )
            _routing_log_write(routing_log_file, initial_norm, actions_hist)
            status = "routed"
            print(
                f"  [{log_key}] cluster={cluster['fingerprint'][:12]}… "
                f"routing computed ({len(actions_hist)} substeps)",
                flush=True,
            )
        else:
            status = "cached"

        cluster_charging_data = [
            (x * coverage_w + coverage_w // 2,
             y * coverage_w + coverage_w // 2)
            for x, y in cluster["stations_opt"]
        ]

        routing_entry = _routing_log_read(routing_log_file)
        scenario = load_scenario_npy(
            sf, grid_height=H, grid_width=W,
            num_timesteps=N_SCENARIO_DATA_STEPS,
        )
        results = run_simulation(
            scenario           = scenario,
            starting_time      = offset,
            routing_entry      = routing_entry,
            ground_locs_data   = ground_locs_data,
            charging_locs_data = cluster_charging_data,
            N=H, M=W,
            coverage_width_cells = coverage_w,
            operational_substeps = operational_substeps,
            max_battery_distance = SIMULATION_PARAMETERS["max_battery_distance"],
            max_battery_time     = SIMULATION_PARAMETERS["max_battery_time"],
        )
        results["routed"] = True

    return {
        "strategy_combo": combo_name,
        "scenario_name":  name,
        "date":           date_str,
        "sim_start_hour": sim_start.hour,
        "log_key":        log_key,
        "offset":         offset,
        "cluster":        cluster["fingerprint"] if cluster else "none",
        "_status":        status,
        **results,
    }


# ── Action normalisation ───────────────────────────────────────────────────────

def _normalise_initial(raw) -> list:
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
    """Run the routing strategy and collect every action taken."""
    strategy = routing_strategy_cls(rescaled_auto_params, rescaled_custom_params)

    rescaled_N           = rescaled_auto_params["N"]
    rescaled_M           = rescaled_auto_params["M"]
    n_drones             = rescaled_auto_params["n_drones"]
    rescaled_max_battery = rescaled_auto_params["max_battery_time"]

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

            new_locs = []
            for i, (act_type, param) in enumerate(acts):
                if act_type in ("charge", "fly"):
                    new_locs.append(tuple(param))
                    drone_batteries[i] = (rescaled_max_battery
                                          if act_type == "charge"
                                          else max(0, drone_batteries[i] - 1))
                elif act_type == "move":
                    ox, oy = drone_locs[i]
                    new_locs.append((
                        max(0, min(rescaled_N - 1, ox + param[0])),
                        max(0, min(rescaled_M - 1, oy + param[1])),
                    ))
                    drone_batteries[i] = max(0, drone_batteries[i] - 1)
                else:
                    new_locs.append(drone_locs[i])
            drone_locs = new_locs
        t += 1

    return initial_norm, actions_history


# ── Ground-sensor-only detection (no drones) ──────────────────────────────────

def check_ground_sensor_detection(scenario: np.ndarray,
                                  ground_locs_data: list,
                                  starting_time: int) -> dict:
    """Check if the fire ever hits a ground sensor during the simulation window.

    Used for fires that are outside all drone clusters.
    """
    if not ground_locs_data:
        return {"delta_t": -1, "device": "undetected",
                "fire_size_cells": int(np.sum(scenario[-1] > 0.5)),
                "fire_size_percentage": np.sum(scenario[-1] > 0.5) /
                                        (scenario.shape[1] * scenario.shape[2]) * 100,
                "total_distance_traveled": 0,
                "percentage_map_explored": 0.0}

    rows_g = [x for x, _ in ground_locs_data]
    cols_g = [y for _, y in ground_locs_data]

    for time_step in range(len(scenario)):
        grid = scenario[time_step]
        if (grid[rows_g, cols_g] == 1).any():
            delta_t = time_step - starting_time
            fire_size_cells = int(np.sum(grid > 0.5))
            return {
                "delta_t":                 delta_t,
                "device":                  "ground sensor",
                "fire_size_cells":         fire_size_cells,
                "fire_size_percentage":    fire_size_cells /
                                           (grid.shape[0] * grid.shape[1]) * 100,
                "total_distance_traveled": 0,
                "percentage_map_explored": 0.0,
            }

    final = scenario[-1]
    return {
        "delta_t":                 -1,
        "device":                  "undetected",
        "fire_size_cells":         int(np.sum(final > 0.5)),
        "fire_size_percentage":    np.sum(final > 0.5) /
                                   (final.shape[0] * final.shape[1]) * 100,
        "total_distance_traveled": 0,
        "percentage_map_explored": 0.0,
    }


# ── Full simulation runner ─────────────────────────────────────────────────────

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
    """Replay cached routing on a fire scenario and return detection metrics."""
    initial_norm = routing_entry["initial_drone_locations"]
    actions_log  = routing_entry["actions_history"]
    action_ptr   = 0

    rescaled_max_battery_time = max_battery_time * operational_substeps

    drones = [
        Drone(
            x * coverage_width_cells + coverage_width_cells // 2,
            y * coverage_width_cells + coverage_width_cells // 2,
            state, charging_locs_data, N, M,
            max_battery_distance, max_battery_time,
            max_battery_distance - 1 * (state == "fly"),
            max_battery_time     - 1 * (state == "fly"),
        )
        for state, (x, y) in initial_norm
    ]

    drone_locs_data     = [drone.get_position() for drone in drones]
    drone_locs_opt      = [(x, y) for _, (x, y) in initial_norm]
    drone_batteries_opt = [rescaled_max_battery_time] * len(drones)

    rows_g = [x for x, _ in ground_locs_data]   if ground_locs_data   else []
    cols_g = [y for _, y in ground_locs_data]    if ground_locs_data   else []
    rows_c = [x for x, _ in charging_locs_data] if charging_locs_data else []
    cols_c = [y for _, y in charging_locs_data] if charging_locs_data else []

    fire_detected        = False
    device               = "undetected"
    fire_size_cells      = 0
    fire_size_percentage = 0.0
    total_distance       = 0
    visited_cells        = set(drone_locs_data)
    t_found              = 0

    for time_step in range(-starting_time,
                           min(N_SCENARIO_DATA_STEPS + starting_time,
                               len(scenario))):

        if time_step >= 0:
            grid = scenario[time_step]

            if rows_g and (grid[rows_g, cols_g] == 1).any():
                fire_detected = True; device = "ground sensor"
                fire_size_cells = int(np.sum(grid > 0.5))
                fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                break

            if rows_c and (grid[rows_c, cols_c] == 1).any():
                fire_detected = True; device = "charging station"
                fire_size_cells = int(np.sum(grid > 0.5))
                fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                break

        for _ in range(operational_substeps):
            if action_ptr < len(actions_log):
                acts = _denormalise_actions(actions_log[action_ptr])
            else:
                acts = [("charge", drone_locs_opt[i]) for i in range(len(drones))]
            action_ptr += 1

            new_locs_opt = []
            for i, (act_type, param) in enumerate(acts):
                if act_type in ("charge", "fly"):
                    new_locs_opt.append(tuple(param))
                    drone_batteries_opt[i] = (rescaled_max_battery_time
                                              if act_type == "charge"
                                              else max(0, drone_batteries_opt[i] - 1))
                elif act_type == "move":
                    ox, oy = drone_locs_opt[i]
                    new_locs_opt.append((
                        max(0, min(N // coverage_width_cells - 1, ox + param[0])),
                        max(0, min(M // coverage_width_cells - 1, oy + param[1])),
                    ))
                    drone_batteries_opt[i] = max(0, drone_batteries_opt[i] - 1)
                else:
                    new_locs_opt.append(drone_locs_opt[i])
            drone_locs_opt = new_locs_opt

            actions_data = []
            for act_type, param in acts:
                if act_type == "fly":
                    c = operational_space_to_dataspace_coordinates(
                        param, coverage=SIMULATION_PARAMETERS["coverage_radius_m"],
                        datacell_size_m=SIMULATION_PARAMETERS["cell_size_m"])
                    actions_data.append((act_type, c))
                elif act_type == "move":
                    actions_data.append(
                        (act_type, (coverage_width_cells * param[0],
                                    coverage_width_cells * param[1])))
                else:
                    c = operational_space_to_dataspace_coordinates(
                        param, coverage=SIMULATION_PARAMETERS["coverage_radius_m"],
                        datacell_size_m=SIMULATION_PARAMETERS["cell_size_m"])
                    actions_data.append((act_type, c))

            for i, (drone, action) in enumerate(zip(drones, actions_data)):
                old_x, old_y   = drone_locs_data[i]
                new_x, new_y, *_ = drone.route(action)
                drone_locs_data[i] = (new_x, new_y)
                total_distance    += abs(new_x - old_x) + abs(new_y - old_y)
                visited_cells.add((new_x, new_y))

            if time_step >= 0:
                for pos in drone_locs_data:
                    if detect_fire_within_coverage(grid, pos, coverage_width_cells):
                        fire_detected = True; device = "drone"
                        fire_size_cells = int(np.sum(grid > 0.5))
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
        final       = scenario[-1]
        fire_size_cells      = int(np.sum(final > 0.5))
        fire_size_percentage = fire_size_cells / (final.shape[0] * final.shape[1]) * 100

    return {
        "delta_t":                 delta_t,
        "device":                  device,
        "fire_size_cells":         fire_size_cells,
        "fire_size_percentage":    fire_size_percentage,
        "total_distance_traveled": total_distance,
        "percentage_map_explored": len(visited_cells) / (N * M) * 100,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(budget_millions: float = 20.0):
    LOG_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    strategy_combinations = build_strategy_combinations(budget_millions)
    print(f"Budget: {int(budget_millions)}M", flush=True)

    # ── Static resources ──────────────────────────────────────────────────────
    print("Loading yearly WFPI map (memory-mapped) ...", flush=True)
    yearly_map = np.load(str(YEARLY_MAP), mmap_mode="r")

    print("Loading avg WFPI map ...", flush=True)
    avg_map = np.load(str(AVG_MAP))

    print("Loading mask ...", flush=True)
    mask = np.load(str(MASK_PATH))
    H, W = mask.shape

    print("Loading config ...", flush=True)
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # ── Rescaling (fixed for all scenarios) ───────────────────────────────────
    cell_size_m  = SIMULATION_PARAMETERS["cell_size_m"]
    speed        = SIMULATION_PARAMETERS["drone_speed_m_per_min"]
    coverage_r_m = SIMULATION_PARAMETERS["coverage_radius_m"]

    operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
    coverage_w = round(coverage_r_m * 2 / cell_size_m)
    if coverage_w % 2 == 0:
        coverage_w -= 1

    rescaled_N            = H // coverage_w
    rescaled_M            = W // coverage_w
    rescaled_max_battery  = SIMULATION_PARAMETERS["max_battery_time"] * operational_substeps

    print(
        f"coverage_width={coverage_w} cells, "
        f"opt grid={rescaled_N}×{rescaled_M}, "
        f"substeps={operational_substeps}, "
        f"max_battery_substeps={rescaled_max_battery}",
        flush=True,
    )

    suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"

    rescaled_mask = pool_mask(mask, coverage_w,
                              mode=SIMULATION_PARAMETERS["mask_pooling_mode"])
    rescaled_mask_path = str(MASK_PATH).replace(".npy", suffix)
    np.save(rescaled_mask_path, rescaled_mask)

    # Apply data-scale mask BEFORE pooling so that non-California cells (which
    # can have very high WFPI values from neighbouring states) do not contaminate
    # the block mean for border opt-cells.
    avg_map_masked = avg_map * mask          # zero out non-CA cells at 1km scale

    if SENSOR_POOLING == "proba":
        # Treat WFPI/255 as per-cell fire probability; pool as P(at least one cell fires).
        rescaled_avg = pool_burnmap_proba_at_least_one(avg_map_masked / 255.0, coverage_w)
        # Re-normalise: the probabilities saturate near 1 for most cells.
        # Stretch valid (non-zero) cells to [0, 255] to recover spatial contrast.
        nz = rescaled_avg > 0
        if nz.any():
            p_min, p_max = rescaled_avg[nz].min(), rescaled_avg[nz].max()
            if p_max > p_min:
                rescaled_avg = np.where(
                    nz,
                    (rescaled_avg - p_min) / (p_max - p_min) * 255.0,
                    0.0,
                )
    else:
        rescaled_avg = pool_burnmap_mean(avg_map_masked, coverage_w)

    rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
    rescaled_avg_path = str(AVG_MAP).replace(".npy", f"_{SENSOR_POOLING}{suffix}")
    np.save(rescaled_avg_path, rescaled_avg)

    print(f"Sensor pooling mode: {SENSOR_POOLING}", flush=True)

    # ── Valid scenario list ───────────────────────────────────────────────────
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf for sf in all_scenario_files
        if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config
               for k in ("offset", "date", "time"))
    ]
    print(
        f"Scenarios with date+time: {len(valid_scenarios)}/{len(all_scenario_files)}",
        flush=True,
    )

    # ── Random subset selection ───────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(rng.choice(len(valid_scenarios), size=BENCHMARK_SUBSET_SIZE, replace=False))
    benchmark_scenarios = [valid_scenarios[i] for i in subset_idx]
    print(
        f"Random subset: {BENCHMARK_SUBSET_SIZE} scenarios (seed={RANDOM_SEED})\n",
        flush=True,
    )

    # ── Base parameter dicts ──────────────────────────────────────────────────
    base_auto = {
        "N": H, "M": W,
        "max_battery_distance": SIMULATION_PARAMETERS["max_battery_distance"],
        "max_battery_time":     SIMULATION_PARAMETERS["max_battery_time"],
        "n_drones":             SIMULATION_PARAMETERS["n_drones"],
        "n_ground_stations":    SIMULATION_PARAMETERS["n_ground_stations"],
        "n_charging_stations":  SIMULATION_PARAMETERS["n_charging_stations"],
        "speed_m_per_min":      speed,
        "coverage_radius_m":    coverage_r_m,
        "cell_size_m":          cell_size_m,
        "transmission_range":   SIMULATION_PARAMETERS["transmission_range"],
        "mask_filename":        rescaled_mask_path,
    }
    base_rescaled_auto = {
        **base_auto,
        "N": rescaled_N, "M": rescaled_M,
        "max_battery_time": rescaled_max_battery,
    }
    base_custom = {
        "mask_filename":        rescaled_mask_path,
        "recompute_logfile":    True,   # benchmark uses RoutingLog; skip inner wrapper cache
        "recompute_kernel":     False,
        "use_linf_cost":        True,
        "regularization_param": 1e5,
    }

    all_results = []

    # ── Strategy loop ─────────────────────────────────────────────────────────
    for combo in strategy_combinations:
        combo_name         = combo["name"]
        SensorCls          = combo["sensor"]
        routing_cls_name   = combo["drone"]          # string
        RoutingCls         = getattr(wrappers, routing_cls_name)
        combo_custom = {
            **base_custom,
            "reevaluation_step":    combo["params"]["reevaluation_step"],
            "optimization_horizon": combo["params"]["optimization_horizon"],
            "burnmap_type":         "dynamic",  # avoid 200x tiling of burn map
        }
        if "time_limit_seconds" in combo["params"]:
            combo_custom["time_limit_seconds"] = combo["params"]["time_limit_seconds"]

        print(f"\n{'='*70}", flush=True)
        print(f"  STRATEGY: {combo_name}", flush=True)
        print(f"{'='*70}\n", flush=True)

        # ── Sensor placement with allocation ──────────────────────────────────
        sensor_cache_name = combo.get("sensor_cache_key", combo_name)
        sensor_log_path = str(
            LOG_DIR / f"sensor_alloc_{sensor_cache_name}_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}.json"
        )
        sensor_custom = {**combo_custom, "burnmap_filename": rescaled_avg_path,
                         **combo.get("sensor_params", {})}

        ground_locs_opt, charging_locs_opt, drones_per_station = \
            load_or_compute_sensor_placement(
                SensorCls, base_rescaled_auto, sensor_custom, sensor_log_path
            )

        ground_locs_data = [
            (x * coverage_w + coverage_w // 2, y * coverage_w + coverage_w // 2)
            for x, y in ground_locs_opt
        ]
        charging_locs_data = [
            (x * coverage_w + coverage_w // 2, y * coverage_w + coverage_w // 2)
            for x, y in charging_locs_opt
        ]

        total_drones = sum(drones_per_station)
        print(
            f"  Ground sensors ({len(ground_locs_opt)}): {ground_locs_opt}\n"
            f"  Charging stations ({len(charging_locs_opt)}): {charging_locs_opt}\n"
            f"  Drones per station: {drones_per_station}  (total: {total_drones})",
            flush=True,
        )

        # ── Cluster computation ───────────────────────────────────────────────
        clusters = compute_clusters(
            charging_locs_opt, drones_per_station, rescaled_max_battery
        )
        print(f"  Clusters: {len(clusters)}", flush=True)
        for i, c in enumerate(clusters):
            print(
                f"    Cluster {i}: stations={c['stations_opt']}, "
                f"drones={c['n_drones']}, fp={c['fingerprint'][:20]}...",
                flush=True,
            )

        # ── Routing log naming ─────────────────────────────────────────────
        base_routing_cls = wrappers._deep_unwrap(RoutingCls).__name__
        oh = combo["params"]["optimization_horizon"]
        rs = combo["params"]["reevaluation_step"]

        # ── Pre-scan: compute metadata for each scenario ────────────────────
        scenario_tasks = []
        for sf in benchmark_scenarios:
            name     = sf.stem.replace("_scenario1", "")
            date_str = config[f"date_{name}"]
            time_str = config[f"time_{name}"]
            offset   = config[f"offset_{name}"]

            discovery_dt = datetime(
                int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]),
                int(time_str[:2]), int(time_str[2:]),
            )
            sim_start = round_to_nearest_hour(
                discovery_dt - timedelta(minutes=30 * offset)
            )
            log_key = f"{sim_start.strftime('%Y%m%d')}_{sim_start.hour:02d}"

            pt = np.load(str(sf))
            fire_row, fire_col = int(pt[0]), int(pt[1])
            fire_opt = (fire_row // coverage_w, fire_col // coverage_w)

            cluster = fire_cluster(fire_opt, clusters, rescaled_max_battery)

            routing_log_file = None
            if cluster is not None:
                routing_log_file = _routing_log_path(
                    str(LOG_DIR), base_routing_cls, oh, rs,
                    cluster["fingerprint"], log_key,
                )

            scenario_tasks.append({
                "sf":              str(sf),
                "name":            name,
                "date_str":        date_str,
                "offset":          offset,
                "sim_start":       sim_start,
                "log_key":         log_key,
                "fire_opt":        fire_opt,
                "cluster":         cluster,
                "routing_log_file": routing_log_file,
            })

        # ── Assign waves so same (cluster, log_key) never run together ────
        # Each (cluster, log_key) pair has its own routing log file (Option B).
        # Two scenarios that share the same pair would both read/write the same
        # file, so they must not run concurrently.
        conflict_groups = defaultdict(list)
        for i, task in enumerate(scenario_tasks):
            if task["cluster"] is not None:
                key = (task["cluster"]["fingerprint"], task["log_key"])
                conflict_groups[key].append(i)
            else:
                task["wave"] = 0

        n_waves = 1
        for indices in conflict_groups.values():
            for wave_idx, idx in enumerate(indices):
                scenario_tasks[idx]["wave"] = wave_idx
                n_waves = max(n_waves, wave_idx + 1)

        wave_sizes = defaultdict(int)
        for task in scenario_tasks:
            wave_sizes[task["wave"]] += 1
        print(
            f"  Pre-scan complete: {len(scenario_tasks)} tasks, "
            f"{n_waves} wave(s) (sizes: {dict(sorted(wave_sizes.items()))})",
            flush=True,
        )

        # ── Shared context for worker processes ───────────────────────────
        shared_ctx = {
            "H":                   H,
            "W":                   W,
            "coverage_w":          coverage_w,
            "operational_substeps": operational_substeps,
            "rescaled_max_battery": rescaled_max_battery,
            "base_rescaled_auto":  base_rescaled_auto,
            "combo_custom":        combo_custom,
            "combo_name":          combo_name,
            "ground_locs_opt":     ground_locs_opt,
            "ground_locs_data":    ground_locs_data,
            "routing_cls_name":    routing_cls_name,
            "tmp_dir":             str(TMP_DIR),
        }

        # ── Execute waves in parallel (ProcessPoolExecutor + spawn) ───────
        n_skipped = n_sensor_only = n_routed = n_cached = 0
        ctx = get_context("spawn")

        for wave in range(n_waves):
            wave_tasks = [t for t in scenario_tasks if t["wave"] == wave]
            if not wave_tasks:
                continue
            n_workers = min(MAX_WORKERS, len(wave_tasks))
            print(
                f"  Wave {wave}/{n_waves-1}: {len(wave_tasks)} scenarios "
                f"(workers={n_workers})",
                flush=True,
            )

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(str(YEARLY_MAP), shared_ctx),
            ) as executor:
                futures = {
                    executor.submit(process_scenario, t): t
                    for t in wave_tasks
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        task = futures[future]
                        print(
                            f"  ERROR processing {task['name']}: {exc}",
                            flush=True,
                        )
                        continue
                    status = result.pop("_status")
                    if status == "routed":
                        n_routed += 1
                    elif status == "cached":
                        n_cached += 1
                    elif status == "sensor_only":
                        n_sensor_only += 1
                        if result["device"] == "undetected":
                            n_skipped += 1
                    all_results.append(result)

        print(
            f"\n  Done: {n_routed} routing computations, "
            f"{n_cached} cached replays, "
            f"{n_sensor_only - n_skipped} sensor-only detections checked, "
            f"{n_skipped} immediately skipped (no cluster, no sensor).",
            flush=True,
        )

    # ── Save results ──────────────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = f"benchmark_results_yearly_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}", flush=True)

        for combo_name, gdf in df.groupby("strategy_combo"):
            detected  = (gdf["delta_t"] != -1).sum()
            n         = len(gdf)
            mean_dt   = gdf.loc[gdf["delta_t"] != -1, "delta_t"].mean()
            print(
                f"  {combo_name}: "
                f"detection rate={detected/n*100:.1f}%  "
                f"mean delta_t={mean_dt:.2f} (detected)  n={n}",
                flush=True,
            )
    else:
        print("No results collected.", flush=True)

    print("\nDone.", flush=True)


def run_sensor_placement_only(budget_millions: float, time_limit_seconds: float = 600.0):
    """Run only sensor placement (Budget strategy) and save to cache. Used for 100M etc."""
    LOG_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)
    print("Loading avg WFPI map ...", flush=True)
    avg_map = np.load(str(AVG_MAP))
    print("Loading mask ...", flush=True)
    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
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
    rescaled_avg_path = str(AVG_MAP).replace(".npy", f"_{SENSOR_POOLING}{suffix}")
    if not Path(rescaled_avg_path).exists():
        avg_map_masked = avg_map * mask
        rescaled_avg = pool_burnmap_mean(avg_map_masked, coverage_w)
        rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
        np.save(rescaled_avg_path, rescaled_avg)
    base_rescaled_auto = {
        "N": H, "M": W,
        "max_battery_distance": SIMULATION_PARAMETERS["max_battery_distance"],
        "max_battery_time":     rescaled_max_battery,
        "n_drones":             SIMULATION_PARAMETERS["n_drones"],
        "n_ground_stations":    SIMULATION_PARAMETERS["n_ground_stations"],
        "n_charging_stations":  SIMULATION_PARAMETERS["n_charging_stations"],
        "speed_m_per_min":      speed,
        "coverage_radius_m":    coverage_r_m,
        "cell_size_m":          cell_size_m,
        "transmission_range":   SIMULATION_PARAMETERS["transmission_range"],
        "mask_filename":        rescaled_mask_path,
        "N": rescaled_N, "M": rescaled_M,
    }
    base_custom = {
        "mask_filename": rescaled_mask_path,
        "recompute_logfile": True,
        "recompute_kernel": False,
        "use_linf_cost": True,
        "regularization_param": 1e5,
    }
    combo_name = f"GaussianBudget{int(budget_millions)}M"
    sensor_log_path = str(LOG_DIR / f"sensor_alloc_{combo_name}_{rescaled_N}x{rescaled_M}_{SENSOR_POOLING}.json")
    sensor_custom = {
        **base_custom,
        "burnmap_filename":    rescaled_avg_path,
        "budget_millions":     budget_millions,
        "cost_sensor":         COST_SENSOR / 1_000_000,
        "cost_station":        COST_STATION / 1_000_000,
        "cost_drone":          COST_DRONE / 1_000_000,
        "time_limit_seconds":   time_limit_seconds,
    }
    print(f"Running sensor placement only: budget={budget_millions}M, time limit={time_limit_seconds}s", flush=True)
    load_or_compute_sensor_placement(
        SensorPlacementMaxCoverageGaussianTimeMaskedBudget,
        base_rescaled_auto,
        sensor_custom,
        sensor_log_path,
    )
    print(f"Saved to {sensor_log_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Yearly benchmark on California 2020 (100 fires, parallel, TOP masked, 1 min routing cap)."
    )
    parser.add_argument(
        "--budget",
        type=int,
        choices=[20, 100, 500],
        default=20,
        help="Total budget in millions (default: 20)",
    )
    parser.add_argument(
        "--sensor-only",
        action="store_true",
        help="Run only sensor placement for the given budget and exit.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="Sensor placement time limit in seconds (default: 600); used only with --sensor-only.",
    )
    args = parser.parse_args()

    if args.sensor_only:
        run_sensor_placement_only(
            budget_millions=float(args.budget),
            time_limit_seconds=args.time_limit,
        )
        sys.exit(0)
    main(budget_millions=float(args.budget))
