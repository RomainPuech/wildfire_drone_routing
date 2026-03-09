#!/usr/bin/env python3
"""
Benchmark on California 2020 Dataset — Pyrologix (Static) Ignition Probability Map

Same pipeline, budget, and fire subset as run_benchmark_california2020_yearly.py
but with Pyrologix ignition probability as the risk map for both sensor placement
and drone routing.

Key differences vs the WFPI yearly benchmark:
  - Risk map: Pyrologix ignition probability, resampled from 10944×6382 to 1309×805
    to match the WFPI grid.  Values scaled to [0, 255] for numerical compatibility.
  - Routing burn map: static (same map for every scenario/date).
    A single routing per cluster is computed with log_key="static" and reused.
  - Sensor placement cache: separate JSON (includes "_pyrologix" tag).
  - Routing logs: separate files (prefix "routing_pyrologix_").

Run from the project root:
  python -u run_benchmark_california2020_pyrologix.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

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


# ── Dataset paths ───────────────────────────────────────────────────────────────
# Fires and config come from the WFPI dataset (same 100 fires).
WFPI_DIR     = PROJECT_ROOT / "California2020Dataset"
MASK_PATH    = WFPI_DIR / "mask.npy"
CONFIG_PATH  = WFPI_DIR / "config_california_2020.json"
SCENARII_DIR = WFPI_DIR / "scenarii"
LOG_DIR      = WFPI_DIR / "logs"
TMP_DIR      = PROJECT_ROOT / "tmp_burnmaps"

# Pyrologix source (native 10944×6382).
PYROLOGIX_RAW = (PROJECT_ROOT / "California2020Dataset_IgnitionProb"
                 / "static_risk_ignition_prob.npy")
# Resampled cache (written here if not present).
PYROLOGIX_RESAMPLED = WFPI_DIR / "static_risk_pyrologix_resampled.npy"

MAX_ROUTING_DATA_STEPS = 24   # offset ≤ 12 + 12 scenario steps
N_SCENARIO_DATA_STEPS  = 12   # 12 × 30 min = 6 h

# Static log_key used for routing: because the burn map never changes,
# every cluster computes its routing exactly once.
STATIC_LOG_KEY = "static"


# ── Simulation parameters (identical to WFPI benchmark) ────────────────────────
SIMULATION_PARAMETERS = {
    "max_battery_distance": -1,
    "max_battery_time":      1,
    "n_drones":              50,
    "n_ground_stations":     100,
    "n_charging_stations":   50,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m":     2900,
    "cell_size_m":           1000,
    "transmission_range":    50000,
    "mask_pooling_mode":     "max",
}

BUDGET_TOTAL  = 20_000_000
COST_SENSOR   = 100_000
COST_DRONE    =  50_000
COST_STATION  = 150_000

SENSOR_POOLING        = "mean"
BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED           = 42


# ── Strategy combinations ───────────────────────────────────────────────────────
STRATEGY_COMBINATIONS = [
    {
        "name":   "GaussianAlloc_TOP",
        "sensor": SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation,
        "drone":  wrappers.DroneRoutingTOPMaskedLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10},
    },
]


# ── Pyrologix resampling ────────────────────────────────────────────────────────

def resample_pyrologix(raw_path: Path, target_shape: tuple, out_path: Path) -> np.ndarray:
    """Resample Pyrologix from its native grid to target_shape (H, W).

    Values are scaled to [0, 255] for numerical compatibility with the
    WFPI-based infrastructure (ILP weights, TOP rewards).
    Result saved as (1, H, W) float32.
    """
    if out_path.exists():
        print(f"  [pyrologix] Loading cached resampled map from {out_path.name}", flush=True)
        return np.load(str(out_path))[0]   # (H, W)

    print(f"  [pyrologix] Resampling {raw_path.name} → {target_shape} ...", flush=True)
    from skimage.transform import resize

    raw = np.load(str(raw_path))[0].astype(np.float32)   # (10944, 6382)
    H_t, W_t = target_shape

    resampled = resize(raw, (H_t, W_t), order=1, anti_aliasing=True,
                       preserve_range=True).astype(np.float32)

    # Scale to [0, 255]
    valid = resampled[np.isfinite(resampled) & (resampled > 0)]
    if valid.size:
        resampled = resampled / valid.max() * 255.0
    resampled = np.nan_to_num(resampled, nan=0.0)

    out = resampled[np.newaxis]   # (1, H, W)
    np.save(str(out_path), out)
    print(f"  [pyrologix] Saved resampled map to {out_path.name}  "
          f"range=[{resampled.min():.2f}, {resampled.max():.2f}]", flush=True)
    return resampled


# ── Build static burn map for routing ──────────────────────────────────────────

def build_burn_map_static(static_map_2d: np.ndarray, num_steps: int) -> np.ndarray:
    """Return (num_steps, H, W) by tiling the static 2-D map."""
    return np.broadcast_to(static_map_2d[np.newaxis], (num_steps,) + static_map_2d.shape).copy()


# ── Cluster helpers (unchanged from WFPI benchmark) ────────────────────────────

def round_to_nearest_hour(dt: datetime) -> datetime:
    if dt.minute >= 30:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)


def compute_clusters(charging_locs_opt, drones_per_station, max_battery_substeps):
    n = len(charging_locs_opt)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = charging_locs_opt[i]; xj, yj = charging_locs_opt[j]
            if max(abs(xi - xj), abs(yi - yj)) <= max_battery_substeps:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    for indices in groups.values():
        stations = [charging_locs_opt[i] for i in indices]
        n_dr     = sum(drones_per_station[i] for i in indices)
        fp       = "_".join(f"{x}-{y}" for x, y in sorted(stations))
        clusters.append({"stations_opt": stations, "n_drones": n_dr, "fingerprint": fp})
    return clusters


def fire_cluster(fire_opt, clusters, max_battery_substeps):
    one_way_reach = max_battery_substeps // 2
    fr, fc = fire_opt
    for cluster in clusters:
        for sx, sy in cluster["stations_opt"]:
            if max(abs(fr - sx), abs(fc - sy)) <= one_way_reach:
                return cluster
    return None


# ── Sensor placement cache ──────────────────────────────────────────────────────

def load_or_compute_sensor_placement(strategy_cls, rescaled_auto_params,
                                     rescaled_custom_params, log_path):
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


# ── Routing log ─────────────────────────────────────────────────────────────────

class RoutingLog:
    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path)) if Path(path).exists() else {}

    def has(self, key, min_steps):
        e = self.data.get(key)
        return e is not None and len(e.get("actions_history", [])) >= min_steps

    def get(self, key):
        return self.data[key]

    def put(self, key, initial_locs, actions_history):
        self.data[key] = {"initial_drone_locations": initial_locs,
                          "actions_history": actions_history}
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Action normalisation (unchanged) ───────────────────────────────────────────

def _normalise_initial(raw):
    if isinstance(raw, list):
        if raw and isinstance(raw[0], tuple) and isinstance(raw[0][0], str):
            return [(st, (int(x), int(y))) for st, (x, y) in raw]
        return [("charge", (int(x), int(y))) for x, y in raw]
    if isinstance(raw, tuple) and len(raw) == 2:
        positions, states = raw
        return [(st, (int(x), int(y))) for (x, y), st in zip(positions, states)]
    raise ValueError(f"Unexpected initial-location format: {type(raw)}")

def _normalise_actions(acts):
    return [[typ, None if param is None else list(param)] for typ, param in acts]

def _denormalise_actions(stored):
    return [(typ, None if param is None else tuple(param)) for typ, param in stored]


# ── Routing computation (unchanged) ────────────────────────────────────────────

def compute_routing(routing_strategy_cls, rescaled_auto_params,
                    rescaled_custom_params, n_data_steps, operational_substeps):
    strategy = routing_strategy_cls(rescaled_auto_params, rescaled_custom_params)
    rescaled_max_battery = rescaled_auto_params["max_battery_time"]
    n_drones             = rescaled_auto_params["n_drones"]

    initial_raw  = strategy.get_initial_drone_locations()
    initial_norm = _normalise_initial(initial_raw)

    drone_locs      = [pos for _, pos in initial_norm]
    drone_batteries = [rescaled_max_battery] * n_drones
    drone_states    = [st  for st, _  in initial_norm]
    actions_history = []
    t = 0

    for _ in range(n_data_steps):
        for _ in range(operational_substeps):
            step_params = {"drone_locations": drone_locs,
                           "drone_batteries": drone_batteries,
                           "drone_states":    drone_states,
                           "t":               t}
            acts = strategy.next_actions(step_params, {})
            actions_history.append(_normalise_actions(acts))

            new_locs    = []
            new_batts   = []
            new_states  = []
            for i, (act_type, param) in enumerate(acts):
                if act_type == "fly":
                    new_locs.append(tuple(param))
                    new_batts.append(max(0, drone_batteries[i] - 1))
                    new_states.append("fly")
                elif act_type == "charge":
                    # Update location to the charge param (the station position),
                    # matching the WFPI benchmark — critical so the reevaluation
                    # check sees the drone at the charging station.
                    new_locs.append(tuple(param))
                    new_batts.append(rescaled_max_battery)
                    new_states.append("charge")
                else:
                    new_locs.append(drone_locs[i])
                    new_batts.append(drone_batteries[i])
                    new_states.append(act_type)
            drone_locs      = new_locs
            drone_batteries = new_batts
            drone_states    = new_states
            t += 1

    return initial_norm, actions_history


# ── Ground-sensor detection (unchanged) ────────────────────────────────────────

def check_ground_sensor_detection(scenario, ground_locs_data, offset):
    H, W = scenario.shape[1], scenario.shape[2]
    sensor_set = set(ground_locs_data)
    for t_rel in range(scenario.shape[0]):
        t_abs = offset + t_rel
        fire_cells = set(zip(*np.where(scenario[t_rel] > 0.5)))
        for fc in fire_cells:
            if fc in sensor_set:
                return {"delta_t": t_abs, "device": "ground_sensor",
                        "fire_size_cells": int(np.sum(scenario[-1] > 0.5)),
                        "fire_size_percentage": int(np.sum(scenario[-1] > 0.5))/(H*W)*100,
                        "total_distance_traveled": 0,
                        "percentage_map_explored": 0}
    return {"delta_t": -1, "device": "undetected",
            "fire_size_cells": int(np.sum(scenario[-1] > 0.5)),
            "fire_size_percentage": int(np.sum(scenario[-1] > 0.5))/(H*W)*100,
            "total_distance_traveled": 0, "percentage_map_explored": 0}


# ── Simulation replay (unchanged) ──────────────────────────────────────────────

def run_simulation(scenario, starting_time, routing_entry, ground_locs_data,
                   charging_locs_data, N, M, coverage_width_cells,
                   operational_substeps, max_battery_distance, max_battery_time):
    from benchmark import detect_fire_within_coverage

    initial_norm    = routing_entry["initial_drone_locations"]
    actions_history = [_denormalise_actions(a) for a in routing_entry["actions_history"]]

    rescaled_max_battery_time = max_battery_time * operational_substeps
    drone_locs   = [pos for _, pos in initial_norm]
    drone_batts  = [rescaled_max_battery_time] * len(drone_locs)
    drone_states = [st  for st, _  in initial_norm]

    delta_t = -1; device = "undetected"
    visited_cells    = set()
    total_distance   = 0
    sensor_set = set(ground_locs_data)
    t = 0

    for t_data in range(starting_time + N_SCENARIO_DATA_STEPS):
        fire_grid = scenario[min(t_data - starting_time, scenario.shape[0]-1)] \
            if t_data >= starting_time else None

        for _ in range(operational_substeps):
            if t >= len(actions_history):
                break
            acts = actions_history[t]

            if delta_t == -1 and fire_grid is not None:
                for i, (old_x, old_y) in enumerate(drone_locs):
                    if detect_fire_within_coverage(
                            fire_grid, old_x, old_y,
                            coverage_width_cells, N, M):
                        delta_t = t_data; device = "drone"; break

            new_locs = []; new_batts = []; new_states = []
            for i, (act_type, param) in enumerate(acts):
                old_x, old_y = drone_locs[i]
                if act_type in ("fly", "charge"):
                    nx, ny = tuple(param)
                    new_locs.append((nx, ny))
                    if act_type == "fly":
                        new_batts.append(max(0, drone_batts[i] - 1))
                        new_states.append("fly")
                        visited_cells.add((nx, ny))
                        total_distance += abs(nx - old_x) + abs(ny - old_y)
                    else:  # charge
                        new_batts.append(rescaled_max_battery_time)
                        new_states.append("charge")
                else:
                    new_locs.append((old_x, old_y))
                    new_batts.append(drone_batts[i])
                    new_states.append(act_type)
            drone_locs   = new_locs
            drone_batts  = new_batts
            drone_states = new_states
            t += 1

        if delta_t == -1 and fire_grid is not None:
            fire_cells = set(zip(*np.where(fire_grid > 0.5)))
            for fc in fire_cells:
                if fc in sensor_set:
                    delta_t = t_data; device = "ground_sensor"; break

    final = scenario[-1]
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


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    # ── Load mask and config (WFPI grid) ──────────────────────────────────────
    print("Loading WFPI mask ...", flush=True)
    mask = np.load(str(MASK_PATH))
    H, W = mask.shape

    print("Loading config ...", flush=True)
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # ── Resample Pyrologix to WFPI grid ───────────────────────────────────────
    print("Preparing Pyrologix risk map ...", flush=True)
    pyrologix_2d = resample_pyrologix(PYROLOGIX_RAW, (H, W), PYROLOGIX_RESAMPLED)
    # pyrologix_2d: (H, W) float32, values in [0, 255]

    # ── Rescaling ─────────────────────────────────────────────────────────────
    cell_size_m  = SIMULATION_PARAMETERS["cell_size_m"]
    speed        = SIMULATION_PARAMETERS["drone_speed_m_per_min"]
    coverage_r_m = SIMULATION_PARAMETERS["coverage_radius_m"]

    operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
    coverage_w = round(coverage_r_m * 2 / cell_size_m)
    if coverage_w % 2 == 0:
        coverage_w -= 1

    rescaled_N           = H // coverage_w
    rescaled_M           = W // coverage_w
    rescaled_max_battery = SIMULATION_PARAMETERS["max_battery_time"] * operational_substeps

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

    # Pool Pyrologix to operational scale for sensor placement
    pyrologix_masked = pyrologix_2d * mask          # (H, W) — zero non-CA cells
    rescaled_avg = pool_burnmap_mean(pyrologix_masked[np.newaxis], coverage_w)  # (1, rescaled_N, rescaled_M)
    rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps

    rescaled_avg_path = str(PYROLOGIX_RESAMPLED).replace(
        ".npy", f"_{SENSOR_POOLING}{suffix}")
    np.save(rescaled_avg_path, rescaled_avg)

    # Pre-build the static routing burn map (operational scale, MAX steps)
    bm_static_data  = build_burn_map_static(pyrologix_masked, MAX_ROUTING_DATA_STEPS)
    bm_static_opt   = pool_burnmap_mean(bm_static_data, coverage_w)
    bm_static_opt   = (np.repeat(bm_static_opt, operational_substeps, axis=0)
                       / operational_substeps)
    static_bm_path  = str(TMP_DIR / "pyrologix_static.npy")
    np.save(static_bm_path, bm_static_opt)
    print(f"Static routing burn map saved: {bm_static_opt.shape}", flush=True)

    # ── Scenario list (same 100 fires as WFPI benchmark) ─────────────────────
    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf for sf in all_scenario_files
        if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config
               for k in ("offset", "date", "time"))
    ]
    print(f"Scenarios with date+time: {len(valid_scenarios)}/{len(all_scenario_files)}",
          flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(rng.choice(len(valid_scenarios),
                                    size=BENCHMARK_SUBSET_SIZE, replace=False))
    benchmark_scenarios = [valid_scenarios[i] for i in subset_idx]
    print(f"Random subset: {BENCHMARK_SUBSET_SIZE} scenarios (seed={RANDOM_SEED})\n",
          flush=True)

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
        "recompute_logfile":    True,
        "recompute_kernel":     False,
        "use_linf_cost":        True,
        "regularization_param": 1e5,
    }

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
            "burnmap_type":         "dynamic",
        }

        print(f"\n{'='*70}", flush=True)
        print(f"  STRATEGY: {combo_name}  [PYROLOGIX RISK MAP]", flush=True)
        print(f"{'='*70}\n", flush=True)

        # ── Sensor placement ──────────────────────────────────────────────────
        sensor_log_path = str(
            LOG_DIR / f"sensor_alloc_{combo_name}_{rescaled_N}x{rescaled_M}_pyrologix.json"
        )
        sensor_custom = {**combo_custom, "burnmap_filename": rescaled_avg_path}

        ground_locs_opt, charging_locs_opt, drones_per_station = \
            load_or_compute_sensor_placement(
                SensorCls, base_rescaled_auto, sensor_custom, sensor_log_path
            )

        ground_locs_data = [
            (x * coverage_w + coverage_w // 2, y * coverage_w + coverage_w // 2)
            for x, y in ground_locs_opt
        ]

        # ── Clusters ──────────────────────────────────────────────────────────
        clusters = compute_clusters(charging_locs_opt, drones_per_station,
                                    rescaled_max_battery)
        print(f"  Clusters: {len(clusters)}", flush=True)
        for i, c in enumerate(clusters):
            print(f"    Cluster {i}: stations={c['stations_opt']}, "
                  f"drones={c['n_drones']}, fp={c['fingerprint'][:20]}...", flush=True)

        # ── Routing logs (one per cluster, keyed "static") ────────────────────
        routing_logs = {}
        base_routing_cls = wrappers._deep_unwrap(RoutingCls).__name__
        oh = combo["params"]["optimization_horizon"]
        rs = combo["params"]["reevaluation_step"]
        for cluster in clusters:
            log_path = str(
                LOG_DIR /
                f"routing_pyrologix_{base_routing_cls}_{oh}OH_{rs}RS_"
                f"cluster_{cluster['fingerprint']}.json"
            )
            routing_logs[cluster["fingerprint"]] = RoutingLog(log_path)

        # ── Scenario loop ─────────────────────────────────────────────────────
        n_skipped = n_sensor_only = n_routed = n_cached = 0

        for sf in benchmark_scenarios:
            name   = sf.stem.replace("_scenario1", "")
            offset = config[f"offset_{name}"]
            date_str = config[f"date_{name}"]
            time_str = config[f"time_{name}"]

            # Convert ignition to opt-space
            pt = np.load(str(sf))
            fire_row, fire_col = int(pt[0]), int(pt[1])
            fire_opt = (fire_row // coverage_w, fire_col // coverage_w)

            # Find cluster
            cluster = fire_cluster(fire_opt, clusters, rescaled_max_battery)

            if cluster is None:
                scenario = load_scenario_npy(
                    str(sf), grid_height=H, grid_width=W,
                    num_timesteps=N_SCENARIO_DATA_STEPS,
                )
                results = check_ground_sensor_detection(
                    scenario, ground_locs_data, offset)
                results["routed"] = False
                n_sensor_only += 1
                if results["device"] == "undetected":
                    n_skipped += 1

            else:
                rlog = routing_logs[cluster["fingerprint"]]
                total_substeps_needed = (offset + N_SCENARIO_DATA_STEPS) * operational_substeps

                if not rlog.has(STATIC_LOG_KEY, total_substeps_needed):
                    # Build cluster-specific auto params
                    cluster_auto = {
                        **base_rescaled_auto,
                        "n_drones":                    cluster["n_drones"],
                        "n_charging_stations":         len(cluster["stations_opt"]),
                        "ground_sensor_locations":     ground_locs_opt,
                        "charging_stations_locations": cluster["stations_opt"],
                    }
                    routing_custom = {**combo_custom, "burnmap_filename": static_bm_path}

                    initial_norm, actions_hist = compute_routing(
                        RoutingCls, cluster_auto, routing_custom,
                        MAX_ROUTING_DATA_STEPS, operational_substeps,
                    )
                    rlog.put(STATIC_LOG_KEY, initial_norm, actions_hist)
                    n_routed += 1
                    print(
                        f"  [static] cluster={cluster['fingerprint'][:12]}… "
                        f"routing computed ({len(actions_hist)} substeps)", flush=True)
                else:
                    n_cached += 1

                cluster_charging_data = [
                    (x * coverage_w + coverage_w // 2, y * coverage_w + coverage_w // 2)
                    for x, y in cluster["stations_opt"]
                ]
                scenario = load_scenario_npy(
                    str(sf), grid_height=H, grid_width=W,
                    num_timesteps=N_SCENARIO_DATA_STEPS,
                )
                results = run_simulation(
                    scenario          = scenario,
                    starting_time     = offset,
                    routing_entry     = rlog.get(STATIC_LOG_KEY),
                    ground_locs_data  = ground_locs_data,
                    charging_locs_data= cluster_charging_data,
                    N=H, M=W,
                    coverage_width_cells  = coverage_w,
                    operational_substeps  = operational_substeps,
                    max_battery_distance  = SIMULATION_PARAMETERS["max_battery_distance"],
                    max_battery_time      = SIMULATION_PARAMETERS["max_battery_time"],
                )
                results["routed"] = True

            all_results.append({
                "strategy_combo": combo_name,
                "scenario_name":  name,
                "date":           date_str,
                "sim_start_hour": "static",
                "log_key":        STATIC_LOG_KEY,
                "offset":         offset,
                "cluster":        cluster["fingerprint"] if cluster else "none",
                **results,
            })

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
        csv_path  = f"benchmark_results_pyrologix_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}", flush=True)

        for cname, gdf in df.groupby("strategy_combo"):
            detected = (gdf["delta_t"] != -1).sum()
            n        = len(gdf)
            mean_dt  = gdf.loc[gdf["delta_t"] != -1, "delta_t"].mean()
            print(
                f"  {cname}: detection rate={detected/n*100:.1f}%  "
                f"mean delta_t={mean_dt:.2f} (detected)  n={n}",
                flush=True,
            )


def run_sensor_placement_only_pyrologix(budgets=(20, 100, 500), time_limit_seconds=600.0):
    """Run Budget sensor placement on Pyrologix map for each budget; save to pyrologix-tagged caches."""
    LOG_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)
    print("Loading WFPI mask ...", flush=True)
    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
    print("Preparing Pyrologix risk map ...", flush=True)
    pyrologix_2d = resample_pyrologix(PYROLOGIX_RAW, (H, W), PYROLOGIX_RESAMPLED)

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
    pyrologix_masked = pyrologix_2d * mask
    rescaled_avg = pool_burnmap_mean(pyrologix_masked[np.newaxis], coverage_w)
    rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
    rescaled_avg_path = str(PYROLOGIX_RESAMPLED).replace(".npy", f"_{SENSOR_POOLING}{suffix}")
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
        "mask_filename":        rescaled_mask_path,
        "recompute_logfile":    True,
        "recompute_kernel":     False,
        "use_linf_cost":        True,
        "regularization_param": 1e5,
    }

    for budget in budgets:
        combo_name = f"GaussianBudget{int(budget)}M"
        sensor_log_path = str(LOG_DIR / f"sensor_alloc_{combo_name}_{rescaled_N}x{rescaled_M}_pyrologix.json")
        sensor_custom = {
            **base_custom,
            "burnmap_filename":   rescaled_avg_path,
            "budget_millions":   float(budget),
            "cost_sensor":       COST_SENSOR / 1_000_000,
            "cost_station":      COST_STATION / 1_000_000,
            "cost_drone":        COST_DRONE / 1_000_000,
            "time_limit_seconds": time_limit_seconds,
        }
        print(f"\n{'='*70}", flush=True)
        print(f"  PYROLOGIX sensor placement: budget={budget}M, time limit={time_limit_seconds}s", flush=True)
        print(f"{'='*70}\n", flush=True)
        load_or_compute_sensor_placement(
            SensorPlacementMaxCoverageGaussianTimeMaskedBudget,
            base_rescaled_auto,
            sensor_custom,
            sensor_log_path,
        )
    print("\nPyrologix sensor placement done for budgets:", budgets, flush=True)


if __name__ == "__main__":
    if "--sensor-only" in sys.argv:
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
        run_sensor_placement_only_pyrologix(budgets=budgets, time_limit_seconds=time_limit)
        sys.exit(0)
    main()
