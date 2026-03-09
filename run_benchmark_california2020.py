#!/usr/bin/env python3
"""
Benchmark: Multiple strategy combinations on California 2020 mini datasets.

This script benchmarks strategies on the mini California 2020 datasets which use
ignition-point-only scenario format. It handles loading scenarios with grid dimensions
from the mask file.

Run from the project root:
    python -u run_benchmark_california2020.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Ensure unbuffered output so Julia debug prints appear in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure we can import from the local `code` and `julia` directories
sys.path.append(str(Path(__file__).resolve().parent / "code"))
sys.path.append(str(Path(__file__).resolve().parent / "julia"))

print("Importing modules...", flush=True)
from displays import create_video_scenario_burnmap
from dataset import load_scenario_npy, load_burn_map
from benchmark import run_benchmark_scenario, return_no_custom_parameters
import wrappers
print("Imports done.\n", flush=True)

# === Configuration: Define all strategy combinations to test ===
STRATEGY_COMBINATIONS = [
    {
        "name": "Gaussian_TOP",
        "sensor": wrappers.SensorPlacementMaxCoverageGaussianTimeMasked,
        "drone": wrappers.DroneRoutingTOPMaskedLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10}
    },
    {
        "name": "Gaussian_GrowingTOP",
        "sensor": wrappers.SensorPlacementMaxCoverageGaussianTimeMasked,
        "drone": wrappers.DroneRoutingTOPMaskedGrowingProbaLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10}
    },
    {
        "name": "Random_GrowingTOP",
        "sensor": wrappers.RandomSensorPlacementStrategyLogged,
        "drone": wrappers.DroneRoutingTOPMaskedGrowingProbaLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10}
    },
    {
        "name": "Random_TOP",
        "sensor": wrappers.RandomSensorPlacementStrategyLogged,
        "drone": wrappers.DroneRoutingTOPMaskedLogged,
        "params": {"reevaluation_step": 5, "optimization_horizon": 10}
    },
    {
        "name": "Random_MaxCoverageGrowingMasked",
        "sensor": wrappers.RandomSensorPlacementStrategyLogged,
        "drone": wrappers.DroneRoutingMaxCoverageGrowingMaskedLogged,
        "params": {"reevaluation_step": 6, "optimization_horizon": 10}
    },
    {
        "name": "Gaussian_MaxCoverageGrowingMasked",
        "sensor": wrappers.SensorPlacementMaxCoverageGaussianTimeMasked,
        "drone": wrappers.DroneRoutingMaxCoverageGrowingMaskedLogged,
        "params": {"reevaluation_step": 6, "optimization_horizon": 10}
    },
]

# === Dataset and simulation parameters ===
DATASET_DIR = Path("MiniCalifornia2020Datasets")
DATASETS = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
print(f"Found {len(DATASETS)} datasets: {DATASETS}\n", flush=True)

simulation_parameters = {
    "max_battery_distance": -1,
    "max_battery_time": 1,
    "n_drones": 2,
    "n_ground_stations": 8,
    "n_charging_stations": 2,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m": 2900,
    "cell_size_m": 30,
    "transmission_range": 50000,
    "mask_pooling_mode": "max",  # "min" = masked if any data cell masked; "max" = masked only if all data cells masked
}

print(f"Simulation parameters: {simulation_parameters}\n", flush=True)
print(f"Will run {len(STRATEGY_COMBINATIONS)} strategy combinations:\n", flush=True)
for i, combo in enumerate(STRATEGY_COMBINATIONS, 1):
    print(f"  {i}. {combo['name']}: {combo['sensor'].__name__} x {combo['drone'].__name__}", flush=True)
print("\n", flush=True)

# === Storage for all results ===
all_results = []

# === Main benchmark loop ===
for combo_idx, combo in enumerate(STRATEGY_COMBINATIONS, 1):
    combo_name = combo["name"]
    SensorPlacementStrategy = combo["sensor"]
    DroneRoutingStrategy = combo["drone"]
    combo_params = combo["params"]
    
    print("\n" + "=" * 80, flush=True)
    print(f"  STRATEGY COMBINATION {combo_idx}/{len(STRATEGY_COMBINATIONS)}: {combo_name}", flush=True)
    print(f"  Sensor: {SensorPlacementStrategy.__name__}", flush=True)
    print(f"  Drone:  {DroneRoutingStrategy.__name__}", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    for dataset_name in DATASETS:
        dataset_dir = DATASET_DIR / dataset_name
        scenario_dir = dataset_dir / "scenarii"
        scenarios = sorted(scenario_dir.glob("*.npy"))
        
        print("-" * 70, flush=True)
        print(f"  DATASET: {dataset_name}  ({len(scenarios)} scenario(s))", flush=True)
        print("-" * 70, flush=True)
        
        # Load mask to get grid dimensions (needed for ignition-point scenarios)
        mask_path = dataset_dir / "mask.npy"
        if not mask_path.exists():
            print(f"  ERROR: mask.npy not found in {dataset_dir}", flush=True)
            continue
        
        mask = np.load(str(mask_path))
        grid_height, grid_width = mask.shape
        print(f"  Grid dimensions: {grid_height} x {grid_width}", flush=True)
        
        # Find risk map (burn map) - could be static_risk_*.npy or wfpi_*.npy files
        risk_map_path = None
        static_risk_files = list(dataset_dir.glob("static_risk_*.npy"))
        if static_risk_files:
            risk_map_path = static_risk_files[0]  # Use first static risk map found
            print(f"  Using static risk map: {risk_map_path.name}", flush=True)
        else:
            # For WFPI datasets, we'll need to load the appropriate WFPI map per scenario
            # For now, we'll use the first WFPI file as a placeholder
            wfpi_files = list(dataset_dir.glob("wfpi_*.npy"))
            if wfpi_files:
                risk_map_path = wfpi_files[0]
                print(f"  Using WFPI map: {risk_map_path.name} (note: should match scenario date)", flush=True)
            else:
                print(f"  WARNING: No risk map found in {dataset_dir}", flush=True)
        
        # Load config file for offsets
        config_path = dataset_dir / f"config_california_2020*.json"
        config_files = list(dataset_dir.glob("config_*.json"))
        config = {}
        if config_files:
            with open(config_files[0], 'r') as f:
                config = json.load(f)
            print(f"  Loaded config from {config_files[0].name}", flush=True)
        
        for scenario_file in scenarios:
            scenario_name = scenario_file.stem
            print(f"\n  Scenario: {scenario_name}", flush=True)
            
            # Load scenario (auto-detects ignition-point format)
            scenario = load_scenario_npy(
                str(scenario_file),
                grid_height=grid_height,
                grid_width=grid_width,
                num_timesteps=12
            )
            print(f"  Scenario shape: {scenario.shape}", flush=True)
            
            # Get offset from config
            fire_key = scenario_name.replace('_scenario1', '')
            offset = config.get(f"offset_{fire_key}", 0)
            starting_time = offset
            
            # For WFPI datasets, we should load the matching WFPI map based on fire date
            # For now, we'll use the risk_map_path we found above
            burnmap_filename = str(risk_map_path) if risk_map_path else None
            
            custom_initialization_parameters = {
                "burnmap_filename": burnmap_filename,
                "mask_filename": str(mask_path),
                "load_from_logfile": False,
                "recompute_logfile": False,  # set True to force recomputation (bypasses cached logs)
                "reevaluation_step": combo_params["reevaluation_step"],
                "optimization_horizon": combo_params["optimization_horizon"],
                "regularization_param": 1e5,
                "recompute_kernel": False,
                "use_linf_cost": True,  # Use L∞ (Chebyshev) cost model instead of binary adjacency
            }
            
            print(f"  Running benchmark (offset: {offset})...", flush=True)
            try:
                results, history = run_benchmark_scenario(
                    scenario=scenario,
                    sensor_placement_strategy=SensorPlacementStrategy,
                    drone_routing_strategy=DroneRoutingStrategy,
                    custom_initialization_parameters=custom_initialization_parameters,
                    custom_step_parameters_function=return_no_custom_parameters,
                    starting_time=starting_time,
                    return_history=True,
                    return_history_scale='operational',
                    input_dir=str(dataset_dir) + "/",
                    simulation_parameters=simulation_parameters,
                    scenario_name=scenario_name,
                )
                
                drone_locations_history, ground_sensor_locations, charging_stations_locations = history
                print(f"  Benchmark complete. Total timesteps: {len(drone_locations_history)}", flush=True)
                
                # Print per-scenario results
                print(f"  Results:", flush=True)
                for key, value in results.items():
                    print(f"    {key}: {value}", flush=True)
                
                # Store results with metadata
                result_row = {
                    "strategy_combo": combo_name,
                    "sensor_strategy": SensorPlacementStrategy.__name__,
                    "drone_strategy": DroneRoutingStrategy.__name__,
                    "dataset_name": dataset_name,
                    "scenario_name": scenario_name,
                    **results
                }
                all_results.append(result_row)
                
                # --- Render video using the latest tmp_burnmap ---
                tmp_dir = Path("tmp_burnmaps")
                tmp_files = sorted(tmp_dir.glob("tmp_burnmap_*.npy"), key=lambda f: f.stat().st_mtime, reverse=True)
                if tmp_files:
                    latest_burnmap = str(tmp_files[0])
                    print(f"  Rendering video (burn map: {latest_burnmap})...", flush=True)
                    
                    burn_map_video = load_burn_map(latest_burnmap)[:len(drone_locations_history)]
                    
                    substeps = results.get("substeps_per_timestep", 1)
                    out_name = f"benchmark_{combo_name}_{dataset_name}_{scenario_name}"
                    
                    # Load mask (data scale) for gray overlay
                    mask_data = np.load(str(mask_path))
                    
                    # Compute coverage_width_cells to match the simulation pooling exactly
                    cwc = round(simulation_parameters["coverage_radius_m"] * 2 / simulation_parameters["cell_size_m"])
                    
                    create_video_scenario_burnmap(
                        burn_map=burn_map_video,
                        drone_locations_history=drone_locations_history,
                        out_filename=out_name,
                        ground_sensor_locations=ground_sensor_locations,
                        charging_stations_locations=charging_stations_locations,
                        frames_per_image=3,
                        maxframes=np.inf,
                        display_zones=True,
                        fire_scenario=scenario,
                        substeps_per_timestep=substeps,
                        mask=mask_data,
                        coverage_width_cells=cwc,
                        mask_pooling_mode=simulation_parameters.get("mask_pooling_mode", "min"),
                    )
                    print(f"  Video saved to display_{out_name}/{out_name}.mp4", flush=True)
                else:
                    print(f"  WARNING: No tmp_burnmap files found. Skipping video.", flush=True)
                    
            except Exception as e:
                print(f"  ERROR running benchmark: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Store error result
                result_row = {
                    "strategy_combo": combo_name,
                    "sensor_strategy": SensorPlacementStrategy.__name__,
                    "drone_strategy": DroneRoutingStrategy.__name__,
                    "dataset_name": dataset_name,
                    "scenario_name": scenario_name,
                    "error": str(e)
                }
                all_results.append(result_row)

# === Generate and save results table ===
print("\n\n" + "=" * 80, flush=True)
print("  GENERATING RESULTS TABLE", flush=True)
print("=" * 80, flush=True)

if all_results:
    df = pd.DataFrame(all_results)
    
    # Reorder columns to put metadata first
    metadata_cols = ["strategy_combo", "sensor_strategy", "drone_strategy", "dataset_name", "scenario_name"]
    other_cols = [c for c in df.columns if c not in metadata_cols]
    df = df[metadata_cols + other_cols]
    
    # Compute averaged results for each strategy combo and dataset
    print("\nComputing averaged results per strategy combo and dataset...", flush=True)
    averaged_rows = []
    
    for combo_name in df["strategy_combo"].unique():
        for dataset_name in df["dataset_name"].unique():
            combo_df = df[(df["strategy_combo"] == combo_name) & (df["dataset_name"] == dataset_name)].copy()
            
            # Skip rows with errors
            if "error" in combo_df.columns:
                combo_df_clean = combo_df[combo_df["error"].isna()]
            else:
                combo_df_clean = combo_df
            
            if len(combo_df_clean) == 0:
                continue
            
            # Get the first row for metadata (same for all rows in a combo)
            first_row = combo_df_clean.iloc[0]
            
            # Create averaged row
            avg_row = {
                "strategy_combo": combo_name,
                "sensor_strategy": first_row["sensor_strategy"],
                "drone_strategy": first_row["drone_strategy"],
                "dataset_name": dataset_name,
                "scenario_name": f"AVERAGE (n={len(combo_df_clean)})",
            }
            
            # Compute detection rate: fraction of runs where fire was detected (delta_t != -1)
            n_total = len(combo_df_clean)
            if "delta_t" in combo_df_clean.columns:
                n_detected = (combo_df_clean["delta_t"] != -1).sum()
                avg_row["detection_rate"] = n_detected / n_total if n_total > 0 else np.nan
            
            # For averaging, replace delta_t == -1 with 0 (undetected = no delay, but counted as 0)
            combo_df_avg = combo_df_clean.copy()
            if "delta_t" in combo_df_avg.columns:
                combo_df_avg.loc[combo_df_avg["delta_t"] == -1, "delta_t"] = 0
            
            # Compute averages for numeric columns (skip metadata and error columns)
            exclude_cols = set(metadata_cols + ["error"])
            for col in combo_df_avg.columns:
                if col in exclude_cols:
                    continue
                
                if combo_df_avg[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                    # Numeric column: compute mean, handling NaNs
                    avg_val = combo_df_avg[col].mean(skipna=True)
                    if pd.isna(avg_val):
                        avg_row[col] = np.nan
                    else:
                        avg_row[col] = avg_val
                elif col == "device":
                    # For device column, show the most common value or "mixed" if multiple
                    devices = combo_df_avg[col].dropna().unique()
                    if len(devices) == 1:
                        avg_row[col] = devices[0]
                    elif len(devices) > 1:
                        avg_row[col] = "mixed"
                    else:
                        avg_row[col] = np.nan
                else:
                    # For other non-numeric columns, leave empty or use a placeholder
                    avg_row[col] = ""
            
            averaged_rows.append(avg_row)
    
    # Combine original results with averaged results
    avg_df = None
    if averaged_rows:
        avg_df = pd.DataFrame(averaged_rows)
        # Ensure all columns from df are present in avg_df (fill missing with empty string or NaN)
        for col in df.columns:
            if col not in avg_df.columns:
                avg_df[col] = "" if df[col].dtype == 'object' else np.nan
        # Reorder columns to match original df
        avg_df = avg_df[df.columns]
        # Append averaged rows to the dataframe
        df = pd.concat([df, avg_df], ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_california2020_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}", flush=True)
    if averaged_rows:
        print(f"  (Includes {len(averaged_rows)} averaged rows)", flush=True)
    
    # Print summary table
    print("\n" + "=" * 80, flush=True)
    print("  SUMMARY TABLE", flush=True)
    print("=" * 80, flush=True)
    
    # Group by strategy combo and dataset, show key metrics
    if "delta_t" in df.columns:
        summary_cols = ["strategy_combo", "dataset_name", "delta_t", "device"]
        if "detection_rate" in df.columns:
            summary_cols.append("detection_rate")
        if "fire_size_cells" in df.columns:
            summary_cols.append("fire_size_cells")
        if "fire_size_percentage" in df.columns:
            summary_cols.append("fire_size_percentage")
        
        summary_df = df[summary_cols].copy()
        print("\n", summary_df.to_string(index=False), flush=True)
        
        # Print averaged results separately
        if averaged_rows:
            print("\n" + "=" * 80, flush=True)
            print("  AVERAGED RESULTS BY STRATEGY COMBO AND DATASET", flush=True)
            print("=" * 80, flush=True)
            avg_summary_df = avg_df[summary_cols].copy()
            print("\n", avg_summary_df.to_string(index=False), flush=True)
    
    # Print full results
    print("\n\n" + "=" * 80, flush=True)
    print("  FULL RESULTS", flush=True)
    print("=" * 80, flush=True)
    print("\n", df.to_string(index=False), flush=True)
    
else:
    print("No results collected.", flush=True)

print("\nDone.", flush=True)
