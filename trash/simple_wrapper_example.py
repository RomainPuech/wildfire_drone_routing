# Simple example using wrapper functions with RandomDroneRoutingStrategy

# Import necessary modules
import os
import sys
import time
import numpy as np
import random

# Add code path
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)

# Import wrapper functions and strategies
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomSensorPlacementStrategy, LoggedDroneRoutingStrategy, RandomDroneRoutingStrategy
from benchmark import run_benchmark_scenario, return_no_custom_parameters
from dataset import load_scenario_npy

# Create the wrapped strategies
WrappedSensorStrategy = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
WrappedDroneStrategy = wrap_log_drone_strategy(RandomDroneRoutingStrategy)

# Define automatic layout parameters function
def my_automatic_layout_parameters(scenario):
    N, M = scenario.shape[1], scenario.shape[2]
    n_charging_stations = 3
    
    # Generate charging station locations as integer tuples
    charging_stations = []
    margin = 2  # Keep stations away from edges
    for _ in range(n_charging_stations):
        x = random.randint(margin, N - margin)
        y = random.randint(margin, M - margin)
        charging_stations.append((x, y))
    
    return {
        "N": N,
        "M": M,
        "max_battery_distance": 20,
        "max_battery_time": 20,
        "n_drones": 5,
        "n_ground_stations": 3,
        "n_charging_stations": n_charging_stations,
        "charging_stations_locations": charging_stations,  # List of integer tuples
    }

# Define custom initialization parameters function
def custom_initialization_parameters_function(input_dir, layout_name=None):
    if isinstance(input_dir, dict):
        # Handle the case where input_dir is actually automatic_params
        # (happens when called from benchmark.py)
        dir_path = input_dir.get("input_dir", ".")
    else:
        dir_path = input_dir
    
    # Extract path components
    path_parts = dir_path.strip('/').split('/')
    if 'scenarii' in path_parts:
        base_dir = '/'.join(path_parts[:path_parts.index('scenarii')])
    else:
        base_dir = '/'.join(path_parts[:-1]) if path_parts else "."
    
    # Create logs directory
    log_dir = f"{base_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    return {
        "burnmap_filename": f"{base_dir}/burn_map.npy",
        "reevaluation_step": 5,
        "optimization_horizon": 5,
        "call_every_n_steps": 5,
        "log_file": f"{log_dir}/{layout_name or 'default'}_log.json"
    }

# Ensure log directories exist
for layout in ["0001", "0002"]:
    os.makedirs(f"MinimalDataset/{layout}/logs", exist_ok=True)

# Patch to fix the wrappers.py issue with set multiplication
def list_wrap_for_automatic_params(scenario, auto_params_func):
    """Wrapper to ensure charging_stations_locations is properly formatted"""
    params = auto_params_func(scenario)
    if "charging_stations_locations" in params:
        # Ensure charging stations are integer tuples
        charging_stations = params["charging_stations_locations"]
        if not isinstance(charging_stations, list):
            charging_stations = list(charging_stations)
        
        # Convert all coordinates to integers
        charging_stations = [
            (int(x), int(y)) if isinstance(x, (str, int)) and isinstance(y, (str, int)) else (x, y)
            for x, y in charging_stations
        ]
        params["charging_stations_locations"] = charging_stations
    return params

if __name__ == "__main__":
    # Test with a single scenario first
    print("Running benchmark with wrapped strategies on a single scenario...")
    scenario_path = "MinimalDataset/0001/scenarii/0001_00002.npy"
    scenario = load_scenario_npy(scenario_path)
    
    # Run benchmark with parameter wrapping to ensure lists
    start_time = time.time()
    results, _ = run_benchmark_scenario(
        scenario, 
        WrappedSensorStrategy, 
        WrappedDroneStrategy, 
        custom_initialization_parameters=custom_initialization_parameters_function(scenario_path),
        custom_step_parameters_function=return_no_custom_parameters,
        automatic_initialization_parameters_function=lambda s: list_wrap_for_automatic_params(s, my_automatic_layout_parameters),
        return_history=False
    )
    
    print(f"Benchmark completed in {time.time() - start_time:.2f} seconds")
    print(f"Results: {results}")
    
    print("\nIf this works, you can uncomment the code below to run on multiple scenarios:")
    """
    from benchmark import benchmark_on_sim2real_dataset_precompute
    
    benchmark_on_sim2real_dataset_precompute(
        "MinimalDataset/",
        WrappedSensorStrategy, 
        WrappedDroneStrategy,
        custom_initialization_parameters_function,
        return_no_custom_parameters,
        max_n_scenarii=10,
        starting_time=0
    )
    """ 