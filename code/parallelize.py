# Import necessary modules
import os
import numpy as np
import sys


from dataset import load_scenario_npy
from benchmark import run_benchmark_scenario
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomSensorPlacementStrategy, DroneRoutingUniformMaxCoverageResetStatic
from new_clustering import get_wrapped_clustering_strategy

# Define paths and parameters
scenario_file = "/Users/josephye/Desktop/UROP/wildfire_drone_routing/MinimalDataset/0001/burn_map.npy"
scenario = load_scenario_npy(scenario_file)

# Define simulation parameters
simulation_parameters = {
    "max_battery_distance": -1,
    "max_battery_time": 1,
    "n_drones": 3,
    "n_ground_stations": 4,
    "n_charging_stations": 2,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m": 300,
    "cell_size_m": 30,
    "transmission_range": 50000,
}

# Define strategies
sensor_strategy = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
drone_strategy = wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingUniformMaxCoverageResetStatic))

# Define custom initialization parameters
def custom_initialization_parameters_function(input_dir: str):
    return {
        "burnmap_filename": os.path.join(input_dir, "burn_map.npy"),
        "reevaluation_step": 5,
        "optimization_horizon": 10,
        "regularization_param": 1,
    }

# Run benchmark on the scenario
results, _ = run_benchmark_scenario(
    scenario,
    sensor_strategy,
    drone_strategy,
    custom_initialization_parameters_function("/Users/josephye/Desktop/UROP/wildfire_drone_routing/MinimalDataset/0001/"), 
    lambda: {},  # No custom step parameters
    simulation_parameters=simulation_parameters,
    return_history=True
)

# Print benchmark results
print("Benchmark Results:")
print(results)