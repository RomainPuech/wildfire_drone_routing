# Import necessary modules
import os
import numpy as np
import sys
import time

from benchmark import run_benchmark_scenarii_parallel
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomSensorPlacementStrategy, DroneRoutingUniformMaxCoverageResetStatic
from new_clustering import get_wrapped_clustering_strategy

# Define paths and parameters
input_dir = "./MinimalDataset/0001/scenarii"
file_format = "npy"

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
    layout_dir = os.path.abspath(os.path.join(input_dir, ".."))
    return {
        "burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy",
        "reevaluation_step": 2,
        "optimization_horizon": 2,
        "regularization_param": 1,
    }

start_time = time.time()
# Run benchmark across all layouts
metrics = run_benchmark_scenarii_parallel(
    input_dir,
    sensor_strategy,
    drone_strategy,
    custom_initialization_parameters_function,
    lambda: {},  # No custom step parameters
    file_format=file_format,
    simulation_parameters=simulation_parameters,
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Benchmark completed in {elapsed_time:.2f} seconds.")

# Print aggregated metrics
print("Aggregated Metrics Across All Layouts:")
for key, val in metrics.items():
        print(f"  {key}: {val}")