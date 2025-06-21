# sequential_runner.py

import os
import time

from benchmark import run_benchmark_scenarii_sequential
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomSensorPlacementStrategy, DroneRoutingUniformMaxCoverageResetStatic
from new_clustering import get_wrapped_clustering_strategy

# --- Configuration --------------------------------------------------------

# Folder containing your .npy scenarii
input_dir = "./MinimalDataset/0001/scenarii"
file_format = "npy"

# Simulation parameters
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

# Wrap and select your strategies
sensor_strategy = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
drone_strategy = wrap_log_drone_strategy(
    get_wrapped_clustering_strategy(DroneRoutingUniformMaxCoverageResetStatic)
)

# Custom initialization parameters (takes only input_dir; sequential runner will detect this)
def custom_initialization_parameters_function(input_dir: str):
    return {
        "burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy",
        "reevaluation_step": 2,
        "optimization_horizon": 2,
        "regularization_param": 1,
    }

# No per‐step overrides
custom_step_parameters_function = lambda: {}

# Optional config for per‐scenario offsets (e.g. {"offset_scenario1.npy": 5})
config = {}

# --------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()

    metrics = run_benchmark_scenarii_sequential(
        input_dir=input_dir,
        sensor_placement_strategy=sensor_strategy,
        drone_routing_strategy=drone_strategy,
        custom_initialization_parameters_function=custom_initialization_parameters_function,
        custom_step_parameters_function=custom_step_parameters_function,
        # tweak any of these if needed:
        file_format=file_format,
        simulation_parameters=simulation_parameters,
    )

    elapsed = time.time() - start
    print(f"\nSequential benchmark completed in {elapsed:.2f} seconds\n")

    print("Aggregated Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")


