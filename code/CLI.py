

# import requred modules
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor
from benchmark import benchmark_on_sim2real_dataset_precompute, return_no_custom_parameters
# from displays import create_scenario_video
from new_clustering import get_wrapped_strategy


PLACEMENT_STRATEGY_TO_TEST = RandomSensorPlacementStrategy
DRONE_STRATEGY_TO_TEST = get_wrapped_strategy(DroneRoutingOptimizationModelReuseIndex)


# change values here to change benchmarking parameters

default_parameters = {
    "max_battery_distance": -1,
    "max_battery_time": 6,
    "n_drones": 3,
    "n_ground_stations": 1,
    "n_charging_stations": 2,
}

def custom_initialization_parameters_function(input_dir:str):
    print(f"input_dir: {input_dir}")
    return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy", "reevaluation_step": 5, "optimization_horizon":5, "strategy_drone": DroneRoutingOptimizationModelReuseIndex, "strategy_sensor": RandomSensorPlacementStrategy}


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmarking script for strategies')
    parser.add_argument('--dataset', type=str, default="MinimalDataset/", help='Dataset folder path')
    parser.add_argument('--n_drones', type=int, default=default_parameters["n_drones"], help='Number of drones')
    parser.add_argument('--n_ground_stations', type=int, default=default_parameters["n_ground_stations"], help='Number of ground stations')
    parser.add_argument('--n_charging_stations', type=int, default=default_parameters["n_charging_stations"], help='Number of charging stations')
    parser.add_argument('--max_battery_distance', type=int, default=default_parameters["max_battery_distance"], help='Maximum battery distance')
    parser.add_argument('--max_battery_time', type=int, default=default_parameters["max_battery_time"], help='Maximum battery time')
    args = parser.parse_args()

    print(f"Running benchmark for dataset: {args.dataset}")
    
    metrics_by_layout = benchmark_on_sim2real_dataset_precompute(
    dataset_folder_name=args.dataset,
    ground_placement_strategy=PLACEMENT_STRATEGY_TO_TEST,
    drone_routing_strategy=DRONE_STRATEGY_TO_TEST,
    custom_initialization_parameters_function=custom_initialization_parameters_function,
    custom_step_parameters_function=return_no_custom_parameters,
    max_n_scenarii=1,
    starting_time=0,
    simulation_parameters = {
        "n_drones": args.n_drones,
        "n_ground_stations": args.n_ground_stations,
        "n_charging_stations": args.n_charging_stations,
        "max_battery_distance": args.max_battery_distance,
        "max_battery_time": args.max_battery_time,
    }
)
    
    print("Done!")

    print(metrics_by_layout)