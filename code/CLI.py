

# import requred modules
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor
from benchmark import benchmark_on_sim2real_dataset_precompute
# from displays import create_scenario_video
from new_clustering import ClusteredDroneStrategyWrapper

PLACEMENT_STRATEGY_TO_TEST = RandomSensorPlacementStrategy
DRONE_STRATEGY_TO_TEST = RandomDroneRoutingStrategy


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmarking script for strategies')
    parser.add_argument('--dataset', type=str, required=True, default="MinimalDataset/", help='Dataset folder path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    metrics_by_layout = benchmark_on_sim2real_dataset_precompute(
    dataset_folder_name=args.dataset,
    ground_placement_strategy=PLACEMENT_STRATEGY_TO_TEST,
    drone_routing_strategy=DRONE_STRATEGY_TO_TEST,
    custom_initialization_parameters_function=custom_initialization_parameters_function,
    custom_step_parameters_function=return_no_custom_parameters,
    max_n_scenarii=1,
    starting_time=0
)
    
    print("Done!")