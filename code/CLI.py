

# import requred modules
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json


from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, DroneRoutingOptimizationModelReuseIndexRegularized, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor
from benchmark import benchmark_on_sim2real_dataset_precompute, return_no_custom_parameters
# from displays import create_scenario_video
from new_clustering import get_wrapped_strategy
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy

PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(RandomDroneRoutingStrategy)


# change values here to change benchmarking parameters

default_parameters =  {
    "max_battery_distance": -1,
    "max_battery_time": 20,
    "n_drones": 20,
    "n_ground_stations": 12,
    "n_charging_stations": 8,
}

def run(placement_strategy:str, drone_strategy:str, default_parameters:dict, custom_initialization_parameters_function:callable):
    dataset = "WideDataset"
    n_drones = default_parameters["n_drones"]
    n_ground_stations = default_parameters["n_ground_stations"]
    n_charging_stations = default_parameters["n_charging_stations"]
    max_battery_distance = default_parameters["max_battery_distance"]
    max_battery_time = default_parameters["max_battery_time"]

    print(f"Running benchmark for strategy: {placement_strategy.strategy_name} and {drone_strategy.strategy_name}")

    start_time = time.time()
    
    metrics_by_layout = benchmark_on_sim2real_dataset_precompute(
    dataset_folder_name=dataset,
    ground_placement_strategy=placement_strategy,
    drone_routing_strategy=drone_strategy,
    custom_initialization_parameters_function=custom_initialization_parameters_function,
    custom_step_parameters_function=return_no_custom_parameters,
    max_n_scenarii=100,
    starting_time=0,
    simulation_parameters = {
        "n_drones": n_drones,
        "n_ground_stations": n_ground_stations,
        "n_charging_stations": n_charging_stations,
        "max_battery_distance": max_battery_distance,
        "max_battery_time": max_battery_time,
    })
    end_time = time.time()
    total_time = end_time - start_time
    metrics_by_layout["total_time"] = total_time
    # save them in a json file
    custom_params = custom_initialization_parameters_function(dataset)
    optimization_horizon = custom_params.get("optimization_horizon", 0)
    reevaluation_step = custom_params.get("reevaluation_step", 0)
    regularization_param = custom_params.get("regularization_param", 0)

    logfile_name = f"results/{placement_strategy.strategy_name}_{drone_strategy.strategy_name}_{"_".join(dataset.strip('/').split('/'))}_{n_drones}_drones_{n_ground_stations}_ground_stations_{n_charging_stations}_charging_stations_{max_battery_distance}_max_battery_distance_{max_battery_time}_max_battery_time_{optimization_horizon}_optimization_horizon_{reevaluation_step}_reevaluation_step_{regularization_param}_regularization_param.json"
    with open(logfile_name, "w") as f:
        json.dump(str(metrics_by_layout), f)

    print(metrics_by_layout)
    print("Done!")
    print(f"Time taken: {end_time - start_time} seconds")
    

def delete_all_logs(dataset:str):
    for layout in os.listdir(f"{dataset}"):
        if layout == ".DS_Store":
            continue
        # check if log directory exists
        if not os.path.exists(f"{dataset}/{layout}/logs/"):
            continue
        for logfile in os.listdir(f"{dataset}/{layout}/logs/"):
            if logfile == ".DS_Store":
                continue
            print(f"Deleting {dataset}/{layout}/logs/{logfile}")
            os.remove(f"{dataset}/{layout}/logs/{logfile}")




if __name__ == "__main__":
    # delete_all_logs("WideDataset")
    #exit()
    # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Benchmarking script for strategies')
    # parser.add_argument('--dataset', type=str, default="WideDataset/", help='Dataset folder path')
    # parser.add_argument('--n_drones', type=int, default=default_parameters["n_drones"], help='Number of drones')
    # parser.add_argument('--n_ground_stations', type=int, default=default_parameters["n_ground_stations"], help='Number of ground stations')
    # parser.add_argument('--n_charging_stations', type=int, default=default_parameters["n_charging_stations"], help='Number of charging stations')
    # parser.add_argument('--max_battery_distance', type=int, default=default_parameters["max_battery_distance"], help='Maximum battery distance')
    # parser.add_argument('--max_battery_time', type=int, default=default_parameters["max_battery_time"], help='Maximum battery time')
    # args = parser.parse_args()

    # print(f"Running benchmark for dataset: {args.dataset}")

    # print("TEST 1 - RANDOM RANDOM")
    # PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
    # DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(RandomDroneRoutingStrategy)
    # run(PLACEMENT_STRATEGY_TO_TEST, DRONE_STRATEGY_TO_TEST, default_parameters)

    def greedy_custom_initialization_parameters_function(input_dir:str):
    # print(f"input_dir: {input_dir}")
        return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy", "reevaluation_step": 2, "optimization_horizon":2, "regularization_param": 0.0001}


    print("TEST 2 - Fixed placement, Greedy Optimization Drone")

    PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
    DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingOptimizationModelReuseIndex))
    
    run(PLACEMENT_STRATEGY_TO_TEST, DRONE_STRATEGY_TO_TEST, default_parameters, greedy_custom_initialization_parameters_function)

    
    
    
    def regular_custom_initialization_parameters_function(input_dir:str):
    # print(f"input_dir: {input_dir}")
        return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy", "reevaluation_step": 6, "optimization_horizon":6, "regularization_param": 0.0001}
     
    print("TEST 3 - Fixed placement, Optimization Drone")

    PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
    DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingOptimizationModelReuseIndex))
    try:
        run(PLACEMENT_STRATEGY_TO_TEST, DRONE_STRATEGY_TO_TEST, default_parameters, regular_custom_initialization_parameters_function)
    except Exception as e:
        print(f"Error: {e}")


    print("TEST 4 - Fixed placement, Optimization Drone (Regularized)")

    PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
    DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingOptimizationModelReuseIndexRegularized))
    try:
        run(PLACEMENT_STRATEGY_TO_TEST, DRONE_STRATEGY_TO_TEST, default_parameters, regular_custom_initialization_parameters_function)
    except Exception as e:
        print(f"Error: {e}")



    def regularized_custom_initialization_parameters_function(input_dir:str):
    # print(f"input_dir: {input_dir}")
        return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy", "reevaluation_step": 6, "optimization_horizon":6, "regularization_param": 1}
     
    print("TEST 5 - Fixed placement, Optimization Drone (VERY Regularized)")

    PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
    DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingOptimizationModelReuseIndexRegularized))
    try:
        run(PLACEMENT_STRATEGY_TO_TEST, DRONE_STRATEGY_TO_TEST, default_parameters, regularized_custom_initialization_parameters_function)
    except Exception as e:
        print(f"Error: {e}")


    print("TEST X - Optim Sensor, Random Drone (Long time to run)")

    PLACEMENT_STRATEGY_TO_TEST = wrap_log_sensor_strategy(SensorPlacementOptimization)
    DRONE_STRATEGY_TO_TEST = wrap_log_drone_strategy(RandomDroneRoutingStrategy)
    try:
        pass
        #run(PLACEMENT_STRATEGY_TO_TEST, DRONE_STRATEGY_TO_TEST, default_parameters, regular_custom_initialization_parameters_function)
    except Exception as e:
        print(f"Error: {e}")


    



    
