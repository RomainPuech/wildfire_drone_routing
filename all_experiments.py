#!/bin/bash

# import requred modules
import sys
import os
import time
import numpy as np
# Add code to path
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset import preprocess_sim2real_dataset, load_scenario_npy, compute_and_save_burn_maps_sim2real_dataset, load_scenario, combine_all_benchmark_results
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from new_clustering import get_wrapped_clustering_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, RandomSensorPlacementStrategy, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor, DroneRoutingUniformMaxCoverageResetStatic, FixedPlacementStrategy, SensorPlacementMaxCoverageGaussianTime, DroneRoutingMaxCoverageResetStatic, DroneRoutingMaxCoverageResetStaticGreedy
from benchmark import run_benchmark_scenario,run_benchmark_scenarii_sequential, get_burnmap_parameters,run_benchmark_scenarii_sequential_precompute, benchmark_on_sim2real_dataset_precompute
from displays import create_scenario_video

# shared parameters
simulation_parameters =  {
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

custom_initialization_parameters = {
    "load_from_logfile": False, 
    "reevaluation_step": 5, 
    "optimization_horizon":10,
    "regularization_param": 1e5
    } #"regularization_param": 0.0001}

layout_folder = "WideDataset/"
dataset_folder_name = "WideDataset/"

def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
        simulation_parameters["N"] = scenario.shape[1]
        simulation_parameters["M"] = scenario.shape[2]
        return simulation_parameters

def return_no_custom_parameters():
    return {}

def run_one_drone_strategy(sensor_strategy, drone_strategy, custom_initialization_parameters_function, nickname=None):
    strategy_name = sensor_strategy.strategy_name + "_" + drone_strategy.strategy_name
    if nickname is None:
        nickname = strategy_name
    print(f"-- Starting {nickname} --")

    time_start = time.time()
    benchmark_on_sim2real_dataset_precompute(dataset_folder_name, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, max_n_scenarii=None, max_n_layouts=None, simulation_parameters = simulation_parameters, file_format="jpg", config_file="config_s2r.json")
    print(f"Time taken to run benchmark on the scenario: {time.time() - time_start} seconds")
    combine_all_benchmark_results("WideDataset/", strategy_name = strategy_name, nickname = nickname)   
    

def run_all_drone_strategies(sensor_strategy):

    def custom_initialization_parameters_function(input_dir:str):
        print(f"input_dir: {input_dir}")
        return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/static_risk_whp.npy", "reevaluation_step": 5, "optimization_horizon":10, "regularization_param": 1}
    def custom_initialization_parameters_function_greedy(input_dir:str):
        print(f"input_dir: {input_dir}")
        return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/static_risk_whp.npy", "reevaluation_step": 2, "optimization_horizon":2, "regularization_param": 1}
    
    #housekeeping : delete temporary burn maps
    # if os.path.exists("tmp_burnmaps"):
    #     for file in os.listdir("tmp_burnmaps"):
    #         os.remove("tmp_burnmaps/" + file)


    run_one_drone_strategy(sensor_strategy, RandomDroneRoutingStrategy, custom_initialization_parameters_function, "KR")
    run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStatic)), custom_initialization_parameters_function, "KMC")
    run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingUniformMaxCoverageResetStatic)), custom_initialization_parameters_function, "KU")
    run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStaticGreedy)), custom_initialization_parameters_function_greedy, "KG")
    
    

if __name__ == "__main__":

    sensor_strategy = wrap_log_sensor_strategy(SensorPlacementMaxCoverageGaussianTime)
    run_all_drone_strategies(sensor_strategy)

