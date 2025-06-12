#!/bin/bash

# import requred modules
print("Starting all_experiments.py")
import sys
import os
import time
import numpy as np
import argparse
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

def run_one_drone_strategy(sensor_strategy, drone_strategy, custom_initialization_parameters_function, experiment_name=""):
    strategy_name = sensor_strategy.strategy_name + "_" + drone_strategy.strategy_name
    
    print(f"-- Starting {experiment_name} --")

    time_start = time.time()
    benchmark_on_sim2real_dataset_precompute(dataset_folder_name, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, max_n_scenarii=None, max_n_layouts=None, simulation_parameters = simulation_parameters, file_format="jpg", config_file="config_s2r.json", experiment_name = experiment_name)
    print(f"Time taken to run benchmark {experiment_name}: {time.time() - time_start} seconds")
    combine_all_benchmark_results("WideDataset/", strategy_name = strategy_name, experiment_name = experiment_name)   
    

def run_all_drone_strategies(sensor_strategy, ss_prefix, bm_prefix):
    print("running experiments with prefix", ss_prefix, bm_prefix)
    bm_prefix_to_bm_name = {"whp":"static_risk_whp.npy", "bm": "burn_map.npy", "bp": "static_risk_bp2024.npy"}

    def custom_initialization_parameters_function(input_dir:str):
        print(f"input_dir: {input_dir}")
        bm_file = f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/{bm_prefix_to_bm_name[bm_prefix]}"
        return {"burnmap_filename": bm_file, "burnmap_type": "dynamic" if bm_file.endswith("burn_map.npy") else "static", "reevaluation_step": 5, "optimization_horizon":10, "regularization_param": 1}
    def custom_initialization_parameters_function_greedy(input_dir:str):
        print(f"input_dir: {input_dir}")
        bm_file = f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/{bm_prefix_to_bm_name[bm_prefix]}"
        return {"burnmap_filename": bm_file, "burnmap_type": "dynamic" if bm_file.endswith("burn_map.npy") else "static", "reevaluation_step": 2, "optimization_horizon":2, "regularization_param": 1}
    
    #housekeeping : delete temporary burn maps
    # if os.path.exists("tmp_burnmaps"):
    #     for file in os.listdir("tmp_burnmaps"):
    #         os.remove("tmp_burnmaps/" + file)


    run_one_drone_strategy(sensor_strategy, RandomDroneRoutingStrategy, custom_initialization_parameters_function, f"{ss_prefix}R{bm_prefix}")
    run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStatic)), custom_initialization_parameters_function, f"{ss_prefix}MC{bm_prefix}")
    run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingUniformMaxCoverageResetStatic)), custom_initialization_parameters_function, f"{ss_prefix}U{bm_prefix}")
    # run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStaticGreedy)), custom_initialization_parameters_function_greedy, "KG")
    
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all drone strategy experiments')
    parser.add_argument('--ss_prefix', type=str, required=True, 
                        help='Sensor strategy prefix (e.g., "S" for sensor)')
    parser.add_argument('--bm_prefix', type=str, required=True, 
                        choices=['whp', 'bm', 'bp'],
                        help='Burn map prefix: "whp" for static_risk_whp.npy, "bm" for burn_map.npy, "bp" for static_risk_bp2024.npy')
    
    args = parser.parse_args()

    sensor_strategy = wrap_log_sensor_strategy(SensorPlacementMaxCoverageGaussianTime)
    run_all_drone_strategies(sensor_strategy, args.ss_prefix, args.bm_prefix)
    # print size of the following layouts: 265, 319, 320, 321, 323, 337 
    # print(load_scenario("WideDataset/0111_03612/Satellite_Images_Mask/0111_00013", extension = ".jpg").shape)
    # print(load_scenario("WideDataset/0265_02487/Satellite_Images_Mask/0265_01500", extension = ".jpg").shape)
    # print(load_scenario("WideDataset/0319_04796/Satellite_Images_Mask/0319_04119", extension = ".jpg").shape)
    # print(load_scenario("WideDataset/0320_02378/Satellite_Images_Mask/0320_00682", extension = ".jpg").shape)
    # print(load_scenario("WideDataset/0321_03136/Satellite_Images_Mask/0321_01452", extension = ".jpg").shape)
    # print(load_scenario("WideDataset/0323_01406/Satellite_Images_Mask/0323_00195", extension = ".jpg").shape)
    # print(load_scenario("WideDataset/0337_02831/Satellite_Images_Mask/0337_01635", extension = ".jpg").shape)


