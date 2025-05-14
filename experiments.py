# import requred modules
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Add code to path
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset import preprocess_sim2real_dataset, load_scenario_npy, compute_and_save_burn_maps_sim2real_dataset
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor, DroneRoutingRegularizedMaxCoverageResetStatic, FixedPlacementStrategy, DroneRoutingRegularizedMaxCoverageResetStatic
from benchmark import run_benchmark_scenario,run_benchmark_scenarii_sequential, get_burnmap_parameters,run_benchmark_scenarii_sequential_precompute, benchmark_on_sim2real_dataset_precompute
from displays import create_scenario_video
from new_clustering import get_wrapped_clustering_strategy

# housekeeping : delete temporary burn maps
if os.path.exists("tmp_burnmaps"):
    for file in os.listdir("tmp_burnmaps"):
        os.remove("tmp_burnmaps/" + file)


print("-- Starting experiments --")

simulation_parameters =  {
    "max_battery_distance": -1,
    "max_battery_time": 2*100000,
    "n_drones": 2,
    "n_ground_stations": 0,
    "n_charging_stations": 2,
    "drone_speed_m_per_min": 1200,
    "coverage_radius_m": 150,
    "cell_size_m": 30,
    "transmission_range": 50000,
    }

custom_initialization_parameters = {
     "load_from_logfile": False, 
     "reevaluation_step": 15, 
     "optimization_horizon":15,
     "regularization_param": 1
     } #"regularization_param": 0.0001}

layout_folder = "WideDataset/"
sensor_strategy = RandomSensorPlacementStrategy
drone_strategy = RandomDroneRoutingStrategy #wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingLinearMinTime))

def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
    simulation_parameters["N"] = scenario.shape[1]
    simulation_parameters["M"] = scenario.shape[2]
    return simulation_parameters

def return_no_custom_parameters():
    return {}

def custom_initialization_parameters_function(input_dir:str):
    print(f"input_dir: {input_dir}")
    return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/static_risk.npy", "reevaluation_step": 5, "optimization_horizon":5}

dataset_folder_name = "WideDataset/"
benchmark_on_sim2real_dataset_precompute(dataset_folder_name, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, max_n_scenarii=100, starting_time=0, max_n_layouts=3, simulation_parameters = simulation_parameters, file_format="jpg")

#run_benchmark_scenarii_sequential_precompute(layout_folder, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, file_format="npy", simulation_parameters=simulation_parameters)

# print(results)
# print(f"Time taken to run benchmark on the scenario: {time.time() - time_start} seconds")
# create_scenario_video(scenario[:len(position_history)],drone_locations_history=position_history,starting_time=0,out_filename='test_simulation', ground_sensor_locations = ground, charging_stations_locations = charging)
# # display a random burn map if there are any
# if os.path.exists("tmp_burnmaps") and len(os.listdir("tmp_burnmaps")) > 0:
#     tmp_burnmap_filename = os.path.join("tmp_burnmaps", os.listdir("tmp_burnmaps")[0])
#     bm = load_scenario_npy(tmp_burnmap_filename)
#     create_scenario_video(bm, burn_map = True, out_filename = "test_burn_map")

# # display the burn map
# bm = load_scenario_npy(custom_initialization_parameters["burnmap_filename"])
# create_scenario_video(bm, burn_map = True, out_filename = "test_burn_map")

# TODO remove the video generation and default param in displays.py