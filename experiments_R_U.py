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
from dataset import preprocess_sim2real_dataset, load_scenario_npy, compute_and_save_burn_maps_sim2real_dataset, load_scenario
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor, DroneRoutingUniformMaxCoverageResetStatic, FixedPlacementStrategy
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
sensor_strategy = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
drone_strategy =  wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingUniformMaxCoverageResetStatic))#DroneRoutingRegularizedMaxCoverageResetStatic#wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingLinearMinTime))

def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
    simulation_parameters["N"] = scenario.shape[1]
    simulation_parameters["M"] = scenario.shape[2]
    return simulation_parameters

def return_no_custom_parameters():
    return {}

def custom_initialization_parameters_function2(input_dir:str):
    print(f"input_dir: {input_dir}")
    return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/static_risk.npy", "reevaluation_step": 5, "optimization_horizon":10, "regularization_param": 1}


def custom_initialization_parameters_function(input_dir:str):
    print(f"input_dir: {input_dir}")
    return {"burnmap_filename": f"WideDataset/0244_03110/static_risk.npy", "reevaluation_step": 5, "optimization_horizon":10, "regularization_param": 1}


dataset_folder_name = "WideDataset/"
time_start = time.time()
run_benchmark_scenarii_sequential_precompute("WideDataset/0244_03110/Satellite_Image_Mask/", sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, file_format="jpg", simulation_parameters=simulation_parameters)
#benchmark_on_sim2real_dataset_precompute(dataset_folder_name, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, max_n_scenarii=1, max_n_layouts=None, simulation_parameters = simulation_parameters, file_format="jpg", config_file="config_s2r.json", skip_folder_names=["WideDataset/0069_03539", "WideDataset/0004_01191", "WideDataset/0058_03866", "WideDataset/0250_02864", "WideDataset/0244_03110"])
print(f"Time taken to run benchmark on the scenario: {time.time() - time_start} seconds")

# burn_map = load_scenario_npy("WideDataset/0004_01191/static_risk.npy")
# #burn_map = load_scenario_npy("WideDataset/tmp_burnmap_863027.npy")
# create_scenario_video(burn_map, burn_map = True, out_filename = "test_burn_map")


# scenario = load_scenario("WideDataset/0004_01191/Satellite_Images_Mask/0004_00205", extension="jpg")
# print(np.argwhere(scenario[0,:,:] == 1))


# scenario = load_scenario("WideDataset/0004_01191/Satellite_Images_Mask/0004_00644", extension="jpg")
# res, (position_history, ground, charging) = run_benchmark_scenario(scenario, sensor_strategy, drone_strategy, custom_initialization_parameters_function("WideDataset/0004_01191/Satellite_Images_Mask/"), return_no_custom_parameters, simulation_parameters=simulation_parameters, return_history=True)
# create_scenario_video(scenario[:len(position_history)],drone_locations_history=position_history,starting_time=0,out_filename='test_simulation', ground_sensor_locations = ground, charging_stations_locations = charging)
# #run_benchmark_scenarii_sequential_precompute(layout_folder, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, file_format="npy", simulation_parameters=simulation_parameters)




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
# Create the congig file

