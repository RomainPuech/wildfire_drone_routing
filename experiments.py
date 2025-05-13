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
    "N": 50,
    "M": 50,
    "max_battery_distance": -1,
    "max_battery_time": 28,
    "n_drones": 2,
    "n_ground_stations": 0,
    "n_charging_stations": 2,
    "speed_m_per_min": 9,
    "coverage_radius_m": 45,
    "cell_size_m": 30,
    "transmission_range": 100,
    }

custom_initialization_parameters = {
    "burnmap_filename": "WideDataset/0101_02057/static_risk.npy",
     "load_from_logfile": False, 
     "reevaluation_step": 15, 
     "optimization_horizon":15,
     "regularization_param": 1
     } #"regularization_param": 0.0001}

layout_folder = "WideDataset/0101_02057/scenarii"
scenario_name = "0101_00001"
sensor_strategy = FixedPlacementStrategy
drone_strategy = get_wrapped_clustering_strategy(DroneRoutingRegularizedMaxCoverageResetStatic) #wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingLinearMinTime))

def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
    simulation_parameters["N"] = scenario.shape[1]
    simulation_parameters["M"] = scenario.shape[2]
    return simulation_parameters

def return_no_custom_parameters():
    return {}
def custom_initialization_parameters_function(input_dir: str, layout_name: str = None):
    return custom_initialization_parameters
scenario = load_scenario_npy(layout_folder + "/" + scenario_name + ".npy")


#run_benchmark_scenarii_sequential_precompute(layout_folder, sensor_strategy, drone_strategy, custom_initialization_parameters_function, return_no_custom_parameters, file_format="npy", simulation_parameters=simulation_parameters)
time_start = time.time()
results, (position_history, ground, charging)  = run_benchmark_scenario(scenario, sensor_strategy, drone_strategy, custom_initialization_parameters, custom_step_parameters_function = return_no_custom_parameters, automatic_initialization_parameters_function=my_automatic_layout_parameters, return_history=True)


print(results)
print(f"Time taken to run benchmark on the scenario: {time.time() - time_start} seconds")
create_scenario_video(scenario[:len(position_history)],drone_locations_history=position_history,starting_time=0,out_filename='test_simulation', ground_sensor_locations = ground, charging_stations_locations = charging)
# display a random burn map if there are any
if os.path.exists("tmp_burnmaps") and len(os.listdir("tmp_burnmaps")) > 0:
    tmp_burnmap_filename = os.path.join("tmp_burnmaps", os.listdir("tmp_burnmaps")[0])
    bm = load_scenario_npy(tmp_burnmap_filename)
    create_scenario_video(bm, burn_map = True, out_filename = "test_burn_map")

# display the burn map
bm = load_scenario_npy(custom_initialization_parameters["burnmap_filename"])
create_scenario_video(bm, burn_map = True, out_filename = "test_burn_map")