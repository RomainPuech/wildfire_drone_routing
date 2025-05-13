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
from new_clustering import get_wrapped_strategy


print("-- Starting experiments --")


# change values here to change benchmarking parameters
def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
    print(scenario.shape[1])
    return {
        "N": scenario.shape[1],
        "M": scenario.shape[2],
        "max_battery_distance": -1,
        "max_battery_time": 28,
        "n_drones": 2,
        "n_ground_stations": 0,
        "n_charging_stations": 2,
    }

simulation_parameters =  {
        "N": 50,
        "M": 50,
        "max_battery_distance": -1,
        "max_battery_time": 28,
        "n_drones": 2,
        "n_ground_stations": 0,
        "n_charging_stations": 2,
    }

def return_no_custom_parameters():
    return {}


def custom_initialization_parameters_function(input_dir: str, layout_name: str = None):
    return {"burnmap_filename": "IP_Dataset/0101_02057/cropped_burn_map_new.npy", "load_from_logfile": False, "reevaluation_step": 25, "optimization_horizon":25,"regularization_param": 1} #"regularization_param": 0.0001}


run_benchmark_scenarii_sequential_precompute("IP_Dataset/0101_02057/cropped_scenarii", 
FixedPlacementStrategy, 
wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingRegularizedMaxCoverageResetStatic)),#wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingLinearMinTime)), 
custom_initialization_parameters_function, 
return_no_custom_parameters, 
file_format="npy", simulation_parameters=simulation_parameters)




# That's very fast to run!
print("starting benchmark")
time_start = time.time()
scenario = load_scenario_npy("IP_Dataset/0101_02057/cropped_scenarii/0101_00002.npy")
results, (position_history, ground, charging)  = run_benchmark_scenario(scenario, FixedPlacementStrategy, wrap_log_drone_strategy(get_wrapped_strategy(DroneRoutingRegularizedMaxCoverageResetStatic)), 
custom_initialization_parameters = {"burnmap_filename": "IP_Dataset/0101_02057/cropped_burn_map.npy", "load_from_logfile": False, "reevaluation_step": 2, "optimization_horizon":11, "regularization_param": 0}, custom_step_parameters_function = return_no_custom_parameters, automatic_initialization_parameters_function=my_automatic_layout_parameters, return_history=True)
print(results)
print(f"Time taken to run benchmark on the scenario: {time.time() - time_start} seconds")
create_scenario_video(scenario[:len(position_history)],drone_locations_history=position_history,starting_time=0,out_filename='test_simulation', ground_sensor_locations = ground, charging_stations_locations = charging)



bm = load_scenario_npy("tmp_burnmaps/tmp_burnmap_251727.npy")
create_scenario_video(bm, burn_map = True, out_filename = "test_burn_map", )