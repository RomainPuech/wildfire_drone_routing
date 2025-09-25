#!/bin/bash

# import requred modules
print("Starting all_experiments.py")
import sys
import os
import time
import numpy as np
import argparse
from functools import partial

# Add code to path
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)
    
from dataset import combine_all_benchmark_results
from wrappers import RandomSensorPlacementStrategyLogged, SensorPlacementMaxCoverageGaussianTimeLogged
from benchmark import benchmark_on_sim2real_dataset_precompute_parallel
from Strategy import RandomDroneRoutingStrategy

# shared parameters
simulation_parameters =  {
    "max_battery_distance": -1,
    "max_battery_time": 1,
    "n_drones": 2,
    "n_ground_stations": 8,
    "n_charging_stations": 2,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m": 300,
    "cell_size_m": 30,
    "transmission_range": 50000,
    }


dataset_folder_name = "./WideDataset"

# Mapping used to resolve burnmap filename
BM_PREFIX_TO_NAME = {
    "whp": "static_risk_whp.npy",
    "bm": "burn_map.npy",
    "bp": "static_risk_bp2024.npy",
}

# === Custom Parameter Functions ===
def custom_initialization_parameters_function(input_dir: str, *, bm_prefix: str):
    layout_dir = os.path.abspath(os.path.join(input_dir, ".."))
    bm_file = os.path.join(layout_dir, BM_PREFIX_TO_NAME[bm_prefix])
    return {
        "burnmap_filename": bm_file,
        "burnmap_type": "dynamic" if bm_file.endswith("burn_map.npy") else "static",
        "reevaluation_step": 5,
        "optimization_horizon": 10,
        "regularization_param": 1
    }


def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
        simulation_parameters["N"] = scenario.shape[1]
        simulation_parameters["M"] = scenario.shape[2]
        return simulation_parameters

def return_no_custom_parameters():
    return {}

def run_one_drone_strategy(sensor_strategy, drone_strategy, custom_initialization_parameters_function, experiment_name=""):
    sensor_strategy_name = str(sensor_strategy.strategy_name) if hasattr(sensor_strategy, "strategy_name") else str(sensor_strategy)
    drone_strategy_name = str(drone_strategy.strategy_name) if hasattr(drone_strategy, "strategy_name") else str(drone_strategy)
    strategy_name = sensor_strategy_name + "_" + drone_strategy_name
    
    print(f"-- Starting {experiment_name} --")
    
    time_start = time.time()
    benchmark_on_sim2real_dataset_precompute_parallel(
        dataset_folder_name,
        sensor_strategy,
        drone_strategy,
        custom_initialization_parameters_function,
        return_no_custom_parameters,
        max_n_scenarii=None,
        max_n_layouts=None,
        simulation_parameters=simulation_parameters,
        # selected_layout_names= ["0081_03471", "0264_02426"],#,"0264_02426","0265_02487"],
        file_format="npy",
        config_file="config_s2r.json",
        experiment_name=experiment_name
    )
    
    print(f"Time taken to run benchmark {experiment_name}: {time.time() - time_start} seconds")
    combine_all_benchmark_results(dataset_folder_name, strategy_name = strategy_name, experiment_name = experiment_name)   
    

def run_all_drone_strategies(sensor_strategy, ss_prefix, bm_prefix):
    print("running experiments with prefix", ss_prefix, bm_prefix)

    init_func = partial(custom_initialization_parameters_function, bm_prefix=bm_prefix)
    

    
    #housekeeping : delete temporary burn maps
    # if os.path.exists("tmp_burnmaps"):
    #     for file in os.listdir("tmp_burnmaps"):
    #         os.remove("tmp_burnmaps/" + file)

    run_one_drone_strategy(sensor_strategy, "RandomDroneRoutingStrategy", custom_initialization_parameters_function, f"{ss_prefix}R{bm_prefix}")
    run_one_drone_strategy(sensor_strategy, "DroneRoutingTOP", init_func, f"{ss_prefix}TOP{bm_prefix}_parallel")
    run_one_drone_strategy(sensor_strategy, "DroneRoutingUniformCoverageResetStatic", init_func, f"{ss_prefix}U{bm_prefix}_parallel")
    run_one_drone_strategy(sensor_strategy, "DroneRoutingMaxCoverageResetStatic", init_func, f"{ss_prefix}M{bm_prefix}_parallel")
     
    

if __name__ == "__main__":
    from julia import Julia
    Julia(compiled_modules=False)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all drone strategy experiments')
    parser.add_argument('--ss_prefix', type=str, required=True, 
                        help='Sensor strategy prefix (e.g., "S" for sensor)')
    parser.add_argument('--bm_prefix', type=str, required=True, 
                        choices=['whp', 'bm', 'bp'],
                        help='Burn map prefix: "whp" for static_risk_whp.npy, "bm" for burn_map.npy, "bp" for static_risk_bp2024.npy')
    
    args = parser.parse_args()
    if args.ss_prefix == "R":
        print("Running random sensor placement")
        sensor_strategy = RandomSensorPlacementStrategyLogged
    else:
        print("Running sensor placement with max coverage")
        sensor_strategy = SensorPlacementMaxCoverageGaussianTimeLogged

    run_all_drone_strategies(sensor_strategy, args.ss_prefix, args.bm_prefix)
