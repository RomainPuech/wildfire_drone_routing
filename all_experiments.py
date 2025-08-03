#!/bin/bash

# import requred modules
print("Starting all_experiments.py")
import sys
import os
import time
import numpy as np
import argparse
import glob
import json
from matplotlib.colors import LogNorm
import imageio
# Add code to path
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset import preprocess_sim2real_dataset, load_scenario_npy, compute_and_save_burn_maps_sim2real_dataset, load_scenario, combine_all_benchmark_results
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from new_clustering import get_wrapped_clustering_strategy
from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, RandomSensorPlacementStrategy, SensorPlacementMaxCoverageGaussianTime, DroneRoutingUniformCoverageGrowingStatic, DroneRoutingMaxCoverageGrowingStatic, DroneRoutingUniformCoverageResetStatic, DroneRoutingMaxCoverageResetStatic, DroneRoutingUniformCoverageResetStatic
from benchmark import run_benchmark_scenario,run_benchmark_scenarii_sequential, get_burnmap_parameters,run_benchmark_scenarii_sequential_precompute, benchmark_on_sim2real_dataset_precompute
from displays import create_scenario_video, create_video_scenario_burnmap

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

custom_initialization_parameters = {
    "load_from_logfile": False, 
    "reevaluation_step": 5, 
    "optimization_horizon":10,
    "regularization_param": 1e5
    } #"regularization_param": 0.0001}


dataset_folder_name = "/home/gridsan/jye/WFDroneBench_local/WideDataset"

def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
    simulation_parameters["N"] = scenario.shape[1]
    simulation_parameters["M"] = scenario.shape[2]
    return simulation_parameters

def return_no_custom_parameters():
    return {}

def load_drone_positions_from_log(json_log_path):
    with open(json_log_path, "r") as f:
        action_history = json.load(f)
    
    drone_positions = []
    for timestep_actions in action_history:
        positions = [tuple(action[1]) for action in timestep_actions]
        drone_positions.append(positions)
    
    return drone_positions

def load_static_locations(log_dir):
    import json
    static_info_path = os.path.join(log_dir, "static_locations.json")
    if os.path.exists(static_info_path):
        with open(static_info_path, "r") as f:
            data = json.load(f)
        return data.get("ground_sensor_locations", []), data.get("charging_station_locations", [])
    else:
        print(f"[!] static_locations.json not found in {log_dir}")
        return [], []
    
def run_one_drone_strategy(sensor_strategy, drone_strategy, custom_initialization_parameters_function, experiment_name, layout_name):

    strategy_name = sensor_strategy.strategy_name + "_" + drone_strategy.strategy_name
    print(f"-- Starting {experiment_name} --")

    time_start = time.time()

    benchmark_on_sim2real_dataset_precompute(
        dataset_folder_name,
        sensor_strategy,
        drone_strategy,
        custom_initialization_parameters_function,
        return_no_custom_parameters,
        max_n_scenarii=1,
        max_n_layouts=1,
        simulation_parameters=simulation_parameters,
        selected_layout_names=[layout_name],
        file_format="npy",
        config_file="config_s2r.json",
        experiment_name=experiment_name
    )

    print(f"Time taken to run benchmark {experiment_name}: {time.time() - time_start} seconds")
    combine_all_benchmark_results("WideDataset/", strategy_name=strategy_name, experiment_name=experiment_name)

    # Create video for the experiment
    try:
        # 🛰️ Find the most recent burnmap
        burnmap_files = glob.glob("tmp_burnmaps/tmp_burnmap_*.npy")
        if not burnmap_files:
            print(f"[!] No burnmap found. Skipping video for {experiment_name}.")
            return

        latest_burnmap = max(burnmap_files, key=os.path.getctime)
        burn_map = load_scenario_npy(latest_burnmap)

        # 🔍 Find the log file ending in logged_drone_routing.json
        layout_log_dir = os.path.join(dataset_folder_name, layout_name, "logs")
        matching_logs = [f for f in os.listdir(layout_log_dir) if f.endswith("logged_drone_routing.json")]
        if not matching_logs:
            print(f"[!] No log file ending with 'logged_drone_routing.json' found in {layout_log_dir}")
            return
        log_path = os.path.join(layout_log_dir, matching_logs[0])

        drone_locations_history = load_drone_positions_from_log(log_path)


        # 📍 Load static locations (ground sensors and charging stations)
        ground_sensor_locations = []
        charging_stations_locations = []
        static_json_candidates = [
            f for f in os.listdir(layout_log_dir)
            if f.endswith("charge.json")
        ]
        if not static_json_candidates:
            print(f"[!] No static_*.json file found in {layout_log_dir}, proceeding without static locations.")
        else:
            static_json_path = os.path.join(layout_log_dir, static_json_candidates[0])
            with open(static_json_path, "r") as f:
                static_data = json.load(f)
                ground_sensor_locations = static_data.get("ground_sensor_locations", [])
                charging_stations_locations = static_data.get("charging_station_locations", [])


        # 🎞️ Create video
        os.makedirs("videos", exist_ok=True)
        create_video_scenario_burnmap(
            burn_map=burn_map,
            drone_locations_history=drone_locations_history,
            ground_sensor_locations=ground_sensor_locations,
            charging_stations_locations=charging_stations_locations,
            out_filename=f"videos/{experiment_name}_burnmap_video"
        )
        print(f"[✔] Video created for {experiment_name}")

    except Exception as e:
        print(f"[!] Error creating video for {experiment_name}: {e}")


def run_all_drone_strategies(sensor_strategy, ss_prefix, bm_prefix):
    print("running experiments with prefix", ss_prefix, bm_prefix)
    bm_prefix_to_bm_name = {"whp":"static_risk_whp.npy", "bm": "burn_map.npy", "bp": "static_risk_bp2024.npy", "ncbm": "burn_map_noncumulative.npy"}

    def custom_initialization_parameters_function(input_dir: str, scenario_name: str = None, canonical_scenario_name: str = None):
        print(f"input_dir: {input_dir}")
        layout_dir = os.path.abspath(os.path.join(input_dir, ".."))
        bm_file = os.path.join(layout_dir, bm_prefix_to_bm_name[bm_prefix])
        print(f"Resolved burnmap file: {bm_file}")
        recompute = scenario_name == canonical_scenario_name if scenario_name and canonical_scenario_name else False

        return {
            "burnmap_filename": bm_file,
            "burnmap_type": "dynamic" if bm_file.endswith("burn_map.npy") else "static",
            "reevaluation_step": 5,
            "optimization_horizon": 10,
            "regularization_param": 1,
            "recompute_logfile_drone": recompute
        }

    def custom_initialization_parameters_function_greedy(input_dir: str):
        print(f"input_dir: {input_dir}")
        layout_dir = os.path.abspath(os.path.join(input_dir, ".."))
        bm_file = os.path.join(layout_dir, bm_prefix_to_bm_name[bm_prefix])
        print(f"Resolved burnmap file: {bm_file}")
        return {
            "burnmap_filename": bm_file,
            "burnmap_type": "dynamic" if bm_file.endswith("burn_map.npy") else "static",
            "reevaluation_step": 2,
            "optimization_horizon": 2,
            "regularization_param": 1
        }
    #housekeeping : delete temporary burn maps
    if os.path.exists("tmp_burnmaps"):
        for file in os.listdir("tmp_burnmaps"):
            os.remove("tmp_burnmaps/" + file)

    layout_name = "0004_01191"  # Replace this with a loop for all layouts if needed
    experiment_name = f"{ss_prefix}MCg{bm_prefix}_{layout_name}"


    # run_one_drone_strategy(sensor_strategy, RandomDroneRoutingStrategy, custom_initialization_parameters_function, f"{ss_prefix}R{bm_prefix}")
    run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStatic)), custom_initialization_parameters_function, experiment_name, layout_name)
    # run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingUniformCoverageResetStatic)), custom_initialization_parameters_function, f"{ss_prefix}Ug{bm_prefix}")
    # run_one_drone_strategy(sensor_strategy, wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStaticGreedy)), custom_initialization_parameters_function_greedy, "KG")
    
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all drone strategy experiments')
    parser.add_argument('--ss_prefix', type=str, required=True, 
                        help='Sensor strategy prefix (e.g., "S" for sensor)')
    parser.add_argument('--bm_prefix', type=str, required=True, 
                        choices=['whp', 'bm', 'bp', 'ncbm'],
                        help='Burn map prefix: "whp" for static_risk_whp.npy, "bm" for burn_map.npy, "ncbm" for noncumulative burn_map.npy, "bp" for static_risk_bp2024.npy')
    
    args = parser.parse_args()
    if args.ss_prefix == "R":
        print("Running random sensor placement")
        sensor_strategy = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
    else:
        print("Running sensor placement with max coverage")
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
