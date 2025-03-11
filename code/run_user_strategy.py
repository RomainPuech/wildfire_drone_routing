"""
Author: Joseph Ye
Date: 3/10/2025
run_user_strategy.py
====================

This script allows users to benchmark their own sensor placement or drone routing
strategies without modifying core system files.

Users can define their strategies in separate Python files inside the "strategy" folder.
Each strategy must be implemented as a Python class.

Features:
- Dynamically loads strategy classes from user-specified files.
- Runs system benchmarks on the loaded strategies.
- Supports sensor placement strategies and can be extended for drone routing strategies.

How It Works:
1. User provides the path to the strategy folder, strategy file name, and class name.
2. The script dynamically loads the strategy class from the file.
3. The strategy is instantiated and passed into the benchmark function.
4. Benchmark results are printed to the console.

Usage:
    $ python run_user_strategy.py

Example call inside the file:
    run_benchmark_for_strategy(
        strategy_folder="strategy",
        strategy_file="logged_sensor_placement.py",
        class_name="LoggedSensorPlacementStrategy",
        automatic_init_params={...},
        custom_init_params={...}
    )

====================
"""

import importlib.util
import os
import tqdm
from dataset import load_scenario_npy
from benchmark import run_benchmark_scenario, get_automatic_layout_parameters, build_custom_init_params, listdir_npy_limited


def load_strategy(strategy_folder: str, strategy_file: str, class_name: str):
    """
    Dynamically loads a strategy class from a file.
    """
    strategy_path = os.path.join(strategy_folder, strategy_file)
    print(f"Looking for strategy file at: {strategy_path}")  # Add this line to debug
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Strategy file {strategy_path} not found!")

    module_name = strategy_file.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise ImportError(f"Class {class_name} not found in {strategy_file}!")

    return getattr(module, class_name)


def run_benchmark_for_strategy(input_dir: str,
                               strategy_folder: str,
                               sensor_strategy_file: str,
                               sensor_class_name: str,
                               drone_strategy_file: str,
                               drone_class_name: str,
                               max_n_scenarii: int = None,
                               starting_time: int = 0):
    """
    Runs benchmarks for the given sensor and drone strategies on all scenarios in input_dir.

    Args:
        input_dir (str): Path to the folder containing scenario .npy files
        strategy_folder (str): Folder where the strategy files are located
        sensor_strategy_file (str): Sensor placement strategy Python file name
        sensor_class_name (str): Name of the sensor placement strategy class
        drone_strategy_file (str): Drone routing strategy Python file name
        drone_class_name (str): Name of the drone routing strategy class
        max_n_scenarii (int, optional): Max number of scenarios to process (None = all)
        starting_time (int, optional): Time step at which wildfire starts
    """

    # Dynamically load the strategies
    SensorPlacementStrategyClass = load_strategy(strategy_folder, sensor_strategy_file, sensor_class_name)
    DroneRoutingStrategyClass = load_strategy(strategy_folder, drone_strategy_file, drone_class_name)

    # List the scenario files from the input_dir
    if not input_dir.endswith('/'):
        input_dir += '/'
    iterable = listdir_npy_limited(input_dir, max_n_scenarii)
    N_SCENARII = max_n_scenarii if max_n_scenarii else len(os.listdir(input_dir))

    # Initialize benchmark counters
    delta_ts = 0
    fails = 0
    devices = {'ground sensor': 0, "charging station": 0, "drone": 0, 'undetected': 0}

    # Process the first scenario to get automatic layout parameters
    first_file = next(iter(iterable), None)
    if first_file is None:
        print(f"No scenarios found in {input_dir}")
        return

    scenario = load_scenario_npy(first_file)
    automatic_init_params = get_automatic_layout_parameters(scenario)
    custom_init_params = build_custom_init_params(input_dir, layout_name="layout_A")


    # Reset the iterable (since we consumed one element already)
    iterable = listdir_npy_limited(input_dir, max_n_scenarii)

    # Create the sensor placement strategy (run it once per layout!)
    print("[run_benchmark_for_strategy] Running sensor placement strategy...")
    sensor_placement_strategy_instance = SensorPlacementStrategyClass(automatic_init_params, custom_init_params)
    ground_sensor_locations, charging_station_locations = sensor_placement_strategy_instance.get_locations()

    # Add placements to the automatic init params so drone strategy can use them
    automatic_init_params["ground_sensor_locations"] = ground_sensor_locations
    automatic_init_params["charging_stations_locations"] = charging_station_locations

    # Loop over scenarios
    for file in tqdm.tqdm(iterable, total=N_SCENARII):
        scenario = load_scenario_npy(file)

        # Drone routing strategy is likely scenario-dependent, so we instantiate per scenario
        drone_routing_strategy_instance = DroneRoutingStrategyClass(automatic_init_params, custom_init_params)

        # Run the benchmark
        delta_t, device, _ = run_benchmark_scenario(
            scenario=scenario,
            sensor_placement_strategy=lambda *_: sensor_placement_strategy_instance,
            drone_routing_strategy=lambda *_: drone_routing_strategy_instance,
            custom_initialization_parameters=custom_init_params,
            custom_step_parameters_function=lambda: {},  # Modify if you have step params
            starting_time=starting_time
        )

        # Aggregate results
        if delta_t == -1:
            fails += 1
            delta_t = 0
        delta_ts += delta_t
        devices[device] += 1

    # Print out summary stats
    avg_detection_time = delta_ts / max(1, (N_SCENARII - fails))
    print(f"\nThis strategy took on average {avg_detection_time:.2f} time steps to find the fire.")
    for device_type, count in devices.items():
        percentage = round((count / N_SCENARII) * 100, 2)
        print(f"Fire found {percentage}% of the time by {device_type}")




if __name__ == "__main__":
    run_benchmark_for_strategy(
        input_dir="MinimalDataset/0001/scenarii",
        strategy_folder="strategy",
        sensor_strategy_file="logged_sensor_placement.py",
        sensor_class_name="LoggedSensorPlacementStrategy",
        drone_strategy_file="logged_drone_routing.py",
        drone_class_name="LoggedDroneRoutingStrategy",
        max_n_scenarii=2,
        starting_time=0
    )
