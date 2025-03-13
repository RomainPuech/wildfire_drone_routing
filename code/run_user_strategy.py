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
from benchmark import run_benchmark_for_strategy2, return_no_custom_parameters, my_custom_init_params



if __name__ == "__main__":
    run_benchmark_for_strategy2(
        input_dir="MinimalDataset/0001/scenarii",
        strategy_folder="code/strategy",
        sensor_strategy_file="logged_sensor_placement.py",
        sensor_class_name="LoggedSensorPlacementStrategy",
        drone_strategy_file="logged_drone_routing.py",
        drone_class_name="LoggedDroneRoutingStrategy",
        max_n_scenarii=4,
        starting_time=0,
        file_format= "npy",
        custom_init_params_fn= my_custom_init_params,  # user-defined initialization params function
        custom_step_params_fn= return_no_custom_parameters
    )
