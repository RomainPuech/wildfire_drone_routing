import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# ----- Imports from codebase:
from benchmark import run_benchmark_scenario, get_automatic_layout_parameters, listdir_npy_limited, build_custom_init_params, return_no_custom_parameters
from dataset import load_scenario_npy
from Strategy import RandomSensorPlacementStrategy, RandomDroneRoutingStrategy, LoggedSensorPlacementStrategy, LoggedDroneRoutingStrategy

###############################################################################
# 1) gather_metrics_for_layout => calls run_benchmark_scenario on each .npy file
###############################################################################
def gather_metrics_for_layout(
    layout_path,
    sensor_strategy_cls,
    drone_strategy_cls,
    custom_init_params_fn,
    custom_step_params_fn,
    starting_time=0,
    max_n_scenarii=None
):
    """
    1) Looks in layout_path/scenarii for .npy scenario files.
    2) Calls run_benchmark_scenario for each scenario.
    3) Aggregates metrics (e.g. avg_execution_time, fire_size_cells, etc.) and returns a dict:
       {
         "avg_execution_time": (mean, std),
         "fire_size_cells": (mean, std),
         ...
         "detection_rates": { "ground sensor": 40, ... },
         "device_counts": {...},
         ...
       }
    or returns None if no .npy files found.
    """

    # Path to the "scenarii" folder
    scenarii_dir = os.path.join(layout_path, "scenarii")
    if not os.path.isdir(scenarii_dir):
        print(f"  [WARNING] No 'scenarii' folder in {layout_path}. Skipping.")
        return None

    # Get the .npy scenario files
    scenario_files = list(listdir_npy_limited(scenarii_dir, max_n_scenarii))
    if not scenario_files:
        print(f"  [WARNING] No .npy scenario files in {scenarii_dir}. Skipping.")
        return None

    # We'll accumulate these metrics for each scenario
    delta_ts = []
    execution_times = []
    fire_sizes_cells = []
    fire_sizes_percent = []
    maps_explored = []
    distances = []
    drone_entropies = []
    sensor_entropies = []

    device_counts = {"ground sensor": 0, "charging station": 0, "drone": 0, "undetected": 0}

    # Usually you build your custom init params once per layout:
    custom_init_params = custom_init_params_fn(layout_path, "generic_layout")

    # Loop over scenario files
    for scenario_file in tqdm.tqdm(scenario_files, desc=f"[{os.path.basename(layout_path)}]"):
        scenario = load_scenario_npy(scenario_file)
        auto_params = get_automatic_layout_parameters(scenario)

        # Actually call the main benchmark function
        results, _ = run_benchmark_scenario(
            scenario=scenario,
            sensor_placement_strategy=sensor_strategy_cls,   # pass the classes
            drone_routing_strategy=drone_strategy_cls,
            custom_initialization_parameters=custom_init_params,
            custom_step_parameters_function=custom_step_params_fn,
            starting_time=starting_time
        )

        # Collect from results
        delta_ts.append(results["delta_t"])
        execution_times.append(results["avg_execution_time"])
        fire_sizes_cells.append(results["fire_size_cells"])
        fire_sizes_percent.append(results["fire_size_percentage"])
        maps_explored.append(results["percentage_map_explored"])
        distances.append(results["total_distance_traveled"])
        drone_entropies.append(results["avg_drone_entropy"])
        sensor_entropies.append(results["sensor_entropy"])

        device_counts[results["device"]] += 1

    # Convert to arrays
    delta_ts = np.array(delta_ts, dtype=float)
    execution_times = np.array(execution_times, dtype=float)
    fire_sizes_cells = np.array(fire_sizes_cells, dtype=float)
    fire_sizes_percent = np.array(fire_sizes_percent, dtype=float)
    maps_explored = np.array(maps_explored, dtype=float)
    distances = np.array(distances, dtype=float)
    drone_entropies = np.array(drone_entropies, dtype=float)
    sensor_entropies = np.array(sensor_entropies, dtype=float)

    # Shorthand for computing (mean, std)
    def mean_std(array):
        return (float(np.mean(array)), float(np.std(array)))

    total_scenarios = len(scenario_files)
    detection_rates = {
        k: 100.0 * device_counts[k] / total_scenarios for k in device_counts
    }

    # Return dict with each metric => (mean, std)
    return {
        "delta_t_mean_std": mean_std(delta_ts),
        "avg_execution_time": mean_std(execution_times),
        "fire_size_cells": mean_std(fire_sizes_cells),
        "fire_size_percentage": mean_std(fire_sizes_percent),
        "percentage_map_explored": mean_std(maps_explored),
        "total_distance_traveled": mean_std(distances),
        "avg_drone_entropy": mean_std(drone_entropies),
        "sensor_entropy": mean_std(sensor_entropies),
        "device_counts": device_counts,
        "detection_rates": detection_rates,
        "n_scenarios": total_scenarios
    }

###############################################################################
# 3) Plot all metrics, each in a separate figure, with numeric x-axis for layouts
###############################################################################
def plot_all_metrics_across_layouts(
    root_folder,
    sensor_strategy_cls,
    drone_strategy_cls,
    max_n_scenarii=None,
    starting_time=0
):
    """
    1) Loops over subfolders in root_folder (e.g., 0001, 0002, etc.).
    2) gather_metrics_for_layout(...) for each folder => aggregated stats
    3) Plots each metric in a separate figure, with x-axis from 1..n (numeric),
       y-axis = (mean ± std) for that metric.
    
    You can pass in your custom sensor/drone strategy classes to run the benchmarks.
    """

    # We'll store aggregated results in layout_results[layout_name] = { ...dict of metrics... }
    layout_results = {}

    for folder_name in sorted(os.listdir(root_folder)):
        layout_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(layout_path):
            continue
        print(f"=== Processing layout folder: {layout_path} ===")
        stats = gather_metrics_for_layout(
            layout_path=layout_path,
            sensor_strategy_cls=sensor_strategy_cls,
            drone_strategy_cls=drone_strategy_cls,
            custom_init_params_fn=build_custom_init_params,
            custom_step_params_fn=return_no_custom_parameters,
            starting_time=starting_time,
            max_n_scenarii=max_n_scenarii
        )
        if stats is not None:
            layout_results[folder_name] = stats

    # If empty, exit
    if not layout_results:
        print("No layout data found. Exiting.")
        return

    # Sort layout names by numeric portion so '0002' < '0010'
    def extract_number(foldername):
        nums = re.findall(r"\d+", foldername)
        return int(nums[0]) if nums else 9999999

    sorted_layout_names = sorted(layout_results.keys(), key=extract_number)

    # We'll define the set of metric keys to plot
    # (each is stored as (mean, std) in gather_metrics_for_layout).
    metric_keys = [
        "delta_t_mean_std",
        "avg_execution_time",
        "fire_size_cells",
        "fire_size_percentage",
        "percentage_map_explored",
        "total_distance_traveled",
        "avg_drone_entropy",
        "sensor_entropy"
    ]

    # For each metric, build an array and plot
    for metric in metric_keys:
        means = []
        stds = []
        for layout_name in sorted_layout_names:
            # Each is (mean_val, std_val)
            mean_val, std_val = layout_results[layout_name][metric]
            means.append(mean_val)
            stds.append(std_val)

        x_positions = np.arange(len(sorted_layout_names))  # e.g. [0..n-1]
        # We'll label them 1..n on the x-axis
        numeric_labels = [i+1 for i in range(len(sorted_layout_names))]

        output_dir = "plots_layouts"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure()
        plt.errorbar(x_positions, means, yerr=stds, fmt='o', capsize=5)
        plt.xticks(x_positions, numeric_labels)
        plt.xlabel("Layout Index")
        plt.ylabel(metric)
        plt.title(f"{metric} per Layout (mean ± std)")

        # save the plot
        out_filename = f"layoutplot_{metric}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"[Saved] {out_path}")

    # Print out final stats for reference
    print("\n=== Final Aggregated Results ===")
    for i, layout_name in enumerate(sorted_layout_names, start=1):
        print(f"Layout {i} => folder '{layout_name}':")
        print(layout_results[layout_name])
        print()

###############################################################################
# 4) Example usage in a main block
###############################################################################
if __name__ == "__main__":
    # Example root folder
    root_dataset = "/Users/josephye/Desktop/wildfire_drone_routing/MinimalDataset"

    # If you want custom strategies, define or import them.
    # For demonstration, we pass RandomSensorPlacementStrategy & RandomDroneRoutingStrategy.
    # You can pass e.g. MySensorPlacementStrategy, MyDroneRoutingStrategy instead if you want.
    sensor_strategy = LoggedSensorPlacementStrategy
    drone_strategy = LoggedDroneRoutingStrategy

    # Just call the function:
    plot_all_metrics_across_layouts(
        root_folder=root_dataset,
        sensor_strategy_cls=sensor_strategy,
        drone_strategy_cls=drone_strategy,
        max_n_scenarii=None,     # or an int
        starting_time=0
    )
