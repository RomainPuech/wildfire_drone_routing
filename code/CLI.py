
# import requred modules
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import json
import math

# from plot_violin import gather_data_from_layouts, plot_violin_for_each_metric
# from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
# from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor
# from benchmark import benchmark_on_sim2real_dataset_precompute, return_no_custom_parameters
# # from displays import create_scenario_video
# from new_clustering import get_wrapped_strategy


# PLACEMENT_STRATEGY_TO_TEST = RandomSensorPlacementStrategy
# DRONE_STRATEGY_TO_TEST = DroneRoutingOptimizationModelReuseIndex


# change values here to change benchmarking parameters

default_parameters = {
    "max_battery_distance": -1,
    "max_battery_time": 6,
    "n_drones": 3,
    "n_ground_stations": 1,
    "n_charging_stations": 2,
}

def custom_initialization_parameters_function(input_dir:str):
    print(f"input_dir: {input_dir}")
    return {"burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy", "reevaluation_step": 5, "optimization_horizon":5, "strategy_drone": DroneRoutingOptimizationModelReuseIndex, "strategy_sensor": RandomSensorPlacementStrategy}



def plot_metrics_from_precomputed(metrics_dict):
    os.makedirs("plots_layouts", exist_ok=True)

    selected_raw_metrics = [
        "raw_execution_times",
        "raw_fire_sizes",
        "raw_fire_percentages",
        "raw_map_explored",
        "raw_distances",
        "raw_drone_entropies",
        "raw_sensor_entropies"
    ]

    for raw_key in selected_raw_metrics:
        metric = raw_key.replace("raw_", "")
        layout_names = []
        means = []
        stds = []

        for layout, metrics in metrics_dict.items():
            if not isinstance(metrics, dict):
                print(f"[warning] Skipping layout '{layout}' — not a metrics dict.")
                continue

            if raw_key in metrics:
                values = metrics[raw_key]
                values = [float(v) for v in values if not np.isnan(v)]
                if not values:
                    continue
                mean_val = np.mean(values)
                std_val = np.std(values)

                layout_names.append(layout)
                means.append(mean_val)
                stds.append(std_val)

        # plot line + std error bars
        plt.figure()
        plt.errorbar(layout_names, means, yerr=stds, fmt='o', capsize=5, markersize=8, linewidth=2)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Across Layouts")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"plots_layouts/{metric}.png")
        plt.close()




def load_and_plot_metrics(filepath):
    """
    Loads a metrics dictionary from a file and plots selected metrics.
    Handles both valid JSON and stringified Python dict formats (even with NaNs).
    """
    with open(filepath, "r") as f:
        text = f.read().strip()

        try:
            data = json.loads(text)
            if isinstance(data, str):
                print("[info] inner string detected — using eval to parse...")
                data = eval(data, {"nan": math.nan})
        except json.JSONDecodeError:
            print("[info] not valid JSON — using eval directly...")
            data = eval(text, {"nan": math.nan})

        if not isinstance(data, dict):
            raise ValueError("Loaded data is not a dictionary")

        plot_metrics_from_precomputed(data)

def generate_violin_plots_from_results(folder_path: str, output_dir: str = "plots_violin"):
    """
    Generates violin plots for selected raw metrics across different strategies.

    Args:
        folder_path (str): Path to the folder containing JSON result files.
        output_dir (str): Path to save violin plots.
    """
    selected_raw_metrics = [
        "raw_execution_times",
        "raw_fire_sizes",
        "raw_fire_percentages",
        "raw_map_explored",
        "raw_distances",
        "raw_drone_entropies",
        "raw_sensor_entropies"
    ]

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            raw_text = f.read().strip()

        try:
            data = json.loads(raw_text)
            if isinstance(data, str):
                data = eval(data, {"nan": float("nan")})
        except Exception:
            data = eval(raw_text, {"nan": float("nan")})

        strategy_name = filename.split("_")[1]

        for layout_id, metrics in data.items():
            if not isinstance(metrics, dict):
                continue

            for raw_metric in selected_raw_metrics:
                if raw_metric in metrics:
                    values = metrics[raw_metric]
                    if isinstance(values, list):
                        for v in values:
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                records.append({
                                    "strategy": strategy_name,
                                    "metric": raw_metric.replace("raw_", ""),
                                    "value": float(v)
                                })

    if not records:
        print("⚠️ No valid data found.")
        return

    df = pd.DataFrame(records)

    for metric in df["metric"].unique():
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df[df["metric"] == metric], x="strategy", y="value", inner="quartile")
        plt.title(f"{metric.replace('_', ' ').title()} Distribution Across Strategies")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xlabel("Strategy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()

    print(f"✅ Violin plots saved to: {output_dir}")

if __name__ == "__main__":
    # Parse command line arguments

    # parser = argparse.ArgumentParser(description='Benchmarking script for strategies')
    # parser.add_argument('--dataset', type=str, default="MinimalDataset/", help='Dataset folder path')
    # parser.add_argument('--n_drones', type=int, default=default_parameters["n_drones"], help='Number of drones')
    # parser.add_argument('--n_ground_stations', type=int, default=default_parameters["n_ground_stations"], help='Number of ground stations')
    # parser.add_argument('--n_charging_stations', type=int, default=default_parameters["n_charging_stations"], help='Number of charging stations')
    # parser.add_argument('--max_battery_distance', type=int, default=default_parameters["max_battery_distance"], help='Maximum battery distance')
    # parser.add_argument('--max_battery_time', type=int, default=default_parameters["max_battery_time"], help='Maximum battery time')
    # args = parser.parse_args()

    # print(f"Running benchmark for dataset: {args.dataset}")
    
    # metrics_by_layout = benchmark_on_sim2real_dataset_precompute(
    #     dataset_folder_name=args.dataset,
    #     ground_placement_strategy=PLACEMENT_STRATEGY_TO_TEST,
    #     drone_routing_strategy=DRONE_STRATEGY_TO_TEST,
    #     custom_initialization_parameters_function=custom_initialization_parameters_function,
    #     custom_step_parameters_function=return_no_custom_parameters,
    #     max_n_scenarii=1,
    #     starting_time=0,
    #     simulation_parameters = {
    #         "n_drones": args.n_drones,
    #         "n_ground_stations": args.n_ground_stations,
    #         "n_charging_stations": args.n_charging_stations,
    #         "max_battery_distance": args.max_battery_distance,
    #         "max_battery_time": args.max_battery_time,
    #     }
    # )
    
    # print("Done!")
    
    # load_and_plot_metrics("/Users/josephye/Desktop/wildfire_drone_routing/code/test.json")
    generate_violin_plots_from_results("code/results/")
    
    
    