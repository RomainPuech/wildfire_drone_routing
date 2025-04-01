import os
import re
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from benchmark import return_no_custom_parameters, build_custom_init_params, run_benchmark_scenario, get_automatic_layout_parameters, listdir_npy_limited
from dataset import load_scenario_npy
from Strategy import RandomSensorPlacementStrategy, RandomDroneRoutingStrategy, LoggedDroneRoutingStrategy, LoggedSensorPlacementStrategy

# -------------------------------------------------------------------
# 2) DEFINE MULTIPLE STRATEGIES
#    Each strategy is: (strategy_name, SensorPlacementClass, DroneRoutingClass)
# -------------------------------------------------------------------
STRATEGIES = [
    ("RandomStrat1", RandomSensorPlacementStrategy, RandomDroneRoutingStrategy),
    ("LoggedStrat1", LoggedSensorPlacementStrategy, LoggedDroneRoutingStrategy),
    # You can add more lines here, e.g.
    # ("MyAwesomeStrategy", MySensorPlacementClass, MyDroneRoutingClass),
]


# -------------------------------------------------------------------
# 4) GATHER ALL SCENARIO DATA ACROSS MULTIPLE LAYOUTS & STRATEGIES
#    We'll build a Pandas DataFrame with columns:
#       ["layout", "scenario", "strategy", "metric", "value"]
# -------------------------------------------------------------------
def gather_data_from_layouts(
    root_folder,
    strategies,
    custom_init_params_fn,
    custom_step_params_fn,
    starting_time=0,
    max_n_scenarii=None
):
    """
    1) Loops over each layout folder in root_folder (e.g. '0001', '0002', etc.).
    2) Finds all .npy scenario files in <layout>/scenarii.
    3) For each scenario and each strategy:
       - run_benchmark_scenario
       - store the raw metric values in a big list of dicts
    4) Returns a DataFrame with one row per run/metric, columns = [layout, scenario, strategy, metric, value].
    """

    # This will accumulate rows for building a DataFrame
    rows = []

    # A list of metrics we want to store from run_benchmark_scenario's output:
    metric_keys = [
        "delta_t",
        "avg_execution_time",
        "fire_size_cells",
        "fire_size_percentage",
        "percentage_map_explored",
        "total_distance_traveled",
        "avg_drone_entropy",
        "sensor_entropy",
    ]

    # -------------------------------------------------------------------
    # Walk through each layout folder: /root_folder/0001, /root_folder/0002, ...
    # -------------------------------------------------------------------
    for layout_folder_name in sorted(os.listdir(root_folder)):
        layout_path = os.path.join(root_folder, layout_folder_name)
        if not os.path.isdir(layout_path):
            continue

        # Scenes are in <layout>/scenarii
        scenarii_dir = os.path.join(layout_path, "scenarii")
        if not os.path.isdir(scenarii_dir):
            print(f"[WARNING] No 'scenarii' folder in {layout_path}. Skipping.")
            continue

        # Grab scenario files
        scenario_files = list(listdir_npy_limited(scenarii_dir, max_n_scenarii))
        scenario_files = list(scenario_files)  # convert generator to list
        if not scenario_files:
            print(f"[WARNING] No .npy files in {scenarii_dir}. Skipping.")
            continue

        print(f"=== Layout folder: {layout_folder_name} => {len(scenario_files)} scenarios found ===")

        for scenario_file in tqdm.tqdm(scenario_files, desc=f"Layout {layout_folder_name}"):
            scenario = load_scenario_npy(scenario_file)
            # If your strategies need layout-based init, do it here or inside each run:
            auto_params = get_automatic_layout_parameters(scenario)

            # We'll also compute custom init params once per layout or scenario:
            custom_init_params = custom_init_params_fn(layout_path, layout_folder_name)

            # For each strategy in STRATEGIES
            for (strat_name, SensorCls, DroneCls) in strategies:
                # Actually run the benchmark
                results, _ = run_benchmark_scenario(
                    scenario=scenario,
                    sensor_placement_strategy=SensorCls,
                    drone_routing_strategy=DroneCls,
                    custom_initialization_parameters=custom_init_params,
                    custom_step_parameters_function=custom_step_params_fn,
                    starting_time=starting_time
                )

                # results is a dict of form:
                # {
                #   "delta_t": ...,
                #   "device": ...,
                #   "avg_execution_time": ...,
                #   ...
                # }

                # Store each metric
                for mkey in metric_keys:
                    val = results[mkey]
                    row = {
                        "layout": layout_folder_name,
                        "scenario": os.path.basename(scenario_file),
                        "strategy": strat_name,
                        "metric": mkey,
                        "value": val
                    }
                    rows.append(row)

    # Build a DataFrame from all rows
    df = pd.DataFrame(rows, columns=["layout", "scenario", "strategy", "metric", "value"])
    print(df.head(100))
    return df

# -------------------------------------------------------------------
# 5) PLOT VIOLIN FOR EACH METRIC, GROUPED BY STRATEGY
#    We'll combine all layouts in one distribution per strategy.
# -------------------------------------------------------------------
def plot_violin_for_each_metric(df, output_dir="plots_violin"):
    """
    1) Takes the DataFrame from gather_data_from_layouts.
    2) For each unique metric in df["metric"], we:
       - Filter the rows for that metric
       - Make a seaborn violin plot with x='strategy', y='value'
       - Save to a PNG file
    """

    # Create output dir to store images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    metric_list = sorted(df["metric"].unique())

    for metric in metric_list:
        subset = df[df["metric"] == metric]

        plt.figure(figsize=(8, 6))
        sns.violinplot(x="strategy", y="value", data=subset, inner="box", cut=0)
        sns.despine()

        plt.title(f"{metric} distribution by strategy\n(All Layouts Combined)")
        plt.xlabel("Strategy")
        plt.ylabel(metric)


        # We'll rotate x-axis labels if you have many strategies
        plt.xticks(rotation=30)

        # Save figure as PNG
        out_filename = f"violin_{metric}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()  # close figure to avoid memory buildup

        print(f"[Saved] {out_path}")

# -------------------------------------------------------------------
# 6) MAIN USAGE
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose your root dataset folder is:
    # /Users/you/Desktop/wildfire_drone_routing/MinimalDataset
    root_dataset = "/Users/josephye/Desktop/wildfire_drone_routing/MinimalDataset"

    # Strategies: we use the global STRATEGIES list
    # If you want to define them here, that's also fine.

    # 1) Gather data from multiple layout folders
    df_all = gather_data_from_layouts(
        root_folder=root_dataset,
        strategies=STRATEGIES,
        custom_init_params_fn=build_custom_init_params,
        custom_step_params_fn=return_no_custom_parameters,
        starting_time=0,
        max_n_scenarii=None
    )

    if df_all is None or df_all.empty:
        print("No data to plot. Exiting.")
    else:
        print("DataFrame shape:", df_all.shape)
        print(df_all.head())

        # 2) Plot violin plots (one figure per metric), saved as PNG
        plot_violin_for_each_metric(df_all, output_dir="plots_violin")
