import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
import logging
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from benchmark import benchmark_on_sim2real_dataset_precompute, return_no_custom_parameters, load_strategy, run_benchmark_for_strategy, print_simulation_parameters, build_custom_init_params

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def setup_directories(base_dir):
    folders = ["logs", "metrics", "reports", "visualizations"]
    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def setup_logging(results_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, "logs", f"run_{timestamp}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return log_path

def save_metrics(metrics, results_dir, dataset_path, model_name):
    """
    Save the benchmark metrics to a JSON file.

    Args:
        metrics (dict): Benchmark results.
        results_dir (str): Path to the results folder.
        dataset_path (str): Full path to the dataset folder.
        model_name (str): Name of the model (e.g., "JULIAX")
    """
    # Extract clean dataset name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_filename = f"benchmark_{dataset_path}_{model_name}_{timestamp}.json"
    metrics_path = os.path.join(metrics_dir, metrics_filename)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    print(f"✅ Metrics saved to {metrics_path}")

def save_latest_results(metrics, results_dir):
    latest_results_path = os.path.join(results_dir, "latest_results.json")

    with open(latest_results_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

    logging.info(f"Saved latest results to {latest_results_path}")

def update_leaderboard(metrics, results_dir, dataset_name, model_name):
    leaderboard_path = os.path.join(results_dir, "leaderboard.csv")
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset": dataset_name,
        "Model": model_name,
        "Average Detection Time": metrics.get("avg_time_to_detection", "N/A"),
        "Detection Rates": metrics.get("device_percentages", "N/A"),
        "Map Explored (%)": metrics.get("avg_map_explored", "N/A"),
        "Execution Time (s)": metrics.get("avg_execution_time", "N/A")
    }

    if os.path.exists(leaderboard_path):
        df = pd.read_csv(leaderboard_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(leaderboard_path, index=False)
    logging.info(f"Updated leaderboard at {leaderboard_path}")

def save_html_leaderboard(results_dir):
    leaderboard_path = os.path.join(results_dir, "leaderboard.csv")
    html_path = os.path.join(results_dir, "reports", "leaderboard.html")

    if os.path.exists(leaderboard_path):
        df = pd.read_csv(leaderboard_path)

        # Basic HTML formatting
        html_content = df.to_html(
            index=False,
            border=0,
            classes="table table-striped table-bordered",
            justify="center"
        )

        full_html = f"""
        <html>
        <head>
            <title>Benchmark Leaderboard</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css">
        </head>
        <body style="margin: 40px;">
            <h1>🔥 Benchmark Leaderboard</h1>
            {html_content}
        </body>
        </html>
        """

        with open(html_path, "w") as f:
            f.write(full_html)

        logging.info(f"Saved HTML leaderboard to {html_path}")
    else:
        logging.warning("Leaderboard CSV not found, skipping HTML generation.")


def save_markdown_report(metrics, results_dir, dataset_name, model_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    report_filename = f"benchmark_summary_{dataset_name}_{model_name}_{timestamp}.md"
    report_path = os.path.join(reports_dir, report_filename)

    with open(report_path, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"**Dataset**: {dataset_name}\n\n")
        f.write(f"**Model**: {model_name}\n\n")
        f.write(f"**Run Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Metrics\n")
        for k, v in metrics.items():
            if isinstance(v, list):
                continue  # Optional: Skip huge lists
            f.write(f"- **{k}**: {v}\n")
    
    logging.info(f"✅ Saved Markdown report to {report_path}")

def save_accuracy_plot(metrics, results_dir, dataset_name, model_name):
    visualizations_dir = os.path.join(results_dir, "visualizations")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(visualizations_dir, f"accuracy_plot_{dataset_name}_{model_name}_{timestamp}.png")

    if "raw_execution_times" in metrics:
        times = metrics["raw_execution_times"]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(times)+1), times, marker='o')
        plt.title(f"Execution Times per Scenario\n{dataset_name} - {model_name}")
        plt.xlabel("Scenario #")
        plt.ylabel("Execution Time (s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot to {plot_path}")
    else:
        logging.warning("No 'raw_execution_times' found in metrics; skipping plot.")

    
def main():
    parser = argparse.ArgumentParser(description="Run benchmark for wildfire drone project.")
    parser.add_argument("--dataset", required=True, help="Dataset folder path")
    parser.add_argument("--model", required=True, help="Model name (e.g., JULIAX, RANDOM)")
    parser.add_argument("--sensor_strategy", required=True, help="Sensor placement strategy file")
    parser.add_argument("--sensor_class", required=True, help="Sensor placement class name")
    parser.add_argument("--drone_strategy", required=True, help="Drone routing strategy file")
    parser.add_argument("--drone_class", required=True, help="Drone routing class name")
    parser.add_argument("--strategy_folder", default="Strategy", help="Folder where strategies are located")
    parser.add_argument("--max_n", type=int, default=None, help="Max number of scenarios to run")
    parser.add_argument("--starting_time", type=int, default=0, help="Starting time of fire")
    parser.add_argument("--file_format", type=str, default="npy", choices=["npy", "jpg"], help="Scenario file format")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store all benchmark results")

    args = parser.parse_args()

    layout_name = os.path.basename(os.path.dirname(args.dataset))
    custom_init_params = build_custom_init_params(args.dataset, layout_name)
    print_simulation_parameters(custom_init_params, title="🚀 Benchmark Settings")

    # Setup folders
    setup_directories(args.results_dir)

    logging.info("Starting benchmark runner...")
    logging.info(f"Parameters: {vars(args)}")


    try:
        metrics = run_benchmark_for_strategy(
            input_dir=args.dataset,
            strategy_folder=args.strategy_folder,
            sensor_strategy_file=args.sensor_strategy,
            sensor_class_name=args.sensor_class,
            drone_strategy_file=args.drone_strategy,
            drone_class_name=args.drone_class,
            max_n_scenarii=args.max_n,
            starting_time=args.starting_time,
            file_format=args.file_format
        )

        # Save outputs
        print("Metrics:", metrics)
        dataset_name = os.path.basename(args.dataset.rstrip('/'))
        save_metrics(metrics, args.results_dir, dataset_name, args.model)
        save_latest_results(metrics, args.results_dir)
        update_leaderboard(metrics, args.results_dir, dataset_name, args.model)
        save_markdown_report(metrics, args.results_dir, dataset_name, args.model)
        save_accuracy_plot(metrics, args.results_dir, dataset_name, args.model)
        save_html_leaderboard(args.results_dir)

        logging.info("Benchmark completed successfully!")

    except Exception as e:
        logging.exception(f"Error during benchmark: {e}")
        raise

if __name__ == "__main__":
    main()
