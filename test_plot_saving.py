import os
import sys
import matplotlib.pyplot as plt

# Add code directory to path
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)

# Import necessary functions
from plot_metrics import plot_all_metrics_across_layouts
from Strategy import SensorPlacementOptimization, DroneRoutingOptimizationModelReuseIndex

# Define the root folder for the dataset
root_folder = "MinimalDataset/"

# Create output directories
os.makedirs("plots", exist_ok=True)

# Use plot_all_metrics_across_layouts to generate and save plots
print("\nGenerating plots across all layouts...")
plot_all_metrics_across_layouts(
    root_folder=root_folder,
    sensor_strategy_cls=SensorPlacementOptimization,
    drone_strategy_cls=DroneRoutingOptimizationModelReuseIndex,
    max_n_scenarii=100,
    starting_time=0
)

print("All plots have been saved to the 'plots' directory!") 