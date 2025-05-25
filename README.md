# WILDFIRE-DRONE-BENCH

<div align="center">
  <img src="https://github.com/user-attachments/assets/b5653f58-ff62-40d8-a422-4af13cd0ccd0" width="40%">
</div>

A comprehensive benchmarking library for evaluating sensor placement and drone routing strategies in wildfire detection scenarios. This library provides tools for testing, visualizing, and comparing different strategies using the "sim2real" dataset.

## 🎥 Video Tutorial
<p align="center">
https://www.loom.com/share/fe6ef53b6c6140fd8c25043bbb0feb0c?sid=73db9697-35a1-4592-b457-27540cf64bd1 
</p>

## 🚀 Features

- **Strategy Development**: Implement and test custom sensor placement and drone routing strategies
- **Dataset Integration**: Seamless integration with the Sim2Real-Fire dataset
- **Benchmarking**: Comprehensive evaluation of strategies with multiple metrics
- **Visualization**: Generate videos and plots of drone movements, fire spread, and sensor placements
- **Performance Optimization**: Support for both JPEG and NPY formats for memory/speed trade-offs

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/RomainPuech/wildfire_drone_routing.git
cd wildfire_drone_routing
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Julia (version 1.11.2 or later) and required packages:
```julia
using Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")  # or your preferred solver
```

4. Download the modified Sim2Real dataset:
```bash
# Download from Hugging Face
https://huggingface.co/datasets/MasterYoda293/DroneBench/tree/main
```

## 📚 Dataset Structure

The library works with the following dataset structure:
```
layout_folder/
├── Satellite_Image_Mask/
│   └── scenario_001/
│       ├── 0001.jpg
│       ├── 0002.jpg
│       └── ...
├── Weather_Data/
│   └── scenario_001.txt
├── static_risk.npy (burn map)
└── other layout info (topography, elevation, etc.)
```

- **Scenarios**: Represent wildfire spread over time
  - JPEG format: Folder of images, one per timestep
  - NPY format: Single file containing all timesteps
- **Burn Maps**: 3D arrays (time × height × width) representing fire probability
- **Weather Data**: Text files containing weather conditions for each scenario

## 🔧 Configuration

### Drone Parameters
- Coverage radius (m)
- Transmission range (m)
- Maximum battery time (hours)
- Speed (m/min)

### Sensor Types
- Ground sensors: Static fire detection
- Drones: Mobile fire detection
- Charging stations: Fire detection + drone charging

### Coverage and Movement
- Square coverage areas (Manhattan distance)
- Drones must start at charging stations
- Multiple drones can charge simultaneously
- Charging takes 1 timestep

## 💻 Usage

### 1. Preprocessing Dataset
```python
from dataset import preprocess_sim2real_dataset

# Convert JPEG scenarios to NPY format
preprocess_sim2real_dataset(
    "./path_to_dataset",
    n_max_scenarii_per_layout=100,  # Optional: limit scenarios per layout
    n_max_layouts=10  # Optional: limit number of layouts
)
```

### 2. Implementing Strategies

Create a new sensor placement strategy:
```python
from Strategy import SensorPlacementStrategy

class MySensorStrategy(SensorPlacementStrategy):
    def get_locations(self):
        """
        Returns two lists of (x,y) coordinates:
        - ground_sensor_locations: List of ground sensor positions
        - charging_station_locations: List of charging station positions
        """
        # Implement your sensor placement logic
        return ground_locations, charging_locations
```

Create a new drone routing strategy:
```python
from Strategy import DroneRoutingStrategy

class MyDroneStrategy(DroneRoutingStrategy):
    def get_initial_drone_locations(self):
        """
        Returns a list of tuples (state, (x,y)) where:
        - state is either 'charge' or 'fly'
        - (x,y) are the initial coordinates
        Drones must start at charging stations (state='charge')
        """
        # Implement initial drone placement
        return initial_locations

    def next_actions(self, automatic_step_parameters, custom_step_parameters):
        """
        Returns a list of tuples (action_type, coordinates) where:
        - action_type is one of: 'fly', 'move', 'charge'
        - coordinates are the target position (x,y)
        
        Parameters:
        - automatic_step_parameters: Dict containing:
            - drone_locations: List of current drone positions
            - drone_batteries: List of current drone battery levels
            - drone_states: List of current drone states
            - t: Current time step
        - custom_step_parameters: Dict for custom strategy parameters
            - Can include any data except actual fire location
            - Useful for ML burn map model inputs or precomputed burn maps
        """
        # Implement drone movement logic
        return actions
```

### Available Strategy Implementations

The library includes several pre-implemented strategies:

1. **Sensor Placement Strategies**:
   - `SensorPlacementOptimization`: Uses Julia optimization to find optimal sensor locations
   - `LoggedSensorPlacementStrategy`: Caches optimization results for faster repeated runs
   - `RandomSensorPlacementStrategy`: Places sensors randomly (for testing)

2. **Drone Routing Strategies**:
   - `DroneRoutingLinearMinTime`: Uses linear programming to minimize detection time
   - `GREEDY_DRONE_STRATEGY`: A heuristic approach for quick routing
   - `RandomDroneRoutingStrategy`: Random drone movements (for testing)

### Custom Parameters

Strategies can be configured using custom parameters:

1. **Required Parameters**:
   - `burnmap_filename`: Path to the burn map file (required for optimization strategies)

2. **Optional Parameters**:
   - `call_every_n_steps` or `reevaluation_step`: Steps between optimization calls
   - `optimization_horizon`: Number of steps to optimize for
   - `log_file`: Path to cache optimization results
   - Any additional parameters needed by your strategy

### Strategy Wrappers

The library provides wrapper functions to enhance your strategies with additional functionality:

#### Logging Wrappers

Use these wrappers to automatically log and cache strategy results:

```python
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy

# Wrap your strategies to add logging
LoggedSensorStrategy = wrap_log_sensor_strategy(MySensorStrategy)
LoggedDroneStrategy = wrap_log_drone_strategy(MyDroneStrategy)

# Use the wrapped strategies
sensor_strategy = LoggedSensorStrategy(...)
drone_strategy = LoggedDroneStrategy(...)
```

Benefits:
- **Reproducibility**: Ensures identical results across runs
- **Performance**: Skips expensive recomputation by caching results
- **Debugging**: Provides detailed logs of all actions and placements

The log files (JSON format) contain:
- Initial sensor and charging station placements
- Complete history of drone movements and actions
- All parameters used for the strategy

#### Clustering/Decomposition Wrappers

For large scenarios, use clustering wrappers to decompose the problem:

```python
from new_clustering import get_wrapped_clustering_strategy

# Wrap your drone routing strategy with clustering
ClusteredDroneStrategy = get_wrapped_clustering_strategy(MyDroneStrategy)
drone_strategy = ClusteredDroneStrategy(...)
```

The clustering wrapper:
1. Divides the environment into manageable clusters
2. Assigns drones to specific clusters
3. Coordinates the sub-strategies for each cluster

### 3. Running Benchmarks

Benchmark a single scenario:
```python
from benchmark import run_benchmark_scenario

results = run_benchmark_scenario(
    scenario=scenario,
    sensor_placement_strategy=MySensorStrategy,
    drone_routing_strategy=MyDroneStrategy,
    custom_initialization_parameters={},  # Optional: Add custom parameters for your strategy
    custom_step_parameters_function=lambda: {},  # Optional: Add custom parameters for each step
    return_history=True  # Optional: Get drone movement history for visualization
)

# results is a dictionary containing metrics:
# - delta_t: Time to fire detection
# - device: Which device detected the fire
# - execution_time: Strategy computation time
# - fire_size_cells: Fire size at detection
# - fire_percentage: Percentage of area burned
# - map_explored: Percentage of area explored
# - total_distance: Total distance traveled by drones
```

Benchmark multiple scenarios:
```python
from benchmark import run_benchmark_scenarii_sequential

results = run_benchmark_scenarii_sequential(
    input_dir="./path_to_scenarios",
    sensor_placement_strategy=MySensorStrategy,
    drone_routing_strategy=MyDroneStrategy,
    custom_initialization_parameters_function=lambda x: {...},
    custom_step_parameters_function=lambda: {...},
    max_n_scenarii=100,  # Optional: limit number of scenarios per layout
    max_n_layouts=10  # Optional: limit number of layouts
)
```

### 4. Visualization

Create a video of drone movements:
```python
from displays import create_scenario_video

# drone_locations_history is a list of lists of (x,y) coordinates
# Each inner list represents drone positions at a specific time step
create_scenario_video(
    scenario_or_filename=scenario,
    drone_locations_history=drone_history,
    ground_sensor_locations=sensor_locations,
    charging_stations_locations=charging_locations,
    out_filename="simulation"  # Outputs MP4 video
)
```

## 📊 Benchmarking Metrics

The library collects the following metrics:
- `delta_t`: Time to fire detection
- `device`: Which device detected the fire
- `execution_time`: Strategy computation time
- `avg_execution_time`: Average time per step
- `fire_size_cells`: Fire size at detection
- `fire_percentage`: Percentage of area burned
- `map_explored`: Percentage of area explored
- `total_distance`: Total distance traveled by drones
- `drone_entropies`: Entropy of drone positions
- `sensor_entropies`: Entropy of sensor positions

## ⚠️ Limitations and Error Handling

- No parallel processing support
- Rectangular coverage areas (Manhattan distance)
- No built-in battery warning system
- No recovery from illegal positions
- Invalid actions from strategies will be flagged during benchmarking
- No validation checks for sensor placement
- Errors during benchmarking are raised as exceptions:
  - Invalid drone positions (outside grid or not starting at charging station)
  - Invalid action types
  - Other strategy-specific errors

## 🔍 Additional Resources

For more details about the library and its implementation, refer to:
- Paper: "WFDroneBench: A Benchmark for Sensor Placement and Drone Routing for Wildfire Detection"
- Dataset: [Sim2Real-Fire](https://github.com/TJU-IDVLab/Sim2Real-Fire)
- Modified Dataset: [DroneBench](https://huggingface.co/datasets/MasterYoda293/DroneBench/tree/main)

## 📝 Notes

- Weather data is provided in the Sim2Real dataset format but is not used in the current strategies
- Users can add their own scenarios by following the Sim2Real dataset format
- Custom parameters can be added to strategies through `custom_initialization_parameters` and `custom_step_parameters`
- Drone movement history can be obtained by setting `return_history=True` in `run_benchmark_scenario`
- Runtime performance varies by strategy - see our paper for detailed benchmarks
- No specific memory usage considerations for large datasets

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions, issues, or collaboration, contact Romain Puech at puech@mit.edu.

## 📑 Citation

If you use this library in your research, please cite:
```bibtex
@misc{wildfire_drone_routing,
  author = {Romain Puech, Joseph Ye, Danique De Moor, Ana Trisovic},
  title = {Wildfire Drone Routing},
  year = {2025},
  howpublished = {\url{https://github.com/RomainPuech/wildfire_drone_routing}},
  note = {Accessed: YYYY-MM-DD}
}
```
