# Wildfire Drone Routing

This library is designed to benchmark sensor placement and drone routing strategies on the "sim2real" dataset. It provides tools for both benchmarking and visualizing the performance of different strategies in wildfire scenarios.

## 🚀 Features 

- **Benchmarking**: Evaluate different sensor placement and drone routing strategies using the "sim2real" dataset.
- **Visualization**: Generate visual representations of wildfire scenarios, including drone movements, fire spread, and sensor placements.
- **Data Processing**: Convert and preprocess data from various formats (e.g., JPG to NPY) for efficient analysis.

## 🛠️ Installation 

To use this library, first download the "sim2real" dataset from [Sim2Real-Fire GitHub repository](https://github.com/TJU-IDVLab/Sim2Real-Fire). Then, clone the repository and install the required dependencies:

```bash
git clone https://github.com/RomainPuech/wildfire_drone_routing.git
cd wildfire_drone_routing
pip install -r requirements.txt
```

## 📚 Usage 

### Preprocessing Data

Before running benchmarks, preprocess the "sim2real" dataset:

```python
from dataset import preprocess_sim2real_dataset

preprocess_sim2real_dataset("./path_to_dataset")
```

### Running Benchmarks

To benchmark a specific strategy, use the following function:

```python
from benchmark import benchmark_on_sim2real_dataset
from Strategy import GroundPlacementStrategy, RoutingStrategy

benchmark_on_sim2real_dataset(
    dataset_folder_name="./path_to_dataset",
    ground_placement_strategy=GroundPlacementStrategy,
    drone_routing_strategy=RoutingStrategy,
    ground_parameters=(10, 10, 100),
    routing_parameters=(10,)
)
```

### Visualization

Visualize the results of a scenario:

```python
from displays import create_scenario_video
from dataset import load_scenario_npy

scenario, starting_time = load_scenario_npy("./path_to_scenario.npy")
create_scenario_video(
    scenario_or_filename=scenario,
    drone_locations_history=None,
    burn_map=False,
    out_filename="simulation",
    starting_time=starting_time
)
```

## 📄 License 

This project is licensed under the MIT License.

## 📧 Contact 

For any questions or issues, please contact Romain Puech at puech@mit.edu.

## 📑 Citation

If you use this library in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{wildfire_drone_routing,
  author = {Romain Puech},
  title = {Wildfire Drone Routing},
  year = {2025},
  howpublished = {\url{https://github.com/your-repo-url}},
  note = {Accessed: YYYY-MM-DD}
}
```
