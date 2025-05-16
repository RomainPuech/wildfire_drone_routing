# WILDFIRE-DRONE-BENCH


<div align="center">
  <img src="https://github.com/user-attachments/assets/b5653f58-ff62-40d8-a422-4af13cd0ccd0" width="40%">

</div>



This library is designed to benchmark sensor placement and drone routing strategies on the "sim2real" dataset. It provides tools for both benchmarking and visualizing the performance of different strategies in wildfire scenarios.

## 🎥 Video Tutorial:
<p align="center">
https://www.loom.com/share/fe6ef53b6c6140fd8c25043bbb0feb0c?sid=73db9697-35a1-4592-b457-27540cf64bd1 
</p>

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
    ground_placement_strategy=<Your placement strategy>,
    drone_routing_strategy=<Your routing strategy>,
    ground_parameters=<(arg_placement1, arg_placement2, ...)>,
    routing_parameters=<(arg_routing1, arg_routing2, ...)>
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
    drone_locations_history=None
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
  author = {Romain Puech, Joseph Ye, Danique De Moor, Ana Trisovic},
  title = {Wildfire Drone Routing},
  year = {2025},
  howpublished = {\url{https://github.com/RomainPuech/wildfire_drone_routing}},
  note = {Accessed: YYYY-MM-DD}
}
```
