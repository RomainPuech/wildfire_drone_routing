# Project Overview

This document provides a high-level overview of the **Wildfire Drone Routing** project: its purpose, architecture, folder structure, dependencies, and how to get started.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Dependencies](#4-dependencies)
5. [Julia Integration](#5-julia-integration)
6. [Configuration Files](#6-configuration-files)
7. [Getting Started](#7-getting-started)

---

## 1. Purpose

**Wildfire Drone Routing** is a benchmarking framework for evaluating **sensor placement** and **drone routing** strategies in wildfire detection scenarios. The system simulates a wildfire spreading on a 2D grid and evaluates how quickly a fleet of drones (and static ground sensors) can detect the fire.

The core question the framework answers is:

> Given a wildfire scenario, a set of ground sensors, charging stations, and drones — how long does it take to detect the fire, and which device detects it?

The framework supports:

- **Multiple strategy implementations**: from random baselines to Julia-based mathematical optimization
- **Sim2Real dataset integration**: real wildfire spread data converted into simulation-ready formats
- **Automated benchmarking**: batch evaluation across hundreds of scenarios and layouts
- **Visualization**: video generation showing fire progression, drone movements, and sensor placements

---

## 2. Architecture

The project follows a modular, strategy-based architecture. The key components are:

```
┌──────────────────────────────────────────────────────────────┐
│                    Experiment Runner                          │
│          (all_experiments_parallel.py / experiments.ipynb)    │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                 Benchmarking Engine                           │
│                   (benchmark.py)                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Operational scaling (data → operational space)    │    │
│  │  • Simulation loop (time steps × substeps)           │    │
│  │  • Fire detection (sensors, drones, charging stations)│   │
│  │  • Metrics collection & CSV export                    │   │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────┬──────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
┌──────────────┐ ┌─────────────┐ ┌───────────────┐
│   Strategy   │ │   Wrappers  │ │   Drone       │
│ (Strategy.py)│ │(wrappers.py)│ │  (Drone.py)   │
│              │ │             │ │               │
│ • Sensor     │ │ • Logging   │ │ • Position    │
│   Placement  │ │ • Caching   │ │ • Battery     │
│ • Drone      │ │ • Clustering│ │ • Movement    │
│   Routing    │ │             │ │ • State       │
└──────┬───────┘ └─────────────┘ └───────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Julia Optimization                         │
│               (my_julia_caller.py → julia/*.jl)              │
│  ┌────────────────────┐ ┌────────────────────┐              │
│  │ground_charging_opt │ │drone_routing_opt   │              │
│  │       .jl          │ │       .jl          │              │
│  └────────────────────┘ └────────────────────┘              │
│  ┌────────────────────┐                                      │
│  │     TOP.jl         │ (PSO for Team Orienteering Problem)  │
│  └────────────────────┘                                      │
└──────────────────────────────────────────────────────────────┘
```

Supporting modules handle data loading (`dataset.py`), visualization (`displays.py`, `video_helpers.py`), and spatial clustering (`new_clustering.py`).

---

## 3. Folder Structure

```
wildfire_drone_routing/
├── code/                          # Python source code
│   ├── benchmark.py               # Benchmarking engine & simulation loop
│   ├── dataset.py                 # Data loading, saving, preprocessing
│   ├── Strategy.py                # Base classes & strategy implementations
│   ├── wrappers.py                # Logging, caching, clustering wrappers
│   ├── Drone.py                   # Drone state machine
│   ├── displays.py                # Visualization (images & videos)
│   ├── video_helpers.py           # Video generation from logs
│   ├── my_julia_caller.py         # Julia session management
│   ├── new_clustering.py          # Spatial clustering wrapper
│   └── dataset_creation/          # Dataset preprocessing scripts
│       ├── Scenario_sampler.py    # Scenario sampling & selection
│       ├── generate_csv.py        # CSV generation
│       ├── move_maps.py           # Risk map organization
│       └── risk_layouts/          # Raw risk map data (.tif, .npy)
│
├── julia/                         # Julia optimization modules
│   ├── ground_charging_opt.jl     # Sensor placement optimization (JuMP/Gurobi)
│   ├── drone_routing_opt.jl       # Drone routing optimization (JuMP/Gurobi)
│   ├── TOP.jl                     # PSO for the Team Orienteering Problem
│   ├── TOP_PSO_multi_depot.jl     # Multi-depot TOP variant
│   ├── helper_functions.jl        # Shared Julia utilities
│   └── test_*.jl                  # Julia test scripts
│
├── julia_env/                     # Julia package environment
│   ├── Project.toml               # Julia dependencies (JuMP, Gurobi, etc.)
│   └── Manifest.toml              # Locked Julia package versions
│
├── MiniTractDataset/              # Small evaluation dataset
├── MiniTractDatasetFull/          # Full evaluation dataset
├── WideDataset/                   # Wide-area evaluation dataset
├── MinimalDataset/                # Minimal test dataset
│
├── all_experiments_parallel.py    # Main experiment entry point
├── experiments.ipynb              # Interactive experiment notebook
├── quicktest.py                   # Quick smoke test
│
├── config_s2r.json                # Sim2Real dataset configuration (scenario offsets)
├── config_tract_mini.json         # Tract mini dataset configuration
├── environment.yml                # Conda environment (Linux)
├── environment_macos.yml          # Conda environment (macOS)
│
├── PSO.md                         # PSO algorithm documentation
├── documentation/                 # Project documentation (this folder)
├── README.md                      # Project README
└── results/                       # Aggregated benchmark results
```

---

## 4. Dependencies

### Python (3.10)

The project uses a Conda environment. Key Python packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, scenario data |
| `pandas` | CSV handling for benchmark results |
| `matplotlib` | Plotting and grid image generation |
| `scipy` | Entropy computation for metrics |
| `pillow` (PIL) | JPEG image I/O for scenarios |
| `rasterio` | GeoTIFF (`.tif`) file I/O |
| `opencv-python` | Video creation |
| `imageio` | Alternative video I/O |
| `shapely` | Geometric operations in clustering |
| `tqdm` | Progress bars |
| `julia` (PyJulia) | Python-Julia bridge |
| `juliacall` / `juliapkg` | Julia package management from Python |

### Julia (1.11.2+)

Julia packages are managed through `julia_env/Project.toml`:

| Package | Purpose |
|---------|---------|
| `JuMP` | Mathematical optimization modeling |
| `Gurobi` | Commercial MIP/LP solver |
| `Graphs` | Graph algorithms |
| `NearestNeighbors` | Spatial queries |
| `Clustering` | K-means and other clustering |
| `NPZ` | Reading `.npy` files from Julia |
| `DataFrames` / `CSV` | Data manipulation |
| `Plots` / `Cairo` | Julia-side plotting |
| `Statistics` | Statistical functions |

---

## 5. Julia Integration

The project uses **PyJulia** to call Julia optimization routines from Python. The Julia session is managed as a singleton through `my_julia_caller.py`:

```python
# my_julia_caller.py — simplified
from julia.api import Julia

def initialize_julia_session():
    # Try fast path first, fall back to slow path
    try:
        jl = Julia(compiled_modules=True, runtime=julia_runtime)
    except Exception:
        jl = Julia(compiled_modules=False, runtime=julia_runtime)

    from julia import Main
    Main.include("julia/ground_charging_opt.jl")
    Main.include("julia/drone_routing_opt.jl")
    Main.include("julia/TOP.jl")
    return jl, Main
```

### Key Design Decisions

1. **Singleton pattern**: The Julia session is initialized once and shared across the entire Python process. This avoids the startup cost on every call.
2. **Eager initialization**: The session initializes at import time (`_julia_session, Main = initialize_julia_session()` at module level), so importing `my_julia_caller` triggers Julia startup.
3. **Shared `Main` object**: Other modules import `Main` from `my_julia_caller` and call Julia functions directly: `jl.some_function(args...)`.
4. **Error logging suppression**: Julia's info-level logs are silenced to keep Python output clean.

### Fast Julia Startup with `python-jl`

By default, Conda's Python is statically linked to libpython, which forces PyJulia to use `compiled_modules=False` (~80s startup). The `python-jl` wrapper (bundled with PyJulia) loads the Julia runtime first, enabling `compiled_modules=True` (~13s startup).

```bash
# Slow (fallback):
python run_benchmark.py          # ~80s Julia init

# Fast (recommended):
python-jl run_benchmark.py       # ~13s Julia init
```

**One-time setup** (already done): PyCall must be built against the project's Python environment:

```bash
conda activate wf
PYTHON=$(which python3) julia -e 'ENV["PYTHON"]=ENV["PYTHON"]; import Pkg; Pkg.build("PyCall")'
```

### Three Julia Modules

| Module | Role |
|--------|------|
| `ground_charging_opt.jl` | Optimizes sensor and charging station placement on the grid using JuMP/Gurobi |
| `drone_routing_opt.jl` | Solves drone routing as a mathematical program (reusable index model) |
| `TOP.jl` | Particle Swarm Optimization for the Team Orienteering Problem (see `PSO.md`) |

---

## 6. Configuration Files

### `config_s2r.json`

A JSON file mapping layout/scenario names to **starting time offsets**. Each offset indicates how many data timesteps before fire ignition the simulation begins (giving drones a head start):

```json
{
  "offset_Lake_Fire": 1,
  "offset_August_Complex_fire": 2,
  "offset_Woolsey_Fire": 1,
  ...
}
```

These offsets are used by the benchmark to set `starting_time` per scenario, allowing drones to begin patrolling before the fire starts.

### Simulation Parameters

Simulation parameters are typically defined in the experiment script:

```python
simulation_parameters = {
    "max_battery_distance": -1,        # Not used (time-based battery)
    "max_battery_time": 1,             # Battery life in hours
    "n_drones": 2,                     # Number of drones
    "n_ground_stations": 8,            # Number of ground sensors
    "n_charging_stations": 2,          # Number of charging stations
    "drone_speed_m_per_min": 600,      # Drone speed (m/min)
    "coverage_radius_m": 300,          # Sensor/drone coverage radius (m)
    "cell_size_m": 30,                 # Data cell size (m)
    "transmission_range": 50000,       # Transmission range (m)
}
```

---

## 7. Getting Started

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/RomainPuech/wildfire_drone_routing.git
cd wildfire_drone_routing
```

2. **Create the Python environment**:
```bash
# Linux
conda env create -f environment.yml
conda activate juliaenv

# macOS
conda env create -f environment_macos.yml
conda activate wf
```

3. **Install Julia packages**:
```julia
using Pkg
Pkg.activate("julia_env")
Pkg.instantiate()
```

4. **Download the dataset** from [HuggingFace](https://huggingface.co/datasets/MasterYoda293/DroneBench/tree/main).

### Running Experiments

**Quick test** (single layout):
```bash
python quicktest.py
```

**Full experiment suite** (all strategies, parallel):
```bash
python all_experiments_parallel.py --ss_prefix S --bm_prefix whp
```

Where:
- `--ss_prefix`: Sensor strategy prefix (`S` for optimized, `R` for random)
- `--bm_prefix`: Burn map type (`whp` for wildfire hazard potential, `bm` for dynamic burn map, `bp` for burn probability)

**Interactive exploration** via Jupyter:
```bash
jupyter notebook experiments.ipynb
```

### Workflow Summary

```
1. Download dataset          →  MiniTractDataset/, WideDataset/
2. Preprocess (JPG → NPY)   →  dataset.preprocess_sim2real_dataset()
3. Compute burn maps         →  dataset.compute_and_save_burn_maps_sim2real_dataset()
4. Choose strategies         →  sensor + drone strategy classes
5. Run benchmarks            →  benchmark_on_sim2real_dataset_precompute_parallel()
6. Collect results           →  dataset.combine_all_benchmark_results()
7. Visualize                 →  displays.create_scenario_video()
```

---

*Next: [02 — Data Pipeline](02_data_pipeline.md)*
