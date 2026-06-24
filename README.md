# Minutes Matter: Rapid Wildfire Detection Through Sensor Placement and Drone Routing

Code for the paper:

> Puech, R., de Moor, D., Trišović, A., & Bertsimas, D. (2026). *Minutes Matter: Rapid Wildfire Detection Through Sensor Placement and Drone Routing.* [Journal TBD]. DOI: [TBD]

This repository contains the infrastructure placement and drone routing optimization models implemented in Julia, and the Python scripts used for data preprocessing, simulation, and figure generation.

## Repository structure

```
run_benchmark_california_yearly.py  — main benchmark runner (all budget levels, all years)
preprocess_benchmark_2021.py        — preprocessing for 2021 dataset
visualize_sensor_placement_2021.py  — single-panel placement maps

code/
  benchmark.py                      — simulation engine and metrics
  Strategy.py                       — all placement and routing strategies
  dataset.py                        — dataset loading and preprocessing
  displays.py                       — publication-quality figure generation
  benchmark_alertcalifornia.py      — ALERTCalifornia camera baseline
  wrappers.py                       — logging and caching wrappers
  new_clustering.py                 — spatial clustering for large grids
  Drone.py, my_julia_caller.py, … — supporting modules
  dataset_creation/
    nature_dataset_creation/        — scripts to build California 2021–2024 datasets

julia/
  TOP.jl                            — Team Orienteering Problem routing (multi-depot)
  TOP_PSO_multi_depot.jl            — PSO heuristic for multi-drone routing
  ground_charging_opt.jl            — ILP placement optimizer (stations + sensors)
  drone_routing_opt.jl              — nonlinear routing optimizer
  drone_routing_opt_linear.jl       — linear-time routing variant
  helper_functions.jl               — shared utilities
  test_*.jl                         — unit and benchmark tests

julia_env/                          — Julia environment (Project.toml + Manifest.toml)

paper/
  Nature_Wildfires/                 — LaTeX source, figures, table-building scripts
  figure4/                          — Fig 4: placement composite map
  figure5bis/                       — Fig 5: cost-sensitivity line plot
  figure6/                          — Fig 6: ALERTCalifornia coverage maps
  final_report/
    generate_final_report.py        — Fig 3: detection frontier data processing
    csv/                            — pre-computed benchmark results (137 CSV files)
  breakeven_report/
    breakeven_sensor_cost_export/
      placement_logs/               — pre-computed placement logs for Fig 5 (48 JSON files)

CODEBASE.md                         — full codebase reference for developers and agents
cameras.json                        — ALERTCalifornia camera metadata
environment.yml                     — Python environment (Linux / HPC)
environment_macos.yml               — Python environment (macOS)
```

## Installation

### Python

We use Python 3.10. Create and activate the conda environment:

```bash
conda env create -f environment.yml   # Linux / HPC
# or
conda env create -f environment_macos.yml  # macOS
conda activate juliaenv
```

### Julia

Install Julia 1.11 or later, then instantiate the environment:

```julia
using Pkg
Pkg.activate("julia_env")
Pkg.instantiate()
```

This installs all exact package versions listed in `julia_env/Manifest.toml`, including JuMP, Gurobi, NPZ, and Plots.

**Note:** The optimization strategies (`ground_charging_opt.jl`, `drone_routing_opt.jl`) require a valid [Gurobi](https://www.gurobi.com/) license. Academic licenses are available for free.

## Data

The California 2021–2024 wildfire datasets (scenarios as `.npy` arrays, per-year WFPI burn maps, burnable masks, and config files) are hosted on HuggingFace:

```
https://huggingface.co/datasets/MasterYoda293/DroneBench
```

Download and place each year's dataset at the repo root so the directory layout is:

```
California2021Dataset/
  config_california_2021.json
  mask.npy
  wfpi_YYYYMMDD.npy  (365 files)
  scenarii/          (fire scenario .npy files)
California2022Dataset/
California2023Dataset/
California2024Dataset/
```

### Rebuilding datasets from raw sources

To reproduce the datasets from USFS ignition records and WFPI/Pyrologix rasters, see the scripts in `code/dataset_creation/nature_dataset_creation/` and the documentation in `documentation/`. Raw input data must be downloaded separately (sources listed in each script's header).

## Reproducing paper results

### Step 1 — Run the benchmark

```bash
python run_benchmark_california_yearly.py \
  --budgets 20000000 50000000 75000000 100000000 500000000 \
  --years 2021 2022 2023 2024 \
  --strategies TOPGrowing MaxCov LinearMinTime \
  --output-dir paper/final_report/csv/
```

This runs the placement optimizer and all routing strategies across all budget levels and years, writing one CSV per (year, budget, strategy) combination. The pre-computed CSVs are already included in `paper/final_report/csv/` so this step can be skipped to reproduce figures directly.

### Step 2 — Reproduce figures

Quick reference for figure reproduction:

| Figure | Script |
|--------|--------|
| Fig 2 — California dataset overview | `code/dataset_creation/nature_dataset_creation/generate_paper_2021_dataset_explainer.py` |
| Fig 3 — Detection frontier | `paper/Nature_Wildfires/make_figure3_frontier.py` |
| Fig 4 — Infrastructure placement maps | `conda run -n wf python paper/figure4/generate_placement_composite_figure.py` |
| Fig 5 — Cost sensitivity | `python paper/figure5bis/make_figure5bis_breakeven_lines.py` |
| Fig 6 — ALERTCalifornia coverage | `conda run -n wf python paper/figure6/generate_alertcalifornia_composite_figure.py` |

Figures 4 and 6 require the `wf` conda environment (geospatial stack: geopandas, rasterio).

### Step 3 — Build tables and PDF

```bash
python paper/Nature_Wildfires/scripts/build_table1_detection.py
python paper/Nature_Wildfires/scripts/build_table2_alertcalifornia.py
python paper/compile_pdf.py
```

## Running on an HPC cluster

The benchmark was run on MIT SuperCloud. See `documentation/` for an overview of the pipeline, and adapt the array-job pattern in `run_benchmark_california_yearly.py` to your scheduler.

## Julia optimization tests

```julia
cd julia/
include("run_extreme_tests_simple.jl")
```

Individual test files (e.g., `test_top_masked.jl`, `test_pso_august_complex_fire.jl`) can be run independently.

## Citation

```bibtex
@article{puech2026minutesmatter,
  author  = {Puech, Romain and de Moor, Danique and Tri\v{s}ovi\'{c}, Ana and Bertsimas, Dimitris},
  title   = {Minutes Matter: Rapid Wildfire Detection Through Sensor Placement and Drone Routing},
  year    = {2026},
  journal = {[Journal TBD]},
  doi     = {[DOI TBD]}
}
```

## License

MIT License. See `LICENSE`.
