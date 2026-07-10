# Minutes Matter: Rapid Wildfire Detection Through Sensor Placement and Drone Routing

Code for the paper:

> Puech, R., de Moor, D., Trišović, A., & Bertsimas, D. (2026). *Minutes Matter: Rapid Wildfire Detection Through Sensor Placement and Drone Routing.* [Journal TBD]. DOI: [TBD]

This repository contains the infrastructure placement and drone routing optimization models implemented in Julia, and the Python scripts used for data preprocessing, simulation, and figure generation.

## Repository structure

```
run_benchmark_california2021_yearly.py — canonical benchmark runner (placement cache + routing + simulation; produced the paper data)
run_benchmark_california_yearly.py     — cleaned multi-year convenience wrapper
test_budget_placement_station_max_greedy_uniform_2021.py  — placement entry point (greedy-uniform StationMax; 20/50/100M)
test_budget_placement_station_max_uniform_fixed_drones_2021.py — placement entry point (uniform fixed drones; 500M)
test_budget_placement_station_max_2021.py, test_budget_placement_station_max_uniform_2021.py — placement variants
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
  Nature_Wildfires/                 — figure/table-building scripts + committed figure PNGs
    make_figure3_frontier.py        — Fig 2: detection frontier plot
    scripts/                        — table builders (build_table1/2, collect_runtimes, …)
  figure4/                          — Fig 3: deployment composite map
  figure5bis/                       — Fig 4: cost-sensitivity line plot
  figure6/                          — Fig 5: ALERTCalifornia coverage maps
  breakeven_figure/                 — shared cost-sensitivity drawing module (used by Fig 3 & 4)
  final_report/
    generate_final_report.py        — Fig 2: detection frontier data processing
    csv/                            — pre-computed benchmark results (137 CSV files)
    placement_data/logs/            — pre-computed panel placement JSONs for Fig 3 (20/50/100/500M)
  breakeven_report/
    breakeven_sensor_cost_export/
      placement_logs/               — pre-computed placement logs for Fig 4 (48 JSON files)

report/
  benchmark_2021_greedy_kernel/     — SLURM submission scripts + pipeline docs (HPC reproduction)

CODEBASE.md                         — full codebase reference for developers and agents
cameras.json                        — ALERTCalifornia camera metadata
environment.yml                     — Python environment (Linux / HPC)
environment_macos.yml               — Python environment (macOS)
```

## Installation

### Python

We use Python 3.10. Create and activate the conda environment:

```bash
conda env create -f environment.yml         # Linux / HPC  -> env name: juliaenv
conda activate juliaenv
# or, on macOS (includes the geopandas/rasterio geospatial stack):
conda env create -f environment_macos.yml   # macOS        -> env name: wf
conda activate wf
```

The Linux/HPC env is named `juliaenv` and the macOS env is named `wf`. The
geo-dependent figures (Fig 3, Fig 5, Fig 6) need `geopandas`, `rasterio`, and
`pyproj`; these are included in the macOS `wf` env. Add them to `juliaenv` if you
generate those figures on Linux.

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

**Everything needed to reproduce the benchmark and Figures 2–5 is committed in this
repository** (~47 MB). No external download is required for the main results. Each
`California<year>Dataset/` (2021–2024) ships the benchmark-critical files:

```
California2021Dataset/
  config_california_2021.json    # per-scenario ignition times / metadata
  mask.npy                       # California burnable-cell mask (1, 1309, 805)
  static_risk_pyrologix.npy      # Pyrologix static risk map — sensor placement + drone routing
  scenarii/                      # per-fire ignition points (*.npy)
California2022Dataset/  …  California2023Dataset/  …  California2024Dataset/
```

The Fig-6 mask `California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy`
is also committed, along with the cached WFPI geo-referencing
(`code/dataset_creation/nature_dataset_creation/wfpi_georef.json`), so the geospatial
figures need no raster archives.

### Large inputs (only for dataset creation / Fig 6)

The following are **not** needed to reproduce the benchmark or Figures 2–5 — they are
only used to rebuild the datasets from scratch or to regenerate the Fig-6 case-study
explainer. Because of their size they are hosted on HuggingFace rather than in Git:

```
https://huggingface.co/datasets/romainpuech/wildfire-drone-routing-data
```

Contents and where to place each item (paths relative to the repo root):

| HuggingFace path | Place at | Used by |
|---|---|---|
| `raw_creation_inputs/*` | `code/dataset_creation/nature_dataset_creation/data/` | dataset creation, Fig 6 (USFS CSVs, Pyrologix GPKG, TIGER shapefiles, WFPI forecast-1 **and** forecast-2 daily zips, `cameras.json`) |
| `day1_wfpi_2020/` | `California2020Dataset_Day1/` | rebuilding the Fig-6 union-burnable mask |
| `day2_wfpi_2021/` | `California2021Dataset/` | rebuilding the 2021 dataset (daily `wfpi_*.npy`, `static_risk_wfpi_yearly.npy`) |

The WFPI **forecast-2** daily archives (which drive the day-2 risk *values* and the
Fig-6 missing-date fire exclusion) are included under
`raw_creation_inputs/2021_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA/`, so the
per-year datasets can be rebuilt from scratch end-to-end. Each script header in
`code/dataset_creation/nature_dataset_creation/` documents the expected format. None of
this is needed for the benchmark or Figures 2–5, which use the committed Tier-1 data.

## Reproducing paper results

> **Fastest path:** all heavy outputs are pre-computed and committed — benchmark
> CSVs in `paper/final_report/csv/` and placement JSONs in
> `paper/final_report/placement_data/logs/` and
> `paper/breakeven_report/breakeven_sensor_cost_export/placement_logs/`. You can
> jump straight to **Step 2** to regenerate the figures. The full pipeline (Step 1)
> only needs to be re-run to reproduce those artifacts from scratch, and requires
> a Gurobi license and ideally an HPC cluster (the datasets are committed).

### Step 1 — Run the benchmark (optional; needs Gurobi)

The benchmark runs in two phases. The canonical entry point that produced the
paper data is `run_benchmark_california2021_yearly.py` (run via `python-jl` so a
single Julia session is reused; the placement scripts also use `python-jl`).

**1a. Infrastructure placement** (once per budget; ILP solved by Gurobi, result
cached to `California2021Dataset/logs/sensor_alloc_GaussianBudget<B>M_*_261x161_mean.json`).
The \$20M/\$50M/\$100M placements use the *greedy-uniform StationMax* optimizer
(7 drones per station); the \$500M placement uses the *uniform-fixed-drones* variant:

```bash
# 20M / 50M / 100M (greedy-uniform StationMax)
python-jl test_budget_placement_station_max_greedy_uniform_2021.py --budget 20  --time-limit 600
python-jl test_budget_placement_station_max_greedy_uniform_2021.py --budget 100 --time-limit 43200

# 500M (uniform fixed drones per station)
python-jl test_budget_placement_station_max_uniform_fixed_drones_2021.py --budget 500 --time-limit 43200

# (equivalently: run_benchmark_california2021_yearly.py --budget <B> --sensor-only --time-limit <S>)
```

**1b. Drone routing + simulation** (per budget × routing strategy; writes one CSV).
The placement cache from 1a is loaded automatically:

```bash
python-jl run_benchmark_california2021_yearly.py --budget 100 --strategy TOPGrowing
python-jl run_benchmark_california2021_yearly.py --budget 100 --strategy MaxCov
python-jl run_benchmark_california2021_yearly.py --budget 100 --strategy LinearMinTime
```

`--budget` ∈ {20, 50, 100, 500}; `--strategy` is a case-insensitive substring
(`TOPGrowing`, `MaxCov`, `LinearMinTime`). See `--help` for the full option list.
On a cluster, use the SLURM scripts in Step 4 rather than running these by hand.

### Step 2 — Reproduce figures

The script-internal names predate the manuscript's final figure numbering, so the
table below maps each manuscript figure to its generator and output. Figures that
need the geospatial stack (California outline via geopandas/rasterio/pyproj) are
marked **geo**; install them in your environment (named `wf` on macOS) or the
outline is simply omitted. Figures 3–5 use the committed Tier-1 datasets; only Fig 6
additionally needs the large raw creation inputs.

| Manuscript figure | Script | Output / data |
|---|---|---|
| **Fig 2** — Detection rate & speed vs. budget | `paper/Nature_Wildfires/make_figure3_frontier.py` | `Figures/frontier.png` — from committed CSVs (no datasets needed) |
| **Fig 3** — Optimized deployment maps | `paper/figure4/generate_placement_composite_figure.py` *(geo)* | `Figures/placement_composite.png` — committed panel JSONs + committed datasets |
| **Fig 4** — Cost sensitivity to sensor cost | `paper/figure5bis/make_figure5bis_breakeven_lines.py` | `Figures/breakeven_costsensitivity_lines.png` — committed JSONs + committed datasets |
| **Fig 5** — ALERTCalifornia coverage | `paper/figure6/generate_alertcalifornia_composite_figure.py` *(geo)* | `Figures/alertcalifornia_coverage_composite.png` — `cameras.json` + committed datasets (georef cached; no raster needed) |
| **Fig 6** — California case-study region | `code/dataset_creation/nature_dataset_creation/generate_paper_2021_dataset_explainer.py` *(geo)* | `Figures/fig0{1..4}_*.png` — committed datasets + large raw WFPI/Pyrologix/USFS inputs |

```bash
# Example (Fig 2 works with no datasets):
python paper/Nature_Wildfires/make_figure3_frontier.py
# Geo figures (run inside the env that has geopandas/rasterio/pyproj):
python paper/figure4/generate_placement_composite_figure.py
```

### Step 3 — Build result tables

```bash
python paper/Nature_Wildfires/scripts/build_table1_detection.py     # -> table1_detection.tex
python paper/Nature_Wildfires/scripts/build_table2_alertcalifornia.py  # -> table2_alertcalifornia.tex
python paper/Nature_Wildfires/scripts/collect_runtimes.py           # -> methods_runtime_table.tex
```

These emit standalone LaTeX table fragments. The manuscript LaTeX source is **not**
part of this code release, so there is no full-PDF build step (`build_pdf.py`
remains for users who supply their own `sn-article.tex`).

### Step 4 — Running on an HPC cluster (MIT SuperCloud / SLURM)

The SLURM submission scripts used for the paper are in
`report/benchmark_2021_greedy_kernel/`:

- `supercloud_2_greedy_uniform_placement_array.sh` — placement array job (budgets 20/100/500).
- `supercloud_3_greedy_uniform_routing_array.sh`, `supercloud_3_greedy_uniform_routing_linear_array.sh` — routing array jobs.
- `supercloud_500M_placement_uniform_fixed_*.sh`, `supercloud_500M_fnature_routing_*.sh` — \$500M placement/routing.
- `supercloud_*_greedy_uniform_placement_breakeven_cs*.sh` — sensor-cost breakeven sweep (Fig 4 inputs).
- `reproduce_benchmark_2021_greedy_kernel.sh` — single-node driver that re-runs placement + placement figures end to end.

Each script loads the cluster toolchain (`module load anaconda/Python-ML-2025a julia gurobi`)
and calls the same `python-jl` entry points as Step 1. Submit with
`sbatch report/benchmark_2021_greedy_kernel/<script>.sh` from the repo root, or
adapt the array-job pattern to your scheduler. See
`report/benchmark_2021_greedy_kernel/benchmark_2021_greedy_kernel.md` for the
pipeline overview.

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
