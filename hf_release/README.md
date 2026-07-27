---
language:
- en
license: cc-by-4.0
task_categories:
- other
tags:
- wildfire
- drones
- benchmark
- routing
- sensor-placement
- risk-map
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/scenarios_index.parquet
- config_name: tables23
  data_files:
  - split: test
    path: data/tables23_scenarios.parquet
---

# WFDroneBench – Wildfire Drone Routing Benchmark

## Dataset Description

WFDroneBench is a semi-synthetic benchmark for proactive wildfire detection
that couples risk maps with sensor/charging-station placement and
battery-constrained drone routing.  The dataset contains fire-spread
simulations derived from real US wildfire data sources (Sim2Real-Fire, USFS
Burn Probability, FPA-FOD).

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| Total layouts | 49 (+ 7 single-scenario layouts = 56) |
| Total scenarios | 7 746 (scenario_summary) / 7 753 (config) |
| Tables 2/3 split | 474 scenarios across 12 layouts |
| Grid resolution | 30 m/cell |
| Fire-spread format | `.npy` arrays (T × N × M) |
| Risk maps | `static_risk_bp2024.npy`, `static_risk_whp.npy`, `burn_map.npy` |

### Configs

- **`default`** — Full scenario index (all 7 746 scenarios with metadata).
- **`tables23`** — The 474-scenario, 12-layout subset used in Tables 2 & 3 of
  the paper, with per-scenario metadata and benchmark results.

### Loading

```python
from datasets import load_dataset

# Full index
ds = load_dataset("MasterYoda293/WFDroneBench", "default")

# Tables 2/3 split only
ds_t23 = load_dataset("MasterYoda293/WFDroneBench", "tables23")
```

### Data Fields

#### `default` config — `scenarios_index.parquet`

| Field | Type | Description |
|-------|------|-------------|
| `layout_id` | string | 4-digit layout identifier |
| `scenario_id` | string | 5-digit scenario identifier |
| `season_number` | float | Season index |
| `seasonal_match` | bool | Whether scenario matches seasonal pattern |
| `historical_match` | bool | Whether scenario matches USFS historical burn probability |
| `big_fire` | bool | Fire-size bin: big |
| `small_fire` | bool | Fire-size bin: small |
| `fast_fire` | bool | Fire-speed bin: fast |
| `slow_fire` | bool | Fire-speed bin: slow |

#### `tables23` config — `tables23_scenarios.parquet`

Same fields as above, plus:

| Field | Type | Description |
|-------|------|-------------|
| `historical_match_pct` | float | Percentage of scenarios with historical match for this layout |
| `per_layout_n` | int | Number of scenarios in this layout used in Tables 2/3 |
| `in_config_s2r` | bool | Whether scenario is present in config_s2r.json (zip offset index) |
| `in_scenario_summary` | bool | Whether scenario is present in scenario_summary.csv |

### Data Splits

| Split | Config | Scenarios | Layouts |
|-------|--------|-----------|---------|
| train | default | 7 746 | 49 |
| test | tables23 | 474 | 12 |

The `tables23` split comprises layouts whose scenarios achieve ≥ 80%
historical ignition match (spatial location and calendar day), not agreement
with the USFS Burn Probability raster. See `splits/SELECTION_RULE.md` for full
details and provenance limits.

## Raw Data Access

The raw fire-spread `.npy` arrays, risk maps, and satellite imagery are in
`DroneBench.zip` (2.25 GB).  Layout folders follow the naming convention
`LLLL_SSSSS/` (layout ID + scenario suffix).  Each contains:

- `scenarii/` — fire-spread `.npy` files (T × N × M grids)
- `static_risk_bp2024.npy` — USFS Burn Probability risk map (int16, ÷10000 for [0,1])
- `static_risk_whp.npy` — Wildfire Hazard Potential risk map
- `burn_map.npy` — dynamic (empirical) burn map
- `Fuel_Map/`, `Topography_Map/`, `Vegetation_Map/`, `Weather_Data/` — ancillary data

## Source Data

| Source | License | Usage |
|--------|---------|-------|
| [Sim2Real-Fire](https://github.com/SebastianGrans/Sim2Real-Fire) | Apache-2.0 | Fire-spread simulation engine |
| [USFS Burn Probability (BP)](https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0034-2) | CC-BY-4.0 | Static risk map (`static_risk_bp2024.npy`) |
| [FPA-FOD](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6) | Public Domain (US Gov) | Fire occurrence data |

## Citation

```bibtex
@inproceedings{wfdronebench2026,
  title     = {WFDroneBench: A Benchmark for Drone-Based Proactive Wildfire Detection},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026}
}
```

## Licensing

The **data** in this repository is released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (the most
restrictive upstream data license).  The **code** (benchmark scripts, loading
utilities) is released under the MIT License.  See `NOTICE` for attribution
details.
