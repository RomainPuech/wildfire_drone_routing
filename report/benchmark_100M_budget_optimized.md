# Benchmark Report — 100M Budget-Optimized, California 2020

**Date:** 2026-03-04 (sensor placement run)  
**Script:** `run_benchmark_california2020_yearly.py --sensor-only --budget 100`  
**Results file:** *(N/A — sensor placement only; full benchmark TBD)*

---

## Setup

### Budget allocation (optimizer-chosen)

The optimizer chose the following allocation under a **100M** budget (costs: sensor 100k, station 150k, drone 50k):

| Component | Count | Unit cost | Subtotal |
|-----------|------:|----------:|--------:|
| Ground sensors | **188** | 100k | 18.8M |
| Charging stations | **86** | 150k | 12.9M |
| Drones | **1366** | 50k | 68.3M |
| **Total** | | | **100M** |

### Simulation parameters

| Parameter | Value |
|-----------|-------|
| Grid | California 2020 WFPI, 1309×805 (~1 km/cell) |
| Operational grid | 261×161 (5×5 coverage blocks) |
| Drone speed | 600 m/min |
| Coverage radius | 2900 m (~2.9 km) |
| Battery (max) | 1 h = 7 operational substeps |
| Transmission range | 50 km |
| Scenario duration | **6 data steps** (3 h) |
| Scenarios benchmarked | 100 random fires (seed=42) from 1530 valid |

### Sensor placement strategy

**Strategy:** `SensorPlacementMaxCoverageGaussianTimeMaskedBudget`  
ILP (Gurobi) maximizes risk-weighted coverage subject to a single budget constraint; device counts and placement are decision variables.

- **Budget:** 100M (costs in millions: sensor 0.1, station 0.15, drone 0.05)
- **MIP gap at termination:** **0.01%** (termination: OPTIMAL)
- **Time limit:** 600 s (10 min); solve: 583 s; preprocessing: 7.6 s; model creation: 14.2 s
- Pre-filtering: top 20% of feasible cells by risk/coverage potential
- Placement saved to: `California2020Dataset/logs/sensor_alloc_GaussianBudget100M_261x161_mean.json`

### Drone routing strategy

*(Full benchmark with routing TBD; same options as 20M report if/when run.)*

---

## Cluster structure

The 86 charging stations form clusters under L∞ distance ≤ 7 (max battery substeps). See overview map for spatial distribution.

A fire scenario is **discoverable** if its ignition point (opt-space) is within L∞ ≤ 3 of at least one charging station (`floor(max_battery / 2) = 3` — one-way drone reach on a single charge).

---

## Benchmark fire locations

Same as 20M report: 100 fires (seed=42). See [benchmark_fire_locations_budget.png](benchmark_fire_locations_budget.png).

---

## Sensor placement map (100M)

![100M sensor placement](benchmark_fire_map_budget_100M.png)

*Charging stations and ground sensors chosen by the 100M budget optimizer. Cluster zones (L∞ ≤ 3 one-way reach). Fire markers: same 100 ignition points (discoverable vs non-discoverable by this placement).*

---

## Detection results

*(N/A until a full benchmark with 100M placement and a routing strategy is run.)*

---

## How to run sensor placement only (100M)

```bash
conda activate wf
cd /path/to/wildfire_drone_routing
python-jl run_benchmark_california2020_yearly.py --sensor-only --budget 100
```

Optional: `--time-limit 600` (default 10 min). Output: `California2020Dataset/logs/sensor_alloc_GaussianBudget100M_261x161_mean.json`.  
To regenerate the 100M placement figure after the run:

```bash
python report/generate_benchmark_report_figures.py
```

(Generates `report/benchmark_fire_map_budget_100M.png` when the 100M sensor cache exists.)
