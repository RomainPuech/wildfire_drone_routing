# Benchmark Report — 500M Budget-Optimized, California 2020

**Date:** *(fill when sensor placement completes)*  
**Script:** `run_benchmark_california2020_yearly.py --sensor-only --budget 500`  
**Results file:** *(N/A — sensor placement only; full benchmark TBD)*

---

## Setup

### Budget allocation (optimizer-chosen)

The optimizer chose the following allocation under a **500M** budget (costs: sensor 100k, station 150k, drone 50k):

| Component | Count | Unit cost | Subtotal |
|-----------|------:|----------:|--------:|
| Ground sensors | *TBD* | 100k | *TBD* |
| Charging stations | *TBD* | 150k | *TBD* |
| Drones | *TBD* | 50k | *TBD* |
| **Total** | | | **500M** |

*Fill from the run log and `California2020Dataset/logs/sensor_alloc_GaussianBudget500M_261x161_mean.json` after the placement completes.*

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

- **Budget:** 500M (costs in millions: sensor 0.1, station 0.15, drone 0.05)
- **MIP gap at termination:** *TBD* (see run log; Julia prints "MIP gap: …%")
- **Time limit:** 600 s (10 min)
- Pre-filtering: top 20% of feasible cells by risk/coverage potential
- Placement saved to: `California2020Dataset/logs/sensor_alloc_GaussianBudget500M_261x161_mean.json`

### Drone routing strategy

*(Full benchmark with routing TBD.)*

---

## Cluster structure

*(Fill after placement: number of charging stations and clusters from the 500M allocation.)*

A fire scenario is **discoverable** if its ignition point (opt-space) is within L∞ ≤ 3 of at least one charging station.

---

## Benchmark fire locations

Same as 20M/100M: 100 fires (seed=42). See [benchmark_fire_locations_budget.png](benchmark_fire_locations_budget.png).

---

## Sensor placement map (500M)

![500M sensor placement](benchmark_fire_map_budget_500M.png)

*Charging stations and ground sensors chosen by the 500M budget optimizer. Cluster zones (L∞ ≤ 3 one-way reach). Fire markers: same 100 ignition points.*

---

## Detection results

*(N/A until a full benchmark with 500M placement is run.)*

---

## How to run sensor placement only (500M)

```bash
conda activate wf
cd /path/to/wildfire_drone_routing
python-jl run_benchmark_california2020_yearly.py --sensor-only --budget 500
```

Optional: `--time-limit 600` (default 10 min).  
After the run, regenerate the figure and fill this report from the log:

```bash
python report/generate_benchmark_report_figures.py
```
