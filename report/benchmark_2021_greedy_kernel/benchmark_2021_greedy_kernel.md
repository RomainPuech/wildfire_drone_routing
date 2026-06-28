# Benchmark Report -- California 2021 Greedy-Uniform Kernel (clustered routing)

**Report date:** 2026-04-13  
**Dataset:** `California2021Dataset` (refreshed build; Pyrologix static risk, operational grid `261×161`, mean-pooled placement map)  
**Archive note:** The 2026-03-27 write-up with **per-station / no-clustering** routing is kept as  
`benchmark_2021_greedy_kernel_20260327_no_clustering.md` for comparison.

---

## Overview

This report summarizes **yearly** placement and routing benchmarks using the **StationMax greedy-uniform kernel** sensor placement (`StationMaxGreedyUniform`) and three routing strategies:

- **MaxCov** -- masked growing coverage (logged)  
- **TOPGrowing** -- time-over-period growing (logged)  
- **LinearMinTime** -- Julia MILP linear routing (`drone_routing_opt_linear.jl`), **120 s** Gurobi cap per solve  

Routing uses **default clustering**: charging stations are merged into connected components (Chebyshev / L-infinity distance within rescaled battery reach), and each fire is routed against the **cluster** it belongs to. This is the mode produced by `run_benchmark_california2021_yearly.py` **with clustering enabled** (Slurm scripts `supercloud_3_greedy_uniform_routing_array.sh` / `supercloud_3_greedy_uniform_routing_linear_array.sh`).

**Budgets:** 20M, 100M, 500M (Gaussian budget curve; 7 drones per station).

---

## Heuristic kernel (unchanged)

Greedy-uniform kernel construction matches the earlier report: per-station `7×7` reach patch, risk-weighted greedy paths up to `Kmax = 7`, cumulative risk fraction applied **uniformly** over all reachable cells, **max-over-stations** aggregation. See the archived March document for full step-by-step prose.

---

## Common experimental setup

| Parameter | Value |
|-----------|--------|
| Benchmark fires | 100 scenarios, `RANDOM_SEED = 42` |
| Data grid | `1309 × 805` (~1 km) |
| Operational grid | `261 × 161` (5 km cells) |
| Drone speed | 600 m/min |
| Coverage radius | 2900 m |
| Battery | 1 h → **7** operational substeps per data hour |
| Routing horizon | `MAX_ROUTING_DATA_STEPS = 24` data hours |
| Re-evaluation | every **5** operational substeps |
| Optimization horizon | **10** substeps |
| **Gurobi time limit (routing)** | **120 s** per optimization (MaxCov, TOPGrowing, LinearMinTime) |
| Placement Gurobi (Slurm) | 600 s (20M); 43200 s (100M / 500M) per `supercloud_2_greedy_uniform_placement_array.sh` |
| Yearly burn map | Static Pyrologix slice (`log_key = pyrologix`) |

---

## Placement summary (sensor logs on disk)

Counts are read from `California2021Dataset/logs/sensor_alloc_GaussianBudget{20,100,500}M_StationMaxGreedyUniform_261x161_mean.json` after the April 2026 refresh.

| Budget | Ground sensors | Charging stations | Clusters (L-infinity merge) | Notes |
|--------|----------------|-------------------|-----------------------------|--------|
| **20M** | 0 | 40 | 6 | Valid placement |
| **100M** | 0 | 200 | 2 | Valid placement |
| **500M** | 0 | **0** | **0** | **Placement did not produce stations** under the current solve (time limit / incumbent). Routing rows are therefore all **no cluster** / undetected -- **not comparable** to 20M/100M until placement is fixed (e.g. warm-start scripts in this folder). |

Cluster counts match `visualize_sensor_placement_2021.py` on the same JSON files (2026-04-13 regenerate).

### Fire detectability from placement alone (benchmark subset, n = 100)

These counts refer to the **same 100 scenarios** as the yearly routing benchmark (`RANDOM_SEED = 42`). They describe **geometry and passive sensors only** -- no drone routing simulation.

**Definitions (aligned with `visualize_sensor_placement_2021.py`):**

- **Ground-detectable:** ignition lies on an operational cell that contains a **ground sensor** (same pooled grid as placement).
- **Drone-reachable:** not ground-covered, but the fire's operational cell is within **one-way Chebyshev distance** `floor(max_battery_substeps / 2) = 3` of **at least one charging station** in the placement (same rule as `visualize_sensor_placement_2021.py`).
- **Not discoverable by placement:** neither of the above (no passive sensor and no station within one-way reach).

| Budget | Ground-detectable | Drone-reachable | Not discoverable | **Total placement-detectable** (ground OR drone-reachable) |
|--------|-------------------|-----------------|------------------|------------------------------------------------------------|
| **20M** | 0 | 26 | 74 | **26 / 100 (26%)** |
| **100M** | 0 | 96 | 4 | **96 / 100 (96%)** |
| **500M** | 0 | 0 | 100 | **0 / 100** (no stations in cached JSON) |

At 20M and 100M there are **no ground sensors** in this greedy-uniform run, so all placement-detectable fires are **drone-reachable** only. The **26** and **96** figures match the routing table column *Among reachable* denominators (fires with `cluster != none` in the yearly driver), which use the same reachability notion after cluster assignment.

---

## Routing results -- summary (Apr 12-13, 2026 CSVs)

Rows are **per scenario** (one row per strategy x fire). Detection = `device != undetected` and `delta_t >= 0`.  
**Among reachable** = fires with `cluster != none` (drone cluster assigned).

| Budget | Strategy | Rows | Overall detection | Among reachable | Mean delta_t (detected) | Median delta_t | Max delta_t |
|--------|----------|------|-------------------|-----------------|---------------------|-----------|--------|
| 20M | MaxCov | 100 | 25% (25/100) | 25/26 (96.2%) | 0.44 | 0 | 3 |
| 20M | TOPGrowing | 100 | 26% (26/100) | **26/26 (100%)** | 0.27 | 0 | 2 |
| 20M | LinearMinTime | 100 | 24% (24/100) | 24/26 (92.3%) | 0.25 | 0 | 2 |
| 100M | MaxCov | 100 | 91% (91/100) | 91/96 (94.8%) | 0.57 | 0 | 4 |
| 100M | TOPGrowing | **98** | 95.9% (94/98) | **94/94 (100%)** | 0.28 | 0 | 2 |
| 100M | LinearMinTime | - | *Not in this bundle* | - | - | - | - |
| 500M | MaxCov / TOP / Linear | 100 each | 0% | 0/0 | - | - | - |

**100M TOPGrowing:** the CSV `benchmark_results_yearly_20260412_085132.csv` contains **98** rows; **Grizzly_2021-CAPNF-000697** and **LONG_2021-CAKNF-006470** are present in the 100M MaxCov file but missing here (run error or incomplete job).

**100M LinearMinTime:** a long Slurm array task was still in progress during this export; **no finished `benchmark_results_yearly_*` CSV** from that run is included. Re-merge the bundle after the job writes `Results saved to:`.

### Paired comparison (MaxCov minus TOPGrowing), fires **detected by both**

| Budget | Pairs (both detected) | Mean delta delta_t (MaxCov - TOP) | Comment |
|--------|------------------------|-------------------------|---------|
| 20M | 25 | **+0.20** data steps | TOP faster on average when both detect |
| 100M | 89 | **+0.28** data steps | TOP faster on average among comparable detects |

Approximate paired *t*-statistics (same pairs): 20M *t* ~ 1.04 (*n* = 25); 100M *t* ~ 2.93 (*n* = 89). For a rigorous *p*-value, re-run with `scipy.stats` or similar in your environment.

### Head-to-head (only fires where **both** strategies detected the fire)

| Budget | TOP faster | Tied | MaxCov faster | Scenarios compared |
|--------|------------|------|---------------|--------------------|
| 20M | 8 | 13 | 4 | 25 |
| 100M | 28 | 50 | 11 | 89 |

### Gurobi optimality gaps (routing MILPs)

Julia prints one line per Gurobi termination inside the yearly routing loop:

- **MaxCov** (`julia/drone_routing_opt.jl`): `Gurobi optimality gap: <pct>% (status: ...)`
- **LinearMinTime** (`julia/drone_routing_opt_linear.jl`): same pattern via `_log_gurobi_gap_linear!`

The table below aggregates **every** such line in the Slurm stdout files from the Apr 2026 greedy-uniform routing jobs (120 s time limit per solve). It is **not** a per-scenario summary: one cluster routing run emits many solves (initial + re-optimizations every 5 operational substeps).

| Strategy | Budget | Log file (under `logs/`) | Solves *n* | Mean gap % | Median gap % | Min % | Max % | OPTIMAL | TIME_LIMIT |
|----------|--------|---------------------------|------------|------------|--------------|-------|-------|---------|------------|
| MaxCov | 20M | `wf_greedy_route-4486528_0.out` | 756 | 8.90 | 0.0 | 0.0 | 96.8 | 400 | 356 |
| MaxCov | 100M | `wf_greedy_route-4486528_2.out` | 2310 | 10.53 | 0.0 | 0.0 | 144.8 | 1212 | 1098 |
| MaxCov | 500M | `wf_greedy_route-4486528_4.out` | 0 | -- | -- | -- | -- | -- | -- |
| LinearMinTime | 20M | `wf_greedy_route_lin-4496394_0.out` | 613 | 17.60 | 17.59 | 4.71 | 40.3 | 0 | 613 |
| LinearMinTime | 100M | `wf_greedy_route_lin-4496394_1.out` | 1847 | 21.61 | 20.06 | 4.25 | 51.9 | 0 | 1847 |
| LinearMinTime | 500M | `wf_greedy_route_lin-4496394_2.out` | 0 | -- | -- | -- | -- | -- | -- |

**TOPGrowing:** the current TOP / PSO routing stack does **not** write `Gurobi optimality gap:` lines to these logs, so **mean/median MIP gaps are not defined** in the same sense. (Placement still uses Gurobi; see placement scripts for sensor MIP gaps.)

**Interpretation:** MaxCov medians are **0%** because a large share of subproblems close to **OPTIMAL** under the cap; LinearMinTime solves in this run overwhelmingly hit **TIME_LIMIT**, so gaps reflect suboptimal incumbent vs. bound at 120 s.

Recompute after new runs (edit the log filenames inside the script if job IDs change):

```bash
python3 report/benchmark_2021_greedy_kernel/parse_routing_mip_gaps.py
```

---

## Data products bundled with this report

| File | Description |
|------|-------------|
| `benchmark_results_yearly_greedy_uniform_20260411_13_merged.csv` | All session rows (798) with `source_csv` / `source_run_tag` provenance |
| Source slices (project root) | See `source_csv` column: `20260412_084333`, `081018`, `094256`, `085132`, `075513`, `075505`, `20260413_134640`, `121953` |

---

## Figures

All PNGs under `figures/` were **regenerated from the current workspace** `California2021Dataset` (not copied from an older bundle):

- **Fire sample:** `python3 report/benchmark_2021_greedy_kernel/make_benchmark_fire_locations_figure.py` (uses `code/displays.py::plot_fire_locations`, same 100 fires as the benchmark driver, seed 42).
- **Cluster / placement maps:** for each budget,  
  `python3 visualize_sensor_placement_2021.py California2021Dataset/logs/sensor_alloc_GaussianBudget{20,100,500}M_StationMaxGreedyUniform_261x161_mean.json --scale both --tag _greedy_uniform_{20M,100M,500M}`  
  then copy into `figures/` (six files). The 500M JSON has zero stations; maps show Pyrologix + fires only.

![Benchmark fire sample (2021)](figures/benchmark_fire_locations_budget_2021.png)

### 20M -- greedy-uniform placement

![20M data scale](figures/california_2021_sensor_clusters_greedy_uniform_20M.png)

![20M operational scale](figures/california_2021_sensor_clusters_opt_greedy_uniform_20M.png)

### 100M -- greedy-uniform placement

![100M data scale](figures/california_2021_sensor_clusters_greedy_uniform_100M.png)

![100M operational scale](figures/california_2021_sensor_clusters_opt_greedy_uniform_100M.png)

### 500M -- greedy-uniform placement (maps from solver output; routing invalid until placement fixed)

![500M data scale](figures/california_2021_sensor_clusters_greedy_uniform_500M.png)

![500M operational scale](figures/california_2021_sensor_clusters_opt_greedy_uniform_500M.png)

---

## Conclusions (this clustered refresh)

1. **Clustered cooperative routing** materially changes detection vs the old **no-clustering** report: at **100M**, overall detection is **~91-96%** in this session, vs the ~33-38% range in the archived per-station mode -- the experiment is **not** directly comparable without relabeling the routing mode.

2. **20M** remains placement-limited: only **~26** fires fall in a non-`none` cluster; among them, **TOPGrowing** attains **100%** detection in this CSV set vs **96%** MaxCov and **92%** Linear.

3. **100M TOPGrowing** CSV is **incomplete** (98/100 scenarios); treat TOPGrowing summary statistics as **lower confidence** until the two missing fires are re-run.

4. **500M** placement currently yields **zero** stations in the cached JSON; **ignore 500M routing numbers** until a valid `sensor_alloc_...500M...json` exists.

5. **LinearMinTime (100M)** should be appended to the merged CSV and this report once the corresponding benchmark finishes.

---

## Reproducibility

From repository root (after dataset + Julia env per cluster docs):

```bash
# Placement (array over budgets)
sbatch report/benchmark_2021_greedy_kernel/supercloud_2_greedy_uniform_placement_array.sh

# Routing -- MaxCov + TOPGrowing
sbatch report/benchmark_2021_greedy_kernel/supercloud_3_greedy_uniform_routing_array.sh

# Routing -- LinearMinTime only
sbatch report/benchmark_2021_greedy_kernel/supercloud_3_greedy_uniform_routing_linear_array.sh
```

Clear caches before a clean refresh:

```bash
bash report/benchmark_2021_greedy_kernel/supercloud_clear_greedy_uniform_cache.sh
```

Recompute placement-detectability counts for the benchmark 100 fires (table in this report):

```bash
python3 report/benchmark_2021_greedy_kernel/print_placement_detectability.py
python3 report/benchmark_2021_greedy_kernel/parse_routing_mip_gaps.py
```

Regenerate report figures (after placement JSONs exist):

```bash
python3 report/benchmark_2021_greedy_kernel/make_benchmark_fire_locations_figure.py
for b in 20 100 500; do
  python3 visualize_sensor_placement_2021.py \
    California2021Dataset/logs/sensor_alloc_GaussianBudget${b}M_StationMaxGreedyUniform_261x161_mean.json \
    --scale both --tag _greedy_uniform_${b}M
done
cp report/california_2021_sensor_clusters*_greedy_uniform_*.png report/benchmark_2021_greedy_kernel/figures/
pandoc report/benchmark_2021_greedy_kernel/benchmark_2021_greedy_kernel.md \
  -o report/benchmark_2021_greedy_kernel/benchmark_2021_greedy_kernel.pdf --pdf-engine=pdflatex -V geometry:margin=1in
```

**Routing outcome maps (optional):** `code/displays.py::plot_benchmark_overview` can draw detected vs missed vs out-of-range fires on the Pyrologix background given a results CSV; there is no separate routing figure in this PDF beyond the cluster maps above. Extend `report/generate_benchmark_report_figures_2021.py` or a sibling script if you want one PNG per strategy combo.
