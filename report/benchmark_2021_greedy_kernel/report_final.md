# Final Nature routing report — California 2021 greedy-uniform kernel

**Date:** 2026-04-13  
**Dataset:** California 2021  
**Risk map:** Pyrologix ignition probability (static)

This note complements **`benchmark_2021_greedy_kernel.md`** (report date 2026-04-13,
clustered yearly routing) and the archived no-clustering write-up
`benchmark_2021_greedy_kernel_20260327_no_clustering.md`. It records
**final_nature** yearly routing runs: **per-station routing with clustering
disabled** (`--no-clustering`), with routing driver settings fixed for the Nature
submission (see below).

---

## Overview

- **Placement:** **StationMax greedy-uniform kernel** JSONs; **20M** / **100M**
  match the **2026-04-13** benchmark report. **50M** placement
  (`sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean.json`,
  Slurm **4516698**) is summarized below with maps.
- **New results (this note):** **MaxCov** and **LinearMinTime** with CSV suffix
  `_final_nature`, produced by `supercloud_final_nature_routing_array.sh`.
- **TOPGrowing (and clustered MaxCov for context):** summary numbers below are
  **taken from** `benchmark_2021_greedy_kernel.md` (**Apr 12–13, 2026** CSVs;
  **default clustering ON**, reeval **5**, horizon **10**, **120 s** Gurobi cap
  per solve). Those runs are **not** the same experiment as final_nature
  (clustering and routing hyperparameters differ); they are included **as in the
  April 13 report** for side-by-side reference.

**Slurm (successful final_nature array):** job `4498086`, job name
`wf_fnature_route`, logs `logs/wf_fnature_route-4498086_{0..3}.out`. (An earlier
submission `4497958` was cancelled and superseded.)

---

## Final_nature routing protocol

| Parameter | final_nature (MaxCov / LinearMinTime here) |
|-----------|---------------------------------------------|
| Station clustering | **OFF** (`--no-clustering`) |
| Re-evaluation step | **7** operational substeps |
| Optimization horizon | **7** substeps |
| Gurobi time limit (routing driver flag) | **300 s** per solve (`--routing-time-limit 300`) |
| Detection horizon | **6** data hours after first burn (`--detection-horizon-data-steps 6`) |
| Benchmark fires | 100 scenarios, `RANDOM_SEED = 42` |
| Other geometry | Same grid / battery / speed assumptions as `benchmark_2021_greedy_kernel.md` |

LinearMinTime uses `julia/drone_routing_opt_linear.jl`; MaxCov uses the masked
coverage MILP. Strategy names in CSVs end with `_final_nature`.

---

## Common setup (aligned with `benchmark_2021_greedy_kernel.md`)

| Parameter | Value |
|-----------|-------|
| Data grid | `1309 × 805` (~1 km/cell) |
| Operational grid | `261 × 161` (5 km/cell) |
| Drone speed | `600 m/min` |
| Coverage radius | `2900 m` |
| Battery | `1 h = 7` operational substeps per data hour |
| One-way reach used in placement pruning | `3` operational cells |
| Benchmark subset | 100 random fires (seed = 42) |

---

## Fire locations

Same figure bundle as the April 13 report (`figures/` under this folder).

![Benchmark fire sample (2021)](figures/benchmark_fire_locations_budget_2021.png)

---

## Summary table — placement (April 2026 refresh)

Counts match **`benchmark_2021_greedy_kernel.md`** (sensor logs on disk after
the April refresh). Drones = 7 per station.

| Budget | Ground sensors | Stations | Drones | Clusters (L-infinity merge) |
|--------|----------------|----------|--------|----------------------------|
| 20M | 0 | 40 | 280 | 6 |
| 50M | 0 | 100 | 700 | 3 |
| 100M | 0 | 200 | 1,400 | 2 |

Counts for **50M** are from `sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean.json` (placement job **4516698**, 2026-04-16). Drones = 7 per station.

---

## Summary table — routing (20M)

**Among reachable** = `cluster != none` in the yearly CSV (same column
convention as the April 13 report). Detection = `device != undetected` and
`delta_t >= 0`.

| Strategy | Mode | Source | Overall detection | Among reachable | Mean delta_t (detected) | 95% CI (mean)\* | Median | Max |
|----------|------|--------|-------------------|-----------------|-------------------------|-----------------|--------|-----|
| MaxCov `_final_nature` | no clustering | `benchmark_results_yearly_20260413_162445.csv` | **26%** (26/100) | **26/26 (100%)** | 0.500 | [0.238, 0.762] | 0.000 | 2.000 |
| LinearMinTime `_final_nature` | no clustering | `benchmark_results_yearly_20260413_175828.csv` | **22%** (22/100) | **22/26 (84.6%)** | 0.545 | [-0.062, 1.153] | 0.000 | 5.000 |
| MaxCov | **clustered** | Apr 12–13 bundle; see `benchmark_2021_greedy_kernel.md` | 25% (25/100) | 25/26 (96.2%) | 0.44 | — | 0 | 3 |
| TOPGrowing | **clustered** | same | 26% (26/100) | **26/26 (100%)** | 0.27 | — | 0 | 2 |

\*Student-t 95% CI on the **detected** subset only, for final_nature rows. The
April 13 main table does not list CIs for clustered MaxCov/TOP.

**Reachable fires missed (20M, final_nature only):** MaxCov misses **none** of
the 26 reachable. LinearMinTime misses **4** reachable fires:
BUCKSKIN_2021-CAENF-020954, Ingalls_2021-CAPNF-001370,
Maddalena_2021-CAPNF-000902, SAPP_2021-CASTF-001710.

### Paired comparison, MaxCov vs LinearMinTime `_final_nature` (both detected, 20M)

| Quantity | Value |
|----------|-------|
| Pairs | 22 |
| Mean (MaxCov - LinearMinTime) | 0.000 data steps |
| Head-to-head (lower delta_t wins) | Linear faster 8, tied 11, MaxCov faster 3 |

### From April 13 report — paired MaxCov vs TOPGrowing (clustered, both detected)

| Budget | Pairs | Mean (MaxCov - TOP) | Head-to-head (TOP / tie / MaxCov) |
|--------|-------|---------------------|-----------------------------------|
| 20M | 25 | **+0.20** data steps | 8 / 13 / 4 |

Approximate paired *t* (April note): 20M *t* ~ 1.04 (*n* = 25). See
`benchmark_2021_greedy_kernel.md` for methodology.

**Comparability:** final_nature rows are **no-clustering**; MaxCov and
TOPGrowing in the April table use **clustering**. Do not merge those columns
into a single “which is best” statement without stating the routing mode.

### 20M — placement figures (same as April 13 report)

![20M data scale](figures/california_2021_sensor_clusters_greedy_uniform_20M.png)

![20M operational scale](figures/california_2021_sensor_clusters_opt_greedy_uniform_20M.png)

---

## Summary table — routing (100M)

| Strategy | Mode | Source | Rows | Overall detection | Among reachable | Mean delta_t (detected) | 95% CI (mean)\* | Median | Max |
|----------|------|--------|------|-------------------|-----------------|-------------------------|-----------------|--------|-----|
| MaxCov `_final_nature` | no clustering | `benchmark_results_yearly_20260413_170201.csv` | 100 | **90%** (90/100) | **90/96 (93.8%)** | 0.422 | [0.302, 0.542] | 0.000 | 2.000 |
| LinearMinTime `_final_nature` | no clustering | `benchmark_results_yearly_20260413_203043.csv` | 100 | **84%** (84/100) | **84/96 (87.5%)** | 0.500 | [0.283, 0.717] | 0.000 | 5.000 |
| MaxCov | **clustered** | Apr 12–13 bundle | 100 | 91% (91/100) | 91/96 (94.8%) | 0.57 | — | 0 | 4 |
| TOPGrowing | **clustered** | `benchmark_results_yearly_20260412_085132.csv` (**98** rows) | **98** | 95.9% (94/98) | **94/94 (100%)** | 0.28 | — | 0 | 2 |

\*Student-t 95% CI on the detected subset (final_nature only).

**April 13 note on 100M TOPGrowing:** the CSV above has **98** scenarios;
**Grizzly_2021-CAPNF-000697** and **LONG_2021-CAKNF-006470** appear in the 100M
MaxCov slice but are **missing** from this TOPGrowing file (run error or
incomplete job). Treat TOPGrowing aggregates as **lower confidence** until
those rows exist.

**Reachable fires missed (100M, final_nature only):**  
**MaxCov** misses 6/96: DEXTER_2021-CAINF-001695, FAWN_2021-CASHU-010480,
GOBBI_2021-CAENF-023392, JULY_2021-CAKNF-003631, RICE_2021-CAMDF-001543,
VASQUEZ_2021-CAANF-002923.  
**LinearMinTime** misses 12/96: BEAR_2021-CAENF-022077, BOX_2021-CAMDF-001391,
CAMPUS_2021-CABDF-012440, CARSON_2021-CAENF-013467, DALE_2021-CASHF-000975,
FLAT_2021-CASHF-000907, GRAND_VIEW_2021-CABDF-012129,
LITTLE_MARBLE_2021-CAKNF-007809, Maddalena_2021-CAPNF-000902,
RAYS_2021-CAENF-013393, SAW_2021-CAINF-001447, SPRUCE_2021-CABDF-012448.

### Paired comparison, MaxCov vs LinearMinTime `_final_nature` (both detected, 100M)

| Quantity | Value |
|----------|-------|
| Pairs | 78 |
| Mean (MaxCov - LinearMinTime) | -0.103 data steps |
| Approx. paired *t* | -0.84 (not significant at alpha = 0.05 without exact *p*) |
| Head-to-head | Linear faster 17, tied 45, MaxCov faster 16 |

### From April 13 report — paired MaxCov vs TOPGrowing (clustered, both detected)

| Budget | Pairs | Mean (MaxCov - TOP) | Head-to-head (TOP / tie / MaxCov) |
|--------|-------|---------------------|-----------------------------------|
| 100M | 89 | **+0.28** data steps | 28 / 50 / 11 |

Approximate paired *t* (April note): 100M *t* ~ 2.93 (*n* = 89).

### 100M — placement figures (same as April 13 report)

![100M data scale](figures/california_2021_sensor_clusters_greedy_uniform_100M.png)

![100M operational scale](figures/california_2021_sensor_clusters_opt_greedy_uniform_100M.png)

---

## 50M greedy-uniform placement

StationMax greedy-uniform kernel at **$50M** (same cost model as other budgets:
sensor / station / drone pricing in the placement driver). Figures regenerated
from the cached JSON with:

`python3 visualize_sensor_placement_2021.py California2021Dataset/logs/sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean.json --scale both --tag _greedy_uniform_50M`

![50M greedy-uniform kernel — data scale](figures/california_2021_sensor_clusters_greedy_uniform_50M.png)

![50M greedy-uniform kernel — operational scale](figures/california_2021_sensor_clusters_opt_greedy_uniform_50M.png)

---

## Data products

**final_nature** (project root):

| File | Strategy |
|------|----------|
| `benchmark_results_yearly_20260413_162445.csv` | GaussianBudget20M_MaxCov_final_nature |
| `benchmark_results_yearly_20260413_170201.csv` | GaussianBudget100M_MaxCov_final_nature |
| `benchmark_results_yearly_20260413_175828.csv` | GaussianBudget20M_LinearMinTime_final_nature |
| `benchmark_results_yearly_20260413_203043.csv` | GaussianBudget100M_LinearMinTime_final_nature |

**TOPGrowing / clustered routing (April 13 report):** merged bundle and slice
list in **`benchmark_2021_greedy_kernel.md`** (e.g.
`benchmark_results_yearly_greedy_uniform_20260411_13_merged.csv`, source tag
`085132` for 100M TOPGrowing).

---

## Reproducibility

```bash
sbatch report/benchmark_2021_greedy_kernel/supercloud_final_nature_routing_array.sh
```

Clustered MaxCov + TOPGrowing (April report):

```bash
sbatch report/benchmark_2021_greedy_kernel/supercloud_3_greedy_uniform_routing_array.sh
```

---

## Conclusions (this note)

1. **final_nature** documents a single **no-clustering** protocol (reeval 7,
   horizon 7, 300 s cap, 6 h detection horizon) for **MaxCov** and
   **LinearMinTime**.

2. **TOPGrowing** (and clustered **MaxCov**) summary statistics in this file are
   **from `benchmark_2021_greedy_kernel.md` (2026-04-13)** — **clustered**
   cooperative routing — not reruns under final_nature.

3. At **20M**, final_nature **MaxCov** matches the **26** placement-reachable
   fires in the April placement table; **LinearMinTime** misses four of them.

4. At **100M**, final_nature **MaxCov** and **LinearMinTime** are both below the
   clustered April **TOPGrowing** headline detection rate on the available
   **98** rows; interpret **TOPGrowing** with the April caveat (two missing
   scenarios).

5. The April 13 report explains why **clustered** vs **no-clustering** numbers
   are **not directly comparable**; this note keeps both on one page with an
   explicit **Mode** column.
