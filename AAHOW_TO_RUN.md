# How to run

Submit all three scripts from the **project root** (`wildfire_drone_routing/`),
waiting for each job to finish before submitting the next.

## MIT Supercloud

From project root, create the logs directory once (job stdout/stderr go to `./logs/`), then submit in order:

```bash
mkdir -p logs

sbatch report/benchmark_2021_greedy_kernel/supercloud_1_preprocess.sh
# wait for it to finish, then:
sbatch report/benchmark_2021_greedy_kernel/supercloud_2_benchmarks.sh
# wait for it to finish, then:
sbatch report/benchmark_2021_greedy_kernel/supercloud_3_wrapup.sh
```

## Engaging

```bash
sbatch report/benchmark_2021_greedy_kernel/engaging_1_preprocess.sh
sbatch report/benchmark_2021_greedy_kernel/engaging_2_benchmarks.sh
sbatch report/benchmark_2021_greedy_kernel/engaging_3_wrapup.sh
```

- **Step 1** – pre-computes the shared `.npy` rescaled files (~5 min)
- **Step 2** – runs 20M, 100M, 500M placements in parallel (array job) and generates plots (~1–2 h)
- **Step 3** – copies figures into the report folder and builds the PDF

---

## 2021 yearly benchmark (drone routing) — MIT Supercloud

Full benchmark: sensor placement + drone routing over 100 fires. Same 3-step pattern; run from project root after `mkdir -p logs`:

```bash
mkdir -p logs

sbatch report/benchmark_2021_yearly_routing/supercloud_1_preprocess.sh
# wait for it to finish, then:
sbatch report/benchmark_2021_yearly_routing/supercloud_2_benchmarks.sh
# wait for it to finish, then:
sbatch report/benchmark_2021_yearly_routing/supercloud_3_wrapup.sh
```

- **Step 1** – same preprocess as above (mask + risk map + routing map); safe to skip if already run for greedy-kernel.
- **Step 2** – runs `run_benchmark_california2021_yearly.py` for budgets 20M, 100M, 500M in parallel (array job). Writes `benchmark_results_yearly_2021_*.csv` to project root.
- **Step 3** – runs `report/generate_benchmark_report_figures_2021.py`; figures go to `report/` (e.g. `benchmark_fire_map_budget_2021_top.png`).
