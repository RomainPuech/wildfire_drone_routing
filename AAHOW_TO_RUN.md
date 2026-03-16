# How to run

From the **project root** (`wildfire_drone_routing/`):

```bash
bash report/benchmark_2021_greedy_kernel/submit_greedy_benchmark.sh
```

This submits a 3-job chain to Slurm:
1. **preprocess** – computes shared `.npy` files once (≤ 15 min)
2. **array[0-2]** – runs 20M / 100M / 500M placements in parallel (≤ 1 h each), starts after step 1
3. **wrapup** – copies figures and builds the PDF, starts after the array finishes (even on partial failure)

Logs are written to `logs/` in the project root.
