# How to run

Run from the **project root** (`wildfire_drone_routing/`).

## Engaging cluster

```bash
bash report/benchmark_2021_greedy_kernel/submit_greedy_benchmark.sh
```

## MIT Supercloud

```bash
bash report/benchmark_2021_greedy_kernel/submit_greedy_benchmark_supercloud.sh
```

---

Both submit scripts run the same 3-job chain:

1. **preprocess** – computes shared `.npy` files once (≤ 15 min, serial)
2. **array[0-2]** – runs 20M / 100M / 500M placements in parallel (≤ 1 h each), starts after step 1
3. **wrapup** – copies figures and builds the PDF, starts after the array finishes (even on partial failure)

Logs are written to `logs/` in the project root.
