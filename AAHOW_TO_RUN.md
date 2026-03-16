# How to run

Submit all three scripts from the **project root** (`wildfire_drone_routing/`),
waiting for each job to finish before submitting the next.

## MIT Supercloud

```bash
sbatch report/benchmark_2021_greedy_kernel/supercloud_1_preprocess.sh
sbatch report/benchmark_2021_greedy_kernel/supercloud_2_benchmarks.sh
sbatch report/benchmark_2021_greedy_kernel/supercloud_3_wrapup.sh
```

## Engaging

```bash
sbatch report/benchmark_2021_greedy_kernel/engaging_1_preprocess.sh
sbatch report/benchmark_2021_greedy_kernel/engaging_2_benchmarks.sh
sbatch report/benchmark_2021_greedy_kernel/engaging_3_wrapup.sh
```

- **Step 1** – pre-computes the shared `.npy` rescaled files (~5 min)
- **Step 2** – runs 20M, 100M, 500M placements sequentially and generates plots (~1–2 h)
- **Step 3** – copies figures into the report folder and builds the PDF
