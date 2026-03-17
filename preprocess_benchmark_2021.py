#!/usr/bin/env python3
"""
Pre-compute the rescaled .npy files that all budget runs share.
Run this once (serially) before launching the benchmark jobs.

NOTE: intentionally does NOT import from benchmark.py / Strategy.py / wrappers.py
so that this script has zero dependency on Julia and can run with plain Python.
"""

import os
import numpy as np
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "California2021Dataset"
PLACEMENT_MAP = DATASET_DIR / "static_risk_pyrologix.npy"
MASK_PATH = DATASET_DIR / "mask.npy"
SENSOR_POOLING = "mean"

# ── Helpers inlined from code/benchmark.py (no Julia / wrappers import needed) ──

def compute_operational_substeps(data_cell_size_m, drone_speed_m_per_min, coverage_radius_m):
    coverage_width_m = 2 * coverage_radius_m
    coverage_width_cells = coverage_width_m / data_cell_size_m
    coverage_width_cells = max(1, round(coverage_width_cells))
    if coverage_width_cells % 2 == 0:
        coverage_width_cells -= 1
    drone_distance_m = 60 * drone_speed_m_per_min
    drone_distance_operational_cells_per_timestep = (
        drone_distance_m // (coverage_width_cells * data_cell_size_m)
    )
    return max(1, round(drone_distance_operational_cells_per_timestep))


def pool_burnmap_mean(burnmap, kernel_size):
    N, M = burnmap.shape[1:]
    N_new = N // kernel_size
    M_new = M // kernel_size
    burnmap_pooled = np.zeros((burnmap.shape[0], N_new, M_new))
    for i in range(N_new):
        for j in range(M_new):
            burnmap_pooled[:, i, j] = np.mean(
                burnmap[:, i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size],
                axis=(1, 2),
            )
    return burnmap_pooled


def pool_mask(mask, kernel_size, mode="min"):
    pool_fn = np.min if mode == "min" else np.max
    N, M = mask.shape
    N_new = N // kernel_size
    M_new = M // kernel_size
    mask_pooled = np.zeros((N_new, M_new))
    for i in range(N_new):
        for j in range(M_new):
            mask_pooled[i, j] = pool_fn(
                mask[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size]
            )
    return mask_pooled

# ─────────────────────────────────────────────────────────────────────────────

def main():
    cell_size_m = 1000
    speed = 600
    coverage_r_m = 2900

    operational_substeps = compute_operational_substeps(cell_size_m, speed, coverage_r_m)
    coverage_w = round(coverage_r_m * 2 / cell_size_m)
    if coverage_w % 2 == 0:
        coverage_w -= 1

    mask = np.load(str(MASK_PATH))
    H, W = mask.shape
    rescaled_N = H // coverage_w
    rescaled_M = W // coverage_w

    print(
        f"Grid: {H}x{W} -> opt {rescaled_N}x{rescaled_M}, substeps={operational_substeps}",
        flush=True,
    )

    suffix = f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy"

    rescaled_mask_path = Path(str(MASK_PATH).replace(".npy", suffix))
    if rescaled_mask_path.exists():
        print(f"Rescaled mask already exists, skipping: {rescaled_mask_path}", flush=True)
    else:
        print(f"Computing rescaled mask -> {rescaled_mask_path}", flush=True)
        rescaled_mask = pool_mask(mask, coverage_w, mode="max")
        tmp = rescaled_mask_path.with_suffix(".tmp.npy")
        np.save(str(tmp), rescaled_mask)
        tmp.rename(rescaled_mask_path)
        print("Done.", flush=True)

    rescaled_avg_path = Path(str(PLACEMENT_MAP).replace(".npy", f"_{SENSOR_POOLING}{suffix}"))
    if rescaled_avg_path.exists():
        print(f"Rescaled risk map already exists, skipping: {rescaled_avg_path}", flush=True)
        rescaled_placement_1frame = np.load(str(rescaled_avg_path))[:1] * operational_substeps
    else:
        print(f"Computing rescaled risk map -> {rescaled_avg_path}", flush=True)
        avg_map = np.load(str(PLACEMENT_MAP))
        avg_map_masked = avg_map * mask
        rescaled_placement_1frame = pool_burnmap_mean(avg_map_masked, coverage_w)
        rescaled_avg = np.repeat(rescaled_placement_1frame, operational_substeps, axis=0) / operational_substeps
        tmp = rescaled_avg_path.with_suffix(".tmp.npy")
        np.save(str(tmp), rescaled_avg)
        tmp.rename(rescaled_avg_path)
        print("Done.", flush=True)

    rescaled_routing_path = Path(str(PLACEMENT_MAP).replace(".npy", f"_{SENSOR_POOLING}_routing{suffix}"))
    if rescaled_routing_path.exists():
        print(f"Rescaled routing map already exists, skipping: {rescaled_routing_path}", flush=True)
    else:
        print(f"Computing rescaled routing map -> {rescaled_routing_path}", flush=True)
        tmp = rescaled_routing_path.with_suffix(".tmp.npy")
        np.save(str(tmp), rescaled_placement_1frame)
        tmp.rename(rescaled_routing_path)
        print("Done.", flush=True)

    print("\nPreprocessing complete.", flush=True)
    print(f"  Rescaled mask:     {rescaled_mask_path}", flush=True)
    print(f"  Rescaled risk map: {rescaled_avg_path}", flush=True)
    print(f"  Rescaled routing:  {rescaled_routing_path}", flush=True)


if __name__ == "__main__":
    main()
