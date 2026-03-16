#!/usr/bin/env python3
"""
Pre-compute the rescaled .npy files that all budget runs share.
Run this once (serially) before launching the parallel array job so that
concurrent workers don't race to write the same files.
"""

import sys
import os
import numpy as np
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "code"))

from benchmark import compute_operational_substeps, pool_burnmap_mean, pool_mask

DATASET_DIR = PROJECT_ROOT / "California2021Dataset"
PLACEMENT_MAP = DATASET_DIR / "static_risk_pyrologix.npy"
MASK_PATH = DATASET_DIR / "mask.npy"
SENSOR_POOLING = "mean"


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
        f"Grid: {H}x{W} -> opt {rescaled_N}x{rescaled_M}, "
        f"substeps={operational_substeps}",
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
    else:
        print(f"Computing rescaled risk map -> {rescaled_avg_path}", flush=True)
        avg_map = np.load(str(PLACEMENT_MAP))
        avg_map_masked = avg_map * mask
        rescaled_avg = pool_burnmap_mean(avg_map_masked, coverage_w)
        rescaled_avg = np.repeat(rescaled_avg, operational_substeps, axis=0) / operational_substeps
        tmp = rescaled_avg_path.with_suffix(".tmp.npy")
        np.save(str(tmp), rescaled_avg)
        tmp.rename(rescaled_avg_path)
        print("Done.", flush=True)

    print("\nPreprocessing complete.", flush=True)
    print(f"  Rescaled mask:     {rescaled_mask_path}", flush=True)
    print(f"  Rescaled risk map: {rescaled_avg_path}", flush=True)


if __name__ == "__main__":
    main()
