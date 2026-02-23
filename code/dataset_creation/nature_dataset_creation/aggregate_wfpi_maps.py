#!/usr/bin/env python3
"""
Aggregate WFPI Maps for Sensor Placement

Creates a yearly-averaged WFPI map by loading all daily WFPI forecast maps
and computing their element-wise mean. The result is a static risk map
suitable for sensor placement strategies.

Processes both Day-1 and Day-2 WFPI forecast datasets.

Run from the project root:
    python code/dataset_creation/nature_dataset_creation/aggregate_wfpi_maps.py
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))

# Datasets to process: (dataset_dir, wfpi_glob_pattern, output_filename)
DATASETS = [
    {
        "name": "California2020Dataset (Day-2 forecasts)",
        "dir": os.path.join(PROJECT_ROOT, "California2020Dataset"),
        "glob": "wfpi_*.npy",
        "output": "static_risk_wfpi_avg.npy",
    },
    {
        "name": "California2020Dataset_Day1 (Day-1 forecasts)",
        "dir": os.path.join(PROJECT_ROOT, "California2020Dataset_Day1"),
        "glob": "wfpi_day1_*.npy",
        "output": "static_risk_wfpi_avg.npy",
    },
]


def aggregate_wfpi_maps(dataset_dir, glob_pattern, output_filename):
    """
    Load all WFPI maps matching the glob pattern in dataset_dir,
    compute their element-wise average, and save the result.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory containing wfpi_*.npy files.
    glob_pattern : str
        Glob pattern to match WFPI files (e.g., "wfpi_*.npy").
    output_filename : str
        Name of the output file to save in dataset_dir.

    Returns
    -------
    np.ndarray or None
        The averaged map, or None if no files were found.
    """
    dataset_path = Path(dataset_dir)
    wfpi_files = sorted(dataset_path.glob(glob_pattern))

    if not wfpi_files:
        print(f"  No files matching '{glob_pattern}' in {dataset_dir}")
        return None

    print(f"  Found {len(wfpi_files)} WFPI files")

    # Load first file to get shape
    first = np.load(str(wfpi_files[0]))
    print(f"  Map shape: {first.shape}, dtype: {first.dtype}")

    # Accumulate sum
    accumulator = np.zeros_like(first, dtype=np.float64)
    count = 0

    for fpath in tqdm(wfpi_files, desc="  Loading"):
        m = np.load(str(fpath))
        accumulator += m.astype(np.float64)
        count += 1

    avg_map = (accumulator / count).astype(np.float32)

    # Save
    output_path = dataset_path / output_filename
    np.save(str(output_path), avg_map)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {avg_map.shape}, min: {avg_map.min():.2f}, max: {avg_map.max():.2f}, mean: {avg_map.mean():.2f}")

    return avg_map


def main():
    for dataset_info in DATASETS:
        name = dataset_info["name"]
        dataset_dir = dataset_info["dir"]
        glob_pattern = dataset_info["glob"]
        output_filename = dataset_info["output"]

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        if not os.path.isdir(dataset_dir):
            print(f"  Directory not found: {dataset_dir} — skipping")
            continue

        aggregate_wfpi_maps(dataset_dir, glob_pattern, output_filename)

    print("\nDone.")


if __name__ == "__main__":
    main()
