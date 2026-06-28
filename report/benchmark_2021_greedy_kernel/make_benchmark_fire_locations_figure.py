#!/usr/bin/env python3
"""Regenerate benchmark_fire_locations_budget_2021.png from current dataset.

Uses the same 100-scenario subset as run_benchmark_california2021_yearly.py
(seed=42). Writes into this directory's figures/ folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from displays import plot_fire_locations  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "California2021Dataset"
PLACEMENT_MAP = DATASET_DIR / "static_risk_pyrologix.npy"
MASK_PATH = DATASET_DIR / "mask.npy"
CONFIG_PATH = DATASET_DIR / "config_california_2021.json"
SCENARII_DIR = DATASET_DIR / "scenarii"

BENCHMARK_SUBSET_SIZE = 100
RANDOM_SEED = 42

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
OUT_PNG = FIGURES_DIR / "benchmark_fire_locations_budget_2021.png"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pyro_map = np.load(str(PLACEMENT_MAP))
    if pyro_map.ndim == 3:
        pyro_map = pyro_map[0]
    mask_arr = np.load(str(MASK_PATH))
    background_vis = np.where(mask_arr > 0, pyro_map.astype(float), 0.0)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    all_scenario_files = sorted(SCENARII_DIR.glob("*.npy"))
    valid_scenarios = [
        sf
        for sf in all_scenario_files
        if all(
            f"{k}_{sf.stem.replace('_scenario1', '')}" in config
            for k in ("offset", "date", "time")
        )
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = np.sort(
        rng.choice(len(valid_scenarios), size=BENCHMARK_SUBSET_SIZE, replace=False)
    )
    benchmark_scenarios = [valid_scenarios[i] for i in subset_idx]

    fire_rows: list[int] = []
    fire_cols: list[int] = []
    for sf in benchmark_scenarios:
        pt = np.load(str(sf))
        fire_rows.append(int(pt[0]))
        fire_cols.append(int(pt[1]))

    plot_fire_locations(
        background_vis,
        fire_rows,
        fire_cols,
        OUT_PNG,
        title="California 2021 benchmark fires (n=100, seed=42) on Pyrologix map",
        marker_size=15,
    )
    print(f"Saved {OUT_PNG.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
