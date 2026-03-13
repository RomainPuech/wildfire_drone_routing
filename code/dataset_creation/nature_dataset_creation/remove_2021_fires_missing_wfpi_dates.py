#!/usr/bin/env python3
"""
Remove from California2021Dataset the fires whose discovery date falls on a
missing 2021 WFPI zip date. Updates scenarii/, config_california_2021.json,
and dataset_summary.json.

Run from project root after create_california_2021_dataset.py has been run once:
    python code/dataset_creation/nature_dataset_creation/remove_2021_fires_missing_wfpi_dates.py
"""

import json
import os
from pathlib import Path
from datetime import date, timedelta

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."
DATA_DIR = SCRIPT_DIR / "data"
WFPI_2021_D2_DIR = DATA_DIR / "2021_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA"
OUTPUT_DIR = PROJECT_ROOT / "California2021Dataset"
SCENARII_DIR = OUTPUT_DIR / "scenarii"
CONFIG_PATH = OUTPUT_DIR / "config_california_2021.json"
SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"


def get_missing_2021_wfpi_dates():
    """Return set of YYYYMMDD for which 2021 D2 zip is missing."""
    if not WFPI_2021_D2_DIR.exists():
        return set()
    have = set()
    for f in WFPI_2021_D2_DIR.glob("wfpi-forecast-2_data_*_*.zip"):
        parts = f.stem.split("_")
        if len(parts) >= 4 and len(parts[3]) == 8:
            have.add(parts[3])
    all_2021 = set((date(2021, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(365))
    return all_2021 - have


def main():
    missing_dates = get_missing_2021_wfpi_dates()
    if not missing_dates:
        print("No missing 2021 WFPI dates; nothing to remove.")
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Find base names (scenario id without _scenario1) whose date_<base> is in missing_dates
    bases_to_remove = []
    for key, value in config.items():
        if not key.startswith("date_"):
            continue
        base = key[5:]  # strip "date_"
        if value in missing_dates:
            bases_to_remove.append(base)

    if not bases_to_remove:
        print("No fires in config on missing WFPI dates; nothing to remove.")
        return

    print(f"Removing {len(bases_to_remove)} fires (discovery on missing WFPI zip date) …")

    # Delete scenario files
    removed = 0
    for base in bases_to_remove:
        npy_path = SCENARII_DIR / f"{base}_scenario1.npy"
        if npy_path.exists():
            npy_path.unlink()
            removed += 1

    # Remove from config: offset_<base>, date_<base>, time_<base> for bases_to_remove
    def drop_key(k):
        if k.startswith("offset_"):
            return k[7:] in bases_to_remove
        if k.startswith("date_") or k.startswith("time_"):
            return k[5:] in bases_to_remove
        return False

    new_config = {k: v for k, v in config.items() if not drop_key(k)}
    with open(CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"  Deleted {removed} scenario files, updated config.")

    # Update dataset_summary.json
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    summary["successful_fires"] = summary["successful_fires"] - len(bases_to_remove)
    summary["excluded_missing_wfpi_date"] = len(bases_to_remove)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Updated dataset_summary.json: successful_fires = {summary['successful_fires']}")

    print("Done.")


if __name__ == "__main__":
    main()
