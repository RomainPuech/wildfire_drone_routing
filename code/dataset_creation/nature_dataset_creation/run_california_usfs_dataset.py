#!/usr/bin/env python3
"""
Single entry point to generate the full California USFS dataset pipeline and report plots.

Runs in order:
  1. Build the Day 1 mask if missing (mask_union_burnable_no_snow_excluded_day1.npy).
  2. Stage-1: explore_california_2020_ignitions.py and explore_california_2021_ignitions.py
     → report/california_2020_ignition_points.png, .md and california_2021_ignition_points.png, .md
  3. Stage-2: filter_wfpi_and_plot.py
     → report/california_2020_ignition_points_wfpi.png, california_2021_ignition_points_wfpi.png
       and appends Stage-2 section to both .md reports

So the whole 2021 (and 2020) California USFS dataset filtering and all plots used in the
report are generatable by running this script once.

Usage:
  python run_california_usfs_dataset.py [--years 2020 2021] [--skip-mask]
  Default: run mask if missing, then both years. --skip-mask skips building the mask.
"""

import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "California2020Dataset")
MASK_PATH = os.path.join(DATASET_DIR, "mask_union_burnable_no_snow_excluded_day1.npy")


def main():
    parser = argparse.ArgumentParser(description="Generate California USFS dataset and report plots.")
    parser.add_argument("--years", nargs="+", default=["2020", "2021"], help="Years to process (default: 2020 2021)")
    parser.add_argument("--skip-mask", action="store_true", help="Do not build Day 1 mask even if missing")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    code_dir = os.path.join(SCRIPT_DIR, "..", "..")  # code/
    if os.path.abspath(code_dir) not in sys.path:
        sys.path.insert(0, os.path.abspath(code_dir))

    # 1. Build Day 1 mask if missing
    if not args.skip_mask and not os.path.isfile(MASK_PATH):
        print("[1/3] Building Day 1 mask …")
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "build_mask_union_burnable_day1.py")],
            cwd=PROJECT_ROOT,
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
    else:
        if args.skip_mask:
            print("[1/3] Skipping mask (--skip-mask).")
        else:
            print("[1/3] Day 1 mask already present.")

    # 2. Stage-1 exploration (per year)
    print("[2/3] Stage-1 filter and plots …")
    for year in args.years:
        script = os.path.join(SCRIPT_DIR, f"explore_california_{year}_ignitions.py")
        if not os.path.isfile(script):
            print(f"  Warning: {script} not found, skipping year {year}")
            continue
        r = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT)
        if r.returncode != 0:
            sys.exit(r.returncode)

    # 3. Stage-2 WFPI filter and WFPI-overlay plots
    print("[3/3] Stage-2 WFPI filter and overlay plots …")
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "filter_wfpi_and_plot.py")],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    print("Done. Report outputs:")
    for year in args.years:
        print(f"  report/california_{year}_ignition_points.png")
        print(f"  report/california_{year}_ignition_points.md")
        print(f"  report/california_{year}_ignition_points_wfpi.png")


if __name__ == "__main__":
    main()
