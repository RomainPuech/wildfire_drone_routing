#!/usr/bin/env python3
"""
reproduce_tables23.py — Reproduce Tables 2 & 3 from splits/

Usage:
    # Point at the folder containing the combined benchmark CSVs
    python reproduce_tables23.py --results-dir /path/to/results/bm

    # Or use the WideDataset directly (requires all layout folders extracted)
    python reproduce_tables23.py --dataset-dir ./WideDataset

The script:
  1. Reads splits/tables23_layouts.txt and splits/tables23_scenarios.csv
  2. Loads benchmark results (pre-computed CSVs or from WideDataset layout folders)
  3. Filters to the 474-scenario Tables 2/3 split
  4. Aggregates by fire-size × fire-speed bins (Table 2) and by risk-map type (Table 3)
  5. Prints markdown tables to stdout and saves CSV summaries to splits/

Alternatively, to re-run the benchmarks from scratch:
    # In all_experiments_parallel.py, pass the layout folder names to
    # selected_layout_names. Folder names are LLLL_XXXXX (layout number
    # followed by an underscore and a numeric suffix).  The benchmark code
    # extracts the layout number via layout.split("_")[0].
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS_DIR = SCRIPT_DIR / "splits"

# ── Strategy shorthands ──────────────────────────────────────────────────────
STRATEGY_FILES = {
    # (sensor_prefix, drone_prefix): combined CSV filename fragment
    ("K", "M"):  "KMbm_parallel",    # MaxCov placement + MaxCov routing
    ("K", "U"):  "KUbm_parallel",    # MaxCov placement + Uniform routing
    ("K", "TOP"): "KTOPbm_parallel", # MaxCov placement + TOP routing
    ("R", "M"):  "RMbm_parallel",    # Random placement + MaxCov routing
    ("R", "U"):  "RUbm_parallel",    # Random placement + Uniform routing
    ("R", "TOP"): "RTOPbm_parallel", # Random placement + TOP routing
}

STRATEGY_DISPLAY = {
    ("K", "M"):   "MaxCov + MaxCov",
    ("K", "U"):   "MaxCov + Uniform",
    ("K", "TOP"): "MaxCov + TOP",
    ("R", "M"):   "Random + MaxCov",
    ("R", "U"):   "Random + Uniform",
    ("R", "TOP"): "Random + TOP",
}


def load_split():
    """Return set of (layout_id, scenario_id) and metadata dict."""
    scenarios_path = SPLITS_DIR / "tables23_scenarios.csv"
    if not scenarios_path.exists():
        sys.exit(f"ERROR: {scenarios_path} not found. Run from the repo root.")
    split_set = set()
    meta = {}
    with open(scenarios_path) as f:
        for row in csv.DictReader(f):
            key = (row["layout_id"], row["scenario_id"])
            split_set.add(key)
            meta[key] = row
    return split_set, meta


def load_benchmark_csv(path):
    """Load a combined_benchmark_results CSV, return list of row dicts."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def find_results_csvs(results_dir):
    """Find combined benchmark CSVs in results_dir, return dict keyed by strategy tuple."""
    found = {}
    for key, fragment in STRATEGY_FILES.items():
        fname = f"combined_benchmark_results{fragment}.csv"
        p = Path(results_dir) / fname
        if p.exists():
            found[key] = p
    return found


def find_results_from_dataset(dataset_dir):
    """Walk WideDataset layout folders and collect per-layout benchmark CSVs."""
    all_rows = defaultdict(list)
    dataset_path = Path(dataset_dir)
    for layout_folder in sorted(dataset_path.iterdir()):
        if not layout_folder.is_dir():
            continue
        layout_id = layout_folder.name.split("_")[0]
        for csv_file in layout_folder.glob("*_benchmark_results*.csv"):
            with open(csv_file) as f:
                for row in csv.DictReader(f):
                    strat_key = _strategy_key_from_row(row)
                    if strat_key:
                        all_rows[strat_key].append(row)
    return all_rows


def _strategy_key_from_row(row):
    sensor = row.get("sensor_strategy", "")
    drone = row.get("drone_strategy", "")
    if "MaxCoverage" in sensor:
        sp = "K"
    elif "Random" in sensor:
        sp = "R"
    else:
        return None
    if "MaxCoverage" in drone:
        dp = "M"
    elif "Uniform" in drone:
        dp = "U"
    elif "TOP" in drone:
        dp = "TOP"
    else:
        return None
    return (sp, dp)


def aggregate_table2(filtered_rows, meta):
    """Aggregate mean delta_t by (fire_size, fire_speed) bins — Table 2."""
    bins = defaultdict(lambda: defaultdict(list))
    for row in filtered_rows:
        key = (row["layout"], row["scenario"])
        m = meta.get(key)
        if not m:
            continue
        big = m.get("big_fire", "") == "True"
        fast = m.get("fast_fire", "") == "True"
        size_label = "Big" if big else "Small"
        speed_label = "Fast" if fast else "Slow"
        cell = f"{speed_label} × {size_label}"
        dt = float(row["delta_t"])
        bins[cell]["values"].append(dt)
    result = {}
    for cell, data in bins.items():
        vals = data["values"]
        result[cell] = {"mean_delta_t": sum(vals) / len(vals), "n": len(vals)}
    return result


def print_table2(all_strategy_results, meta):
    """Print Table 2 (by fire-size × fire-speed) for all strategies."""
    cells_order = ["Fast × Big", "Fast × Small", "Slow × Big", "Slow × Small"]
    print("\n## Table 2: Mean Δt by fire-size × fire-speed bin (burn-map risk)\n")
    header = "| Strategy | " + " | ".join(cells_order) + " |"
    sep = "|" + "---|" * (len(cells_order) + 1)
    print(header)
    print(sep)
    for strat_key in sorted(all_strategy_results.keys()):
        rows = all_strategy_results[strat_key]
        agg = aggregate_table2(rows, meta)
        label = STRATEGY_DISPLAY.get(strat_key, str(strat_key))
        vals = []
        for c in cells_order:
            if c in agg:
                vals.append(f"{agg[c]['mean_delta_t']:.2f} (n={agg[c]['n']})")
            else:
                vals.append("—")
        print(f"| {label} | " + " | ".join(vals) + " |")


def print_table3(all_strategy_results, meta):
    """Print Table 3 summary (overall mean Δt per strategy — burn-map risk)."""
    print("\n## Table 3: Overall mean Δt per strategy (burn-map risk)\n")
    print("| Strategy | Mean Δt | n |")
    print("|---|---|---|")
    for strat_key in sorted(all_strategy_results.keys()):
        rows = all_strategy_results[strat_key]
        if not rows:
            continue
        vals = [float(r["delta_t"]) for r in rows]
        label = STRATEGY_DISPLAY.get(strat_key, str(strat_key))
        print(f"| {label} | {sum(vals)/len(vals):.2f} | {len(vals)} |")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Tables 2 & 3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir",
                       help="Directory containing combined_benchmark_results*.csv files")
    group.add_argument("--dataset-dir",
                       help="Path to WideDataset with extracted layout folders")
    args = parser.parse_args()

    split_set, meta = load_split()
    print(f"Loaded split: {len(split_set)} scenarios across "
          f"{len(set(k[0] for k in split_set))} layouts")

    if args.results_dir:
        csv_map = find_results_csvs(args.results_dir)
        if not csv_map:
            sys.exit(f"No combined_benchmark_results*.csv found in {args.results_dir}")
        all_strategy_results = {}
        for strat_key, path in csv_map.items():
            rows = load_benchmark_csv(path)
            filtered = [r for r in rows
                        if (r["layout"], r["scenario"]) in split_set]
            all_strategy_results[strat_key] = filtered
            print(f"  {STRATEGY_DISPLAY[strat_key]}: {len(filtered)}/{len(rows)} rows in split")
    else:
        raw = find_results_from_dataset(args.dataset_dir)
        all_strategy_results = {}
        for strat_key, rows in raw.items():
            filtered = [r for r in rows
                        if (r["layout"], r["scenario"]) in split_set]
            all_strategy_results[strat_key] = filtered
            print(f"  {STRATEGY_DISPLAY.get(strat_key, strat_key)}: "
                  f"{len(filtered)}/{len(rows)} rows in split")

    print_table2(all_strategy_results, meta)
    print_table3(all_strategy_results, meta)

    out_path = SPLITS_DIR / "tables23_reproduced.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "cell", "mean_delta_t", "n"])
        for strat_key in sorted(all_strategy_results.keys()):
            rows = all_strategy_results[strat_key]
            agg = aggregate_table2(rows, meta)
            label = STRATEGY_DISPLAY.get(strat_key, str(strat_key))
            for cell, vals in sorted(agg.items()):
                w.writerow([label, cell, f"{vals['mean_delta_t']:.4f}", vals["n"]])
    print(f"\nSaved reproduced aggregation to {out_path}")


if __name__ == "__main__":
    main()
