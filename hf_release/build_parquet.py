#!/usr/bin/env python3
"""
Build Parquet index files for the HuggingFace dataset release.

Reads:
  - notebooks/scenario_summary.csv  (or the copy in the repo)
  - splits/tables23_scenarios.csv

Writes:
  - hf_release/data/scenarios_index.parquet     (default config, train split)
  - hf_release/data/tables23_scenarios.parquet   (tables23 config, test split)
"""

import csv
import os
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required: pip install pyarrow")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"


def _to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def _to_float(val):
    if val is None or val == "":
        return None
    return float(val)


def _to_int(val):
    if val is None or val == "":
        return None
    return int(val)


def build_scenarios_index():
    """Build the full scenarios_index.parquet from scenario_summary.csv."""
    summary_paths = [
        REPO_ROOT / "notebooks" / "scenario_summary.csv",
        Path(os.environ.get("SCENARIO_SUMMARY_CSV", "")) if os.environ.get("SCENARIO_SUMMARY_CSV") else None,
    ]
    summary_path = None
    for p in summary_paths:
        if p and p.exists():
            summary_path = p
            break

    if summary_path is None:
        print("WARNING: scenario_summary.csv not found. "
              "Set SCENARIO_SUMMARY_CSV env var or place it in notebooks/.")
        print("Skipping scenarios_index.parquet generation.")
        return False

    rows = []
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "layout_id": row["layout_number"],
                "scenario_id": row["scenario_number"],
                "season_number": _to_float(row.get("season_number")),
                "seasonal_match": _to_bool(row.get("seasonal_match", False)),
                "historical_match": _to_bool(row.get("historical_match", False)),
                "big_fire": _to_bool(row.get("big_fire", False)),
                "small_fire": _to_bool(row.get("small_fire", False)),
                "fast_fire": _to_bool(row.get("fast_fire", False)),
                "slow_fire": _to_bool(row.get("slow_fire", False)),
            })

    schema = pa.schema([
        ("layout_id", pa.string()),
        ("scenario_id", pa.string()),
        ("season_number", pa.float32()),
        ("seasonal_match", pa.bool_()),
        ("historical_match", pa.bool_()),
        ("big_fire", pa.bool_()),
        ("small_fire", pa.bool_()),
        ("fast_fire", pa.bool_()),
        ("slow_fire", pa.bool_()),
    ])

    table = pa.table({
        col: [r[col] for r in rows] for col in schema.names
    }, schema=schema)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "scenarios_index.parquet"
    pq.write_table(table, out_path)
    print(f"✅ Wrote {len(rows)} rows to {out_path}")
    return True


def build_tables23():
    """Build tables23_scenarios.parquet from splits/tables23_scenarios.csv."""
    csv_path = REPO_ROOT / "splits" / "tables23_scenarios.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return False

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "layout_id": row["layout_id"],
                "scenario_id": row["scenario_id"],
                "historical_match_pct": _to_float(row.get("historical_match_pct")),
                "big_fire": _to_bool(row.get("big_fire")) if row.get("big_fire") else None,
                "small_fire": _to_bool(row.get("small_fire")) if row.get("small_fire") else None,
                "fast_fire": _to_bool(row.get("fast_fire")) if row.get("fast_fire") else None,
                "slow_fire": _to_bool(row.get("slow_fire")) if row.get("slow_fire") else None,
                "per_layout_n": _to_int(row.get("per_layout_n")),
                "in_config_s2r": _to_bool(row.get("in_config_s2r", True)),
                "in_scenario_summary": _to_bool(row.get("in_scenario_summary", True)),
            })

    schema = pa.schema([
        ("layout_id", pa.string()),
        ("scenario_id", pa.string()),
        ("historical_match_pct", pa.float32()),
        ("big_fire", pa.bool_()),
        ("small_fire", pa.bool_()),
        ("fast_fire", pa.bool_()),
        ("slow_fire", pa.bool_()),
        ("per_layout_n", pa.int32()),
        ("in_config_s2r", pa.bool_()),
        ("in_scenario_summary", pa.bool_()),
    ])

    table = pa.table({
        col: [r[col] for r in rows] for col in schema.names
    }, schema=schema)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "tables23_scenarios.parquet"
    pq.write_table(table, out_path)
    print(f"✅ Wrote {len(rows)} rows to {out_path}")
    return True


if __name__ == "__main__":
    ok1 = build_scenarios_index()
    ok2 = build_tables23()
    if ok1 and ok2:
        print("\n✅ All parquet files built. Ready for HuggingFace upload.")
    elif ok2:
        print("\n⚠️  tables23 parquet built. scenarios_index skipped (no scenario_summary.csv).")
    else:
        print("\n❌ Some files could not be built.")
        sys.exit(1)
