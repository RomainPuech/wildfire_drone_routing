#!/usr/bin/env python3
"""
Augment California 2020 Config with Discovery Date and Time

For each scenario already in config_california_2020.json, looks up
DISCOVERY_DATE and DISCOVERY_TIME from the FPA FOD database and adds:
  date_<name>: "YYYYMMDD"   (fire discovery date)
  time_<name>: "HHMM"       (fire discovery time, 24-hour format)

Scenarios whose fire has no DISCOVERY_TIME recorded are removed from the
config (they cannot be used with the time-aware yearly WFPI benchmark).

Run from the project root:
    python code/dataset_creation/nature_dataset_creation/augment_config_with_times.py
"""

import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."

FIRE_DB_PATH = SCRIPT_DIR / "data/RDS-2013-0009.6_Data_Format3_GPKG/FPA_FOD_20221014.gpkg"
CONFIG_PATH  = PROJECT_ROOT / "California2020Dataset/config_california_2020.json"


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_fire_date(date_str) -> datetime | None:
    """Parse fire discovery date from various formats."""
    if not date_str:
        return None
    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y"]:
        try:
            return datetime.strptime(date_str.split()[0], fmt)
        except Exception:
            continue
    return None


def extract_fod_id(scenario_name: str) -> int | None:
    """Extract the numeric FOD_ID from the end of a scenario name.

    The scenario name format is: {safe_fire_name}_{fod_id}
    e.g. "GULCH_TRL__JONESVALLEY_400612285" → 400612285
    """
    m = re.search(r"(\d+)$", scenario_name)
    return int(m.group(1)) if m else None


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading config from {CONFIG_PATH} ...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Collect all offset entries and their scenario names / FOD IDs
    offset_entries = {}   # name → fod_id
    for key, value in config.items():
        if not key.startswith("offset_"):
            continue
        name = key[len("offset_"):]
        fod_id = extract_fod_id(name)
        if fod_id is not None:
            offset_entries[name] = fod_id

    print(f"  {len(offset_entries)} offset entries found")

    all_fod_ids = list(set(offset_entries.values()))

    # Query the database
    print(f"Querying FPA FOD database for {len(all_fod_ids)} unique FOD IDs ...")
    conn = sqlite3.connect(str(FIRE_DB_PATH))
    cur  = conn.cursor()
    placeholders = ",".join(["?"] * len(all_fod_ids))
    cur.execute(
        f"SELECT FOD_ID, DISCOVERY_DATE, DISCOVERY_TIME "
        f"FROM Fires WHERE FOD_ID IN ({placeholders})",
        all_fod_ids,
    )
    db_rows = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.close()
    print(f"  Retrieved {len(db_rows)} rows from database")

    # Build augmented config
    new_config  = {}
    kept        = 0
    no_time     = 0
    no_db_row   = 0
    bad_date    = 0

    for name, fod_id in offset_entries.items():
        if fod_id not in db_rows:
            no_db_row += 1
            continue

        disc_date_raw, disc_time_raw = db_rows[fod_id]

        if not disc_time_raw:
            no_time += 1
            continue

        parsed_date = parse_fire_date(disc_date_raw)
        if parsed_date is None:
            bad_date += 1
            continue

        time_str = str(disc_time_raw).zfill(4)   # ensure "HHMM" with leading zeros

        new_config[f"offset_{name}"] = config[f"offset_{name}"]
        new_config[f"date_{name}"]   = parsed_date.strftime("%Y%m%d")
        new_config[f"time_{name}"]   = time_str
        kept += 1

    print(f"\nResults:")
    print(f"  Kept (have date + time): {kept}")
    print(f"  Skipped — no DISCOVERY_TIME:  {no_time}")
    print(f"  Skipped — not in database:    {no_db_row}")
    print(f"  Skipped — unparseable date:   {bad_date}")

    # Save
    with open(CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"\nSaved augmented config ({len(new_config)} keys) → {CONFIG_PATH}")


if __name__ == "__main__":
    main()
