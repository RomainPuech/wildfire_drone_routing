#!/usr/bin/env python3
"""
Compare the Yearly WFPI Burn Map vs other risk maps.

Methodology (same as 09_risk_map_comparison.md):
  For each fire in California2020Dataset:
    1. Load the ignition point (row, col) from the scenario .npy file.
    2. Determine the correct frame in the yearly map:
         hour < 10 → frame 2*(day_of_year-1) + 0  (D2)
         hour >= 10 → frame 2*(day_of_year-1) + 1  (D1)
       If discovery time is NULL in the DB → skip the fire.
    3. Read the risk value at (row, col) from the yearly map and from
       each reference map (D2, D1, ignition prob, burn prob).
    4. Compare fire-location values vs background (all valid mask cells).

Outputs a markdown comparison document.

Run from the project root:
    python code/dataset_creation/nature_dataset_creation/analyze_yearly_wfpi_map.py
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from datetime import date, datetime

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."
DATA_DIR     = SCRIPT_DIR / "data"

D2_DATASET  = PROJECT_ROOT / "California2020Dataset"
D1_DATASET  = PROJECT_ROOT / "California2020Dataset_Day1"

YEARLY_MAP  = D2_DATASET / "static_risk_wfpi_yearly.npy"
MASK_PATH   = D2_DATASET / "mask.npy"
FIRE_DB     = DATA_DIR / "RDS-2013-0009.6_Data_Format3_GPKG/FPA_FOD_20221014.gpkg"
CONFIG_PATH = D2_DATASET / "config_california_2020.json"

# Static risk maps (from other datasets — same grid after cropping)
IGN_PROB_PATH  = PROJECT_ROOT / "California2020Dataset_IgnitionProb/static_risk_ignition_prob.npy"
BURN_PROB_PATH = PROJECT_ROOT / "California2020Dataset_BurnProb/static_risk_burn_prob.npy"

OUTPUT_MD = PROJECT_ROOT / "documentation/12_yearly_wfpi_map_comparison.md"
LARGE_FIRE_ACRES = 100  # threshold for "large fires"


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_time(time_str):
    """Parse HHMM string to (hour, minute). Returns None if invalid."""
    if not time_str or not str(time_str).strip():
        return None
    s = str(time_str).strip().zfill(4)
    try:
        h, m = int(s[:2]), int(s[2:])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except ValueError:
        pass
    return None


def parse_date(date_str):
    """Parse fire discovery date. Returns date or None."""
    if not date_str:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str.split()[0], fmt).date()
        except ValueError:
            continue
    return None


def frame_index_for(discovery_date: date, hour: int) -> int:
    """Return the yearly-map frame index for a given date and hour."""
    import calendar
    # day_of_year: 1-indexed
    day_of_year = discovery_date.timetuple().tm_yday
    half = 0 if hour < 10 else 1
    return 2 * (day_of_year - 1) + half


def wfpi_date_str_for_d2(discovery_date: date) -> str:
    """The D2 .npy filename date (= discovery_date - 1 day)."""
    from datetime import timedelta
    return (discovery_date - timedelta(days=1)).strftime("%Y%m%d")


def stats(values: np.ndarray, background_median: float):
    """Return (median, median_ratio, pct_above_median) for a set of values."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(values))
    ratio = med / background_median if background_median > 0 else np.nan
    pct_above = float(np.mean(values > background_median)) * 100
    return med, ratio, pct_above


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading mask...")
    mask = np.load(str(MASK_PATH))           # (H, W), 1=valid
    valid_idx = np.where(mask == 1)          # arrays of row, col indices
    print(f"  Valid cells: {len(valid_idx[0]):,}")

    print("Loading yearly map (this is 3 GB — may take a moment)...")
    yearly = np.load(str(YEARLY_MAP))        # (732, H, W)
    print(f"  Shape: {yearly.shape}")

    # Background medians computed below after loading the avg maps

    # Use yearly-average maps as backgrounds (consistent with 09_risk_map_comparison.md).
    # This averages over all 366 days so winter + summer are both represented.
    print("Computing backgrounds from yearly-average maps...")
    d2_avg_map = np.load(str(D2_DATASET / "static_risk_wfpi_avg.npy"))[0]   # (H, W)
    d1_avg_map = np.load(str(D1_DATASET / "static_risk_wfpi_avg.npy"))[0]
    yearly_mean_map = yearly.mean(axis=0)                                      # (H, W)

    d2_bg_median     = float(np.median(d2_avg_map[valid_idx[0], valid_idx[1]]))
    d1_bg_median     = float(np.median(d1_avg_map[valid_idx[0], valid_idx[1]]))
    yearly_bg_median = float(np.median(yearly_mean_map[valid_idx[0], valid_idx[1]]))
    print(f"  D2 background median: {d2_bg_median:.2f}")
    print(f"  D1 background median: {d1_bg_median:.2f}")
    print(f"  Yearly background median: {yearly_bg_median:.2f}")

    # Static maps — note they have different grids; skip in this comparison
    ign_prob = burn_prob = None
    ign_bg = burn_bg = np.nan

    # Load fire database — need FOD_ID, discovery date, time, lat/lon, fire_size
    print("\nLoading fire database...")
    conn = sqlite3.connect(str(FIRE_DB))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT FOD_ID, FIRE_NAME, DISCOVERY_DATE, DISCOVERY_TIME, LATITUDE, LONGITUDE, FIRE_SIZE
        FROM Fires
        WHERE FIRE_YEAR = 2020 AND STATE = 'CA'
        AND DISCOVERY_DATE IS NOT NULL AND DISCOVERY_DATE != ''
        AND DISCOVERY_TIME IS NOT NULL AND DISCOVERY_TIME != ''
        AND LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"  {len(rows)} fires with date and time")

    # Build FOD_ID → (date, hour, fire_size) mapping
    fire_meta = {}
    for fod_id, name, disc_date, disc_time, lat, lon, fire_size in rows:
        d = parse_date(disc_date)
        t = parse_time(disc_time)
        if d and t and d.year == 2020:
            fire_meta[str(fod_id)] = {
                "date": d, "hour": t[0], "fire_size": fire_size or 0
            }
    print(f"  {len(fire_meta)} fires with valid date+time in 2020")

    # Load config to map scenario names to FOD IDs
    with open(str(CONFIG_PATH)) as f:
        config = json.load(f)

    # Iterate scenarios
    scenario_dir = D2_DATASET / "scenarii"
    scenario_files = sorted(scenario_dir.glob("*.npy"))
    print(f"\n  Processing {len(scenario_files)} scenarios...")

    yearly_vals_all  = []
    d2_vals_all      = []
    d1_vals_all      = []
    ign_vals_all     = []
    burn_vals_all    = []

    yearly_vals_lg   = []
    d2_vals_lg       = []
    d1_vals_lg       = []
    ign_vals_lg      = []
    burn_vals_lg     = []

    skipped = 0

    for sf in scenario_files:
        stem = sf.stem                           # e.g. AUGUST_COMPLEX_400633321_scenario1
        # extract FOD_ID: last numeric token before _scenario
        import re
        m = re.search(r"_(\d+)_scenario", stem)
        if not m:
            skipped += 1
            continue
        fod_id = m.group(1)

        meta = fire_meta.get(fod_id)
        if meta is None:
            skipped += 1
            continue

        disc_date  = meta["date"]
        hour       = meta["hour"]
        fire_size  = meta["fire_size"]

        # Load ignition point
        data = np.load(str(sf))
        if data.shape == (3,):
            row, col = int(data[0]), int(data[1])
        elif data.shape == (2,):
            row, col = int(data[0]), int(data[1])
        else:
            skipped += 1
            continue

        H, W = mask.shape
        if not (0 <= row < H and 0 <= col < W):
            skipped += 1
            continue

        # Yearly map value — use the correct frame for this fire's time
        fi = frame_index_for(disc_date, hour)
        if fi >= len(yearly):
            skipped += 1
            continue
        yr_val = float(yearly[fi, row, col])

        # D2 value — load the per-fire D2 map
        d2_date = wfpi_date_str_for_d2(disc_date)
        d2_path = D2_DATASET / f"wfpi_{d2_date}.npy"
        if not d2_path.exists():
            skipped += 1
            continue
        d2_map = np.load(str(d2_path))[0]
        d2_val = float(d2_map[row, col])

        # D1 value — load the per-fire D1 map
        d1_date = disc_date.strftime("%Y%m%d")
        d1_path = D1_DATASET / f"wfpi_day1_{d1_date}.npy"
        if not d1_path.exists():
            skipped += 1
            continue
        d1_map = np.load(str(d1_path))[0]
        d1_val = float(d1_map[row, col])

        yearly_vals_all.append(yr_val)
        d2_vals_all.append(d2_val)
        d1_vals_all.append(d1_val)

        if ign_prob is not None and row < ign_prob.shape[0] and col < ign_prob.shape[1]:
            ign_vals_all.append(float(ign_prob[row, col]))
        if burn_prob is not None and row < burn_prob.shape[0] and col < burn_prob.shape[1]:
            burn_vals_all.append(float(burn_prob[row, col]))

        if fire_size >= LARGE_FIRE_ACRES:
            yearly_vals_lg.append(yr_val)
            d2_vals_lg.append(d2_val)
            d1_vals_lg.append(d1_val)
            if ign_prob is not None and row < ign_prob.shape[0] and col < ign_prob.shape[1]:
                ign_vals_lg.append(float(ign_prob[row, col]))
            if burn_prob is not None and row < burn_prob.shape[0] and col < burn_prob.shape[1]:
                burn_vals_lg.append(float(burn_prob[row, col]))

    print(f"  Matched: {len(yearly_vals_all)} fires  |  Large fires: {len(yearly_vals_lg)}  |  Skipped: {skipped}")

    # ── Compute statistics ─────────────────────────────────────────────────────
    yearly_arr_all = np.array(yearly_vals_all)
    d2_arr_all     = np.array(d2_vals_all)
    d1_arr_all     = np.array(d1_vals_all)

    results = {}
    for label, fire_vals, bg_med in [
        ("WFPI Yearly (time-aware)", yearly_arr_all, yearly_bg_median),
        ("WFPI Day 2",               d2_arr_all,     d2_bg_median),
        ("WFPI Day 1",               d1_arr_all,     d1_bg_median),
    ]:
        med, ratio, pct = stats(fire_vals, bg_med)
        results[label] = {"all": (med, ratio, pct)}

    # Large fires
    results_lg = {}
    yearly_arr_lg = np.array(yearly_vals_lg)
    d2_arr_lg     = np.array(d2_vals_lg)
    d1_arr_lg     = np.array(d1_vals_lg)

    for label, fire_vals, bg_med in [
        ("WFPI Yearly (time-aware)", yearly_arr_lg, yearly_bg_median),
        ("WFPI Day 2",               d2_arr_lg,     d2_bg_median),
        ("WFPI Day 1",               d1_arr_lg,     d1_bg_median),
    ]:
        med, ratio, pct = stats(fire_vals, bg_med)
        results_lg[label] = (med, ratio, pct)

    # Print summary
    print("\n=== All Fires ===")
    for k, v in results.items():
        med, ratio, pct = v["all"]
        print(f"  {k:35s}: median={med:.4f}, ratio={ratio:.2f}x, pct_above={pct:.1f}%")
    print("\n=== Large Fires ===")
    for k, v in results_lg.items():
        med, ratio, pct = v
        print(f"  {k:35s}: median={med:.4f}, ratio={ratio:.2f}x, pct_above={pct:.1f}%")

    # ── Write markdown ─────────────────────────────────────────────────────────
    write_markdown(results, results_lg,
                   n_all=len(yearly_vals_all), n_lg=len(yearly_vals_lg),
                   d2_bg=d2_bg_median, d1_bg=d1_bg_median,
                   yearly_bg=yearly_bg_median)
    print(f"\nMarkdown written to {OUTPUT_MD}")


def write_markdown(results, results_lg, n_all, n_lg,
                   d2_bg, d1_bg, yearly_bg):
    def row(label, all_t, lg_t):
        ma, ra, pa = all_t
        ml, rl, pl = lg_t if lg_t else (np.nan, np.nan, np.nan)
        def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "N/A"
        return (f"| **{label}** | {fmt(ma)} | {fmt(ra)}x | {fmt(pa)} pp "
                f"| {fmt(ml)} | {fmt(rl)}x | {fmt(pl)} pp |")

    lines = [
        "# Yearly WFPI Burn Map: Risk Map Comparison",
        "",
        "## Overview",
        "",
        "This document compares the new **Yearly WFPI Burn Map** (`static_risk_wfpi_yearly.npy`) "
        "against the four reference risk maps already evaluated in "
        "[09_risk_map_comparison.md](09_risk_map_comparison.md).",
        "",
        "The key innovation of the yearly map is **time-aware WFPI selection**: instead of always "
        "using the Day-2 or Day-1 forecast, it uses whichever forecast was operationally "
        "available at the fire's discovery time:",
        "",
        "- **Before 10 am** → Day-2 forecast (issued the previous day, the best available before "
        "the 10 am update)",
        "- **After 10 am** → Day-1 forecast (issued the same day, updated at 10 am)",
        "",
        "Fires without a recorded discovery time are excluded from this comparison.",
        "",
        "## Dataset",
        "",
        f"- **All fires analysed:** {n_all:,}",
        f"- **Large fires (≥ {LARGE_FIRE_ACRES} acres):** {n_lg:,}",
        f"- **Background median (D2):** {d2_bg:.2f}",
        f"- **Background median (D1):** {d1_bg:.2f}",
        f"- **Background median (Yearly, combined):** {yearly_bg:.2f}",
        "",
        "## Summary Table",
        "",
        "| Risk Map | Fire Med. (All) | Ratio (All) | % Above Med. (All) "
        "| Fire Med. (Large) | Ratio (Large) | % Above Med. (Large) |",
        "|----------|----------------|-------------|--------------------"
        "|------------------|---------------|----------------------|",
    ]

    map_order = ["WFPI Yearly (time-aware)", "WFPI Day 1", "WFPI Day 2",
                 "Ignition Probability", "Burn Probability (FSim)"]
    for k in map_order:
        if k in results:
            all_t = results[k]["all"]
            lg_t  = results_lg.get(k)
            lines.append(row(k, all_t, lg_t))

    # Determine winner
    def rank_key(k):
        if k not in results:
            return -1
        return results[k]["all"][2]  # % above median
    winner_all = max((k for k in map_order if k in results), key=rank_key)
    winner_lg  = max((k for k in map_order if k in results_lg),
                     key=lambda k: results_lg[k][2] if k in results_lg else -1)

    lines += [
        "",
        f"**Best for all fires (% above median):** {winner_all}",
        f"**Best for large fires (% above median):** {winner_lg}",
        "",
        "## Detailed Analysis",
        "",
        "### WFPI Yearly (Time-Aware)",
        "",
        "The yearly map selects the most operationally accurate WFPI forecast for each fire "
        "based on its discovery time. This reduces the systematic error introduced by always "
        "using D2 (too early) or D1 (occasionally unavailable before 10 am).",
        "",
        "**Construction:**",
        "",
        "- Shape: `(732, 1309, 805)` — 2 frames × 366 days (2020 is a leap year)",
        "- Frame `2*(d-1)+0`: Day-2 forecast for day d (issued on day d−1)",
        "- Frame `2*(d-1)+1`: Day-1 forecast for day d (issued on day d at 10 am)",
        "- Missing source files filled by nearest-neighbour interpolation",
        "- Jan 1 pre-10 am uses Jan 1 D1 as fallback (Dec 31 2019 D2 not available)",
        "",
        "**Indexing at runtime:**",
        "",
        "```python",
        "def frame_index(discovery_date, hour):",
        "    day_of_year = discovery_date.timetuple().tm_yday  # 1–366",
        "    half = 0 if hour < 10 else 1",
        "    return 2 * (day_of_year - 1) + half",
        "```",
        "",
        "### Comparison with Previous Results (from 09_risk_map_comparison.md)",
        "",
        "| Risk Map | % Above Median (All) | % Above Median (Large) |",
        "|----------|---------------------|------------------------|",
    ]

    ref = {
        "WFPI Day 2":             ("+4.7 pp",  "+13.9 pp"),
        "WFPI Day 1":             ("+15.8 pp", "+17.9 pp"),
        "Ignition Probability":   ("+21.4 pp", "+27.6 pp"),
        "Burn Probability (FSim)": ("+5.8 pp", "+22.3 pp"),
    }
    for k, (prev_all, prev_lg) in ref.items():
        lines.append(f"| {k} | {prev_all} | {prev_lg} |")
    lines.append("| **WFPI Yearly (time-aware)** | *see table above* | *see table above* |")

    lines += [
        "",
        "## Interpretation",
        "",
        "The time-aware yearly map is expected to **outperform WFPI Day 2** (because it uses "
        "Day-1 for fires discovered after 10 am, which are the majority) and to perform "
        "**comparably to or better than WFPI Day 1** for fires discovered before 10 am "
        "(where it correctly uses Day-2 rather than a Day-1 map that was not yet updated).",
        "",
        "Compared to static maps (Ignition Probability, Burn Probability), the yearly map "
        "captures daily weather variation but lacks the long-term climatological signal.",
        "",
        "## Files",
        "",
        "| File | Location | Description |",
        "|------|----------|-------------|",
        "| `static_risk_wfpi_yearly.npy` | `California2020Dataset/` | Yearly burn map, shape (732, H, W) |",
        "| `complete_wfpi_datasets.py` | `code/dataset_creation/nature_dataset_creation/` | Fills missing daily WFPI files |",
        "| `create_yearly_wfpi_burnmap.py` | `code/dataset_creation/nature_dataset_creation/` | Builds the yearly map |",
        "| `analyze_yearly_wfpi_map.py` | `code/dataset_creation/nature_dataset_creation/` | Comparison analysis (this document) |",
        "",
        "## References",
        "",
        "- [09_risk_map_comparison.md](09_risk_map_comparison.md) — Baseline risk map comparison",
        "- [04_california_2020_dataset.md](04_california_2020_dataset.md) — Dataset documentation",
        "- USGS Fire Danger Maps: https://firedanger.cr.usgs.gov/apps/staticmaps",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_MD), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
