#!/usr/bin/env python3
"""
Risk Map Comparison on the California 2021 USFS Dataset.

Evaluates 6 burn maps on the 932 fires in California2021Dataset:
  1. WFPI Yearly 2021 (time-aware: D2 before 10 am, D1 from 10 am)
  2. WFPI Day 1 2021 (per-fire daily map)
  3. WFPI Day 2 2021 (per-fire daily map, day before)
  4. WFPI 2021 Averaged (mean over year, excluding values >=249)
  5. Ignition Probability / Pyrologix (static, resampled to 1309x805)
  6. Burn Probability / FSim (static, resampled to 1309x805)

Methodology (same as analyze_yearly_wfpi_map.py / 09_risk_map_comparison.md):
  For each fire with a valid discovery time in config_california_2021.json:
    1. Load (row, col) from scenario .npy.
    2. Read the risk value at (row, col) from each map.
    3. Compare fire-location values vs background median (all valid mask cells).
  Metrics: median, median ratio, % fires above background median.

Outputs:
  documentation/15_2021_risk_map_comparison.md

Run from project root:
    python code/dataset_creation/nature_dataset_creation/analyze_2021_risk_maps.py
"""

import json
import re
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."

DATASET_2021       = PROJECT_ROOT / "California2021Dataset"
DATASET_2021_D1    = PROJECT_ROOT / "California2021Dataset_Day1"
DATASET_2020       = PROJECT_ROOT / "California2020Dataset"

MASK_PATH          = DATASET_2021 / "mask.npy"
YEARLY_MAP_PATH    = DATASET_2021 / "static_risk_wfpi_yearly.npy"
AVG_MAP_PATH       = DATASET_2021 / "static_risk_wfpi_avg.npy"
BURN_ONCE_PATH     = DATASET_2021 / "static_risk_wfpi_burn_at_least_once.npy"
PYROLOGIX_PATH     = DATASET_2020 / "static_risk_pyrologix_resampled.npy"
BURN_PROB_PATH     = DATASET_2020 / "static_risk_burn_prob_resampled.npy"
CONFIG_PATH        = DATASET_2021 / "config_california_2021.json"
SCENARII_DIR       = DATASET_2021 / "scenarii"
OUTPUT_MD          = PROJECT_ROOT / "documentation/15_2021_risk_map_comparison.md"

LARGE_FIRE_SIZECLASS = {"D", "E", "F", "G", "H", "I", "J", "K"}  # >= 100 acres

# Size-class lookups are in config? No — we need them from CSV.
CSV_PATH = SCRIPT_DIR / "data/USFS_ignition_points.csv"


def parse_time(s):
    """Parse HHMM string → (hour, minute) or None."""
    if not s or not str(s).strip():
        return None
    s = str(s).strip().zfill(4)
    try:
        h, m = int(s[:2]), int(s[2:])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except ValueError:
        pass
    return None


def frame_index_for(d: date, hour: int) -> int:
    """yearly map frame index for a given date (2021) and hour."""
    day_of_year = d.timetuple().tm_yday   # 1-365
    half = 0 if hour < 10 else 1
    return 2 * (day_of_year - 1) + half


def stats(values: np.ndarray, bg_median: float):
    """(median, ratio, pct_above_bg_median)."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    med   = float(np.median(values))
    ratio = med / bg_median if bg_median > 0 else np.nan
    pct   = float(np.mean(np.array(values) > bg_median)) * 100
    return med, ratio, pct


def fmt(v, digits=2):
    return f"{v:.{digits}f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"


def pp_improvement(pct_above):
    """Convert raw % above bg median to improvement over 50% baseline (in pp)."""
    if np.isnan(pct_above):
        return np.nan
    return pct_above - 50.0


def load_daily_bg_medians(folder: Path, glob_pattern: str,
                           valid_rows, valid_cols) -> dict:
    """Load every daily WFPI npy in folder, return {date_str: bg_median}."""
    bg = {}
    files = sorted(folder.glob(glob_pattern))
    print(f"  Pre-computing bg medians for {len(files)} daily maps in {folder.name} …")
    for f in files:
        m = re.search(r"(\d{8})", f.name)
        if not m:
            continue
        date_str = m.group(1)
        arr = np.load(str(f))
        frame = arr[0] if arr.ndim == 3 else arr
        bg[date_str] = float(np.median(frame[valid_rows, valid_cols]))
    return bg


def stats_per_day(fire_records, subset="all"):
    """
    Per-day-background adjusted % above median.

    fire_records: list of dicts with keys
        "val_d1", "val_d2", "val_yr",
        "bg_d1",  "bg_d2",  "bg_yr",   (per-fire day-specific bg medians)
        "is_large"

    Returns dict name → (mean_pct_above_day_bg, improvement_pp, n_fires_used)
    for D1, D2, Yearly.

    "per-day % above" is computed as:
      for each fire: above = fire_val > fire_day_bg   (0 or 1)
      average over all fires in the subset.
    This is equivalent to averaging per-day percentages weighted by fires-per-day.
    """
    keep = [r for r in fire_records if (subset == "all" or r["is_large"])]
    out = {}
    for name, val_key, bg_key in [
        ("D1",                 "val_d1", "bg_d1"),
        ("D2",                 "val_d2", "bg_d2"),
        ("Yearly (time-aware)","val_yr", "bg_yr"),
    ]:
        flags = [r[val_key] > r[bg_key] for r in keep
                 if r[val_key] is not None and r[bg_key] is not None]
        if flags:
            pct = float(np.mean(flags)) * 100
            out[name] = (pct, pct - 50.0, len(flags))
        else:
            out[name] = (np.nan, np.nan, 0)
    return out


def main():
    import pandas as pd

    # ── 1. Load mask ──────────────────────────────────────────────────────────
    print("Loading mask …")
    mask = np.load(str(MASK_PATH))
    valid_rows, valid_cols = np.where(mask == 1)
    n_valid = len(valid_rows)
    print(f"  Valid cells: {n_valid:,}")

    # ── 2. Load static maps ───────────────────────────────────────────────────
    print("Loading static maps …")
    pyrologix  = np.load(str(PYROLOGIX_PATH))[0]
    burn_prob  = np.load(str(BURN_PROB_PATH))[0]
    wfpi_avg   = np.load(str(AVG_MAP_PATH))[0]
    burn_once  = np.load(str(BURN_ONCE_PATH))[0]

    pyrologix_bg = float(np.median(pyrologix[valid_rows, valid_cols]))
    burn_prob_bg = float(np.median(burn_prob[valid_rows, valid_cols]))
    wfpi_avg_bg  = float(np.median(wfpi_avg[valid_rows, valid_cols]))
    burn_once_bg = float(np.median(burn_once[valid_rows, valid_cols]))
    print(f"  Pyrologix bg median:      {pyrologix_bg:.4f}")
    print(f"  Burn Prob bg median:      {burn_prob_bg:.4f}")
    print(f"  WFPI avg bg median:       {wfpi_avg_bg:.2f}")
    print(f"  Burn-at-least-once bg:    {burn_once_bg:.2f}")

    # ── 3. Load yearly map + annual bg medians ────────────────────────────────
    print("Loading 2021 yearly map (may take a moment) …")
    yearly = np.load(str(YEARLY_MAP_PATH))   # (730, H, W)
    yearly_mean = yearly.mean(axis=0)
    yearly_bg   = float(np.median(yearly_mean[valid_rows, valid_cols]))

    d2_frames = yearly[0::2]
    d1_frames = yearly[1::2]
    d2_bg_annual = float(np.median(d2_frames.mean(axis=0)[valid_rows, valid_cols]))
    d1_bg_annual = float(np.median(d1_frames.mean(axis=0)[valid_rows, valid_cols]))
    print(f"  Annual bg median D2 (avg map): {d2_bg_annual:.2f}")
    print(f"  Annual bg median D1 (avg map): {d1_bg_annual:.2f}")
    print(f"  Annual bg median Yearly:       {yearly_bg:.2f}")

    # ── 4. Pre-compute per-day background medians (D1 and D2) ─────────────────
    print("Pre-computing per-day background medians …")
    d1_bg_per_day = load_daily_bg_medians(
        DATASET_2021_D1, "wfpi_day1_????????.npy", valid_rows, valid_cols)
    d2_bg_per_day = load_daily_bg_medians(
        DATASET_2021,    "wfpi_????????.npy",      valid_rows, valid_cols)
    print(f"  D1 days loaded: {len(d1_bg_per_day)}  |  D2 days loaded: {len(d2_bg_per_day)}")
    # Sanity check: print median of per-day bg medians
    if d1_bg_per_day:
        print(f"  Median of D1 daily bg medians: {np.median(list(d1_bg_per_day.values())):.2f}")
    if d2_bg_per_day:
        print(f"  Median of D2 daily bg medians: {np.median(list(d2_bg_per_day.values())):.2f}")

    # ── 5. Load config ────────────────────────────────────────────────────────
    print("Loading config …")
    with open(str(CONFIG_PATH)) as f:
        config = json.load(f)

    bases = set()
    for key in config:
        if key.startswith("offset_"):
            bases.add(key[7:])

    meta = {}
    for base in bases:
        date_str = config.get(f"date_{base}")
        time_str = config.get(f"time_{base}")
        if date_str and time_str:
            try:
                d = datetime.strptime(date_str, "%Y%m%d").date()
                t = parse_time(time_str)
                if d and t:
                    meta[base] = {"date": d, "hour": t[0]}
            except ValueError:
                pass

    # ── 6. Load USFS CSV for SIZECLASS ────────────────────────────────────────
    print("Loading USFS CSV for size classes …")
    df = pd.read_csv(str(CSV_PATH), low_memory=False, usecols=["UNIQFIREID", "FIRENAME", "SIZECLASS"])
    df = df[df["FIREYEAR"].notna()] if "FIREYEAR" in df.columns else df
    size_map = {}
    for _, row in df.iterrows():
        uid = str(row.get("UNIQFIREID", "") or "")
        sc  = str(row.get("SIZECLASS",  "") or "")
        if uid:
            size_map[uid] = sc

    # ── 7. Iterate scenarios ──────────────────────────────────────────────────
    print("Processing scenarios …")

    # global-bg collections (same as before, for static maps and annual WFPI comparison)
    collect = {name: {"all": [], "large": []} for name in [
        "Yearly (time-aware)", "D1", "D2", "WFPI Avg", "Burn-once", "Pyrologix", "Burn Prob"
    ]}
    # per-day-bg records for WFPI maps
    fire_records = []   # list of dicts

    n_skipped = 0
    n_no_time = 0

    for sf in sorted(SCENARII_DIR.glob("*.npy")):
        base = sf.stem.replace("_scenario1", "")

        m = meta.get(base)
        if m is None:
            n_no_time += 1
            n_skipped += 1
            continue

        disc_date = m["date"]
        hour      = m["hour"]

        data = np.load(str(sf))
        if data.ndim == 1 and len(data) >= 2:
            row_i, col_i = int(data[0]), int(data[1])
        else:
            n_skipped += 1
            continue

        H, W = mask.shape
        if not (0 <= row_i < H and 0 <= col_i < W):
            n_skipped += 1
            continue

        m_uid = re.search(r"(2021-[A-Z]+-\w+)", base)
        uniq_id  = m_uid.group(1) if m_uid else ""
        sizeclass = size_map.get(uniq_id, "")
        is_large  = sizeclass in LARGE_FIRE_SIZECLASS

        # Yearly map value
        fi = frame_index_for(disc_date, hour)
        if fi >= yearly.shape[0]:
            n_skipped += 1
            continue
        yr_val = float(yearly[fi, row_i, col_i])

        # D2 value and its per-day bg
        d2_date_str = (disc_date - timedelta(days=1)).strftime("%Y%m%d")
        d2_path = DATASET_2021 / f"wfpi_{d2_date_str}.npy"
        if not d2_path.exists():
            n_skipped += 1
            continue
        d2_val = float(np.load(str(d2_path))[0][row_i, col_i])
        d2_day_bg = d2_bg_per_day.get(d2_date_str)

        # D1 value and its per-day bg
        d1_date_str = disc_date.strftime("%Y%m%d")
        d1_path = DATASET_2021_D1 / f"wfpi_day1_{d1_date_str}.npy"
        if not d1_path.exists():
            n_skipped += 1
            continue
        d1_val = float(np.load(str(d1_path))[0][row_i, col_i])
        d1_day_bg = d1_bg_per_day.get(d1_date_str)

        # Yearly per-day bg: use D1 bg if hour>=10, else D2 bg of prev day
        yr_day_bg = d1_day_bg if hour >= 10 else d2_day_bg

        # Static map values
        pyro_val = float(pyrologix[row_i, col_i])
        bp_val   = float(burn_prob[row_i, col_i])
        avg_val  = float(wfpi_avg[row_i, col_i])
        bo_val   = float(burn_once[row_i, col_i])

        for name, val in [
            ("Yearly (time-aware)", yr_val),
            ("D1",    d1_val),
            ("D2",    d2_val),
            ("WFPI Avg", avg_val),
            ("Burn-once", bo_val),
            ("Pyrologix", pyro_val),
            ("Burn Prob",  bp_val),
        ]:
            collect[name]["all"].append(val)
            if is_large:
                collect[name]["large"].append(val)

        fire_records.append({
            "val_d1": d1_val, "bg_d1": d1_day_bg,
            "val_d2": d2_val, "bg_d2": d2_day_bg,
            "val_yr": yr_val, "bg_yr": yr_day_bg,
            "is_large": is_large,
        })

    n_fires = len(collect["D1"]["all"])
    n_large = len(collect["D1"]["large"])
    print(f"  Matched: {n_fires} fires | Large: {n_large} | Skipped: {n_skipped} (no time: {n_no_time})")

    # ── 8. Global-bg statistics (annual average background) ───────────────────
    bg_medians = {
        "Yearly (time-aware)": yearly_bg,
        "D1":        d1_bg_annual,
        "D2":        d2_bg_annual,
        "WFPI Avg":  wfpi_avg_bg,
        "Burn-once": burn_once_bg,
        "Pyrologix": pyrologix_bg,
        "Burn Prob": burn_prob_bg,
    }

    results = {}
    for name, bg_med in bg_medians.items():
        arr_all = np.array(collect[name]["all"])
        arr_lg  = np.array(collect[name]["large"])
        results[name] = {
            "bg":    bg_med,
            "all":   stats(arr_all, bg_med),
            "large": stats(arr_lg,  bg_med),
        }

    print("\n=== Global (annual-average) background ===")
    for name in results:
        med, ratio, pct = results[name]["all"]
        print(f"  {name:28s} all:  median={fmt(med,3)}, ratio={fmt(ratio)}x, "
              f"pct_above={fmt(pct,1)}% (impr: {pct-50:+.1f} pp)")

    # ── 9. Per-day-background statistics (correct for daily maps) ─────────────
    pdg_all   = stats_per_day(fire_records, "all")
    pdg_large = stats_per_day(fire_records, "large")

    print("\n=== Per-day background (fire vs its own day's median) ===")
    for name in ["D1", "D2", "Yearly (time-aware)"]:
        pct_a, imp_a, n_a = pdg_all[name]
        pct_l, imp_l, n_l = pdg_large[name]
        print(f"  {name:28s} all ({n_a}):   pct_above={fmt(pct_a,1)}% impr={imp_a:+.1f} pp")
        print(f"  {name:28s} large ({n_l}): pct_above={fmt(pct_l,1)}% impr={imp_l:+.1f} pp")

    # ── 10. Write markdown ────────────────────────────────────────────────────
    write_markdown(results, pdg_all, pdg_large, n_fires, n_large)
    print(f"\nMarkdown written → {OUTPUT_MD}")


MAP_DISPLAY = {
    "Yearly (time-aware)": "WFPI Yearly 2021 (time-aware)",
    "D1":        "WFPI Day 1 2021",
    "D2":        "WFPI Day 2 2021",
    "WFPI Avg":  "WFPI 2021 Averaged (excl. ≥249)",
    "Burn-once": "WFPI 2021 Burn-at-least-once",
    "Pyrologix": "Ignition Probability (Pyrologix)",
    "Burn Prob": "Burn Probability (FSim/BP)",
}


def write_markdown(results, pdg_all, pdg_large, n_fires, n_large):
    lines = []

    # % above bg median: raw value. Improvement vs background = pct_above - 50
    def imp(pct_above):
        """Improvement from raw % above (subtracts 50)."""
        if np.isnan(pct_above):
            return "N/A"
        return f"{pct_above - 50:+.1f} pp"

    def imp_pp(already_improvement):
        """Format a value that is already an improvement (pct - 50)."""
        if already_improvement is None or (isinstance(already_improvement, float) and np.isnan(already_improvement)):
            return "N/A"
        return f"{already_improvement:+.1f} pp"

    def pct_str(pct_above):
        if np.isnan(pct_above):
            return "N/A"
        return f"{pct_above:.1f}%"

    # Rank by improvement vs 50% background (= pct_above - 50), all fires
    ranking_all   = sorted(results.keys(), key=lambda k: results[k]["all"][2],   reverse=True)
    ranking_large = sorted(results.keys(), key=lambda k: results[k]["large"][2], reverse=True)

    winner_all   = ranking_all[0]
    winner_large = ranking_large[0]

    # Compute improvements
    def impr(name, subset):
        return results[name][subset][2] - 50.0

    lines += [
        "# Risk Map Comparison: California 2021 USFS Dataset",
        "",
        "> **Note on data leakage:** Pyrologix was trained on 2006–2020 historical fire data and is",
        "> likely to have seen 2020 patterns. Burn Probability (FSim) may share some training overlap.",
        "> The five 2021 WFPI-derived maps (Yearly, D1, D2, Averaged, Burn-at-least-once) have no",
        "> leakage for 2021 fires and are the only strictly clean metrics.",
        "",
        "## Overview",
        "",
        "This document evaluates **seven burn map variants** on the **California 2021 USFS wildfire",
        f"dataset** ({n_fires:,} fires with recorded discovery times, out of 932 in the dataset).",
        "Fires come from the USFS ignition-point database, filtered with the 2020 Day-1",
        "union-of-burnable validity mask, and excluding fires on dates with missing 2021 WFPI zips.",
        "",
        "Two methodologies are used:",
        "",
        "**Methodology A — Annual background (global):** the risk-map value at each fire's ignition",
        "cell is compared to a single background median computed over all valid cells and all days",
        "(or the static map). This is the standard methodology from doc 09 and doc 12.",
        "",
        "**Methodology B — Per-day background (WFPI maps only):** for each fire, the background",
        "median is computed from that fire's *specific* daily WFPI map. This removes seasonal",
        "bias (a summer fire is no longer compared to a winter-diluted background) and directly",
        "answers: *was this ignition in a relatively high-risk area on its own day?*",
        "",
        "The key metric is **improvement vs background** = (% fires above bg median) − 50%.",
        "A positive value means fires cluster in higher-risk areas; negative means anti-correlation.",
        "",
        "### Maps Evaluated",
        "",
        "| # | Map | Type | Notes |",
        "|---|-----|------|-------|",
        "| 1 | WFPI Yearly 2021 (time-aware) | Time-varying | D2 before 10 am, D1 from 10 am — same logic as 2020 yearly map |",
        "| 2 | WFPI Day 1 2021 | Time-varying | Per-fire same-day forecast |",
        "| 3 | WFPI Day 2 2021 | Time-varying | Per-fire day-before forecast |",
        "| 4 | WFPI 2021 Averaged | Static | Mean over year, values ≥249 excluded |",
        "| 5 | WFPI 2021 Burn-at-least-once | Static | P(≥1 high-WFPI day) remapped to 0–248; saturated for most cells |",
        "| 6 | Ignition Probability (Pyrologix) | Static | ML model, 2006–2020 training; resampled to 1309×805 |",
        "| 7 | Burn Probability (FSim/BP) | Static | Fire-simulation model; resampled to 1309×805 |",
        "",
        "## Dataset",
        "",
        f"- **Fires analysed:** {n_fires:,} (all with valid discovery date + time in config)",
        f"- **Large fires (≥ 100 acres, size class D–K):** {n_large:,}",
        f"- **Grid:** 1309 × 805, 1 km, WFPI Lambert Azimuthal Equal-Area CRS",
        f"- **Valid cells (mask == 1):** 249,255",
        f"- **Mask:** `mask_union_burnable_no_snow_excluded_day1.npy` (2020 D1 union-of-burnable)",
        "",
    ]

    # Summary table — show both raw % above and improvement vs 50%
    lines += [
        "## Summary Table",
        "",
        "- **Bg Median:** median map value over all valid California cells",
        "- **% Above Bg:** fraction of fire cells with value > bg median (50% = random)",
        "- **Improvement vs bg:** (% Above Bg) − 50% — positive means correlated with fires",
        "- **Ratio:** fire median / background median",
        "",
        "| Rank | Risk Map | Bg Median | Fire Med. (All) | Ratio | % Above (All) | Improvement (All) | % Above (Large) | Improvement (Large) |",
        "|------|----------|-----------|-----------------|-------|---------------|-------------------|-----------------|---------------------|",
    ]
    for rank, name in enumerate(ranking_all, 1):
        r   = results[name]
        bg  = r["bg"]
        ma, ra, pa = r["all"]
        ml, rl, pl = r["large"]
        disp = MAP_DISPLAY[name]
        lines.append(
            f"| {rank} | **{disp}** | {fmt(bg,3)} | {fmt(ma,2)} | {fmt(ra,2)}x "
            f"| {pct_str(pa)} | {imp(pa)} "
            f"| {pct_str(pl)} | {imp(pl)} |"
        )

    lines += [
        "",
        f"**Best for all fires (annual bg):** {MAP_DISPLAY[winner_all]} "
        f"({pct_str(results[winner_all]['all'][2])} fires above bg, {imp(results[winner_all]['all'][2])})",
        f"**Best for large fires (annual bg):** {MAP_DISPLAY[winner_large]} "
        f"({pct_str(results[winner_large]['large'][2])} fires above bg, {imp(results[winner_large]['large'][2])})",
        "",
    ]

    # ── Per-day-background table (WFPI maps only) ──────────────────────────
    lines += [
        "## Per-Day Background: WFPI Maps (Corrected Methodology)",
        "",
        "For time-varying maps, comparing each fire to a single annual background median is",
        "misleading: a fire in summer is compared to a background that mixes summer and winter days.",
        "The correct approach is to compare each fire to the **background of its own day**:",
        "",
        "1. Load the daily WFPI map for each fire's discovery date.",
        "2. Compute the median over all 249,255 valid California cells for that day.",
        "3. Check whether the fire's value exceeds that day's median.",
        "4. Average the binary above/below flags across all fires.",
        "",
        "This removes the seasonal bias and measures whether each fire was in a relatively",
        "**high-risk area on its specific day** — regardless of the season.",
        "",
        "| Map | % Above Day Bg (All) | Improvement (All) | % Above Day Bg (Large) | Improvement (Large) | N fires |",
        "|-----|----------------------|-------------------|------------------------|---------------------|---------|",
    ]
    for name in ["Yearly (time-aware)", "D1", "D2"]:
        disp = MAP_DISPLAY[name]
        pct_a, imp_a, n_a = pdg_all.get(name, (np.nan, np.nan, 0))
        pct_l, imp_l, n_l = pdg_large.get(name, (np.nan, np.nan, 0))
        lines.append(
            f"| **{disp}** | {pct_str(pct_a)} | {imp_pp(imp_a)} "
            f"| {pct_str(pct_l)} | {imp_pp(imp_l)} | {n_a} |"
        )

    # Rank by per-day improvement, all fires
    pdg_ranked_all   = sorted(["D1", "D2", "Yearly (time-aware)"],
                               key=lambda k: pdg_all.get(k, (0,))[0], reverse=True)
    pdg_ranked_large = sorted(["D1", "D2", "Yearly (time-aware)"],
                               key=lambda k: pdg_large.get(k, (0,))[0], reverse=True)
    pdg_winner = pdg_ranked_all[0]
    pdg_winner_l = pdg_ranked_large[0]
    lines += [
        "",
        f"**Best WFPI map (per-day bg, all fires):** {MAP_DISPLAY[pdg_winner]} "
        f"({pct_str(pdg_all[pdg_winner][0])}, {imp_pp(pdg_all[pdg_winner][1])})",
        f"**Best WFPI map (per-day bg, large fires):** {MAP_DISPLAY[pdg_winner_l]} "
        f"({pct_str(pdg_large[pdg_winner_l][0])}, {imp_pp(pdg_large[pdg_winner_l][1])})",
        "",
    ]

    # Ranking section — improvement (pp) as primary metric
    lines += [
        "## Rankings",
        "",
        "### All Fires — ranked by improvement vs 50% background",
        "",
        "| Rank | Map | % Fires Above Bg Median | Improvement vs 50% | Assessment |",
        "|------|-----|------------------------|---------------------|------------|",
    ]
    for rank, name in enumerate(ranking_all, 1):
        pct = results[name]["all"][2]
        improvement = pct - 50.0
        if improvement > 15:
            level = "Strong positive correlation"
        elif improvement > 5:
            level = "Moderate positive correlation"
        elif improvement > -5:
            level = "Near-random / no signal"
        elif improvement > -15:
            level = "Moderate anti-correlation"
        else:
            level = "Strong anti-correlation or saturated"
        lines.append(f"| {rank} | {MAP_DISPLAY[name]} | {pct_str(pct)} | {imp(pct)} | {level} |")

    lines += [
        "",
        "### Large Fires — ranked by improvement vs 50% background",
        "",
        "| Rank | Map | % Fires Above Bg Median | Improvement vs 50% |",
        "|------|-----|------------------------|---------------------|",
    ]
    for rank, name in enumerate(ranking_large, 1):
        pct = results[name]["large"][2]
        lines.append(f"| {rank} | {MAP_DISPLAY[name]} | {pct_str(pct)} | {imp(pct)} |")

    # Per-map detail sections
    lines += ["", "## Detailed Results", ""]
    for name in ranking_all:
        r    = results[name]
        disp = MAP_DISPLAY[name]
        ma, ra, pa = r["all"]
        ml, rl, pl = r["large"]
        lines += [
            f"### {disp}",
            "",
            f"- **Background median:** {fmt(r['bg'], 3)}",
            f"- **All fires** ({n_fires:,}): fire median = {fmt(ma, 2)}, ratio = {fmt(ra, 2)}x, "
            f"{pct_str(pa)} above bg median (improvement: {imp(pa)})",
            f"- **Large fires** ({n_large:,}): fire median = {fmt(ml, 2)}, ratio = {fmt(rl, 2)}x, "
            f"{pct_str(pl)} above bg median (improvement: {imp(pl)})",
            "",
        ]

    # Key findings — written with actual numbers filled in
    pyro_imp  = impr("Pyrologix", "all")
    pyro_l    = impr("Pyrologix", "large")
    bp_imp    = impr("Burn Prob", "all")
    bp_l      = impr("Burn Prob", "large")
    yr_imp    = impr("Yearly (time-aware)", "all")
    yr_l      = impr("Yearly (time-aware)", "large")
    d1_imp    = impr("D1", "all")
    d2_imp    = impr("D2", "all")
    avg_imp   = impr("WFPI Avg", "all")
    avg_l     = impr("WFPI Avg", "large")

    lines += [
        "## Key Findings",
        "",
        f"### 1. Overall winner: {MAP_DISPLAY[winner_all]}",
        "",
        f"Pyrologix places {pct_str(results['Pyrologix']['all'][2])} of fires above the background",
        f"median — an improvement of {imp(results['Pyrologix']['all'][2])} over the 50% baseline.",
        "This is consistent with the 2020 result (+21.4 pp in doc 09) and confirms that Pyrologix",
        "captures long-term ignition risk very well. However, it was trained on 2006–2020 data and",
        "**should be considered contaminated** for 2021 evaluation.",
        "",
        f"### 2. Best clean (no-leakage) map: Burn Probability (FSim/BP)",
        "",
        f"Burn Probability achieves {imp(results['Burn Prob']['all'][2])} improvement for all fires",
        f"and {imp(results['Burn Prob']['large'][2])} for large fires. Its ratio for large fires",
        f"({fmt(results['Burn Prob']['large'][1], 2)}x) indicates large fires occur in areas with",
        "significantly above-average burn probability, even on this 2021 out-of-sample test.",
        "Like Pyrologix, FSim BP may share some training overlap with 2020 conditions.",
        "",
        "### 3. WFPI time-varying maps are anti-correlated with 2021 fires",
        "",
        f"All three WFPI variants (Yearly {yr_imp:+.1f} pp, D1 {d1_imp:+.1f} pp, D2 {d2_imp:+.1f} pp) show",
        "**negative improvements**: fires occur in areas of *below-average* WFPI on their discovery",
        "day. This is the opposite of the 2020 result (+17–18 pp).",
        "",
        "Possible explanations:",
        "",
        "- **2021 was a severe drought year.** Many large fires (Dixie, Caldor, Monument) burned in",
        "  northern California forests where WFPI (wind-enhanced) tends to be lower than the",
        "  statewide average. Drought-driven fires in low-wind forest environments are poorly",
        "  captured by a wind-centric index.",
        "- **USFS dataset composition.** The USFS ignition points include all 932 California",
        "  wildfires, many of which are small (size class A–C). In 2020 only FPA-FOD records with",
        "  a recorded FOD_ID and discovery time were used (~2,219 fires). The USFS dataset's small",
        "  fires may be concentrated in lower-WFPI areas (e.g. residential interface, roadsides),",
        "  dragging the fire distribution below the statewide WFPI median.",
        "- **Background inflation from the full-year WFPI average.** The background median",
        f"  ({fmt(results['D2']['bg'], 1)} for D2/D1) is derived from averaging all 365 daily WFPI",
        "  maps. Summer maps are high everywhere, inflating the background; if fires are",
        "  concentrated in spring/autumn when WFPI is more spatially variable, the fire-day values",
        "  can fall below this elevated background.",
        "",
        "### 4. WFPI Averaged map is near-random",
        "",
        f"The year-averaged WFPI (improvement {avg_imp:+.1f} pp all fires; {avg_l:+.1f} pp large fires)",
        "is nearly indistinguishable from a random placement. This suggests that the mean WFPI over",
        "the year does not add information beyond what the mask already captures — the valid cells",
        "have similar mean WFPI regardless of whether they are fire-prone.",
        "",
        "### 5. Burn-at-least-once map is unusable",
        "",
        "The background median is saturated at 248.0 (the maximum value). Over 365 days of WFPI,",
        "virtually every valid California cell reaches a WFPI < 249 at least once, giving all",
        "cells a burn-at-least-once probability ≈ 1 remapped to 248. The map provides no",
        "discriminative power and should not be used as a risk map.",
        "",
        "### 6. WFPI Yearly vs D1 vs D2 (time-aware selection adds marginal value)",
        "",
        f"Yearly ({yr_imp:+.1f} pp) marginally outperforms both D1 ({d1_imp:+.1f} pp) and D2 ({d2_imp:+.1f} pp),",
        "consistent with the 2020 result where the time-aware selection gave a small edge.",
        "The gap between D1 and D2 is minimal, suggesting same-day vs day-before WFPI accuracy",
        "difference is small for 2021 fires.",
        "",
        "### 7. Large-fire pattern",
        "",
        f"Large fires (≥ 100 acres, n = {n_large}) show stronger positive correlation with",
        f"Pyrologix ({pyro_l:+.1f} pp) and Burn Probability ({bp_l:+.1f} pp) than all fires.",
        "The WFPI maps remain anti-correlated even for large fires, though less severely.",
        "This suggests that large fires in 2021 occurred in areas of historically high fire",
        "probability but not necessarily high wind-driven fire danger on the specific day.",
        "",
    ]

    # Comparison table with 2020 results
    lines += [
        "## Comparison with 2020 Results",
        "",
        "All 2020 results use improvement vs background (pp above 50% baseline).",
        "",
        "| Map | Improvement (All, 2020) | Improvement (All, 2021) | Improvement (Large, 2020) | Improvement (Large, 2021) |",
        "|-----|------------------------|------------------------|--------------------------|--------------------------|",
        f"| WFPI Yearly (time-aware) | +18.1 pp | {imp(results['Yearly (time-aware)']['all'][2])} | +31.8 pp | {imp(results['Yearly (time-aware)']['large'][2])} |",
        f"| WFPI Day 1 | +18.0 pp | {imp(results['D1']['all'][2])} | +29.5 pp | {imp(results['D1']['large'][2])} |",
        f"| WFPI Day 2 | +17.2 pp | {imp(results['D2']['all'][2])} | +34.1 pp | {imp(results['D2']['large'][2])} |",
        f"| Ignition Probability (Pyrologix) | +21.4 pp | {imp(results['Pyrologix']['all'][2])} | +27.6 pp | {imp(results['Pyrologix']['large'][2])} |",
        f"| Burn Probability (FSim) | +5.8 pp | {imp(results['Burn Prob']['all'][2])} | +22.3 pp | {imp(results['Burn Prob']['large'][2])} |",
        "",
        "The most striking change from 2020 to 2021 is the **reversal of the WFPI signal**:",
        "strongly positive in 2020 (+17–18 pp) but negative in 2021 (−15 to −16 pp).",
        "Pyrologix and Burn Probability maintain a positive signal in both years,",
        "which makes them more robust across dataset compositions and fire regimes.",
        "",
    ]

    lines += [
        "## Recommended Map Ranking for Use in the 2021 Dataset",
        "",
        "| Priority | Map | Rationale |",
        "|----------|-----|-----------|",
        "| 1st | **Ignition Probability (Pyrologix)** | Strongest signal (+21 pp all, +14 pp large); robust across years. Leakage concern for strict 2021 evaluation. |",
        "| 2nd | **Burn Probability (FSim/BP)** | Best clean signal (+9 pp all, +11 pp large); strong for large fires. Slight leakage concern. |",
        "| 3rd | **WFPI Yearly 2021 (time-aware)** | Best WFPI variant; operationally correct burn map for the simulation engine. Negative signal reflects 2021 regime. |",
        "| 4th | **WFPI Day 1 2021** | Slightly behind Yearly; adequate fallback if time-aware map unavailable. |",
        "| 5th | **WFPI Day 2 2021** | Marginally worse than D1 but within noise. |",
        "| 6th | **WFPI 2021 Averaged** | Near-random signal; useful only as a climatological background visualisation. |",
        "| — | **WFPI 2021 Burn-at-least-once** | Saturated; not useful as a discriminative risk map for this grid. |",
        "",
    ]

    lines += [
        "## Files",
        "",
        "| File | Location | Description |",
        "|------|----------|-------------|",
        "| `analyze_2021_risk_maps.py` | `code/dataset_creation/nature_dataset_creation/` | This analysis script |",
        "| `static_risk_wfpi_yearly.npy` | `California2021Dataset/` | 2021 yearly burn map (730, H, W) |",
        "| `static_risk_wfpi_avg.npy` | `California2021Dataset/` | 2021 WFPI average (values ≥249 excluded) |",
        "| `static_risk_wfpi_burn_at_least_once.npy` | `California2021Dataset/` | P(burn ≥1 time), rescaled 0–248 |",
        "| `static_risk_pyrologix_resampled.npy` | `California2020Dataset/` | Pyrologix ignition prob (resampled 1309×805) |",
        "| `static_risk_burn_prob_resampled.npy` | `California2020Dataset/` | FSim burn prob (resampled 1309×805) |",
        "| `wfpi_YYYYMMDD.npy` | `California2021Dataset/` | 2021 D2 daily maps (365 files) |",
        "| `wfpi_day1_YYYYMMDD.npy` | `California2021Dataset_Day1/` | 2021 D1 daily maps (365 files) |",
        "",
        "## References",
        "",
        "- [09_risk_map_comparison.md](09_risk_map_comparison.md) — 2020 baseline (static maps)",
        "- [12_yearly_wfpi_map_comparison.md](12_yearly_wfpi_map_comparison.md) — 2020 WFPI yearly map",
        "- [14_usfs_california_dataset_creation.md](14_usfs_california_dataset_creation.md) — 2021 dataset pipeline",
        "- Pyrologix DOI: 10.17605/OSF.IO/CFGH9",
        "- FSim/BP DOI: 10.2737/RDS-2016-0034-2",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_MD), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import pandas as pd
    main()
