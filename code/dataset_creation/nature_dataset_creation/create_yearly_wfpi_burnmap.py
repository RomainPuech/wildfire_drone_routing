#!/usr/bin/env python3
"""
Create Yearly WFPI Burn Map for California 2020

Builds a (732, H, W) float32 array that covers every day of 2020 with
2 frames per day:

  Frame layout for calendar day d  (d=1 for Jan 1, d=366 for Dec 31):
    Frame 2*(d-1) + 0  →  "before 10 am"  =  D2 forecast for day d
                           File: California2020Dataset/wfpi_{date_of_(d-1)}.npy
                           Rationale: before the daily 10-am WFPI update we
                           only have the Day-2 forecast issued the previous day.
    Frame 2*(d-1) + 1  →  "after 10 am"   =  D1 forecast for day d
                           File: California2020Dataset_Day1/wfpi_day1_{date_of_d}.npy
                           Rationale: at 10 am the Day-1 (same-day) forecast
                           becomes available and supersedes the Day-2 forecast.

Indexing a scenario at runtime
-------------------------------
Given fire discovery date D, discovery time T (HH:MM), and starting offset of
k half-hour steps:

    from datetime import date, timedelta

    sim_start = datetime(D.year, D.month, D.day, T.hour, T.minute) - timedelta(minutes=30*k)

    def frame_index(dt):
        day_of_year = dt.timetuple().tm_yday   # 1-366
        half = 0 if dt.hour < 10 else 1
        return 2 * (day_of_year - 1) + half

    # Build the per-scenario burn map  (shape: num_steps × H × W)
    frames = [year_map[frame_index(sim_start + timedelta(minutes=30*t))]
              for t in range(num_steps)]
    scenario_burnmap = np.stack(frames)

Output
------
  California2020Dataset/static_risk_wfpi_yearly.npy
  Shape  : (732, 1309, 805)   (732 = 2 × 366 days in leap year 2020)
  Dtype  : float32
  Values : 0 – 255  (same scale as individual WFPI maps)

Run from the project root:
    python code/dataset_creation/nature_dataset_creation/create_yearly_wfpi_burnmap.py
"""

import numpy as np
from pathlib import Path
from datetime import date, timedelta
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."

D2_DATASET  = PROJECT_ROOT / "California2020Dataset"
D1_DATASET  = PROJECT_ROOT / "California2020Dataset_Day1"
OUTPUT_PATH = D2_DATASET / "static_risk_wfpi_yearly.npy"

# ── helpers ────────────────────────────────────────────────────────────────────

def load_frame(path: Path) -> np.ndarray:
    """Load a (1, H, W) WFPI .npy and return the (H, W) slice."""
    arr = np.load(str(path))
    return arr[0] if arr.ndim == 3 else arr


def main():
    # Discover all 366 days of 2020
    all_days: list[date] = []
    d = date(2020, 1, 1)
    while d <= date(2020, 12, 31):
        all_days.append(d)
        d += timedelta(days=1)

    n_days   = len(all_days)   # 366  (2020 is a leap year)
    n_frames = 2 * n_days      # 732

    # Peek at one file to get (H, W)
    sample = load_frame(D2_DATASET / f"wfpi_{all_days[1].strftime('%Y%m%d')}.npy")
    H, W = sample.shape
    print(f"Grid: {H} × {W}")
    print(f"Frames: {n_frames}  ({n_days} days × 2 frames/day)")
    estimated_gb = n_frames * H * W * 4 / 1e9
    print(f"Estimated file size: {estimated_gb:.2f} GB")

    # Allocate output array
    yearly = np.empty((n_frames, H, W), dtype=np.float32)

    missing_d2 = []
    missing_d1 = []

    for i, day in enumerate(tqdm(all_days, desc="Building yearly map")):
        d_str      = day.strftime("%Y%m%d")
        prev_d_str = (day - timedelta(days=1)).strftime("%Y%m%d")

        # ── Frame 0: before 10 am → D2 forecast for `day`
        #    D2 file is named after the ISSUE date (day before the fire date),
        #    so for calendar day `day` we use the file named `prev_d_str`.
        d2_path = D2_DATASET / f"wfpi_{prev_d_str}.npy"
        if not d2_path.exists():
            # Fallback: use the D2 file for `day` itself if previous day missing
            d2_path = D2_DATASET / f"wfpi_{d_str}.npy"
            missing_d2.append(day)
        yearly[2 * i] = load_frame(d2_path)

        # ── Frame 1: after 10 am → D1 forecast for `day`
        #    D1 file is named after the ISSUE date (same as fire date).
        d1_path = D1_DATASET / f"wfpi_day1_{d_str}.npy"
        if not d1_path.exists():
            # Fallback: use D2 for the same day
            d1_path = D2_DATASET / f"wfpi_{d_str}.npy"
            missing_d1.append(day)
        yearly[2 * i + 1] = load_frame(d1_path)

    if missing_d2:
        print(f"\nWARNING: {len(missing_d2)} days used fallback for D2 frame: "
              f"{[d.isoformat() for d in missing_d2]}")
    if missing_d1:
        print(f"WARNING: {len(missing_d1)} days used fallback for D1 frame: "
              f"{[d.isoformat() for d in missing_d1]}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH} ...")
    np.save(str(OUTPUT_PATH), yearly)

    actual_gb = OUTPUT_PATH.stat().st_size / 1e9
    print(f"Saved. File size: {actual_gb:.2f} GB")
    print(f"Shape: {yearly.shape}, dtype: {yearly.dtype}")
    print(f"Value range: [{yearly.min():.1f}, {yearly.max():.1f}], "
          f"mean: {yearly.mean():.2f}")


if __name__ == "__main__":
    main()
