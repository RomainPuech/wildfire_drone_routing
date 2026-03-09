#!/usr/bin/env python3
"""
Create a video of California WFPI burn map + daily fire ignition points.

For each of the 732 half-days of 2020 (before/after 10 am = 2 frames/day):
  - WFPI risk map masked to California
  - Blue dots for fires ignited on that calendar day

Output: california_wfpi_fires_2020.mp4

Usage:
    python make_wfpi_video.py [--fps N] [--dpi N]
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / "California2020Dataset"
YEARLY_MAP   = DATASET_DIR / "static_risk_wfpi_yearly.npy"
MASK_PATH    = DATASET_DIR / "mask.npy"
CONFIG_PATH  = DATASET_DIR / "config_california_2020.json"
SCENARII_DIR = DATASET_DIR / "scenarii"


def load_fires_by_date(config: dict) -> dict:
    """Return {date_str: [(row, col), ...]} for all dated scenarios."""
    fires = {}
    for key, date_str in config.items():
        if not key.startswith("date_"):
            continue
        scenario_name = key[len("date_"):]
        p = SCENARII_DIR / f"{scenario_name}_scenario1.npy"
        if not p.exists():
            continue
        pt = np.load(str(p))
        row, col = int(pt[0]), int(pt[1])
        fires.setdefault(date_str, []).append((row, col))
    return fires


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=int, default=12,
                        help="Frames per second (default 12 → ~1 min video)")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Output DPI (default 100)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading WFPI yearly map (memory-mapped)…", flush=True)
    yearly = np.load(str(YEARLY_MAP), mmap_mode="r")   # (732, H, W)
    T, H, W = yearly.shape
    print(f"  shape: {T} frames × {H}×{W}", flush=True)

    mask      = np.load(str(MASK_PATH)).astype(bool)   # (H, W)
    anti_mask = ~mask

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Building fire lookup…", flush=True)
    fires_by_date = load_fires_by_date(config)
    total_fires   = sum(len(v) for v in fires_by_date.values())
    print(f"  {total_fires} fires across {len(fires_by_date)} unique dates", flush=True)

    # ── Figure setup ───────────────────────────────────────────────────────────
    aspect  = W / H
    fig_h   = 9.0
    fig_w   = fig_h * aspect + 2.0          # extra width for colourbar + title
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colourmap: YlOrRd, NaN cells rendered white
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")

    # Initial display frame
    disp = np.where(mask, yearly[0].astype(float), np.nan)
    im   = ax.imshow(disp, cmap=cmap, origin="upper",
                     interpolation="nearest",
                     vmin=0, vmax=255,
                     extent=[0, W, H, 0])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="WFPI (0–255)")

    # Fire scatter (starts empty)
    sc = ax.scatter([], [], s=12, color="royalblue",
                    edgecolors="navy", linewidths=0.4,
                    alpha=0.85, zorder=5)

    title_obj = ax.set_title("", fontsize=12, fontweight="bold")
    ax.set_xlabel("Column (1 km / cell)")
    ax.set_ylabel("Row (1 km / cell)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()

    # ── Month boundaries (day-of-year, 0-based) ────────────────────────────────
    # 2020 is a leap year
    month_starts_doy = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names      = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── Video writer ───────────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "california_wfpi_fires_2020.mp4"
    writer   = FFMpegWriter(
        fps=args.fps,
        metadata={"title": "California WFPI 2020", "artist": "wildfire_drone_routing"},
        codec="h264",
        extra_args=["-pix_fmt", "yuv420p"],      # broad player compatibility
    )

    print(f"Rendering {T} frames at {args.fps} fps → {T/args.fps:.0f}s video…", flush=True)

    start_date = date(2020, 1, 1)

    with writer.saving(fig, str(out_path), dpi=args.dpi):
        for t in range(T):
            doy  = t // 2          # 0-based day of year
            half = t % 2           # 0 = before 10 am, 1 = after 10 am

            current_date = start_date + timedelta(days=doy)
            date_str     = current_date.strftime("%Y%m%d")
            half_label   = "before 10 am" if half == 0 else "after 10 am"

            # ── Update WFPI map ────────────────────────────────────────────────
            frame = yearly[t].astype(float)
            frame[anti_mask] = np.nan
            im.set_data(frame)

            # ── Update fire dots ───────────────────────────────────────────────
            day_fires = fires_by_date.get(date_str, [])
            if day_fires:
                f_rows = [r for r, c in day_fires]
                f_cols = [c for r, c in day_fires]
                sc.set_offsets(np.c_[f_cols, f_rows])
            else:
                sc.set_offsets(np.empty((0, 2)))

            # ── Title ──────────────────────────────────────────────────────────
            n_fires = len(day_fires)
            title_obj.set_text(
                f"California WFPI — {current_date.strftime('%B %d, %Y')}  ({half_label})\n"
                f"{n_fires} fire{'s' if n_fires != 1 else ''} ignited this day"
            )

            writer.grab_frame()

            if t % 100 == 0 or t == T - 1:
                pct = 100 * (t + 1) / T
                print(f"  [{pct:5.1f}%] frame {t+1}/{T}  {current_date}", flush=True)

    print(f"\nSaved → {out_path}", flush=True)
    print(f"Duration: {T/args.fps:.1f}s  |  Frames: {T}  |  FPS: {args.fps}", flush=True)


if __name__ == "__main__":
    main()
