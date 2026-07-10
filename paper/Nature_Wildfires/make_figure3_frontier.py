#!/usr/bin/env python3
"""Generate Figure 3 (frontier.png) for the Nature Wildfires manuscript.

Single-panel figure showing two curves for the TOP routing strategy:
  • Overall detection rate (% of benchmark fires detected within 6 h)
  • Within-1h detection rate (% of benchmark fires detected within the first hour, Δt = 0)
Both share the same y-axis (0–100 %) against realized hardware spend.

Output: paper/Nature_Wildfires/Figures/frontier.png

Usage (from repo root):
    python paper/Nature_Wildfires/make_figure3_frontier.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

# Allow importing from paper/final_report/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final_report"))
from generate_final_report import compute_frontier_detection_curves  # noqa: E402

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
import os
OUT_PNG = Path(__file__).resolve().parent / "Figures" / "frontier.png"
# Optional override (e.g. to export a vector PDF): FIG_OUT=/path/frontier.pdf
OUT = Path(os.environ.get("FIG_OUT", str(OUT_PNG)))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE = "#2166AC"       # "Detected within 6 h" curve
ORANGE = "#D6604D"     # "Detected within 1 h" curve
# 1 h series dash (must match plot + legend proxy)
DASH_1H = (0, (6, 3))

# Axis / tick text 1.3× prior (14/12 pt); legend kept at 12 via explicit `legend` call
_AX_LBL = 14 * 1.3
_TCK_LBL = 12 * 1.3
LEGEND_FONTSIZE = 12 * 1.1  # 13.2

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": _AX_LBL,
    "axes.titlesize": 14,  # unused (no suptitle); keep prior size
    "xtick.labelsize": _TCK_LBL,
    "ytick.labelsize": _TCK_LBL,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": LEGEND_FONTSIZE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = compute_frontier_detection_curves()

x = np.array(data["spent_millions"])          # [0, 20, 50, 100, 172.6]
top6h = np.array(data["top_pct"])             # overall detection %
w1h = np.array(data["top_within1h_pct"])      # within-1h detection %
w1h_lo = np.array(data["top_within1h_ci_lo"])
w1h_hi = np.array(data["top_within1h_ci_hi"])

# Also get Wilson CI for the overall detection rate
from generate_final_report import ROUTING_CSV_SPECS, load_rows_multi, routing_block, placement_table  # noqa: E402

placements = placement_table()
disc_map = {int(p["budget"]): int(p["discoverable"]) for p in placements}

top6h_lo: list[float] = [float("nan")]
top6h_hi: list[float] = [float("nan")]
for budget, strat, _mode, paths in ROUTING_CSV_SPECS:
    if strat != "TOPGrowing":
        continue
    rows = load_rows_multi(paths)
    st = routing_block(rows, disc_map[budget])
    ci = st["ci_p"]
    if ci is not None:
        top6h_lo.append(100.0 * ci[0])
        top6h_hi.append(100.0 * ci[1])
    else:
        top6h_lo.append(float("nan"))
        top6h_hi.append(float("nan"))

# Anchor CI bands at the origin so shading extends from (0, 0)
top6h_lo[0] = 0.0
top6h_hi[0] = 0.0
top6h_lo_arr = np.array(top6h_lo)
top6h_hi_arr = np.array(top6h_hi)

w1h_lo[0] = 0.0
w1h_hi[0] = 0.0

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Subtle CIs (~10% opacity) — lines drive the read
CI_ALPHA = 0.10

# ── Overall detection (TOP, within 6 h) ─────────────────────────────────────
ax.fill_between(
    x, top6h_lo_arr, top6h_hi_arr,
    color=BLUE, alpha=CI_ALPHA, linewidth=0, zorder=1,
)
ax.plot(
    x, top6h, color=BLUE, linestyle="-", linewidth=2.2,
    marker="o", markersize=7, zorder=3,
)

# ── Within-1h detection (TOP) — dashed (6,3) + markers ────────────────────
ax.fill_between(
    x, w1h_lo, w1h_hi,
    color=ORANGE, alpha=CI_ALPHA, linewidth=0, zorder=1,
)
ax.plot(
    x, w1h, color=ORANGE, linewidth=2.2,
    linestyle=DASH_1H,
    marker="s", markersize=7,
    zorder=3,
)

# ── Data labels ──────────────────────────────────────────────────────────────
def label_curve(xv, yv, color, yoffset=4, fmt="{:.0f}%", per_point_offsets=None):
    for i, (xi, yi) in enumerate(zip(xv, yv)):
        if math.isnan(yi) or xi == 0:
            continue
        off = per_point_offsets[i] if per_point_offsets is not None else yoffset
        va = "top" if off < 0 else "bottom"
        ax.annotate(
            fmt.format(yi),
            xy=(xi, yi),
            xytext=(0, off),
            textcoords="offset points",
            ha="center", va=va,
            color=color, fontsize=14, fontweight="normal",
        )

# 75M is index 3; put its label below the curve to avoid legend overlap
blue_offsets = [15, 15, 15, 15, 15, 15]
label_curve(x, top6h, BLUE, per_point_offsets=blue_offsets)
label_curve(x, w1h, ORANGE, yoffset=-30)

# ── Axes — minimal spines, ticks out (not into the data), labels tight ──────
ax.set_xlim(-5, x[-1] + 12)
ax.set_ylim(0, 108)
ax.set_ylabel("Share of benchmark fires (%)", labelpad=4)
ax.set_xlabel(r"Budget spent (M$)", labelpad=4)

# x-tick at each data point
ax.set_xticks(np.concatenate([[0], x[1:]]))
ax.set_xticklabels(["0"] + [f"{round(xi)}" for xi in x[1:]])

ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

# Horizontal rules only, ~6% black (Tufte / Nature-style, high data-ink)
GRID_ALPHA = 0.06
ax.grid(
    True,
    which="major",
    axis="y",
    color="k",
    alpha=GRID_ALPHA,
    linewidth=0.6,
    linestyle="-",
    zorder=0,
)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Ticks point outward, never into the data area; no top/right spine
for _side in ("left", "bottom"):
    ax.spines[_side].set_linewidth(0.8)
    ax.spines[_side].set_color("0.15")
ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=3.2,
    width=0.6,
    top=False,
    right=False,
    labelsize=_TCK_LBL,
)
ax.tick_params(axis="y", which="minor", length=0)  # no minor tick lines
ax.tick_params(axis="x", which="minor", length=0)

# ── Legend (explicit Line2D so dashed 1 h pattern renders in the key) ───────
_legend_handles = [
    Line2D(
        [0], [0], color=BLUE, linestyle="-", linewidth=2.2,
        marker="o", markersize=7, markerfacecolor=BLUE, markeredgecolor=BLUE,
        label="Detected within 6 h (TOP)",
    ),
    Line2D(
        [0], [0], color=ORANGE, linestyle=DASH_1H, linewidth=2.2,
        marker="s", markersize=7, markerfacecolor=ORANGE, markeredgecolor=ORANGE,
        label="Detected within 1 h (TOP)",
    ),
]
ax.legend(
    handles=_legend_handles,
    loc="lower right",
    frameon=False,
    handlelength=3.6,
    fontsize=LEGEND_FONTSIZE,
)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print(f"Saved: {OUT}")
