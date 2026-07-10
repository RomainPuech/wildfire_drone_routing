#!/usr/bin/env python3
"""Generate Figure 5-bis: sensor & drone counts vs. ground-sensor unit cost.

Two-panel figure (20M budget | 50M budget).  Each panel plots (left y-axis):
  • Number of ground sensors  (blue solid, circles)
  • Number of drones          (orange dashed, squares)
And on a right y-axis:
  • % of CA fires reachable, pooled over 2021–2024 (n = 3,693)  (#1B7837 green, dotted, squares)

X-axis: ground-sensor unit cost (k USD).

Data source: paper/breakeven_report/breakeven_sensor_cost_export/placement_logs/*.json
Output:      paper/Nature_Wildfires/Figures/breakeven_costsensitivity_lines.png

Run from repo root:
    python paper/figure5bis/make_figure5bis_breakeven_lines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = (
    REPO_ROOT
    / "paper"
    / "breakeven_report"
    / "breakeven_sensor_cost_export"
    / "placement_logs"
)
import os
OUT_PNG = REPO_ROOT / "paper" / "Nature_Wildfires" / "Figures" / "breakeven_costsensitivity_lines.png"
# Optional override (e.g. to export a vector PDF): FIG_OUT=/path/breakeven.pdf
OUT = Path(os.environ.get("FIG_OUT", str(OUT_PNG)))

# ---------------------------------------------------------------------------
# Reachability helpers — imported from visualize_sensor_placement_2021.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))
from visualize_sensor_placement_2021 import classify_fires_opt, compute_clusters  # noqa: E402

COVERAGE_W = 5  # same constant as in visualize_sensor_placement_2021.py

# Reachability is reported over the full California fire population pooled across all
# benchmark years (2021–2024, n = 3,693) — consistent with Table 1, Figure 3, and
# Figure 4. The placement geometry is California-wide and year-independent.
FIRE_YEARS = (2021, 2022, 2023, 2024)


def _load_all_years_fire_cells() -> tuple[np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    for year in FIRE_YEARS:
        scen_dir = REPO_ROOT / f"California{year}Dataset" / "scenarii"
        cfg_path = REPO_ROOT / f"California{year}Dataset" / f"config_california_{year}.json"
        if not scen_dir.is_dir() or not cfg_path.is_file():
            print(f"  WARNING: skipping year {year} (missing scenarii/ or config)", flush=True)
            continue
        config = json.loads(cfg_path.read_text())
        valid = {
            k[len("offset_"):]
            for k in config
            if k.startswith("offset_")
            and f"date_{k[len('offset_'):]}" in config
            and f"time_{k[len('offset_'):]}" in config
        }
        for fp in sorted(scen_dir.glob("*.npy")):
            if fp.stem.replace("_scenario1", "") not in valid:
                continue
            pt = np.load(str(fp))
            rows.append(int(pt[0]))
            cols.append(int(pt[1]))
    return (
        np.asarray(rows, dtype=np.int64) // COVERAGE_W,
        np.asarray(cols, dtype=np.int64) // COVERAGE_W,
    )


FIRE_ROWS_OPT, FIRE_COLS_OPT = _load_all_years_fire_cells()
N_FIRES = len(FIRE_ROWS_OPT)

# ---------------------------------------------------------------------------
# Style — matches make_figure3_frontier.py exactly
# ---------------------------------------------------------------------------
BLUE   = "#2166AC"   # ground sensors
ORANGE = "#D6604D"   # drones
GREEN  = "#1B7837"   # % reachable fires
DASH_DRONES = (0, (6, 3))

_AX_LBL       = 14 * 1.3
_TCK_LBL      = 12 * 1.3
LEGEND_FONTSIZE = 12 * 1.1

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": _AX_LBL,
    "axes.titlesize": 14,
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

GRID_ALPHA = 0.06

# ---------------------------------------------------------------------------
# Reachability computation
# ---------------------------------------------------------------------------

def pct_reachable_from_json(p_json: Path) -> float:
    """Return % of CA fires (pooled 2021–2024) reachable by a given placement JSON."""
    d = json.loads(p_json.read_text())
    charging_locs = [tuple(s) for s in d["charging_station_locations"]]
    drones_per_st = list(d["drones_per_charging_station"])
    ground_locs   = [tuple(g) for g in d.get("ground_sensor_locations", [])]

    ground_opt_set = set(ground_locs)

    if charging_locs:
        clusters = compute_clusters(charging_locs, drones_per_st)
    else:
        clusters = []

    detected_gs, discoverable, _ = classify_fires_opt(
        FIRE_ROWS_OPT, FIRE_COLS_OPT, clusters, ground_opt_set
    )
    n_reach = len(detected_gs) + len(discoverable)
    return 100.0 * n_reach / N_FIRES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_breakeven_data() -> dict[float, list[dict]]:
    """
    Read all canonical (non-filt80, non-baseline) JSONs and return a dict
    mapping budget_millions -> sorted list of records.

    Each record: {cost_k, n_sensors, n_stations, n_drones, pct_reachable}
    """
    records: dict[float, list[dict]] = {}
    for p in sorted(JSON_DIR.glob("*.json")):
        name = p.stem
        # Skip _filt80 duplicates and the baseline warm-start cache
        if name.endswith("_filt80"):
            continue
        if name.endswith("_mean"):
            continue
        with open(p) as f:
            d = json.load(f)
        dc = d.get("device_counts", {})
        budget = float(d["budget_millions"])
        cost_k = float(d["cost_sensor_millions"]) * 1000  # MUSD → k USD
        rec = {
            "cost_k":        cost_k,
            "n_sensors":     int(dc.get("n_ground_sensors", 0)),
            "n_stations":    int(dc.get("n_charging_stations", 0)),
            "n_drones":      int(dc.get("n_drones", 0)),
            "pct_reachable": pct_reachable_from_json(p),
        }
        records.setdefault(budget, []).append(rec)

    # Sort each budget's list by sensor cost
    for budget in records:
        records[budget].sort(key=lambda r: r["cost_k"])
    return records


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _style_ax(ax):
    """Apply frontier-style spine / tick / grid settings."""
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("0.15")
    ax.tick_params(
        axis="both", which="major",
        direction="out", length=3.2, width=0.6,
        top=False, right=False,
        labelsize=_TCK_LBL,
    )
    ax.tick_params(axis="y", which="minor", length=0)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(
        True, which="major", axis="y",
        color="k", alpha=GRID_ALPHA, linewidth=0.6, linestyle="-", zorder=0,
    )
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def draw_panel(
    ax,
    rows: list[dict],
    budget_label: str,
    *,
    x_max: float | None = None,
) -> mpl.axes.Axes:
    """Draw one budget panel; returns the third (right) axis for legend assembly."""
    # Crop to informative range: drop points beyond x_max (if given), then
    # keep up to and including the first zero-sensor point.
    if x_max is not None:
        rows = [r for r in rows if r["cost_k"] <= x_max]

    first_zero_idx = next(
        (i for i, r in enumerate(rows) if r["n_sensors"] == 0), None
    )

    if first_zero_idx is not None:
        rows = rows[: first_zero_idx + 1]
        # Pad with constant values up to x_max.
        if x_max is not None and rows[-1]["cost_k"] < x_max:
            last = rows[-1]
            c = last["cost_k"] + 1.0
            while c <= x_max + 1e-9:
                rows = rows + [{
                    "cost_k":        c,
                    "n_sensors":     0,
                    "n_stations":    last["n_stations"],
                    "n_drones":      last["n_drones"],
                    "pct_reachable": last["pct_reachable"],
                }]
                c += 1.0

    cost = np.array([r["cost_k"]        for r in rows])
    sens = np.array([r["n_sensors"]     for r in rows])
    dron = np.array([r["n_drones"]      for r in rows])
    pcts = np.array([r["pct_reachable"] for r in rows])

    ax.plot(cost, sens, color=BLUE,   linestyle="-",         linewidth=2.2,
            marker="o", markersize=7, zorder=3)
    ax.plot(cost, dron, color=ORANGE, linestyle=DASH_DRONES, linewidth=2.2,
            marker="s", markersize=7, zorder=3)

    _style_ax(ax)

    x_min_int = int(cost[0])
    x_max_int = int(round(cost[-1]))
    all_ticks = list(range(x_min_int, x_max_int + 1))
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(
        [str(t) if t % 2 == 0 else "" for t in all_ticks],
        rotation=45, ha="right",
    )
    ax.set_xlim(cost[0] - 0.5, cost[-1] + 0.5)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Ground-sensor unit cost (k USD)", labelpad=4)
    ax.set_ylabel("Number of devices", labelpad=4)
    ax.set_title(f"Total budget: ${budget_label}", fontsize=14, pad=8)

    # Mixed-regime shading
    mixed = [r for r in rows if r["n_sensors"] > 0 and r["n_stations"] > 0]
    if mixed:
        ax.axvspan(mixed[0]["cost_k"] - 0.3, mixed[-1]["cost_k"] + 0.3,
                   color="#DDDDDD", alpha=0.35, zorder=0)

    # Third axis: % reachable fires (right spine)
    ax3 = ax.twinx()
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_linewidth(0.8)
    ax3.spines["right"].set_color(GREEN)
    ax3.spines["top"].set_visible(False)
    ax3.plot(cost, pcts, color=GREEN, linestyle=":", linewidth=1.6,
             marker="s", markersize=5, zorder=4, label="% reachable fires")
    ax3.set_ylabel("% reachable fires", color=GREEN, labelpad=6, fontsize=_AX_LBL)
    ax3.tick_params(axis="y", labelcolor=GREEN, direction="out",
                    length=3.2, width=0.6, labelsize=_TCK_LBL)
    ax3.set_ylim(0, 100)
    # Share x-limits
    ax3.set_xlim(ax.get_xlim())

    return ax3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading fire data ({N_FIRES} fires)...", flush=True)
    data = load_breakeven_data()

    budgets_present = sorted(data.keys())
    if not budgets_present:
        sys.exit(f"No JSON files found in {JSON_DIR}")

    # Keep the two main budgets in order: 20M and 50M
    target = [20.0, 50.0]
    panels = [(b, data[b]) for b in target if b in data]
    if not panels:
        sys.exit("Could not find 20M or 50M budget data.")

    # Print reachability stats for validation
    for budget, rows in panels:
        label = f"{int(budget)}M"
        # Only use non-padded rows (those that came from real JSONs before padding)
        real_pcts = [r["pct_reachable"] for r in rows
                     if isinstance(r.get("pct_reachable"), float)]
        print(f"  {label}: n_placements={len(rows)}, "
              f"min_pct={min(real_pcts):.2f}%, max_pct={max(real_pcts):.2f}%",
              flush=True)

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7.5 * n_panels, 5.2),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    x_max_map = {20.0: 23.0, 50.0: None}
    ax3_list = []
    for ax, (budget, rows) in zip(axes, panels):
        ax3 = draw_panel(ax, rows, f"{int(budget)}M", x_max=x_max_map.get(budget))
        ax3_list.append(ax3)

    # Shared legend below both panels
    handles = [
        Line2D([0], [0], color=BLUE,   linestyle="-",         linewidth=2.2,
               marker="o", markersize=7, label="Ground sensors"),
        Line2D([0], [0], color=ORANGE, linestyle=DASH_DRONES,  linewidth=2.2,
               marker="s", markersize=7, label="Drones"),
        mpl.patches.Patch(color="#DDDDDD", alpha=0.6, label="Mixed regime"),
        Line2D([0], [0], color=GREEN,  linestyle=":",          linewidth=1.6,
               marker="s", markersize=5, label="% reachable fires"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=3.0,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=int(os.environ.get("FIG_DPI", "300")), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
