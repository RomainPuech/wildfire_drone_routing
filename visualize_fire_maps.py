#!/usr/bin/env python3
"""
Generate fire-on-risk-map comparison plots (no sensor/charging station overlay).

Outputs:
  california_fires_2019_on_wfpi.png   — 2019 fires on avg WFPI (Day 2) map
  california_fires_pyrologix.png      — 2020 fires on Pyrologix ignition prob map
  california_fires_2020_large.png     — Large (>=100 ac) 2020 fires on avg WFPI map

Usage:
    python visualize_fire_maps.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Dataset directories ──────────────────────────────────────────────────────
WFPI_DIR        = PROJECT_ROOT / "California2020Dataset"
IGNPROB_2020    = PROJECT_ROOT / "California2020Dataset_IgnitionProb"
IGNPROB_2019    = PROJECT_ROOT / "California2019Dataset_IgnitionProb"
LARGE_FIRES_DIR = PROJECT_ROOT / "California2020Dataset_LargeFires"


def load_fire_points(scenarii_dir: Path):
    """Return (rows, cols) arrays from all scenario .npy files in a directory."""
    rows, cols = [], []
    for f in sorted(scenarii_dir.glob("*.npy")):
        pt = np.load(str(f))
        rows.append(int(pt[0]))
        cols.append(int(pt[1]))
    return np.array(rows), np.array(cols)


def make_plot(background, mask, fire_rows, fire_cols,
              title, cbar_label, out_path,
              vmin=None, vmax=None, cmap="YlOrRd",
              fire_label="Fire ignition points"):
    """Render background map + fire scatter and save."""
    bmap = background.astype(float).copy()
    bmap[mask == 0] = np.nan
    H, W = bmap.shape

    aspect = W / H
    fig_h  = 12
    fig, ax = plt.subplots(figsize=(fig_h * aspect + 1.8, fig_h))
    im = ax.imshow(bmap, cmap=cmap, origin="upper", interpolation="nearest",
                   vmin=vmin, vmax=vmax, extent=[0, W, H, 0])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=cbar_label)

    n_fires = len(fire_rows)
    ax.scatter(fire_cols, fire_rows, s=10, color="black",
               alpha=0.5, zorder=2, linewidths=0)

    leg = [mpatches.Patch(facecolor="black", alpha=0.7,
                          label=f"{fire_label} (n={n_fires})")]
    ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.85)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}", flush=True)


def main():
    # ── Load WFPI avg map & mask ──────────────────────────────────────────────
    print("Loading WFPI avg map and mask ...", flush=True)
    wfpi_avg  = np.load(str(WFPI_DIR / "static_risk_wfpi_avg.npy"))[0]   # (H, W)
    wfpi_mask = np.load(str(WFPI_DIR / "mask.npy"))
    H_wfpi, W_wfpi = wfpi_mask.shape

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 1: 2019 fires on WFPI avg map
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1] 2019 fires on avg WFPI map", flush=True)
    fire_rows_pyro, fire_cols_pyro = load_fire_points(IGNPROB_2019 / "scenarii")
    pyro_mask = np.load(str(IGNPROB_2019 / "mask.npy"))
    H_pyro, W_pyro = pyro_mask.shape

    # Scale Pyrologix coords → WFPI coords
    fire_rows_wfpi = (fire_rows_pyro * (H_wfpi / H_pyro)).astype(int)
    fire_cols_wfpi = (fire_cols_pyro * (W_wfpi / W_pyro)).astype(int)
    fire_rows_wfpi = np.clip(fire_rows_wfpi, 0, H_wfpi - 1)
    fire_cols_wfpi = np.clip(fire_cols_wfpi, 0, W_wfpi - 1)
    print(f"  {len(fire_rows_pyro)} fires (Pyrologix grid → scaled to WFPI grid)")

    make_plot(
        wfpi_avg, wfpi_mask, fire_rows_wfpi, fire_cols_wfpi,
        title=f"Avg yearly WFPI (2020 Day 2) + 2019 California fires\n"
              f"{len(fire_rows_pyro)} fires  ·  grid {H_wfpi}×{W_wfpi}  (1 km / cell)",
        cbar_label="WFPI (0–255)", vmin=0, vmax=255,
        out_path=PROJECT_ROOT / "california_fires_2019_on_wfpi.png",
        fire_label="2019 fire ignition points",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 2: 2020 fires on Pyrologix ignition probability map
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[2] 2020 fires on Pyrologix ignition probability map", flush=True)
    ign_prob = np.load(str(IGNPROB_2020 / "static_risk_ignition_prob.npy"))[0]
    ign_mask = np.load(str(IGNPROB_2020 / "mask.npy"))
    fire_rows_ip20, fire_cols_ip20 = load_fire_points(IGNPROB_2020 / "scenarii")
    print(f"  {len(fire_rows_ip20)} fires (native Pyrologix grid)")

    make_plot(
        ign_prob, ign_mask, fire_rows_ip20, fire_cols_ip20,
        title=f"Pyrologix ignition probability + 2020 California fires\n"
              f"{len(fire_rows_ip20)} fires  ·  grid {ign_prob.shape[0]}×{ign_prob.shape[1]}",
        cbar_label="Ignition probability",
        vmin=0, vmax=np.nanpercentile(ign_prob[ign_mask > 0], 99),
        cmap="YlOrRd",
        out_path=PROJECT_ROOT / "california_fires_pyrologix.png",
        fire_label="2020 fire ignition points",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 3: Large 2020 fires on WFPI avg map
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[3] Large (≥100 ac) 2020 fires on avg WFPI map", flush=True)
    fire_rows_lg, fire_cols_lg = load_fire_points(LARGE_FIRES_DIR / "scenarii")
    print(f"  {len(fire_rows_lg)} large fires (WFPI grid)")

    make_plot(
        wfpi_avg, wfpi_mask, fire_rows_lg, fire_cols_lg,
        title=f"Avg yearly WFPI (2020 Day 2) + large 2020 fires (≥100 acres)\n"
              f"{len(fire_rows_lg)} fires  ·  grid {H_wfpi}×{W_wfpi}  (1 km / cell)",
        cbar_label="WFPI (0–255)", vmin=0, vmax=255,
        out_path=PROJECT_ROOT / "california_fires_2020_large.png",
        fire_label="Large fire ignition points (≥100 ac)",
    )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
