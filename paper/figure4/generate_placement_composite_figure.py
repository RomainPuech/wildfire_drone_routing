#!/usr/bin/env python3
"""
Single composite figure for Nature Fig.~4: 2×2 budget panels, **one** shared colorbar
(top row height), **one** shared legend in **2 rows** (``ncol=2``), and per-map
annotations for discoverable / unreachable benchmark ignitions.

Reuses the same oper­ational drawing stack as ``visualize_sensor_placement_2021.draw_operational_fig4_map_on_ax``.

**California state outline** comes from geospatial data (geopandas, rasterio, etc.). The default system ``python``
usually cannot build the boundary, so the map is missing the state delimiter. **Always** generate with the
project conda env, from repo root::

  conda run -n wf python paper/figure4/generate_placement_composite_figure.py
  conda run -n wf python paper/figure4/generate_placement_composite_figure.py --out paper/Nature_Wildfires/Figures/placement_composite.png
"""
from __future__ import annotations

import argparse
import os
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

import matplotlib.font_manager as fm
import placement_map_style as pms
import visualize_sensor_placement_2021 as viz

_LM = (
    "/Library/TeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    str(Path.home() / "texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
)


def _load_breakeven_module():
    p = (
        REPO_ROOT
        / "paper"
        / "breakeven_figure"
        / "generate_breakeven_cost_sensitivity_figures.py"
    )
    spec = importlib.util.spec_from_file_location("breakeven_csf", p)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(m)
    return m


def _publication_rc() -> dict:
    for path in _LM:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except (OSError, ValueError, RuntimeError):
            continue
        break
    return {
        "font.family": "serif",
        "font.serif": [
            "Latin Modern Roman",
            "Latin Modern",
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }


def _figsize_2x2(r_w: int, r_h: int) -> tuple[float, float]:
    """Choose figsize so map panels fill subplot cells horizontally."""
    # Must stay in sync with ``fig.add_gridspec`` parameters below.
    fw = 18.75 * MAP_LINEAR_SCALE
    inner_w_frac = GS_RIGHT - GS_LEFT
    sum_wr = sum(GS_WIDTH_RATIOS)
    w_map_in = fw * inner_w_frac * (GS_WIDTH_RATIOS[0] / sum_wr)
    target_row_h = w_map_in * (r_h / r_w)

    # GridSpec geometry for 2 rows with relative gap ``GS_HSPACE``:
    # row_h = (fh * (top-bottom)) / (2 + hspace)
    # Solve fh so row_h equals the data-aspect target.
    fh = target_row_h * (2.0 + GS_HSPACE) / (GS_TOP - GS_BOTTOM)
    return (fw, float(np.clip(fh, 8.0, 28.0)))


# Typography aligned with ``generate_breakeven_cost_sensitivity_figures`` (Nature Fig.~5)
FS_SUBTITLE = 15.75 * 1.5 * 2.0  # panel / budget line below each map
FS_CB_LABEL = 25.0
FS_CB_TICKS = 23.0
FS_LEGEND = 46.0
LINESP_SUB = 1.2
LEGEND_Y_ANCHOR = -0.032
# Fire-count box position in axes fraction (upper-right area near CA corner).
ANNO_X_AXES = 0.73
ANNO_Y_AXES = 0.90
ANNO_FONTSIZE = 33.0

# Composite layout tuning.
MAP_LINEAR_SCALE = 1.4
GS_LEFT = 0.02
GS_RIGHT = 0.995
GS_TOP = 0.95
GS_BOTTOM = 0.11
GS_HSPACE = 0.18
GS_WSPACE = -0.06
GS_WIDTH_RATIOS = (1.0, 1.0, 0.02)


def _shared_legend_entries_with_fires() -> list:
    h_gs = pms.line2d_ground_legend()
    h_cs = pms.line2d_charging_legend()
    h_zone = pms.line2d_drone_zone_legend(pms.DEFAULT_FIG4_DRONE_ZONE_ALPHA)
    h_nd = Line2D(
        [],
        [],
        marker=".",
        linestyle="None",
        color="gray",
        markerfacecolor="gray",
        markersize=5,
        alpha=0.75,
        label="_nolegend_",
    )
    h_dx = Line2D(
        [],
        [],
        marker=".",
        linestyle="None",
        color="black",
        markerfacecolor="black",
        markersize=5,
        alpha=0.9,
        label="_nolegend_",
    )
    return [
        h_gs,
        h_cs,
        h_zone,
        (h_nd, h_dx, "Ignitions: unreachable; discoverable"),
    ]


# Benchmark years pooled for fire discoverability counts (placement geometry is the
# same California-wide layout across years; only ignition points differ per year).
FIRE_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024)


def _load_all_years_fires_opt() -> tuple[np.ndarray, np.ndarray]:
    """Pool fire ignition points across all benchmark years, in operational (5 km) coords.

    Mirrors ``generate_breakeven_cost_sensitivity_figures._load_fires`` but iterates
    over every ``California<year>Dataset`` (per-year config name) so discoverable /
    unreachable counts reflect the full 2021–2024 fire population (n = 3,693), matching
    Table 1 and Figure 3.
    """
    import json

    rows: list[int] = []
    cols: list[int] = []
    found_years: list[int] = []
    for year in FIRE_YEARS:
        ds = REPO_ROOT / f"California{year}Dataset"
        cfg_path = ds / f"config_california_{year}.json"
        scen_dir = ds / "scenarii"
        if not cfg_path.is_file() or not scen_dir.is_dir():
            print(
                f"generate_placement_composite: WARNING: skipping year {year} "
                f"(missing {cfg_path.name} or scenarii/)",
                file=sys.stderr,
            )
            continue
        with cfg_path.open() as f:
            config = json.load(f)
        valid = {
            k[len("offset_"):]
            for k in config
            if k.startswith("offset_")
            and f"date_{k[len('offset_'):]}" in config
            and f"time_{k[len('offset_'):]}" in config
        }
        n_before = len(rows)
        for fp in sorted(scen_dir.glob("*.npy")):
            name = fp.stem.replace("_scenario1", "")
            if name not in valid:
                continue
            pt = np.load(str(fp))
            rows.append(int(pt[0]))
            cols.append(int(pt[1]))
        found_years.append(year)
        print(
            f"generate_placement_composite: {year}: {len(rows) - n_before} fires",
            file=sys.stderr,
        )
    if not found_years:
        raise FileNotFoundError(
            "No per-year California<year>Dataset/scenarii found for fire discoverability."
        )
    rows_arr = np.asarray(rows, dtype=int) // viz.COVERAGE_W
    cols_arr = np.asarray(cols, dtype=int) // viz.COVERAGE_W
    print(
        f"generate_placement_composite: pooled {len(rows_arr)} fires across "
        f"{found_years}",
        file=sys.stderr,
    )
    return rows_arr, cols_arr


DEFAULT_JSON_DIR = (
    REPO_ROOT / "paper" / "final_report" / "placement_data" / "logs"
)
DEFAULT_OUT = (
    REPO_ROOT / "paper" / "Nature_Wildfires" / "Figures" / "placement_composite.png"
)
PANELS: list[tuple[str, str]] = [
    (
        "sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean.json",
        r"$20 million",
    ),
    (
        "sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean.json",
        r"$50 million",
    ),
    (
        "sensor_alloc_GaussianBudget100M_StationMaxGreedyUniform_261x161_mean.json",
        r"$100 million",
    ),
    (
        "sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps1_6h_pruned.json",
        r"$500 million",
    ),
]


def render(
    out_path: Path,
    dataset_dir: Path,
    json_log_dir: Path,
) -> None:
    br = _load_breakeven_module()
    bmap_opt, boundary_paths_opt, rH, rW, _fr_opt_2021, _fc_opt_2021 = (
        br._prepare_layout_data(dataset_dir)  # noqa: SLF001
    )
    # Layout (risk map / mask / CA outline) is California-wide and year-independent;
    # only the ignition points change. Pool fires across all benchmark years so
    # discoverable / unreachable counts match Table 1 and Figure 3 (n = 3,693).
    fire_rows_opt, fire_cols_opt = _load_all_years_fires_opt()
    if not boundary_paths_opt:
        print(
            "generate_placement_composite: WARNING: California state outline is empty. "
            "The PNG will have no state border. Regenerate with: "
            "conda run -n wf python paper/figure4/generate_placement_composite_figure.py",
            file=sys.stderr,
        )

    pairs: list[tuple[Path, str]] = [
        (json_log_dir / j, cap) for j, cap in PANELS
    ]
    for p, _s in pairs:
        if not p.is_file():
            raise FileNotFoundError(f"Missing panel JSON: {p}")

    fw, fh = _figsize_2x2(rW, rH)
    with plt.rc_context(_publication_rc()):
        fig = plt.figure(figsize=(fw, fh), facecolor="white", layout="none")
        # 2×3: maps always in columns 0–1 (top and bottom rows share the same horizontal
        # layout). Column 2 is **only** the colorbar in row 0; row 1 col 2 is empty. Using
        # ``fig.colorbar(ax=top_row)`` on a 2×2 grid instead shrinks only the top row and
        # misaligns the two map rows.
        # Extra hspace: fire counts + budget lines sit in the margin *below* each map (not
        # in axes), so the rows need a larger vertical gap to avoid clashing.
        gs = fig.add_gridspec(
            2,
            3,
            # Very narrow colorbar column → maximize map-panel width.
            width_ratios=GS_WIDTH_RATIOS,
            # ``wspace`` is the gap between adjacent subplot columns (maps | maps | colorbar).
            wspace=GS_WSPACE,
            hspace=GS_HSPACE,
            left=GS_LEFT,
            right=GS_RIGHT,
            top=GS_TOP,
            bottom=GS_BOTTOM,
        )
        ax_20 = fig.add_subplot(gs[0, 0])
        ax_50 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        ax_100 = fig.add_subplot(gs[1, 0])
        ax_500 = fig.add_subplot(gs[1, 1])
        ax_pad = fig.add_subplot(gs[1, 2])
        ax_pad.set_axis_off()

        panel_axes = (ax_20, ax_50, ax_100, ax_500)
        # Crop to burnable/risk footprint so California occupies more of each panel.
        valid = np.isfinite(bmap_opt)
        rr, cc = np.where(valid)
        if rr.size and cc.size:
            pad = 4
            x0 = max(0, int(cc.min()) - pad)
            x1 = min(int(rW), int(cc.max()) + 1 + pad)
            y0 = max(0, int(rr.min()) - pad)
            y1 = min(int(rH), int(rr.max()) + 1 + pad)
        else:
            x0, x1, y0, y1 = 0, int(rW), 0, int(rH)

        ims: list = []
        for (json_path, sublabel), ax in zip(pairs, panel_axes):
            d, g_loc, c_loc, dps, clusters, gset, _dc = br._load_panel_solution(  # noqa: SLF001
                json_path
            )
            _ = d
            det, disc, ndisc = viz.classify_fires_opt(
                fire_rows_opt, fire_cols_opt, clusters, gset
            )
            im, _, n_disc, n_unreach, _ = viz.draw_operational_fig4_map_on_ax(
                ax,
                bmap_opt,
                boundary_paths_opt,
                rH,
                rW,
                g_loc,
                c_loc,
                dps,
                clusters,
                det,
                disc,
                ndisc,
                marker_scale=2.0,
                discoverable_marker=".",
                show_drone_count_labels=False,
            )
            ims.append(im)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            # Fire counts: small white box **inside** the axes, bottom-left (as in early composites).
            # Place the count box on the right side of each map and make it larger.
            tb = ax.text(
                ANNO_X_AXES,
                ANNO_Y_AXES,
                f"Discoverable: {n_disc}\nUnreachable: {n_unreach}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNO_FONTSIZE,
                linespacing=1.15,
                color="#212121",
                bbox=dict(
                    boxstyle="round,pad=0.38",
                    facecolor="white",
                    edgecolor="0.78",
                    alpha=0.96,
                ),
                zorder=30,
            )
            tb.set_clip_on(False)
            # Budget line below the map (Fig.5 subtitle size)
            _cap_pt = 10.0
            _sub_linesp = 1.2
            _subtitle_y_pt = _cap_pt + 0.5 * FS_SUBTITLE * _sub_linesp
            ax.annotate(
                sublabel,
                xy=(0.5, 0.0),
                xycoords="axes fraction",
                xytext=(0, -_subtitle_y_pt),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=FS_SUBTITLE,
                linespacing=_sub_linesp,
                clip_on=False,
            )

        ref_im = ims[0]
        cbar = fig.colorbar(ref_im, cax=cax)
        cbar.set_label("Ignition probability (0–1)", fontsize=FS_CB_LABEL, labelpad=20)
        cbar.ax.tick_params(labelsize=FS_CB_TICKS)
        cbar.outline.set_visible(False)

        entries = _shared_legend_entries_with_fires()
        hnd, labs, hmap = pms.legend_entries_to_handles_labels(entries)
        leg = fig.legend(
            handles=hnd,
            labels=labs,
            loc="lower center",
            bbox_to_anchor=(0.5, LEGEND_Y_ANCHOR),
            ncol=2,
            frameon=True,
            fontsize=FS_LEGEND,
            markerscale=2.0,
            columnspacing=1.35,
            handletextpad=0.4,
            handlelength=1.55,
            handleheight=0.92,
            alignment="center",
            labelspacing=0.45,
            borderaxespad=0.2,
            handler_map=hmap,
        )
        # Bordered legend box, matching annotation callout style.
        leg_frame = leg.get_frame()
        leg_frame.set_facecolor("white")
        leg_frame.set_edgecolor("0.78")
        leg_frame.set_linewidth(1.0)
        leg_frame.set_alpha(0.96)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.patch.set_facecolor("white")
        fig.savefig(
            str(out_path),
            dpi=int(os.environ.get("FIG_DPI", "320")),
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "California2021Dataset",
    )
    ap.add_argument(
        "--json-log-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
    )
    a = ap.parse_args()
    ds = a.dataset_dir.resolve()
    if not (ds / "static_risk_pyrologix.npy").is_file():
        alt = REPO_ROOT / "paper" / "final_report" / "placement_data"
        if (alt / "static_risk_pyrologix.npy").is_file():
            ds = alt
        else:
            print("Missing static_risk_pyrologix.npy; pass --dataset-dir", file=sys.stderr)
            return 1
    render(
        a.out.resolve(),
        ds,
        a.json_log_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
