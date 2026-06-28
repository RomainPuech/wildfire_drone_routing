#!/usr/bin/env python3
"""
Regenerate Nature cost-sensitivity placement figure (operational 5 km grid).

Primary output: **one composite** PNG with a 2×3 map grid, **one shared** ignition
probability colorbar, **one shared** symbol legend (sensors, stations, drone-reach
zones; **no** benchmark ignition scatter), and **per-panel** text giving
ground-sensor / station / drone counts from each placement JSON.

Uses the same drawing stack as ``visualize_sensor_placement_2021.py`` (Pyrologix
background on the **operational 5 km pooled** grid, cluster unions, ground sensors
as **exact 1×1 pooled cells**, and charging stations **without** per-station
drone-count labels on this figure).

Publication styling (Figure 2–aligned):
  - serif rc (Latin Modern when available)
  - white figure background, ``bbox_inches="tight"`` with modest padding
  - California outline #444444, 1.0 pt

Run from repo root:
  conda run -n wf python paper/breakeven_figure/generate_breakeven_cost_sensitivity_figures.py

Optional:
  --panels            Also write the six separate PNGs (maps only, no colorbar/legend)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
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

_LM_OTF = (
    "/Library/TeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    str(Path.home() / "texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
)

DEFAULT_JSON_DIR = (
    REPO_ROOT
    / "paper"
    / "breakeven_report"
    / "breakeven_sensor_cost_export"
    / "placement_logs"
)
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "Nature_Wildfires" / "Figures"
COMPOSITE_NAME = "breakeven_costsensitivity_composite.png"


def _placement_json_candidates(primary: str) -> tuple[str, ...]:
    """Alternate filenames for the same 50M / 0.015 MUSD sensor-cost log in ``placement_logs``."""
    if primary.endswith("mean_breakeven_50M_cs0p015.json"):
        root = primary[: -len(".json")]
        return (
            primary,
            f"{root}_filt80.json",
            primary.replace(
                "mean_breakeven_50M_cs0p015.json",
                "mean_thresh50M_m15.json",
            ),
        )
    return (primary,)


def _resolve_placement_json(json_dir: Path, json_name: str) -> Path:
    for name in _placement_json_candidates(json_name):
        p = json_dir / name
        if p.is_file():
            return p
    tried = "\n  ".join(str(json_dir / n) for n in _placement_json_candidates(json_name))
    raise FileNotFoundError(f"Missing placement JSON (tried):\n  {tried}")


def _composite_figsize_inches(r_w: int, r_h: int) -> tuple[float, float]:
    """Large raster for print; aspect matches the operational 2×3 map grid."""
    ar = r_w / r_h
    fw = 33.0
    fh = fw * (2.25) / (3.0 * ar + 0.08)
    return (fw, float(np.clip(fh, 10.0, 22.0)))

# (json filename, subtitle below map)
PANELS: tuple[tuple[str, str], ...] = (
    (
        "sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean_breakeven_cs0p010.json",
        "20M USD total,\n10k USD per sensor",
    ),
    (
        "sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean_breakeven_cs0p011.json",
        "20M USD total,\n11k USD per sensor",
    ),
    (
        "sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean_breakeven_cs0p012.json",
        "20M USD total,\n12k USD per sensor",
    ),
    (
        "sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean_breakeven_50M_cs0p010.json",
        "50M USD total,\n10k USD per sensor",
    ),
    (
        "sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean_breakeven_50M_cs0p015.json",
        "50M USD total,\n15k USD per sensor",
    ),
    (
        "sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean_thresh50M_m23.json",
        "50M USD total,\n23k USD per sensor",
    ),
)

def _register_latin_modern() -> None:
    for path in _LM_OTF:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except (OSError, ValueError, RuntimeError):
            continue
        return


def _publication_rc() -> dict:
    _register_latin_modern()
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


def _add_sensors_stations_opt_no_counts(
    ax,
    ground_locs: list[tuple[int, int]],
    charging_locs: list[tuple[int, int]],
    drones_per_station: list[int],
    legend_items: list | None,
    *,
    marker_scale: float = 0.30,
) -> None:
    """
    Ground sensors: exact **1×1 risk cells**; charging: cyan diamonds (``placement_map_style``).
    """
    if ground_locs:
        pms.add_ground_sensor_cells(ax, ground_locs)
        if legend_items is not None:
            legend_items.append(pms.line2d_ground_legend())
    if charging_locs:
        pms.add_charging_stations(
            ax,
            charging_locs,
            drones_per_station,
            marker_scale=marker_scale,
            show_drone_count_labels=False,
        )
        if legend_items is not None:
            legend_items.append(pms.line2d_charging_legend())


def _load_fires(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    config_path = dataset_dir / "config_california_2021.json"
    with open(config_path) as f:
        config = json.load(f)
    valid_names = {
        key[len("offset_") :]
        for key in config
        if key.startswith("offset_")
        and f"date_{key[len('offset_'):]}" in config
        and f"time_{key[len('offset_'):]}" in config
    }
    scenarii_dir = dataset_dir / "scenarii"
    fire_rows: list[int] = []
    fire_cols: list[int] = []
    for fp in sorted(scenarii_dir.glob("*.npy")):
        name = fp.stem.replace("_scenario1", "")
        if name not in valid_names:
            continue
        pt = np.load(str(fp))
        fire_rows.append(int(pt[0]))
        fire_cols.append(int(pt[1]))
    return np.asarray(fire_rows, dtype=int), np.asarray(fire_cols, dtype=int)


def _draw_map_on_ax(
    ax,
    bmap_opt: np.ndarray,
    boundary_paths_opt: list[np.ndarray],
    rH: int,
    rW: int,
    clusters,
    ground_locs_opt: list[tuple[int, int]],
    charging_locs_opt: list[tuple[int, int]],
    drones_per_station: list[int],
    det_gnd_opt,
    disc_opt,
    ndisc_opt,
    *,
    marker_scale: float = 0.30,
    show_fires: bool = False,
    drone_zone_fill_alpha: float = 0.25,
):
    """Draw pooled Pyrologix + overlays on ``ax``. Returns the ``imshow`` mappable."""
    W, H = rW, rH
    bmap01 = np.asarray(bmap_opt, dtype=float) / 255.0
    im = ax.imshow(
        bmap01,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        extent=[0, W, H, 0],
        aspect="equal",
        zorder=1,
    )
    # Match Figure 4: axes box aspect = data aspect (height / width) so maps fill panels.
    ax.set_box_aspect(rH / rW)
    # Keep explicit x/y limits authoritative (used to crop to CA footprint in the composite).
    ax.set_adjustable("box")

    leg_sink: list = []
    zone_half_opt = viz.DRONE_REACH
    viz.add_cluster_unions(
        ax,
        clusters,
        rH,
        rW,
        zone_half_opt,
        leg_sink,
        fill_alpha=drone_zone_fill_alpha,
    )
    if show_fires:
        viz.add_fire_markers(ax, det_gnd_opt, disc_opt, ndisc_opt, leg_sink)
    _add_sensors_stations_opt_no_counts(
        ax,
        ground_locs_opt,
        charging_locs_opt,
        drones_per_station,
        leg_sink,
        marker_scale=marker_scale,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    pms.draw_california_state_outline(ax, boundary_paths_opt)
    return im


def _shared_legend_entries(*, include_fires: bool = False):
    """Handles/labels for one figure-level legend (no per-panel counts)."""
    h_gs = pms.line2d_ground_legend()
    h_cs = pms.line2d_charging_legend()
    h_zone = pms.line2d_drone_zone_legend(pms.DEFAULT_FIG4_DRONE_ZONE_ALPHA)
    out = [h_gs, h_cs, h_zone]
    if include_fires:
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
            marker="x",
            linestyle="None",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=7,
            label="_nolegend_",
        )
        out.append((h_nd, h_dx, "Ignitions: beyond range; discoverable"))
    return out


def _prepare_layout_data(dataset_dir: Path):
    """Pooled operational (5 km) Pyrologix and mask — **not** 1 km data-scale rasters."""
    pyro_map = np.load(str(dataset_dir / "static_risk_pyrologix.npy"))
    mask = np.load(str(dataset_dir / "mask.npy"))
    H, W = mask.shape
    pyro_2d = pyro_map[0].astype(float)
    rH, rW = H // viz.COVERAGE_W, W // viz.COVERAGE_W

    boundary_paths_opt: list[np.ndarray] = []
    try:
        gr = viz.get_pyrologix_georef()
        if gr is not None:
            _ct, _crs, gh, gw = gr
            if (gh, gw) == (H, W):
                boundary_paths_data = viz.california_boundary_pixel_paths(
                    _ct, _crs, gh, gw
                )
                boundary_paths_opt = viz.data_pixel_paths_to_opt_plot(
                    boundary_paths_data, rH, rW
                )
    except Exception as exc:
        print(f"[breakeven] CA outline skipped: {exc}", file=sys.stderr)

    pyro_masked = pyro_2d * mask.astype(float)
    mask_opt = viz.pool_max_2d(mask.astype(float), viz.COVERAGE_W)
    bmap_opt_raw = viz.pool_mean_2d(pyro_masked, viz.COVERAGE_W)
    bmap_opt = bmap_opt_raw.copy()
    bmap_opt[mask_opt == 0] = np.nan

    fire_rows_data, fire_cols_data = _load_fires(dataset_dir)
    fire_rows_opt = fire_rows_data // viz.COVERAGE_W
    fire_cols_opt = fire_cols_data // viz.COVERAGE_W

    return bmap_opt, boundary_paths_opt, rH, rW, fire_rows_opt, fire_cols_opt


def _load_panel_solution(json_path: Path):
    with open(json_path) as f:
        d = json.load(f)
    ground_locs_opt = [tuple(x) for x in d["ground_sensor_locations"]]
    charging_locs_opt = [tuple(x) for x in d["charging_station_locations"]]
    drones_per_station = d["drones_per_charging_station"]
    clusters = viz.compute_clusters(charging_locs_opt, drones_per_station)
    ground_opt_set = set(ground_locs_opt)
    dc = d.get("device_counts", {})
    return d, ground_locs_opt, charging_locs_opt, drones_per_station, clusters, ground_opt_set, dc


def render_composite_figure(
    json_dir: Path,
    out_path: Path,
    dataset_dir: Path,
) -> None:
    bmap_opt, boundary_paths_opt, rH, rW, _, _ = _prepare_layout_data(dataset_dir)

    with plt.rc_context(_publication_rc()):
        fig = plt.figure(
            figsize=_composite_figsize_inches(rW, rH),
            facecolor="white",
            layout="none",
        )
        # 2×4 grid: maps always in cols 0..2 (top/bottom rows left-aligned), colorbar only
        # in top-right col; bottom-right col left empty as padding.
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=(1.0, 1.0, 1.0, 0.034),
            wspace=-0.04,
            hspace=0.27,
            left=0.01,
            right=0.95,
            top=0.962,
            bottom=0.125,
        )
        ax_00 = fig.add_subplot(gs[0, 0])
        ax_01 = fig.add_subplot(gs[0, 1])
        ax_02 = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])
        ax_10 = fig.add_subplot(gs[1, 0])
        ax_11 = fig.add_subplot(gs[1, 1])
        ax_12 = fig.add_subplot(gs[1, 2])
        ax_pad = fig.add_subplot(gs[1, 3])
        ax_pad.set_axis_off()
        axes = [ax_00, ax_01, ax_02, ax_10, ax_11, ax_12]
        # Crop to valid burnable footprint so California fills each panel.
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

        ims = []
        for (json_name, subtitle), ax in zip(PANELS, axes):
            cache_path = _resolve_placement_json(json_dir, json_name)

            _, ground_locs_opt, charging_locs_opt, drones_per_station, clusters, _, dc = (
                _load_panel_solution(cache_path)
            )
            im = _draw_map_on_ax(
                ax,
                bmap_opt,
                boundary_paths_opt,
                rH,
                rW,
                clusters,
                ground_locs_opt,
                charging_locs_opt,
                drones_per_station,
                [],
                [],
                [],
                marker_scale=1.0,
                show_fires=False,
                drone_zone_fill_alpha=pms.DEFAULT_FIG4_DRONE_ZONE_ALPHA,
            )
            ims.append(im)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)

            # Scenario line below the map (no (a)--(f) panel letters).
            _cap_pt = 10
            _sub_fs = 46.0
            _sub_linespacing = 1.2
            # Half a line of space between map bottom and subtitle (top-aligned text).
            _subtitle_y_pt = _cap_pt + 0.5 * _sub_fs * _sub_linespacing
            subtitle_ann = ax.annotate(
                subtitle,
                xy=(0.5, 0.0),
                xycoords="axes fraction",
                xytext=(0, -_subtitle_y_pt),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=_sub_fs,
                linespacing=_sub_linespacing,
                clip_on=False,
            )
            subtitle_ann.set_clip_on(False)

            ng = int(dc.get("n_ground_sensors", len(ground_locs_opt)))
            ns = int(dc.get("n_charging_stations", len(charging_locs_opt)))
            nd = int(dc.get("n_drones", sum(drones_per_station)))
            stats = f"Sensors: {ng}\nStations: {ns}\nDrones: {nd}"
            tb = ax.text(
                0.73,
                0.90,
                stats,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=29.9,
                linespacing=1.12,
                bbox=dict(
                    boxstyle="round,pad=0.36",
                    facecolor="white",
                    edgecolor="0.78",
                    alpha=0.96,
                ),
                zorder=20,
            )
            tb.set_clip_on(False)

        ref_im = ims[0]
        # Colorbar height ~= top-row map height via dedicated cax.
        cb = fig.colorbar(ref_im, cax=cax)
        cb.set_label("Ignition probability (0–1)", fontsize=25.0, labelpad=20)
        cb.ax.tick_params(labelsize=23.0)
        cb.outline.set_visible(False)

        legend_entries = _shared_legend_entries(include_fires=False)
        handles, labels, handler_map = pms.legend_entries_to_handles_labels(legend_entries)
        leg = fig.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.048),
            ncol=3,
            frameon=True,
            fontsize=46.0,
            markerscale=2.0,
            columnspacing=1.35,
            handletextpad=0.4,
            handlelength=1.55,
            handleheight=0.92,
            alignment="center",
            handler_map=handler_map,
        )
        leg_frame = leg.get_frame()
        leg_frame.set_facecolor("white")
        leg_frame.set_edgecolor("0.78")
        leg_frame.set_linewidth(1.0)
        leg_frame.set_alpha(0.96)

        fig.patch.set_facecolor("white")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(out_path),
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
        plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}", flush=True)


def render_standalone_panel(
    cache_path: Path,
    out_path: Path,
    dataset_dir: Path,
    *,
    show_colorbar: bool,
) -> None:
    """Original one-map PNG (optional)."""
    try:
        from displays import _pyrologix_legend_below_map
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import `displays`. Use: conda run -n wf python ..."
        ) from exc

    _, ground_locs_opt, charging_locs_opt, drones_per_station, clusters, _, _ = (
        _load_panel_solution(cache_path)
    )

    pyro_map = np.load(str(dataset_dir / "static_risk_pyrologix.npy"))
    mask = np.load(str(dataset_dir / "mask.npy"))
    H, W = mask.shape
    pyro_2d = pyro_map[0].astype(float)
    rH, rW = H // viz.COVERAGE_W, W // viz.COVERAGE_W

    boundary_paths_opt: list[np.ndarray] = []
    try:
        gr = viz.get_pyrologix_georef()
        if gr is not None:
            _ct, _crs, gh, gw = gr
            if (gh, gw) == (H, W):
                boundary_paths_data = viz.california_boundary_pixel_paths(
                    _ct, _crs, gh, gw
                )
                boundary_paths_opt = viz.data_pixel_paths_to_opt_plot(
                    boundary_paths_data, rH, rW
                )
    except Exception as exc:
        print(f"[breakeven] CA outline skipped: {exc}", file=sys.stderr)

    pyro_masked = pyro_2d * mask.astype(float)
    mask_opt = viz.pool_max_2d(mask.astype(float), viz.COVERAGE_W)
    bmap_opt_raw = viz.pool_mean_2d(pyro_masked, viz.COVERAGE_W)
    bmap_opt = bmap_opt_raw.copy()
    bmap_opt[mask_opt == 0] = np.nan

    with plt.rc_context(_publication_rc()):
        fig, ax, _im = viz.make_base_axes(
            bmap_opt,
            boundary_paths=boundary_paths_opt or None,
            show_colorbar=show_colorbar,
        )
        _ = _im
        leg: list = []
        viz.add_cluster_unions(ax, clusters, rH, rW, viz.DRONE_REACH, leg)
        _add_sensors_stations_opt_no_counts(
            ax,
            ground_locs_opt,
            charging_locs_opt,
            drones_per_station,
            leg,
            marker_scale=0.32,
        )
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        pms.style_ca_outline_figure2(ax)
        ax.set_xlim(0, rW)
        ax.set_ylim(rH, 0)
        _pyrologix_legend_below_map(fig, ax, leg, legend_fontsize=11, framed=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(out_path),
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor="white",
        )
        plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "California2021Dataset",
    )
    p.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--panels",
        action="store_true",
        help="Also write six separate breakeven_costsensitivity_*.png maps.",
    )
    p.add_argument(
        "--no-colorbar-panels",
        action="store_true",
        help="With --panels: omit per-panel colorbar (default: on).",
    )
    args = p.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    json_dir = args.json_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not dataset_dir.is_dir():
        sys.exit(f"Missing dataset directory: {dataset_dir}")

    for json_name, _ in PANELS:
        try:
            _resolve_placement_json(json_dir, json_name)
        except FileNotFoundError as exc:
            sys.exit(str(exc))

    print(f"Composite → {out_dir / COMPOSITE_NAME}", flush=True)
    render_composite_figure(json_dir, out_dir / COMPOSITE_NAME, dataset_dir)

    if args.panels:
        show_cb = not args.no_colorbar_panels
        panel_names = [p[0] for p in PANELS]
        out_names = [
            "breakeven_costsensitivity_20M_sensors_only.png",
            "breakeven_costsensitivity_20M_mixed.png",
            "breakeven_costsensitivity_20M_drones_only.png",
            "breakeven_costsensitivity_50M_sensors_only.png",
            "breakeven_costsensitivity_50M_mixed.png",
            "breakeven_costsensitivity_50M_drones_only.png",
        ]
        for json_name, out_name in zip(panel_names, out_names):
            render_standalone_panel(
                _resolve_placement_json(json_dir, json_name),
                out_dir / out_name,
                dataset_dir,
                show_colorbar=show_cb,
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
