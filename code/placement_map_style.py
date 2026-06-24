"""
Shared drawing and legend style for placement maps (Nature Fig.~4 and Fig.~5).

Kept in sync with ``paper/breakeven_figure/generate_breakeven_cost_sensitivity_figures.py``
visual language: ground = blue 1×1 cells, stations = cyan diamonds, drone zones = green wash.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

# Aligned with ``visualize_sensor_placement_2021.DRONE_REACH_AREA_COLOR``
DRONE_REACH_AREA_COLOR = "#2e7d32"
DRONE_REACH_AREA_LEGEND = "Area reachable by drones"

# Ground + charging (same as breakeven composite)
GROUND_SENSOR_MARKER = "s"
GROUND_SENSOR_FACE = "#1565C0"
CHARGING_FACE = "#00BCD4"
CHARGING_EDGE = "#263238"
# ``scatter(..., s=…)`` is area in points²; 1.5× smaller linear size than older viz
CHARGING_DIAMOND_LINEAR_SCALE = 1.0 / 1.5
CHARGING_DIAMOND_AREA_SCALE = CHARGING_DIAMOND_LINEAR_SCALE**2
DEFAULT_FIG4_DRONE_ZONE_ALPHA = 0.34

# Pooled 5~km map (``add_cluster_unions``): imshow z=1, fill z=3, zone edges z=4, scatters 5+.
# Draw the state outline *above* the green wash and zone LineCollections, *below* ignitions/stations.
ZORDER_CALIFORNIA_STATE_OUTLINE = 4.2

__all__ = [
    "DRONE_REACH_AREA_COLOR",
    "DRONE_REACH_AREA_LEGEND",
    "DEFAULT_FIG4_DRONE_ZONE_ALPHA",
    "GROUND_SENSOR_FACE",
    "CHARGING_FACE",
    "CHARGING_EDGE",
    "ZORDER_CALIFORNIA_STATE_OUTLINE",
    "legend_entries_to_handles_labels",
    "add_ground_sensor_cells",
    "station_scatter_area_pts2",
    "add_charging_stations",
    "draw_california_state_outline",
    "style_ca_outline_figure2",
    "line2d_ground_legend",
    "line2d_charging_legend",
    "line2d_drone_zone_legend",
]


def legend_entries_to_handles_labels(entries: list) -> tuple[list, list, dict | None]:
    """Same contract as ``displays._legend_entries_to_handles_labels``."""
    handles: list = []
    labels: list = []
    need_tuple_handler = False
    for e in entries:
        if (
            isinstance(e, tuple)
            and len(e) == 3
            and isinstance(e[0], Line2D)
            and isinstance(e[1], Line2D)
            and isinstance(e[2], str)
        ):
            handles.append((e[0], e[1]))
            labels.append(e[2])
            need_tuple_handler = True
        else:
            handles.append(e)
            labels.append(e.get_label())
    handler_map = {tuple: HandlerTuple(ndivide=None)} if need_tuple_handler else None
    return handles, labels, handler_map


def add_ground_sensor_cells(ax, ground_locs: list[tuple[int, int]]) -> None:
    """One filled square per pooled (opt) cell; extent matches ``imshow(..., [0,W,H,0])``."""
    if not ground_locs:
        return
    n = len(ground_locs)
    verts = np.empty((n, 4, 2), dtype=float)
    for i, (row, col) in enumerate(ground_locs):
        c, r = float(col), float(row)
        verts[i, 0, :] = (c, r)
        verts[i, 1, :] = (c + 1.0, r)
        verts[i, 2, :] = (c + 1.0, r + 1.0)
        verts[i, 3, :] = (c, r + 1.0)
    pc = PolyCollection(
        verts,
        closed=True,
        facecolors=GROUND_SENSOR_FACE,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        zorder=5,
    )
    ax.add_collection(pc)


def station_scatter_area_pts2(n_stations: int, *, marker_scale: float) -> float:
    """Matplotlib scatter ``s`` (points²) for charging diamonds."""
    ms = float(marker_scale)
    if n_stations >= 85:
        s_s = 10.2
    elif n_stations >= 40:
        s_s = 12.4
    elif n_stations >= 15:
        s_s = 16.0
    else:
        s_s = 20.5
    base = max(8.0, s_s * ms)
    return base * CHARGING_DIAMOND_AREA_SCALE


def add_charging_stations(
    ax,
    charging_locs: list[tuple[int, int]],
    drones_per_station: list[int],
    *,
    marker_scale: float = 0.5,
    show_drone_count_labels: bool = True,
) -> None:
    """Cyan diamonds with dark edges; optional per-station drone count in **charging** cyan."""
    if not charging_locs:
        return
    n_s = len(charging_locs)
    s_s = station_scatter_area_pts2(n_s, marker_scale=marker_scale)
    lw_s = max(0.28, 0.58 * float(marker_scale) * CHARGING_DIAMOND_LINEAR_SCALE)
    ms = float(marker_scale)
    for (r, c), nd in zip(charging_locs, drones_per_station):
        ax.scatter(
            c,
            r,
            marker="D",
            s=s_s,
            color=CHARGING_FACE,
            edgecolors=CHARGING_EDGE,
            linewidths=lw_s,
            zorder=6,
        )
        if show_drone_count_labels:
            off = max(1, int(4 * ms))
            ax.text(
                c + off,
                r - off,
                str(int(nd)),
                color=CHARGING_FACE,
                fontsize=max(6, int(7 * ms * 1.1)),
                fontweight="bold",
                zorder=6,
            )


def draw_california_state_outline(
    ax,
    boundary_paths: list | None,
    *,
    zorder: float | None = None,
) -> None:
    """
    State boundary in pooled (operational) pixel coordinates, matching Fig.~5: #444444, 1.0 pt.

    Must be drawn *after* the YlOrRd + drone-zone layers so the stroke is not hidden under
    the green imshow (which uses z=3) or zone edges (z=4).
    """
    if not boundary_paths:
        return
    z = ZORDER_CALIFORNIA_STATE_OUTLINE if zorder is None else zorder
    for path in boundary_paths:
        if path is None or len(path) < 2:
            continue
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#444444",
            linewidth=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=z,
        )


def style_ca_outline_figure2(ax) -> None:
    """Narrow + dark outline (#444444, 1.0 pt) to match breakeven / publication."""
    old_hex = mcolors.to_hex("#7a7a7a").lower()
    for ln in ax.lines:
        try:
            hx = mcolors.to_hex(ln.get_color()).lower()
        except ValueError:
            continue
        if hx == old_hex:
            ln.set_color("#444444")
            ln.set_linewidth(1.0)


def line2d_ground_legend() -> Line2D:
    return Line2D(
        [],
        [],
        marker=GROUND_SENSOR_MARKER,
        linestyle="None",
        color=GROUND_SENSOR_FACE,
        markerfacecolor=GROUND_SENSOR_FACE,
        markeredgecolor=GROUND_SENSOR_FACE,
        markeredgewidth=0.0,
        markersize=11.0,
        label="Ground sensor",
    )


def line2d_charging_legend() -> Line2D:
    return Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        color=CHARGING_FACE,
        markerfacecolor=CHARGING_FACE,
        markeredgecolor=CHARGING_EDGE,
        markeredgewidth=0.65,
        markersize=10.0,
        label="Charging station",
    )


def line2d_drone_zone_legend(
    fill_alpha: float = DEFAULT_FIG4_DRONE_ZONE_ALPHA,
) -> Line2D:
    colour = DRONE_REACH_AREA_COLOR
    r_, g_, b_ = mcolors.to_rgb(colour)
    fa = float(np.clip(fill_alpha, 0.05, 0.85))
    return Line2D(
        [],
        [],
        marker="s",
        linestyle="None",
        markerfacecolor=(r_, g_, b_, fa),
        markeredgecolor=(r_, g_, b_, 1.0),
        markeredgewidth=1.0,
        markersize=11.0,
        label=DRONE_REACH_AREA_LEGEND,
    )


def add_side_colorbar_single_row(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    *,
    label: str = "Ignition probability (0–1)",
    label_fontsize: float = 16.0,
    tick_fontsize: float = 15.0,
) -> None:
    """
    Vertical colorbar with height tied to **this** map axes only (not the full figure).

    ``fraction`` / ``aspect`` follow the breakeven composite; ``ax=`` pins height to the axes.
    """
    cb = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.046,
        pad=0.02,
        aspect=20.0,
    )
    cb.set_label(label, fontsize=label_fontsize, labelpad=10)
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.outline.set_visible(False)


def save_white_tight(
    out_path, fig: plt.Figure, *, dpi: int = 320, pad: float = 0.12
) -> None:
    fig.patch.set_facecolor("white")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_path),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad,
        facecolor="white",
    )
    plt.close(fig)
