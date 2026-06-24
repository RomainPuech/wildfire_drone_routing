#!/usr/bin/env python3
"""
Generate Nature Figure 6 composite for ALERTCalifornia benchmark coverage.

Output: one 1x2 composite with shared Pyrologix colorbar, shared legend, and
per-panel detected/undetected callouts.

Run from repo root:
  conda run -n wf python paper/figure6/generate_alertcalifornia_composite_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pyproj import Transformer
from rasterio.transform import xy as raster_xy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

import benchmark_alertcalifornia as _bm
import matplotlib.font_manager as fm
import plot_alertcalifornia_coverage as _pac

_LM_OTF = (
    "/Library/TeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    str(Path.home() / "texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
)

DEFAULT_OUT = (
    REPO_ROOT
    / "paper"
    / "Nature_Wildfires"
    / "Figures"
    / "alertcalifornia_coverage_composite.png"
)
RADII_KM = (10.0, 32.0)

# Styling aligned with Figures 4/5.
FS_SUBTITLE = 40.0
FS_CB_LABEL = 25.0
FS_CB_TICKS = 23.0
FS_LEGEND = 46.0
FS_ANNO = 29.9


def _publication_rc() -> dict:
    for path in _LM_OTF:
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


def _detected_by_radius(
    fire_xy_proj: np.ndarray,
    cam_xy_proj: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    radius_m = float(radius_km) * 1000.0
    # (N_fires, N_cams, 2) -> min distance by fire
    d = fire_xy_proj[:, None, :] - cam_xy_proj[None, :, :]
    md = np.sqrt(np.sum(d * d, axis=2)).min(axis=1)
    return md <= radius_m


def render(out_path: Path, dataset_root: Path) -> None:
    cropped_t, wfpi_crs = _bm.get_wfpi_georef()
    cam_xy_proj, cam_lons, cam_lats = _bm.load_unique_camera_positions(wfpi_crs)
    fires = _bm.load_benchmark_fires(dataset_root, all_fires=True)

    to_wgs = Transformer.from_crs(wfpi_crs, "EPSG:4326", always_xy=True)
    fire_xy_proj = np.zeros((len(fires), 2), dtype=float)
    fire_lons = np.zeros(len(fires), dtype=float)
    fire_lats = np.zeros(len(fires), dtype=float)
    for i, (_name, row, col) in enumerate(fires):
        x, y = raster_xy(cropped_t, row, col)
        fire_xy_proj[i, 0] = x
        fire_xy_proj[i, 1] = y
        lon, lat = to_wgs.transform(x, y)
        fire_lons[i] = lon
        fire_lats[i] = lat

    pyro_geo, extent = _pac._load_pyrologix_wgs84(dataset_root, cropped_t, wfpi_crs)
    if pyro_geo is None or extent is None:
        raise RuntimeError("Pyrologix raster missing; cannot render Figure 6 composite.")
    bmap01 = np.asarray(pyro_geo, dtype=float) / 255.0

    ca_boundary = _pac._load_ca_boundary(wfpi_crs)
    circles = {
        r: _pac._build_camera_circles_wgs84(cam_lons, cam_lats, wfpi_crs, r * 1000.0)
        for r in RADII_KM
    }

    # Crop to valid raster footprint to enlarge California in each panel.
    valid = np.isfinite(bmap01)
    rr, cc = np.where(valid)
    if rr.size and cc.size:
        pad = 6
        x0 = max(0, int(cc.min()) - pad)
        x1 = min(bmap01.shape[1], int(cc.max()) + 1 + pad)
        y0 = max(0, int(rr.min()) - pad)
        y1 = min(bmap01.shape[0], int(rr.max()) + 1 + pad)
        lon0 = extent[0] + (extent[1] - extent[0]) * (x0 / bmap01.shape[1])
        lon1 = extent[0] + (extent[1] - extent[0]) * (x1 / bmap01.shape[1])
        lat_top = extent[3] + (extent[2] - extent[3]) * (y0 / bmap01.shape[0])
        lat_bot = extent[3] + (extent[2] - extent[3]) * (y1 / bmap01.shape[0])
    else:
        lon0, lon1, lat_bot, lat_top = extent[0], extent[1], extent[2], extent[3]

    with plt.rc_context(_publication_rc()):
        fig = plt.figure(figsize=(26.0, 14.0), facecolor="white", layout="none")
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=(1.0, 1.0, 0.04),
            wspace=-0.02,
            left=0.018,
            right=0.986,
            top=0.955,
            bottom=0.19,
        )
        ax20 = fig.add_subplot(gs[0, 0])
        ax32 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        axes = [ax20, ax32]

        ims = []
        for radius_km, ax in zip(RADII_KM, axes):
            det = _detected_by_radius(fire_xy_proj, cam_xy_proj, radius_km)
            n_det = int(det.sum())
            n_und = int((~det).sum())
            pct = 100.0 * n_det / float(len(det))

            im = ax.imshow(
                bmap01,
                extent=extent,
                origin="upper",
                cmap="YlOrRd",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                zorder=1,
            )
            ims.append(im)
            circles[radius_km].plot(
                ax=ax,
                facecolor=(0.20, 0.45, 0.72, 0.28),
                edgecolor=(0.20, 0.45, 0.72, 0.62),
                linewidth=0.35,
                zorder=3,
            )
            ca_boundary.plot(ax=ax, color="none", edgecolor="#444444", linewidth=1.0, zorder=4)

            ax.scatter(
                fire_lons[~det],
                fire_lats[~det],
                marker=".",
                s=144,
                c="gray",
                alpha=0.75,
                zorder=6,
            )
            ax.scatter(
                fire_lons[det],
                fire_lats[det],
                marker=".",
                s=160,
                c="black",
                alpha=0.9,
                zorder=7,
            )

            # Top-right callout.
            tb = ax.text(
                0.73,
                0.90,
                f"Detected: {n_det}\nUndetected: {n_und}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=FS_ANNO,
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

            subtitle = f"r = {int(radius_km)} km,\n{pct:.0f}% detected"
            ax.annotate(
                subtitle,
                xy=(0.5, 0.0),
                xycoords="axes fraction",
                xytext=(0, -42),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=FS_SUBTITLE,
                linespacing=1.2,
                clip_on=False,
            )

            ax.set_xlim(lon0, lon1)
            ax.set_ylim(lat_bot, lat_top)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", which="both", length=0, width=0, labelbottom=False, labelleft=False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        cb = fig.colorbar(ims[0], cax=cax)
        cb.set_label("Ignition probability (0–1)", fontsize=FS_CB_LABEL, labelpad=20)
        cb.ax.tick_params(labelsize=FS_CB_TICKS)
        cb.outline.set_visible(False)

        circle_patch = mpatches.Patch(
            facecolor=(0.20, 0.45, 0.72, 0.28),
            edgecolor=(0.20, 0.45, 0.72, 0.62),
            linewidth=0.8,
            label="Camera coverage",
        )
        h_det = Line2D(
            [],
            [],
            marker=".",
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markersize=10,
            label="Ignitions: detected",
        )
        h_und = Line2D(
            [],
            [],
            marker=".",
            linestyle="None",
            color="gray",
            markerfacecolor="gray",
            alpha=0.75,
            markersize=10,
            label="Ignitions: undetected",
        )
        leg = fig.legend(
            handles=[circle_patch, h_det, h_und],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.065),
            ncol=3,
            frameon=True,
            fontsize=FS_LEGEND,
            markerscale=2.0,
            columnspacing=1.35,
            handletextpad=0.4,
            handlelength=1.55,
            handleheight=0.92,
            alignment="center",
        )
        leg_frame = leg.get_frame()
        leg_frame.set_facecolor("white")
        leg_frame.set_edgecolor("0.78")
        leg_frame.set_linewidth(1.0)
        leg_frame.set_alpha(0.96)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(out_path),
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
        plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    a = ap.parse_args()
    ds = _bm.resolve_dataset_root(str(a.dataset_dir) if a.dataset_dir else None)
    render(a.out.resolve(), ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
