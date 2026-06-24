#!/usr/bin/env python3
"""
Plot ALERTCalifornia camera coverage on the California 2021 map.

Overlays on the masked Pyrologix ignition-probability map:
  - ALERTCalifornia camera coverage circles (one per unique camera site, radius = --radius km)
  - The 100-fire benchmark subset (seed=42), coloured by detection status

Usage (from project root):
    python code/plot_alertcalifornia_coverage.py --radius 20
    python code/plot_alertcalifornia_coverage.py --radius 20 --out results/my_plot.png
    python code/plot_alertcalifornia_coverage.py --radius 20 --no-pyrologix

Arguments
---------
--radius FLOAT        Camera detection radius in kilometres (required).
--out PATH            Output PNG path.  Default:
                        results/alertcalifornia_coverage_<radius>km.png
--dataset-dir PATH    Path to California2021Dataset (default: project root auto-detection).
--no-pyrologix        Skip Pyrologix background (faster; plain white background).
--dpi INT             Output DPI (default: 150).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import shared functions from benchmark_alertcalifornia (same directory)
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))
import benchmark_alertcalifornia as _bm  # noqa: E402

PROJECT_ROOT = _bm.PROJECT_ROOT
CAMERAS_JSON = _bm.CAMERAS_JSON
RESULTS_DIR = _bm.RESULTS_DIR

# ---------------------------------------------------------------------------
# Matplotlib setup — paper-style serif font (Latin Modern / Computer Modern)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

_LM_OTF_CANDIDATES = (
    "/Library/TeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
)
for _p in _LM_OTF_CANDIDATES:
    if Path(_p).is_file():
        try:
            import matplotlib.font_manager as _fm
            _fm.fontManager.addfont(_p)
        except Exception:
            pass
        break

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Latin Modern", "Computer Modern Roman",
                       "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)

# Okabe–Ito palette (colour-blind safe)
_C_DETECTED   = "#009E73"   # bluish green
_C_UNDETECTED = "#D55E00"   # vermillion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_camera_circles_wgs84(cam_lons, cam_lats, wfpi_crs, radius_m: float):
    """Return a GeoDataFrame of camera circles in EPSG:4326."""
    import geopandas as gpd
    from shapely.geometry import Point

    cam_pts_wgs = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(cam_lons, cam_lats)],
        crs="EPSG:4326",
    )
    cam_pts_proj = cam_pts_wgs.to_crs(wfpi_crs)
    circles_proj = cam_pts_proj.buffer(radius_m)
    circles_wgs  = gpd.GeoDataFrame(geometry=circles_proj, crs=wfpi_crs).to_crs("EPSG:4326")
    return circles_wgs


def _load_pyrologix_wgs84(dataset_root: Path, cropped_t, wfpi_crs):
    """Return (img_array, extent) for the masked Pyrologix map in WGS84.

    img_array has NaN where the mask is 0.
    extent = [lon_left, lon_right, lat_bottom, lat_top] for imshow.
    Returns (None, None) if data files are missing.
    """
    pyro_path = dataset_root / "static_risk_pyrologix.npy"
    mask_path  = dataset_root / "mask.npy"
    if not pyro_path.is_file() or not mask_path.is_file():
        return None, None

    try:
        import rasterio
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        from rasterio.crs import CRS as RioCRS
    except ImportError:
        return None, None

    pyro_raw = np.load(str(pyro_path))
    mask     = np.load(str(mask_path))
    pyro_2d  = (pyro_raw[0] if pyro_raw.ndim == 3 else pyro_raw).astype(np.float32)
    pyro_masked = np.where(mask == 1, pyro_2d, np.nan)

    H, W = pyro_masked.shape
    dst_crs = RioCRS.from_epsg(4326)
    dst_t, dst_W, dst_H = calculate_default_transform(
        wfpi_crs, dst_crs, W, H,
        left   = cropped_t.c,
        top    = cropped_t.f,
        right  = cropped_t.c + W * cropped_t.a,
        bottom = cropped_t.f + H * cropped_t.e,
    )
    pyro_geo = np.full((dst_H, dst_W), np.nan, dtype=np.float32)
    reproject(
        source      = pyro_masked,
        destination = pyro_geo,
        src_transform = cropped_t,
        src_crs     = wfpi_crs,
        dst_transform = dst_t,
        dst_crs     = dst_crs,
        resampling  = Resampling.bilinear,
        src_nodata  = np.nan,
        dst_nodata  = np.nan,
    )
    extent = [
        dst_t.c,
        dst_t.c + dst_W * dst_t.a,
        dst_t.f + dst_H * dst_t.e,
        dst_t.f,
    ]
    return pyro_geo, extent


def _load_ca_boundary(wfpi_crs):
    """Return the dissolved California boundary GeoDataFrame in EPSG:4326."""
    import geopandas as gpd

    ca_tracts = gpd.read_file(str(_bm.CA_TRACTS_SHP)).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    return ca_tracts.dissolve()


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def make_plot(
    radius_km: float,
    out_png: Path,
    dataset_root: Path,
    use_pyrologix: bool = True,
    dpi: int = 150,
) -> None:
    import rasterio.transform
    from pyproj import Transformer

    radius_m = radius_km * 1_000.0

    print("[1/6] Loading geo-referencing …")
    cropped_t, wfpi_crs = _bm.get_wfpi_georef()

    print("[2/6] Loading camera positions …")
    cam_xy, cam_lons, cam_lats = _bm.load_unique_camera_positions(wfpi_crs)
    print(f"      {len(cam_lons)} unique camera sites.")

    print("[3/6] Loading 100-fire benchmark subset (seed=42) …")
    fires = _bm.load_benchmark_fires(dataset_root)

    # Compute per-fire detection and lat/lon
    to_wgs84 = Transformer.from_crs(wfpi_crs, "EPSG:4326", always_xy=True)
    fire_lons, fire_lats, fire_detected = [], [], []
    n_detected = 0
    for _name, row, col in fires:
        x, y = rasterio.transform.xy(cropped_t, row, col)
        dists = np.sqrt((cam_xy[:, 0] - x) ** 2 + (cam_xy[:, 1] - y) ** 2)
        det = bool(np.min(dists) <= radius_m)
        lon, lat = to_wgs84.transform(x, y)
        fire_lons.append(lon)
        fire_lats.append(lat)
        fire_detected.append(det)
        if det:
            n_detected += 1

    detection_pct = 100.0 * n_detected / len(fires)
    print(f"      Detection rate: {n_detected}/{len(fires)} = {detection_pct:.1f}%")

    print("[4/6] Building camera coverage circles …")
    import geopandas as gpd
    circles_wgs = _build_camera_circles_wgs84(cam_lons, cam_lats, wfpi_crs, radius_m)

    print("[5/6] Loading California boundary …")
    ca_boundary = _load_ca_boundary(wfpi_crs)

    print("[6/6] Rendering plot …")
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_aspect("equal")

    if use_pyrologix:
        pyro_geo, extent = _load_pyrologix_wgs84(dataset_root, cropped_t, wfpi_crs)
        if pyro_geo is not None:
            ax.imshow(
                pyro_geo,
                extent=extent,
                origin="upper",
                cmap="YlOrRd",
                alpha=0.80,
                zorder=1,
                aspect="auto",
                vmin=0,
                vmax=255,
            )
        else:
            print("      [warn] Pyrologix raster not found — using white background.")
            ax.set_facecolor("white")
    else:
        ax.set_facecolor("#f5f5f0")

    # California state outline
    ca_boundary.plot(ax=ax, color="none", edgecolor="#444444", linewidth=0.8, zorder=2)

    # Camera coverage circles
    circles_wgs.plot(
        ax=ax,
        facecolor=(0.20, 0.45, 0.72, 0.18),   # steelblue, very transparent
        edgecolor=(0.20, 0.45, 0.72, 0.50),
        linewidth=0.30,
        zorder=3,
    )

    # Camera site positions (small dots)
    ax.scatter(
        cam_lons, cam_lats,
        c="#1565C0", s=3, alpha=0.6, linewidths=0, zorder=4,
    )

    # Fire ignition points — colour by detection status
    fire_lons = np.array(fire_lons)
    fire_lats = np.array(fire_lats)
    fire_detected = np.array(fire_detected)

    ax.scatter(
        fire_lons[fire_detected], fire_lats[fire_detected],
        c=_C_DETECTED, s=38, alpha=0.95, linewidths=0.6,
        edgecolors="white", zorder=6, label=f"Detected  ({fire_detected.sum()})",
    )
    ax.scatter(
        fire_lons[~fire_detected], fire_lats[~fire_detected],
        c=_C_UNDETECTED, s=38, alpha=0.95, linewidths=0.6,
        edgecolors="white", zorder=6, label=f"Undetected  ({(~fire_detected).sum()})",
    )

    # Legend
    circle_patch = mpatches.Patch(
        facecolor=(0.20, 0.45, 0.72, 0.25),
        edgecolor=(0.20, 0.45, 0.72, 0.7),
        linewidth=0.8,
        label=f"Camera coverage ({radius_km:g} km radius)",
    )
    cam_dot = Line2D(
        [], [], marker="o", color="w", markerfacecolor="#1565C0",
        markersize=5, label=f"Camera site  ({len(cam_lons)})",
    )
    det_patch = mpatches.Patch(facecolor=_C_DETECTED,   label=f"Detected  ({fire_detected.sum()})")
    und_patch = mpatches.Patch(facecolor=_C_UNDETECTED, label=f"Undetected  ({(~fire_detected).sum()})")

    ax.legend(
        handles=[circle_patch, cam_dot, det_patch, und_patch],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.92,
        edgecolor="#cccccc",
        title="ALERTCalifornia baseline",
        title_fontsize=9,
    )

    ax.set_xlim(-124.6, -113.9)
    ax.set_ylim(32.3, 42.1)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(
        f"ALERTCalifornia camera coverage — {radius_km:g} km radius\n"
        f"Detection rate on 100-fire benchmark: {n_detected}/100 = {detection_pct:.1f}%",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ALERTCalifornia camera coverage circles on the California 2021 map."
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Camera detection radius in kilometres.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output PNG path.  Default: "
            "results/alertcalifornia_coverage_<radius>km.png"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to California2021Dataset directory.",
    )
    parser.add_argument(
        "--no-pyrologix",
        action="store_true",
        help="Skip the Pyrologix background raster (faster).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150).",
    )
    args = parser.parse_args()

    if args.radius <= 0:
        sys.exit("Error: --radius must be a positive number.")

    dataset_root = _bm.resolve_dataset_root(args.dataset_dir)

    if args.out:
        out_png = Path(args.out)
    else:
        radius_tag = f"{args.radius:g}km"
        out_png = RESULTS_DIR / f"alertcalifornia_coverage_{radius_tag}.png"

    make_plot(
        radius_km=args.radius,
        out_png=out_png,
        dataset_root=dataset_root,
        use_pyrologix=not args.no_pyrologix,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
