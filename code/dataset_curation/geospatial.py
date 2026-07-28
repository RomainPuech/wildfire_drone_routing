"""Geospatial stages for layout footprints, fire records, and risk rasters."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_geom

LOGGER = logging.getLogger(__name__)
VEGETATION_RASTER = Path("Vegetation_Map") / "Existing_Vegetation_Cover.tif"


def layout_rasters(dataset_root: Path) -> Iterable[tuple[str, Path]]:
    for layout in sorted(path for path in Path(dataset_root).iterdir() if path.is_dir()):
        raster = layout / VEGETATION_RASTER
        if raster.is_file():
            yield layout.name, raster


def extract_layout_footprints(
    dataset_root: Path,
    output: Path,
    *,
    expected_resolution: float = 30.0,
    minimum_width: int = 500,
    include_small: bool = False,
) -> int:
    """Write a GeoJSON layout-grid inventory without copying source rasters."""
    features: list[dict[str, Any]] = []
    for identifier, path in layout_rasters(dataset_root):
        with rasterio.open(path) as source:
            xres, yres = abs(source.transform.a), abs(source.transform.e)
            if not np.isclose([xres, yres], expected_resolution).all():
                raise ValueError(f"{path}: expected {expected_resolution} m cells, got {xres}×{yres}")
            if source.crs is None:
                raise ValueError(f"{path}: CRS is missing")
            if source.width < minimum_width and not include_small:
                LOGGER.info("Excluding %s: width %d < %d", identifier, source.width, minimum_width)
                continue
            left, bottom, right, top = source.bounds
            native = {
                "type": "Polygon",
                "coordinates": [[
                    [left, top], [right, top], [right, bottom], [left, bottom], [left, top]
                ]],
            }
            geometry = transform_geom(source.crs, "EPSG:4326", native, precision=12)
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "identifier": identifier,
                        "height": source.height,
                        "width": source.width,
                        "resolution_m": expected_resolution,
                        # Keep inventories portable across machines. Consumers
                        # resolve this path relative to dataset_root.
                        "grid_raster": str(path.relative_to(dataset_root)),
                    },
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(features)


def merge_fire_records(old_gpkg: Path, newer_data: Path, output: Path) -> int:
    """Normalize and stable-deduplicate FPA-FOD and newer USFS records."""
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("merge-fires requires geopandas (install environment.yml)") from error

    old = gpd.read_file(old_gpkg, layer="Fires")
    newer = gpd.read_file(newer_data)
    old = old.to_crs("EPSG:4326")
    newer = newer.to_crs("EPSG:4326")

    old_norm = gpd.GeoDataFrame(
        {
            "source_fire_id": old["FOD_ID"].astype("string"),
            "discovery_date": pd.to_datetime(old["DISCOVERY_DATE"], errors="coerce", utc=True),
            "longitude": old.geometry.x,
            "latitude": old.geometry.y,
            "source": "FPA_FOD_2022",
            "geometry": old.geometry,
        },
        crs="EPSG:4326",
    )
    newer_id = "OBJECTID" if "OBJECTID" in newer else newer.index.astype(str)
    newer_norm = gpd.GeoDataFrame(
        {
            "source_fire_id": newer[newer_id].astype("string") if isinstance(newer_id, str) else newer_id,
            "discovery_date": pd.to_datetime(newer["DISCOVERYDATETIME"], errors="coerce", utc=True),
            "longitude": newer.geometry.x,
            "latitude": newer.geometry.y,
            "source": "USFS_newer",
            "geometry": newer.geometry,
        },
        crs="EPSG:4326",
    )
    combined = gpd.GeoDataFrame(
        pd.concat([old_norm, newer_norm], ignore_index=True), crs="EPSG:4326"
    )
    combined["fire_id"] = combined["source"] + ":" + combined["source_fire_id"].astype(str)
    combined = combined.dropna(subset=["discovery_date", "longitude", "latitude"])
    combined["_date_key"] = combined["discovery_date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    combined = combined.sort_values(
        ["source", "source_fire_id"], kind="stable"
    ).drop_duplicates(["latitude", "longitude", "_date_key"], keep="first")
    combined = combined.drop(columns="_date_key")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        combined.to_parquet(output, index=False)
    else:
        combined.to_file(output, layer="fires", driver="GPKG")
    return len(combined)


def align_risk_raster(
    continental_raster: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    risk_name: str,
    resampling: str | None = None,
    fill_nodata: float = 0.0,
) -> int:
    """Reproject a continental BP/WHP raster onto every exact layout grid.

    Values remain in the source product's native units. BP is continuous and
    defaults to bilinear resampling; WHP is an index and defaults to nearest
    neighbor. Missing source coverage is represented by ``fill_nodata``.
    """
    if risk_name not in {"bp", "whp"}:
        raise ValueError("risk_name must be 'bp' or 'whp'")
    method_name = resampling or ("bilinear" if risk_name == "bp" else "nearest")
    method = Resampling[method_name]
    output_stem = "static_risk_bp2024" if risk_name == "bp" else "static_risk_whp"
    count = 0
    with rasterio.open(continental_raster) as source:
        if source.crs is None:
            raise ValueError(f"{continental_raster}: CRS is missing")
        for identifier, template_path in layout_rasters(dataset_root):
            with rasterio.open(template_path) as template:
                destination = np.full(
                    (template.height, template.width),
                    fill_nodata,
                    dtype=np.float32,
                )
                reproject(
                    source=rasterio.band(source, 1),
                    destination=destination,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=template.transform,
                    dst_crs=template.crs,
                    dst_nodata=fill_nodata,
                    resampling=method,
                )
                profile = template.profile.copy()
                profile.update(
                    count=1,
                    dtype="float32",
                    nodata=fill_nodata,
                    compress="deflate",
                )
            target = output_root / identifier
            target.mkdir(parents=True, exist_ok=True)
            tif_path = target / f"{output_stem}.tif"
            with rasterio.open(tif_path, "w", **profile) as sink:
                sink.write(destination, 1)
            np.save(target / f"{output_stem}.npy", destination[np.newaxis])
            count += 1
    return count
