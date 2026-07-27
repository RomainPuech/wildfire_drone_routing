"""Dataset-level selection, preprocessing, and summary stages."""

from __future__ import annotations

import csv
from datetime import date, datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform

from .matching import FireRecord, ScenarioRecord, match_scenarios
from .preprocess import jpg_sequence_to_array, preprocess_layout

LOGGER = logging.getLogger(__name__)
VEGETATION_RASTER = Path("Vegetation_Map") / "Existing_Vegetation_Cover.tif"


def _mask_directory(layout: Path) -> Path:
    for name in ("Satellite_Images_Mask", "Satellite_Image_Mask"):
        candidate = layout / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No JPG mask directory in {layout}")


def _weather_date(layout: Path, scenario_id: str) -> datetime.date | None:
    path = layout / "Weather_Data" / f"{scenario_id}.txt"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        LOGGER.warning("Empty weather file: %s", path)
        return None
    fields = lines[0].split()
    try:
        return datetime.strptime(" ".join(fields[:3]), "%Y %m %d").date()
    except (ValueError, IndexError):
        LOGGER.warning("Could not parse first weather date in %s", path)
        return None


def index_scenarios(layout: Path, *, date_aware: bool) -> list[ScenarioRecord]:
    records: list[ScenarioRecord] = []
    for folder in sorted(path for path in _mask_directory(layout).iterdir() if path.is_dir()):
        first = jpg_sequence_to_array(folder, first_frame_only=True)
        cells = tuple((int(row), int(col)) for row, col in np.argwhere(first >= 0.5))
        if cells:
            records.append(
                ScenarioRecord(
                    scenario_id=folder.name,
                    ignition_cells=cells,
                    start_date=_weather_date(layout, folder.name) if date_aware else None,
                )
            )
    return records


def select_scenarios(
    dataset_root: Path,
    fires_path: Path,
    footprints_path: Path,
    output_manifest: Path,
    *,
    date_aware: bool = False,
    max_distance: int | None = None,
    max_day_difference: int = 1,
    max_unmatched_fraction: float = 0.2,
    seed: int = 0,
    fire_source: str | None = None,
    minimum_discovery_date: date | None = None,
    restrict_to_weather_range: bool = False,
    apply_layout_filter: bool = True,
) -> tuple[int, int]:
    """Spatially join fires, match scenarios, and write auditable CSVs."""
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("select requires geopandas (install environment.yml)") from error

    fires = gpd.read_parquet(fires_path) if fires_path.suffix == ".parquet" else gpd.read_file(fires_path)
    layouts = gpd.read_file(footprints_path)
    fires = fires.to_crs("EPSG:4326")
    layouts = layouts.to_crs("EPSG:4326")
    if fire_source is not None:
        if "source" not in fires:
            raise ValueError("--fire-source requires a 'source' column from merge-fires")
        fires = fires[fires["source"] == fire_source]
    if minimum_discovery_date is not None:
        discovered = pd.to_datetime(fires["discovery_date"], utc=True, errors="coerce")
        fires = fires[discovered.dt.date >= minimum_discovery_date]
    joined = gpd.sjoin(fires, layouts, how="inner", predicate="within")
    distance_limit = max_distance if max_distance is not None else (10 if date_aware else 5)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for layout_id in sorted(layouts["identifier"]):
        layout = Path(dataset_root) / layout_id
        layout_fires = joined[joined["identifier"] == layout_id]
        if layout_fires.empty:
            continue
        scenarios = index_scenarios(layout, date_aware=date_aware)
        if date_aware and restrict_to_weather_range:
            scenario_dates = [item.start_date for item in scenarios if item.start_date is not None]
            if not scenario_dates:
                LOGGER.warning("Skipping %s: no parseable scenario weather dates", layout_id)
                continue
            first_date, last_date = min(scenario_dates), max(scenario_dates)
            discovered = pd.to_datetime(
                layout_fires["discovery_date"], utc=True, errors="coerce"
            )
            layout_fires = layout_fires[
                (discovered.dt.date >= first_date)
                & (discovered.dt.date <= last_date)
            ]
            if layout_fires.empty:
                continue
        footprint = layouts[layouts["identifier"] == layout_id].iloc[0]
        grid_path = Path(footprint["grid_raster"])
        if not grid_path.is_absolute():
            grid_path = Path(dataset_root) / grid_path
        # Older footprint manifests may contain stale absolute paths.
        if not grid_path.is_file():
            grid_path = layout / VEGETATION_RASTER
        with rasterio.open(grid_path) as grid:
            fire_records: list[FireRecord] = []
            for _, item in layout_fires.iterrows():
                xs, ys = transform("EPSG:4326", grid.crs, [item.geometry.x], [item.geometry.y])
                row, col = rowcol(grid.transform, xs[0], ys[0])
                discovery = pd.to_datetime(item["discovery_date"], utc=True, errors="coerce")
                fire_records.append(
                    FireRecord(
                        fire_id=str(item["fire_id"]),
                        row=int(row),
                        col=int(col),
                        discovery_date=None if pd.isna(discovery) else discovery.date(),
                    )
                )
        matches, unmatched = match_scenarios(
            fire_records,
            scenarios,
            max_distance=distance_limit,
            date_aware=date_aware,
            max_day_difference=max_day_difference,
            seed=seed,
        )
        fraction = len(unmatched) / max(len(fire_records), 1)
        threshold_pass = fraction <= max_unmatched_fraction
        accepted = threshold_pass if apply_layout_filter else True
        reports.append(
            {
                "layout_id": layout_id,
                "historical_fire_count": len(fire_records),
                "matched_count": len(matches),
                "unmatched_count": len(unmatched),
                "unmatched_fraction": fraction,
                "threshold_pass": threshold_pass,
                "accepted": accepted,
            }
        )
        by_fire = {match.fire_id: match for match in matches}
        for fire in sorted(fire_records, key=lambda value: value.fire_id):
            match = by_fire.get(fire.fire_id)
            rows.append(
                {
                    "layout_id": layout_id,
                    "fire_id": fire.fire_id,
                    "discovery_date": fire.discovery_date.isoformat() if fire.discovery_date else "",
                    "scenario_id": match.scenario_id if match else "",
                    "matched": match is not None,
                    "layout_accepted": accepted,
                    "fire_row": fire.row,
                    "fire_col": fire.col,
                    "ignition_row": match.ignition_row if match else "",
                    "ignition_col": match.ignition_col if match else "",
                    "chebyshev_cells": match.chebyshev_cells if match else "",
                    "manhattan_cells": match.manhattan_cells if match else "",
                    "day_difference": match.day_difference if match and match.day_difference is not None else "",
                    "mode": "date-aware" if date_aware else "space-only",
                    "seed": seed,
                }
            )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "layout_id", "fire_id", "discovery_date", "scenario_id", "matched",
        "layout_accepted", "fire_row", "fire_col", "ignition_row", "ignition_col",
        "chebyshev_cells", "manhattan_cells", "day_difference", "mode", "seed",
    ]
    with output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_manifest.with_name(f"{output_manifest.stem}_layouts.csv")
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "layout_id", "historical_fire_count", "matched_count",
                "unmatched_count", "unmatched_fraction", "threshold_pass",
                "accepted",
            ],
        )
        writer.writeheader()
        writer.writerows(reports)
    return sum(bool(row["matched"]) and bool(row["layout_accepted"]) for row in rows), len(reports)


def preprocess_from_manifest(
    dataset_root: Path,
    manifest: Path,
    output_root: Path,
    *,
    config_file: Path | None = None,
    threshold: float = 0.5,
) -> int:
    config = json.loads(config_file.read_text()) if config_file else {}
    selected: dict[str, set[str]] = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["matched"].lower() == "true"
                and row["layout_accepted"].lower() == "true"
                and row["scenario_id"]
            ):
                selected.setdefault(row["layout_id"], set()).add(row["scenario_id"])
    count = 0
    for layout_id, scenario_ids in sorted(selected.items()):
        count += preprocess_layout(
            dataset_root / layout_id,
            scenario_ids,
            output_root,
            offsets=config,
            threshold=threshold,
        )
    return count


def fire_radius_cells(final_frame: np.ndarray) -> float:
    points = np.argwhere(np.asarray(final_frame) >= 0.5)
    if not len(points):
        return 0.0
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).max())


def scenario_characteristics(
    scenario: np.ndarray,
    *,
    cell_size_km: float = 0.03,
    big_radius_km: float = 20.0,
    fast_timestep: int = 10,
    fast_fraction: float = 0.5,
) -> dict[str, Any]:
    final_count = int((scenario[-1] >= 0.5).sum())
    time_index = min(fast_timestep, len(scenario) - 1)
    early_count = int((scenario[time_index] >= 0.5).sum())
    radius_km = fire_radius_cells(scenario[-1]) * cell_size_km
    return {
        "final_burned_cells": final_count,
        "radius_km": radius_km,
        "big_fire": radius_km >= big_radius_km,
        "fast_fire": early_count >= fast_fraction * final_count,
    }


def build_summary(
    dataset_root: Path,
    manifest: Path,
    output: Path,
    *,
    cell_size_km: float = 0.03,
    big_radius_km: float = 20.0,
    fast_timestep: int = 10,
    fast_fraction: float = 0.5,
    historical_manifest: Path | None = None,
) -> int:
    rows: list[dict[str, Any]] = []
    historical_scenarios: set[tuple[str, str]] = set()
    if historical_manifest is not None:
        with historical_manifest.open(newline="", encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                if item.get("matched", "").lower() == "true" and item.get("scenario_id"):
                    historical_scenarios.add((item["layout_id"], item["scenario_id"]))
    with manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    for item in manifest_rows:
        if item["matched"].lower() != "true" or item["layout_accepted"].lower() != "true":
            continue
        layout = dataset_root / item["layout_id"]
        scenario_id = item["scenario_id"]
        npy = layout / "scenarii" / f"{scenario_id}.npy"
        scenario = np.load(npy) if npy.is_file() else jpg_sequence_to_array(_mask_directory(layout) / scenario_id)
        characteristics = scenario_characteristics(
            scenario,
            cell_size_km=cell_size_km,
            big_radius_km=big_radius_km,
            fast_timestep=fast_timestep,
            fast_fraction=fast_fraction,
        )
        start_date = _weather_date(layout, scenario_id)
        fire_date = datetime.fromisoformat(item["discovery_date"]).date() if item["discovery_date"] else None
        rows.append(
            {
                "layout_id": item["layout_id"],
                "scenario_id": scenario_id,
                "layout_number": item["layout_id"].split("_")[0],
                "scenario_number": scenario_id.split("_")[-1],
                "fire_id": item["fire_id"],
                "season_number": ((start_date.month % 12) // 3 + 1) if start_date else "",
                "seasonal_match": bool(start_date and fire_date and start_date.strftime("%m-%d") == fire_date.strftime("%m-%d")),
                "historical_match": (item["layout_id"], scenario_id)
                in historical_scenarios,
                **characteristics,
                "small_fire": not characteristics["big_fire"],
                "slow_fire": not characteristics["fast_fire"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            "layout_id", "scenario_id", "layout_number", "scenario_number",
            "fire_id", "season_number", "seasonal_match", "historical_match",
            "final_burned_cells", "radius_km", "big_fire", "fast_fire",
            "small_fire", "slow_fire",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
