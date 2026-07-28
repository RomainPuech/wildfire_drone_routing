"""Lightweight tests for the reviewer-facing curation package."""

import csv
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin, xy
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dataset_curation.matching import (  # noqa: E402
    FireRecord,
    ScenarioRecord,
    chebyshev_distance,
    match_scenarios,
)
from dataset_curation.preprocess import empirical_burn_map, jpg_sequence_to_array  # noqa: E402
from dataset_curation.workflow import (  # noqa: E402
    build_summary,
    scenario_characteristics,
    select_scenarios,
)
from dataset_curation.geospatial import align_risk_raster, extract_layout_footprints  # noqa: E402


def test_matching_is_order_independent_and_has_no_reuse():
    fires = [
        FireRecord("fire-b", 10, 10),
        FireRecord("fire-a", 10, 10),
    ]
    scenarios = [
        ScenarioRecord("scenario-b", ((10, 10),)),
        ScenarioRecord("scenario-a", ((10, 10),)),
    ]
    first, unmatched = match_scenarios(fires, scenarios, seed=17)
    second, _ = match_scenarios(list(reversed(fires)), list(reversed(scenarios)), seed=17)
    assert first == second
    assert not unmatched
    assert len({match.scenario_id for match in first}) == 2


def test_date_matching_and_exclusion():
    fire = FireRecord("f", 5, 5, date(2023, 8, 4))
    scenarios = [
        ScenarioRecord("excluded", ((5, 5),), date(2023, 8, 4)),
        ScenarioRecord("near-date", ((6, 5),), date(2023, 8, 5)),
        ScenarioRecord("wrong-date", ((5, 5),), date(2023, 8, 7)),
    ]
    matches, unmatched = match_scenarios(
        [fire],
        scenarios,
        date_aware=True,
        max_distance=10,
        max_day_difference=1,
        excluded_scenarios={"excluded"},
    )
    assert not unmatched
    assert matches[0].scenario_id == "near-date"
    assert matches[0].day_difference == 1
    assert chebyshev_distance((5, 5), (6, 5)) == 1


def test_jpg_threshold_is_binary_float32(tmp_path):
    folder = tmp_path / "scenario"
    folder.mkdir()
    Image.fromarray(np.full((8, 8), 128, dtype=np.uint8)).save(folder / "10.jpg")
    Image.fromarray(np.full((8, 8), 127, dtype=np.uint8)).save(folder / "2.jpg")
    array = jpg_sequence_to_array(folder)
    assert array.dtype == np.float32
    assert not array[0].any()
    assert array[1].all()


def test_empirical_map_and_summary_characteristics():
    short = np.zeros((2, 3, 3), dtype=np.float32)
    long = np.zeros((3, 3, 3), dtype=np.float32)
    short[:, 1, 1] = 1
    long[:, 1, 1] = 1
    result = empirical_burn_map([short, long])
    assert result.shape == (3, 3, 3)
    assert result[:, 1, 1].tolist() == [1.0, 1.0, 1.0]
    offset_result = empirical_burn_map([short], offsets=[1])
    assert offset_result[:, 1, 1].tolist() == [0.0, 1.0, 1.0]
    attributes = scenario_characteristics(long, cell_size_km=1, big_radius_km=0)
    assert attributes["big_fire"] is True
    assert attributes["fast_fire"] is True


def test_footprint_and_risk_alignment_use_exact_template_grid(tmp_path):
    dataset = tmp_path / "raw"
    vegetation = dataset / "0001_00001" / "Vegetation_Map"
    vegetation.mkdir(parents=True)
    template_path = vegetation / "Existing_Vegetation_Cover.tif"
    transform_grid = from_origin(500000, 4100000, 30, 30)
    profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 4,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32611",
        "transform": transform_grid,
    }
    with rasterio.open(template_path, "w", **profile) as sink:
        sink.write(np.ones((3, 4), dtype=np.uint8), 1)
    footprints = tmp_path / "layouts.geojson"
    assert extract_layout_footprints(dataset, footprints, include_small=True) == 1
    properties = json.loads(footprints.read_text())["features"][0]["properties"]
    assert properties["width"] == 4
    assert not Path(properties["grid_raster"]).is_absolute()

    risk = tmp_path / "risk.tif"
    risk_profile = {**profile, "dtype": "float32"}
    with rasterio.open(risk, "w", **risk_profile) as sink:
        sink.write(np.arange(12, dtype=np.float32).reshape(3, 4), 1)
    output = tmp_path / "curated"
    assert align_risk_raster(risk, dataset, output, risk_name="whp") == 1
    with rasterio.open(output / "0001_00001" / "static_risk_whp.tif") as aligned:
        assert aligned.shape == (3, 4)
        assert aligned.transform == transform_grid
        assert aligned.crs == rasterio.crs.CRS.from_epsg(32611)
    saved = np.load(output / "0001_00001" / "static_risk_whp.npy")
    assert saved.shape == (1, 3, 4)


def test_summary_uses_separate_historical_manifest(tmp_path):
    dataset = tmp_path / "raw"
    layout = dataset / "0016_03070"
    scenarios = layout / "scenarii"
    weather = layout / "Weather_Data"
    scenarios.mkdir(parents=True)
    weather.mkdir()
    scenario_id = "0016_00001"
    np.save(scenarios / f"{scenario_id}.npy", np.ones((2, 2, 2), dtype=np.float32))
    (weather / f"{scenario_id}.txt").write_text("2020 07 03 0000\n")

    selection = tmp_path / "selection.csv"
    selection.write_text(
        "layout_id,fire_id,discovery_date,scenario_id,matched,layout_accepted\n"
        f"0016_03070,f,2020-07-03,{scenario_id},True,True\n"
    )
    historical = tmp_path / "historical.csv"
    historical.write_text(
        "layout_id,scenario_id,matched\n"
        f"0016_03070,{scenario_id},True\n"
    )
    output = tmp_path / "summary.csv"
    assert build_summary(
        dataset,
        selection,
        output,
        historical_manifest=historical,
        big_radius_km=0,
    ) == 1
    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["layout_number"] == "0016"
    assert row["scenario_number"] == "00001"
    assert row["historical_match"] == "True"


def test_georeferenced_selection_uses_portable_grid_path(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    dataset = tmp_path / "raw"
    layout = dataset / "0016_03070"
    vegetation = layout / "Vegetation_Map"
    masks_root = layout / "Satellite_Images_Mask"
    space_masks = masks_root / "0016_00001"
    date_masks = masks_root / "0016_00002"
    weather = layout / "Weather_Data"
    vegetation.mkdir(parents=True)
    space_masks.mkdir(parents=True)
    date_masks.mkdir(parents=True)
    weather.mkdir()
    grid_transform = from_origin(500000, 4100000, 30, 30)
    profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 4,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32611",
        "transform": grid_transform,
    }
    with rasterio.open(vegetation / "Existing_Vegetation_Cover.tif", "w", **profile) as sink:
        sink.write(np.ones((3, 4), dtype=np.uint8), 1)
    space_ignition = np.zeros((3, 4), dtype=np.uint8)
    space_ignition[2, 1] = 255
    Image.fromarray(space_ignition).save(
        space_masks / "out1.jpg", quality=100, subsampling=0
    )
    date_ignition = np.zeros((3, 4), dtype=np.uint8)
    date_ignition[1, 2] = 255
    Image.fromarray(date_ignition).save(
        date_masks / "out1.jpg", quality=100, subsampling=0
    )
    (weather / "0016_00001.txt").write_text("2020 07 03 0000\n")
    (weather / "0016_00002.txt").write_text("2020 07 03 0000\n")

    east, north = xy(grid_transform, 1, 2, offset="center")
    lon, lat = transform("EPSG:32611", "EPSG:4326", [east], [north])
    fires = gpd.GeoDataFrame(
        {
            "fire_id": ["f"],
            "discovery_date": ["2020-07-03T00:00:00Z"],
            "source": ["FPA_FOD_2022"],
        },
        geometry=[Point(lon[0], lat[0])],
        crs="EPSG:4326",
    )
    fires_path = tmp_path / "fires.gpkg"
    fires.to_file(fires_path, layer="fires", driver="GPKG")
    footprints = tmp_path / "layouts.geojson"
    extract_layout_footprints(dataset, footprints, include_small=True)

    manifest = tmp_path / "selection.csv"
    assert select_scenarios(dataset, fires_path, footprints, manifest) == (1, 1)
    with manifest.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["scenario_id"] == "0016_00001"
    assert row["fire_row"] == row["ignition_row"] == "2"
    assert row["fire_col"] == row["ignition_col"] == "1"

    historical_manifest = tmp_path / "selection_historical.csv"
    assert select_scenarios(
        dataset,
        fires_path,
        footprints,
        historical_manifest,
        date_aware=True,
    ) == (1, 1)
    with historical_manifest.open(newline="") as stream:
        historical_row = next(csv.DictReader(stream))
    assert historical_row["scenario_id"] == "0016_00002"
    assert historical_row["fire_row"] == historical_row["ignition_row"] == "1"
    assert historical_row["fire_col"] == historical_row["ignition_col"] == "2"
