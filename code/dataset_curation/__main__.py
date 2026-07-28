"""Command-line interface for the WFDroneBench curation stages."""

from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path

from .geospatial import align_risk_raster, extract_layout_footprints, merge_fire_records
from .workflow import build_summary, preprocess_from_manifest, select_scenarios


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _date(value: str) -> date:
    return date.fromisoformat(value)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python code/curate_dataset.py",
        description="Non-destructive WFDroneBench dataset curation.",
    )
    result.add_argument("--verbose", action="store_true")
    commands = result.add_subparsers(dest="command", required=True)

    footprints = commands.add_parser("footprints", help="inventory exact 30 m layout grids")
    footprints.add_argument("dataset_root", type=_path)
    footprints.add_argument("output", type=_path)
    footprints.add_argument("--resolution", type=float, default=30.0, help="expected metres/cell (default: 30)")
    footprints.add_argument("--minimum-width", type=int, default=500, help="exclude width below this (default: 500)")
    footprints.add_argument("--include-small", action="store_true", help="retain layouts narrower than --minimum-width")

    fires = commands.add_parser("merge-fires", help="normalize and deduplicate fire records")
    fires.add_argument("fpa_fod_2022", type=_path, help="FPA_FOD_20221014.gpkg")
    fires.add_argument("usfs_newer", type=_path, help="newer USFS occurrence vector data")
    fires.add_argument("output", type=_path, help=".gpkg or .parquet")

    select = commands.add_parser("select", help="spatially match fires to scenarios without reuse")
    select.add_argument("dataset_root", type=_path)
    select.add_argument("fires", type=_path)
    select.add_argument("footprints", type=_path)
    select.add_argument("output_manifest", type=_path)
    select.add_argument("--date-aware", action="store_true", help="require scenario date within ±1 day by default")
    select.add_argument(
        "--max-distance",
        type=int,
        help="Chebyshev cells (default: 5 space-only; 10 date-aware)",
    )
    select.add_argument("--max-day-difference", type=int, default=1, help="date leeway in days (default: 1)")
    select.add_argument(
        "--max-unmatched-fraction",
        type=float,
        default=0.2,
        help="reject layout above this fraction (default: 0.20)",
    )
    select.add_argument("--seed", type=int, default=0, help="deterministic tie seed (default: 0)")
    select.add_argument(
        "--fire-source",
        help="source label to retain (date-aware default: USFS_newer)",
    )
    select.add_argument(
        "--all-fire-sources",
        action="store_true",
        help="date-aware mode only: do not default to USFS_newer records",
    )
    select.add_argument(
        "--minimum-discovery-date",
        type=_date,
        help="ISO date lower bound (date-aware default: 2019-01-01)",
    )
    select.add_argument(
        "--all-weather-dates",
        action="store_true",
        help="do not restrict date-aware fires to the layout's weather-date range",
    )
    select.add_argument(
        "--enforce-layout-filter",
        action="store_true",
        help="also apply the 20%% unmatched-layout rejection in classification-only date-aware mode",
    )

    risk = commands.add_parser("risk", help="align continental BP/WHP to exact layout grids")
    risk.add_argument("continental_raster", type=_path)
    risk.add_argument("dataset_root", type=_path)
    risk.add_argument("output_root", type=_path)
    risk.add_argument("--risk-name", choices=("bp", "whp"), required=True)
    risk.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "cubic"),
        help="default: bilinear for BP, nearest for WHP",
    )
    risk.add_argument(
        "--fill-nodata",
        type=float,
        default=0.0,
        help="value outside source coverage (default: 0)",
    )

    preprocess = commands.add_parser("preprocess", help="convert selected JPGs and compute burn maps")
    preprocess.add_argument("dataset_root", type=_path)
    preprocess.add_argument("manifest", type=_path)
    preprocess.add_argument("output_root", type=_path)
    preprocess.add_argument(
        "--config",
        type=_path,
        help="root config_s2r.json containing optional scenario offsets",
    )
    preprocess.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold after JPG /255 normalization (default: >=0.5)",
    )

    summary = commands.add_parser("summary", help="build scenario_summary.csv")
    summary.add_argument("dataset_root", type=_path, help="raw or preprocessed layout root")
    summary.add_argument("manifest", type=_path)
    summary.add_argument("output", type=_path)
    summary.add_argument(
        "--cell-size-km",
        type=float,
        default=0.03,
        help="physical cell size for radius (default: 0.03 for 30 m)",
    )
    summary.add_argument("--big-radius-km", type=float, default=20.0, help="big-fire radius threshold (default: 20)")
    summary.add_argument("--fast-timestep", type=int, default=10, help="early timestep (default: 10)")
    summary.add_argument("--fast-fraction", type=float, default=0.5, help="early/final burned-cell ratio (default: 0.5)")
    summary.add_argument(
        "--historical-manifest",
        type=_path,
        help="date-aware selection manifest used to set historical_match",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "footprints":
        count = extract_layout_footprints(
            args.dataset_root,
            args.output,
            expected_resolution=args.resolution,
            minimum_width=args.minimum_width,
            include_small=args.include_small,
        )
    elif args.command == "merge-fires":
        count = merge_fire_records(args.fpa_fod_2022, args.usfs_newer, args.output)
    elif args.command == "select":
        date_source = None
        minimum_date = args.minimum_discovery_date
        if args.date_aware:
            if not args.all_fire_sources:
                date_source = args.fire_source or "USFS_newer"
            if minimum_date is None:
                minimum_date = date(2019, 1, 1)
        count, layouts = select_scenarios(
            args.dataset_root,
            args.fires,
            args.footprints,
            args.output_manifest,
            date_aware=args.date_aware,
            max_distance=args.max_distance,
            max_day_difference=args.max_day_difference,
            max_unmatched_fraction=args.max_unmatched_fraction,
            seed=args.seed,
            fire_source=date_source if args.date_aware else args.fire_source,
            minimum_discovery_date=minimum_date,
            restrict_to_weather_range=args.date_aware and not args.all_weather_dates,
            apply_layout_filter=not args.date_aware or args.enforce_layout_filter,
        )
        logging.warning("Selected %d scenarios across %d layouts", count, layouts)
        return 0
    elif args.command == "risk":
        count = align_risk_raster(
            args.continental_raster,
            args.dataset_root,
            args.output_root,
            risk_name=args.risk_name,
            resampling=args.resampling,
            fill_nodata=args.fill_nodata,
        )
    elif args.command == "preprocess":
        count = preprocess_from_manifest(
            args.dataset_root,
            args.manifest,
            args.output_root,
            config_file=args.config,
            threshold=args.threshold,
        )
    else:
        count = build_summary(
            args.dataset_root,
            args.manifest,
            args.output,
            cell_size_km=args.cell_size_km,
            big_radius_km=args.big_radius_km,
            fast_timestep=args.fast_timestep,
            fast_fraction=args.fast_fraction,
            historical_manifest=args.historical_manifest,
        )
    logging.warning("Wrote %d records/layouts", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
