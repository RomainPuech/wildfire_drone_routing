#!/usr/bin/env python3
"""
Fill 100M and 500M benchmark report placeholders from sensor placement JSON.

Run from project root after sensor placement has produced:
  California2021Dataset/logs/sensor_alloc_GaussianBudget100M_TOP_261x161_mean.json
  California2021Dataset/logs/sensor_alloc_GaussianBudget500M_TOP_261x161_mean.json

Usage:
  python report/fill_placement_report_2021.py
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "California2021Dataset" / "logs"
COST_SENSOR = 100_000
COST_STATION = 150_000
COST_DRONE = 50_000
MAX_BATTERY_SUBSTEPS = 7


def cluster_stats(charging_locs_opt):
    n = len(charging_locs_opt)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = charging_locs_opt[i]
            xj, yj = charging_locs_opt[j]
            if max(abs(xi - xj), abs(yi - yj)) <= MAX_BATTERY_SUBSTEPS:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    sizes = [len(g) for g in groups.values()]
    singletons = sum(1 for s in sizes if s == 1)
    multi = sum(1 for s in sizes if s > 1)
    return singletons, multi, len(sizes)


def main():
    for budget in (100, 500):
        cache_name = f"sensor_alloc_GaussianBudget{budget}M_TOP_261x161_mean.json"
        path = LOG_DIR / cache_name
        if not path.exists():
            print(f"Skip {budget}M: {path.name} not found")
            continue

        with open(path) as f:
            d = json.load(f)

        n_gs = len(d["ground_sensor_locations"])
        n_cs = len(d["charging_station_locations"])
        drones_per_station = d["drones_per_charging_station"]
        n_dr = sum(drones_per_station)

        sub_gs = n_gs * COST_SENSOR / 1e6
        sub_cs = n_cs * COST_STATION / 1e6
        sub_dr = n_dr * COST_DRONE / 1e6

        charging_locs = [tuple(x) for x in d["charging_station_locations"]]
        singletons, multi, total_clusters = cluster_stats(charging_locs)

        report_path = PROJECT_ROOT / "report" / f"benchmark_{budget}M_budget_2021.md"
        if not report_path.exists():
            print(f"Report not found: {report_path}")
            continue

        text = report_path.read_text()

        # Replace allocation table TBDs
        text = text.replace(
            "| Ground sensors | TBD | 100k | TBD |",
            f"| Ground sensors | {n_gs} | 100k | {sub_gs:.2f}M |",
        )
        text = text.replace(
            "| Charging stations | TBD | 150k | TBD |",
            f"| Charging stations | {n_cs} | 150k | {sub_cs:.2f}M |",
        )
        text = text.replace(
            "| Drones | TBD | 50k | TBD |",
            f"| Drones | {n_dr} | 50k | {sub_dr:.2f}M |",
        )

        # Replace cluster TBDs
        text = text.replace(
            "| Singleton clusters (1 station) | TBD |",
            f"| Singleton clusters (1 station) | {singletons} |",
        )
        text = text.replace(
            "| Multi-station clusters | TBD |",
            f"| Multi-station clusters | {multi} |",
        )
        text = text.replace(
            "| Total clusters | **TBD** |",
            f"| Total clusters | **{total_clusters}** |",
        )

        # MIP gap: try to leave as TBD or fill if we had it (we don't from JSON)
        # So leave "MIP gap at termination: TBD" as is.

        report_path.write_text(text)
        print(f"Updated {report_path.name}: {n_gs} sensors, {n_cs} stations, {n_dr} drones, {total_clusters} clusters")


if __name__ == "__main__":
    main()
