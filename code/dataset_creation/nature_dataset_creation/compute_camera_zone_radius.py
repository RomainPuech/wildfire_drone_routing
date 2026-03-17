#!/usr/bin/env python3
"""Compute the radius (in miles) of the circle arcs in cameras.json. Each zone is a sector; radius = max distance from camera (center) to the sector boundary."""

import os
import geopandas as gpd
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CAMERAS_JSON = os.path.join(DATA_DIR, "cameras.json")
CA_ALBERS = "EPSG:3310"
METERS_TO_MILES = 1.0 / 1609.344


def main():
    zones = gpd.read_file(CAMERAS_JSON)
    if zones.crs is None:
        zones.set_crs("EPSG:4326", inplace=True)
    zones_m = zones.to_crs(CA_ALBERS)
    radii_m = []
    for idx, row in zones_m.iterrows():
        lon = row.get("longitude", None)
        lat = row.get("latitude", None)
        if lon is None or lat is None:
            continue
        center_wgs = Point(float(lon), float(lat))
        center_m = gpd.GeoSeries([center_wgs], crs="EPSG:4326").to_crs(CA_ALBERS).iloc[0]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        boundary = geom.boundary
        if boundary is None:
            pts = list(geom.exterior.coords) if hasattr(geom, "exterior") else []
        elif boundary.geom_type == "LineString":
            pts = list(boundary.coords)
        else:
            pts = []
            for g in boundary.geoms:
                pts.extend(g.coords)
        if not pts:
            continue
        r = max(center_m.distance(Point(x, y)) for x, y in pts)
        radii_m.append(r)
    radii_mi = [r * METERS_TO_MILES for r in radii_m]
    print(f"Camera zones: {len(radii_mi)} with valid radius")
    print(f"Radius (miles) — min: {min(radii_mi):.2f}, max: {max(radii_mi):.2f}, mean: {sum(radii_mi)/len(radii_mi):.2f}")
    print(f"Radius (miles) — median: {sorted(radii_mi)[len(radii_mi)//2]:.2f}")


if __name__ == "__main__":
    main()
