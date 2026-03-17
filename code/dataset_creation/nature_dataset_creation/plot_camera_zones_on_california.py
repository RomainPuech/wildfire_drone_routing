#!/usr/bin/env python3
"""
Plot camera zones from cameras.json (GeoJSON) on top of a California map.

Zone polygons (sectors) are converted to full circles (center from properties,
radius in miles, default 20). Output: camera_zones_california.png (or _60mi.png etc.)
"""

import argparse
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CAMERAS_JSON = os.path.join(DATA_DIR, "cameras.json")
CA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract", "tl_2024_06_tract.shp")
CA_ALBERS = "EPSG:3310"  # California Albers (meters) for buffer
MILES_TO_METERS = 1609.344


def zones_to_full_circles(zones_gdf, radius_miles=20):
    """Replace each zone polygon (sector) with a full circle. Center from properties; radius = radius_miles (default 20)."""
    radius_m = radius_miles * MILES_TO_METERS
    zones_m = zones_gdf.to_crs(CA_ALBERS)
    circles = []
    for idx, row in zones_m.iterrows():
        lon = row.get("longitude", None)
        lat = row.get("latitude", None)
        if lon is None or lat is None or (hasattr(lon, "__float__") and (lon != lon or lat != lat)):
            circles.append(row.geometry)
            continue
        center_wgs = Point(float(lon), float(lat))
        center_m = gpd.GeoSeries([center_wgs], crs="EPSG:4326").to_crs(CA_ALBERS).iloc[0]
        circles.append(center_m.buffer(radius_m))
    out = gpd.GeoDataFrame(
        zones_gdf.drop(columns=["geometry"]),
        geometry=circles,
        crs=CA_ALBERS,
    )
    return out.to_crs("EPSG:4326")


def main():
    parser = argparse.ArgumentParser(description="Plot camera zones (circles) on California map.")
    parser.add_argument("--radius-miles", type=float, default=20, help="Camera zone circle radius in miles (default: 20)")
    args = parser.parse_args()
    radius_miles = args.radius_miles

    out_png = os.path.join(SCRIPT_DIR, "camera_zones_california.png")
    if radius_miles != 20:
        out_png = os.path.join(SCRIPT_DIR, f"camera_zones_california_{int(radius_miles)}mi.png")

    print("Loading camera zones GeoJSON …")
    zones = gpd.read_file(CAMERAS_JSON)
    if zones.crs is None:
        zones.set_crs("EPSG:4326", inplace=True)
    else:
        zones = zones.to_crs("EPSG:4326")
    print(f"Converting zone sectors to full circles (radius={radius_miles} mi) …")
    zones = zones_to_full_circles(zones, radius_miles=radius_miles)

    print("Loading California boundary …")
    if not os.path.exists(CA_TRACTS):
        raise FileNotFoundError(
            f"California tract shapefile not found at {CA_TRACTS}. "
            "Download tl_2024_06_tract (Census TIGER) and extract to data/tl_2024_06_tract/."
        )
    ca_tracts = gpd.read_file(CA_TRACTS).to_crs("EPSG:4326")
    ca_tracts["geometry"] = ca_tracts.buffer(0)
    ca_boundary = ca_tracts.dissolve()

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_aspect("equal")

    # Basemap: California outline and light fill
    ca_boundary.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#333333", linewidth=0.8)

    # Camera zones: semi-transparent polygons (many overlapping zones)
    zones.plot(
        ax=ax,
        facecolor=to_rgba("steelblue", 0.35),
        edgecolor=to_rgba("steelblue", 0.7),
        linewidth=0.4,
    )

    ax.set_xlim(ca_boundary.total_bounds[0], ca_boundary.total_bounds[2])
    ax.set_ylim(ca_boundary.total_bounds[1], ca_boundary.total_bounds[3])
    ax.set_axis_off()
    ax.set_title(f"AlertCalifornia camera zones on California ({radius_miles:.0f} mi radius)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
