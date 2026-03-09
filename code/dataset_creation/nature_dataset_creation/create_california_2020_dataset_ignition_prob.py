#!/usr/bin/env python3
"""
Create California 2020 Wildfire Dataset using Ignition Probability Map

This script creates a dataset with:
- One statewide layout using static ignition probability map
- One ignition-point scenario per fire
- Filters: CA 2020, non-prescriptive, non-urban areas
"""

import os
import sys
import sqlite3
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform
from shapely.geometry import Point, box
from shapely.geometry import mapping
from datetime import datetime, timedelta
import json
import random
from pathlib import Path
from tqdm import tqdm

# Add code directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
code_dir = os.path.join(project_root, "code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dataset import save_scenario_ignition_point

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../California2020Dataset_IgnitionProb")

# Paths
FIRE_DB_PATH = os.path.join(DATA_DIR, "RDS-2013-0009.6_Data_Format3_GPKG/FPA_FOD_20221014.gpkg")
URBAN_SHAPEFILE = os.path.join(DATA_DIR, "tl_2025_us_uac20/tl_2025_us_uac20.shp")
CALIFORNIA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract/tl_2024_06_tract.shp")
IGNITION_PROB_PATH = os.path.join(DATA_DIR, "W_Tot_Ign_Prob_COG/W_Tot_Ign_Prob_COG.tif")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "scenarii"), exist_ok=True)

def parse_fire_date(date_str):
    """Parse fire discovery date from various formats."""
    if not date_str or date_str == '':
        return None
    
    formats = [
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m-%d-%Y",
    ]
    
    for fmt in formats:
        try:
            date_part = date_str.split()[0] if ' ' in date_str else date_str
            return datetime.strptime(date_part, fmt)
        except:
            continue
    
    return None

def crop_to_california(data, transform, crs, california_gdf):
    """
    Crop data to California state bounding box.
    """
    from affine import Affine

    # Reproject California boundary to raster CRS
    california_raster_crs = california_gdf.to_crs(crs)

    # Bounding box in raster CRS
    minx, miny, maxx, maxy = california_raster_crs.total_bounds
    
    # Add buffer
    buffer_meters = 50000  # 50 km buffer
    minx -= buffer_meters
    miny -= buffer_meters
    maxx += buffer_meters
    maxy += buffer_meters

    # Convert to pixel row/col
    row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)

    row_min, col_min = int(np.floor(row_min)), int(np.floor(col_min))
    row_max, col_max = int(np.ceil(row_max)) + 1, int(np.ceil(col_max)) + 1

    # Clamp to grid bounds
    H = data.shape[-2]
    W = data.shape[-1]
    row_min = max(0, row_min)
    row_max = min(H, row_max)
    col_min = max(0, col_min)
    col_max = min(W, col_max)

    print(f"  California crop: rows {row_min}:{row_max}, cols {col_min}:{col_max}  "
          f"(full grid: {H}×{W} → cropped: {row_max - row_min}×{col_max - col_min})")

    # Slice
    if data.ndim == 2:
        result = data[row_min:row_max, col_min:col_max]
    else:
        result = data[..., row_min:row_max, col_min:col_max]

    # Update transform
    new_transform = Affine(
        transform.a, transform.b, transform.c + col_min * transform.a,
        transform.d, transform.e, transform.f + row_min * transform.e
    )

    return result, new_transform, (row_min, row_max - 1, col_min, col_max - 1)

def create_california_mask(ignition_prob_data, ignition_prob_transform, ignition_prob_crs, urban_gdf=None, california_gdf=None):
    """
    Create a mask for California from ignition probability data.
    Mask is 1 where data is valid AND not in urban areas, 0 otherwise.
    Cropped to California state boundary.
    """
    from rasterio.features import rasterize
    from scipy import ndimage
    
    # Crop to California first if boundary provided
    if california_gdf is not None:
        ignition_prob_data_3d = ignition_prob_data[np.newaxis, :, :] if ignition_prob_data.ndim == 2 else ignition_prob_data
        cropped_data, new_transform, crop_bounds = crop_to_california(
            ignition_prob_data_3d, ignition_prob_transform, ignition_prob_crs, california_gdf
        )
        # Ensure 2D
        if cropped_data.ndim == 3:
            ignition_prob_data = cropped_data[0]
        else:
            ignition_prob_data = cropped_data
        ignition_prob_transform = new_transform
    else:
        crop_bounds = (0, ignition_prob_data.shape[0], 0, ignition_prob_data.shape[1])
        new_transform = ignition_prob_transform
    
    # Start with validity mask (not NaN, not negative) - ensure 2D
    if ignition_prob_data.ndim == 3:
        ignition_prob_data_2d = ignition_prob_data[0]
    else:
        ignition_prob_data_2d = ignition_prob_data
    mask = np.ones_like(ignition_prob_data_2d, dtype=np.float32)
    mask[np.isnan(ignition_prob_data_2d)] = 0
    mask[ignition_prob_data_2d < 0] = 0
    
    # Mask areas outside California boundary (if provided)
    if california_gdf is not None:
        california_raster_crs = california_gdf.to_crs(ignition_prob_crs)
        shapes = [mapping(geom) for geom in california_raster_crs.geometry]
        
        # Create a temporary rasterio dataset for masking
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds
        
        # Use 2D dimensions
        height, width = ignition_prob_data_2d.shape
        left = ignition_prob_transform[2]
        top = ignition_prob_transform[5]
        right = left + width * ignition_prob_transform[0]
        bottom = top + height * ignition_prob_transform[4]
        
        # Rasterize California boundary
        california_mask_raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=ignition_prob_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        mask[california_mask_raster == 0] = 0
    
    # Mask urban areas (if provided)
    if urban_gdf is not None:
        urban_raster_crs = urban_gdf.to_crs(ignition_prob_crs)
        urban_shapes = [mapping(geom) for geom in urban_raster_crs.geometry]
        
        # Use 2D dimensions
        height, width = ignition_prob_data_2d.shape
        left = ignition_prob_transform[2]
        top = ignition_prob_transform[5]
        right = left + width * ignition_prob_transform[0]
        bottom = top + height * ignition_prob_transform[4]
        
        # Rasterize urban areas
        urban_mask_raster = rasterize(
            urban_shapes,
            out_shape=(height, width),
            transform=ignition_prob_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        mask[urban_mask_raster == 1] = 0  # Set urban areas to 0
    
    # Keep only the largest connected component
    labeled_array, num_features = ndimage.label(mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # Ignore background (0)
        largest_component_label = np.argmax(component_sizes)
        mask[labeled_array != largest_component_label] = 0
    
    return mask, new_transform, crop_bounds

def main():
    print("=" * 80)
    print("California 2020 Wildfire Dataset Creation (Ignition Probability Map)")
    print("=" * 80)
    
    # Step 1: Load fire database
    print("\n[1/7] Loading fire database...")
    conn = sqlite3.connect(FIRE_DB_PATH)
    cursor = conn.cursor()
    
    # Get all CA 2020 fires with valid dates
    cursor.execute("""
        SELECT FOD_ID, FIRE_NAME, DISCOVERY_DATE, LATITUDE, LONGITUDE, FIRE_SIZE
        FROM Fires
        WHERE FIRE_YEAR = 2020
        AND STATE = 'CA'
        AND DISCOVERY_DATE IS NOT NULL
        AND DISCOVERY_DATE != ''
        AND LATITUDE IS NOT NULL
        AND LONGITUDE IS NOT NULL
        ORDER BY DISCOVERY_DATE
    """)
    
    all_fires = cursor.fetchall()
    print(f"Found {len(all_fires)} CA 2020 fires with valid dates")
    
    # Parse dates and filter
    fires_with_dates = []
    skipped_jan1 = 0
    for fire in all_fires:
        fire_id, fire_name, discovery_date, lat, lon, fire_size = fire
        parsed_date = parse_fire_date(discovery_date)
        if parsed_date:
            # Skip fires on January 1st
            if parsed_date.month == 1 and parsed_date.day == 1:
                skipped_jan1 += 1
                continue
            
            fires_with_dates.append({
                'fod_id': fire_id,
                'fire_name': fire_name or f"Fire_{fire_id}",
                'discovery_date': parsed_date,
                'date_str': parsed_date.strftime("%Y-%m-%d"),
                'latitude': lat,
                'longitude': lon,
                'fire_size': fire_size,
            })
    
    print(f"Parsed {len(fires_with_dates)} fires with valid dates")
    if skipped_jan1 > 0:
        print(f"Skipped {skipped_jan1} fires on January 1st")
    
    # Step 2: Load urban areas and California boundary
    print("\n[2/7] Loading urban areas and California boundary...")
    urban_gdf = gpd.read_file(URBAN_SHAPEFILE)
    urban_gdf = urban_gdf.to_crs("EPSG:4326")
    urban_gdf["geometry"] = urban_gdf.buffer(0)
    
    california_tracts = gpd.read_file(CALIFORNIA_TRACTS)
    california_tracts = california_tracts.to_crs("EPSG:4326")
    california_tracts["geometry"] = california_tracts.buffer(0)
    california_boundary = california_tracts.dissolve()
    california_boundary_gdf = gpd.GeoDataFrame([1], geometry=[california_boundary.geometry.iloc[0]], crs="EPSG:4326")
    
    # Create fire points GeoDataFrame
    fire_points = []
    for fire in fires_with_dates:
        fire_points.append({
            'fod_id': fire['fod_id'],
            'geometry': Point(fire['longitude'], fire['latitude'])
        })
    
    fire_gdf = gpd.GeoDataFrame(fire_points, crs="EPSG:4326")
    
    # Spatial join to find fires in urban areas
    fires_in_urban = gpd.sjoin(fire_gdf, urban_gdf, how='inner', predicate='within')
    urban_fire_ids = set(fires_in_urban['fod_id'].values)
    
    # Filter out urban fires
    non_urban_fires = [f for f in fires_with_dates if f['fod_id'] not in urban_fire_ids]
    print(f"Filtered out {len(fires_with_dates) - len(non_urban_fires)} urban fires")
    print(f"Remaining: {len(non_urban_fires)} non-urban fires")
    
    # Step 3: Load ignition probability map
    print("\n[3/7] Loading ignition probability map...")
    with rasterio.open(IGNITION_PROB_PATH) as src:
        ignition_prob_data = src.read(1)
        ignition_prob_meta = src.meta.copy()
        ignition_prob_transform = src.transform
        ignition_prob_crs = src.crs
    
    print(f"Ignition probability map dimensions: {ignition_prob_data.shape}")
    print(f"CRS: {ignition_prob_crs}")
    print(f"Resolution: {src.res}")
    print(f"Data range: [{np.nanmin(ignition_prob_data):.6f}, {np.nanmax(ignition_prob_data):.6f}]")
    
    # Step 4: Create mask
    print("\n[4/7] Creating California mask...")
    print("  - Cropping to California bounding box (with buffer)...")
    print("  - Masking areas outside California boundary...")
    print("  - Masking invalid data (NaN, negative)...")
    print("  - Masking urban areas...")
    print("  - Keeping only largest connected component...")
    california_mask, mask_transform, crop_bounds = create_california_mask(
        ignition_prob_data, ignition_prob_transform, ignition_prob_crs, urban_gdf, california_boundary_gdf
    )
    mask_path = os.path.join(OUTPUT_DIR, "mask.npy")
    np.save(mask_path, california_mask)
    
    total_cells = california_mask.size
    valid_cells = np.sum(california_mask == 1)
    masked_cells = total_cells - valid_cells
    print(f"Saved mask to {mask_path}")
    print(f"  Mask shape (cropped): {california_mask.shape}")
    print(f"  Valid operational area: {valid_cells:,} cells ({100*valid_cells/total_cells:.1f}%)")
    print(f"  Masked area: {masked_cells:,} cells ({100*masked_cells/total_cells:.1f}%)")
    
    # Step 5: Crop and save ignition probability map
    print("\n[5/7] Cropping and saving ignition probability map...")
    ignition_prob_data_3d = ignition_prob_data[np.newaxis, :, :] if ignition_prob_data.ndim == 2 else ignition_prob_data
    cropped_ignition_prob, cropped_transform, _ = crop_to_california(
        ignition_prob_data_3d, ignition_prob_transform, ignition_prob_crs, california_boundary_gdf
    )
    
    # Save cropped ignition probability map
    ignition_prob_path = os.path.join(OUTPUT_DIR, "static_risk_ignition_prob.npy")
    ignition_prob_3d = cropped_ignition_prob[np.newaxis, :, :].astype(np.float32) if cropped_ignition_prob.ndim == 2 else cropped_ignition_prob.astype(np.float32)
    np.save(ignition_prob_path, ignition_prob_3d)
    print(f"Saved ignition probability map to {ignition_prob_path}")
    print(f"  Shape: {ignition_prob_3d.shape}")
    
    # Update transform and data for coordinate conversion
    ignition_prob_transform = cropped_transform
    # Ensure 2D
    if cropped_ignition_prob.ndim == 3:
        ignition_prob_data = cropped_ignition_prob[0]
    else:
        ignition_prob_data = cropped_ignition_prob
    grid_height, grid_width = ignition_prob_data.shape
    
    # Step 6: Process fires and create scenarios
    print("\n[6/7] Processing fires and creating scenarios...")
    
    # Create transformer for coordinate conversion
    from pyproj import Transformer
    if isinstance(ignition_prob_crs, str):
        ignition_prob_crs_obj = rasterio.crs.CRS.from_string(ignition_prob_crs)
    else:
        ignition_prob_crs_obj = ignition_prob_crs
    transformer = Transformer.from_crs("EPSG:4326", ignition_prob_crs_obj, always_xy=True)
    
    successful_fires = []
    failed_fires = []
    
    for fire in tqdm(non_urban_fires, desc="Processing fires"):
        try:
            # Convert lat/lon to grid coordinates
            x, y = transformer.transform(fire['longitude'], fire['latitude'])
            row, col = rasterio.transform.rowcol(ignition_prob_transform, x, y)
            
            # Validate coordinates
            if not (0 <= row < grid_height and 0 <= col < grid_width):
                print(f"WARNING: Fire {fire['fod_id']} at ({row}, {col}) is outside grid bounds")
                failed_fires.append(fire)
                continue
            
            # Check if point is in valid mask area
            if california_mask[row, col] == 0:
                print(f"WARNING: Fire {fire['fod_id']} at ({row}, {col}) is in masked area")
                failed_fires.append(fire)
                continue
            
            # Create scenario filename
            safe_fire_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in fire['fire_name'])
            scenario_name = f"{safe_fire_name}_{fire['fod_id']}_scenario1"
            scenario_path = os.path.join(OUTPUT_DIR, "scenarii", f"{scenario_name}.npy")
            
            # Save ignition point
            save_scenario_ignition_point(
                row=row,
                col=col,
                start_timestep=0,
                out_filename=scenario_path
            )
            
            successful_fires.append({
                'fire': fire,
                'scenario_name': scenario_name,
                'row': row,
                'col': col,
            })
            
        except Exception as e:
            print(f"ERROR processing fire {fire['fod_id']}: {e}")
            failed_fires.append(fire)
    
    print(f"\nSuccessfully processed: {len(successful_fires)} fires")
    print(f"Failed: {len(failed_fires)} fires")
    
    # Step 7: Create config file
    print("\n[7/7] Creating config file...")
    config = {}
    for fire_info in successful_fires:
        fire_key = f"offset_{fire_info['scenario_name'].replace('_scenario1', '')}"
        config[fire_key] = random.randint(1, 12)
    
    config_path = os.path.join(OUTPUT_DIR, "config_california_2020_ignition_prob.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Step 8: Create summary
    print("\n[8/8] Creating dataset summary...")
    summary = {
        'dataset_name': 'California2020_IgnitionProb',
        'total_fires': len(non_urban_fires),
        'successful_fires': len(successful_fires),
        'failed_fires': len(failed_fires),
        'grid_dimensions': {
            'height': int(grid_height),
            'width': int(grid_width)
        },
        'crs': str(ignition_prob_crs),
        'resolution_meters': 120.0,
        'date_range': {
            'start': min(f['discovery_date'].strftime("%Y-%m-%d") for f in non_urban_fires),
            'end': max(f['discovery_date'].strftime("%Y-%m-%d") for f in non_urban_fires)
        }
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    print("\n" + "=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Successful fires: {len(successful_fires)}")
    print(f"Failed fires: {len(failed_fires)}")
    
    conn.close()

if __name__ == "__main__":
    main()
