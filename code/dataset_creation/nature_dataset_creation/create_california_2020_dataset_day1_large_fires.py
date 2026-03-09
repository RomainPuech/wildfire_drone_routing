#!/usr/bin/env python3
"""
Create California 2020 Wildfire Dataset (Day 1 Forecast Version)

This script creates a dataset with:
- One statewide layout using WFPI Day 1 forecast maps (same-day forecast)
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
import zipfile
import tempfile
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
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../California2020Dataset_Day1_LargeFires")

# Paths
FIRE_DB_PATH = os.path.join(DATA_DIR, "RDS-2013-0009.6_Data_Format3_GPKG/FPA_FOD_20221014.gpkg")
URBAN_SHAPEFILE = os.path.join(DATA_DIR, "tl_2025_us_uac20/tl_2025_us_uac20.shp")
CALIFORNIA_TRACTS = os.path.join(DATA_DIR, "tl_2024_06_tract/tl_2024_06_tract.shp")  # California tracts for state boundary
WFPI_DATA_DIR = os.path.join(DATA_DIR, "2020_Wind-enhanced_Fire_Potential_Index_Forecast_1_DATA")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "scenarii"), exist_ok=True)

def parse_fire_date(date_str):
    """Parse fire discovery date from various formats."""
    if not date_str or date_str == '':
        return None
    
    # Try different date formats
    formats = [
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m-%d-%Y",
    ]
    
    for fmt in formats:
        try:
            # Handle time components
            date_part = date_str.split()[0] if ' ' in date_str else date_str
            return datetime.strptime(date_part, fmt)
        except:
            continue
    
    return None

def get_wfpi_date_for_fire(fire_date):
    """
    Get the WFPI date to use for a fire.
    We use Day 1 forecast from the SAME DAY as the fire (same-day forecast).
    """
    if fire_date is None:
        return None
    
    # Same day as the fire
    return fire_date.strftime("%Y%m%d")

def find_wfpi_file(target_date):
    """Find the WFPI zip file for a given date."""
    zip_pattern = f"wfpi-forecast-1_data_{target_date}_{target_date}.zip"
    zip_path = os.path.join(WFPI_DATA_DIR, zip_pattern)
    
    if os.path.exists(zip_path):
        return zip_path
    return None

def load_wfpi_raster(date_str):
    """
    Load WFPI raster for a given date.
    Returns (raster_data, raster_meta, raster_transform, raster_crs)
    """
    zip_path = find_wfpi_file(date_str)
    if not zip_path:
        return None
    
    # Extract to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Find the .tif file (exclude .xml files)
        tif_files = [f for f in Path(tmpdir).rglob("*.tif") if not f.name.endswith('.xml')]
        if not tif_files:
            # Try .tiff extension
            tif_files = [f for f in Path(tmpdir).rglob("*.tiff") if not f.name.endswith('.xml')]
        if not tif_files:
            return None
        
        tif_path = str(tif_files[0])
        
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
            
            return (data, meta, transform, crs)

def crop_to_california(data, wfpi_transform, crs, california_gdf):
    """
    Crop data to California state bounding box.

    Uses the bounding box of California in the WFPI CRS to find the
    pixel range that covers California, then slices the array directly.
    This is reliable regardless of the data dtype.

    Returns:
        cropped_data (2-D if input was 2-D or 3-D with 1 band leading dim),
        new_transform (Affine),
        (row_min, row_max, col_min, col_max)
    """
    from affine import Affine

    # Reproject California boundary to WFPI CRS
    california_wfpi_crs = california_gdf.to_crs(crs)

    # Bounding box in WFPI CRS  (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = california_wfpi_crs.total_bounds
    
    # Add buffer to ensure we capture all of California (especially southern tip and irregular edges)
    # Buffer in meters (WFPI CRS is typically in meters)
    buffer_meters = 50000  # 50 km buffer
    minx -= buffer_meters
    miny -= buffer_meters
    maxx += buffer_meters
    maxy += buffer_meters

    # Convert geographic corners to pixel row/col
    # maxy (northernmost) → smallest row index; miny → largest row index
    row_min, col_min = rasterio.transform.rowcol(wfpi_transform, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(wfpi_transform, maxx, miny)

    # rowcol can return floats; cast and add a small buffer (round down for min, up for max)
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
        cropped = data[row_min:row_max, col_min:col_max]
    else:
        cropped = data[:, row_min:row_max, col_min:col_max]

    # Build new transform whose origin is the top-left of the cropped window
    new_transform = Affine(
        wfpi_transform.a, wfpi_transform.b,
        wfpi_transform.c + col_min * wfpi_transform.a,
        wfpi_transform.d, wfpi_transform.e,
        wfpi_transform.f + row_min * wfpi_transform.e,
    )

    result = cropped[0] if data.ndim == 3 else cropped
    return result, new_transform, (row_min, row_max - 1, col_min, col_max - 1)

def create_california_mask(wfpi_data, wfpi_meta, wfpi_transform, wfpi_crs, urban_gdf=None, california_gdf=None):
    """
    Create a mask for California from WFPI data.
    Mask is 1 where WFPI data is valid (>= 0) AND not in urban areas, 0 otherwise.
    Cropped to California state boundary.
    
    Args:
        wfpi_data: WFPI raster data
        wfpi_meta: WFPI metadata
        wfpi_transform: WFPI transform
        wfpi_crs: WFPI CRS
        urban_gdf: GeoDataFrame of urban areas (optional)
        california_gdf: GeoDataFrame of California boundary (optional)
    
    Returns:
        mask: Binary mask (1 = valid operational area, 0 = masked out)
        new_transform: Transform for cropped mask
        crop_bounds: (row_min, row_max, col_min, col_max) of crop
    """
    from rasterio.features import rasterize
    from scipy import ndimage
    
    # Crop to California first if boundary provided
    if california_gdf is not None:
        wfpi_data_3d = wfpi_data[np.newaxis, :, :] if wfpi_data.ndim == 2 else wfpi_data
        cropped_data, new_transform, crop_bounds = crop_to_california(
            wfpi_data_3d, wfpi_transform, wfpi_crs, california_gdf
        )
        wfpi_data = cropped_data
        wfpi_transform = new_transform
    else:
        crop_bounds = (0, wfpi_data.shape[0], 0, wfpi_data.shape[1])
        new_transform = wfpi_transform
    
    # Start with WFPI validity mask
    mask = np.ones_like(wfpi_data, dtype=np.float32)
    mask[wfpi_data < 0] = 0
    
    # Mask nodata values (NoData is typically 255 for uint8 WFPI data)
    if wfpi_meta.get('nodata') is not None:
        mask[wfpi_data == wfpi_meta['nodata']] = 0
    # Also explicitly mask 255 values (common NoData value for WFPI)
    mask[wfpi_data == 255] = 0
    
    # Mask special values >= 249 (outside US, deserts with no ignition risk, etc.)
    mask[wfpi_data >= 249] = 0
    
    # Mask areas outside California boundary (if provided)
    if california_gdf is not None:
        # Reproject California boundary to WFPI CRS (already done in crop, but need fresh for mask)
        california_wfpi_crs = california_gdf.to_crs(wfpi_crs)
        
        # Get union of all California geometries
        california_geom = california_wfpi_crs.unary_union
        
        # Create shapes for rasterization (inverse: 1 = inside CA, 0 = outside)
        shapes = [(california_geom, 1)]
        
        # Rasterize California boundary (1 = inside CA, 0 = outside CA)
        ca_mask = rasterize(
            shapes,
            out_shape=wfpi_data.shape,
            transform=wfpi_transform,
            fill=0,
            dtype=np.float32,
            all_touched=True
        )
        
        # Set mask to 0 where outside California
        mask[ca_mask == 0] = 0
    
    # Mask urban areas if provided
    if urban_gdf is not None:
        # Reproject urban areas to WFPI CRS
        urban_wfpi_crs = urban_gdf.to_crs(wfpi_crs)
        
        # Create shapes for rasterization
        shapes = [(geom, 1) for geom in urban_wfpi_crs.geometry]
        
        # Rasterize urban areas (1 = urban, 0 = non-urban)
        urban_mask = rasterize(
            shapes,
            out_shape=wfpi_data.shape,
            transform=wfpi_transform,
            fill=0,
            dtype=np.float32,
            all_touched=True
        )
        
        # Set mask to 0 where urban areas are
        mask[urban_mask == 1] = 0
    
    # Keep only the largest connected component (mainland California)
    # Label connected components
    labeled_mask, num_features = ndimage.label(mask == 1)
    
    if num_features > 0:
        # Find the largest component
        component_sizes = ndimage.sum(mask == 1, labeled_mask, range(1, num_features + 1))
        largest_component = np.argmax(component_sizes) + 1
        
        # Keep only the largest component, mask out all others (islands and isolated pixels)
        mask = (labeled_mask == largest_component).astype(np.float32)
    
    return mask, new_transform, crop_bounds

def latlon_to_grid_coords(lat, lon, transform, crs):
    """
    Convert lat/lon to grid row/col coordinates.
    """
    # Transform lat/lon to raster CRS
    x, y = transform(lon, lat, src_crs='EPSG:4326', dst_crs=crs)
    
    # Convert to row/col
    row, col = rasterio.transform.rowcol(transform, x, y)
    
    return row, col

def main():
    print("=" * 80)
    print("California 2020 Wildfire Dataset Creation (Day 1 Forecast, Large Fires Only, >= 100 acres)")
    print("=" * 80)
    
    # Step 1: Load fire database
    print("\n[1/7] Loading fire database...")
    conn = sqlite3.connect(FIRE_DB_PATH)
    cursor = conn.cursor()
    
    # Get all CA 2020 fires with valid dates and FIRE_SIZE >= 100 acres (large fires)
    cursor.execute("""
        SELECT FOD_ID, FIRE_NAME, DISCOVERY_DATE, LATITUDE, LONGITUDE, FIRE_SIZE
        FROM Fires
        WHERE FIRE_YEAR = 2020
        AND STATE = 'CA'
        AND DISCOVERY_DATE IS NOT NULL
        AND DISCOVERY_DATE != ''
        AND LATITUDE IS NOT NULL
        AND LONGITUDE IS NOT NULL
        AND FIRE_SIZE >= 100
        ORDER BY DISCOVERY_DATE
    """)
    
    all_fires = cursor.fetchall()
    print(f"Found {len(all_fires)} CA 2020 fires with valid dates")
    
    # Parse dates and filter
    fires_with_dates = []
    for fire in all_fires:
        fire_id, fire_name, discovery_date, lat, lon, fire_size = fire
        parsed_date = parse_fire_date(discovery_date)
        if parsed_date:
            # No need to skip Jan 1st for Day 1 forecast (uses same day)
            
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
    
    # Step 2: Load urban areas and California boundary
    print("\n[2/7] Loading urban areas and California boundary...")
    urban_gdf = gpd.read_file(URBAN_SHAPEFILE)
    urban_gdf = urban_gdf.to_crs("EPSG:4326")
    urban_gdf["geometry"] = urban_gdf.buffer(0)  # Fix invalid geometries
    
    # Load California tracts to get state boundary
    california_tracts = gpd.read_file(CALIFORNIA_TRACTS)
    california_tracts = california_tracts.to_crs("EPSG:4326")
    california_tracts["geometry"] = california_tracts.buffer(0)  # Fix invalid geometries
    # Create union of all tracts to get California boundary
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
    
    # Step 3: Load first WFPI file to get grid dimensions
    print("\n[3/7] Loading WFPI data structure...")
    # Find first fire with available WFPI data
    wfpi_result = None
    sample_fire = None
    for fire in non_urban_fires:
        wfpi_date = get_wfpi_date_for_fire(fire['discovery_date'])
        if wfpi_date:
            wfpi_result = load_wfpi_raster(wfpi_date)
            if wfpi_result:
                sample_fire = fire
                break
    
    if not wfpi_result:
        print(f"ERROR: Could not find any WFPI data for the fires")
        return
    
    wfpi_data, wfpi_meta, wfpi_transform, wfpi_crs = wfpi_result
    grid_height, grid_width = wfpi_data.shape
    print(f"WFPI grid dimensions: {grid_height} x {grid_width}")
    print(f"WFPI CRS: {wfpi_crs}")
    # Calculate bounds from transform
    left = wfpi_transform[2]
    top = wfpi_transform[5]
    right = left + grid_width * wfpi_transform[0]
    bottom = top + grid_height * wfpi_transform[4]
    print(f"WFPI bounds: ({left}, {bottom}, {right}, {top})")
    
    # Step 4: Create mask (cropped to California, with urban areas excluded, keeping only main component)
    print("\n[4/7] Creating California mask...")
    print("  - Cropping to California bounding box (with buffer)...")
    print("  - Masking areas outside California boundary...")
    print("  - Masking invalid WFPI data...")
    print("  - Masking urban areas...")
    print("  - Keeping only largest connected component (removing islands)...")
    california_mask, mask_transform, crop_bounds = create_california_mask(
        wfpi_data, wfpi_meta, wfpi_transform, wfpi_crs, urban_gdf, california_boundary_gdf
    )
    mask_path = os.path.join(OUTPUT_DIR, "mask.npy")
    np.save(mask_path, california_mask)
    
    # Calculate statistics
    total_cells = california_mask.size
    valid_cells = np.sum(california_mask == 1)
    masked_cells = total_cells - valid_cells
    print(f"Saved mask to {mask_path}")
    print(f"  Mask shape (cropped): {california_mask.shape}")
    print(f"  Valid operational area: {valid_cells:,} cells ({100*valid_cells/total_cells:.1f}%)")
    print(f"  Masked area: {masked_cells:,} cells ({100*masked_cells/total_cells:.1f}%)")
    
    # Step 5: Process fires and create scenarios
    print("\n[5/7] Processing fires and creating scenarios...")
    
    # Group fires by date to minimize WFPI loading
    fires_by_date = {}
    for fire in non_urban_fires:
        wfpi_date = get_wfpi_date_for_fire(fire['discovery_date'])
        if wfpi_date:
            if wfpi_date not in fires_by_date:
                fires_by_date[wfpi_date] = []
            fires_by_date[wfpi_date].append(fire)
    
    print(f"Processing {len(non_urban_fires)} fires across {len(fires_by_date)} unique dates")
    
    # Create transformer for coordinate conversion (will be created per date)
    
    successful_fires = []
    failed_fires = []
    
    for wfpi_date, fires in tqdm(fires_by_date.items(), desc="Processing dates"):
        # Load WFPI for this date
        wfpi_result = load_wfpi_raster(wfpi_date)
        if not wfpi_result:
            print(f"WARNING: Could not load WFPI for {wfpi_date}, skipping {len(fires)} fires")
            failed_fires.extend(fires)
            continue
        
        wfpi_data, wfpi_meta, wfpi_transform, wfpi_crs = wfpi_result
        
        # Crop WFPI to California
        wfpi_data_3d = wfpi_data[np.newaxis, :, :] if wfpi_data.ndim == 2 else wfpi_data
        cropped_wfpi, cropped_transform, _ = crop_to_california(
            wfpi_data_3d, wfpi_transform, wfpi_crs, california_boundary_gdf
        )
        
        # Save WFPI map (cropped)
        wfpi_path = os.path.join(OUTPUT_DIR, f"wfpi_day1_{wfpi_date}.npy")
        # Reshape to (1, H, W) for consistency
        wfpi_3d = cropped_wfpi[np.newaxis, :, :].astype(np.float32)
        # Cap negative values at 0
        wfpi_3d = np.maximum(wfpi_3d, 0)
        # Values > 249 are special values (outside US, deserts with no ignition risk, etc.)
        # Keep them as-is (they will be masked out by the mask)
        np.save(wfpi_path, wfpi_3d)
        
        # Update transform and data for coordinate conversion
        wfpi_transform = cropped_transform
        wfpi_data = cropped_wfpi
        
        # Create transformer for this date's CRS
        from pyproj import Transformer
        if isinstance(wfpi_crs, str):
            wfpi_crs_obj = rasterio.crs.CRS.from_string(wfpi_crs)
        else:
            wfpi_crs_obj = wfpi_crs
        transformer = Transformer.from_crs("EPSG:4326", wfpi_crs_obj, always_xy=True)
        
        # Update grid dimensions for cropped data
        grid_height, grid_width = wfpi_data.shape
        
        # Process each fire for this date
        for fire in fires:
            try:
                # Convert lat/lon to grid coordinates (using cropped transform)
                x, y = transformer.transform(fire['longitude'], fire['latitude'])
                row, col = rasterio.transform.rowcol(wfpi_transform, x, y)
                
                # Validate coordinates (using cropped dimensions)
                if not (0 <= row < grid_height and 0 <= col < grid_width):
                    print(f"WARNING: Fire {fire['fod_id']} at ({row}, {col}) is outside grid bounds")
                    failed_fires.append(fire)
                    continue
                
                # Check if point is in valid mask area
                if california_mask[row, col] == 0:
                    print(f"WARNING: Fire {fire['fod_id']} at ({row}, {col}) is in masked area")
                    failed_fires.append(fire)
                    continue
                
                # Create scenario filename (sanitize fire name)
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
                    'wfpi_date': wfpi_date,
                })
                
            except Exception as e:
                print(f"ERROR processing fire {fire['fod_id']}: {e}")
                failed_fires.append(fire)
    
    print(f"\nSuccessfully processed: {len(successful_fires)} fires")
    print(f"Failed: {len(failed_fires)} fires")
    
    # Step 6: Create config file
    print("\n[6/7] Creating config file...")
    config = {}
    for fire_info in successful_fires:
        fire = fire_info['fire']
        fire_key = f"offset_{fire_info['scenario_name'].replace('_scenario1', '')}"
        config[fire_key] = random.randint(1, 12)
    
    config_path = os.path.join(OUTPUT_DIR, "config_california_2020_day1_large_fires.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Step 7: Create summary
    print("\n[7/7] Creating dataset summary...")
    summary = {
        'dataset_name': 'California2020_Day1_LargeFires',
        'total_fires': len(non_urban_fires),
        'successful_fires': len(successful_fires),
        'failed_fires': len(failed_fires),
        'unique_wfpi_dates': len(fires_by_date),
        'grid_dimensions': {
            'height': int(grid_height),
            'width': int(grid_width)
        },
        'crs': str(wfpi_crs),
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
    print(f"WFPI maps: {len(fires_by_date)}")
    
    conn.close()

if __name__ == "__main__":
    main()
