#!/usr/bin/env python3
"""
Plot 50 fires on their corresponding WFPI maps.

Creates individual PNG files showing each fire's location on its WFPI map.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta

# Add code directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
code_dir = os.path.join(project_root, "code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../California2020Dataset")
FIRE_DB_PATH = os.path.join(DATA_DIR, "RDS-2013-0009.6_Data_Format3_GPKG/FPA_FOD_20221014.gpkg")

def parse_fire_date(date_str):
    """Parse fire discovery date from various formats."""
    if not date_str or date_str == '':
        return None
    
    formats = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y"]
    for fmt in formats:
        try:
            date_part = date_str.split()[0] if ' ' in date_str else date_str
            return datetime.strptime(date_part, fmt)
        except:
            continue
    return None

def get_wfpi_date_for_fire(fire_date):
    """Get the WFPI date to use for a fire (day before)."""
    if fire_date is None:
        return None
    day_before = fire_date - timedelta(days=1)
    return day_before.strftime("%Y%m%d")

def main():
    print("=" * 80)
    print("Plotting 50 fires on their WFPI maps")
    print("=" * 80)
    
    # Load dataset summary to get successful fires
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load fire database
    conn = sqlite3.connect(FIRE_DB_PATH)
    cursor = conn.cursor()
    
    # Get successful fires from scenarios directory
    scenario_files = list(Path(OUTPUT_DIR, "scenarii").glob("*.npy"))
    print(f"\nFound {len(scenario_files)} scenario files")
    
    # Select 50 random fires
    import random
    selected_scenarios = random.sample(scenario_files, min(50, len(scenario_files)))
    
    # Create output directory for plots
    plots_dir = os.path.join(OUTPUT_DIR, "fire_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load mask to get dimensions
    mask = np.load(os.path.join(OUTPUT_DIR, "mask.npy"))
    mask_height, mask_width = mask.shape
    
    print(f"\nProcessing {len(selected_scenarios)} fires...")
    
    for idx, scenario_file in enumerate(selected_scenarios, 1):
        try:
            # Extract FOD_ID from filename (format: FireName_FODID_scenario1.npy)
            filename_parts = scenario_file.stem.replace("_scenario1", "").rsplit("_", 1)
            if len(filename_parts) < 2:
                continue
            
            fod_id = filename_parts[-1]
            
            # Load ignition point
            ignition = np.load(scenario_file)
            row, col = int(ignition[0]), int(ignition[1])
            
            # Get fire info from database
            cursor.execute("""
                SELECT FIRE_NAME, DISCOVERY_DATE, LATITUDE, LONGITUDE, FIRE_SIZE
                FROM Fires
                WHERE FOD_ID = ?
            """, (fod_id,))
            
            fire_info = cursor.fetchone()
            if not fire_info:
                continue
            
            fire_name, discovery_date, lat, lon, fire_size = fire_info
            parsed_date = parse_fire_date(discovery_date)
            wfpi_date = get_wfpi_date_for_fire(parsed_date)
            
            # Load WFPI map
            wfpi_path = os.path.join(OUTPUT_DIR, f"wfpi_{wfpi_date}.npy")
            if not os.path.exists(wfpi_path):
                continue
            
            wfpi_data = np.load(wfpi_path)
            if wfpi_data.ndim == 3:
                wfpi_2d = wfpi_data[0]
            else:
                wfpi_2d = wfpi_data
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot WFPI map
            wfpi_masked = wfpi_2d.copy()
            wfpi_masked[mask == 0] = np.nan  # Mask invalid areas
            
            im = ax.imshow(wfpi_masked, cmap='inferno', vmin=0, 
                          vmax=np.nanpercentile(wfpi_2d[wfpi_2d > 0], 98))
            
            # Plot fire location
            ax.plot(col, row, 'r*', markersize=20, markeredgecolor='none', 
                   label='Fire Location', zorder=10)
            
            # Add circle around fire for visibility
            circle = plt.Circle((col, row), radius=50, fill=False, 
                              edgecolor='red', linewidth=2, linestyle='--', zorder=9)
            ax.add_patch(circle)
            
            # Add title and info
            safe_fire_name = fire_name or f"Fire_{fod_id}"
            title = f"{safe_fire_name}\nFOD ID: {fod_id}"
            if parsed_date:
                title += f"\nDiscovery: {parsed_date.strftime('%Y-%m-%d')}"
            if fire_size:
                title += f" | Size: {fire_size:.1f} acres"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            ax.set_xlabel('Column (pixels)', fontsize=12)
            ax.set_ylabel('Row (pixels)', fontsize=12)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Wildland Fire Potential Index', fontsize=12)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            
            # Add grid info
            info_text = f"WFPI Date: {wfpi_date}\nGrid: {mask_height}×{mask_width}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure
            safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_fire_name)
            output_path = os.path.join(plots_dir, f"{idx:02d}_{safe_filename}_{fod_id}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(selected_scenarios)} fires...")
                
        except Exception as e:
            print(f"  ERROR processing {scenario_file}: {e}")
            continue
    
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"Successfully created plots for {len(selected_scenarios)} fires")
    print(f"Plots saved to: {plots_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
