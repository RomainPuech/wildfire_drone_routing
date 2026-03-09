#!/usr/bin/env python3
"""
Visualize California 2020 Dataset Mask

Shows the operational zones (valid areas) with urban areas and islands removed.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add code directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
code_dir = os.path.join(project_root, "code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../California2020Dataset")

def visualize_mask_and_wfpi():
    """Visualize the mask with a sample WFPI map."""
    
    # Load mask
    mask_path = os.path.join(OUTPUT_DIR, "mask.npy")
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask file not found at {mask_path}")
        return
    
    mask = np.load(mask_path)
    print(f"Mask shape: {mask.shape}")
    print(f"Valid cells: {np.sum(mask == 1):,} ({100*np.sum(mask == 1)/mask.size:.1f}%)")
    print(f"Masked cells: {np.sum(mask == 0):,} ({100*np.sum(mask == 0)/mask.size:.1f}%)")
    
    # Find a sample WFPI file
    wfpi_files = list(Path(OUTPUT_DIR).glob("wfpi_*.npy"))
    if not wfpi_files:
        print("ERROR: No WFPI files found")
        return
    
    # Load first WFPI file
    sample_wfpi = wfpi_files[0]
    wfpi_data = np.load(sample_wfpi)
    wfpi_date = sample_wfpi.stem.replace("wfpi_", "")
    print(f"\nUsing WFPI from: {wfpi_date}")
    
    # WFPI is (1, H, W), extract first band
    if wfpi_data.ndim == 3:
        wfpi_2d = wfpi_data[0]
    else:
        wfpi_2d = wfpi_data
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: WFPI map
    ax1 = axes[0]
    im1 = ax1.imshow(wfpi_2d, cmap='inferno', vmin=0, vmax=np.nanpercentile(wfpi_2d[wfpi_2d > 0], 98))
    ax1.set_title(f'WFPI Day 2 Forecast\n{wfpi_date}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Fire Potential Index', fraction=0.046, pad=0.04)
    
    # Plot 2: Mask
    ax2 = axes[1]
    im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('Operational Mask\n(Green=Valid, Red=Masked)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Mask Value', fraction=0.046, pad=0.04, ticks=[0, 1])
    
    # Plot 3: WFPI masked by operational area
    ax3 = axes[2]
    wfpi_masked = wfpi_2d.copy()
    wfpi_masked[mask == 0] = np.nan  # Set masked areas to NaN
    im3 = ax3.imshow(wfpi_masked, cmap='inferno', vmin=0, vmax=np.nanpercentile(wfpi_2d[wfpi_2d > 0], 98))
    ax3.set_title('WFPI in Operational Zones\n(Urban areas & islands removed)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='Fire Potential Index', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "mask_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Also create a detailed view
    fig2, ax = plt.subplots(figsize=(12, 12))
    
    # Create a combined visualization
    # Use WFPI as base, overlay mask as transparency
    wfpi_normalized = (wfpi_2d - np.nanmin(wfpi_2d[wfpi_2d > 0])) / (np.nanmax(wfpi_2d[wfpi_2d > 0]) - np.nanmin(wfpi_2d[wfpi_2d > 0]) + 1e-10)
    wfpi_normalized[wfpi_2d <= 0] = 0
    
    # Create RGB image: WFPI in red channel, mask in green channel
    rgb = np.zeros((*wfpi_2d.shape, 3))
    rgb[:, :, 0] = wfpi_normalized  # Red = WFPI
    rgb[:, :, 1] = mask  # Green = Valid operational area
    rgb[:, :, 2] = 0.2  # Blue = constant
    
    ax.imshow(rgb)
    ax.set_title(f'California 2020 Operational Zones\nWFPI (Red) + Valid Areas (Green)\nDate: {wfpi_date}', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Valid Operational Area'),
        Patch(facecolor='red', label='Masked (Urban/Invalid/Islands)'),
        Patch(facecolor='yellow', label='High WFPI in Valid Area')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    output_path2 = os.path.join(OUTPUT_DIR, "mask_visualization_detailed.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved detailed visualization to: {output_path2}")
    
    plt.show()

if __name__ == "__main__":
    visualize_mask_and_wfpi()
