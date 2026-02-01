#!/usr/bin/env python3
"""
Quick script to create rescaled mask file for testing.
Run from the project root: python create_rescaled_mask.py
"""
import numpy as np

def pool_mask_min(mask, kernel_size):
    """
    Pool the mask to the new size by using a min operation.
    """
    N, M = mask.shape
    N_new = N // kernel_size
    M_new = M // kernel_size
    mask_pooled = np.zeros((N_new, M_new))
    for i in range(N_new):
        for j in range(M_new):
            mask_pooled[i, j] = np.min(mask[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size])
    return mask_pooled

# Parameters matching the benchmark
coverage_radius_m = 300
cell_size_m = 30
coverage_width_cells = round(coverage_radius_m * 2 / cell_size_m)  # = 20
operational_substeps = 63

print(f"Coverage width cells: {coverage_width_cells}")

# Load and rescale mask
mask_path = "MiniTractDataset/AugustComplexFire/mask.npy"
mask = np.load(mask_path)
print(f"Original mask shape: {mask.shape}")

rescaled_mask = pool_mask_min(mask, coverage_width_cells)
print(f"Rescaled mask shape: {rescaled_mask.shape}")

# Save
N_new, M_new = rescaled_mask.shape
output_path = f"MiniTractDataset/AugustComplexFire/mask_rescaled_{N_new}x{M_new}_{operational_substeps}substeps.npy"
np.save(output_path, rescaled_mask)
print(f"Saved rescaled mask to: {output_path}")

# Stats
print(f"Feasible cells: {np.sum(rescaled_mask == 1)}")
print(f"Blocked cells: {np.sum(rescaled_mask == 0)}")
