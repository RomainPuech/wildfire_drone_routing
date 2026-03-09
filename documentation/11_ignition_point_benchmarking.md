# Ignition-Point-Only Benchmarking Support

## Overview

This document describes the implementation of support for ignition-point-only scenario format in the benchmarking system. This allows the benchmarking code to work with the new California 2020 datasets that use a storage-efficient ignition-point format instead of full grid arrays.

## Changes Made

### 1. Dataset Loading (`code/dataset.py`)

#### New Function: `load_scenario_ignition_point()`

Added a new function to load ignition-point-only scenarios and convert them to full grids:

```python
def load_scenario_ignition_point(filename, grid_height, grid_width, num_timesteps=12):
    """
    Load an ignition-point-only scenario and convert it to a full grid.
    
    Args:
        filename (str): Path to the .npy file containing ignition point
        grid_height (int): Height of the full grid (from mask or risk map)
        grid_width (int): Width of the full grid
        num_timesteps (int): Number of timesteps in the scenario (default: 12)
    
    Returns:
        numpy.ndarray: Full scenario grid with shape (num_timesteps, grid_height, grid_width)
    """
```

**Behavior:**
- Loads the ignition point coordinates `[row, col, start_timestep]` from the `.npy` file
- Validates coordinates are within grid bounds
- Creates a full `(num_timesteps, grid_height, grid_width)` array of zeros
- Sets the ignition point cell to 1.0 for all timesteps from `start_timestep` onwards

#### Updated Function: `load_scenario_npy()`

Modified to auto-detect ignition-point format and delegate to the new function:

```python
def load_scenario_npy(filename, grid_height=None, grid_width=None, num_timesteps=12):
    """
    Load a scenario from a NumPy binary file.
    Auto-detects ignition-point-only format and converts to full grid.
    """
```

**Auto-Detection Logic:**
- Checks if loaded data has shape `(2,)` or `(3,)` → ignition-point format
- If ignition-point format and `grid_height`/`grid_width` provided → calls `load_scenario_ignition_point()`
- Otherwise → uses existing full-grid loading logic
- **Backward compatible:** Existing full-grid scenarios continue to work

### 2. Benchmark Functions (`code/benchmark.py`)

#### Updated Function: `get_automatic_layout_parameters()`

Modified to handle cases where scenario might be `None` (for ignition-point format):

```python
def get_automatic_layout_parameters(scenario: np.ndarray, input_dir: str, ...):
    # Get grid dimensions from scenario if available, otherwise from mask
    if scenario is not None and scenario.ndim >= 2:
        N, M = scenario.shape[1], scenario.shape[2]
    elif mask_filename and os.path.exists(mask_filename):
        mask = np.load(mask_filename)
        N, M = mask.shape[0], mask.shape[1]
    else:
        raise ValueError("Cannot determine grid dimensions")
```

**Key Change:**
- Can now determine grid dimensions from mask when scenario is not yet loaded
- This is necessary for ignition-point scenarios where we need dimensions before loading

### 3. New Benchmark Script (`run_benchmark_california2020.py`)

Created a new benchmark script specifically for California 2020 datasets:

**Key Features:**
- **Multi-dataset support:** Benchmarks across all mini datasets in `MiniCalifornia2020Datasets/`
- **Ignition-point handling:** Loads mask first to get grid dimensions, then loads scenarios
- **Risk map detection:** Automatically finds static risk maps (`static_risk_*.npy`) or WFPI maps (`wfpi_*.npy`)
- **Config loading:** Loads offset configuration from `config_*.json` files
- **Same strategy combinations:** Uses the same 6 strategy combinations as `run_benchmark.py`
- **Results aggregation:** Groups results by strategy combo and dataset

**Workflow:**
1. Iterate through all datasets in `MiniCalifornia2020Datasets/`
2. For each dataset:
   - Load mask to get grid dimensions
   - Find risk map (static or WFPI)
   - Load config file for offsets
   - For each scenario:
     - Load scenario with grid dimensions (auto-detects ignition-point format)
     - Get offset from config
     - Run benchmark
     - Save results and generate video

## Usage

### Running the Benchmark

```bash
python -u run_benchmark_california2020.py
```

### Expected Output

The script will:
1. Process all 8 mini datasets
2. Run 6 strategy combinations on each dataset
3. Generate results CSV: `benchmark_results_california2020_YYYYMMDD_HHMMSS.csv`
4. Create videos for each scenario: `display_benchmark_*_*/`

### Results Structure

Results include:
- **Metadata:** strategy_combo, sensor_strategy, drone_strategy, dataset_name, scenario_name
- **Metrics:** delta_t, device, fire_size_cells, fire_size_percentage, etc.
- **Averaged results:** Per strategy combo and dataset

## Backward Compatibility

All changes maintain **full backward compatibility**:

- ✅ Existing full-grid scenarios (e.g., `MiniTractDataset`) continue to work
- ✅ `load_scenario_npy()` auto-detects format based on array shape
- ✅ No changes required to existing benchmark scripts
- ✅ Existing datasets don't need migration

## Dataset Format Detection

The system automatically detects scenario format:

| Format | Array Shape | Detection | Action |
|--------|-------------|-----------|--------|
| **Ignition-point** | `(2,)` or `(3,)` | Auto-detected | Convert to full grid |
| **Full grid (new)** | `(T, H, W)` | Auto-detected | Use directly |
| **Full grid (old)** | 0-dim dict | Auto-detected | Extract `scenario` key |

## Limitations and Future Improvements

### Current Limitations

1. **WFPI Map Selection:** For WFPI datasets, the script currently uses the first WFPI map found. Ideally, it should load the WFPI map matching each scenario's fire date.

2. **Burn Map Creation:** For WFPI datasets, we use the WFPI map directly as the burn map. This works but might not be optimal.

### Future Improvements

1. **Date-based WFPI Loading:** Extract fire date from scenario name/config and load matching WFPI map
2. **Burn Map Generation:** Create proper burn maps from WFPI data if needed
3. **Performance Optimization:** Cache loaded scenarios if running multiple strategies on same dataset

## Testing

### Test Cases

1. ✅ Load ignition-point scenario with grid dimensions → converts to full grid
2. ✅ Load full-grid scenario → works as before
3. ✅ Auto-detection in `load_scenario_npy()` → correctly identifies format
4. ✅ Benchmark script runs on mini datasets → processes all scenarios
5. ✅ Backward compatibility → existing datasets still work

### Verification

To verify the implementation:

```python
from code.dataset import load_scenario_npy
import numpy as np

# Test ignition-point loading
mask = np.load("MiniCalifornia2020Datasets/California2020Dataset_BurnProb/mask.npy")
h, w = mask.shape

scenario = load_scenario_npy(
    "MiniCalifornia2020Datasets/California2020Dataset_BurnProb/scenarii/Fire_001_scenario1.npy",
    grid_height=h,
    grid_width=w,
    num_timesteps=12
)

print(f"Scenario shape: {scenario.shape}")  # Should be (12, H, W)
print(f"Ignition point: {np.where(scenario[0] > 0)}")  # Should show one cell
```

## Summary

The ignition-point-only format support has been successfully integrated into the benchmarking system:

- ✅ **New function:** `load_scenario_ignition_point()` for format conversion
- ✅ **Auto-detection:** `load_scenario_npy()` automatically detects format
- ✅ **Grid dimensions:** `get_automatic_layout_parameters()` handles mask-based dimensions
- ✅ **New benchmark script:** `run_benchmark_california2020.py` for California 2020 datasets
- ✅ **Backward compatible:** All existing functionality preserved

The system is ready to benchmark strategies on the new California 2020 datasets while maintaining full compatibility with existing datasets.
