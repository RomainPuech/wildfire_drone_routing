# Ignition-Point-Only Scenario Format

## Overview

This document describes the **ignition-point-only** scenario format, a storage-efficient alternative to full grid-based scenarios. Instead of storing complete `(T, H, W)` arrays where most values are zero, we store only the ignition point coordinates and generate the scenario grid on-the-fly during benchmarking.

## Storage Format

### File Structure

Each scenario is stored as a NumPy array with shape `(3,)` containing:
- `[0]`: Row index (int32) of the ignition point
- `[1]`: Column index (int32) of the ignition point  
- `[2]`: Starting timestep (int32, typically 0)

**File naming convention:** `{fire_name}_scenario{N}.npy`

**Example:**
```python
import numpy as np
# Save ignition point
ignition_point = np.array([521, 989, 0], dtype=np.int32)  # (row, col, start_timestep)
np.save("LakeFire_scenario1.npy", ignition_point)

# Load ignition point
ignition = np.load("LakeFire_scenario1.npy")
row, col, start_timestep = ignition[0], ignition[1], ignition[2]
```

### Storage Savings

**Before (full grid format):**
- Each scenario: `(12, H, W)` float32 array
- Example: `(12, 800, 1000)` = 38.4 MB per scenario
- 10,198 scenarios = **~366 GB**

**After (ignition-point-only format):**
- Each scenario: `(3,)` int32 array = 12 bytes
- 10,198 scenarios = **~0.12 MB** (negligible!)
- **Total dataset: ~1-2 GB** (dominated by WFPI maps and mask)

## Implementation Requirements

### 1. New Function: `load_scenario_ignition_point()`

**Location:** `code/dataset.py`

**Purpose:** Detect and load ignition-point-only scenarios, converting them to full grid format on-the-fly.

**Signature:**
```python
def load_scenario_ignition_point(filename, grid_height, grid_width, num_timesteps=12):
    """
    Load an ignition-point-only scenario and convert it to a full grid.
    
    Args:
        filename (str): Path to the .npy file containing ignition point
        grid_height (int): Height of the full grid (from mask or WFPI map)
        grid_width (int): Width of the full grid
        num_timesteps (int): Number of timesteps in the scenario (default: 12 for 6 hours)
    
    Returns:
        numpy.ndarray: Full scenario grid with shape (num_timesteps, grid_height, grid_width)
                     where the ignition point is set to 1.0 for all timesteps
    """
```

**Implementation:**
```python
def load_scenario_ignition_point(filename, grid_height, grid_width, num_timesteps=12):
    """
    Load an ignition-point-only scenario and convert it to a full grid.
    """
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    ignition_data = np.load(filename, allow_pickle=True)
    
    # Check if this is an ignition-point-only format (shape is (3,) or (2,))
    if ignition_data.shape in [(2,), (3,)]:
        # Extract ignition point coordinates
        row = int(ignition_data[0])
        col = int(ignition_data[1])
        start_timestep = int(ignition_data[2]) if len(ignition_data) > 2 else 0
        
        # Validate coordinates
        if not (0 <= row < grid_height and 0 <= col < grid_width):
            raise ValueError(
                f"Ignition point ({row}, {col}) is outside grid bounds "
                f"({grid_height}, {grid_width})"
            )
        
        # Create full scenario grid
        scenario = np.zeros((num_timesteps, grid_height, grid_width), dtype=np.float32)
        
        # Set ignition point to 1.0 for all timesteps from start_timestep onwards
        scenario[start_timestep:, row, col] = 1.0
        
        return scenario
    else:
        # Fall back to existing format (full grid)
        return load_scenario_npy(filename)
```

### 2. Modify `load_scenario_npy()` for Auto-Detection

**Location:** `code/dataset.py`

**Change:** Update `load_scenario_npy()` to automatically detect ignition-point-only format and delegate to the new function when needed.

**Modified Implementation:**
```python
def load_scenario_npy(filename, grid_height=None, grid_width=None, num_timesteps=12):
    """
    Load a scenario from a NumPy binary file.
    Auto-detects ignition-point-only format and converts to full grid.
    
    Args:
        filename (str): Name of the file to load (with or without .npy extension)
        grid_height (int, optional): Height of the grid (required for ignition-point format)
        grid_width (int, optional): Width of the grid (required for ignition-point format)
        num_timesteps (int, optional): Number of timesteps (default: 12)
    
    Returns:
        numpy.ndarray: TxNxM array representing the fire progression
    """
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    try:
        loaded_data = np.load(filename, allow_pickle=True)
        
        # Check if this is ignition-point-only format
        if loaded_data.shape in [(2,), (3,)]:
            if grid_height is None or grid_width is None:
                raise ValueError(
                    "grid_height and grid_width must be provided for "
                    "ignition-point-only scenarios"
                )
            return load_scenario_ignition_point(
                filename, grid_height, grid_width, num_timesteps
            )
        
        # Existing format handling
        if loaded_data.ndim > 0:  # Regular array (new format)
            return loaded_data
        else:  # 0-dim array containing dictionary (old format)
            return loaded_data.item()['scenario']
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error loading scenario: {str(e)}")
```

### 3. Update `get_automatic_layout_parameters()` to Pass Grid Dimensions

**Location:** `code/benchmark.py`

**Change:** When loading scenarios, we need to know the grid dimensions (from mask or WFPI map) before loading the scenario. The current flow loads the scenario first, then gets layout parameters. We need to reverse this or pass grid dimensions.

**Option A (Recommended):** Load mask/WFPI first to get dimensions, then load scenario:
```python
def get_automatic_layout_parameters(scenario: np.ndarray, input_dir: str, 
                                    simulation_parameters: dict, 
                                    scenario_name: str = ""):
    # ... existing code ...
    
    # Load mask to get grid dimensions (needed for ignition-point scenarios)
    mask_filename = os.path.join(os.path.abspath(input_dir), "mask.npy")
    if not os.path.exists(mask_filename):
        mask_filename = os.path.join(layout_dir, "mask.npy")
    
    grid_height, grid_width = None, None
    if mask_filename and os.path.exists(mask_filename):
        mask = np.load(mask_filename)
        grid_height, grid_width = mask.shape
    
    # If scenario is None or not yet loaded, we'll need dimensions
    # This happens when we're loading ignition-point scenarios
    if scenario is None and grid_height and grid_width:
        # Scenario will be loaded later with these dimensions
        pass
    
    return {
        # ... existing parameters ...
        "grid_height": grid_height,  # NEW: Add grid dimensions
        "grid_width": grid_width,    # NEW: Add grid dimensions
        "mask_filename": mask_filename,
        # ... rest of parameters ...
    }
```

**Option B:** Load mask/WFPI in `run_benchmark_scenario()` before loading scenario:
```python
def run_benchmark_scenario(...):
    # 0. Get layout parameters (but scenario might not be loaded yet)
    # Load mask/WFPI first to get dimensions
    layout_dir = os.path.abspath(os.path.join(input_dir, ".."))
    mask_filename = os.path.join(layout_dir, "mask.npy")
    
    if os.path.exists(mask_filename):
        mask = np.load(mask_filename)
        grid_height, grid_width = mask.shape
    else:
        # Fallback: try to infer from scenario if it's already loaded
        if scenario is not None:
            grid_height, grid_width = scenario.shape[1], scenario.shape[2]
        else:
            raise ValueError("Cannot determine grid dimensions")
    
    # Now load scenario if it's a path (ignition-point format)
    if isinstance(scenario, str):
        scenario = load_scenario_npy(
            scenario, 
            grid_height=grid_height, 
            grid_width=grid_width,
            num_timesteps=12
        )
    
    # Continue with existing logic...
```

### 4. Update Scenario Loading in Benchmark Functions

**Location:** `code/benchmark.py`

**Functions to modify:**
- `run_benchmark_scenarii_sequential()`
- `run_benchmark_scenarii_sequential_precompute()`
- `run_benchmark_scenarii_sequential_no_precompute()`

**Change:** Update the scenario loading logic to pass grid dimensions:

```python
# In run_benchmark_scenarii_sequential_precompute() and similar functions:

# Load mask to get grid dimensions
mask_filename = os.path.join(layout_dir, "mask.npy")
grid_height, grid_width = None, None
if os.path.exists(mask_filename):
    mask = np.load(mask_filename)
    grid_height, grid_width = mask.shape
else:
    # Fallback: load first scenario to get dimensions (if not ignition-point)
    first_scenario_path = os.path.join(input_dir, iterable[0])
    first_scenario = load_scenario_fn(first_scenario_path)
    grid_height, grid_width = first_scenario.shape[1], first_scenario.shape[2]

# Load scenarios with grid dimensions
for scenario_file in iterable:
    scenario_path = os.path.join(input_dir, scenario_file)
    scenario = load_scenario_fn(
        scenario_path,
        grid_height=grid_height,
        grid_width=grid_width,
        num_timesteps=12
    )
    # ... continue processing ...
```

### 5. Update `listdir_npy_limited()` if Needed

**Location:** `code/benchmark.py`

**Status:** No changes needed. The function already lists `.npy` files, which works for both formats.

### 6. Backward Compatibility

**Strategy:** Maintain full backward compatibility with existing full-grid scenarios.

**Implementation:**
- `load_scenario_npy()` auto-detects format based on array shape
- If shape is `(2,)` or `(3,)` → ignition-point format
- If shape is `(T, H, W)` or `(H, W)` → full grid format
- Existing datasets continue to work without modification

### 7. Dataset Creation Helper Function

**Location:** `code/dataset.py` (new function)

**Purpose:** Save scenarios in ignition-point-only format during dataset creation.

```python
def save_scenario_ignition_point(row, col, start_timestep=0, out_filename="scenario"):
    """
    Save an ignition point as a scenario file.
    
    Args:
        row (int): Row index of the ignition point
        col (int): Column index of the ignition point
        start_timestep (int): Timestep when fire starts (default: 0)
        out_filename (str): Output filename (.npy extension added if missing)
    """
    if not out_filename.endswith('.npy'):
        out_filename += '.npy'
    
    ignition_point = np.array([row, col, start_timestep], dtype=np.int32)
    np.save(out_filename, ignition_point)
```

## Testing Checklist

- [ ] Test loading ignition-point-only scenarios
- [ ] Test loading full-grid scenarios (backward compatibility)
- [ ] Test auto-detection in `load_scenario_npy()`
- [ ] Test scenario loading in `run_benchmark_scenarii_sequential()`
- [ ] Test with existing `MiniTractDataset` (should still work)
- [ ] Test with new California 2020 dataset
- [ ] Verify fire detection logic works correctly with generated grids
- [ ] Verify scenario videos can be generated (if applicable)

## Migration Notes

**For existing datasets:**
- No migration needed. Existing full-grid scenarios continue to work.

**For new datasets:**
- Use `save_scenario_ignition_point()` when creating scenarios
- Ensure mask/WFPI map is available to provide grid dimensions during loading

## Performance Considerations

**Memory:**
- **Before:** Each scenario loaded into memory as full `(12, H, W)` array
- **After:** Ignition point loaded as `(3,)` array, grid generated on-demand
- **Benefit:** Reduced memory footprint when processing many scenarios

**Computation:**
- Grid generation is O(1) per scenario (just setting one cell)
- Negligible overhead compared to simulation loop

## Example Usage

```python
# Dataset creation
from code.dataset import save_scenario_ignition_point

# Save ignition point for a fire
save_scenario_ignition_point(
    row=521, 
    col=989, 
    start_timestep=0,
    out_filename="California2020/Fire_001_scenario1.npy"
)

# Benchmark loading (automatic)
from code.dataset import load_scenario_npy
from code.benchmark import run_benchmark_scenario

# Load scenario (auto-detects format)
scenario = load_scenario_npy(
    "California2020/Fire_001_scenario1.npy",
    grid_height=800,
    grid_width=1000,
    num_timesteps=12
)

# scenario is now a (12, 800, 1000) array with ignition point set
# Continue with normal benchmarking...
```

## Summary

The ignition-point-only format reduces scenario storage from **~366 GB to ~0.12 MB** (a **99.97% reduction**) while maintaining full functionality. The implementation requires:

1. New `load_scenario_ignition_point()` function
2. Auto-detection in `load_scenario_npy()`
3. Grid dimension passing through benchmark functions
4. Backward compatibility with existing datasets

The total dataset size is now dominated by WFPI maps (~1-2 GB) rather than scenarios, making the California 2020 dataset much more manageable.
