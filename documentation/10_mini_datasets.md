# Mini California 2020 Datasets

## Overview

The **Mini California 2020 Datasets** are reduced versions of the full California 2020 wildfire datasets, each containing **10 randomly selected fires**. These mini datasets are designed for quick benchmarking and testing of routing strategies without the computational overhead of processing thousands of fires.

## Location

All mini datasets are located in:
```
MiniCalifornia2020Datasets/
```

## Available Mini Datasets

The following 8 mini datasets have been created:

1. **California2020Dataset** - WFPI Day 2 forecast, all fires
2. **California2020Dataset_Day1** - WFPI Day 1 forecast, all fires
3. **California2020Dataset_LargeFires** - WFPI Day 2 forecast, large fires only (>= 100 acres)
4. **California2020Dataset_Day1_LargeFires** - WFPI Day 1 forecast, large fires only (>= 100 acres)
5. **California2020Dataset_IgnitionProb** - Ignition Probability map, all fires
6. **California2020Dataset_IgnitionProb_LargeFires** - Ignition Probability map, large fires only (>= 100 acres)
7. **California2020Dataset_BurnProb** - Burn Probability (FSim) map, all fires
8. **California2020Dataset_BurnProb_LargeFires** - Burn Probability (FSim) map, large fires only (>= 100 acres)

## Dataset Structure

Each mini dataset follows the same structure as the full dataset:

```
MiniCalifornia2020Datasets/
└── {DatasetName}/
    ├── mask.npy                          # California mask (1 = valid, 0 = invalid)
    ├── {risk_map}.npy                    # Risk map (WFPI or static risk)
    ├── scenarii/                         # Ignition-point scenarios
    │   ├── FireName_FODID_scenario1.npy  # 10 selected scenarios
    │   └── ...
    ├── config_{dataset_name}.json        # Configuration with offsets for selected fires
    └── dataset_summary.json              # Dataset metadata
```

### Risk Map Files

- **WFPI datasets**: Multiple `wfpi_YYYYMMDD.npy` files (one per unique fire date in selected fires)
- **Ignition Probability datasets**: `static_risk_ignition_prob.npy`
- **Burn Probability datasets**: `static_risk_burn_prob.npy`

## Dataset Characteristics

### Fire Selection

- **Selection method**: Random sampling (seed = 42 for reproducibility)
- **Number of fires**: 10 fires per dataset
- **Selection criteria**: Same as full dataset (CA 2020, non-prescriptive, non-urban)

### Configuration

Each mini dataset includes a configuration file with random offsets (1-12) for each selected fire, matching the format of the full dataset.

### Summary Metadata

Each mini dataset includes a `dataset_summary.json` file with:
- `is_mini_dataset: true`
- `source_dataset`: Name of the source full dataset
- `total_fires`: 10
- `selected_fires`: List of fire keys for the selected scenarios
- Other metadata from the source dataset (grid dimensions, CRS, resolution, etc.)

## Usage

### Loading a Mini Dataset

Mini datasets can be loaded using the same code as full datasets:

```python
from code.dataset import load_scenario_npy
import numpy as np

# Load a scenario
scenario_path = "MiniCalifornia2020Datasets/California2020Dataset_BurnProb/scenarii/FireName_FODID_scenario1.npy"
ignition = np.load(scenario_path)
row, col, start_timestep = ignition[0], ignition[1], ignition[2]

# Load mask
mask = np.load("MiniCalifornia2020Datasets/California2020Dataset_BurnProb/mask.npy")

# Load risk map
risk_map = np.load("MiniCalifornia2020Datasets/California2020Dataset_BurnProb/static_risk_burn_prob.npy")
```

### Benchmarking

Mini datasets are ideal for:
- **Quick testing** of new strategies
- **Development** and debugging
- **Rapid iteration** on algorithm improvements
- **Comparison** across different risk maps

For final evaluation, use the full datasets.

## Creation Script

Mini datasets were created using:
```bash
python code/dataset_creation/nature_dataset_creation/create_mini_datasets.py
```

The script:
1. Randomly selects 10 fires from each full dataset (seed = 42)
2. Copies necessary files (mask, risk map, config, selected scenarios)
3. Updates config to include only selected fires
4. Updates summary metadata

## Reproducibility

The random selection uses a fixed seed (42), ensuring:
- Same fires are selected each time the script is run
- Results are reproducible across different runs
- Mini datasets remain consistent

## File Sizes

Approximate sizes per mini dataset:
- **WFPI datasets**: ~50-100 MB (includes multiple WFPI maps)
- **Static risk datasets**: ~20-30 MB (single risk map)
- **Total**: ~400-500 MB for all 8 mini datasets

## Comparison with Full Datasets

| Dataset | Full Dataset Fires | Mini Dataset Fires | Reduction |
|---------|-------------------|-------------------|-----------|
| California2020Dataset | 4,166 | 10 | 99.8% |
| California2020Dataset_Day1 | 2,325 | 10 | 99.6% |
| California2020Dataset_LargeFires | 146 | 10 | 93.2% |
| California2020Dataset_Day1_LargeFires | 137 | 10 | 92.7% |
| California2020Dataset_IgnitionProb | 4,547 | 10 | 99.8% |
| California2020Dataset_IgnitionProb_LargeFires | 210 | 10 | 95.2% |
| California2020Dataset_BurnProb | 4,550 | 10 | 99.8% |
| California2020Dataset_BurnProb_LargeFires | 211 | 10 | 95.3% |

## Notes

- **WFPI maps**: For WFPI datasets, all WFPI maps from the source dataset are copied (not just those needed for the 10 selected fires). This ensures compatibility but increases file size. Future optimization could copy only needed maps.

- **Scenario format**: All mini datasets use the ignition-point-only format (see [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md)).

- **Mask and risk maps**: These are identical to the full datasets (not cropped or modified).

## References

- Full dataset documentation: [04_california_2020_dataset.md](04_california_2020_dataset.md)
- Ignition point format: [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md)
- Risk map comparison: [09_risk_map_comparison.md](09_risk_map_comparison.md)
