# Auto-generated counts (see paper_2021_dataset_creation_explainer.md in this folder)

## Mask steps (cell counts)

| Step | Valid cells |
|------|-------------|
| (1) California boundary | 426,469 |
| (2) Excluding urban | 406,688 |
| (3) Excluding always WFPI-invalid | 272,168 |
| (4) Components ≥ 9×9 km² + 1 px dilation | 283,790 |

## 2021 USFS wildfires (ignition points)

| Category | Count |
|----------|-------|
| Raw CA wildfires (CSV filters) | 1,086 |
| Outside CA boundary | 13 |
| Urban | 25 |
| Non-urban, off grid or mask | 10 |
| Excluded — missing WFPI zip date | 56 |
| **In dataset (filter pipeline)** | **982** |
| Scenario files `*.npy` on disk | 981 |

## Benchmark pool

| Scenarios with date/time in config | 981 |
| Random benchmark subset | 100 (seed 42) |
