---
language:
- en
license: cc-by-4.0
task_categories:
- other
tags:
- wildfire
- drones
- benchmark
- routing
- sensor-placement
- risk-map
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/scenarios_index.parquet
- config_name: tables23
  data_files:
  - split: test
    path: data/tables23_scenarios.parquet
---

# DroneBench Dataset (clean release)

Wildfire monitoring layouts for benchmarking sensor placement and drone routing
(NeurIPS 2026 Datasets & Benchmarks anonymous submission).

## Summary
| Attribute | Value |
|-----------|-------|
| Layouts in data archive | 56 (49 + 7 tables23 extras) |
| Tables 2/3 split | 471 scenarios / 12 layouts |
| Fire-spread | JPG in `Satellite_Images_Mask/` |
| Risk maps | `burn_map.npy` (oracle), `static_risk_bp2024.npy` (BP) |

## Large archives
- `DroneBench_data.tar` — layouts (JPG + burn/BP + ancillary); no `scenarii/*.npy`

## Loading indices
```python
from datasets import load_dataset
REPO = "anonymoussubmission2/anonymous-submission-neurips26-2831"
ds = load_dataset(REPO, "default")       # full index
ds_t23 = load_dataset(REPO, "tables23")  # Tables 2/3 split
```

Selection rule for the Tables 2/3 split: layouts with ≥80% historical ignition match
(`splits/SELECTION_RULE.md`).

## Risk-map encoding (important)

- **`static_risk_bp2024.npy`** — USFS Burn Probability (BP), the independent risk map used in **Table 2**.
  Stored as **`int16`** with raw grid values typically in **~11–385** (not unit probabilities).
  These are FSim annual burn probability × **10 000** (USFS BP scaling).
  Convert to \([0,1]\) with:
  ```python
  import numpy as np
  bp = np.load("static_risk_bp2024.npy").astype(np.float32) / 10000.0
  ```
- **`burn_map.npy`** — oracle / ground-truth **dynamic** risk map used in **Table 3**.
  Stored as **`float32`** in **[0, 1]**, shape `(T, H, W)` (mean burn fraction over scenarios at each time).

Legacy `static_risk.npy` is **not** shipped (it was a duplicate of BP under another name).

## Fire-spread JPG encoding / thresholding

Scenarios live in `Satellite_Images_Mask/<layout>_<scenario>/` as **grayscale JPG** frames (lossy).

Official preprocessing (`code/dataset.py` → `load_scenario_jpg`, used by `preprocess_sim2real_dataset` / `jpg_scenario_to_npy`):

1. Load each frame as grayscale (`L`).
2. Normalize: `pixel / 255.0` → float in \([0,1]\).
3. **Binarize with threshold 0.5**: `(arr >= 0.5).astype(float)`.
4. Stack frames → NPY array shape `(T, H, W)`.

JPEG compression can introduce mild artifacts; regenerating `scenarii/*.npy` with the library before experiments is recommended. Weather and other modalities in the release are unused by the current baseline strategies.

## Ignition time offsets (`config_s2r.json`)

`config_s2r.json` maps `offset_<layout>_<scenario>` → integer hours to prepend empty frames when building burn maps / aligning scenarios. Coverage: all **7 746** scenarios in the default index and all **56** layouts (plus 7 single-scenario extras → 7 753 offset keys).

## Citation
```bibtex
@inproceedings{anonymous_neurips2026_dronebench,
  title     = {A Benchmark for Sensor Placement and Drone Routing for Wildfire Detection},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026},
  note      = {Anonymous submission}
}
```
