# WFDroneBench dataset encoding

This document specifies how risk maps and fire-spread trajectories are stored in the released WFDroneBench layouts, how they relate to the paper’s model notation \(r_{it} \in [0,1]\), and which loader/transform code paths the benchmark applies before optimization.

## Layout structure

Each layout folder (e.g. `0111_03612/` in the full release, or `MinimalDataset/0001/` in the repo sample) typically contains:

```
layout_folder/
├── Satellite_Images_Mask/   # fire-spread trajectories (JPG frame sequences)
│   └── <scenario_id>/       # one folder per scenario, frames sorted naturally
├── scenarii/                # optional preprocessed fire trajectories (.npy)
├── burn_map.npy             # dynamic “ground-truth” risk map (computed)
├── static_risk.npy          # legacy static risk filename (see below)
├── static_risk_bp2024.npy   # USDA Burn Probability (BP), used in Table 2 experiments
├── static_risk_whp.npy      # USDA Wildfire Hazard Potential (WHP) index
├── Weather_Data/
└── …                        # fuel, topography, vegetation, satellite preview, etc.
```

**Minimal examples in this repository:** see [`MinimalDataset/0001/`](MinimalDataset/0001/) and [`MinimalDataset/0002/`](MinimalDataset/0002/). Layout `0001` includes JPG masks, preprocessed `scenarii/*.npy`, and `burn_map.npy`. Layout `0002` adds benchmark log outputs under `logs/`.

---

## Risk maps

### Paper model vs on-disk files

In the paper, \(r_{it} \in [0,1]\) denotes the **model interface**: a burn probability (or risk weight) at grid cell \(i\) and time \(t\). Optimization objectives in Appendix E are written for values in this range.

**On-disk arrays may use different native encodings.** The loaders below read values **as stored** (no automatic unit conversion to \([0,1]\)). To reproduce the paper’s probabilistic interpretation when loading data outside the benchmark pipeline, apply the native-unit conversions in the table below before treating entries as probabilities.

| File | Role in experiments | Typical shape | Typical dtype | Native units (USFS / dataset) | Maps to paper \(r_{it}\) |
|------|---------------------|---------------|---------------|--------------------------------|---------------------------|
| `static_risk.npy` | Legacy release name (same role as BP in older zips) | `(1, N, M)` or `(N, M)` | `int16` | **FSim annual burn probability × 10 000** (USFS BP scaling). Observed range in the release is roughly 11–385, i.e. about 0.001–0.039 annual BP. | \(r = \texttt{value} / 10000\) |
| `static_risk_bp2024.npy` | Static BP map for Table 2 (`--bm_prefix bp`) | `(1, N, M)` | `int16` or `float32` | Same BP source as above, cropped/resampled to the layout grid. | \(r = \texttt{value} / 10000\) if integer; already in \([0,1]\) if float |
| `static_risk_whp.npy` | WHP ablations (`--bm_prefix whp`; see `paper/.../table.md`) | `(1, N, M)` | `int16` | **USDA WHP continuous integer index** (relative high-intensity fire potential; not a probability). USFS documents baseline indices up to ~100 000. | Use as a **non-negative risk weight**; not a calibrated \(r_{it}\) unless you define a normalization (e.g. divide by layout max) |
| `burn_map.npy` | Dynamic ground-truth map for Table 3 (`--bm_prefix bm`) | `(T, N, M)` | `float32` | **Empirical burn fraction** per cell and hour: mean of binary fire masks across scenarios (see below). | Already in \([0,1]\) |

**USFS references:** Burn Probability and WHP rasters from the USDA Forest Service FSim / WHP products ([WHP 2023 metadata](https://doi.org/10.2737/RDS-2015-0047-4)). BP is an annual burn probability; WHP is a composite hazard index derived from BP and flame-length probabilities.

Filenames used by the experiment driver are defined in [`all_experiments_parallel.py`](all_experiments_parallel.py) (`BM_PREFIX_TO_NAME`).

---

### Loading (Python)

| Step | Function | File |
|------|----------|------|
| Load any `.npy` risk/scenario array | `load_scenario_npy` | [`code/dataset.py`](code/dataset.py) |
| Unified loader (`.npy` path or JPG folder) | `load_scenario` / `load_burn_map` | [`code/dataset.py`](code/dataset.py) |
| Build dynamic `burn_map.npy` from scenarios | `compute_burn_map` → `save_burn_map` | [`code/dataset.py`](code/dataset.py) |

`load_burn_map(filename)` is a thin alias: it calls `load_scenario(..., extension=".npy")` and returns a `T×N×M` array unchanged.

Static maps with a single time slice (`T=1`) are **time-duplicated inside strategies** before Julia routing (e.g. `DroneRoutingUniformCoverageResetStatic` in [`code/Strategy.py`](code/Strategy.py) expands the map to a long horizon for rolling re-optimization).

### Loading (Julia)

Julia optimizers read the same `.npy` files via NPZ:

- `load_burn_map(filename)` in [`julia/helper_functions.jl`](julia/helper_functions.jl) — no value scaling; optional `static_map=true` repeats `(1,N,M)` → `(100,N,M)`.
- Sensor placement: `SENSOR_MAXCOV_STRATEGY`, `Max_Coverage_Kernel` in [`julia/ground_charging_opt.jl`](julia/ground_charging_opt.jl).
- Drone routing: `create_index_routing_model` in [`julia/drone_routing_opt.jl`](julia/drone_routing_opt.jl).

### Transforms applied **before optimization** (benchmark pipeline)

After loading, [`code/benchmark.py`](code/benchmark.py) **does not** rescale USFS integer units to \([0,1]\). It applies **grid/time rescaling** tied to drone coverage and sub-stepping:

1. **Load** — `load_burn_map(burnmap_filename)` (raw values).
2. **Spatial pooling** to the operational grid (one cell per drone coverage footprint):
   - `run_benchmark_scenario`: `pool_burnmap_mean` (arithmetic mean over each `coverage_width_cells × coverage_width_cells` block).
   - `run_drone_routing_strategy`: `pool_burnmap_proba_at_least_one` using \(1 - \prod(1-p)\) over each block (assumes block entries behave like probabilities in \([0,1]\); appropriate for `burn_map.npy`, less so for raw integer BP unless pre-normalized).
3. **Temporal refinement** — `np.repeat(..., operational_substeps, axis=0) / operational_substeps` so one data timestep (1 hour) is split into `operational_substeps` optimizer steps.
4. **Persist** rescaled map next to the source file as `*_rescaled_{N}x{M}_{k}substeps.npy` and pass that path to Julia.

Relevant symbols: `compute_operational_substeps`, `coverage_width_cells` from drone `coverage_radius_m` and `cell_size_m` (default 30 m).

**Practical guidance:** treat `burn_map.npy` as the canonical \([0,1]\) dynamic risk field. For static BP files stored as `int16`, convert to \(r=\texttt{value}/10000\) when interpreting or exporting probabilities; the shipped benchmark code uses stored values directly as optimization weights (ranking is preserved under positive scaling).

---

## Fire-spread trajectories (Sim2Real masks)

Fire spread is provided as **grayscale JPG sequences** under `Satellite_Images_Mask/<scenario_id>/`. Optionally, preprocessed **`scenarii/<scenario_id>.npy`** files are provided or can be generated.

### JPG decoding (canonical loader)

[`load_scenario_jpg`](code/dataset.py) (also used by `load_scenario(..., extension=".jpg")`):

1. Read each frame as grayscale (`PIL.Image.convert('L')`).
2. Convert to float: **`pixel / 255.0`** → values in \([0,1]\).
3. If `binary=True` (**default**): **`mask = (pixel >= 0.5).astype(float)`** → `{0,1}` fire mask.

Frames are sorted with natural numeric order (`out2.jpg` before `out10.jpg`).

### JPEG caveats

- JPG is **lossy**. Boundary pixels may sit strictly between 0 and 1 before thresholding; after `>= 0.5` they become 0 or 1.
- Re-saving or recompressing frames can shift edge pixels across the 0.5 threshold.
- For bit-identical reproduction, prefer **`scenarii/*.npy`** produced by [`jpg_scenario_to_npy`](code/dataset.py) / [`preprocess_sim2real_dataset`](code/dataset.py).

### NPY format and precedence

- **New format:** `float32` array of shape `(T, N, M)` (see `save_scenario_npy`).
- **Legacy format:** 0-d object array wrapping `{'scenario': ...}`; `load_scenario_npy` unwraps it.
- **When both JPG and NPY exist:** the benchmark precompute path and [`all_experiments_parallel.py`](all_experiments_parallel.py) use `file_format="npy"` and read `scenarii/*.npy`. Treat **NPY as canonical** for experiments; JPG is the human-readable source export from Sim2Real-Fire.
- **`burn_map.npy`** is derived from scenario masks via `compute_burn_map` (mean of binary masks over scenarios, optionally with per-scenario start offsets from `config_s2r.json`).

### Fire detection in simulation

[`code/benchmark.py`](code/benchmark.py) treats a cell as burning when **`grid[i,j] == 1`** (strict equality) for sensor/drone detection; burned-area metrics use **`grid > 0.5`**. Preprocessed masks should therefore be binary `{0,1}` (as produced by `load_scenario_jpg` with default `binary=True`).

---

## Quick start (MinimalDataset)

```python
from dataset import load_scenario, load_burn_map, load_scenario_jpg

# Fire trajectory — prefer preprocessed NPY
scenario = load_scenario("MinimalDataset/0001/scenarii/0001_00002.npy")

# Same scenario from JPG (default: /255 then threshold >= 0.5)
scenario_jpg = load_scenario(
    "MinimalDataset/0001/Satellite_Images_Mask/0001_00002",
    extension=".jpg",
)

# Dynamic ground-truth risk map (float32, values in [0,1])
burn_map = load_burn_map("MinimalDataset/0001/burn_map.npy")

# Normalize legacy/static USFS BP integers to paper probabilities (external use)
import numpy as np
bp = np.load("path/to/layout/static_risk_bp2024.npy")
if bp.dtype != np.float32:
    r_it = bp.astype(np.float64) / 10000.0
else:
    r_it = bp
```

See [`README.md`](README.md) for preprocessing (`preprocess_sim2real_dataset`) and benchmarking entry points.
