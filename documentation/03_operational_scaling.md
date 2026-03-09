# Operational Scaling: Data Space → Operational Space

This document explains the critical conversion between **data space** (the raw fire scenario grid) and **operational space** (the coarser grid where drone routing decisions are made). This rescaling bridges the gap between high-resolution fire data and actionable drone plans.

---

## Table of Contents

1. [Why Two Spaces?](#1-why-two-spaces)
2. [Key Concepts](#2-key-concepts)
3. [Scaling Parameters](#3-scaling-parameters)
4. [Grid Rescaling](#4-grid-rescaling)
5. [Temporal Rescaling (Substeps)](#5-temporal-rescaling-substeps)
6. [Battery Rescaling](#6-battery-rescaling)
7. [Burn Map Pooling](#7-burn-map-pooling)
8. [Mask Pooling](#8-mask-pooling)
9. [Coordinate Conversion](#9-coordinate-conversion)
10. [Complete Scaling Pipeline](#10-complete-scaling-pipeline)

---

## 1. Why Two Spaces?

The raw fire scenario data is a high-resolution grid where each cell represents a small physical area (e.g., 30×30 meters). However, a drone's coverage area is much larger than a single cell — for example, a drone with a 300m coverage radius covers a 20×20 data-cell area.

Making routing decisions at the data-cell level would be:
- **Computationally expensive**: The optimization problem scales with grid size
- **Semantically meaningless**: A drone covers many cells at once, so cell-level planning is wasteful
- **Inconsistent with reality**: A drone "visits" an area, not a single 30m cell

The solution: **rescale the grid** so that one operational cell equals one drone coverage area. Strategies operate on this coarser operational grid, and actions are converted back to data space for simulation.

```
Data Space                           Operational Space
(N × M cells, 30m each)              (N' × M' cells, one per coverage area)
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐             ┌───────┬───────┐
│ │ │ │ │ │ │ │ │ │ │              │       │       │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤             │   ●   │   ●   │  ● = drone covers
│ │ │ │ │ │ │ │ │ │ │              │       │       │      entire cell
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   ──────►  ├───────┼───────┤
│ │ │ │ │ │ │ │ │ │ │              │       │       │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤             │   ●   │       │
│ │ │ │ │ │ │ │ │ │ │              │       │       │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘             └───────┴───────┘
  10 × 10 data cells                    2 × 2 operational cells
  (30m each = 300m)                     (150m each = 300m)
```

---

## 2. Key Concepts

| Concept | Symbol | Description |
|---------|--------|-------------|
| Data cell size | `cell_size_m` | Physical size of one data grid cell (typically 30m) |
| Coverage radius | `coverage_radius_m` | Drone sensor coverage radius (typically 300m) |
| Coverage width | `coverage_width_cells` | Number of data cells in one coverage diameter |
| Drone speed | `drone_speed_m_per_min` | Drone movement speed (typically 600 m/min) |
| Operational substeps | `operational_substeps` | Drone actions per data timestep |
| Data timestep | — | 60 minutes (1 hour per fire frame) |

---

## 3. Scaling Parameters

The conversion is governed by a few physical parameters from the simulation configuration:

```python
simulation_parameters = {
    "cell_size_m": 30,              # Each data cell = 30m × 30m
    "coverage_radius_m": 300,        # Drone covers a 300m radius
    "drone_speed_m_per_min": 600,    # Drone flies at 600 m/min
    "max_battery_time": 1,           # Battery lasts 1 hour (in data timesteps)
}
```

From these, the benchmark computes:

```python
# Coverage width in data cells (diameter of coverage area)
coverage_width_cells = round(coverage_radius_m * 2 / cell_size_m)
# = round(300 * 2 / 30) = 20

# Rescaled grid dimensions
rescaled_N = N // coverage_width_cells    # e.g., 200 // 20 = 10
rescaled_M = M // coverage_width_cells    # e.g., 200 // 20 = 10
```

---

## 4. Grid Rescaling

The operational grid is obtained by dividing the data grid dimensions by `coverage_width_cells`:

```
N_operational = N_data // coverage_width_cells
M_operational = M_data // coverage_width_cells
```

Each operational cell corresponds to a `coverage_width_cells × coverage_width_cells` block of data cells. The center of each operational cell `(i, j)` in data coordinates is:

```python
data_x = i * coverage_width_cells + coverage_width_cells // 2
data_y = j * coverage_width_cells + coverage_width_cells // 2
```

**Note**: Cells at the edges of the data grid that don't form a complete coverage block are truncated (integer division discards the remainder).

---

## 5. Temporal Rescaling (Substeps)

One data timestep equals **60 minutes** of real time. During that hour, a drone can make multiple movements. The number of drone actions per data timestep is called `operational_substeps`:

### `compute_operational_substeps(data_cell_size_m, drone_speed_m_per_min, coverage_radius_m)`

```python
def compute_operational_substeps(data_cell_size_m, drone_speed_m_per_min, coverage_radius_m):
    # Width of one coverage area in data cells
    coverage_width_cells = round(2 * coverage_radius_m / data_cell_size_m)
    if coverage_width_cells % 2 == 0:
        coverage_width_cells -= 1  # Make odd for symmetry

    # Distance the drone can travel in 60 minutes
    drone_distance_m = 60 * drone_speed_m_per_min

    # How many operational cells the drone can cross in one data timestep
    drone_distance_operational_cells = drone_distance_m // (coverage_width_cells * data_cell_size_m)

    return max(1, round(drone_distance_operational_cells))
```

**Example**:
- `coverage_width_cells` = 19 (after making odd)
- `drone_distance_m` = 60 × 600 = 36,000m
- One operational cell = 19 × 30m = 570m
- `operational_substeps` = 36,000 // 570 ≈ 63

This means: **within each 1-hour data timestep, the drone takes 63 routing decisions** (each moving it by one operational cell).

### Simulation Loop Structure

```
for each data timestep t:
    check fire detection by static sensors
    for each substep s in range(operational_substeps):
        strategy.next_actions()     # Get drone actions at operational scale
        convert actions to data scale
        move drones
        check fire detection by drones
```

---

## 6. Battery Rescaling

Battery capacity is measured in **data timesteps** (hours) but consumed in **operational substeps** (drone moves). The conversion is:

```python
rescaled_max_battery_time = max_battery_time * operational_substeps
```

**Example**: If `max_battery_time = 1` (1 hour) and `operational_substeps = 63`:
- The drone has 63 operational moves before needing to recharge
- Each move costs 1 unit of operational battery
- Charging fully restores the battery to `rescaled_max_battery_time`

The strategy receives `rescaled_max_battery_time` in its initialization parameters and manages the battery budget at the operational level.

---

## 7. Burn Map Pooling

Burn maps must be rescaled from data space to operational space. The project implements two pooling methods:

### `pool_burnmap_mean(burnmap, kernel_size)`

Averages the burn probabilities within each `kernel_size × kernel_size` block:

```python
burnmap_pooled[t, i, j] = mean(burnmap[t, i*k:(i+1)*k, j*k:(j+1)*k])
```

This provides a smooth estimate of fire risk in each operational cell.

### `pool_burnmap_proba_at_least_one(burnmap, kernel_size)`

Computes the probability that **at least one** data cell within the block is on fire, using the complement rule:

```python
burnmap_pooled[t, i, j] = 1 - prod(1 - burnmap[t, i*k:(i+1)*k, j*k:(j+1)*k])
```

This is more appropriate for fire detection: if any cell in the drone's coverage area is burning, the drone will detect it. This method gives higher probabilities than simple averaging and better reflects the detection semantics.

**Note**: If the input burn map is 2D (static risk map), it is expanded to 3D by repeating 100 times along the time axis before pooling.

### Temporal Pooling

After spatial pooling, the burn map is also replicated along the time axis to match the substep resolution:

```python
# Duplicate each hourly frame operational_substeps times
rescaled_burnmap = np.repeat(rescaled_burnmap, operational_substeps, axis=0)

# Rescale probabilities to substep time scale
rescaled_burnmap /= operational_substeps
```

The division by `operational_substeps` adjusts the per-hour fire probability to a per-substep probability, assuming uniform distribution of fire ignition within each hour.

---

## 8. Mask Pooling

Some layouts include a **mask** (e.g., land/water or burnable/non-burnable) that is also pooled to operational scale:

### `pool_mask_min(mask, kernel_size)`

Uses a **min** pooling operation:

```python
mask_pooled[i, j] = min(mask[i*k:(i+1)*k, j*k:(j+1)*k])
```

A pooled cell is only considered valid (1) if **all** data cells in the block are valid. This is conservative: if any data cell is invalid (water, rock, etc.), the entire operational cell is marked as invalid.

---

## 9. Coordinate Conversion

### Operational → Data Space

When the strategy produces actions in operational coordinates, they must be converted to data coordinates for the actual drone simulation.

#### For `fly` actions (absolute position)

```python
def operational_space_to_dataspace_coordinates(coordinate, coverage, datacell_size_m):
    n_data_cells = round(2 * coverage / datacell_size_m)
    if n_data_cells % 2 == 0:
        n_data_cells -= 1

    x, y = coordinate
    half = n_data_cells // 2
    return (x * n_data_cells + half, y * n_data_cells + half)
```

This maps operational cell `(i, j)` to the **center** of the corresponding data-cell block.

#### For `move` actions (relative displacement)

```python
# move actions are scaled by coverage_width_cells
converted = (coverage_width_cells * dx, coverage_width_cells * dy)
```

#### For `charge` actions

Same as `fly` — the charging station's operational coordinates are converted to data space.

### Sensor Position Conversion

Sensor and charging station positions are placed in operational space by the sensor placement strategy, then converted to data space for the simulation:

```python
# Operational → Data
ground_sensor_locations_data = [
    (x * coverage_width_cells + coverage_width_cells // 2,
     y * coverage_width_cells + coverage_width_cells // 2)
    for x, y in ground_sensor_locations_opt
]
```

### Drone Position Tracking

The benchmark maintains drone positions in **both** coordinate systems simultaneously:

- `drone_locations_data_scale`: Used for fire detection (comparing against data grid)
- `drone_locations_opt_scale`: Fed back to the strategy for its next decision

This dual tracking avoids coordinate conversion errors accumulating over time.

---

## 10. Complete Scaling Pipeline

Here is the full sequence of operations performed at the start of each benchmark scenario:

```
Input:
  scenario       (T, N_data, M_data)     Raw fire grid
  burn_map       (T, N_data, M_data)     Fire probability map
  mask           (N_data, M_data)         Land/water mask (optional)
  simulation_parameters                   Physical drone parameters

Step 1: Compute scaling factors
  coverage_width_cells  = round(2 * coverage_radius_m / cell_size_m)
  operational_substeps  = compute_operational_substeps(...)
  N_op = N_data // coverage_width_cells
  M_op = M_data // coverage_width_cells

Step 2: Rescale burn map
  burn_map_op = pool_burnmap_proba_at_least_one(burn_map, coverage_width_cells)
                           →  shape (T, N_op, M_op)
  burn_map_op = np.repeat(burn_map_op, operational_substeps, axis=0) / operational_substeps
                           →  shape (T * operational_substeps, N_op, M_op)
  np.save(rescaled_burnmap_filename, burn_map_op)

Step 3: Rescale mask (if present)
  mask_op = pool_mask_min(mask, coverage_width_cells)
                           →  shape (N_op, M_op)
  np.save(rescaled_mask_filename, mask_op)

Step 4: Rescale parameters
  rescaled_params = {
      N: N_op,
      M: M_op,
      max_battery_time: max_battery_time * operational_substeps,
      burnmap_filename: rescaled_burnmap_filename,
      mask_filename: rescaled_mask_filename,
      ...
  }

Step 5: Place sensors (operational scale)
  sensor_strategy(rescaled_params) → ground_sensors_op, charging_stations_op

Step 6: Convert sensor positions to data scale
  ground_sensors_data = [(x*cwc + cwc//2, y*cwc + cwc//2) for x,y in ground_sensors_op]
  charging_stations_data = [(x*cwc + cwc//2, y*cwc + cwc//2) for x,y in charging_stations_op]

Step 7: Initialize drones (operational scale)
  routing_strategy(rescaled_params) → initial_drone_positions_op

Step 8: Convert drone positions to data scale
  drones = [Drone(x*cwc + cwc//2, y*cwc + cwc//2, state, ...) for state,(x,y) in positions_op]

Step 9: Simulation loop
  for each data timestep t:
      check static sensor detection (data scale)
      for each substep:
          actions_op = strategy.next_actions(positions_op, batteries_op, ...)
          actions_data = convert_to_data_scale(actions_op)
          move drones (data scale)
          update positions_op from actions
          check drone detection (data scale)
```

### Saved Rescaled Files

The rescaled burn maps and masks are saved to disk with descriptive filenames:

```
burn_map_rescaled_{N_op}x{M_op}_{substeps}substeps.npy
mask_rescaled_{N_op}x{M_op}_{substeps}substeps.npy
```

This allows them to be reused across scenarios within the same layout (since the grid dimensions and scaling factors are identical).

---

*Previous: [02 — Data Pipeline](02_data_pipeline.md) · Next: [04 — Strategy Architecture](04_strategy_architecture.md)*
