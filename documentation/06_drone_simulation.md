# Drone Simulation

This document describes the `Drone` class, which models the physical behavior of a single drone during simulation: its position, battery, state machine, and movement mechanics.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Drone State](#2-drone-state)
3. [Initialization](#3-initialization)
4. [Action Types](#4-action-types)
5. [Battery Management](#5-battery-management)
6. [Boundary Clamping](#6-boundary-clamping)
7. [State Machine](#7-state-machine)
8. [Integration with the Benchmark](#8-integration-with-the-benchmark)

---

## 1. Overview

The `Drone` class (in `code/Drone.py`) is a lightweight state machine that represents a single UAV in the wildfire detection simulation. It operates entirely in **data space** — the high-resolution grid — while the strategy operates in **operational space**. The benchmark engine mediates between the two coordinate systems.

Each drone tracks:
- Its **(x, y) position** on the data grid
- Two **battery levels** (distance and time)
- Its **state** (`'charge'` or `'fly'`)
- Whether it is **alive** (operational)

---

## 2. Drone State

```python
class Drone:
    x: int                    # Current row on the data grid
    y: int                    # Current column on the data grid
    N: int                    # Grid height
    M: int                    # Grid width
    state: str                # 'charge' or 'fly'
    alive: bool               # Whether the drone is operational
    distance_battery: int     # Remaining distance budget (Manhattan)
    time_battery: int         # Remaining time budget (steps)
    max_distance_battery: int # Full distance budget
    max_time_battery: int     # Full time budget
    charging_stations_locations: list  # List of (x, y) charging stations
```

---

## 3. Initialization

```python
Drone(x, y, state, charging_stations_locations, N, M,
      max_distance_battery=100, max_time_battery=100,
      current_distance_battery=None, current_time_battery=None)
```

**Constraints**:
- The drone **must start at a charging station**. If `(x, y)` is not in `charging_stations_locations`, a `ValueError` is raised.
- If `state == 'charge'`, the battery is immediately set to full, regardless of `current_distance_battery` and `current_time_battery`.
- If no current battery is provided, it defaults to the maximum.

**In the benchmark**, drones are initialized like this:

```python
Drone(
    x * coverage_width_cells + coverage_width_cells // 2,   # data-space x
    y * coverage_width_cells + coverage_width_cells // 2,    # data-space y
    state,                    # 'charge' (initial state)
    charging_stations_data,   # all charging stations in data coords
    N_data,                   # data grid height
    M_data,                   # data grid width
    max_battery_distance,     # distance budget
    max_battery_time,         # time budget (data-space hours)
    max_battery_distance - 1*(state=='fly'),   # initial distance battery
    max_battery_time - 1*(state=='fly')        # initial time battery
)
```

---

## 4. Action Types

The drone's `route()` method dispatches to one of three actions:

```python
def route(self, action):
    if action[0] == 'move':
        return self.move(*action[1])      # Relative movement
    elif action[0] == 'fly':
        return self.fly(*action[1])        # Absolute position
    elif action[0] == 'charge':
        return self.recharge(*action[1])   # Go to charging station
    else:
        raise ValueError(f"Invalid action: {action}")
```

All three return a 5-tuple: `(x, y, distance_battery, time_battery, state)`.

### `move(dx, dy)` — Relative Movement

Moves the drone by a displacement vector:

```python
def move(self, dx, dy):
    self.state = "fly"
    self.x += dx
    self.y += dy
    self.x = max(0, min(self.x, self.N-1))   # Clamp to grid
    self.y = max(0, min(self.y, self.M-1))
    self.distance_battery -= (abs(dx) + abs(dy))   # Manhattan cost
    self.time_battery -= 1
    return self.x, self.y, self.distance_battery, self.time_battery, self.state
```

- **Displacement**: `(dx, dy)` are typically in `{-1, 0, +1}` for single-cell moves, but can be larger when converted from operational scale
- **Battery cost**: Manhattan distance `|dx| + |dy|` for distance, 1 for time
- **State**: Always set to `'fly'` after a move

### `fly(x, y)` — Absolute Position

Teleports the drone to the specified position:

```python
def fly(self, x, y):
    self.state = "fly"
    self.x = x
    self.y = y
    self.distance_battery -= (abs(self.x - x) + abs(self.y - y))
    self.time_battery -= 1
    return self.x, self.y, self.distance_battery, self.time_battery, self.state
```

**Note**: The distance battery deduction uses `abs(self.x - x)` *after* setting `self.x = x`, so it always evaluates to 0 for distance. This means `fly` actions only cost 1 time unit. This is by design — in operational space, each `fly` action moves the drone by one operational cell, and the physical distance is subsumed by the time cost.

### `recharge(x, y)` — Charging

Moves the drone to a charging station and fully restores the battery:

```python
def recharge(self, x, y):
    self.x = x
    self.y = y
    self.state = "charge"
    self.distance_battery = self.max_distance_battery
    self.time_battery = self.max_time_battery
    return self.x, self.y, self.distance_battery, self.time_battery, self.state
```

- Charging is **instantaneous** — the drone is fully recharged in one substep
- The drone doesn't need to already be at the station; the action moves it there
- State is set to `'charge'`

---

## 5. Battery Management

The drone has two independent battery systems:

| Battery | Depleted by | Cost per move | Cost per fly | Recharged by |
|---------|------------|---------------|--------------|--------------|
| `distance_battery` | `move` actions | `\|dx\| + \|dy\|` (Manhattan) | 0 (see note) | `recharge` → full |
| `time_battery` | Both `move` and `fly` | 1 | 1 | `recharge` → full |

In practice, **time-based battery** (`max_battery_time`) is the primary constraint used by all strategies. The distance battery (`max_battery_distance`) is typically set to `-1` (disabled) in the simulation parameters.

### Battery Depletion at Operational Scale

The benchmark separately tracks battery at the operational scale:

```python
# After each action in the simulation loop:
if action[0] in ['move', 'fly']:
    drone_batteries_opt_scale[drone_index] -= 1
elif action[0] in ['charge']:
    drone_batteries_opt_scale[drone_index] = rescaled_max_battery_time
```

This operational battery is what the strategy sees. The data-scale battery on the `Drone` object is tracked separately for completeness but doesn't currently trigger drone death.

### Battery Death (Not Yet Implemented)

The `_check_battery()` method and `alive` flag exist but are currently disabled:

```python
def _check_battery(self):
    return True  # Always alive (TODO)
    if self.time_battery <= 0:
        self.alive = False
        return False
    return True
```

---

## 6. Boundary Clamping

When a `move` action would push the drone outside the grid, coordinates are clamped:

```python
self.x = max(0, min(self.x, self.N - 1))
self.y = max(0, min(self.y, self.M - 1))
```

This prevents index errors during fire detection. Drones hitting a boundary simply stop at the edge.

**Note**: `fly` actions don't have boundary clamping — they are expected to always target valid coordinates (as generated by the optimization).

---

## 7. State Machine

The drone has a simple two-state machine:

```
         move/fly          charge
  ┌──────────────────────────────────┐
  │                                   │
  ▼                                   │
┌──────┐                        ┌─────────┐
│ fly  │───── recharge() ──────►│ charge  │
│      │◄──── move()/fly() ────│         │
└──────┘                        └─────────┘
  │  ▲                              │
  │  │  move()/fly()                │
  │  └──────────────────────────────┘
```

- Starting state: `'charge'` (at a charging station)
- Any `move` or `fly` action transitions to `'fly'`
- Any `charge` action transitions to `'charge'`
- The strategy sees the state and decides when to return to a charging station

---

## 8. Integration with the Benchmark

The benchmark creates and manages `Drone` objects. Here's how the pieces fit together:

```python
# 1. Strategy decides actions in operational space
actions_opt = strategy.next_actions(state_opt, params)

# 2. Benchmark converts to data space
actions_data = convert_to_data_scale(actions_opt)

# 3. Benchmark applies actions to Drone objects
for drone, action in zip(drones, actions_data):
    new_x, new_y, dist_bat, time_bat, new_state = drone.route(action)

    # 4. Update tracked positions
    drone_locations_data[i] = (new_x, new_y)
    drone_locations_opt[i] = new_position_opt[i]

    # 5. Update operational batteries separately
    if action[0] in ['move', 'fly']:
        batteries_opt[i] -= 1
    elif action[0] == 'charge':
        batteries_opt[i] = max_battery_opt

    # 6. Track metrics
    total_distance += abs(new_x - old_x) + abs(new_y - old_y)
    visited_cells.add((new_x, new_y))
```

The `Drone` class is intentionally simple — it's a physical simulation object, not a decision-maker. All intelligence lives in the strategy.

---

*Previous: [05 — Clustering and Wrappers](05_clustering_and_wrappers.md) · Next: [07 — Benchmarking Pipeline](07_benchmarking_pipeline.md)*
