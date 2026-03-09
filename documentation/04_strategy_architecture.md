# Strategy Architecture

This document describes the strategy design pattern used in the project, from the abstract base classes to the concrete implementations. Strategies are the pluggable intelligence of the system: they decide **where** sensors go and **how** drones move.

---

## Table of Contents

1. [Design Pattern](#1-design-pattern)
2. [Base Classes](#2-base-classes)
3. [Sensor Placement Strategies](#3-sensor-placement-strategies)
4. [Drone Routing Strategies](#4-drone-routing-strategies)
5. [Julia Integration Pattern](#5-julia-integration-pattern)
6. [Burn Map Management Patterns](#6-burn-map-management-patterns)
7. [Strategy Index Conversion](#7-strategy-index-conversion)
8. [Strategy Summary](#8-strategy-summary)

---

## 1. Design Pattern

The project uses a **Strategy Pattern** where:

1. Abstract base classes define the interface (`SensorPlacementStrategy`, `DroneRoutingStrategy`)
2. Concrete classes implement specific algorithms (random, greedy, optimization-based, TOP-based)
3. The benchmark engine accepts any strategy class and calls it through the uniform interface
4. Wrappers add cross-cutting concerns (logging, caching, clustering) without modifying strategy logic

```
                      SensorPlacementStrategy (ABC)
                     /           |             \
    RandomSensor     SensorPlacementOpt    LoggedOptimization
    Placement         (Julia optimizer)      SensorPlacement
    Strategy                                   Strategy

                      DroneRoutingStrategy (ABC)
                     /          |          \           \
    RandomDrone      DroneRouting    DroneRouting     DroneRouting
    Routing          Optimization   MaxCoverage      TOP
    Strategy         ModelReuse     ResetStatic      (PSO-based)
```

---

## 2. Base Classes

### `SensorPlacementStrategy`

Decides where to place **ground sensors** and **charging stations** on the grid.

```python
class SensorPlacementStrategy:
    def __init__(self, automatic_initialization_parameters: dict,
                       custom_initialization_parameters: dict):
        # Must set: self.ground_sensor_locations, self.charging_station_locations
        raise NotImplementedError

    def get_locations(self):
        # Returns: (ground_sensor_locations, charging_station_locations)
        # Each is a list of (x, y) tuples in operational coordinates
        return self.ground_sensor_locations, self.charging_station_locations
```

**Initialization Parameters** (automatic):

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | int | Grid height (operational scale) |
| `M` | int | Grid width (operational scale) |
| `n_ground_stations` | int | Target number of ground sensors to place |
| `n_charging_stations` | int | Target number of charging stations to place |
| `max_battery_time` | int | Drone battery capacity (operational substeps) |
| `input_dir` | str | Directory containing scenario data |
| `mask_filename` | str | Path to land/water mask (optional) |

### `DroneRoutingStrategy`

Decides **initial drone positions** and **actions at each timestep**.

```python
class DroneRoutingStrategy:
    def __init__(self, automatic_initialization_parameters: dict,
                       custom_initialization_parameters: dict):
        raise NotImplementedError

    def get_initial_drone_locations(self):
        # Returns: list of (state, (x, y)) tuples
        # state: 'charge' or 'fly'
        # (x, y): coordinates in operational scale
        # Drones must start at charging stations
        raise NotImplementedError

    def next_actions(self, automatic_step_parameters: dict,
                           custom_step_parameters: dict):
        # Returns: list of (action_type, (x, y)) tuples
        # action_type: 'fly', 'move', or 'charge'
        raise NotImplementedError
```

**Step Parameters** (automatic, provided at each timestep):

| Parameter | Type | Description |
|-----------|------|-------------|
| `drone_locations` | list[(x,y)] | Current drone positions (operational scale) |
| `drone_batteries` | list[int] | Remaining battery for each drone (operational substeps) |
| `drone_states` | list[str] | Current state of each drone (`'charge'` or `'fly'`) |
| `t` | int | Current operational timestep |

### Action Types

| Action | Semantics | Coordinates |
|--------|-----------|-------------|
| `'fly'` | Move to absolute position | `(x, y)` in operational scale |
| `'move'` | Move by relative displacement | `(dx, dy)` in {-1, 0, +1} |
| `'charge'` | Go to charging station | `(x, y)` of the charging station |

---

## 3. Sensor Placement Strategies

### `RandomSensorPlacementStrategy`

The simplest baseline. Places sensors at random positions on the grid.

```python
class RandomSensorPlacementStrategy(SensorPlacementStrategy):
    def __init__(self, auto_params, custom_params):
        self.ground_sensor_locations = [
            (random.randint(0, N-1), random.randint(0, M-1))
            for _ in range(n_ground_stations)
        ]
        self.charging_station_locations = [
            (random.randint(0, N-1), random.randint(0, M-1))
            for _ in range(n_charging_stations)
        ]
```

### `SensorPlacementOptimization`

Uses Julia's `ground_charging_opt.jl` module to solve a mathematical optimization problem for sensor placement.

```python
class SensorPlacementOptimization(SensorPlacementStrategy):
    def __init__(self, auto_params, custom_params):
        # Calls Julia optimization:
        x_vars, y_vars = jl.NEW_SENSOR_STRATEGY(
            burnmap_filename,
            n_ground_stations,
            n_charging_stations
        )
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
```

The Julia optimizer (using JuMP + Gurobi) maximizes coverage of high-risk areas while ensuring charging stations are reachable by drones.

### `SensorPlacementMaxCoverageGaussianTime` / `SensorPlacementMaxCoverageGaussianTimeMasked`

Advanced sensor placement strategies that consider:
- Gaussian-weighted coverage areas
- Temporal fire probability from burn maps
- Land/water masks to avoid placing sensors on invalid terrain

These strategies also call Julia optimization but with more sophisticated objective functions.

### `FixedPlacementStrategy`

A trivial strategy that uses hardcoded positions (useful for debugging):

```python
class FixedPlacementStrategy(SensorPlacementStrategy):
    def __init__(self, auto_params, custom_params):
        self.charging_station_locations = [(35, 14), (30, 42)]
        self.ground_sensor_locations = []
```

---

## 4. Drone Routing Strategies

### `RandomDroneRoutingStrategy`

Baseline strategy where drones move randomly, with a heuristic to return to charging stations when battery is low:

```
for each drone:
    if battery == 0:
        charge at current position
    elif distance_to_nearest_charging_station == remaining_battery:
        move towards nearest charging station (Chebyshev distance)
    else:
        move in a random direction (dx, dy ∈ {-1, 0, +1})
```

Drones are initially distributed uniformly across charging stations.

### `DroneRoutingOptimizationModelReuseIndex`

Uses Julia's `drone_routing_opt.jl` to solve a mathematical program for drone routing. Key features:

1. **Model reuse**: Creates the JuMP optimization model once, then updates and re-solves it
2. **Rolling horizon**: Optimizes for `optimization_horizon` steps ahead
3. **Periodic re-evaluation**: Re-solves every `reevaluation_step` steps with updated drone states

**Lifecycle**:
```
get_initial_drone_locations():
    model = jl.create_index_routing_model(burnmap, n_drones, charging_stations, ...)
    solution = jl.solve_index_init_routing(model, reevaluation_step)
    return solution[0]  # initial positions

next_actions():
    if call_counter == reevaluation_step - 1:
        solution = jl.solve_index_next_move_routing(
            model, reevaluation_step, drone_locations, drone_states, batteries, t
        )
        call_counter = 0
    call_counter += 1
    return solution[call_counter]  # pre-computed plan
```

### `DroneRoutingUniformCoverageResetStatic` / `DroneRoutingMaxCoverageResetStatic`

The main coverage-based strategies. They maintain a **mutable copy** of the burn map and zero out visited cells to avoid revisiting:

**Key Innovation — Burn Map Reset**:
```python
# After visiting cell (x, y), zero out its burn probability
# for the next reset_time steps
self.current_burnmap[t : t + reset_time, x, y] = 0
```

This encourages the optimizer to send drones to unvisited areas. At each re-evaluation, the modified burn map is saved to a temporary file and passed to Julia.

**Variants**:
- **Uniform Coverage**: Starts with a uniform (flat) burn map — drones explore uniformly
- **Max Coverage**: Starts with the actual burn map — drones prioritize high-risk areas

Both share the same re-evaluation pattern:
1. Save the current (modified) burn map to a temp file
2. Call Julia's `solve_index_next_move_routing()` with updated state
3. Execute the pre-computed plan for `reevaluation_step` steps
4. Zero out visited cells in the burn map

### `DroneRoutingTOP` (Team Orienteering Problem)

Uses the **PSO-based TOP solver** (documented in `PSO.md`) instead of the JuMP optimizer:

```python
class DroneRoutingTOP(DroneRoutingStrategy):
    def get_initial_drone_locations(self):
        self.current_solution = jl.compute_TOP_plan_multiple_depots(
            burnmap_filename, n_drones, charging_stations,
            ground_sensors, max_battery_time, t=0, verbose=False, []
        )
```

**Key differences from the Coverage-based strategies**:
- The reevaluation step equals the battery lifetime (drones plan entire charge cycles)
- At re-evaluation, drones must be at charging stations (hard constraint)
- Uses PSO metaheuristic instead of MILP optimization

**Burn Map Handling Types**:

| Type | Behavior |
|------|----------|
| `fixed_reset` | Zero out visited cells for `reset_time_periods` re-evaluation cycles |
| `growing` | Zero out visited cells permanently, but add back the initial burn map each data timestep |
| `growing_proba` | Like `growing`, but uses probabilistic combination: `1 - (1-current)(1-initial)` |

**Derived Classes**:
- `DroneRoutingTOPwarm` — identical to `DroneRoutingTOP` (warm-start variant)
- `DroneRoutingTOPGrowing` — uses `growing` burn map handling
- `DroneRoutingTOPGrowingProba` — uses `growing_proba` burn map handling

### `DroneRoutingTOPMasked`

Extension of `DroneRoutingTOP` that supports **terrain masks**. Blocked cells (where `mask == 0`) are avoided during routing. Calls `jl.compute_TOP_plan_multiple_depots_masked()`.

### `GREEDY_DRONE_STRATEGY`

A simple template strategy that calls a Julia function (`jl.NEW_drone_routing_example`) every `call_every_n_steps`:

```python
class GREEDY_DRONE_STRATEGY(DroneRoutingStrategy):
    def next_actions(self, auto_params, custom_params):
        if self.call_counter % self.call_every_n_steps == 0:
            self.current_solution = jl.NEW_drone_routing_example(
                drone_locations, drone_batteries, burnmap_filename, horizon
            )
        return self.current_solution[self.call_counter % self.call_every_n_steps]
```

---

## 5. Julia Integration Pattern

All optimization-based strategies follow the same integration pattern with Julia:

### 1. Index Conversion

Julia uses **1-based indexing**, Python uses **0-based indexing**. Every strategy that calls Julia handles this conversion explicitly:

```python
# Python → Julia (before calling Julia)
julia_locations = [(x+1, y+1) for x, y in python_locations]

# Julia → Python (after receiving results)
python_solution = [
    [(code, (x-1, y-1)) if code != "move" else (code, (x, y))
     for code, (x, y) in plan]
    for plan in julia_solution
]
```

Note: `move` actions contain **relative** displacements, so they don't need index conversion.

### 2. Temp Burn Map Files

Strategies that modify burn maps (all coverage/reset variants) save modified burn maps to temporary files in `./tmp_burnmaps/`:

```python
self.current_burnmap_filename = f"./tmp_burnmaps/tmp_burnmap_{random_id}.npy"
save_burn_map(self.current_burnmap, self.current_burnmap_filename)
```

The random ID prevents file conflicts when running strategies in parallel.

### 3. Solution Structure

Julia optimization returns a **nested list**:
```
solution[timestep][drone_index] = (action_type, (x, y))
```

The strategy stores this as `self.current_solution` and indexes into it each timestep.

---

## 6. Burn Map Management Patterns

The coverage-based strategies dynamically modify the burn map to guide exploration. Three patterns exist:

### Fixed Reset

```
Visit cell (x,y) at time t:
    burnmap[t : t + reset_time, x, y] = 0

Effect: Cell becomes "uninteresting" for reset_time steps, then reverts
```

### Growing

```
Visit cell (x,y) at time t:
    burnmap[t:, x, y] = 0

Every data timestep t:
    burnmap[t:] += initial_burnmap[t]

Effect: Visited cells are zeroed, but risk accumulates over time from the base burn map
```

### Growing Probability

```
Visit cell (x,y) at time t:
    burnmap[t:, x, y] = 0

Every data timestep t:
    burnmap[t:] = 1 - (1 - burnmap[t:]) * (1 - initial_burnmap[t])

Effect: Like growing, but uses probabilistic combination assuming independence
```

---

## 7. Strategy Index Conversion

A subtle but critical detail: the strategy operates entirely in **operational space**, but the benchmark engine handles all conversions. The flow is:

```
Benchmark Engine                Strategy (Operational Space)
──────────────                  ──────────────────────────────
                                
rescaled_params = {              
  N: N_op, M: M_op,            
  max_battery_time: rescaled,   
  ...                           
}                               
                                ┌─────────────────────────────┐
 ───────rescaled_params──────►  │  __init__(rescaled_params)   │
                                │  Places sensors at (i,j)     │
                                │  in [0, N_op) × [0, M_op)   │
                                └─────────────────────────────┘
 ◄──── sensor positions ──────  
                                
 Convert: data_pos = op_pos *   
   coverage_width + offset      
                                
 ────── drone state (op) ────►  ┌─────────────────────────────┐
                                │  next_actions(state_op)      │
                                │  Returns fly/move/charge     │
                                │  in operational coords       │
                                └─────────────────────────────┘
 ◄──── actions (op) ──────────  
                                
 Convert: data_action =         
   scale(op_action)             
                                
 Execute on Drone objects       
   (data scale)                 
```

---

## 8. Strategy Summary

| Strategy | Type | Optimizer | Burn Map | Re-evaluation |
|----------|------|-----------|----------|---------------|
| `RandomSensorPlacement` | Sensor | None | — | — |
| `SensorPlacementOptimization` | Sensor | Julia/Gurobi | Static | One-shot |
| `SensorPlacementMaxCoverageGaussianTime` | Sensor | Julia/Gurobi | Time-weighted | One-shot |
| `RandomDroneRouting` | Drone | None | — | Every step |
| `DroneRoutingOptimizationModelReuseIndex` | Drone | Julia/JuMP | Static | Every N steps |
| `DroneRoutingUniformCoverageResetStatic` | Drone | Julia/JuMP | Uniform + reset | Every N steps |
| `DroneRoutingMaxCoverageResetStatic` | Drone | Julia/JuMP | Risk-based + reset | Every N steps |
| `DroneRoutingTOP` | Drone | Julia/PSO | Fixed reset | Every battery cycle |
| `DroneRoutingTOPGrowing` | Drone | Julia/PSO | Growing | Every battery cycle |
| `DroneRoutingTOPMasked` | Drone | Julia/PSO | Fixed reset + mask | Every battery cycle |
| `GREEDY_DRONE_STRATEGY` | Drone | Julia (custom) | Static | Every N steps |

---

*Previous: [03 — Operational Scaling](03_operational_scaling.md) · Next: [05 — Clustering and Wrappers](05_clustering_and_wrappers.md)*
