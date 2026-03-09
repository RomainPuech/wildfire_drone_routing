# Clustering and Wrappers

This document describes the wrapper system that adds **logging**, **caching**, and **spatial clustering** to any strategy without modifying the strategy code. These wrappers are essential for reproducibility, performance, and scaling to larger scenarios.

---

## Table of Contents

1. [Wrapper Architecture](#1-wrapper-architecture)
2. [Logging/Caching Wrappers](#2-loggingcaching-wrappers)
3. [Clustering Wrapper](#3-clustering-wrapper)
4. [Wrapper Composition](#4-wrapper-composition)
5. [Pre-registered Wrapped Strategies](#5-pre-registered-wrapped-strategies)
6. [Multiprocessing Considerations](#6-multiprocessing-considerations)

---

## 1. Wrapper Architecture

The project uses the **Decorator Pattern** via dynamic class creation (`type()`) to wrap strategies transparently. The wrapped class has the same interface as the original, so the benchmark engine doesn't need to know about the wrapping.

```
                    Strategy Interface
                    ┌──────────────────┐
                    │ __init__()       │
                    │ get_locations()  │ (sensor)
                    │ get_initial_     │
                    │   drone_locs()   │ (drone)
                    │ next_actions()   │ (drone)
                    └──────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Logging    │ │  Clustering  │ │    Base       │
    │   Wrapper    │ │   Wrapper    │ │  Strategy     │
    │              │ │              │ │              │
    │ Check cache  │ │ Partition    │ │ Actual       │
    │ Log actions  │ │ into clusters│ │ computation  │
    │ Replay from  │ │ Run sub-     │ │              │
    │ JSON file    │ │ strategies   │ │              │
    └──────────────┘ └──────────────┘ └──────────────┘
```

All wrappers are created by factory functions:
- `make_loggable_sensor_strategy(cls)` → sensor logging wrapper
- `make_loggable_drone_strategy(cls)` → drone logging wrapper
- `get_wrapped_clustering_strategy(cls)` → clustering wrapper

---

## 2. Logging/Caching Wrappers

### Sensor Strategy Logging (`LoggableSensorStrategyWrapper`)

Wraps any `SensorPlacementStrategy` to log and cache sensor positions.

**On first call**: Runs the base strategy, saves results to a JSON log file.
**On subsequent calls**: Loads positions from the log file, skipping the expensive optimization.

```python
# Factory usage:
LoggedSensor = make_loggable_sensor_strategy(SensorPlacementOptimization)
# Now LoggedSensor behaves like SensorPlacementOptimization but with caching
```

**Log file location**:
```
{layout_dir}/logs/{StrategyName}_{burnmap_nickname}_{N}N_{M}M_{n_ground}ground_{n_charge}charge.json
```

**Log file format**:
```json
{
  "ground_sensor_locations": [[3, 7], [12, 5], ...],
  "charging_station_locations": [[8, 2], [15, 10]]
}
```

**Caching logic**:
```
if log file exists:
    load ground_sensor_locations and charging_station_locations from JSON
    return cached positions
else:
    run base_strategy_cls(auto_params, custom_params)
    save positions to JSON
    return computed positions
```

### Drone Strategy Logging (`LoggableDroneStrategyWrapper`)

Wraps any `DroneRoutingStrategy` to log and cache the complete action history.

```python
# Factory usage:
LoggedDrone = make_loggable_drone_strategy(DroneRoutingMaxCoverageResetStatic)
```

**Log file location**:
```
{layout_dir}/logs/{layout}_{StrategyName}_{burnmap}_{n_drones}_drones_{n_charging}_charging_stations_{n_ground}_ground_stations_{positions}_{horizon}_{reeval}_{regularization}_logged_drone_routing.json
```

The detailed filename encodes all relevant parameters, ensuring that changing any parameter creates a new log file.

**Log file format**:
```json
{
  "initial_drone_locations": [
    ["charge", [3, 7]],
    ["charge", [12, 5]]
  ],
  "actions_history": [
    [["fly", [4, 8]], ["fly", [13, 6]]],
    [["fly", [5, 9]], ["charge", [12, 5]]],
    ...
  ]
}
```

**Caching logic for `next_actions()`**:
```
if loaded from disk AND step_counter < len(actions_history):
    return actions_history[step_counter]  # replay from cache
else:
    actions = inner_strategy.next_actions(...)
    append to actions_history
    flush to disk
    return actions
```

**Key features**:
- **Incremental flushing**: The log file is written after every step, so partial runs are preserved
- **Attribute delegation**: `__getattr__` proxies to the inner strategy, so wrappers are fully transparent
- **Normalisation**: Handles different return formats from strategies (list of tuples, tuple of lists)

### `_deep_unwrap(cls)` — Getting the Base Strategy Name

Wrappers can be nested (clustering + logging). This utility function recursively peels through wrapper layers to find the original strategy class name:

```python
def _deep_unwrap(cls):
    while True:
        if hasattr(cls, "base_cls"):        # clustering wrapper
            cls = cls.base_cls
        elif hasattr(cls, "base_strategy_cls"):  # logging wrapper
            cls = cls.base_strategy_cls
        else:
            break
    return cls
```

This ensures log file names reflect the actual strategy, not the wrapper chain.

---

## 3. Clustering Wrapper

### Purpose

When charging stations are spread across a large area, a single optimization problem covering the entire grid can be expensive and may produce poor solutions. The **clustering wrapper** decomposes the problem by:

1. Grouping nearby charging stations into **clusters**
2. Assigning drones proportionally to each cluster
3. Running an independent sub-strategy for each cluster
4. Combining actions from all clusters into a single action list

### `ClusteredDroneStrategyWrapped`

The core clustering class, defined in `new_clustering.py`.

#### Initialization

```python
class ClusteredDroneStrategyWrapped:
    def __init__(self, auto_params, custom_params):
        # 1. Cluster charging stations
        self.clusters = self.find_clusters(
            charging_stations, max_battery_time
        )

        # 2. Allocate drones proportionally
        drones_per_cluster = allocate_drones(total_drones, clusters)

        # 3. Create sub-strategy for each cluster
        for cid, stations in enumerate(clusters):
            sub_params = auto_params.copy()
            sub_params["charging_stations_locations"] = stations
            sub_params["n_drones"] = drones_per_cluster[cid]
            sub_params["n_charging_stations"] = len(stations)
            sub_params["ground_sensor_locations"] = sensors_in_cluster

            strategy_instances.append(BaseStrategy(sub_params, custom_params))
```

#### Clustering Algorithm (`find_clusters`)

Uses **BFS-based connected components** on a graph where stations are connected if their Euclidean distance is within the drone battery range:

```
1. Build adjacency graph:
   - For each pair of charging stations (i, j):
     - If distance(i, j) ≤ drone_battery:
       - Add edge i ↔ j

2. Find connected components via BFS:
   - Each component is a cluster
   - Isolated stations form singleton clusters
```

The threshold `radius = drone_battery` ensures that within a cluster, a drone can fly from any station to any other station in the cluster (directly or with intermediate recharges).

#### Drone Allocation

Drones are distributed proportionally to the number of charging stations per cluster:

```python
drones_per_cluster = [
    max(1, round(total_drones * (stations / total_charging_stations)))
    for stations in charging_stations_per_cluster
]

# Adjust to match total exactly
while sum(drones_per_cluster) > total_drones:
    # Remove from clusters with >1 drone
while sum(drones_per_cluster) < total_drones:
    # Add to clusters
```

Every cluster gets at least 1 drone.

#### Boundary Computation

Each cluster's operational area is defined by the union of bounding boxes around its stations:

```python
def get_cluster_boundary_boxes(stations, half_extent):
    boxes = []
    for x, y in stations:
        box = Polygon([
            (x - half_extent, y - half_extent),
            (x + half_extent, y - half_extent),
            (x + half_extent, y + half_extent),
            (x - half_extent, y + half_extent),
        ])
        boxes.append(box)
    return list(unary_union(boxes))  # Merge overlapping boxes
```

The `half_extent` is `min(transmission_range, battery/2)` — the maximum distance a drone can fly from a station and return.

Ground sensors are assigned to the cluster whose boundary polygon contains them.

#### Action Dispatch

At each timestep, the wrapper:
1. Slices the global drone state by cluster
2. Calls each sub-strategy independently
3. Concatenates the actions

```python
def next_actions(self, auto_step_params, custom_step_params):
    actions = []
    idx = 0
    for count, strat in zip(self.drones_per_cluster, self.strategy_instances):
        sliced_params = {
            "drone_locations": auto_step_params["drone_locations"][idx:idx+count],
            "drone_batteries": auto_step_params["drone_batteries"][idx:idx+count],
            "drone_states":    auto_step_params["drone_states"][idx:idx+count],
            "t": auto_step_params["t"]
        }
        cluster_actions = strat.next_actions(sliced_params, custom_step_params)
        actions.extend(cluster_actions)
        idx += count
    return actions
```

#### Visualization

The clustering wrapper includes a `plot_clusters()` method to visualize the cluster layout with colored boundaries, station markers, and drone zones.

### Factory Function

```python
def get_wrapped_clustering_strategy(BaseStrategy):
    name = BaseStrategy.__name__
    Wrapped = type(
        name,
        (ClusteredDroneStrategyWrapped,),
        {'base_cls': BaseStrategy, '__module__': 'wrappers'}
    )
    globals()[name] = Wrapped
    return Wrapped
```

The returned class has the same `__name__` as the base strategy but inherits from `ClusteredDroneStrategyWrapped`, adding clustering behavior transparently.

---

## 4. Wrapper Composition

In practice, strategies are wrapped in a specific order: **Clustering first, then Logging**. This means:

```python
# 1. Start with the base strategy
BaseStrategy = DroneRoutingMaxCoverageResetStatic

# 2. Wrap with clustering
ClusteredStrategy = get_wrapped_clustering_strategy(BaseStrategy)
# ClusteredStrategy clusters stations, runs BaseStrategy per cluster

# 3. Wrap with logging
LoggedClusteredStrategy = make_loggable_drone_strategy(ClusteredStrategy)
# LoggedClusteredStrategy caches the entire clustered strategy's output
```

The composition chain is:

```
LoggableDroneStrategyWrapper
  └── base_strategy_cls = ClusteredDroneStrategyWrapped
        └── base_cls = DroneRoutingMaxCoverageResetStatic
```

When the logging wrapper looks up the strategy name for the log filename, `_deep_unwrap()` traverses this chain to find `DroneRoutingMaxCoverageResetStatic`.

---

## 5. Pre-registered Wrapped Strategies

At the bottom of `wrappers.py`, all standard strategy combinations are pre-registered:

```python
# Sensor strategies (logged)
RandomSensorPlacementStrategyLogged = make_loggable_sensor_strategy(RandomSensorPlacementStrategy)
SensorPlacementMaxCoverageGaussianTimeLogged = make_loggable_sensor_strategy(SensorPlacementMaxCoverageGaussianTime)

# Drone strategies (clustered + logged)
ClusteredMaxCoverage = get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStatic)
DroneRoutingMaxCoverageResetStaticLogged = make_loggable_drone_strategy(ClusteredMaxCoverage)

ClusteredTOP = get_wrapped_clustering_strategy(DroneRoutingTOP)
DroneRoutingTOPLogged = make_loggable_drone_strategy(ClusteredTOP)
# ... etc
```

These are registered in `globals()` so that:
1. They can be looked up by string name (needed for parallel processing with `getattr(wrappers, name)`)
2. `pickle` can serialize them (needed for `ProcessPoolExecutor`)

```python
# Make strategies accessible by their short names
globals()["DroneRoutingMaxCoverageResetStatic"] = DroneRoutingMaxCoverageResetStaticLogged
globals()["DroneRoutingTOP"] = DroneRoutingTOPLogged
# ...
```

---

## 6. Multiprocessing Considerations

The wrapper system is designed to work with `ProcessPoolExecutor`:

1. **Spawn context**: The benchmark uses `get_context("spawn")` to avoid fork-related issues with Julia
2. **Pickle compatibility**: Wrapped classes are registered in `globals()` with `__module__ = 'wrappers'` so `pickle` can find them
3. **String-based strategy passing**: In parallel mode, drone strategy names are passed as strings and resolved via `getattr(wrappers, drone_strategy_name)` in each worker process
4. **Independent Julia sessions**: Each spawned process initializes its own Julia session
5. **Log file isolation**: Each layout's logs are stored in its own `logs/` directory, and random IDs in temp burn map filenames prevent write conflicts

---

*Previous: [04 — Strategy Architecture](04_strategy_architecture.md) · Next: [06 — Drone Simulation](06_drone_simulation.md)*
