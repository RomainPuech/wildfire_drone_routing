# Particle Swarm Optimization for the Team Orienteering Problem (TOP)

This document provides a detailed technical description of the PSO-inspired algorithm used to solve the Team Orienteering Problem (TOP) for drone routing in wildfire monitoring scenarios.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Algorithm Architecture](#2-algorithm-architecture)
3. [Data Structures](#3-data-structures)
4. [Solution Representation: The Giant Tour](#4-solution-representation-the-giant-tour)
5. [The Split Procedure](#5-the-split-procedure)
6. [Swarm Initialization](#6-swarm-initialization)
7. [PSO Update Mechanism](#7-pso-update-mechanism)
8. [Local Search Operators](#8-local-search-operators)
9. [Diversity Management](#9-diversity-management)
10. [Algorithm Flow](#10-algorithm-flow)
11. [Integration with TOP.jl](#11-integration-with-topjl)
12. [Performance Optimizations](#12-performance-optimizations)
13. [Planned Optimizations](#13-planned-optimizations)

---

## 1. Problem Overview

### The Team Orienteering Problem (TOP)

The TOP is a combinatorial optimization problem where:

- **Objective**: Maximize total profit collected by a fleet of vehicles (drones)
- **Constraints**:
  - Each drone has a limited battery/time budget (`max_battery_time`)
  - Each customer (grid cell) can be visited at most once across all drones
  - All drones must start and end at depot locations (charging stations)
  - Drones move on a grid using Chebyshev distance (8-directional movement)

### Application Context

In this wildfire monitoring application:
- **Customers**: Grid cells with associated risk values (profits)
- **Depots**: Charging stations where drones recharge
- **Profit**: Risk value of each cell (from burn maps)
- **Travel Cost**: Chebyshev distance (L∞ norm) between cells

---

## 2. Algorithm Architecture

The implementation spans two main files:

| File | Purpose |
|------|---------|
| `TOP.jl` | Entry points, problem setup, route patching, greedy fallback |
| `TOP_PSO_multi_depot.jl` | Core PSO algorithm, split procedure, local search |

### High-Level Flow

```
Python API Call
      ↓
compute_TOP_plan_multiple_depots() [TOP.jl]
      ↓
CPA_multiple_depots() [TOP.jl]
      ↓
get_PSO_solution_multiple_depots() [TOP.jl]
      ↓
solve_PSO_TOP_multiple_depots() [TOP_PSO_multi_depot.jl]
      ↓
Movement Plan (routes for each drone)
```

---

## 3. Data Structures

### 3.1 Particle Structure

```julia
mutable struct Particle
    position::Vector{Int}          # Current permutation (giant tour)
    local_best::Vector{Int}        # Best known position for this particle
    local_best_profit::Float64     # Profit of local best
    current_profit::Float64        # Current profit
end
```

Each particle represents a **candidate solution** encoded as a permutation of customer indices.

### 3.2 PSO State Structure

```julia
mutable struct PSOiA_TOP_multiple_depots
    # Swarm state
    swarm::Vector{Particle}
    global_best::Vector{Int}
    global_best_profit::Float64
    
    # Algorithm parameters
    swarm_size::Int
    max_iterations::Int
    w::Float64                     # Inertia weight
    c1::Float64                    # Cognitive factor
    c2::Float64                    # Social factor
    ph::Float64                    # Probability of random move (IDCH)
    pm::Float64                    # Probability of local search
    
    # Problem parameters
    n_drones::Int
    n_pure_customers::Int
    max_battery_time::Int
    
    # Problem data
    customers::Vector{Tuple{Int,Int}}      # Customer coordinates
    profits::Vector{Float64}               # Customer profits (risk values)
    costs::Dict{Tuple{Int,Int}, Float64}   # Travel costs between nodes
    left_neighbors::Dict{Int, Vector{Int}} # Adjacency for shift optimization
    
    # Optimization helpers
    node_to_position::Vector{Dict{Int, Int}}  # Position lookup per particle
    accessible_customers::Vector{Int}          # Reachable customer indices
    depot_coord::Vector{Tuple{Int,Int}}        # Depot locations
    closest_depot_distance::Vector{Float64}    # Precomputed return distances
end
```

---

## 4. Solution Representation: The Giant Tour

### Concept

Instead of maintaining separate routes for each drone, the algorithm uses a **giant tour** representation:

```
Giant Tour: [depot₁, c₃, c₇, c₂, depot₂, c₁, c₅, c₄, ...]
```

This is a permutation of all customers (and depot markers) that encodes the visiting order. The actual routes for each drone are extracted using the **Split Procedure**.

### Benefits

1. **Uniform search space**: All solutions are permutations of the same length
2. **Crossover-friendly**: PSO operators work directly on permutations
3. **Flexible route assignment**: Split optimally assigns customers to drones

### Encoding Rules

- Customer indices: `1` to `n_pure_customers`
- Depot indices: `> n_pure_customers` (duplicated for multi-depot support)
- Artificial node: `0` (represents start/end in the cost matrix)

---

## 5. The Split Procedure

The Split Procedure is the **core evaluation mechanism** that converts a giant tour into optimal drone routes.

### Algorithm: `fast_split_with_routes_multiple_depots()`

#### Phase 1: Compute Saturated Tours

For each starting position `i` in the permutation:

```julia
for i in 1:n
    if permutation[i] <= n_pure_customers  # Skip non-depot starts
        succ[i] = i + 1
        continue
    end
    
    # Build maximal feasible tour from position i
    current_cost = 0.0
    current_profit = 0.0
    j = i + 1
    
    while j <= n
        customer_idx = permutation[j]
        travel_cost = costs[(prev_customer, customer_idx)]
        return_distance = closest_depot_distance[customer_idx]
        
        # Feasibility check: can we visit and return?
        if current_cost + travel_cost + return_distance > L
            break
        end
        
        current_cost += travel_cost
        current_profit += profits[customer_idx]
        j += 1
    end
    
    P[i] = current_profit      # Profit of saturated tour
    succ[i] = j <= n ? j : 0   # Next starting position
    tour_lengths[i] = j - i
end
```

**Key insight**: A saturated tour starting at position `i` visits as many consecutive customers as possible while respecting the battery constraint.

#### Phase 2: Dynamic Programming

Build a DP table `Γ[i, j]` = maximum profit using `j` drones from position `i` onwards:

```julia
Γ = zeros(n + 1, m + 1)

for i in n:-1:1
    for j in 0:m
        if j == 0
            Γ[i, j+1] = 0.0  # No drones, no profit
        else
            # Option 1: Skip position i
            Γ[i, j+1] = Γ[i + 1, j + 1]
            
            # Option 2: Use saturated tour starting at i
            profit_with_tour = P[i] + (succ[i] == 0 ? 0.0 : Γ[succ[i], j])
            Γ[i, j+1] = max(Γ[i, j+1], profit_with_tour)
        end
    end
end
```

**Recurrence relation**:
```
Γ(i, m) = max { Γ(i+1, m),  Γ(succ[i], m-1) + P[i] }
              ↑ Skip i      ↑ Use tour from i
```

#### Phase 3: Backtracking

Reconstruct the optimal routes by backtracking through the DP table:

```julia
routes = Vector{Vector{Int}}()
i = 1
j = m

while i <= n && j > 0 && length(routes) < m
    option1 = Γ[i + 1, j + 1]
    option2 = P[i] + (succ[i] == 0 ? 0.0 : Γ[succ[i], j])
    
    if abs(option2 - Γ[i, j + 1]) < 1e-10
        # Extract saturated tour from position i
        route = permutation[i : i + tour_lengths[i] - 1]
        push!(routes, route)
        i = succ[i] > 0 ? succ[i] : n + 1
        j -= 1
    else
        i += 1
    end
end
```

#### Complexity

- **Time**: O(n) for Phase 1, O(n × m) for Phase 2, O(n) for Phase 3
- **Space**: O(n × m) for the DP table

---

## 6. Swarm Initialization

### Three-Phase Initialization

```julia
function initialize_swarm(pso, use_greedy_init; skip_idch)
```

#### Phase 1: Random Initialization

Create `swarm_size` particles with random permutations:

```julia
for i in 1:swarm_size
    position = shuffle(accessible_customers)
    profit = fast_split_multiple_depots(position, pso)
    particle = Particle(position, position, profit, profit)
    push!(swarm, particle)
end
```

#### Phase 2: IDCH Heuristic (Optional)

Replace first few particles with solutions from **Iterative Destruction/Construction Heuristic**:

```julia
function idch_heuristic(pso, slow_version)
    current_solution = shuffle(accessible_customers)
    best_solution = current_solution
    best_profit = fast_split(best_solution, pso)
    
    while no_improvement < max_iter
        # Destruction: remove 1-3 random customers
        removed = remove_random(current_solution, 1:3)
        
        # Construction: reinsert using Best Insertion Algorithm
        reconstructed = best_insertion_algorithm(destroyed, removed, pso)
        
        profit = fast_split(reconstructed, pso)
        if profit > best_profit
            best_solution = reconstructed
            best_profit = profit
            no_improvement = 0
        else
            no_improvement += 1
        end
    end
    return best_solution
end
```

#### Phase 3: Greedy Fallback

Generate high-quality solutions using a greedy approach:

1. Create synthetic risk map from customer profits
2. For each drone, greedily select highest-risk reachable points
3. Convert coordinate-based routes to customer indices
4. Insert depot markers between drone routes

```julia
function initialize_with_greedy_fallback_two(pso)
    # First solution: greedy for all drones
    for drone_idx in 1:n_drones
        route = get_greedy_fallback_solution(
            risk_pertime, 
            previous_routes,  # Avoid overlap
            GridpointsDronesDetecting,
            ChargingStations,
            max_battery_time
        )
        tours_coordinates[drone_idx] = route
    end
    
    # Second solution: greedy avoiding first solution
    for drone_idx in 1:n_drones
        route = get_greedy_fallback_solution(
            risk_pertime,
            previous_routes + first_solution_routes,
            ...
        )
    end
    
    return [first_position, second_position], [profit1, profit2]
end
```

---

## 7. PSO Update Mechanism

### Position Update: `update_position!()`

Unlike traditional PSO with velocity vectors, this uses a **crossover-like operator** on permutations:

```julia
function update_position!(particle, global_best, pso)
    n = length(particle.position)
    
    # Calculate extraction sizes from each source
    r1, r2 = rand(), rand()
    n_current = floor(Int, w * n)
    n_local = floor(Int, (1 - w) * n * c1 * r1 / (c1 * r1 + c2 * r2))
    n_global = n - n_current - n_local
    
    # Phase 1: Extract subsequences in random order
    sources = [
        (particle.position, n_current),    # Inertia: keep from current
        (particle.local_best, n_local),     # Cognitive: learn from personal best
        (global_best, n_global)             # Social: learn from swarm best
    ]
    
    shuffle!(sources)
    M = Set{Int}()  # Marked (already extracted) customers
    subsequences = []
    
    for (source, target_length) in sources
        extracted = extract_subsequence(source, target_length, M)
        push!(subsequences, extracted)
    end
    
    # Phase 2: Link subsequences in random order
    shuffle!(subsequences)
    new_position = vcat(subsequences...)
    
    # Add remaining customers randomly
    remaining = setdiff(accessible_customers, M)
    append!(new_position, shuffle(remaining))
    
    particle.position = new_position
end
```

### Subsequence Extraction

```julia
function extract_subsequence(permutation, target_length, marked)
    n = length(permutation)
    r = rand(1:n)  # Random starting point
    extracted = Int[]
    
    # Browse from r to end
    for i in r:n
        if !(permutation[i] in marked) && length(extracted) < target_length
            push!(extracted, permutation[i])
            push!(marked, permutation[i])
        end
    end
    
    # Browse from r-1 down to 1
    for i in (r-1):-1:1
        if !(permutation[i] in marked) && length(extracted) < target_length
            pushfirst!(extracted, permutation[i])
            push!(marked, permutation[i])
        end
    end
    
    return extracted
end
```

---

## 8. Local Search Operators

### 8.1 Shift Operator

Move a single customer (or depot) from position `i` to position `j`:

```julia
function shift_operator!(particle, particle_idx, pso)
    for i in shuffle(1:n)
        node_i = particle.position[i]
        is_depot = node_i > n_pure_customers
        
        for j in shuffle(setdiff(1:n, [i]))
            # Optimization: skip if move won't help
            if !is_depot
                if is_blocking_once_inserted(i, j) && is_blocking_once_removed(i)
                    continue
                end
            end
            
            new_position = move_element(particle.position, i, j)
            new_profit = fast_split(new_position, pso)
            
            if new_profit > particle.current_profit
                particle.position = new_position
                particle.current_profit = new_profit
                # Update node-to-position mapping incrementally
                return true
            end
        end
    end
    return false
end
```

**Blocking check optimization**: Skip moves that don't change route structure:

```julia
function is_blocking_once_inserted(particle, i, j, pso)
    # Check if inserting node i at position j would block (cost > L)
    if j == 1 return false end
    L = pso.max_battery_time
    node_i = particle.position[i]
    node_j_pred = particle.position[j-1]
    return get(pso.costs, (node_j_pred, node_i), L*4) > L
end
```

### 8.2 Swap Operator

Exchange two customers at positions `i` and `j`:

```julia
function swap_operator!(particle, particle_idx, pso)
    pos = particle.position
    
    for i in shuffle(1:n)
        for j in shuffle((i+1):n)
            # Skip if both positions would block
            if is_blocking_once_inserted(i, j) && is_blocking_once_inserted(j, i)
                continue
            end
            
            # Trial swap
            pos[i], pos[j] = pos[j], pos[i]
            new_profit = fast_split(pos, pso)
            
            if new_profit > particle.current_profit
                particle.current_profit = new_profit
                # Update mapping: O(1) - just 2 updates
                pso.node_to_position[particle_idx][pos[i]] = i
                pso.node_to_position[particle_idx][pos[j]] = j
                return true
            else
                # Revert swap
                pos[i], pos[j] = pos[j], pos[i]
            end
        end
    end
    return false
end
```

### 8.3 Destruction/Repair Operator

Remove random customers and reinsert using Best Insertion Algorithm:

```julia
function destruction_repair_operator!(particle, particle_idx, pso)
    n = length(particle.position)
    max_remove = max(1, n ÷ n_drones)
    n_remove = rand(1:max_remove)
    
    # Remove random customers
    new_position = copy(particle.position)
    removed = Int[]
    for _ in 1:n_remove
        idx = rand(1:length(new_position))
        push!(removed, new_position[idx])
        deleteat!(new_position, idx)
    end
    
    # Reconstruct using Best Insertion Algorithm
    reconstructed = best_insertion_algorithm(new_position, removed, pso)
    new_profit = fast_split(reconstructed, pso)
    
    if new_profit > particle.current_profit
        particle.position = reconstructed
        particle.current_profit = new_profit
        # Recompute full mapping (O(n))
        pso.node_to_position[particle_idx] = compute_node_to_position(reconstructed)
        return true
    end
    return false
end
```

### Best Insertion Algorithm (BIA)

Used in IDCH and destruction/repair:

```julia
function best_insertion_algorithm(partial_solution, unrouted, pso)
    solution = copy(partial_solution)
    remaining = copy(unrouted)
    α = rand() * 2.0  # Random parameter for profit weighting
    
    while !isempty(remaining)
        best_customer = -1
        best_position = -1
        best_cost = Inf
        
        for customer in remaining
            customer_profit = pso.profits[customer]
            
            for pos in 1:(length(solution) + 1)
                # Calculate insertion cost
                # C_i,z + C_z,j - C_i,j - (P_z)^α
                cost_iz = cost_to_customer(solution, pos-1, customer)
                cost_zj = cost_from_customer(customer, solution, pos)
                cost_ij = direct_cost(solution, pos-1, pos)
                
                insertion_cost = cost_iz + cost_zj - cost_ij - (customer_profit^α)
                
                if insertion_cost < best_cost
                    best_cost = insertion_cost
                    best_customer = customer
                    best_position = pos
                end
            end
        end
        
        if best_customer != -1
            insert!(solution, best_position, best_customer)
            filter!(x -> x != best_customer, remaining)
        else
            break
        end
    end
    
    return solution
end
```

### Local Search Integration

```julia
function local_search!(particle, particle_idx, pso)
    improved = true
    
    while improved
        improved = false
        neighborhoods = shuffle([1, 2])  # shift, swap (destruction/repair optional)
        
        for neighborhood in neighborhoods
            if neighborhood == 1
                improved = shift_operator!(particle, particle_idx, pso)
            elseif neighborhood == 2
                improved = swap_operator!(particle, particle_idx, pso)
            end
            
            if improved
                break  # Restart from first neighborhood
            end
        end
    end
end
```

---

## 9. Diversity Management

### Local Best Update with Similarity Check

```julia
function update_local_bests!(pso, δ = 1e-6)
    sorted_indices = sortperm([p.local_best_profit for p in pso.swarm])
    worst_idx = sorted_indices[1]
    
    for particle in pso.swarm
        # Rule 1: Only update if better than worst
        if particle.current_profit > pso.swarm[worst_idx].local_best_profit
            current_cost = calculate_travel_cost(particle.position, pso)
            
            # Rule 2: Find similar particle
            similar_found = false
            for (i, other) in enumerate(pso.swarm)
                other_cost = calculate_travel_cost(other.local_best, pso)
                
                # Similarity: same profit AND similar travel cost
                if abs(other.local_best_profit - particle.current_profit) < δ &&
                   abs(other_cost - current_cost) < δ
                    pso.swarm[i].local_best = copy(particle.position)
                    pso.swarm[i].local_best_profit = particle.current_profit
                    similar_found = true
                    break
                end
            end
            
            # Rule 3: Replace worst if no similar found
            if !similar_found
                pso.swarm[worst_idx].local_best = copy(particle.position)
                pso.swarm[worst_idx].local_best_profit = particle.current_profit
            end
        end
        
        # Update personal best
        if particle.current_profit > particle.local_best_profit
            particle.local_best = copy(particle.position)
            particle.local_best_profit = particle.current_profit
        end
        
        # Update global best
        if particle.current_profit > pso.global_best_profit
            pso.global_best = copy(particle.position)
            pso.global_best_profit = particle.current_profit
        end
    end
end
```

---

## 10. Algorithm Flow

### Main PSO Loop

```julia
function solve_PSO_TOP_multiple_depots(...)
    # 1. Setup: Determine accessible customers
    accessible_customers = compute_accessible_customers(customers, depot_coord, max_battery_time, blocked)
    
    # 2. Precompute closest depot distances for feasibility checks
    closest_depot_distance = precompute_return_distances(customers, depot_coord, blocked)
    
    # 3. Initialize PSO state
    pso = PSOiA_TOP_multiple_depots(...)
    
    # 4. Initialize swarm (random + IDCH + greedy)
    initialize_swarm(pso, use_greedy_init)
    
    # 5. Main loop
    iter = 1
    while iter <= max_iterations && elapsed_time < max_time
        improvement_found = false
        
        for x in 1:swarm_size
            # Random move with probability ph
            if rand() < ph
                pso.swarm[x].position = idch_heuristic(pso)
            else
                update_position!(pso.swarm[x], pso.global_best, pso)
            end
            
            # Local search with probability pm
            if rand() < pm
                local_search!(pso.swarm[x], x, pso)
            end
            
            # Evaluate solution
            pso.swarm[x].current_profit = fast_split(pso.swarm[x].position, pso)
            
            # Update local bests and check for improvement
            prev_best = pso.global_best_profit
            update_local_bests!(pso)
            
            if pso.global_best_profit > prev_best
                improvement_found = true
            end
        end
        
        # Reset iteration counter on improvement (intensification)
        if improvement_found
            iter = 1
        else
            iter += 1
        end
    end
    
    return pso.global_best, pso.global_best_profit, pso
end
```

---

## 11. Integration with TOP.jl

### Entry Point: `compute_TOP_plan_multiple_depots()`

```julia
function compute_TOP_plan_multiple_depots(
    risk_pertime_file,    # Path to burn map (.npy)
    n_drones,             # Number of drones
    ChargingStations,     # List of depot coordinates
    GroundStations,       # Cells to avoid (ground assets)
    max_battery_time,     # Battery limit in time steps
    t,                    # Current time step
    verbose,
    initial_drone_positions,
    mask_filename         # Optional mask for blocked cells
)
```

### Route Post-Processing

After PSO returns the giant tour, routes are converted to actual movements:

```julia
# 1. Extract routes from giant tour
routes, tours_coordinates = CPA_multiple_depots(...)

# 2. Patch routes to include intermediate cells
tours_coordinates = get_patched_tours_coordinates(routes, GridpointsDronesDetecting, ChargingStations, n_drones, blocked)

# 3. Handle short tours with greedy extension
for s in 1:n_drones
    if length(tours_coordinates[s]) <= max_battery_time - 1
        remaining_time = max_battery_time - length(tours_coordinates[s])
        extension = get_greedy_fallback_solution(...)
        tours_coordinates[s] = [tours_coordinates[s]; extension[2:end]]
    end
end

# 4. Convert to movement plan format
movement_plan = build_movement_plan(tours_coordinates, max_battery_time, n_drones)
```

### Path Patching with Highest Risk

When patching between waypoints, the algorithm uses dynamic programming to find the path that maximizes collected risk:

```julia
function patch_path_with_highest_risk!(route, target_point, risk_pertime, blocked)
    # DP to find highest-risk path using Chebyshev distance
    # Only considers points on shortest paths (start_dist + target_dist == total_dist)
    # Handles blocked cells by falling back to BFS
end
```

---

## 12. Performance Optimizations

### 12.1 Precomputed Data Structures

| Structure | Purpose | Complexity |
|-----------|---------|------------|
| `closest_depot_distance` | Feasibility check in split | O(1) lookup |
| `left_neighbors` | Skip impossible shifts | O(1) lookup |
| `node_to_position` | Track node positions in permutation | O(1) lookup, incremental update |
| `accessible_customers` | Prune unreachable nodes | One-time O(n) computation |

### 12.2 Blocking Check Optimization

Before evaluating a shift/swap, check if it would actually change the route structure:

```julia
# Only evaluate if the move might improve the solution
if !is_depot
    if is_blocking_once_inserted(i, j) && is_blocking_once_removed(i)
        continue  # Skip evaluation
    end
end
```

### 12.3 Incremental Mapping Updates

Instead of recomputing `node_to_position` after every move:

```julia
# Shift: O(j - i) updates instead of O(n)
# Swap: O(1) - just 2 updates
# Destruction/Repair: O(n) - full recompute (unavoidable)
```

### 12.4 BFS for Masked Grids

When a mask defines blocked cells, use BFS instead of L∞ distance:

```julia
if !isempty(blocked)
    GridpointsDrones_set, _ = get_drone_gridpoints_BFS(ChargingStations, max_battery_time/2, I, N, M)
else
    GridpointsDrones_set = get_drone_gridpoints(ChargingStations, max_battery_time/2, I)
end
```

### 12.5 Time Limit Enforcement

The algorithm respects a `max_time` parameter:

```julia
while iter <= max_iterations
    elapsed = time() - start_time
    if elapsed > max_time
        println("[TIME CHECK] Maximum time limit reached. Stopping.")
        break
    end
    # ... rest of iteration
end
```

---

## Summary

This PSO-inspired algorithm for TOP combines several key innovations:

1. **Giant Tour Representation**: Unified encoding for multi-drone routes
2. **Efficient Split Procedure**: O(nm) optimal route extraction via dynamic programming
3. **Hybrid Initialization**: Random + IDCH + Greedy for solution diversity
4. **Crossover-based Update**: Permutation-friendly position update operator
5. **Multi-neighborhood Local Search**: Shift, swap, and destruction/repair operators
6. **Diversity Management**: Similarity-based local best updates
7. **Adaptive Iteration**: Counter reset on improvement for intensification

The algorithm efficiently handles real-world constraints including multiple depots, blocked cells (via mask), and time limits for online decision-making in wildfire drone routing scenarios.

---

## 13. Planned Optimizations

This section documents optimizations to be implemented to improve algorithm performance, particularly for grid-based instances where nodes have limited valid neighbors.

### 13.1 Split Procedure Optimization via Sparse Data Structures

#### Motivation

The current Split Procedure has complexity O(n) for Phase 1 and O(n × m) for Phase 2, where n is the permutation length and m is the number of drones. However, for grid-based TOP instances:

1. **Depot nodes are sparse**: Only k depot nodes exist in the permutation, where k = (number of physical depots) × (depot duplicates per depot). Typically k << n.
2. **Tours can only start at depots**: Non-depot positions cannot be tour starting points, yet we iterate over all n positions.
3. **DP values propagate unchanged between depots**: For non-depot positions, the DP recurrence simplifies to identity.

By exploiting this structure, we can reduce complexity from O(n × m) to O(k² × m + L) (or O(k × m × log k + L) with binary search), where L is the sum of saturated tour lengths.

#### Data Structure Changes

##### 1. Move `node_to_position` into the Particle struct

**Current structure:**
```julia
mutable struct Particle
    position::Vector{Int}
    local_best::Vector{Int}
    local_best_profit::Float64
    current_profit::Float64
end

mutable struct PSOiA_TOP_multiple_depots
    # ...
    node_to_position::Vector{Dict{Int, Int}}  # One dict per particle
    # ...
end
```

**New structure:**
```julia
mutable struct Particle
    position::Vector{Int}
    local_best::Vector{Int}
    local_best_profit::Float64
    current_profit::Float64
    node_to_position::Vector{Int}  # Maps node → position in this particle's permutation
end
```

This change:
- Improves data locality (particle data is co-located)
- Makes the mapping directly accessible from the particle
- Simplifies function signatures (no need to pass particle index)

**Vector vs Dict for `node_to_position`:**
- Prefer `Vector{Int}` because node IDs are contiguous `1:n` (no gaps).
- This yields better cache locality and faster indexing.
- A `Dict` is only beneficial if node IDs are sparse or non-contiguous.

##### 2. Accessing depot positions on-the-fly

Depot nodes are identified as indices > `n_pure_customers`. To get all depot positions for a particle:

```julia
function get_sorted_depot_positions(node_to_position::Vector{Int}, n_pure_customers::Int)
    depot_positions = Int[]
    for node in (n_pure_customers + 1):length(node_to_position)
        push!(depot_positions, node_to_position[node])
    end
    sort!(depot_positions)
    return depot_positions
end
```

This runs in O(k) time where k is the number of depot nodes (iterating only over depot nodes),
plus O(k log k) for sorting. Sorting is required because depot positions are not ordered in the
permutation even if node IDs are.

**Why sorting is OK:**
- k is small (typically D × m) and `k log k` is negligible compared to Phase 2's `O(k² × m)`.

#### Phase 1 Optimization: Sparse Saturated Tour Computation

**Current implementation** (baseline) iterates over all n positions:
```julia
for i in 1:n
    if permutation[i] <= n_pure_customers
        succ[i] = i + 1
        continue
    end
    # Build saturated tour...
end
```

**Optimized implementation** iterates only over depot positions:

```julia
function compute_saturated_tours_sparse(
    permutation::Vector{Int},
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    n_pure_customers = pso.n_pure_customers
    L = pso.max_battery_time
    
    # Get sorted depot positions from the node-to-position mapping
    sorted_depot_positions = get_sorted_depot_positions(particle.node_to_position, n_pure_customers)
    k = length(sorted_depot_positions)
    
    # Sparse arrays indexed by depot index (1 to k), not position
    P_sparse = zeros(Float64, k)           # Profit of saturated tour
    succ_sparse = zeros(Int, k)            # Next position after tour ends (0 if extends to end)
    tour_lengths_sparse = zeros(Int, k)    # Length of saturated tour
    
    for idx in 1:k
        depot_pos = sorted_depot_positions[idx]
        
        # Build saturated tour starting from depot_pos
        current_cost = 0.0
        current_profit = 0.0
        prev_customer = permutation[depot_pos]  # The depot node itself
        j = depot_pos + 1
        
        while j <= n
            customer_idx = permutation[j]
            
            # Get travel cost (default to infeasible if not found)
            travel_cost = get(pso.costs, (prev_customer, customer_idx), L * 4)
            
            # Get return distance to closest depot (precomputed)
            return_distance = pso.closest_depot_distance[customer_idx]
            
            # Feasibility check
            if current_cost + travel_cost + return_distance > L
                break
            end
            
            current_cost += travel_cost
            current_profit += pso.profits[customer_idx]
            prev_customer = customer_idx
            j += 1
        end
        
        P_sparse[idx] = current_profit
        tour_lengths_sparse[idx] = j - depot_pos
        succ_sparse[idx] = (j <= n) ? j : 0  # 0 indicates tour extends to end
    end
    
    return P_sparse, succ_sparse, tour_lengths_sparse, sorted_depot_positions
end
```

**Complexity analysis:**
- Extracting depot positions: O(k) where k = number of depot nodes
- Sorting depot positions: O(k log k)
- Building saturated tours: O(L_total) where L_total = Σ(tour lengths)
  - Each depot extends its tour until battery is exhausted
  - Maximum extension per depot is `max_battery_time`
  - Total work ≤ k × max_battery_time, but typically less

**Total Phase 1 complexity: O(k log k + L_total)** instead of O(n)

#### Phase 2 Optimization: Sparse Dynamic Programming

**Key insight**: For non-depot positions, the DP value propagates unchanged.

Consider the recurrence at a non-depot position i where `P[i] = 0` and `succ[i] = i + 1`:
```
Γ[i, j] = max(Γ[i+1, j], P[i] + Γ[succ[i], j-1])
        = max(Γ[i+1, j], 0 + Γ[i+1, j-1])
        = max(Γ[i+1, j], Γ[i+1, j-1])
```

Since having more drones available never decreases profit (Γ[i+1, j] ≥ Γ[i+1, j-1]), this simplifies to:
```
Γ[i, j] = Γ[i+1, j]   (for non-depot positions)
```

**Consequence**: Between consecutive depot positions p_a and p_b (where p_a < p_b), all intermediate positions have the same DP value as p_b:
```
Γ[p_a + 1, j] = Γ[p_a + 2, j] = ... = Γ[p_b - 1, j] = Γ[p_b, j]
```

This means we only need to compute and store DP values at depot positions.

**Sparse DP implementation:**

```julia
function sparse_dp_phase2(
    P_sparse::Vector{Float64},
    succ_sparse::Vector{Int},
    sorted_depot_positions::Vector{Int},
    m::Int,  # Number of drones
    n::Int   # Permutation length
)
    k = length(sorted_depot_positions)
    
    # Γ_sparse[idx, j] = max profit using j drones from depot index idx onwards
    # Dimensions: (k+1) × (m+1) to handle boundary conditions
    # idx ranges from 1 to k, j ranges from 0 to m
    # We use 1-based indexing for j as well: Γ_sparse[idx, j+1] stores Γ(idx, j)
    Γ_sparse = zeros(Float64, k + 1, m + 1)
    
    # Boundary condition: Γ_sparse[k+1, :] = 0 (no more depots)
    # Already initialized to 0
    
    # Fill DP table in reverse order of depot indices
    for idx in k:-1:1
        depot_pos = sorted_depot_positions[idx]
        
        for j in 1:m
            # Option 1: Skip this depot (go to next depot)
            # If idx == k, there's no next depot, so skip_value = 0
            skip_value = Γ_sparse[idx + 1, j + 1]
            
            # Option 2: Use the saturated tour starting at this depot
            succ_pos = succ_sparse[idx]
            
            if succ_pos == 0
                # Tour extends to end of permutation, no remaining profit
                remaining_value = 0.0
            else
                # Find the first depot at or after succ_pos (binary search)
                remaining_value = lookup_Γ_sparse(
                    succ_pos, j - 1,
                    sorted_depot_positions, Γ_sparse
                )
            end
            
            use_value = P_sparse[idx] + remaining_value
            
            Γ_sparse[idx, j + 1] = max(skip_value, use_value)
        end
    end
    
    return Γ_sparse
end
```

**Lookup function for sparse DP:**

The key challenge is that `succ_sparse[idx]` returns a *position* in the permutation, but we store DP values by *depot index*. We need to find the depot index corresponding to the first depot at or after a given position.

```julia
function lookup_Γ_sparse(
    position::Int,
    j::Int,  # Number of drones (0 to m)
    sorted_depot_positions::Vector{Int},
    Γ_sparse::Matrix{Float64}
)
    # Handle boundary: no drones left
    if j < 0
        return 0.0
    end

    # Binary search for first depot at or after 'position'
    idx = searchsortedfirst(sorted_depot_positions, position)
    if idx > length(sorted_depot_positions)
        return 0.0
    end
    return Γ_sparse[idx, j + 1]
end
```

**Complexity analysis:**
- DP table fill: O(k × m) iterations
- Each iteration: O(1) for skip option, O(log k) for lookup in use option
- Total Phase 2 complexity: O(k × m × log k)

For small k (which is typical), this is **much better than O(n × m)**.

**Boundary conditions explained:**

| Condition | Handling | Justification |
|-----------|----------|---------------|
| `idx == k` (last depot), skip option | `Γ_sparse[k+1, j+1] = 0` | No more depots after the last one |
| `succ_pos == 0` (tour extends to end) | `remaining_value = 0` | No customers left to visit after this tour |
| `succ_pos > sorted_depot_positions[k]` | `lookup_Γ_sparse` returns 0 | Position is after all depots |
| `j == 0` (no drones left) | `Γ_sparse[idx, 1] = 0` | Cannot collect profit without drones |

**Final answer extraction:**

The final answer is `Γ[1, m]` in the original dense formulation, which represents the maximum profit starting from position 1 with m drones.

In sparse form, position 1 may be before the first depot. Since values propagate unchanged between depots, `Γ[1, m] = Γ[first_depot_pos, m] = Γ_sparse[1, m+1]`.

However, if the first depot is at position p₁ > 1, and there are customers at positions 1 to p₁-1, these customers cannot be in any tour (no tour can start before the first depot). The propagation rule handles this correctly.

```julia
# Final answer
optimal_profit = Γ_sparse[1, m + 1]
```

Alternatively, using the lookup function for consistency:
```julia
optimal_profit = lookup_Γ_sparse(1, m, sorted_depot_positions, Γ_sparse)
```

#### Phase 3 Optimization: Sparse Backtracking

**Current implementation** iterates position-by-position:
```julia
i = 1
while i <= n && j > 0
    if "use tour at i"
        # extract route
        i = succ[i]
        j -= 1
    else
        i += 1
    end
end
```

**Optimized implementation** iterates depot-by-depot:

```julia
function sparse_backtracking(
    P_sparse::Vector{Float64},
    succ_sparse::Vector{Int},
    tour_lengths_sparse::Vector{Int},
    sorted_depot_positions::Vector{Int},
    Γ_sparse::Matrix{Float64},
    permutation::Vector{Int},
    m::Int,
    n::Int
)
    routes = Vector{Vector{Int}}()
    
    idx = 1  # Current depot index
    j = m    # Remaining drones
    k = length(sorted_depot_positions)
    
    while idx <= k && j > 0 && length(routes) < m
        depot_pos = sorted_depot_positions[idx]
        
        # Compute both options to determine which was chosen
        skip_value = Γ_sparse[idx + 1, j + 1]
        
        succ_pos = succ_sparse[idx]
        if succ_pos == 0
            remaining_value = 0.0
        else
            remaining_value = lookup_Γ_sparse(succ_pos, j - 1, sorted_depot_positions, Γ_sparse)
        end
        use_value = P_sparse[idx] + remaining_value
        
        # Check which option matches the stored DP value
        if abs(use_value - Γ_sparse[idx, j + 1]) < 1e-10
            # Use the tour starting at this depot
            tour_start = depot_pos
            tour_end = depot_pos + tour_lengths_sparse[idx] - 1
            route = permutation[tour_start:tour_end]
            push!(routes, route)
            
            # Move to next depot at or after succ_pos
            if succ_pos == 0 || succ_pos > n
                break  # Tour extends to end, we're done
            end
            
            # Find next depot index at or after succ_pos
            next_idx = find_next_depot_index(succ_pos, sorted_depot_positions)
            if next_idx == 0
                break  # No more depots
            end
            idx = next_idx
            j -= 1
        else
            # Skip this depot, move to next
            idx += 1
        end
    end
    
    return routes
end

function find_next_depot_index(position::Int, sorted_depot_positions::Vector{Int})
    # Binary search for first depot at or after 'position'
    idx = searchsortedfirst(sorted_depot_positions, position)
    return idx <= length(sorted_depot_positions) ? idx : 0
end
```

**Complexity**: O(k) iterations with O(log k) lookup per iteration → O(k log k) total.

#### Complete Sparse Split Procedure

```julia
function fast_split_sparse(
    permutation::Vector{Int},
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    m = pso.n_drones
    
    # Empty TourIntervals for edge cases
    empty_intervals = TourIntervals(Tuple{Int,Int}[], 0)
    
    if n == 0
        return 0.0, Vector{Vector{Int}}(), empty_intervals
    end
    
    # Phase 1: Compute saturated tours (sparse)
    P_sparse, succ_sparse, tour_lengths_sparse, sorted_depot_positions = 
        compute_saturated_tours_sparse(permutation, particle, pso)
    
    k = length(sorted_depot_positions)
    if k == 0
        return 0.0, Vector{Vector{Int}}(), empty_intervals  # No depots in permutation
    end
    
    # Phase 2: Dynamic programming (sparse)
    Γ_sparse = sparse_dp_phase2(P_sparse, succ_sparse, sorted_depot_positions, m, n)
    
    # Phase 3: Backtracking (sparse)
    routes = sparse_backtracking(
        P_sparse, succ_sparse, tour_lengths_sparse,
        sorted_depot_positions, Γ_sparse, permutation, m, n
    )
    
    optimal_profit = Γ_sparse[1, m + 1]
    
    # Build tour intervals for boundary optimization (Section 13.2)
    tour_intervals = build_tour_intervals(sorted_depot_positions, tour_lengths_sparse)
    
    return optimal_profit, routes, tour_intervals
end
```

#### Complexity Summary

| Phase | Original | Optimized | Improvement |
|-------|----------|-----------|-------------|
| Phase 1 | O(n) | O(k log k + L) | Significant when k << n |
| Phase 2 | O(n × m) | O(k × m × log k) | Significant when k << n |
| Phase 3 | O(n) | O(k log k) | Significant when k << n |
| **Total** | **O(n × m)** | **O(k × m × log k + L)** | **k << n typically** |

Where:
- n = permutation length (number of customers + depots)
- m = number of drones
- k = number of depot nodes in permutation
- L = sum of saturated tour lengths ≤ k × max_battery_time

**When is this optimization effective?**
- k << n: Many customers, few depots (typical in grid-based wildfire monitoring)
- Example: n = 1000 customers, k = 10 depot nodes, m = 5 drones
  - Original: O(1000 × 5) = O(5000)
  - Optimized: O(10 × 5 × log(10) + 500) ≈ O(170 + 500) = O(670)
  - ~7x improvement

#### Implementation Checklist

1. **Modify `Particle` struct** to include `node_to_position::Vector{Int}`

2. **Remove `node_to_position` from `PSOiA_TOP_multiple_depots`** struct

3. **Update all particle creation code** to initialize the mapping:
   ```julia
   particle = Particle(position, position, profit, profit, compute_node_to_position(position))
   ```

4. **Add helper function** `get_sorted_depot_positions(particle, n_pure_customers)`

5. **Implement sparse Phase 1**: `compute_saturated_tours_sparse()`

6. **Implement sparse Phase 2**: `sparse_dp_phase2()` and `lookup_Γ_sparse()`

7. **Implement sparse Phase 3**: `sparse_backtracking()` and `find_next_depot_index()`

8. **Replace `fast_split_with_routes_multiple_depots()`** with the new sparse version

9. **Update all callers** that pass `particle_idx` to access `node_to_position` to use `particle.node_to_position` directly

10. **Update incremental mapping updates** in shift/swap operators to use the new particle field

11. **Handle `fast_split_sparse` on arbitrary permutations:**
    
    The sparse split requires a `node_to_position` mapping. For functions that evaluate 
    candidate permutations (not particle positions), we need different strategies:
    
    - **IDCH, best_insertion_algorithm, update_position!**: 
      Compute a fresh mapping for the candidate permutation, and reuse it during split:
      ```julia
      candidate_mapping = compute_node_to_position(candidate_permutation)
      # Option A: call an overload that accepts a particle (mapping stored inside)
      temp_particle = Particle(candidate_permutation, candidate_permutation, 0.0, 0.0, candidate_mapping)
      profit, routes, tour_intervals = fast_split_sparse(candidate_permutation, temp_particle, pso)
      ```
      If accepted, assign the mapping to the particle's `node_to_position` field.
    
    - **shift_operator!, swap_operator!**: 
      These already have the particle's mapping. For trial evaluations:
      1. Temporarily modify the mapping to reflect the move
      2. Evaluate `fast_split_sparse` using the modified mapping
      3. If rejected, revert the mapping changes
      
      For swap, reverting is O(1) (swap back two entries).
      For shift, reverting is O(|j - i|) (shift entries back).

12. **Alternative: Overload `fast_split_sparse` with two signatures**
    
    ```julia
    # Version 1: Use provided particle (has node_to_position mapping)
    function fast_split_sparse(permutation, particle, pso)
        sorted_depot_positions = get_sorted_depot_positions(particle.node_to_position, pso.n_pure_customers)
        # ... rest of sparse split ...
        tour_intervals = build_tour_intervals(sorted_depot_positions, tour_lengths_sparse)
        return optimal_profit, routes, tour_intervals
    end
    
    # Version 2: Compute mapping on-the-fly (for IDCH, etc.)
    function fast_split_sparse(permutation, pso)
        node_to_position = compute_node_to_position(permutation)
        # Create a temporary particle-like object or pass mapping directly
        sorted_depot_positions = get_sorted_depot_positions(node_to_position, pso.n_pure_customers)
        # ... rest of sparse split ...
        tour_intervals = build_tour_intervals(sorted_depot_positions, tour_lengths_sparse)
        return optimal_profit, routes, tour_intervals
    end
    ```
    
    This allows callers to choose between efficiency (reuse mapping) and convenience (auto-compute).
    Both versions return `(profit, routes, tour_intervals)` for consistency.

### 13.2 Shift/Swap Boundary Optimization (Skip No-Op Moves)

This optimization reduces the number of expensive split evaluations during local search by
skipping moves that provably cannot change the split profit.

#### Key Idea

Only customers that lie inside at least one **saturated tour interval** can contribute to the
profit computed by the split procedure. If a move only affects positions that are **outside**
all saturated tours, the split profit is guaranteed to stay unchanged.

We can precompute the union of saturated tour intervals and use it to quickly reject no-op
shift/swap moves.

#### Definitions

For each depot start position `dpos`, Phase 1 yields a saturated tour length `l`.
That tour covers the interval:

- `I = (dpos, dpos + l - 1]`  (positions after the depot node)

Let `U` be the union of all such intervals across depots (merged and sorted).
To ensure shifts that cross depot positions are not skipped, include depot positions
as zero-length intervals `(dpos, dpos)` in `U`. This is built from **all Phase 1
saturated tours**, not just the routes selected by Phase 3.

#### Diagram (tours vs. dead zones)

```
Permutation: [D1, c1, c2, c3, c4, c5, D2, c6, c7, c8, D3, c9, ...]
              ^   |---------|  ^^^^^   ^   |---|  ^^^^^^
              |   tour 1        dead    |   tour2  dead
              depot1                   depot2
```

- `tour 1` covers the interval `(D1, end1]`
- `dead` positions are outside all saturated tours

#### Safe Skip Conditions

- **Shift (move position `i` to `j`)**:
  - This affects all positions in `R = [min(i, j), max(i, j)]`.
  - If `R` does **not** intersect `U`, the move cannot change the split profit.
  - Since `U` includes depot positions, this also prevents skipping shifts that cross depots.

- **Swap (positions `i` and `j`)**:
  - This affects only positions `i` and `j`.
  - If **both** `i` and `j` are outside `U`, the move cannot change the split profit.

These checks are **conservative and correct**. If the move might change profit, it is still
evaluated normally.

#### Why the Range Check is Necessary for Shifts

A shift from position `i` to position `j` (where `i < j`) works as follows:
1. The node at position `i` is removed
2. All nodes at positions `i+1, i+2, ..., j` shift LEFT by 1
3. The removed node is inserted at position `j`

This means **intermediate positions change content**, even if `i` and `j` are both outside `U`.

**Example:**
```
Before: [n1, n2, n3, n4, n5, n6]   Tour interval U = [3, 4, 5]
         i=1                j=6   (both outside U)
         
After:  [n2, n3, n4, n5, n6, n1]
              ^^^^^^^^^^^
              Positions 3,4,5 now contain n4,n5,n6 instead of n3,n4,n5
```

The tour content changes because nodes shifted into/out of the tour interval. Therefore, we 
must check if the **entire range** `[min(i,j), max(i,j)]` intersects `U`, not just the endpoints.

For **swap**, only positions `i` and `j` exchange content—no intermediate positions are 
affected. Therefore, checking just those two positions is sufficient.

#### Implementation Sketch

1. **Build saturated tours** during Phase 1:
   - For each depot start position, compute `endpos = dpos + l - 1`.
2. **Build and merge intervals**:
   - Sort intervals by start, merge overlaps into a compact union list `U`.
3. **Intersection check**:
   - Use binary search on `U` to test whether a position (swap) or range (shift) intersects.

#### Notes

- Always evaluate moves involving a **depot node**, since depot positions define tour starts
  and can alter DP structure even when profit is zero.
- A cheap fallback check is: if the move is strictly after `max(endpos)`, it is a no-op.
  This is weaker but nearly free.

#### Detailed Implementation

**Note on modified function signatures**: The optimized `shift_operator!` and `swap_operator!` 
functions take an additional `tour_intervals` parameter and return `(Bool, TourIntervals)` 
instead of just `Bool`. This allows the `local_search!` function to pass updated intervals 
between operator calls without recomputing them from scratch.

##### Data Structure for Tour Intervals

```julia
struct TourIntervals
    intervals::Vector{Tuple{Int,Int}}  # Sorted, non-overlapping (start, end) pairs
    max_end::Int                        # Maximum endpoint for quick rejection
end

function build_tour_intervals(
    sorted_depot_positions::Vector{Int},
    tour_lengths_sparse::Vector{Int}
)
    intervals = Tuple{Int,Int}[]
    
    for (idx, depot_pos) in enumerate(sorted_depot_positions)
        # Include depot positions as zero-length intervals to prevent depot-crossing skips
        push!(intervals, (depot_pos, depot_pos))
        tour_start = depot_pos + 1  # First customer after depot
        tour_end = depot_pos + tour_lengths_sparse[idx] - 1
        if tour_end >= tour_start  # Non-empty tour
            push!(intervals, (tour_start, tour_end))
        end
    end
    
    # Sort by start position
    sort!(intervals, by=first)
    
    # Merge overlapping intervals
    merged = Tuple{Int,Int}[]
    for (s, e) in intervals
        if isempty(merged) || s > merged[end][2] + 1
            push!(merged, (s, e))
        else
            # Extend the last interval
            merged[end] = (merged[end][1], max(merged[end][2], e))
        end
    end
    
    max_end = isempty(merged) ? 0 : merged[end][2]
    return TourIntervals(merged, max_end)
end
```

##### Intersection Check Functions

```julia
"""
Check if a range [range_start, range_end] intersects any tour interval.
Used for shift operations.
"""
function intersects_range(ti::TourIntervals, range_start::Int, range_end::Int)
    # Quick rejection: range is entirely after all tours
    if range_start > ti.max_end
        return false
    end
    
    # Quick rejection: empty intervals
    if isempty(ti.intervals)
        return false
    end
    
    # Binary search for first interval with start >= range_start
    idx = searchsortedfirst(ti.intervals, (range_start, 0), by=first)
    
    # Check interval before idx (if exists): does its end reach into our range?
    if idx > 1 && ti.intervals[idx-1][2] >= range_start
        return true
    end
    
    # Check interval at idx (if exists): does its start fall within our range?
    if idx <= length(ti.intervals) && ti.intervals[idx][1] <= range_end
        return true
    end
    
    return false
end

"""
Check if a single position is inside any tour interval.
Used for swap operations.
"""
function is_active(ti::TourIntervals, pos::Int)
    return intersects_range(ti, pos, pos)
end
```

##### Integration with Shift Operator

```julia
function shift_operator!(
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots,
    tour_intervals::TourIntervals
)
    n = length(particle.position)
    
    for i in shuffle(1:n)
        node_i = particle.position[i]
        is_depot = node_i > pso.n_pure_customers
        
        for j in shuffle(setdiff(1:n, [i]))
            # === BOUNDARY OPTIMIZATION ===
            # Skip if move cannot affect any tour (customer moves in dead zone)
            if !is_depot
                range_start = min(i, j)
                range_end = max(i, j)
                if !intersects_range(tour_intervals, range_start, range_end)
                    continue  # No-op move, skip evaluation
                end
            end
            
            # === EXISTING BLOCKING CHECK ===
            if !is_depot
                if is_blocking_once_inserted(particle, i, j, pso) && 
                   is_blocking_once_removed(particle, i, pso)
                    continue
                end
            end
            
            # === EVALUATE SHIFT ===
            new_position = move_element(particle.position, i, j)
            new_profit, _, new_tour_intervals = fast_split_sparse(new_position, pso)
            
            if new_profit > particle.current_profit
                particle.position = new_position
                particle.current_profit = new_profit
                # Update particle's node_to_position mapping
                update_mapping_after_shift!(particle.node_to_position, i, j)
                # Return new tour intervals for subsequent iterations
                return true, new_tour_intervals
            end
        end
    end
    
    return false, tour_intervals
end
```

##### Integration with Swap Operator

```julia
function swap_operator!(
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots,
    tour_intervals::TourIntervals
)
    n = length(particle.position)
    pos = particle.position
    
    for i in shuffle(1:n)
        node_i = pos[i]
        is_depot_i = node_i > pso.n_pure_customers
        
        for j in shuffle((i+1):n)
            node_j = pos[j]
            is_depot_j = node_j > pso.n_pure_customers
            
            # === BOUNDARY OPTIMIZATION ===
            # Skip if both positions are outside all tours (and neither is a depot)
            if !is_depot_i && !is_depot_j
                if !is_active(tour_intervals, i) && !is_active(tour_intervals, j)
                    continue  # No-op swap, skip evaluation
                end
            end
            
            # === EXISTING BLOCKING CHECK ===
            if !is_depot_i && !is_depot_j
                if is_blocking_once_inserted(particle, i, j, pso) && 
                   is_blocking_once_inserted(particle, j, i, pso)
                    continue
                end
            end
            
            # === EVALUATE SWAP ===
            pos[i], pos[j] = pos[j], pos[i]  # Trial swap
            new_profit, _, new_tour_intervals = fast_split_sparse(pos, pso)
            
            if new_profit > particle.current_profit
                particle.current_profit = new_profit
                # Update mapping: O(1)
                particle.node_to_position[node_i] = j
                particle.node_to_position[node_j] = i
                return true, new_tour_intervals
            else
                # Revert swap
                pos[i], pos[j] = pos[j], pos[i]
            end
        end
    end
    
    return false, tour_intervals
end
```

##### Integration with Local Search

```julia
function local_search!(particle::Particle, pso::PSOiA_TOP_multiple_depots)
    # Initial split to get tour intervals
    # fast_split_sparse returns (profit, routes, tour_intervals)
    _, _, tour_intervals = fast_split_sparse(particle.position, particle, pso)
    
    improved = true
    while improved
        improved = false
        neighborhoods = shuffle([1, 2])
        
        for neighborhood in neighborhoods
            if neighborhood == 1
                improved, tour_intervals = shift_operator!(particle, pso, tour_intervals)
            else
                improved, tour_intervals = swap_operator!(particle, pso, tour_intervals)
            end
            
            if improved
                break  # Restart from first neighborhood
            end
        end
    end
end
```

#### Expected Savings

The savings depend on what fraction of positions are "active" (inside tour intervals):

| Scenario | n | Active positions | Pairs skipped | Savings |
|----------|---|-----------------|---------------|---------|
| Dense routes (60% active) | 500 | 300 | ~40,000 | ~16% |
| Sparse routes (20% active) | 500 | 100 | ~160,000 | ~64% |
| Very sparse (10% active) | 500 | 50 | ~200,000 | ~81% |

The optimization is most effective when battery constraints limit route lengths, leaving 
many customers in "dead zones" that can never be visited regardless of permutation order.

