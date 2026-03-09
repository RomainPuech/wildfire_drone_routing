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
13. [Implemented Optimizations](#13-implemented-optimizations)
    - 13.1 [Sparse Split Procedure](#131-split-procedure-optimization-via-sparse-data-structures--production)
    - 13.2 [Boundary Optimizations](#132-boundary-optimizations-for-local-search--production)
    - 13.3 [Production Features](#133-additional-production-features--production)
    - 13.4 [Incremental Tour Updates](#134-incremental-tour-updates--production)
    - 13.5 [Cost Matrix](#135-cost-matrix-optimization--production)
    - 13.6 [Allocation-Free Iteration](#136-allocation-free-iteration--production)
    - 13.7 [Lazy Dead Filter (Swap Only)](#137-lazy-dead-filter-swap-only--production)
    - 13.8 [Live Zone Filter](#138-live-zone-filter--not-recommended)
14. [Summary](#summary)
15. [Appendix: Deprecated Optimization Approaches](#appendix-deprecated-optimization-approaches)

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
    node_to_position::Vector{Int}  # Maps node → position in this particle's permutation
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

## 13. Implemented Optimizations

This section documents optimizations that have been fully implemented and are in production use. These provide significant performance improvements for sparse graphs and grid-based routing problems typical in wildfire monitoring scenarios.

### 13.1 Split Procedure Optimization via Sparse Data Structures ✅ PRODUCTION

**Status**: Fully implemented and in production use since optimization phase.

**Reference**: See `sparse_split_optimization.md` for complete theoretical foundations and proofs.

#### Motivation

The baseline Split Procedure has complexity O(n) for Phase 1 and O(n × m) for Phase 2, where n is the permutation length and m is the number of drones. However, for grid-based TOP instances:

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

#### Implementation Status

**All components are implemented and operational:**

1. ✅ **`Particle` struct** includes `node_to_position::Vector{Int}` (line 32-38 in TOP_PSO_multi_depot.jl)

2. ✅ **`PSOiA_TOP_multiple_depots`** no longer contains `node_to_position` (line 41-62)

3. ✅ **Helper functions implemented**:
   - `get_sorted_depot_positions()` (line 284)
   - `compute_saturated_tours_sparse()` (line 300)
   - `sparse_dp_phase2()` and `lookup_Γ_sparse()` (integrated into fast_split_sparse)
   - `sparse_backtracking()` (integrated into fast_split_sparse)

4. ✅ **`fast_split_sparse()` fully implemented** (line 770)
   - Returns `(profit, routes, tour_intervals)` tuple
   - Used throughout the main PSO loop

5. ✅ **Local search uses sparse operators**:
   - `local_search_sparse!()` is the primary local search function (line 2376)
   - Called with probability `pm` in the main PSO loop (line 3153)
   - Integrates with boundary optimization via `tour_intervals`

#### Performance Impact

For typical grid-based instances with:
- n = 1000 customers
- k = 10 depot nodes  
- m = 5 drones

**Complexity reduction**:
- Original: O(1000 × 5) = O(5000)
- Sparse: O(10 × 5 × log(10) + 500) ≈ O(670)
- **~7× speedup in split procedure**

The optimization is most effective when k << n, which is typical for grid-based wildfire monitoring scenarios where depots are sparse but customers are dense.

### 13.2 Boundary Optimizations for Local Search ✅ PRODUCTION

**Status**: Fully implemented and actively used in production.

**References**: 
- `boundary_optimization_paper.md` - Formal proofs and theoretical foundations
- `boundary_optimizations.md` - Extended analysis and implementation details

This optimization reduces expensive split evaluations during local search by identifying moves that provably cannot improve the solution. Based on formal theoretical analysis, we use a **conservative filtering strategy** that guarantees zero false negatives.

#### Theoretical Foundation

**Key Observations** (proven formally in `boundary_optimization_paper.md`):

1. **Swap Boundary (Observation 4.1)**: If positions `i` and `j` are non-depot nodes, then only tours that include positions `i-1` or `j-1` can change profit after swapping.

2. **Shift Boundary (Observation 4.2)**: If positions `i < j` are non-depot nodes, then only tours that include positions `i-1` or `j-1` can change profit after shifting `i → j`.

3. **Tour Membership**: A position belongs to a saturated tour if it falls within the tour's interval `[start, start + length)`. Positions belonging to no tour are called "dead".

#### Implementation Strategy: Three-Tier Filtering

The implementation uses three complementary filters that work hierarchically:

##### Tier 1: Tour Interval Filtering (Coarse Filter)

**Purpose**: Quickly reject moves affecting only dead zones.

**Data Structure**: `TourIntervals` - stores merged, non-overlapping intervals representing all saturated tours.

**Skip Conditions**:
- **Shift `i → j`**: Skip if `[min(i,j), max(i,j)]` doesn't intersect any tour interval
- **Swap `i, j`**: Skip if **both** positions are outside all tour intervals

**Complexity**: O(log k) binary search per check

**Implementation**: 
- `struct TourIntervals` (TOP_PSO_multi_depot.jl:267)
- `build_tour_intervals()` (line 623)
- `intersects_range()` (line 662), `is_active()` (line 693)

##### Tier 2: Irrelevance-Based Filtering (Fine Filter for Shift)

**Purpose**: Use blocking and deadness analysis to skip provably irrelevant moves.

**Definitions**:

1. **`irrelevant_once_removed(i)`**: True if removing node at position `i` doesn't change any tour:
   - Node must be blocking or dead, AND
   - `is_blocking_once_removed(i)` must be true (removal doesn't unblock tours)

2. **`irrelevant_once_inserted_at_position_j(i, j)`**: True if inserting node `i` before position `j` doesn't change any tour:
   - Requires `irrelevant_once_removed(i)` = true, AND
   - Node at position `j-1` must be dead (cannot extend tours)

**Skip Criterion**: Skip shift `i → j` if BOTH conditions hold.

**Candidate Reduction**: When `irrelevant_once_removed(i)` is true, restrict candidates to:
```julia
j ∈ {position(neighbor) + 1 | neighbor ∈ left_neighbors(node_i)}
```
This reduces search from O(n) to O(degree) in sparse graphs.

**Complexity**: O(1) per blocking check + O(deg) candidate iteration

**Implementation**:
- `is_blocking()`, `is_blocking_once_removed()`, `is_blocking_once_inserted()` (lines 2489-2515)
- `compute_dead_positions()` (used in shift_operator_sparse!)
- Left-neighbor restriction via `pso.left_neighbors`

##### Tier 3: Blocking-Based Filter (Fine Filter for Swap)

**Purpose**: Lightweight feasibility check for swaps.

**Key Insight**: Swaps can only improve if they change blocking status of relevant edges.

**Skip Condition**: Skip if blocking status unchanged:
```julia
is_blocking_once_inserted(i, j) && is_blocking_once_removed(i) &&
is_blocking_once_inserted(j, i) && is_blocking_once_removed(j)
```

**Complexity**: O(1) - four constant-time edge feasibility checks

**Implementation**: Integrated into `swap_operator_sparse!()` (line 2301)

#### Performance Analysis

**Complexity Comparison**:

| Operation | Without Filter | With 3-Tier Filter | Improvement |
|-----------|---------------|-------------------|-------------|
| Rejected move | O(k × m × log k + L) | O(log k + 1) | ~1000× |
| Accepted move | O(k × m × log k + L) | O(k × m × log k + L) | Same |

**Effectiveness by Route Density**:

| Scenario | Active Positions | Skip Rate | Time Saved |
|----------|-----------------|-----------|------------|
| Dense routes (60%) | 300/500 | ~40% | ~16% |
| Sparse routes (20%) | 100/500 | ~70% | ~64% |
| Very sparse (10%) | 50/500 | ~85% | ~81% |

Battery-constrained wildfire monitoring typically produces sparse routes (10-20% active positions), making these optimizations highly effective.

#### Correctness Guarantee

**Theorem** (boundary_optimization_paper.md, Observation 5.3): The combined filtering strategy is **conservative** - it never skips a move that could improve the final optimal profit.

**Proof Sketch**:
1. **Tour intervals**: Moves affecting only dead zones cannot change tour structure (Observation 3)
2. **Irrelevance**: If removal and insertion are both irrelevant, no tour changes (Theorem 5.3)
3. **Blocking**: Unchanged blocking status implies no tour feasibility changes (Definition 7.1-7.3)

Complete formal proofs available in `boundary_optimization_paper.md`.

### 13.3 Additional Production Features ✅ PRODUCTION

Beyond the core optimizations, several features enhance the algorithm's real-world applicability:

#### Flexible Cost Models

**Binary Cost Model** (default):
- Cost = 1 for adjacent cells (Chebyshev distance ≤ 1)
- Cost = `max_battery_time × 4` (infeasible) for non-adjacent
- Enables dynamic path patching between waypoints

**L∞ Distance Model** (`use_linf_cost=true`):
- Cost = actual Chebyshev distance between cells
- More realistic for flight time estimation
- Adjustable via parameter in `get_PSO_solution_multiple_depots()`

**Implementation**: TOP.jl lines 739-788

#### Blocked Cell Support (Obstacle Avoidance)

**Purpose**: Handle environments with obstacles (buildings, no-fly zones, terrain).

**Features**:
- When `mask_filename` is provided, loads blocked cell mask from file
- Switches from L∞ distance to BFS pathfinding for:
  - Accessibility computation (which cells can drones reach?)
  - Path patching between waypoints
  - Return-to-depot distance calculations
- Ensures valid paths avoiding obstacles

**Functions**:
- `get_drone_gridpoints_BFS()` (helper_functions.jl)
- `bfs_path()` for pathfinding (TOP.jl:1042)
- `patch_path_with_highest_risk!()` with blocked cell handling (TOP.jl:1336)

**Implementation**: TOP.jl lines 2386-2415, TOP_PSO_multi_depot.jl lines 2941+

#### Performance Monitoring

**Statistics Tracking**:
- `SHIFT_STATS`: Tracks candidates evaluated, skipped by filters, timing
- `SWAP_STATS`: Similar metrics for swap operator
- Enables performance profiling and optimization validation

**Time Management**:
- `max_time` parameter enforces time limits for online decision-making
- Algorithm checks elapsed time and terminates gracefully when limit approached
- Critical for real-time wildfire response scenarios

**Implementation**: TOP_PSO_multi_depot.jl throughout main loop

#### Greedy Fallback Mechanism

**Purpose**: Ensure all drones have valid routes even when PSO produces short tours.

**Features**:
- `get_greedy_fallback_solution()` extends routes greedily when needed
- Uses dynamic programming for path patching with maximum risk collection  
- Handles initial tour generation and route extension
- Accounts for already-visited cells to avoid redundant coverage

**Integration**:
- Called when PSO tour is too short (< `max_battery_time - 1`)
- Ensures movement plans are always complete and valid

**Implementation**: TOP.jl lines 1517-1799

### 13.4 Incremental Tour Updates ✅ PRODUCTION

**Status**: Fully implemented and in production use.

**Reference**: See `incremental_tours_optimization.md` for complete algorithm details, correctness proofs, and benchmarks.

**Toggle**: `ENABLE_INCREMENTAL_LOCAL_SEARCH[] = true`

#### Motivation

The standard local search evaluates every trial move by running the full split procedure (Phase 1 + Phase 2 DP). With O(n²) candidate moves per operator call, this dominates the PSO runtime. However, most moves only affect a small subset of the saturated tours — the rest are unchanged.

#### Approach

Maintain a **Tour Cache** (`TourCache` struct) that stores Phase 1 results across evaluations:

```julia
mutable struct TourCache
    sorted_depot_positions::Vector{Int}
    P_sparse::Vector{Float64}        # Profit of each saturated tour
    succ_sparse::Vector{Int}         # Successor position of each tour
    tour_lengths_sparse::Vector{Int} # Length of each tour
end
```

For each trial move, instead of a full split:

1. **Identify affected tours** via a lightweight O(k) check (only tours whose influence range contains a modified position).
2. **Recompute only affected tours** — O(L_affected) instead of O(L).
3. **Skip the DP entirely** when no cached value actually changed (~75–81% of the time).
4. **Accept/reject** and update the cache in-place.

#### Incremental Swap

A swap(i, j) can only affect tours whose influence range (d, d+ℓ] contains position i or j. For non-depot swaps, this is checked in O(k). Depot swaps (rare: k/n fraction) fall back to a full split and cache rebuild.

#### Incremental Shift

A shift(i, j) preserves the relative order of nodes within the shifted block — only the two "breakpoint" positions (removal at i, insertion at j) introduce new node adjacencies. A tour is affected if and only if one of the breakpoints falls in its influence range:

$$i \in [d, d+\ell] \quad \text{or} \quad j \in (d, d+\ell]$$

This is **much tighter** than the naïve range-overlap check and results in a 75% DP skip rate (vs 28% with range overlap).

Shifts are performed in-place with O(|i−j|) element moves and a matching revert, avoiding O(n) allocations.

#### Performance (AugustComplexFire, n=900, k=2)

| Metric | Full split | Incremental | Improvement |
|---|---|---|---|
| Per-swap eval | 2.0μs | 0.04μs | **46×** |
| Per-shift eval | 2.5μs | 0.11μs | **23×** |
| DP skip rate (swap) | 0% | 81% | — |
| DP skip rate (shift) | 0% | 75% | — |
| Full local search | 0.576s | 0.135s | **4.3×** |
| Solution quality | baseline | identical | **0% loss** |

#### Correctness

Verified with zero violations across:
- Exhaustive tests on small instances (n=34, 56, 64)
- 5,000 sampled evaluations on large instances (n=304, 508, 904)
- 20,000 per-evaluation tests on the real AugustComplexFire instance

#### Implementation

| Function | Purpose |
|---|---|
| `TourCache` | Cached Phase 1 arrays |
| `init_tour_cache` | Initialize from full split |
| `find_affected_tour_indices` | Swap: find tours covering positions i or j |
| `recompute_single_tour!` | Recompute one saturated tour in-place |
| `swap_operator_incremental!` | Full incremental swap with depot fallback |
| `compute_shifted_depot_positions` | New depot positions after shift |
| `compute_tour_at` | Compute single tour at arbitrary depot position |
| `shift_in_place!` / `revert_shift_in_place!` | O(|i−j|) in-place shift and undo |
| `shift_operator_incremental!` | Full incremental shift with breakpoint check |
| `local_search_fully_incremental!` | Combined incremental local search |

### 13.5 Cost Matrix Optimization ✅ PRODUCTION

**Status**: Fully implemented and in production use.

**Toggle**: `ENABLE_COST_MATRIX[] = true`

#### Motivation

Travel costs between nodes are stored in a `Dict{Tuple{Int,Int}, Float64}`, which is accessed millions of times per PSO run (once per edge in every split evaluation). Dictionary lookups involve hashing and collision resolution, adding overhead to the innermost loop.

#### Approach

Replace the dictionary with a dense `Matrix{Float64}` for O(1) indexed lookup:

```julia
# Added to PSOiA_TOP_multiple_depots struct:
cost_matrix::Matrix{Float64}   # (n_total+1) × (n_total+1), pre-filled with penalty value

# Inline lookup function:
@inline function lookup_cost(pso, from, to, default)
    if ENABLE_COST_MATRIX[]
        return @inbounds pso.cost_matrix[from+1, to+1]
    else
        return get(pso.costs, (from, to), default)
    end
end
```

The matrix is built once during PSO initialization from the existing `costs` dictionary. Entries not present in the dictionary are pre-filled with the infeasibility penalty (`max_battery_time × 4`). Node indices are offset by +1 to handle the artificial node 0.

#### Memory

For the AugustComplexFire instance (n ≈ 907 nodes): 907² × 8 bytes ≈ **6.6 MB**. Negligible.

#### Performance (AugustComplexFire)

| Config | Split avg time (Dict) | Split avg time (Matrix) | Speedup |
|---|---|---|---|
| Binary cost | 1.74μs | 1.61μs | **8%** |
| L∞ cost | 1.76μs | 1.48μs | **16%** |

The per-split speedup is modest but compounds over millions of calls. Overall wall-clock improvement: **1.02×** for the full PSO run.

#### Implementation

- `cost_matrix` field in `PSOiA_TOP_multiple_depots` struct
- `lookup_cost()` inline helper (replaces all ~30 `get(pso.costs, ...)` call sites)
- Matrix construction in `solve_PSO_TOP_multiple_depots()` initialization

### 13.6 Allocation-Free Iteration ✅ PRODUCTION

**Status**: Fully implemented and in production use.

**Reference**: See `alloc_free_iteration_optimization.md` for detailed correctness argument and benchmarks.

#### Motivation

The shift and swap operators iterate over candidate positions in random order. The original implementation allocated new arrays on every iteration of the outer loop:

```julia
# Shift: 2 allocations per outer iteration
inner_j = shuffle(setdiff(1:n, [i]))

# Swap: 2 allocations per outer iteration  
inner_j = shuffle(collect(i+1:n))
```

For n=900, this produces ~13 MB of GC pressure per shift call and ~3.2 MB per swap call. Over thousands of local search invocations, the cumulative allocation pressure is substantial.

#### Approach

Replace with **pre-allocated reusable buffers** and in-place shuffling:

```julia
# Shift: pre-allocate once, shuffle in-place (0 allocations per iteration)
inner_j_buf = collect(1:n)
shuffle!(inner_j_buf)
for j_idx in 1:n
    @inbounds j = inner_j_buf[j_idx]
    j == i && continue   # Equivalent to setdiff

# Swap: fill buffer, shuffle view (0 allocations per iteration)
swap_j_buf = Vector{Int}(undef, n)
for k in 1:inner_len; swap_j_buf[k] = i + k; end
shuffle!(view(swap_j_buf, 1:inner_len))
```

A `buf_dirty` flag tracks whether the buffer needs resetting after live zone filter usage.

#### Performance (AugustComplexFire)

| Config | Before | After | Speedup |
|---|---|---|---|
| OPT_ON (non-incremental) | 76.76s | 67.91s | **1.13×** |
| LINF_COST | 68.34s | 65.00s | **1.05×** |
| CM_INCR (incremental) | 60.17s | 60.51s | ~1.0× |

The optimization primarily benefits non-incremental configurations where candidate iteration is a larger fraction of runtime. For incremental configurations, the effect is negligible since the incremental evaluator already reduces split calls by 95%+.

Micro-benchmarks show **3.6–4.5× speedup** for shift candidate generation and **~100% allocation elimination**.

#### Correctness

The optimization produces statistically identical search behavior. `shuffle!(buf)` with `j == i` skip produces a uniform random permutation of {1,…,n}\{i}, identical to `shuffle(setdiff(1:n, [i]))`. The only difference is one extra RNG call per iteration (shuffling n vs n−1 elements), causing trajectory divergence with identical seeds but identical distributional properties.

#### Modified Operators

All six shift/swap operators were modified:
- `shift_operator_sparse!`, `swap_operator_sparse!` (non-incremental path)
- `shift_operator_incremental!`, `swap_operator_incremental!` (incremental path)
- `shift_operator!`, `swap_operator!` (legacy path)

### 13.7 Lazy Dead Filter (Swap Only) ✅ PRODUCTION

**Status**: Fully implemented and **enabled by default** for swap operators only. Disabled for shift operators.

**Toggle**: `ENABLE_LAZY_DEAD_FILTER[] = true` (applies to swap operators only)

**Reference**: See `live_zone_optimization.md` for the underlying dead-zone theory and correctness proofs.

#### Concept

Positions deep inside the "dead zone" (not covered by any saturated tour ± 1 boundary position) are provably irrelevant — swapping two dead positions cannot affect the split result. The lazy dead filter performs an on-the-fly O(k) check for each swap candidate pair, skipping dead–dead swaps without precomputation.

Unlike the full Live Zone Filter (which also restructures candidate iteration for shift), this filter:
- **Only applies to swaps** — shift operators are unaffected
- **No precomputation** — each check is done lazily using the tour cache's `sorted_depot_positions` and `tour_lengths_sparse`
- **Preserves shift diversification** — dead-zone shifts still shuffle nodes, maintaining PSO exploration breadth

#### Why Swap Only?

Benchmarking showed that the lazy dead filter provides a **1.20× per-call speedup for swaps** (3.47ms → 2.89ms) by eliminating ~7% additional dead–dead pairs before the expensive split evaluation. However, for shifts, the filter adds overhead (O(k) range check per candidate) without catching any additional skips beyond the irrelevance filter, resulting in a **0.94× slowdown**.

| Operator | Per-call speedup | Skip rate change | Verdict |
|---|---|---|---|
| **Swap** | **1.20×** | 83.6% → 90.7% | ✅ Beneficial |
| Shift | 0.94× | 76.4% → 76.3% | ❌ Overhead only |

#### Profit Impact

The swap-only lazy dead filter has **negligible profit impact** (−0.31% vs no filter), far better than the full Live Zone Filter (−1.81%):

| Config | Profit | Δ vs no filter |
|---|---|---|
| No filter | 0.047092 | — |
| Full LZ (precomputed) | 0.046240 | −1.81% |
| **Lazy dead (swap only)** | **0.046944** | **−0.31%** |

#### Implementation

- `is_position_safe_dead(p, sorted_depot_positions, tour_lengths_sparse)` — O(k) check if position p is outside all tour coverage ranges
- Applied in `swap_operator_sparse!` and `swap_operator_incremental!` inner loops
- Removed from `shift_operator_sparse!` and `shift_operator_incremental!`

### 13.8 Live Zone Filter ⚠️ NOT RECOMMENDED

**Status**: Implemented but **disabled by default**. Hurts solution quality in practice.

**Toggle**: `ENABLE_LIVE_ZONE_FILTER[] = false`

**Reference**: See `live_zone_optimization.md` for full analysis and correctness proofs.

#### Concept

The full Live Zone Filter restricts candidate iteration to "live" positions only (precomputed):

- **Swap**: Dead outer position → iterate only over live inner positions
- **Shift**: Dead outer position → skip all inner positions within the same dead block

#### Why Not Recommended

Despite being provably correct and delivering significant candidate reduction (70.8% swap, 42.1% shift), the filter **degrades final solution quality** by −1.81% due to loss of diversification from dead-zone moves. The swap-only lazy dead filter (§13.7) captures the swap benefit without the shift drawback.

**Recommendation**: Use `ENABLE_LAZY_DEAD_FILTER[] = true` (swap only) instead.

---

## Summary

This PSO-inspired algorithm for TOP combines several key innovations to efficiently solve wildfire drone routing problems:

### Core Algorithm Components

1. **Giant Tour Representation**: Unified encoding for multi-drone routes
2. **Efficient Split Procedure**: O(k × m × log k) sparse optimal route extraction via dynamic programming
3. **Hybrid Initialization**: Random + IDCH + Greedy for solution diversity
4. **Crossover-based Update**: Permutation-friendly position update operator
5. **Multi-neighborhood Local Search**: Shift and swap operators with sophisticated filtering
6. **Diversity Management**: Similarity-based local best updates
7. **Adaptive Iteration**: Counter reset on improvement for intensification

### Production-Ready Optimizations

**Sparse Split (§13.1)**:
- Depot-only computation reduces complexity from O(n × m) to O(k × m × log k)
- Typical speedup: 7× for grid-based instances
- Based on observation that tours can only start at depot positions

**Boundary Optimization (§13.2)**:
- Three-tier conservative filtering: tour intervals → irrelevance → blocking
- Skips 40-85% of no-op moves depending on route density
- Proven to never skip improving moves (zero false negatives)
- Exploits graph sparsity via left-neighbor restriction (O(n) → O(degree))

**Incremental Tour Updates (§13.4)**:
- Cache Phase 1 results and recompute only affected tours per move
- Skip DP entirely when no tour changes (75-81% of the time)
- 4.3× local search speedup with zero quality loss

**Cost Matrix (§13.5)**:
- Dense matrix replaces dictionary for travel cost lookups
- 8-16% per-split speedup, ~6.6 MB memory for n=900

**Allocation-Free Iteration (§13.6)**:
- Pre-allocated buffers eliminate GC pressure in candidate iteration
- 1.13× wall-clock speedup for non-incremental path
- ~100% allocation elimination in shift/swap inner loops

**Additional Features (§13.3)**:
- Flexible cost models (binary pathing vs L∞ distance)
- BFS-based obstacle avoidance for blocked cells
- Time-bounded execution for real-time response
- Greedy fallback for robustness
- Performance monitoring and statistics

**Lazy Dead Filter — Swap Only (§13.7)**:
- On-the-fly O(k) dead–dead check eliminates ~7% additional swap candidates
- 1.20× per-call swap speedup with negligible profit impact (−0.31%)
- Enabled by default (`ENABLE_LAZY_DEAD_FILTER[] = true`)

**Not Recommended — Full Live Zone Filter (§13.8)**:
- Reduces candidate counts by 42-71% but hurts solution quality (−1.81%)
- Disabled by default (`ENABLE_LIVE_ZONE_FILTER[] = false`)

### Performance Profile

| Component | Speedup | Applicability |
|-----------|---------|---------------|
| Sparse Split | 7× | Always (depot-sparse graphs) |
| Boundary Filters | 5-15× | Battery-constrained scenarios |
| Left-neighbor Restriction | 2-5× | Sparse client graphs |
| Incremental Tour Updates | 4.3× | Local search phase |
| Allocation-Free Iteration | 1.13× | Non-incremental path |
| Cost Matrix | 1.02× | Always |
| Lazy Dead Filter (swap) | 1.20× per swap call | Swap operator |
| **Combined** | **significant** | Typical wildfire scenarios |

The algorithm achieves these performance gains while maintaining solution quality through mathematically proven conservative filtering strategies. See individual optimization docs for complete details:
- `sparse_split_optimization.md` — Sparse split theory and proofs
- `boundary_optimization_paper.md` / `boundary_optimizations.md` — Boundary filter theory
- `incremental_tours_optimization.md` — Incremental tour update algorithm
- `alloc_free_iteration_optimization.md` — Allocation-free iteration details
- `live_zone_optimization.md` — Live zone filter analysis (swap-only lazy version recommended; full version not recommended)

---

## Appendix: Deprecated Optimization Approaches

This appendix documents optimization strategies that were considered during development but ultimately **not implemented** or **deprecated** in favor of the approaches described in Section 13.

### A.1 Perfect Profit Filter (NOT IMPLEMENTED)

**Status**: ❌ Considered but rejected due to fundamental correctness issues.

**Original Idea**: Cache all Phase 1 saturated tour information and use profit-only filtering to skip local search moves.

#### Proposed Approach

The idea was to maintain complete tour information for each particle:

```julia
struct TourInfo
    start_pos::Int
    depot_node::Int
    length::Int
    profit::Float64
    node_sequence::Vector{Int}
end

struct ParticleTourCache
    tours::Vector{TourInfo}
    position_to_tour_ids::Vector{Vector{Int}}
end
```

**Filtering Strategy**: Before evaluating a move (swap or shift), recompute only the profits of affected tours. If no affected tour increases in profit, reject the move without running the full split procedure.

#### Why It Was Rejected

**Critical Flaw**: The filter is **not conservative** - it can produce false negatives.

**Counterexample**:

```
Original permutation: [D1, c1, c2, D2, c3, c4]
Battery = 3

Phase 1 tours:
- Tour from D1: [D1, c1, c2] - profit = 10
- Tour from D2: [D2, c3, c4] - profit = 10

Phase 3 selects both tours: total profit = 20
```

After swap(2, 5) (swapping c1 and c3):
```
New permutation: [D1, c3, c2, D2, c1, c4]

Phase 1 tours:
- Tour from D1: [D1, c3, c2] - profit = 10 (unchanged)
- Tour from D2: [D2, c1, c4] - profit = 10 (unchanged)

BUT: Phase 3 now selects ONLY first tour due to overlap!
Final profit = 10 (worse!)
```

**The Problem**: Individual tour profits stayed the same, but **tour overlap structure changed**, causing the DP to select a different (worse) combination. The profit-only filter would have approved this move, incorrectly predicting improvement.

**Worse**: The opposite can also occur - a move might be rejected because no tour improves individually, yet the DP finds a better non-overlapping combination, producing higher total profit.

**Conclusion**: Checking only individual tour profits is **insufficient**. Changes in tour lengths and overlaps affect DP decisions independently of individual tour profits.

### A.2 Tour Cache with Perfect Filter (NOT IMPLEMENTED)

**Status**: ❌ Theoretically sound but impractical due to overhead.

**Refined Idea**: Use full per-particle tour caching with complete DP-aware filtering.

#### Proposed Approach

Maintain not just tour profits but also:
- Complete DP table `Γ[i, j]` for each particle
- Selected tour IDs from Phase 3 backtracking
- Position-to-tour membership mappings

**Filtering Strategy**: Before a move, identify affected tours, recompute their profits, update the DP incrementally, and check if the final DP value improves.

#### Why It Was Rejected

**Memory Overhead**:
- Per particle: ~8-15 KB for tour cache
- For swarm_size=50: ~400-750 KB total
- Acceptable, but...

**Computational Overhead**:
- Incremental DP updates are complex to implement correctly
- Each rejected move still requires O(k) tour profit recomputation
- With boundary optimizations (Section 13.2), we can skip moves with O(log k + 1) checks
- The perfect filter doesn't provide enough additional benefit

**Maintenance Burden**:
- Cache invalidation logic is error-prone
- Every local search operator must update the cache
- Debugging cache inconsistencies is difficult

**Empirical Testing**: Preliminary experiments showed the three-tier boundary optimization (Section 13.2) achieved 80-85% rejection rate without caching, making the perfect filter's additional 5-10% improvement not worth the complexity.

**Decision**: Implement the simpler, faster, provably correct three-tier boundary optimization instead.

### A.3 Dense Split with Incremental Updates (NOT IMPLEMENTED)

**Status**: ❌ Superseded by sparse split (Section 13.1).

**Idea**: Keep the dense O(n × m) split but update it incrementally after local search moves.

#### Why It Was Rejected

**Complexity Issues**:
- Swap moves change 2 positions → up to O(n) tours affected → O(n × m) DP update
- Shift moves change O(n) positions (due to left-shifting) → O(n × m) DP update
- Not actually incremental enough to beat full recomputation

**Maintenance Burden**:
- Incremental DP update logic is complex and bug-prone
- Easier to just run sparse split (O(k × m × log k)) after every move

**Empirical Testing**: Sparse split (Section 13.1) turned out to be faster than any incremental dense approach, making this optimization obsolete.

### A.4 Summary of Deprecated Approaches

| Approach | Status | Reason for Rejection |
|----------|--------|---------------------|
| Perfect Profit Filter (§A.1) | ❌ Not Implemented | Fundamentally incorrect (false negatives) |
| Tour Cache with DP (§A.2) | ❌ Not Implemented | Too complex, insufficient benefit over boundary optimization |
| Incremental Dense Split (§A.3) | ❌ Not Implemented | Sparse split is faster and simpler |

**Key Lesson**: Correctness and simplicity matter more than theoretical sophistication. The three-tier boundary optimization (Section 13.2) achieves 80-85% skip rates with provable correctness and minimal overhead, making more complex caching schemes unnecessary.

---

**Document Version**: 3.0  
**Last Updated**: February 2026  
**Status**: Production - Actively Maintained