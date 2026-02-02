using Random
using DataStructures
using Plots

# Include the existing TOP.jl for sample generation and plotting
# include("TOP.jl")

mutable struct Particle
    position::Vector{Int}          # Current permutation (giant tour)
    local_best::Vector{Int}        # Best known position for this particle
    local_best_profit::Float64     # Profit of local best
    current_profit::Float64        # Current profit
    node_to_position::Vector{Int}  # Maps node → position in this particle's permutation
end


mutable struct PSOiA_TOP_multiple_depots
    swarm::Vector{Particle}
    global_best::Vector{Int}
    global_best_profit::Float64
    swarm_size::Int
    max_iterations::Int
    w::Float64                     # Inertia weight
    c1::Float64                    # Cognitive factor
    c2::Float64                    # Social factor
    ph::Float64                    # Probability of random move
    pm::Float64                    # Probability of local search
    n_drones::Int
    n_pure_customers::Int
    max_battery_time::Int
    customers::Vector{Tuple{Int,Int}}  # Customer coordinates
    profits::Vector{Float64}       # Customer profits
    costs::Dict{Tuple{Int,Int}, Float64}  # Travel costs
    left_neighbors::Dict{Int, Vector{Int}}  # Left neighbors
    accessible_customers::Vector{Int}  # Indices of accessible customers
    depot_coord::Vector{Tuple{Int,Int}}    # Depot coordinates
    closest_depot_distance::Vector{Float64}  # Pre-computed min return distance to closest depot (Chebyshev)
end

"""
Fast split procedure for GRID (grid-based) multiple depots
"""
function fast_split_with_routes_multiple_depots(permutation::Vector{Int}, pso_multiple_depots::PSOiA_TOP_multiple_depots)
    n = length(permutation)
    m = pso_multiple_depots.n_drones
    L = pso_multiple_depots.max_battery_time
    
    if n == 0
        return 0.0, Vector{Vector{Int}}()
    end
    
    # Debug: log when fast_split is called with large permutations
    # if n > 50
    #     println("[FAST_SPLIT-DEBUG] Called with large permutation: n=$n, first 10=$(permutation[1:min(10, n)]), costs_dict_size=$(length(pso_multiple_depots.costs))")
    # end

    # start_time_phase_1 = time()
    
    # Calculate saturated tours P[i] and first successor succ[i]
    P = zeros(n)  # Profit of saturated tour starting at position i
    succ = zeros(Int, n)  # First successor of saturated tour starting at position i
    tour_lengths = zeros(Int, n)  # Length of each saturated tour
    
    # Debug: track missing costs
    missing_costs_count = 0
    missing_costs_pairs = Tuple{Int,Int}[]
    
    for i in 1:n # HERE! instead of 1:n, you could i += (j-i)... (but then you say can never go from a node to a depot (which is fine?)) #TODO
        # if i is a not a depot, skip
        #println("n_pure_customers: $(pso_multiple_depots.n_pure_customers)")
        #error("STOP")
        if permutation[i] <= pso_multiple_depots.n_pure_customers
            succ[i] = i + 1
            continue
        end 
        #else
        current_cost = 0.0
        current_profit = 0.0
        travel_cost = 0.0
        prev_customer = permutation[i]
        j = i + 1
        # all the rest
    
        # Build maximal feasible tour starting from position i
        while j <= n
            customer_idx = permutation[j]
            cost_key = (prev_customer, customer_idx)
            if !haskey(pso_multiple_depots.costs, cost_key)
                missing_costs_count += 1
                if missing_costs_count <= 5  # Only store first 5 for logging
                    push!(missing_costs_pairs, cost_key)
                end
            end
            travel_cost = get(pso_multiple_depots.costs, cost_key, L*4)
            # Feasibility: ensure we can still return to the closest depot using precomputed distance
            return_distance = pso_multiple_depots.closest_depot_distance[customer_idx]
            current_cost += travel_cost
            if current_cost + return_distance > L
                break
            end
            
            current_profit += pso_multiple_depots.profits[customer_idx]
            prev_customer = customer_idx
            j += 1
        end
        
        P[i] = current_profit
        tour_lengths[i] = j - i
        # Apply Equation 3: succ[i] = i + l_i^max + 1 if i + l_i^max + 1 ≤ n, else 0
        # Here, l_i^max = j - i, so i + l_i^max + 1 = i + (j - i) + 1 = j + 1
        if j <= n
            succ[i] = j
        else
            succ[i] = 0  # According to Equation 3
        end
        
    end
    # time_phase_1 = time() - start_time_phase_1
    # println("time to run phase 1: $time_phase_1")

    # start_time_phase_2 = time()
    # Dynamic programming table Γ[i,j] = max profit using j drones from position i onwards
    # Using (n+1) × (m+1) to handle boundary conditions as per Equation 4
    Γ = zeros(n + 1, m + 1)
    
    # Fill DP table in reverse order (as shown in Figure 1c)
    for i in n:-1:1
        for j in 0:m
            if j == 0
                # No drones left, no profit possible
                Γ[i, j + 1] = 0.0
            else
                # Option 1: Don't use tour starting at i
                Γ[i, j + 1] = Γ[i + 1, j + 1]
                
                # Option 2: Use saturated tour starting at i
                # According to Equation 4: max{Γ(succ[i], j-1) + P[i], Γ(i+1, j)}
                if succ[i] == 0
                    # When succ[i] = 0, Γ(succ[i], j-1) = 0 according to Equation 4
                    profit_with_tour = P[i] + 0.0
                else
                    # Normal case: access Γ[succ[i], j] which corresponds to Γ(succ[i], j-1) in paper notation
                    profit_with_tour = P[i] + Γ[succ[i], j]
                end
                
                Γ[i, j + 1] = max(Γ[i, j + 1], profit_with_tour)
            end
        end
    end
    # time_phase_2 = time() - start_time_phase_2
    # println("time to run phase 2: $time_phase_2")

    # start_time_phase_3 = time()
    # Backtrack to find the optimal routes (as described in the paper)
    # routes actually never used as output!! but we keep it for now #TODO
    routes = Vector{Vector{Int}}()
    i = 1
    j = m
    
    while i <= n && j > 0 && length(routes) < m
        # Check which option was chosen in the DP
        option1 = Γ[i + 1, j + 1]  # Don't use tour starting at i
        
        option2 = 0.0
        if succ[i] == 0
            option2 = P[i] + 0.0
        else
            option2 = P[i] + Γ[succ[i], j]
        end
        
        if abs(option2 - Γ[i, j + 1]) < 1e-10  # Use saturated tour starting at i
            # Extract the saturated tour starting at position i
            tour_end = i + tour_lengths[i] - 1
            route = permutation[i:tour_end]
            push!(routes, route)
            
            i = succ[i] > 0 ? succ[i] : n + 1
            j -= 1
        else  # Skip this position
            i += 1
        end
    end
    # time_phase_3 = time() - start_time_phase_3
    # println("time to run phase 3: $time_phase_3")
    # total_time = time_phase_1 + time_phase_2 + time_phase_3
    # if total_time > 0.01
    #     println("total time: $(time_phase_1 + time_phase_2 + time_phase_3)")
    #     # relative time
    #     println("relative time: $(time_phase_1 / (time_phase_1 + time_phase_2 + time_phase_3))")
    #     println("relative time: $(time_phase_2 / (time_phase_1 + time_phase_2 + time_phase_3))")
    #     println("relative time: $(time_phase_3 / (time_phase_1 + time_phase_2 + time_phase_3))")
    #     println("n: $n")
    # end
    
    # Log missing costs if any
    if missing_costs_count > 0
        println("[FAST_SPLIT-DEBUG] WARNING: $missing_costs_count missing cost entries (using default L*4=$(L*4))")
        if !isempty(missing_costs_pairs)
            println("[FAST_SPLIT-DEBUG] First few missing pairs: $(missing_costs_pairs[1:min(5, length(missing_costs_pairs))])")
        end
    end

    return Γ[1, m + 1], routes
end



# Use sparse split for all profit-only evaluations (Section 13.1)
function fast_split_multiple_depots(permutation::Vector{Int}, pso_multiple_depots::PSOiA_TOP_multiple_depots)
    profit, _, _ = fast_split_sparse(permutation, pso_multiple_depots)
    return profit
end

"""
Compute node-to-position mapping for a given permutation
Returns a Vector where node_to_pos[node] = position of that node in the permutation.
The vector is sized to hold all nodes (1 to max node ID).
"""
function compute_node_to_position(permutation::Vector{Int})
    if isempty(permutation)
        return Int[]
    end
    max_node = maximum(permutation)
    node_to_pos = zeros(Int, max_node)
    for (pos, node) in enumerate(permutation)
        node_to_pos[node] = pos
    end
    return node_to_pos
end

# ============================================================================
# SPARSE SPLIT PROCEDURE (Optimized for grid-based instances with sparse depots)
# ============================================================================

"""
TourIntervals: Stores the union of saturated tour intervals for boundary optimization.
Used to skip no-op shift/swap moves that cannot affect the split profit.
"""
struct TourIntervals
    intervals::Vector{Tuple{Int,Int}}  # Sorted, non-overlapping (start, end) pairs
    max_end::Int                        # Maximum endpoint for quick rejection
end

"""
Build an empty TourIntervals
"""
function empty_tour_intervals()
    return TourIntervals(Tuple{Int,Int}[], 0)
end

"""
Get sorted depot positions from a node_to_position mapping.
Depot nodes are those with index > n_pure_customers.
Returns positions sorted in ascending order.
"""
function get_sorted_depot_positions(node_to_position::Vector{Int}, n_pure_customers::Int)
    depot_positions = Int[]
    for node in (n_pure_customers + 1):length(node_to_position)
        pos = node_to_position[node]
        if pos > 0  # Only include if the node is in the permutation
            push!(depot_positions, pos)
        end
    end
    sort!(depot_positions)
    return depot_positions
end

"""
Phase 1 (Sparse): Compute saturated tours only at depot positions.
Returns sparse arrays indexed by depot index (1 to k), not position.
"""
function compute_saturated_tours_sparse(
    permutation::Vector{Int},
    node_to_position::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    n_pure_customers = pso.n_pure_customers
    L = pso.max_battery_time
    
    # Get sorted depot positions
    sorted_depot_positions = get_sorted_depot_positions(node_to_position, n_pure_customers)
    k = length(sorted_depot_positions)
    
    if k == 0
        return Float64[], Int[], Int[], Int[]
    end
    
    # Sparse arrays indexed by depot index (1 to k)
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
            
            # Accumulate cost first
            current_cost += travel_cost
            
            # Feasibility check
            if current_cost + return_distance > L
                break
            end
            
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

"""
Lookup function for sparse DP: find Γ value at first depot at or after 'position'.
Uses binary search for O(log k) lookup.
"""
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

"""
Phase 2 (Sparse): Dynamic programming on depot positions only.
Γ_sparse[idx, j+1] = max profit using j drones from depot index idx onwards.
"""
function sparse_dp_phase2(
    P_sparse::Vector{Float64},
    succ_sparse::Vector{Int},
    sorted_depot_positions::Vector{Int},
    m::Int,  # Number of drones
    n::Int   # Permutation length
)
    k = length(sorted_depot_positions)
    
    # Γ_sparse dimensions: (k+1) × (m+1) to handle boundary conditions
    Γ_sparse = zeros(Float64, k + 1, m + 1)
    
    # Boundary condition: Γ_sparse[k+1, :] = 0 (no more depots) - already initialized
    
    # Fill DP table in reverse order of depot indices
    for idx in k:-1:1
        for j in 1:m
            # Option 1: Skip this depot (go to next depot)
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

"""
Find next depot index at or after a given position (binary search).
Returns 0 if no depot exists at or after the position.
"""
function find_next_depot_index(position::Int, sorted_depot_positions::Vector{Int})
    idx = searchsortedfirst(sorted_depot_positions, position)
    return idx <= length(sorted_depot_positions) ? idx : 0
end

"""
Phase 3 (Sparse): Backtrack through sparse DP table to extract routes.
"""
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

"""
Build TourIntervals from Phase 1 results.
Includes depot positions as zero-length intervals to prevent skipping depot-crossing moves.
"""
function build_tour_intervals(
    sorted_depot_positions::Vector{Int},
    tour_lengths_sparse::Vector{Int}
)
    intervals = Tuple{Int,Int}[]
    
    for (idx, depot_pos) in enumerate(sorted_depot_positions)
        # Include depot position as zero-length interval
        push!(intervals, (depot_pos, depot_pos))
        # Include tour interval (positions after depot)
        tour_start = depot_pos + 1
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

"""
Check if a range [range_start, range_end] intersects any tour interval.
Used for shift operations to detect no-op moves.
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

"""
Sparse split procedure: Version 1 - uses node_to_position from particle.
Returns (profit, routes, tour_intervals).
"""
function fast_split_sparse(
    permutation::Vector{Int},
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots
)
    return fast_split_sparse_with_mapping(permutation, particle.node_to_position, pso)
end

"""
Sparse split procedure: Version 2 - computes mapping on-the-fly.
Returns (profit, routes, tour_intervals).
"""
function fast_split_sparse(
    permutation::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    node_to_position = compute_node_to_position(permutation)
    return fast_split_sparse_with_mapping(permutation, node_to_position, pso)
end

"""
Core sparse split implementation with explicit node_to_position mapping.
Returns (profit, routes, tour_intervals).
"""
function fast_split_sparse_with_mapping(
    permutation::Vector{Int},
    node_to_position::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    m = pso.n_drones
    
    empty_intervals = empty_tour_intervals()
    
    if n == 0
        return 0.0, Vector{Vector{Int}}(), empty_intervals
    end
    
    # Phase 1: Compute saturated tours (sparse)
    P_sparse, succ_sparse, tour_lengths_sparse, sorted_depot_positions = 
        compute_saturated_tours_sparse(permutation, node_to_position, pso)
    
    k = length(sorted_depot_positions)
    if k == 0
        return 0.0, Vector{Vector{Int}}(), empty_intervals
    end
    
    # Phase 2: Dynamic programming (sparse)
    Γ_sparse = sparse_dp_phase2(P_sparse, succ_sparse, sorted_depot_positions, m, n)
    
    # Phase 3: Backtracking (sparse)
    routes = sparse_backtracking(
        P_sparse, succ_sparse, tour_lengths_sparse,
        sorted_depot_positions, Γ_sparse, permutation, m, n
    )
    
    # Build tour intervals for boundary optimization
    tour_intervals = build_tour_intervals(sorted_depot_positions, tour_lengths_sparse)
    
    optimal_profit = lookup_Γ_sparse(1, m, sorted_depot_positions, Γ_sparse)
    
    return optimal_profit, routes, tour_intervals
end

# ============================================================================
# END SPARSE SPLIT PROCEDURE
# ============================================================================

"""
Initialize particle swarm
"""
function initialize_swarm(pso::PSOiA_TOP_multiple_depots, use_greedy_init::Bool = true; skip_idch::Bool = false)
    total_start_time = time()
    
    # === Phase 1: Random Initialization ===
    println("\n[SWARM INIT] Phase 1: Random initialization ($(pso.swarm_size) particles)")
    random_start = time()
    for i in 1:pso.swarm_size
        # Create random permutation of accessible customers
        position = shuffle(pso.accessible_customers)
        profit = fast_split_multiple_depots(position, pso)
        
        # Compute node-to-position mapping for this particle
        node_to_pos = compute_node_to_position(position)
        
        particle = Particle(
            copy(position),
            copy(position),
            profit,
            profit,
            node_to_pos
        )
        
        push!(pso.swarm, particle)
        
        # Update global best
        if profit > pso.global_best_profit
            pso.global_best = copy(position)
            pso.global_best_profit = profit
            # println("Initial swarm: New best = $(round(pso.global_best_profit, digits=3))")
        end
    end
    random_time = time() - random_start
    println("[SWARM INIT] Phase 1 completed in $(round(random_time, digits=3))s (best profit: $(round(pso.global_best_profit, digits=3)))")
    
    # === Phase 2: IDCH Heuristic Initialization ===
    n_idch = min(5, pso.swarm_size ÷ 2)
    if n_idch > 0 && !skip_idch
        println("\n[SWARM INIT] Phase 2: IDCH heuristic initialization ($n_idch particles)")
        idch_start = time()
        for i in 1:n_idch
            idch_particle_start = time()
            position = idch_heuristic(pso, false)  # Fast version
            profit, routes, _ = fast_split_sparse(position, pso)
            
            # Calculate profit breakdown per route
            route_profits = Float64[]
            route_lengths = Int[]
            route_customer_counts = Int[]
            total_customers = 0
            for (drone_idx, route) in enumerate(routes)
                route_profit = 0.0
                route_customers = 0
                for customer_idx in route
                    # Skip depot nodes (artificial node 0 and depot nodes > n_pure_customers)
                    if customer_idx > 0 && customer_idx <= pso.n_pure_customers
                        route_profit += pso.profits[customer_idx]
                        route_customers += 1
                    end
                end
                push!(route_profits, route_profit)
                push!(route_lengths, length(route))
                push!(route_customer_counts, route_customers)
                total_customers += route_customers
            end
            
            pso.swarm[i].position = copy(position)
            pso.swarm[i].position = copy(position)
            pso.swarm[i].local_best = copy(position)
            pso.swarm[i].local_best_profit = profit
            pso.swarm[i].current_profit = profit
            pso.swarm[i].node_to_position = compute_node_to_position(position)
            
            # Print detailed information
            println("  [IDCH] Particle $i:")
            println("    - Position length: $(length(position)), first 5 customers: $(position[1:min(5, length(position))])")
            println("    - Total profit: $(round(profit, digits=6))")
            println("    - Number of routes: $(length(routes))")
            println("    - Total customers visited: $total_customers")
            for (drone_idx, route_profit) in enumerate(route_profits)
                println("    - Route $drone_idx: profit=$(round(route_profit, digits=6)), length=$(route_lengths[drone_idx]), customers=$(route_customer_counts[drone_idx])")
                if length(routes[drone_idx]) <= 20
                    println("      Path: $(routes[drone_idx])")
                else
                    println("      Path (first 10): $(routes[drone_idx][1:10]) ... (last 5): $(routes[drone_idx][(end-4):end])")
                end
            end
            println("    - Time: $(round(time() - idch_particle_start, digits=3))s")
            
            if profit > pso.global_best_profit
                pso.global_best = copy(position)
                pso.global_best_profit = profit
                println("    - Status: NEW BEST")
            else
                println("    - Status: Profit = $(round(profit, digits=6))")
            end
        end
        idch_time = time() - idch_start
        println("[SWARM INIT] Phase 2 completed in $(round(idch_time, digits=3))s")
    end
    
    # === Phase 3: Greedy Fallback Initialization ===
    if pso.swarm_size >= 2 && use_greedy_init
        # Store best profit before greedy phase for comparison (from IDCH or random if IDCH skipped)
        best_profit_before_greedy = pso.global_best_profit
        
        println("\n[SWARM INIT] Phase 3: Greedy fallback initialization (2 particles)")
        greedy_start = time()
        positions, expected_profits = initialize_with_greedy_fallback_two(pso)
        greedy_compute_time = time() - greedy_start
        println("[SWARM INIT] Greedy solutions computed in $(round(greedy_compute_time, digits=3))s")
        
        # Replace the last two particles with the greedy solutions
        # IMPORTANT: Use expected_profits from greedy route building, NOT fast_split!
        # This is because greedy routes use actual path distances, while fast_split uses L-infinity costs.
        eval_start = time()
        best_greedy_profit = -Inf
        best_greedy_idx = 0
        
        for (idx, position) in enumerate(positions)
            particle_index = pso.swarm_size - (2 - idx)  # Last two particles
            
            # Use the expected profit from greedy route building directly
            profit = expected_profits[idx]
            
            # Debug: Check position before evaluation
            println("  [GREEDY-DEBUG] Particle $idx: position length = $(length(position)), first 5 = $(position[1:min(5, length(position))])")
            println("  [GREEDY-DEBUG] Particle $idx: expected profit from greedy = $(round(profit, digits=6))")
            
            # Also compute fast_split profit for comparison (but don't use it)
            fast_split_profit, routes, _ = fast_split_sparse(position, pso)
            println("  [GREEDY-DEBUG] Particle $idx: fast_split profit = $(round(fast_split_profit, digits=6)) (NOT USED)")
            if fast_split_profit < profit * 0.5
                println("  [GREEDY-DEBUG] Particle $idx: NOTE: fast_split gives much lower profit because it uses L-infinity costs,")
                println("                               but greedy routes were built with actual path distances through intermediate cells.")
            end
            
            pso.swarm[particle_index].position = copy(position)
            pso.swarm[particle_index].local_best = copy(position)
            pso.swarm[particle_index].local_best_profit = profit
            pso.swarm[particle_index].current_profit = profit
            pso.swarm[particle_index].node_to_position = compute_node_to_position(position)
            
            # Track best greedy solution
            if profit > best_greedy_profit
                best_greedy_profit = profit
                best_greedy_idx = idx
            end
            
            if profit > pso.global_best_profit
                pso.global_best = copy(position)
                pso.global_best_profit = profit
                println("  [GREEDY] Particle $idx: New best = $(round(pso.global_best_profit, digits=6))")
            else
                println("  [GREEDY] Particle $idx: Profit = $(round(profit, digits=6))")
            end
        end
        
        # Print summary comparison
        println("\n  [GREEDY] === GREEDY SOLUTION SUMMARY ($(pso.n_drones) drones) ===")
        println("  [GREEDY] Best greedy solution (particle $best_greedy_idx):")
        println("    - Total profit: $(round(best_greedy_profit, digits=6))")
        println("    - (Note: profit calculated from greedy route building, not fast_split)")
        println("  [GREEDY] Comparison with previous best:")
        println("    - Previous best profit: $(round(best_profit_before_greedy, digits=6))")
        println("    - Greedy best profit: $(round(best_greedy_profit, digits=6))")
        if best_greedy_profit > best_profit_before_greedy && best_profit_before_greedy > 0
            improvement = best_greedy_profit - best_profit_before_greedy
            improvement_pct = 100.0 * improvement / best_profit_before_greedy
            println("    - Greedy is BETTER by $(round(improvement, digits=6)) ($(round(improvement_pct, digits=1))% improvement)")
        elseif best_greedy_profit > best_profit_before_greedy
            println("    - Greedy is BETTER (previous was 0)")
        elseif best_greedy_profit < best_profit_before_greedy
            gap = best_profit_before_greedy - best_greedy_profit
            gap_pct = 100.0 * gap / best_profit_before_greedy
            println("    - Previous best is BETTER by $(round(gap, digits=6)) ($(round(gap_pct, digits=1))% gap)")
        else
            println("    - Greedy and previous best are EQUAL")
        end
        println("  [GREEDY] ===========================================\n")
        
        eval_time = time() - eval_start
        println("[SWARM INIT] Phase 3 evaluation completed in $(round(eval_time, digits=3))s")
        println("[SWARM INIT] Phase 3 total time: $(round(greedy_compute_time + eval_time, digits=3))s")
    elseif pso.swarm_size >= 1 && use_greedy_init
        println("\n[SWARM INIT] Phase 3: Greedy fallback initialization (1 particle)")
        greedy_start = time()
        position = initialize_with_greedy_fallback(pso)
        greedy_compute_time = time() - greedy_start
        println("[SWARM INIT] Greedy solution computed in $(round(greedy_compute_time, digits=3))s")
        
        eval_start = time()
        profit = fast_split_multiple_depots(position, pso)
        eval_time = time() - eval_start
        
        # Replace the last particle with the greedy solution
        pso.swarm[end].position = copy(position)
        pso.swarm[end].local_best = copy(position)
        pso.swarm[end].local_best_profit = profit
        pso.swarm[end].current_profit = profit
        pso.swarm[end].node_to_position = compute_node_to_position(position)
        
        if profit > pso.global_best_profit
            pso.global_best = copy(position)
            pso.global_best_profit = profit
            println("  [GREEDY] New best = $(round(pso.global_best_profit, digits=3)) (eval time: $(round(eval_time, digits=3))s)")
        else
            println("  [GREEDY] Profit = $(round(profit, digits=3)) (eval time: $(round(eval_time, digits=3))s)")
        end
        println("[SWARM INIT] Phase 3 total time: $(round(greedy_compute_time + eval_time, digits=3))s")
    end
    
    total_time = time() - total_start_time
    println("\n[SWARM INIT] Total initialization time: $(round(total_time, digits=3))s")
    println("[SWARM INIT] Final best profit: $(round(pso.global_best_profit, digits=3))\n")
end

"""
Initialize two particles using greedy fallback solutions where the second takes the first into account
"""
function initialize_with_greedy_fallback_two(pso::PSOiA_TOP_multiple_depots)
    try
        total_start = time()
        
        # === Setup: Create Risk Map ===
        setup_start = time()
        println("  [GREEDY] Step 1: Creating risk map from customer profits")
        # Create a synthetic risk map where profits correspond to risk values
        # The greedy fallback expects a 3D array with dimensions (time, x, y)
        # We'll create a simple grid based on customer coordinates
        
        # Find grid bounds from customer coordinates
        max_x = maximum(coord[1] for coord in pso.customers if coord[1] > 0)
        max_y = maximum(coord[2] for coord in pso.customers if coord[2] > 0)
        
        # Create risk map
        risk_pertime = zeros(1, max_x, max_y)
        
        # Set risk values based on customer profits
        test_point = (36, 46)  # The point the user added with high profit
        test_point_found = false
        test_point_profit = 0.0
        test_point_customer_idx = 0
        for i in 1:pso.n_pure_customers
            if i <= length(pso.customers) && i <= length(pso.profits)
                coord = pso.customers[i]
                if coord[1] > 0 && coord[2] > 0 && coord[1] <= max_x && coord[2] <= max_y
                    risk_pertime[1, coord[1], coord[2]] = pso.profits[i]
                    if coord == test_point
                        test_point_found = true
                        test_point_profit = pso.profits[i]
                        test_point_customer_idx = i
                        println("  [GREEDY-DEBUG] Test point $test_point found as customer $i with profit: $(test_point_profit)")
                    end
                end
            end
        end
        if !test_point_found
            println("  [GREEDY-DEBUG] WARNING: Test point $test_point is NOT in the customer list!")
            println("  [GREEDY-DEBUG] Checking if it would be accessible if it were a customer...")
            # Check if it would be accessible if it were a customer
            if test_point[1] > 0 && test_point[1] <= max_x && test_point[2] > 0 && test_point[2] <= max_y
                println("  [GREEDY-DEBUG] Test point $test_point is within grid bounds ($max_x, $max_y)")
                # Check distance to charging stations (we'll get ChargingStation below)
            end
        end
        
        # Extract GridpointsDronesDetecting (pure customers only)
        GridpointsDronesDetecting = pso.customers[1:pso.n_pure_customers]
        
        # Extract ChargingStation from depot coordinates
        ChargingStation = pso.depot_coord
        
        # Check accessibility if point was found
        if test_point_found
            println("  [GREEDY-DEBUG] Checking accessibility of test point $test_point...")
            for (idx, cs) in enumerate(ChargingStation)
                dist = max(abs(test_point[1] - cs[1]), abs(test_point[2] - cs[2]))
                return_dist = dist  # Same distance to return
                total_dist = dist + return_dist
                accessible = total_dist <= pso.max_battery_time
                println("  [GREEDY-DEBUG] Distance from charging station $idx ($cs) to $test_point: $dist, round trip: $total_dist (battery limit: $(pso.max_battery_time), accessible: $accessible)")
            end
            # Check if it's in accessible_customers
            if test_point_customer_idx in pso.accessible_customers
                println("  [GREEDY-DEBUG] Test point IS in accessible_customers list")
            else
                println("  [GREEDY-DEBUG] WARNING: Test point is NOT in accessible_customers list!")
            end
        else
            # Check distance to charging stations even if not a customer
            for (idx, cs) in enumerate(ChargingStation)
                dist = max(abs(test_point[1] - cs[1]), abs(test_point[2] - cs[2]))
                return_dist = dist
                total_dist = dist + return_dist
                println("  [GREEDY-DEBUG] Distance from charging station $idx ($cs) to $test_point: $dist, round trip: $total_dist (battery limit: $(pso.max_battery_time))")
            end
        end
        
        # Empty ground stations for this case
        GroundStations = Tuple{Int,Int}[]
        setup_time = time() - setup_start
        println("  [GREEDY] Step 1 completed in $(round(setup_time, digits=3))s")
        
        # === FIRST SOLUTION ===
        println("  [GREEDY] Step 2: Computing first greedy solution for $(pso.n_drones) drones")
        first_solution_start = time()
        # Generate routes for all drones sequentially
        greedy_routes_first = Vector{Vector{Tuple{Int,Int}}}()
        tours_coordinates_first = [Tuple{Int,Int}[] for _ in 1:pso.n_drones]
        
        for drone_idx in 1:pso.n_drones
            println("    [GREEDY] Generating route for drone $drone_idx")
            drone_start = time()
            # Call greedy fallback solution for this drone
            greedy_route = get_greedy_fallback_solution(
                risk_pertime, 
                tours_coordinates_first,  # Contains routes for previous drones
                GridpointsDronesDetecting, 
                ChargingStation, 
                GroundStations, 
                pso.max_battery_time, 
                1,  # Single drone route
                ChargingStation  # Initial drone positions
            )
            drone_time = time() - drone_start
            
            # Handle case where greedy_route is empty or nothing
            if isempty(greedy_route) || all(coord in ChargingStation for coord in greedy_route)
                println("    [GREEDY] Warning: Drone $drone_idx route is empty or only contains depots")
                greedy_route = [ChargingStation[1], ChargingStation[1]]  # Minimal valid route
            end
            
            push!(greedy_routes_first, greedy_route)
            tours_coordinates_first[drone_idx] = greedy_route  # Mark as visited for next drone
            println("    [GREEDY] Drone $drone_idx route: length=$(length(greedy_route)), time=$(round(drone_time, digits=3))s")
        end
        first_solution_time = time() - first_solution_start
        println("  [GREEDY] Step 2 completed in $(round(first_solution_time, digits=3))s")
        println("  [GREEDY] First solution routes summary:")
        for (drone_idx, route) in enumerate(greedy_routes_first)
            route_customers = [coord for coord in route if !(coord in ChargingStation)]
            println("    - Drone $drone_idx: $(length(route)) points, $(length(route_customers)) customers")
        end
        
        # Calculate expected profit from greedy routes directly (before conversion)
        expected_profit_first = 0.0
        customers_in_routes_first = Set{Int}()
        for (drone_idx, greedy_route) in enumerate(greedy_routes_first)
            route_profit = 0.0
            for coord in greedy_route
                if coord in ChargingStation
                    continue
                end
                # Find customer index and add profit
                for i in 1:pso.n_pure_customers
                    if i <= length(pso.customers) && pso.customers[i] == coord
                        if i in pso.accessible_customers
                            route_profit += pso.profits[i]
                            push!(customers_in_routes_first, i)
                        end
                        break
                    end
                end
            end
            expected_profit_first += route_profit
            println("    [GREEDY] Drone $drone_idx route expected profit: $(round(route_profit, digits=6))")
        end
        println("    [GREEDY] Total expected profit from greedy routes: $(round(expected_profit_first, digits=6))")
        
        # Convert each drone's route to customer indices, keeping them separate
        println("  [GREEDY] Step 3: Converting first solution to customer indices (per-drone)")
        convert_first_start = time()
        
        # Get depot indices for separating routes
        depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
        
        # Build position with depot separators: [depot; drone1_customers; depot; drone2_customers; ...]
        position_first_with_depots = Int[]
        for (drone_idx, greedy_route) in enumerate(greedy_routes_first)
            # Add a depot marker before each drone's customers
            if !isempty(depot_indices)
                # Use different depots if available, otherwise reuse
                depot_idx = min(drone_idx, length(depot_indices))
                push!(position_first_with_depots, depot_indices[depot_idx])
            end
            
            # Add customers for this drone
            drone_customers = Int[]
            for coord in greedy_route
                # Skip depot coordinates
                if coord in ChargingStation
                    continue
                end
                
                # Find the customer index for this coordinate
                for i in 1:pso.n_pure_customers
                    if i <= length(pso.customers) && pso.customers[i] == coord
                        if i in pso.accessible_customers
                            push!(drone_customers, i)
                        end
                        break
                    end
                end
            end
            
            # Remove duplicates within this drone's route while preserving order
            seen_in_drone = Set{Int}()
            for customer in drone_customers
                if !(customer in seen_in_drone)
                    push!(position_first_with_depots, customer)
                    push!(seen_in_drone, customer)
                end
            end
            
            println("    [GREEDY] Drone $drone_idx: $(length(seen_in_drone)) unique customers")
        end
        
        convert_first_time = time() - convert_first_start
        
        # Calculate profit from the converted position
        position_profit_first = sum(pso.profits[c] for c in position_first_with_depots if c > 0 && c <= pso.n_pure_customers; init=0.0)
        customer_count_first = sum(1 for c in position_first_with_depots if c > 0 && c <= pso.n_pure_customers)
        println("  [GREEDY] Step 3 completed in $(round(convert_first_time, digits=3))s")
        println("    [GREEDY] Converted position: $(length(position_first_with_depots)) total elements ($customer_count_first customers + $(length(depot_indices)) depots)")
        println("    [GREEDY] Position profit (sum of customer profits): $(round(position_profit_first, digits=6))")
        println("    [GREEDY] Customers in routes: $(length(customers_in_routes_first)), in position: $customer_count_first")
        
        unique_position_first = position_first_with_depots
        
        # === SECOND SOLUTION ===
        println("  [GREEDY] Step 4: Computing second greedy solution for $(pso.n_drones) drones (avoiding first)")
        second_solution_start = time()
        # Generate routes for all drones sequentially, avoiding first solution
        greedy_routes_second = Vector{Vector{Tuple{Int,Int}}}()
        tours_coordinates_second = deepcopy(tours_coordinates_first)  # Start with first solution marked as visited
        
        for drone_idx in 1:pso.n_drones
            println("    [GREEDY] Generating route for drone $drone_idx (second solution)")
            drone_start = time()
            # Call greedy fallback solution for this drone
            greedy_route = get_greedy_fallback_solution(
                risk_pertime, 
                tours_coordinates_second,  # Contains routes for previous drones and first solution
                GridpointsDronesDetecting, 
                ChargingStation, 
                GroundStations, 
                pso.max_battery_time, 
                1,  # Single drone route
                ChargingStation  # Initial drone positions
            )
            drone_time = time() - drone_start
            
            # Handle case where greedy_route is empty or nothing
            if isempty(greedy_route) || all(coord in ChargingStation for coord in greedy_route)
                println("    [GREEDY] Warning: Drone $drone_idx route (second) is empty or only contains depots")
                greedy_route = [ChargingStation[1], ChargingStation[1]]  # Minimal valid route
            end
            
            push!(greedy_routes_second, greedy_route)
            tours_coordinates_second[drone_idx] = greedy_route  # Mark as visited for next drone
            println("    [GREEDY] Drone $drone_idx route (second): length=$(length(greedy_route)), time=$(round(drone_time, digits=3))s")
        end
        second_solution_time = time() - second_solution_start
        println("  [GREEDY] Step 4 completed in $(round(second_solution_time, digits=3))s")
        println("  [GREEDY] Second solution routes summary:")
        for (drone_idx, route) in enumerate(greedy_routes_second)
            route_customers = [coord for coord in route if !(coord in ChargingStation)]
            println("    - Drone $drone_idx: $(length(route)) points, $(length(route_customers)) customers")
        end
        
        # Calculate expected profit from greedy routes directly (before conversion)
        expected_profit_second = 0.0
        customers_in_routes_second = Set{Int}()
        for (drone_idx, greedy_route) in enumerate(greedy_routes_second)
            route_profit = 0.0
            for coord in greedy_route
                if coord in ChargingStation
                    continue
                end
                # Find customer index and add profit
                for i in 1:pso.n_pure_customers
                    if i <= length(pso.customers) && pso.customers[i] == coord
                        if i in pso.accessible_customers
                            route_profit += pso.profits[i]
                            push!(customers_in_routes_second, i)
                        end
                        break
                    end
                end
            end
            expected_profit_second += route_profit
            println("    [GREEDY] Drone $drone_idx route (second) expected profit: $(round(route_profit, digits=6))")
        end
        println("    [GREEDY] Total expected profit from greedy routes (second): $(round(expected_profit_second, digits=6))")
        
        # Convert each drone's route to customer indices, keeping them separate
        println("  [GREEDY] Step 5: Converting second solution to customer indices (per-drone)")
        convert_second_start = time()
        
        # Get depot indices for separating routes
        depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
        
        # Build position with depot separators: [depot; drone1_customers; depot; drone2_customers; ...]
        position_second_with_depots = Int[]
        for (drone_idx, greedy_route) in enumerate(greedy_routes_second)
            # Add a depot marker before each drone's customers
            if !isempty(depot_indices)
                # Use different depots if available, otherwise reuse
                depot_idx = min(drone_idx, length(depot_indices))
                push!(position_second_with_depots, depot_indices[depot_idx])
            end
            
            # Add customers for this drone
            drone_customers = Int[]
            for coord in greedy_route
                # Skip depot coordinates
                if coord in ChargingStation
                    continue
                end
                
                # Find the customer index for this coordinate
                for i in 1:pso.n_pure_customers
                    if i <= length(pso.customers) && pso.customers[i] == coord
                        if i in pso.accessible_customers
                            push!(drone_customers, i)
                        end
                        break
                    end
                end
            end
            
            # Remove duplicates within this drone's route while preserving order
            seen_in_drone = Set{Int}()
            for customer in drone_customers
                if !(customer in seen_in_drone)
                    push!(position_second_with_depots, customer)
                    push!(seen_in_drone, customer)
                end
            end
            
            println("    [GREEDY] Drone $drone_idx (second): $(length(seen_in_drone)) unique customers")
        end
        
        convert_second_time = time() - convert_second_start
        
        # Calculate profit from the converted position
        position_profit_second = sum(pso.profits[c] for c in position_second_with_depots if c > 0 && c <= pso.n_pure_customers; init=0.0)
        customer_count_second = sum(1 for c in position_second_with_depots if c > 0 && c <= pso.n_pure_customers)
        println("  [GREEDY] Step 5 completed in $(round(convert_second_time, digits=3))s")
        println("    [GREEDY] Converted position: $(length(position_second_with_depots)) total elements ($customer_count_second customers + $(length(depot_indices)) depots)")
        println("    [GREEDY] Position profit (sum of customer profits): $(round(position_profit_second, digits=6))")
        println("    [GREEDY] Customers in routes: $(length(customers_in_routes_second)), in position: $customer_count_second")
        
        unique_position_second = position_second_with_depots
        
        # === PROCESS BOTH SOLUTIONS ===
        println("  [GREEDY] Step 6: Finalizing solutions")
        finalize_start = time()
        final_positions = []
        
        # Process first solution (depot markers already included in conversion)
        if !isempty(unique_position_first) && length(unique_position_first) >= 1
            customer_count = sum(1 for c in unique_position_first if c > 0 && c <= pso.n_pure_customers)
            depot_count = sum(1 for c in unique_position_first if c > pso.n_pure_customers)
            println("Greedy fallback first generated position with $(length(unique_position_first)) elements ($customer_count customers + $depot_count depots)")
            # first 5 nodes:
            println("first 5 nodes: $(unique_position_first[1:min(5, length(unique_position_first))])")
            println("  [GREEDY-DEBUG] n_pure_customers: $(pso.n_pure_customers), total customers: $(length(pso.customers))")
            push!(final_positions, unique_position_first)
        else
            println("Greedy fallback first failed, using random initialization")
            # Ensure we have a valid accessible customers list
            if !isempty(pso.accessible_customers)
                push!(final_positions, shuffle(pso.accessible_customers))
            else
                # Last resort: create a minimal solution with just depot nodes if available
                depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
                if !isempty(depot_indices)
                    push!(final_positions, [depot_indices[1]])
                else
                    push!(final_positions, [1])  # Absolute fallback
                end
            end
        end
        
        # Process second solution (depot markers already included in conversion)
        if !isempty(unique_position_second) && length(unique_position_second) >= 1
            customer_count = sum(1 for c in unique_position_second if c > 0 && c <= pso.n_pure_customers)
            depot_count = sum(1 for c in unique_position_second if c > pso.n_pure_customers)
            println("Greedy fallback second generated position with $(length(unique_position_second)) elements ($customer_count customers + $depot_count depots)")
            # first 5 nodes:
            println("first 5 nodes: $(unique_position_second[1:min(5, length(unique_position_second))])")
            push!(final_positions, unique_position_second)
        else
            println("Greedy fallback second failed, using random initialization")
            # Ensure we have a valid accessible customers list
            if !isempty(pso.accessible_customers)
                push!(final_positions, shuffle(pso.accessible_customers))
            else
                # Last resort: create a minimal solution with just depot nodes if available
                depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
                if !isempty(depot_indices)
                    push!(final_positions, [depot_indices[1]])
                else
                    push!(final_positions, [1])  # Absolute fallback
                end
            end
        end
        finalize_time = time() - finalize_start
        println("  [GREEDY] Step 6 completed in $(round(finalize_time, digits=3))s")
        
        total_time = time() - total_start
        println("  [GREEDY] Total greedy fallback time: $(round(total_time, digits=3))s")
        println("    - Setup: $(round(setup_time, digits=3))s ($(round(100*setup_time/total_time, digits=1))%)")
        println("    - First solution: $(round(first_solution_time, digits=3))s ($(round(100*first_solution_time/total_time, digits=1))%)")
        println("    - First conversion: $(round(convert_first_time, digits=3))s ($(round(100*convert_first_time/total_time, digits=1))%)")
        println("    - Second solution: $(round(second_solution_time, digits=3))s ($(round(100*second_solution_time/total_time, digits=1))%)")
        println("    - Second conversion: $(round(convert_second_time, digits=3))s ($(round(100*convert_second_time/total_time, digits=1))%)")
        println("    - Finalization: $(round(finalize_time, digits=3))s ($(round(100*finalize_time/total_time, digits=1))%)")
        
        # Return positions AND expected profits (from greedy route building, not fast_split)
        expected_profits = [expected_profit_first, expected_profit_second]
        return final_positions, expected_profits
        
    catch e
        println("Error in greedy fallback two initialization: $e")
        println("Falling back to random initialization for both particles")
        # Ensure we have valid fallbacks for both particles
        fallback_positions = []
        for i in 1:2
            if !isempty(pso.accessible_customers)
                push!(fallback_positions, shuffle(pso.accessible_customers))
            else
                # Last resort: create a minimal solution
                depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
                if !isempty(depot_indices)
                    push!(fallback_positions, [depot_indices[1]])
                else
                    push!(fallback_positions, [1])  # Absolute fallback
                end
            end
        end
        # Return zero expected profits for fallback
        return fallback_positions, [0.0, 0.0]
    end
end

"""
Initialize one particle using greedy fallback solution
"""
function initialize_with_greedy_fallback(pso::PSOiA_TOP_multiple_depots)
    try
        # Create a synthetic risk map where profits correspond to risk values
        # The greedy fallback expects a 3D array with dimensions (time, x, y)
        # We'll create a simple grid based on customer coordinates
        
        # Find grid bounds from customer coordinates
        max_x = maximum(coord[1] for coord in pso.customers if coord[1] > 0)
        max_y = maximum(coord[2] for coord in pso.customers if coord[2] > 0)
        
        # Create risk map
        risk_pertime = zeros(1, max_x, max_y)
        
        # Set risk values based on customer profits
        for i in 1:pso.n_pure_customers
            if i <= length(pso.customers) && i <= length(pso.profits)
                coord = pso.customers[i]
                if coord[1] > 0 && coord[2] > 0 && coord[1] <= max_x && coord[2] <= max_y
                    risk_pertime[1, coord[1], coord[2]] = pso.profits[i]
                end
            end
        end
        
        # Extract GridpointsDronesDetecting (pure customers only)
        GridpointsDronesDetecting = pso.customers[1:pso.n_pure_customers]
        
        # Extract ChargingStation from depot coordinates
        ChargingStation = pso.depot_coord
        
        # Empty ground stations for this case
        GroundStations = Tuple{Int,Int}[]
        
        # Generate routes for all drones sequentially
        greedy_routes = Vector{Vector{Tuple{Int,Int}}}()
        tours_coordinates = [Tuple{Int,Int}[] for _ in 1:pso.n_drones]
        
        for drone_idx in 1:pso.n_drones
            # Call greedy fallback solution for this drone
            greedy_route = get_greedy_fallback_solution(
                risk_pertime, 
                tours_coordinates,  # Contains routes for previous drones
                GridpointsDronesDetecting, 
                ChargingStation, 
                GroundStations, 
                pso.max_battery_time, 
                1,  # Single drone route
                ChargingStation  # Initial drone positions
            )
            
            # Handle case where greedy_route is empty or nothing
            if isempty(greedy_route) || all(coord in ChargingStation for coord in greedy_route)
                println("Warning: Greedy route for drone $drone_idx is empty or only contains depots")
                greedy_route = [ChargingStation[1], ChargingStation[1]]  # Minimal valid route
            end
            
            push!(greedy_routes, greedy_route)
            tours_coordinates[drone_idx] = greedy_route  # Mark as visited for next drone
        end
        
        # Convert all routes to customer indices and combine into single permutation
        position = Int[]
        for greedy_route in greedy_routes
            for coord in greedy_route
                # Skip depot coordinates - they shouldn't be in the customer list
                if coord in ChargingStation
                    continue
                end
                
                # Find the customer index for this coordinate
                for i in 1:pso.n_pure_customers
                    if i <= length(pso.customers) && pso.customers[i] == coord
                        if i in pso.accessible_customers
                            push!(position, i)
                        end
                        break
                    end
                end
            end
        end
        
        # Remove duplicates while preserving order
        unique_position = Int[]
        seen = Set{Int}()
        for customer in position
            if !(customer in seen)
                push!(unique_position, customer)
                push!(seen, customer)
            end
        end
        
        # If we got a good solution, return it; otherwise fall back to random
        if !isempty(unique_position) && length(unique_position) >= 1
            println("Greedy fallback generated position with $(length(unique_position)) customers")
            
            # CRITICAL FIX: Insert depot nodes to make the permutation compatible with fast_split
            # The fast_split function expects depot nodes (> n_pure_customers) to start tours
            # We need to add a depot node at the beginning of our sequence
            depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
            if !isempty(depot_indices)
                # Add a depot at the beginning to start the tour
                final_position = [depot_indices[1]; unique_position]
                return final_position
            else
                return unique_position
            end
        else
            println("Greedy fallback failed, using random initialization")
            # Ensure we have a valid accessible customers list
            if !isempty(pso.accessible_customers)
                return shuffle(pso.accessible_customers)
            else
                # Last resort: create a minimal solution with just depot nodes if available
                depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
                if !isempty(depot_indices)
                    return [depot_indices[1]]
                else
                    return [1]  # Absolute fallback
                end
            end
        end
        
    catch e
        println("Error in greedy fallback initialization: $e")
        println("Falling back to random initialization")
        # Ensure we have a valid accessible customers list
        if !isempty(pso.accessible_customers)
            return shuffle(pso.accessible_customers)
        else
            # Last resort: create a minimal solution
            depot_indices = collect((pso.n_pure_customers + 1):length(pso.customers))
            if !isempty(depot_indices)
                return [depot_indices[1]]
            else
                return [1]  # Absolute fallback
            end
        end
    end
end

"""
Iterative Destruction/Construction Heuristic (IDCH)
"""
function idch_heuristic(pso::PSOiA_TOP_multiple_depots, slow_version::Bool = false)
    max_iter = slow_version ? length(pso.accessible_customers)^2 : length(pso.accessible_customers)
    
    # Start with random permutation
    current_solution = shuffle(pso.accessible_customers)
    best_solution = copy(current_solution)
    best_profit = fast_split_multiple_depots(best_solution, pso)
    
    no_improvement = 0
    
    while no_improvement < max_iter
        # Destruction phase: remove random customers
        n_remove = rand(1:min(3, length(current_solution) ÷ 2))
        destroyed = copy(current_solution)
        removed_customers = Int[]
        
        for _ in 1:n_remove
            if length(destroyed) > 1
                idx = rand(1:length(destroyed))
                push!(removed_customers, destroyed[idx])
                deleteat!(destroyed, idx)
            end
        end
        
        # Construction phase: reinsert customers using Best Insertion
        reconstructed = best_insertion_algorithm(destroyed, removed_customers, pso)
        profit = fast_split_multiple_depots(reconstructed, pso)
        
        if profit > best_profit
            best_solution = copy(reconstructed)
            best_profit = profit
            no_improvement = 0
        else
            no_improvement += 1
        end
        
        current_solution = reconstructed
    end
    
    return best_solution
end

"""
Best Insertion Algorithm (BIA)
"""
function best_insertion_algorithm(partial_solution::Vector{Int}, unrouted::Vector{Int}, pso::PSOiA_TOP_multiple_depots)
    solution = copy(partial_solution)
    remaining = copy(unrouted)
    
    while !isempty(remaining)
        best_customer = -1
        best_position = -1
        best_cost = Inf
        
        # Random α parameter as mentioned in paper
        α = rand() * 2.0
        
        for customer in remaining
            customer_profit = pso.profits[customer]
            
            # Try inserting at each position
            for pos in 1:(length(solution) + 1)
                # Calculate insertion cost: C_i,z + C_z,j - C_i,j - (P_z)^α
                if pos == 1 && pos > length(solution)
                    # Only customer in route
                    cost_iz = get(pso.costs, (0, customer), 0.0)  # depot to customer
                    cost_zj = get(pso.costs, (customer, 0), 0.0)  # customer to depot
                    cost_ij = 0.0  # no direct connection to remove
                elseif pos == 1
                    # Insert at beginning
                    cost_iz = get(pso.costs, (0, customer), 0.0)  # depot to customer
                    cost_zj = get(pso.costs, (customer, solution[1]), 0.0)  # customer to next
                    cost_ij = get(pso.costs, (0, solution[1]), 0.0)  # depot to next (to remove)
                elseif pos > length(solution)
                    # Insert at end
                    cost_iz = get(pso.costs, (solution[end], customer), 0.0)  # prev to customer
                    cost_zj = get(pso.costs, (customer, 0), 0.0)  # customer to depot
                    cost_ij = get(pso.costs, (solution[end], 0), 0.0)  # prev to depot (to remove)
                else
                    # Insert in middle
                    cost_iz = get(pso.costs, (solution[pos-1], customer), 0.0)  # prev to customer
                    cost_zj = get(pso.costs, (customer, solution[pos]), 0.0)  # customer to next
                    cost_ij = get(pso.costs, (solution[pos-1], solution[pos]), 0.0)  # prev to next (to remove)
                end
                
                # Paper's formula: C_i,z + C_z,j - C_i,j - (P_z)^α
                insertion_cost = cost_iz + cost_zj - cost_ij - (customer_profit^α)
                
                if insertion_cost < best_cost
                    best_cost = insertion_cost
                    best_customer = customer
                    best_position = pos
                end
            end
        end
        
        # Insert best customer if found
        if best_customer != -1
            insert!(solution, best_position, best_customer)
            filter!(x -> x != best_customer, remaining)
        else
            break  # No feasible insertion found
        end
    end
    
    return solution
end

"""
Position update using genetic crossover-like operator
"""
function update_position!(particle::Particle, global_best::Vector{Int}, pso::PSOiA_TOP_multiple_depots)
    n = length(particle.position)
    
    # Calculate number of customers to extract from each source
    r1 = rand()
    r2 = rand()
    
    n_current = floor(Int, pso.w * n)
    n_local = floor(Int, (1 - pso.w) * n * pso.c1 * r1 / (pso.c1 * r1 + pso.c2 * r2))
    n_global = n - n_current - n_local
    
    # Phase 1: Extract subsequences in random order
    M = Set{Int}()  # Marked customers
    sources = [(particle.position, n_current, "current"), 
               (particle.local_best, n_local, "local"), 
               (global_best, n_global, "global")]
    
    # Random extraction order as specified in paper
    shuffle!(sources)
    extracted_subsequences = []
    
    for (source, target_length, name) in sources
        if target_length > 0
            extracted = extract_subsequence(source, target_length, M)
            push!(extracted_subsequences, (extracted, name))
        end
    end
    
    # Phase 2: Link extracted subsequences in random order
    shuffle!(extracted_subsequences)
    new_position = Int[]
    
    for (subsequence, _) in extracted_subsequences
        append!(new_position, subsequence)
    end
    
    # Add remaining customers randomly
    remaining = setdiff(pso.accessible_customers, M)
    append!(new_position, shuffle(remaining))
    
    particle.position = new_position
    particle.current_profit = fast_split_multiple_depots(new_position, pso)
end

"""
Extract subsequence from permutation (core component of PSO update)
"""
function extract_subsequence(permutation::Vector{Int}, target_length::Int, marked::Set{Int})
    if target_length <= 0 || isempty(permutation)
        return Int[]
    end
    
    n = size(permutation, 1)
    r = rand(1:n)  # Random starting location
    extracted = Int[]
    
    # Step 2: Browse from r to end
    for i in r:n
        if !(permutation[i] in marked) && length(extracted) < target_length
            push!(extracted, permutation[i])
            push!(marked, permutation[i])
        end
        if length(extracted) >= target_length
            return extracted
        end
    end
    
    # Step 3: Browse from r down to 1
    for i in (r-1):-1:1
        if !(permutation[i] in marked) && length(extracted) < target_length
            pushfirst!(extracted, permutation[i])
            push!(marked, permutation[i])
        end
        if length(extracted) >= target_length
            return extracted
        end
    end
    
    return extracted
end

"""
Local search with three neighborhoods
"""
function local_search!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    improved = true
    total_time_shift = 0.0
    total_time_swap = 0.0
    total_time_destruction_repair = 0.0
    improved_count_shift = 0
    improved_count_swap = 0
    improved_count_destruction_repair = 0
    call_count_shift = 0
    call_count_swap = 0
    call_count_destruction_repair = 0
    
    while improved
        improved = false
        neighborhoods = [1, 2]#, 3]  # shift, swap, destruction/repair
        shuffle!(neighborhoods)
        
        for neighborhood in neighborhoods
            if neighborhood == 1  # Shift operator
                time_before_shift = time()
                improved = shift_operator!(particle, particle_idx, pso)
                time_after_shift = time()
                total_time_shift += time_after_shift - time_before_shift
                improved_count_shift += improved ? 1 : 0
                call_count_shift += 1
            elseif neighborhood == 2  # Swap operator
                time_before_swap = time()
                improved = swap_operator!(particle, particle_idx, pso)
                time_after_swap = time()
                total_time_swap += time_after_swap - time_after_swap - time_before_swap
                improved_count_swap += improved ? 1 : 0
                call_count_swap += 1
            else  # Destruction/repair operator
                time_before_destruction_repair = time()
                improved = destruction_repair_operator!(particle, particle_idx, pso)
                time_after_destruction_repair = time()
                total_time_destruction_repair += time_after_destruction_repair - time_before_destruction_repair
                improved_count_destruction_repair += improved ? 1 : 0
                call_count_destruction_repair += 1
            end
            
            if improved
                break
            end
        end
    end
    # println("total time shift / call count shift: $(total_time_shift / max(call_count_shift, 1))")
    # println("total time swap / call count swap: $(total_time_swap / max(call_count_swap, 1))")
    # println("total time destruction repair / call count destruction repair: $(total_time_destruction_repair / max(call_count_destruction_repair, 1))")
    # println("improved count shift / call count shift: $(improved_count_shift / max(call_count_shift, 1))")
    # println("improved count swap / call count swap: $(improved_count_swap / max(call_count_swap, 1))")
    # println("improved count destruction repair / call count destruction repair: $(improved_count_destruction_repair / max(call_count_destruction_repair, 1))")
    return total_time_shift, total_time_swap, total_time_destruction_repair
end

# ============================================================================
# SPARSE OPERATORS (with boundary optimization)
# These are new versions that use fast_split_sparse and skip no-op moves.
# ============================================================================

"""
Shift operator with sparse split and boundary optimization.
Skips moves that cannot affect the split profit (moves entirely in dead zones).
Returns (improved::Bool, new_tour_intervals::TourIntervals).
"""
function shift_operator_sparse!(
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots,
    tour_intervals::TourIntervals
)
    n = length(particle.position)
    positions = shuffle(1:n)
    
    for i in positions
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
                if is_blocking_once_inserted(particle, i, j, pso) && is_blocking_once_removed(particle, i, pso)
                    continue
                end
            end
            
            # === EVALUATE SHIFT ===
            new_position = move_element(particle.position, i, j)
            new_profit, _, new_tour_intervals = fast_split_sparse(new_position, pso)
            
            if new_profit > particle.current_profit
                particle.position = new_position
                particle.current_profit = new_profit
                
                # Update node-to-position mapping incrementally
                node_moved = particle.position[j]
                if i < j
                    for pos in (i+1):j
                        node_at_pos = particle.position[pos]
                        particle.node_to_position[node_at_pos] = pos
                    end
                else
                    for pos in j:(i-1)
                        node_at_pos = particle.position[pos]
                        particle.node_to_position[node_at_pos] = pos
                    end
                end
                particle.node_to_position[node_moved] = j
                
                return true, new_tour_intervals
            end
        end
    end
    
    return false, tour_intervals
end

"""
Swap operator with sparse split and boundary optimization.
Skips swaps where both positions are outside all tour intervals.
Returns (improved::Bool, new_tour_intervals::TourIntervals).
Counters: (total_swaps, skipped_swaps) are updated in-place.
"""
function swap_operator_sparse!(
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots,
    tour_intervals::TourIntervals,
    counters::Ref{Tuple{Int, Int}}  # (total_swaps, skipped_swaps)
)
    n = length(particle.position)
    positions = shuffle(1:n)
    pos = particle.position
    
    for i in positions
        node_i = pos[i]
        is_depot_i = node_i > pso.n_pure_customers
        
        for j in shuffle((i+1):n)
            node_j = pos[j]
            is_depot_j = node_j > pso.n_pure_customers
            
            # Increment total swap attempts
            counters[] = (counters[][1] + 1, counters[][2])
            
            # === BOUNDARY OPTIMIZATION ===
            # Skip if both positions are outside all tours (and neither is a depot)
            if !is_depot_i && !is_depot_j
                if !is_active(tour_intervals, i) && !is_active(tour_intervals, j)
                    # Increment skipped swaps
                    counters[] = (counters[][1], counters[][2] + 1)
                    continue  # No-op swap, skip evaluation
                end
            end
            
            # === EXISTING BLOCKING CHECK ===
            if !is_depot_i && !is_depot_j
                if is_blocking_once_inserted(particle, i, j, pso) && is_blocking_once_inserted(particle, j, i, pso)
                    continue
                end
            end
            
            # === EVALUATE SWAP ===
            node_at_i = pos[i]
            node_at_j = pos[j]
            pos[i], pos[j] = pos[j], pos[i]  # Trial swap
            new_profit, _, new_tour_intervals = fast_split_sparse(pos, pso)
            
            if new_profit > particle.current_profit
                particle.current_profit = new_profit
                # Update mapping: O(1)
                particle.node_to_position[node_at_i] = j
                particle.node_to_position[node_at_j] = i
                return true, new_tour_intervals
            else
                # Revert swap
                pos[i], pos[j] = pos[j], pos[i]
            end
        end
    end
    
    return false, tour_intervals
end

"""
Local search using sparse operators with boundary optimization.
Returns timing info (time_shift, time_swap, time_destruction_repair) for compatibility with original local_search!.
Note: particle_idx is accepted for compatibility but not used (mapping is in particle).
"""
function local_search_sparse!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    total_time_shift = 0.0
    total_time_swap = 0.0
    total_time_destruction_repair = 0.0  # Not used in sparse version, kept for compatibility
    
    # Counters for swap boundary optimization statistics
    swap_counters = Ref((0, 0))  # (total_swaps, skipped_swaps)
    
    # Initial split to get tour intervals
    _, _, tour_intervals = fast_split_sparse(particle.position, particle, pso)
    
    improved = true
    while improved
        improved = false
        neighborhoods = shuffle([1, 2])
        
        for neighborhood in neighborhoods
            if neighborhood == 1
                time_before = time()
                improved, tour_intervals = shift_operator_sparse!(particle, pso, tour_intervals)
                total_time_shift += time() - time_before
            else
                time_before = time()
                improved, tour_intervals = swap_operator_sparse!(particle, pso, tour_intervals, swap_counters)
                total_time_swap += time() - time_before
            end
            
            if improved
                break  # Restart from first neighborhood
            end
        end
    end
    
    # Print swap boundary optimization statistics
    total_swaps, skipped_swaps = swap_counters[]
    if total_swaps > 0
        skip_percentage = 100.0 * skipped_swaps / total_swaps
        println("[BOUNDARY-OPT] Swap operations: $total_swaps total, $skipped_swaps skipped ($(round(skip_percentage, digits=2))%)")
    end
    
    return total_time_shift, total_time_swap, total_time_destruction_repair
end

# ============================================================================
# END SPARSE OPERATORS
# ============================================================================

"""
Move element from position i to position j in one pass with no extra copy
"""
function move_element(vec, i, j)
    n = length(vec)
    new_vec = Vector{eltype(vec)}(undef, n)
    customer = vec[i]
    
    # Adjust j if it's after the removed element
    target_j = j > i ? j - 1 : j
    
    new_idx = 1
    for old_idx in 1:n
        if old_idx == i
            continue  # Skip the element we're moving
        end
        
        if new_idx == target_j
            new_vec[target_j] = customer
            new_idx += 1
        end
        
        new_vec[new_idx] = vec[old_idx]
        new_idx += 1
    end
    
    # If target position is at the end
    if target_j == n
        new_vec[n] = customer
    end
    
    return new_vec
end

"""
Check if a node is blocking at its current position (i.e if it is neighbor of its predecessor)
"""
function is_blocking(particle::Particle, i::Int, pso::PSOiA_TOP_multiple_depots)
    if i == 1
        return false
    end
    L = pso.max_battery_time
    node_i = particle.position[i]
    node_i_pred = particle.position[i-1]
    if get(pso.costs, (node_i_pred, node_i), L*4) > L
        return true
    end
    return false
end

"""
Check if a node is blocking at its new position (i.e if it is neighbor of the previous node)
"""
function is_blocking_once_inserted(particle::Particle, i::Int, j::Int, pso::PSOiA_TOP_multiple_depots)
    if j == 1
        return false
    end
    L = pso.max_battery_time
    node_i = particle.position[i]
    node_j_pred = particle.position[j-1]
    if get(pso.costs, (node_j_pred, node_i), L*4) > L
        return true
    end
    return false
end

"""
Check if a node is blocking once removed (i.e if it is neighbor of the previous node)
"""
function is_blocking_once_removed(particle::Particle, i::Int, pso::PSOiA_TOP_multiple_depots)
    if i == length(particle.position) || i == 1
        return false
    end
    L = pso.max_battery_time
    node_i_succ = particle.position[i+1]
    node_i_pred = particle.position[i-1]
    if get(pso.costs, (node_i_pred, node_i_succ), L*4) > L
        return true
    end
    return false
end

"""
Shift operator: move node (customer or depot) to different position
Note: Blocking optimization only applies to customer nodes, not depot nodes,
since depot nodes are route separators and moving them can improve profit
by restructuring routes even if they appear "blocking" in terms of cost.
Updates node-to-position mapping incrementally for efficiency.
"""
function shift_operator!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    time_split = 0.0
    start_time = time()
    n = length(particle.position)
    positions = shuffle(1:n)  # Random order evaluation
    
    for i in positions
        node_i = particle.position[i]
        is_depot = node_i > pso.n_pure_customers

        # if it's a depot, we iterate over all customers. If not, only over the left neighbors.
        candidates = Vector{Int}()
        if is_depot
            candidates = shuffle(1:n)
        else
            candidates = shuffle(pso.left_neighbors[node_i])
        end
        
        for j in shuffle(setdiff(1:n, [i])) # already this is wasteful because the setdiff is not efficient. you can just skip if i ==j, which we do anyway in the is_blocking_once_inserted and is_blocking_once_removed functions. #TODO
            # GRID based optimization: is it worth trying this shift or not?
            # if the shift won't change the current tours, then we don't need to try it
            # easy version first: IF:
            
            # Only apply blocking optimization to customer nodes (not depot nodes)
            # Depot nodes are route separators, so moving them can improve profit by restructuring routes
            # even if they appear "blocking" in terms of cost
            if !is_depot
                # simplified but equivalent: if moving it blocks both, then we don't try it
                if is_blocking_once_inserted(particle, i, j, pso) && is_blocking_once_removed(particle, i, pso)
                    # then we don't try it
                    continue
                end
            end
            # For depot nodes, always try the move (they affect route structure, not just cost feasibility)
            # slightly more complex strategy: TODO implement it just after you improved the swap prcedure.
            ##########
            # # if it blocks there, but there was already blocked, the only thing that matters is here.
            # if is_blocking(particle, j, pso) && is_blocking_once_inserted(particle, i, j, pso)
            #     # the only thing that matters is here.
            #     if !is_blocking_once_removed(particle, i, pso) && !is_blocking(particle, i, pso)
            #         # then we need to know if it increaes profits. if not, no need to try it.
            #         #it increases profits if it connects to a part that increases profits OR if it doesn't connect but has better own profit.
            #         # if the new one is blocking on the right, then we need to check if its profit is less than the old one's profit (eventuellement + ceux du node a droite). If so, we skip.

            #         continue
            #     end
            # end
            ##########





            ## Try moving customer from position i to position j
            # new_position = copy(particle.position)
            # customer = new_position[i]
            # deleteat!(new_position, i)
            # insert!(new_position, j > i ? j-1 : j, customer)
            new_position = move_element(particle.position, i, j)
            
            time_before_split = time()
            new_profit = fast_split_multiple_depots(new_position, pso)
            time_after_split = time()
            time_split += time_after_split - time_before_split
            if new_profit > particle.current_profit
                particle.position = new_position
                particle.current_profit = new_profit
                
                # Update node-to-position mapping incrementally (more efficient than recomputing)
                # When moving from position i to j, positions between min(i,j) and max(i,j) shift
                node_moved = particle.position[j]
                if i < j
                    # Moving forward: positions from i+1 to j shift left by 1
                    for pos in (i+1):j
                        node_at_pos = particle.position[pos]
                        particle.node_to_position[node_at_pos] = pos
                    end
                else  # i > j
                    # Moving backward: positions from j to i-1 shift right by 1
                    for pos in j:(i-1)
                        node_at_pos = particle.position[pos]
                        particle.node_to_position[node_at_pos] = pos
                    end
                end
                particle.node_to_position[node_moved] = j
                
                return true
            end
        end
    end
    ending_time = time()
    #println("time to run split: $time_split")
    #println("time to run shift without split: $(ending_time - start_time - time_split)")
    return false
end

"""
Swap operator: exchange two customers
Updates node-to-position mapping incrementally for efficiency (O(1) update).
"""
function swap_operator!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    n = length(particle.position)
    positions = shuffle(1:n)
    pos = particle.position
    for i in positions
        for j in shuffle((i+1):n)
            # same as for shifts: if both are blocking in their respective new positions, then we don't try it
            if is_blocking_once_inserted(particle, i, j, pso) && is_blocking_once_inserted(particle, j, i, pso)
                continue
            end
            node_at_i = pos[i]
            node_at_j = pos[j]
            pos[i], pos[j] = pos[j], pos[i]  # trial swap
            new_profit = fast_split_multiple_depots(pos, pso)
            if new_profit > particle.current_profit
                particle.current_profit = new_profit
                # Update node-to-position mapping incrementally (O(1) - just 2 updates)
                particle.node_to_position[node_at_i] = j
                particle.node_to_position[node_at_j] = i
                return true  # keep swap; pos already updated
            else
                pos[i], pos[j] = pos[j], pos[i]  # revert
            end
        end
    end
    return false
end

"""
Destruction/repair operator
Since this creates a new permutation, we recompute the mapping (but only when accepting the change).
"""
function destruction_repair_operator!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    n = length(particle.position)
    # Paper specifies: "between 1 and n/m" customers
    max_remove = max(1, n ÷ pso.n_drones)
    n_remove = rand(1:max_remove)
    
    # Remove random customers
    new_position = copy(particle.position)
    removed = Int[]
    
    for _ in 1:n_remove
        if length(new_position) > 1
            idx = rand(1:length(new_position))
            push!(removed, new_position[idx])
            deleteat!(new_position, idx)
        end
    end
    
    # Reconstruct using BIA
    reconstructed = best_insertion_algorithm(new_position, removed, pso)
    new_profit = fast_split_multiple_depots(reconstructed, pso)
    
    if new_profit > particle.current_profit
        particle.position = reconstructed
        particle.current_profit = new_profit
        # Recompute mapping since we created a new permutation
        particle.node_to_position = compute_node_to_position(reconstructed)
        return true
    end
    
    return false
end

"""
Update local best positions with diversity management
"""
function update_local_bests!(pso::PSOiA_TOP_multiple_depots, δ::Float64 = 1e-6)
    # Sort particles by local best profit
    sorted_indices = sortperm([p.local_best_profit for p in pso.swarm])
    worst_idx = sorted_indices[1]
    
    for particle in pso.swarm
        # Rule 1: Apply update only if better than worst local best
        if particle.current_profit > pso.swarm[worst_idx].local_best_profit
            # Calculate travel cost for current particle position
            current_cost = calculate_travel_cost(particle.position, pso)
            
            # Rule 2: Find similar particle
            similar_found = false
            
            for (i, other) in enumerate(pso.swarm)
                other_cost = calculate_travel_cost(other.local_best, pso)
                
                # Similarity check: same profit AND travel cost difference < δ
                if abs(other.local_best_profit - particle.current_profit) < δ && 
                   abs(other_cost - current_cost) < δ
                    # Replace similar particle
                    pso.swarm[i].local_best = copy(particle.position)
                    pso.swarm[i].local_best_profit = particle.current_profit
                    # Note: node_to_position tracks current position, not local_best, so no update needed here
                    similar_found = true
                    break
                end
            end
            
            # Rule 3: If no similar particle found, replace worst
            if !similar_found
                pso.swarm[worst_idx].local_best = copy(particle.position)
                pso.swarm[worst_idx].local_best_profit = particle.current_profit
                # Note: node_to_position tracks current position, not local_best, so no update needed here
            end
        end
        
        # Update personal best
        if particle.current_profit > particle.local_best_profit
            particle.local_best = copy(particle.position)
            particle.local_best_profit = particle.current_profit
            # Note: node_to_position mapping tracks current position, not local_best
            # So we don't need to update it here since it's already updated when position changes
        end
        
        # Update global best
        if particle.current_profit > pso.global_best_profit
            pso.global_best = copy(particle.position)
            pso.global_best_profit = particle.current_profit
        end
    end
end

"""
Calculate travel cost for a given solution (needed for similarity measure)
"""
function calculate_travel_cost(permutation::Vector{Int}, pso::PSOiA_TOP_multiple_depots)
    if isempty(permutation)
        return 0.0
    end
    
    routes = extract_routes(permutation, pso)
    total_cost = 0.0
    
    for route in routes
        if !isempty(route)
            # Cost from depot to first customer
            total_cost += get(pso.costs, (0, route[1]), 0.0)
            
            # Cost between customers
            for i in 1:(length(route)-1)
                total_cost += get(pso.costs, (route[i], route[i+1]), 0.0)
            end
            
            # Cost from last customer to depot
            total_cost += get(pso.costs, (route[end], 0), 0.0)
        end
    end
    
    return total_cost
end
# """
# Main PSO algorithm - following Algorithm 1 from the paper exactly
# """
# function solve_PSO_TOP_multiple_depots(customers::Vector{Tuple{Int,Int}}, profits::Vector{Float64}, 
#                        costs::Dict{Tuple{Int,Int}, Float64}, n_drones::Int, 
#                        max_battery_time::Int, depot_coord::Tuple{Int,Int} = (0, 0);
#                        swarm_size::Int = 50, max_iterations::Int = 1000,
#                        w::Float64 = 0.3, c1::Float64 = 0.5, c2::Float64 = 0.3,
#                        ph::Float64 = 0.1, pm::Float64 = 0.3)
    
#     # Start timing the algorithm execution
#     start_time = time()
    
#     # Determine accessible customers using L-infinity distance instead of cost matrix
#     accessible_customers = Int[]
#     println("Customers: $(customers)")
#     println("See, we have the depots above...")
#     for i in 1:length(customers)
#         # Calculate L-infinity distance (minimum hops) to visit customer and return
#         customer_coord = customers[i]
#         depot_x, depot_y = depot_coord
#         customer_x, customer_y = customer_coord
        
#         # L-infinity distance: max(|x1-x2|, |y1-y2|)
#         distance_to = max(abs(customer_x - depot_x), abs(customer_y - depot_y))
#         distance_from = max(abs(customer_x - depot_x), abs(customer_y - depot_y))  # Same for return
        
#         # Check if customer can be visited and returned within battery limit
#         total_distance = distance_to + distance_from
#         if total_distance <= max_battery_time
#             push!(accessible_customers, i)
#         end
#     end
    
#     # Initialize PSO
#     pso = PSOiA_TOP_multiple_depots(
#         Particle[], Int[], -Inf, swarm_size, max_iterations,
#         w, c1, c2, ph, pm, n_drones, max_battery_time,
#         customers, profits, costs, accessible_customers, depot_coord
#     )
    
#     # println("=== PSO SETUP ===")
#     # println("Total customers: $(length(customers))")
#     # println("Accessible customers: $(length(accessible_customers))")
#     # println("Max battery time: $max_battery_time")
#     # println("Number of drones: $n_drones")
#     # println("==================")
    
#     # Initialize and evaluate each particle in swarm (see Section 2.3)
#     initialize_swarm(pso)

#     iter = 1
#     itermax = max_iterations #* length(accessible_customers) * n_drones  # As mentioned in paper
    
#     # println("Starting PSO with $(pso.swarm_size) particles, initial best: $(pso.global_best_profit)")
#     # Main algorithm loop following Algorithm 1
#     while iter <= itermax
#         improvement_found = false
        
#         for x in 1:pso.swarm_size
#             # Random move with probability ph
#             if rand() < pso.ph
#                 # Move S[x] to a new position (see Section 2.3)
#                 pso.swarm[x].position = idch_heuristic(pso, false)  # Fast version
#             else
#                 # Update S[x].pos (see Section 2.5)
#                 update_position!(pso.swarm[x], pso.global_best, pso)
#             end
            
#             # Local search with probability pm
#             if rand() < pso.pm
#                 # Apply local search on S[x].pos (see Section 2.4)
#                 local_search!(pso.swarm[x], pso)
#             end
            
#             # Evaluate S[x].pos (see Section 2.2)
#             pso.swarm[x].current_profit = fast_split_multiple_depots(pso.swarm[x].position, pso)
            
#             # Update lbest of S (see Section 2.6)
#             prev_global_best = pso.global_best_profit
#             update_local_bests!(pso)

#             # Check if update Rule 3 is applied (new local best discovered)
#             if pso.global_best_profit > prev_global_best
#                 improvement_found = true
#                 println("Iter $iter: New best = $(round(pso.global_best_profit, digits=3)), Solution: $(pso.global_best)")
#             end
#         end

#         if improvement_found
#             iter = 1  # Reset counter when improvement found
#         else
#             iter += 1  # Increment counter when no improvement
#         end
#     end
    
#     # Calculate and print execution time
#     end_time = time()
#     execution_time = end_time - start_time
#     println("=== PSO ALGORITHM COMPLETED ===")
#     println("Final best profit: $(round(pso.global_best_profit, digits=3))")
#     println("Total execution time: $(round(execution_time, digits=3)) seconds")
#     println("==============================")
    
#     return pso.global_best, pso.global_best_profit, pso
# end







"""
Main PSO algorithm adapted for multiple depots
"""
function solve_PSO_TOP_multiple_depots(customers::Vector{Tuple{Int,Int}}, profits::Vector{Float64}, 
                       costs::Dict{Tuple{Int,Int}, Float64}, left_neighbors::Dict{Int, Vector{Int}}, 
                       n_drones::Int, n_pure_customers::Int,
                       max_battery_time::Int, depot_coord::Vector{Tuple{Int,Int}} = [(0, 0)],
                       blocked::Set{Tuple{Int,Int}} = Set{Tuple{Int,Int}}();
                       swarm_size::Int = 50, max_iterations::Int = 1000, max_time::Float64 = 3600.0,
                       w::Float64 = 0.3, c1::Float64 = 0.5, c2::Float64 = 0.3,
                       ph::Float64 = 0.1, pm::Float64 = 0.3, use_greedy_init::Bool = true)
    # println("Starting solve_PSO_TOP_multiple_depots in TOP_PSO_multi_depot.jl...")
    # Start timing the algorithm execution
    start_time = time()
    println("[TIME CHECK] Algorithm started: max_time=$(max_time)s")
    
    # Determine accessible customers
    accessible_customers = Int[]
    closest_depot_distance = Vector{Float64}(undef, length(customers))
    
    if !isempty(blocked)
        # Use BFS-based accessibility when blocked cells exist
        bfs_start_time = time()
        elapsed = bfs_start_time - start_time
        println("[TIME CHECK] Starting BFS computation: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")
        
        # First, determine grid bounds from customer coordinates
        max_x = max(maximum(coord[1] for coord in customers), maximum(coord[1] for coord in depot_coord))
        max_y = max(maximum(coord[2] for coord in customers), maximum(coord[2] for coord in depot_coord))
        N, M = max_x, max_y
        
        # Compute BFS distances from all depots to all customers
        # We use BFS to find actual path distances respecting blocked cells
        inbounds(x, y) = 1 <= x <= N && 1 <= y <= M
        neighbors(x,y) = ((x+1,y), (x-1,y), (x, y+1), (x, y-1), (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1))
        
        # Initialize distance array
        dist = fill(Inf, N, M)
        Q = DataStructures.Queue{Tuple{Int,Int}}()
        
        # Initialize all depots as sources
        for (sx,sy) in depot_coord
            if inbounds(sx,sy) && !((sx,sy) in blocked)
                dist[sx,sy] = 0
                DataStructures.enqueue!(Q, (sx,sy))
            end
        end
        
        # BFS to compute distances with periodic time checks
        bfs_iterations = 0
        last_check_time = time()
        while !isempty(Q)
            # Periodic time check every 0.5 seconds
            current_check_time = time()
            if current_check_time - last_check_time > 0.5
                elapsed = current_check_time - start_time
                remaining = max_time - elapsed
                queue_size = length(Q)
                println("[TIME CHECK] BFS in progress: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(remaining, digits=2))s, queue_size=$queue_size, iterations=$bfs_iterations")
                if elapsed > max_time
                    println("[TIME CHECK] Time limit exceeded during BFS! Stopping.")
                    return pso.global_best, pso.global_best_profit, pso
                end
                last_check_time = current_check_time
            end
            
            (x,y) = DataStructures.dequeue!(Q)
            dxy = dist[x,y]
            for (nx,ny) in neighbors(x,y)
                if inbounds(nx,ny) && !((nx,ny) in blocked) && isinf(dist[nx,ny])
                    dist[nx,ny] = dxy + 1
                    DataStructures.enqueue!(Q, (nx,ny))
                end
            end
            bfs_iterations += 1
        end
        bfs_end_time = time()
        elapsed = bfs_end_time - start_time
        println("[TIME CHECK] BFS completed: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s, iterations=$bfs_iterations")
        
        # Check accessibility for each customer using BFS distances
        for i in 1:length(customers)
            customer_coord = customers[i]
            customer_x, customer_y = customer_coord
            
            if inbounds(customer_x, customer_y) && !isinf(dist[customer_x, customer_y])
                bfs_distance = dist[customer_x, customer_y]
                closest_depot_distance[i] = bfs_distance
                
                # Check if customer can be visited and returned within battery limit (2 * distance for round trip)
                if 2 * bfs_distance <= max_battery_time
                    push!(accessible_customers, i)
                end
            else
                closest_depot_distance[i] = Inf
            end
        end
    else
        # Original L-infinity distance based accessibility
        for i in 1:length(customers)
            # Calculate L-infinity distance (minimum hops) to visit customer and return
            customer_coord = customers[i]
            min_distance = max_battery_time*2
            for depot in depot_coord
                depot_x, depot_y = depot
                customer_x, customer_y = customer_coord
                # L-infinity distance: max(|x1-x2|, |y1-y2|)
                distance_to = max(abs(customer_x - depot_x), abs(customer_y - depot_y))
                distance_from = max(abs(customer_x - depot_x), abs(customer_y - depot_y))
                if distance_to + distance_from < min_distance
                    min_distance = distance_to + distance_from
                end
            end
                
            # Check if customer can be visited and returned within battery limit
            if min_distance <= max_battery_time
                push!(accessible_customers, i)
            end
        end
        
        # Precompute closest depot return distance (Chebyshev) for every customer
        for i in 1:length(customers)
            customer_x, customer_y = customers[i]
            min_d = typemax(Int)
            for (depot_x, depot_y) in depot_coord
                d = max(abs(depot_x - customer_x), abs(depot_y - customer_y))
                if d < min_d
                    min_d = d
                end
            end
            closest_depot_distance[i] = Float64(min_d)
        end
    end
    elapsed = time() - start_time
    println("[TIME CHECK] After accessibility check: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")
    println("accessible_customers: $(length(accessible_customers))")
    println("n_pure_customers: $(n_pure_customers)")
    
    # Initialize PSO (node_to_position is now stored in each Particle)
    pso = PSOiA_TOP_multiple_depots(
        Particle[], Int[], -Inf, swarm_size, max_iterations,
        w, c1, c2, ph, pm, n_drones, n_pure_customers, max_battery_time,
        customers, profits, costs, left_neighbors,
        accessible_customers, depot_coord, closest_depot_distance
    )
    
    elapsed = time() - start_time
    println("[TIME CHECK] After PSO object creation: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")

    time_to_initialize_pso = time() - start_time
    println("time to initialize pso: $(time_to_initialize_pso)")
    time_before_swarm_sampling = time()
    elapsed = time_before_swarm_sampling - start_time
    println("[TIME CHECK] Before swarm initialization: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")
    
    # Initialize and evaluate each particle in swarm (see Section 2.3)
    # Skip IDCH for testing purposes
    initialize_swarm(pso, use_greedy_init, skip_idch=true)
    time_after_swarm_sampling = time()
    elapsed = time_after_swarm_sampling - start_time
    println("time to initialize swarm: $(time_after_swarm_sampling - time_before_swarm_sampling)")
    println("[TIME CHECK] After swarm initialization: elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")
    
    # Check if we've already exceeded the time limit during initialization
    if elapsed > max_time
        println("[TIME CHECK] Maximum time limit of $(max_time) seconds reached during initialization. Stopping algorithm.")
        return pso.global_best, pso.global_best_profit, pso
    end

    iter = 1
    itermax = max_iterations #* length(accessible_customers) * n_drones  # As mentioned in paper
    
    # println("Starting PSO with $(pso.swarm_size) particles, initial best: $(pso.global_best_profit)")
    # Main algorithm loop following Algorithm 1
    total_time_local_search = 0.0
    total_time_swap = 0.0
    total_time_shift = 0.0
    total_time_destruction_repair = 0.0
    while iter <= itermax
        # Check if max_time has been exceeded
        current_time = time()
        elapsed_time = current_time - start_time
        remaining_time = max_time - elapsed_time
        if iter % 10 == 1 || elapsed_time > max_time * 0.9  # Print every 10 iterations or when 90% time used
            println("[TIME CHECK] Iteration $iter: elapsed=$(round(elapsed_time, digits=2))s, remaining=$(round(remaining_time, digits=2))s, best_profit=$(round(pso.global_best_profit, digits=6))")
        end
        if elapsed_time > max_time
            println("[TIME CHECK] Maximum time limit of $(max_time) seconds reached at iteration $iter. Stopping algorithm.")
            break
        end
        
        improvement_found = false
        time_limit_exceeded = false
        
        for x in 1:pso.swarm_size
            # Time check at start of each particle
            particle_start_time = time()
            elapsed = particle_start_time - start_time
            if elapsed > max_time
                println("[TIME CHECK] Time limit exceeded at start of particle $x of iteration $iter. Stopping.")
                time_limit_exceeded = true
                break
            end
            if x <= 3 || x % 10 == 1  # Log first 3 particles and every 10th
                println("[TIME CHECK] Processing particle $x/$(pso.swarm_size) (iter $iter): elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")
            end
            
            # Random move with probability ph
            if rand() < pso.ph
                # Move S[x] to a new position (see Section 2.3)
                time_before_idch = time()
                pso.swarm[x].position = idch_heuristic(pso, false)  # Fast version
                time_after_idch = time()
                idch_time = time_after_idch - time_before_idch
                elapsed = time_after_idch - start_time
                if idch_time > 0.1  # Log if IDCH takes more than 0.1 seconds
                    println("[TIME CHECK] IDCH took $(round(idch_time, digits=3))s (particle $x, iter $iter), elapsed=$(round(elapsed, digits=2))s")
                end
                # Update node-to-position mapping after position change
                pso.swarm[x].node_to_position = compute_node_to_position(pso.swarm[x].position)
            else
                # Update S[x].pos (see Section 2.5)
                update_position!(pso.swarm[x], pso.global_best, pso)
                # Update node-to-position mapping after position change
                pso.swarm[x].node_to_position = compute_node_to_position(pso.swarm[x].position)
            end
            
            # Local search with probability pm
            if rand() < pso.pm
                # Apply local search on S[x].pos (see Section 2.4)
                # Using sparse operators with boundary optimization
                time_before_local_search = time()
                time_shift, time_swap, time_destruction_repair = local_search_sparse!(pso.swarm[x], x, pso)
                time_after_local_search = time()
                local_search_time = time_after_local_search - time_before_local_search
                total_time_local_search += local_search_time
                total_time_shift += time_shift
                total_time_swap += time_swap
                total_time_destruction_repair += time_destruction_repair
                elapsed = time_after_local_search - start_time
                if local_search_time > 0.1  # Log if local search takes more than 0.1 seconds
                    println("[TIME CHECK] Local search took $(round(local_search_time, digits=3))s (particle $x, iter $iter), elapsed=$(round(elapsed, digits=2))s")
                end
                # Note: node-to-position mapping is updated incrementally within sparse shift/swap operators
            end
            
            # Evaluate S[x].pos (see Section 2.2)
            time_before_split = time()
            pso.swarm[x].current_profit = fast_split_multiple_depots(pso.swarm[x].position, pso)
            time_after_split = time()
            split_time = time_after_split - time_before_split
            elapsed = time_after_split - start_time
            if split_time > 0.1  # Log if fast_split takes more than 0.1 seconds
                println("[TIME CHECK] fast_split took $(round(split_time, digits=3))s (particle $x, iter $iter), elapsed=$(round(elapsed, digits=2))s")
            end
            
            # Check time limit after expensive operations
            if elapsed > max_time
                println("[TIME CHECK] Time limit exceeded after particle $x of iteration $iter. Stopping.")
                time_limit_exceeded = true
                break
            end
            
            # Update lbest of S (see Section 2.6)
            prev_global_best = pso.global_best_profit
            update_local_bests!(pso)

            # Check if update Rule 3 is applied (new local best discovered)
            if pso.global_best_profit > prev_global_best
                improvement_found = true
                println("Iter $iter: New best = $(round(pso.global_best_profit, digits=3))")#, Solution: $(pso.global_best)")
            end
        end
        
        # Break out of outer loop if time limit was exceeded
        if time_limit_exceeded
            break
        end

        if improvement_found
            iter = 1  # Reset counter when improvement found
        else
            iter += 1  # Increment counter when no improvement
        end
    end
    println("total time local search: $total_time_local_search")
    println("total time shift: $total_time_shift")
    println("total time swap: $total_time_swap")
    println("total time destruction repair: $total_time_destruction_repair")
    println("total time sum: $(total_time_shift + total_time_swap + total_time_destruction_repair)")
    # Calculate and print execution time
    end_time = time()
    execution_time = end_time - start_time
    println("=== PSO ALGORITHM COMPLETED ===")
    println("Final best profit: $(round(pso.global_best_profit, digits=3))")
    println("Total execution time: $(round(execution_time, digits=3)) seconds")
    println("==============================")
    
    return pso.global_best, pso.global_best_profit, pso
end









"""
Convert giant tour to actual routes using the optimal split procedure from the paper
"""
function extract_routes(giant_tour::Vector{Int}, pso::PSOiA_TOP_multiple_depots)
    # Use the sparse split procedure (Section 13.1)
    optimal_profit, routes, _ = fast_split_sparse(giant_tour, pso)
    
    #println("=== OPTIMAL ROUTES ===")
    for (i, route) in enumerate(routes)
        if !isempty(route)
            # Calculate actual costs for verification
            route_cost = 0.0
            if !isempty(route)
                # Cost from depot to first customer
                route_cost += get(pso.costs, (0, route[1]), 0.0)
                
                # Cost between customers
                for j in 1:(length(route)-1)
                    route_cost += get(pso.costs, (route[j], route[j+1]), 0.0)
                end
                
                # Cost from last customer to depot
                route_cost += get(pso.costs, (route[end], 0), 0.0)
            end
            
            route_profit = sum(pso.profits[c] for c in route)
            # println("  Drone $i: $route (cost: $route_cost, profit: $(round(route_profit, digits=3)))")
        end
    end
    # println("Total profit: $(round(optimal_profit, digits=3))")
    
    return routes
end