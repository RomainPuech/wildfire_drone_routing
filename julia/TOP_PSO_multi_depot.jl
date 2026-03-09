using Random
using DataStructures
using Plots

# Runtime toggles for boundary optimizations
const ENABLE_SHIFT_IRRELEVANCE_FILTER = Ref(true)
const ENABLE_SWAP_BLOCKING_FILTER = Ref(true)
const ENABLE_IDCH = Ref(false)
const ENABLE_INCREMENTAL_LOCAL_SEARCH = Ref(true)
const ENABLE_COST_MATRIX = Ref(true)
const ENABLE_LIVE_ZONE_FILTER = Ref(false)
const ENABLE_LAZY_DEAD_FILTER = Ref(true)   # Applies to swap operators only (not shift)
const ENABLE_SPARSE_SPLIT = Ref(true)       # When false, use dense O(n²) split instead of sparse O(k·n/k)

# Boundary optimization stats (candidate-level)
const SHIFT_STATS = Ref((candidates=0, skipped=0, time=0.0, calls=0))
const SWAP_STATS = Ref((candidates=0, skipped=0, time=0.0, calls=0))
const SPLIT_SPARSE_STATS = Ref((calls=0, time=0.0))
const SPLIT_SPARSE_PROFIT_STATS = Ref((calls=0, time=0.0))
const SPLIT_DENSE_STATS = Ref((calls=0, time=0.0))

function reset_boundary_stats!()
    SHIFT_STATS[] = (candidates=0, skipped=0, time=0.0, calls=0)
    SWAP_STATS[] = (candidates=0, skipped=0, time=0.0, calls=0)
    SPLIT_SPARSE_STATS[] = (calls=0, time=0.0)
    SPLIT_SPARSE_PROFIT_STATS[] = (calls=0, time=0.0)
    SPLIT_DENSE_STATS[] = (calls=0, time=0.0)
    if isdefined(@__MODULE__, :INCREMENTAL_SWAP_STATS)
        INCREMENTAL_SWAP_STATS[] = (candidates=0, skipped_blocking=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
    end
    if isdefined(@__MODULE__, :INCREMENTAL_SHIFT_STATS)
        INCREMENTAL_SHIFT_STATS[] = (candidates=0, skipped_filter=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
    end
end

function get_boundary_stats()
    return SHIFT_STATS[], SWAP_STATS[], SPLIT_SPARSE_STATS[], SPLIT_SPARSE_PROFIT_STATS[], SPLIT_DENSE_STATS[]
end

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
    costs::Dict{Tuple{Int,Int}, Float64}  # Travel costs (dict, legacy)
    cost_matrix::Matrix{Float64}           # Travel costs (dense matrix, cost_matrix[i+1,j+1] for nodes i,j; 0-indexed artificial node → row/col 1)
    left_neighbors::Dict{Int, Vector{Int}}  # Left neighbors
    accessible_customers::Vector{Int}  # Indices of accessible customers
    depot_coord::Vector{Tuple{Int,Int}}    # Depot coordinates
    closest_depot_distance::Vector{Float64}  # Pre-computed min return distance to closest depot (Chebyshev)
end

"""
Inline cost lookup.  When `ENABLE_COST_MATRIX[]` is true, uses the dense O(1) matrix;
otherwise falls back to the Dict `get` with the given default.
"""
@inline function lookup_cost(pso::PSOiA_TOP_multiple_depots, from::Int, to::Int, default::Real)::Float64
    if ENABLE_COST_MATRIX[]
        return @inbounds pso.cost_matrix[from + 1, to + 1]
    else
        return get(pso.costs, (from, to), Float64(default))
    end
end

"""
Fast split procedure for GRID (grid-based) multiple depots
"""
function fast_split_with_routes_multiple_depots(permutation::Vector{Int}, pso_multiple_depots::PSOiA_TOP_multiple_depots)
    split_start_time = time()
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
            if !ENABLE_COST_MATRIX[] && !haskey(pso_multiple_depots.costs, cost_key)
                missing_costs_count += 1
                if missing_costs_count <= 5  # Only store first 5 for logging
                    push!(missing_costs_pairs, cost_key)
                end
            end
            travel_cost = lookup_cost(pso_multiple_depots, prev_customer, customer_idx, L*4)
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

    result = (Γ[1, m + 1], routes)
    SPLIT_DENSE_STATS[] = (calls=SPLIT_DENSE_STATS[].calls + 1,
                           time=SPLIT_DENSE_STATS[].time + (time() - split_start_time))
    return result
end



# Use sparse split for all profit-only evaluations (Section 13.1)
function fast_split_multiple_depots(permutation::Vector{Int}, pso_multiple_depots::PSOiA_TOP_multiple_depots)
    if ENABLE_SPARSE_SPLIT[]
        return fast_split_sparse_profit(permutation, pso_multiple_depots)
    else
        profit, _ = fast_split_with_routes_multiple_depots(permutation, pso_multiple_depots)
        return profit
    end
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
            travel_cost = lookup_cost(pso, prev_customer, customer_idx, L * 4)
            
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
Compute saturated tour profit starting from a given depot position.
Returns 0.0 if start_pos is invalid or does not point to a depot.
"""
function compute_saturated_tour_profit(
    permutation::Vector{Int},
    start_pos::Int,
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    if start_pos <= 0 || start_pos > n
        return 0.0
    end
    if permutation[start_pos] <= pso.n_pure_customers
        return 0.0
    end
    L = pso.max_battery_time
    current_cost = 0.0
    current_profit = 0.0
    prev_customer = permutation[start_pos]
    j = start_pos + 1

    while j <= n
        customer_idx = permutation[j]
        travel_cost = lookup_cost(pso, prev_customer, customer_idx, L * 4)
        return_distance = pso.closest_depot_distance[customer_idx]

        current_cost += travel_cost
        if current_cost + return_distance > L
            break
        end

        current_profit += pso.profits[customer_idx]
        prev_customer = customer_idx
        j += 1
    end

    return current_profit
end

"""
Return indices of saturated tours (by depot index) whose intervals contain position pos.
"""
function get_tours_covering_position(
    sorted_depot_positions::Vector{Int},
    tour_lengths_sparse::Vector{Int},
    pos::Int
)
    if pos < 1
        return Int[]
    end
    covered = Int[]
    for idx in eachindex(sorted_depot_positions)
        depot_pos = sorted_depot_positions[idx]
        tour_end = depot_pos + tour_lengths_sparse[idx] - 1
        if depot_pos <= pos && pos <= tour_end
            push!(covered, idx)
        end
    end
    return covered
end

"""
Return a Bool vector dead_positions where dead_positions[pos] = true
if no saturated tour covers position pos.
"""
function compute_dead_positions(
    n::Int,
    sorted_depot_positions::Vector{Int},
    tour_lengths_sparse::Vector{Int}
)
    covered = falses(n)
    for idx in eachindex(sorted_depot_positions)
        start_pos = sorted_depot_positions[idx]
        end_pos = start_pos + tour_lengths_sparse[idx] - 1
        for pos in start_pos:min(end_pos, n)
            covered[pos] = true
        end
    end
    return .!covered
end

"""
Compute safe dead positions for the live zone filter.
Unlike compute_dead_positions, this extends each tour's coverage by +1 to include
the *boundary* position — the first position the greedy tour reads but does NOT include
in the tour. Changing the node at the boundary can alter the tour's extension decision,
so it must be treated as "live" for the purpose of filtering shift/swap candidates.
"""
function compute_safe_dead_positions(
    n::Int,
    sorted_depot_positions::Vector{Int},
    tour_lengths_sparse::Vector{Int}
)
    covered = falses(n)
    for idx in eachindex(sorted_depot_positions)
        start_pos = sorted_depot_positions[idx]
        end_pos = start_pos + tour_lengths_sparse[idx]  # +1 vs compute_dead_positions
        for pos in start_pos:min(end_pos, n)
            covered[pos] = true
        end
    end
    return .!covered
end

"""
Compute dead block boundaries for each position.
Returns (block_start, block_end) where:
  - block_start[p] = first position in the contiguous dead block containing p (0 if p is live)
  - block_end[p]   = last  position in the contiguous dead block containing p (0 if p is live)
"""
function compute_dead_block_boundaries(dead_positions::BitVector)
    n = length(dead_positions)
    block_start = zeros(Int, n)
    block_end   = zeros(Int, n)
    cur_start = 0
    for p in 1:n
        if dead_positions[p]
            if cur_start == 0
                cur_start = p
            end
        else
            if cur_start > 0
                for q in cur_start:p-1
                    block_start[q] = cur_start
                    block_end[q]   = p - 1
                end
                cur_start = 0
            end
        end
    end
    if cur_start > 0
        for q in cur_start:n
            block_start[q] = cur_start
            block_end[q]   = n
        end
    end
    return block_start, block_end
end

"""
Check if position `pos` is safe-dead (no tour reads it, with +1 boundary extension).
O(k) where k = number of depots.
"""
@inline function is_position_safe_dead(pos::Int, sorted_depot_positions::Vector{Int}, tour_lengths_sparse::Vector{Int})
    @inbounds for idx in eachindex(sorted_depot_positions)
        d = sorted_depot_positions[idx]
        if d <= pos <= d + tour_lengths_sparse[idx]   # +1 extension: [d, d+len] includes boundary
            return false
        end
    end
    return true
end

"""
Check if the range [lo, hi] is entirely safe-dead (no tour's extended range overlaps it).
O(k) where k = number of depots.
"""
@inline function is_range_safe_dead(lo::Int, hi::Int, sorted_depot_positions::Vector{Int}, tour_lengths_sparse::Vector{Int})
    @inbounds for idx in eachindex(sorted_depot_positions)
        d = sorted_depot_positions[idx]
        tour_end = d + tour_lengths_sparse[idx]  # +1 extension
        if d <= hi && lo <= tour_end  # interval overlap test
            return false
        end
    end
    return true
end

"""
Compute the new start position of a depot tour after shifting i -> j (assumes i < j).
"""
function shift_new_depot_pos(depot_pos::Int, i::Int, j::Int)
    # move_element inserts at position j-1 when j > i (before original j)
    # Positions in (i, j) shift left by 1; positions >= j are unchanged.
    if depot_pos <= i
        return depot_pos
    elseif depot_pos < j
        return depot_pos - 1
    else
        return depot_pos
    end
end

"""
Remove element at position i (1-based) and return a new vector.
"""
function remove_element(vec::Vector{Int}, i::Int)
    n = length(vec)
    if i < 1 || i > n
        return copy(vec)
    end
    new_vec = Vector{Int}(undef, n - 1)
    new_idx = 1
    for old_idx in 1:n
        if old_idx == i
            continue
        end
        new_vec[new_idx] = vec[old_idx]
        new_idx += 1
    end
    return new_vec
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
    if !ENABLE_SPARSE_SPLIT[]
        profit, routes = fast_split_with_routes_multiple_depots(permutation, pso)
        return profit, routes, empty_tour_intervals()
    end
    split_start_time = time()
    result = fast_split_sparse_with_mapping(permutation, particle.node_to_position, pso)
    SPLIT_SPARSE_STATS[] = (calls=SPLIT_SPARSE_STATS[].calls + 1,
                            time=SPLIT_SPARSE_STATS[].time + (time() - split_start_time))
    return result
end

"""
Sparse split procedure (profit-only): Version 1 - uses node_to_position from particle.
Returns profit only.
"""
function fast_split_sparse_profit(
    permutation::Vector{Int},
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots
)
    split_start_time = time()
    result = fast_split_sparse_profit_with_mapping(permutation, particle.node_to_position, pso)
    SPLIT_SPARSE_PROFIT_STATS[] = (calls=SPLIT_SPARSE_PROFIT_STATS[].calls + 1,
                                   time=SPLIT_SPARSE_PROFIT_STATS[].time + (time() - split_start_time))
    return result
end

"""
Sparse split procedure: Version 2 - computes mapping on-the-fly.
Returns (profit, routes, tour_intervals).
"""
function fast_split_sparse(
    permutation::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    if !ENABLE_SPARSE_SPLIT[]
        profit, routes = fast_split_with_routes_multiple_depots(permutation, pso)
        return profit, routes, empty_tour_intervals()
    end
    split_start_time = time()
    node_to_position = compute_node_to_position(permutation)
    result = fast_split_sparse_with_mapping(permutation, node_to_position, pso)
    SPLIT_SPARSE_STATS[] = (calls=SPLIT_SPARSE_STATS[].calls + 1,
                            time=SPLIT_SPARSE_STATS[].time + (time() - split_start_time))
    return result
end

"""
Sparse split procedure (profit-only): Version 2 - computes mapping on-the-fly.
Returns profit only.
"""
function fast_split_sparse_profit(
    permutation::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    split_start_time = time()
    node_to_position = compute_node_to_position(permutation)
    result = fast_split_sparse_profit_with_mapping(permutation, node_to_position, pso)
    SPLIT_SPARSE_PROFIT_STATS[] = (calls=SPLIT_SPARSE_PROFIT_STATS[].calls + 1,
                                   time=SPLIT_SPARSE_PROFIT_STATS[].time + (time() - split_start_time))
    return result
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

"""
Core sparse split implementation (profit-only) with explicit node_to_position mapping.
Returns profit only.
"""
function fast_split_sparse_profit_with_mapping(
    permutation::Vector{Int},
    node_to_position::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    m = pso.n_drones
    
    if n == 0
        return 0.0
    end
    
    # Phase 1: Compute saturated tours (sparse)
    P_sparse, succ_sparse, _, sorted_depot_positions = 
        compute_saturated_tours_sparse(permutation, node_to_position, pso)
    
    k = length(sorted_depot_positions)
    if k == 0
        return 0.0
    end
    
    # Phase 2: Dynamic programming (sparse)
    Γ_sparse = sparse_dp_phase2(P_sparse, succ_sparse, sorted_depot_positions, m, n)
    
    return lookup_Γ_sparse(1, m, sorted_depot_positions, Γ_sparse)
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
        positions, expected_profits, greedy_routes_list = initialize_with_greedy_fallback_two(pso)
        greedy_compute_time = time() - greedy_start
        println("[SWARM INIT] Greedy solutions computed in $(round(greedy_compute_time, digits=3))s")
        
        # Replace the last two particles with the greedy solutions
        # IMPORTANT: Use profit computed by fast_split procedure, not from greedy route building.
        # This ensures consistency with how all other particles are evaluated.
        eval_start = time()
        best_greedy_profit = -Inf
        best_greedy_idx = 0
        
        for (idx, position) in enumerate(positions)
            particle_index = pso.swarm_size - (2 - idx)  # Last two particles
            
            # Compute profit using fast_split procedure (same as all other particles)
            # Also compute node_to_position mapping for debug code and particle storage
            node_to_pos = compute_node_to_position(position)
            fast_split_profit, fast_split_routes, _ = fast_split_sparse(position, pso)
            profit = fast_split_profit
            
            # Debug: Check position before evaluation
            # println("  [GREEDY-DEBUG] Particle $idx: position length = $(length(position)), first 10 = $(position[1:min(10, length(position))])")
            # println("  [GREEDY-DEBUG] Particle $idx: fast_split profit = $(round(profit, digits=6)) (USED)")
            # println("  [GREEDY-DEBUG] Particle $idx: expected profit from greedy = $(round(expected_profits[idx], digits=6)) (for comparison only)")
            
            # Show depot positions in the giant tour
            depot_positions = Int[]
            customer_positions = Int[]
            depot_nodes_seen = Set{Int}()
            for (pos_idx, node) in enumerate(position)
                if node > pso.n_pure_customers
                    push!(depot_positions, pos_idx)
                    push!(depot_nodes_seen, node)
                else
                    push!(customer_positions, pos_idx)
                end
            end
            # println("  [GREEDY-DEBUG] Giant tour structure: $(length(depot_positions)) depots at positions $(depot_positions[1:min(5, length(depot_positions))])$(length(depot_positions) > 5 ? "..." : ""), $(length(customer_positions)) customer positions")
            # println("  [GREEDY-DEBUG] Depot nodes in giant tour: $(sort(collect(depot_nodes_seen)))")
            # println("  [GREEDY-DEBUG] n_pure_customers = $(pso.n_pure_customers), so depot indices are > $(pso.n_pure_customers)")
            
            # Show what happens around depot boundaries
            if !isempty(depot_positions)
                for depot_pos in depot_positions[1:min(2, length(depot_positions))]
                    # println("  [GREEDY-DEBUG] Around depot at position $depot_pos:")
                    start_idx = max(1, depot_pos - 2)
                    end_idx = min(length(position), depot_pos + 5)
                    println("    Positions $start_idx-$end_idx: $(position[start_idx:end_idx])")
                end
            end
            
            # Additional debug: manually compute what saturated tours fast_split would see
            sorted_depot_positions = get_sorted_depot_positions(node_to_pos, pso.n_pure_customers)
            # println("  [GREEDY-DEBUG] Fast_split will see $(length(sorted_depot_positions)) depot positions: $(sorted_depot_positions[1:min(5, length(sorted_depot_positions))])$(length(sorted_depot_positions) > 5 ? "..." : "")")
            
            # Compute saturated tours manually to see what fast_split sees
            if !isempty(sorted_depot_positions)
                # println("  [GREEDY-DEBUG] Saturated tours that fast_split computes:")
                for (depot_idx, depot_pos) in enumerate(sorted_depot_positions[1:min(3, length(sorted_depot_positions))])
                    current_cost = 0.0
                    current_profit = 0.0
                    prev_node = position[depot_pos]
                    j = depot_pos + 1
                    customers_in_tour = Int[]
                    
                    while j <= length(position)
                        customer_idx = position[j]
                        travel_cost = lookup_cost(pso, prev_node, customer_idx, Float64(pso.max_battery_time * 4))
                        return_distance = pso.closest_depot_distance[customer_idx]
                        
                        current_cost += travel_cost
                        if current_cost + return_distance > pso.max_battery_time
                            break
                        end
                        
                        if customer_idx <= pso.n_pure_customers
                            push!(customers_in_tour, customer_idx)
                            current_profit += pso.profits[customer_idx]
                        end
                        prev_node = customer_idx
                        j += 1
                    end
                    
                    println("    Depot $depot_idx (pos $depot_pos): profit=$(round(current_profit, digits=6)), $(length(customers_in_tour)) customers, cost=$(round(current_cost, digits=2))")
                    if length(customers_in_tour) > 0
                        println("      First 10 customers: $(customers_in_tour[1:min(10, length(customers_in_tour))])")
                    end
                end
            end
            
            # Debug: Show what fast_split actually received and computed
            # println("  [GREEDY-DEBUG] Fast_split received giant tour of length $(length(position))")
            # println("  [GREEDY-DEBUG] Fast_split returned $(length(fast_split_routes)) routes")
            for (r_idx, route) in enumerate(fast_split_routes)
                customer_nodes = [node for node in route if node <= pso.n_pure_customers]
                println("    Route $r_idx: length=$(length(route)) ($(length(customer_nodes)) customers), nodes=$(route[1:min(10, length(route))])$(length(route) > 10 ? "..." : "")")
                if !isempty(customer_nodes)
                    println("      Customer nodes: $(customer_nodes[1:min(10, length(customer_nodes))])$(length(customer_nodes) > 10 ? "..." : "")")
                end
            end
            
            # Get greedy routes for this particle
            greedy_routes = idx <= length(greedy_routes_list) ? greedy_routes_list[idx] : Vector{Vector{Tuple{Int,Int}}}()
            ChargingStation = pso.depot_coord
            
            # Analyze why fast_split might be failing
            # println("  [GREEDY-DEBUG] === INVESTIGATING FAST_SPLIT BEHAVIOR ===")
            
            # Check if customers in greedy routes are adjacent in the giant tour
            all_greedy_customer_indices = Set{Int}()
            for greedy_route in greedy_routes
                for coord in greedy_route
                    if coord in ChargingStation
                        continue
                    end
                    for i in 1:pso.n_pure_customers
                        if i <= length(pso.customers) && pso.customers[i] == coord
                            if i in pso.accessible_customers
                                push!(all_greedy_customer_indices, i)
                            end
                            break
                        end
                    end
                end
            end
            
            # Find positions of greedy customers in the giant tour
            greedy_customer_positions = Dict{Int, Int}()  # customer -> position in giant tour
            for (pos_idx, node) in enumerate(position)
                if node in all_greedy_customer_indices
                    greedy_customer_positions[node] = pos_idx
                end
            end
            
            # println("  [GREEDY-DEBUG] Greedy customers found in giant tour: $(length(greedy_customer_positions)) out of $(length(all_greedy_customer_indices))")
            
            # Check adjacency: are consecutive greedy customers adjacent in the giant tour?
            # Also check if they're adjacent according to cost dictionary
            adjacency_issues = []
            for greedy_route in greedy_routes
                prev_customer_idx = nothing
                for coord in greedy_route
                    if coord in ChargingStation
                        continue
                    end
                    customer_idx = nothing
                    for i in 1:pso.n_pure_customers
                        if i <= length(pso.customers) && pso.customers[i] == coord
                            if i in pso.accessible_customers
                                customer_idx = i
                            end
                            break
                        end
                    end
                    if customer_idx === nothing
                        continue
                    end
                    
                    if prev_customer_idx !== nothing
                        # Check if they're adjacent in giant tour
                        pos_prev = get(greedy_customer_positions, prev_customer_idx, -1)
                        pos_curr = get(greedy_customer_positions, customer_idx, -1)
                        are_adjacent_in_tour = (pos_prev != -1 && pos_curr != -1 && abs(pos_prev - pos_curr) == 1)
                        
                        # Check if cost dictionary has entry
                        cost_key = (prev_customer_idx, customer_idx)
                        has_cost_entry = ENABLE_COST_MATRIX[] ? true : haskey(pso.costs, cost_key)
                        cost_value = lookup_cost(pso, prev_customer_idx, customer_idx, Float64(pso.max_battery_time * 4))
                        is_adjacent_by_cost = (cost_value <= 1.0)  # Adjacent means cost = 1.0
                        
                        if !are_adjacent_in_tour || !has_cost_entry || !is_adjacent_by_cost
                            push!(adjacency_issues, (
                                prev_customer_idx, customer_idx,
                                are_adjacent_in_tour, has_cost_entry, is_adjacent_by_cost,
                                pos_prev, pos_curr, cost_value
                            ))
                        end
                    end
                    prev_customer_idx = customer_idx
                end
            end
            
            if !isempty(adjacency_issues)
                # println("  [GREEDY-DEBUG] Found $(length(adjacency_issues)) adjacency issues:")
                for (i, issue) in enumerate(adjacency_issues[1:min(10, length(adjacency_issues))])
                    prev, curr, adj_tour, has_cost, adj_cost, pos_p, pos_c, cost_val = issue
                    println("    Issue $i: $prev -> $curr: adj_in_tour=$adj_tour, has_cost=$has_cost, adj_by_cost=$adj_cost, positions=($pos_p,$pos_c), cost=$cost_val")
                end
                if length(adjacency_issues) > 10
                    println("    ... and $(length(adjacency_issues) - 10) more issues")
                end
            else
                # println("  [GREEDY-DEBUG] All consecutive greedy customers are adjacent in tour and cost dict")
            end
            
            # Print detailed route comparison
            # println("  [GREEDY-DEBUG] === ROUTE COMPARISON FOR PARTICLE $idx ===")
            # println("  [GREEDY-DEBUG] === CONVERSION PROCESS ANALYSIS ===")
            # println("  [GREEDY-DEBUG] n_pure_customers = $(pso.n_pure_customers)")
            # println("  [GREEDY-DEBUG] Total customers in pso.customers = $(length(pso.customers))")
            # println("  [GREEDY-DEBUG] Depot indices range: $(pso.n_pure_customers + 1) to $(length(pso.customers))")
            # println("  [GREEDY-DEBUG] Depot nodes in giant tour: $(sort(collect(depot_nodes_seen)))")
            
            # println("  [GREEDY-DEBUG] Greedy routes ($(length(greedy_routes)) drones):")
            greedy_customers_by_drone = Vector{Set{Int}}()
            for (drone_idx, greedy_route) in enumerate(greedy_routes)
                println("    Drone $drone_idx greedy route (coordinates): length=$(length(greedy_route)), first 5 = $(greedy_route[1:min(5, length(greedy_route))])")
                
                # Convert greedy route coordinates to customer indices
            greedy_customers = Int[]
            coordinates_not_found = Vector{Tuple{Int,Int}}()
                for coord in greedy_route
                    if coord in ChargingStation
                        continue
                    end
                    found = false
                    for i in 1:pso.n_pure_customers
                        if i <= length(pso.customers) && pso.customers[i] == coord
                            if i in pso.accessible_customers
                                push!(greedy_customers, i)
                                found = true
                            end
                            break
                        end
                    end
                    if !found
                        push!(coordinates_not_found, coord)
                    end
                end
                
                if !isempty(coordinates_not_found)
                    println("      WARNING: $(length(coordinates_not_found)) coordinates not found in customer list: $(coordinates_not_found[1:min(5, length(coordinates_not_found))])$(length(coordinates_not_found) > 5 ? "..." : "")")
                end
                
                push!(greedy_customers_by_drone, Set(greedy_customers))
                route_str = isempty(greedy_customers) ? "[]" : "[$(join(greedy_customers, ", "))]"
                println("    Drone $drone_idx (customer indices): $route_str ($(length(greedy_customers)) customers)")
                
                # Check if these customers are in the giant tour and show their positions
                customer_positions_in_tour = Int[]
                for cust in greedy_customers[1:min(10, length(greedy_customers))]
                    for (pos_idx, node) in enumerate(position)
                        if node == cust
                            push!(customer_positions_in_tour, pos_idx)
                            break
                        end
                    end
                end
                if !isempty(customer_positions_in_tour)
                    println("      First $(min(10, length(greedy_customers))) customers at positions in giant tour: $customer_positions_in_tour")
                    # Check if they're consecutive
                    if length(customer_positions_in_tour) > 1
                        gaps = [customer_positions_in_tour[i+1] - customer_positions_in_tour[i] for i in 1:(length(customer_positions_in_tour)-1)]
                        non_consecutive = [g for g in gaps if g > 1]
                        if !isempty(non_consecutive)
                            println("      GAPS between consecutive customers: $non_consecutive (customers are NOT adjacent in giant tour!)")
                        end
                    end
                end
            end
            
            # println("  [GREEDY-DEBUG] Fast_split routes ($(length(fast_split_routes)) drones):")
            fast_split_customers_by_drone = Vector{Set{Int}}()
            for (drone_idx, fast_route) in enumerate(fast_split_routes)
                # Show the FULL route including depot nodes
                println("    Drone $drone_idx FULL route: $fast_route (length=$(length(fast_route)))")
                
                # Extract customer indices from fast_split route (skip depot nodes)
                fast_customers = [node for node in fast_route if node <= pso.n_pure_customers && node in pso.accessible_customers]
                depot_nodes = [node for node in fast_route if node > pso.n_pure_customers]
                push!(fast_split_customers_by_drone, Set(fast_customers))
                route_str = isempty(fast_customers) ? "[]" : "[$(join(fast_customers, ", "))]"
                println("    Drone $drone_idx customers only: $route_str ($(length(fast_customers)) customers)")
                if !isempty(depot_nodes)
                    println("    Drone $drone_idx depot nodes: $depot_nodes")
                end
            end
            
            # Compare and highlight differences
            # println("  [GREEDY-DEBUG] Differences:")
            all_greedy_customers = Set{Int}()
            for s in greedy_customers_by_drone
                union!(all_greedy_customers, s)
            end
            all_fast_split_customers = Set{Int}()
            for s in fast_split_customers_by_drone
                union!(all_fast_split_customers, s)
            end
            
            only_in_greedy = setdiff(all_greedy_customers, all_fast_split_customers)
            only_in_fast_split = setdiff(all_fast_split_customers, all_greedy_customers)
            in_both = intersect(all_greedy_customers, all_fast_split_customers)
            
            if !isempty(only_in_greedy)
                println("    Customers ONLY in greedy (missing from fast_split): [$(join(sort(collect(only_in_greedy)), ", "))] ($(length(only_in_greedy)) customers)")
                missing_profit = sum(pso.profits[c] for c in only_in_greedy)
                println("      Missing profit: $(round(missing_profit, digits=6))")
            end
            if !isempty(only_in_fast_split)
                println("    Customers ONLY in fast_split (not in greedy): [$(join(sort(collect(only_in_fast_split)), ", "))] ($(length(only_in_fast_split)) customers)")
            end
            if !isempty(in_both)
                in_both_sorted = sort(collect(in_both))
                display_count = min(20, length(in_both_sorted))
                println("    Customers in BOTH: [$(join(in_both_sorted[1:display_count], ", "))]$(length(in_both_sorted) > 20 ? " ... ($(length(in_both_sorted)) total)" : "")")
            end
            
            # Per-drone comparison
            max_drones = max(length(greedy_routes), length(fast_split_routes))
            for drone_idx in 1:max_drones
                greedy_set = drone_idx <= length(greedy_customers_by_drone) ? greedy_customers_by_drone[drone_idx] : Set{Int}()
                fast_set = drone_idx <= length(fast_split_customers_by_drone) ? fast_split_customers_by_drone[drone_idx] : Set{Int}()
                
                if greedy_set != fast_set
                    only_greedy = setdiff(greedy_set, fast_set)
                    only_fast = setdiff(fast_set, greedy_set)
                    if !isempty(only_greedy) || !isempty(only_fast)
                        println("    Drone $drone_idx differences:")
                        if !isempty(only_greedy)
                            println("      Only in greedy: [$(join(sort(collect(only_greedy)), ", "))]")
                        end
                        if !isempty(only_fast)
                            println("      Only in fast_split: [$(join(sort(collect(only_fast)), ", "))]")
                        end
                    end
                end
            end
            
            # println("  [GREEDY-DEBUG] ==========================================")
            
            # Compare fast_split profit with expected greedy profit
            if fast_split_profit < expected_profits[idx] * 0.5
                # println("  [GREEDY-DEBUG] Particle $idx: NOTE: fast_split gives lower profit than greedy route building.")
                println("                               This is expected because fast_split uses L-infinity costs,")
                println("                               while greedy routes were built with actual path distances through intermediate cells.")
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
        println("    - (Note: profit calculated by fast_split procedure, consistent with other particles)")
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
        flush(stdout)
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
                        # println("  [GREEDY-DEBUG] Test point $test_point found as customer $i with profit: $(test_point_profit)")
                    end
                end
            end
        end
        if !test_point_found
            # println("  [GREEDY-DEBUG] WARNING: Test point $test_point is NOT in the customer list!")
            # println("  [GREEDY-DEBUG] Checking if it would be accessible if it were a customer...")
            # Check if it would be accessible if it were a customer
            if test_point[1] > 0 && test_point[1] <= max_x && test_point[2] > 0 && test_point[2] <= max_y
                # println("  [GREEDY-DEBUG] Test point $test_point is within grid bounds ($max_x, $max_y)")
                # Check distance to charging stations (we'll get ChargingStation below)
            end
        end
        
        # Extract GridpointsDronesDetecting (pure customers only)
        GridpointsDronesDetecting = pso.customers[1:pso.n_pure_customers]
        
        # Extract ChargingStation from depot coordinates
        ChargingStation = pso.depot_coord
        
        # Check accessibility if point was found
        if test_point_found
            # println("  [GREEDY-DEBUG] Checking accessibility of test point $test_point...")
            for (idx, cs) in enumerate(ChargingStation)
                dist = max(abs(test_point[1] - cs[1]), abs(test_point[2] - cs[2]))
                return_dist = dist  # Same distance to return
                total_dist = dist + return_dist
                accessible = total_dist <= pso.max_battery_time
                # println("  [GREEDY-DEBUG] Distance from charging station $idx ($cs) to $test_point: $dist, round trip: $total_dist (battery limit: $(pso.max_battery_time), accessible: $accessible)")
            end
            # Check if it's in accessible_customers
            if test_point_customer_idx in pso.accessible_customers
                # println("  [GREEDY-DEBUG] Test point IS in accessible_customers list")
            else
                # println("  [GREEDY-DEBUG] WARNING: Test point is NOT in accessible_customers list!")
            end
        else
            # Check distance to charging stations even if not a customer
            for (idx, cs) in enumerate(ChargingStation)
                dist = max(abs(test_point[1] - cs[1]), abs(test_point[2] - cs[2]))
                return_dist = dist
                total_dist = dist + return_dist
                # println("  [GREEDY-DEBUG] Distance from charging station $idx ($cs) to $test_point: $dist, round trip: $total_dist (battery limit: $(pso.max_battery_time))")
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
            # println("  [GREEDY-DEBUG] n_pure_customers: $(pso.n_pure_customers), total customers: $(length(pso.customers))")
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
        
        # Return positions, expected profits, AND greedy routes (from greedy route building, not fast_split)
        expected_profits = [expected_profit_first, expected_profit_second]
        greedy_routes_list = [greedy_routes_first, greedy_routes_second]
        return final_positions, expected_profits, greedy_routes_list
        
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
        return fallback_positions, [0.0, 0.0], [Vector{Vector{Tuple{Int,Int}}}(), Vector{Vector{Tuple{Int,Int}}}()]
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
                    cost_iz = lookup_cost(pso, 0, customer, 0.0)  # depot to customer
                    cost_zj = lookup_cost(pso, customer, 0, 0.0)  # customer to depot
                    cost_ij = 0.0  # no direct connection to remove
                elseif pos == 1
                    # Insert at beginning
                    cost_iz = lookup_cost(pso, 0, customer, 0.0)  # depot to customer
                    cost_zj = lookup_cost(pso, customer, solution[1], 0.0)  # customer to next
                    cost_ij = lookup_cost(pso, 0, solution[1], 0.0)  # depot to next (to remove)
                elseif pos > length(solution)
                    # Insert at end
                    cost_iz = lookup_cost(pso, solution[end], customer, 0.0)  # prev to customer
                    cost_zj = lookup_cost(pso, customer, 0, 0.0)  # customer to depot
                    cost_ij = lookup_cost(pso, solution[end], 0, 0.0)  # prev to depot (to remove)
                else
                    # Insert in middle
                    cost_iz = lookup_cost(pso, solution[pos-1], customer, 0.0)  # prev to customer
                    cost_zj = lookup_cost(pso, customer, solution[pos], 0.0)  # customer to next
                    cost_ij = lookup_cost(pso, solution[pos-1], solution[pos], 0.0)  # prev to next (to remove)
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
    start_time = time()
    n = length(particle.position)
    positions = shuffle(1:n)
    # Precompute saturated tours for deadness checks
    _, _, tour_lengths_sparse, sorted_depot_positions = compute_saturated_tours_sparse(
        particle.position, particle.node_to_position, pso
    )
    dead_positions = compute_dead_positions(n, sorted_depot_positions, tour_lengths_sparse)

    # Live zone filter precomputation: safe dead positions (extended by +1) and dead block boundaries
    if ENABLE_LIVE_ZONE_FILTER[]
        safe_dead = compute_safe_dead_positions(n, sorted_depot_positions, tour_lengths_sparse)
        dbs, dbe = compute_dead_block_boundaries(safe_dead)
    end

    # Pre-allocate inner candidate buffer (avoids O(n) allocations per outer iteration)
    inner_j_buf = collect(1:n)
    buf_dirty = false
    inner_len = n

    for i in positions
        node_i = particle.position[i]
        is_depot = node_i > pso.n_pure_customers

        # Build inner candidate list — live zone filter restricts dead i to outside its dead block
        if ENABLE_LIVE_ZONE_FILTER[] && !is_depot && safe_dead[i]
            bs = dbs[i]; be = dbe[i]
            inner_len = 0
            for p in 1:bs-1; inner_len += 1; inner_j_buf[inner_len] = p; end
            for p in be+1:n; inner_len += 1; inner_j_buf[inner_len] = p; end
            shuffle!(view(inner_j_buf, 1:inner_len))
            buf_dirty = true
        else
            if buf_dirty
                for p in 1:n; inner_j_buf[p] = p; end
                buf_dirty = false
            end
            shuffle!(inner_j_buf)
            inner_len = n
        end

        for j_idx in 1:inner_len
            @inbounds j = inner_j_buf[j_idx]
            if i == j
                continue
            end
            SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates + 1,
                             skipped=SHIFT_STATS[].skipped,
                             time=SHIFT_STATS[].time,
                             calls=SHIFT_STATS[].calls)

            # === IRRELEVANCE-BASED SKIP (customers only) ===
            if ENABLE_SHIFT_IRRELEVANCE_FILTER[] && !is_depot
                # Check 1: Original blocking-based skip (works well with binary costs)
                is_blocking_or_dead = is_blocking(particle, i, pso) || dead_positions[i]
                irrelevant_removed = false
                if is_blocking_or_dead && is_blocking_once_removed(particle, i, pso)
                    irrelevant_removed = true
                end

                if irrelevant_removed && j > 1 && dead_positions[j - 1]
                    SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                                     skipped=SHIFT_STATS[].skipped + 1,
                                     time=SHIFT_STATS[].time,
                                     calls=SHIFT_STATS[].calls)
                    continue
                end
                
                # Check 2: Dead-zone skip (works with both binary and L-infinity costs)
                # If both source and target positions are in dead zones, the move cannot affect profit
                if dead_positions[i] && j > 1 && dead_positions[j - 1]
                    SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                                     skipped=SHIFT_STATS[].skipped + 1,
                                     time=SHIFT_STATS[].time,
                                     calls=SHIFT_STATS[].calls)
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
                SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                                 skipped=SHIFT_STATS[].skipped,
                                 time=SHIFT_STATS[].time + (time() - start_time),
                                 calls=SHIFT_STATS[].calls + 1)
                return true, new_tour_intervals
            end
        end
    end

    SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                     skipped=SHIFT_STATS[].skipped,
                     time=SHIFT_STATS[].time + (time() - start_time),
                     calls=SHIFT_STATS[].calls + 1)
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
    start_time = time()
    n = length(particle.position)
    positions = shuffle(1:n)
    pos = particle.position

    # Precompute depot positions / tour lengths for live-zone or lazy-dead filter
    local tl_sparse_lz::Vector{Int}, sdp_lz::Vector{Int}
    if ENABLE_LIVE_ZONE_FILTER[] || ENABLE_LAZY_DEAD_FILTER[]
        _, _, tl_sparse_lz, sdp_lz = compute_saturated_tours_sparse(
            particle.position, particle.node_to_position, pso
        )
    end

    # Live zone filter: precompute safe dead positions (extended by +1) and sorted live positions
    if ENABLE_LIVE_ZONE_FILTER[]
        lz_dead = compute_safe_dead_positions(n, sdp_lz, tl_sparse_lz)
        lz_live_sorted = sort([p for p in 1:n if !lz_dead[p]])
    end

    # Pre-allocate inner candidate buffer (avoids O(n) allocations per outer iteration)
    swap_j_buf = Vector{Int}(undef, n)
    inner_len = 0

    for i in positions
        node_i = pos[i]
        is_depot_i = node_i > pso.n_pure_customers

        # Lazy dead filter: O(k) check for outer position (cached per outer iteration)
        i_safe_dead = ENABLE_LAZY_DEAD_FILTER[] && !is_depot_i && is_position_safe_dead(i, sdp_lz, tl_sparse_lz)

        # Build inner candidate list — when i is dead customer, only pair with live j
        if ENABLE_LIVE_ZONE_FILTER[] && !is_depot_i && lz_dead[i]
            lo = searchsortedfirst(lz_live_sorted, i + 1)
            lo > length(lz_live_sorted) && continue
            inner_len = length(lz_live_sorted) - lo + 1
            for k in 1:inner_len; swap_j_buf[k] = lz_live_sorted[lo + k - 1]; end
            shuffle!(view(swap_j_buf, 1:inner_len))
        else
            inner_len = n - i
            for k in 1:inner_len; swap_j_buf[k] = i + k; end
            shuffle!(view(swap_j_buf, 1:inner_len))
        end
        
        for j_idx in 1:inner_len
            @inbounds j = swap_j_buf[j_idx]
            node_j = pos[j]
            is_depot_j = node_j > pso.n_pure_customers
            
            # Increment total swap attempts
            counters[] = (counters[][1] + 1, counters[][2])
            SWAP_STATS[] = (candidates=SWAP_STATS[].candidates + 1,
                            skipped=SWAP_STATS[].skipped,
                            time=SWAP_STATS[].time,
                            calls=SWAP_STATS[].calls)

            # === LAZY DEAD FILTER (swap): skip dead-dead pairs, O(k) per check ===
            if i_safe_dead && !is_depot_j && is_position_safe_dead(j, sdp_lz, tl_sparse_lz)
                counters[] = (counters[][1], counters[][2] + 1)
                SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                                skipped=SWAP_STATS[].skipped + 1,
                                time=SWAP_STATS[].time,
                                calls=SWAP_STATS[].calls)
                continue
            end
            
            # === BLOCKING CHECK (swap) ===
            if ENABLE_SWAP_BLOCKING_FILTER[] && !is_depot_i && !is_depot_j
                if is_blocking_once_inserted(particle, i, j, pso) &&
                   is_blocking_once_removed(particle, i, pso) &&
                   is_blocking_once_inserted(particle, j, i, pso) &&
                   is_blocking_once_removed(particle, j, pso)
                    SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                                    skipped=SWAP_STATS[].skipped + 1,
                                    time=SWAP_STATS[].time,
                                    calls=SWAP_STATS[].calls)
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
                SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                                skipped=SWAP_STATS[].skipped,
                                time=SWAP_STATS[].time + (time() - start_time),
                                calls=SWAP_STATS[].calls + 1)
                return true, new_tour_intervals
            else
                # Revert swap
                pos[i], pos[j] = pos[j], pos[i]
            end
        end
    end

    SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                    skipped=SWAP_STATS[].skipped,
                    time=SWAP_STATS[].time + (time() - start_time),
                    calls=SWAP_STATS[].calls + 1)
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
    # Track shift stats deltas for this local search
    shift_start_candidates = SHIFT_STATS[].candidates
    shift_start_skipped = SHIFT_STATS[].skipped
    
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
        avg_swap_time = total_time_swap / max(total_swaps, 1)
        println("[BOUNDARY-OPT] Swap operations: $total_swaps total, $skipped_swaps skipped ($(round(skip_percentage, digits=2))%), avg time=$(round(avg_swap_time, digits=6))s")
    end
    
    # Print shift boundary optimization statistics
    shift_total = SHIFT_STATS[].candidates - shift_start_candidates
    shift_skipped = SHIFT_STATS[].skipped - shift_start_skipped
    if shift_total > 0
        skip_percentage = 100.0 * shift_skipped / shift_total
        avg_shift_time = total_time_shift / max(shift_total, 1)
        println("[BOUNDARY-OPT] Shift operations: $shift_total total, $shift_skipped skipped ($(round(skip_percentage, digits=2))%), avg time=$(round(avg_shift_time, digits=6))s")
    end
    
    return total_time_shift, total_time_swap, total_time_destruction_repair
end

# ============================================================================
# END SPARSE OPERATORS
# ============================================================================

# ============================================================================
# INCREMENTAL TOUR UPDATE FOR SWAP (avoids full split recomputation)
# ============================================================================

# Stats for incremental swap
const INCREMENTAL_SWAP_STATS = Ref((candidates=0, skipped_blocking=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0))

function reset_incremental_swap_stats!()
    INCREMENTAL_SWAP_STATS[] = (candidates=0, skipped_blocking=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
end

"""
Tour cache: stores Phase 1 results for incremental updates.
sorted_depot_positions is invariant during swap (depots are never swapped).
"""
mutable struct TourCache
    sorted_depot_positions::Vector{Int}   # Invariant during swap
    P_sparse::Vector{Float64}             # Profit per depot tour
    succ_sparse::Vector{Int}              # Successor position per depot tour
    tour_lengths_sparse::Vector{Int}      # Length per depot tour
end

"""
Initialize a TourCache from a particle by running full Phase 1.
"""
function init_tour_cache(particle::Particle, pso::PSOiA_TOP_multiple_depots)::TourCache
    P, succ, lens, depots = compute_saturated_tours_sparse(
        particle.position, particle.node_to_position, pso
    )
    return TourCache(depots, copy(P), copy(succ), copy(lens))
end

"""
Find depot indices whose tours are affected by swap(i, j).

A tour t covering [d_t, d_t + len_t - 1] with successor at d_t + len_t
is affected if position i or j falls in the influence range (d_t, d_t + len_t].

Why this range:
  - (d_t, d_t + len_t - 1]: positions inside the tour (excluding depot — depots aren't swapped).
    Swapping a node here changes intermediate costs → tour may shrink or its profit changes.
  - d_t + len_t (successor position): the node here was too expensive to include.
    If swapped with a cheaper node, the tour might extend.
  - d_t itself: depot position, not swapped, so excluded.
"""
function find_affected_tour_indices(
    cache::TourCache,
    i::Int,
    j::Int
)::Vector{Int}
    affected = Int[]
    k = length(cache.sorted_depot_positions)
    for t in 1:k
        d_t = cache.sorted_depot_positions[t]
        len_t = cache.tour_lengths_sparse[t]
        # Influence range: (d_t, d_t + len_t] — i.e. d_t < pos ≤ d_t + len_t
        if (d_t < i <= d_t + len_t) || (d_t < j <= d_t + len_t)
            push!(affected, t)
        end
    end
    return affected
end

"""
Recompute a single saturated tour in the cache (same greedy logic as Phase 1).
"""
function recompute_single_tour!(
    cache::TourCache,
    t::Int,
    permutation::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    L = pso.max_battery_time
    depot_pos = cache.sorted_depot_positions[t]

    current_cost = 0.0
    current_profit = 0.0
    prev_customer = permutation[depot_pos]  # The depot node
    j = depot_pos + 1

    while j <= n
        customer_idx = permutation[j]
        travel_cost = lookup_cost(pso, prev_customer, customer_idx, L * 4)
        return_distance = pso.closest_depot_distance[customer_idx]
        current_cost += travel_cost

        if current_cost + return_distance > L
            break
        end

        current_profit += pso.profits[customer_idx]
        prev_customer = customer_idx
        j += 1
    end

    cache.P_sparse[t] = current_profit
    cache.tour_lengths_sparse[t] = j - depot_pos
    cache.succ_sparse[t] = (j <= n) ? j : 0
end

"""
Swap operator with incremental tour updates.
Instead of calling fast_split_sparse for every trial, we:
  1. Maintain a TourCache across evaluations
  2. For each trial swap, identify affected tours (O(k))
  3. Recompute only affected tours (O(L_affected) instead of O(L))
  4. Skip the DP entirely if no cached tour actually changed
  5. Rerun DP only when needed (O(k·m·log k), always cheap)
  6. Skip Phase 3 (backtracking) for rejected moves

Returns (improved::Bool, new_tour_intervals::TourIntervals, updated_cache::TourCache).
"""
function swap_operator_incremental!(
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots,
    tour_intervals::TourIntervals,
    cache::TourCache,
    counters::Ref{Tuple{Int, Int}}  # (total_swaps, skipped_swaps)
)
    start_time = time()
    n = length(particle.position)
    m = pso.n_drones
    positions = shuffle(1:n)
    pos = particle.position

    # Live zone filter: precompute safe dead positions (extended by +1) and sorted live positions from cache
    if ENABLE_LIVE_ZONE_FILTER[]
        lz_dead = compute_safe_dead_positions(n, cache.sorted_depot_positions, cache.tour_lengths_sparse)
        lz_live_sorted = sort([p for p in 1:n if !lz_dead[p]])
    end

    # Pre-allocate inner candidate buffer (avoids O(n) allocations per outer iteration)
    swap_j_buf = Vector{Int}(undef, n)
    inner_len = 0

    for i in positions
        node_i = pos[i]
        is_depot_i = node_i > pso.n_pure_customers

        # Lazy dead filter: O(k) check for outer position (cached per outer iteration)
        i_safe_dead = ENABLE_LAZY_DEAD_FILTER[] && !is_depot_i && is_position_safe_dead(i, cache.sorted_depot_positions, cache.tour_lengths_sparse)

        # Build inner candidate list — when i is dead customer, only pair with live j
        if ENABLE_LIVE_ZONE_FILTER[] && !is_depot_i && lz_dead[i]
            lo = searchsortedfirst(lz_live_sorted, i + 1)
            lo > length(lz_live_sorted) && continue
            inner_len = length(lz_live_sorted) - lo + 1
            for k in 1:inner_len; swap_j_buf[k] = lz_live_sorted[lo + k - 1]; end
            shuffle!(view(swap_j_buf, 1:inner_len))
        else
            inner_len = n - i
            for k in 1:inner_len; swap_j_buf[k] = i + k; end
            shuffle!(view(swap_j_buf, 1:inner_len))
        end

        for j_idx in 1:inner_len
            @inbounds j = swap_j_buf[j_idx]
            node_j = pos[j]
            is_depot_j = node_j > pso.n_pure_customers

            # Increment total swap attempts
            counters[] = (counters[][1] + 1, counters[][2])
            INCREMENTAL_SWAP_STATS[] = (
                candidates=INCREMENTAL_SWAP_STATS[].candidates + 1,
                skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
                skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
                accepted=INCREMENTAL_SWAP_STATS[].accepted,
                time=INCREMENTAL_SWAP_STATS[].time,
                calls=INCREMENTAL_SWAP_STATS[].calls
            )

            # === LAZY DEAD FILTER (swap): skip dead-dead pairs, O(k) per check ===
            if i_safe_dead && !is_depot_j && is_position_safe_dead(j, cache.sorted_depot_positions, cache.tour_lengths_sparse)
                counters[] = (counters[][1], counters[][2] + 1)
                INCREMENTAL_SWAP_STATS[] = (
                    candidates=INCREMENTAL_SWAP_STATS[].candidates,
                    skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking + 1,
                    skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                    evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
                    accepted=INCREMENTAL_SWAP_STATS[].accepted,
                    time=INCREMENTAL_SWAP_STATS[].time,
                    calls=INCREMENTAL_SWAP_STATS[].calls
                )
                continue
            end

            # === TIER 3: Blocking filter (unchanged from original) ===
            if ENABLE_SWAP_BLOCKING_FILTER[] && !is_depot_i && !is_depot_j
                if is_blocking_once_inserted(particle, i, j, pso) &&
                   is_blocking_once_removed(particle, i, pso) &&
                   is_blocking_once_inserted(particle, j, i, pso) &&
                   is_blocking_once_removed(particle, j, pso)
                    INCREMENTAL_SWAP_STATS[] = (
                        candidates=INCREMENTAL_SWAP_STATS[].candidates,
                        skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking + 1,
                        skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                        evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
                        accepted=INCREMENTAL_SWAP_STATS[].accepted,
                        time=INCREMENTAL_SWAP_STATS[].time,
                        calls=INCREMENTAL_SWAP_STATS[].calls
                    )
                    continue
                end
            end

            # === INCREMENTAL EVALUATION ===
            # Step A: Perform trial swap
            pos[i], pos[j] = pos[j], pos[i]

            # Step A': If a depot is involved, the depot position changes and
            # the cache's sorted_depot_positions would be invalid. Fall back to
            # full split evaluation (rare: only k/n fraction of pairs).
            if is_depot_i || is_depot_j
                new_profit, _, new_tour_intervals = fast_split_sparse(pos, pso)
                INCREMENTAL_SWAP_STATS[] = (
                    candidates=INCREMENTAL_SWAP_STATS[].candidates,
                    skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
                    skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                    evaluated=INCREMENTAL_SWAP_STATS[].evaluated + 1,
                    accepted=INCREMENTAL_SWAP_STATS[].accepted,
                    time=INCREMENTAL_SWAP_STATS[].time,
                    calls=INCREMENTAL_SWAP_STATS[].calls
                )
                if new_profit > particle.current_profit
                    particle.current_profit = new_profit
                    particle.node_to_position[pos[i]] = i
                    particle.node_to_position[pos[j]] = j
                    # Rebuild full cache for new depot layout
                    cache = init_tour_cache(particle, pso)
                    INCREMENTAL_SWAP_STATS[] = (
                        candidates=INCREMENTAL_SWAP_STATS[].candidates,
                        skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
                        skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                        evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
                        accepted=INCREMENTAL_SWAP_STATS[].accepted + 1,
                        time=INCREMENTAL_SWAP_STATS[].time + (time() - start_time),
                        calls=INCREMENTAL_SWAP_STATS[].calls + 1
                    )
                    return true, new_tour_intervals, cache
                else
                    pos[i], pos[j] = pos[j], pos[i]
                end
                continue
            end

            # Step B: Find affected tours (customer-customer swap only)
            affected = find_affected_tour_indices(cache, i, j)

            # Step C: Save old values for potential revert
            n_affected = length(affected)
            old_P = Vector{Float64}(undef, n_affected)
            old_len = Vector{Int}(undef, n_affected)
            old_succ = Vector{Int}(undef, n_affected)
            for (idx, t) in enumerate(affected)
                old_P[idx] = cache.P_sparse[t]
                old_len[idx] = cache.tour_lengths_sparse[t]
                old_succ[idx] = cache.succ_sparse[t]
            end

            # Step D: Recompute only affected tours
            for t in affected
                recompute_single_tour!(cache, t, pos, pso)
            end

            # Step E: Check if any cached tour actually changed (DP skip optimization)
            cache_changed = false
            for (idx, t) in enumerate(affected)
                if cache.P_sparse[t] != old_P[idx] ||
                   cache.succ_sparse[t] != old_succ[idx] ||
                   cache.tour_lengths_sparse[t] != old_len[idx]
                    cache_changed = true
                    break
                end
            end

            if !cache_changed
                # No tour changed → profit is identical, skip DP
                # Revert swap
                pos[i], pos[j] = pos[j], pos[i]
                # No need to revert cache (values unchanged)
                INCREMENTAL_SWAP_STATS[] = (
                    candidates=INCREMENTAL_SWAP_STATS[].candidates,
                    skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
                    skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp + 1,
                    evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
                    accepted=INCREMENTAL_SWAP_STATS[].accepted,
                    time=INCREMENTAL_SWAP_STATS[].time,
                    calls=INCREMENTAL_SWAP_STATS[].calls
                )
                continue
            end

            # Step F: Rerun DP (always cheap: O(k·m·log k))
            Γ_sparse = sparse_dp_phase2(
                cache.P_sparse, cache.succ_sparse,
                cache.sorted_depot_positions, m, n
            )
            new_profit = lookup_Γ_sparse(
                1, m, cache.sorted_depot_positions, Γ_sparse
            )

            INCREMENTAL_SWAP_STATS[] = (
                candidates=INCREMENTAL_SWAP_STATS[].candidates,
                skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
                skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                evaluated=INCREMENTAL_SWAP_STATS[].evaluated + 1,
                accepted=INCREMENTAL_SWAP_STATS[].accepted,
                time=INCREMENTAL_SWAP_STATS[].time,
                calls=INCREMENTAL_SWAP_STATS[].calls
            )

            # Step G: Accept or reject
            if new_profit > particle.current_profit
                # ACCEPT — finalize
                particle.current_profit = new_profit
                node_at_i = pos[i]  # After swap: this is the old node_j
                node_at_j = pos[j]  # After swap: this is the old node_i
                particle.node_to_position[node_at_i] = i
                particle.node_to_position[node_at_j] = j
                # Rebuild tour intervals (O(k))
                new_ti = build_tour_intervals(
                    cache.sorted_depot_positions, cache.tour_lengths_sparse
                )
                INCREMENTAL_SWAP_STATS[] = (
                    candidates=INCREMENTAL_SWAP_STATS[].candidates,
                    skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
                    skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
                    evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
                    accepted=INCREMENTAL_SWAP_STATS[].accepted + 1,
                    time=INCREMENTAL_SWAP_STATS[].time + (time() - start_time),
                    calls=INCREMENTAL_SWAP_STATS[].calls + 1
                )
                return true, new_ti, cache
            else
                # REJECT — revert swap and cache
                pos[i], pos[j] = pos[j], pos[i]
                for (idx, t) in enumerate(affected)
                    cache.P_sparse[t] = old_P[idx]
                    cache.tour_lengths_sparse[t] = old_len[idx]
                    cache.succ_sparse[t] = old_succ[idx]
                end
            end
        end
    end

    INCREMENTAL_SWAP_STATS[] = (
        candidates=INCREMENTAL_SWAP_STATS[].candidates,
        skipped_blocking=INCREMENTAL_SWAP_STATS[].skipped_blocking,
        skipped_dp=INCREMENTAL_SWAP_STATS[].skipped_dp,
        evaluated=INCREMENTAL_SWAP_STATS[].evaluated,
        accepted=INCREMENTAL_SWAP_STATS[].accepted,
        time=INCREMENTAL_SWAP_STATS[].time + (time() - start_time),
        calls=INCREMENTAL_SWAP_STATS[].calls + 1
    )
    return false, tour_intervals, cache
end

"""
Local search using sparse operators with incremental swap updates.
Uses the original shift_operator_sparse! but replaces swap with swap_operator_incremental!.
Returns timing info (time_shift, time_swap, time_destruction_repair) for compatibility.
"""
function local_search_incremental_swap!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    total_time_shift = 0.0
    total_time_swap = 0.0
    total_time_destruction_repair = 0.0

    swap_counters = Ref((0, 0))
    shift_start_candidates = SHIFT_STATS[].candidates
    shift_start_skipped = SHIFT_STATS[].skipped

    # Initial split to get tour intervals AND initialize cache
    _, _, tour_intervals = fast_split_sparse(particle.position, particle, pso)
    cache = init_tour_cache(particle, pso)

    improved = true
    while improved
        improved = false
        neighborhoods = shuffle([1, 2])

        for neighborhood in neighborhoods
            if neighborhood == 1
                time_before = time()
                improved, tour_intervals = shift_operator_sparse!(particle, pso, tour_intervals)
                total_time_shift += time() - time_before
                if improved
                    # Shift changed the permutation and depot positions may have moved;
                    # rebuild cache from scratch after shift acceptance
                    cache = init_tour_cache(particle, pso)
                end
            else
                time_before = time()
                improved, tour_intervals, cache = swap_operator_incremental!(
                    particle, pso, tour_intervals, cache, swap_counters
                )
                total_time_swap += time() - time_before
            end

            if improved
                break  # Restart from first neighborhood
            end
        end
    end

    total_swaps, skipped_swaps = swap_counters[]
    if total_swaps > 0
        skip_percentage = 100.0 * skipped_swaps / total_swaps
        avg_swap_time = total_time_swap / max(total_swaps, 1)
        println("[INCREMENTAL-SWAP] Swap operations: $total_swaps total, $skipped_swaps skipped ($(round(skip_percentage, digits=2))%), avg time=$(round(avg_swap_time, digits=6))s")
    end

    shift_total = SHIFT_STATS[].candidates - shift_start_candidates
    shift_skipped = SHIFT_STATS[].skipped - shift_start_skipped
    if shift_total > 0
        skip_percentage = 100.0 * shift_skipped / shift_total
        avg_shift_time = total_time_shift / max(shift_total, 1)
        println("[INCREMENTAL-SWAP] Shift operations: $shift_total total, $shift_skipped skipped ($(round(skip_percentage, digits=2))%), avg time=$(round(avg_shift_time, digits=6))s")
    end

    return total_time_shift, total_time_swap, total_time_destruction_repair
end

# ============================================================================
# END INCREMENTAL TOUR UPDATE FOR SWAP
# ============================================================================

# ============================================================================
# INCREMENTAL TOUR UPDATE FOR SHIFT
# ============================================================================

# Stats for incremental shift
const INCREMENTAL_SHIFT_STATS = Ref((candidates=0, skipped_filter=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0))

function reset_incremental_shift_stats!()
    INCREMENTAL_SHIFT_STATS[] = (candidates=0, skipped_filter=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
end

"""
Perform shift(i, j) in-place on pos.
Semantics match move_element: remove element at i, insert at target position.
  i < j: element goes to j-1; positions (i, j) shift left by 1.
  i > j: element goes to j; positions [j, i) shift right by 1.
"""
function shift_in_place!(pos::Vector{Int}, i::Int, j::Int)
    if i == j; return; end
    saved = pos[i]
    if i < j
        for p in i:j-2
            pos[p] = pos[p+1]
        end
        pos[j-1] = saved
    else  # i > j
        for p in i:-1:j+1
            pos[p] = pos[p-1]
        end
        pos[j] = saved
    end
end

"""
Revert a shift(i, j) that was applied in-place.
"""
function revert_shift_in_place!(pos::Vector{Int}, i::Int, j::Int)
    if i == j; return; end
    if i < j
        # Element is at j-1, put it back at i
        saved = pos[j-1]
        for p in j-1:-1:i+1
            pos[p] = pos[p-1]
        end
        pos[i] = saved
    else  # i > j
        # Element is at j, put it back at i
        saved = pos[j]
        for p in j:i-1
            pos[p] = pos[p+1]
        end
        pos[i] = saved
    end
end

"""
Compute new depot positions after shift(i, j).
Returns a new vector of shifted depot positions (NOT sorted).
"""
function compute_shifted_depot_positions(
    old_sorted_depot_positions::Vector{Int},
    i::Int,
    j::Int
)::Vector{Int}
    k = length(old_sorted_depot_positions)
    new_positions = Vector{Int}(undef, k)

    for t in 1:k
        d = old_sorted_depot_positions[t]
        if i < j
            if d == i
                new_positions[t] = j - 1
            elseif d > i && d < j
                new_positions[t] = d - 1
            else
                new_positions[t] = d
            end
        else  # i > j
            if d == i
                new_positions[t] = j
            elseif d >= j && d < i
                new_positions[t] = d + 1
            else
                new_positions[t] = d
            end
        end
    end

    return new_positions
end

"""
Compute a single saturated tour at a given depot position in the permutation.
Returns (profit, tour_length, succ).
"""
function compute_tour_at(
    depot_pos::Int,
    permutation::Vector{Int},
    pso::PSOiA_TOP_multiple_depots
)
    n = length(permutation)
    L = pso.max_battery_time

    current_cost = 0.0
    current_profit = 0.0
    prev_customer = permutation[depot_pos]
    jj = depot_pos + 1

    while jj <= n
        customer_idx = permutation[jj]
        travel_cost = lookup_cost(pso, prev_customer, customer_idx, L * 4)
        return_distance = pso.closest_depot_distance[customer_idx]
        current_cost += travel_cost
        if current_cost + return_distance > L
            break
        end
        current_profit += pso.profits[customer_idx]
        prev_customer = customer_idx
        jj += 1
    end

    tour_length = jj - depot_pos
    succ = (jj <= n) ? jj : 0
    return current_profit, tour_length, succ
end

"""
Shift operator with incremental tour updates.
Instead of calling move_element + fast_split_sparse per trial, we:
  1. Shift in-place (O(|i-j|), no allocation)
  2. Compute new depot positions (O(k))
  3. Find affected tours: those whose range overlaps the change range
  4. Recompute only affected tours
  5. Skip DP when nothing changed
  6. Revert in-place on rejection

Returns (improved::Bool, new_tour_intervals::TourIntervals, updated_cache::TourCache).
"""
function shift_operator_incremental!(
    particle::Particle,
    pso::PSOiA_TOP_multiple_depots,
    cache::TourCache,
    tour_intervals::TourIntervals
)
    start_time = time()
    n = length(particle.position)
    m = pso.n_drones
    k = length(cache.sorted_depot_positions)
    positions = shuffle(1:n)
    pos = particle.position

    # Precompute dead positions from cache
    dead_positions = compute_dead_positions(n, cache.sorted_depot_positions, cache.tour_lengths_sparse)

    # Live zone filter precomputation: safe dead positions (extended by +1) and dead block boundaries
    if ENABLE_LIVE_ZONE_FILTER[]
        safe_dead = compute_safe_dead_positions(n, cache.sorted_depot_positions, cache.tour_lengths_sparse)
        dbs, dbe = compute_dead_block_boundaries(safe_dead)
    end

    # Pre-allocate inner candidate buffer (avoids O(n) allocations per outer iteration)
    inner_j_buf = collect(1:n)
    buf_dirty = false
    inner_len = n

    for i in positions
        node_i = pos[i]
        is_depot = node_i > pso.n_pure_customers

        # Build inner candidate list — live zone filter restricts dead i to outside its dead block
        if ENABLE_LIVE_ZONE_FILTER[] && !is_depot && safe_dead[i]
            bs = dbs[i]; be = dbe[i]
            inner_len = 0
            for p in 1:bs-1; inner_len += 1; inner_j_buf[inner_len] = p; end
            for p in be+1:n; inner_len += 1; inner_j_buf[inner_len] = p; end
            shuffle!(view(inner_j_buf, 1:inner_len))
            buf_dirty = true
        else
            if buf_dirty
                for p in 1:n; inner_j_buf[p] = p; end
                buf_dirty = false
            end
            shuffle!(inner_j_buf)
            inner_len = n
        end

        for j_idx in 1:inner_len
            @inbounds j = inner_j_buf[j_idx]
            if i == j
                continue
            end

            INCREMENTAL_SHIFT_STATS[] = (
                candidates=INCREMENTAL_SHIFT_STATS[].candidates + 1,
                skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter,
                skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp,
                evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
                accepted=INCREMENTAL_SHIFT_STATS[].accepted,
                time=INCREMENTAL_SHIFT_STATS[].time,
                calls=INCREMENTAL_SHIFT_STATS[].calls
            )

            # === IRRELEVANCE-BASED SKIP (customers only) — same as original ===
            if ENABLE_SHIFT_IRRELEVANCE_FILTER[] && !is_depot
                is_blocking_or_dead = is_blocking(particle, i, pso) || dead_positions[i]
                irrelevant_removed = false
                if is_blocking_or_dead && is_blocking_once_removed(particle, i, pso)
                    irrelevant_removed = true
                end

                if irrelevant_removed && j > 1 && dead_positions[j - 1]
                    INCREMENTAL_SHIFT_STATS[] = (
                        candidates=INCREMENTAL_SHIFT_STATS[].candidates,
                        skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter + 1,
                        skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp,
                        evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
                        accepted=INCREMENTAL_SHIFT_STATS[].accepted,
                        time=INCREMENTAL_SHIFT_STATS[].time,
                        calls=INCREMENTAL_SHIFT_STATS[].calls
                    )
                    continue
                end

                if dead_positions[i] && j > 1 && dead_positions[j - 1]
                    INCREMENTAL_SHIFT_STATS[] = (
                        candidates=INCREMENTAL_SHIFT_STATS[].candidates,
                        skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter + 1,
                        skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp,
                        evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
                        accepted=INCREMENTAL_SHIFT_STATS[].accepted,
                        time=INCREMENTAL_SHIFT_STATS[].time,
                        calls=INCREMENTAL_SHIFT_STATS[].calls
                    )
                    continue
                end
            end

            # === INCREMENTAL EVALUATION ===

            # Step A: Find affected tours using breakpoint check.
            #
            # A shift(i,j) (move_element semantics) only breaks two "connections"
            # in the node sequence:
            #   1. Removal at position i: the edge from perm[i-1] to perm[i] breaks
            #   2. Insertion at position j-1 (i<j) or j (i>j): a new node appears
            # All intermediate node-pair relationships are preserved (just slid).
            #
            # In OLD-permutation coordinates, a tour at depot d with succ at d+len
            # is affected iff:
            #   - i ∈ [d, d+len]  (removal breakpoint; d<= to catch depot at i)
            #   - j ∈ (d, d+len]  (insertion breakpoint; the boundary of changed zone)

            bp1 = i
            bp2 = j

            any_affected = false
            affected_mask = falses(k)
            for t in 1:k
                d = cache.sorted_depot_positions[t]
                succ_pos = d + cache.tour_lengths_sparse[t]  # d + len
                # bp1 uses ≤ (not <) because d = bp1 = i means the depot itself
                # is being removed/relocated — always affected.
                # bp2 uses < because a depot at bp2 is just outside the insertion
                # point and its tour sequence is preserved.
                if (d <= bp1 <= succ_pos) || (d < bp2 <= succ_pos)
                    affected_mask[t] = true
                    any_affected = true
                end
            end

            # Step B: Early exit — no tours affected → profit unchanged
            if !any_affected
                INCREMENTAL_SHIFT_STATS[] = (
                    candidates=INCREMENTAL_SHIFT_STATS[].candidates,
                    skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter,
                    skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp + 1,
                    evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
                    accepted=INCREMENTAL_SHIFT_STATS[].accepted,
                    time=INCREMENTAL_SHIFT_STATS[].time,
                    calls=INCREMENTAL_SHIFT_STATS[].calls
                )
                continue
            end

            # Step C: Compute new depot positions
            new_depot_positions_unsorted = compute_shifted_depot_positions(
                cache.sorted_depot_positions, i, j
            )
            sorted_perm = sortperm(new_depot_positions_unsorted)
            new_sorted_depots = new_depot_positions_unsorted[sorted_perm]

            # Step D: Perform shift in-place
            shift_in_place!(pos, i, j)

            # Step E: Build new P/len/succ arrays
            new_P = Vector{Float64}(undef, k)
            new_len = Vector{Int}(undef, k)
            new_succ = Vector{Int}(undef, k)

            for new_idx in 1:k
                old_idx = sorted_perm[new_idx]
                if affected_mask[old_idx]
                    # Recompute on shifted permutation
                    depot_pos = new_sorted_depots[new_idx]
                    p, l, s = compute_tour_at(depot_pos, pos, pso)
                    new_P[new_idx] = p
                    new_len[new_idx] = l
                    new_succ[new_idx] = s
                else
                    # Unaffected: P and len are unchanged (same node sequence).
                    # But succ is an ABSOLUTE position that shifts with the depot.
                    new_P[new_idx] = cache.P_sparse[old_idx]
                    old_len = cache.tour_lengths_sparse[old_idx]
                    new_len[new_idx] = old_len
                    succ_val = new_sorted_depots[new_idx] + old_len
                    new_succ[new_idx] = (succ_val <= n) ? succ_val : 0
                end
            end

            # Step F: Check if DP-relevant state changed
            cache_changed = false
            if new_sorted_depots != cache.sorted_depot_positions
                cache_changed = true
            else
                for t in 1:k
                    if new_P[t] != cache.P_sparse[t] ||
                       new_len[t] != cache.tour_lengths_sparse[t] ||
                       new_succ[t] != cache.succ_sparse[t]
                        cache_changed = true
                        break
                    end
                end
            end

            if !cache_changed
                # Revert shift
                revert_shift_in_place!(pos, i, j)
                INCREMENTAL_SHIFT_STATS[] = (
                    candidates=INCREMENTAL_SHIFT_STATS[].candidates,
                    skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter,
                    skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp + 1,
                    evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
                    accepted=INCREMENTAL_SHIFT_STATS[].accepted,
                    time=INCREMENTAL_SHIFT_STATS[].time,
                    calls=INCREMENTAL_SHIFT_STATS[].calls
                )
                continue
            end

            # Step G: Run DP
            Γ_sparse = sparse_dp_phase2(new_P, new_succ, new_sorted_depots, m, n)
            new_profit = lookup_Γ_sparse(1, m, new_sorted_depots, Γ_sparse)

            INCREMENTAL_SHIFT_STATS[] = (
                candidates=INCREMENTAL_SHIFT_STATS[].candidates,
                skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter,
                skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp,
                evaluated=INCREMENTAL_SHIFT_STATS[].evaluated + 1,
                accepted=INCREMENTAL_SHIFT_STATS[].accepted,
                time=INCREMENTAL_SHIFT_STATS[].time,
                calls=INCREMENTAL_SHIFT_STATS[].calls
            )

            # Step H: Accept or reject
            if new_profit > particle.current_profit
                # ACCEPT
                particle.current_profit = new_profit

                # Update node_to_position for depot nodes (used by get_sorted_depot_positions)
                # The shift affects positions [lo, hi]; update all nodes in that range
                if i < j
                    for p in i:j-1
                        particle.node_to_position[pos[p]] = p
                    end
                else
                    for p in j:i
                        particle.node_to_position[pos[p]] = p
                    end
                end

                # Update cache
                cache.sorted_depot_positions = new_sorted_depots
                cache.P_sparse = new_P
                cache.tour_lengths_sparse = new_len
                cache.succ_sparse = new_succ

                new_ti = build_tour_intervals(new_sorted_depots, new_len)

                INCREMENTAL_SHIFT_STATS[] = (
                    candidates=INCREMENTAL_SHIFT_STATS[].candidates,
                    skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter,
                    skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp,
                    evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
                    accepted=INCREMENTAL_SHIFT_STATS[].accepted + 1,
                    time=INCREMENTAL_SHIFT_STATS[].time + (time() - start_time),
                    calls=INCREMENTAL_SHIFT_STATS[].calls + 1
                )
                return true, new_ti, cache
            else
                # REJECT — revert shift
                revert_shift_in_place!(pos, i, j)
            end
        end
    end

    INCREMENTAL_SHIFT_STATS[] = (
        candidates=INCREMENTAL_SHIFT_STATS[].candidates,
        skipped_filter=INCREMENTAL_SHIFT_STATS[].skipped_filter,
        skipped_dp=INCREMENTAL_SHIFT_STATS[].skipped_dp,
        evaluated=INCREMENTAL_SHIFT_STATS[].evaluated,
        accepted=INCREMENTAL_SHIFT_STATS[].accepted,
        time=INCREMENTAL_SHIFT_STATS[].time + (time() - start_time),
        calls=INCREMENTAL_SHIFT_STATS[].calls + 1
    )
    return false, tour_intervals, cache
end

"""
Fully incremental local search: both shift and swap use incremental tour updates.
Returns timing info for compatibility.
"""
function local_search_fully_incremental!(particle::Particle, particle_idx::Int, pso::PSOiA_TOP_multiple_depots)
    total_time_shift = 0.0
    total_time_swap = 0.0
    total_time_destruction_repair = 0.0

    swap_counters = Ref((0, 0))

    # Initialize cache once
    _, _, tour_intervals = fast_split_sparse(particle.position, particle, pso)
    cache = init_tour_cache(particle, pso)

    improved = true
    while improved
        improved = false
        neighborhoods = shuffle([1, 2])

        for neighborhood in neighborhoods
            if neighborhood == 1
                time_before = time()
                improved, tour_intervals, cache = shift_operator_incremental!(
                    particle, pso, cache, tour_intervals
                )
                total_time_shift += time() - time_before
            else
                time_before = time()
                improved, tour_intervals, cache = swap_operator_incremental!(
                    particle, pso, tour_intervals, cache, swap_counters
                )
                total_time_swap += time() - time_before
            end

            if improved
                break
            end
        end
    end

    return total_time_shift, total_time_swap, total_time_destruction_repair
end

# ============================================================================
# END INCREMENTAL TOUR UPDATE FOR SHIFT
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
    if lookup_cost(pso, node_i_pred, node_i, Float64(L*4)) > L
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
    if lookup_cost(pso, node_j_pred, node_i, Float64(L*4)) > L
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
    if lookup_cost(pso, node_i_pred, node_i_succ, Float64(L*4)) > L
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
    # Precompute saturated tours for deadness checks
    _, _, tour_lengths_sparse, sorted_depot_positions = compute_saturated_tours_sparse(
        particle.position, particle.node_to_position, pso
    )
    dead_positions = compute_dead_positions(n, sorted_depot_positions, tour_lengths_sparse)
    
    # Pre-allocate inner candidate buffer (avoids O(n) allocations per outer iteration)
    inner_j_buf = collect(1:n)

    for i in positions
        node_i = particle.position[i]
        is_depot = node_i > pso.n_pure_customers
        shuffle!(inner_j_buf)
        for j_idx in 1:n
            @inbounds j = inner_j_buf[j_idx]
            if i == j
                continue
            end
            SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates + 1,
                             skipped=SHIFT_STATS[].skipped,
                             time=SHIFT_STATS[].time,
                             calls=SHIFT_STATS[].calls)

            # === IRRELEVANCE-BASED SKIP (customers only) ===
            if ENABLE_SHIFT_IRRELEVANCE_FILTER[] && !is_depot
                # Check 1: Original blocking-based skip (works well with binary costs)
                is_blocking_or_dead = is_blocking(particle, i, pso) || dead_positions[i]
                irrelevant_removed = false
                if is_blocking_or_dead && is_blocking_once_removed(particle, i, pso)
                    irrelevant_removed = true
                end

                if irrelevant_removed && j > 1 && dead_positions[j - 1]
                    SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                                     skipped=SHIFT_STATS[].skipped + 1,
                                     time=SHIFT_STATS[].time,
                                     calls=SHIFT_STATS[].calls)
                    continue
                end
                
                # Check 2: Dead-zone skip (works with both binary and L-infinity costs)
                # If both source and target positions are in dead zones, the move cannot affect profit
                if dead_positions[i] && j > 1 && dead_positions[j - 1]
                    SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                                     skipped=SHIFT_STATS[].skipped + 1,
                                     time=SHIFT_STATS[].time,
                                     calls=SHIFT_STATS[].calls)
                    continue
                end
            end

            new_position = move_element(particle.position, i, j)
            time_before_split = time()
            new_profit = fast_split_multiple_depots(new_position, pso)
            time_after_split = time()
            time_split += time_after_split - time_before_split
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
                SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                                 skipped=SHIFT_STATS[].skipped,
                                 time=SHIFT_STATS[].time + (time() - start_time),
                                 calls=SHIFT_STATS[].calls + 1)
                return true
            end
        end
    end
    ending_time = time()
    #println("time to run split: $time_split")
    #println("time to run shift without split: $(ending_time - start_time - time_split)")
    SHIFT_STATS[] = (candidates=SHIFT_STATS[].candidates,
                     skipped=SHIFT_STATS[].skipped,
                     time=SHIFT_STATS[].time + (ending_time - start_time),
                     calls=SHIFT_STATS[].calls + 1)
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
    # Precompute saturated tours for deadness checks
    _, _, tour_lengths_sparse, sorted_depot_positions = compute_saturated_tours_sparse(
        particle.position, particle.node_to_position, pso
    )
    dead_positions = compute_dead_positions(n, sorted_depot_positions, tour_lengths_sparse)
    
    # Pre-allocate inner candidate buffer (avoids O(n) allocations per outer iteration)
    swap_j_buf = Vector{Int}(undef, n)

    for i in positions
        inner_len = n - i
        for k in 1:inner_len; swap_j_buf[k] = i + k; end
        shuffle!(view(swap_j_buf, 1:inner_len))
        for j_idx in 1:inner_len
            j = swap_j_buf[j_idx]
            SWAP_STATS[] = (candidates=SWAP_STATS[].candidates + 1,
                            skipped=SWAP_STATS[].skipped,
                            time=SWAP_STATS[].time,
                            calls=SWAP_STATS[].calls)
            node_at_i = pos[i]
            node_at_j = pos[j]
            is_depot_i = node_at_i > pso.n_pure_customers
            is_depot_j = node_at_j > pso.n_pure_customers
            if ENABLE_SWAP_BLOCKING_FILTER[] && !is_depot_i && !is_depot_j
                # Check 1: Original blocking-based skip (works well with binary costs)
                if is_blocking_once_inserted(particle, i, j, pso) &&
                   is_blocking_once_removed(particle, i, pso) &&
                   is_blocking_once_inserted(particle, j, i, pso) &&
                   is_blocking_once_removed(particle, j, pso)
                    SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                                    skipped=SWAP_STATS[].skipped + 1,
                                    time=SWAP_STATS[].time,
                                    calls=SWAP_STATS[].calls)
                    continue
                end
                
                # Check 2: Dead-zone skip (works with both binary and L-infinity costs)
                # If both positions are in dead zones, the swap cannot affect profit
                if dead_positions[i] && dead_positions[j]
                    SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                                    skipped=SWAP_STATS[].skipped + 1,
                                    time=SWAP_STATS[].time,
                                    calls=SWAP_STATS[].calls)
                    continue
                end
            end
            pos[i], pos[j] = pos[j], pos[i]  # trial swap
            new_profit = fast_split_multiple_depots(pos, pso)
            if new_profit > particle.current_profit
                particle.current_profit = new_profit
                # Update node-to-position mapping incrementally (O(1) - just 2 updates)
                particle.node_to_position[node_at_i] = j
                particle.node_to_position[node_at_j] = i
                SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                                skipped=SWAP_STATS[].skipped,
                                time=SWAP_STATS[].time,
                                calls=SWAP_STATS[].calls + 1)
                return true  # keep swap; pos already updated
            else
                pos[i], pos[j] = pos[j], pos[i]  # revert
            end
        end
    end
    SWAP_STATS[] = (candidates=SWAP_STATS[].candidates,
                    skipped=SWAP_STATS[].skipped,
                    time=SWAP_STATS[].time,
                    calls=SWAP_STATS[].calls + 1)
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
            total_cost += lookup_cost(pso, 0, route[1], 0.0)
            
            # Cost between customers
            for i in 1:(length(route)-1)
                total_cost += lookup_cost(pso, route[i], route[i+1], 0.0)
            end
            
            # Cost from last customer to depot
            total_cost += lookup_cost(pso, route[end], 0, 0.0)
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
                       swarm_size::Int = 50, max_iterations::Int = 1000, max_time::Float64 = 120.0,
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
    
    # Build dense cost matrix from the costs dictionary.
    # Node indices: 0 (artificial) .. length(customers). Matrix is 1-indexed: cost_matrix[i+1, j+1].
    n_total_nodes = length(customers)  # includes depot nodes; index 0 is artificial
    default_cost = Float64(max_battery_time * 4)
    cost_matrix = fill(default_cost, n_total_nodes + 1, n_total_nodes + 1)
    for ((i, j), c) in costs
        cost_matrix[i + 1, j + 1] = c
    end

    # Initialize PSO (node_to_position is now stored in each Particle)
    pso = PSOiA_TOP_multiple_depots(
        Particle[], Int[], -Inf, swarm_size, max_iterations,
        w, c1, c2, ph, pm, n_drones, n_pure_customers, max_battery_time,
        customers, profits, costs, cost_matrix, left_neighbors,
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
    # IDCH can be disabled via ENABLE_IDCH
    initialize_swarm(pso, use_greedy_init, skip_idch=!ENABLE_IDCH[])
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
        if elapsed_time > max_time * 0.9
            println("[TIME CHECK] Iteration $iter: elapsed=$(round(elapsed_time, digits=2))s, remaining=$(round(remaining_time, digits=2))s, best_profit=$(round(pso.global_best_profit, digits=6))")
            flush(stdout)
        end
        if iter % 100 == 0
            flush(stdout)
        end
        if elapsed_time > max_time
            println("[TIME CHECK] Maximum time limit of $(max_time) seconds reached at iteration $iter. Stopping algorithm.")
            flush(stdout)
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
            # if x <= 3 || x % 10 == 1  # Log first 3 particles and every 10th
            #     println("[TIME CHECK] Processing particle $x/$(pso.swarm_size) (iter $iter): elapsed=$(round(elapsed, digits=2))s, remaining=$(round(max_time - elapsed, digits=2))s")
            # end
            
            # Random move with probability ph (IDCH) if enabled
            if ENABLE_IDCH[] && rand() < pso.ph
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
                time_before_local_search = time()
                if ENABLE_INCREMENTAL_LOCAL_SEARCH[]
                    time_shift, time_swap, time_destruction_repair = local_search_fully_incremental!(pso.swarm[x], x, pso)
                else
                    # Using sparse operators with boundary optimization
                    time_shift, time_swap, time_destruction_repair = local_search_sparse!(pso.swarm[x], x, pso)
                end
                time_after_local_search = time()
                local_search_time = time_after_local_search - time_before_local_search
                total_time_local_search += local_search_time
                total_time_shift += time_shift
                total_time_swap += time_swap
                total_time_destruction_repair += time_destruction_repair
                elapsed = time_after_local_search - start_time
                if local_search_time > 10.0
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
                route_cost += lookup_cost(pso, 0, route[1], 0.0)
                
                # Cost between customers
                for j in 1:(length(route)-1)
                    route_cost += lookup_cost(pso, route[j], route[j+1], 0.0)
                end
                
                # Cost from last customer to depot
                route_cost += lookup_cost(pso, route[end], 0, 0.0)
            end
            
            route_profit = sum(pso.profits[c] for c in route)
            # println("  Drone $i: $route (cost: $route_cost, profit: $(round(route_profit, digits=3)))")
        end
    end
    # println("Total profit: $(round(optimal_profit, digits=3))")
    
    return routes
end