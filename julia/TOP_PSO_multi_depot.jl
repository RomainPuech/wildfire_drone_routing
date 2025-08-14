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
    accessible_customers::Vector{Int}  # Indices of accessible customers
    depot_coord::Vector{Tuple{Int,Int}}    # Depot coordinates
    closest_depot_distance::Vector{Float64}  # Pre-computed min return distance to closest depot (Chebyshev)
end


"""
Fast split procedure for multiple depots
"""
function fast_split_with_routes_multiple_depots_XX(permutation::Vector{Int}, pso_multiple_depots::PSOiA_TOP_multiple_depots)
    n = length(permutation)
    m = pso_multiple_depots.n_drones
    L = pso_multiple_depots.max_battery_time
    
    if n == 0
        return 0.0, Vector{Vector{Int}}()
    end

    start_time_phase_1 = time()
    
    # Calculate saturated tours P[i] and first successor succ[i]
    P = zeros(n)  # Profit of saturated tour starting at position i
    succ = zeros(Int, n)  # First successor of saturated tour starting at position i
    tour_lengths = zeros(Int, n)  # Length of each saturated tour
    
    for i in 1:n
        current_cost = 0.0
        current_profit = 0.0
        j = i
        
        # Build maximal feasible tour starting from position i
        while j <= n
            customer_idx = permutation[j]
            
            # Add travel cost to this customer
            if j == i
                # Cost from depot to first customer
                travel_cost = get(pso_multiple_depots.costs, (0, customer_idx), L*4)
            else
                prev_customer = permutation[j-1]
                travel_cost = get(pso_multiple_depots.costs, (prev_customer, customer_idx), L*4)
            end
            
            # Feasibility: ensure we can still return to the closest depot using precomputed distance
            return_distance = pso_multiple_depots.closest_depot_distance[customer_idx]
            if current_cost + travel_cost + return_distance > L
                break
            end
            
            current_cost += travel_cost
            current_profit += pso_multiple_depots.profits[customer_idx]
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
    time_phase_1 = time() - start_time_phase_1
    # println("time to run phase 1: $time_phase_1")

    start_time_phase_2 = time()
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
    time_phase_2 = time() - start_time_phase_2
    # println("time to run phase 2: $time_phase_2")

    start_time_phase_3 = time()
    # Backtrack to find the optimal routes (as described in the paper)
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
    time_phase_3 = time() - start_time_phase_3
    # println("time to run phase 3: $time_phase_3")
    total_time = time_phase_1 + time_phase_2 + time_phase_3
    if total_time > 0.01
        println("total time: $(time_phase_1 + time_phase_2 + time_phase_3)")
        # relative time
        println("relative time: $(time_phase_1 / (time_phase_1 + time_phase_2 + time_phase_3))")
        println("relative time: $(time_phase_2 / (time_phase_1 + time_phase_2 + time_phase_3))")
        println("relative time: $(time_phase_3 / (time_phase_1 + time_phase_2 + time_phase_3))")
        println("n: $n")
    end

    return Γ[1, m + 1], routes
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

    # start_time_phase_1 = time()
    
    # Calculate saturated tours P[i] and first successor succ[i]
    P = zeros(n)  # Profit of saturated tour starting at position i
    succ = zeros(Int, n)  # First successor of saturated tour starting at position i
    tour_lengths = zeros(Int, n)  # Length of each saturated tour
    
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
            travel_cost = get(pso_multiple_depots.costs, (prev_customer, customer_idx), L*4)
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

    return Γ[1, m + 1], routes
end



# Update the original fast_split to use the new function
function fast_split_multiple_depots(permutation::Vector{Int}, pso_multiple_depots::PSOiA_TOP_multiple_depots)
    profit, _ = fast_split_with_routes_multiple_depots(permutation, pso_multiple_depots)
    return profit
end

"""
Initialize particle swarm
"""
function initialize_swarm(pso::PSOiA_TOP_multiple_depots)
    for i in 1:pso.swarm_size
        # Create random permutation of accessible customers
        position = shuffle(pso.accessible_customers)
        profit = fast_split_multiple_depots(position, pso)
        
        particle = Particle(
            copy(position),
            copy(position),
            profit,
            profit
        )
        
        push!(pso.swarm, particle)
        
        # Update global best
        if profit > pso.global_best_profit
            pso.global_best = copy(position)
            pso.global_best_profit = profit
            # println("Initial swarm: New best = $(round(pso.global_best_profit, digits=3))")
        end
    end
    
    # Initialize some particles with IDCH heuristic (better quality)
    n_idch = min(5, pso.swarm_size ÷ 2)
    for i in 1:n_idch
        position = idch_heuristic(pso, false)  # Fast version
        profit = fast_split_multiple_depots(position, pso)
        
        pso.swarm[i].position = copy(position)
        pso.swarm[i].local_best = copy(position)
        pso.swarm[i].local_best_profit = profit
        pso.swarm[i].current_profit = profit
        
        if profit > pso.global_best_profit
            pso.global_best = copy(position)
            pso.global_best_profit = profit
            # println("IDCH initialization: New best = $(round(pso.global_best_profit, digits=3)), Solution: $(pso.global_best)")
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
function local_search!(particle::Particle, pso::PSOiA_TOP_multiple_depots)
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
                improved = shift_operator!(particle, pso)
                time_after_shift = time()
                total_time_shift += time_after_shift - time_before_shift
                improved_count_shift += improved ? 1 : 0
                call_count_shift += 1
            elseif neighborhood == 2  # Swap operator
                time_before_swap = time()
                improved = swap_operator!(particle, pso)
                time_after_swap = time()
                total_time_swap += time_after_swap - time_before_swap
                improved_count_swap += improved ? 1 : 0
                call_count_swap += 1
            else  # Destruction/repair operator
                time_before_destruction_repair = time()
                improved = destruction_repair_operator!(particle, pso)
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
Shift operator: move customer to different position
"""
function shift_operator!(particle::Particle, pso::PSOiA_TOP_multiple_depots)
    time_split = 0.0
    start_time = time()
    n = length(particle.position)
    positions = shuffle(1:n)  # Random order evaluation
    
    for i in positions
        for j in shuffle(setdiff(1:n, [i]))
            # GRID based optimization: is it worth trying this shift or not?
            # if the shift won't change the current tours, then we don't need to try it
            # easy version first: IF:
            
            # simplified but equivalent: if moving it blocks both, then we don't try it
            if is_blocking_once_inserted(particle, i, j, pso) && is_blocking_once_removed(particle, i, pso)
                # then we don't try it
                continue
            end
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
                ending_time = time()
                #println("time to run split: $time_split")
                #println("time to run shift without split: $(ending_time - start_time - time_split)")
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
"""
function swap_operator!(particle::Particle, pso::PSOiA_TOP_multiple_depots)
    n = length(particle.position)
    positions = shuffle(1:n)
    pos = particle.position
    for i in positions
        for j in shuffle((i+1):n)
            # same as for shifts: if both are blocking in their respective new positions, then we don't try it
            if is_blocking_once_inserted(particle, i, j, pso) && is_blocking_once_inserted(particle, j, i, pso)
                continue
            end
            pos[i], pos[j] = pos[j], pos[i]  # trial swap
            new_profit = fast_split_multiple_depots(pos, pso)
            if new_profit > particle.current_profit
                particle.current_profit = new_profit
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
"""
function destruction_repair_operator!(particle::Particle, pso::PSOiA_TOP_multiple_depots)
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
                    similar_found = true
                    break
                end
            end
            
            # Rule 3: If no similar particle found, replace worst
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
                       costs::Dict{Tuple{Int,Int}, Float64}, n_drones::Int, n_pure_customers::Int,
                       max_battery_time::Int, depot_coord::Vector{Tuple{Int,Int}} = [(0, 0)];
                       swarm_size::Int = 50, max_iterations::Int = 1000,
                       w::Float64 = 0.3, c1::Float64 = 0.5, c2::Float64 = 0.3,
                       ph::Float64 = 0.1, pm::Float64 = 0.3)
    # println("Starting solve_PSO_TOP_multiple_depots in TOP_PSO_multi_depot.jl...")
    # Start timing the algorithm execution
    start_time = time()
    
    # Determine accessible customers using L-infinity distance to closest depot instead of cost matrix
    accessible_customers = Int[]
    # println("Customers: $(customers)")
    # println("See, we have the depots above...")
    for i in 1:length(customers)
        # Calculate L-infinity distance (minimum hops) to visit customer and return
        customer_coord = customers[i]
        min_distance = max_battery_time*2
        for depot in depot_coord
            depot_x, depot_y = depot
            customer_x, customer_y = customer_coord
            # L-infinity distance: max(|x1-x2|, |y1-y2|)
            distance_to = max(abs(customer_x - depot_x), abs(customer_y - depot_y)) #+ 4 # +4 because we have the artificial node
            distance_from = max(abs(customer_x - depot_x), abs(customer_y - depot_y)) #+ 4 # +4 because we have the artificial node
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
    closest_depot_distance = Vector{Float64}(undef, length(customers))
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
    println("accessible_customers: $(length(accessible_customers))")
    println("n_pure_customers: $(n_pure_customers)")
    # Initialize PSO
    pso = PSOiA_TOP_multiple_depots(
        Particle[], Int[], -Inf, swarm_size, max_iterations,
        w, c1, c2, ph, pm, n_drones, n_pure_customers, max_battery_time,
        customers, profits, costs, accessible_customers, depot_coord, closest_depot_distance
    )
    
    # println("=== PSO SETUP ===")
    # println("Total customers: $(length(customers))")
    # println("Accessible customers: $(length(accessible_customers))")
    # println("Max battery time: $max_battery_time")
    # println("Number of drones: $n_drones")
    # println("==================")

    time_to_initialize_pso = time() - start_time
    println("time to initialize pso: $(time_to_initialize_pso)")
    time_before_swarm_sampling = time()
    
    # Initialize and evaluate each particle in swarm (see Section 2.3)
    initialize_swarm(pso)
    time_after_swarm_sampling = time()
    println("time to initialize swarm: $(time_after_swarm_sampling - time_before_swarm_sampling)")
    

    iter = 1
    itermax = max_iterations #* length(accessible_customers) * n_drones  # As mentioned in paper
    
    # println("Starting PSO with $(pso.swarm_size) particles, initial best: $(pso.global_best_profit)")
    # Main algorithm loop following Algorithm 1
    total_time_local_search = 0.0
    total_time_swap = 0.0
    total_time_shift = 0.0
    total_time_destruction_repair = 0.0
    while iter <= itermax
        improvement_found = false
        
        for x in 1:pso.swarm_size
            # Random move with probability ph
            if rand() < pso.ph
                # Move S[x] to a new position (see Section 2.3)
                time_before_idch = time()
                pso.swarm[x].position = idch_heuristic(pso, false)  # Fast version
                time_after_idch = time()
                #println("time to run idch: $(time_after_idch - time_before_idch)")
            else
                # Update S[x].pos (see Section 2.5)
                update_position!(pso.swarm[x], pso.global_best, pso)
            end
            
            # Local search with probability pm
            if rand() < pso.pm
                # Apply local search on S[x].pos (see Section 2.4)
                time_before_local_search = time()
                time_shift, time_swap, time_destruction_repair = local_search!(pso.swarm[x], pso)
                time_after_local_search = time()
                total_time_local_search += time_after_local_search - time_before_local_search
                total_time_shift += time_shift
                total_time_swap += time_swap
                total_time_destruction_repair += time_destruction_repair
                #println("time to run local search: $(time_after_local_search - time_before_local_search)")
            end
            
            # Evaluate S[x].pos (see Section 2.2)
            time_before_split = time()
            pso.swarm[x].current_profit = fast_split_multiple_depots(pso.swarm[x].position, pso)
            time_after_split = time()
            #println("time to run split: $(time_after_split - time_before_split)")
            # Update lbest of S (see Section 2.6)
            prev_global_best = pso.global_best_profit
            update_local_bests!(pso)

            # Check if update Rule 3 is applied (new local best discovered)
            if pso.global_best_profit > prev_global_best
                improvement_found = true
                println("Iter $iter: New best = $(round(pso.global_best_profit, digits=3))")#, Solution: $(pso.global_best)")
            end
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
    # Use the proper split procedure from the paper
    optimal_profit, routes = fast_split_with_routes_multiple_depots(giant_tour, pso)
    
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