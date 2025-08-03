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

mutable struct PSOiA_TOP
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
    max_battery_time::Int
    customers::Vector{Tuple{Int,Int}}  # Customer coordinates
    profits::Vector{Float64}       # Customer profits
    costs::Dict{Tuple{Int,Int}, Float64}  # Travel costs
    accessible_customers::Vector{Int}  # Indices of accessible customers
    depot_coord::Tuple{Int,Int}    # Depot coordinates
end

"""
O(m*n) Fast split procedure based on interval graph model
Follows the algorithm described in the paper exactly - Figure 1
Precomputes all saturated tours in O(n) time, then uses O(m*n) DP
"""
function fast_split_with_routes(permutation::Vector{Int}, pso::PSOiA_TOP)
    n = length(permutation)
    m = pso.n_drones
    L = pso.max_battery_time
    
    if n == 0
        return 0.0, Vector{Vector{Int}}()
    end
    
    # Phase 1: Precompute all saturated tours in O(n) time
    # This is the key improvement over the O(n²) version
    P = zeros(n)  # Profit of saturated tour starting at position i
    succ = zeros(Int, n)  # First successor of saturated tour starting at position i
    tour_lengths = zeros(Int, n)  # Length of each saturated tour
    
    # Single forward pass to compute all saturated tours - O(n) total
    j = 1  # Current position being evaluated
    for i in 1:n
        # Extend tour starting at i as far as possible
        current_cost = 0.0
        current_profit = 0.0
        
        # Ensure j starts at least at position i
        j = max(j, i)
        
        # Extend tour from current j position
        while j <= n
            customer_idx = permutation[j]
            
            # Calculate travel cost to this customer
            if j == i
                # Cost from depot to first customer
                travel_cost = get(pso.costs, (0, customer_idx), L*4)
            else
                prev_customer = permutation[j-1]
                travel_cost = get(pso.costs, (prev_customer, customer_idx), L*4)
            end
            
            # Calculate return cost to depot (L-infinity distance)
            customer_coord = pso.customers[customer_idx]
            depot_x, depot_y = pso.depot_coord
            customer_x, customer_y = customer_coord
            return_distance = max(abs(customer_x - depot_x), abs(customer_y - depot_y))
            
            # Check feasibility
            if current_cost + travel_cost + return_distance > L
                break
            end
            
            # Add this customer to the tour
            current_cost += travel_cost
            current_profit += pso.profits[customer_idx]
            j += 1
        end
        
        # Store results for tour starting at position i
        P[i] = current_profit
        tour_lengths[i] = j - i
        
        # Apply Equation 3: succ[i] = i + l_i^max + 1 if i + l_i^max + 1 ≤ n, else 0
        next_pos = i + tour_lengths[i]
        if next_pos <= n
            succ[i] = next_pos
        else
            succ[i] = 0  # According to Equation 3
        end
        
        # Key optimization: if tour starting at i+1 would start after j,
        # we can reuse the current j position for the next iteration
        # This ensures O(n) total complexity
    end
    
    # Phase 2: Dynamic programming - O(m*n) as per Equation 4
    # Γ[i,j] = max profit using j drones from position i onwards
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
    
    # Phase 3: Backtrack to find the optimal routes (as described in the paper)
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
    
    return Γ[1, m + 1], routes
end

# Update the original fast_split to use the new function
function fast_split(permutation::Vector{Int}, pso::PSOiA_TOP)
    profit, _ = fast_split_with_routes(permutation, pso)
    return profit
end

"""
Initialize particle swarm
"""
function initialize_swarm(pso::PSOiA_TOP)
    for i in 1:pso.swarm_size
        # Create random permutation of accessible customers
        position = shuffle(pso.accessible_customers)
        profit = fast_split(position, pso)
        
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
            println("Initial swarm: New best = $(round(pso.global_best_profit, digits=3))")
        end
    end
    
    # Initialize some particles with IDCH heuristic (better quality)
    # Paper mentions using slow IDCH for initialization with diversification
    n_idch = min(5, pso.swarm_size ÷ 2)
    for i in 1:n_idch
        position = idch_heuristic(pso, true)  # Slow version with diversification for initialization
        profit = fast_split(position, pso)
        
        pso.swarm[i].position = copy(position)
        pso.swarm[i].local_best = copy(position)
        pso.swarm[i].local_best_profit = profit
        pso.swarm[i].current_profit = profit
        
        if profit > pso.global_best_profit
            pso.global_best = copy(position)
            pso.global_best_profit = profit
            println("IDCH initialization: New best = $(round(pso.global_best_profit, digits=3)), Solution: $(pso.global_best)")
        end
    end
end

"""
2-opt tour optimization procedure
Improves individual routes by removing two edges and reconnecting in the best way
"""
function two_opt_improvement!(route::Vector{Int}, pso::PSOiA_TOP)
    if length(route) <= 3
        return false  # Cannot improve tours with 3 or fewer customers
    end
    
    improved = true
    overall_improved = false
    
    while improved
        improved = false
        n = length(route)
        
        for i in 1:(n-2)
            for j in (i+2):n
                # Skip if j == i+1 (adjacent edges) or creates invalid swap
                if j == i + 1
                    continue
                end
                
                # Calculate current cost of edges (i, i+1) and (j, j+1)
                current_cost = 0.0
                new_cost = 0.0
                
                # Current edges
                if i == 1
                    # Edge from depot to route[1]
                    current_cost += get(pso.costs, (0, route[i]), 0.0)
                else
                    # Edge from route[i-1] to route[i]
                    current_cost += get(pso.costs, (route[i-1], route[i]), 0.0)
                end
                
                if i < n
                    # Edge from route[i] to route[i+1]
                    current_cost += get(pso.costs, (route[i], route[i+1]), 0.0)
                end
                
                if j < n
                    # Edge from route[j] to route[j+1]
                    current_cost += get(pso.costs, (route[j], route[j+1]), 0.0)
                else
                    # Edge from route[j] to depot
                    current_cost += get(pso.costs, (route[j], 0), 0.0)
                end
                
                # Try 2-opt swap: reverse the order between positions i+1 and j
                new_route = copy(route)
                reverse!(new_route, i+1, j)
                
                # Calculate new cost after 2-opt
                if i == 1
                    # Edge from depot to new_route[1]
                    new_cost += get(pso.costs, (0, new_route[i]), 0.0)
                else
                    # Edge from new_route[i-1] to new_route[i]
                    new_cost += get(pso.costs, (new_route[i-1], new_route[i]), 0.0)
                end
                
                if i < n
                    # Edge from new_route[i] to new_route[i+1]
                    new_cost += get(pso.costs, (new_route[i], new_route[i+1]), 0.0)
                end
                
                if j < n
                    # Edge from new_route[j] to new_route[j+1]
                    new_cost += get(pso.costs, (new_route[j], new_route[j+1]), 0.0)
                else
                    # Edge from new_route[j] to depot
                    new_cost += get(pso.costs, (new_route[j], 0), 0.0)
                end
                
                # If improvement found, apply it
                if new_cost < current_cost
                    route[:] = new_route
                    improved = true
                    overall_improved = true
                    break
                end
            end
            if improved
                break
            end
        end
    end
    
    return overall_improved
end

"""
Apply 2-opt to all routes in a solution
"""
function apply_two_opt_to_solution!(permutation::Vector{Int}, pso::PSOiA_TOP)
    # Extract current routes from the permutation
    _, routes = fast_split_with_routes(permutation, pso)
    
    improved = false
    
    # Apply 2-opt to each route
    for route in routes
        if length(route) > 3
            if two_opt_improvement!(route, pso)
                improved = true
            end
        end
    end
    
    # If any route was improved, reconstruct the permutation
    if improved
        # Reconstruct permutation from improved routes
        new_permutation = Int[]
        for route in routes
            append!(new_permutation, route)
        end
        
        # Add any missing customers that were in the original permutation
        # but not included in the optimal routes
        used_customers = Set(new_permutation)
        remaining_customers = Int[]
        for customer in permutation
            if !(customer in used_customers)
                push!(remaining_customers, customer)
            end
        end
        
        # Append remaining customers to maintain same permutation length
        append!(new_permutation, remaining_customers)
        
        # Ensure we have the exact same length as input
        if length(new_permutation) == length(permutation)
            permutation[:] = new_permutation
        else
            # If lengths don't match, don't modify (safety fallback)
            improved = false
        end
    end
    
    return improved
end

"""
Iterative Destruction/Construction Heuristic (IDCH)
Now includes 2-opt optimization as mentioned in the paper
"""
function idch_heuristic(pso::PSOiA_TOP, slow_version::Bool = false)
    n = length(pso.accessible_customers)
    max_iter = slow_version ? n^2 : n
    diversify_every = n  # For slow version, diversify every n iterations
    
    # Start with random permutation
    current_solution = shuffle(pso.accessible_customers)
    best_solution = copy(current_solution)
    best_profit = fast_split(best_solution, pso)
    
    no_improvement = 0
    iteration = 0
    
    while no_improvement < max_iter
        iteration += 1
        
        # Slow version: apply diversification every n iterations
        if slow_version && (iteration % diversify_every == 0)
            # Diversification phase as described in paper
            # Remove up to n/m customers (large destruction)
            max_remove_diversify = max(1, n ÷ pso.n_drones)
            n_remove = rand(1:max_remove_diversify)
            
            destroyed = copy(current_solution)
            removed_customers = Int[]
            
            for _ in 1:n_remove
                if length(destroyed) > 1
                    idx = rand(1:length(destroyed))
                    push!(removed_customers, destroyed[idx])
                    deleteat!(destroyed, idx)
                end
            end
            
            # Apply 2-opt to each tour after destruction (paper mentions this)
            apply_two_opt_to_solution!(destroyed, pso)
            
            # Reconstruction phase
            reconstructed = best_insertion_algorithm(destroyed, removed_customers, pso)
            current_solution = reconstructed
        else
            # Regular destruction phase: remove 1-3 customers
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
            
            # Apply 2-opt procedure to reduce travel cost (as mentioned in paper)
            apply_two_opt_to_solution!(destroyed, pso)
            
            # Construction phase: reinsert customers using Best Insertion
            reconstructed = best_insertion_algorithm(destroyed, removed_customers, pso)
            current_solution = reconstructed
        end
        
        profit = fast_split(current_solution, pso)
        
        if profit > best_profit
            best_solution = copy(current_solution)
            best_profit = profit
            no_improvement = 0
        else
            no_improvement += 1
        end
    end
    
    return best_solution
end

"""
Best Insertion Algorithm (BIA)
"""
function best_insertion_algorithm(partial_solution::Vector{Int}, unrouted::Vector{Int}, pso::PSOiA_TOP)
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
function update_position!(particle::Particle, global_best::Vector{Int}, pso::PSOiA_TOP)
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
    particle.current_profit = fast_split(new_position, pso)
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
function local_search!(particle::Particle, pso::PSOiA_TOP)
    improved = true
    
    while improved
        improved = false
        neighborhoods = [1, 2, 3]  # shift, swap, destruction/repair
        shuffle!(neighborhoods)
        
        for neighborhood in neighborhoods
            if neighborhood == 1  # Shift operator
                improved = shift_operator!(particle, pso)
            elseif neighborhood == 2  # Swap operator
                improved = swap_operator!(particle, pso)
            else  # Destruction/repair operator
                improved = destruction_repair_operator!(particle, pso)
            end
            
            if improved
                break
            end
        end
    end
end

"""
Shift operator: move customer to different position
"""
function shift_operator!(particle::Particle, pso::PSOiA_TOP)
    n = length(particle.position)
    positions = shuffle(1:n)  # Random order evaluation
    
    for i in positions
        for j in shuffle(setdiff(1:n, [i]))
            # Try moving customer from position i to position j
            new_position = copy(particle.position)
            customer = new_position[i]
            deleteat!(new_position, i)
            insert!(new_position, j > i ? j-1 : j, customer)
            
            new_profit = fast_split(new_position, pso)
            if new_profit > particle.current_profit
                particle.position = new_position
                particle.current_profit = new_profit
                return true
            end
        end
    end
    return false
end

"""
Swap operator: exchange two customers
"""
function swap_operator!(particle::Particle, pso::PSOiA_TOP)
    n = length(particle.position)
    positions = shuffle(1:n)
    
    for i in positions
        for j in shuffle((i+1):n)
            # Try swapping customers at positions i and j
            new_position = copy(particle.position)
            new_position[i], new_position[j] = new_position[j], new_position[i]
            
            new_profit = fast_split(new_position, pso)
            if new_profit > particle.current_profit
                particle.position = new_position
                particle.current_profit = new_profit
                return true
            end
        end
    end
    return false
end

"""
Destruction/repair operator
Now includes 2-opt after reconstruction as mentioned in the paper
"""
function destruction_repair_operator!(particle::Particle, pso::PSOiA_TOP)
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
    
    # Apply 2-opt to optimize tour costs after reconstruction (as mentioned in paper)
    apply_two_opt_to_solution!(reconstructed, pso)
    
    new_profit = fast_split(reconstructed, pso)
    
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
function update_local_bests!(pso::PSOiA_TOP, δ::Float64 = 1e-6)
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
function calculate_travel_cost(permutation::Vector{Int}, pso::PSOiA_TOP)
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

"""
Main PSO algorithm - following Algorithm 1 from the paper exactly
"""
function solve_PSO_TOP(customers::Vector{Tuple{Int,Int}}, profits::Vector{Float64}, 
                       costs::Dict{Tuple{Int,Int}, Float64}, n_drones::Int, 
                       max_battery_time::Int, depot_coord::Tuple{Int,Int} = (0, 0);
                       swarm_size::Int = 50, max_iterations::Int = 1000,
                       w::Float64 = 0.3, c1::Float64 = 0.5, c2::Float64 = 0.3,
                       ph::Float64 = 0.1, pm::Float64 = 0.3)
    
    # Start timing the algorithm execution
    start_time = time()
    
    # Determine accessible customers using L-infinity distance instead of cost matrix
    accessible_customers = Int[]
    for i in 1:length(customers)
        # Calculate L-infinity distance (minimum hops) to visit customer and return
        customer_coord = customers[i]
        depot_x, depot_y = depot_coord
        customer_x, customer_y = customer_coord
        
        # L-infinity distance: max(|x1-x2|, |y1-y2|)
        distance_to = max(abs(customer_x - depot_x), abs(customer_y - depot_y))
        distance_from = max(abs(customer_x - depot_x), abs(customer_y - depot_y))  # Same for return
        
        # Check if customer can be visited and returned within battery limit
        total_distance = distance_to + distance_from
        if total_distance <= max_battery_time
            push!(accessible_customers, i)
        end
    end
    
    # Initialize PSO
    pso = PSOiA_TOP(
        Particle[], Int[], -Inf, swarm_size, max_iterations,
        w, c1, c2, ph, pm, n_drones, max_battery_time,
        customers, profits, costs, accessible_customers, depot_coord
    )
    
    # println("=== PSO SETUP ===")
    # println("Total customers: $(length(customers))")
    # println("Accessible customers: $(length(accessible_customers))")
    # println("Max battery time: $max_battery_time")
    # println("Number of drones: $n_drones")
    # println("==================")
    
    # Initialize and evaluate each particle in swarm (see Section 2.3)
    initialize_swarm(pso)

    iter = 1
    itermax = max_iterations #* length(accessible_customers) * n_drones  # As mentioned in paper
    
    # println("Starting PSO with $(pso.swarm_size) particles, initial best: $(pso.global_best_profit)")
    # Main algorithm loop following Algorithm 1
    while iter <= itermax
        improvement_found = false
        
        for x in 1:pso.swarm_size
            # Random move with probability ph
            if rand() < pso.ph
                # Move S[x] to a new position (see Section 2.3)
                pso.swarm[x].position = idch_heuristic(pso, false)  # Fast version
            else
                # Update S[x].pos (see Section 2.5)
                update_position!(pso.swarm[x], pso.global_best, pso)
            end
            
            # Local search with probability pm
            if rand() < pso.pm
                # Apply local search on S[x].pos (see Section 2.4)
                local_search!(pso.swarm[x], pso)
            end
            
            # Evaluate S[x].pos (see Section 2.2)
            pso.swarm[x].current_profit = fast_split(pso.swarm[x].position, pso)
            
            # Update lbest of S (see Section 2.6)
            prev_global_best = pso.global_best_profit
            update_local_bests!(pso)

            # Check if update Rule 3 is applied (new local best discovered)
            if pso.global_best_profit > prev_global_best
                improvement_found = true
                println("Iter $iter: New best = $(round(pso.global_best_profit, digits=3)), Solution: $(pso.global_best)")
            end
        end

        if improvement_found
            iter = 1  # Reset counter when improvement found
        else
            iter += 1  # Increment counter when no improvement
        end
    end
    
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
function extract_routes(giant_tour::Vector{Int}, pso::PSOiA_TOP)
    # Use the proper split procedure from the paper
    optimal_profit, routes = fast_split_with_routes(giant_tour, pso)
    
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