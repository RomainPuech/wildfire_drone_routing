"""
Standalone test for PSO-inspired algorithm for Team Orienteering Problem
Includes only the necessary functions without graphics
"""

using Random
using DataStructures

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
end

"""
Fast split procedure based on interval graph model (O(m*n) complexity)
"""
function fast_split(permutation::Vector{Int}, pso::PSOiA_TOP)
    n = length(permutation)
    m = pso.n_drones
    L = pso.max_battery_time
    
    if n == 0
        return 0.0
    end
    
    # Calculate saturated tours P[i] and first successor succ[i]
    P = zeros(n)  # Profit of saturated tour starting at position i
    succ = zeros(Int, n)  # First successor of saturated tour starting at position i
    
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
                travel_cost = get(pso.costs, (0, customer_idx), L*4)
            else
                prev_customer = permutation[j-1]
                travel_cost = get(pso.costs, (prev_customer, customer_idx), L*4)
            end
            
            # Check if we can add this customer and still return to depot
            return_cost = get(pso.costs, (customer_idx, 0), L*4)
            
            if current_cost + travel_cost + return_cost > L
                break
            end
            
            current_cost += travel_cost
            current_profit += pso.profits[customer_idx]
            j += 1
        end
        
        P[i] = current_profit
        succ[i] = j
    end
    
    # Dynamic programming table Γ[i,j] = max profit using j drones from position i onwards
    Γ = zeros(n + 1, m + 1)
    
    # Fill DP table in reverse order
    for i in n:-1:1
        for j in 0:m
            if j == 0
                Γ[i, j + 1] = 0.0  # No drones left, no profit
            else
                # Option 1: Don't use tour starting at i
                Γ[i, j + 1] = Γ[i + 1, j + 1]
                
                # Option 2: Use saturated tour starting at i (if we have remaining drones)
                if succ[i] <= n + 1
                    profit_with_tour = P[i] + Γ[succ[i], j]
                    Γ[i, j + 1] = max(Γ[i, j + 1], profit_with_tour)
                end
            end
        end
    end
    
    return Γ[1, m + 1]
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
                # Calculate insertion cost
                if pos == 1
                    # Insert at beginning
                    cost_before = 0.0
                    cost_after = pos <= length(solution) ? get(pso.costs, (customer, solution[pos]), 0.0) : 0.0
                elseif pos > length(solution)
                    # Insert at end
                    cost_before = get(pso.costs, (solution[end], customer), 0.0)
                    cost_after = 0.0
                else
                    # Insert in middle
                    cost_before = get(pso.costs, (solution[pos-1], customer), 0.0)
                    cost_after = get(pso.costs, (customer, solution[pos]), 0.0)
                    # Remove old direct cost
                    cost_before -= get(pso.costs, (solution[pos-1], solution[pos]), 0.0)
                end
                
                insertion_cost = cost_before + cost_after - customer_profit^α
                
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
Iterative Destruction/Construction Heuristic (IDCH)
"""
function idch_heuristic(pso::PSOiA_TOP, slow_version::Bool = false)
    max_iter = slow_version ? length(pso.accessible_customers)^2 : length(pso.accessible_customers)
    
    # Start with random permutation
    current_solution = shuffle(pso.accessible_customers)
    best_solution = copy(current_solution)
    best_profit = fast_split(best_solution, pso)
    
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
        profit = fast_split(reconstructed, pso)
        
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
    
    # Extract subsequences
    M = Set{Int}()  # Marked customers
    new_position = Int[]
    
    # Phase 1: Extract from current position
    if n_current > 0
        extracted = extract_subsequence(particle.position, n_current, M)
        append!(new_position, extracted)
    end
    
    # Phase 2: Extract from local best
    if n_local > 0
        extracted = extract_subsequence(particle.local_best, n_local, M)
        append!(new_position, extracted)
    end
    
    # Phase 3: Extract from global best
    if n_global > 0
        extracted = extract_subsequence(global_best, n_global, M)
        append!(new_position, extracted)
    end
    
    # Add remaining customers randomly
    remaining = setdiff(pso.accessible_customers, M)
    append!(new_position, shuffle(remaining))
    
    particle.position = new_position
    particle.current_profit = fast_split(new_position, pso)
end

"""
Local search with shift operator only (simplified)
"""
function local_search!(particle::Particle, pso::PSOiA_TOP)
    improved = true
    max_attempts = 10  # Limit to prevent infinite loops
    attempts = 0
    
    while improved && attempts < max_attempts
        improved = false
        attempts += 1
        
        n = length(particle.position)
        positions = shuffle(1:n)  # Random order evaluation
        
        for i in positions[1:min(5, length(positions))]  # Limit to first 5 positions
            for j in shuffle(setdiff(1:n, [i]))[1:min(3, n-1)]  # Limit to first 3 alternatives
                # Try moving customer from position i to position j
                new_position = copy(particle.position)
                customer = new_position[i]
                deleteat!(new_position, i)
                insert!(new_position, j > i ? j-1 : j, customer)
                
                new_profit = fast_split(new_position, pso)
                if new_profit > particle.current_profit
                    particle.position = new_position
                    particle.current_profit = new_profit
                    improved = true
                    break
                end
            end
            if improved
                break
            end
        end
    end
end

"""
Update local best positions
"""
function update_local_bests!(pso::PSOiA_TOP)
    for particle in pso.swarm
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
        end
    end
    
    # Initialize some particles with IDCH heuristic (better quality)
    n_idch = min(2, pso.swarm_size ÷ 2)  # Reduced for speed
    for i in 1:n_idch
        position = idch_heuristic(pso, false)  # Fast version
        profit = fast_split(position, pso)
        
        pso.swarm[i].position = copy(position)
        pso.swarm[i].local_best = copy(position)
        pso.swarm[i].local_best_profit = profit
        pso.swarm[i].current_profit = profit
        
        if profit > pso.global_best_profit
            pso.global_best = copy(position)
            pso.global_best_profit = profit
        end
    end
end

"""
Main PSO algorithm - simplified version
"""
function solve_PSO_TOP(customers::Vector{Tuple{Int,Int}}, profits::Vector{Float64}, 
                       costs::Dict{Tuple{Int,Int}, Float64}, n_drones::Int, 
                       max_battery_time::Int; swarm_size::Int = 10, max_iterations::Int = 20,
                       w::Float64 = 0.3, c1::Float64 = 0.5, c2::Float64 = 0.3,
                       ph::Float64 = 0.1, pm::Float64 = 0.3)
    
    # Determine accessible customers
    accessible_customers = Int[]
    for i in 1:length(customers)
        # Check if customer can be visited and returned within battery limit
        cost_to = get(costs, (0, i), max_battery_time * 4)  # From depot
        cost_from = get(costs, (i, 0), max_battery_time * 4)  # To depot
        if cost_to + cost_from <= max_battery_time
            push!(accessible_customers, i)
        end
    end
    
    println("Accessible customers: $(length(accessible_customers)) out of $(length(customers))")
    
    # Initialize PSO
    pso = PSOiA_TOP(
        Particle[], Int[], -Inf, swarm_size, max_iterations,
        w, c1, c2, ph, pm, n_drones, max_battery_time,
        customers, profits, costs, accessible_customers
    )
    
    # Initialize swarm
    initialize_swarm(pso)
    
    iter = 1
    itermax = max_iterations
    
    println("Starting PSO with $(pso.swarm_size) particles")
    println("Initial best profit: $(pso.global_best_profit)")
    
    # Main algorithm loop
    while iter <= itermax
        improvement_found = false
        
        for x in 1:pso.swarm_size
            # Random move with probability ph
            if rand() < pso.ph
                pso.swarm[x].position = idch_heuristic(pso, false)  # Fast version
            else
                # Update position
                update_position!(pso.swarm[x], pso.global_best, pso)
            end
            
            # Local search with probability pm
            if rand() < pso.pm
                local_search!(pso.swarm[x], pso)
            end
            
            # Evaluate position
            pso.swarm[x].current_profit = fast_split(pso.swarm[x].position, pso)
            
            # Update local bests
            prev_global_best = pso.global_best_profit
            update_local_bests!(pso)
            
            # Check if improvement found
            if pso.global_best_profit > prev_global_best
                improvement_found = true
                println("Iteration $iter: New best profit = $(pso.global_best_profit)")
            end
        end
        
        iter += 1
    end
    
    println("Final best profit: $(pso.global_best_profit)")
    return pso.global_best, pso.global_best_profit, pso
end

"""
Extract routes from giant tour
"""
function extract_routes(giant_tour::Vector{Int}, pso::PSOiA_TOP)
    routes = Vector{Vector{Int}}()
    
    if isempty(giant_tour)
        return routes
    end
    
    current_route = Int[]
    current_cost = 0.0
    L = pso.max_battery_time
    m = pso.n_drones
    
    for customer_idx in giant_tour
        # Calculate cost to add this customer
        additional_cost = 0.0
        if isempty(current_route)
            additional_cost = get(pso.costs, (0, customer_idx), L*4)  # From depot
        else
            additional_cost = get(pso.costs, (current_route[end], customer_idx), L*4)
        end
        
        # Check if we can add this customer and still return
        return_cost = get(pso.costs, (customer_idx, 0), L*4)
        
        if current_cost + additional_cost + return_cost <= L && length(routes) < m
            push!(current_route, customer_idx)
            current_cost += additional_cost
        else
            # Start new route if we have remaining drones
            if !isempty(current_route)
                push!(routes, copy(current_route))
            end
            
            if length(routes) < m
                current_route = [customer_idx]
                current_cost = get(pso.costs, (0, customer_idx), L*4)
            else
                break  # No more drones available
            end
        end
    end
    
    # Add last route if not empty
    if !isempty(current_route) && length(routes) < m
        push!(routes, current_route)
    end
    
    return routes
end

# Simple test function
function simple_test()
    println("=== Standalone PSO Test ===")
    
    Random.seed!(42)
    
    # Simple 4x4 grid test case
    customers = [(1,1), (1,2), (2,1), (2,2), (3,3), (4,4)]
    profits = [10.0, 15.0, 8.0, 12.0, 20.0, 25.0]
    
    # Simple cost matrix (Manhattan distance)
    costs = Dict{Tuple{Int,Int}, Float64}()
    depot_pos = (2, 2)  # Center position
    
    n_customers = length(customers)
    
    # Costs from depot
    for i in 1:n_customers
        x, y = customers[i]
        dist = abs(x - depot_pos[1]) + abs(y - depot_pos[2])
        costs[(0, i)] = Float64(dist)
        costs[(i, 0)] = Float64(dist)
    end
    
    # Costs between customers
    for i in 1:n_customers
        for j in 1:n_customers
            if i != j
                x1, y1 = customers[i]
                x2, y2 = customers[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                costs[(i, j)] = Float64(dist)
            else
                costs[(i, j)] = 0.0
            end
        end
    end
    
    println("Customers: $customers")
    println("Profits: $profits")
    println("Testing with 2 drones, battery limit 10")
    
    # Run PSO with small parameters for quick test
    start_time = time()
    giant_tour, profit, pso_obj = solve_PSO_TOP(
        customers, profits, costs, 2, 10;
        swarm_size=5, max_iterations=10,
        w=0.3, c1=0.5, c2=0.3, ph=0.1, pm=0.3
    )
    end_time = time()
    
    println("Solution found in $(round(end_time - start_time, digits=2)) seconds")
    println("Best giant tour: $giant_tour")
    println("Best profit: $profit")
    
    # Extract and display routes
    routes = extract_routes(giant_tour, pso_obj)
    println("\nExtracted routes:")
    total_check = 0.0
    for (drone_id, route) in enumerate(routes)
        if !isempty(route)
            route_coords = [customers[idx] for idx in route]
            route_profit = sum(profits[idx] for idx in route)
            total_check += route_profit
            println("  Drone $drone_id: Depot -> $(join(route_coords, " -> ")) -> Depot (profit: $route_profit)")
        else
            println("  Drone $drone_id: No route (stays at depot)")
        end
    end
    println("Total profit check: $total_check")
    
    return giant_tour, profit
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    simple_test()
end