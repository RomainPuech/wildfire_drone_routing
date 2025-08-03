"""
Test script for PSO-inspired algorithm for Team Orienteering Problem
Uses sample generation and plotting from TOP.jl
"""

include("TOP_PSO.jl")

# Test the PSO algorithm with the data from TOP.jl
function test_PSO_TOP()
    println("=== Testing PSO-inspired Algorithm for TOP ===")
    
    # Use the same data generation as in TOP.jl
    Random.seed!(42)
    n_drones = 2
    max_battery_time = 5
    N = 8
    M = 8
    
    # Generate charging station and ground stations
    ChargingStation = generate_random_charging_stations(N, M, 1)
    GroundStations = generate_random_ground_stations(N, M, 5)
    
    # Generate risk map (profits for customers)
    risk_pertime = rand(1, N, M)
    
    println("Grid size: $(N)x$(M)")
    println("Number of drones: $n_drones")
    println("Max battery time: $max_battery_time")
    println("Charging station: $(ChargingStation[1])")
    println("Ground stations: $GroundStations")
    
    # Prepare data for PSO algorithm
    # Convert coordinates to customer list
    customers = Tuple{Int,Int}[]
    profits = Float64[]
    
    # Add all grid points as potential customers (except charging station)
    for i in 1:N
        for j in 1:M
            if (i, j) != ChargingStation[1]  # Exclude charging station
                push!(customers, (i, j))
                push!(profits, risk_pertime[1, i, j])
            end
        end
    end
    
    # Create cost matrix (travel times between customers)
    costs = Dict{Tuple{Int,Int}, Float64}()
    n_customers = length(customers)
    
    # Add costs from depot to customers and back
    for i in 1:n_customers
        xi, yi = customers[i]
        
        # Distance from depot (charging station) to customer
        depot_x, depot_y = ChargingStation[1]
        inf_dist_from_depot = max(abs(xi - depot_x), abs(yi - depot_y))
        costs[(0, i)] = inf_dist_from_depot <= 1 ? 1.0 : max_battery_time * 4
        costs[(i, 0)] = costs[(0, i)]  # Symmetric
        
        # Distance between customers
        for j in 1:n_customers
            if i != j
                xj, yj = customers[j]
                inf_dist = max(abs(xi - xj), abs(yi - yj))
                costs[(i, j)] = inf_dist <= 1 ? 1.0 : max_battery_time * 4
            else
                costs[(i, j)] = 0.0
            end
        end
    end
    
    println("Number of potential customers: $n_customers")
    
    # Test different PSO configurations
    configurations = [
        (swarm_size=20, max_iterations=50, w=0.3, c1=0.5, c2=0.3, ph=0.1, pm=0.3),
        (swarm_size=30, max_iterations=100, w=0.4, c1=0.6, c2=0.4, ph=0.15, pm=0.4),
    ]
    
    best_overall_profit = -Inf
    best_configuration = nothing
    best_solution = nothing
    
    for (i, config) in enumerate(configurations)
        println("\n--- Configuration $i ---")
        println("Swarm size: $(config.swarm_size)")
        println("Max iterations: $(config.max_iterations)")  
        println("PSO parameters: w=$(config.w), c1=$(config.c1), c2=$(config.c2)")
        println("Probabilities: ph=$(config.ph), pm=$(config.pm)")
        
        # Run PSO algorithm
        start_time = time()
        giant_tour, profit, pso_obj = solve_PSO_TOP(
            customers, profits, costs, n_drones, max_battery_time;
            swarm_size=config.swarm_size,
            max_iterations=config.max_iterations,
            w=config.w, c1=config.c1, c2=config.c2,
            ph=config.ph, pm=config.pm
        )
        end_time = time()
        
        println("Solution found in $(round(end_time - start_time, digits=2)) seconds")
        println("Best profit: $profit")
        
        if profit > best_overall_profit
            best_overall_profit = profit
            best_configuration = i
            best_solution = (giant_tour, profit, pso_obj)
        end
        
        # Extract routes from giant tour
        routes = extract_routes(giant_tour, pso_obj)
        println("Routes extracted:")
        for (drone_id, route) in enumerate(routes)
            if !isempty(route)
                route_coords = [customers[idx] for idx in route]
                total_cost = calculate_route_cost(route, costs, 0)
                route_profit = sum(profits[idx] for idx in route)
                println("  Drone $drone_id: $route_coords (cost: $total_cost, profit: $route_profit)")
            end
        end
    end
    
    println("\n=== BEST CONFIGURATION RESULTS ===")
    println("Best configuration: #$best_configuration")
    println("Best profit: $best_overall_profit")
    
    if best_solution !== nothing
        giant_tour, profit, pso_obj = best_solution
        routes = extract_routes(giant_tour, pso_obj)
        
        println("\nFinal solution routes:")
        total_profit_check = 0.0
        for (drone_id, route) in enumerate(routes)
            if !isempty(route)
                route_coords = [customers[idx] for idx in route]
                total_cost = calculate_route_cost(route, costs, 0)
                route_profit = sum(profits[idx] for idx in route)
                total_profit_check += route_profit
                
                println("  Drone $drone_id:")
                println("    Route: Depot -> $(join(route_coords, " -> ")) -> Depot")
                println("    Cost: $total_cost / $max_battery_time")
                println("    Profit: $route_profit")
                println("    Feasible: $(total_cost <= max_battery_time)")
            end
        end
        println("Total profit check: $total_profit_check")
        
        # Compare with greedy solution from TOP.jl
        println("\n=== COMPARISON WITH GREEDY SOLUTION ===")
        
        # Prepare data for greedy comparison (need to convert to TOP.jl format)
        coords_for_greedy = deepcopy(customers)
        push!(coords_for_greedy, ChargingStation[1])  # Add depot at end
        push!(coords_for_greedy, ChargingStation[1])  # Add depot copy
        
        Begin_CS = length(coords_for_greedy) - 1
        End_CS = length(coords_for_greedy)
        
        # Create cost matrix for greedy (using indices)
        c_greedy = Dict{Tuple{Int,Int}, Float64}()
        for i in 1:length(coords_for_greedy), j in 1:length(coords_for_greedy)
            if i <= n_customers && j <= n_customers
                c_greedy[(i, j)] = get(costs, (i, j), max_battery_time * 4)
            elseif i <= n_customers && (j == Begin_CS || j == End_CS)
                c_greedy[(i, j)] = get(costs, (i, 0), max_battery_time * 4)
            elseif (i == Begin_CS || i == End_CS) && j <= n_customers
                c_greedy[(i, j)] = get(costs, (0, j), max_battery_time * 4)
            else
                c_greedy[(i, j)] = 0.0
            end
        end
        
        # Create risk matrix for greedy
        risk_for_greedy = zeros(1, length(coords_for_greedy))
        for i in 1:n_customers
            risk_for_greedy[1, i] = profits[i]
        end
        
        # Run greedy algorithm
        greedy_routes = greedy_TOP_multiple_drones(risk_for_greedy, coords_for_greedy, 
                                                  Begin_CS, End_CS, max_battery_time, 
                                                  n_drones, c_greedy)
        greedy_profit = compute_objective_greedy(greedy_routes, coords_for_greedy, 
                                               risk_for_greedy, Begin_CS, End_CS)
        
        println("Greedy solution profit: $greedy_profit")
        println("PSO solution profit: $best_overall_profit")
        improvement = best_overall_profit > greedy_profit ? 
                     round(((best_overall_profit - greedy_profit) / greedy_profit) * 100, digits=2) : 0.0
        println("PSO improvement over greedy: $(improvement)%")
    end
    
    return best_solution
end

"""
Calculate total cost of a route
"""
function calculate_route_cost(route::Vector{Int}, costs::Dict{Tuple{Int,Int}, Float64}, depot_idx::Int)
    if isempty(route)
        return 0.0
    end
    
    total_cost = 0.0
    
    # Cost from depot to first customer
    total_cost += get(costs, (depot_idx, route[1]), 0.0)
    
    # Cost between consecutive customers
    for i in 1:(length(route)-1)
        total_cost += get(costs, (route[i], route[i+1]), 0.0)
    end
    
    # Cost from last customer back to depot
    total_cost += get(costs, (route[end], depot_idx), 0.0)
    
    return total_cost
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_PSO_TOP()
end