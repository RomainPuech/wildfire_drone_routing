"""
Realistic test for PSO-inspired algorithm for Team Orienteering Problem
Uses the data generation approach from TOP.jl but without graphics
"""

using Random
using DataStructures

# Include our PSO implementation
include("TOP_PSO.jl")

# Helper functions from TOP.jl (recreated to avoid graphics)
function generate_random_charging_stations(N::Int, M::Int, num_stations::Int)
    selected = rand(1:N*M, num_stations)
    return [(div(i-1, M)+1, mod(i-1, M)+1) for i in selected]
end

function generate_random_ground_stations(N::Int, M::Int, num_stations::Int)
    selected = rand(1:N*M, num_stations)
    return [(div(i-1, M)+1, mod(i-1, M)+1) for i in selected]
end

function get_drone_gridpoints(charging_stations, max_range, all_points)
    reachable = Set{Tuple{Int,Int}}()
    
    for cs in charging_stations
        for point in all_points
            # Use infinity norm (Chebyshev distance)
            dist = max(abs(point[1] - cs[1]), abs(point[2] - cs[2]))
            if dist <= max_range
                push!(reachable, point)
            end
        end
    end
    
    return reachable
end

function test_PSO_realistic()
    println("=== Testing PSO-inspired Algorithm for TOP (Realistic) ===")
    
    # Use the same data generation as in TOP.jl
    Random.seed!(42)
    n_drones = 2
    max_battery_time = 5  # Reduced from 15 to make problem more constrained
    N = 8
    M = 8
    
    # Generate charging station and ground stations
    ChargingStation = generate_random_charging_stations(N, M, 1)
    GroundStations = generate_random_ground_stations(N, M, 5)
    
    # Generate risk map (profits for customers)
    risk_pertime = rand(1, N, M)  # 1 time step, values between 0 and 1
    
    println("Grid size: $(N)x$(M)")
    println("Number of drones: $n_drones")
    println("Max battery time: $max_battery_time")
    println("Charging station: $(ChargingStation[1])")
    println("Ground stations: $GroundStations")
    
    # Process data similar to TOP.jl
    H, N, M = size(risk_pertime)
    
    # Define grid points and drone-accessible points
    I = [(x, y) for x in 1:N for y in 1:M] # All feasible grid points
    GridpointsDrones_set = get_drone_gridpoints(ChargingStation, floor(max_battery_time/2), I)
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStation)
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))
    
    println("Total grid points: $(length(I))")
    println("Drone-accessible points: $(length(GridpointsDrones_set))")
    println("Detecting points (excl. charging): $(length(GridpointsDronesDetecting))")
    
    # Prepare data for PSO algorithm
    customers = GridpointsDronesDetecting
    profits = Float64[]
    
    # Extract profits for accessible customers
    for (x, y) in customers
        push!(profits, risk_pertime[1, x, y])
    end
    
    # Create cost matrix (travel times between customers) - using infinity norm
    costs = Dict{Tuple{Int,Int}, Float64}()
    n_customers = length(customers)
    
    # Add costs from depot to customers and back
    depot_x, depot_y = ChargingStation[1]
    for i in 1:n_customers
        xi, yi = customers[i]
        
        # Distance from depot (charging station) to customer
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
    println("Total profit available: $(round(sum(profits), digits=2))")
    
    # Create a greedy baseline for comparison
    println("\n=== Greedy Baseline ===")
    greedy_profit = greedy_baseline(customers, profits, costs, n_drones, max_battery_time)
    println("Greedy solution profit: $(round(greedy_profit, digits=3))")
    
    # Test PSO with multiple configurations
    println("\n=== PSO Configurations ===")
    configurations = [
        (name="Small Fast", swarm_size=10, max_iterations=20, w=0.3, c1=0.5, c2=0.3, ph=0.1, pm=0.2),
        (name="Medium", swarm_size=20, max_iterations=50, w=0.4, c1=0.6, c2=0.4, ph=0.15, pm=0.3),
        (name="Large Thorough", swarm_size=30, max_iterations=100, w=0.3, c1=0.5, c2=0.3, ph=0.2, pm=0.4),
    ]
    
    best_overall_profit = -Inf
    best_configuration = nothing
    best_solution = nothing
    
    for (i, config) in enumerate(configurations)
        println("\n--- Configuration $(i): $(config.name) ---")
        println("Swarm size: $(config.swarm_size), Max iterations: $(config.max_iterations)")  
        println("PSO params: w=$(config.w), c1=$(config.c1), c2=$(config.c2)")
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
        
        execution_time = round(end_time - start_time, digits=2)
        println("Execution time: $(execution_time) seconds")
        println("Final profit: $(round(profit, digits=3))")
        
        # Calculate improvement over greedy
        improvement = greedy_profit > 0 ? round(((profit - greedy_profit) / greedy_profit) * 100, digits=1) : 0.0
        println("Improvement over greedy: $(improvement)%")
        
        if profit > best_overall_profit
            best_overall_profit = profit
            best_configuration = config.name
            best_solution = (giant_tour, profit, pso_obj)
        end
        
        # Extract and analyze routes
        routes = extract_routes(giant_tour, pso_obj)
        println("Number of routes: $(length(routes))")
        
        total_route_profit = 0.0
        for (drone_id, route) in enumerate(routes)
            if !isempty(route)
                route_coords = [customers[idx] for idx in route]
                total_cost = calculate_route_cost(route, costs, 0)
                route_profit = sum(profits[idx] for idx in route)
                total_route_profit += route_profit
                feasible = total_cost <= max_battery_time
                
                println("  Drone $drone_id: $(length(route)) customers, cost: $(round(total_cost, digits=1))/$max_battery_time, profit: $(round(route_profit, digits=3)), feasible: $feasible")
            else
                println("  Drone $drone_id: No route (stays at depot)")
            end
        end
        println("Total route profit check: $(round(total_route_profit, digits=3))")
        
        # Efficiency metrics
        profit_per_second = profit / execution_time
        println("Efficiency: $(round(profit_per_second, digits=2)) profit/second")
    end
    
    println("\n=== FINAL RESULTS ===")
    println("Best configuration: $best_configuration")
    println("Best profit: $(round(best_overall_profit, digits=3))")
    println("Greedy baseline: $(round(greedy_profit, digits=3))")
    
    if best_overall_profit > greedy_profit
        improvement = round(((best_overall_profit - greedy_profit) / greedy_profit) * 100, digits=1)
        println("PSO improvement: $(improvement)%")
    else
        println("PSO did not improve over greedy baseline")
    end
    
    if best_solution !== nothing
        giant_tour, profit, pso_obj = best_solution
        routes = extract_routes(giant_tour, pso_obj)
        
        println("\nBest solution details:")
        println("Giant tour: $giant_tour")
        for (drone_id, route) in enumerate(routes)
            if !isempty(route)
                route_coords = [customers[idx] for idx in route]
                total_cost = calculate_route_cost(route, costs, 0)
                route_profit = sum(profits[idx] for idx in route)
                
                println("  Drone $drone_id:")
                println("    Route: Depot$(ChargingStation[1]) -> $(join(route_coords, " -> ")) -> Depot$(ChargingStation[1])")
                println("    Cost: $(round(total_cost, digits=1)) / $max_battery_time")
                println("    Profit: $(round(route_profit, digits=3))")
                println("    Customers: $(length(route))")
            end
        end
    end
    
    return best_solution
end

"""
Simple greedy baseline for comparison
"""
function greedy_baseline(customers::Vector{Tuple{Int,Int}}, profits::Vector{Float64}, 
                        costs::Dict{Tuple{Int,Int}, Float64}, n_drones::Int, max_battery_time::Int)
    
    visited = Set{Int}()
    total_profit = 0.0
    
    for drone in 1:n_drones
        current_cost = 0.0
        
        while true
            best_customer = -1
            best_ratio = -Inf
            best_cost = Inf
            
            # Find best profit/cost ratio customer
            for i in 1:length(customers)
                if i in visited
                    continue
                end
                
                # Cost to visit this customer and return to depot
                cost_to = get(costs, (0, i), max_battery_time * 4)
                cost_from = get(costs, (i, 0), max_battery_time * 4)
                total_cost_needed = current_cost + cost_to + cost_from
                
                if total_cost_needed <= max_battery_time
                    ratio = profits[i] / cost_to
                    if ratio > best_ratio
                        best_ratio = ratio
                        best_customer = i
                        best_cost = cost_to
                    end
                end
            end
            
            if best_customer == -1
                break  # No more feasible customers
            end
            
            # Visit best customer
            push!(visited, best_customer)
            total_profit += profits[best_customer]
            current_cost += best_cost
        end
    end
    
    return total_profit
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
    test_PSO_realistic()
end