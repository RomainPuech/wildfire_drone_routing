"""
Simple test for PSO-inspired algorithm for Team Orienteering Problem
"""

using Random

# Minimal version without graphics
include("TOP_PSO.jl")

function simple_test()
    println("=== Simple PSO Test ===")
    
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
        swarm_size=10, max_iterations=20,
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