"""
Test script to verify PSO runs for more iterations and finds better solutions
"""

include("TOP_PSO.jl")

function test_longer_PSO()
    println("=== Testing Longer PSO Execution ===")
    
    Random.seed!(42)
    
    # Create a larger problem to see the difference
    customers = [(i, j) for i in 1:6 for j in 1:6 if (i, j) != (3, 3)]  # 6x6 grid minus depot
    depot_coord = (3, 3)  # Center depot
    
    # Generate profits
    profits = [rand() * 10 for _ in customers]
    
    # Create proper grid connectivity cost matrix
    costs = Dict{Tuple{Int,Int}, Float64}()
    n_customers = length(customers)
    max_battery_time = 12
    
    # Costs from depot to customers and back
    for i in 1:n_customers
        xi, yi = customers[i]
        inf_dist_from_depot = max(abs(xi - depot_coord[1]), abs(yi - depot_coord[2]))
        if inf_dist_from_depot <= 1
            costs[(0, i)] = 1.0
            costs[(i, 0)] = 1.0
        else
            costs[(0, i)] = max_battery_time * 4
            costs[(i, 0)] = max_battery_time * 4
        end
    end
    
    # Costs between customers - only neighbors
    for i in 1:n_customers
        for j in 1:n_customers
            if i != j
                xi, yi = customers[i]
                xj, yj = customers[j]
                inf_dist = max(abs(xi - xj), abs(yi - yj))
                if inf_dist <= 1
                    costs[(i, j)] = 1.0
                else
                    costs[(i, j)] = max_battery_time * 4
                end
            else
                costs[(i, j)] = 0.0
            end
        end
    end
    
    println("Problem size: $(length(customers)) customers")
    println("Total profit available: $(round(sum(profits), digits=2))")
    
    # Test 1: Short PSO run (like before)
    println("\n--- Short PSO (10 iterations) ---")
    start_time = time()
    giant_tour_short, profit_short, pso_obj_short = solve_PSO_TOP(
        customers, profits, costs, 2, max_battery_time;
        swarm_size=15, max_iterations=10,
        w=0.3, c1=0.5, c2=0.3, ph=0.15, pm=0.3
    )
    short_time = time() - start_time
    
    routes_short = extract_routes(giant_tour_short, pso_obj_short)
    println("Short PSO result: $(round(profit_short, digits=3)) in $(round(short_time, digits=2))s")
    
    # Test 2: Longer PSO run
    println("\n--- Longer PSO (50 iterations) ---")
    start_time = time()
    giant_tour_long, profit_long, pso_obj_long = solve_PSO_TOP(
        customers, profits, costs, 2, max_battery_time;
        swarm_size=20, max_iterations=50,
        w=0.3, c1=0.5, c2=0.3, ph=0.15, pm=0.4
    )
    long_time = time() - start_time
    
    routes_long = extract_routes(giant_tour_long, pso_obj_long)
    println("Long PSO result: $(round(profit_long, digits=3)) in $(round(long_time, digits=2))s")
    
    # Compare results
    println("\n=== COMPARISON ===")
    println("Short PSO (10 iter): $(round(profit_short, digits=3))")
    println("Long PSO (50 iter):  $(round(profit_long, digits=3))")
    
    improvement = profit_long > profit_short ? round(((profit_long - profit_short) / profit_short) * 100, digits=1) : 0.0
    
    if profit_long > profit_short
        println("✅ Longer PSO improved by $(improvement)%")
    else
        println("⚖️  Both PSO runs found similar quality solutions")
    end
    
    # Show route details
    println("\nShort PSO routes:")
    for (i, route) in enumerate(routes_short)
        if !isempty(route)
            coords = [customers[idx] for idx in route]
            route_profit = sum(profits[idx] for idx in route)
            println("  Drone $i: $(length(route)) customers, profit $(round(route_profit, digits=2))")
            println("    Route: Depot$(depot_coord) -> $(join(coords, " -> ")) -> Depot$(depot_coord)")
        end
    end
    
    println("\nLong PSO routes:")
    for (i, route) in enumerate(routes_long)
        if !isempty(route)
            coords = [customers[idx] for idx in route]
            route_profit = sum(profits[idx] for idx in route)
            println("  Drone $i: $(length(route)) customers, profit $(round(route_profit, digits=2))")
            println("    Route: Depot$(depot_coord) -> $(join(coords, " -> ")) -> Depot$(depot_coord)")
        end
    end
    
    return profit_short, profit_long
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_longer_PSO()
end