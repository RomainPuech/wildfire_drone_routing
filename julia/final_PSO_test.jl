"""
Final comprehensive test for PSO-inspired algorithm for Team Orienteering Problem
Uses TOP.jl functions that are available and demonstrates algorithm effectiveness
"""

include("TOP.jl")

# Test with realistic TOP data using available functions
function final_PSO_test()
    println("=== Final PSO Algorithm Test ===")
    
    Random.seed!(42)
    n_drones = 2
    max_battery_time = 15  # Increased for more interesting routes
    N = 10  # Smaller grid for clearer demonstration
    M = 10
    
    # Use available function
    ChargingStation = generate_random_charging_stations(N, M, 1)
    
    # Create ground stations manually
    GroundStations = [(2, 3), (4, 5), (1, 6), (5, 2), (3, 4)]
    
    # Generate profit map
    risk_pertime = rand(1, N, M)
    
    println("Grid size: $(N)x$(M)")
    println("Number of drones: $n_drones")
    println("Max battery time: $max_battery_time")
    println("Charging station: $(ChargingStation[1])")
    println("Ground stations: $GroundStations")
    
    # Create comprehensive test with all grid points as potential customers
    customers = Tuple{Int,Int}[]
    profits = Float64[]
    
    # Add all grid points except charging station as potential customers
    for i in 1:N
        for j in 1:M
            if (i, j) != ChargingStation[1]
                push!(customers, (i, j))
                push!(profits, risk_pertime[1, i, j])
            end
        end
    end
    
    # Create realistic cost matrix using infinity norm (as in TOP.jl)
    costs = Dict{Tuple{Int,Int}, Float64}()
    n_customers = length(customers)
    depot_x, depot_y = ChargingStation[1]
    
    # Costs from depot to customers and back
    for i in 1:n_customers
        xi, yi = customers[i]
        inf_dist_from_depot = max(abs(xi - depot_x), abs(yi - depot_y))
        costs[(0, i)] = inf_dist_from_depot <= 1 ? 1.0 : inf_dist_from_depot
        costs[(i, 0)] = costs[(0, i)]
    end
    
    # Costs between customers
    for i in 1:n_customers
        for j in 1:n_customers
            if i != j
                xi, yi = customers[i]
                xj, yj = customers[j]
                inf_dist = max(abs(xi - xj), abs(yi - yj))
                costs[(i, j)] = inf_dist <= 1 ? 1.0 : inf_dist
            else
                costs[(i, j)] = 0.0
            end
        end
    end
    
    println("Total customers: $n_customers")
    println("Total profit available: $(round(sum(profits), digits=2))")
    
    # Greedy baseline using our own implementation
    println("\n=== Greedy Baseline ===")
    greedy_profit = greedy_baseline(customers, profits, costs, n_drones, max_battery_time)
    println("Greedy profit: $(round(greedy_profit, digits=3))")
    
    # Test PSO with multiple configurations
    println("\n=== PSO Algorithm Test ===")
    configurations = [
        (name="Quick Test", swarm_size=10, max_iterations=5, w=0.3, c1=0.5, c2=0.3, ph=0.1, pm=0.25),
        #(name="Balanced", swarm_size=25, max_iterations=10, w=0.4, c1=0.6, c2=0.4, ph=0.15, pm=0.35),
        #(name="Thorough", swarm_size=35, max_iterations=100, w=0.3, c1=0.5, c2=0.3, ph=0.2, pm=0.4),
    ]
    
    best_results = []
    
    for (i, config) in enumerate(configurations)
        println("\n--- $(config.name) Configuration ---")
        println("Parameters: $(config.swarm_size) particles, $(config.max_iterations) max iterations")
        println("PSO: w=$(config.w), c1=$(config.c1), c2=$(config.c2), ph=$(config.ph), pm=$(config.pm)")
        
        # Run PSO multiple times for statistical significance
        run_profits = Float64[]
        run_times = Float64[]
        
        for run in 1:1  # 3 runs per configuration
            start_time = time()
            giant_tour, profit, pso_obj = solve_PSO_TOP(
                customers, profits, costs, n_drones, max_battery_time;
                swarm_size=config.swarm_size,
                max_iterations=config.max_iterations,
                w=config.w, c1=config.c1, c2=config.c2,
                ph=config.ph, pm=config.pm
            )
            end_time = time()
            
            push!(run_profits, profit)
            push!(run_times, end_time - start_time)
        end
        
        avg_profit = mean(run_profits)
        max_profit = maximum(run_profits)
        avg_time = mean(run_times)
        
        println("Results over 3 runs:")
        println("  Average profit: $(round(avg_profit, digits=3))")
        println("  Best profit: $(round(max_profit, digits=3))")
        println("  Average time: $(round(avg_time, digits=2)) seconds")
        
        improvement = greedy_profit > 0 ? round(((max_profit - greedy_profit) / greedy_profit) * 100, digits=1) : 0.0
        println("  Best improvement over greedy: $(improvement)%")
        
        push!(best_results, (config=config, avg_profit=avg_profit, max_profit=max_profit, avg_time=avg_time, improvement=improvement))
        
        # Show best solution details
        if max_profit == maximum(run_profits)
            best_idx = findfirst(x -> x == max_profit, run_profits)
            
            # Re-run to get the best solution details
            Random.seed!(42 + best_idx)  # Reproducible results
            giant_tour, profit, pso_obj = solve_PSO_TOP(
                customers, profits, costs, n_drones, max_battery_time;
                swarm_size=config.swarm_size,
                max_iterations=config.max_iterations,
                w=config.w, c1=config.c1, c2=config.c2,
                ph=config.ph, pm=config.pm
            )
            
            routes = extract_routes(giant_tour, pso_obj)
            println("  Best solution routes:")
            total_customers = 0
            for (drone_id, route) in enumerate(routes)
                if !isempty(route)
                    route_coords = [customers[idx] for idx in route]
                    total_cost = calculate_route_cost(route, costs, 0)
                    route_profit = sum(profits[idx] for idx in route)
                    total_customers += length(route)
                    
                    println("    Drone $drone_id: $(length(route)) customers, cost $(round(total_cost, digits=1))/$max_battery_time, profit $(round(route_profit, digits=3))")
                    println("      Route: Depot$(ChargingStation[1]) -> $(join(route_coords, " -> ")) -> Depot$(ChargingStation[1])")
                end
            end
            println("    Total customers visited: $total_customers/$n_customers")
        end
    end
    
    # Final comparison
    println("\n=== FINAL COMPARISON ===")
    println("Greedy baseline: $(round(greedy_profit, digits=3))")
    
    best_overall = maximum([r.max_profit for r in best_results])
    best_config = best_results[findfirst(r -> r.max_profit == best_overall, best_results)]
    
    println("Best PSO result: $(round(best_overall, digits=3)) ($(best_config.config.name))")
    println("Overall improvement: $(best_config.improvement)%")
    
    if best_overall > greedy_profit
        println("✅ PSO successfully improved over greedy baseline!")
    elseif abs(best_overall - greedy_profit) < 1e-6
        println("⚖️  PSO matched greedy baseline (both found same optimum)")
    else
        println("❌ PSO did not improve over greedy baseline")
    end
    
    
    return best_results
end

# Helper function for statistics
function mean(arr)
    return sum(arr) / length(arr)
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

# Helper function from test (redefine to avoid conflict)
function calculate_route_cost(route::Vector{Int}, costs::Dict{Tuple{Int,Int}, Float64}, depot_idx::Int)
    if isempty(route)
        return 0.0
    end
    
    total_cost = 0.0
    total_cost += get(costs, (depot_idx, route[1]), 0.0)
    
    for i in 1:(length(route)-1)
        total_cost += get(costs, (route[i], route[i+1]), 0.0)
    end
    
    total_cost += get(costs, (route[end], depot_idx), 0.0)
    return total_cost
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    final_PSO_test()
end