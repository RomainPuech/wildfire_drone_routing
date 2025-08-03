using Random
using Printf

# Include both implementations
include("TOP_PSO.jl")
include("Improved_TOP_PSO.jl")

# Rename functions to avoid conflicts
const OriginalPSO = Main
const ImprovedPSO = Main

"""
Simple benchmark function replacement for @belapsed
"""
function time_function(f, args...; samples=5)
    # Warm up
    f(args...)
    
    times = Float64[]
    for _ in 1:samples
        t = @elapsed f(args...)
        push!(times, t)
    end
    
    return minimum(times)  # Return minimum time like @belapsed
end

"""
Generate test instance for TOP with proper grid movement costs
"""
function generate_test_instance(n_customers::Int, grid_size::Int, seed::Int = 42)
    Random.seed!(seed)
    
    # Generate customer positions on a grid
    customers = [(rand(1:grid_size), rand(1:grid_size)) for _ in 1:n_customers]
    
    # Generate random profits (higher profits for customers further from depot)
    depot = (grid_size ÷ 2, grid_size ÷ 2)
    profits = Float64[]
    
    for (x, y) in customers
        # Distance-based profit with some randomness
        dist_from_depot = max(abs(x - depot[1]), abs(y - depot[2]))
        base_profit = 10.0 + dist_from_depot * 5.0
        profit = base_profit + rand() * 10.0  # Add randomness
        push!(profits, profit)
    end
    
    # Generate PROPER GRID cost matrix: 1 for L-inf neighbors, infinity otherwise
    costs = Dict{Tuple{Int,Int}, Float64}()
    LARGE_COST = 1000.0  # Use large cost instead of Inf for numerical stability
    
    """
    Check if two points are L-infinity neighbors (distance ≤ 1)
    """
    function are_neighbors(pos1::Tuple{Int,Int}, pos2::Tuple{Int,Int})
        x1, y1 = pos1
        x2, y2 = pos2
        return max(abs(x2 - x1), abs(y2 - y1)) <= 1
    end
    
    # Costs from depot (customer 0) to all customers
    for i in 1:n_customers
        customer_pos = customers[i]
        if are_neighbors(depot, customer_pos)
            costs[(0, i)] = 1.0
            costs[(i, 0)] = 1.0
        else
            costs[(0, i)] = LARGE_COST
            costs[(i, 0)] = LARGE_COST
        end
    end
    
    # Costs between customers
    for i in 1:n_customers
        for j in 1:n_customers
            if i != j
                pos_i = customers[i]
                pos_j = customers[j]
                if are_neighbors(pos_i, pos_j)
                    costs[(i, j)] = 1.0
                else
                    costs[(i, j)] = LARGE_COST
                end
            end
        end
    end
    
    return customers, profits, costs, depot
end

"""
Test a single instance with both implementations
"""
function test_single_instance(customers, profits, costs, depot, n_drones, max_battery_time; 
                             max_iterations=100, swarm_size=20, test_name="")
    
    println("=" ^ 60)
    println("Testing: $test_name")
    println("Customers: $(length(customers)), Drones: $n_drones, Battery: $max_battery_time")
    println("Grid movement: cost=1 for neighbors, cost=∞ for non-neighbors")
    println("=" ^ 60)
    
    # Test original implementation
    println("\n🔄 Running Original Implementation...")
    original_time = @elapsed begin
        original_best, original_profit, original_pso = OriginalPSO.solve_PSO_TOP(
            customers, profits, costs, n_drones, max_battery_time, depot;
            swarm_size=swarm_size, max_iterations=max_iterations,
            w=0.3, c1=0.5, c2=0.3, ph=0.1, pm=0.3
        )
    end
    
    # Extract routes for original
    original_routes = OriginalPSO.extract_routes(original_best, original_pso)
    
    println("\n🚀 Running Improved Implementation...")
    improved_time = @elapsed begin
        improved_best, improved_profit, improved_pso = ImprovedPSO.solve_PSO_TOP(
            customers, profits, costs, n_drones, max_battery_time, depot;
            swarm_size=swarm_size, max_iterations=max_iterations,
            w=0.3, c1=0.5, c2=0.3, ph=0.1, pm=0.3
        )
    end
    
    # Extract routes for improved
    improved_routes = ImprovedPSO.extract_routes(improved_best, improved_pso)
    
    # Compare results
    println("\n📊 COMPARISON RESULTS")
    println("-" ^ 40)
    @printf "Original Profit:     %.3f\n" original_profit
    @printf "Improved Profit:     %.3f\n" improved_profit
    @printf "Profit Difference:   %.6f\n" abs(original_profit - improved_profit)
    println()
    @printf "Original Time:       %.3f seconds\n" original_time
    @printf "Improved Time:       %.3f seconds\n" improved_time
    @printf "Speedup:             %.2fx\n" (original_time / improved_time)
    println()
    
    # Verify solution quality
    profit_diff = abs(original_profit - improved_profit)
    if profit_diff < 1e-6
        println("✅ PROFIT MATCH: Solutions have identical profit")
    elseif profit_diff < 0.01
        println("⚠️  PROFIT CLOSE: Solutions have very similar profit (diff < 0.01)")
    else
        println("❌ PROFIT DIFFER: Solutions have different profits!")
    end
    
    # Verify route feasibility and show costs
    println("\n🛣️  ROUTE ANALYSIS")
    println("Original routes: $(length(original_routes)) routes")
    for (i, route) in enumerate(original_routes)
        if !isempty(route)
            route_profit = sum(profits[c] for c in route)
            # Calculate actual grid cost
            route_cost = 0
            if !isempty(route)
                route_cost += costs[(0, route[1])]  # depot to first
                for j in 1:(length(route)-1)
                    route_cost += costs[(route[j], route[j+1])]  # between customers
                end
                route_cost += costs[(route[end], 0)]  # last to depot
            end
            println("  Route $i: $(length(route)) customers, profit: $(round(route_profit, digits=2)), cost: $route_cost")
        end
    end
    
    println("Improved routes: $(length(improved_routes)) routes")
    for (i, route) in enumerate(improved_routes)
        if !isempty(route)
            route_profit = sum(profits[c] for c in route)
            # Calculate actual grid cost
            route_cost = 0
            if !isempty(route)
                route_cost += costs[(0, route[1])]  # depot to first
                for j in 1:(length(route)-1)
                    route_cost += costs[(route[j], route[j+1])]  # between customers
                end
                route_cost += costs[(route[end], 0)]  # last to depot
            end
            println("  Route $i: $(length(route)) customers, profit: $(round(route_profit, digits=2)), cost: $route_cost")
        end
    end
    
    return (
        original_profit=original_profit,
        improved_profit=improved_profit,
        original_time=original_time,
        improved_time=improved_time,
        speedup=original_time/improved_time,
        profit_diff=profit_diff
    )
end

"""
Run comprehensive test suite
"""
function run_test_suite()
    println("🧪 PSO-iA Implementation Comparison Test Suite")
    println("Testing Original vs Improved Fast Split Algorithm")
    println("Using proper grid movement costs: 1 for neighbors, ∞ for non-neighbors")
    println()
    
    # Test cases with increasing complexity
    test_cases = [
        (n_customers=10, grid_size=10, n_drones=2, max_battery=8, name="Small Instance"),
        (n_customers=20, grid_size=15, n_drones=3, max_battery=12, name="Medium Instance"),
        (n_customers=30, grid_size=20, n_drones=3, max_battery=15, name="Large Instance"),
        (n_customers=100, grid_size=10, n_drones=4, max_battery=20, name="Very Large Instance"),
    ]
    
    results = []
    
    for (i, test_case) in enumerate(test_cases)
        customers, profits, costs, depot = generate_test_instance(
            test_case.n_customers, test_case.grid_size, 42 + i
        )
        
        result = test_single_instance(
            customers, profits, costs, depot, 
            test_case.n_drones, test_case.max_battery;
            max_iterations=50,  # Reduced for faster testing
            swarm_size=15,      # Reduced for faster testing
            test_name=test_case.name
        )
        
        push!(results, (name=test_case.name, result=result))
        
        # Small delay between tests
        sleep(1)
    end
    
    # Summary
    println("\n" * "=" ^ 80)
    println("📈 SUMMARY OF ALL TESTS")
    println("=" ^ 80)
    
    total_speedup = 0.0
    max_profit_diff = 0.0
    
    for (test_name, result) in results
        @printf "%-20s | Speedup: %6.2fx | Profit Diff: %8.6f\n" test_name result.speedup result.profit_diff
        total_speedup += result.speedup
        max_profit_diff = max(max_profit_diff, result.profit_diff)
    end
    
    avg_speedup = total_speedup / length(results)
    
    println("-" ^ 80)
    @printf "Average Speedup: %.2fx\n" avg_speedup
    @printf "Max Profit Difference: %.6f\n" max_profit_diff
    
    if max_profit_diff < 1e-6
        println("✅ All tests passed: Implementations produce identical results")
    elseif max_profit_diff < 0.01
        println("⚠️  Most tests passed: Minor differences in some results")
    else
        println("❌ Some tests failed: Significant differences detected")
    end
    
    println("\n🎯 CONCLUSION:")
    if avg_speedup > 1.0
        println("✅ Improved implementation is faster on average")
    else
        println("❌ Improved implementation is not consistently faster")
    end
    
    return results
end

"""
Benchmark specific split function performance with proper grid costs
"""
function benchmark_split_functions()
    println("\n🔬 DETAILED SPLIT FUNCTION BENCHMARK")
    println("Using proper grid movement costs: 1 for neighbors, ∞ for non-neighbors")
    println("=" ^ 60)
    
    # Generate test permutations of different sizes
    sizes = [10, 20, 30, 50, 100]
    
    for n in sizes
        customers, profits, costs, depot = generate_test_instance(n, 20, 42)
        
        # Create PSO objects for both implementations
        original_pso = OriginalPSO.PSOiA_TOP(
            OriginalPSO.Particle[], Int[], -Inf, 20, 100,
            0.3, 0.5, 0.3, 0.1, 0.3, 3, 15,
            customers, profits, costs, collect(1:n), depot
        )
        
        improved_pso = ImprovedPSO.PSOiA_TOP(
            ImprovedPSO.Particle[], Int[], -Inf, 20, 100,
            0.3, 0.5, 0.3, 0.1, 0.3, 3, 15,
            customers, profits, costs, collect(1:n), depot
        )
        
        # Random permutation
        permutation = shuffle(collect(1:n))
        
        # Benchmark original split
        original_time = time_function(() -> OriginalPSO.fast_split(permutation, original_pso))
        original_profit = OriginalPSO.fast_split(permutation, original_pso)
        
        # Benchmark improved split
        improved_time = time_function(() -> ImprovedPSO.fast_split(permutation, improved_pso))
        improved_profit = ImprovedPSO.fast_split(permutation, improved_pso)
        
        profit_diff = abs(original_profit - improved_profit)
        speedup = original_time / improved_time
        
        @printf "n=%3d | Original: %8.2f ms | Improved: %8.2f ms | Speedup: %6.2fx | Profit diff: %.6f\n" n (original_time*1000) (improved_time*1000) speedup profit_diff
    end
end

# Run the test suite
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting PSO Implementation Comparison Tests...")
    println("Using PROPER GRID COSTS: 1 for L-∞ neighbors, ∞ for non-neighbors")
    println()
    
    # Run main test suite
    results = run_test_suite()
    
    # Run detailed benchmark
    benchmark_split_functions()
    
    println("\n🏁 All tests completed!")
end 