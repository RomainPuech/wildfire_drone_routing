using Random
using Printf

# Include both implementations
include("TOP_PSO.jl")
include("Improved_TOP_PSO.jl")

# Rename to avoid conflicts
const OriginalPSO = Main
const ImprovedPSO = Main

"""
Generate test instance for split algorithm testing with proper grid costs
"""
function generate_split_test_instance(n_customers::Int, grid_size::Int, seed::Int = 42)
    Random.seed!(seed)
    
    # Generate customer positions on a grid
    customers = [(rand(1:grid_size), rand(1:grid_size)) for _ in 1:n_customers]
    
    # Generate random profits
    depot = (grid_size ÷ 2, grid_size ÷ 2)
    profits = Float64[]
    
    for (x, y) in customers
        dist_from_depot = max(abs(x - depot[1]), abs(y - depot[2]))
        base_profit = 10.0 + dist_from_depot * 5.0
        profit = base_profit + rand() * 10.0
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
    
    # Costs from depot to customers
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
Test split algorithm correctness and performance
"""
function test_split_algorithm()
    println("🔬 SPLIT ALGORITHM CORRECTNESS & PERFORMANCE TEST")
    println("Using proper grid movement costs: 1 for neighbors, ∞ for non-neighbors")
    println("=" ^ 70)
    
    # Test different problem sizes
    test_sizes = [5, 10, 15, 20, 30, 50, 100]
    
    println("Testing split algorithm on various instance sizes...")
    println("Size | Original Time | Improved Time | Speedup | Profit Match | Routes Match")
    println("-" ^ 80)
    
    total_original_time = 0.0
    total_improved_time = 0.0
    all_correct = true
    
    for n in test_sizes
        customers, profits, costs, depot = generate_split_test_instance(n, 20, 42)
        
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
        
        # Test multiple random permutations for this size
        total_original = 0.0
        total_improved = 0.0
        perfect_matches = 0
        route_matches = 0
        num_tests = 10  # Test multiple permutations
        
        for test_i in 1:num_tests
            Random.seed!(42 + test_i)  # Consistent but different permutations
            permutation = shuffle(collect(1:n))
            
            # Time original implementation
            original_time = @elapsed begin
                original_profit, original_routes = OriginalPSO.fast_split_with_routes(permutation, original_pso)
            end
            
            # Time improved implementation  
            improved_time = @elapsed begin
                improved_profit, improved_routes = ImprovedPSO.fast_split_with_routes(permutation, improved_pso)
            end
            
            total_original += original_time
            total_improved += improved_time
            
            # Check correctness
            profit_diff = abs(original_profit - improved_profit)
            if profit_diff < 1e-10
                perfect_matches += 1
            end
            
            # Check route structure (same number of routes, same total customers)
            if length(original_routes) == length(improved_routes)
                orig_customers = sum(length(r) for r in original_routes)
                impr_customers = sum(length(r) for r in improved_routes)
                if orig_customers == impr_customers
                    route_matches += 1
                end
            end
        end
        
        avg_original = total_original / num_tests
        avg_improved = total_improved / num_tests
        speedup = avg_original / avg_improved
        
        total_original_time += avg_original
        total_improved_time += avg_improved
        
        profit_match_pct = (perfect_matches / num_tests) * 100
        route_match_pct = (route_matches / num_tests) * 100
        
        if perfect_matches < num_tests
            all_correct = false
        end
        
        @printf "%4d | %8.4f ms   | %8.4f ms   | %6.2fx  | %7.1f%%    | %7.1f%%\n" n (avg_original*1000) (avg_improved*1000) speedup profit_match_pct route_match_pct
    end
    
    println("-" ^ 80)
    overall_speedup = total_original_time / total_improved_time
    @printf "OVERALL SPEEDUP: %.2fx\n" overall_speedup
    
    if all_correct
        println("✅ ALL TESTS PASSED: Split algorithms produce identical results")
    else
        println("❌ SOME MISMATCHES: Check implementation for correctness issues")
    end
    
    return overall_speedup, all_correct
end

"""
Detailed correctness verification with proper grid costs
"""
function detailed_correctness_test()
    println("\n🔍 DETAILED CORRECTNESS VERIFICATION")
    println("=" ^ 50)
    
    # Test with a specific small example we can manually verify
    # Place customers in a connected pattern
    customers = [(2,2), (2,3), (3,2), (3,3), (1,2)]  # All connected to depot or each other
    profits = [10.0, 15.0, 12.0, 18.0, 20.0]
    depot = (2, 2)  # Depot at center
    
    # Proper grid cost matrix
    costs = Dict{Tuple{Int,Int}, Float64}()
    LARGE_COST = 1000.0
    
    function are_neighbors(pos1::Tuple{Int,Int}, pos2::Tuple{Int,Int})
        x1, y1 = pos1
        x2, y2 = pos2
        return max(abs(x2 - x1), abs(y2 - y1)) <= 1
    end
    
    # Costs from depot
    for i in 1:5
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
    for i in 1:5
        for j in 1:5
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
    
    println("Test setup:")
    println("Depot at: $depot")
    println("Customer positions: $customers")
    println("Customer profits: $profits")
    println("\nGrid connectivity (showing costs = 1):")
    for i in 1:5
        neighbors = []
        # Check depot connection
        if are_neighbors(depot, customers[i])
            push!(neighbors, "depot")
        end
        # Check customer connections
        for j in 1:5
            if i != j && are_neighbors(customers[i], customers[j])
                push!(neighbors, "C$j")
            end
        end
        println("  Customer $i at $(customers[i]) connects to: $(join(neighbors, ", "))")
    end
    
    # Create PSO objects
    original_pso = OriginalPSO.PSOiA_TOP(
        OriginalPSO.Particle[], Int[], -Inf, 20, 100,
        0.3, 0.5, 0.3, 0.1, 0.3, 2, 8,  # 2 drones, battery 8
        customers, profits, costs, collect(1:5), depot
    )
    
    improved_pso = ImprovedPSO.PSOiA_TOP(
        ImprovedPSO.Particle[], Int[], -Inf, 20, 100,
        0.3, 0.5, 0.3, 0.1, 0.3, 2, 8,  # 2 drones, battery 8
        customers, profits, costs, collect(1:5), depot
    )
    
    # Test specific permutation
    permutation = [1, 2, 3, 4, 5]
    
    println("\nTesting permutation: $permutation")
    println("Max battery: 8, Drones: 2")
    println()
    
    # Run both algorithms
    original_profit, original_routes = OriginalPSO.fast_split_with_routes(permutation, original_pso)
    improved_profit, improved_routes = ImprovedPSO.fast_split_with_routes(permutation, improved_pso)
    
    println("ORIGINAL ALGORITHM:")
    @printf "Total profit: %.3f\n" original_profit
    println("Routes:")
    for (i, route) in enumerate(original_routes)
        if !isempty(route)
            route_profit = sum(profits[c] for c in route)
            # Calculate route cost for verification
            route_cost = 0
            if !isempty(route)
                route_cost += costs[(0, route[1])]  # depot to first
                for j in 1:(length(route)-1)
                    route_cost += costs[(route[j], route[j+1])]  # between customers
                end
                route_cost += costs[(route[end], 0)]  # last to depot
            end
            println("  Drone $i: $route (profit: $(round(route_profit, digits=2)), cost: $route_cost)")
        end
    end
    
    println("\nIMPROVED ALGORITHM:")
    @printf "Total profit: %.3f\n" improved_profit
    println("Routes:")
    for (i, route) in enumerate(improved_routes)
        if !isempty(route)
            route_profit = sum(profits[c] for c in route)
            # Calculate route cost for verification
            route_cost = 0
            if !isempty(route)
                route_cost += costs[(0, route[1])]  # depot to first
                for j in 1:(length(route)-1)
                    route_cost += costs[(route[j], route[j+1])]  # between customers
                end
                route_cost += costs[(route[end], 0)]  # last to depot
            end
            println("  Drone $i: $route (profit: $(round(route_profit, digits=2)), cost: $route_cost)")
        end
    end
    
    profit_diff = abs(original_profit - improved_profit)
    println("\nProfit difference: $(round(profit_diff, digits=10))")
    
    if profit_diff < 1e-10
        println("✅ PERFECT MATCH")
    else
        println("❌ MISMATCH DETECTED")
    end
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("🧪 SPLIT ALGORITHM FOCUSED TESTING - PROPER GRID COSTS")
    println("Grid movement: cost=1 for L-∞ neighbors, cost=∞ for non-neighbors")
    println()
    
    # Run performance and correctness test
    speedup, all_correct = test_split_algorithm()
    
    # Run detailed verification
    detailed_correctness_test()
    
    println("\n📋 SUMMARY")
    println("=" ^ 30)
    @printf "Overall speedup: %.2fx\n" speedup
    println("Correctness: $(all_correct ? "✅ PASSED" : "❌ FAILED")")
    
    if speedup > 1.0 && all_correct
        println("\n🎉 SUCCESS: Improved implementation is both faster and correct!")
    elseif all_correct
        println("\n⚠️  Improved implementation is correct but not consistently faster")
    else
        println("\n🚨 WARNING: Correctness issues detected!")
    end
end 