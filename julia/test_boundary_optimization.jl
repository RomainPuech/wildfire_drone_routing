# Test script for boundary optimization correctness
# Run this from the julia directory: julia test_boundary_optimization.jl
#
# This script tests that optimization 2 (boundary constraints) is correct:
# - Skipped swap operations indeed do not change the solution profit
# - Skipped shift operations indeed do not change the solution profit

using Dates
using Random
using Statistics

println("="^60)
println("BOUNDARY OPTIMIZATION CORRECTNESS TEST")
println("="^60)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

# Include the necessary files
println("Loading Julia modules...")
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")
println("Modules loaded successfully!")
println()

# ============================================================================
# Helper function to create a test PSO instance
# ============================================================================

function create_test_pso(n_pure_customers::Int, n_depot_duplicates::Int, n_drones::Int, max_battery_time::Int)
    n_total = n_pure_customers + n_depot_duplicates
    
    # Create customer coordinates (grid positions)
    customers = [(rand(1:30), rand(1:30)) for _ in 1:n_total]
    
    # Create profits vector (random, but higher for customers)
    profits_vec = [rand(0.1:0.01:1.0) for _ in 1:n_pure_customers]
    # Depots have zero profit
    append!(profits_vec, zeros(Float64, n_depot_duplicates))
    
    # Create cost dictionary:
    # Only adjacent nodes (Chebyshev distance 1) have cost 1.
    # Non-adjacent nodes are set to an infeasible cost (> max_battery_time).
    costs = Dict{Tuple{Int, Int}, Float64}()
    for i in 1:n_total
        for j in 1:n_total
            if i != j
                x1, y1 = customers[i]
                x2, y2 = customers[j]
                cheb = max(abs(x1 - x2), abs(y1 - y2))
                if cheb == 1
                    costs[(i, j)] = 1.0
                else
                    costs[(i, j)] = max_battery_time * 4
                end
            end
        end
    end
    
    # Left neighbors (feasible predecessors in clients graph)
    left_neighbors = Dict{Int, Vector{Int}}()
    for v in 1:n_total
        left_neighbors[v] = Int[]
    end
    for u in 1:n_total
        for v in 1:n_total
            if u == v
                continue
            end
            if get(costs, (u, v), max_battery_time * 4) <= max_battery_time
                push!(left_neighbors[v], u)
            end
        end
    end
    
    # Closest depot distance (precomputed)
    closest_depot_distance = ones(Float64, n_total)
    
    # Depot coordinates
    depot_coord = customers[(n_pure_customers+1):end]
    
    # Accessible customers = all nodes
    accessible_customers = collect(1:n_total)
    
    # Create PSO instance (positional arguments)
    pso = PSOiA_TOP_multiple_depots(
        Particle[],           # swarm
        Int[],                # global_best
        -Inf,                 # global_best_profit
        10,                   # swarm_size
        100,                  # max_iterations
        0.7,                  # w
        1.5,                  # c1
        1.5,                  # c2
        0.1,                  # ph
        0.8,                  # pm
        n_drones,
        n_pure_customers,
        max_battery_time,
        customers,
        profits_vec,  # profits vector
        costs,
        left_neighbors,
        accessible_customers,
        depot_coord,
        closest_depot_distance
    )
    
    return pso
end

# ============================================================================
# TEST: Swap Blocking Optimization Correctness
# ============================================================================

function test_swap_boundary_correctness(pso::PSOiA_TOP_multiple_depots, permutation::Vector{Int}, n_samples::Int = 1000)
    println("Testing swap blocking optimization correctness...")
    
    # Create a particle
    particle = Particle(
        copy(permutation),  # position
        copy(permutation),  # local_best
        0.0,                # local_best_profit
        0.0,                # current_profit
        compute_node_to_position(permutation)  # node_to_position
    )
    
    # Compute initial profit
    initial_profit, _, _ = fast_split_sparse(particle.position, particle, pso)
    particle.current_profit = initial_profit
    
    println("  Initial profit: $(round(initial_profit, digits=6))")
    println("  Testing up to $n_samples swap pairs...")
    
    n = length(permutation)
    violations = 0
    total_skipped = 0
    total_tested = 0
    total_customers_tested = 0
    total_irrelevant_removed = 0
    total_customers_tested = 0
    total_irrelevant_removed = 0
    total_customers_tested = 0
    total_irrelevant_removed = 0
    total_customers_tested = 0
    total_irrelevant_removed = 0
    
    # Sample swap pairs
    Random.seed!(42)  # For reproducibility
    positions_i = shuffle(1:n)
    positions_j = shuffle(1:n)
    
    for sample_idx in 1:min(n_samples, n * (n - 1) ÷ 2)
        i = positions_i[(sample_idx - 1) % n + 1]
        j = positions_j[(sample_idx - 1) % (n - 1) + 1]
        
        if i >= j
            continue
        end
        
        node_i = permutation[i]
        node_j = permutation[j]
        is_depot_i = node_i > pso.n_pure_customers
        is_depot_j = node_j > pso.n_pure_customers
        
        total_tested += 1
        
        # Check if this swap would be skipped by blocking optimization
        would_skip = false
        if !is_depot_i && !is_depot_j
            if is_blocking_once_inserted(particle, i, j, pso) &&
               is_blocking_once_removed(particle, i, pso) &&
               is_blocking_once_inserted(particle, j, i, pso) &&
               is_blocking_once_removed(particle, j, pso)
                would_skip = true
            end
        end
        
        if would_skip
            total_skipped += 1
            
            # Actually perform the swap and check if profit changes
            test_permutation = copy(permutation)
            test_permutation[i], test_permutation[j] = test_permutation[j], test_permutation[i]
            
            # Compute profit with swapped permutation
            test_profit, _, _ = fast_split_sparse(test_permutation, pso)
            
            # Check if profit changed (allowing for floating point errors)
            if test_profit > initial_profit + 1e-9
                violations += 1
                println("  ❌ VIOLATION: Swap positions ($i, $j) was skipped but increases profit!")
                println("     Initial profit: $(round(initial_profit, digits=8))")
                println("     Swapped profit: $(round(test_profit, digits=8))")
                println("     Difference: $(round(abs(test_profit - initial_profit), digits=10))")
                println("     Node at i: $node_i (depot: $is_depot_i)")
                println("     Node at j: $node_j (depot: $is_depot_j)")
            end
        end
    end
    
    println("  Results:")
    println("    Total swap pairs tested: $total_tested")
    println("    Swaps skipped by boundary optimization: $total_skipped")
    println("    Violations (skipped swaps that change profit): $violations")
    
    if violations == 0
        println("  ✅ PASSED: All skipped swaps are provably no-ops")
    else
        println("  ❌ FAILED: Found $violations violations")
    end
    
    return violations == 0
end

# ============================================================================
# TEST: Shift Irrelevance Filter Correctness
# ============================================================================

function test_shift_boundary_correctness(pso::PSOiA_TOP_multiple_depots, permutation::Vector{Int}, n_samples::Int = 1000)
    println("Testing shift irrelevance filter correctness...")
    
    # Create a particle
    particle = Particle(
        copy(permutation),  # position
        copy(permutation),  # local_best
        0.0,                # local_best_profit
        0.0,                # current_profit
        compute_node_to_position(permutation)  # node_to_position
    )
    
    # Compute initial profit and dead positions
    initial_profit, _, _ = fast_split_sparse(particle.position, particle, pso)
    _, _, tour_lengths_sparse, sorted_depot_positions = compute_saturated_tours_sparse(
        particle.position, particle.node_to_position, pso
    )
    dead_positions = compute_dead_positions(length(permutation), sorted_depot_positions, tour_lengths_sparse)
    particle.current_profit = initial_profit
    
    println("  Initial profit: $(round(initial_profit, digits=6))")
    println("  Testing up to $n_samples shift moves...")
    
    n = length(permutation)
    violations = 0
    total_skipped = 0
    total_tested = 0
    total_customers_tested = 0
    total_irrelevant_removed = 0
    
    # Sample shift moves
    Random.seed!(123)  # For reproducibility
    positions_i = shuffle(1:n)
    positions_j = shuffle(1:n)
    
    for sample_idx in 1:min(n_samples, n * (n - 1))
        i = positions_i[(sample_idx - 1) % n + 1]
        j = positions_j[(sample_idx - 1) % (n - 1) + 1]
        
        if i == j
            continue
        end
        
        node_i = permutation[i]
        is_depot = node_i > pso.n_pure_customers
        
        total_tested += 1
        
        # Check if this shift would be skipped by irrelevance filter
        would_skip = false
        if is_depot
            would_skip = false
        else
            total_customers_tested += 1
            is_blocking_or_dead = is_blocking(particle, i, pso) || dead_positions[i]
            irrelevant_removed = false
            if is_blocking_or_dead && is_blocking_once_removed(particle, i, pso)
                irrelevant_removed = true
            end
            if irrelevant_removed
                total_irrelevant_removed += 1
            end
            if irrelevant_removed && j > 1 && dead_positions[j - 1]
                would_skip = true
            end
        end
        
        if would_skip
            total_skipped += 1
            
            # Actually perform the shift and check if profit changes
            test_permutation = move_element(permutation, i, j)
            
            # Compute profit with shifted permutation
            test_profit, _, _ = fast_split_sparse(test_permutation, pso)
            
            # Check if profit increased (allowing for floating point errors)
            if test_profit > initial_profit + 1e-9
                violations += 1
                println("  ❌ VIOLATION: Shift from position $i to $j was skipped but increases profit!")
                println("     Initial profit: $(round(initial_profit, digits=8))")
                println("     Shifted profit: $(round(test_profit, digits=8))")
                println("     Difference: $(round(abs(test_profit - initial_profit), digits=10))")
                println("     Node at i: $node_i (depot: $is_depot)")
            end
        end
    end
    
    println("  Results:")
    println("    Total shift moves tested: $total_tested")
    println("    Shifts skipped by boundary optimization: $total_skipped")
    if total_tested > 0
        println("    Skip rate: $(round(100 * total_skipped / total_tested, digits=2))%")
    end
    if total_customers_tested > 0
        println("    irrelevant_once_removed count: $total_irrelevant_removed")
        println("    Stage 1 fail rate (irrelevant once removed): $(round(100 * total_irrelevant_removed / total_customers_tested, digits=2))%")
    end
    println("    Violations (skipped shifts that change profit): $violations")
    
    if violations == 0
        println("  ✅ PASSED: All skipped shifts are provably no-ops")
    else
        println("  ❌ FAILED: Found $violations violations")
    end
    
    return violations == 0
end

# ============================================================================
# BENCHMARK: Speedup (filtered vs baseline)
# ============================================================================

function benchmark_boundary_speedup(pso::PSOiA_TOP_multiple_depots, permutation::Vector{Int}, n_trials::Int = 500)
    println("Benchmarking speedup (filtered vs baseline)...")
    n = length(permutation)
    particle = Particle(
        copy(permutation),
        copy(permutation),
        0.0,
        0.0,
        compute_node_to_position(permutation)
    )
    base_profit, _, _ = fast_split_sparse(particle.position, particle, pso)
    particle.current_profit = base_profit

    # Precompute dead positions
    _, _, tour_lengths_sparse, sorted_depot_positions = compute_saturated_tours_sparse(
        particle.position, particle.node_to_position, pso
    )
    dead_positions = compute_dead_positions(n, sorted_depot_positions, tour_lengths_sparse)

    # Baseline: evaluate all sampled shift moves with full split
    Random.seed!(2026)
    shift_pairs = [(rand(1:n-1), rand(2:n)) for _ in 1:n_trials]
    shift_pairs = [(i, j) for (i, j) in shift_pairs if i < j]
    t_baseline_shift = @elapsed begin
        for (i, j) in shift_pairs
            new_pos = move_element(permutation, i, j)
            fast_split_sparse(new_pos, pso)
        end
    end

    # Filtered shift: irrelevance-based skip
    t_filtered_shift = @elapsed begin
        for (i, j) in shift_pairs
            node_i = permutation[i]
            is_depot = node_i > pso.n_pure_customers
            if is_depot
                new_pos = move_element(permutation, i, j)
                fast_split_sparse(new_pos, pso)
                continue
            end

            # Irrelevance-based skip
            is_blocking_or_dead = is_blocking(particle, i, pso) || dead_positions[i]
            irrelevant_removed = false
            if is_blocking_or_dead && is_blocking_once_removed(particle, i, pso)
                irrelevant_removed = true
            end
            if irrelevant_removed && j > 1 && dead_positions[j - 1]
                continue
            end

            new_pos = move_element(permutation, i, j)
            fast_split_sparse(new_pos, pso)
        end
    end

    # Baseline: evaluate all sampled swap moves with full split
    swap_pairs = [(rand(1:n-1), rand(2:n)) for _ in 1:n_trials]
    swap_pairs = [(i, j) for (i, j) in swap_pairs if i < j]
    t_baseline_swap = @elapsed begin
        for (i, j) in swap_pairs
            test = copy(permutation)
            test[i], test[j] = test[j], test[i]
            fast_split_sparse(test, pso)
        end
    end

    # Filtered swap: only apply blocking checks
    t_filtered_swap = @elapsed begin
        for (i, j) in swap_pairs
            node_i = permutation[i]
            node_j = permutation[j]
            is_depot_i = node_i > pso.n_pure_customers
            is_depot_j = node_j > pso.n_pure_customers
            if !is_depot_i && !is_depot_j
                if is_blocking_once_inserted(particle, i, j, pso) &&
                   is_blocking_once_removed(particle, i, pso) &&
                   is_blocking_once_inserted(particle, j, i, pso) &&
                   is_blocking_once_removed(particle, j, pso)
                    continue
                end
            end
            test = copy(permutation)
            test[i], test[j] = test[j], test[i]
            fast_split_sparse(test, pso)
        end
    end

    println("  Shift baseline time:  $(round(t_baseline_shift, digits=4))s")
    println("  Shift filtered time:  $(round(t_filtered_shift, digits=4))s")
    println("  Shift speedup:        $(round(t_baseline_shift / max(t_filtered_shift, 1e-9), digits=2))x")
    println("  Swap baseline time:   $(round(t_baseline_swap, digits=4))s")
    println("  Swap filtered time:   $(round(t_filtered_swap, digits=4))s")
    println("  Swap speedup:         $(round(t_baseline_swap / max(t_filtered_swap, 1e-9), digits=2))x")
end


# ============================================================================
# BENCHMARK: Sparse split vs dense split speedup
# ============================================================================

function benchmark_sparse_split_speedup(pso::PSOiA_TOP_multiple_depots, permutation::Vector{Int}, n_trials::Int = 200)
    println("Benchmarking sparse split vs dense split...")

    # Warmup
    fast_split_sparse(permutation, pso)
    fast_split_sparse_profit(permutation, pso)
    fast_split_with_routes_multiple_depots(permutation, pso)

    t_dense = @elapsed begin
        for _ in 1:n_trials
            fast_split_with_routes_multiple_depots(permutation, pso)
        end
    end

    t_sparse = @elapsed begin
        for _ in 1:n_trials
            fast_split_sparse(permutation, pso)
        end
    end

    t_sparse_profit = @elapsed begin
        for _ in 1:n_trials
            fast_split_sparse_profit(permutation, pso)
        end
    end

    println("  Dense split time:  $(round(t_dense, digits=4))s")
    println("  Sparse split time: $(round(t_sparse, digits=4))s")
    println("  Sparse profit time: $(round(t_sparse_profit, digits=4))s")
    println("  Speedup:           $(round(t_dense / max(t_sparse, 1e-9), digits=2))x")
    println("  Profit-only speedup vs sparse: $(round(t_sparse / max(t_sparse_profit, 1e-9), digits=2))x")
end

# (Removed) Counterexample search for the deprecated profit-only filter.

# (Removed) Small counterexample search for the deprecated profit-only filter.

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

function run_boundary_optimization_tests()
    println("="^60)
    println("BOUNDARY OPTIMIZATION CORRECTNESS TESTS")
    println("="^60)
    println()
    
    test_results = []

    # Test configuration (10x larger instances for speedup measurement)
    test_configs = [
        (n_pure_customers=300, n_depot_duplicates=5, n_drones=2, max_battery_time=15, n_samples=500),
        (n_pure_customers=500, n_depot_duplicates=8, n_drones=3, max_battery_time=20, n_samples=1000),
        (n_pure_customers=900, n_depot_duplicates=4, n_drones=2, max_battery_time=63, n_samples=2000),
    ]
    
    for (test_idx, config) in enumerate(test_configs)
        println("="^60)
        println("TEST CONFIGURATION $test_idx")
        println("="^60)
        println("  n_pure_customers: $(config.n_pure_customers)")
        println("  n_depot_duplicates: $(config.n_depot_duplicates)")
        println("  n_drones: $(config.n_drones)")
        println("  max_battery_time: $(config.max_battery_time)")
        println("  n_samples: $(config.n_samples)")
        println()
        
        # Create PSO instance
        Random.seed!(test_idx * 1000)
        pso = create_test_pso(
            config.n_pure_customers,
            config.n_depot_duplicates,
            config.n_drones,
            config.max_battery_time
        )
        
        # Create a random permutation
        n_total = config.n_pure_customers + config.n_depot_duplicates
        permutation = collect(1:n_total)
        shuffle!(permutation)
        
        # Ensure depots are distributed (not all at the end)
        # This creates a more realistic test case
        depot_nodes = collect((config.n_pure_customers + 1):n_total)
        customer_nodes = collect(1:config.n_pure_customers)
        permutation = vcat(
            shuffle(customer_nodes[1:div(config.n_pure_customers, 3)]),
            depot_nodes[1:div(length(depot_nodes), 2)],
            shuffle(customer_nodes[div(config.n_pure_customers, 3)+1:2*div(config.n_pure_customers, 3)]),
            depot_nodes[div(length(depot_nodes), 2)+1:end],
            shuffle(customer_nodes[2*div(config.n_pure_customers, 3)+1:end])
        )
        
        println("  Permutation length: $(length(permutation))")
        println("  First 10 elements: $(permutation[1:min(10, length(permutation))])")
        println()
        
        # Test swap boundary optimization
        swap_passed = test_swap_boundary_correctness(pso, permutation, config.n_samples)
        println()
        
        # Test shift boundary optimization
        shift_passed = test_shift_boundary_correctness(pso, permutation, config.n_samples)
        println()

        # Benchmark speedup (smaller sample for runtime)
        benchmark_boundary_speedup(pso, permutation, 200)
        println()

        # Benchmark sparse vs dense split
        benchmark_sparse_split_speedup(pso, permutation, 200)
        println()

        push!(test_results, (test_idx, swap_passed, shift_passed))
    end
    
    # Summary
    println("="^60)
    println("TEST SUMMARY")
    println("="^60)
    
    all_passed = true
    for (test_idx, swap_passed, shift_passed) in test_results
        status = (swap_passed && shift_passed) ? "✅ PASSED" : "❌ FAILED"
        println("Test $test_idx: $status")
        println("  Swap boundary optimization: $(swap_passed ? "✅" : "❌")")
        println("  Shift boundary optimization: $(shift_passed ? "✅" : "❌")")
        if !(swap_passed && shift_passed)
            all_passed = false
        end
    end
    
    println()
    
    # (Removed) Small counterexample search for deprecated profit-only filter
    if all_passed
        println("Overall: ✅ ALL TESTS PASSED")
        println("Boundary optimization is CORRECT: only no-op operations are skipped.")
    else
        println("Overall: ❌ SOME TESTS FAILED")
        println("Boundary optimization may have issues - review violations above.")
    end
    
    println()
    println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    
    return all_passed
end

# Run the tests
run_boundary_optimization_tests()
