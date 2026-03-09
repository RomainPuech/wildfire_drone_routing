# Test script for sparse split optimization
# Run this from the julia directory: julia test_sparse_optimization.jl
#
# This script tests:
# 1. Sparse split produces the same results as dense split
# 2. Sparse operators produce correct results
# 3. Overall TOP implementation still works correctly

using Dates
using Random
using Statistics

println("="^60)
println("SPARSE SPLIT OPTIMIZATION TEST")
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
# TEST 1: Sparse Split vs Dense Split Equivalence
# ============================================================================

println("="^60)
println("TEST 1: Sparse Split vs Dense Split Equivalence")
println("="^60)

function test_split_equivalence(n_tests::Int, seed::Int)
    Random.seed!(seed)
    
    # Test parameters
    n_pure_customers = 50
    n_depot_duplicates = 10  # Total depot nodes
    n_drones = 3
    max_battery_time = 20
    
    # Create mock data
    n_total = n_pure_customers + n_depot_duplicates
    
    # Create customer coordinates (grid positions)
    customers = [(rand(1:20), rand(1:20)) for _ in 1:n_total]
    
    # Create profits (random)
    profits = rand(Float64, n_total)
    profits[(n_pure_customers+1):end] .= 0.0  # Depot nodes have 0 profit
    
    # Create costs dictionary (Chebyshev distance between adjacent nodes)
    costs = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n_total
        for j in 1:n_total
            if i != j
                dist = max(abs(customers[i][1] - customers[j][1]), 
                          abs(customers[i][2] - customers[j][2]))
                if dist <= 2  # Only adjacent nodes have costs
                    costs[(i, j)] = Float64(dist)
                end
            end
        end
        # Cost from artificial depot (0) to this node
        costs[(0, i)] = 1.0
    end
    
    # Left neighbors (for blocking check - not used in this test)
    left_neighbors = Dict{Int, Vector{Int}}()
    for i in 1:n_total
        left_neighbors[i] = Int[]
    end
    
    # Closest depot distance (precomputed)
    closest_depot_distance = ones(Float64, n_total)
    
    # Depot coordinates
    depot_coord = customers[(n_pure_customers+1):end]
    
    # Accessible customers = all nodes
    accessible_customers = collect(1:n_total)
    
    # Create mock PSO object (we just need it for the split function)
    pso = PSOiA_TOP_multiple_depots(
        Particle[],  # empty swarm
        Int[],       # empty global_best
        -Inf,        # global_best_profit
        10,          # swarm_size
        100,         # max_iterations
        0.7, 1.5, 1.5, 0.1, 0.8,  # w, c1, c2, ph, pm
        n_drones,
        n_pure_customers,
        max_battery_time,
        customers,
        profits,
        costs,
        left_neighbors,
        accessible_customers,
        depot_coord,
        closest_depot_distance
    )
    
    # Run tests
    passed = 0
    failed = 0
    max_diff = 0.0
    
    for test_idx in 1:n_tests
        # Create random permutation
        permutation = shuffle(accessible_customers)
        
        # Compute node_to_position mapping
        node_to_position = compute_node_to_position(permutation)
        
        # Dense split
        dense_profit, dense_routes = fast_split_with_routes_multiple_depots(permutation, pso)
        
        # Sparse split
        sparse_profit, sparse_routes, tour_intervals = fast_split_sparse_with_mapping(permutation, node_to_position, pso)
        
        # Compare results
        profit_diff = abs(dense_profit - sparse_profit)
        max_diff = max(max_diff, profit_diff)
        
        if profit_diff < 1e-9
            passed += 1
        else
            failed += 1
            if failed <= 5  # Only print first 5 failures
                println("  FAIL test $test_idx: dense=$dense_profit, sparse=$sparse_profit, diff=$profit_diff")
            end
        end
    end
    
    println("Results: $passed passed, $failed failed out of $n_tests tests")
    println("Maximum profit difference: $max_diff")
    
    return failed == 0
end

# Run the equivalence test
test1_passed = test_split_equivalence(100, 42)
println("TEST 1: $(test1_passed ? "PASSED ✓" : "FAILED ✗")")
println()

# ============================================================================
# TEST 2: TourIntervals Boundary Checks
# ============================================================================

println("="^60)
println("TEST 2: TourIntervals Boundary Checks")
println("="^60)

function test_tour_intervals()
    passed = true
    
    # Test case 1: Simple intervals
    intervals1 = [(3, 5), (8, 10), (15, 20)]
    ti1 = TourIntervals(intervals1, 20)
    
    # Test intersects_range
    tests = [
        # (range_start, range_end, expected)
        (1, 2, false),    # Before all intervals
        (3, 5, true),     # Exactly first interval
        (4, 4, true),     # Inside first interval
        (6, 7, false),    # Between intervals
        (7, 9, true),     # Overlaps second interval
        (21, 25, false),  # After all intervals
        (1, 25, true),    # Spans everything
        (10, 15, true),   # Spans gap but touches both ends
    ]
    
    for (rs, re, expected) in tests
        result = intersects_range(ti1, rs, re)
        if result != expected
            println("  FAIL: intersects_range([$rs, $re]) = $result, expected $expected")
            passed = false
        end
    end
    
    # Test is_active
    active_tests = [
        (1, false),
        (3, true),
        (4, true),
        (5, true),
        (6, false),
        (8, true),
        (11, false),
        (15, true),
        (20, true),
        (21, false),
    ]
    
    for (pos, expected) in active_tests
        result = is_active(ti1, pos)
        if result != expected
            println("  FAIL: is_active($pos) = $result, expected $expected")
            passed = false
        end
    end
    
    # Test case 2: Empty intervals
    ti_empty = empty_tour_intervals()
    if intersects_range(ti_empty, 1, 10)
        println("  FAIL: empty intervals should not intersect anything")
        passed = false
    end
    
    # Test case 3: Merged intervals from build_tour_intervals
    depot_positions = [5, 10, 20]
    tour_lengths = [4, 6, 3]  # Tours cover: depot+1 to depot+len-1
    ti3 = build_tour_intervals(depot_positions, tour_lengths)
    
    # Expected merged intervals:
    # Depot 5 -> (5,5) and (6, 8)
    # Depot 10 -> (10,10) and (11, 15)
    # Depot 20 -> (20,20) and (21, 22)
    # After merge: (5,8), (10,15), (20,22)
    
    if !is_active(ti3, 5) || !is_active(ti3, 7) || !is_active(ti3, 12)
        println("  FAIL: Expected positions to be active in merged intervals")
        passed = false
    end
    
    if is_active(ti3, 9) || is_active(ti3, 16) || is_active(ti3, 25)
        println("  FAIL: Expected positions to be inactive (outside intervals)")
        passed = false
    end
    
    if passed
        println("All boundary check tests passed!")
    end
    
    return passed
end

test2_passed = test_tour_intervals()
println("TEST 2: $(test2_passed ? "PASSED ✓" : "FAILED ✗")")
println()

# ============================================================================
# TEST 3: Sparse vs Dense Split Performance
# ============================================================================

println("="^60)
println("TEST 3: Performance Comparison (Sparse vs Dense)")
println("="^60)

function test_performance(n_iterations::Int)
    Random.seed!(123)
    
    # Larger test case for performance comparison
    n_pure_customers = 200
    n_depot_duplicates = 15
    n_drones = 5
    max_battery_time = 30
    
    n_total = n_pure_customers + n_depot_duplicates
    
    customers = [(rand(1:50), rand(1:50)) for _ in 1:n_total]
    profits = rand(Float64, n_total)
    profits[(n_pure_customers+1):end] .= 0.0
    
    costs = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n_total
        for j in 1:n_total
            if i != j
                dist = max(abs(customers[i][1] - customers[j][1]), 
                          abs(customers[i][2] - customers[j][2]))
                if dist <= 3
                    costs[(i, j)] = Float64(dist)
                end
            end
        end
        costs[(0, i)] = 1.0
    end
    
    left_neighbors = Dict{Int, Vector{Int}}()
    for i in 1:n_total
        left_neighbors[i] = Int[]
    end
    
    closest_depot_distance = ones(Float64, n_total)
    depot_coord = customers[(n_pure_customers+1):end]
    accessible_customers = collect(1:n_total)
    
    pso = PSOiA_TOP_multiple_depots(
        Particle[], Int[], -Inf, 10, 100,
        0.7, 1.5, 1.5, 0.1, 0.8,
        n_drones, n_pure_customers, max_battery_time,
        customers, profits, costs, left_neighbors,
        accessible_customers, depot_coord, closest_depot_distance
    )
    
    # Generate test permutations
    permutations = [shuffle(accessible_customers) for _ in 1:n_iterations]
    mappings = [compute_node_to_position(p) for p in permutations]
    
    # Warm-up
    for i in 1:min(5, n_iterations)
        fast_split_with_routes_multiple_depots(permutations[i], pso)
        fast_split_sparse_with_mapping(permutations[i], mappings[i], pso)
    end
    
    # Time dense split
    dense_start = time()
    for i in 1:n_iterations
        fast_split_with_routes_multiple_depots(permutations[i], pso)
    end
    dense_time = time() - dense_start
    
    # Time sparse split (with precomputed mapping)
    sparse_start = time()
    for i in 1:n_iterations
        fast_split_sparse_with_mapping(permutations[i], mappings[i], pso)
    end
    sparse_time = time() - sparse_start
    
    # Time sparse split (computing mapping on-the-fly)
    sparse_full_start = time()
    for i in 1:n_iterations
        fast_split_sparse(permutations[i], pso)
    end
    sparse_full_time = time() - sparse_full_start
    
    println("Configuration: n=$n_total (k=$n_depot_duplicates depots), m=$n_drones drones")
    println("Iterations: $n_iterations")
    println()
    println("Dense split:              $(round(dense_time * 1000, digits=2)) ms total, $(round(dense_time / n_iterations * 1000, digits=3)) ms/iter")
    println("Sparse split (w/ map):    $(round(sparse_time * 1000, digits=2)) ms total, $(round(sparse_time / n_iterations * 1000, digits=3)) ms/iter")
    println("Sparse split (full):      $(round(sparse_full_time * 1000, digits=2)) ms total, $(round(sparse_full_time / n_iterations * 1000, digits=3)) ms/iter")
    println()
    println("Speedup (sparse w/ map vs dense): $(round(dense_time / sparse_time, digits=2))x")
    println("Speedup (sparse full vs dense):   $(round(dense_time / sparse_full_time, digits=2))x")
    
    return true
end

test_performance(500)
println("TEST 3: COMPLETED (performance comparison)")
println()

# ============================================================================
# TEST 4: Full TOP Integration Test
# ============================================================================

println("="^60)
println("TEST 4: Full TOP Integration Test")
println("="^60)

function test_full_top()
    # Use the same test data as test_top_masked.jl
    burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
    mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"
    
    if !isfile(burnmap_filename)
        println("  SKIP: Test data not found at $burnmap_filename")
        return true  # Skip but don't fail
    end
    
    charging_stations = [(28, 36), (66, 32)]
    ground_stations = [(8, 26), (9, 26)]
    n_drones = 2
    max_battery_time = 63
    t = 0
    verbose = false
    initial_drone_positions = Vector{Tuple{Int,Int}}()
    
    println("Running full TOP computation...")
    println("  Burnmap: $burnmap_filename")
    println("  Mask: $mask_filename")
    println("  Charging stations: $charging_stations")
    println("  Drones: $n_drones, Battery: $max_battery_time")
    
    try
        start_time = time()
        movement_plan = compute_TOP_plan_multiple_depots(
            burnmap_filename,
            n_drones,
            charging_stations,
            ground_stations,
            max_battery_time,
            t,
            verbose,
            initial_drone_positions,
            mask_filename
        )
        elapsed = time() - start_time
        
        println()
        println("Results:")
        println("  Computation time: $(round(elapsed, digits=2))s")
        println("  Movement plan length: $(length(movement_plan))")
        
        if length(movement_plan) > 0
            println("  First 3 steps: $(movement_plan[1:min(3, end)])")
            println("  Last 3 steps: $(movement_plan[max(1, end-2):end])")
            
            # Basic sanity checks
            if length(movement_plan) != max_battery_time
                println("  WARNING: Expected movement_plan length = $max_battery_time, got $(length(movement_plan))")
            end
            
            return true
        else
            println("  ERROR: Empty movement plan!")
            return false
        end
    catch e
        println("  ERROR: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

test4_passed = test_full_top()
println("TEST 4: $(test4_passed ? "PASSED ✓" : "FAILED ✗")")
println()

# ============================================================================
# SUMMARY
# ============================================================================

println("="^60)
println("TEST SUMMARY")
println("="^60)

all_passed = test1_passed && test2_passed && test4_passed

println("TEST 1 (Split Equivalence):    $(test1_passed ? "PASSED ✓" : "FAILED ✗")")
println("TEST 2 (Boundary Checks):      $(test2_passed ? "PASSED ✓" : "FAILED ✗")")
println("TEST 3 (Performance):          COMPLETED (informational)")
println("TEST 4 (Full TOP Integration): $(test4_passed ? "PASSED ✓" : "FAILED ✗")")
println()
println("Overall: $(all_passed ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗")")
println()
println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
