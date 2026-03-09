# Comprehensive test for drone allocation optimization with sanity checks
# Run this from the julia directory: julia test_allocation_comprehensive.jl

using Dates
using Random
using Statistics

println("="^70)
println("COMPREHENSIVE DRONE ALLOCATION TEST WITH SANITY CHECKS")
println("="^70)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

# Include the necessary files
println("Loading Julia modules...")
include("helper_functions.jl")
include("ground_charging_opt.jl")
println("Modules loaded successfully!")
println()

# ============================================================================
# Helper functions
# ============================================================================

function create_synthetic_burnmap(N::Int, M::Int, T::Int=1)
    burnmap = zeros(Float64, T, N, M)
    corner_size = min(5, N÷3, M÷3)
    for t in 1:T
        for i in 1:corner_size, j in 1:corner_size
            burnmap[t, i, j] = 0.8
        end
        for i in 1:corner_size, j in (M-corner_size+1):M
            burnmap[t, i, j] = 0.7
        end
        for i in (N-corner_size+1):N, j in 1:corner_size
            burnmap[t, i, j] = 0.6
        end
        for i in (N-corner_size+1):N, j in (M-corner_size+1):M
            burnmap[t, i, j] = 0.9
        end
        center_i = N ÷ 2
        center_j = M ÷ 2
        for i in (center_i-2):(center_i+2), j in (center_j-2):(center_j+2)
            if 1 <= i <= N && 1 <= j <= M
                burnmap[t, i, j] = 0.5
            end
        end
    end
    return burnmap
end

function create_simple_kernel(kernel_size::Int)
    kernel = Dict{Tuple{Int,Int}, Float64}()
    for dx in -kernel_size:kernel_size
        for dy in -kernel_size:kernel_size
            dist = max(abs(dx), abs(dy))
            if dist <= kernel_size
                weight = max(0.0, 1.0 - dist / (kernel_size + 1))
                if weight > 0.01
                    kernel[(dx, dy)] = weight
                end
            end
        end
    end
    return kernel
end

# ============================================================================
# Comprehensive Test Function
# ============================================================================

function run_comprehensive_test(test_num::Int, N::Int, M::Int, N_grounds::Int, N_charging::Int, n_drones::Int, max_battery::Int)
    println("="^70)
    println("TEST $test_num: Grid $(N)x$(M), $(N_grounds) ground, $(N_charging) charging, $(n_drones) drones")
    println("="^70)
    
    # Create synthetic burn map
    burnmap = create_synthetic_burnmap(N, M, 1)
    temp_burnmap_file = "temp_comp_burnmap_$(test_num).npy"
    npzwrite(temp_burnmap_file, burnmap)
    
    # Create kernel
    kernel = create_simple_kernel(max_battery)
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    println("Running ORIGINAL version...")
    start_time = time_ns() / 1e9
    result_orig = Max_Coverage_Kernel(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    time_orig = (time_ns() / 1e9) - start_time
    
    ground_orig, charging_orig = result_orig
    println("  Time: $(round(time_orig, digits=3)) seconds")
    println("  Ground sensors: $(length(ground_orig))")
    println("  Charging stations: $(length(charging_orig))")
    
    # Sanity checks for original
    if length(ground_orig) != N_grounds
        println("  ⚠ WARNING: Expected $(N_grounds) ground sensors, got $(length(ground_orig))")
    end
    if length(charging_orig) != N_charging
        println("  ⚠ WARNING: Expected $(N_charging) charging stations, got $(length(charging_orig))")
    end
    
    println()
    println("Running WITH ALLOCATION version...")
    start_time = time_ns() / 1e9
    result_alloc = Max_Coverage_Kernel_WithAllocation(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    time_alloc = (time_ns() / 1e9) - start_time
    
    ground_alloc, charging_alloc, drone_allocations = result_alloc
    println("  Time: $(round(time_alloc, digits=3)) seconds")
    println("  Ground sensors: $(length(ground_alloc))")
    println("  Charging stations: $(length(charging_alloc))")
    println("  Drone allocations: $drone_allocations")
    
    # Sanity checks for allocation version
    issues = String[]
    
    if length(ground_alloc) != N_grounds
        push!(issues, "Expected $(N_grounds) ground sensors, got $(length(ground_alloc))")
    end
    if length(charging_alloc) != N_charging
        push!(issues, "Expected $(N_charging) charging stations, got $(length(charging_alloc))")
    end
    if length(drone_allocations) != length(charging_alloc)
        push!(issues, "Drone allocations ($(length(drone_allocations))) don't match charging stations ($(length(charging_alloc)))")
    end
    
    total_allocated = sum(drone_allocations)
    if total_allocated != n_drones
        push!(issues, "Total drones allocated ($total_allocated) doesn't equal n_drones ($n_drones)")
    end
    
    if any(a < 0 for a in drone_allocations)
        push!(issues, "Found negative drone allocation")
    end
    
    if any(a > n_drones for a in drone_allocations)
        push!(issues, "Found drone allocation exceeding n_drones")
    end
    
    if any(!isinteger(a) for a in drone_allocations)
        push!(issues, "Found non-integer drone allocation")
    end
    
    # Compare solutions
    println()
    println("Solution Comparison:")
    println("  Ground sensors match: $(ground_orig == ground_alloc)")
    println("  Charging stations match: $(charging_orig == charging_alloc)")
    
    if ground_orig != ground_alloc
        println("  Original ground: $ground_orig")
        println("  Allocation ground: $ground_alloc")
    end
    
    if charging_orig != charging_alloc
        println("  Original charging: $charging_orig")
        println("  Allocation charging: $charging_alloc")
    end
    
    # Performance comparison
    println()
    println("Performance:")
    if time_alloc < time_orig
        speedup = time_orig / time_alloc
        println("  ✓ Allocation version is $(round(speedup, digits=2))x FASTER")
        println("  Time difference: $(round(time_orig - time_alloc, digits=3)) seconds")
    elseif time_alloc > time_orig
        slowdown = time_alloc / time_orig
        println("  ⚠ Allocation version is $(round(slowdown, digits=2))x SLOWER")
        println("  Time difference: $(round(time_alloc - time_orig, digits=3)) seconds")
    else
        println("  ≈ Performance is similar")
    end
    
    # Report issues
    if length(issues) > 0
        println()
        println("  ⚠ SANITY CHECK ISSUES:")
        for issue in issues
            println("    - $issue")
        end
    else
        println()
        println("  ✓ All sanity checks passed")
    end
    
    # Clean up
    rm(temp_burnmap_file, force=true)
    
    println("="^70)
    println()
    
    return (time_orig, time_alloc, length(issues) == 0, issues)
end

# ============================================================================
# Run Multiple Tests
# ============================================================================

println("Running comprehensive tests on multiple instances...")
println()

test_configs = [
    (20, 20, 2, 3, 10, 5),   # Small
    (25, 25, 3, 4, 12, 6),    # Small-medium
    (30, 30, 4, 5, 15, 7),    # Medium
    (35, 35, 4, 6, 18, 8),    # Medium-large
]

all_results = []
all_times_orig = Float64[]
all_times_alloc = Float64[]

for (idx, (N, M, N_grounds, N_charging, n_drones, max_battery)) in enumerate(test_configs)
    result = run_comprehensive_test(idx, N, M, N_grounds, N_charging, n_drones, max_battery)
    push!(all_results, result)
    push!(all_times_orig, result[1])
    push!(all_times_alloc, result[2])
end

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("OVERALL SUMMARY")
println("="^70)
println()

all_passed = true
for (i, (time_orig, time_alloc, passed, issues)) in enumerate(all_results)
    status = passed ? "✓ PASSED" : "✗ FAILED"
    println("Test $i: $status")
    if !passed
        all_passed = false
        for issue in issues
            println("  - $issue")
        end
    end
end

println()
println("Performance Summary:")
println("  Average time (original): $(round(mean(all_times_orig), digits=3)) seconds")
println("  Average time (allocation): $(round(mean(all_times_alloc), digits=3)) seconds")

avg_orig = mean(all_times_orig)
avg_alloc = mean(all_times_alloc)

if avg_alloc < avg_orig
    speedup = avg_orig / avg_alloc
    println("  ✓ Allocation version is $(round(speedup, digits=2))x FASTER on average")
    println("  Average speedup: $(round((avg_orig - avg_alloc), digits=3)) seconds per test")
elseif avg_alloc > avg_orig
    slowdown = avg_alloc / avg_orig
    println("  ⚠ Allocation version is $(round(slowdown, digits=2))x SLOWER on average")
    println("  Average slowdown: $(round((avg_alloc - avg_orig), digits=3)) seconds per test")
else
    println("  ≈ Performance is similar")
end

println()
if all_passed
    println("✓ ALL TESTS PASSED - All sanity checks successful!")
else
    println("✗ SOME TESTS FAILED - Review issues above")
end

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
