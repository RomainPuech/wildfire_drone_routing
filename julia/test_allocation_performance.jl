# Performance comparison test for drone allocation optimization
# Run this from the julia directory: julia test_allocation_performance.jl
#
# This script compares the running times of:
# 1. Max_Coverage_Kernel vs Max_Coverage_Kernel_WithAllocation
# 2. Max_Coverage_Kernel_Masked vs Max_Coverage_Kernel_Masked_WithAllocation

using Dates
using Random
using Statistics

println("="^70)
println("DRONE ALLOCATION PERFORMANCE COMPARISON TEST")
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
# Helper functions (same as test_drone_allocation.jl)
# ============================================================================

function create_synthetic_burnmap(N::Int, M::Int, T::Int=1)
    """
    Creates a simple synthetic burn map with high-risk areas in the corners.
    """
    burnmap = zeros(Float64, T, N, M)
    
    # Create high-risk areas in corners
    corner_size = min(5, N÷3, M÷3)
    for t in 1:T
        # Top-left corner
        for i in 1:corner_size, j in 1:corner_size
            burnmap[t, i, j] = 0.8
        end
        # Top-right corner
        for i in 1:corner_size, j in (M-corner_size+1):M
            burnmap[t, i, j] = 0.7
        end
        # Bottom-left corner
        for i in (N-corner_size+1):N, j in 1:corner_size
            burnmap[t, i, j] = 0.6
        end
        # Bottom-right corner
        for i in (N-corner_size+1):N, j in (M-corner_size+1):M
            burnmap[t, i, j] = 0.9
        end
        # Center area (medium risk)
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
    """
    Creates a simple kernel that decreases with distance.
    """
    kernel = Dict{Tuple{Int,Int}, Float64}()
    
    for dx in -kernel_size:kernel_size
        for dy in -kernel_size:kernel_size
            # L-infinity distance
            dist = max(abs(dx), abs(dy))
            if dist <= kernel_size
                # Coverage decreases with distance, normalized
                weight = max(0.0, 1.0 - dist / (kernel_size + 1))
                if weight > 0.01  # Only store non-negligible values
                    kernel[(dx, dy)] = weight
                end
            end
        end
    end
    
    return kernel
end

# ============================================================================
# Performance Test Function
# ============================================================================

function run_performance_test(test_name::String, N::Int, M::Int, N_grounds::Int, N_charging::Int, n_drones::Int, max_battery::Int, use_mask::Bool=false)
    println("="^70)
    println("$test_name")
    println("="^70)
    println("Test parameters:")
    println("  Grid size: $(N) x $(M)")
    println("  Ground stations: $(N_grounds)")
    println("  Charging stations: $(N_charging)")
    println("  Total drones: $(n_drones)")
    println("  Max battery (kernel size): $(max_battery)")
    println()
    
    # Create synthetic burn map
    burnmap = create_synthetic_burnmap(N, M, 1)
    temp_burnmap_file = "temp_perf_burnmap.npy"
    npzwrite(temp_burnmap_file, burnmap)
    
    # Create kernel
    kernel = create_simple_kernel(max_battery)
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    # Create mask if needed
    mask_file = nothing
    if use_mask
        mask = ones(Bool, N, M)
        temp_mask_file = "temp_perf_mask.npy"
        npzwrite(temp_mask_file, mask)
        mask_file = temp_mask_file
    end
    
    times_original = Float64[]
    times_with_allocation = Float64[]
    
    n_runs = 1  # Run each version once (optimization is expensive)
    
    println("Running each version once (optimization is expensive)...")
    println()
    
    # Test original version (without allocation)
    println("Testing ORIGINAL version (without allocation):")
    start_time = time_ns() / 1e9
    if use_mask
        result_orig = Max_Coverage_Kernel_Masked(
            temp_burnmap_file, N_grounds, N_charging, n_drones,
            kernel, kernel_size_x, kernel_size_y, mask_file, false, max_battery
        )
    else
        result_orig = Max_Coverage_Kernel(
            temp_burnmap_file, N_grounds, N_charging, n_drones,
            kernel, kernel_size_x, kernel_size_y, mask_file
        )
    end
    elapsed_orig = (time_ns() / 1e9) - start_time
    push!(times_original, elapsed_orig)
    println("  Time: $(round(elapsed_orig, digits=3)) seconds")
    
    println()
    println("Testing WITH ALLOCATION version:")
    start_time = time_ns() / 1e9
    if use_mask
        result_alloc = Max_Coverage_Kernel_Masked_WithAllocation(
            temp_burnmap_file, N_grounds, N_charging, n_drones,
            kernel, kernel_size_x, kernel_size_y, mask_file, false, max_battery
        )
    else
        result_alloc = Max_Coverage_Kernel_WithAllocation(
            temp_burnmap_file, N_grounds, N_charging, n_drones,
            kernel, kernel_size_x, kernel_size_y, mask_file
        )
    end
    elapsed_alloc = (time_ns() / 1e9) - start_time
    push!(times_with_allocation, elapsed_alloc)
    println("  Time: $(round(elapsed_alloc, digits=3)) seconds")
    
    # Calculate statistics
    avg_original = mean(times_original)
    avg_allocation = mean(times_with_allocation)
    std_original = length(times_original) > 1 ? std(times_original) : 0.0
    std_allocation = length(times_with_allocation) > 1 ? std(times_with_allocation) : 0.0
    
    speedup = avg_original / avg_allocation
    slowdown = avg_allocation / avg_original
    
    println()
    println("="^70)
    println("RESULTS")
    println("="^70)
    println("Original version (without allocation):")
    println("  Time: $(round(avg_original, digits=3)) seconds")
    println()
    println("With allocation version:")
    println("  Time: $(round(avg_allocation, digits=3)) seconds")
    println()
    
    if avg_allocation < avg_original
        speedup_factor = avg_original / avg_allocation
        println("✓ With allocation version is $(round(speedup_factor, digits=2))x FASTER")
    elseif avg_allocation > avg_original
        slowdown_factor = avg_allocation / avg_original
        println("⚠ With allocation version is $(round(slowdown_factor, digits=2))x SLOWER")
    else
        println("≈ Performance is similar")
    end
    
    # Check if difference is significant (more than 10%)
    if abs(avg_original - avg_allocation) / max(avg_original, avg_allocation) > 0.1
        if avg_allocation > avg_original
            println("⚠ SIGNIFICANT SLOWDOWN: Allocation version is $(round((avg_allocation/avg_original - 1) * 100, digits=1))% slower")
        else
            println("✓ SIGNIFICANT SPEEDUP: Allocation version is $(round((avg_original/avg_allocation - 1) * 100, digits=1))% faster")
        end
    else
        println("≈ Difference is not significant (< 10%)")
    end
    
    # Clean up
    rm(temp_burnmap_file, force=true)
    if use_mask && mask_file !== nothing
        rm(mask_file, force=true)
    end
    
    println("="^70)
    println()
    
    return (avg_original, avg_allocation, speedup, slowdown)
end

# ============================================================================
# Run Performance Tests
# ============================================================================

println("Running performance comparison tests on larger instances...")
println()

# Test 1: Medium instance without mask
results1 = run_performance_test(
    "TEST 1: Medium Instance (No Mask)",
    30, 30,  # Grid size
    4,       # Ground stations
    6,       # Charging stations
    15,      # Drones
    8        # Max battery
)

# Test 2: Large instance without mask
results2 = run_performance_test(
    "TEST 2: Large Instance (No Mask)",
    40, 40,  # Grid size
    5,       # Ground stations
    8,       # Charging stations
    20,      # Drones
    10       # Max battery
)

# Test 3: Medium instance with mask
results3 = run_performance_test(
    "TEST 3: Medium Instance (With Mask)",
    30, 30,  # Grid size
    4,       # Ground stations
    6,       # Charging stations
    15,      # Drones
    8,       # Max battery
    true     # Use mask
)

# Test 4: Large instance with mask
results4 = run_performance_test(
    "TEST 4: Large Instance (With Mask)",
    40, 40,  # Grid size
    5,       # Ground stations
    8,       # Charging stations
    20,      # Drones
    10,      # Max battery
    true     # Use mask
)

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("OVERALL SUMMARY")
println("="^70)
println()

all_results = [results1, results2, results3, results4]
test_names = [
    "Test 1: Medium (No Mask)",
    "Test 2: Large (No Mask)",
    "Test 3: Medium (With Mask)",
    "Test 4: Large (With Mask)"
]

for (i, (name, (orig, alloc, speedup, slowdown))) in enumerate(zip(test_names, all_results))
    println("$name:")
    println("  Original: $(round(orig, digits=3))s | Allocation: $(round(alloc, digits=3))s")
    if alloc > orig
        println("  Slowdown: $(round((alloc/orig - 1) * 100, digits=1))%")
    else
        println("  Speedup: $(round((orig/alloc - 1) * 100, digits=1))%")
    end
    println()
end

# Overall average
avg_orig_all = mean([r[1] for r in all_results])
avg_alloc_all = mean([r[2] for r in all_results])

println("Overall Average:")
println("  Original version: $(round(avg_orig_all, digits=3)) seconds")
println("  Allocation version: $(round(avg_alloc_all, digits=3)) seconds")
if avg_alloc_all > avg_orig_all
    overall_slowdown = (avg_alloc_all / avg_orig_all - 1) * 100
    println("  Overall slowdown: $(round(overall_slowdown, digits=1))%")
    if overall_slowdown > 10
        println("  ⚠ SIGNIFICANT OVERALL SLOWDOWN")
    end
else
    overall_speedup = (avg_orig_all / avg_alloc_all - 1) * 100
    println("  Overall speedup: $(round(overall_speedup, digits=1))%")
end

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
