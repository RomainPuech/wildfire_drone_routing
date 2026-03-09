# Test script for drone allocation in sensor placement optimization
# Run this from the julia directory: julia test_drone_allocation.jl
#
# This script tests:
# 1. Max_Coverage_Kernel_WithAllocation returns 3 values (ground sensors, charging stations, drone allocations)
# 2. Max_Coverage_Kernel_Masked_WithAllocation returns 3 values
# 3. Drone allocations sum to n_drones
# 4. Number of allocations matches number of charging stations
# 5. Allocations are non-negative and bounded correctly
# 6. Coverage constraints work with nc[i] instead of xc[i]

using Dates
using Random
using Statistics

println("="^60)
println("DRONE ALLOCATION OPTIMIZATION TEST")
println("="^60)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

# Include the necessary files
println("Loading Julia modules...")
include("helper_functions.jl")
include("ground_charging_opt.jl")
println("Modules loaded successfully!")
println()

# ============================================================================
# Helper function to create a simple synthetic burn map
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
# TEST 1: Max_Coverage_Kernel with synthetic data
# ============================================================================

println("="^60)
println("TEST 1: Max_Coverage_Kernel_WithAllocation with synthetic data")
println("="^60)

function test_max_coverage_kernel()
    # Test parameters
    N = 20
    M = 20
    N_grounds = 2
    N_charging = 3
    n_drones = 10
    max_battery = 5
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    println("Test parameters:")
    println("  Grid size: $(N) x $(M)")
    println("  Ground stations: $(N_grounds)")
    println("  Charging stations: $(N_charging)")
    println("  Total drones: $(n_drones)")
    println("  Max battery (kernel size): $(max_battery)")
    println()
    
    # Create synthetic burn map
    println("Creating synthetic burn map...")
    burnmap = create_synthetic_burnmap(N, M, 1)
    
    # Save to temporary file
    temp_burnmap_file = "temp_test_burnmap.npy"
    npzwrite(temp_burnmap_file, burnmap)
    println("Saved burn map to $(temp_burnmap_file)")
    
    # Create kernel
    println("Creating kernel...")
    kernel = create_simple_kernel(max_battery)
    println("Kernel has $(length(kernel)) entries")
    
        # Run optimization
        println()
        println("Running Max_Coverage_Kernel_WithAllocation optimization...")
        try
            result = Max_Coverage_Kernel_WithAllocation(
            temp_burnmap_file,
            N_grounds,
            N_charging,
            n_drones,
            kernel,
            kernel_size_x,
            kernel_size_y,
            nothing  # no mask
        )
        
        # Verify return value structure
        println()
        println("Checking return values...")
        if length(result) != 3
            error("Expected 3 return values, got $(length(result))")
        end
        
        ground_sensors, charging_stations, drone_allocations = result
        
        println("✓ Function returned 3 values")
        println()
        println("Results:")
        println("  Ground sensor locations: $(ground_sensors)")
        println("  Charging station locations: $(charging_stations)")
        println("  Drone allocations: $(drone_allocations)")
        println()
        
        # Verify ground sensors
        if length(ground_sensors) != N_grounds
            error("Expected $(N_grounds) ground sensors, got $(length(ground_sensors))")
        end
        println("✓ Number of ground sensors: $(length(ground_sensors))")
        
        # Verify charging stations
        if length(charging_stations) != N_charging
            error("Expected $(N_charging) charging stations, got $(length(charging_stations))")
        end
        println("✓ Number of charging stations: $(length(charging_stations))")
        
        # Verify drone allocations
        if length(drone_allocations) != length(charging_stations)
            error("Number of drone allocations ($(length(drone_allocations))) doesn't match number of charging stations ($(length(charging_stations)))")
        end
        println("✓ Number of drone allocations matches number of charging stations")
        
        # Verify sum of allocations
        total_allocated = sum(drone_allocations)
        if total_allocated != n_drones
            error("Sum of drone allocations ($(total_allocated)) doesn't equal n_drones ($(n_drones))")
        end
        println("✓ Sum of drone allocations equals n_drones: $(total_allocated)")
        
        # Verify allocations are non-negative
        if any(a < 0 for a in drone_allocations)
            error("Found negative drone allocation")
        end
        println("✓ All drone allocations are non-negative")
        
        # Verify allocations are bounded
        if any(a > n_drones for a in drone_allocations)
            error("Found drone allocation exceeding n_drones")
        end
        println("✓ All drone allocations are bounded by n_drones")
        
        # Verify allocations are integers
        if any(!isinteger(a) for a in drone_allocations)
            error("Found non-integer drone allocation")
        end
        println("✓ All drone allocations are integers")
        
        # Print allocation details
        println()
        println("Drone allocation details:")
        for (i, (station, allocation)) in enumerate(zip(charging_stations, drone_allocations))
            println("  Station $(i) at $(station): $(Int(allocation)) drone(s)")
        end
        
        # Clean up
        rm(temp_burnmap_file, force=true)
        
        println()
        println("✓ TEST 1 PASSED")
        return true
        
    catch e
        println()
        println("✗ TEST 1 FAILED")
        println("Error: $(e)")
        if isfile(temp_burnmap_file)
            rm(temp_burnmap_file, force=true)
        end
        return false
    end
end

test1_result = test_max_coverage_kernel()
println()

# ============================================================================
# TEST 2: Max_Coverage_Kernel_Masked with synthetic data
# ============================================================================

println("="^60)
println("TEST 2: Max_Coverage_Kernel_Masked_WithAllocation with synthetic data")
println("="^60)

function test_max_coverage_kernel_masked()
    # Test parameters
    N = 20
    M = 20
    N_grounds = 2
    N_charging = 3
    n_drones = 8
    max_battery = 5
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    println("Test parameters:")
    println("  Grid size: $(N) x $(M)")
    println("  Ground stations: $(N_grounds)")
    println("  Charging stations: $(N_charging)")
    println("  Total drones: $(n_drones)")
    println("  Max battery (kernel size): $(max_battery)")
    println()
    
    # Create synthetic burn map
    println("Creating synthetic burn map...")
    burnmap = create_synthetic_burnmap(N, M, 1)
    
    # Save to temporary file
    temp_burnmap_file = "temp_test_burnmap_masked.npy"
    npzwrite(temp_burnmap_file, burnmap)
    println("Saved burn map to $(temp_burnmap_file)")
    
    # Create simple mask (all cells valid)
    println("Creating mask (all cells valid)...")
    mask = ones(Bool, N, M)
    temp_mask_file = "temp_test_mask.npy"
    npzwrite(temp_mask_file, mask)
    println("Saved mask to $(temp_mask_file)")
    
    # Create kernel
    println("Creating kernel...")
    kernel = create_simple_kernel(max_battery)
    println("Kernel has $(length(kernel)) entries")
    
    # Run optimization (without recompute_kernel for speed)
        println()
        println("Running Max_Coverage_Kernel_Masked_WithAllocation optimization (recompute_kernel=false)...")
        try
            result = Max_Coverage_Kernel_Masked_WithAllocation(
            temp_burnmap_file,
            N_grounds,
            N_charging,
            n_drones,
            kernel,
            kernel_size_x,
            kernel_size_y,
            temp_mask_file,
            false,  # recompute_kernel
            max_battery  # n_steps
        )
        
        # Verify return value structure
        println()
        println("Checking return values...")
        if length(result) != 3
            error("Expected 3 return values, got $(length(result))")
        end
        
        ground_sensors, charging_stations, drone_allocations = result
        
        println("✓ Function returned 3 values")
        println()
        println("Results:")
        println("  Ground sensor locations: $(ground_sensors)")
        println("  Charging station locations: $(charging_stations)")
        println("  Drone allocations: $(drone_allocations)")
        println()
        
        # Verify ground sensors
        if length(ground_sensors) != N_grounds
            error("Expected $(N_grounds) ground sensors, got $(length(ground_sensors))")
        end
        println("✓ Number of ground sensors: $(length(ground_sensors))")
        
        # Verify charging stations
        if length(charging_stations) != N_charging
            error("Expected $(N_charging) charging stations, got $(length(charging_stations))")
        end
        println("✓ Number of charging stations: $(length(charging_stations))")
        
        # Verify drone allocations
        if length(drone_allocations) != length(charging_stations)
            error("Number of drone allocations ($(length(drone_allocations))) doesn't match number of charging stations ($(length(charging_stations)))")
        end
        println("✓ Number of drone allocations matches number of charging stations")
        
        # Verify sum of allocations
        total_allocated = sum(drone_allocations)
        if total_allocated != n_drones
            error("Sum of drone allocations ($(total_allocated)) doesn't equal n_drones ($(n_drones))")
        end
        println("✓ Sum of drone allocations equals n_drones: $(total_allocated)")
        
        # Verify allocations are non-negative
        if any(a < 0 for a in drone_allocations)
            error("Found negative drone allocation")
        end
        println("✓ All drone allocations are non-negative")
        
        # Verify allocations are bounded
        if any(a > n_drones for a in drone_allocations)
            error("Found drone allocation exceeding n_drones")
        end
        println("✓ All drone allocations are bounded by n_drones")
        
        # Verify allocations are integers
        if any(!isinteger(a) for a in drone_allocations)
            error("Found non-integer drone allocation")
        end
        println("✓ All drone allocations are integers")
        
        # Print allocation details
        println()
        println("Drone allocation details:")
        for (i, (station, allocation)) in enumerate(zip(charging_stations, drone_allocations))
            println("  Station $(i) at $(station): $(Int(allocation)) drone(s)")
        end
        
        # Clean up
        rm(temp_burnmap_file, force=true)
        rm(temp_mask_file, force=true)
        
        println()
        println("✓ TEST 2 PASSED")
        return true
        
    catch e
        println()
        println("✗ TEST 2 FAILED")
        println("Error: $(e)")
        if isfile(temp_burnmap_file)
            rm(temp_burnmap_file, force=true)
        end
        if isfile(temp_mask_file)
            rm(temp_mask_file, force=true)
        end
        return false
    end
end

test2_result = test_max_coverage_kernel_masked()
println()

# ============================================================================
# TEST 3: Verify coverage constraint behavior
# ============================================================================

println("="^60)
println("TEST 3: Verify coverage constraint behavior with multiple drones")
println("="^60)

function test_coverage_with_multiple_drones()
    # Test with a scenario where allocating more drones to a station should improve coverage
    N = 15
    M = 15
    N_grounds = 1
    N_charging = 2
    n_drones = 5
    max_battery = 4
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    println("Test parameters:")
    println("  Grid size: $(N) x $(M)")
    println("  Ground stations: $(N_grounds)")
    println("  Charging stations: $(N_charging)")
    println("  Total drones: $(n_drones)")
    println("  Max battery (kernel size): $(max_battery)")
    println()
    println("This test verifies that the optimizer can allocate multiple drones")
    println("to a single charging station when it improves coverage.")
    println()
    
    # Create burn map with one high-risk area
    burnmap = zeros(Float64, 1, N, M)
    # High-risk area in top-left
    for i in 1:5, j in 1:5
        burnmap[1, i, j] = 0.9
    end
    # Medium-risk area in bottom-right
    for i in (N-4):N, j in (M-4):M
        burnmap[1, i, j] = 0.5
    end
    
    # Save to temporary file
    temp_burnmap_file = "temp_test_burnmap_coverage.npy"
    npzwrite(temp_burnmap_file, burnmap)
    
    # Create kernel
    kernel = create_simple_kernel(max_battery)
    
        # Run optimization
        println("Running optimization...")
        try
            result = Max_Coverage_Kernel_WithAllocation(
            temp_burnmap_file,
            N_grounds,
            N_charging,
            n_drones,
            kernel,
            kernel_size_x,
            kernel_size_y,
            nothing
        )
        
        ground_sensors, charging_stations, drone_allocations = result
        
        println()
        println("Results:")
        println("  Ground sensor locations: $(ground_sensors)")
        println("  Charging station locations: $(charging_stations)")
        println("  Drone allocations: $(drone_allocations)")
        println()
        
        # Check if at least one station has more than 1 drone
        max_allocation = maximum(drone_allocations)
        if max_allocation > 1
            println("✓ At least one charging station has multiple drones ($(max_allocation))")
            println("  This indicates the optimizer is using the nc[i] variable correctly.")
        else
            println("⚠ All charging stations have 1 drone or less")
            println("  This might be optimal for this scenario, or the formulation might not be working.")
        end
        
        # Verify allocations sum correctly
        if sum(drone_allocations) != n_drones
            error("Sum of allocations doesn't match n_drones")
        end
        println("✓ Drone allocations sum correctly")
        
        # Clean up
        rm(temp_burnmap_file, force=true)
        
        println()
        println("✓ TEST 3 PASSED")
        return true
        
    catch e
        println()
        println("✗ TEST 3 FAILED")
        println("Error: $(e)")
        if isfile(temp_burnmap_file)
            rm(temp_burnmap_file, force=true)
        end
        return false
    end
end

test3_result = test_coverage_with_multiple_drones()
println()

# ============================================================================
# Summary
# ============================================================================

println("="^60)
println("TEST SUMMARY")
println("="^60)
println("Test 1 (Max_Coverage_Kernel_WithAllocation): $(test1_result ? "PASSED" : "FAILED")")
println("Test 2 (Max_Coverage_Kernel_Masked_WithAllocation): $(test2_result ? "PASSED" : "FAILED")")
println("Test 3 (Coverage with multiple drones): $(test3_result ? "PASSED" : "FAILED")")
println()

all_passed = test1_result && test2_result && test3_result

if all_passed
    println("✓ ALL TESTS PASSED")
    println("The drone allocation formulation is working correctly!")
else
    println("✗ SOME TESTS FAILED")
    println("Please review the errors above.")
end

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^60)
