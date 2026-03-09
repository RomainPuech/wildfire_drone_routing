# Test with uniform grid and equal drones/charging stations
# Run this from the julia directory: julia test_uniform_grid.jl

using Dates
using Random

println("="^70)
println("UNIFORM GRID TEST - Equal Drones and Charging Stations")
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

function create_uniform_burnmap(N::Int, M::Int, risk_value::Float64=0.5, T::Int=1)
    """
    Creates a uniform burn map where all cells have the same risk value.
    """
    burnmap = fill(risk_value, T, N, M)
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
# Test Function
# ============================================================================

function run_uniform_test(test_num::Int, N::Int, M::Int, N_grounds::Int, N_charging::Int, max_battery::Int)
    println("="^70)
    println("TEST $test_num: Uniform Grid $(N)x$(M), $(N_grounds) ground, $(N_charging) charging, $(N_charging) drones")
    println("="^70)
    println("Note: Number of drones = number of charging stations (1 drone per station expected)")
    println()
    
    # Create uniform burn map
    burnmap = create_uniform_burnmap(N, M, 0.5, 1)
    temp_burnmap_file = "temp_uniform_burnmap_$(test_num).npy"
    npzwrite(temp_burnmap_file, burnmap)
    
    kernel = create_simple_kernel(max_battery)
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    n_drones = N_charging  # Equal to number of charging stations
    
    println("Running ORIGINAL version...")
    result_orig = Max_Coverage_Kernel(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_orig, charging_orig = result_orig
    
    println("  Ground sensors: $(length(ground_orig))")
    println("  Charging stations: $(length(charging_orig))")
    println("  Ground locations: $ground_orig")
    println("  Charging locations: $charging_orig")
    
    println()
    println("Running WITH ALLOCATION version...")
    result_alloc = Max_Coverage_Kernel_WithAllocation(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_alloc, charging_alloc, drone_allocations = result_alloc
    
    println("  Ground sensors: $(length(ground_alloc))")
    println("  Charging stations: $(length(charging_alloc))")
    println("  Ground locations: $ground_alloc")
    println("  Charging locations: $charging_alloc")
    println("  Drone allocations: $drone_allocations")
    
    println()
    println("="^70)
    println("COMPARISON")
    println("="^70)
    
    # Compare results
    ground_match = ground_orig == ground_alloc
    charging_match = charging_orig == charging_alloc
    
    println("Ground sensors match: $ground_match")
    if !ground_match
        println("  Original: $ground_orig")
        println("  Allocation: $ground_alloc")
        println("  Difference: $(setdiff(Set(ground_orig), Set(ground_alloc))) vs $(setdiff(Set(ground_alloc), Set(ground_orig)))")
    end
    
    println("Charging stations match: $charging_match")
    if !charging_match
        println("  Original: $charging_orig")
        println("  Allocation: $charging_alloc")
        println("  Difference: $(setdiff(Set(charging_orig), Set(charging_alloc))) vs $(setdiff(Set(charging_alloc), Set(charging_orig)))")
    end
    
    # Check drone allocations
    println()
    println("Drone Allocation Analysis:")
    println("  Allocations: $drone_allocations")
    println("  Sum: $(sum(drone_allocations)) (expected: $n_drones)")
    println("  Expected per station: 1 (since n_drones = n_charging)")
    
    all_one = all(a == 1 for a in drone_allocations)
    if all_one
        println("  ✓ All stations have exactly 1 drone (as expected)")
    else
        println("  ⚠ Not all stations have 1 drone:")
        for (i, alloc) in enumerate(drone_allocations)
            if alloc != 1
                println("    Station $i: $alloc drones")
            end
        end
    end
    
    # Check if solutions are equivalent
    println()
    if ground_match && charging_match
        println("✓ SOLUTIONS ARE IDENTICAL")
    else
        println("⚠ SOLUTIONS DIFFER")
        
        # Check if they're just permutations (same set, different order)
        if Set(ground_orig) == Set(ground_alloc) && Set(charging_orig) == Set(charging_alloc)
            println("  But they contain the same locations (just different order)")
        else
            println("  They contain different locations")
        end
    end
    
    # Sanity check: with uniform grid and 1 drone per station, 
    # allocation version should allocate 1 to each
    if n_drones == N_charging && !all_one
        println()
        println("⚠ UNEXPECTED: With equal drones and stations, expected 1 per station")
        println("  This suggests the allocation formulation may be finding a different optimum")
    end
    
    rm(temp_burnmap_file, force=true)
    
    println("="^70)
    println()
    
    return (ground_match, charging_match, all_one, ground_orig, ground_alloc, charging_orig, charging_alloc, drone_allocations)
end

# ============================================================================
# Run Tests
# ============================================================================

println("Running tests with uniform grids and equal drones/charging stations...")
println()

test_configs = [
    (30, 30, 3, 5, 8),   # Medium grid, 5 charging = 5 drones
    (40, 40, 4, 6, 10),  # Larger grid, 6 charging = 6 drones
    (50, 50, 5, 8, 12),  # Large grid, 8 charging = 8 drones
]

all_results = []

for (idx, (N, M, N_grounds, N_charging, max_battery)) in enumerate(test_configs)
    result = run_uniform_test(idx, N, M, N_grounds, N_charging, max_battery)
    push!(all_results, result)
end

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("OVERALL SUMMARY")
println("="^70)
println()

all_ground_match = all(r[1] for r in all_results)
all_charging_match = all(r[2] for r in all_results)
all_one_drone = all(r[3] for r in all_results)

println("Results across all tests:")
println("  Ground sensors match in all tests: $all_ground_match")
println("  Charging stations match in all tests: $all_charging_match")
println("  All stations have 1 drone in all tests: $all_one_drone")
println()

if all_ground_match && all_charging_match
    println("✓ Both methods produce IDENTICAL solutions")
else
    println("⚠ Methods produce DIFFERENT solutions")
    println()
    println("Detailed differences:")
    for (i, (ground_match, charging_match, all_one, g_orig, g_alloc, c_orig, c_alloc, drones)) in enumerate(all_results)
        println("  Test $i:")
        if !ground_match
            println("    Ground sensors differ")
        end
        if !charging_match
            println("    Charging stations differ")
            println("      Original: $c_orig")
            println("      Allocation: $c_alloc")
        end
        if !all_one
            println("    Drone allocations: $drones")
        end
    end
end

println()
println("Analysis:")
println("  With uniform risk and 1 drone per station, both formulations should")
println("  theoretically produce similar results. Differences may indicate:")
println("  - Different optimal solutions (multiple optima exist)")
println("  - Formulation differences affecting the optimization path")
println("  - Solver behavior differences")

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
