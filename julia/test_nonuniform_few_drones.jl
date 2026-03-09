# Test with non-uniform risk and fewer drones than charging stations
# Run this from the julia directory: julia test_nonuniform_few_drones.jl

using Dates

println("="^70)
println("NON-UNIFORM RISK TEST - Fewer Drones Than Charging Stations")
println("="^70)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

# Include the necessary files
println("Loading Julia modules...")
include("helper_functions.jl")
include("ground_charging_opt.jl")
include("test_objective_comparison.jl")  # For objective extraction functions

println("Modules loaded successfully!")
println()

# ============================================================================
# Helper functions
# ============================================================================

function create_synthetic_burnmap(N::Int, M::Int, T::Int=1)
    """
    Creates a non-uniform burn map with high-risk areas in the corners.
    """
    burnmap = zeros(Float64, T, N, M)
    corner_size = min(5, N÷3, M÷3)
    for t in 1:T
        # Top-left corner - high risk
        for i in 1:corner_size, j in 1:corner_size
            burnmap[t, i, j] = 0.8
        end
        # Top-right corner - high risk
        for i in 1:corner_size, j in (M-corner_size+1):M
            burnmap[t, i, j] = 0.7
        end
        # Bottom-left corner - medium risk
        for i in (N-corner_size+1):N, j in 1:corner_size
            burnmap[t, i, j] = 0.6
        end
        # Bottom-right corner - very high risk
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

function run_test(test_num::Int, N::Int, M::Int, N_grounds::Int, N_charging::Int, n_drones::Int, max_battery::Int)
    println("="^70)
    println("TEST $test_num: Grid $(N)x$(M), $(N_grounds) ground, $(N_charging) charging, $(n_drones) drones")
    println("="^70)
    println("Note: Fewer drones ($n_drones) than charging stations ($N_charging)")
    println("      Allocation version should concentrate drones at high-value stations")
    println()
    
    # Create non-uniform burn map
    burnmap = create_synthetic_burnmap(N, M, 1)
    temp_burnmap_file = "temp_nonuniform_burnmap_$(test_num).npy"
    npzwrite(temp_burnmap_file, burnmap)
    
    kernel = create_simple_kernel(max_battery)
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    println("Running ORIGINAL version...")
    result_orig = Max_Coverage_Kernel_WithObjective(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_orig, charging_orig, obj_orig, covered_risk_orig, total_risk, coverage_pct_orig = result_orig
    
    println("  Objective: $(round(obj_orig, digits=4))")
    println("  Coverage: $(round(coverage_pct_orig, digits=2))%")
    println("  Ground sensors: $(length(ground_orig))")
    println("  Charging stations: $(length(charging_orig))")
    println("  Charging locations: $charging_orig")
    
    println()
    println("Running WITH ALLOCATION version...")
    result_alloc = Max_Coverage_Kernel_WithAllocation_WithObjective(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_alloc, charging_alloc, drone_allocations, obj_alloc, covered_risk_alloc, total_risk2, coverage_pct_alloc = result_alloc
    
    println("  Objective: $(round(obj_alloc, digits=4))")
    println("  Coverage: $(round(coverage_pct_alloc, digits=2))%")
    println("  Ground sensors: $(length(ground_alloc))")
    println("  Charging stations: $(length(charging_alloc))")
    println("  Charging locations: $charging_alloc")
    println("  Drone allocations: $drone_allocations")
    
    # Analyze drone allocations
    println()
    println("Drone Allocation Analysis:")
    println("  Total drones: $n_drones")
    println("  Total charging stations: $(length(charging_alloc))")
    println("  Allocations: $drone_allocations")
    println("  Sum: $(sum(drone_allocations)) (expected: $n_drones)")
    
    # Check if drones are concentrated
    max_allocation = maximum(drone_allocations)
    min_allocation = minimum(drone_allocations)
    stations_with_drones = count(a > 0 for a in drone_allocations)
    stations_without_drones = count(a == 0 for a in drone_allocations)
    
    println("  Max drones at one station: $max_allocation")
    println("  Min drones at one station: $min_allocation")
    println("  Stations with drones: $stations_with_drones")
    println("  Stations without drones: $stations_without_drones")
    
    if stations_without_drones > 0
        println("  ✓ Some stations have 0 drones (expected when n_drones < n_charging)")
    end
    
    if max_allocation > 1
        println("  ✓ Drones are concentrated: at least one station has $max_allocation drones")
    else
        println("  ⚠ All stations have at most 1 drone (may not be optimal)")
    end
    
    # Check which stations got drones
    println()
    println("Station Analysis:")
    for (i, (station, alloc)) in enumerate(zip(charging_alloc, drone_allocations))
        println("  Station $i at $station: $alloc drone(s)")
    end
    
    # Compare objectives
    println()
    println("="^70)
    println("OBJECTIVE COMPARISON")
    println("="^70)
    
    obj_diff = obj_alloc - obj_orig
    obj_diff_pct = (obj_diff / obj_orig) * 100
    
    println("Original objective: $(round(obj_orig, digits=4))")
    println("Allocation objective: $(round(obj_alloc, digits=4))")
    println("Difference: $(round(obj_diff, digits=4)) ($(round(obj_diff_pct, digits=2))%)")
    
    if obj_alloc > obj_orig
        improvement = ((obj_alloc / obj_orig) - 1) * 100
        println("✓ Allocation version achieves $(round(improvement, digits=2))% HIGHER objective")
        println("  This shows the benefit of concentrating drones at optimal stations")
    elseif obj_alloc < obj_orig
        decrease = (1 - (obj_alloc / obj_orig)) * 100
        println("⚠ Allocation version achieves $(round(decrease, digits=2))% LOWER objective")
    else
        println("≈ Objectives are equal")
    end
    
    println()
    println("Coverage Comparison:")
    println("  Original: $(round(coverage_pct_orig, digits=2))%")
    println("  Allocation: $(round(coverage_pct_alloc, digits=2))%")
    println("  Improvement: $(round(coverage_pct_alloc - coverage_pct_orig, digits=2)) percentage points")
    
    # Sanity checks
    println()
    println("Sanity Checks:")
    
    if sum(drone_allocations) == n_drones
        println("  ✓ Total drones allocated equals n_drones")
    else
        println("  ⚠ Total drones allocated ($(sum(drone_allocations))) != n_drones ($n_drones)")
    end
    
    if all(a >= 0 for a in drone_allocations)
        println("  ✓ All allocations are non-negative")
    else
        println("  ⚠ Found negative allocation")
    end
    
    if all(a <= n_drones for a in drone_allocations)
        println("  ✓ All allocations are bounded by n_drones")
    else
        println("  ⚠ Found allocation exceeding n_drones")
    end
    
    if length(drone_allocations) == length(charging_alloc)
        println("  ✓ Number of allocations matches number of stations")
    else
        println("  ⚠ Mismatch in allocation count")
    end
    
    # Check if allocation makes sense (stations near high-risk areas should get more drones)
    println()
    println("Allocation Rationale Check:")
    println("  High-risk areas are in corners: (0,0), (0,M-1), (N-1,0), (N-1,M-1)")
    println("  Stations near high-risk areas should ideally get more drones")
    
    high_risk_corners = [(0, 0), (0, M-1), (N-1, 0), (N-1, M-1)]
    for (i, (station, alloc)) in enumerate(zip(charging_alloc, drone_allocations))
        # Check distance to nearest high-risk corner
        min_dist = minimum([max(abs(station[1] - corner[1]), abs(station[2] - corner[2])) 
                           for corner in high_risk_corners])
        if min_dist <= max_battery && alloc > 0
            println("    Station $i at $station: $alloc drone(s) - near high-risk area (dist: $min_dist)")
        end
    end
    
    rm(temp_burnmap_file, force=true)
    
    println("="^70)
    println()
    
    return (obj_orig, obj_alloc, coverage_pct_orig, coverage_pct_alloc, max_allocation, stations_without_drones)
end

# ============================================================================
# Run Tests
# ============================================================================

println("Running tests with non-uniform risk and fewer drones than charging stations...")
println()

test_configs = [
    (30, 30, 3, 6, 5, 8),   # 5 drones, 6 charging stations
    (35, 35, 4, 8, 5, 10),  # 5 drones, 8 charging stations
    (40, 40, 4, 10, 5, 12), # 5 drones, 10 charging stations
]

all_results = []

for (idx, (N, M, N_grounds, N_charging, n_drones, max_battery)) in enumerate(test_configs)
    result = run_test(idx, N, M, N_grounds, N_charging, n_drones, max_battery)
    push!(all_results, result)
end

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("OVERALL SUMMARY")
println("="^70)
println()

avg_obj_orig = mean([r[1] for r in all_results])
avg_obj_alloc = mean([r[2] for r in all_results])
avg_coverage_orig = mean([r[3] for r in all_results])
avg_coverage_alloc = mean([r[4] for r in all_results])

println("Average Performance:")
println("  Original objective: $(round(avg_obj_orig, digits=4))")
println("  Allocation objective: $(round(avg_obj_alloc, digits=4))")
println("  Improvement: $(round((avg_obj_alloc/avg_obj_orig - 1) * 100, digits=2))%")
println()
println("  Original coverage: $(round(avg_coverage_orig, digits=2))%")
println("  Allocation coverage: $(round(avg_coverage_alloc, digits=2))%")
println("  Improvement: $(round(avg_coverage_alloc - avg_coverage_orig, digits=2)) percentage points")
println()

println("Allocation Behavior:")
all_max_allocations = [r[5] for r in all_results]
all_stations_without = [r[6] for r in all_results]

avg_max_allocation = mean(all_max_allocations)
avg_stations_without = mean(all_stations_without)

println("  Average max drones per station: $(round(avg_max_allocation, digits=2))")
println("  Average stations without drones: $(round(avg_stations_without, digits=2))")

if avg_max_allocation > 1
    println("  ✓ Drones are being concentrated (max > 1)")
end

if avg_stations_without > 0
    println("  ✓ Some stations get 0 drones (expected when n_drones < n_charging)")
end

println()
if avg_obj_alloc > avg_obj_orig
    println("✓ Allocation version consistently achieves HIGHER objectives")
    println("  This demonstrates the benefit of intelligent drone allocation")
else
    println("⚠ Results are mixed - review individual tests")
end

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
