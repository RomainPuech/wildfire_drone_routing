# Test with extreme scenarios where drone concentration should clearly help
# Run this from the julia directory: julia test_extreme_scenarios.jl

using Dates
using Statistics

println("="^70)
println("EXTREME SCENARIOS TEST - Drone Concentration")
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

function create_extreme_burnmap_scenario1(N::Int, M::Int, T::Int=1)
    """
    Scenario 1: Single very high-risk hotspot in one corner
    Most of the grid has low risk, but one small area has very high risk.
    """
    burnmap = fill(0.1, T, N, M)  # Low baseline risk
    
    # Create a very high-risk hotspot in bottom-right corner
    hotspot_size = 3
    for t in 1:T
        for i in (N-hotspot_size):N, j in (M-hotspot_size):M
            burnmap[t, i, j] = 0.95  # Very high risk
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario2(N::Int, M::Int, T::Int=1)
    """
    Scenario 2: Two high-risk areas far apart
    Two hotspots that are far from each other, requiring concentration.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    # Hotspot 1: Top-left corner
    hotspot_size = 4
    for t in 1:T
        for i in 1:hotspot_size, j in 1:hotspot_size
            burnmap[t, i, j] = 0.9
        end
    end
    
    # Hotspot 2: Bottom-right corner
    for t in 1:T
        for i in (N-hotspot_size+1):N, j in (M-hotspot_size+1):M
            burnmap[t, i, j] = 0.9
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario3(N::Int, M::Int, T::Int=1)
    """
    Scenario 3: Single very high-risk area in center
    One large high-risk area in the center that benefits from multiple drones.
    Reduced size for faster computation.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    # Large high-risk area in center (smaller radius for smaller grids)
    center_i = N ÷ 2
    center_j = M ÷ 2
    hotspot_radius = min(4, N ÷ 4, M ÷ 4)  # Adaptive radius, max 4
    for t in 1:T
        for i in (center_i-hotspot_radius):(center_i+hotspot_radius)
            for j in (center_j-hotspot_radius):(center_j+hotspot_radius)
                if 1 <= i <= N && 1 <= j <= M
                    burnmap[t, i, j] = 0.95
                end
            end
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario4(N::Int, M::Int, T::Int=1)
    """
    Scenario 4: Large extended high-risk area
    A large rectangular high-risk area that extends beyond distance 2 from a single station.
    Cells at distance 3-5 from the station center will benefit from multiple drones.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    # Large rectangular high-risk area in center (adaptive size)
    center_i = N ÷ 2
    center_j = M ÷ 2
    # Make it large enough that cells at distance 3-5 from station center are high-risk
    width = min(8, M - 4)
    height = min(8, N - 4)
    for t in 1:T
        for i in (center_i-height÷2):(center_i+height÷2)
            for j in (center_j-width÷2):(center_j+width÷2)
                if 1 <= i <= N && 1 <= j <= M
                    burnmap[t, i, j] = 0.95
                end
            end
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario5(N::Int, M::Int, T::Int=1)
    """
    Scenario 5: High-risk ring at distance 3-4
    High-risk cells are specifically at distance 3-4 from where a station would be placed.
    These cells have lower coverage (50-60% with kernel size 6-8) and benefit from multiple drones.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    # Place high-risk in a ring pattern around center
    center_i = N ÷ 2
    center_j = M ÷ 2
    for t in 1:T
        for i in 1:N
            for j in 1:M
                # Calculate L-infinity distance from center
                dist = max(abs(i - center_i), abs(j - center_j))
                # High risk at distance 3-5 (where coverage is lower and benefits from multiple drones)
                if 3 <= dist <= 5
                    burnmap[t, i, j] = 0.95
                end
            end
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario6(N::Int, M::Int, T::Int=1)
    """
    Scenario 6: Gradient risk with peak at distance 3
    Risk increases with distance from center, peaking at distance 3-4.
    This tests if drones concentrate to cover the medium-distance high-risk cells.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    center_i = N ÷ 2
    center_j = M ÷ 2
    for t in 1:T
        for i in 1:N
            for j in 1:M
                dist = max(abs(i - center_i), abs(j - center_j))
                # Risk peaks at distance 3-4
                if dist <= 6
                    # Higher risk at distance 3-4 (where multiple drones help)
                    if 3 <= dist <= 4
                        burnmap[t, i, j] = 0.95
                    elseif dist == 2 || dist == 5
                        burnmap[t, i, j] = 0.7
                    elseif dist == 1 || dist == 6
                        burnmap[t, i, j] = 0.4
                    else
                        burnmap[t, i, j] = 0.2
                    end
                end
            end
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario7(N::Int, M::Int, T::Int=1)
    """
    Scenario 7: Two large overlapping high-risk areas
    Two large high-risk areas that overlap, creating a very large region.
    A single station can't cover all cells well, multiple drones help.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    # First large area (adaptive size)
    center1_i = N ÷ 3
    center1_j = M ÷ 3
    radius1 = min(5, N ÷ 3, M ÷ 3)
    for t in 1:T
        for i in (center1_i-radius1):(center1_i+radius1)
            for j in (center1_j-radius1):(center1_j+radius1)
                if 1 <= i <= N && 1 <= j <= M
                    dist = max(abs(i - center1_i), abs(j - center1_j))
                    if dist <= radius1
                        burnmap[t, i, j] = max(burnmap[t, i, j], 0.95)
                    end
                end
            end
        end
    end
    
    # Second large area (overlapping, adaptive size)
    center2_i = 2*N ÷ 3
    center2_j = 2*M ÷ 3
    radius2 = min(5, N ÷ 3, M ÷ 3)
    for t in 1:T
        for i in (center2_i-radius2):(center2_i+radius2)
            for j in (center2_j-radius2):(center2_j+radius2)
                if 1 <= i <= N && 1 <= j <= M
                    dist = max(abs(i - center2_i), abs(j - center2_j))
                    if dist <= radius2
                        burnmap[t, i, j] = max(burnmap[t, i, j], 0.95)
                    end
                end
            end
        end
    end
    
    return burnmap
end

function create_extreme_burnmap_scenario8(N::Int, M::Int, T::Int=1)
    """
    Scenario 8: High-risk corridor
    A long, narrow high-risk corridor. Stations placed along it will have
    high-risk cells at various distances, including distance 3+ that benefit from multiple drones.
    """
    burnmap = fill(0.05, T, N, M)  # Very low baseline risk
    
    # Create a diagonal corridor (adaptive size)
    center_i = N ÷ 2
    center_j = M ÷ 2
    corridor_width = 2
    corridor_length = min(10, min(N, M) - 4)
    
    for t in 1:T
        for offset in -corridor_length÷2:corridor_length÷2
            for w in -corridor_width:corridor_width
                i = center_i + offset
                j = center_j + offset + w
                if 1 <= i <= N && 1 <= j <= M
                    burnmap[t, i, j] = 0.95
                end
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

function run_extreme_test(scenario_name::String, create_burnmap_fn, N::Int, M::Int, 
                         N_grounds::Int, N_charging::Int, n_drones::Int, max_battery::Int)
    println("="^70)
    println("SCENARIO: $scenario_name")
    println("="^70)
    println("Grid: $(N)x$(M), $(N_grounds) ground, $(N_charging) charging, $(n_drones) drones")
    println("Kernel size: $max_battery")
    println("Expected: Drones should concentrate at high-risk areas")
    println("Note: Distance 0-2 cells already well-covered (85-100%), distance 3+ benefit from multiple drones")
    println()
    
    # Create extreme burn map
    burnmap = create_burnmap_fn(N, M, 1)
    temp_burnmap_file = "temp_extreme_burnmap_$(scenario_name).npy"
    npzwrite(temp_burnmap_file, burnmap)
    
    # Calculate total risk for context
    total_risk = sum(burnmap)
    max_risk = maximum(burnmap)
    high_risk_cells = count(x -> x > 0.5, burnmap)
    println("Risk Map Statistics:")
    println("  Total risk: $(round(total_risk, digits=2))")
    println("  Max risk: $(round(max_risk, digits=2))")
    println("  High-risk cells (>0.5): $high_risk_cells")
    println()
    
    kernel = create_simple_kernel(max_battery)
    kernel_size_x = max_battery
    kernel_size_y = max_battery
    
    # Print kernel coverage info for context
    center_coverage = get(kernel, (0, 0), 0.0)
    dist1_coverage = get(kernel, (1, 0), 0.0)
    dist2_coverage = get(kernel, (2, 0), 0.0)
    dist3_coverage = get(kernel, (3, 0), 0.0)
    dist4_coverage = get(kernel, (4, 0), 0.0)
    println("Kernel Coverage (for reference):")
    println("  Distance 0: $(round(center_coverage, digits=3)) (1 drone), $(round(min(1.0, center_coverage*2), digits=3)) (2 drones)")
    println("  Distance 1: $(round(dist1_coverage, digits=3)) (1 drone), $(round(min(1.0, dist1_coverage*2), digits=3)) (2 drones)")
    println("  Distance 2: $(round(dist2_coverage, digits=3)) (1 drone), $(round(min(1.0, dist2_coverage*2), digits=3)) (2 drones)")
    println("  Distance 3: $(round(dist3_coverage, digits=3)) (1 drone), $(round(min(1.0, dist3_coverage*2), digits=3)) (2 drones), $(round(min(1.0, dist3_coverage*3), digits=3)) (3 drones)")
    println("  Distance 4: $(round(dist4_coverage, digits=3)) (1 drone), $(round(min(1.0, dist4_coverage*2), digits=3)) (2 drones), $(round(min(1.0, dist4_coverage*3), digits=3)) (3 drones)")
    println()
    
    println("Running ORIGINAL version...")
    result_orig = Max_Coverage_Kernel_WithObjective(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_orig, charging_orig, obj_orig, covered_risk_orig, total_risk2, coverage_pct_orig = result_orig
    
    println("  Objective: $(round(obj_orig, digits=4))")
    println("  Coverage: $(round(coverage_pct_orig, digits=2))%")
    println("  Charging stations: $charging_orig")
    
    println()
    println("Running WITH ALLOCATION version...")
    result_alloc = Max_Coverage_Kernel_WithAllocation_WithObjective(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_alloc, charging_alloc, drone_allocations, obj_alloc, covered_risk_alloc, total_risk3, coverage_pct_alloc = result_alloc
    
    println("  Objective: $(round(obj_alloc, digits=4))")
    println("  Coverage: $(round(coverage_pct_alloc, digits=2))%")
    println("  Charging stations: $charging_alloc")
    println("  Drone allocations: $drone_allocations")
    
    # Analyze concentration
    println()
    println("="^70)
    println("CONCENTRATION ANALYSIS")
    println("="^70)
    
    max_allocation = maximum(drone_allocations)
    min_allocation = minimum(drone_allocations)
    stations_with_drones = count(a > 0 for a in drone_allocations)
    stations_without_drones = count(a == 0 for a in drone_allocations)
    concentration_ratio = max_allocation / (n_drones / max(1, stations_with_drones))
    
    println("Drone Distribution:")
    println("  Total drones: $n_drones")
    println("  Max at one station: $max_allocation")
    println("  Min at one station: $min_allocation")
    println("  Stations with drones: $stations_with_drones")
    println("  Stations without drones: $stations_without_drones")
    println("  Concentration ratio: $(round(concentration_ratio, digits=2))x")
    println("    (1.0 = even distribution, >1.0 = concentrated)")
    
    # Check if concentration happened
    if max_allocation > 1
        println("  ✓ CONCENTRATION DETECTED: At least one station has $max_allocation drones")
    else
        println("  ⚠ NO CONCENTRATION: All stations have ≤1 drone")
    end
    
    if stations_without_drones > 0
        println("  ✓ Some stations have 0 drones (drones concentrated at others)")
    end
    
    # Analyze which stations got drones
    println()
    println("Station Analysis:")
    for (i, (station, alloc)) in enumerate(zip(charging_alloc, drone_allocations))
        # Check if station is near high-risk area
        # station is in 0-based coordinates, burnmap is 1-based
        station_x = station[1] + 1
        station_y = station[2] + 1
        if 1 <= station_x <= N && 1 <= station_y <= M
            station_risk = burnmap[1, station_x, station_y]
            near_high_risk = station_risk > 0.5
            
            marker = near_high_risk ? "🔥" : "  "
            println("  $marker Station $i at $station: $alloc drone(s) (local risk: $(round(station_risk, digits=2)))")
        else
            println("  Station $i at $station: $alloc drone(s) (coordinates out of bounds)")
        end
    end
    
    # Check if high-risk areas got more drones
    println()
    println("High-Risk Area Coverage:")
    high_risk_stations = []
    for (i, (station, alloc)) in enumerate(zip(charging_alloc, drone_allocations))
        station_x = station[1] + 1
        station_y = station[2] + 1
        if 1 <= station_x <= N && 1 <= station_y <= M
            station_risk = burnmap[1, station_x, station_y]
            if station_risk > 0.5
                push!(high_risk_stations, (i, station, alloc, station_risk))
            end
        end
    end
    
    if length(high_risk_stations) > 0
        total_drones_at_high_risk = sum(s[3] for s in high_risk_stations)
        println("  Stations in high-risk areas: $(length(high_risk_stations))")
        println("  Total drones at high-risk stations: $total_drones_at_high_risk")
        println("  Percentage of drones at high-risk: $(round((total_drones_at_high_risk/n_drones)*100, digits=1))%")
        
        for (i, station, alloc, risk) in high_risk_stations
            println("    Station $i at $station: $alloc drone(s) (risk: $(round(risk, digits=2)))")
        end
    else
        println("  ⚠ No stations placed in high-risk areas!")
    end
    
    # Objective comparison
    println()
    println("="^70)
    println("OBJECTIVE COMPARISON")
    println("="^70)
    
    obj_diff = obj_alloc - obj_orig
    obj_diff_pct = (obj_diff / obj_orig) * 100
    
    println("Original objective: $(round(obj_orig, digits=4))")
    println("Allocation objective: $(round(obj_alloc, digits=4))")
    println("Difference: $(round(obj_diff, digits=4)) ($(round(obj_diff_pct, digits=2))%)")
    
    # Note: When n_drones < n_charging_stations, allocation version may have lower objective
    # because stations with 0 drones provide no coverage, while original version counts
    # coverage from all stations regardless of drone assignment
    println("Note: If n_drones < n_charging_stations, allocation version may have lower")
    println("      objective because stations with 0 drones provide no coverage.")
    println("      Original version counts coverage from all stations.")
    println()
    
    if obj_alloc > obj_orig
        improvement = ((obj_alloc / obj_orig) - 1) * 100
        println("✓ Allocation version achieves $(round(improvement, digits=2))% HIGHER objective")
        if max_allocation > 1
            println("  This confirms that concentrating drones improves coverage!")
        end
    elseif obj_alloc < obj_orig
        decrease = (1 - (obj_alloc / obj_orig)) * 100
        println("⚠ Allocation version achieves $(round(decrease, digits=2))% LOWER objective")
        if n_drones < N_charging
            println("  Expected: Fewer drones than stations means some stations have 0 coverage")
            println("  Original version counts coverage from all stations (even without drones)")
        else
            println("  This is unexpected - investigate why concentration isn't helping")
        end
    else
        println("≈ Objectives are equal")
    end
    
    println()
    println("Coverage Comparison:")
    println("  Original: $(round(coverage_pct_orig, digits=2))%")
    println("  Allocation: $(round(coverage_pct_alloc, digits=2))%")
    println("  Difference: $(round(coverage_pct_alloc - coverage_pct_orig, digits=2)) percentage points")
    
    rm(temp_burnmap_file, force=true)
    
    println("="^70)
    println()
    
    return (obj_orig, obj_alloc, max_allocation, stations_without_drones, 
            length(high_risk_stations), obj_diff_pct)
end

# ============================================================================
# Run Extreme Tests
# ============================================================================

println("Running extreme scenario tests...")
println()

# Original scenarios (may not show concentration due to kernel behavior)
# Using very small grids for fast computation
test_configs_original = [
    ("Single Hotspot", create_extreme_burnmap_scenario1, 12, 12, 1, 2, 3, 5),
    ("Two Distant Hotspots", create_extreme_burnmap_scenario2, 12, 12, 1, 3, 3, 5),
    ("Large Center Hotspot", create_extreme_burnmap_scenario3, 12, 12, 1, 2, 3, 5),
]

# New scenarios designed with kernel coverage in mind
# These focus on distance 3+ cells that benefit from multiple drones
# Using very small grids for fast computation
test_configs_new = [
    # Scenario 4: Large extended area - cells at distance 3-5 are high-risk
    ("Large Extended Area", create_extreme_burnmap_scenario4, 12, 12, 1, 2, 3, 5),
    
    # Scenario 5: High-risk ring at distance 3-4 - specifically targets cells that benefit from multiple drones
    ("High-Risk Ring (dist 3-4)", create_extreme_burnmap_scenario5, 12, 12, 1, 2, 3, 5),
    
    # Scenario 6: Gradient with peak at distance 3-4
    ("Gradient Peak (dist 3-4)", create_extreme_burnmap_scenario6, 12, 12, 1, 2, 3, 5),
    
    # Scenario 7: Two large overlapping areas
    ("Two Overlapping Areas", create_extreme_burnmap_scenario7, 15, 15, 1, 3, 3, 5),
    
    # Scenario 8: High-risk corridor
    ("High-Risk Corridor", create_extreme_burnmap_scenario8, 12, 12, 1, 2, 3, 5),
    
    # Test with larger kernel where distance 3+ cells have more room for improvement
    ("Large Extended Area (kernel 6)", create_extreme_burnmap_scenario4, 15, 15, 1, 2, 3, 6),
    ("High-Risk Ring (kernel 6)", create_extreme_burnmap_scenario5, 15, 15, 1, 2, 3, 6),
]

# Combine all scenarios
test_configs = vcat(test_configs_original, test_configs_new)

all_results = []

total_tests = length(test_configs)
println("Total tests to run: $total_tests")
println("="^70)
println()

for (test_idx, (scenario_name, create_fn, N, M, N_grounds, N_charging, n_drones, max_battery)) in enumerate(test_configs)
    println(">>> TEST $test_idx/$total_tests: $scenario_name <<<")
    println("Grid: $(N)x$(M), $(N_grounds) ground, $(N_charging) charging, $(n_drones) drones, kernel: $max_battery")
    flush(stdout)
    
    result = run_extreme_test(scenario_name, create_fn, N, M, N_grounds, N_charging, n_drones, max_battery)
    push!(all_results, (scenario_name, result))
    
    println(">>> Completed test $test_idx/$total_tests <<<")
    println()
    flush(stdout)
end

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("OVERALL SUMMARY")
println("="^70)
println()

println("Results by Scenario:")
for (scenario_name, (obj_orig, obj_alloc, max_alloc, stations_without, high_risk_count, obj_diff_pct)) in all_results
    println("  $scenario_name:")
    println("    Objective improvement: $(round(obj_diff_pct, digits=2))%")
    println("    Max drones at one station: $max_alloc")
    println("    Stations without drones: $stations_without")
    println("    Stations in high-risk areas: $high_risk_count")
    
    if max_alloc > 1
        println("    ✓ Concentration occurred")
    else
        println("    ⚠ No concentration")
    end
    println()
end

avg_improvement = mean([r[2][6] for r in all_results])
avg_max_alloc = mean([r[2][3] for r in all_results])

println("Average Performance:")
println("  Average objective improvement: $(round(avg_improvement, digits=2))%")
println("  Average max drones per station: $(round(avg_max_alloc, digits=2))")

if avg_max_alloc > 1.5
    println("  ✓ Strong concentration detected across scenarios")
elseif avg_max_alloc > 1.0
    println("  ⚠ Weak concentration - may need formulation review")
else
    println("  ⚠ No concentration - formulation may not be working as intended")
end

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
