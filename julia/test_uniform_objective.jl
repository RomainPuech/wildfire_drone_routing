# Test objective values for uniform grid with equal drones/charging stations
# Run this from the julia directory: julia test_uniform_objective.jl

using Dates

println("="^70)
println("UNIFORM GRID OBJECTIVE COMPARISON")
println("="^70)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

include("helper_functions.jl")
include("ground_charging_opt.jl")
include("test_objective_comparison.jl")  # Reuse the objective extraction functions

function create_uniform_burnmap(N::Int, M::Int, risk_value::Float64=0.5, T::Int=1)
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

println("Testing with uniform grid and equal drones/charging stations...")
println()

N, M = 30, 30
N_grounds = 3
N_charging = 5
n_drones = N_charging  # Equal to number of charging stations
max_battery = 8

burnmap = create_uniform_burnmap(N, M, 0.5, 1)
temp_burnmap_file = "temp_uniform_obj.npy"
npzwrite(temp_burnmap_file, burnmap)

kernel = create_simple_kernel(max_battery)
kernel_size_x = max_battery
kernel_size_y = max_battery

println("Grid: $(N)x$(M), uniform risk = 0.5")
println("Ground stations: $N_grounds")
println("Charging stations: $N_charging")
println("Drones: $n_drones (1 per station expected)")
println()

println("Running ORIGINAL version...")
result_orig = Max_Coverage_Kernel_WithObjective(
    temp_burnmap_file, N_grounds, N_charging, n_drones,
    kernel, kernel_size_x, kernel_size_y, nothing
)
ground_orig, charging_orig, obj_orig, covered_risk_orig, total_risk, coverage_pct_orig = result_orig

println("  Objective: $(round(obj_orig, digits=6))")
println("  Coverage: $(round(coverage_pct_orig, digits=4))%")
println("  Ground: $ground_orig")
println("  Charging: $charging_orig")

println()
println("Running WITH ALLOCATION version...")
result_alloc = Max_Coverage_Kernel_WithAllocation_WithObjective(
    temp_burnmap_file, N_grounds, N_charging, n_drones,
    kernel, kernel_size_x, kernel_size_y, nothing
)
ground_alloc, charging_alloc, drone_allocations, obj_alloc, covered_risk_alloc, total_risk2, coverage_pct_alloc = result_alloc

println("  Objective: $(round(obj_alloc, digits=6))")
println("  Coverage: $(round(coverage_pct_alloc, digits=4))%")
println("  Ground: $ground_alloc")
println("  Charging: $charging_alloc")
println("  Drone allocations: $drone_allocations")

println()
println("="^70)
println("OBJECTIVE COMPARISON")
println("="^70)

obj_diff = abs(obj_alloc - obj_orig)
obj_diff_pct = (obj_diff / obj_orig) * 100

println("Original objective: $(round(obj_orig, digits=6))")
println("Allocation objective: $(round(obj_alloc, digits=6))")
println("Absolute difference: $(round(obj_diff, digits=6))")
println("Relative difference: $(round(obj_diff_pct, digits=4))%")

if obj_diff < 0.0001
    println()
    println("✓ OBJECTIVES ARE ESSENTIALLY IDENTICAL (difference < 0.01%)")
    println("  This confirms both methods find optimal solutions with the same objective value")
    println("  The different locations are just different choices from equivalent optima")
elseif obj_diff < 0.01
    println()
    println("≈ OBJECTIVES ARE VERY SIMILAR (difference < 1%)")
    println("  Small differences may be due to numerical precision or solver tolerances")
else
    println()
    if obj_alloc > obj_orig
        println("⚠ Allocation version has HIGHER objective")
        println("  This suggests the allocation formulation may find better solutions")
    else
        println("⚠ Allocation version has LOWER objective")
        println("  This suggests the original formulation may find better solutions")
    end
end

println()
println("Coverage Comparison:")
println("  Original: $(round(coverage_pct_orig, digits=4))%")
println("  Allocation: $(round(coverage_pct_alloc, digits=4))%")
println("  Difference: $(round(abs(coverage_pct_alloc - coverage_pct_orig), digits=4))%")

println()
println("Solution Analysis:")
println("  Ground sensors match: $(ground_orig == ground_alloc)")
println("  Charging stations match: $(charging_orig == charging_alloc)")

if ground_orig != ground_alloc || charging_orig != charging_alloc
    println("  Different locations but same objective suggests:")
    println("    - Multiple optimal solutions exist (common in uniform grids)")
    println("    - Both formulations find optimal solutions")
    println("    - Solver chooses different optima based on formulation structure")
end

println()
println("Drone Allocation:")
println("  Allocations: $drone_allocations")
println("  All equal to 1: $(all(a == 1 for a in drone_allocations))")
if all(a == 1 for a in drone_allocations)
    println("  ✓ As expected: 1 drone per station when n_drones = n_charging")
end

rm(temp_burnmap_file, force=true)

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
