# Objective function comparison test for drone allocation optimization
# Run this from the julia directory: julia test_objective_comparison.jl

using Dates
using Random
using Statistics

println("="^70)
println("OBJECTIVE FUNCTION COMPARISON TEST")
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
# Modified functions to return objective values
# ============================================================================

function Max_Coverage_Kernel_WithObjective(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y, mask_file)
    time_start = time_ns() / 1e9 
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    if T != 1 
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1/10) * sum(static_map[t,i,j] for t in 1:10)
        end
        static_map = avg_risk
    else
        static_map = static_map[1,:,:]
    end

    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end

    I = [(x, y) for x in 1:N for y in 1:M]
    I_prime = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    I_second = I_prime
    I_common = intersect(I_prime, I_second)
    I_ground_only = setdiff(I_prime, I_common)

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    xg = @variable(model, [i in I_prime], Bin)
    xc = @variable(model, [i in I_second], Bin)
    nc = @variable(model, [i in I_second], Int)
    theta = @variable(model, [i in I])

    @objective(model, Max, sum(static_map[point...] * theta[point] for point in I))

    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1)
    @constraint(model, sum(xg) == N_grounds)
    @constraint(model, sum(xc) == N_charging)   
    @constraint(model, sum(nc) == n_drones)
    @constraint(model, [i in I_second], nc[i]<=n_drones*xc[i])

    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i])
    
    @constraint(model, [(i_point,j_point) in I; (i_point,j_point) in I_prime], 
        theta[(i_point,j_point)] <= sum(
            get(kernel, (-dx,-dy), 0.0) * xc[(i_point+dx,j_point+dy)]
            for dx in max(-i_point+1,-kernel_size_x):min(N-i_point,kernel_size_x)
            for dy in max(-j_point+1,-kernel_size_y):min(M-j_point,kernel_size_y)
            if (i_point+dx,j_point+dy) in I_second && haskey(kernel, (-dx,-dy))
        ) + xg[(i_point,j_point)]
    )
    @constraint(model, [(i_point,j_point) in I; (i_point,j_point) ∉ I_prime], 
        theta[(i_point,j_point)] <= sum(
            get(kernel, (-dx,-dy), 0.0) * xc[(i_point+dx,j_point+dy)]
            for dx in max(-i_point+1,-kernel_size_x):min(N-i_point,kernel_size_x)
            for dy in max(-j_point+1,-kernel_size_y):min(M-j_point,kernel_size_y)
            if (i_point+dx,j_point+dy) in I_second && haskey(kernel, (-dx,-dy))
        )
    )

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(xg[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(xc[i]) > 0.5]
    
    obj_value = objective_value(model)
    
    # Calculate coverage breakdown
    total_risk = sum(static_map)
    covered_risk = sum(static_map[point...] * value(theta[point]) for point in I)
    coverage_percentage = (covered_risk / total_risk) * 100
    
    return selected_x_indices, selected_y_indices, obj_value, covered_risk, total_risk, coverage_percentage
end

function Max_Coverage_Kernel_WithAllocation_WithObjective(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y, mask_file)
    time_start = time_ns() / 1e9 
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    if T != 1 
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1/10) * sum(static_map[t,i,j] for t in 1:10)
        end
        static_map = avg_risk
    else
        static_map = static_map[1,:,:]
    end

    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end

    I = [(x, y) for x in 1:N for y in 1:M]
    I_prime = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    I_second = I_prime
    I_common = intersect(I_prime, I_second)
    I_ground_only = setdiff(I_prime, I_common)

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    xg = @variable(model, [i in I_prime], Bin)
    xc = @variable(model, [i in I_second], Bin)
    nc = @variable(model, [i in I_second], Int)
    theta = @variable(model, [i in I])

    @objective(model, Max, sum(static_map[point...] * theta[point] for point in I))

    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1)
    @constraint(model, sum(xg) == N_grounds)
    @constraint(model, sum(xc) == N_charging)   
    @constraint(model, sum(nc) == n_drones)
    @constraint(model, [i in I_second], nc[i] <= n_drones * xc[i])
    @constraint(model, [i in I_second], nc[i] >= 0)

    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i])
    
    @constraint(model, [(i_point,j_point) in I; (i_point,j_point) in I_prime], 
        theta[(i_point,j_point)] <= sum(
            get(kernel, (-dx,-dy), 0.0) * nc[(i_point+dx,j_point+dy)]
            for dx in max(-i_point+1,-kernel_size_x):min(N-i_point,kernel_size_x)
            for dy in max(-j_point+1,-kernel_size_y):min(M-j_point,kernel_size_y)
            if (i_point+dx,j_point+dy) in I_second && haskey(kernel, (-dx,-dy))
        ) + xg[(i_point,j_point)]
    )
    @constraint(model, [(i_point,j_point) in I; (i_point,j_point) ∉ I_prime], 
        theta[(i_point,j_point)] <= sum(
            get(kernel, (-dx,-dy), 0.0) * nc[(i_point+dx,j_point+dy)]
            for dx in max(-i_point+1,-kernel_size_x):min(N-i_point,kernel_size_x)
            for dy in max(-j_point+1,-kernel_size_y):min(M-j_point,kernel_size_y)
            if (i_point+dx,j_point+dy) in I_second && haskey(kernel, (-dx,-dy))
        )
    )

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(xg[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(xc[i]) > 0.5]
    drone_allocations = [Int(round(value(nc[i]))) for i in I_second if value(xc[i]) > 0.5]
    
    obj_value = objective_value(model)
    
    # Calculate coverage breakdown
    total_risk = sum(static_map)
    covered_risk = sum(static_map[point...] * value(theta[point]) for point in I)
    coverage_percentage = (covered_risk / total_risk) * 100
    
    return selected_x_indices, selected_y_indices, drone_allocations, obj_value, covered_risk, total_risk, coverage_percentage
end

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
# Test Function
# ============================================================================

function run_objective_test(test_num::Int, N::Int, M::Int, N_grounds::Int, N_charging::Int, n_drones::Int, max_battery::Int)
    println("="^70)
    println("TEST $test_num: Grid $(N)x$(M), $(N_grounds) ground, $(N_charging) charging, $(n_drones) drones")
    println("="^70)
    
    burnmap = create_synthetic_burnmap(N, M, 1)
    temp_burnmap_file = "temp_obj_burnmap_$(test_num).npy"
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
    
    println("  Objective value: $(round(obj_orig, digits=4))")
    println("  Covered risk: $(round(covered_risk_orig, digits=4)) / $(round(total_risk, digits=4))")
    println("  Coverage: $(round(coverage_pct_orig, digits=2))%")
    
    println()
    println("Running WITH ALLOCATION version...")
    result_alloc = Max_Coverage_Kernel_WithAllocation_WithObjective(
        temp_burnmap_file, N_grounds, N_charging, n_drones,
        kernel, kernel_size_x, kernel_size_y, nothing
    )
    ground_alloc, charging_alloc, drone_allocations, obj_alloc, covered_risk_alloc, total_risk2, coverage_pct_alloc = result_alloc
    
    println("  Objective value: $(round(obj_alloc, digits=4))")
    println("  Covered risk: $(round(covered_risk_alloc, digits=4)) / $(round(total_risk2, digits=4))")
    println("  Coverage: $(round(coverage_pct_alloc, digits=2))%")
    println("  Drone allocations: $drone_allocations")
    
    # Verify total risk matches
    if abs(total_risk - total_risk2) > 0.001
        println("  ⚠ WARNING: Total risk mismatch!")
    end
    
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
    println("  Difference: $(round(coverage_pct_alloc - coverage_pct_orig, digits=2))%")
    
    # Sanity checks
    println()
    println("Sanity Checks:")
    
    # Check if coverage is reasonable (should be between 0 and 100%)
    if coverage_pct_orig < 0 || coverage_pct_orig > 100
        println("  ⚠ Original coverage percentage out of bounds: $(coverage_pct_orig)%")
    else
        println("  ✓ Original coverage is in valid range")
    end
    
    if coverage_pct_alloc < 0 || coverage_pct_alloc > 100
        println("  ⚠ Allocation coverage percentage out of bounds: $(coverage_pct_alloc)%")
    else
        println("  ✓ Allocation coverage is in valid range")
    end
    
    # Check if covered risk <= total risk
    if covered_risk_orig > total_risk * 1.01  # Allow small floating point error
        println("  ⚠ Original covered risk exceeds total risk!")
    else
        println("  ✓ Original covered risk is valid")
    end
    
    if covered_risk_alloc > total_risk * 1.01
        println("  ⚠ Allocation covered risk exceeds total risk!")
    else
        println("  ✓ Allocation covered risk is valid")
    end
    
    # Check if objective matches covered risk (should be equal)
    if abs(obj_orig - covered_risk_orig) > 0.01
        println("  ⚠ Original objective doesn't match covered risk!")
        println("    Objective: $(obj_orig), Covered: $(covered_risk_orig)")
    else
        println("  ✓ Original objective matches covered risk")
    end
    
    if abs(obj_alloc - covered_risk_alloc) > 0.01
        println("  ⚠ Allocation objective doesn't match covered risk!")
        println("    Objective: $(obj_alloc), Covered: $(covered_risk_alloc)")
    else
        println("  ✓ Allocation objective matches covered risk")
    end
    
    # Check if coverage makes sense given constraints
    max_possible_coverage = min(1.0, (N_grounds + N_charging * n_drones) / (N * M))  # Rough estimate
    if coverage_pct_orig > max_possible_coverage * 100 * 2  # Allow some margin
        println("  ⚠ Original coverage seems unrealistically high")
    end
    
    if coverage_pct_alloc > max_possible_coverage * 100 * 2
        println("  ⚠ Allocation coverage seems unrealistically high")
    end
    
    rm(temp_burnmap_file, force=true)
    
    println("="^70)
    println()
    
    return (obj_orig, obj_alloc, coverage_pct_orig, coverage_pct_alloc, obj_diff_pct)
end

# ============================================================================
# Run Tests
# ============================================================================

println("Running objective comparison tests...")
println()

test_configs = [
    (20, 20, 2, 3, 10, 5),
    (25, 25, 3, 4, 12, 6),
    (30, 30, 4, 5, 15, 7),
]

all_results = []

for (idx, (N, M, N_grounds, N_charging, n_drones, max_battery)) in enumerate(test_configs)
    result = run_objective_test(idx, N, M, N_grounds, N_charging, n_drones, max_battery)
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

println("Average Objective Values:")
println("  Original: $(round(avg_obj_orig, digits=4))")
println("  Allocation: $(round(avg_obj_alloc, digits=4))")
println("  Difference: $(round(avg_obj_alloc - avg_obj_orig, digits=4)) ($(round((avg_obj_alloc/avg_obj_orig - 1) * 100, digits=2))%)")
println()

println("Average Coverage:")
println("  Original: $(round(avg_coverage_orig, digits=2))%")
println("  Allocation: $(round(avg_coverage_alloc, digits=2))%")
println("  Difference: $(round(avg_coverage_alloc - avg_coverage_orig, digits=2))%")
println()

if avg_obj_alloc > avg_obj_orig
    println("✓ Allocation version achieves HIGHER objective on average")
elseif avg_obj_alloc < avg_obj_orig
    println("⚠ Allocation version achieves LOWER objective on average")
else
    println("≈ Objectives are similar on average")
end

println()
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^70)
