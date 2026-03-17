# println("installing packages")
# import Pkg
# Pkg.add("IJulia")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Distances")
# Pkg.add("MAT")
# Pkg.add("Plots")
# Pkg.add("FFMPEG")
# Pkg.add("JuMP")
# Pkg.add("Gurobi")
# Pkg.add("Clustering")
# Pkg.add("NPZ")
# Pkg.add("NearestNeighbors")
# Pkg.add("Statistics")
# Pkg.add("AxisArrays")
# Pkg.add("Cairo")

using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ, Statistics

include("helper_functions.jl")

function load_parameters(risk_pertime_file)
    risk_pertime, _ = load_burn_map(risk_pertime_file)
    println("risk_pertime dimensions: ", size(risk_pertime), ndims(risk_pertime))
    T, N, _ = size(risk_pertime)
    M = N
    I = [(x, y) for x in 1:N for y in 1:M]
    if I_prime === nothing
        I_prime = I
    end

    if I_second === nothing
        I_second = I
    end

    return (risk_pertime=risk_pertime, T=T, N=N, M=M, I=I, I_prime=I_prime, I_second=I_second)
end

function NEW_SENSOR_STRATEGY(risk_pertime_file, N_grounds, N_charging)

    time_start = time_ns() / 1e9 

    # Load burn map and extract dimensions
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    println("risk_pertime_file=", risk_pertime_file)
    println("T=", T)
    println("N=", N)
    println("M=", M)

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]

    # Precompute average wildfire risk for each cell to avoid recalculating it multiple times
    avg_risk = zeros(N, M)
    for i in 1:N, j in 1:M
        avg_risk[i,j] = (1/T) * sum(risk_pertime[t,i,j] for t in 1:T)
    end

    # prefilter: keep only cells with risk > 90% of other cells
    #first_quartile_risk = quantile(vec(avg_risk), 0.0)
    # I_prime = [(i, j) for i in 1:N, j in 1:M if avg_risk[i,j] > 0.0] # >first_quartile_risk
    
    I_prime = [(i, j) for i in 1:N, j in 1:M if avg_risk[i,j] > 0.0] # Feasible grid points for ground stations
    I_second = I_prime #Feasible grid points for charging stations

    # print how many cells are discarded
    # println("Number of cells discarded: ", length(I) - length(I_prime))

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    
    # Variables 
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables

    # Objective - use precomputed average risk
    @objective(model, Max, 
        sum(avg_risk[i...] * x[i] for i in I_prime) + 
        sum(avg_risk[i...] * y[i] for i in I_second))

    # Constraints
    @constraint(model, [i in I_prime], x[i] + y[i] <= 1) # Can't place both devices at the same location
    @constraint(model, sum(x) <= N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(y) <= N_charging) # Capacity constraint on the charging stations

    close_pairs = [(i, j) for i in I_second for j in I_second if i != j && maximum(abs.(i .- j)) <= 10]  # Precompute valid (i, j) pairs where L∞ distance ≤ 10 
    @constraint(model, [(i, j) in close_pairs], y[i] + y[j] <= 1) # Spatial exclusion constraint between two charging stations
    cs_pairs = [(i, j) for (i, j) in close_pairs if j in I_prime]  
    @constraint(model, [(i,j) in cs_pairs], y[i] + x[j] <= 1) # Spatial exclusion constraint between a ground sensor and a charging station

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    #Extract selected sensor and charging station placements
    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) > 0.5]

    println("selected_x_indices=", selected_x_indices)
    println("selected_y_indices=", selected_y_indices)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices
end



function Max_Coverage_Kernel(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y, mask_file) # next variant: we add how many drones are in the area

    # kernel is a map (dx,dy) -> value that gives you the coverage if you are dx,dy away from the charging station, |dx| <= kernel_size_x, |dy| <= kernel_size_y

    time_start = time_ns() / 1e9 

    # Load burn map and extract dimensions
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("static_map_file=", static_map_file)
    if T != 1 
        println("static_map_file must be a single time step")
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
    static_map .*= mask

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]
    
    I_prime = [(i[1], i[2]) for i in findall(mask .> 0.0)] # Feasible grid points for ground stations
    println("I_prime number of points: ", length(I_prime))
    I_second = I_prime #Feasible grid points for charging stations
    I_common = intersect(I_prime, I_second)
    I_charging_only = setdiff(I_second, I_common)
    I_ground_only = setdiff(I_prime, I_common)



    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    
    # Variables 
    xg = @variable(model, [i in I_prime], Bin) # ground sensor variables
    xc = @variable(model, [i in I_second], Bin) # charging station variables
    nc = @variable(model, [i in I_second], Int) # number of drones from charging station i
    theta = @variable(model, [i in I]) # coverage variables

    # Objective - maximize coverage
    @objective(model, Max, 
        sum(static_map[point...] * theta[point] for point in I))

    # Placement constraints
    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1) # exclusion constraint on both ground sensors and charging stations
    @constraint(model, sum(xg) == N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(xc) == N_charging) # Capacity constraint on the charging stations   
    @constraint(model, sum(nc) == n_drones) # we use all the drones

    # linking constraint
    @constraint(model, [i in I_second], nc[i]<=n_drones*xc[i]) # number of drones in the area of the ground sensor is the sum of the charging stations in the area

    # Coverage constraints
    @constraint(model, [i in I], 0 <= theta[i] <= 1) # coverage variables are between 0 and 1
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i]) # coverage constraint on ground sensors
    
    # HERE WE ASSUME I = I_prime for efficiency, just change how the sum is indexed on depending on what is the most efficient in your case.
    # coverage = zeros(N,M)
    # for (i_point,j_point) in I
    #     for dx in max(-i_point+1,-kernel_size_x):min(N-i_point+1,kernel_size_x), dy in max(-j_point+1,-kernel_size_y):min(M-j_point+1,kernel_size_y)
    #         coverage_percentage = kernel[(-dx,-dy)] # - because here we compte the delta from point to the charging station and kernel is from charging station to point
    #         coverage[i_point, j_point] += coverage_percentage * xc[i_point]
    #     end
    # end
    # @constraint(model, [i in I], theta[i] >= coverage[i]) # coverage constraint on charging stations
    # Single constraint for charging station coverage
    # Split into two cases: points in I_prime and points not in I_prime
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

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    #Extract selected sensor and charging station placements
    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(xg[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(xc[i]) > 0.5]

    # println("selected_x_indices=", selected_x_indices)
    # println("selected_y_indices=", selected_y_indices)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices
end


"""
Compute coverage kernel from a single starting point using iterative DP that respects the mask.
Returns a dictionary mapping (target_x, target_y) -> coverage_probability
"""
function compute_masked_kernel_from_point(start_x, start_y, n_steps, mask, N, M)
    # Uniform coverage kernel for coordinated drone patrol, mask-aware.
    #
    # Weight = 1/B for every reachable cell within L∞ distance n_steps
    # that is not blocked by the mask.  B = 2*n_steps + 1 (full battery).
    # With B coordinated drones, all reachable cells are fully saturated.

    R = n_steps               # patrol range = max_battery // 2
    B = 2 * R + 1             # full battery = zone side length
    w = 1.0 / B

    result = Dict{Tuple{Int,Int}, Float64}()
    for di in -R:R, dj in -R:R
        ni, nj = start_x + di, start_y + dj
        if 1 <= ni <= N && 1 <= nj <= M && mask[ni, nj] > 0
            result[(ni, nj)] = w
        end
    end

    return result
end


function shortest_masked_path_within_reach(start::Tuple{Int,Int}, goal::Tuple{Int,Int}, mask, N::Int, M::Int, reach::Int)
    if start == goal
        return [start]
    end

    min_x = max(1, start[1] - reach)
    max_x = min(N, start[1] + reach)
    min_y = max(1, start[2] - reach)
    max_y = min(M, start[2] + reach)

    moves = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    queue = [start]
    head = 1
    parent = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    dist = Dict{Tuple{Int,Int}, Int}(start => 0)

    while head <= length(queue)
        current = queue[head]
        head += 1
        current_dist = dist[current]
        if current_dist >= reach
            continue
        end

        for (dx, dy) in moves
            nxt = (current[1] + dx, current[2] + dy)
            if !(min_x <= nxt[1] <= max_x && min_y <= nxt[2] <= max_y)
                continue
            end
            if mask[nxt...] <= 0 || haskey(dist, nxt)
                continue
            end
            parent[nxt] = current
            dist[nxt] = current_dist + 1
            if nxt == goal
                path = [goal]
                node = goal
                while node != start
                    node = parent[node]
                    push!(path, node)
                end
                reverse!(path)
                return path
            end
            push!(queue, nxt)
        end
    end

    return Tuple{Int,Int}[]
end


function build_station_greedy_kernels(station::Tuple{Int,Int}, mask, N::Int, M::Int, reach::Int, kmax::Int)
    min_x = max(1, station[1] - reach)
    max_x = min(N, station[1] + reach)
    min_y = max(1, station[2] - reach)
    max_y = min(M, station[2] + reach)

    templates = Vector{Vector{Tuple{Int,Int}}}()
    template_keys = Set{String}()

    for i_point in min_x:max_x
        for j_point in min_y:max_y
            if mask[i_point, j_point] <= 0
                continue
            end
            goal = (i_point, j_point)
            path = shortest_masked_path_within_reach(station, goal, mask, N, M, reach)
            if isempty(path)
                continue
            end
            key = join(["$(p[1])_$(p[2])" for p in path], ";")
            if !(key in template_keys)
                push!(templates, path)
                push!(template_keys, key)
            end
        end
    end

    if isempty(templates)
        return Dict{Tuple{Int,Int}, Int}(), Dict{Tuple{Int,Int}, Vector{Int}}()
    end

    first_cover_level = Dict{Tuple{Int,Int}, Int}()
    delta_levels = Dict{Tuple{Int,Int}, Vector{Int}}()
    covered = Set{Tuple{Int,Int}}()
    used = falses(length(templates))

    for level in 1:kmax
        best_idx = 0
        best_gain = -1
        best_length = -1

        for idx in eachindex(templates)
            if used[idx]
                continue
            end
            gain = 0
            for cell in templates[idx]
                if !(cell in covered)
                    gain += 1
                end
            end
            tpl_len = length(templates[idx])
            if gain > best_gain || (gain == best_gain && tpl_len > best_length)
                best_gain = gain
                best_length = tpl_len
                best_idx = idx
            end
        end

        if best_idx == 0 || best_gain <= 0
            break
        end

        used[best_idx] = true
        for cell in templates[best_idx]
            if !(cell in covered)
                push!(covered, cell)
                first_cover_level[cell] = level
            end
        end
    end

    for (cell, first_level) in first_cover_level
        levels = zeros(Int, kmax)
        levels[first_level] = 1
        delta_levels[cell] = levels
    end

    return first_cover_level, delta_levels
end


function reachable_masked_cells_within_reach(station::Tuple{Int,Int}, mask, N::Int, M::Int, reach::Int)
    if mask[station...] <= 0
        return Tuple{Int,Int}[]
    end

    min_x = max(1, station[1] - reach)
    max_x = min(N, station[1] + reach)
    min_y = max(1, station[2] - reach)
    max_y = min(M, station[2] + reach)

    moves = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    queue = [station]
    head = 1
    dist = Dict{Tuple{Int,Int}, Int}(station => 0)

    while head <= length(queue)
        current = queue[head]
        head += 1
        current_dist = dist[current]
        if current_dist >= reach
            continue
        end

        for (dx, dy) in moves
            nxt = (current[1] + dx, current[2] + dy)
            if !(min_x <= nxt[1] <= max_x && min_y <= nxt[2] <= max_y)
                continue
            end
            if mask[nxt...] <= 0 || haskey(dist, nxt)
                continue
            end
            dist[nxt] = current_dist + 1
            push!(queue, nxt)
        end
    end

    return collect(keys(dist))
end


function build_station_uniform_kernels(station::Tuple{Int,Int}, mask, N::Int, M::Int, reach::Int, kmax::Int, full_battery::Int)
    reachable_cells = reachable_masked_cells_within_reach(station, mask, N, M, reach)
    n_reachable = length(reachable_cells)
    if n_reachable == 0
        return Dict{Tuple{Int,Int}, Int}(), Dict{Tuple{Int,Int}, Vector{Float64}}()
    end

    first_cover_level = Dict{Tuple{Int,Int}, Int}()
    delta_levels = Dict{Tuple{Int,Int}, Vector{Float64}}()

    increments = zeros(Float64, kmax)
    previous = 0.0
    for level in 1:kmax
        current = min(1.0, level * full_battery / n_reachable)
        increments[level] = current - previous
        previous = current
    end

    for cell in reachable_cells
        first_cover_level[cell] = 1
        delta_levels[cell] = copy(increments)
    end

    return first_cover_level, delta_levels
end


function build_station_greedy_uniform_kernels(station::Tuple{Int,Int}, mask, N::Int, M::Int, reach::Int, kmax::Int, static_map, full_battery::Int)
    reachable_cells = reachable_masked_cells_within_reach(station, mask, N, M, reach)
    n_reachable = length(reachable_cells)
    if n_reachable == 0
        return Dict{Tuple{Int,Int}, Int}(), Dict{Tuple{Int,Int}, Vector{Float64}}()
    end

    total_risk = sum(static_map[cell...] for cell in reachable_cells)
    if total_risk <= 0.0
        first_cover_level = Dict{Tuple{Int,Int}, Int}()
        delta_levels = Dict{Tuple{Int,Int}, Vector{Float64}}()
        for cell in reachable_cells
            first_cover_level[cell] = 1
            delta_levels[cell] = zeros(Float64, kmax)
        end
        return first_cover_level, delta_levels
    end

    min_x = max(1, station[1] - reach)
    max_x = min(N, station[1] + reach)
    min_y = max(1, station[2] - reach)
    max_y = min(M, station[2] + reach)

    templates = Vector{Vector{Tuple{Int,Int}}}()
    template_keys = Set{String}()

    for i_point in min_x:max_x
        for j_point in min_y:max_y
            if mask[i_point, j_point] <= 0
                continue
            end
            goal = (i_point, j_point)
            path = shortest_masked_path_within_reach(station, goal, mask, N, M, reach)
            if isempty(path)
                continue
            end
            key = join(["$(p[1])_$(p[2])" for p in path], ";")
            if !(key in template_keys)
                push!(templates, path)
                push!(template_keys, key)
            end
        end
    end

    if isempty(templates)
        first_cover_level = Dict{Tuple{Int,Int}, Int}()
        delta_levels = Dict{Tuple{Int,Int}, Vector{Float64}}()
        for cell in reachable_cells
            first_cover_level[cell] = 1
            delta_levels[cell] = zeros(Float64, kmax)
        end
        return first_cover_level, delta_levels
    end

    covered = Set{Tuple{Int,Int}}()
    used = falses(length(templates))
    risk_per_level = zeros(Float64, kmax)

    for level in 1:kmax
        best_idx = 0
        best_gain = -1.0
        best_length = -1

        for idx in eachindex(templates)
            if used[idx]
                continue
            end
            gain = 0.0
            for cell in templates[idx]
                if !(cell in covered)
                    gain += static_map[cell...]
                end
            end
            tpl_len = length(templates[idx])
            if gain > best_gain || (gain == best_gain && tpl_len > best_length)
                best_gain = gain
                best_length = tpl_len
                best_idx = idx
            end
        end

        if best_idx == 0 || best_gain <= 0.0
            break
        end

        used[best_idx] = true
        level_risk = 0.0
        for cell in templates[best_idx]
            if !(cell in covered)
                push!(covered, cell)
                level_risk += static_map[cell...]
            end
        end
        risk_per_level[level] = level_risk
    end

    increments = zeros(Float64, kmax)
    cum_risk = 0.0
    prev_frac = 0.0
    for level in 1:kmax
        cum_risk += risk_per_level[level]
        frac = min(1.0, cum_risk / total_risk)
        if level == full_battery
            frac = 1.0
        end
        increments[level] = frac - prev_frac
        prev_frac = frac
    end

    first_cover_level = Dict{Tuple{Int,Int}, Int}()
    delta_levels = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for cell in reachable_cells
        first_cover_level[cell] = 1
        delta_levels[cell] = copy(increments)
    end

    return first_cover_level, delta_levels
end


"""
Max_Coverage_Kernel_Masked: Sensor placement optimization with mask-aware coverage kernels.

Arguments:
- static_map_file: Path to the risk/static map file
- N_grounds: Number of ground sensors to place
- N_charging: Number of charging stations to place
- n_drones: Number of drones
- kernel: Pre-computed kernel (used if recompute_kernel=false)
- kernel_size_x, kernel_size_y: Kernel dimensions
- mask_file: Path to mask file (cells with mask=0 are blocked)
- recompute_kernel: If true, compute per-location kernels using masked DP
- n_steps: Number of DP steps for kernel computation (only used if recompute_kernel=true)
"""
function Max_Coverage_Kernel_Masked(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y, mask_file, recompute_kernel=false, n_steps=3, time_limit_seconds=600.0)

    time_start = time_ns() / 1e9 
    time_preprocessing_start = time_ns() / 1e9

    # Load burn map and extract dimensions
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("static_map_file=", static_map_file)
    println("recompute_kernel=", recompute_kernel)
    
    if T != 1 
        println("static_map_file must be a single time step, averaging first 10 steps")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1/min(10,T)) * sum(static_map[t,i,j] for t in 1:min(10,T))
        end
        static_map = avg_risk
    else
        static_map = static_map[1,:,:]
    end

    # Load mask
    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end
    static_map .*= mask

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]
    
    # All feasible grid points (where mask > 0)
    I_all_feasible = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    println("Total feasible points: ", length(I_all_feasible))
    
    # ========== PRE-FILTERING FOR EFFICIENCY ==========
    # Pre-filtering: keep top X% candidates to reduce problem size if needed
    candidate_percentile = 0.00  # Keep 100% of cells (no filtering)
    
    # --- Filter ground sensor candidates by immediate cell risk (top 20%) ---
    ground_risks = [(loc, static_map[loc...]) for loc in I_all_feasible]
    ground_risk_values = [r for (_, r) in ground_risks]
    ground_risk_threshold = length(ground_risk_values) > 0 ? quantile(ground_risk_values, candidate_percentile) : 0.0
    I_prime = [loc for (loc, risk) in ground_risks if risk >= ground_risk_threshold]
    println("Ground sensor candidates (top 20% by immediate risk): ", length(I_prime), " / ", length(I_all_feasible))
    
    # --- Filter charging station candidates by coverage potential (top 20%) ---
    # Coverage potential = sum of (kernel_weight × risk) for all reachable cells
    charging_potentials = Dict{Tuple{Int,Int}, Float64}()
    for (cx, cy) in I_all_feasible
        coverage_potential = 0.0
        # Sum kernel-weighted risk over all cells reachable from this location
        for dx in max(-cx + 1, -kernel_size_x):min(N - cx, kernel_size_x)
            for dy in max(-cy + 1, -kernel_size_y):min(M - cy, kernel_size_y)
                target_x, target_y = cx + dx, cy + dy
                if 1 <= target_x <= N && 1 <= target_y <= M
                    kernel_weight = get(kernel, (dx, dy), 0.0)
                    if kernel_weight > 0
                        coverage_potential += kernel_weight * static_map[target_x, target_y]
                    end
                end
            end
        end
        charging_potentials[(cx, cy)] = coverage_potential
    end
    
    charging_potential_values = collect(values(charging_potentials))
    charging_potential_threshold = length(charging_potential_values) > 0 ? quantile(charging_potential_values, candidate_percentile) : 0.0
    I_second = [loc for (loc, potential) in charging_potentials if potential >= charging_potential_threshold]
    println("Charging station candidates (top 20% by coverage potential): ", length(I_second), " / ", length(I_all_feasible))
    
    # Safety check: if fewer feasible cells than sensors+charging stations requested,
    # cap the counts (one placement per cell). Prioritise charging stations (drones need depots).
    n_feasible = length(I_all_feasible)
    if n_feasible < N_grounds + N_charging
        effective_N_charging = min(N_charging, n_feasible)
        effective_N_grounds  = min(N_grounds, max(0, n_feasible - effective_N_charging))
        println("WARNING: Only $n_feasible feasible cell(s) for $(N_grounds) sensors + $(N_charging) charging stations.")
        println("         Capping to $effective_N_grounds sensor(s) + $effective_N_charging charging station(s).")
        N_grounds  = effective_N_grounds
        N_charging = effective_N_charging
    end

    # Edge case: nothing to place → return empty results immediately
    if N_grounds == 0 && N_charging == 0
        println("WARNING: No feasible cells at all. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[]
    end

    # Safety check: ensure we have enough candidates in the filtered sets
    if length(I_prime) < N_grounds
        println("WARNING: Not enough ground sensor candidates ($(length(I_prime))) for N_grounds=$N_grounds. Using all feasible points.")
        I_prime = I_all_feasible
    end
    if length(I_second) < N_charging
        println("WARNING: Not enough charging station candidates ($(length(I_second))) for N_charging=$N_charging. Using all feasible points.")
        I_second = I_all_feasible
    end
    
    # Compute set intersections (use Sets for faster membership checking)
    I_prime_set = Set(I_prime)
    I_second_set = Set(I_second)
    I_common = [loc for loc in I_prime if loc in I_second_set]
    I_ground_only = [loc for loc in I_prime if !(loc in I_second_set)]
    
    println("Common candidates: ", length(I_common))
    println("Ground-only candidates: ", length(I_ground_only))
    flush(stdout)

    # If recompute_kernel is true, compute per-location kernels using masked DP
    # Now only for the filtered I_second candidates
    per_location_kernels = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, Float64}}()
    if recompute_kernel
        println("Computing per-location kernels using masked DP (n_steps=$n_steps)...")
        kernel_start_time = time_ns() / 1e9
        for (idx, (cx, cy)) in enumerate(I_second)
            per_location_kernels[(cx, cy)] = compute_masked_kernel_from_point(cx, cy, n_steps, mask, N, M)
            if idx % 100 == 0
                println("  Computed kernel for $idx / $(length(I_second)) locations")
            end
        end
        println("Kernel computation took ", (time_ns() / 1e9) - kernel_start_time, " seconds")
    end

    time_preprocessing_end = time_ns() / 1e9
    println("Preprocessing took ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    flush(stdout)

    time_model_creation_start = time_ns() / 1e9

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    if time_limit_seconds > 0
        set_time_limit_sec(model, time_limit_seconds)
        println("Gurobi time limit set to ", time_limit_seconds, " seconds")
        flush(stdout)
    end

    # Variables
    xg = @variable(model, [i in I_prime], Bin) # ground sensor variables
    xc = @variable(model, [i in I_second], Bin) # charging station variables
    nc = @variable(model, [i in I_second], Int) # number of drones from charging station i
    theta = @variable(model, [i in I]) # coverage variables

    # Objective - maximize coverage
    @objective(model, Max,
        sum(static_map[point...] * theta[point] for point in I))

    # Placement constraints
    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1) # exclusion constraint
    @constraint(model, sum(xg) == N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(xc) == N_charging) # Capacity constraint on the charging stations
    @constraint(model, sum(nc) == n_drones) # we use all the drones

    # linking constraint
    @constraint(model, [i in I_second], nc[i] <= n_drones * xc[i])

    # Coverage constraints
    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i])
    
    if recompute_kernel
        # Use per-location kernels computed with masked DP
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) in I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(per_location_kernels[(cx, cy)], (i_point, j_point), 0.0) * xc[(cx, cy)]
                for (cx, cy) in I_second
                if haskey(per_location_kernels[(cx, cy)], (i_point, j_point))
            ) + xg[(i_point, j_point)]
        )
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) ∉ I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(per_location_kernels[(cx, cy)], (i_point, j_point), 0.0) * xc[(cx, cy)]
                for (cx, cy) in I_second
                if haskey(per_location_kernels[(cx, cy)], (i_point, j_point))
            )
        )
    else
        # Use the provided fixed kernel (same as original Max_Coverage_Kernel)
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) in I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(kernel, (-dx, -dy), 0.0) * xc[(i_point + dx, j_point + dy)]
                for dx in max(-i_point + 1, -kernel_size_x):min(N - i_point, kernel_size_x)
                for dy in max(-j_point + 1, -kernel_size_y):min(M - j_point, kernel_size_y)
                if (i_point + dx, j_point + dy) in I_second_set && haskey(kernel, (-dx, -dy))
            ) + xg[(i_point, j_point)]
        )
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) ∉ I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(kernel, (-dx, -dy), 0.0) * xc[(i_point + dx, j_point + dy)]
                for dx in max(-i_point + 1, -kernel_size_x):min(N - i_point, kernel_size_x)
                for dy in max(-j_point + 1, -kernel_size_y):min(M - j_point, kernel_size_y)
                if (i_point + dx, j_point + dy) in I_second_set && haskey(kernel, (-dx, -dy))
            )
        )
    end

    time_model_creation_end = time_ns() / 1e9
    println("Model creation took ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    flush(stdout)

    time_solve_start = time_ns() / 1e9
    optimize!(model)
    time_solve_end = time_ns() / 1e9
    println("Solving took ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    status = termination_status(model)
    println("Termination status: ", status)
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        gap = abs(obj_bound - obj_val) / max(abs(obj_val), 1e-10) * 100.0
        println("Objective value:  ", round(obj_val, digits=4))
        println("Objective bound:  ", round(obj_bound, digits=4))
        println("MIP gap:          ", round(gap, digits=2), "%")
    else
        println("No integer-feasible solution found within time limit!")
    end
    flush(stdout)

    # Extract selected sensor and charging station placements
    selected_x_indices = [(i[1] - 1, i[2] - 1) for i in I_prime if value(xg[i]) > 0.5] 
    selected_y_indices = [(i[1] - 1, i[2] - 1) for i in I_second if value(xc[i]) > 0.5]

    println("\n=== TIMING SUMMARY ===")
    println("  Preprocessing:   ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    println("  Model creation:  ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    println("  Solving:         ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    println("  TOTAL:           ", round((time_ns() / 1e9) - time_start, digits=2), " seconds")
    println("======================\n")
    
    return selected_x_indices, selected_y_indices
end


# ============================================================================
# NEW FUNCTIONS WITH DRONE ALLOCATION
# ============================================================================

function Max_Coverage_Kernel_WithAllocation(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y, mask_file)
    """
    Same as Max_Coverage_Kernel but uses nc[i] (number of drones) in coverage constraints
    and returns drone allocations as a third value.
    """

    time_start = time_ns() / 1e9 

    # Load burn map and extract dimensions
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("static_map_file=", static_map_file)
    if T != 1 
        println("static_map_file must be a single time step")
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
    static_map .*= mask

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]
    
    I_prime = [(i[1], i[2]) for i in findall(mask .> 0.0)] # Feasible grid points for ground stations
    println("I_prime number of points: ", length(I_prime))
    I_second = I_prime #Feasible grid points for charging stations
    I_common = intersect(I_prime, I_second)
    I_charging_only = setdiff(I_second, I_common)
    I_ground_only = setdiff(I_prime, I_common)



    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    
    # Variables 
    xg = @variable(model, [i in I_prime], Bin) # ground sensor variables
    xc = @variable(model, [i in I_second], Bin) # charging station variables
    nc = @variable(model, [i in I_second], Int) # number of drones from charging station i
    theta = @variable(model, [i in I]) # coverage variables

    # Objective - maximize coverage
    @objective(model, Max, 
        sum(static_map[point...] * theta[point] for point in I))

    # Placement constraints
    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1) # exclusion constraint on both ground sensors and charging stations
    @constraint(model, sum(xg) == N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(xc) == N_charging) # Capacity constraint on the charging stations   
    @constraint(model, sum(nc) == n_drones) # we use all the drones

    # linking constraint
    @constraint(model, [i in I_second], nc[i] <= n_drones * xc[i]) # number of drones at charging station i
    @constraint(model, [i in I_second], nc[i] >= 0) # explicit lower bound on number of drones

    # Coverage constraints
    @constraint(model, [i in I], 0 <= theta[i] <= 1) # coverage variables are between 0 and 1
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i]) # coverage constraint on ground sensors
    
    # Use nc[i] (number of drones) instead of xc[i] (binary), multiply kernel weight by number of drones
    # Coverage is capped at 1.0 by the constraint theta[i] <= 1
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

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    #Extract selected sensor and charging station placements
    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(xg[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(xc[i]) > 0.5]
    
    # Extract drone allocations for each selected charging station (matching order of selected_y_indices)
    drone_allocations = [Int(round(value(nc[i]))) for i in I_second if value(xc[i]) > 0.5]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices, drone_allocations
end


function Max_Coverage_Kernel_Masked_WithAllocation(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y, mask_file, recompute_kernel=false, n_steps=3, time_limit_seconds=600.0)
    """
    Same as Max_Coverage_Kernel_Masked but uses nc[i] (number of drones) in coverage constraints
    and returns drone allocations as a third value.
    """

    time_start = time_ns() / 1e9 
    time_preprocessing_start = time_ns() / 1e9

    # Load burn map and extract dimensions
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("static_map_file=", static_map_file)
    println("recompute_kernel=", recompute_kernel)
    
    if T != 1 
        println("static_map_file must be a single time step, averaging first 10 steps")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1/min(10,T)) * sum(static_map[t,i,j] for t in 1:min(10,T))
        end
        static_map = avg_risk
    else
        static_map = static_map[1,:,:]
    end

    # Load mask
    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end
    static_map .*= mask

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]
    
    # All feasible grid points (where mask > 0)
    I_all_feasible = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    println("Total feasible points: ", length(I_all_feasible))
    
    # ========== PRE-FILTERING FOR EFFICIENCY ==========
    # Pre-filtering: keep top X% candidates to reduce problem size if needed
    candidate_percentile = 0.00  # Keep 100% of cells (no filtering)
    
    # --- Filter ground sensor candidates by immediate cell risk (top 20%) ---
    ground_risks = [(loc, static_map[loc...]) for loc in I_all_feasible]
    ground_risk_values = [r for (_, r) in ground_risks]
    ground_risk_threshold = length(ground_risk_values) > 0 ? quantile(ground_risk_values, candidate_percentile) : 0.0
    I_prime = [loc for (loc, risk) in ground_risks if risk >= ground_risk_threshold]
    println("Ground sensor candidates (top 20% by immediate risk): ", length(I_prime), " / ", length(I_all_feasible))
    
    # --- Filter charging station candidates by coverage potential (top 20%) ---
    # Coverage potential = sum of (kernel_weight × risk) for all reachable cells
    charging_potentials = Dict{Tuple{Int,Int}, Float64}()
    for (cx, cy) in I_all_feasible
        coverage_potential = 0.0
        # Sum kernel-weighted risk over all cells reachable from this location
        for dx in max(-cx + 1, -kernel_size_x):min(N - cx, kernel_size_x)
            for dy in max(-cy + 1, -kernel_size_y):min(M - cy, kernel_size_y)
                target_x, target_y = cx + dx, cy + dy
                if 1 <= target_x <= N && 1 <= target_y <= M
                    kernel_weight = get(kernel, (dx, dy), 0.0)
                    if kernel_weight > 0
                        coverage_potential += kernel_weight * static_map[target_x, target_y]
                    end
                end
            end
        end
        charging_potentials[(cx, cy)] = coverage_potential
    end
    
    charging_potential_values = collect(values(charging_potentials))
    charging_potential_threshold = length(charging_potential_values) > 0 ? quantile(charging_potential_values, candidate_percentile) : 0.0
    I_second = [loc for (loc, potential) in charging_potentials if potential >= charging_potential_threshold]
    println("Charging station candidates (top 20% by coverage potential): ", length(I_second), " / ", length(I_all_feasible))
    
    # Safety check: if fewer feasible cells than sensors+charging stations requested,
    # cap the counts (one placement per cell). Prioritise charging stations (drones need depots).
    n_feasible = length(I_all_feasible)
    if n_feasible < N_grounds + N_charging
        effective_N_charging = min(N_charging, n_feasible)
        effective_N_grounds  = min(N_grounds, max(0, n_feasible - effective_N_charging))
        println("WARNING: Only $n_feasible feasible cell(s) for $(N_grounds) sensors + $(N_charging) charging stations.")
        println("         Capping to $effective_N_grounds sensor(s) + $effective_N_charging charging station(s).")
        N_grounds  = effective_N_grounds
        N_charging = effective_N_charging
    end

    # Edge case: nothing to place → return empty results immediately
    if N_grounds == 0 && N_charging == 0
        println("WARNING: No feasible cells at all. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    # Safety check: ensure we have enough candidates in the filtered sets
    if length(I_prime) < N_grounds
        println("WARNING: Not enough ground sensor candidates ($(length(I_prime))) for N_grounds=$N_grounds. Using all feasible points.")
        I_prime = I_all_feasible
    end
    if length(I_second) < N_charging
        println("WARNING: Not enough charging station candidates ($(length(I_second))) for N_charging=$N_charging. Using all feasible points.")
        I_second = I_all_feasible
    end
    
    # Compute set intersections (use Sets for faster membership checking)
    I_prime_set = Set(I_prime)
    I_second_set = Set(I_second)
    I_common = [loc for loc in I_prime if loc in I_second_set]
    I_ground_only = [loc for loc in I_prime if !(loc in I_second_set)]
    
    println("Common candidates: ", length(I_common))
    println("Ground-only candidates: ", length(I_ground_only))
    flush(stdout)

    # If recompute_kernel is true, compute per-location kernels using masked DP
    # Now only for the filtered I_second candidates
    per_location_kernels = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, Float64}}()
    if recompute_kernel
        println("Computing per-location kernels using masked DP (n_steps=$n_steps)...")
        kernel_start_time = time_ns() / 1e9
        for (idx, (cx, cy)) in enumerate(I_second)
            per_location_kernels[(cx, cy)] = compute_masked_kernel_from_point(cx, cy, n_steps, mask, N, M)
            if idx % 100 == 0
                println("  Computed kernel for $idx / $(length(I_second)) locations")
            end
        end
        println("Kernel computation took ", (time_ns() / 1e9) - kernel_start_time, " seconds")
    end

    time_preprocessing_end = time_ns() / 1e9
    println("Preprocessing took ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    flush(stdout)

    time_model_creation_start = time_ns() / 1e9

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    if time_limit_seconds > 0
        set_time_limit_sec(model, time_limit_seconds)
        println("Gurobi time limit set to ", time_limit_seconds, " seconds")
        flush(stdout)
    end

    # Variables
    xg = @variable(model, [i in I_prime], Bin) # ground sensor variables
    xc = @variable(model, [i in I_second], Bin) # charging station variables
    nc = @variable(model, [i in I_second], Int) # number of drones from charging station i
    theta = @variable(model, [i in I]) # coverage variables

    # Objective - maximize coverage
    @objective(model, Max,
        sum(static_map[point...] * theta[point] for point in I))

    # Placement constraints
    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1) # exclusion constraint
    @constraint(model, sum(xg) == N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(xc) == N_charging) # Capacity constraint on the charging stations
    @constraint(model, sum(nc) == n_drones) # we use all the drones

    # linking constraint
    @constraint(model, [i in I_second], nc[i] <= n_drones * xc[i])
    @constraint(model, [i in I_second], nc[i] >= 0) # explicit lower bound on number of drones

    # Coverage constraints
    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i])
    
    # Use nc[i] (number of drones) instead of xc[i] (binary), multiply kernel weight by number of drones
    # Coverage is capped at 1.0 by the constraint theta[i] <= 1
    if recompute_kernel
        # Use per-location kernels computed with masked DP
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) in I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(per_location_kernels[(cx, cy)], (i_point, j_point), 0.0) * nc[(cx, cy)]
                for (cx, cy) in I_second
                if haskey(per_location_kernels[(cx, cy)], (i_point, j_point))
            ) + xg[(i_point, j_point)]
        )
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) ∉ I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(per_location_kernels[(cx, cy)], (i_point, j_point), 0.0) * nc[(cx, cy)]
                for (cx, cy) in I_second
                if haskey(per_location_kernels[(cx, cy)], (i_point, j_point))
            )
        )
    else
        # Use the provided fixed kernel (same as original Max_Coverage_Kernel)
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) in I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(kernel, (-dx, -dy), 0.0) * nc[(i_point + dx, j_point + dy)]
                for dx in max(-i_point + 1, -kernel_size_x):min(N - i_point, kernel_size_x)
                for dy in max(-j_point + 1, -kernel_size_y):min(M - j_point, kernel_size_y)
                if (i_point + dx, j_point + dy) in I_second_set && haskey(kernel, (-dx, -dy))
            ) + xg[(i_point, j_point)]
        )
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) ∉ I_prime_set], 
            theta[(i_point, j_point)] <= sum(
                get(kernel, (-dx, -dy), 0.0) * nc[(i_point + dx, j_point + dy)]
                for dx in max(-i_point + 1, -kernel_size_x):min(N - i_point, kernel_size_x)
                for dy in max(-j_point + 1, -kernel_size_y):min(M - j_point, kernel_size_y)
                if (i_point + dx, j_point + dy) in I_second_set && haskey(kernel, (-dx, -dy))
            )
        )
    end

    time_model_creation_end = time_ns() / 1e9
    println("Model creation took ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    flush(stdout)

    time_solve_start = time_ns() / 1e9
    optimize!(model)
    time_solve_end = time_ns() / 1e9
    println("Solving took ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    status = termination_status(model)
    println("Termination status: ", status)
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        gap = abs(obj_bound - obj_val) / max(abs(obj_val), 1e-10) * 100.0
        println("Objective value:  ", round(obj_val, digits=4))
        println("Objective bound:  ", round(obj_bound, digits=4))
        println("MIP gap:          ", round(gap, digits=2), "%")
    else
        println("No integer-feasible solution found within time limit!")
    end
    flush(stdout)

    # Extract selected sensor and charging station placements
    if !has_values(model)
        println("ERROR: no feasible solution — returning empty placements.")
        flush(stdout)
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    selected_x_indices = [(i[1] - 1, i[2] - 1) for i in I_prime if value(xg[i]) > 0.5]
    selected_y_indices = [(i[1] - 1, i[2] - 1) for i in I_second if value(xc[i]) > 0.5]

    # Extract drone allocations for each selected charging station (matching order of selected_y_indices)
    drone_allocations = [Int(round(value(nc[i]))) for i in I_second if value(xc[i]) > 0.5]

    println("\n=== TIMING SUMMARY ===")
    println("  Preprocessing:   ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    println("  Model creation:  ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    println("  Solving:         ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    println("  TOTAL:           ", round((time_ns() / 1e9) - time_start, digits=2), " seconds")
    println("======================\n")
    flush(stdout)

    return selected_x_indices, selected_y_indices, drone_allocations
end


function Max_Coverage_Kernel_Masked_Budget(static_map_file, budget_millions, cost_sensor_millions, cost_station_millions, cost_drone_millions, kernel, kernel_size_x, kernel_size_y, mask_file, recompute_kernel=false, n_steps=3, time_limit_seconds=600.0)
    """
    Budget-constrained sensor/drone placement.  Instead of fixing the number of
    ground sensors, charging stations and drones, the optimiser allocates them
    subject to:  cost_sensor·Ng + cost_station·Nc + cost_drone·Nd ≤ budget.
    All costs and budget are expressed in millions.

    Returns (ground_sensor_locations, charging_station_locations, drone_allocations)
    with 0-based coordinates for Python interop.
    """

    time_start = time_ns() / 1e9
    time_preprocessing_start = time_ns() / 1e9

    # ── Load burn map ─────────────────────────────────────────────────────────
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("=== Max_Coverage_Kernel_Masked_Budget ===")
    println("static_map_file=", static_map_file)
    println("budget=", budget_millions, "M")
    println("costs: sensor=", cost_sensor_millions, "M, station=", cost_station_millions, "M, drone=", cost_drone_millions, "M")
    println("recompute_kernel=", recompute_kernel)
    epsilon = budget_millions > 300.0 ? 0.1 : 0.0
    println("budget regularization epsilon=", epsilon)

    if T != 1
        println("Averaging first ", min(10, T), " time steps")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1/min(10,T)) * sum(static_map[t,i,j] for t in 1:min(10,T))
        end
        static_map = avg_risk
    else
        static_map = static_map[1,:,:]
    end

    # ── Load mask ─────────────────────────────────────────────────────────────
    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end
    static_map .*= mask

    # ── Grid points ───────────────────────────────────────────────────────────
    I = [(x, y) for x in 1:N for y in 1:M]
    I_all_feasible = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    println("Total feasible points: ", length(I_all_feasible))

    # ── Upper bounds on device counts (from budget) ───────────────────────────
    max_possible_grounds  = floor(Int, budget_millions / cost_sensor_millions)
    max_possible_charging = floor(Int, budget_millions / cost_station_millions)
    max_possible_drones   = floor(Int, budget_millions / cost_drone_millions)
    println("Max possible: sensors=", max_possible_grounds,
            ", stations=", max_possible_charging,
            ", drones=", max_possible_drones)

    if max_possible_grounds == 0 && max_possible_charging == 0
        println("WARNING: Budget too small for any device. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    # ── Pre-filtering: keep top X% candidates to reduce problem size if needed
    candidate_percentile = 0.00  # Keep 100% of cells (no filtering)

    ground_risks = [(loc, static_map[loc...]) for loc in I_all_feasible]
    ground_risk_values = [r for (_, r) in ground_risks]
    ground_risk_threshold = length(ground_risk_values) > 0 ? quantile(ground_risk_values, candidate_percentile) : 0.0
    I_prime = [loc for (loc, risk) in ground_risks if risk >= ground_risk_threshold]
    println("Ground sensor candidates (top 20% by risk): ", length(I_prime), " / ", length(I_all_feasible))

    charging_potentials = Dict{Tuple{Int,Int}, Float64}()
    for (cx, cy) in I_all_feasible
        coverage_potential = 0.0
        for dx in max(-cx + 1, -kernel_size_x):min(N - cx, kernel_size_x)
            for dy in max(-cy + 1, -kernel_size_y):min(M - cy, kernel_size_y)
                target_x, target_y = cx + dx, cy + dy
                if 1 <= target_x <= N && 1 <= target_y <= M
                    kernel_weight = get(kernel, (dx, dy), 0.0)
                    if kernel_weight > 0
                        coverage_potential += kernel_weight * static_map[target_x, target_y]
                    end
                end
            end
        end
        charging_potentials[(cx, cy)] = coverage_potential
    end

    charging_potential_values = collect(values(charging_potentials))
    charging_potential_threshold = length(charging_potential_values) > 0 ? quantile(charging_potential_values, candidate_percentile) : 0.0
    I_second = [loc for (loc, potential) in charging_potentials if potential >= charging_potential_threshold]
    println("Charging station candidates (top 20% by coverage potential): ", length(I_second), " / ", length(I_all_feasible))

    # Safety: ensure enough candidates for the budget-feasible device counts
    if length(I_prime) < max_possible_grounds
        println("WARNING: Not enough ground candidates (", length(I_prime), ") for max possible (", max_possible_grounds, "). Using all feasible.")
        I_prime = I_all_feasible
    end
    if length(I_second) < max_possible_charging
        println("WARNING: Not enough charging candidates (", length(I_second), ") for max possible (", max_possible_charging, "). Using all feasible.")
        I_second = I_all_feasible
    end

    n_feasible = length(I_all_feasible)
    if n_feasible == 0
        println("WARNING: No feasible cells. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    I_prime_set  = Set(I_prime)
    I_second_set = Set(I_second)
    I_common      = [loc for loc in I_prime  if loc in I_second_set]
    I_ground_only = [loc for loc in I_prime  if !(loc in I_second_set)]

    println("Common candidates: ", length(I_common))
    println("Ground-only candidates: ", length(I_ground_only))
    flush(stdout)

    # ── Per-location kernels (if requested) ───────────────────────────────────
    per_location_kernels = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, Float64}}()
    if recompute_kernel
        println("Computing per-location kernels using masked DP (n_steps=$n_steps)...")
        kernel_start_time = time_ns() / 1e9
        for (idx, (cx, cy)) in enumerate(I_second)
            per_location_kernels[(cx, cy)] = compute_masked_kernel_from_point(cx, cy, n_steps, mask, N, M)
            if idx % 100 == 0
                println("  Computed kernel for $idx / $(length(I_second)) locations")
            end
        end
        println("Kernel computation took ", (time_ns() / 1e9) - kernel_start_time, " seconds")
    end

    time_preprocessing_end = time_ns() / 1e9
    println("Preprocessing took ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    flush(stdout)

    # ── Build optimisation model ──────────────────────────────────────────────
    time_model_creation_start = time_ns() / 1e9

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    if time_limit_seconds > 0
        set_time_limit_sec(model, time_limit_seconds)
        println("Gurobi time limit set to ", time_limit_seconds, " seconds")
        flush(stdout)
    end

    # Variables
    xg    = @variable(model, [i in I_prime],  Bin)   # ground sensor at cell i
    xc    = @variable(model, [i in I_second], Bin)   # charging station at cell i
    nc    = @variable(model, [i in I_second], Int)   # drones at station i
    theta = @variable(model, [i in I])               # coverage at cell i

    budget_used_expr = @expression(model,
        cost_sensor_millions  * sum(xg) +
        cost_station_millions * sum(xc) +
        cost_drone_millions   * sum(nc)
    )

    # Objective: maximise risk-weighted coverage, with a small budget penalty
    # only for very large budgets to prefer cheaper solutions among coverage ties.
    @objective(model, Max,
        sum(static_map[point...] * theta[point] for point in I) - epsilon * budget_used_expr)

    # ── Budget constraint (replaces fixed device counts) ──────────────────────
    @constraint(model, budget_used_expr <= budget_millions)

    # Placement exclusion: a cell cannot host both a sensor and a station
    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1)

    # Linking: drones only at selected stations
    @constraint(model, [i in I_second], nc[i] <= max_possible_drones * xc[i])
    @constraint(model, [i in I_second], nc[i] >= 0)

    # Coverage bounds
    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i])

    # ── Coverage constraints ──────────────────────────────────────────────────
    if recompute_kernel
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) in I_prime_set],
            theta[(i_point, j_point)] <= sum(
                get(per_location_kernels[(cx, cy)], (i_point, j_point), 0.0) * nc[(cx, cy)]
                for (cx, cy) in I_second
                if haskey(per_location_kernels[(cx, cy)], (i_point, j_point))
            ) + xg[(i_point, j_point)]
        )
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) ∉ I_prime_set],
            theta[(i_point, j_point)] <= sum(
                get(per_location_kernels[(cx, cy)], (i_point, j_point), 0.0) * nc[(cx, cy)]
                for (cx, cy) in I_second
                if haskey(per_location_kernels[(cx, cy)], (i_point, j_point))
            )
        )
    else
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) in I_prime_set],
            theta[(i_point, j_point)] <= sum(
                get(kernel, (-dx, -dy), 0.0) * nc[(i_point + dx, j_point + dy)]
                for dx in max(-i_point + 1, -kernel_size_x):min(N - i_point, kernel_size_x)
                for dy in max(-j_point + 1, -kernel_size_y):min(M - j_point, kernel_size_y)
                if (i_point + dx, j_point + dy) in I_second_set && haskey(kernel, (-dx, -dy))
            ) + xg[(i_point, j_point)]
        )
        @constraint(model, [(i_point, j_point) in I; (i_point, j_point) ∉ I_prime_set],
            theta[(i_point, j_point)] <= sum(
                get(kernel, (-dx, -dy), 0.0) * nc[(i_point + dx, j_point + dy)]
                for dx in max(-i_point + 1, -kernel_size_x):min(N - i_point, kernel_size_x)
                for dy in max(-j_point + 1, -kernel_size_y):min(M - j_point, kernel_size_y)
                if (i_point + dx, j_point + dy) in I_second_set && haskey(kernel, (-dx, -dy))
            )
        )
    end

    time_model_creation_end = time_ns() / 1e9
    println("Model creation took ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    flush(stdout)

    # ── Solve ─────────────────────────────────────────────────────────────────
    time_solve_start = time_ns() / 1e9
    optimize!(model)
    time_solve_end = time_ns() / 1e9
    println("Solving took ", round(time_solve_end - time_solve_start, digits=2), " seconds")

    status = termination_status(model)
    println("Termination status: ", status)
    if has_values(model)
        obj_val  = objective_value(model)
        obj_bound = objective_bound(model)
        gap = abs(obj_bound - obj_val) / max(abs(obj_val), 1e-10) * 100.0
        println("Objective value:  ", round(obj_val, digits=4))
        println("Objective bound:  ", round(obj_bound, digits=4))
        println("MIP gap:          ", round(gap, digits=2), "%")
    else
        println("No integer-feasible solution found within time limit!")
    end
    flush(stdout)

    # ── Extract results ───────────────────────────────────────────────────────
    if !has_values(model)
        println("ERROR: no feasible solution — returning empty placements.")
        flush(stdout)
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    selected_x_indices = [(i[1] - 1, i[2] - 1) for i in I_prime  if value(xg[i]) > 0.5]
    selected_y_indices = [(i[1] - 1, i[2] - 1) for i in I_second if value(xc[i]) > 0.5]
    drone_allocations  = [Int(round(value(nc[i]))) for i in I_second if value(xc[i]) > 0.5]

    n_sensors  = length(selected_x_indices)
    n_stations = length(selected_y_indices)
    n_drones   = sum(drone_allocations; init=0)
    budget_used = n_sensors * cost_sensor_millions + n_stations * cost_station_millions + n_drones * cost_drone_millions

    println("\n=== BUDGET ALLOCATION ===")
    println("  Ground sensors:     ", n_sensors,  " (cost ", round(n_sensors * cost_sensor_millions, digits=2), "M)")
    println("  Charging stations:  ", n_stations, " (cost ", round(n_stations * cost_station_millions, digits=2), "M)")
    println("  Drones:             ", n_drones,   " (cost ", round(n_drones * cost_drone_millions, digits=2), "M)")
    println("  Budget used:        ", round(budget_used, digits=2), "M / ", budget_millions, "M")

    println("\n=== TIMING SUMMARY ===")
    println("  Preprocessing:   ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    println("  Model creation:  ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    println("  Solving:         ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    println("  TOTAL:           ", round((time_ns() / 1e9) - time_start, digits=2), " seconds")
    println("======================\n")
    flush(stdout)

    return selected_x_indices, selected_y_indices, drone_allocations
end


function Max_Coverage_Kernel_Masked_Budget_StationMax(static_map_file, budget_millions, cost_sensor_millions, cost_station_millions, cost_drone_millions, kernel, kernel_size_x, kernel_size_y, mask_file, recompute_kernel=false, n_steps=63, time_limit_seconds=600.0, max_drones_per_station=7, candidate_percentile=0.80)
    """
    Budget-constrained placement variant where coverage from different charging
    stations does not add. For each cell, only the best contributing station is
    counted; multi-drone coverage at the same station still grows with capped
    linear increments derived from the single-drone kernel.

    Only cell-station pairs with L∞ distance <= floor(kernel_size_x / 2) are
    kept, matching a one-way reach based on half the battery life.
    """

    time_start = time_ns() / 1e9
    time_preprocessing_start = time_ns() / 1e9

    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("=== Max_Coverage_Kernel_Masked_Budget_StationMax ===")
    println("static_map_file=", static_map_file)
    println("budget=", budget_millions, "M")
    println("costs: sensor=", cost_sensor_millions, "M, station=", cost_station_millions, "M, drone=", cost_drone_millions, "M")
    println("recompute_kernel=", recompute_kernel)
    println("max_drones_per_station=", max_drones_per_station)
    println("candidate_percentile=", candidate_percentile)
    epsilon = budget_millions > 300.0 ? 0.1 : 0.0
    println("budget regularization epsilon=", epsilon)

    if T != 1
        println("Averaging first ", min(10, T), " time steps")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1 / min(10, T)) * sum(static_map[t, i, j] for t in 1:min(10, T))
        end
        static_map = avg_risk
    else
        static_map = static_map[1, :, :]
    end

    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end

    I_all_feasible = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    I = I_all_feasible
    println("Total feasible points: ", length(I_all_feasible))

    max_possible_grounds  = floor(Int, budget_millions / cost_sensor_millions)
    max_possible_charging = floor(Int, budget_millions / cost_station_millions)
    max_possible_drones   = floor(Int, budget_millions / cost_drone_millions)
    println("Max possible: sensors=", max_possible_grounds,
            ", stations=", max_possible_charging,
            ", drones=", max_possible_drones)

    if max_possible_grounds == 0 && max_possible_charging == 0
        println("WARNING: Budget too small for any device. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    ground_risks = [(loc, static_map[loc...]) for loc in I_all_feasible]
    ground_risk_values = [r for (_, r) in ground_risks]
    ground_risk_threshold = length(ground_risk_values) > 0 ? quantile(ground_risk_values, candidate_percentile) : 0.0
    I_prime = [loc for (loc, risk) in ground_risks if risk >= ground_risk_threshold]
    println("Ground sensor candidates (top ", round((1 - candidate_percentile) * 100, digits=1), "% by risk): ", length(I_prime), " / ", length(I_all_feasible))

    charging_potentials = Dict{Tuple{Int,Int}, Float64}()
    for (cx, cy) in I_all_feasible
        coverage_potential = 0.0
        for dx in max(-cx + 1, -kernel_size_x):min(N - cx, kernel_size_x)
            for dy in max(-cy + 1, -kernel_size_y):min(M - cy, kernel_size_y)
                target_x, target_y = cx + dx, cy + dy
                if 1 <= target_x <= N && 1 <= target_y <= M
                    if mask[target_x, target_y] <= 0
                        continue
                    end
                    kernel_weight = get(kernel, (dx, dy), 0.0)
                    if kernel_weight > 0
                        coverage_potential += kernel_weight * static_map[target_x, target_y]
                    end
                end
            end
        end
        charging_potentials[(cx, cy)] = coverage_potential
    end

    charging_potential_values = collect(values(charging_potentials))
    charging_potential_threshold = length(charging_potential_values) > 0 ? quantile(charging_potential_values, candidate_percentile) : 0.0
    I_second = [loc for (loc, potential) in charging_potentials if potential >= charging_potential_threshold]
    println("Charging station candidates (top ", round((1 - candidate_percentile) * 100, digits=1), "% by coverage potential): ", length(I_second), " / ", length(I_all_feasible))

    if length(I_prime) < max_possible_grounds
        println("WARNING: Not enough ground candidates (", length(I_prime), ") for max possible (", max_possible_grounds, "). Using all feasible.")
        I_prime = I_all_feasible
    end
    if length(I_second) < max_possible_charging
        println("WARNING: Not enough charging candidates (", length(I_second), ") for max possible (", max_possible_charging, "). Using all feasible.")
        I_second = I_all_feasible
    end

    if isempty(I_all_feasible)
        println("WARNING: No feasible cells. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    I_prime_set = Set(I_prime)

    per_location_kernels = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, Float64}}()
    if recompute_kernel
        println("Computing per-location kernels using masked DP (n_steps=$n_steps)...")
        kernel_start_time = time_ns() / 1e9
        for (idx, (cx, cy)) in enumerate(I_second)
            per_location_kernels[(cx, cy)] = compute_masked_kernel_from_point(cx, cy, n_steps, mask, N, M)
            if idx % 100 == 0
                println("  Computed kernel for $idx / $(length(I_second)) locations")
            end
        end
        println("Kernel computation took ", (time_ns() / 1e9) - kernel_start_time, " seconds")
    end

    one_way_reach = floor(Int, kernel_size_x / 2)
    println("Pair pruning radius (L∞ one-way reach): ", one_way_reach)

    feasible_pairs = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
    cell_to_pairs = Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}}()
    station_to_pairs = Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}}()
    pair_first_cover_level = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, Int}()
    pair_delta_levels = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, Vector{Int}}()
    station_drone_caps = Dict{Tuple{Int,Int}, Int}()

    for station in I_second
        local_pairs = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
        first_cover_level, delta_levels = build_station_greedy_kernels(
            station, mask, N, M, one_way_reach, max_drones_per_station
        )

        for (cell, first_level) in first_cover_level
            pair = (cell, station)
            push!(feasible_pairs, pair)
            push!(local_pairs, pair)
            if !haskey(cell_to_pairs, cell)
                cell_to_pairs[cell] = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
            end
            push!(cell_to_pairs[cell], pair)
            pair_first_cover_level[pair] = first_level
            pair_delta_levels[pair] = delta_levels[cell]
        end

        if !isempty(local_pairs)
            station_to_pairs[station] = local_pairs
            station_drone_caps[station] = min(max_possible_drones, max(1, max_drones_per_station))
        end
    end

    I_second = [station for station in I_second if haskey(station_to_pairs, station)]
    I_second_set = Set(I_second)
    I_common = [loc for loc in I_prime if loc in I_second_set]

    println("Stations with at least one feasible pair: ", length(I_second))
    println("Feasible station-cell pairs kept: ", length(feasible_pairs))
    if !isempty(I_second)
        cap_values = [station_drone_caps[s] for s in I_second]
        println("Per-station drone cap: min=", minimum(cap_values), ", max=", maximum(cap_values), ", mean=", round(mean(cap_values), digits=2))
    end

    station_levels = Tuple{Tuple{Int,Int}, Int}[]
    for station in I_second
        for level in 1:station_drone_caps[station]
            push!(station_levels, (station, level))
        end
    end
    println("Station-drone binary levels: ", length(station_levels))

    time_preprocessing_end = time_ns() / 1e9
    println("Preprocessing took ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    flush(stdout)

    time_model_creation_start = time_ns() / 1e9

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    if time_limit_seconds > 0
        set_time_limit_sec(model, time_limit_seconds)
        println("Gurobi time limit set to ", time_limit_seconds, " seconds")
        flush(stdout)
    end

    xg    = @variable(model, [i in I_prime], Bin)
    z     = @variable(model, [sl in station_levels], Bin)
    u     = @variable(model, [pair in feasible_pairs], Bin)
    w     = @variable(model, [pair in feasible_pairs])
    theta = @variable(model, [i in I])

    budget_used_expr = @expression(model,
        cost_sensor_millions * sum(xg) +
        cost_station_millions * sum(z[(station, 1)] for station in I_second) +
        cost_drone_millions * sum(z[sl] for sl in station_levels)
    )

    @objective(model, Max,
        sum(static_map[point...] * theta[point] for point in I) - epsilon * budget_used_expr)

    @constraint(model, budget_used_expr <= budget_millions)

    @constraint(model, [i in I_common], xg[i] + z[(i, 1)] <= 1)
    @constraint(model, [station in I_second, level in 1:(station_drone_caps[station] - 1)],
        z[(station, level)] >= z[(station, level + 1)])

    @constraint(model, [pair in feasible_pairs], u[pair] <= z[(pair[2], pair_first_cover_level[pair])])
    @constraint(model, [pair in feasible_pairs], w[pair] >= 0)
    @constraint(model, [pair in feasible_pairs], w[pair] <= u[pair])
    @constraint(model, [pair in feasible_pairs],
        w[pair] <= sum(
            pair_delta_levels[pair][level] *
            z[(pair[2], level)]
            for level in 1:station_drone_caps[pair[2]]
        ))

    @constraint(model, [cell in keys(cell_to_pairs)],
        sum(u[pair] for pair in cell_to_pairs[cell]) <= 1)

    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [cell in I; cell in I_prime_set],
        theta[cell] <= xg[cell] + sum(w[pair] for pair in get(cell_to_pairs, cell, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[])))
    @constraint(model, [cell in I; !(cell in I_prime_set)],
        theta[cell] <= sum(w[pair] for pair in get(cell_to_pairs, cell, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[])))

    time_model_creation_end = time_ns() / 1e9
    println("Model creation took ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    flush(stdout)

    time_solve_start = time_ns() / 1e9
    optimize!(model)
    time_solve_end = time_ns() / 1e9
    println("Solving took ", round(time_solve_end - time_solve_start, digits=2), " seconds")

    status = termination_status(model)
    println("Termination status: ", status)
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        gap = abs(obj_bound - obj_val) / max(abs(obj_val), 1e-10) * 100.0
        println("Objective value:  ", round(obj_val, digits=4))
        println("Objective bound:  ", round(obj_bound, digits=4))
        println("MIP gap:          ", round(gap, digits=2), "%")
    else
        println("No integer-feasible solution found within time limit!")
    end
    flush(stdout)

    if !has_values(model)
        println("ERROR: no feasible solution — returning empty placements.")
        flush(stdout)
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    selected_x_indices = [(i[1] - 1, i[2] - 1) for i in I_prime if value(xg[i]) > 0.5]
    selected_y_indices = [(station[1] - 1, station[2] - 1) for station in I_second if value(z[(station, 1)]) > 0.5]
    drone_allocations = [sum(Int(round(value(z[(station, level)]))) for level in 1:station_drone_caps[station])
                         for station in I_second if value(z[(station, 1)]) > 0.5]

    n_sensors = length(selected_x_indices)
    n_stations = length(selected_y_indices)
    n_drones = sum(drone_allocations; init=0)
    budget_used = n_sensors * cost_sensor_millions + n_stations * cost_station_millions + n_drones * cost_drone_millions

    println("\n=== BUDGET ALLOCATION ===")
    println("  Ground sensors:     ", n_sensors, " (cost ", round(n_sensors * cost_sensor_millions, digits=2), "M)")
    println("  Charging stations:  ", n_stations, " (cost ", round(n_stations * cost_station_millions, digits=2), "M)")
    println("  Drones:             ", n_drones, " (cost ", round(n_drones * cost_drone_millions, digits=2), "M)")
    println("  Budget used:        ", round(budget_used, digits=2), "M / ", budget_millions, "M")

    println("\n=== TIMING SUMMARY ===")
    println("  Preprocessing:   ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    println("  Model creation:  ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    println("  Solving:         ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    println("  TOTAL:           ", round((time_ns() / 1e9) - time_start, digits=2), " seconds")
    println("======================\n")
    flush(stdout)

    return selected_x_indices, selected_y_indices, drone_allocations
end


function Max_Coverage_Kernel_Masked_Budget_StationMax_Uniform(static_map_file, budget_millions, cost_sensor_millions, cost_station_millions, cost_drone_millions, kernel, kernel_size_x, kernel_size_y, mask_file, recompute_kernel=false, n_steps=63, time_limit_seconds=600.0, max_drones_per_station=7, candidate_percentile=0.80)
    """
    Budget-constrained StationMax placement variant with a mask-aware uniform
    same-station coverage model. Coverage from different charging stations does
    not add; each cell can be assigned to at most one station. For a station
    with k drones and R reachable feasible cells in its masked one-way-reach
    zone, each reachable cell receives coverage min(1, k * B / R), where
    B = 2 * reach + 1 is the full battery in operational substeps.
    """

    time_start = time_ns() / 1e9
    time_preprocessing_start = time_ns() / 1e9

    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("=== Max_Coverage_Kernel_Masked_Budget_StationMax_Uniform ===")
    println("static_map_file=", static_map_file)
    println("budget=", budget_millions, "M")
    println("costs: sensor=", cost_sensor_millions, "M, station=", cost_station_millions, "M, drone=", cost_drone_millions, "M")
    println("recompute_kernel=", recompute_kernel)
    println("max_drones_per_station=", max_drones_per_station)
    println("candidate_percentile=", candidate_percentile)
    epsilon = budget_millions > 300.0 ? 0.1 : 0.0
    println("budget regularization epsilon=", epsilon)

    if T != 1
        println("Averaging first ", min(10, T), " time steps")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1 / min(10, T)) * sum(static_map[t, i, j] for t in 1:min(10, T))
        end
        static_map = avg_risk
    else
        static_map = static_map[1, :, :]
    end

    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end

    I_all_feasible = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    I = I_all_feasible
    println("Total feasible points: ", length(I_all_feasible))

    max_possible_grounds  = floor(Int, budget_millions / cost_sensor_millions)
    max_possible_charging = floor(Int, budget_millions / cost_station_millions)
    max_possible_drones   = floor(Int, budget_millions / cost_drone_millions)
    println("Max possible: sensors=", max_possible_grounds,
            ", stations=", max_possible_charging,
            ", drones=", max_possible_drones)

    if max_possible_grounds == 0 && max_possible_charging == 0
        println("WARNING: Budget too small for any device. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    ground_risks = [(loc, static_map[loc...]) for loc in I_all_feasible]
    ground_risk_values = [r for (_, r) in ground_risks]
    ground_risk_threshold = length(ground_risk_values) > 0 ? quantile(ground_risk_values, candidate_percentile) : 0.0
    I_prime = [loc for (loc, risk) in ground_risks if risk >= ground_risk_threshold]
    println("Ground sensor candidates (top ", round((1 - candidate_percentile) * 100, digits=1), "% by risk): ", length(I_prime), " / ", length(I_all_feasible))

    charging_potentials = Dict{Tuple{Int,Int}, Float64}()
    for (cx, cy) in I_all_feasible
        coverage_potential = 0.0
        for dx in max(-cx + 1, -kernel_size_x):min(N - cx, kernel_size_x)
            for dy in max(-cy + 1, -kernel_size_y):min(M - cy, kernel_size_y)
                target_x, target_y = cx + dx, cy + dy
                if 1 <= target_x <= N && 1 <= target_y <= M
                    if mask[target_x, target_y] <= 0
                        continue
                    end
                    kernel_weight = get(kernel, (dx, dy), 0.0)
                    if kernel_weight > 0
                        coverage_potential += kernel_weight * static_map[target_x, target_y]
                    end
                end
            end
        end
        charging_potentials[(cx, cy)] = coverage_potential
    end

    charging_potential_values = collect(values(charging_potentials))
    charging_potential_threshold = length(charging_potential_values) > 0 ? quantile(charging_potential_values, candidate_percentile) : 0.0
    I_second = [loc for (loc, potential) in charging_potentials if potential >= charging_potential_threshold]
    println("Charging station candidates (top ", round((1 - candidate_percentile) * 100, digits=1), "% by coverage potential): ", length(I_second), " / ", length(I_all_feasible))

    if length(I_prime) < max_possible_grounds
        println("WARNING: Not enough ground candidates (", length(I_prime), ") for max possible (", max_possible_grounds, "). Using all feasible.")
        I_prime = I_all_feasible
    end
    if length(I_second) < max_possible_charging
        println("WARNING: Not enough charging candidates (", length(I_second), ") for max possible (", max_possible_charging, "). Using all feasible.")
        I_second = I_all_feasible
    end

    if isempty(I_all_feasible)
        println("WARNING: No feasible cells. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    I_prime_set = Set(I_prime)

    one_way_reach = floor(Int, kernel_size_x / 2)
    full_battery = 2 * one_way_reach + 1
    println("Pair pruning radius (L∞ one-way reach): ", one_way_reach)
    println("Uniform station battery budget B=", full_battery)

    feasible_pairs = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
    cell_to_pairs = Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}}()
    station_to_pairs = Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}}()
    pair_first_cover_level = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, Int}()
    pair_delta_levels = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, Vector{Float64}}()
    station_drone_caps = Dict{Tuple{Int,Int}, Int}()

    for station in I_second
        local_pairs = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
        first_cover_level, delta_levels = build_station_uniform_kernels(
            station, mask, N, M, one_way_reach, max_drones_per_station, full_battery
        )

        for (cell, first_level) in first_cover_level
            pair = (cell, station)
            push!(feasible_pairs, pair)
            push!(local_pairs, pair)
            if !haskey(cell_to_pairs, cell)
                cell_to_pairs[cell] = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
            end
            push!(cell_to_pairs[cell], pair)
            pair_first_cover_level[pair] = first_level
            pair_delta_levels[pair] = delta_levels[cell]
        end

        if !isempty(local_pairs)
            station_to_pairs[station] = local_pairs
            station_drone_caps[station] = min(max_possible_drones, max(1, max_drones_per_station))
        end
    end

    I_second = [station for station in I_second if haskey(station_to_pairs, station)]
    I_second_set = Set(I_second)
    I_common = [loc for loc in I_prime if loc in I_second_set]

    println("Stations with at least one feasible pair: ", length(I_second))
    println("Feasible station-cell pairs kept: ", length(feasible_pairs))
    if !isempty(I_second)
        cap_values = [station_drone_caps[s] for s in I_second]
        println("Per-station drone cap: min=", minimum(cap_values), ", max=", maximum(cap_values), ", mean=", round(mean(cap_values), digits=2))
    end

    station_levels = Tuple{Tuple{Int,Int}, Int}[]
    for station in I_second
        for level in 1:station_drone_caps[station]
            push!(station_levels, (station, level))
        end
    end
    println("Station-drone binary levels: ", length(station_levels))

    time_preprocessing_end = time_ns() / 1e9
    println("Preprocessing took ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    flush(stdout)

    time_model_creation_start = time_ns() / 1e9

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    if time_limit_seconds > 0
        set_time_limit_sec(model, time_limit_seconds)
        println("Gurobi time limit set to ", time_limit_seconds, " seconds")
        flush(stdout)
    end

    xg    = @variable(model, [i in I_prime], Bin)
    z     = @variable(model, [sl in station_levels], Bin)
    u     = @variable(model, [pair in feasible_pairs], Bin)
    w     = @variable(model, [pair in feasible_pairs])
    theta = @variable(model, [i in I])

    budget_used_expr = @expression(model,
        cost_sensor_millions * sum(xg) +
        cost_station_millions * sum(z[(station, 1)] for station in I_second) +
        cost_drone_millions * sum(z[sl] for sl in station_levels)
    )

    @objective(model, Max,
        sum(static_map[point...] * theta[point] for point in I) - epsilon * budget_used_expr)

    @constraint(model, budget_used_expr <= budget_millions)

    @constraint(model, [i in I_common], xg[i] + z[(i, 1)] <= 1)
    @constraint(model, [station in I_second, level in 1:(station_drone_caps[station] - 1)],
        z[(station, level)] >= z[(station, level + 1)])

    @constraint(model, [pair in feasible_pairs], u[pair] <= z[(pair[2], pair_first_cover_level[pair])])
    @constraint(model, [pair in feasible_pairs], w[pair] >= 0)
    @constraint(model, [pair in feasible_pairs], w[pair] <= u[pair])
    @constraint(model, [pair in feasible_pairs],
        w[pair] <= sum(
            pair_delta_levels[pair][level] *
            z[(pair[2], level)]
            for level in 1:station_drone_caps[pair[2]]
        ))

    @constraint(model, [cell in keys(cell_to_pairs)],
        sum(u[pair] for pair in cell_to_pairs[cell]) <= 1)

    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [cell in I; cell in I_prime_set],
        theta[cell] <= xg[cell] + sum(w[pair] for pair in get(cell_to_pairs, cell, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[])))
    @constraint(model, [cell in I; !(cell in I_prime_set)],
        theta[cell] <= sum(w[pair] for pair in get(cell_to_pairs, cell, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[])))

    time_model_creation_end = time_ns() / 1e9
    println("Model creation took ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    flush(stdout)

    time_solve_start = time_ns() / 1e9
    optimize!(model)
    time_solve_end = time_ns() / 1e9
    println("Solving took ", round(time_solve_end - time_solve_start, digits=2), " seconds")

    status = termination_status(model)
    println("Termination status: ", status)
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        gap = abs(obj_bound - obj_val) / max(abs(obj_val), 1e-10) * 100.0
        println("Objective value:  ", round(obj_val, digits=4))
        println("Objective bound:  ", round(obj_bound, digits=4))
        println("MIP gap:          ", round(gap, digits=2), "%")
    else
        println("No integer-feasible solution found within time limit!")
    end
    flush(stdout)

    if !has_values(model)
        println("ERROR: no feasible solution — returning empty placements.")
        flush(stdout)
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    selected_x_indices = [(i[1] - 1, i[2] - 1) for i in I_prime if value(xg[i]) > 0.5]
    selected_y_indices = [(station[1] - 1, station[2] - 1) for station in I_second if value(z[(station, 1)]) > 0.5]
    drone_allocations = [sum(Int(round(value(z[(station, level)]))) for level in 1:station_drone_caps[station])
                         for station in I_second if value(z[(station, 1)]) > 0.5]

    n_sensors = length(selected_x_indices)
    n_stations = length(selected_y_indices)
    n_drones = sum(drone_allocations; init=0)
    budget_used = n_sensors * cost_sensor_millions + n_stations * cost_station_millions + n_drones * cost_drone_millions

    println("\n=== BUDGET ALLOCATION ===")
    println("  Ground sensors:     ", n_sensors, " (cost ", round(n_sensors * cost_sensor_millions, digits=2), "M)")
    println("  Charging stations:  ", n_stations, " (cost ", round(n_stations * cost_station_millions, digits=2), "M)")
    println("  Drones:             ", n_drones, " (cost ", round(n_drones * cost_drone_millions, digits=2), "M)")
    println("  Budget used:        ", round(budget_used, digits=2), "M / ", budget_millions, "M")

    println("\n=== TIMING SUMMARY ===")
    println("  Preprocessing:   ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    println("  Model creation:  ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    println("  Solving:         ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    println("  TOTAL:           ", round((time_ns() / 1e9) - time_start, digits=2), " seconds")
    println("======================\n")
    flush(stdout)

    return selected_x_indices, selected_y_indices, drone_allocations
end


function Max_Coverage_Kernel_Masked_Budget_StationMax_GreedyUniform(static_map_file, budget_millions, cost_sensor_millions, cost_station_millions, cost_drone_millions, kernel, kernel_size_x, kernel_size_y, mask_file, recompute_kernel=false, n_steps=63, time_limit_seconds=600.0, max_drones_per_station=7, candidate_percentile=0.80)
    """
    Budget-constrained StationMax placement with greedy-uniform coverage.

    For each candidate station the greedy set-cover heuristic determines the
    fraction of the zone's total risk that k drones can cover.  That aggregate
    risk fraction is then applied *uniformly* to every reachable cell in the
    station's zone, instead of being limited to the specific greedy paths.
    Coverage from different stations does not add (at-most-one assignment).
    """

    time_start = time_ns() / 1e9
    time_preprocessing_start = time_ns() / 1e9

    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("=== Max_Coverage_Kernel_Masked_Budget_StationMax_GreedyUniform ===")
    println("static_map_file=", static_map_file)
    println("budget=", budget_millions, "M")
    println("costs: sensor=", cost_sensor_millions, "M, station=", cost_station_millions, "M, drone=", cost_drone_millions, "M")
    println("max_drones_per_station=", max_drones_per_station)
    println("candidate_percentile=", candidate_percentile)
    epsilon = budget_millions > 300.0 ? 0.1 : 0.0
    println("budget regularization epsilon=", epsilon)

    if T != 1
        println("Averaging first ", min(10, T), " time steps")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1 / min(10, T)) * sum(static_map[t, i, j] for t in 1:min(10, T))
        end
        static_map = avg_risk
    else
        static_map = static_map[1, :, :]
    end

    if !isnothing(mask_file) && mask_file != ""
        mask = load_mask(mask_file)
    else
        mask = ones(N, M)
    end

    I_all_feasible = [(i[1], i[2]) for i in findall(mask .> 0.0)]
    I = I_all_feasible
    println("Total feasible points: ", length(I_all_feasible))

    max_possible_grounds  = floor(Int, budget_millions / cost_sensor_millions)
    max_possible_charging = floor(Int, budget_millions / cost_station_millions)
    max_possible_drones   = floor(Int, budget_millions / cost_drone_millions)
    println("Max possible: sensors=", max_possible_grounds,
            ", stations=", max_possible_charging,
            ", drones=", max_possible_drones)

    if max_possible_grounds == 0 && max_possible_charging == 0
        println("WARNING: Budget too small for any device. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    ground_risks = [(loc, static_map[loc...]) for loc in I_all_feasible]
    ground_risk_values = [r for (_, r) in ground_risks]
    ground_risk_threshold = length(ground_risk_values) > 0 ? quantile(ground_risk_values, candidate_percentile) : 0.0
    I_prime = [loc for (loc, risk) in ground_risks if risk >= ground_risk_threshold]
    println("Ground sensor candidates (top ", round((1 - candidate_percentile) * 100, digits=1), "% by risk): ", length(I_prime), " / ", length(I_all_feasible))

    charging_potentials = Dict{Tuple{Int,Int}, Float64}()
    for (cx, cy) in I_all_feasible
        coverage_potential = 0.0
        for dx in max(-cx + 1, -kernel_size_x):min(N - cx, kernel_size_x)
            for dy in max(-cy + 1, -kernel_size_y):min(M - cy, kernel_size_y)
                target_x, target_y = cx + dx, cy + dy
                if 1 <= target_x <= N && 1 <= target_y <= M
                    if mask[target_x, target_y] <= 0
                        continue
                    end
                    kernel_weight = get(kernel, (dx, dy), 0.0)
                    if kernel_weight > 0
                        coverage_potential += kernel_weight * static_map[target_x, target_y]
                    end
                end
            end
        end
        charging_potentials[(cx, cy)] = coverage_potential
    end

    charging_potential_values = collect(values(charging_potentials))
    charging_potential_threshold = length(charging_potential_values) > 0 ? quantile(charging_potential_values, candidate_percentile) : 0.0
    I_second = [loc for (loc, potential) in charging_potentials if potential >= charging_potential_threshold]
    println("Charging station candidates (top ", round((1 - candidate_percentile) * 100, digits=1), "% by coverage potential): ", length(I_second), " / ", length(I_all_feasible))

    if length(I_prime) < max_possible_grounds
        println("WARNING: Not enough ground candidates (", length(I_prime), ") for max possible (", max_possible_grounds, "). Using all feasible.")
        I_prime = I_all_feasible
    end
    if length(I_second) < max_possible_charging
        println("WARNING: Not enough charging candidates (", length(I_second), ") for max possible (", max_possible_charging, "). Using all feasible.")
        I_second = I_all_feasible
    end

    if isempty(I_all_feasible)
        println("WARNING: No feasible cells. Returning empty placements.")
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    I_prime_set = Set(I_prime)

    one_way_reach = floor(Int, kernel_size_x / 2)
    println("Pair pruning radius (L∞ one-way reach): ", one_way_reach)

    feasible_pairs = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
    cell_to_pairs = Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}}()
    station_to_pairs = Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}}()
    pair_first_cover_level = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, Int}()
    pair_delta_levels = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, Vector{Float64}}()
    station_drone_caps = Dict{Tuple{Int,Int}, Int}()

    for station in I_second
        local_pairs = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
        first_cover_level, delta_levels = build_station_greedy_uniform_kernels(
            station, mask, N, M, one_way_reach, max_drones_per_station, static_map, kernel_size_x
        )

        for (cell, first_level) in first_cover_level
            pair = (cell, station)
            push!(feasible_pairs, pair)
            push!(local_pairs, pair)
            if !haskey(cell_to_pairs, cell)
                cell_to_pairs[cell] = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
            end
            push!(cell_to_pairs[cell], pair)
            pair_first_cover_level[pair] = first_level
            pair_delta_levels[pair] = delta_levels[cell]
        end

        if !isempty(local_pairs)
            station_to_pairs[station] = local_pairs
            station_drone_caps[station] = min(max_possible_drones, max(1, max_drones_per_station))
        end
    end

    I_second = [station for station in I_second if haskey(station_to_pairs, station)]
    I_second_set = Set(I_second)
    I_common = [loc for loc in I_prime if loc in I_second_set]

    println("Stations with at least one feasible pair: ", length(I_second))
    println("Feasible station-cell pairs kept: ", length(feasible_pairs))
    if !isempty(I_second)
        cap_values = [station_drone_caps[s] for s in I_second]
        println("Per-station drone cap: min=", minimum(cap_values), ", max=", maximum(cap_values), ", mean=", round(mean(cap_values), digits=2))
    end

    station_levels = Tuple{Tuple{Int,Int}, Int}[]
    for station in I_second
        for level in 1:station_drone_caps[station]
            push!(station_levels, (station, level))
        end
    end
    println("Station-drone binary levels: ", length(station_levels))

    time_preprocessing_end = time_ns() / 1e9
    println("Preprocessing took ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    flush(stdout)

    time_model_creation_start = time_ns() / 1e9

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")))
    if time_limit_seconds > 0
        set_time_limit_sec(model, time_limit_seconds)
        println("Gurobi time limit set to ", time_limit_seconds, " seconds")
        flush(stdout)
    end

    xg    = @variable(model, [i in I_prime], Bin)
    z     = @variable(model, [sl in station_levels], Bin)
    u     = @variable(model, [pair in feasible_pairs], Bin)
    w     = @variable(model, [pair in feasible_pairs])
    theta = @variable(model, [i in I])

    budget_used_expr = @expression(model,
        cost_sensor_millions * sum(xg) +
        cost_station_millions * sum(z[(station, 1)] for station in I_second) +
        cost_drone_millions * sum(z[sl] for sl in station_levels)
    )

    @objective(model, Max,
        sum(static_map[point...] * theta[point] for point in I) - epsilon * budget_used_expr)

    @constraint(model, budget_used_expr <= budget_millions)

    @constraint(model, [i in I_common], xg[i] + z[(i, 1)] <= 1)
    @constraint(model, [station in I_second, level in 1:(station_drone_caps[station] - 1)],
        z[(station, level)] >= z[(station, level + 1)])

    @constraint(model, [pair in feasible_pairs], u[pair] <= z[(pair[2], pair_first_cover_level[pair])])
    @constraint(model, [pair in feasible_pairs], w[pair] >= 0)
    @constraint(model, [pair in feasible_pairs], w[pair] <= u[pair])
    @constraint(model, [pair in feasible_pairs],
        w[pair] <= sum(
            pair_delta_levels[pair][level] *
            z[(pair[2], level)]
            for level in 1:station_drone_caps[pair[2]]
        ))

    @constraint(model, [cell in keys(cell_to_pairs)],
        sum(u[pair] for pair in cell_to_pairs[cell]) <= 1)

    @constraint(model, [i in I], 0 <= theta[i] <= 1)
    @constraint(model, [cell in I; cell in I_prime_set],
        theta[cell] <= xg[cell] + sum(w[pair] for pair in get(cell_to_pairs, cell, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[])))
    @constraint(model, [cell in I; !(cell in I_prime_set)],
        theta[cell] <= sum(w[pair] for pair in get(cell_to_pairs, cell, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[])))

    time_model_creation_end = time_ns() / 1e9
    println("Model creation took ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    flush(stdout)

    time_solve_start = time_ns() / 1e9
    optimize!(model)
    time_solve_end = time_ns() / 1e9
    println("Solving took ", round(time_solve_end - time_solve_start, digits=2), " seconds")

    status = termination_status(model)
    println("Termination status: ", status)
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        gap = abs(obj_bound - obj_val) / max(abs(obj_val), 1e-10) * 100.0
        println("Objective value:  ", round(obj_val, digits=4))
        println("Objective bound:  ", round(obj_bound, digits=4))
        println("MIP gap:          ", round(gap, digits=2), "%")
    else
        println("No integer-feasible solution found within time limit!")
    end
    flush(stdout)

    if !has_values(model)
        println("ERROR: no feasible solution — returning empty placements.")
        flush(stdout)
        return Tuple{Int,Int}[], Tuple{Int,Int}[], Int[]
    end

    selected_x_indices = [(i[1] - 1, i[2] - 1) for i in I_prime if value(xg[i]) > 0.5]
    selected_y_indices = [(station[1] - 1, station[2] - 1) for station in I_second if value(z[(station, 1)]) > 0.5]
    drone_allocations = [sum(Int(round(value(z[(station, level)]))) for level in 1:station_drone_caps[station])
                         for station in I_second if value(z[(station, 1)]) > 0.5]

    n_sensors = length(selected_x_indices)
    n_stations = length(selected_y_indices)
    n_drones = sum(drone_allocations; init=0)
    budget_used = n_sensors * cost_sensor_millions + n_stations * cost_station_millions + n_drones * cost_drone_millions

    println("\n=== BUDGET ALLOCATION ===")
    println("  Ground sensors:     ", n_sensors, " (cost ", round(n_sensors * cost_sensor_millions, digits=2), "M)")
    println("  Charging stations:  ", n_stations, " (cost ", round(n_stations * cost_station_millions, digits=2), "M)")
    println("  Drones:             ", n_drones, " (cost ", round(n_drones * cost_drone_millions, digits=2), "M)")
    println("  Budget used:        ", round(budget_used, digits=2), "M / ", budget_millions, "M")

    println("\n=== TIMING SUMMARY ===")
    println("  Preprocessing:   ", round(time_preprocessing_end - time_preprocessing_start, digits=2), " seconds")
    println("  Model creation:  ", round(time_model_creation_end - time_model_creation_start, digits=2), " seconds")
    println("  Solving:         ", round(time_solve_end - time_solve_start, digits=2), " seconds")
    println("  TOTAL:           ", round((time_ns() / 1e9) - time_start, digits=2), " seconds")
    println("======================\n")
    flush(stdout)

    return selected_x_indices, selected_y_indices, drone_allocations
end


# using ImageFiltering  # For the equivalent of scipy.ndimage.convolve

# function count_paths_convolution(N, M, origin, n)
#     # Initialize the dynamic programming array
#     dp = zeros(Float64, N, M)
#     dp[origin[1], origin[2]] = 1.0  # Note: Julia is 1-based indexing

#     # Create the 3x3 kernel
#     kernel = centered(ones(Float64, 3, 3))  # Use centered kernel

#     # Apply convolution n times
#     for _ in 1:n
#         dp = imfilter(dp, kernel, Fill(0.0))  # Use Fill(0.0) instead of "constant"
#     end

#     # Get the origin value for normalization
#     origin_value = dp[origin[1], origin[2]]

#     # Create the mapping dictionary
#     mapping = Dict{Tuple{Int,Int}, Float64}()
#     for x in 1:N, y in 1:M
#         mapping[(x - origin[1], y - origin[2])] = dp[x,y]/origin_value
#     end

#     return mapping
# end


# # Test parameters
# N = 60  # Grid size
# M = 60
# N_grounds = 5  # Number of ground stations
# N_charging = 3  # Number of charging stations

# # Generate the kernel using the translated function
# origin = (N÷2, M÷2)  # Center point
# n = 20  # Number of steps for the convolution
# kernel = count_paths_convolution(N, M, origin, n)
# println("kernel=", kernel)

# # Get kernel size
# kernel_size_x = 20
# kernel_size_y = 20

# println("kernel_size_x=", kernel_size_x)
# println("kernel_size_y=", kernel_size_y)

# # Call the optimization function
# ground_locations, charging_locations = Max_Coverage_Kernel(
#     "WideDataset/0016_03070/burn_map_rescaled_26x30_substeps_63.npy",
#     N_grounds,
#     N_charging,
#     kernel,
#     kernel_size_x,
#     kernel_size_y
# )

# # Print results
# println("Ground station locations: ", ground_locations)
# println("Charging station locations: ", charging_locations)

# # Optional: Visualize the results
# using Plots

# # Create a heatmap of the static map
# static_map = load_burn_map("WideDataset/0016_03070/burn_map_rescaled_26x30_substeps_63.npy")
# heatmap(static_map[1,:,:], title="Burn Map with Station Placements")

# # Add ground stations
# for (x, y) in ground_locations
#     scatter!([x+1], [y+1], label="Ground Station", color=:blue, markersize=8)
# end

# # Add charging stations
# for (x, y) in charging_locations
#     scatter!([x+1], [y+1], label="Charging Station", color=:red, markersize=8)
# end

# # Save the plot
# savefig("station_placements.png")