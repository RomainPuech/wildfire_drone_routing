# 2025, Formulations and code by Danique De Moor, adapted by Romain Puech

# TODO strict typing

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
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ, Statistics

include("helper_functions.jl")

function load_parameters(risk_pertime_file)
    risk_pertime, _ = load_burn_map(risk_pertime_file)
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

#charging stations detect by rate detection_rate in neighboring grids until L infinity norm 4
function NEW_SENSOR_STRATEGY(risk_pertime_file, N_grounds, N_charging)
    println("NEW STRATEGY")

    I_prime = nothing
    I_second = nothing
    I_third = nothing

    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    println("N=", N)
    println("M=", M)
    detection_rate = 0.7

    I = [(x, y) for x in 1:N for y in 1:M]


    if I_prime === nothing
        I_prime = I
    end

    if I_second === nothing
        I_second = I
    end
    
    if I_third === nothing
        I_third = I
    end

    I_possible = union(I_prime,I_second,I_third)

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables
    theta = @variable(model, [i in I_possible])

    t1 = time_ns() / 1e9  # Convert nanoseconds to seconds

    close_pairs = [(i, j) for i in I_possible for j in I_second if i != j && maximum(abs.(i .- j)) <= 4]
    println("Step 1 took ", (time_ns() / 1e9) - t1, " seconds")
    t4 = time_ns() / 1e9
    # close_dict = Dict(i => [k for (j, k) in close_pairs if j == i] for i in I_prime)
    close_dict = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (i, j) in close_pairs
    if haskey(close_dict, i)
        push!(close_dict[i], j)
    else
        close_dict[i] = [j]
    end
    end
    println("Step 1 took ", (time_ns() / 1e9) - t4, " seconds")


    # risk_pertime is not indexed by I but rather is 2 dimensional with coordinates given by i, so we have to splat i with '...'
    # @objective(model, Max, sum(risk_pertime[1,i...]*theta[i] for i in I_possible))

    # risk_pertime is averaged instead of looking at risk_pertime at t = 1
    @objective(model, Max, sum((1/T)*sum(risk_pertime[t,i...] for t in 1:T)*theta[i] for i in I_possible))

    #constraints defining the epigraph variable theta[i] = min(1,x[i] + sum_k y[i,k]*y[k])
    @constraint(model, [i in I_possible], theta[i] <= 1)
    t2 = time_ns() / 1e9  # Convert nanoseconds to seconds
    @constraint(model, [i in I_prime], theta[i] <= x[i] + sum(detection_rate*phi(i,k)*y[k] for k in get(close_dict,i,[])))
    println("Step 2 took ", (time_ns() / 1e9) - t2, " seconds")
    t3 = time_ns() / 1e9  # Convert nanoseconds to seconds
    @constraint(model, [i in setdiff(union(I_second,I_third),I_prime)], theta[i] <= sum(y[k] for k in I_second))
    println("Step 3 took ", (time_ns() / 1e9) - t3, " seconds")
    @constraint(model, theta >= 0)
    @constraint(model, [i in intersect(I_prime, I_second)], x[i] + y[i] <= 1) # 2b
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) # modified: contraint on the total number of ground/charging stations instead of a budget.
    
    optimize!(model)
    println("Optimizing model took ", (time_ns() / 1e9) - t5, " seconds")

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) ≈ 1]
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) ≈ 1]

    return selected_x_indices, selected_y_indices

end

#no charging stations allowed within L infinity distance of 4 of each other
function NEW_SENSOR_STRATEGY_2(risk_pertime_file, N_grounds, N_charging)
    println("NEW STRATEGY 2")


    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)

    I = [(x, y) for x in 1:N for y in 1:M]

    # I_prime = [(i, j) for (i, j) in I if risk_pertime[5, i, j] > 0.05]
    I_prime = I
    I_second = I_prime


    if I_prime === nothing
        I_prime = I
    end

    if I_second === nothing
        I_second = I
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables

    @objective(model, Max, sum((1/T)*sum(risk_pertime[t,i...] for t in 1:T)*x[i] for i in I_prime) + sum((1/T)*sum(risk_pertime[t,k...] for t in 1:T)*y[k] for k in I_second))

    @constraint(model, [i in intersect(I_prime, I_second)], x[i] + y[i] <= 1) # 2b
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) 
    # Precompute valid (i, j) pairs where L∞ distance ≤ 4
    close_pairs = [(i, j) for i in I_second for j in I_second if i != j && maximum(abs.(i .- j)) <= 1]
    # Add constraints efficiently
    @constraint(model, [(i, j) in close_pairs], y[i] + y[j] <= 1)

    cs_pairs = [(i, j) for (i, j) in close_pairs if j in I_prime]  # charging-sensor
    @constraint(model, [(i,j) in cs_pairs], y[i] + x[j] <= 1)

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) ≈ 1]
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds")
    println("Average risk equals")

    return selected_x_indices, selected_y_indices
end

function NEW_SENSOR_STRATEGY_3(risk_pertime_file, N_grounds, N_charging)
    println("NEW STRATEGY 3 - Optimized for speed")

    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    println("risk_pertime_file=", risk_pertime_file)
    println("T=", T)
    println("N=", N)
    println("M=", M)

    I = [(x, y) for x in 1:N for y in 1:M]

    # Precompute average risk for each cell to avoid recalculating it multiple times
    avg_risk = zeros(N, M)
    for i in 1:N, j in 1:M
        avg_risk[i,j] = (1/T) * sum(risk_pertime[t,i,j] for t in 1:T)
    end

    # prerfilter: keep only cells with risk > 90% of other cells
    #first_quartile_risk = quantile(vec(avg_risk), 0.0)
    I_prime = [(i, j) for i in 1:N, j in 1:M if avg_risk[i,j] > 0.0] # >first_quartile_risk
    I_second = I_prime

    # prrint how many cells are discarded
    println("Number of cells discarded: ", length(I) - length(I_prime))

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Variables 
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables

    # Objective - use precomputed average risk
    @objective(model, Max, 
        sum(avg_risk[i...] * x[i] for i in I_prime) + 
        sum(avg_risk[i...] * y[i] for i in I_second))

    # Constraints
    @constraint(model, [i in I_prime], x[i] + y[i] <= 1) # Can't place both at same location
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) 

    # Precompute valid (i, j) pairs where L∞ distance ≤ 1 (reverted to NEW_SENSOR_STRATEGY_2 approach)
    close_pairs = [(i, j) for i in I_second for j in I_second if i != j && maximum(abs.(i .- j)) <= 5]
    # Add constraints efficiently
    @constraint(model, [(i, j) in close_pairs], y[i] + y[j] <= 1)

    cs_pairs = [(i, j) for (i, j) in close_pairs if j in I_prime]  # charging-sensor
    @constraint(model, [(i,j) in cs_pairs], y[i] + x[j] <= 1)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) > 0.5]
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) > 0.5]

    println("selected_x_indices=", selected_x_indices)
    println("selected_y_indices=", selected_y_indices)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices
end

function NEW_SENSOR_STRATEGY_4(risk_pertime_file, N_grounds, N_charging)
    println("NEW STRATEGY 3 - Optimized for speed")

    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    optimization_horizon = 3
    n_drones = 5
    T_opt = optimization_horizon
    max_battery_time = 10

    I = [(x, y) for x in 1:N for y in 1:M]

    # Precompute average risk for each cell to avoid recalculating it multiple times
    avg_risk = zeros(N, M)
    for i in 1:N, j in 1:M
        avg_risk[i,j] = (1/T) * sum(risk_pertime[t,i,j] for t in 1:T)
    end

    # prerfilter: keep only cells with risk > 90% of other cells
    first_quartile_risk = quantile(vec(avg_risk), 0.9)
    I_prime = [(i, j) for i in 1:N, j in 1:M if avg_risk[i,j] > first_quartile_risk]
    I_second = I_prime

    I_third = get_drone_gridpoints(I_second, floor(max_battery_time/2), I)
    GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(I_third))

    # prrint how many cells are discarded
    println("Number of cells discarded: ", length(I) - length(I_prime))

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Variables 
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables
    a = @variable(model, [i in I_third, t=1:T, s=1:n_drones], Bin)
    c = @variable(model, [i in I_prime, t=1:T, s=1:n_drones], Bin)
    b = @variable(model, [t=1:T, s=1:n_drones], Int)
    theta = @variable(model, [t=1:T_opt, k in I_third])


    # Objective - use precomputed average risk
    @objective(model, Max, 
        sum(avg_risk[i...] * x[i] for i in I_prime) + 
        sum(avg_risk[i...] * y[i] for i in I_second) + sum(risk_pertime[30+t,k[1],k[2]]*theta[t,k] for t=1:T_opt, k in I_third))

    # Constraints
    @constraint(model, [i in I_prime], x[i] + y[i] <= 1) # Can't place both at same location
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) 


    # Precompute valid (i, j) pairs where L∞ distance ≤ 1 (reverted to NEW_SENSOR_STRATEGY_2 approach)
    close_pairs = [(i, j) for i in I_second for j in I_second if i != j && maximum(abs.(i .- j)) <= 4]
    # Add constraints efficiently
    @constraint(model, [(i, j) in close_pairs], y[i] + y[j] <= 1)

    cs_pairs = [(i, j) for (i, j) in close_pairs if j in I_prime]  # charging-sensor
    @constraint(model, [(i,j) in cs_pairs], y[i] + x[j] <= 1)

    # Common constraints - using tuple indices directly
    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T_opt, s=1:n_drones], 
               sum(a[i,t,s] for i in I_third) + 
               sum(c[i,t,s] for i in I_second) == 1)

    # Movement constraints - using tuple indexing
    # Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in I_second, t in 1:T_opt-1, s in 1:n_drones], 
                c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in I if i in neighbors_and_point(j)) + c[j,t,s])
    
    @constraint(model, [j in setdiff(I_third,I_second), t in 1:T_opt-1, s in 1:n_drones], 
                a[j,t+1,s] <= sum(a[i,t,s] for i in I_third if i in neighbors_and_point(j)) + sum(c[i,t,s] for i in I_second if i in neighbors_and_point(j)))
    
    @constraint(model, [t=1:T_opt, s=1:n_drones], 0 <= b[t,s] <= max_battery_time)

    @constraint(model, [s in 1:n_drones, t in 1:T_opt],
    b[t,s] >= max_battery_time*sum(c[i,t,s] for i in I_second))
    @constraint(model, [t in 1:T_opt-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - 1 + (max_battery_time+1) * sum(c[i,t+1,s] for i in I_second))

    # Objective function with theta variables
    @constraint(model, [t=1:T_opt, k in I_second], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones) + x[k] + y[k])
    @constraint(model, [t=1:T_opt, k in setdiff(I_third,I_second)], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones))

    @constraint(model, [t=1:T_opt, k in I_third], 0 <= theta[t,k] <= 1)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) > 0.5]
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) > 0.5]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices
end