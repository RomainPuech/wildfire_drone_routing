# 2025, Formulations and code by Danique De Moor, adapted by Romain Puech

# TODO strict typing

println("installing packages")
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
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ

include("helper_functions.jl")

function gamma_matrix(I, K, max_battery_time)
    max_distance = floor(max_battery_time / 2) + 1
    gamma_values = zeros(length(I), length(K))  # Preallocate matrix

    # Precompute index lookup tables
    I_map = Dict(i => idx for (idx, i) in enumerate(I))
    K_map = Dict(k => idx for (idx, k) in enumerate(K))

    for (ii, i) in enumerate(I)
        for (kk, k) in enumerate(K)
            r = L_inf_distance(i, k)
            if r <= max_distance - 1
                gamma_values[ii, kk] = (1 / max_distance) * (max_distance - r - 1)
            end
        end
    end

    # Return the full matrix and lookup maps
    return gamma_values, I_map, K_map
end

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
    @objective(model, Max, sum(risk_pertime[1,i...]*theta[i] for i in I_possible))

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

    I_prime = nothing
    I_second = nothing

    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)

    I = [(x, y) for x in 1:N for y in 1:M]


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

    @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(risk_pertime[1,k...]*y[k] for k in I_second))

    @constraint(model, [i in intersect(I_prime, I_second)], x[i] + y[i] <= 1) # 2b
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) 
    # Precompute valid (i, j) pairs where L∞ distance ≤ 4
    close_pairs = [(i, j) for i in I for j in I if i != j && maximum(abs.(i .- j)) <= 4]
    # Add constraints efficiently
    @constraint(model, [(i, j) in close_pairs], y[i] + y[j] <= 1)

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) ≈ 1]
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds")

    return selected_x_indices, selected_y_indices
end

