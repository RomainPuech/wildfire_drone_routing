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
function NEW_SENSOR_STRATEGY_2(risk_pertime_file, N_grounds, N_charging, max_battery_time)
    println("NEW STRATEGY 2")


    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)

    I = [(x, y) for x in 1:N for y in 1:M]

    # I_prime = [(i, j) for (i, j) in I if risk_pertime[T, i, j] > 0.3]
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

    # @objective(model, Max, sum((1/T)*sum(risk_pertime[t,i...] for t in 1:T)*x[i] for i in I_prime) + sum((1/T)*sum(risk_pertime[t,k...] for t in 1:T)*y[k] for k in I_second))
    @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(risk_pertime[1,k...]*y[k] for k in I_second))

    @constraint(model, [i in intersect(I_prime, I_second)], x[i] + y[i] <= 1) # 2b
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) 
    # Precompute valid (i, j) pairs where L∞ distance ≤ 4
    close_pairs = [(i, j) for i in I_second for j in I_second if i != j && maximum(abs.(i .- j)) <= floor(max_battery_time/2)]
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

