# 2025, Formulations and code by Danique De Moor, adapted by Romain Puech

# TODO strict typing

println("installing packages")
import Pkg
Pkg.add("IJulia")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Distances")
Pkg.add("MAT")
Pkg.add("Plots")
Pkg.add("FFMPEG")
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("Clustering")
Pkg.add("NPZ")
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ

include("helper_functions.jl")

function L_inf_distance(a,b)
    """
    Returns the L-infinity distance between a and b in R^n
    """
    return maximum(abs.(a .- b))
end

function neighbors(i, I=nothing)
    """
    Returns the L-infinity norm-neighbors of i in Z^n, intersected with feasible set I if provided
    (returns the feasible cells directly around i)
    """
    n = length(i)
    neighbors_list = []
    
    # Generate all possible combinations of -1, 0, 1 in n dimensions
    for moves in Iterators.product(fill((-1,0,1), n)...)
        if any(m != 0 for m in moves)  # Skip the point itself
            point = [i[j] + moves[j] for j in 1:n]
            if I === nothing || point in I # if the point belongs to the original set I
                push!(neighbors_list, Tuple(i[j] + moves[j] for j in 1:n))
            end
        end
    end
    
    return neighbors_list
end

function phi(x,y)
    return L_inf_distance(x, y) <= 4 ? 1 : 0
end

function test()
    println("test")
end

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


function ground_charging_opt_model_grid(risk_pertime_file, N_grounds, N_charging)
    """
    Returns the locations of all ground sensors and charging stations.

    Input:
    I: Set of all positions
    I_prime: Set of feasible positions for ground sensors. Default to I
    I_second: Set of all feasible positions for charging stations. Defaults to I
    risk_pertime_dir: Folder containing burn maps
    """
    println("In julia fuinction")
    # Print current working directory contents
    # println("Current working directory: ", pwd())
    # println("Directory contents:")
    # for (root, dirs, files) in walkdir(pwd())
    #     println("Directory: ", root)
    #     for dir in dirs
    #         println("  Dir: ", dir)
    #     end
    #     for file in files
    #         println("  File: ", file)
    #     end
    # end
    I_prime = nothing
    I_second = nothing

    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)

    nu = 1
    omega = 1
    b = 1.6
    B = 700 # total budget

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

    # risk_pertime is not indexed by I but rather is 2 dimensional with coordinates given by i, so we have to splat i with '...'
    # @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(phi[i,k]*risk_pertime[1,i...]*y[k] for k in I_second for i in neighbors(k,I))) # 2a
    # modifyed 2a: we say that charging stations / ground sensors can only detect ON their cell
    @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(risk_pertime[1,k...]*y[k] for k in I_second))

    @constraint(model, [i in intersect(I_prime, I_second)], x[i] + y[i] <= 1) # 2b

    # We don't need the following since stations can only see their own cell
    # for i in I_prime
    #     @constraint(model, x[i] + sum(phi[i,k]*y[k] for k in I_second) <= b) # 2c
    # end

    #CHANGE!! ---------
    #for i in I_second
    @constraint(model, sum(y[k] for k in I_second) <= 1) # 2d
    #@constraint(model, sum(phi(i,k)*y[k] for k in I_second) <= 1) # 2d
    #end
    #-------------------

    #@constraint(model, nu*sum(y[i] for i in I_second) + omega*sum(x[i] for i in I_prime) <= B) # 2e, budget constraint
    # modified: contraint on the total number of ground/charging stations instead of a budget.
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) # modified: contraint on the total number of ground/charging stations instead of a budget.

    optimize!(model)

    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) ≈ 1]
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds")

    return selected_x_indices, selected_y_indices
end

function drone_routing_example(drones, batteries, risk_pertime_file, time_horizon)
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    
    # Generate random moves for each drone
    # output should have this format: [("move", (dx, dy)), ("move", (dx, dy)), ...]
    return [[("move", (rand(-5:5), rand(-5:5))) for _ in 1:length(drones)] for _ in 1:time_horizon]
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

#Can we make rollinghorizon_step initialization parameter?
function NEW_ROUTING_STRATEGY(risk_pertime_file,n_drones,ChargingStations,reevaluation_step, T, max_battery_time, battery_level,fly_indices,charge_indices, movement_plan_prev) #T here is optimization_horizon: should be > max_battery_time, reevaluation_step < floor(optimization_horizon/2)

    time_start = time_ns() / 1e9 

    params = load_parameters(risk_pertime_file)

    risk_pertime = params.risk_pertime
    I = params.I
    I_prime = params.I_prime
    I_second = params.I_second
    B_max = 1
    B_min = 0.2

    GridpointsDrones = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I) # = allowed_gridpoints_drones
    GridpointsDronesDetecting = setdiff(covered_points,ChargingStations) # = NonChargingstations_allowed

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    #Defining the variables
    a = @variable(model, [i in GridpointsDrones, t in 1:T, s in n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t in 1:T, s in n_drones], Bin)
    b = @variable(model, [t in 1:T, s in n_drones])
    
    @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))

    #Defining the constraints
    @constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1)
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:S], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if neighbors[i,j]) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if neighbors[i,j]))
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if neighbors[i,j]) + c[j,t+1,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in S], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if neighbors[i,j]))
    @constraint(model, [t in 1:T, s in 1:n_drones], B_min <= b[t,s] <= B_max)
    @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - 0.1*sum(a[i,t,s] for i in GridpointsDrones) + B_max*sum(c[i,t,s] for i in ChargingStations))
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(c[i,t,s] for i in ChargingStations) >= 1 - b[t,s]/B_min)
    # @constraint(model, [s in 1:n_drones], sum(c[i,1,s] for i in ChargingStations) == 1) #First run
    @constraint(model, [s in 1:n_drones, i in Chargingstations], c[i,1,s] == charge_indices[i,1,s])
    @constraint(model, [s in 1:n_drones, i in GridpointsDrones], a[i,1,s] == fly_indices[i,1,s])
    # @constraint(model, [s in 1:n_drones], b[1,s] == B_max) #First run
    @constraint(model, [s in 1:n_drones], b[1,s] == battery_level[reevaluation_step,s]) #makes sure that the battery_level is 
    @constraint(model, [s in 1:n_drones], b[T,s] >= B_min)
    @constraint(model, [s in 1:n_drones], sum(c[i,T,s] for i in ChargingStations) == 1)

    @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))
    optimize!(model)

    fly_indices = value(c);
    charge_indices = value(a);
    selected_fly_indices = [(i,t,s) for i in GridpointsDronesDetecting, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]
    battery_level = value(b);

    println("Took ", (time_ns() / 1e9) - time_start, " seconds")

    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("move", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end

    updated_movement_plan = vcat(movement_plan_prev, movement_plan[1:reevaluation_step])

    return fly_indices, charge_indices, battery_level, updated_movement_plan

end

function NEW_ROUTING_STRATEGY_FULL_HORIZON(risk_pertime_file,n_drones,ChargingStations,reevaluation_step,T,max_battery_time)
    battery_level = ones(optimization_horizon,n_drones)
    movement_plan_full = []
    fly_indices = nothing
    charge_indices = nothing
    movement_plan_prev = []
    full_horizon, N, M = size(risk_pertime)
    for t in 1:reevaluation_step:full_horizon
        println("Running optimization for time step $t")

        # Run optimization strategy for next reevaluation_step steps
        fly_indices, charge_indices, battery_level, movement_plan_updated = NEW_ROUTING_STRATEGY(risk_pertime_file, n_drones, ChargingStations, reevaluation_step, T, max_battery_time, battery_level, fly_indices, charge_indices, movement_plan_prev)
        append!(movement_plan_full,movement_plan_updated)
        movement_plan_prev = movement_plan_full
    end

    return movement_plan_full
end

function GREEDY_ROUTING_STRATEGY(risk_pertime_file,T,n_drones)
    params = load_parameters(risk_pertime_file)

    @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))
    
end

