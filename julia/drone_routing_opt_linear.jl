# import helper_functions such as load_burn_map
include("helper_functions.jl")
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ
using AxisArrays: AxisArray
# Index-based implementation for model reuse
# -----------------------------------------

if !isdefined(Main, :IndexRoutingModel)
  struct IndexRoutingModelLinear
    model::Model
    a::Array{VariableRef, 3}
    c::Array{VariableRef, 3}
    b::Array{VariableRef, 2}
    w::AbstractArray{VariableRef, 3}
    θ::Array{VariableRef, 2}
    init_constraints::Vector{ConstraintRef}
    next_move_constraints::Vector{ConstraintRef}
    GridpointsDrones::Vector{Tuple{Int,Int}}
    ChargingStations::Vector{Tuple{Int,Int}}
    risk_idx::Array{Float64, 2}
    T::Int
    n_drones::Int
    grid_to_idx::Dict{Tuple{Int,Int}, Int}
    charging_map::Dict{Int, Int}
    max_battery_time::Int
  end
end

function create_index_routing_model_linear(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time, objective_type="min_cumulative_prob") # 
    t1 = time_ns() / 1e9
    risk_pertime = load_burn_map(risk_pertime_file)
    println("risk_pertime_file: ", risk_pertime_file)
    H, N, M = size(risk_pertime)
    T = optimization_horizon
    
    # Convert Python lists of tuples to Julia Vector of tuples if needed
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
    GroundStationSet = Set(GroundStations)  # faster lookup
    
    # println("ChargingStations: ", ChargingStations)
    
    I = [(x, y) for x in 1:N for y in 1:M]
    
    # Get grid points and convert from Set to Vector
    GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
    GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDrones_set))
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))
    
    # println("GridpointsDrones[1:5]: ", GridpointsDrones[1:min(5, length(GridpointsDrones))])

    # precomputing the closest distance to a charging station for each gridpoint
    precomputed_closest_distance_to_charging_station = closest_distances(ChargingStations, GridpointsDrones)
    # println("precomputed_closest_distance_to_charging_station: \n", precomputed_closest_distance_to_charging_station)
    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Defining the variables using simple integers for position indices
    # Transform grid points to integer indices
    grid_to_idx = Dict(point => i for (i, point) in enumerate(GridpointsDrones))
    charging_idx = [grid_to_idx[point] for point in ChargingStations]
    ground_idx = [grid_to_idx[point] for point in intersect(GridpointsDrones,GroundStations)]
    c_index_to_grid_idx = Dict(i => grid_to_idx[ChargingStations[i]] for i in 1:length(ChargingStations))
    
    # Create variables with integer indices
    a = @variable(model, a[i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], Bin)
    c = @variable(model, c[i=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin)
    b = @variable(model, b[t=1:T, s=1:n_drones], Int)
    
    # Common constraints
    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T, s=1:n_drones], sum(a[i,t,s] for i=1:length(GridpointsDrones)) + sum(c[i,t,s] for i=1:length(ChargingStations)) == 1)
    
    # Movement constraints need to be updated to use integer indices
    # Map each grid point to its neighbors using integer indices
    neighbors_map = Dict()
    for (i, point) in enumerate(GridpointsDrones)
        neighbors_idx = [grid_to_idx[p] for p in GridpointsDrones if p in neighbors_and_point(point) && haskey(grid_to_idx, p)]
        neighbors_map[i] = neighbors_idx
    end
    
    # Charging stations map
    charging_map = Dict()
    for (i, point) in enumerate(ChargingStations)
        charging_map[i] = grid_to_idx[point]
    end
    
    for (i, point) in enumerate(ChargingStations)
        j = grid_to_idx[point]
        for t in 1:T-1, s in 1:n_drones
            @constraint(model, c[i,t+1,s] + a[j,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j]) + c[i,t,s])
            #@constraint(model, a[j,t,s] <= sum(a[k,t+1,s] for k in neighbors_map[j]) + c[i,t+1,s]) #TODO is it correct to remove this constraint?
        end
    end
    
    for j_idx in 1:length(GridpointsDrones)
        point = GridpointsDrones[j_idx]
        if !(point in ChargingStations)  # If not a charging station
            for t in 1:T-1, s in 1:n_drones
                @constraint(model, a[j_idx,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j_idx]))
                #@constraint(model, a[j_idx,t,s] <= sum(a[k,t+1,s] for k in neighbors_map[j_idx])) #TODO is it correct to remove this constraint?
            end
        end
    end
    
    # Min/max battery level constraints
    @constraint(model, [t=1:T, s=1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
    # Battery dynamics
    # bilinear version
    # @constraint(model, [t=1:T-1, s=1:n_drones], 
    #     b[t+1,s] == b[t,s] - sum(a[i,t+1,s] for i=1:length(GridpointsDrones)) + 
    #     (max_battery_time - b[t,s]) * sum(c[i,t+1,s] for i=1:length(ChargingStations)))
    # linear version
    @constraint(model, [s in 1:n_drones, t in 1:T], b[t,s] >= max_battery_time*sum(c[i,t,s] for i in 1:length(ChargingStations)))
    @constraint(model, [t in 1:T-1, s in 1:n_drones], 
        b[t+1,s] <= b[t,s] - 1 + (max_battery_time+1) * sum(c[i,t+1,s] for i in 1:length(ChargingStations)))

    # No suicide constraint
    # distance from closest charging station
    @constraint(model, [s=1:n_drones, i_idx=1:length(GridpointsDrones)], 
                b[T,s] >= a[i_idx,T,s]*precomputed_closest_distance_to_charging_station[i_idx])


    # Nico's special edit
    # objective_types : ["max_coverage", "max_cumulative_probability"]
    # if objective_type == "max_coverage"
        # Create objective variables with integer indices
    theta = @variable(model, [t=1:T, k=1:length(GridpointsDrones)])
    @constraint(model, [t=1:T, k=1:length(GridpointsDrones)], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones))
    @constraint(model, [t=1:T, k=1:length(GridpointsDrones)], 0 <= theta[t,k] <= 1)
    @constraint(model, [t=1:T, k=1:length(ChargingStations)], theta[t,grid_to_idx[ChargingStations[k]]]==0)
    if objective_type == "min_cumulative_prob"
        # Linear approximation of the routing: Variable of the visit time:
    
        # w_itt2:= 1 if in t2 between last z_it visit and t // 0 otherwise 
        zeta = @variable(model, zeta[i=1:length(GridpointsDrones), t=1:T], Int)
        w = @variable(model, w[i=1:length(GridpointsDrones), t=1:T, t2=1:t], Bin)

        @constraint(model, [t=2:T, i=1:length(GridpointsDrones)], zeta[i,t] >= zeta[i,t-1])
        @constraint(model,        [i=1:length(GridpointsDrones)], zeta[i,1] >= 0)
        
        
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones), s=1:n_drones], zeta[i,t] <= t) # Future visits
        @constraint(model, [t=2:T, i=1:length(GridpointsDrones), s=1:n_drones], zeta[i,t] <= zeta[i,t-1]+t*a[i,t,s]) # Future visits
        @constraint(model,        [i=1:length(GridpointsDrones), s=1:n_drones], zeta[i,1] <= a[i,1,s]) # Future visits
    
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones), t2=1:t], (t2 - zeta[i,t]) <= t*w[i,t,t2] )
        @constraint(model, [t=1:T, k=1:length(ChargingStations), t2=1:t],  w[k,t,t2] == 1 ) # always visiting this ones
        # @constraint(model, [t=1:T, k=1:length(ChargingStations)], theta[t,grid_to_idx[ChargingStations[k]]]==0)


        # @constraint(model, )
    end


    #Risk objective with integer indices
    risk_idx = zeros(H, length(GridpointsDrones))
    for t in 1:H
        for (k, point) in enumerate(GridpointsDrones)
            # if k in charging_idx
            #     risk_idx[t, k] = 0.0  # no reward for drone coverage
            # else
            risk_idx[t, k] = risk_pertime[t, point[1], point[2]]
            # end

            # Nico's edit

            # for k2 in 1:length(ChargingStations)
            #     if grid_to_idx[ChargingStations[k2]] == k # if is charging station, I don't care about its risk
            #         risk_idx[t,k] == 0 # reset to 0 the reward 
            #     end
            # end
        end
    end

    # #capacity constraint on charging stations
    # @constraint(model, [i in 1:length(ChargingStations), t in 1:T], sum(c[i,t,s] for s in 1:n_drones) <= 2)
    
    # Nico's edit
    if objective_type == "max_coverage"
        @objective(model, Max, sum(risk_idx[t,k]*theta[t,k] for t=1:T, k=1:length(GridpointsDrones)))
    elseif objective_type == "min_cumulative_prob"
        # @objective(model, Max, sum(( t*theta[t,i]-(t-zeta[i,t,s]) ) * risk_idx[t,i] for t=1:T, i=1:length(GridpointsDrones), s=1:n_drones))
        @objective(model, Min, sum(( w[i,t,t2] * risk_idx[t2,i] + theta[t,i]/t ) for t=1:T, t2=1:t,i=1:length(GridpointsDrones)))
    end
    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]
    t2 = time_ns() / 1e9
    println("Model created in ", t2 - t1, " seconds")
    # println("DEBUG: Charging Stations received in Julia:")
    println(ChargingStations)
    return IndexRoutingModelLinear(model, a, c, b, w, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, risk_idx, T, n_drones, grid_to_idx, charging_map, max_battery_time)
end

function solve_index_init_routing_linear(routing_model::IndexRoutingModelLinear, reevaluation_step)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    w = routing_model.w
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    
    # Clear any existing next_move constraints
    for con in routing_model.next_move_constraints
        delete(model, con)
    end
    empty!(routing_model.next_move_constraints)
    
    # Clear any existing init constraints
    for con in routing_model.init_constraints
        delete(model, con)
    end
    empty!(routing_model.init_constraints)
    
    # Add init-specific constraints
    t1 = time_ns() / 1e9
    
    # All drones start from a charging station at t=1
    for s in 1:n_drones
        # For each drone, sum over charging stations (by index)
        charging_station_idx = [grid_to_idx[station] for station in ChargingStations]
        charging_station_idxs = 1:length(ChargingStations)  # Indices into c array
        c_index_to_grid_idx = Dict(i => grid_to_idx[ChargingStations[i]] for i in 1:length(ChargingStations))
        
        constraint = @constraint(model, 
                               sum(c[i,1,s] for i in charging_station_idxs) + 
                               sum(a[grid_to_idx[ChargingStations[i]],1,s] for i in charging_station_idxs) == 1)
        push!(routing_model.init_constraints, constraint)
    end
    
    # All drones start with full battery
    max_battery_time = routing_model.max_battery_time
    for s in 1:n_drones
        push!(routing_model.init_constraints, @constraint(model, b[1,s] == max_battery_time - sum(a[i,1,s] for i in 1:length(GridpointsDrones))))
    end

    #Capacity of each charging stationin the beginning is at most capacity_charging
    capacity_charging = 30
    for i in 1:length(ChargingStations)
        
        constraint = @constraint(model, sum(c[i,1,s] for s in 1:n_drones) + sum(a[grid_to_idx[ChargingStations[i]],1,s] for s in 1:n_drones) <= capacity_charging)
        push!(routing_model.init_constraints, constraint)
    end
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    # println("Creating init constraints took ", t2 - t1, " seconds")
    # println("Optimizing model took ", t3 - t2, " seconds")

    # # # # # Nico's print after optimizing 
    # zeta = model[:zeta]
    # w = model[:w]

    # for i=1:length(GridpointsDrones), t=1:T
    #     println("zeta_"*string(i,t)*": ", value(zeta[i,t]))
    #     println("w_"*string(i,t,t)*": ", value(w[i,t,t]))
    #     println("w_"*string(i,t,t+1)*": ", value(w[i,t,t+1]))
    # end

    # violated_sum = 0
    # sum_violations = []
    # for i=1:length(GridpointsDrones), t=1:T, s=1:n_drones
        
    #     if abs((t-zeta[i,t,s]) - sum(w[i,t,t2,s] for t2 in 1:t)) > 1e-6
    #         violated_sum+=1
    #         push!(sum_violations, (t-zeta[i,t,s]) - sum(w[i,t,t2,s] for t2 in 1:t))
    #     end
    # end
    # println("VIOLATED SUMS: ", violated_sum)
    # println("Violations: ", sum_violations)
    # a = model[:a]
    # println("First coordinate:")
    # for t in 1:T
    #     println("--------------")
    #     for t2 in 1:t
    #         println("w: ", value(w[1,t,t2]))
    #     end
    # end
    
    # for t in 1:T
    #     println("zeta: ", value(zeta[1,t]))
    #     println("drone 1 fly: ", value(a[1,t,1]))
    #     println("drone 1 charge: ", value(a[1,t,1]))
    # end


    for s in 1:n_drones
        for i in 1:length(ChargingStations)
            if value(a[grid_to_idx[ChargingStations[i]],1,s]) >= 0.9
                # println("Drone flying positions (Julia): ", s, [GridpointsDrones[grid_to_idx[ChargingStations[i]]] for station in ChargingStations])
                # println("Drone flying index (Julia): ", [grid_to_idx[ChargingStations[i]] for station in ChargingStations])
            end
            if value(c[i,1,s]) >= 0.9
            end               
        end
    end

    # Extract results
    # println("Solver Status: ", termination_status(model))
    # println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    
    # Generate movement plan using integer indices
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    
    # Process results for fly actions
    for t in 1:reevaluation_step
        for s in 1:n_drones
            # Check fly actions
            for i in 1:length(GridpointsDrones)
                if value(a[i,t,s]) >= 0.9 1
                    movement_plan[t][s] = ("fly", GridpointsDrones[i])
                end
            end
            # Check charge actions
            for i in 1:length(ChargingStations)
                if value(c[i,t,s]) >= 0.9 1
                    movement_plan[t][s] = ("charge", ChargingStations[i])
                end
            end
        end
    end
    # println("movement_plan: ", movement_plan)
    return movement_plan[1:reevaluation_step]
end

function solve_index_next_move_routing_linear(routing_model::IndexRoutingModelLinear, reevaluation_step, drone_locations, drone_states, battery_level, offset)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    w = routing_model.w
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    risk_idx = routing_model.risk_idx
    # Clear any existing init constraints
    for con in routing_model.init_constraints
        delete(model, con)
    end
    empty!(routing_model.init_constraints)
    
    # Clear any existing next-move constraints
    for con in routing_model.next_move_constraints
        delete(model, con)
    end
    empty!(routing_model.next_move_constraints)
    
    # Add next-move specific constraints
    t1 = time_ns() / 1e9
    
    # Set drone starting positions based on previous locations
    for (s, state) in enumerate(drone_states)
        loc = drone_locations[s]  # This is a tuple (x,y)
        
        # First make sure the location is in our grid points
        if !haskey(grid_to_idx, loc)
            println("Error: Drone $s is at location $loc which is not in the grid points")
            error("Drone $s is at location $loc which is not in the grid points")
        end
        
        loc_idx = grid_to_idx[loc]
        
        if state == "charge"
            # Find which charging station index corresponds to this location
            for (i, cs) in enumerate(ChargingStations)
                if cs == loc
                    push!(routing_model.next_move_constraints, @constraint(model, c[i,1,s] == 1))
                    break
                end
            end
        elseif state == "fly"
            push!(routing_model.next_move_constraints, @constraint(model, a[loc_idx,1,s] == 1))
        end
    end
    
    # Set starting battery levels
    for s in 1:n_drones
        push!(routing_model.next_move_constraints, @constraint(model, b[1,s] == Int(battery_level[s])))
    end

    # objective with offset
    @objective(model, Min, sum(( w[i,t,t2] * risk_idx[offset+t2,i]) for t=1:T, t2=1:t,i=1:length(GridpointsDrones)))
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    # println("Creating next_move constraints took ", t2 - t1, " seconds")
    # println("Optimizing model took ", t3 - t2, " seconds")
    
    # Extract results
    # println("Solver Status: ", termination_status(model))
    # println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    
    # Generate movement plan using integer indices
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    
    # Process results for fly actions
    for t in 1:reevaluation_step
        for s in 1:n_drones
            # Check fly actions
            for i in 1:length(GridpointsDrones)
                if value(a[i,t,s]) >= 0.9
                    movement_plan[t][s] = ("fly", GridpointsDrones[i])
                end
            end
            # Check charge actions
            for i in 1:length(ChargingStations)
                if value(c[i,t,s]) >= 0.9
                    movement_plan[t][s] = ("charge", ChargingStations[i])
                end
            end
        end
    end
    # println("movement_plan: ", movement_plan)
    return movement_plan[1:reevaluation_step]
end



















# # Example parameters
# risk_pertime_file = "/Users/puech/Desktop/IPbackupcode/julia/burn_map_rescaled_53x54.npy"
# n_drones = 2
# ChargingStations = [(35,14), (30,42)]
# GroundStations = [(10,10), (20,20), (30,30), (40,40)]
# optimization_horizon = 10
# max_battery_time = 20
# reevaluation_step = 3

# # 1. Create the model
# routing_model = create_index_routing_model_linear(
#     risk_pertime_file,
#     n_drones,
#     ChargingStations,
#     GroundStations,
#     optimization_horizon,
#     max_battery_time
# )

# println("Model created successfully!")

# # 2. Solve initial routing
# init_plan = solve_index_init_routing_linear(routing_model, reevaluation_step)
# println("Initial plan:")
# println(init_plan)

# # 3. Prepare arguments for next-move routing
# # Extract drone locations and states from the last step of the initial plan
# offset = reevaluation_step
# last_step = init_plan[end]
# drone_locations = [action[2] for action in last_step]  # [(x, y), ...]
# drone_states = [action[1] for action in last_step]     # "charge" or "fly"
# # For this example, set battery_level to max for all drones
# battery_level = fill(max_battery_time, n_drones)

# # 4. Solve next move routing
# next_plan = solve_index_next_move_routing_linear(
#     routing_model,
#     reevaluation_step,
#     drone_locations,
#     drone_states,
#     battery_level,
#     offset
# )
# println("Next move plan:")
# println(next_plan)
