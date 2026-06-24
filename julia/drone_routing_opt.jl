# import helper_functions such as load_burn_map
include("helper_functions.jl")
# ] add AxisArrays
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ
using AxisArrays: AxisArray


# Index-based implementation for model reuse
# -----------------------------------------

struct IndexRoutingModel
    model::Model
    a::Array{VariableRef, 3}
    c::Array{VariableRef, 3}
    b::Array{VariableRef, 2}
    theta::Array{VariableRef, 2}
    init_constraints::Vector{ConstraintRef}
    next_move_constraints::Vector{ConstraintRef}
    GridpointsDrones::Vector{Tuple{Int,Int}}
    ChargingStations::Vector{Tuple{Int,Int}}
    GridpointsDronesDetecting::Vector{Tuple{Int,Int}}
    risk_pertime_file::String
    T::Int
    n_drones::Int
    grid_to_idx::Dict{Tuple{Int,Int}, Int}
    charging_map::Dict{Int, Int}
    max_battery_time::Int
end

function create_index_routing_model(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time)
    println("Creating index routing model")
    t1 = time_ns() / 1e9

    # Load burn map and extract dimensions
    risk_pertime = load_burn_map(risk_pertime_file)
    H, N, M = size(risk_pertime)
    T = optimization_horizon
    # println("N: ", N)
    # println("M: ", M)
    # println("T: ", T)
    if H == 1 # we duplicate the risk per time for 100 time steps
        println("Warning: risk_pertime has shape (1,N,M), we duplicate it for 100 time steps")
        risk_pertime = repeat(risk_pertime, 100, 1, 1)
        H = 100
    end
    for (x,y) in ChargingStations
        println("risk_pertime[1,x,y]: ", risk_pertime[1,x,y])
    end
    # Convert Python lists of tuples to Julia Vector of tuples if needed
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
    GroundStationSet = Set(GroundStations)  # faster lookup
        
    I = [(x, y) for x in 1:N for y in 1:M] # All feasible grid points
    
    # Get grid points and convert from Set to Vector
    GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
    GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDrones_set)) # All feasible grid points for drones
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
    GridpointsDronesDetecting_set = setdiff(GridpointsDronesDetecting_set, GroundStations) 
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set)) # All feasible grid points for drones minus the grid points in which a ground sensor or charging station is placed

    # Precomputing the closest distance to a charging station for each gridpoint
    precomputed_closest_distance_to_charging_station = closest_distances(ChargingStations, GridpointsDrones)

    
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    
    # Defining the variables using simple integers for position indices
    # Transform grid points to integer indices
    grid_to_idx = Dict(point => i for (i, point) in enumerate(GridpointsDrones))
    
    # Create variables with integer indices
    a = @variable(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], Bin) # Variable denoting if drone s flies at grid point i at time t
    c = @variable(model, [j=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin) # Variable denoting if drnoe s charges at grid point i at time t
    b = @variable(model, [t=1:T, s=1:n_drones], Int) # Variable denoting the battery of drone s at time t, defined as the # of time steps drone s can operate without recharging
    theta = @variable(model, [t=1:T, k=1:length(GridpointsDronesDetecting)], Bin) # Variable denoting if grid point k is covered by a drone at time t    

    # Constraints

    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T, s=1:n_drones], sum(a[i,t,s] for i=1:length(GridpointsDrones)) + sum(c[j,t,s] for j=1:length(ChargingStations)) == 1)
    
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
    
    # A drone can only fly or charge at location j at time t+1 if itw as charging already in the same location or the drnoe was in a neighboring location at time t
    for (j, point) in enumerate(ChargingStations)
        i = grid_to_idx[point]
        for t in 1:T-1, s in 1:n_drones
            @constraint(model, c[j,t+1,s] + a[i,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[i]) + c[j,t,s])
        end
    end
    
    for i_idx in 1:length(GridpointsDrones)
        point = GridpointsDrones[i_idx]
        if !(point in ChargingStations)  # If not a charging station
            for t in 1:T-1, s in 1:n_drones
                @constraint(model, a[i_idx,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[i_idx]))
            end
        end
    end
    
    # Min/max battery level constraints
    @constraint(model, [t=1:T, s=1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
    # Battery dynamics
    @constraint(model, [s in 1:n_drones, t in 1:T], b[t,s] >= max_battery_time*sum(c[i,t,s] for i in 1:length(ChargingStations)))
    @constraint(model, [t in 1:T-1, s in 1:n_drones], 
        b[t+1,s] <= b[t,s] - 1 + (max_battery_time+1) * sum(c[i,t+1,s] for i in 1:length(ChargingStations)))

    # No suicide constraint
    @constraint(model, [s=1:n_drones, i_idx=1:length(GridpointsDrones)], 
                b[T,s] >= a[i_idx,T,s]*precomputed_closest_distance_to_charging_station[i_idx])

    # Coverage constraints 
    @constraint(model, [t=1:T, k=1:length(GridpointsDronesDetecting), s=1:n_drones], theta[t,k] >= a[grid_to_idx[GridpointsDronesDetecting[k]],t,s]) # it's not the same k! We link with the grid coordinates.
    @constraint(model, [k=1:length(GridpointsDronesDetecting)], theta[1,k] <= sum(a[grid_to_idx[GridpointsDronesDetecting[k]],1,s] for s=1:n_drones))
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] <= sum(a[grid_to_idx[GridpointsDronesDetecting[k]],t,s] for s=1:n_drones) + theta[t-1,k])
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] >= theta[t-1,k]) 
    
    # Objective
    if length(GridpointsDronesDetecting) == 0
        @objective(model, Max, 0)
    else
        @objective(model, Max, sum([risk_pertime[1,GridpointsDronesDetecting[k]...]*(theta[1,k]) for k in 1:length(GridpointsDronesDetecting)]) + sum(risk_pertime[t,GridpointsDronesDetecting[k]...]*(theta[t,k] - theta[t-1,k]) for t in 2:T, k in 1:length(GridpointsDronesDetecting))) # plain max coverage
    end


    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]
    t2 = time_ns() / 1e9
    println("Model created in ", t2 - t1, " seconds")
    println(ChargingStations)
    return IndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, GridpointsDronesDetecting, risk_pertime_file, T, n_drones, grid_to_idx, charging_map, max_battery_time)
end

function solve_index_init_routing(routing_model::IndexRoutingModel, reevaluation_step)
    # println("Solving index init routing")
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    theta = routing_model.theta
    offset = 0
    # println("in solve_index_init_routing")
    # println(axes(b))
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    GridpointsDronesDetecting = routing_model.GridpointsDronesDetecting
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    risk_pertime = load_burn_map(routing_model.risk_pertime_file)
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
        charging_station_idxs = 1:length(ChargingStations)  # Indices into c array
        
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

    #Capacity of each charging station in the beginning is at most capacity_charging
    capacity_charging = 30
    for i in 1:length(ChargingStations)
        constraint = @constraint(model, sum(c[i,1,s] for s in 1:n_drones) + sum(a[grid_to_idx[ChargingStations[i]],1,s] for s in 1:n_drones) <= capacity_charging)
        push!(routing_model.init_constraints, constraint)
    end
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        denom = max(abs(obj_val), 1e-10)
        gap_pct = 100.0 * (obj_bound - obj_val) / denom
        println("Gurobi optimality gap: ", round(gap_pct, digits=4), "% (status: ", termination_status(model), ")")
    end
    # check if the model has a solution (accept OPTIMAL or TIME_LIMIT with incumbent)
    if termination_status(model) != MOI.OPTIMAL && termination_status(model) != MOI.TIME_LIMIT
        println("No solution found")
        println("Termination status: ", termination_status(model))
        # print the input parameters
        println("Input parameters:")
        println("Charging Stations: ", ChargingStations)
        println("T: ", T)
        println("n_drones: ", n_drones)
        println("max_battery_time: ", max_battery_time)
        
        return
    end
    if !has_values(model)
        return
    end

    for s in 1:n_drones
        for i in 1:length(ChargingStations)
            if value(a[grid_to_idx[ChargingStations[i]],1,s]) >= 0.9
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

    # #print objective value
    # println("Objective value: ", objective_value(model))
    # println("b: ", value.(b))
    # println("movement_plan: ", movement_plan)

    # println("Objective value: ", objective_value(model), " because: ")

    # for k in 1:length(GridpointsDronesDetecting)
    #     if value(theta[1,k]) >= 0.9
    #         println("Theta variable: ", 1, " ", GridpointsDronesDetecting[k], " ", value(theta[1,k]), " and risk_pertime: ", risk_pertime[1+offset, GridpointsDronesDetecting[k]...])
    #     end
    #     if value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,1]) >= 0.9
    #         println("A variable 1: ", 1, " ", GridpointsDronesDetecting[grid_to_idx[GridpointsDronesDetecting[k]]], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,1]))
    #     end
    #     if value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,2]) >= 0.9
    #         println("A variable 2: ", 1, " ", GridpointsDronesDetecting[grid_to_idx[GridpointsDronesDetecting[k]]], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,2]))
    #     end
    # end
    # for t in 2:T
    #     for k in 1:length(GridpointsDronesDetecting)
    #         if value(theta[t,k])-value(theta[t-1,k]) >= 0.9
    #             println("Theta variable: ", t, " ", GridpointsDronesDetecting[k], " ", value(theta[t,k]) - value(theta[t-1,k]), " and risk_pertime: ", risk_pertime[t+offset, GridpointsDronesDetecting[k]...])
    #         end
    #         if value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,1]) >= 0.9
    #             println("A variable 1: ", t, " ", GridpointsDronesDetecting[k], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,1]))
    #         end
    #         if value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,2]) >= 0.9
    #             println("A variable 2: ", t, " ", GridpointsDronesDetecting[k], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,2]))
    #         end
    #     end
    # end

    return movement_plan[1:reevaluation_step]
end

function solve_index_next_move_routing(routing_model::IndexRoutingModel, reevaluation_step, drone_locations, drone_states, battery_level, offset=0)
    # println("Solving index next move routing")
    # println("Reevaluation step: ", reevaluation_step)
    # println("Drone locations: ", drone_locations)
    # println("Drone states: ", drone_states)
    # println("Battery level: ", battery_level)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    theta = routing_model.theta
    # println("in solve_index_next_move_routing")
    # println(axes(b))
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    risk_pertime = load_burn_map(routing_model.risk_pertime_file)
    GridpointsDronesDetecting = routing_model.GridpointsDronesDetecting
    
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
        if drone_states[s] != "charge"
            push!(routing_model.next_move_constraints, @constraint(model, b[1,s] == Int(battery_level[s]))) # or full if you are currently charging
        else
            push!(routing_model.next_move_constraints, @constraint(model, b[1,s] == routing_model.max_battery_time))
        end
    end

    # Update objective with offset
    # Check that offset doesn't exceed the time horizon of risk_pertime
    H = size(risk_pertime, 1)
    if T + offset > H
        println("Warning: T + offset = $(T + offset) exceeds risk_pertime time horizon H = $H. Adjusting T to $(H - offset)")
        T = H - offset
        if offset +1 > H
            println("Warning: offset = $offset exceeds risk_pertime time horizon H = $H. Adjusting offset to $(H)")
            offset = H - 1
        end
    end
    
    @objective(model, Max, 
        sum([risk_pertime[1+offset, GridpointsDronesDetecting[k]...]*(theta[1,k]) for k in 1:length(GridpointsDronesDetecting)]) + 
        sum(risk_pertime[t+offset, GridpointsDronesDetecting[k]...]*(theta[t,k] - theta[t-1,k]) for t in 2:T, k in 1:length(GridpointsDronesDetecting))
    )
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    if has_values(model)
        obj_val = objective_value(model)
        obj_bound = objective_bound(model)
        denom = max(abs(obj_val), 1e-10)
        gap_pct = 100.0 * (obj_bound - obj_val) / denom
        println("Gurobi optimality gap: ", round(gap_pct, digits=4), "% (status: ", termination_status(model), ")")
    end

    #println("Creating next_move constraints took ", t2 - t1, " seconds")
    #println("Optimizing model took ", t3 - t2, " seconds")
    
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
    # print the battery variable
    #println("Battery variable: ", value.(b))
    #println("movement_plan: ", movement_plan)
    # print theta variable as an array
    # println("Objective value: ", objective_value(model), " because: ")
    # println("Max obj val possible at 1: ", sum(risk_pertime[1+offset, GridpointsDronesDetecting[k]...] for k in 1:length(GridpointsDronesDetecting)))

    # for k in 1:length(GridpointsDronesDetecting)
    #     if value(theta[1,k]) >= 0.9
    #         println("Theta variable: ", 1, " ", GridpointsDronesDetecting[k], " ", value(theta[1,k]), " and risk_pertime: ", risk_pertime[1+offset, GridpointsDronesDetecting[k]...])
    #     end
    #     if value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,1]) >= 0.9
    #         println("A variable 1: ", 1, " ", GridpointsDronesDetecting[grid_to_idx[GridpointsDronesDetecting[k]]], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,1]))
    #     end
    #     if value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,2]) >= 0.9
    #         println("A variable 2: ", 1, " ", GridpointsDronesDetecting[grid_to_idx[GridpointsDronesDetecting[k]]], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],1,2]))
    #     end
    # end
    # for t in 2:T
    #     for k in 1:length(GridpointsDronesDetecting)
    #         if value(theta[t,k])-value(theta[t-1,k]) >= 0.9
    #             println("Theta variable: ", t, " ", GridpointsDronesDetecting[k], " ", value(theta[t,k]) - value(theta[t-1,k]), " and risk_pertime: ", risk_pertime[t+offset, GridpointsDronesDetecting[k]...])
    #         end
    #         if value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,1]) >= 0.9
    #             println("A variable: ", t, " ", GridpointsDronesDetecting[k], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,1]))
    #         end
    #         if value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,2]) >= 0.9
    #             println("A variable 2: ", t, " ", GridpointsDronesDetecting[k], " ", value(a[grid_to_idx[GridpointsDronesDetecting[k]],t,2]))
    #         end
    #     end
    # end
    return movement_plan[1:reevaluation_step]
end

##### MASKED VERSION #####

struct IndexRoutingModel
    model::Model
    a::Array{VariableRef, 3}
    c::Array{VariableRef, 3}
    b::Array{VariableRef, 2}
    theta::Array{VariableRef, 2}
    init_constraints::Vector{ConstraintRef}
    next_move_constraints::Vector{ConstraintRef}
    GridpointsDrones::Vector{Tuple{Int,Int}}
    ChargingStations::Vector{Tuple{Int,Int}}
    GridpointsDronesDetecting::Vector{Tuple{Int,Int}}
    risk_pertime_file::String
    T::Int
    n_drones::Int
    grid_to_idx::Dict{Tuple{Int,Int}, Int}
    charging_map::Dict{Int, Int}
    max_battery_time::Int
end

function create_index_routing_model_masked(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time, mask_filename)
    println("Creating index routing model (masked)")
    t1 = time_ns() / 1e9

    # Load burn map and extract dimensions
    risk_pertime = load_burn_map(risk_pertime_file)
    H, N, M = size(risk_pertime)
    T = optimization_horizon
    # println("N: ", N)
    # println("M: ", M)
    # println("T: ", T)
    if H == 1 # we duplicate the risk per time for 100 time steps
        println("Warning: risk_pertime has shape (1,N,M), we duplicate it for 100 time steps")
        risk_pertime = repeat(risk_pertime, 100, 1, 1)
        H = 100
    end
    for (x,y) in ChargingStations
        println("risk_pertime[1,x,y]: ", risk_pertime[1,x,y])
    end
    # Convert Python lists of tuples to Julia Vector of tuples if needed
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
    GroundStationSet = Set(GroundStations)  # faster lookup

    if mask_filename != nothing
        mask = load_mask(mask_filename)
    else
        mask = ones(N, M)
    end
        
    I = [(x, y) for x in 1:N for y in 1:M if mask[x,y] == 1] # All feasible grid points
    
    # Get grid points and convert from Set to Vector
    # Use BFS-based get_drone_gridpoints that accounts for blocked cells
    GridpointsDrones_set, distance_dict = get_drone_gridpoints_BFS(ChargingStations, floor(max_battery_time/2), I, N, M)
    # print max value of distance_dict
    println("max value of distance_dict: ", maximum(values(distance_dict)))
    GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDrones_set)) # All feasible grid points for drones
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
    GridpointsDronesDetecting_set = setdiff(GridpointsDronesDetecting_set, GroundStations) 
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set)) # All feasible grid points for drones minus the grid points in which a ground sensor or charging station is placed

    # Precomputing the closest distance to a charging station for each gridpoint
    # Use the distances computed during BFS instead of recomputing
    precomputed_closest_distance_to_charging_station = [distance_dict[point] for point in GridpointsDrones]


    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 120.0)  # 2 minutes for drone routing (max coverage masked)
    println("Gurobi time limit set to 120 seconds for index routing (masked)")

    # Defining the variables using simple integers for position indices
    # Transform grid points to integer indices
    grid_to_idx = Dict(point => i for (i, point) in enumerate(GridpointsDrones))
    
    # Create variables with integer indices
    a = @variable(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], Bin) # Variable denoting if drone s flies at grid point i at time t
    c = @variable(model, [j=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin) # Variable denoting if drnoe s charges at grid point i at time t
    b = @variable(model, [t=1:T, s=1:n_drones], Int) # Variable denoting the battery of drone s at time t, defined as the # of time steps drone s can operate without recharging
    theta = @variable(model, [t=1:T, k=1:length(GridpointsDronesDetecting)], Bin) # Variable denoting if grid point k is covered by a drone at time t    

    # Constraints

    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T, s=1:n_drones], sum(a[i,t,s] for i=1:length(GridpointsDrones)) + sum(c[j,t,s] for j=1:length(ChargingStations)) == 1)
    
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
    
    # A drone can only fly or charge at location j at time t+1 if itw as charging already in the same location or the drnoe was in a neighboring location at time t
    for (j, point) in enumerate(ChargingStations)
        i = grid_to_idx[point]
        for t in 1:T-1, s in 1:n_drones
            @constraint(model, c[j,t+1,s] + a[i,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[i]) + c[j,t,s])
        end
    end
    
    for i_idx in 1:length(GridpointsDrones)
        point = GridpointsDrones[i_idx]
        if !(point in ChargingStations)  # If not a charging station
            for t in 1:T-1, s in 1:n_drones
                @constraint(model, a[i_idx,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[i_idx]))
            end
        end
    end
    
    # Min/max battery level constraints
    @constraint(model, [t=1:T, s=1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
    # Battery dynamics
    @constraint(model, [s in 1:n_drones, t in 1:T], b[t,s] >= max_battery_time*sum(c[i,t,s] for i in 1:length(ChargingStations)))
    @constraint(model, [t in 1:T-1, s in 1:n_drones], 
        b[t+1,s] <= b[t,s] - 1 + (max_battery_time+1) * sum(c[i,t+1,s] for i in 1:length(ChargingStations)))

    # No suicide constraint
    @constraint(model, [s=1:n_drones, i_idx=1:length(GridpointsDrones)], 
                b[T,s] >= a[i_idx,T,s]*precomputed_closest_distance_to_charging_station[i_idx])

    # Coverage constraints 
    @constraint(model, [t=1:T, k=1:length(GridpointsDronesDetecting), s=1:n_drones], theta[t,k] >= a[grid_to_idx[GridpointsDronesDetecting[k]],t,s]) # it's not the same k! We link with the grid coordinates.
    @constraint(model, [k=1:length(GridpointsDronesDetecting)], theta[1,k] <= sum(a[grid_to_idx[GridpointsDronesDetecting[k]],1,s] for s=1:n_drones))
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] <= sum(a[grid_to_idx[GridpointsDronesDetecting[k]],t,s] for s=1:n_drones) + theta[t-1,k])
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] >= theta[t-1,k]) 
    
    # Objective
    if length(GridpointsDronesDetecting) == 0
        @objective(model, Max, 0)
    else
        @objective(model, Max, sum([risk_pertime[1,GridpointsDronesDetecting[k]...]*(theta[1,k]) for k in 1:length(GridpointsDronesDetecting)]) + sum(risk_pertime[t,GridpointsDronesDetecting[k]...]*(theta[t,k] - theta[t-1,k]) for t in 2:T, k in 1:length(GridpointsDronesDetecting))) # plain max coverage
    end


    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]
    t2 = time_ns() / 1e9
    println("Model created in ", t2 - t1, " seconds")
    println(ChargingStations)
    return IndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, GridpointsDronesDetecting, risk_pertime_file, T, n_drones, grid_to_idx, charging_map, max_battery_time)
end
