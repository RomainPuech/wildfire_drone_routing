# import helper_functions such as load_burn_map
include("helper_functions.jl")
# using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ


function drone_routing_next_move_example(drones, batteries, risk_pertime_file, time_horizon)
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    
    # Generate random moves for each drone
    # output should have this format: [("move", (dx, dy)), ("move", (dx, dy)), ...]
    return [[("move", (rand(-5:5), rand(-5:5))) for _ in 1:length(drones)] for _ in 1:time_horizon]
end

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
    risk_pertime::Array{Float64, 3}
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
    println("risk_pertime: ", risk_pertime[1,1,1])
    H, N, M = size(risk_pertime)
    T = optimization_horizon
    println("N: ", N)
    println("M: ", M)
    println("T: ", T)
    if H == 1 # we duplicate the risk per time for 100 time steps
        println("Duplicating risk per time for 100 time steps")
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
    c = @variable(model, [i=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin) # Variable denoting if drnoe s charges at grid point i at time t
    b = @variable(model, [t=1:T, s=1:n_drones], Int) # Variable denoting the battery of drone s at time t, defined as the # of time steps drone s can operate without recharging
    theta = @variable(model, [t=1:T, k=1:length(GridpointsDronesDetecting)], Bin) # Variable denoting if grid point k is covered by a drone at time t    

    # Constraints

    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T, s=1:n_drones], sum(a[i,t,s] for i=1:length(GridpointsDrones)) + sum(c[i,t,s] for i=1:length(ChargingStations)) == 1)
    
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
    for (i, point) in enumerate(ChargingStations)
        j = grid_to_idx[point]
        for t in 1:T-1, s in 1:n_drones
            @constraint(model, c[i,t+1,s] + a[j,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j]) + c[i,t,s])
        end
    end
    
    for j_idx in 1:length(GridpointsDrones)
        point = GridpointsDrones[j_idx]
        if !(point in ChargingStations)  # If not a charging station
            for t in 1:T-1, s in 1:n_drones
                @constraint(model, a[j_idx,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j_idx]))
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
    @constraint(model, [t=1:T, k=1:length(GridpointsDronesDetecting), s=1:n_drones], theta[t,k] >= a[k,t,s])
    @constraint(model, [k=1:length(GridpointsDronesDetecting)], theta[1,k] <= sum(a[k,1,s] for s=1:n_drones))
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones) + theta[t-1,k])
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] >= theta[t-1,k]) 
    
    # Objective
    @objective(model, Max, sum([risk_pertime[1,GridpointsDronesDetecting[k]...]*(theta[1,k]) for k in 1:length(GridpointsDronesDetecting)]) + sum(risk_pertime[t,GridpointsDronesDetecting[k]...]*(theta[t,k] - theta[t-1,k]) for t in 2:T, k in 1:length(GridpointsDronesDetecting))) # plain max coverage

######
    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]
    t2 = time_ns() / 1e9
    println("Model created in ", t2 - t1, " seconds")
    println(ChargingStations)
    return IndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, risk_pertime, T, n_drones, grid_to_idx, charging_map, max_battery_time)
end

function solve_index_init_routing(routing_model::IndexRoutingModel, reevaluation_step)
    # println("Solving index init routing")
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
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
    # check if the model has a solution
    if termination_status(model) != MOI.OPTIMAL
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

    #print objective value
    println("Objective value: ", objective_value(model))

    return movement_plan[1:reevaluation_step]
end

function solve_index_next_move_routing(routing_model::IndexRoutingModel, reevaluation_step, drone_locations, drone_states, battery_level)
    # println("Solving index next move routing")
    # println("Reevaluation step: ", reevaluation_step)
    # println("Drone locations: ", drone_locations)
    # println("Drone states: ", drone_states)
    # println("Battery level: ", battery_level)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    
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
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
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
    #println("movement_plan: ", movement_plan[1:reevaluation_step])
    return movement_plan[1:reevaluation_step]
end

# ------------------ MAX COVERAGE WITH REGULARIZATION ------------------

struct RegularizedIndexRoutingModel
    model::Model
    a::Array{VariableRef, 3}
    c::Array{VariableRef, 3}
    b::Array{VariableRef, 2}
    theta::Array{VariableRef, 2}
    init_constraints::Vector{ConstraintRef}
    next_move_constraints::Vector{ConstraintRef}
    GridpointsDrones::Vector{Tuple{Int,Int}}
    ChargingStations::Vector{Tuple{Int,Int}}
    risk_pertime::Array{Float64, 3}
    T::Int
    n_drones::Int
    grid_to_idx::Dict{Tuple{Int,Int}, Int}
    charging_map::Dict{Int, Int}
    max_battery_time::Int
    regularization_param::Float64
end

function create_regularized_index_routing_model(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time, regularization_param)
    println("Creating regularized index routing model")
    t1 = time_ns() / 1e9
    risk_pertime = load_burn_map(risk_pertime_file)
    H, N, M = size(risk_pertime)
    T = optimization_horizon
    println("N: ", N)
    println("M: ", M)
    println("T: ", T)
    if H == 1 # we duplicate the risk per time for 100 time steps
        risk_pertime = repeat(risk_pertime, 100, 1, 1)
        H = 100
    end
    
    # Convert Python lists of tuples to Julia Vector of tuples if needed
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
    
    I = [(x, y) for x in 1:N for y in 1:M] # All grid points
    
    # Get grid points and convert from Set to Vector
    GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
    GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDrones_set))
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
    GridpointsDronesDetecting_set = setdiff(GridpointsDronesDetecting_set, GroundStations)
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))
    
    # precomputing the closest distance to a charging station for each gridpoint
    precomputed_closest_distance_to_charging_station = closest_distances(ChargingStations, GridpointsDrones)
    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Defining the variables using simple integers for position indices
    # Transform grid points to integer indices
    grid_to_idx = Dict(point => i for (i, point) in enumerate(GridpointsDrones))
    
    # Create variables with integer indices
    a = @variable(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], Bin) # Variable denoting if drone s flies at grid point i at time t
    c = @variable(model, [i=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin) # Variable denoting if drnoe s charges at grid point i at time t
    b = @variable(model, [t=1:T, s=1:n_drones], Int) # Variable denoting the battery of drone s at time t, defined as the # of time steps drone s can operate without recharging
    theta = @variable(model, [t=1:T, k=1:length(GridpointsDronesDetecting)], Bin) # Variable denoting if grid point k is covered by a drone at time t    
  
    ### pariwise distances

    # Coordinates of each object
    @variable(model, xa[1:n_drones, 1:T], Int)
    @variable(model, ya[1:n_drones, 1:T], Int)
    @variable(model, xc[1:n_drones, 1:T], Int)
    @variable(model, yc[1:n_drones, 1:T], Int)
    @variable(model, x[1:n_drones, 1:T], Int)
    @variable(model, y[1:n_drones, 1:T], Int)

    # Link position variables to coordinates
    @constraint(model, [s=1:n_drones, t=1:T], xa[s,t] == sum(GridpointsDrones[i][1] * a[i,t,s] for i in 1:length(GridpointsDrones)))
    @constraint(model, [s=1:n_drones, t=1:T], ya[s,t] == sum(GridpointsDrones[i][2] * a[i,t,s] for i in 1:length(GridpointsDrones)))
    @constraint(model, [s=1:n_drones, t=1:T], xc[s,t] == sum(ChargingStations[i][1] * c[i,t,s] for i in 1:length(ChargingStations)))
    @constraint(model, [s=1:n_drones, t=1:T], yc[s,t] == sum(ChargingStations[i][2] * c[i,t,s] for i in 1:length(ChargingStations)))
    @constraint(model, [s=1:n_drones, t=1:T], x[s,t] == xa[s,t] + xc[s,t])
    @constraint(model, [s=1:n_drones, t=1:T], y[s,t] == ya[s,t] + yc[s,t]) # because one of the 2 is 0

    M_x = N - 1
    M_y = M - 1
    M_inf = max(M_x, M_y) # For the L∞ "max(dX, dY)" constraints

    # zX, zY ∈ {0,1} for each pair (s1<s2) and time t
    @variable(model, zX[1:n_drones, 1:n_drones, 1:T], Bin)
    @variable(model, zY[1:n_drones, 1:n_drones, 1:T], Bin)

    # dX, dY ≥ 0 track the absolute differences
    @variable(model, dX[1:n_drones, 1:n_drones, 1:T] >= 0)
    @variable(model, dY[1:n_drones, 1:n_drones, 1:T] >= 0)

    # For each s1 < s2, we do constraints so dX = | x[s1,t] - x[s2,t] |.
    # Because x[...] are Int ∈ [1..N], we can do big-M with M_x = N-1.
    #  dX ≥  x[s1,t] - x[s2,t]
    #  dX ≤  x[s1,t] - x[s2,t] + M_x*(1 - zX)
    #  dX ≥  x[s2,t] - x[s1,t]
    #  dX ≤  x[s2,t] - x[s1,t] + M_x*zX
    # Similarly for dY using zY and M_y.

    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones], 
        dX[s1,s2,t] ≥ x[s1,t] - x[s2,t])
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones],
        dX[s1,s2,t] ≤ (x[s1,t] - x[s2,t]) + M_x*(1 - zX[s1,s2,t]))

    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones], 
        dX[s1,s2,t] ≥ x[s2,t] - x[s1,t])
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones],
        dX[s1,s2,t] ≤ (x[s2,t] - x[s1,t]) + M_x*zX[s1,s2,t])

    # Repeat the pattern for dY:
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones], 
        dY[s1,s2,t] ≥ y[s1,t] - y[s2,t])
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones],
        dY[s1,s2,t] ≤ (y[s1,t] - y[s2,t]) + M_y*(1 - zY[s1,s2,t]))

    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones], 
        dY[s1,s2,t] ≥ y[s2,t] - y[s1,t])
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones],
        dY[s1,s2,t] ≤ (y[s2,t] - y[s1,t]) + M_y*zY[s1,s2,t])

    @variable(model, dInf[1:n_drones, 1:n_drones, 1:T] >= 0)
    @variable(model, zInf[1:n_drones, 1:n_drones, 1:T], Bin)
    
    # We want: dInf = max(dX, dY). The constraints:
    #   dInf >= dX
    #   dInf >= dY
    #   dInf ≤ dX + M_inf * (1 - zInf)
    #   dInf ≤ dY + M_inf * zInf
    # If zInf=0 => dInf ≤ dX        (implies dX >= dY)
    # If zInf=1 => dInf ≤ dY        (implies dY >= dX)
    
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones], dInf[s1,s2,t] ≥ dX[s1,s2,t])
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones], dInf[s1,s2,t] ≥ dY[s1,s2,t])
    
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones],
        dInf[s1,s2,t] ≤ dX[s1,s2,t] + M_inf*(1 - zInf[s1,s2,t]))
    @constraint(model, [t=1:T, s1=1:n_drones, s2=s1+1:n_drones],
        dInf[s1,s2,t] ≤ dY[s1,s2,t] + M_inf*zInf[s1,s2,t])
        
     @expression(model, total_linf_dist,
        sum(dInf[s1,s2,t] for s1 in 1:n_drones for s2 in s1+1:n_drones for t in 1:T)
    )
    

    # Common constraints
    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T, s=1:n_drones], sum(a[i,t,s] for i=1:length(GridpointsDrones)) + sum(c[i,t,s] for i=1:length(ChargingStations)) == 1)

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
    for (i, point) in enumerate(ChargingStations)
        j = grid_to_idx[point]
        for t in 1:T-1, s in 1:n_drones
            @constraint(model, c[i,t+1,s] + a[j,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j]) + c[i,t,s])
        end
    end
    
    for j_idx in 1:length(GridpointsDrones)
        point = GridpointsDrones[j_idx]
        if !(point in ChargingStations)  # If not a charging station
            for t in 1:T-1, s in 1:n_drones
                @constraint(model, a[j_idx,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j_idx]))
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
    @constraint(model, [t=1:T, k=1:length(GridpointsDronesDetecting), s=1:n_drones], theta[t,k] >= a[k,t,s])
    @constraint(model, [k=1:length(GridpointsDronesDetecting)], theta[1,k] <= sum(a[k,1,s] for s=1:n_drones))
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones) + theta[t-1,k])
    @constraint(model, [t=2:T, k=1:length(GridpointsDronesDetecting)], theta[t,k] >= theta[t-1,k]) 

    @objective(model, Max, sum([risk_pertime[1,GridpointsDronesDetecting[k]...]*(theta[1,k]) for k in 1:length(GridpointsDronesDetecting)]) + sum(risk_pertime[t,GridpointsDronesDetecting[k]...]*(theta[t,k] - theta[t-1,k]) for t in 2:T, k in 1:length(GridpointsDronesDetecting)) + regularization_param*total_linf_dist ) # plain max coverage

    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]
    t2 = time_ns() / 1e9
    println("Model created in ", t2 - t1, " seconds")
    # println("DEBUG: Charging Stations received in Julia:")
    println(ChargingStations)
    return RegularizedIndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, risk_pertime, T, n_drones, grid_to_idx, charging_map, max_battery_time, regularization_param)
end

function solve_regularized_index_init_routing(routing_model::RegularizedIndexRoutingModel, reevaluation_step)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    regularization_param = routing_model.regularization_param
    
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
    set_optimizer_attribute(model, "OutputFlag", 1)
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    # println("Creating init constraints took ", t2 - t1, " seconds")
    # println("Optimizing model took ", t3 - t2, " seconds")

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

function solve_regularized_index_next_move_routing(routing_model::RegularizedIndexRoutingModel, reevaluation_step, drone_locations, drone_states, battery_level)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
    grid_to_idx = routing_model.grid_to_idx
    T = routing_model.T
    n_drones = routing_model.n_drones
    
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