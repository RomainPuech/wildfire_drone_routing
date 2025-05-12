# import helper_functions such as load_burn_map
include("helper_functions.jl")
# using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ

# Uncomment to run the example
# example_routing_model_reuse()


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

function create_index_routing_model_linear(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time, objective_type="max_coverage") #min_cumulative_prob 
    t1 = time_ns() / 1e9
    risk_pertime = load_burn_map(risk_pertime_file)
    _, N, M = size(risk_pertime)
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
        zeta =  @variable(model, zeta[i=1:length(GridpointsDrones), t=0:T], Int)
        y_bar = @variable(model, y_bar[i=1:length(GridpointsDrones), t=1:T], Bin)
        w = @variable(model, w[i=1:length(GridpointsDrones), t=1:T, t2=1:t], Bin)

        @constraint(model, [t=1:T, i=1:length(GridpointsDrones)              ], y_bar[i,t] <= sum(a[i,t,s] for s=1:n_drones))
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones), s=1:n_drones], a[i,t,s]   <= y_bar[i,t] )
        
        # w = @variable(model, w[i=1:length(GridpointsDrones), t=1:T, t2=1:t, s=1:n_drones], Bin)
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, t2=1:t, s=1:n_drones], zeta[i,t,s] <= t2*w[i,t,t2,s]) # can be merged
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, t2=1:t, s=1:n_drones], t2*w[i,t,t2,s] <= t)
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], (t-zeta[i,t,s]) == sum(w[i,t,t2,s] for t2 in 1:t))


        # New version (harder constraints & not very tight)
        # t2 a_it2 <= z_it <= t, for t2<=t
        # @constraint(model, [t=1:T, i=1:length(GridpointsDrones), s=1:n_drones, t2=1:t], t >= zeta[i,t,s]                ) # Past visits

        # @constraint(model, [t=1:T, i=1:length(GridpointsDrones), s=1:n_drones, t2=1:t], zeta[i,t] >= t2*a[i,t2,s]) # Past visits # NOT NECESSARY?
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones)], zeta[i,t] >= zeta[i,t-1])
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones)], zeta[i,t] >= t*y_bar[i,t])
        # @constraint(model,        [i=1:length(GridpointsDrones)], zeta[i,1] >= 0)
        
        # @constraint(model, [t=1:T, i=1:length(not_g_ch_index), s=1:n_drones, t2=1:t], zeta[not_g_ch_index[i],t] >= t2*a[not_g_ch_index[i],t2,s]) # Past visits
        # @constraint(model, [t=1:T, i=1:length(GridpointsDrones), s=1:n_drones, t3=t:T], 0 <= zeta[i,t,s]                ) # Future visits
        # @constraint(model, [t=1:T, i=1:length(GridpointsDrones), s=1:n_drones, t3=t:T], zeta[i,t] <= t3*a[i,t3,s]) # Future visits
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones)], zeta[i,t] <= t) # Future visits
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones)], zeta[i,t] <= zeta[i,t-1]+t*y_bar[i,t]) # Future visits
        # @constraint(model,        [i=1:length(GridpointsDrones), s=1:n_drones], zeta[i,1] <= a[i,1,s]) # Future visits
        @constraint(model,        [i=1:length(GridpointsDrones)], zeta[i,0] == 0) # Future visits
            
        # @constraint(model, [t=1:T, i=1:length(not_g_ch_index), s=1:n_drones, t3=t:T], zeta[not_g_ch_index[i],t] <= t3*a[not_g_ch_index[i],t3,s]) # Future visits
        # If ground or charging, always visited at t 
        # @constraint(model, [t=1:T, i=1:length(g_ch_index), s=1:n_drones, t2=1:t], zeta[g_ch_index[i],t] == t) # Past visits


        

        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], (t-zeta[i,t,s]) == sum(w[i,t,t2,s] for t2 in 1:t)) # Exact difference
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], (1-a[i,t,s])+zeta[i,t,s]-a[i,t,s]*T  <= t*w[i,t,t,s]                  ) # border condition: t2=t when visited a_it=1. This case is already secured z_it = z_it
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones],                                         t*w[i,t,t,s] <= t*(1-a[i,t,s])) # border condition: t2=t when visited a_it=1. This case is already secured z_it = z_it
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, t2=1:(t-1), s=1:n_drones], (zeta[i,t,s]/t)+zeta[i,t,s] - T*(zeta[i,t,s]-zeta[i,t2,s]) <= t2*w[i,t,t2,s]                                    ) # z_it < t2 <= t case. Only bound if z_it = z_it2 (in between)
        # @constraint(model, [i=1:length(GridpointsDrones), t=1:T, t2=1:(t-1), s=1:n_drones],                                                 t2*w[i,t,t2,s] <= t + T*(zeta[i,t,s]-zeta[i,t2,s])) # z_it < t2 <= t case. Only bound if z_it = z_it2 (in between)
        
        @constraint(model, [t=1:T, i=1:length(GridpointsDrones), t2=1:t], (t2 - zeta[i,t]) <= t*w[i,t,t2] )
        # @constraint(model, [t=1:T, i=1:length(GridpointsDrones)        ], zeta[i,t] <= t*w[i,t,t2] )

        # @constraint(model, [t=1:T, k=1:length(ChargingStations), t2=1:t],  w[k,t,t2] == 1 ) # always visiting this ones
        @constraint(model, [t=1:T, k=1:length(ChargingStations)], zeta[grid_to_idx[ChargingStations[k]], t]==t)
        @constraint(model, [t=1:T, k in ground_idx], zeta[k, t]==t)



        # @constraint(model, )
    end



    # #Take into account ground sensors in Julia Objective function
    # if !isempty(ground_idx)
    #     @constraint(model, [t=1:T, k in ground_idx], theta[t,k] <= 0.1)
    # end

    # ground_idx_set = Set(ground_idx)  # for faster lookup

    # risk_idx = zeros(T, length(GridpointsDrones))
    # for t in 1:T
    #     for (k, point) in enumerate(GridpointsDrones)
    #         if k in ground_idx_set
    #             risk_idx[t, k] = 0.0  # no reward for drone coverage
    #         else
    #             risk_idx[t, k] = risk_pertime[t, point[1], point[2]]
    #         end
    #     end
    # end

   #once point is covered by drone, the next tau steps the risk is zero
   #MAKES IT VERY SLOW
    # tau = 2
    # @constraint(model, [k in 1:length(GridpointsDrones), t in 1:T-tau, delta in 1:tau], theta[t+delta,k] <= 1 - sum(a[k,t,s] for s in 1:n_drones))
    
    #Risk objective with integer indices
    risk_idx = zeros(T, length(GridpointsDrones))
    for t in 1:T
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
    return IndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, risk_pertime, T, n_drones, grid_to_idx, charging_map, max_battery_time)
end

function solve_index_init_routing_linear(routing_model::IndexRoutingModel, reevaluation_step)
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
    println("Optimizing model took ", t3 - t2, " seconds")

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

function solve_index_next_move_routing_linear(routing_model::IndexRoutingModel, reevaluation_step, drone_locations, drone_states, battery_level)
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
        push!(routing_model.next_move_constraints, @constraint(model, b[1,s] == Int(battery_level[s][2])))
    end
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    # println("Creating next_move constraints took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")
    
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


# MIN DETECTION time

# function create_index_routing_model_mindetection(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time)
#     t1 = time_ns() / 1e9
#     risk_pertime = load_burn_map(risk_pertime_file)
#     _, N, M = size(risk_pertime)
#     T = optimization_horizon
    
#     # Convert Python lists of tuples to Julia Vector of tuples if needed
#     ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
#     GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
#     GroundStationSet = Set(GroundStations)  # faster lookup
        
#     I = [(x, y) for x in 1:N for y in 1:M]
    
#     # Get grid points and convert from Set to Vector
#     GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
#     GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDrones_set))
#     GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
#     GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))
    
#     # precomputing the closest distance to a charging station for each gridpoint
#     precomputed_closest_distance_to_charging_station = closest_distances(ChargingStations, GridpointsDrones)
    
#     model = Model(Gurobi.Optimizer)
#     set_silent(model)
    
#     # Defining the variables using simple integers for position indices
#     # Transform grid points to integer indices
#     grid_to_idx = Dict(point => i for (i, point) in enumerate(GridpointsDrones))
#     charging_idx = [grid_to_idx[point] for point in ChargingStations]
#     ground_idx = [grid_to_idx[point] for point in intersect(GridpointsDrones,GroundStations)]
#     c_index_to_grid_idx = Dict(i => grid_to_idx[ChargingStations[i]] for i in 1:length(ChargingStations))
    
#     # Create variables with integer indices
#     a = @variable(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], Bin)
#     c = @variable(model, [i=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin)
#     b = @variable(model, [t=1:T, s=1:n_drones], Int)
#     tau = @variable(model, [i=1:length(GridpointsDrones), s = 1:n_drones], Int)
    
#     # Common constraints
#     # Each drone either charges or flies, not both
#     @constraint(model, [t=1:T, s=1:n_drones], sum(a[i,t,s] for i=1:length(GridpointsDrones)) + sum(c[i,t,s] for i=1:length(ChargingStations)) == 1)

#     # Movement constraints need to be updated to use integer indices
#     # Map each grid point to its neighbors using integer indices
#     neighbors_map = Dict()
#     for (i, point) in enumerate(GridpointsDrones)
#         neighbors_idx = [grid_to_idx[p] for p in GridpointsDrones if p in neighbors_and_point(point) && haskey(grid_to_idx, p)]
#         neighbors_map[i] = neighbors_idx
#     end
    
#     # Charging stations map
#     charging_map = Dict()
#     for (i, point) in enumerate(ChargingStations)
#         charging_map[i] = grid_to_idx[point]
#     end
    
#     for (i, point) in enumerate(ChargingStations)
#         j = grid_to_idx[point]
#         for t in 1:T-1, s in 1:n_drones
#             @constraint(model, c[i,t+1,s] + a[j,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j]) + c[i,t,s])
#             #@constraint(model, a[j,t,s] <= sum(a[k,t+1,s] for k in neighbors_map[j]) + c[i,t+1,s]) #TODO is it correct to remove this constraint?
#         end
#     end
    
#     for j_idx in 1:length(GridpointsDrones)
#         point = GridpointsDrones[j_idx]
#         if !(point in ChargingStations)  # If not a charging station
#             for t in 1:T-1, s in 1:n_drones
#                 @constraint(model, a[j_idx,t+1,s] <= sum(a[k,t,s] for k in neighbors_map[j_idx]))
#                 #@constraint(model, a[j_idx,t,s] <= sum(a[k,t+1,s] for k in neighbors_map[j_idx])) #TODO is it correct to remove this constraint?
#             end
#         end
#     end
    
#     # Min/max battery level constraints
#     @constraint(model, [t=1:T, s=1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
#     # Battery dynamics
#     # bilinear version
#     # @constraint(model, [t=1:T-1, s=1:n_drones], 
#     #     b[t+1,s] == b[t,s] - sum(a[i,t+1,s] for i=1:length(GridpointsDrones)) + 
#     #     (max_battery_time - b[t,s]) * sum(c[i,t+1,s] for i=1:length(ChargingStations)))
#     # linear version
#     @constraint(model, [s in 1:n_drones, t in 1:T], b[t,s] >= max_battery_time*sum(c[i,t,s] for i in 1:length(ChargingStations)))
#     @constraint(model, [t in 1:T-1, s in 1:n_drones], 
#         b[t+1,s] <= b[t,s] - 1 + (max_battery_time+1) * sum(c[i,t+1,s] for i in 1:length(ChargingStations)))

#     # No suicide constraint
#     # distance from closest charging station
#     @constraint(model, [s=1:n_drones, i_idx=1:length(GridpointsDrones)], 
#                 b[T,s] >= a[i_idx,T,s]*precomputed_closest_distance_to_charging_station[i_idx])

#     # Create objective variables with integer indices
#     theta = @variable(model, [t=1:T, k=1:length(GridpointsDrones)])
#     @constraint(model, [t=1:T, k=1:length(GridpointsDrones)], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones))
#     @constraint(model, [t=1:T, k=1:length(GridpointsDrones)], 0 <= theta[t,k] <= 1)
#     @constraint(model, [t=1:T, k=1:length(ChargingStations)], theta[t,grid_to_idx[ChargingStations[k]]]==0)

#     #Risk objective with integer indices
#     risk_idx = zeros(T, length(GridpointsDrones))
#     for t in 1:T
#         for (k, point) in enumerate(GridpointsDrones)
#             # if k in charging_idx
#             #     risk_idx[t, k] = 0.0  # no reward for drone coverage
#             # else
#             risk_idx[t, k] = risk_pertime[t, point[1], point[2]]
#             # end
#         end
#     end

#     #difference with previous due to minimizing detection time
#     wildfire_exp = []
#     for i in 1:length(GridpointsDrones)
#         wildfire_exp[i] = sum(t*risk_idx[t,i] for t = 1:T) 
#     end

#     @constraint(model, [i in 1:length(GridpointDrones), s in 1:n_drones], tau[i,s] >= wildfire_exp[i])

#     for i in 1:length(GridpointsDrones), s in 1:n_drones
#         lower_bound_t = max(1, ceil(wildfire_exp[i]))  # Find the minimum time to consider for detection
    
#         @constraint(model, tau[i, s] >= sum(t * a[i, t, s] for t in lower_bound_t:T))
#     end
    

#     @objective(model, Min, sum(risk_idx[t,k]*(tau[k,s] - wildfire_exp[k]) for t=1:T, k=1:length(GridpointsDrones)))

    
#     # Initialize constraint containers
#     init_constraints = ConstraintRef[]
#     next_move_constraints = ConstraintRef[]
#     t2 = time_ns() / 1e9
#     println("Model created in ", t2 - t1, " seconds")
#     println("DEBUG: Charging Stations received in Julia:")
#     println(ChargingStations)
#     return IndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
#                         GridpointsDrones, ChargingStations, risk_pertime, T, n_drones, grid_to_idx, charging_map, max_battery_time)
# end
