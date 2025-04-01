# import helper_functions such as load_burn_map
include("helper_functions.jl")
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ


function drone_routing_next_move_example(drones, batteries, risk_pertime_file, time_horizon)
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    
    # Generate random moves for each drone
    # output should have this format: [("move", (dx, dy)), ("move", (dx, dy)), ...]
    return [[("move", (rand(-5:5), rand(-5:5))) for _ in 1:length(drones)] for _ in 1:time_horizon]
end


# ---------------------------------------------------------------------------------------------------------------------------------------
function NEW_ROUTING_STRATEGY_INIT(risk_pertime_file,n_drones,ChargingStations,GroundStations,optimization_horizon,max_battery_time,reevaluation_step) #T here is optimization_horizon: should be > max_battery_time, reevaluation_step < floor(optimization_horizon/2)

    risk_pertime = load_burn_map(risk_pertime_file)
    _, N, M = size(risk_pertime)
    T = optimization_horizon
    t1 = time_ns() / 1e9 

    detection_rate = 0.7
    I_prime = nothing
    I_second = nothing
    I_third = nothing
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
    B_max = 1
    B_min = 0.2

    GridpointsDrones = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I) # = allowed_gridpoints_drones
    GridpointsDronesDetecting = setdiff(GridpointsDrones,ChargingStations) # = NonChargingstations_allowed


    model = Model(Gurobi.Optimizer)
    set_silent(model)
    #Defining the variables
    a = @variable(model, [i in GridpointsDrones, t in 1:T, s in 1:n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t in 1:T, s in 1:n_drones], Bin)
    b = @variable(model, [t in 1:T, s in 1:n_drones])
    
    #DEFINING THE CONSTRAINTS 
    #No 2 drones flying in the same place at the same time
    #@constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1) 
    #Each drone either charges or flies, not both
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    #Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    #Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t+1,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    #Min/max battery level constraints
    @constraint(model, [t in 1:T, s in 1:n_drones], B_min <= b[t,s] <= B_max)
    #Battery level at t+1 is less than battery level at t - 0.2 if drone flies at t + B_max if drone charges at t 
    @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - (1-B_min)/max_battery_time*sum(a[i,t,s] for i in GridpointsDrones) + B_max*sum(c[i,t,s] for i in ChargingStations))
    #Drones need to charge if battery level falls below B_min
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(c[i,t,s] for i in ChargingStations) >= 1 - b[t,s]/B_min)
    #All drones start to fly from a charging station at t=1
    @constraint(model, [s in 1:n_drones], sum(c[i,1,s] for i in ChargingStations) + sum(a[i,1,s] for i in ChargingStations) == 1) #First run
    #@constraint(model, [s in 1:n_drones], sum(a[i,1,s] for i in GridpointsDrones) == 1) #First run
    # @constraint(model, [s in 1:n_drones], sum(a[i,2,s] for i in GridpointsDrones) == 1) #First run

    #All drones start with full battery at t=1
    @constraint(model, [s in 1:n_drones], b[1,s] == B_max) #First run

    @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.00001*sum(b[t,s] for t in 1:T, s in 1:n_drones))
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    println("Creating model took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")

    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    selected_fly_indices = [(i,t,s) for i in GridpointsDrones, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - t1, " seconds")

    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("fly", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end

    updated_movement_plan = movement_plan[1:reevaluation_step]

    return updated_movement_plan

end

## new battery Definition

function NEW_ROUTING_STRATEGY_INIT_INTEGER_BATTERY(risk_pertime_file,n_drones,ChargingStations,GroundStations,optimization_horizon,max_battery_time,reevaluation_step) #T here is optimization_horizon: should be > max_battery_time, reevaluation_step < floor(optimization_horizon/2)

    risk_pertime = load_burn_map(risk_pertime_file)
    _, N, M = size(risk_pertime)
    T = optimization_horizon
    t1 = time_ns() / 1e9 

    detection_rate = 0.7
    I_prime = nothing
    I_second = nothing
    I_third = nothing
    I = [(x, y) for x in 1:N for y in 1:M]

    # Convert Python lists of tuples to Julia Vector of tuples
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]

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

    GridpointsDrones = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I) # = allowed_gridpoints_drones
    GridpointsDronesDetecting = setdiff(GridpointsDrones,ChargingStations) # = NonChargingstations_allowed


    model = Model(Gurobi.Optimizer)
    set_silent(model)
    #Defining the variables
    a = @variable(model, [i in GridpointsDrones, t in 1:T, s in 1:n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t in 1:T, s in 1:n_drones], Bin)
    b = @variable(model, [t in 1:T, s in 1:n_drones], Int)
    
    #DEFINING THE CONSTRAINTS 
    #No 2 drones flying in the same place at the same time -> why?
    #@constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1) 
    #Each drone either charges or flies, not both
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    #Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    #Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1 #TODO is it correct to remove these 2 constraints?
    # @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t+1,s])
    # @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    
    # Battery level constraints: Integer between 0 and max_battery_time
    @constraint(model, [t in 1:T, s in 1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
    # Battery dynamics: decreases by 1 when flying, resets to max when charging
    @constraint(model, [t in 1:T-1, s in 1:n_drones], 
        b[t+1,s] == b[t,s] - sum(a[i,t,s] for i in GridpointsDrones) + 
        (max_battery_time - b[t,s]) * sum(c[i,t,s] for i in ChargingStations))

    # @constraint(model, [t in 1:T-1, s in 1:n_drones], 
    #     b[t+1,s] <= b[t,s] - sum(a[i,t,s] for i in GridpointsDrones) + 
    #     max_battery_time * sum(c[i,t,s] for i in ChargingStations))
    
    ########## Constraints specific to the init problem
    #All drones start to fly from a charging station at t=1
    @constraint(model, [s in 1:n_drones], sum(c[i,1,s] for i in ChargingStations) + sum(a[i,1,s] for i in ChargingStations) == 1) #First run
    #All drones start with full battery at t=1
    @constraint(model, [s in 1:n_drones], b[1,s] == max_battery_time) #First run
    ########## End of constraints specific to the init problem

    # @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones))
    # objective that doesn't count twice the risk of the same cell if multiple drones fly there
    @variable(model, theta[t in 1:T, k in GridpointsDrones])
    @constraint(model, [t in 1:T, k in GridpointsDrones], theta[t,k] <= sum(a[k,t,s] for s in 1:n_drones))
    @constraint(model, [t in 1:T, k in GridpointsDrones], 0 <= theta[t,k] <= 1)
    @objective(model, Max, sum(risk_pertime[t,k...]*theta[t,k] for k in GridpointsDrones, t in 1:T))

    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    println("Creating model took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")

    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    selected_fly_indices = [(i,t,s) for i in GridpointsDrones, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - t1, " seconds")

    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("fly", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end

    updated_movement_plan = movement_plan[1:reevaluation_step]

    return updated_movement_plan

end



#######








function NEW_ROUTING_STRATEGY_NEXTMOVE(risk_pertime_file,n_drones,ChargingStations,GroundStations,optimization_horizon,max_battery_time,reevaluation_step,drone_locations,drone_states,battery_level) #T here is optimization_horizon: should be > max_battery_time, reevaluation_step < floor(optimization_horizon/2)

    risk_pertime = load_burn_map(risk_pertime_file)
    _, N, M = size(risk_pertime)
    T = optimization_horizon
    t1 = time_ns() / 1e9 

    detection_rate = 0.7
    I_prime = nothing
    I_second = nothing
    I_third = nothing
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
    B_max = 1
    B_min = 0.2

    GridpointsDrones = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I) # = allowed_gridpoints_drones
    GridpointsDronesDetecting = setdiff(GridpointsDrones,ChargingStations) # = NonChargingstations_allowed


    model = Model(Gurobi.Optimizer)
    set_silent(model)
    #Defining the variables
    a = @variable(model, [i in GridpointsDrones, t in 1:T, s in 1:n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t in 1:T, s in 1:n_drones], Bin)
    b = @variable(model, [t in 1:T, s in 1:n_drones])
    
    #DEFINING THE CONSTRAINTS 
    #No 2 drones flying in the same place at the same time
    #@constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1) 
    #Each drone either charges or flies, not both
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    #Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    #Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t+1,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    #Min/max battery level constraints
    @constraint(model, [t in 1:T, s in 1:n_drones], B_min <= b[t,s] <= B_max)
    #Battery level at t+1 is less than battery level at t - 0.2 if drone flies at t + B_max if drone charges at t 
    @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - (1-B_min)/max_battery_time*sum(a[i,t,s] for i in GridpointsDrones) + B_max*sum(c[i,t,s] for i in ChargingStations))
    #Drones need to charge if battery level falls below B_min
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(c[i,t,s] for i in ChargingStations) >= 1 - b[t,s]/B_min)

    ## Constraints specific to the the next move
    #All drones start from location of previous drone
    for (s, state) in enumerate(drone_states)
        loc = drone_locations[s]  # This is a tuple (x,y)
        if state == "charge"
            @constraint(model, c[loc,1,s] == 1)
        elseif state == "fly"
            @constraint(model, a[loc,1,s] == 1)
        end
    end
    #All drones start with battery level as given in input
    @constraint(model, [s in 1:n_drones], b[1,s] == battery_level[s][2]) #First run. [1] because battery_level is a list of tuples (distance_battery,time_battery) We use time here. #TODO use distance as well

    @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))
    optimize!(model)
    println("Optimizing model took ", (time_ns() / 1e9) - t1, " seconds")


    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    selected_fly_indices = [(i,t,s) for i in GridpointsDrones, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - t1, " seconds")

    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("fly", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end

    updated_movement_plan = movement_plan[1:reevaluation_step]

    return updated_movement_plan

end

function NEW_ROUTING_STRATEGY_NEXTMOVE_INTEGER_BATTERY(risk_pertime_file,n_drones,ChargingStations,GroundStations,optimization_horizon,max_battery_time,reevaluation_step,drone_locations,drone_states,battery_level) #T here is optimization_horizon: should be > max_battery_time, reevaluation_step < floor(optimization_horizon/2)

    risk_pertime = load_burn_map(risk_pertime_file)
    _, N, M = size(risk_pertime)
    T = optimization_horizon
    t1 = time_ns() / 1e9 

    detection_rate = 0.7
    I_prime = nothing
    I_second = nothing
    I_third = nothing
    I = [(x, y) for x in 1:N for y in 1:M]

    # Convert Python lists of tuples to Julia Vector of tuples
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]

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

    GridpointsDrones = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I) # = allowed_gridpoints_drones
    GridpointsDronesDetecting = setdiff(GridpointsDrones,ChargingStations) # = NonChargingstations_allowed

    


    model = Model(Gurobi.Optimizer)
    set_silent(model)
    #Defining the variables
    a = @variable(model, [i in GridpointsDrones, t in 1:T, s in 1:n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t in 1:T, s in 1:n_drones], Bin)
    b = @variable(model, [t in 1:T, s in 1:n_drones], Int)
    
    #DEFINING THE CONSTRAINTS 
    #No 2 drones flying in the same place at the same time
    #@constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1) 
    #Each drone either charges or flies, not both
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    #Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    #Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t+1,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    
    # Battery level constraints: Integer between 0 and max_battery_time
    @constraint(model, [t in 1:T, s in 1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
    # Battery dynamics: decreases by 1 when flying, resets to max when charging
    # @constraint(model, [t in 1:T-1, s in 1:n_drones], 
    #     b[t+1,s] == b[t,s] - sum(a[i,t,s] for i in GridpointsDrones) + 
    #     (max_battery_time - b[t,s]) * sum(c[i,t,s] for i in ChargingStations)) # bilinear!
    
    @constraint(model, [t in 1:T-1, s in 1:n_drones], 
        b[t+1,s] <= b[t,s] - sum(a[i,t,s] for i in GridpointsDrones) + 
        max_battery_time * sum(c[i,t,s] for i in ChargingStations))




    ########## Constraints specific to the the next move problem
    #All drones start from location of previous drone
    for (s, state) in enumerate(drone_states)
        loc = drone_locations[s]  # This is a tuple (x,y)
        if state == "charge"
            @constraint(model, c[loc,1,s] == 1)
        elseif state == "fly"
            @constraint(model, a[loc,1,s] == 1)
        end
    end
    #All drones start with battery level as given in input
    @constraint(model, [s in 1:n_drones], b[1,s] == Int(battery_level[s][2])) 
    ########## End of constraints specific to the the next move problem

    # @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones))
    # objective that doesn't count twice the risk of the same cell if multiple drones fly there
    @variable(model, theta[t in 1:T, k in GridpointsDrones])
    @constraint(model, [t in 1:T, k in GridpointsDrones], theta[t,k] <= sum(a[k,t,s] for s in 1:n_drones))
    @constraint(model, [t in 1:T, k in GridpointsDrones], 0 <= theta[t,k] <= 1)
    @objective(model, Max, sum(risk_pertime[t,k...]*theta[t,k] for k in GridpointsDrones, t in 1:T))
    

    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    println("Creating model took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")


    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    selected_fly_indices = [(i,t,s) for i in GridpointsDrones, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - t1, " seconds")

    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("fly", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end

    updated_movement_plan = movement_plan[1:reevaluation_step]

    return updated_movement_plan

end



# a = println(NEW_ROUTING_STRATEGY_INIT("./WideDataset/0001/burn_map.npy", 2, [(101, 99)], [(1,1)], 10, 10, 10))
# b = println(NEW_ROUTING_STRATEGY_INIT_INTEGER_BATTERY("./WideDataset/0001/burn_map.npy", 2, [(101, 99)], [(1,1)], 10, 10, 10))

# Add a test for the new function
#NEW_ROUTING_STRATEGY_NEXTMOVE_INTEGER_BATTERY(risk_pertime_file,n_drones,ChargingStations,GroundStations,optimization_horizon,max_battery_time,reevaluation_step,drone_locations,drone_states,battery_level) #T here is optimization_horizon: should be > max_battery_time, reevaluation_step < floor(optimization_horizon/2)
# c = println(NEW_ROUTING_STRATEGY_NEXTMOVE_INTEGER_BATTERY("./WideDataset/0001/burn_map.npy", 2, [(101, 99)], [(1,1)], 12, 12, 10, [(103, 100), (97, 99)], ["fly", "fly"], [(8, 8), (8, 8)]))
#d = println(NEW_ROUTING_STRATEGY_NEXTMOVE(                "./WideDataset/0001/burn_map.npy", 2, [(101, 99)], [(1,1)], 12, 12, 10, [(103, 100), (97, 99)], ["fly", "fly"], [(8, 8), (8, 8)]))
#c == d

# New functions for optimized model handling
# -----------------------------------------

struct RoutingModel
    model::Model
    a::JuMP.Containers.DenseAxisArray{VariableRef, 3}  # Changed from Array to DenseAxisArray
    c::JuMP.Containers.DenseAxisArray{VariableRef, 3}  # Changed from Array to DenseAxisArray
    b::Array{VariableRef, 2}
    theta::JuMP.Containers.DenseAxisArray{VariableRef, 2}  # Changed from Array to DenseAxisArray
    init_constraints::Vector{ConstraintRef}
    next_move_constraints::Vector{ConstraintRef}
    GridpointsDrones::Set{Tuple{Int,Int}}
    ChargingStations::Vector{Tuple{Int,Int}}
    risk_pertime::Array{Float64, 3}
    T::Int
    n_drones::Int
    max_battery_time::Int
end

function create_routing_model(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time)
    t1 = time_ns() / 1e9
    risk_pertime = load_burn_map(risk_pertime_file)
    _, N, M = size(risk_pertime)
    T = optimization_horizon
    
    # Convert Python lists of tuples to Julia Vector of tuples if needed
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
    
    # println("ChargingStations: ", ChargingStations)
    
    I = [(x, y) for x in 1:N for y in 1:M]
    
    # Get grid points as a Set (not converting to Vector anymore)
    GridpointsDrones = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
    GridpointsDronesDetecting = setdiff(GridpointsDrones, ChargingStations)
    
    # println("GridpointsDrones examples: ", collect(GridpointsDrones)[1:min(5, length(GridpointsDrones))])

    # precomputing the closest distance to a charging station for each gridpoint
    precomputed_closest_distance_to_charging_station = closest_distances_tuple_index(ChargingStations, GridpointsDrones)
    # println("precomputed_closest_distance_to_charging_station: \n", precomputed_closest_distance_to_charging_station)
    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Define variables using tuple indices directly
    a = @variable(model, [i in GridpointsDrones, t=1:T, s=1:n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t=1:T, s=1:n_drones], Bin)
    b = @variable(model, [t=1:T, s=1:n_drones], Int)
    
    # Common constraints - using tuple indices directly
    # Each drone either charges or flies, not both
    @constraint(model, [t=1:T, s=1:n_drones], 
               sum(a[i,t,s] for i in GridpointsDrones) + 
               sum(c[i,t,s] for i in ChargingStations) == 1)
    
    # Movement constraints - using tuple indexing
    # Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], 
                c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t,s])
    
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], 
                a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + sum(c[i,t,s] for i in ChargingStations if i in neighbors_and_point(j)))
    
    # Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1
    # @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], 
    #             a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)) + c[j,t+1,s])
    
    # @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], 
    #             a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors_and_point(j)))
    
    # Min/max battery level constraints
    @constraint(model, [t=1:T, s=1:n_drones], 0 <= b[t,s] <= max_battery_time)
    
    # Battery dynamics
    # bilinear version
    # @constraint(model, [t=1:T-1, s=1:n_drones], 
    #     b[t+1,s] == b[t,s] - sum(a[i,t+1,s] for i in GridpointsDrones) + 
    #     (max_battery_time - b[t,s]) * sum(c[i,t+1,s] for i in ChargingStations))

    # linear version
    @constraint(model, [s in 1:n_drones, t in 1:T],
        b[t,s] >= max_battery_time*sum(c[i,t,s] for i in ChargingStations))
    @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - 1 + (max_battery_time+1) * sum(c[i,t+1,s] for i in ChargingStations))

    # no suicide constraint
    # wrong version: distance from all charging stations rather than minimum distance
    # @constraint(model, [s=1:n_drones, i in GridpointsDrones, j in ChargingStations], 
    # b[T,s] >= L_inf_distance(i,j)*a[i,T,s])
    # correct version: distance from closest charging station
    @constraint(model, [s=1:n_drones], 
    b[T,s] >= sum(a[i,T,s]*(precomputed_closest_distance_to_charging_station[i]) for i in GridpointsDrones))
    
    
    
    # Objective function with theta variables
    theta = @variable(model, [t=1:T, k in GridpointsDrones])
    @constraint(model, [t=1:T, k in GridpointsDrones], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones))
    @constraint(model, [t=1:T, k in GridpointsDrones], 0 <= theta[t,k] <= 1)
    
    # Risk objective - using tuple indices
    @objective(model, Max, sum(risk_pertime[t,k[1],k[2]]*theta[t,k] for t=1:T, k in GridpointsDrones))
    
    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]

    println("Model created in ", time_ns() / 1e9 - t1, " seconds")
    
    return RoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, risk_pertime, T, n_drones, max_battery_time)
end

function solve_init_routing(routing_model::RoutingModel, reevaluation_step)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
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
    for s in 1:n_drones #TODO add neighbors
        constraint = @constraint(model, 
                             sum(c[i,1,s] for i in ChargingStations) + 
                             sum(a[i,1,s] for i in ChargingStations) == 1)
        push!(routing_model.init_constraints, constraint)
    end
    
    # All drones start with full battery
    max_battery_time = routing_model.max_battery_time
    for s in 1:n_drones
        push!(routing_model.init_constraints, @constraint(model, b[1,s] == max_battery_time - sum(a[i,1,s] for i in GridpointsDrones))) # TODO replace with state input
    end
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    println("Creating init constraints took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")
    
    # Extract results
    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    # println("Battery levels: ", value.(b))
    
    # Generate movement plan using tuple indices directly
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    
    # Process results for fly actions using tuple indices
    selected_fly_indices = [(i,t,s) for i in GridpointsDrones, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]
    
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("fly", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end
    
    return movement_plan[1:reevaluation_step]
end

function solve_next_move_routing(routing_model::RoutingModel, reevaluation_step, drone_locations, drone_states, battery_level)
    model = routing_model.model
    a = routing_model.a
    c = routing_model.c
    b = routing_model.b
    ChargingStations = routing_model.ChargingStations
    GridpointsDrones = routing_model.GridpointsDrones
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
        if !(loc in GridpointsDrones)
            println("Error: Drone $s is starting at location $loc which is not in the grid points")
            error("Drone $s is starting at location $loc which is not in the grid points")
        end
        
        if state == "charge"
            # Find which charging station corresponds to this location
            if loc in ChargingStations
                push!(routing_model.next_move_constraints, @constraint(model, c[loc,1,s] == 1))
            else
                println("Error: Drone $s is charging but location $loc is not a charging station")
                error("Drone $s is charging but location $loc is not a charging station")
            end
        elseif state == "fly"
            push!(routing_model.next_move_constraints, @constraint(model, a[loc,1,s] == 1))
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
    println("Creating next_move constraints took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")
    
    # Extract results
    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    # println("Battery levels: ", value.(b))
    
    # Generate movement plan using tuple indices directly
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    
    # Process results for fly and charge actions using tuple indices
    selected_fly_indices = [(i,t,s) for i in GridpointsDrones, t in 1:reevaluation_step, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
    selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:reevaluation_step, s in 1:n_drones if value(c[i,t,s]) ≈ 1]
    
    # Replace random movements with optimized drone movements
    for (i, t, s) in selected_fly_indices
        movement_plan[t][s] = ("fly", i)  # Move to gridpoint i
    end
    for (i, t, s) in selected_charge_indices
        movement_plan[t][s] = ("charge", i)  # Charge at station i
    end
    
    return movement_plan[1:reevaluation_step]
end

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

function create_index_routing_model(risk_pertime_file, n_drones, ChargingStations, GroundStations, optimization_horizon, max_battery_time)
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
    
    # println("grid_to_idx for ChargingStations: ", [grid_to_idx[point] for point in ChargingStations])
    
    # Create variables with integer indices
    a = @variable(model, [i=1:length(GridpointsDrones), t=1:T, s=1:n_drones], Bin)
    c = @variable(model, [i=1:length(ChargingStations), t=1:T, s=1:n_drones], Bin)
    b = @variable(model, [t=1:T, s=1:n_drones], Int)
    
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
    # wrong version: distance from all charging stations rather than minimum distance
    # @constraint(model, [s=1:n_drones, i_idx=1:length(GridpointsDrones), j_idx=1:length(ChargingStations)], 
    #             b[T,s] >= L_inf_distance(GridpointsDrones[i_idx], ChargingStations[j_idx])*a[i_idx,T,s])
    # correct version: distance from closest charging station
    @constraint(model, [s=1:n_drones, i_idx=1:length(GridpointsDrones)], 
                b[T,s] >= a[i_idx,T,s]*precomputed_closest_distance_to_charging_station[i_idx])

    # Create objective variables with integer indices
    theta = @variable(model, [t=1:T, k=1:length(GridpointsDrones)])
    @constraint(model, [t=1:T, k=1:length(GridpointsDrones)], theta[t,k] <= sum(a[k,t,s] for s=1:n_drones))
    @constraint(model, [t=1:T, k=1:length(GridpointsDrones)], 0 <= theta[t,k] <= 1)

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
            # if point in GroundStationSet #Ground stations don't add to objective, so no need for drones to go there. Risk is set to 0.
            #     risk_idx[t,k] = 0.0
            # else
            risk_idx[t, k] = risk_pertime[t, point[1], point[2]]
            # end
        end
    end

    #capacity constraint on charging stations
    # @constraint(model, [i in 1:length(ChargingStations), t in 1:T], sum(c[i,t,s] for s in 1:n_drones) <= 2)
    

    @objective(model, Max, sum(risk_idx[t,k]*theta[t,k] for t=1:T, k=1:length(GridpointsDrones)))
    
    # Initialize constraint containers
    init_constraints = ConstraintRef[]
    next_move_constraints = ConstraintRef[]
    t2 = time_ns() / 1e9
    println("Model created in ", t2 - t1, " seconds")
    return IndexRoutingModel(model, a, c, b, theta, init_constraints, next_move_constraints, 
                        GridpointsDrones, ChargingStations, risk_pertime, T, n_drones, grid_to_idx, charging_map, max_battery_time)
end

function solve_index_init_routing(routing_model::IndexRoutingModel, reevaluation_step)
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
                               sum(a[c_index_to_grid_idx[i],1,s] for i in charging_station_idx) == 1)
        push!(routing_model.init_constraints, constraint)
    end
    
    # All drones start with full battery
    max_battery_time = routing_model.max_battery_time
    for s in 1:n_drones
        push!(routing_model.init_constraints, @constraint(model, b[1,s] == max_battery_time - sum(a[i,1,s] for i in 1:length(GridpointsDrones))))
    end
    
    # Optimize
    t2 = time_ns() / 1e9
    optimize!(model)
    t3 = time_ns() / 1e9
    println("Creating init constraints took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")

    println("Drone starting positions (Julia): ", [grid_to_idx[station] for station in ChargingStations])
    println("Charging Stations (Julia): ", ChargingStations)
    
    # Extract results
    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    
    # Generate movement plan using integer indices
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    
    # Process results for fly actions
    for t in 1:reevaluation_step
        for s in 1:n_drones
            # Check fly actions
            for i in 1:length(GridpointsDrones)
                if value(a[i,t,s]) ≈ 1
                    movement_plan[t][s] = ("fly", GridpointsDrones[i])
                end
            end
            # Check charge actions
            for i in 1:length(ChargingStations)
                if value(c[i,t,s]) ≈ 1
                    movement_plan[t][s] = ("charge", ChargingStations[i])
                end
            end
        end
    end
    
    return movement_plan[1:reevaluation_step]
end

function solve_index_next_move_routing(routing_model::IndexRoutingModel, reevaluation_step, drone_locations, drone_states, battery_level)
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
    println("Creating next_move constraints took ", t2 - t1, " seconds")
    println("Optimizing model took ", t3 - t2, " seconds")
    
    # Extract results
    println("Solver Status: ", termination_status(model))
    println("Objective Value: ", has_values(model) ? objective_value(model) : "No solution found")
    
    # Generate movement plan using integer indices
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:reevaluation_step]
    
    # Process results for fly actions
    for t in 1:reevaluation_step
        for s in 1:n_drones
            # Check fly actions
            for i in 1:length(GridpointsDrones)
                if value(a[i,t,s]) ≈ 1
                    movement_plan[t][s] = ("fly", GridpointsDrones[i])
                end
            end
            # Check charge actions
            for i in 1:length(ChargingStations)
                if value(c[i,t,s]) ≈ 1
                    movement_plan[t][s] = ("charge", ChargingStations[i])
                end
            end
        end
    end
    
    return movement_plan[1:reevaluation_step]
end
