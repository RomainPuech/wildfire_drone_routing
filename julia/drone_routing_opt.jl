# import helper_functions such as load_burn_map
include("helper_functions.jl")

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
    @constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1) 
    #Each drone either charges or flies, not both
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    #Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors(j)) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors(j)))
    #Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors(j)) + c[j,t+1,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors(j)))
    #Min/max battery level constraints
    @constraint(model, [t in 1:T, s in 1:n_drones], B_min <= b[t,s] <= B_max)
    #Battery level at t+1 is less than battery level at t - 0.2 if drone flies at t + B_max if drone charges at t 
    @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - (1-B_min)/max_battery_time*sum(a[i,t,s] for i in GridpointsDrones) + B_max*sum(c[i,t,s] for i in ChargingStations))
    #Drones need to charge if battery level falls below B_min
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(c[i,t,s] for i in ChargingStations) >= 1 - b[t,s]/B_min)
    #All drones start to fly from a charging station at t=1
    @constraint(model, [s in 1:n_drones], sum(c[i,1,s] for i in ChargingStations) + sum(a[i,1,s] for i in ChargingStations) == 1) #First run
    # @constraint(model, [s in 1:n_drones], sum(a[i,2,s] for i in GridpointsDrones) == 1) #First run

    #All drones start with full battery at t=1
    @constraint(model, [s in 1:n_drones], b[1,s] == B_max) #First run

    @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))
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

    println("GridpointsDrones: ", GridpointsDrones)
    println("GridpointsDronesDetecting: ", GridpointsDronesDetecting)
    println("ChargingStations: ", ChargingStations)


    model = Model(Gurobi.Optimizer)
    set_silent(model)
    #Defining the variables
    a = @variable(model, [i in GridpointsDrones, t in 1:T, s in 1:n_drones], Bin)
    c = @variable(model, [i in ChargingStations, t in 1:T, s in 1:n_drones], Bin)
    b = @variable(model, [t in 1:T, s in 1:n_drones])
    
    #DEFINING THE CONSTRAINTS 
    #All drones start from location of previous drone
    for (s, state) in enumerate(drone_states)
        loc = drone_locations[s]  # This is a tuple (x,y)
        if state == "charge"
            @constraint(model, c[loc,1,s] == 1)
        elseif state == "fly"
            @constraint(model, a[loc,1,s] == 1)
        end
    end

    #No 2 drones flying in the same place at the same time
    @constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1) 
    #Each drone either charges or flies, not both
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
    #Drone can only charge/fly at j at t+1 if it already charged at j or if it flew in a neighboring gridpoint at t
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors(j)) + c[j,t,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if i in neighbors(j)))
    #Drone can only fly at j at t if it is flying at a neighboring grid point at t+1 or charging at j at t+1
    @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors(j)) + c[j,t+1,s])
    @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if i in neighbors(j)))
    #Min/max battery level constraints
    @constraint(model, [t in 1:T, s in 1:n_drones], B_min <= b[t,s] <= B_max)
    #Battery level at t+1 is less than battery level at t - 0.2 if drone flies at t + B_max if drone charges at t 
    @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - (1-B_min)/max_battery_time*sum(a[i,t,s] for i in GridpointsDrones) + B_max*sum(c[i,t,s] for i in ChargingStations))
    #Drones need to charge if battery level falls below B_min
    @constraint(model, [t in 1:T, s in 1:n_drones], sum(c[i,t,s] for i in ChargingStations) >= 1 - b[t,s]/B_min)



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



# NEW_ROUTING_STRATEGY_INIT("./WideDataset/0001/burn_map.npy", 1, [(113, 187)], [(1,1)], 10, 10, 3)