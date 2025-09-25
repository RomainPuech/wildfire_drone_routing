# comment to install the Julia packages the first time you run the code

# import Pkg
# Pkg.add("JuMP")
# Pkg.add("Gurobi")
# Pkg.add("Graphs")
# Pkg.add("GraphPlot")
# Pkg.add("Colors")
# Pkg.add("Plots")
# Pkg.add("Compose")
# Pkg.add("Cairo")
# Pkg.add("Fontconfig")
# Pkg.add("DataStructures")

import Cairo
import Fontconfig

using JuMP
using Gurobi
using Graphs
using GraphPlot
using Colors
using Plots
using Compose
using DataStructures
using Random

include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")



function milp_relaxed(risk_pertime,n_drones,ChargingStation,GroundStations,max_battery_time, L)

    # ---------- parameters ----------
    
    # Extract dimensions from risk_pertime 
    H, N, M = size(risk_pertime)
    if H == 1 # we duplicate the risk per time for 100 time steps
        # println("Duplicating risk per time for 100 time steps")
        risk_pertime = repeat(risk_pertime, 100, 1, 1)
        H = 100
    end

    # Convert Python lists of tuples to Julia Vector of tuples if needed
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStation]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]

    I = [(x, y) for x in 1:N for y in 1:M] # All feasible grid points
    GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
    # GridpointsDrones = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDrones_set)) # All feasible grid points for drones
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
    #GridpointsDronesDetecting_set = setdiff(GridpointsDronesDetecting_set, GroundStations) 
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set)) # All feasible grid points for drones minus the grid points in which a charging station is placed
    GridpointsDrones = 1:(length(GridpointsDronesDetecting) + 2)
    GridpointsDrones_begin = 1:(length(GridpointsDronesDetecting) + 1)
    GridpointsDrones_end = setdiff(GridpointsDrones,[length(GridpointsDronesDetecting) + 1])
    TransitGridpoints = 1:length(GridpointsDronesDetecting)
    Begin_CS  = length(GridpointsDronesDetecting) + 1
    End_CS = length(GridpointsDronesDetecting) + 2

    #define c[i,j] as 1 if drone can fly in one timestep from i to j, otherwise set c[i,j] > L, where L is limit
    coords = deepcopy(GridpointsDronesDetecting)
    push!(coords, ChargingStations[1])  # For Begin_CS
    push!(coords, ChargingStations[1])  # For End_CS

    # Define number of total drone nodes
    n_nodes = length(coords)
    c = Dict{Tuple{Int,Int}, Float64}()

    for i in 1:n_nodes, j in 1:n_nodes
        xi, yi = coords[i]
        xj, yj = coords[j]

        inf_dist = max(abs(xi - xj), abs(yi - yj))
        if inf_dist <= 1
            c[(i, j)] = 1.0
        else
            c[(i, j)] = L*4
        end
    end

    c[(Begin_CS,End_CS)] = L*4

    return GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints
    # what comes after, we don't need it anymore since we don't use the model

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # ---------- variables ----------

    x = @variable(model, [i in GridpointsDrones, j in GridpointsDrones, s = 1:n_drones], Bin)
    y = @variable(model, [i in TransitGridpoints, s = 1:n_drones], Bin)

    # ---------- constraints ----------

    #Each gridpoint is visited at most once by one drone
    @constraint(model, [i in TransitGridpoints], sum(y[i,s] for s in 1:n_drones) <= 1) # (2)

    #Each vehicle starts its path at charging station and ends at charging station, modeled as different charging stations
    @constraint(model, [s=1:n_drones], sum(x[Begin_CS,i,s] for i in GridpointsDrones_end) == 1) # (3)
    @constraint(model, [s=1:n_drones], sum(x[i,End_CS,s] for i in GridpointsDrones_begin) == 1) # (3)

    # No incoming arc to Begin_CS
    @constraint(model, [j in GridpointsDrones, s in 1:n_drones], x[j, Begin_CS, s] == 0) # (?)

    # No outgoing arc from End_CS
    @constraint(model, [j in GridpointsDrones, s in 1:n_drones], x[End_CS, j, s] == 0) # (?)


    # #Ensure connectivity of each tour            
    @constraint(model, [k in TransitGridpoints, s=1:n_drones], 
                sum(x[k,i,s] for i in setdiff(GridpointsDrones_end,[k])) == y[k,s]) # (4)
    @constraint(model, [k in TransitGridpoints, s=1:n_drones], 
                sum(x[j,k,s] for j in setdiff(GridpointsDrones_begin,[k])) == y[k,s]) # (4)

    #Impose travel length restriction
    for s in 1:n_drones
        @constraint(model,
            sum(
                c[(i, j)] * x[i, j, s]
                for i in GridpointsDrones_begin
                for j in GridpointsDrones_end
                if i != j && haskey(c, (i, j))
            ) <= L # (5)
        )
    end

    # symmetry breaking constraints: we order the drones by decreasing profit
    @constraint(model, [s in 1:n_drones-1], sum(risk_pertime[1,GridpointsDronesDetecting[k]...]*(y[k,s+1]) for k in TransitGridpoints) <= sum(risk_pertime[1,GridpointsDronesDetecting[k]...]*(y[k,s]) for k in TransitGridpoints)) # (11)
    
    # @objective(model, Max, 0)
    @objective(model, Max, sum(risk_pertime[1,GridpointsDronesDetecting[k]...]*(y[k,s]) for k in TransitGridpoints for s in 1:n_drones)) # (1)

    return GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints
end

# model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y = milp_relaxed(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)
# optimize!(model)

function solve_TOP_init_routing(model)
    optimize!(model)
    # extract solution in format [[code, (x,y)], [code, (x,y)], ...]
    # Generate movement plan using integer indices
    movement_plan = [[("stay", (0, 0)) for _ in 1:n_drones] for _ in 1:max_battery_time]
    # x gives us the transition from i to j for drone s. We have one transition per drone per time step.
    # We need to extract the transition from i to j for each drone at each time step.
    for s in 1:n_drones
        t = 1
        movement_plan[t][s] = ("charge", ChargingStations[1])
        current_node = Begin_CS
        while current_node != End_CS
            next_nodes = [j for j in GridpointsDrones if value(x[current_node, j, s]) > 0.8]
            if isempty(next_nodes)
                println("Drone $s could not return to End_CS")
                break
            elseif length(next_nodes) > 1
                println("Drone $s has multiple next nodes")
                break
            else
                current_node = next_nodes[1]
                t += 1
                movement_plan[t][s] = ("fly", coords[current_node])
                
            end
        end
        t += 1
        movement_plan[t][s] = ("charge", ChargingStations[1])
    end
    return movement_plan
end

function solve_TOP_next_move_routing(model, drone_locations, t)
    # update the model with the new initial drone locations
    optimize!(model)
    return solution
end


# --------------- FIND SUBTOURS ---------------
function subtours(n_drones, GridpointsDrones, Begin_CS, End_CS, x)
    subtours_per_drone = OrderedDict{Int, Vector{Vector{Int}}}()

    for s in 1:n_drones
        used_nodes = Set{Int}()
        edges_s = Tuple{Int, Int}[]
        for i in GridpointsDrones, j in GridpointsDrones
            if value(x[i, j, s]) > 0.8
                push!(edges_s, (i, j))
                push!(used_nodes, i)
                push!(used_nodes, j)
            end
        end

        node_map = Dict(node => idx for (idx, node) in enumerate(sort(collect(used_nodes))))
        reverse_map = Dict(v => k for (k, v) in node_map)

        G = DiGraph(length(node_map))
        for (i, j) in edges_s
            add_edge!(G, node_map[i], node_map[j])
        end

        components = strongly_connected_components(G)
        subtours = Vector{Vector{Int}}()

        for comp in components
            node_ids = [reverse_map[v] for v in comp]
            if !(Begin_CS in node_ids || End_CS in node_ids) && length(node_ids) > 1
                push!(subtours, node_ids)
            end
        end

        subtours_per_drone[s] = subtours
    end

    return subtours_per_drone
end

# subtours_per_drone = subtours(n_drones, GridpointsDrones, Begin_CS, End_CS,x)
# sorted_subtours = OrderedDict(k => v for (k, v) in sort(collect(subtours_per_drone)))





# --------------- INITIAL GREEDY SOLUTION ---------------
function greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)

    #track all visited nodes so no two drones visit the same one
    visited = Set{Int}()

    #initialize route storage for each drone
    routes = Vector{Vector{Int}}(undef, n_drones)

    #loop over each drone 
    for s in 1:n_drones
        current_node = Begin_CS         #start at charging station
        battery = max_battery_time      #initialize battery
        route = [current_node]          #start route with depot

        while true
            best_node = nothing         #best candidate for next node
            best_reward = -Inf          #max reward so far
            best_cost = Inf             #cost to reach best candidate

            #try all available nodes to find the best next one 
            for (j_idx, j_coords) in enumerate(coords)
                #skip if already visited, same as current, or is the end depot
                if j_idx in visited || j_idx == current_node || j_idx == End_CS
                    continue
                end

                #check if rachable and if the drone can still return afterward
                if haskey(c, (current_node, j_idx)) && c[(current_node, j_idx)] <= battery
                    reward = risk_pertime[1, j_coords...]
                    cost_to_end = haskey(c, (j_idx, End_CS)) ? c[(j_idx, End_CS)] : Inf
                    total_cost = c[(current_node, j_idx)] + cost_to_end

                    # Select node if reward is better and it's feasible to return
                    if reward > best_reward && total_cost <= battery
                        best_reward = reward
                        best_node = j_idx
                        best_cost = c[(current_node, j_idx)]
                    end
                end
            end

            # If no feasible node to visit, break the loop
            if best_node === nothing
                break
            end

            # Visit selected node
            push!(route, best_node)
            union!(visited, [best_node])    #mark as visited
            battery -= best_cost            #reduce battery
            current_node = best_node        #move to new node 
        end

        # Always return to End_CS
        if haskey(c, (current_node, End_CS)) && c[(current_node, End_CS)] <= battery
            push!(route, End_CS)
        else
            println("Drone $s could not return to End_CS")
        end

        routes[s] = route
    end

    return routes
end

function compute_objective_greedy(routes, coords, risk_pertime, Begin_CS, End_CS)
    total_reward = 0.0
    for route in routes
        for node in route
            if node != Begin_CS && node != End_CS  # exclude depots
                coord = coords[node]
                total_reward += risk_pertime[1, coord...]
            end
        end
    end
    return total_reward
end



# --------------- CUTTING PLANE ALGORITHM ---------------

function extract_tours_from_solution(x, drones, GridpointsDrones, Begin_CS, End_CS)
    tours = Dict{Int, Vector{Int}}()

    for s in drones
        # check if the drone has a tour, i.e. a path that starts at Begin_CS and ends at End_CS
        if sum(x[Begin_CS, j, s] for j in GridpointsDrones) == 0 || sum(x[j, End_CS, s] for j in GridpointsDrones) == 0
            continue
        end

        route = Int[]
        current_node = Begin_CS

        while true
            push!(route, current_node)
            next_nodes = [j for j in GridpointsDrones if value(x[current_node, j, s]) > 0.5]
            
            if isempty(next_nodes)
                break
            end
            if length(next_nodes) > 1
                println("PROBLEM !! Drone $s has multiple next nodes")
                break
            end

            current_node = next_nodes[1]

            # Stop if we reached the End_CS
            if current_node == End_CS
                push!(route, current_node)
                break
            end
        end

        tours[s] = route
    end

    return tours
end


"""
Convert TOP.jl format to PSO format and run PSO algorithm
"""
function get_PSO_solution_multiple_depots(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time, initial_drone_positions = []; use_greedy_init::Bool = true, max_time::Float64 = 3000.0, max_iterations::Int = 500, swarm_size::Int = 10)
    start_time = time()
    # println("Starting get_PSO_solution_multiple_depots...")
    # Convert GridpointsDronesDetecting to customer format for PSO
    customers = GridpointsDronesDetecting
    # println("Customers: $(customers)")
    profits = Float64[]

    # find how many drones start at each depot
    if length(initial_drone_positions) > 0 
        if length(initial_drone_positions) != n_drones
            error("Initial drone positions must be of length n_drones or empty")
        end
        n_duplicates_array = [0 for _ in 1:length(ChargingStation)]
        for i in 1:n_drones
            for (depot_idx, depot) in enumerate(ChargingStation)
                if initial_drone_positions[i] == depot
                    n_duplicates_array[depot_idx] += 1
                    break
                end 
            end
        end
        println("n_duplicates_array: $n_duplicates_array")
    else
        # How many times we duplicate the depot depends on the number of drones.
        # We want as many copies of any depot as there are drones.
        n_duplicates_array = [n_drones for _ in 1:length(ChargingStation)]
    end


    # add the depot and duplicate depots to the customers
    for (depot_idx, depot) in enumerate(ChargingStation)
        for _ in 1:n_duplicates_array[depot_idx]
            push!(customers, depot)
        end
    end
    # println("Customers: $(customers)")
    # println("len Customers: $(length(customers))")

    # Extract profits for each customer and clamp to non-negative values
    for (x, y) in customers
        profit_value = risk_pertime[1, x, y]
        # Clamp profits to be non-negative to avoid complex number errors in PSO algorithm
        clamped_profit = max(0.0, profit_value)
        push!(profits, clamped_profit)
    end
    n_customers = length(customers) - sum(n_duplicates_array)
    # println("n_customers: $n_customers")
    # Create cost matrix using infinity norm (as in TOP.jl)
    costs = Dict{Tuple{Int,Int}, Float64}()

    # IF there are multiple depots, we create an artificial node connecting all depots
    artificial_node = 0 #length(customers) + length(ChargingStation) + 1
    costs[(artificial_node, artificial_node)] = 0.0

    
    # cost from any customer to artificial node is infinite cost
    for i in 1:n_customers
        costs[(artificial_node, i)] = max_battery_time*4 
        costs[(i, artificial_node)] = max_battery_time*4 
    end

    # cost from artificial node to all depots is 0 and from all depots to artificial node is 0
    # we also duplicate all depots to allow going back to artificial node at the end
    depot_node = n_customers + 1  # Start indexing depot nodes after customers
    for depot_idx in 1:length(ChargingStation)
        for duplicate_idx in 1:n_duplicates_array[depot_idx]
            costs[(artificial_node, depot_node)] = 0.0 # should I put a cost higher than the battery and add this cost to the battery?
            costs[(depot_node, artificial_node)] = 0.0
            depot_node += 1
        end
    end # /!\ BREAKS TRIANGLE INEQUALITY
    
    # Create a helper function to check if two depot nodes are duplicates of the same depot
    function are_same_depot_duplicates(i, j, n_customers, n_duplicates_array)
        if i <= n_customers || j <= n_customers
            return false  # At least one is not a depot
        end
        
        # Find which depot each node belongs to
        depot_i = -1
        depot_j = -1
        current_depot_node = n_customers + 1
        
        for depot_idx in 1:length(n_duplicates_array)
            for duplicate_idx in 1:n_duplicates_array[depot_idx]
                if current_depot_node == i
                    depot_i = depot_idx
                end
                if current_depot_node == j
                    depot_j = depot_idx
                end
                current_depot_node += 1
            end
        end
        
        return depot_i == depot_j && depot_i != -1
    end
    
    # Costs between customers (this includes depots, but the cost will be overwritten later)
    for i in 1:length(customers) # this includes the depots but the cost will be overwritten later
        for j in 1:length(customers) # this includes the depots but the cost will be overwritten later
            if i != j
                # if both are duplicate of the same depot, we put an infinite cost
                if are_same_depot_duplicates(i, j, n_customers, n_duplicates_array)
                    costs[(i, j)] = max_battery_time*4
                    costs[(j, i)] = max_battery_time*4
                else
                    xi, yi = customers[i]
                    xj, yj = customers[j]
                    inf_dist = max(abs(xi - xj), abs(yi - yj))
                    costs[(i, j)] = inf_dist <= 1 ? 1.0 : max_battery_time*4
                end
            else
                costs[(i, j)] = 0.0
            end
        end
    end

    depot_node = n_customers + 1  # Start indexing depot nodes after customers
    for (depot_idx, depot) in enumerate(ChargingStation)
        depot_x, depot_y = depot
        for duplicate_idx in 1:n_duplicates_array[depot_idx]
            # Costs from depot to customers and back
            for i in 1:n_customers
                xi, yi = customers[i]
                inf_dist_from_depot = max(abs(xi - depot_x), abs(yi - depot_y))
                costs[(depot_node, i)] = inf_dist_from_depot <= 1 ? 1.0 : max_battery_time*4
                costs[(i, depot_node)] = inf_dist_from_depot <= 1 ? 1.0 : max_battery_time*4
            end
            depot_node += 1
        end
    end
    
    # Run PSO algorithm with proper parameters for CPA initialization
    println("=== PROBLEM SETUP ===")
    println("Customers: $(length(customers)), Battery limit: $max_battery_time, Drones: $n_drones")
    # println("Depot: $(ChargingStation[1])")
    # println("======================")
    time_before_pso = time()
    total_time_without_pso = time_before_pso - start_time
    # println("Running PSO for initial solution...")
    giant_tour, pso_profit, pso_obj = solve_PSO_TOP_multiple_depots(
        customers, profits, costs, n_drones, n_customers, max_battery_time, ChargingStation;
        swarm_size=swarm_size, max_iterations=max_iterations,  # Use passed parameters
        max_time=max_time, # Use passed parameter
        w=0.3, c1=0.5, c2=0.3, ph=0.15, pm=0.3, use_greedy_init=use_greedy_init
    )
    time_after_pso = time()
    println("execution time for PSO algorithm: $(time_after_pso - time_before_pso)")
   
    # Convert PSO routes back to TOP.jl format
    pso_routes = extract_routes(giant_tour, pso_obj)
    
    # Convert to TOP.jl route format (with Begin_CS and End_CS indices)
    top_routes = Vector{Vector{Int}}(undef, n_drones)
    Begin_CS = 0 # length(GridpointsDronesDetecting) + 1
    End_CS = 0 # length(GridpointsDronesDetecting) + 2
    
    for s in 1:n_drones
        if s <= length(pso_routes) && !isempty(pso_routes[s])
            # Convert customer indices to TOP.jl format
            route = [Begin_CS]  # Start at charging station
            append!(route, pso_routes[s])  # Add customer indices (already correct)
            push!(route, End_CS)  # End at charging station
            top_routes[s] = route
        else
            # Empty route: just go from Begin_CS to End_CS
            top_routes[s] = [Begin_CS, End_CS]
        end
    end

    # print the top routes
    # println("Top routes:")
    # for s in 1:n_drones
    #     println("Drone $s: $(top_routes[s])")
    # end

    # here, we replace the duplicate nodes with the original nodes
    # Create a mapping from duplicate depot indices to original depot indices
    function map_duplicate_to_original(node_idx, n_customers, n_duplicates_array)
        if node_idx <= n_customers
            return node_idx  # Customer node, no change
        end
        
        # This is a depot node, find which original depot it corresponds to
        current_depot_node = n_customers + 1
        for depot_idx in 1:length(n_duplicates_array)
            for duplicate_idx in 1:n_duplicates_array[depot_idx]
                if current_depot_node == node_idx
                    # Return the index of the first duplicate of this depot
                    first_duplicate_idx = n_customers + sum(n_duplicates_array[1:depot_idx-1]) + 1
                    return first_duplicate_idx
                end
                current_depot_node += 1
            end
        end
        
        return node_idx  # Should not reach here
    end
    
    for s in 1:n_drones
        if s <= length(top_routes) && length(top_routes[s]) >= 2
            route = top_routes[s]
            for i in 1:length(route)
                if route[i] > n_customers
                    route[i] = map_duplicate_to_original(route[i], n_customers, n_duplicates_array)
                end
            end
        end
    end
    time_after_mapping = time()
    total_time_without_pso += time_after_mapping - time_after_pso
    println("total time without PSO: $total_time_without_pso")
    return top_routes, pso_profit
end


"""
Warm start the MILP model with a given solution
"""
function warm_start_with_solution!(model, x, y, routes, n_drones, GridpointsDrones, TransitGridpoints)
    try
        # Reset all variables to 0 first (using JuMP DenseAxisArray syntax)
        for i in GridpointsDrones, j in GridpointsDrones, s in 1:n_drones
            try
                set_start_value(x[i, j, s], 0.0)
            catch
                # Variable may not exist, continue
            end
        end
        
        for k in TransitGridpoints, s in 1:n_drones
            try
                set_start_value(y[k, s], 0.0)
            catch
                # Variable may not exist, continue
            end
        end
        
        # Set variables based on the PSO solution routes
        for s in 1:n_drones
            if s <= length(routes) && length(routes[s]) >= 2
                route = routes[s]
                
                # Set x variables for transitions in this route
                for idx in 1:(length(route)-1)
                    i = route[idx]
                    j = route[idx+1]
                    
                    try
                        set_start_value(x[i, j, s], 1.0)
                        println("  Setting x[$i,$j,$s] = 1.0")
                    catch e
                        println("  Could not set x[$i,$j,$s]: $e")
                    end
                end
                
                # Set y variables for visited transit points
                for node in route
                    # Only set y for transit gridpoints (not depots)
                    if node in TransitGridpoints
                        try
                            set_start_value(y[node, s], 1.0)
                            println("  Setting y[$node,$s] = 1.0")
                        catch e
                            println("  Could not set y[$node,$s]: $e")
                        end
                    end
                end
            end
        end
        
        println("Warm start completed successfully")
        
    catch e
        println("Warning: Could not set warm start values: $e")
        println("Continuing without warm start...")
    end
end

function print_routes(routes, coords, n_drones, filename_suffix="")
    println("\n=== Routes $filename_suffix ===")
    for s in 1:n_drones
        if s <= length(routes) && length(routes[s]) >= 2
            route = routes[s]
            route_str = ""
            for (idx, node_id) in enumerate(route)
                if node_id == 0
                    x, y = -1, -1 # ARTIFICIAL NODE
                    # route_str += "(A,A)"
                else
                    x, y = coords[node_id]
                end
                if idx == 1
                    route_str = "($x,$y)"
                else
                    route_str *= " -> ($x,$y)"
                end
            end
            println("Drone $s: $route_str")
        else
            println("Drone $s: No route")
        end
    end
    println()
end

"""
Get the coordinates of the tours, remove the artificial node, and patches the path to the depot
"""
function get_patched_tours_coordinates(routes, GridpointsDronesDetecting, ChargingStations, n_drones)
    tours_coordinates = Vector{Vector{Tuple{Int,Int}}}(undef, n_drones)
    println("routes: $routes")
    for s in 1:n_drones
        tours_coordinates[s] = Vector{Tuple{Int,Int}}()  # Initialize with empty vector
        if s <= length(routes) && length(routes[s]) >= 3
            route = routes[s]
            # println("route: $route")
            if route[1] != 0
                error("First node should be the artificial node")
            end
            if route[end] != 0
                error("Last node should be the artificial node")
            end
            if route[2] > length(GridpointsDronesDetecting)
                println("Warning: Route index $(route[2]) exceeds GridpointsDronesDetecting length $(length(GridpointsDronesDetecting))")
                continue  # Skip this route
            end
            current_node = GridpointsDronesDetecting[route[2]]
            if !(current_node in ChargingStations)
                error("First node should be a charging station")
            end
            push!(tours_coordinates[s], current_node)

            if length(route) > 3
                for next_node_index in 3:length(route)-1 # last node is artificial node
                    if route[next_node_index] > length(GridpointsDronesDetecting)
                        println("Warning: Route index $(route[next_node_index]) exceeds GridpointsDronesDetecting length $(length(GridpointsDronesDetecting))")
                        continue  # Skip this node
                    end
                    next_node = GridpointsDronesDetecting[route[next_node_index]]

                    # patch the path between 2 clients
                    while abs(current_node[1] - next_node[1]) > 1 || abs(current_node[2] - next_node[2]) > 1
                        current_node = (current_node[1] + sign(next_node[1] - current_node[1]), current_node[2] + sign(next_node[2] - current_node[2]))
                        push!(tours_coordinates[s], current_node)
                    end
                    push!(tours_coordinates[s], next_node)
                    current_node = next_node
                end

                # patch the path to the last depot
                last_node = current_node
                if !(last_node in ChargingStations)
                    # We identify the closest charging station
                    closest_charging_station_idx = findmin(x -> sum(abs.(x .- last_node)), ChargingStations)[2]
                    closest_charging_station = ChargingStations[closest_charging_station_idx]
                    while abs(current_node[1] - closest_charging_station[1]) > 1 || abs(current_node[2] - closest_charging_station[2]) > 1
                        current_node = (current_node[1] + sign(closest_charging_station[1] - current_node[1]), current_node[2] + sign(closest_charging_station[2] - current_node[2]))
                        push!(tours_coordinates[s], current_node)
                    end
                    push!(tours_coordinates[s], closest_charging_station)
                end

            end

        end
    end
    return tours_coordinates
end

function plot_routes(routes, coords, Begin_CS, End_CS, GridpointsDronesDetecting, n_drones, filename_suffix="", verbose::Bool = true)
    # Node index to coordinates mapping
    node_index_to_coords = Dict(i => coords[i] for i in 1:length(coords))

    # Precompute layout positions
    locs_x = [node_index_to_coords[i][1] for i in 1:length(node_index_to_coords)]
    locs_y = [node_index_to_coords[i][2] for i in 1:length(node_index_to_coords)]

    # Node colors (green for transit, red for depots)
    n_nodes = length(GridpointsDronesDetecting) + 2
    nodefillc = fill(colorant"green", n_nodes)
    nodefillc[Begin_CS] = colorant"red"
    nodefillc[End_CS] = colorant"red"

    # Define distinct colors for each drone's path
    edge_colors = [RGB(1,1,1), RGB(1,0.5,0), RGB(0.5,0.5,1), RGB(0,1,0), RGB(1,0,1)]  # Add more if needed

    # Store graphs and plots
    drone_graphs = Dict{Int, SimpleDiGraph}()
    drone_plots = Vector{Compose.Context}(undef, n_drones)

    for s in 1:n_drones
        G = SimpleDiGraph(n_nodes)
        stroke_colors = RGB[]  # Edge colors for this drone

        # Create edges from the route
        if s <= length(routes) && length(routes[s]) >= 2
            route = routes[s]
            for idx in 1:(length(route)-1)
                i = route[idx]
                j = route[idx+1]
                add_edge!(G, i, j)
                push!(stroke_colors, edge_colors[s ≤ length(edge_colors) ? s : end])
            end
        end

        # Store graph
        drone_graphs[s] = G

        # Plot
        drone_plots[s] = gplot(
            G,
            locs_x,
            locs_y;
            nodefillc = nodefillc,
            edgestrokec = stroke_colors,
            nodelabel = 1:nv(G),
            arrowlengthfrac = 0.05,
            nodesize = 0.8,
            title = "Drone $s"
        )
    end

    # Combine plots side by side
    side_by_side_plot = hstack(drone_plots...)

    # Save as PNG (requires Cairo & Fontconfig)
    filename = isempty(filename_suffix) ? "drones_side_by_side.png" : "drones_side_by_side_$filename_suffix.png"
    draw(PNG(filename, 300 * n_drones, 500), side_by_side_plot)

    # Show plot only if verbose mode is enabled
    if verbose
        display(side_by_side_plot)
        println("Plot saved as: $filename")
    end
end

function find_highest_risk_point_within_radius(risk_pertime, possible_centers, radius, possible_points, ChargingStations = [])
    if length(ChargingStations) == 0
        ChargingStations = possible_centers
    end
    #println("ChargingStations: $ChargingStations")
    best_risk = -1.0
    best_point = nothing
    best_center = nothing
    best_cost = 0.0
    best_return_cost = 0.0
    best_return_point = nothing
    for center in possible_centers
        for point in possible_points
            if point == center
                continue
            end
            cost = max(abs(center[1] - point[1]), abs(center[2] - point[2]))
            if cost <= radius && point[1] > 0 && point[1] <= size(risk_pertime, 2) && point[2] > 0 && point[2] <= size(risk_pertime, 3) && risk_pertime[1, point...] > best_risk
                # and if you can come back to  a charging station, i.e if the distance to the closest charging station + distance to center <= radius
                return_cost, closest_charging_station_index = findmin(x -> max(abs(x[1] - point[1]), abs(x[2] - point[2])), ChargingStations)
                closest_charging_station = ChargingStations[closest_charging_station_index]
                # println("closest_charging_station: $closest_charging_station")
                # println("point:", point)
                # println("return_cost:", return_cost)
                if cost + return_cost <= radius
                    best_risk = risk_pertime[1, point...]
                    best_point = point
                    best_center = center
                    best_cost = cost
                    best_return_cost = return_cost
                    best_return_point = closest_charging_station
                end
            end
        end
    end
    return best_point, best_center, best_cost, best_return_cost, best_return_point
end

function patch_path_to_route!(route, target_point)
    """
    Patch path from the last point in route to target_point by adding intermediate points.
    Modifies route in-place and adds target_point at the end.
    """
    current_point = route[end]
    while abs(current_point[1] - target_point[1]) > 1 || abs(current_point[2] - target_point[2]) > 1
        current_point = (current_point[1] + sign(target_point[1] - current_point[1]), current_point[2] + sign(target_point[2] - current_point[2]))
        push!(route, current_point)
    end
    push!(route, target_point)
end

function set_point_risk_to_zero!(risk_pertime, point)
    """
    Set the risk of a specific point to zero at time 1.
    """
    if point === nothing
        return
    end
    if point[1] >= 1 && point[1] <= size(risk_pertime, 2) && 
       point[2] >= 1 && point[2] <= size(risk_pertime, 3)
        risk_pertime[1, point[1], point[2]] = 0.0
    end
end

function patch_path_with_highest_risk!(route, target_point, risk_pertime)
    """
    Patch path from the last point in route to target_point using dynamic programming
    to find the path with the highest cumulative risk using Chebyshev distance.
    Modifies route in-place and adds target_point at the end.
    """
    start_point = route[end]
    
    # If points are adjacent or the same, just add the target point
    if max(abs(start_point[1] - target_point[1]), abs(start_point[2] - target_point[2])) <= 1
        if start_point != target_point
            push!(route, target_point)
        end
        return
    end
    
    # Get dimensions of risk_pertime
    N, M = size(risk_pertime, 2), size(risk_pertime, 3)
    
    # Ensure both points are within bounds
    if start_point[1] < 1 || start_point[1] > N || start_point[2] < 1 || start_point[2] > M ||
       target_point[1] < 1 || target_point[1] > N || target_point[2] < 1 || target_point[2] > M
        # Fallback to simple patching if points are out of bounds
        patch_path_to_route!(route, target_point)
        return
    end
    
    # Get bounds for the search area - for Chebyshev distance, we need to expand bounds
    # to include all points that could be on a shortest path
    chebyshev_dist = max(abs(target_point[1] - start_point[1]), abs(target_point[2] - start_point[2]))
    min_x = min(start_point[1], target_point[1]) - chebyshev_dist
    max_x = max(start_point[1], target_point[1]) + chebyshev_dist
    min_y = min(start_point[2], target_point[2]) - chebyshev_dist
    max_y = max(start_point[2], target_point[2]) + chebyshev_dist
    
    # Clamp bounds to grid dimensions
    min_x = max(1, min_x)
    max_x = min(N, max_x)
    min_y = max(1, min_y)
    max_y = min(M, max_y)
    
    # Initialize DP table: dp[x][y] = (max_cumulative_risk, parent_x, parent_y)
    dp = Dict{Tuple{Int,Int}, Tuple{Float64, Int, Int}}()
    
    # Initialize starting point with its risk at time 1
    start_risk = risk_pertime[1, start_point[1], start_point[2]]
    dp[start_point] = (start_risk, -1, -1)  # -1, -1 indicates no parent
    
    # Use a priority queue (simulated with sorted processing) to process points
    # in order of decreasing cumulative risk to ensure optimal substructure
    
    # Process all points in the bounding rectangle
    changed = true
    max_iterations = (max_x - min_x + 1) * (max_y - min_y + 1) * 10  # Prevent infinite loops
    iteration = 0
    
    while changed && iteration < max_iterations
        changed = false
        iteration += 1
        
        for x in min_x:max_x
            for y in min_y:max_y
                current = (x, y)
                
                # Skip if not on a valid path (Chebyshev distance constraint)
                start_dist = max(abs(x - start_point[1]), abs(y - start_point[2]))
                target_dist = max(abs(target_point[1] - x), abs(target_point[2] - y))
                total_dist = max(abs(target_point[1] - start_point[1]), abs(target_point[2] - start_point[2]))
                
                # Point must be on a shortest path between start and target for Chebyshev distance
                if start_dist + target_dist != total_dist
                    continue
                end
                
                # Get current point's risk at time 1
                current_risk = risk_pertime[1, x, y]
                
                
                if current == start_point
                    continue  # Already initialized
                end
                
                best_total_risk = -Inf
                best_parent_x, best_parent_y = -1, -1
                
                # Check all possible previous positions (8-connected neighbors)
                for dx in -1:1, dy in -1:1
                    if dx == 0 && dy == 0
                        continue
                    end
                    
                    prev_x, prev_y = x - dx, y - dy
                    prev = (prev_x, prev_y)
                    
                    # Check if previous position is in bounds and in DP table
                    if prev_x >= min_x && prev_x <= max_x && prev_y >= min_y && prev_y <= max_y && haskey(dp, prev)
                        # Check if this move is valid (towards target) using Chebyshev distance
                        prev_start_dist = max(abs(prev_x - start_point[1]), abs(prev_y - start_point[2]))
                        prev_target_dist = max(abs(target_point[1] - prev_x), abs(target_point[2] - prev_y))
                        
                        # Previous point must also be on a shortest path and closer to start
                        if prev_start_dist + prev_target_dist == total_dist && prev_start_dist < start_dist
                            total_risk = dp[prev][1] + current_risk
                            
                            if total_risk > best_total_risk
                                best_total_risk = total_risk
                                best_parent_x, best_parent_y = prev_x, prev_y
                            end
                        end
                    end
                end
                
                # Update DP table if we found a better path
                if best_parent_x != -1 && (!haskey(dp, current) || best_total_risk > dp[current][1])
                    dp[current] = (best_total_risk, best_parent_x, best_parent_y)
                    changed = true
                end
            end
        end
    end
    
    # Reconstruct path from target back to start
    if !haskey(dp, target_point)
        # Fallback to simple patching if DP fails
        patch_path_to_route!(route, target_point)
        return
    end
    
    # Build path from target to start
    path = []
    current = target_point
    while current != start_point
        pushfirst!(path, current)
        parent_x, parent_y = dp[current][2], dp[current][3]
        if parent_x == -1 && parent_y == -1
            break
        end
        current = (parent_x, parent_y)
    end
    
    # Add the path to the route (excluding start_point which is already in route)
    for point in path
        push!(route, point)
    end
end

function get_greedy_fallback_solution(risk_pertime, tours_coordinates, GridpointsDronesDetecting, ChargingStations, GroundStations, max_battery_time, n_drones, initial_drone_positions, override_allowed_initial_positions = [])
    # Make a copy of risk_pertime to avoid modifying the original
    risk_pertime_copy = copy(risk_pertime)
    for station in  GroundStations
        set_point_risk_to_zero!(risk_pertime_copy, station)
    end
    for station in ChargingStations
        set_point_risk_to_zero!(risk_pertime_copy, station)
    end
    
    possible_points = setdiff(GridpointsDronesDetecting, ChargingStations)
    possible_points = setdiff(possible_points, GroundStations)
    for s in 1:n_drones
        possible_points = setdiff(possible_points, tours_coordinates[s])
    end
    
    # Set risk to zero for all points already visited by existing tours
    for s in 1:n_drones
        for point in tours_coordinates[s]
            set_point_risk_to_zero!(risk_pertime_copy, point)
        end
    end
    
    # check which initial positions we are still allowed to use. These are the ones in the initial_drone_positions that are not in the first node of tours_coordinates counted with multiplicity
    allowed_initial_positions = copy(initial_drone_positions)
    for s in 1:n_drones
        if length(tours_coordinates[s]) > 0
            start_node = tours_coordinates[s][1]
            idx = findfirst(x -> x == start_node, allowed_initial_positions)
            if idx !== nothing
                deleteat!(allowed_initial_positions, idx)
            end
        end
    end
    # now make allowed_initial_positions a set to avoid duplicates
    allowed_initial_positions = Set(allowed_initial_positions)
    if length(allowed_initial_positions) == 0
        # we can use any initial position
        allowed_initial_positions = ChargingStations
    end
    # if override_allowed_initial_positions is not empty, then we use it instead of allowed_initial_positions
    if length(override_allowed_initial_positions) > 0
        allowed_initial_positions = Set(override_allowed_initial_positions)
    end
    # we now have the list of allowed initial positions. We now find the chain of points
    if max_battery_time <= 1
        element = first(allowed_initial_positions)
        return [element, element]
    end
    best_first_point, first_center, current_cumulative_cost, current_return_cost, best_return_point = find_highest_risk_point_within_radius(risk_pertime_copy, allowed_initial_positions, max_battery_time, possible_points, ChargingStations)
    println("best_first_point: $best_first_point")
    println("first_center: $first_center")
    println("current_cumulative_cost: $current_cumulative_cost")
    println("current_return_cost: $current_return_cost")
    println("best_return_point: $best_return_point")
    
    # Handle case where find_highest_risk_point_within_radius returns nothing
    if best_first_point === nothing || first_center === nothing
        println("Warning: No valid points found in greedy fallback, returning minimal route")
        if !isempty(allowed_initial_positions)
            start_depot = first(allowed_initial_positions)
            return [start_depot, start_depot]  # Minimal route: start and end at same depot
        else
            # Ultimate fallback if no allowed positions
            if !isempty(ChargingStations)
                fallback_depot = ChargingStations[1]
                return [fallback_depot, fallback_depot]
            else
                # Should never happen, but just in case
                return [(1, 1), (1, 1)]
            end
        end
    end
    
    # Initialize final route and patch between first_center and best_first_point
    final_route = [first_center]
    # Set first_center risk to zero in this time step
    set_point_risk_to_zero!(risk_pertime_copy, first_center)
    
    patch_path_with_highest_risk!(final_route, best_first_point, risk_pertime_copy)
    
    # Set risk to zero for all newly visited points
    for point in final_route
        set_point_risk_to_zero!(risk_pertime_copy, point)
    end
    
    # Remove points we've used from possible_points
    route_main_points = [first_center, best_first_point]
    possible_points = setdiff(possible_points, final_route)
    
    while current_cumulative_cost + current_return_cost < max_battery_time
        best_next_point, _, best_cost, best_return_cost, best_return_point = find_highest_risk_point_within_radius(risk_pertime_copy, [final_route[end]], max_battery_time - current_cumulative_cost, possible_points, ChargingStations)
        # if nothing
        if best_next_point === nothing
            break
        end
        
        # Patch from current end of route to the new point
        route_length_before = length(final_route)
        patch_path_with_highest_risk!(final_route, best_next_point, risk_pertime_copy)
        
        # Set risk to zero for all newly visited points in this segment
        for i in (route_length_before + 1):length(final_route)
            point = final_route[i]
            set_point_risk_to_zero!(risk_pertime_copy, point)
        end
        
        push!(route_main_points, best_next_point)
        possible_points = setdiff(possible_points, final_route)
        current_cumulative_cost += best_cost
    end
    
    if best_return_point === nothing
        println("WARNING: best_return_point is nothing")
        if ChargingStations == []
            ChargingStations = allowed_initial_positions
        end
        # Ensure we have charging stations to work with
        if !isempty(ChargingStations)
            # find the closest charging station to the last point
            closest_charging_station, closest_charging_station_index = findmin(x -> max(abs(x[1] - final_route[end][1]), abs(x[2] - final_route[end][2])), ChargingStations)
            closest_charging_station = ChargingStations[closest_charging_station_index]
            best_return_point = closest_charging_station
        else
            # Ultimate fallback - return to where we started
            println("WARNING: No charging stations available, returning to start point")
            best_return_point = final_route[1]
        end
    end
    
    # Patch from the last point to the return point
    patch_path_with_highest_risk!(final_route, best_return_point, risk_pertime_copy) # add the last return charging point again to close the loop
    
    return final_route
end

function CPA_multiple_depots(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L, verbose::Bool = false, initial_drone_positions = [])
    # Create plural version for function calls that expect plural
    ChargingStations = ChargingStation
    
    # println("Starting CPA...")
    # Initial upper bound (UB) and initial PSO lower bound (LB)
    GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints = milp_relaxed(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)
    
    # UB = sum(risk_pertime[1, GridpointsDronesDetecting[k]...] for k in TransitGridpoints)
    
    # Create cost matrix c for greedy comparison
    n_nodes = length(coords)
    c = Dict{Tuple{Int,Int}, Float64}()
    
    for i in 1:n_nodes, j in 1:n_nodes
        xi, yi = coords[i]
        xj, yj = coords[j]
        
        inf_dist = max(abs(xi - xj), abs(yi - yj))
        if inf_dist <= 1
            c[(i, j)] = 1.0
        else
            c[(i, j)] = L*4
        end
    end
    
    # Prevent direct connection between Begin_CS and End_CS
    c[(Begin_CS, End_CS)] = L*4
    
    # Use PSO instead of greedy for initialization
    # TODO TEMPORARY: ALWAYS CALL THE MULTI-DEPOT VERSION
    routes, best_LB = get_PSO_solution_multiple_depots(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time, initial_drone_positions)
    # if length(ChargingStation) == 1
    #     routes, best_LB = get_PSO_solution(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time)
    # else
    #     routes, best_LB = get_PSO_solution_multiple_depots(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time)
    # end

    
    # Also compute greedy for comparison
    #greedy_routes = greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)
    #greedy_LB = compute_objective_greedy(greedy_routes, coords, risk_pertime, Begin_CS, End_CS)

    # println("Initial LB from PSO = $best_LB, LB from greedy = $greedy_LB, UB = $UB")
    # println("PSO improvement over greedy: $(round(((best_LB - greedy_LB) / greedy_LB) * 100, digits=2))%")
    

    # print GridpointsDronesDetecting
    # println("GridpointsDronesDetecting: $(GridpointsDronesDetecting)")
    # println("length(GridpointsDronesDetecting): $(length(GridpointsDronesDetecting))")


    # Print the initial PSO solution routes
    # print_routes(routes, GridpointsDronesDetecting, n_drones, "(PSO Initial)")
    tours_coordinates = get_patched_tours_coordinates(routes, GridpointsDronesDetecting, ChargingStations,n_drones)
    # fallback mechanism: if one of the tours is empty, we use a greedy solution
    for s in 1:n_drones
        if length(tours_coordinates[s]) < 3
            println("WARNING:We use the FALLBACK SOLUTION for drone $s")
            tours_coordinates[s] = get_greedy_fallback_solution(risk_pertime, tours_coordinates, GridpointsDronesDetecting, ChargingStations, GroundStations, max_battery_time, n_drones, initial_drone_positions)
        end
    end
    #println("tours_coordinates: $(tours_coordinates)")
    
    # Plot the initial PSO solution
    if verbose
        println("Plotting initial PSO solution...")
        plot_routes(routes, coords, Begin_CS, End_CS, GridpointsDronesDetecting, n_drones, "pso_initial", verbose)
    end
    # also log these routes to a file, append if the file already exists   
    # open("pso_initial_routes.txt", "a") do f
    #     for s in 1:n_drones
    #         write(f, "Drone $s: ")
    #         for (i, route_idx) in enumerate(routes[s])
    #             coord = get(GridpointsDronesDetecting, route_idx, (-1, -1))
    #             write(f, "$coord")
    #             if i < length(routes[s])
    #                 write(f, " -> ")
    #             end
    #         end
    #         write(f, "\n")
    #     end
    # end

    # RETURN THE PSO SOLUTION DIRECTLY (bypassing CPA algorithm)
    # println("\n=== RETURNING PSO SOLUTION DIRECTLY ===")
    # println("Final PSO objective value: $best_LB")
    # println("Skipping CPA algorithm")
    return routes, tours_coordinates

    # CDELETED: The rest of the CPA algorithm
    # ...
end


#### MULTIPLE DEPOTS
function compute_TOP_plan_multiple_depots(risk_pertime_file::String,
    n_drones::Int,
    ChargingStations::Vector{Tuple{Int,Int}},
    GroundStations::Vector{Tuple{Int,Int}},
    max_battery_time::Int,
    t::Int,
    verbose::Bool = false,
    initial_drone_positions = [])

    start_time = time()
    if n_drones == 0
        return []
    end
   # julia-indexing for the burnmap is 1-based, so we need to shift the time index by 1
   t += 1
   # Load the burn-map (.npy)
   risk_pertime = load_burn_map(risk_pertime_file)
   risk_pertime = risk_pertime[t:end, :, :]

   # put risk of 0 for the charging stations
   for cs in ChargingStations
      risk_pertime[:, cs[1], cs[2]] .= 0
   end

   # put risk of 0 for the ground stations
   for gs in GroundStations
      risk_pertime[:, gs[1], gs[2]] .= 0
   end

   # The TOP horizon (L) equals the max battery time by assumption
   L = max_battery_time

   # Create GridpointsDronesDetecting for use in extensions
   _, N, M = size(risk_pertime)
   I = [(x, y) for x in 1:N for y in 1:M] # All feasible grid points
   GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
   GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
   GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))

   # ------------------------------------------------------------------
   # 1) Solve the Team-Orienteering Problem via CPA (returns routes)
   # ------------------------------------------------------------------
   time_before_cpa = time()
   routes, tours_coordinates = CPA_multiple_depots(risk_pertime, n_drones,
                          ChargingStations,
                          GroundStations,
                          max_battery_time,
                          L,
                          verbose,
                          initial_drone_positions)
    time_after_cpa = time()
    cpa_time = time_after_cpa - time_before_cpa
    println("execution time for CPA: $cpa_time")
    

    movement_plan = [ [("stay", (0,0)) for _ in 1:n_drones] for _ in 1:max_battery_time+1]
    println("tours_coordinates: $tours_coordinates")
    for s in 1:n_drones
        t = 1

        # here we check if the tour is too short and extend it if necessary
        if length(tours_coordinates[s]) <= max_battery_time - 1 # -1 because we need at least 2 battery steps remaining for the extension
            # we extend it using the extension solution
            println("extending tour, length(tours_coordinates[s]): $(length(tours_coordinates[s]))")
            remaining_time = max_battery_time - length(tours_coordinates[s])
            overwritten_allowed_initial_positions = [tours_coordinates[s][end]] # the extension has to start from where the drone currently is
            extension_solution = get_greedy_fallback_solution(risk_pertime, tours_coordinates, GridpointsDronesDetecting, ChargingStations, GroundStations, remaining_time, n_drones, initial_drone_positions, overwritten_allowed_initial_positions)
            # append the extension solution to the original tour
            tours_coordinates[s] = [tours_coordinates[s]; extension_solution[2:end]] # skiping the duplicate of the last node
        end
        
        movement_plan[1][s] = ("charge", tours_coordinates[s][t])
        
        while t < max_battery_time
            t += 1
            #println("t: $t, s: $s")
            if t > length(tours_coordinates[s])
                #warn("WARNING: tours_coordinates[s] is too short")
                println("WARNING: tours_coordinates[s] is too short")
                println("tours_coordinates[s]: $tours_coordinates[s]")
                println("t: $t")
                println("s: $s")
                println("max_battery_time: $max_battery_time")
                println("length(tours_coordinates[s]): $(length(tours_coordinates[s]))")
                movement_plan[t][s] = ("fly", tours_coordinates[s][end])
            else
                movement_plan[t][s] = ("fly", tours_coordinates[s][t])
            end
        end
        movement_plan[max_battery_time+1][s] = ("charge", tours_coordinates[s][end])
    end
    # println("movement_plan: $movement_plan")
    total_time = time() - start_time
    println("total julia time: $total_time")
    println("total time without CPA: $(total_time - cpa_time)")
    return movement_plan

#    # ------------------------------------------------------------------
#    # 2) Re-build the coordinate vector that maps node indices → (x,y)
#    #    so we can convert the integer routes into explicit actions
#    # ------------------------------------------------------------------
#    _, N, M = size(risk_pertime)
#    I = [(x, y) for x in 1:N for y in 1:M]            # all grid cells
#    Grid_set  = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
#    Grid_det  = setdiff(Grid_set, ChargingStations)
#    Grid_det_vec = convert(Vector{Tuple{Int,Int}}, collect(Grid_det))

#    coords = deepcopy(Grid_det_vec)
#    push!(coords, ChargingStations[1])  # Begin_CS
#    push!(coords, ChargingStations[1])  # End_CS

#    Begin_CS = length(Grid_det_vec) + 1
#    End_CS   = length(Grid_det_vec) + 2

#    # ------------------------------------------------------------------
#    # 3) Build the time-indexed movement plan expected by Python
#    # ------------------------------------------------------------------
#    horizon = max_battery_time                    # optimization horizon
#    movement_plan = [ [("stay", (0,0)) for _ in 1:n_drones] for _ in 1:horizon+1]

#    for s in 1:n_drones
#        route = s <= length(routes) && !isempty(routes[s]) ? routes[s] : [Begin_CS, End_CS]
#        movement_plan[1][s] = ("charge", ChargingStations[1])
#        t = 1
#        for node_idx in route[2:end-1]  # skip initial and final depots
#            t += 1
#            if t > horizon
#                break
#            end

           
#            next_node = coords[node_idx]
#            current_node = get(coords, route[t-1], (0,0)) # 0,0 here for the python plot
           
#            # Check if we can fly directly (Chebyshev distance = 1)
#            if t >=3 && max(abs(next_node[1] - current_node[1]), abs(next_node[2] - current_node[2])) != 1
#                # We need to "patch" the path with intermediate steps
#                current_pos = (current_node[1], current_node[2])  # Create a copy to avoid mutating original
               
#                while max(abs(next_node[1] - current_pos[1]), abs(next_node[2] - current_pos[2])) > 1
#                    # Move one step toward the target
#                    new_x = current_pos[1]
#                    new_y = current_pos[2]
#                    if abs(next_node[1] - current_pos[1]) > 0
#                        new_x += sign(next_node[1] - current_pos[1])
#                    end
#                    if abs(next_node[2] - current_pos[2]) > 0
#                        new_y += sign(next_node[2] - current_pos[2])
#                    end
#                    current_pos = (new_x, new_y)
                   
#                    movement_plan[t][s] = ("fly", current_pos)
#                    t += 1
                   
#                    # Safety check to prevent infinite loops
#                    if t > horizon
#                        break
#                    end
#                end
#            end
#            movement_plan[t][s] = ("fly", next_node)
#        end
#     #    if t < horizon + 1 # we include the final depot manually
#     #        t += 1
#     #        movement_plan[t][s] = ("charge", ChargingStations[1])
#     #    end
#    end

#    return movement_plan[2:end] # no need to include starting depot in the movement plan
#    # return movement_plan[2:end] # no need to include starting depot in the movement plan
end











# Overloaded method to handle Vector{Any} for ground stations (empty case from PyCall)
function compute_TOP_plan_single_depot(risk_pertime_file::String,
                          n_drones::Int,
                          ChargingStations::Vector{Tuple{Int,Int}},
                          GroundStations::Vector{Any},  # Allow Vector{Any} for empty case
                          max_battery_time::Int,
                          t::Int,
                          verbose::Bool = false)
    # Convert Vector{Any} to Vector{Tuple{Int,Int}}
    typed_ground_stations = Vector{Tuple{Int,Int}}()
    for gs in GroundStations
        if isa(gs, Tuple{Int,Int})
            push!(typed_ground_stations, gs)
        end
    end
    
    # Call the main method with properly typed arguments
    return compute_TOP_plan_single_depot(risk_pertime_file, n_drones, ChargingStations, typed_ground_stations, max_battery_time, t, verbose)
end


function compute_TOP_plan_multiple_depots(risk_pertime_file::String,
    n_drones::Int,
    ChargingStations::Vector{Tuple{Int,Int}},
    GroundStations::Vector{Any},  # Allow Vector{Any} for empty case
    max_battery_time::Int,
    t::Int,
    verbose::Bool = false,
    initial_drone_positions = [])
    # Convert Vector{Any} to Vector{Tuple{Int,Int}}
    typed_ground_stations = Vector{Tuple{Int,Int}}()
    for gs in GroundStations
        if isa(gs, Tuple{Int,Int})
            push!(typed_ground_stations, gs)
        end
    end
    return compute_TOP_plan_multiple_depots(risk_pertime_file, n_drones, ChargingStations, typed_ground_stations, max_battery_time, t, verbose, initial_drone_positions)
end
