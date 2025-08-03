#input as we can do it for every charging station separately.
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
include("TOP_PSO.jl")


# EXAMPLE ON GENERATED DATA 

Random.seed!(42)
n_drones = 2
max_battery_time = 8
N = 8
M = 8
function generate_random_charging_stations(N::Int, M::Int, num_stations::Int)
    selected = rand(1:N*M, num_stations)
    return [(div(i-1, M)+1, mod(i-1, M)+1) for i in selected]
end

# Example: generate 1 random charging station on a 20x20 grid
N = 8
M = 8
ChargingStation = generate_random_charging_stations(N, M, 1)
risk_pertime = rand(1, N, M)  # 1 time step, values between 0 and 1
function generate_random_ground_stations(N::Int, M::Int, num_stations::Int)
    selected = rand(1:N*M, num_stations)
    return [(div(i-1, M)+1, mod(i-1, M)+1) for i in selected]
end
GroundStations = generate_random_ground_stations(N, M, 5)

L = 8

# ---------- parameters ----------

# risk_pertime = load_burn_map(risk_pertime_file)
H, N, M = size(risk_pertime)
if H == 1 # we duplicate the risk per time for 100 time steps
    println("Duplicating risk per time for 100 time steps")
    risk_pertime = repeat(risk_pertime, 100, 1, 1)
    H = 100
end

# Define ChargingStation beforehand
ChargingStations = ChargingStation
# Convert Python lists of tuples to Julia Vector of tuples if needed
ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
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


function milp_relaxed(risk_pertime,n_drones,ChargingStation,GroundStations,max_battery_time, L)

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

    return model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y
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

# --------------- PLOT ROUTES ---------------

# node_index_to_coords = Dict(i => coords[i] for i in 1:length(coords))

# # Precompute layout positions
# locs_x = [node_index_to_coords[i][1] for i in 1:length(node_index_to_coords)]
# locs_y = [node_index_to_coords[i][2] for i in 1:length(node_index_to_coords)]

# # Node colors (green for transit, red for depots)
# n_nodes = length(GridpointsDronesDetecting) + 2
# nodefillc = fill(colorant"green", n_nodes)
# nodefillc[Begin_CS] = colorant"red"
# nodefillc[End_CS] = colorant"red"

# # Define distinct colors for each drone's path
# edge_colors = [RGB(1,1,1), RGB(1,0.5,0), RGB(0.5,0.5,1), RGB(0,1,0), RGB(1,0,1)]  # Add more if needed

# # Store graphs and plots
# drone_graphs = Dict{Int, SimpleDiGraph}()
# drone_plots = Vector{Compose.Context}(undef, n_drones)

# for s in 1:n_drones
#     G = SimpleDiGraph(n_nodes)
#     stroke_colors = RGB[]  # Edge colors for this drone

#     for i in GridpointsDrones, j in GridpointsDrones
#         if value(x[i, j, s]) > 0.8
#             add_edge!(G, i, j)
#             push!(stroke_colors, edge_colors[s ≤ length(edge_colors) ? s : end])
#         end
#     end

#     # Store graph
#     drone_graphs[s] = G

#     # Plot
#     drone_plots[s] = gplot(
#         G,
#         locs_x,
#         locs_y;
#         nodefillc = nodefillc,
#         edgestrokec = stroke_colors,
#         nodelabel = 1:nv(G),
#         arrowlengthfrac = 0.05,
#         nodesize = 0.8,
#         title = "Drone $s"
#     )
# end

# # Combine plots side by side
# side_by_side_plot = hstack(drone_plots...)

# # Save as PNG (requires Cairo & Fontconfig)
# draw(PNG("drones_side_by_side_2.png", 300 * n_drones, 500), side_by_side_plot)

# # Show plot
# display(side_by_side_plot)



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


# routes = greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)
# obj_value = compute_objective_greedy(routes, coords, risk_pertime, Begin_CS, End_CS)
# println("Objective value = $obj_value")



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
function get_PSO_solution(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time)
    # Convert GridpointsDronesDetecting to customer format for PSO
    customers = GridpointsDronesDetecting
    profits = Float64[]
    
    # Extract profits for each customer
    for (x, y) in customers
        push!(profits, risk_pertime[1, x, y])
    end
    
    # Create cost matrix using infinity norm (as in TOP.jl)
    costs = Dict{Tuple{Int,Int}, Float64}()
    n_customers = length(customers)
    depot_x, depot_y = ChargingStation[1]
    
    # Costs from depot to customers and back
    for i in 1:n_customers
        xi, yi = customers[i]
        inf_dist_from_depot = max(abs(xi - depot_x), abs(yi - depot_y))
        costs[(0, i)] = inf_dist_from_depot <= 1 ? 1.0 : max_battery_time*4
        costs[(i, 0)] = costs[(0, i)]
    end
    
    # Costs between customers
    for i in 1:n_customers
        for j in 1:n_customers
            if i != j
                xi, yi = customers[i]
                xj, yj = customers[j]
                inf_dist = max(abs(xi - xj), abs(yi - yj))
                costs[(i, j)] = inf_dist <= 1 ? 1.0 : max_battery_time*4
            else
                costs[(i, j)] = 0.0
            end
        end
    end
    
    # Run PSO algorithm with proper parameters for CPA initialization
    println("Running PSO for CPA initialization...")
    giant_tour, pso_profit, pso_obj = solve_PSO_TOP(
        customers, profits, costs, n_drones, max_battery_time;
        swarm_size=10, max_iterations=1,
        w=0.3, c1=0.5, c2=0.3, ph=0.15, pm=0.3
    )
    
    # Convert PSO routes back to TOP.jl format
    pso_routes = extract_routes(giant_tour, pso_obj)
    
    # Convert to TOP.jl route format (with Begin_CS and End_CS indices)
    top_routes = Vector{Vector{Int}}(undef, n_drones)
    Begin_CS = length(GridpointsDronesDetecting) + 1
    End_CS = length(GridpointsDronesDetecting) + 2
    
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
                x, y = coords[node_id]
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

function plot_routes(routes, coords, Begin_CS, End_CS, GridpointsDronesDetecting, n_drones, filename_suffix="")
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

    # Show plot
    display(side_by_side_plot)
    
    println("Plot saved as: $filename")
end

function CPA(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)

    # Initial upper bound (UB) and initial PSO lower bound (LB)
    model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y = milp_relaxed(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)
    
    UB = sum(risk_pertime[1, GridpointsDronesDetecting[k]...] for k in TransitGridpoints)
    
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
    routes, best_LB = get_PSO_solution(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time)
    
    # Also compute greedy for comparison
    greedy_routes = greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)
    greedy_LB = compute_objective_greedy(greedy_routes, coords, risk_pertime, Begin_CS, End_CS)

    println("Initial LB from PSO = $best_LB, LB from greedy = $greedy_LB, UB = $UB")
    println("PSO improvement over greedy: $(round(((best_LB - greedy_LB) / greedy_LB) * 100, digits=2))%")
    
    # Print the initial PSO solution routes
    print_routes(routes, coords, n_drones, "(PSO Initial)")
    
    # Plot the initial PSO solution
    println("Plotting initial PSO solution...")
    plot_routes(routes, coords, Begin_CS, End_CS, GridpointsDronesDetecting, n_drones, "pso_initial")

    # WARM START THE MODEL WITH THE PSO SOLUTION
    println("Warm starting MILP with PSO solution...")
    error("Stop here") # keep this for now
    warm_start_with_solution!(model, x, y, routes, n_drones, GridpointsDrones, TransitGridpoints)

    iteration = 1
    println("\n--- Iteration $iteration ---")

        while true
            optimize!(model)

            opt_val = objective_value(model) # this is P(SOL)
            if opt_val < UB
                UB = opt_val
            end
            
            println("Iteration $iteration: LB = $best_LB, UB = $UB, Gap = $(round(((UB - best_LB) / UB) * 100, digits=2))%")

            # --------------- PLOT THE GRAPH ---------------
            # Convert MILP solution to routes format for plotting
            milp_routes = extract_tours_from_solution(x, 1:n_drones, GridpointsDrones, Begin_CS, End_CS)
            route_vectors = Vector{Vector{Int}}(undef, n_drones)
            for s in 1:n_drones
                if haskey(milp_routes, s)
                    route_vectors[s] = milp_routes[s]
                else
                    route_vectors[s] = [Begin_CS, End_CS]  # Empty route
                end
            end
            
            # Plot using the reusable function
            plot_routes(route_vectors, coords, Begin_CS, End_CS, GridpointsDronesDetecting, n_drones, "iter$(iteration)")
            # --------------- END OF PLOT ---------------

            # Here we assume that the relaxed problem has been solved to optimality (i.e gurobi did not timeout or anything)
            # extract subtours from the solution
            subtours_per_drone = subtours(n_drones, GridpointsDrones, Begin_CS, End_CS, x)

            # extract tours from the solution
            tours_per_drone = extract_tours_from_solution(x, 1:n_drones, GridpointsDrones, Begin_CS, End_CS)
            # this becomes our new feasible solution if it improves the current LB
            if !isempty(tours_per_drone)
                profit = 0.0
                for s in 1:n_drones
                    # if a tour exists for drone s, then we add the profit of the tour to the total profit
                    if haskey(tours_per_drone, s) && length(tours_per_drone[s]) > 2
                        profit += sum(risk_pertime[1, coords[k]...] for k in tours_per_drone[s][2:end-1]) # skip the depots
                    end
                end
                if profit > best_LB
                    best_LB = profit
                    # cast dict to vector of vectors    
                    routes = Vector{Vector{Int}}(undef, n_drones)
                    for s in 1:n_drones
                        if haskey(tours_per_drone, s)
                            routes[s] = tours_per_drone[s]
                        else
                            routes[s] = []
                        end
                    end
                end
            end

            # if no subtour, then we can stop the algorithm!
            has_subtours = any(s -> !isempty(subtours_per_drone[s]), 1:n_drones)
            if !has_subtours
                println("✓ No subtours found - optimal solution reached!")
                println("Final objective value: $best_LB")
                return routes, UB, x, y
            end
            
            total_subtours = sum(length(subtours_per_drone[s]) for s in 1:n_drones)
            println("Found $total_subtours subtours across $(length(subtours_per_drone)) drones - adding constraints...")

            # if lower bound is equal to upper bound, then we can stop the algorithm!
            if best_LB == UB
                println("✓ Optimal solution found (LB = UB)!")
                println("Final objective value: $best_LB")
                return routes, UB, x, y
            end

            # if there are subtours, then we need to add GSEC constraints to eliminate them
            # THE BIG QUESTION IS: Do we use only the tour for the GSEC constraint, or do we use the whole relaxed solution (tour + subtours)? TODO
            # OR do we use the subtours, so we take their complement??!!! TODO

            for s in 1:n_drones
                for T in subtours_per_drone[s]
                    # S is V \ T
                    S = setdiff(GridpointsDrones,T)
                    # check that both depots are in S: raise error if not
                    if !(Begin_CS in S && End_CS in S)
                        error("Begin_CS or End_CS not in S")
                    end
                    # check that S is not empty: raise error if it is
                    if isempty(S)
                        error("S is empty")
                    end

                    outside_S = T
    
                    delta_plus = [(u, v) for u in S for v in outside_S if c[(u,v)] < L]
                    delta_min = [(u,v) for u in outside_S for v in S if c[(u,v)] <L]

                    delta = [delta_plus; delta_min]

                    gamma = [(u,v) for u in S for v in S if c[(u,v)] <L]

                    gamma_T = [(u,v) for u in T for v in T if c[(u,v)] <L]
                    
                    S_minus_depots = setdiff(S, [Begin_CS, End_CS])
                    # println("S_minus_depots = $S_minus_depots")


                    if !isempty(delta)
                        @constraint(model, [i in T, s in 1:n_drones], sum(x[u, v, s] for (u, v) in delta) >= 2*y[i, s]) # (8) in the paper
                        @constraint(model, [j in T, s in 1:n_drones], sum(x[u, v, s] for (u, v) in gamma) <= sum(y[i, s] for i in S_minus_depots) - y[j,s] + 1) # (9) in the paper
                        @constraint(model, [j in T, s in 1:n_drones], sum(x[u, v, s] for (u, v) in gamma_T) <= sum(y[i, s] for i in T) - y[j,s]) # (10) in the paper
                    end
                end
            end

            optimize!(model)
            iteration += 1
            
        end
end

routes, UB, x, y = CPA(risk_pertime,n_drones,ChargingStation,GroundStations,max_battery_time, L)