#input as we can do it for every charging station separately.
import Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("Graphs")
Pkg.add("GraphPlot")
Pkg.add("Colors")
Pkg.add("Plots")
Pkg.add("Compose")
Pkg.add("Cairo")
Pkg.add("Fontconfig")
Pkg.add("DataStructures")

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


# EXAMPLE ON GENERATED DATA 

Random.seed!(42)
n_drones = 3
max_battery_time = 10
N = 20
M = 20
function generate_random_charging_stations(N::Int, M::Int, num_stations::Int)
    selected = rand(1:N*M, num_stations)
    return [(div(i-1, M)+1, mod(i-1, M)+1) for i in selected]
end

# Example: generate 1 random charging station on a 20x20 grid
N = 20
M = 20
ChargingStation = generate_random_charging_stations(N, M, 1)
risk_pertime = rand(1, N, M)  # 1 time step, values between 0 and 1
function generate_random_ground_stations(N::Int, M::Int, num_stations::Int)
    selected = rand(1:N*M, num_stations)
    return [(div(i-1, M)+1, mod(i-1, M)+1) for i in selected]
end
GroundStations = generate_random_ground_stations(N, M, 5)

L = 20

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

c[(121,122)] = L*4


function milp_relaxed(risk_pertime,n_drones,ChargingStation,GroundStations,max_battery_time, L)

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # ---------- variables ----------

    x = @variable(model, [i in GridpointsDrones, j in GridpointsDrones, s = 1:n_drones], Bin)
    y = @variable(model, [i in TransitGridpoints, s = 1:n_drones], Bin)

    # ---------- constraints ----------

    #Each gridpoint is visited at most once by one drone
    @constraint(model, [i in TransitGridpoints], sum(y[i,s] for s in 1:n_drones) <= 1)

    #Each vehicle starts its path at charging station and ends at charging station, modeled as different charging stations
    @constraint(model, [s=1:n_drones], sum(x[Begin_CS,i,s] for i in GridpointsDrones_end) == 1)
    @constraint(model, [s=1:n_drones], sum(x[i,End_CS,s] for i in GridpointsDrones_begin) == 1)

    # No incoming arc to Begin_CS
    @constraint(model, [j in GridpointsDrones, s in 1:n_drones], x[j, Begin_CS, s] == 0)

    # No outgoing arc from End_CS
    @constraint(model, [j in GridpointsDrones, s in 1:n_drones], x[End_CS, j, s] == 0)


    # #Ensure connectivity of each tour            
    @constraint(model, [k in TransitGridpoints, s=1:n_drones], 
                sum(x[k,i,s] for i in setdiff(GridpointsDrones_end,[k])) == y[k,s])
    @constraint(model, [k in TransitGridpoints, s=1:n_drones], 
                sum(x[j,k,s] for j in setdiff(GridpointsDrones_begin,[k])) == y[k,s])

    #Impose travel length restriction
    for s in 1:n_drones
        @constraint(model,
            sum(
                c[(i, j)] * x[i, j, s]
                for i in GridpointsDrones_begin
                for j in GridpointsDrones_end
                if i != j && haskey(c, (i, j))
            ) <= L
        )
    end
    
    # @objective(model, Max, 0)
    @objective(model, Max, sum(risk_pertime[1,GridpointsDronesDetecting[k]...]*(y[k,s]) for k in TransitGridpoints for s in 1:n_drones)) 

    return model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y
end

model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y = milp_relaxed(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)
optimize!(model)





# --------------- PLOT ROUTES ---------------

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

    for i in GridpointsDrones, j in GridpointsDrones
        if value(x[i, j, s]) > 0.8
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
draw(PNG("drones_side_by_side_2.png", 300 * n_drones, 500), side_by_side_plot)

# Show plot
display(side_by_side_plot)



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

subtours_per_drone = subtours(n_drones, GridpointsDrones, Begin_CS, End_CS)
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


routes = greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)
obj_value = compute_objective_greedy(routes, coords, risk_pertime, Begin_CS, End_CS)
println("Objective value = $obj_value")



# --------------- CUTTING PLANE ALGORITHM ---------------

function extract_tours_from_solution(x, valid_drones, GridpointsDrones, Begin_CS, End_CS)
    tours = Dict{Int, Vector{Int}}()

    for s in valid_drones
        route = Int[]
        current_node = Begin_CS

        while true
            push!(route, current_node)
            next_nodes = [j for j in GridpointsDrones if value(x[current_node, j, s]) > 0.5]
            
            if isempty(next_nodes)
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

function CPA(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)

    # Initial upper bound (UB) and initial greedy lower bound (LB)
    UB = sum(risk_pertime[1, GridpointsDronesDetecting[k]...] for k in TransitGridpoints)
    routes = greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)
    best_LB = compute_objective_greedy(routes, coords, risk_pertime, Begin_CS, End_CS)

    println("Initial LB from greedy = $best_LB, UB = $UB")

    iteration = 1
    model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y = milp_relaxed(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)
    println("\n--- Iteration $iteration ---")
    optimize!(model)


    while iteration <= 15
        while true
            opt_val = objective_value(model)
            if opt_val < UB
                UB = opt_val
                print(UB)
            end

            # --------------- PLOT THE GRAPH ---------------
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

                for i in GridpointsDrones, j in GridpointsDrones
                    if value(x[i, j, s]) > 0.8
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
            draw(PNG("drones_side_by_side_iter$(iteration).png", 300 * n_drones, 500), side_by_side_plot)

            # Show plot
            display(side_by_side_plot)

            subtours_per_drone = subtours(n_drones, GridpointsDrones, Begin_CS, End_CS, x)
            print(subtours_per_drone)

            valid_drones = [s for s in 1:n_drones if isempty(subtours_per_drone[s])]
            invalid_drones = [s for s in 1:n_drones if !isempty(subtours_per_drone[s])]

            #If there are valid tours (i.e., when there is a drone without any subtours) then
            if !isempty(valid_drones)
                println("Drones with valid tours: ", valid_drones)       
                profit = sum(risk_pertime[1, GridpointsDronesDetecting[k]...] * value(y[k, s]) 
                                for k in TransitGridpoints, s in valid_drones)
                if profit > best_LB
                    best_LB = profit
                    print(best_LB)
                    routes = Vector{Vector{Int}}(undef, n_drones)  # Re-initialize to avoid stale entries

                    routes_valid = extract_tours_from_solution(x, valid_drones, GridpointsDrones, Begin_CS, End_CS)
                    for s in 1:n_drones
                        routes[s] = s in valid_drones ? routes_valid[s] : []
                    end
                end
            end

            if isempty(invalid_drones) || best_LB == UB
                return routes, UB, x, y
            else
            #If there are drones with subtours, then
                # Add GSEC constraints to eliminate detected subtours
                for s in invalid_drones
                    for S in subtours_per_drone[s]
                        outside_S = setdiff(GridpointsDrones,S)
                        if isempty(outside_S)
                            continue  # avoid error on empty set
                        end
                        delta_plus = [(u, v) for u in S for v in outside_S if c[(u,v)] < L]
                        if !isempty(delta_plus)
                                @constraint(model, sum(x[u, v, s] for (u, v) in delta_plus) >= 2 * sum(y[i, s] for i in S))
                            # for i in S
                            # # for i in setdiff(outside_S,121:122)
                            # #     @assert isa(i, Int) "i is not an integer: got i = $i of type $(typeof(i))"
                            # #     if i in axes(y, 1) && s in axes(y, 2)
                            #         # @constraint(model, sum(x[u, v, s] for (u, v) in delta_plus) >= 1)
                            #         # @show size(y)
                            #         # @show typeof(y)
                            #         @constraint(model, sum(x[u, v, s] for (u, v) in delta_plus) >= 2 * y[i, s])
                            # end
                            #     end
                            # end                        
                        end
                    end
                end

                optimize!(model)
                iteration += 1
            end
        end
    end
end

routes, UB, x, y = CPA(risk_pertime,n_drones,ChargingStation,GroundStations,max_battery_time, L)

