"""
Test script to demonstrate PSO warm start for MILP solver in TOP
"""

include("TOP.jl")

function test_warm_start()
    println("=== Testing PSO Warm Start for MILP Solver ===")
    
    # Use smaller parameters for quick test
    Random.seed!(42)
    n_drones = 2
    max_battery_time = 10
    N = 6
    M = 6
    L = max_battery_time
    
    # Generate test data
    ChargingStation = generate_random_charging_stations(N, M, 1)
    GroundStations = generate_random_ground_stations(N, M, 3)
    risk_pertime = rand(1, N, M)
    
    println("Grid size: $(N)x$(M)")
    println("Number of drones: $n_drones")
    println("Max battery time: $max_battery_time")
    println("Charging station: $(ChargingStation[1])")
    println("Ground stations: $GroundStations")
    
    # Set up the problem similar to TOP.jl
    H, N, M = size(risk_pertime)
    if H == 1
        println("Duplicating risk per time for 100 time steps")
        risk_pertime = repeat(risk_pertime, 100, 1, 1)
        H = 100
    end
    
    ChargingStations = ChargingStation
    ChargingStations = [(Int(x), Int(y)) for (x,y) in ChargingStations]
    GroundStations = [(Int(x), Int(y)) for (x,y) in GroundStations]
    
    I = [(x, y) for x in 1:N for y in 1:M]
    GridpointsDrones_set = get_drone_gridpoints(ChargingStations, floor(max_battery_time/2), I)
    GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, ChargingStations)
    GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))
    
    coords = deepcopy(GridpointsDronesDetecting)
    push!(coords, ChargingStations[1])  # For Begin_CS
    push!(coords, ChargingStations[1])  # For End_CS
    
    Begin_CS = length(GridpointsDronesDetecting) + 1
    End_CS = length(GridpointsDronesDetecting) + 2
    TransitGridpoints = 1:length(GridpointsDronesDetecting)
    
    println("Number of transit gridpoints: $(length(GridpointsDronesDetecting))")
    println("Begin_CS: $Begin_CS, End_CS: $End_CS")
    
    # Create cost matrix
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
    
    c[(Begin_CS, End_CS)] = L*4
    
    # Test 1: Run greedy baseline
    println("\n=== Greedy Baseline ===")
    greedy_routes = greedy_TOP_multiple_drones(risk_pertime, coords, Begin_CS, End_CS, max_battery_time, n_drones, c)
    greedy_profit = compute_objective_greedy(greedy_routes, coords, risk_pertime, Begin_CS, End_CS)
    println("Greedy profit: $(round(greedy_profit, digits=3))")
    
    # Test 2: Run PSO
    println("\n=== PSO Solution ===")
    pso_routes, pso_profit = get_PSO_solution(risk_pertime, GridpointsDronesDetecting, ChargingStation, n_drones, max_battery_time)
    println("PSO profit: $(round(pso_profit, digits=3))")
    println("PSO improvement: $(round(((pso_profit - greedy_profit) / greedy_profit) * 100, digits=1))%")
    
    # Test 3: Create MILP model and test warm start
    println("\n=== Testing Warm Start ===")
    model, x, GridpointsDrones, GridpointsDronesDetecting, coords, Begin_CS, End_CS, TransitGridpoints, y = milp_relaxed(risk_pertime, n_drones, ChargingStation, GroundStations, max_battery_time, L)
    
    println("PSO routes to be used for warm start:")
    for (s, route) in enumerate(pso_routes)
        if length(route) >= 2
            route_coords = [coords[i] for i in route if i <= length(coords)]
            println("  Drone $s: $route -> coords: $route_coords")
        else
            println("  Drone $s: Empty route")
        end
    end
    
    # Apply warm start
    warm_start_with_solution!(model, x, y, pso_routes, n_drones, GridpointsDrones, TransitGridpoints)
    
    # Test optimization with warm start
    println("\n=== MILP Optimization with Warm Start ===")
    start_time = time()
    optimize!(model)
    end_time = time()
    
    if termination_status(model) == MOI.OPTIMAL
        opt_val = objective_value(model)
        println("MILP optimal value: $(round(opt_val, digits=3))")
        println("Optimization time: $(round(end_time - start_time, digits=3)) seconds")
        
        # Compare results
        println("\nComparison:")
        println("  Greedy: $(round(greedy_profit, digits=3))")
        println("  PSO: $(round(pso_profit, digits=3))")
        println("  MILP (with PSO warm start): $(round(opt_val, digits=3))")
        
        if opt_val >= pso_profit * 0.999  # Account for numerical precision
            println("✅ MILP found solution at least as good as PSO warm start!")
        else
            println("⚠️  MILP solution is worse than PSO warm start (this shouldn't happen)")
        end
        
    else
        println("MILP optimization failed or was not optimal")
        println("Termination status: $(termination_status(model))")
    end
    
    return pso_routes, pso_profit, greedy_profit
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_warm_start()
end