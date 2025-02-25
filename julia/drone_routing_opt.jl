# import helper_functions such as load_burn_map
include("helper_functions.jl")

function drone_routing_example(drones, batteries, risk_pertime_file, time_horizon)
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    
    # Generate random moves for each drone
    # output should have this format: [("move", (dx, dy)), ("move", (dx, dy)), ...]
    return [[("move", (rand(-5:5), rand(-5:5))) for _ in 1:length(drones)] for _ in 1:time_horizon]
end

function NEW_drone_routing_example(drones, batteries, risk_pertime_file, time_horizon)
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    
    # Generate random moves for each drone
    # output should have this format: [ [("move", (dx, dy)), ("move", (dx, dy)), ...] for each timesteps]
    return [[("move", (rand(-5:5), rand(-5:5))) for _ in 1:length(drones)] for _ in 1:time_horizon]
end