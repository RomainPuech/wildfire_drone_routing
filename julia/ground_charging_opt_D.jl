# 2025, Formulations and code by Danique De Moor, adapted by Romain Puech

# TODO strict typing

import Pkg
Pkg.add("IJulia")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Distances")
Pkg.add("MAT")
Pkg.add("Plots")
Pkg.add("FFMPEG")
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("Clustering")
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP



function L_inf_distance(a,b)
    """
    Returns the L-infinity distance between a and b in R^n
    """
    return maximum(abs.(a .- b))
end

function neighbors(i, I=nothing)
    """
    Returns the L-infinity norm-neighbors of i in Z^n, intersected with feasible set I if provided
    (returns the feasible cells directly around i)
    """
    n = length(i)
    neighbors_list = []
    
    # Generate all possible combinations of -1, 0, 1 in n dimensions
    for moves in Iterators.product(fill((-1,0,1), n)...)
        if any(m != 0 for m in moves)  # Skip the point itself
            point = [i[j] + moves[j] for j in 1:n]
            if I === nothing || point in I # if the point belongs to the original set I
                push!(neighbors_list, Tuple(i[j] + moves[j] for j in 1:n))
            end
        end
    end
    
    return neighbors_list
end

function phi(x,y)
    return L_inf_distance(i, k) <= 1 ? 1 : 0
end

# Detection in gridpoint i of charging station placed at k, when a drone can fly at maximum T_max time steps.
function gamma(i,k,T_max)
    return max(0,1-L_inf_distance(i,k)/ceil(T_max/2)) 
end

function load_burn_map(filename)
    # Add .txt extension if not present
    if !endswith(filename, ".txt")
        filename = filename * ".txt"
    end
    
    try
        # Read the file
        lines = readlines(filename)
        
        # Read starting time from first line
        starting_time = parse(Int, lines[1])
        
        # Read dimensions from second line
        T, N = parse.(Int, split(lines[2], ","))
        
        # Initialize burn_map array
        burn_map = zeros(Float64, (T, N, N))
        
        # Current line index (skip first two header lines)
        line_idx = 3
        
        # Read burn_map data
        for t in 1:T
            for i in 1:N
                # Parse row values
                row = parse.(Float64, split(lines[line_idx], ","))
                burn_map[t, i, :] = row
                line_idx += 1
            end
        end
        
        return burn_map, starting_time
    catch e
        error("Error loading burn map: $e")
    end
end

function load_parameters(risk_pertime_file)
    risk_pertime, _ = load_burn_map(risk_pertime_file)
    T, N, _ = size(risk_pertime)
    M = N
    I = [(x, y) for x in 1:N for y in 1:M]
    if I_prime === nothing
        I_prime = I
    end

    if I_second === nothing
        I_second = I
    end

    return (risk_pertime=risk_pertime, T=T, N=N, M=M, I=I, I_prime=I_prime, I_second=I_second)
end

function neighbors_chargingcoverage(k, I, coveragearea_maxdistance)
    return [j for j in I if L_inf_distance(k,j) <= coveragearea_maxdistance]
end

function ground_charging_opt_model_grid(risk_pertime_file, n_ground_stations, n_charging_stations)
    """
    Returns the locations of all ground sensors and charging stations.

    Input:
    I: Set of all positions
    I_prime: Set of feasible positions for ground sensors. Default to I
    I_second: Set of all feasible positions for charging stations. Defaults to I
    risk_pertime_dir: Folder containing burn maps
    """
    time_start = time_ns() / 1e9 

    params = load_parameters(risk_pertime_file)

    risk_pertime = params.risk_pertime
    I = params.I 
    I_prime = params.I_prime
    I_second = params.I_second

    charging_mindistance = 4 #min distance between two charging stations
    T_max = 5 #max time a drone can be in the air
    C_min = 10
        
    phi = Dict((i, k) => (L_inf_distance(i, k) <= charging_mindistance ? 1 : 0) for i in I, k in I)
    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables

    # risk_pertime is not indexed by I but rather is 2 dimensional with coordinates given by i, so we have to splat i with '...'
    # @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(phi[i,k]*risk_pertime[1,i...]*y[k] for k in I_second for i in neighbors(k,I))) # 2a
    # modifyed 2a: we say that ground sensors can only detect ON their cell, while charging stations can detect L inf distance of ceil(T_max/2) away with probability gamma(i,k,T_max) 
    @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(sum(gamma(i,k,T_max)*risk_pertime[1,k...]*y[k] for k in I_second) for i in neighbors_chargingcoverage(k,I,ceil(T_max/2))))

    @constraint(model, [i in intersect(I_prime, I_second)], x[i] + y[i] <= 1) # 2b

    # We don't need the following since stations can only see their own cell
    # for i in I_prime
    #     @constraint(model, x[i] + sum(phi[i,k]*y[k] for k in I_second) <= b) # 2c
    # end

    
    for i in I_second
        @constraint(model, sum(phi[i,k]*y[k] for k in I_second) <= 1) # 2d
    end

    #@constraint(model, nu*sum(y[i] for i in I_second) + omega*sum(x[i] for i in I_prime) <= B) # 2e, budget constraint
    # modified: contraint on the total number of ground/charging stations instead of a budget.
    @constraint(model, sum(x) <= n_ground_stations)
    @constraint(model, sum(y) <= n_charging_stations) # modified: contraint on the total number of ground/charging stations instead of a budget.

    # additional constraint: each charging station provides coverage for at least C_min unique gridpoints not covered by ground sensors
    @constraint(model, [k in I_second], sum(coverage_area[k, j] * (1 - x[j]) for j in neighbors_chargingcoverage(k, I, floor(T_max/2))) >= C_min * y[k])

    optimize!(model)

    selected_x_indices = [i for i in I_prime if value(x[i]) ≈ 1]
    selected_y_indices = [i for i in I_second if value(y[i]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds")

    return selected_x_indices, selected_y_indices
end

#COMMENTS DANIQUE 
# - in constraint 2d, we can remove phi[i,k] as we assume charging stations only detect in own gridpoint.
# - Ground stations & charging stations placement based on population, accessibility, safety. In simulation not possible (is this true?) but on real dataset it is.
# - In our case, if we don't assume charging stations have detection range outside of it's own grid, we should add constraint stating minimum distance between charging stations or similar.
# - Charging stations now do not take into account proximity to ground stations. Ways to fix this
#   --> Multiply charging station part in objective with weight factor that prioritizes locations where drones are most needed (i.e., where ground sensors cannot cover). How?
#   --> Ensure each charging station provides coverage for at least C_min unique grid points in neighborhood not covered by ground sensors