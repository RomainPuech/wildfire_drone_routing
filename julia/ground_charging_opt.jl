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
Pkg.add("NPZ")
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ


function L_inf_distance(a,b)
    """
    Returns the L-infinity distance between a and b in R^n
    """
    return return maximum(abs.(a .- b))
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

function load_burn_map(filename)
    
    try
        # Read the file
        println("Loading burn map from $filename")
        burn_map = npzread(filename)
        println("Burn map loaded")
        println("Burn map: $burn_map")
        return burn_map
    catch e
        error("Error loading burn map: $e")
    end
end


function ground_charging_opt_model_grid(risk_pertime_file, N_grounds, N_charging, I_prime = nothing, I_second = nothing,)
    """
    Returns the locations of all ground sensors and charging stations.

    Input:
    I: Set of all positions
    I_prime: Set of feasible positions for ground sensors. Default to I
    I_second: Set of all feasible positions for charging stations. Defaults to I
    risk_pertime_dir: Folder containing burn maps
    """
    # Print current working directory contents
    # println("Current working directory: ", pwd())
    # println("Directory contents:")
    # for (root, dirs, files) in walkdir(pwd())
    #     println("Directory: ", root)
    #     for dir in dirs
    #         println("  Dir: ", dir)
    #     end
    #     for file in files
    #         println("  File: ", file)
    #     end
    # end
    time_start = time_ns() / 1e9 

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)

    nu = 1
    omega = 1
    b = 1.6
    B = 700 # total budget

    I = [(x, y) for x in 1:N for y in 1:M]


    if I_prime === nothing
        I_prime = I
    end

    if I_second === nothing
        I_second = I
    end
    
    phi = Dict((i, k) => (L_inf_distance(i, k) <= 1 ? 1 : 0) for i in I, k in I)

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables

    # risk_pertime is not indexed by I but rather is 2 dimensional with coordinates given by i, so we have to splat i with '...'
    # @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(phi[i,k]*risk_pertime[1,i...]*y[k] for k in I_second for i in neighbors(k,I))) # 2a
    # modifyed 2a: we say that charging stations / ground sensors can only detect ON their cell
    @objective(model, Max, sum(risk_pertime[1,i...]*x[i] for i in I_prime) + sum(risk_pertime[1,k...]*y[k] for k in I_second))

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
    @constraint(model, sum(x) <= N_grounds)
    @constraint(model, sum(y) <= N_charging) # modified: contraint on the total number of ground/charging stations instead of a budget.

    optimize!(model)

    selected_x_indices = [i for i in I_prime if value(x[i]) ≈ 1]
    selected_y_indices = [i for i in I_second if value(y[i]) ≈ 1]

    println("Took ", (time_ns() / 1e9) - time_start, " seconds")

    return selected_x_indices, selected_y_indices
end