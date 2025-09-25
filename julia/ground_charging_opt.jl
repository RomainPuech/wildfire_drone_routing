# println("installing packages")
# import Pkg
# Pkg.add("IJulia")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Distances")
# Pkg.add("MAT")
# Pkg.add("Plots")
# Pkg.add("FFMPEG")
# Pkg.add("JuMP")
# Pkg.add("Gurobi")
# Pkg.add("Clustering")
# Pkg.add("NPZ")
# Pkg.add("NearestNeighbors")
# Pkg.add("Statistics")
using SparseArrays, Pkg, MAT, CSV, DataFrames, Distances, SparseArrays, Random, Plots, Gurobi, JuMP, NPZ, Statistics

include("helper_functions.jl")

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

function NEW_SENSOR_STRATEGY(risk_pertime_file, N_grounds, N_charging)

    time_start = time_ns() / 1e9 

    # Load burn map and extract dimensions
    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = size(risk_pertime)
    println("risk_pertime_file=", risk_pertime_file)
    println("T=", T)
    println("N=", N)
    println("M=", M)

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]

    # Precompute average wildfire risk for each cell to avoid recalculating it multiple times
    avg_risk = zeros(N, M)
    for i in 1:N, j in 1:M
        avg_risk[i,j] = (1/T) * sum(risk_pertime[t,i,j] for t in 1:T)
    end

    # prefilter: keep only cells with risk > 90% of other cells
    #first_quartile_risk = quantile(vec(avg_risk), 0.0)
    # I_prime = [(i, j) for i in 1:N, j in 1:M if avg_risk[i,j] > 0.0] # >first_quartile_risk
    
    I_prime = [(i, j) for i in 1:N, j in 1:M if avg_risk[i,j] > 0.0] # Feasible grid points for ground stations
    I_second = I_prime #Feasible grid points for charging stations

    # print how many cells are discarded
    # println("Number of cells discarded: ", length(I) - length(I_prime))

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Variables 
    x = @variable(model, [i in I_prime], Bin) # ground sensor variables
    y = @variable(model, [i in I_second], Bin) # charging station variables

    # Objective - use precomputed average risk
    @objective(model, Max, 
        sum(avg_risk[i...] * x[i] for i in I_prime) + 
        sum(avg_risk[i...] * y[i] for i in I_second))

    # Constraints
    @constraint(model, [i in I_prime], x[i] + y[i] <= 1) # Can't place both devices at the same location
    @constraint(model, sum(x) <= N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(y) <= N_charging) # Capacity constraint on the charging stations

    close_pairs = [(i, j) for i in I_second for j in I_second if i != j && maximum(abs.(i .- j)) <= 10]  # Precompute valid (i, j) pairs where L∞ distance ≤ 10 
    @constraint(model, [(i, j) in close_pairs], y[i] + y[j] <= 1) # Spatial exclusion constraint between two charging stations
    cs_pairs = [(i, j) for (i, j) in close_pairs if j in I_prime]  
    @constraint(model, [(i,j) in cs_pairs], y[i] + x[j] <= 1) # Spatial exclusion constraint between a ground sensor and a charging station

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    #Extract selected sensor and charging station placements
    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(x[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(y[i]) > 0.5]

    println("selected_x_indices=", selected_x_indices)
    println("selected_y_indices=", selected_y_indices)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices
end



function Max_Coverage_Kernel(static_map_file, N_grounds, N_charging, n_drones, kernel, kernel_size_x, kernel_size_y) # next variant: we add how many drones are in the area

    # kernel is a map (dx,dy) -> value that gives you the coverage if you are dx,dy away from the charging station, |dx| <= kernel_size_x, |dy| <= kernel_size_y

    time_start = time_ns() / 1e9 

    # Load burn map and extract dimensions
    static_map = load_burn_map(static_map_file)
    T, N, M = size(static_map)

    println("static_map_file=", static_map_file)
    if T != 1 
        println("static_map_file must be a single time step")
        avg_risk = zeros(N, M)
        for i in 1:N, j in 1:M
            avg_risk[i,j] = (1/10) * sum(static_map[t,i,j] for t in 1:10)
        end
        static_map = avg_risk
    else
        static_map = static_map[1,:,:]
    end

    # Grid points
    I = [(x, y) for x in 1:N for y in 1:M]
    
    I_prime = I # Feasible grid points for ground stations
    I_second = I_prime #Feasible grid points for charging stations
    I_common = intersect(I_prime, I_second)
    I_charging_only = setdiff(I_second, I_common)
    I_ground_only = setdiff(I_prime, I_common)



    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Variables 
    xg = @variable(model, [i in I_prime], Bin) # ground sensor variables
    xc = @variable(model, [i in I_second], Bin) # charging station variables
    nc = @variable(model, [i in I_second], Int) # number of drones from charging station i
    theta = @variable(model, [i in I]) # coverage variables

    # Objective - maximize coverage
    @objective(model, Max, 
        sum(static_map[point...] * theta[point] for point in I))

    # Placement constraints
    @constraint(model, [i in I_common], xg[i] + xc[i] <= 1) # exclusion constraint on both ground sensors and charging stations
    @constraint(model, sum(xg) == N_grounds) # Capacity constraint on the ground sensors
    @constraint(model, sum(xc) == N_charging) # Capacity constraint on the charging stations   
    @constraint(model, sum(nc) == n_drones) # we use all the drones

    # linking constraint
    @constraint(model, [i in I_second], nc[i]<=n_drones*xc[i]) # number of drones in the area of the ground sensor is the sum of the charging stations in the area

    # Coverage constraints
    @constraint(model, [i in I], 0 <= theta[i] <= 1) # coverage variables are between 0 and 1
    @constraint(model, [i in I_ground_only], theta[i] >= xg[i]) # coverage constraint on ground sensors
    
    # HERE WE ASSUME I = I_prime for efficiency, just change how the sum is indexed on depending on what is the most efficient in your case.
    # coverage = zeros(N,M)
    # for (i_point,j_point) in I
    #     for dx in max(-i_point+1,-kernel_size_x):min(N-i_point+1,kernel_size_x), dy in max(-j_point+1,-kernel_size_y):min(M-j_point+1,kernel_size_y)
    #         coverage_percentage = kernel[(-dx,-dy)] # - because here we compte the delta from point to the charging station and kernel is from charging station to point
    #         coverage[i_point, j_point] += coverage_percentage * xc[i_point]
    #     end
    # end
    # @constraint(model, [i in I], theta[i] >= coverage[i]) # coverage constraint on charging stations
    # Single constraint for charging station coverage
    @constraint(model, [(i_point,j_point) in I], 
        theta[(i_point,j_point)] <= sum(
            kernel[(-dx,-dy)] * xc[(i_point+dx,j_point+dy)]
            for dx in max(-i_point+1,-kernel_size_x):min(N-i_point,kernel_size_x)
            for dy in max(-j_point+1,-kernel_size_y):min(M-j_point,kernel_size_y)
            #if (i_point+dx,j_point+dy) in I_second
        ) + xg[(i_point,j_point)]
    )

    println("Took ", (time_ns() / 1e9) - time_start, " seconds to create model")

    optimize!(model)

    #Extract selected sensor and charging station placements
    selected_x_indices = [(i[1]-1, i[2]-1) for i in I_prime if value(xg[i]) > 0.5] 
    selected_y_indices = [(i[1]-1, i[2]-1) for i in I_second if value(xc[i]) > 0.5]

    # println("selected_x_indices=", selected_x_indices)
    # println("selected_y_indices=", selected_y_indices)

    println("Took ", (time_ns() / 1e9) - time_start, " seconds total")
    
    return selected_x_indices, selected_y_indices
end



# using ImageFiltering  # For the equivalent of scipy.ndimage.convolve

# function count_paths_convolution(N, M, origin, n)
#     # Initialize the dynamic programming array
#     dp = zeros(Float64, N, M)
#     dp[origin[1], origin[2]] = 1.0  # Note: Julia is 1-based indexing

#     # Create the 3x3 kernel
#     kernel = centered(ones(Float64, 3, 3))  # Use centered kernel

#     # Apply convolution n times
#     for _ in 1:n
#         dp = imfilter(dp, kernel, Fill(0.0))  # Use Fill(0.0) instead of "constant"
#     end

#     # Get the origin value for normalization
#     origin_value = dp[origin[1], origin[2]]

#     # Create the mapping dictionary
#     mapping = Dict{Tuple{Int,Int}, Float64}()
#     for x in 1:N, y in 1:M
#         mapping[(x - origin[1], y - origin[2])] = dp[x,y]/origin_value
#     end

#     return mapping
# end


# # Test parameters
# N = 60  # Grid size
# M = 60
# N_grounds = 5  # Number of ground stations
# N_charging = 3  # Number of charging stations

# # Generate the kernel using the translated function
# origin = (N÷2, M÷2)  # Center point
# n = 20  # Number of steps for the convolution
# kernel = count_paths_convolution(N, M, origin, n)
# println("kernel=", kernel)

# # Get kernel size
# kernel_size_x = 20
# kernel_size_y = 20

# println("kernel_size_x=", kernel_size_x)
# println("kernel_size_y=", kernel_size_y)

# # Call the optimization function
# ground_locations, charging_locations = Max_Coverage_Kernel(
#     "WideDataset/0016_03070/burn_map_rescaled_26x30_substeps_63.npy",
#     N_grounds,
#     N_charging,
#     kernel,
#     kernel_size_x,
#     kernel_size_y
# )

# # Print results
# println("Ground station locations: ", ground_locations)
# println("Charging station locations: ", charging_locations)

# # Optional: Visualize the results
# using Plots

# # Create a heatmap of the static map
# static_map = load_burn_map("WideDataset/0016_03070/burn_map_rescaled_26x30_substeps_63.npy")
# heatmap(static_map[1,:,:], title="Burn Map with Station Placements")

# # Add ground stations
# for (x, y) in ground_locations
#     scatter!([x+1], [y+1], label="Ground Station", color=:blue, markersize=8)
# end

# # Add charging stations
# for (x, y) in charging_locations
#     scatter!([x+1], [y+1], label="Charging Station", color=:red, markersize=8)
# end

# # Save the plot
# savefig("station_placements.png")