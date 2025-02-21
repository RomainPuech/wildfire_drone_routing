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

## ------------------- LOAD THE DATA --------------------
# ------------------------------------------------------

file_path_population = "/Users/demoor/Dropbox (MIT)/Rubicon/Wildfire project/Data/grid_20km_risk_map_CA.csv"
df_population = CSV.read(file_path_population, DataFrame);
delete!(df_population,[28])

file_path = "/Users/demoor/Documents/MIT Projects/Wildfire project/Matlab files/Data/risk_grid.csv"
df = CSV.read(file_path, DataFrame);

file_path_population = "/Users/demoor/Dropbox (MIT)/Rubicon/Wildfire project/Data/grid_20km_risk_map_CA.csv"
df_population = CSV.read(file_path_population, DataFrame);
delete!(df_population,[28])

file_path = "/Users/demoor/Documents/MIT Projects/Wildfire project/Matlab files/Data/risk_grid.csv"
df = CSV.read(file_path, DataFrame);

population = df_population[:,4]
urban = findall(p -> p >= 450, population)
rural = findall(p -> p < 450, population)
urban_high = findall(p -> p >= 15000, population)

x_grid = df[:,1];
y_grid = df[:,2];
risk = df[:,3:13];
wfday = df[:,14:23];
x_grid = x_grid/10000;
y_grid = y_grid/10000;
xy_grid = hcat(x_grid,y_grid);

x_grid_urban = x_grid[urban]
y_grid_urban = y_grid[urban]
x_grid_rural = x_grid[rural]
y_grid_rural = y_grid[rural]
x_grid_urban_high = x_grid[urban_high]
y_grid_urban_high = y_grid[urban_high]
xy_grid_rural = hcat(x_grid_rural, y_grid_rural)
xy_grid_urban = hcat(x_grid_urban, y_grid_urban);

num_time_steps = 5
risk_pertime = risk[:, 1] .+ risk[:, 2:(num_time_steps+1)]


# ------------------- GROUND SENSORS AND CHARGING STATIONS --------------------
# -----------------------------------------------------------------------------

## ------------------- define the parameters --------------------

distance_treshold = 5;
coverage_treshold = 0.8

I = 

##  ------------------- define the optimization model ------------------- 

function ground_charging_opt_model_2()
    model = Model(Gurobi.Optimizer)
    x = @variable(model, [i in urban], Bin) #ground sensor variables
    c = @variable(model, [i in setdiff(I,urban_high)], Bin) #charging station variables

    @constraint(model, [i in setdiff(urban, urban_high)], x[i] + c[i] <= 1)
    for i in setdiff(urban)
        for k in setdiff(I, urban_high)
            if neighbors_cs[i,k]
                @constraint(model, x[i] + w[i,k]*c[k] <= 1.6)
            end
        end
    end

    for i in setdiff(I,urban_high)
        @constraint(model, sum(w[i,k]*c[k] for k in setdiff(I,urban_high)) <= 1)
    end

    # @constraint(model, sum(c[i] for i in setdiff(I,urban_high)) <= 12)
    # @constraint(model, sum(x[i] for i in urban) <= 300)
    @constraint(model, 50*sum(c[i] for i in setdiff(I,urban_high)) + sum(x[i] for i in urban) <= 700) #budget constraint
    @objective(model, Max, sum(risk_pertime[k,1]*x[k] for k in urban) + sum(w[i,k]*risk_pertime[i,1]*c[k] for k in setdiff(I,urban_high), i in setdiff(I,urban_high) if neighbors_cs[i,k]))
    
    optimize!(model)
    
    return x, c
end

## 

x, c = ground_charging_opt_model_2();

##  ------------------- plot the ground sensors and the charging stations on the grid ------------------- 



plot(x_grid, y_grid, seriestype=:scatter, marker=:circle, color=:white, markerstrokecolor=:black, markercolor=:white, label="", size=(800,1000))
    plot!(x_grid_rural, y_grid_rural, seriestype=:scatter, marker =:diamond, markercolor=:white, markerstrokecolor=:yellow, label="rural area")
    plot!(x_grid_urban_high, y_grid_urban_high, seriestype=:scatter, marker=:diamond, markercolor=:white, markerstrokecolor=:orange, label="high urban area")

    x_location_x = Float64[]
    x_location_y = Float64[]
    c_location_x = Float64[]
    c_location_y = Float64[]
    x_vals = JuMP.value.(x)
    charging_vals = JuMP.value.(c)
    for i in urban
        if x_vals[i] >= 0.9
            push!(x_location_x, x_grid[i])
            push!(x_location_y, y_grid[i])
        end
    end
    if !isempty(x_location_x) && !isempty(x_location_y)
        plot!(x_location_x, x_location_y, seriestype=:scatter, label = "ground sensors", markercolor=:red, color = :red, lw = 3, marker = :hexagon, markersize=:7)
    end
    for i in setdiff(I,urban_high)
        if charging_vals[i] >= 0.9
            push!(c_location_x, x_grid[i])
            push!(c_location_y, y_grid[i])
        end
    end
    if !isempty(c_location_x) && !isempty(c_location_y)
        plot!(c_location_x, c_location_y, seriestype=:scatter, label = "charging stations", markercolor =:green, color = :green, lw = 3, marker = :hexagon, markersize =:7)
    end

## ----------------------------
    c_location_idx = Int64[]
for i in setdiff(I,urban_high)
    if charging_vals[i] >= 0.9
        push!(c_location_idx, i)
    end
end
x_location_idx = Int64[]
for i in urban
    if x_vals[i] >= 0.9
        push!(x_location_idx, i)
    end
end

## ---------------------------

# ------------------- DRONE ROUTING, METHOD 1 --------------------
# -----------------------------------------------------------------------------

## ------------------- define the parameters and indexes used for the optimization model --------------------

GroundStations = x_location_idx
ChargingStations = c_location_idx

# define the parameters
num_time_steps = 10;
num_drones = 3;
D_max = 10;
T_max = 4;
D_delta = 4;
M = 15;

risk_pertime = risk[:, 1] .+ risk[:, 2:(num_time_steps+1)]

D = pairwise(Euclidean(), xy_grid');
distance_treshold = 3
coverage_contribution_treshold = 3
coverage_treshold = 0.7

#neighbor matrix  where entries are 1 only if the distance from i to j is smaller than coverage_contribution_treshold
neighbors = sparse(D .<= distance_treshold);
coverage_contribution = sparse(D .<= coverage_contribution_treshold);

# detection matrix
w = @. 1 / ((1+D)^2);
w_sparse = sparse(w .>= 0.1);


I = 1:num_grid_points;
J = 1:num_grid_points;
T = 1:num_time_steps;
T1 = 1:(num_time_steps +1)
S = 1:num_drones;
M1 = D_max + 1
M2 = T_max + 1

B_min = 0.3
B_max = 1.0
## ------------------- define the areas of which drones are allowed to fly, i.e., within d_cs distance of charging station & plot --------------------
D_subset = D[:, ChargingStations]
cs_prox = findall(x -> minimum(x) <= ceil(Int,0.6*D_max), eachrow(D_subset))
restricted_gridpoints_drones = union(urban_high,setdiff(I,cs_prox))
allowed_gridpoints_drones = setdiff(cs_prox, urban_high)

x_grid_cs_prox = x_grid[allowed_gridpoints_drones]
y_grid_cs_prox = y_grid[allowed_gridpoints_drones]

NonChargingStations_allowed = setdiff(allowed_gridpoints_drones,ChargingStations)
GroundStations_allowed = intersect(allowed_gridpoints_drones,GroundStations)

plot!(x_grid_cs_prox, y_grid_cs_prox, seriestype=:scatter, marker=:diamond, markercolor=:white, markerstrokecolor=:blue, label="within proximity charging stations")


## ------------------- define the optimization model --------------------
BigM = 10

function drone_routing_maxcoverage(risk_pertime_file,n_drones,n_ground_stations,n_charging_stations)

        time_start = time_ns() / 1e9 

        selected_x_indices, selected_y_indices = ground_charging_opt_model_grid(risk_pertime_file, n_ground_stations, n_charging_stations)

        ChargingStations = selected_y_indices #coming from ground_charging_opt
        GroundStations = selected_x_indices #coming from ground_charging_opt
        GridpointsDrones # = allowed_gridpoints_drones
        GridpointsDronesDetecting # = NonChargingstations_allowed

        params = load_parameters(risk_pertime_file)

        risk_pertime = params.risk_pertime
        T = params.T

        model = Model(Gurobi.Optimizer)
        set_silent(model)
        #Defining the variables
        a = @variable(model, [i in GridpointsDrones, t in 1:T, s in n_drones], Bin)
        c = @variable(model, [i in ChargingStations, t in 1:T, s in n_drones], Bin)
        b = @variable(model, [t in 1:T, s in n_drones])
    
        @objective(model, Max, sum(sum(risk_pertime[t,k...]*x1[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))


        #Defining the constraints
        @constraint(model, [i in GridpointsDrones, t in 1:T], sum(a[i,t,s] for s in 1:n_drones) <= 1)
        @constraint(model, [t in 1:T, s in 1:n_drones], sum(a[i,t,s] for i in GridpointsDrones) + sum(c[i,t,s] for i in ChargingStations) == 1)
        @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:S], c[j,t+1,s] + a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if neighbors[i,j]) + c[j,t,s])
        @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in 1:n_drones], a[j,t+1,s] <= sum(a[i,t,s] for i in GridpointsDrones if neighbors[i,j]))
        @constraint(model, [j in ChargingStations, t in 1:T-1, s in 1:n_drones], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if neighbors[i,j]) + c[j,t+1,s])
        @constraint(model, [j in GridpointsDronesDetecting, t in 1:T-1, s in S], a[j,t,s] <= sum(a[i,t+1,s] for i in GridpointsDrones if neighbors[i,j]))
        @constraint(model, [t in 1:T, s in 1:n_drones], B_min <= b[t,s] <= B_max)
        @constraint(model, [t in 1:T-1, s in 1:n_drones], b[t+1,s] <= b[t,s] - 0.1*sum(a[i,t,s] for i in GridpointsDrones) + B_max*sum(c[i,t,s] for i in ChargingStations))
        # @constraint(model, [t in 1:num_time_steps-1, s in S], b[t+1,s] >= B_max - (1 - sum(c1[i,t,s] for i in ChargingStations)) * BigM)
        @constraint(model, [t in 1:T, s in 1:n_drones], sum(c[i,t,s] for i in ChargingStations) >= 1 - b[t,s]/B_min)
        @constraint(model, [s in 1:n_drones], sum(c[i,1,s] for i in ChargingStations) == 1)
        @constraint(model, [s in 1:n_drones], b[1,s] == B_max)
        @constraint(model, [s in 1:n_drones], b[T,s] >= B_min)
        # @constraint(model, [i in ChargingStations, t in 2:num_time_steps-1, s in S], c1[i,t,s] + c1[i,t+1,s] <= 1)
        @constraint(model, [s in 1:n_drones], sum(c[i,T,s] for i in ChargingStations) == 1)


        @objective(model, Max, sum(sum(risk_pertime[t,k...]*a[k,t,s] for k in GridpointsDrones, t in 1:T) for s in 1:n_drones) + 0.0001*sum(b[t,s] for t in 1:T, s in 1:n_drones))
        optimize!(model)

        solver_time = MOI.get(model, MOI.SolveTimeSec())

        selected_fly_indices = [(i,t,s) for i in GridpointsDronesDetecting, t in 1:T, s in 1:n_drones if value(a[i,t,s]) ≈ 1]
        selected_charge_indices = [(i,t,s) for i in ChargingStations, t in 1:T, s in 1:n_drones if value(c1[i,t,s]) ≈ 1]
        battery_level = value(b);

        println("Took ", (time_ns() / 1e9) - time_start, " seconds")

        return selected_fly_indices, selected_charge_indices, battery_level

    end
    
# -----D: change neighbors[i,j] definition, or take as input?

##

x1, c1, b = drone_routing_maxcoverage(risk_pertime, S, T, T1, allowed_gridpoints_drones, ChargingStations, NonChargingStations_allowed);

x1_vals = JuMP.value.(x1)
c1_vals = JuMP.value.(c1)
b_vals = JuMP.value.(b)

##

anim = @animate for t in 1:num_time_steps
    plot(x_grid, y_grid, seriestype=:scatter, marker=:circle, color=:white, markerstrokecolor=:black, markercolor=:white, label="", size=(800,1000))

    x_location_x = Float64[]
    x_location_y = Float64[]
    c_location_x = Float64[]
    c_location_y = Float64[]
    x_vals = JuMP.value.(x)
    c_vals = JuMP.value.(c)
    for i in urban
        if x_vals[i] >= 0.9
            push!(x_location_x, x_grid[i])
            push!(x_location_y, y_grid[i])
        end
    end
    if !isempty(x_location_x) && !isempty(x_location_y)
        plot!(x_location_x, x_location_y, seriestype=:scatter, label = "ground sensors", markercolor=:green, color = :green, lw = 3, marker = :hexagon, markersize=:5)
    end
    for i in setdiff(I,urban_high)
        if c_vals[i] >= 0.9
            push!(c_location_x, x_grid[i])
            push!(c_location_y, y_grid[i])
        end
    end
    if !isempty(c_location_x) && !isempty(c_location_y)
        plot!(c_location_x, c_location_y, seriestype=:scatter, label = "", markercolor =:red, color = :red, lw = 3, marker = :hexagon, markersize=:5)
    end

    plot!(x_grid_cs_prox, y_grid_cs_prox, seriestype=:scatter, marker=:diamond, markercolor=:white, markerstrokecolor=:blue, label="within proximity charging stations")

        # Loop over the drones
        for s in S
            # Store the drone's trajectory for flying (red) and charging (green)
            drone_flying_x = Float64[]
            drone_flying_y = Float64[]
            drone_charging_x = Float64[]
            drone_charging_y = Float64[]
            
            # Find where the drone is flying or charging at time t
            for i in allowed_gridpoints_drones
                if x1_vals[i, t, s] >= 0.9  # Drone is flying
                    push!(drone_flying_x, x_grid[i])
                    push!(drone_flying_y, y_grid[i])
                end
            end
            for i in ChargingStations
                if c1_vals[i, t, s] >= 0.9  # Drone is charging
                    push!(drone_charging_x, x_grid[i])
                    push!(drone_charging_y, y_grid[i])
                end
            end
            
            # Plot the drone's path when flying (red) and charging (green)
            if !isempty(drone_flying_x) && !isempty(drone_flying_y)
                plot!(drone_flying_x, drone_flying_y, label = "", color = :blue, lw = 2, marker = :star, markersize=:15)
            end
            if !isempty(drone_charging_x) && !isempty(drone_charging_y)
                plot!(drone_charging_x, drone_charging_y, label = "", color = :red, lw = 2, marker = :star, markersize=:9)
            end
        end
        
    title!("Time period: $t")
end


    # Save the animation as a gif
    gif(anim, "drone_flight_charging_animation.gif", fps = 1)  


##


