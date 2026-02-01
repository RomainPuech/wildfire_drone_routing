# Test script for compute_TOP_plan_multiple_depots with mask support
# Run this from the julia directory: julia test_top_masked.jl

using Dates

# Redirect all output to a log file
log_filename = "test_top_masked_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
log_file = open(log_filename, "w")
original_stdout = stdout
redirect_stdout(log_file)

println("Loading Julia modules...")
println("(All output is being written to: $log_filename)")

# Include the necessary files
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")

println("Modules loaded successfully!")

# Test parameters at OPERATIONAL SCALE (matching what run_benchmark_scenario produces)
# The benchmark.py rescales from data scale (2067x2252) to operational scale (103x112)
# using coverage_width_cells = 20 (600m coverage / 30m cell size)
burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

# Check if rescaled mask exists, if not use the one that might have been created
if !isfile(mask_filename)
    println("WARNING: Rescaled mask not found at $mask_filename")
    println("You may need to run the Python benchmark first to create the rescaled files.")
    println("Trying alternative path...")
    mask_filename = "../MiniTractDataset/mask_rescaled_103x112_63substeps.npy"
    if !isfile(mask_filename)
        println("Alternative not found either. Will try to proceed anyway.")
    end
end

# Julia uses 1-based indexing, these are the charging station locations
# At operational scale (103x112), positions from the sensor placement logs
charging_stations = [(28, 36), (66, 32)]  # Python (27,35) and (65,31) + 1
ground_stations = [(8, 26), (9, 26), (8, 27), (9, 27), (8, 28), (9, 28), (8, 29), (9, 29)]  # Example

n_drones = 2
max_battery_time = 63  # Already at operational scale
t = 0
verbose = false
initial_drone_positions = Vector{Tuple{Int,Int}}()

println("\n=== Test Parameters ===")
println("Burnmap file: $burnmap_filename")
println("Mask file: $mask_filename")
println("Charging stations (Julia indexing): $charging_stations")
println("Ground stations (Julia indexing): $ground_stations")
println("Number of drones: $n_drones")
println("Max battery time: $max_battery_time")
println("========================\n")

# First, let's test loading the mask
println("Loading mask...")
mask = load_mask(mask_filename)
println("Mask shape: $(size(mask))")
println("Mask min: $(minimum(mask)), max: $(maximum(mask))")
println("Number of feasible cells (mask==1): $(sum(mask .== 1))")
println("Number of blocked cells (mask==0): $(sum(mask .== 0))")

# Test loading the burn map
println("\nLoading burn map...")
burnmap = load_burn_map(burnmap_filename)
println("Burn map shape: $(size(burnmap))")

# Check that charging stations are in feasible cells
println("\n=== Checking charging station validity ===")
for (i, cs) in enumerate(charging_stations)
    if mask[cs[1], cs[2]] == 1
        println("Charging station $i at $cs: OK (feasible)")
    else
        println("WARNING: Charging station $i at $cs is in a BLOCKED cell!")
    end
end

# Now test the full function
println("\n=== Calling compute_TOP_plan_multiple_depots (MASKED) ===")
try
    movement_plan = compute_TOP_plan_multiple_depots(
        burnmap_filename,
        n_drones,
        charging_stations,
        ground_stations,
        max_battery_time,
        t,
        verbose,
        initial_drone_positions,
        mask_filename  # Pass the mask filename
    )
    
    println("\n=== Results ===")
    println("Movement plan length: $(length(movement_plan))")
    if length(movement_plan) > 0
        println("First few steps of movement plan:")
        for i in 1:min(5, length(movement_plan))
            println("  Step $i: $(movement_plan[i])")
        end
    end
catch e
    println("\nERROR occurred:")
    println(e)
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(log_file, exc, bt)
        println()
    end
    # Restore stdout temporarily to show error
    redirect_stdout(original_stdout)
    println("ERROR occurred - see log file: $log_filename")
    redirect_stdout(log_file)
end

println("\n=== Test WITHOUT mask (for comparison) ===")
try
    movement_plan_no_mask = compute_TOP_plan_multiple_depots(
        burnmap_filename,
        n_drones,
        charging_stations,
        ground_stations,
        max_battery_time,
        t,
        verbose,
        initial_drone_positions,
        nothing  # No mask
    )
    
    println("Movement plan (no mask) length: $(length(movement_plan_no_mask))")
catch e
    println("ERROR (no mask): $e")
end

println("\nTest completed!")

# Restore stdout and close log file
try
    redirect_stdout(original_stdout)
    close(log_file)
    println("Output written to: $log_filename")
catch
    # If something goes wrong, at least try to restore stdout
    redirect_stdout(original_stdout)
    println("Warning: Could not properly close log file, but output should be in: $log_filename")
end
