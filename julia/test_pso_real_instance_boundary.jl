# Real-instance PSO benchmark with boundary optimizations on/off
# Run from julia/: julia test_pso_real_instance_boundary.jl

using Dates
using Random

println("="^60)
println("REAL INSTANCE PSO BENCHMARK (BOUNDARY OPT ON/OFF)")
println("="^60)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

# Include the necessary files
println("Loading Julia modules...")
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")
println("Modules loaded successfully!")
println()

# ---------------------------------------------------------------------------
# Problem instance (same as test_top_masked.jl)
# ---------------------------------------------------------------------------
burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

if !isfile(mask_filename)
    println("WARNING: Rescaled mask not found at $mask_filename")
    println("Trying alternative path...")
    mask_filename = "../MiniTractDataset/mask_rescaled_103x112_63substeps.npy"
end

charging_stations = [(28, 36), (66, 32)]
ground_stations = [(8, 26), (9, 26), (8, 27), (9, 27), (8, 28), (9, 28), (8, 29), (9, 29)]

n_drones = 2
max_battery_time = 63
t = 0
initial_drone_positions = Vector{Tuple{Int,Int}}()

# PSO caps
max_time = 60.0
max_iterations = 300
swarm_size = 10

println("Burnmap file: $burnmap_filename")
println("Mask file: $mask_filename")
println("Charging stations: $charging_stations")
println("Ground stations: $ground_stations")
println("n_drones: $n_drones, max_battery_time: $max_battery_time")
println("PSO caps: max_time=$(max_time)s, max_iterations=$max_iterations, swarm_size=$swarm_size")
println()

# ---------------------------------------------------------------------------
# Build inputs similarly to compute_TOP_plan_multiple_depots
# ---------------------------------------------------------------------------
println("Loading burn map and mask...")
t_idx = t + 1
risk_pertime = load_burn_map(burnmap_filename)
risk_pertime = risk_pertime[t_idx:end, :, :]

for cs in charging_stations
    risk_pertime[:, cs[1], cs[2]] .= 0
end
for gs in ground_stations
    risk_pertime[:, gs[1], gs[2]] .= 0
end

_, N, M = size(risk_pertime)
if mask_filename !== nothing
    mask = load_mask(mask_filename)
    I = [(x, y) for x in 1:N for y in 1:M if mask[x,y] == 1]
    blocked = Set([(x, y) for x in 1:N for y in 1:M if mask[x,y] != 1])
else
    I = [(x, y) for x in 1:N for y in 1:M]
    blocked = Set{Tuple{Int,Int}}()
end

if mask_filename !== nothing
    GridpointsDrones_set, _ = get_drone_gridpoints_BFS(charging_stations, floor(max_battery_time/2), I, N, M)
else
    GridpointsDrones_set = get_drone_gridpoints(charging_stations, floor(max_battery_time/2), I)
end

GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, charging_stations)
GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))

println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")
println()

# ---------------------------------------------------------------------------
# Run PSO twice: optimizations ON vs OFF
# ---------------------------------------------------------------------------

function run_pso(label; shift_filter::Bool, swap_filter::Bool)
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = shift_filter
    ENABLE_SWAP_BLOCKING_FILTER[] = swap_filter
    reset_boundary_stats!()
    println("[$label] shift_filter=$(shift_filter), swap_filter=$(swap_filter)")
    Random.seed!(1234)
    elapsed = @elapsed begin
        routes, best_LB = get_PSO_solution_multiple_depots(
            risk_pertime,
            GridpointsDronesDetecting,
            charging_stations,
            n_drones,
            max_battery_time,
            initial_drone_positions,
            blocked;
            use_greedy_init = true,
            max_time = max_time,
            max_iterations = max_iterations,
            swarm_size = swarm_size
        )
        println("[$label] best_LB = $(round(best_LB, digits=6)), routes = $(length(routes))")
    end
    println("[$label] elapsed = $(round(elapsed, digits=3))s")
    shift_stats, swap_stats, split_sparse_stats, split_sparse_profit_stats, split_dense_stats = get_boundary_stats()
    if shift_stats.calls > 0
        avg_shift = shift_stats.time / shift_stats.calls
        skip_pct = shift_stats.candidates == 0 ? 0.0 : 100.0 * shift_stats.skipped / shift_stats.candidates
        println("[$label] shift avg time: $(round(avg_shift, digits=6))s")
        println("[$label] shift skipped: $(shift_stats.skipped) / $(shift_stats.candidates) ($(round(skip_pct, digits=2))%)")
    end
    if swap_stats.calls > 0
        avg_swap = swap_stats.time / swap_stats.calls
        skip_pct = swap_stats.candidates == 0 ? 0.0 : 100.0 * swap_stats.skipped / swap_stats.candidates
        println("[$label] swap avg time: $(round(avg_swap, digits=6))s")
        println("[$label] swap skipped: $(swap_stats.skipped) / $(swap_stats.candidates) ($(round(skip_pct, digits=2))%)")
    end
    if split_sparse_stats.calls > 0
        avg_split = split_sparse_stats.time / split_sparse_stats.calls
        println("[$label] split_sparse avg time: $(round(avg_split, digits=6))s")
        println("[$label] split_sparse calls: $(split_sparse_stats.calls)")
    end
    if split_sparse_profit_stats.calls > 0
        avg_split = split_sparse_profit_stats.time / split_sparse_profit_stats.calls
        println("[$label] split_sparse_profit avg time: $(round(avg_split, digits=6))s")
        println("[$label] split_sparse_profit calls: $(split_sparse_profit_stats.calls)")
    end
    if split_dense_stats.calls > 0
        avg_split = split_dense_stats.time / split_dense_stats.calls
        println("[$label] split_dense avg time: $(round(avg_split, digits=6))s")
        println("[$label] split_dense calls: $(split_dense_stats.calls)")
    end
    println()
end

run_pso("OPT_ON"; shift_filter=true, swap_filter=true)
run_pso("OPT_OFF"; shift_filter=false, swap_filter=false)

println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
