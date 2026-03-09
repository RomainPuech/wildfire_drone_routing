#
# PSO benchmark on MiniTractDataset/AugustComplexFire instance
# Run from julia/:  julia test_pso_august_complex_fire.jl
#

using Dates
using Random

println("="^60)
println("PSO BENCHMARK – AugustComplexFire (with mask)")
println("="^60)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

println("Loading Julia modules...")
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")
println("Modules loaded successfully!")
println()

# ---------------------------------------------------------------------------
# Problem instance: MiniTractDataset/AugustComplexFire (same as test_pso_real_instance_boundary.jl)
# ---------------------------------------------------------------------------

burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

if !isfile(mask_filename)
    println("WARNING: Rescaled mask not found at $mask_filename")
    println("Trying alternative path...")
    mask_filename = "../MiniTractDataset/mask_rescaled_103x112_63substeps.npy"
end

charging_stations = [(28, 36), (12,26), (8, 28), (20, 27)] 
ground_stations = [(8, 26), (9, 26), (8, 27), (9, 27), (9, 28), (8, 29), (9, 29)]

n_drones = 4
max_battery_time = 63
t = 0                      # starting time index (Python-style); Julia will use t+1
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

# Zero-out risk at depots and ground assets
for cs in charging_stations
    risk_pertime[:, cs[1], cs[2]] .= 0
end
for gs in ground_stations
    risk_pertime[:, gs[1], gs[2]] .= 0
end

_, N, M = size(risk_pertime)

# Load mask if provided
if mask_filename !== nothing && isfile(mask_filename)
    mask = load_mask(mask_filename)
    I = [(x, y) for x in 1:N for y in 1:M if mask[x,y] == 1]
    blocked = Set([(x, y) for x in 1:N for y in 1:M if mask[x,y] != 1])
else
    I = [(x, y) for x in 1:N for y in 1:M]
    blocked = Set{Tuple{Int,Int}}()
end

# Use BFS-based get_drone_gridpoints when mask is provided
if mask_filename !== nothing && isfile(mask_filename)
    GridpointsDrones_set, _ = get_drone_gridpoints_BFS(charging_stations, floor(max_battery_time/2), I, N, M)
else
    GridpointsDrones_set = get_drone_gridpoints(charging_stations, floor(max_battery_time/2), I)
end

# Cap transmission range to 30x30 square centered on each charging station
transmission_range_size = 60
half_range = transmission_range_size ÷ 2  # 15 cells on each side
filtered_gridpoints = Set{Tuple{Int,Int}}()
original_count = length(GridpointsDrones_set)
for point in GridpointsDrones_set
    # Check if point is within 30x30 square of any charging station
    within_range = false
    for (cs_x, cs_y) in charging_stations
        x_min = cs_x - (half_range - 1)  # 14 cells to the left (center + 15 to right = 30 total)
        x_max = cs_x + half_range          # 15 cells to the right
        y_min = cs_y - (half_range - 1)   # 14 cells below
        y_max = cs_y + half_range          # 15 cells above
        
        if x_min <= point[1] <= x_max && y_min <= point[2] <= y_max
            within_range = true
            break
        end
    end
    if within_range
        push!(filtered_gridpoints, point)
    end
end
GridpointsDrones_set = filtered_gridpoints
filtered_count = length(GridpointsDrones_set)
println("Transmission range capped to $(transmission_range_size)x$(transmission_range_size) square per charging station")
println("  Original accessible points: $original_count")
println("  After 30x30 filter: $filtered_count")

GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, charging_stations)
GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))

println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")
println()

# ---------------------------------------------------------------------------
# Run PSO tests: OPT_OFF and LINF_COST_NO_OPT
# ---------------------------------------------------------------------------

"""
Print timing results for a given PSO run (shared by both OPT_ON / OPT_OFF and summary).
"""
function print_pso_timing(label, best_LB, routes_count, elapsed,
                          shift_stats, swap_stats,
                          split_sparse_stats, split_sparse_profit_stats,
                          split_dense_stats)
    println("[$label] best_LB = $(round(best_LB, digits=6)), routes = $routes_count")
    println("[$label] elapsed = $(round(elapsed, digits=3))s")

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

function run_pso(label; shift_filter::Bool, swap_filter::Bool, use_linf_cost::Bool = false, incremental::Bool = false, cost_matrix::Bool = false, live_zone::Bool = false, lazy_dead::Bool = false)
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = shift_filter
    ENABLE_SWAP_BLOCKING_FILTER[] = swap_filter
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = incremental
    ENABLE_COST_MATRIX[] = cost_matrix
    ENABLE_LIVE_ZONE_FILTER[] = live_zone
    ENABLE_LAZY_DEAD_FILTER[] = lazy_dead
    ENABLE_SPARSE_SPLIT[] = true
    reset_boundary_stats!()

    println("[$label] shift_filter=$(shift_filter), swap_filter=$(swap_filter), use_linf_cost=$(use_linf_cost), incremental=$(incremental), cost_matrix=$(cost_matrix), live_zone=$(live_zone), lazy_dead=$(lazy_dead)")
    Random.seed!(1234)

    best_LB_ref = Ref{Float64}(0.0)
    routes_count_ref = Ref{Int}(0)

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
            swarm_size = swarm_size,
            use_linf_cost = use_linf_cost,
        )
        best_LB_ref[] = best_LB
        routes_count_ref[] = length(routes)
    end

    shift_stats, swap_stats, split_sparse_stats, split_sparse_profit_stats, split_dense_stats = get_boundary_stats()
    incr_swap_stats = INCREMENTAL_SWAP_STATS[]
    incr_shift_stats = INCREMENTAL_SHIFT_STATS[]

    # Print timing for this run
    print_pso_timing(
        label,
        best_LB_ref[],
        routes_count_ref[],
        elapsed,
        shift_stats,
        swap_stats,
        split_sparse_stats,
        split_sparse_profit_stats,
        split_dense_stats,
    )

    if incremental
        println("[$label] === Incremental Stats ===")
        if incr_swap_stats.calls > 0
            println("[$label] incr_swap: $(incr_swap_stats.candidates) candidates, " *
                    "$(incr_swap_stats.skipped_blocking) skipped_blocking, " *
                    "$(incr_swap_stats.skipped_dp) skipped_dp, " *
                    "$(incr_swap_stats.evaluated) evaluated, " *
                    "$(incr_swap_stats.accepted) accepted, " *
                    "$(incr_swap_stats.calls) calls, " *
                    "$(round(incr_swap_stats.time, digits=3))s total")
        end
        if incr_shift_stats.calls > 0
            println("[$label] incr_shift: $(incr_shift_stats.candidates) candidates, " *
                    "$(incr_shift_stats.skipped_filter) skipped_filter, " *
                    "$(incr_shift_stats.skipped_dp) skipped_dp, " *
                    "$(incr_shift_stats.evaluated) evaluated, " *
                    "$(incr_shift_stats.accepted) accepted, " *
                    "$(incr_shift_stats.calls) calls, " *
                    "$(round(incr_shift_stats.time, digits=3))s total")
        end
        println()
    end

    return (
        best_LB = best_LB_ref[],
        routes = routes_count_ref[],
        elapsed = elapsed,
        shift_stats = shift_stats,
        swap_stats = swap_stats,
        split_sparse_stats = split_sparse_stats,
        split_sparse_profit_stats = split_sparse_profit_stats,
        split_dense_stats = split_dense_stats,
        incr_swap_stats = incr_swap_stats,
        incr_shift_stats = incr_shift_stats,
    )
end

# ---------------------------------------------------------------------------
# Run: OPT_OFF (no optimizations, sparse split) — commented out
# ---------------------------------------------------------------------------
# opt_off_stats = run_pso("OPT_OFF"; shift_filter = false, swap_filter = false)

# ---------------------------------------------------------------------------
# Run: BEST (all optimizations: boundary + incremental + cost matrix + lazy dead swap-only)
# ---------------------------------------------------------------------------

best_stats = run_pso("BEST"; shift_filter = true, swap_filter = true, incremental = true, cost_matrix = true, lazy_dead = true)

# # ---------------------------------------------------------------------------
# # Previous configs (commented out to save time)
# # ---------------------------------------------------------------------------
# opt_on_stats = run_pso("OPT_ON"; shift_filter = true, swap_filter = true)
# linf_cost_stats = run_pso("LINF_COST"; shift_filter = true, swap_filter = true, use_linf_cost = true)
# incr_stats = run_pso("INCREMENTAL"; shift_filter = true, swap_filter = true, incremental = true)
# incr_linf_stats = run_pso("INCREMENTAL_LINF"; shift_filter = true, swap_filter = true, use_linf_cost = true, incremental = true)
# cm_stats = run_pso("COST_MATRIX"; shift_filter = true, swap_filter = true, cost_matrix = true)
# cm_linf_stats = run_pso("CM_LINF"; shift_filter = true, swap_filter = true, use_linf_cost = true, cost_matrix = true)
# cm_incr_stats = run_pso("CM_INCR"; shift_filter = true, swap_filter = true, incremental = true, cost_matrix = true)
# cm_incr_linf_stats = run_pso("CM_INCR_LINF"; shift_filter = true, swap_filter = true, use_linf_cost = true, incremental = true, cost_matrix = true)
# lz_incr_stats = run_pso("LZ_INCR"; shift_filter = true, swap_filter = true, incremental = true, cost_matrix = true, live_zone = true)
# lz_incr_linf_stats = run_pso("LZ_INCR_LINF"; shift_filter = true, swap_filter = true, use_linf_cost = true, incremental = true, cost_matrix = true, live_zone = true)

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

println("="^70)
println("==== SUMMARY ====")
println("="^70)

total_split_best = best_stats.split_sparse_stats.calls + best_stats.split_sparse_profit_stats.calls + best_stats.split_dense_stats.calls
println("  BEST profit:    $(round(best_stats.best_LB, digits=6))")
println("  BEST elapsed:   $(round(best_stats.elapsed, digits=2))s")
println("  BEST routes:    $(best_stats.routes)")
println("  Split calls:    $(total_split_best)")
println()

best_swap_cands = best_stats.incr_swap_stats.candidates > 0 ? best_stats.incr_swap_stats.candidates : best_stats.swap_stats.candidates
best_shift_cands = best_stats.incr_shift_stats.candidates > 0 ? best_stats.incr_shift_stats.candidates : best_stats.shift_stats.candidates
println("--- Candidate throughput ---")
println("  Swap candidates:  $(best_swap_cands)")
println("  Shift candidates: $(best_shift_cands)")
println()

println("--- Detailed timing ---")
println()

print_pso_timing(
    "BEST (SUMMARY)",
    best_stats.best_LB,
    best_stats.routes,
    best_stats.elapsed,
    best_stats.shift_stats,
    best_stats.swap_stats,
    best_stats.split_sparse_stats,
    best_stats.split_sparse_profit_stats,
    best_stats.split_dense_stats,
)

if best_stats.incr_swap_stats.calls > 0
    println("[BEST] === Incremental Stats ===")
    incr_swap = best_stats.incr_swap_stats
    incr_shift = best_stats.incr_shift_stats
    println("[BEST] incr_swap: $(incr_swap.candidates) cands, $(incr_swap.skipped_blocking) skip_block, $(incr_swap.skipped_dp) skip_dp, $(incr_swap.evaluated) eval, $(incr_swap.accepted) accept, $(incr_swap.calls) calls, $(round(incr_swap.time, digits=3))s")
    println("[BEST] incr_shift: $(incr_shift.candidates) cands, $(incr_shift.skipped_filter) skip_filter, $(incr_shift.skipped_dp) skip_dp, $(incr_shift.evaluated) eval, $(incr_shift.accepted) accept, $(incr_shift.calls) calls, $(round(incr_shift.time, digits=3))s")
    println()
end

println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
