#
# PSO benchmark on WideDataset/0321_03136 using Gaussian max-coverage placement
# Run from julia/:  julia test_pso_gaussian_0321_03136.jl
#

using Dates
using Random
using JSON

println("="^60)
println("PSO BENCHMARK – WideDataset/0321_03136 (Gaussian placement)")
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
# Problem instance: WideDataset/0321_03136 with Gaussian max-coverage layout
# ---------------------------------------------------------------------------

const DATASET_DIR = "../WideDataset/0321_03136"

burnmap_filename = joinpath(DATASET_DIR, "static_risk_whp_rescaled_13x13_63substeps.npy")
gaussian_log_filename = joinpath(
    DATASET_DIR,
    "logs",
    "SensorPlacementMaxCoverageGaussianTime_static_risk_whp_rescaled_13x13_63substeps_13N_13M_8ground_2charge.json",
)

println("Burnmap file: $burnmap_filename")
println("Gaussian placement log: $gaussian_log_filename")

# Load Gaussian max-coverage placement (0-based coordinates) and convert to 1-based Julia coords
println("Loading Gaussian max-coverage placement...")
placement_log = JSON.parsefile(gaussian_log_filename)

raw_ground = placement_log["ground_sensor_locations"]
raw_charging = placement_log["charging_station_locations"]

ground_stations = [(Int(p[1]) + 1, Int(p[2]) + 1) for p in raw_ground]
charging_stations = [(Int(p[1]) + 1, Int(p[2]) + 1) for p in raw_charging]

println("Charging stations (1-based): $charging_stations")
println("Ground stations   (1-based): $ground_stations")
println()

n_drones = 2
max_battery_time = 63
t = 0                      # starting time index (Python-style); Julia will use t+1
initial_drone_positions = Vector{Tuple{Int,Int}}()

# PSO caps
max_time = 60.0
max_iterations = 300
swarm_size = 10

println("n_drones: $n_drones, max_battery_time: $max_battery_time")
println("PSO caps: max_time=$(max_time)s, max_iterations=$max_iterations, swarm_size=$swarm_size")
println()

# ---------------------------------------------------------------------------
# Build inputs similarly to compute_TOP_plan_multiple_depots
# ---------------------------------------------------------------------------

println("Loading risk map...")
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

# No explicit mask / blocked cells for this test
blocked = Set{Tuple{Int,Int}}()

# Feasible grid points (no mask)
I = [(x, y) for x in 1:N for y in 1:M]

# Reachable grid points from charging stations within half the battery
GridpointsDrones_set = get_drone_gridpoints(charging_stations, floor(max_battery_time / 2), I)
GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, charging_stations)
GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))

println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")
println()

# ---------------------------------------------------------------------------
# Run PSO twice: boundary optimizations ON vs OFF
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

function run_pso(label; shift_filter::Bool, swap_filter::Bool, use_linf_cost::Bool = false)
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = shift_filter
    ENABLE_SWAP_BLOCKING_FILTER[] = swap_filter
    reset_boundary_stats!()

    println("[$label] shift_filter=$(shift_filter), swap_filter=$(swap_filter), use_linf_cost=$(use_linf_cost)")
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

    return (
        best_LB = best_LB_ref[],
        routes = routes_count_ref[],
        elapsed = elapsed,
        shift_stats = shift_stats,
        swap_stats = swap_stats,
        split_sparse_stats = split_sparse_stats,
        split_sparse_profit_stats = split_sparse_profit_stats,
        split_dense_stats = split_dense_stats,
    )
end

# Commented out for now
# opt_on_stats = run_pso("OPT_ON"; shift_filter = true, swap_filter = true)
opt_off_stats = run_pso("OPT_OFF"; shift_filter = false, swap_filter = false)

# ---------------------------------------------------------------------------
# Third run: PSO without sparse split optimization (force dense split everywhere)
# COMMENTED OUT FOR NOW
# ---------------------------------------------------------------------------

# @eval begin
#     # Override profit-only entry point to use dense split
#     function fast_split_multiple_depots(
#         permutation::Vector{Int},
#         pso_multiple_depots::PSOiA_TOP_multiple_depots,
#     )
#         profit, _routes = fast_split_with_routes_multiple_depots(permutation, pso_multiple_depots)
#         return profit
#     end

#     # Override sparse split (with particle) to delegate to dense split
#     function fast_split_sparse(
#         permutation::Vector{Int},
#         particle::Particle,
#         pso::PSOiA_TOP_multiple_depots,
#     )
#         profit, routes = fast_split_with_routes_multiple_depots(permutation, pso)
#         empty_intervals = empty_tour_intervals()
#         return profit, routes, empty_intervals
#     end

#     # Override sparse split (mapping computed internally) to delegate to dense split
#     function fast_split_sparse(
#         permutation::Vector{Int},
#         pso::PSOiA_TOP_multiple_depots,
#     )
#         profit, routes = fast_split_with_routes_multiple_depots(permutation, pso)
#         empty_intervals = empty_tour_intervals()
#         return profit, routes, empty_intervals
#     end

#     # Override sparse profit-only splits to delegate to dense split
#     function fast_split_sparse_profit(
#         permutation::Vector{Int},
#         particle::Particle,
#         pso::PSOiA_TOP_multiple_depots,
#     )
#         profit, _routes = fast_split_with_routes_multiple_depots(permutation, pso)
#         return profit
#     end

#     function fast_split_sparse_profit(
#         permutation::Vector{Int},
#         pso::PSOiA_TOP_multiple_depots,
#     )
#         profit, _routes = fast_split_with_routes_multiple_depots(permutation, pso)
#         return profit
#     end
# end

# no_sparse_stats = run_pso("NO_SPARSE"; shift_filter = true, swap_filter = true)

# ---------------------------------------------------------------------------
# Second run: PSO with L-infinity cost matrix (uses sparse split and boundary optimizations)
# COMMENTED OUT FOR NOW
# ---------------------------------------------------------------------------

# linf_cost_stats = run_pso("LINF_COST"; shift_filter = true, swap_filter = true, use_linf_cost = true)

# ---------------------------------------------------------------------------
# Third run: PSO with L-infinity cost matrix WITHOUT boundary optimizations
# (for comparison with OPT_OFF)
# ---------------------------------------------------------------------------

linf_cost_no_opt_stats = run_pso("LINF_COST_NO_OPT"; shift_filter = false, swap_filter = false, use_linf_cost = true)

println("==== SUMMARY COMPARISON (OPT_OFF / LINF_COST_NO_OPT) ====")
# Commented out for now
# print_pso_timing(
#     "OPT_ON (SUMMARY)",
#     opt_on_stats.best_LB,
#     opt_on_stats.routes,
#     opt_on_stats.elapsed,
#     opt_on_stats.shift_stats,
#     opt_on_stats.swap_stats,
#     opt_on_stats.split_sparse_stats,
#     opt_on_stats.split_sparse_profit_stats,
#     opt_on_stats.split_dense_stats,
# )
print_pso_timing(
    "OPT_OFF (SUMMARY)",
    opt_off_stats.best_LB,
    opt_off_stats.routes,
    opt_off_stats.elapsed,
    opt_off_stats.shift_stats,
    opt_off_stats.swap_stats,
    opt_off_stats.split_sparse_stats,
    opt_off_stats.split_sparse_profit_stats,
    opt_off_stats.split_dense_stats,
)
# Commented out for now
# print_pso_timing(
#     "LINF_COST (SUMMARY)",
#     linf_cost_stats.best_LB,
#     linf_cost_stats.routes,
#     linf_cost_stats.elapsed,
#     linf_cost_stats.shift_stats,
#     linf_cost_stats.swap_stats,
#     linf_cost_stats.split_sparse_stats,
#     linf_cost_stats.split_sparse_profit_stats,
#     linf_cost_stats.split_dense_stats,
# )
print_pso_timing(
    "LINF_COST_NO_OPT (SUMMARY)",
    linf_cost_no_opt_stats.best_LB,
    linf_cost_no_opt_stats.routes,
    linf_cost_no_opt_stats.elapsed,
    linf_cost_no_opt_stats.shift_stats,
    linf_cost_no_opt_stats.swap_stats,
    linf_cost_no_opt_stats.split_sparse_stats,
    linf_cost_no_opt_stats.split_sparse_profit_stats,
    linf_cost_no_opt_stats.split_dense_stats,
)

println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

