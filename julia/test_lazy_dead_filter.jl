#
# Test: All Optimizations ON vs OFF, and Lazy Dead Filter (swap only)
# Runs PSO on AugustComplexFire under multiple configurations and compares profit + speed.
#
# Configurations:
#   ALL_OFF: No boundary filters, no incremental, no cost matrix, no lazy dead
#   ALL_ON:  All optimizations ON (boundary + incremental + cost matrix + lazy dead swap)
#   NO_LAZY: All optimizations ON except lazy dead filter
#
# Run from julia/:  julia test_lazy_dead_filter.jl
#

using Dates
using Random
using Printf

println("="^70)
println("ALL OPTIMIZATIONS BENCHMARK – AugustComplexFire")
println("="^70)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
println("Loading Julia modules...")
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")
println("Modules loaded successfully!")
println()

# ---------------------------------------------------------------------------
# Problem instance: AugustComplexFire
# ---------------------------------------------------------------------------
burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

if !isfile(mask_filename)
    mask_filename = "../MiniTractDataset/mask_rescaled_103x112_63substeps.npy"
end

charging_stations = [(28, 36)]
ground_stations = [(8, 26), (9, 26), (8, 27), (9, 27), (8, 28), (9, 28), (8, 29), (9, 29)]

n_drones = 2
max_battery_time = 63
t = 0
initial_drone_positions = Vector{Tuple{Int,Int}}()

max_time = 60.0
max_iterations = 300
swarm_size = 10

println("Instance: AugustComplexFire (n≈900, k=2, L=63)")
println("PSO caps: max_time=$(max_time)s, max_iterations=$max_iterations, swarm_size=$swarm_size")
println()

# ---------------------------------------------------------------------------
# Build inputs
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

if mask_filename !== nothing && isfile(mask_filename)
    mask = load_mask(mask_filename)
    I = [(x, y) for x in 1:N for y in 1:M if mask[x,y] == 1]
    blocked = Set([(x, y) for x in 1:N for y in 1:M if mask[x,y] != 1])
else
    I = [(x, y) for x in 1:N for y in 1:M]
    blocked = Set{Tuple{Int,Int}}()
end

if mask_filename !== nothing && isfile(mask_filename)
    GridpointsDrones_set, _ = get_drone_gridpoints_BFS(charging_stations, floor(max_battery_time/2), I, N, M)
else
    GridpointsDrones_set = get_drone_gridpoints(charging_stations, floor(max_battery_time/2), I)
end

# Cap transmission range to 30x30 square
transmission_range_size = 30
half_range = transmission_range_size ÷ 2
filtered_gridpoints = Set{Tuple{Int,Int}}()
for point in GridpointsDrones_set
    for (cs_x, cs_y) in charging_stations
        x_min = cs_x - (half_range - 1)
        x_max = cs_x + half_range
        y_min = cs_y - (half_range - 1)
        y_max = cs_y + half_range
        if x_min <= point[1] <= x_max && y_min <= point[2] <= y_max
            push!(filtered_gridpoints, point)
            break
        end
    end
end
GridpointsDrones_set = filtered_gridpoints
GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, charging_stations)
GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))

println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")
println()

# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------
function run_pso_config(label;
        boundary_filters::Bool=true,
        incremental::Bool=true,
        cost_matrix::Bool=true,
        lazy_dead::Bool=true)
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = boundary_filters
    ENABLE_SWAP_BLOCKING_FILTER[] = boundary_filters
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = incremental
    ENABLE_COST_MATRIX[] = cost_matrix
    ENABLE_LIVE_ZONE_FILTER[] = false              # Always off (not recommended)
    ENABLE_LAZY_DEAD_FILTER[] = lazy_dead           # Swap-only filter
    reset_boundary_stats!()

    println("[$label] boundary=$boundary_filters, incremental=$incremental, cost_matrix=$cost_matrix, lazy_dead=$lazy_dead")
    Random.seed!(1234)

    local routes, best_LB
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
            use_linf_cost = false,
        )
    end

    shift_stats, swap_stats, split_sparse_stats, _, _ = get_boundary_stats()
    incr_swap = INCREMENTAL_SWAP_STATS[]
    incr_shift = INCREMENTAL_SHIFT_STATS[]

    println("[$label] profit = $(round(best_LB, digits=6)), elapsed = $(round(elapsed, digits=2))s, routes = $(length(routes))")

    # Combined stats (sparse or incremental depending on config)
    total_swap_cands = swap_stats.candidates + incr_swap.candidates
    total_shift_cands = shift_stats.candidates + incr_shift.candidates
    total_swap_skip = swap_stats.skipped + incr_swap.skipped_blocking
    total_shift_skip = shift_stats.skipped + incr_shift.skipped_filter

    # Per-call timing
    if incr_swap.calls > 0
        avg_swap_ms = 1000.0 * incr_swap.time / incr_swap.calls
        println("[$label] incr_swap: $(incr_swap.candidates) cands, $(incr_swap.skipped_blocking) skip_block, $(incr_swap.skipped_dp) skip_dp, $(incr_swap.evaluated) eval, $(incr_swap.accepted) acc, $(incr_swap.calls) calls, avg $(round(avg_swap_ms, digits=3))ms/call, total $(round(incr_swap.time, digits=2))s")
    end
    if incr_shift.calls > 0
        avg_shift_ms = 1000.0 * incr_shift.time / incr_shift.calls
        println("[$label] incr_shift: $(incr_shift.candidates) cands, $(incr_shift.skipped_filter) skip_filt, $(incr_shift.skipped_dp) skip_dp, $(incr_shift.evaluated) eval, $(incr_shift.accepted) acc, $(incr_shift.calls) calls, avg $(round(avg_shift_ms, digits=3))ms/call, total $(round(incr_shift.time, digits=2))s")
    end
    if swap_stats.calls > 0 && incr_swap.calls == 0
        avg_swap_ms = swap_stats.time > 0 ? 1000.0 * swap_stats.time / swap_stats.calls : 0.0
        println("[$label] sparse_swap: $(swap_stats.candidates) cands, $(swap_stats.skipped) skipped, $(swap_stats.calls) calls, avg $(round(avg_swap_ms, digits=3))ms/call, total $(round(swap_stats.time, digits=2))s")
    end
    if shift_stats.calls > 0 && incr_shift.calls == 0
        avg_shift_ms = shift_stats.time > 0 ? 1000.0 * shift_stats.time / shift_stats.calls : 0.0
        println("[$label] sparse_shift: $(shift_stats.candidates) cands, $(shift_stats.skipped) skipped, $(shift_stats.calls) calls, avg $(round(avg_shift_ms, digits=3))ms/call, total $(round(shift_stats.time, digits=2))s")
    end
    if split_sparse_stats.calls > 0
        avg_split = split_sparse_stats.time / split_sparse_stats.calls
        println("[$label] split_sparse: $(split_sparse_stats.calls) calls, avg $(round(avg_split*1e6, digits=1))μs")
    end
    println()

    return (label=label, profit=best_LB, elapsed=elapsed, routes=length(routes),
            shift_stats=shift_stats, swap_stats=swap_stats,
            incr_swap=incr_swap, incr_shift=incr_shift,
            split_calls=split_sparse_stats.calls,
            total_swap_cands=total_swap_cands, total_shift_cands=total_shift_cands,
            total_swap_skip=total_swap_skip, total_shift_skip=total_shift_skip)
end

# ---------------------------------------------------------------------------
# Warmup (small run to JIT-compile everything)
# ---------------------------------------------------------------------------
println("="^70)
println("WARMUP (short run to JIT-compile)")
println("="^70)
begin
    # Warmup ALL_OFF path
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = false
    ENABLE_SWAP_BLOCKING_FILTER[] = false
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = false
    ENABLE_COST_MATRIX[] = false
    ENABLE_LIVE_ZONE_FILTER[] = false
    ENABLE_LAZY_DEAD_FILTER[] = false
    reset_boundary_stats!()
    Random.seed!(9999)
    get_PSO_solution_multiple_depots(
        risk_pertime, GridpointsDronesDetecting, charging_stations,
        n_drones, max_battery_time, initial_drone_positions, blocked;
        use_greedy_init=true, max_time=5.0, max_iterations=5, swarm_size=3,
        use_linf_cost=false)

    # Warmup ALL_ON path
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = true
    ENABLE_SWAP_BLOCKING_FILTER[] = true
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = true
    ENABLE_COST_MATRIX[] = true
    ENABLE_LIVE_ZONE_FILTER[] = false
    ENABLE_LAZY_DEAD_FILTER[] = true
    reset_boundary_stats!()
    Random.seed!(9999)
    get_PSO_solution_multiple_depots(
        risk_pertime, GridpointsDronesDetecting, charging_stations,
        n_drones, max_battery_time, initial_drone_positions, blocked;
        use_greedy_init=true, max_time=5.0, max_iterations=5, swarm_size=3,
        use_linf_cost=false)
end
println("Warmup complete.\n")

# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------
println("="^70)
println("BENCHMARK RUNS (60s each, seed=1234)")
println("="^70)
println()

results = []

# 1) ALL OFF: no boundary filters, no incremental, no cost matrix, no lazy dead
push!(results, run_pso_config("ALL_OFF";
    boundary_filters=false, incremental=false, cost_matrix=false, lazy_dead=false))

# 2) ALL ON (without lazy dead): boundary + incremental + cost matrix
push!(results, run_pso_config("NO_LAZY";
    boundary_filters=true, incremental=true, cost_matrix=true, lazy_dead=false))

# 3) ALL ON (with lazy dead swap): boundary + incremental + cost matrix + lazy dead
push!(results, run_pso_config("ALL_ON";
    boundary_filters=true, incremental=true, cost_matrix=true, lazy_dead=true))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("="^80)
println("SUMMARY TABLE")
println("="^80)
println()
@printf("%-12s  %10s  %8s  %12s  %12s  %12s\n",
        "Config", "Profit", "Time(s)", "Split calls", "Swap cands", "Shift cands")
println("-"^80)
for r in results
    @printf("%-12s  %10.6f  %8.2f  %12d  %12d  %12d\n",
            r.label, r.profit, r.elapsed, r.split_calls,
            r.total_swap_cands, r.total_shift_cands)
end

println()
println("="^80)
println("SPEEDUP & PROFIT COMPARISON")
println("="^80)

baseline = results[1]  # ALL_OFF
for r in results[2:end]
    diff = r.profit - baseline.profit
    sign = diff >= 0 ? "+" : ""
    pct = baseline.profit > 0 ? 100.0 * diff / baseline.profit : 0.0
    @printf("%-12s vs ALL_OFF: Δprofit = %s%.6f (%s%.2f%%), split calls %dx → %dx\n",
            r.label, sign, diff, sign, pct, baseline.split_calls, r.split_calls)
end

# Also compare NO_LAZY vs ALL_ON (lazy dead effect)
no_lazy = results[2]
all_on  = results[3]
lazy_diff = all_on.profit - no_lazy.profit
lazy_sign = lazy_diff >= 0 ? "+" : ""
lazy_pct = no_lazy.profit > 0 ? 100.0 * lazy_diff / no_lazy.profit : 0.0
println()
@printf("ALL_ON vs NO_LAZY (lazy dead effect): Δprofit = %s%.6f (%s%.2f%%)\n",
        lazy_sign, lazy_diff, lazy_sign, lazy_pct)

# Swap per-call comparison for NO_LAZY vs ALL_ON
if no_lazy.incr_swap.calls > 0 && all_on.incr_swap.calls > 0
    avg_no_lazy = 1000.0 * no_lazy.incr_swap.time / no_lazy.incr_swap.calls
    avg_all_on  = 1000.0 * all_on.incr_swap.time / all_on.incr_swap.calls
    speedup = avg_no_lazy > 0 ? avg_no_lazy / avg_all_on : 0.0
    @printf("  Swap per-call: NO_LAZY=%.3fms, ALL_ON=%.3fms → %.2f× speedup\n",
            avg_no_lazy, avg_all_on, speedup)
end

println()
println("="^80)
println("CANDIDATE SKIP RATES")
println("="^80)
for r in results
    swap_pct = r.total_swap_cands > 0 ? round(100.0 * r.total_swap_skip / r.total_swap_cands, digits=1) : 0.0
    shift_pct = r.total_shift_cands > 0 ? round(100.0 * r.total_shift_skip / r.total_shift_cands, digits=1) : 0.0
    @printf("%-12s: swap %.1f%% skipped, shift %.1f%% skipped\n", r.label, swap_pct, shift_pct)
end

println()
println("Done!")
