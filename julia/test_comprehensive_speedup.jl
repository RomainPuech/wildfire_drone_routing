#
# Comprehensive PSO speedup benchmark – AugustComplexFire
#
# Compares:
#   DENSE       – Dense O(n²) split, no optimizations
#   SPARSE_ONLY – Sparse split, no other optimizations
#   BEST        – All optimizations: sparse + boundary + incremental + cost matrix + lazy dead (swap-only)
#   BEST_LINF   – Same as BEST with L∞ cost model
#
# Run from julia/:  julia test_comprehensive_speedup.jl
#

using Dates
using Random
using Printf

println("="^70)
println("COMPREHENSIVE PSO SPEEDUP BENCHMARK – AugustComplexFire")
println("="^70)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

println("Loading Julia modules...")
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")
println("Modules loaded successfully!")
println()

# ---------------------------------------------------------------------------
# Problem instance: MiniTractDataset/AugustComplexFire
# ---------------------------------------------------------------------------

burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

if !isfile(mask_filename)
    println("WARNING: Rescaled mask not found at $mask_filename")
    println("Trying alternative path...")
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

println("Instance: AugustComplexFire (103×112, mask-aware)")
println("n_drones=$n_drones, max_battery_time=$max_battery_time")
println("PSO: max_time=$(max_time)s, max_iterations=$max_iterations, swarm_size=$swarm_size")
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
println("Accessible points after 30×30 filter: $(length(GridpointsDrones_set))")

GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, charging_stations)
GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))
println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")
println()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

function print_pso_timing(label, best_LB, routes_count, elapsed,
                          shift_stats, swap_stats,
                          split_sparse_stats, split_sparse_profit_stats,
                          split_dense_stats)
    println("[$label] profit=$(round(best_LB, digits=6)), routes=$routes_count, elapsed=$(round(elapsed, digits=2))s")

    if shift_stats.calls > 0
        skip_pct = shift_stats.candidates == 0 ? 0.0 : 100.0 * shift_stats.skipped / shift_stats.candidates
        println("[$label]   shift: $(shift_stats.calls) calls, $(shift_stats.candidates) cands, $(round(skip_pct, digits=1))% skipped, $(round(shift_stats.time, digits=2))s total")
    end
    if swap_stats.calls > 0
        skip_pct = swap_stats.candidates == 0 ? 0.0 : 100.0 * swap_stats.skipped / swap_stats.candidates
        println("[$label]   swap:  $(swap_stats.calls) calls, $(swap_stats.candidates) cands, $(round(skip_pct, digits=1))% skipped, $(round(swap_stats.time, digits=2))s total")
    end
    total_split = split_sparse_stats.calls + split_sparse_profit_stats.calls + split_dense_stats.calls
    if split_sparse_stats.calls > 0
        avg = split_sparse_stats.time / split_sparse_stats.calls * 1e6
        println("[$label]   split_sparse: $(split_sparse_stats.calls) calls, avg=$(round(avg, digits=2))μs")
    end
    if split_sparse_profit_stats.calls > 0
        avg = split_sparse_profit_stats.time / split_sparse_profit_stats.calls * 1e6
        println("[$label]   split_sparse_profit: $(split_sparse_profit_stats.calls) calls, avg=$(round(avg, digits=2))μs")
    end
    if split_dense_stats.calls > 0
        avg = split_dense_stats.time / split_dense_stats.calls * 1e6
        println("[$label]   split_dense: $(split_dense_stats.calls) calls, avg=$(round(avg, digits=2))μs")
    end
    println("[$label]   total split calls: $total_split")
    println()
end

function run_pso_config(label;
        boundary::Bool, incremental::Bool, cost_matrix::Bool,
        lazy_dead::Bool, sparse_split::Bool, use_linf_cost::Bool=false)

    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = boundary
    ENABLE_SWAP_BLOCKING_FILTER[] = boundary
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = incremental
    ENABLE_COST_MATRIX[] = cost_matrix
    ENABLE_LIVE_ZONE_FILTER[] = false
    ENABLE_LAZY_DEAD_FILTER[] = lazy_dead
    ENABLE_SPARSE_SPLIT[] = sparse_split
    reset_boundary_stats!()

    println("─"^70)
    println("[$label] boundary=$boundary, incremental=$incremental, cost_matrix=$cost_matrix, " *
            "lazy_dead=$lazy_dead, sparse_split=$sparse_split, linf=$use_linf_cost")
    Random.seed!(1234)

    best_LB_ref = Ref(0.0)
    routes_ref = Ref(0)

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
        routes_ref[] = length(routes)
    end

    shift_stats, swap_stats, split_sparse_stats, split_sparse_profit_stats, split_dense_stats = get_boundary_stats()
    incr_swap = INCREMENTAL_SWAP_STATS[]
    incr_shift = INCREMENTAL_SHIFT_STATS[]

    print_pso_timing(label, best_LB_ref[], routes_ref[], elapsed,
                     shift_stats, swap_stats,
                     split_sparse_stats, split_sparse_profit_stats, split_dense_stats)

    if incremental
        println("[$label] === Incremental stats ===")
        if incr_swap.calls > 0
            skip_pct = incr_swap.candidates == 0 ? 0.0 : 100.0 * (incr_swap.skipped_blocking + incr_swap.skipped_dp) / incr_swap.candidates
            println("[$label]   incr_swap: $(incr_swap.calls) calls, $(incr_swap.candidates) cands, " *
                    "$(incr_swap.skipped_blocking) skip_block, $(incr_swap.skipped_dp) skip_dp, " *
                    "$(incr_swap.evaluated) eval, $(incr_swap.accepted) accept, $(round(incr_swap.time, digits=2))s")
        end
        if incr_shift.calls > 0
            println("[$label]   incr_shift: $(incr_shift.calls) calls, $(incr_shift.candidates) cands, " *
                    "$(incr_shift.skipped_filter) skip_filter, $(incr_shift.skipped_dp) skip_dp, " *
                    "$(incr_shift.evaluated) eval, $(incr_shift.accepted) accept, $(round(incr_shift.time, digits=2))s")
        end
        println()
    end

    total_split = split_sparse_stats.calls + split_sparse_profit_stats.calls + split_dense_stats.calls

    return (label=label, profit=best_LB_ref[], elapsed=elapsed, routes=routes_ref[],
            shift_stats=shift_stats, swap_stats=swap_stats,
            split_sparse_stats=split_sparse_stats,
            split_sparse_profit_stats=split_sparse_profit_stats,
            split_dense_stats=split_dense_stats,
            incr_swap=incr_swap, incr_shift=incr_shift,
            total_split=total_split)
end

# ---------------------------------------------------------------------------
# Warmup (small run to JIT-compile everything)
# ---------------------------------------------------------------------------

println("Warming up JIT...")
let
    ENABLE_SPARSE_SPLIT[] = true
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = true
    ENABLE_SWAP_BLOCKING_FILTER[] = true
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = true
    ENABLE_COST_MATRIX[] = true
    ENABLE_LAZY_DEAD_FILTER[] = true
    Random.seed!(9999)
    get_PSO_solution_multiple_depots(
        risk_pertime, GridpointsDronesDetecting, charging_stations,
        n_drones, max_battery_time, initial_drone_positions, blocked;
        use_greedy_init=true, max_time=5.0, max_iterations=5, swarm_size=3)
    # Also warmup dense path
    ENABLE_SPARSE_SPLIT[] = false
    ENABLE_INCREMENTAL_LOCAL_SEARCH[] = false
    ENABLE_SHIFT_IRRELEVANCE_FILTER[] = false
    ENABLE_SWAP_BLOCKING_FILTER[] = false
    ENABLE_COST_MATRIX[] = false
    ENABLE_LAZY_DEAD_FILTER[] = false
    Random.seed!(9999)
    get_PSO_solution_multiple_depots(
        risk_pertime, GridpointsDronesDetecting, charging_stations,
        n_drones, max_battery_time, initial_drone_positions, blocked;
        use_greedy_init=true, max_time=5.0, max_iterations=5, swarm_size=3)
end
println("Warmup done.")
println()

# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------

results = []

# 1) DENSE: Dense split, no optimizations
push!(results, run_pso_config("DENSE";
    boundary=false, incremental=false, cost_matrix=false,
    lazy_dead=false, sparse_split=false))

# 2) SPARSE_ONLY: Sparse split, no other optimizations
push!(results, run_pso_config("SPARSE_ONLY";
    boundary=false, incremental=false, cost_matrix=false,
    lazy_dead=false, sparse_split=true))

# 3) BEST: All optimizations (boundary + incremental + cost_matrix + lazy_dead swap-only)
push!(results, run_pso_config("BEST";
    boundary=true, incremental=true, cost_matrix=true,
    lazy_dead=true, sparse_split=true))

# 4) BEST_LINF: Same as BEST with L∞ cost
push!(results, run_pso_config("BEST_LINF";
    boundary=true, incremental=true, cost_matrix=true,
    lazy_dead=true, sparse_split=true, use_linf_cost=true))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

println()
println("="^90)
println("SUMMARY TABLE")
println("="^90)
@printf("%-14s │ %10s │ %8s │ %12s │ %12s │ %12s\n",
        "Config", "Profit", "Time(s)", "Split calls", "Swap cands", "Shift cands")
println("─"^90)
for r in results
    swap_cands = r.swap_stats.candidates > 0 ? r.swap_stats.candidates :
                 (r.incr_swap.candidates > 0 ? r.incr_swap.candidates : 0)
    shift_cands = r.shift_stats.candidates > 0 ? r.shift_stats.candidates :
                  (r.incr_shift.candidates > 0 ? r.incr_shift.candidates : 0)
    @printf("%-14s │ %10.6f │ %8.2f │ %12d │ %12d │ %12d\n",
            r.label, r.profit, r.elapsed, r.total_split, swap_cands, shift_cands)
end
println("="^90)
println()

# ---------------------------------------------------------------------------
# Speedup & profit comparisons
# ---------------------------------------------------------------------------

println("─── Speedup comparisons ───")
dense = results[1]
sparse_only = results[2]
best = results[3]
best_linf = results[4]

# Speedup in terms of split throughput (calls per second)
dense_split_rate = dense.total_split / dense.elapsed
sparse_split_rate = sparse_only.total_split / sparse_only.elapsed
best_split_rate = best.total_split / best.elapsed

println()
println("Split throughput (calls/sec):")
@printf("  DENSE:       %12.0f calls / %.2fs = %10.0f calls/s\n", dense.total_split, dense.elapsed, dense_split_rate)
@printf("  SPARSE_ONLY: %12.0f calls / %.2fs = %10.0f calls/s\n", sparse_only.total_split, sparse_only.elapsed, sparse_split_rate)
@printf("  BEST:        %12.0f calls / %.2fs = %10.0f calls/s\n", best.total_split, best.elapsed, best_split_rate)
println()

# Per-split time
if dense.split_dense_stats.calls > 0
    dense_per_split = dense.split_dense_stats.time / dense.split_dense_stats.calls * 1e6
    @printf("  DENSE avg split time:       %.2f μs\n", dense_per_split)
end
if sparse_only.split_sparse_stats.calls + sparse_only.split_sparse_profit_stats.calls > 0
    sparse_calls = sparse_only.split_sparse_stats.calls + sparse_only.split_sparse_profit_stats.calls
    sparse_time = sparse_only.split_sparse_stats.time + sparse_only.split_sparse_profit_stats.time
    @printf("  SPARSE_ONLY avg split time: %.2f μs\n", sparse_time / sparse_calls * 1e6)
end
if best.split_sparse_stats.calls + best.split_sparse_profit_stats.calls > 0
    best_calls = best.split_sparse_stats.calls + best.split_sparse_profit_stats.calls
    best_time = best.split_sparse_stats.time + best.split_sparse_profit_stats.time
    @printf("  BEST avg split time:        %.2f μs\n", best_time / best_calls * 1e6)
end
println()

# Wall-clock "effective speedup" — more split calls in same time
println("Effective speedup (split calls done in 60s):")
@printf("  SPARSE_ONLY vs DENSE:  %.2f× more split calls\n", sparse_only.total_split / max(dense.total_split, 1))
@printf("  BEST vs DENSE:         %.2f× more split calls\n", best.total_split / max(dense.total_split, 1))
@printf("  BEST vs SPARSE_ONLY:   %.2f× more split calls\n", best.total_split / max(sparse_only.total_split, 1))
println()

# Per-split speedup (dense vs sparse)
if dense.split_dense_stats.calls > 0 && (sparse_only.split_sparse_stats.calls + sparse_only.split_sparse_profit_stats.calls) > 0
    dense_avg = dense.split_dense_stats.time / dense.split_dense_stats.calls
    sparse_calls = sparse_only.split_sparse_stats.calls + sparse_only.split_sparse_profit_stats.calls
    sparse_avg = (sparse_only.split_sparse_stats.time + sparse_only.split_sparse_profit_stats.time) / sparse_calls
    @printf("Per-split speedup (sparse vs dense): %.2f×\n", dense_avg / sparse_avg)
end
println()

# Profit comparison
println("─── Profit comparison ───")
@printf("  DENSE:       %.6f\n", dense.profit)
@printf("  SPARSE_ONLY: %.6f  (Δ = %+.6f vs DENSE)\n", sparse_only.profit, sparse_only.profit - dense.profit)
@printf("  BEST:        %.6f  (Δ = %+.6f vs DENSE, Δ = %+.6f vs SPARSE_ONLY)\n",
        best.profit, best.profit - dense.profit, best.profit - sparse_only.profit)
@printf("  BEST_LINF:   %.6f  (Δ = %+.6f vs BEST)\n",
        best_linf.profit, best_linf.profit - best.profit)
println()

println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
