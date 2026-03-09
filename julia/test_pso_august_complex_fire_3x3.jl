#
# PSO benchmark on MiniTractDataset/AugustComplexFire instance
# with 3×3 spatial downscaling (9× fewer cells, aggregated profits)
#
# Run from julia/:  julia test_pso_august_complex_fire_3x3.jl
#

using Dates
using Random

println("="^60)
println("PSO BENCHMARK – AugustComplexFire 3×3 DOWNSCALED (with mask)")
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
# Problem instance: MiniTractDataset/AugustComplexFire
# Original grid: 103×112, downscaled to ~35×38 with 3×3 aggregation
# ---------------------------------------------------------------------------

burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

if !isfile(mask_filename)
    println("WARNING: Rescaled mask not found at $mask_filename")
    println("Trying alternative path...")
    mask_filename = "../MiniTractDataset/mask_rescaled_103x112_63substeps.npy"
end

# Original positions (will be remapped after downscaling)
orig_charging_stations = [(28, 36), (12,26), (8, 28), (20, 27)]
orig_ground_stations = [(8, 26), (9, 26), (8, 27), (9, 27), (8, 28), (9, 28), (8, 29), (9, 29)]

n_drones = 6
orig_max_battery_time = 63
t = 0

# PSO caps
max_time = 60.0
max_iterations = 300
swarm_size = 10

# ---------------------------------------------------------------------------
# 3×3 Downscaling
# ---------------------------------------------------------------------------

const KERNEL_SIZE = 3

"""Map an original 1-based coordinate to the downscaled grid."""
downscale_coord(c::Int) = div(c - 1, KERNEL_SIZE) + 1

"""Map an (x,y) tuple to the downscaled grid."""
downscale_pos(pos::Tuple{Int,Int}) = (downscale_coord(pos[1]), downscale_coord(pos[2]))

println("Loading burn map and mask...")
t_idx = t + 1
risk_pertime_orig = load_burn_map(burnmap_filename)
risk_pertime_orig = risk_pertime_orig[t_idx:end, :, :]

# Zero-out risk at original depots and ground assets BEFORE aggregation
for cs in orig_charging_stations
    risk_pertime_orig[:, cs[1], cs[2]] .= 0
end
for gs in orig_ground_stations
    risk_pertime_orig[:, gs[1], gs[2]] .= 0
end

T_orig, N_orig, M_orig = size(risk_pertime_orig)
println("Original grid: $(N_orig)×$(M_orig), T=$(T_orig)")

# Compute downscaled dimensions
N_ds = div(N_orig - 1, KERNEL_SIZE) + 1  # ceil division
M_ds = div(M_orig - 1, KERNEL_SIZE) + 1
println("Downscaled grid: $(N_ds)×$(M_ds) ($(KERNEL_SIZE)×$(KERNEL_SIZE) kernel)")

# Aggregate risk_pertime: sum over each 3×3 block
risk_pertime = zeros(T_orig, N_ds, M_ds)
for t_step in 1:T_orig
    for bx in 1:N_ds
        for by in 1:M_ds
            # Original cell range for this block
            x_start = (bx - 1) * KERNEL_SIZE + 1
            x_end   = min(bx * KERNEL_SIZE, N_orig)
            y_start = (by - 1) * KERNEL_SIZE + 1
            y_end   = min(by * KERNEL_SIZE, M_orig)
            risk_pertime[t_step, bx, by] = sum(risk_pertime_orig[t_step, x_start:x_end, y_start:y_end])
        end
    end
end

# Aggregate mask: a downscaled cell is accessible if ANY original cell in the block is accessible
if mask_filename !== nothing && isfile(mask_filename)
    mask_orig = load_mask(mask_filename)
    mask_ds = zeros(Int, N_ds, M_ds)
    for bx in 1:N_ds
        for by in 1:M_ds
            x_start = (bx - 1) * KERNEL_SIZE + 1
            x_end   = min(bx * KERNEL_SIZE, N_orig)
            y_start = (by - 1) * KERNEL_SIZE + 1
            y_end   = min(by * KERNEL_SIZE, M_orig)
            # Accessible if any cell in the block is accessible
            if any(mask_orig[x_start:x_end, y_start:y_end] .== 1)
                mask_ds[bx, by] = 1
            end
        end
    end
    I = [(x, y) for x in 1:N_ds for y in 1:M_ds if mask_ds[x, y] == 1]
    blocked = Set([(x, y) for x in 1:N_ds for y in 1:M_ds if mask_ds[x, y] != 1])
else
    I = [(x, y) for x in 1:N_ds for y in 1:M_ds]
    blocked = Set{Tuple{Int,Int}}()
end

# Remap charging stations and ground stations to downscaled coordinates
charging_stations = unique([downscale_pos(cs) for cs in orig_charging_stations])
ground_stations   = unique([downscale_pos(gs) for gs in orig_ground_stations])

# Scale battery time: each step in the downscaled grid ≈ KERNEL_SIZE original steps
max_battery_time = div(orig_max_battery_time, KERNEL_SIZE)

# Zero out risk at remapped depots and ground stations in the downscaled grid
for cs in charging_stations
    risk_pertime[:, cs[1], cs[2]] .= 0
end
for gs in ground_stations
    risk_pertime[:, gs[1], gs[2]] .= 0
end

println()
println("=== Downscaled Instance Parameters ===")
println("Charging stations: $(orig_charging_stations) → $(charging_stations)")
println("Ground stations:   $(length(orig_ground_stations)) original → $(length(ground_stations)) unique downscaled: $(ground_stations)")
println("Battery time:      $(orig_max_battery_time) → $(max_battery_time)")
println("Grid:              $(N_orig)×$(M_orig) → $(N_ds)×$(M_ds)  ($(N_orig*M_orig) → $(N_ds*M_ds) cells)")
println("n_drones: $n_drones")
println("PSO caps: max_time=$(max_time)s, max_iterations=$max_iterations, swarm_size=$swarm_size")
println()

# ---------------------------------------------------------------------------
# Build accessible gridpoints (same logic as original, on the downscaled grid)
# ---------------------------------------------------------------------------

initial_drone_positions = Vector{Tuple{Int,Int}}()

# Use BFS-based get_drone_gridpoints when mask is provided
N = N_ds
M = M_ds
if mask_filename !== nothing && isfile(mask_filename)
    GridpointsDrones_set, _ = get_drone_gridpoints_BFS(charging_stations, floor(max_battery_time / 2), I, N, M)
else
    GridpointsDrones_set = get_drone_gridpoints(charging_stations, floor(max_battery_time / 2), I)
end

# Cap transmission range (scaled down by KERNEL_SIZE)
transmission_range_size_orig = 120
transmission_range_size = div(transmission_range_size_orig, KERNEL_SIZE)  # 60/3 = 20
half_range = transmission_range_size ÷ 2

filtered_gridpoints = Set{Tuple{Int,Int}}()
original_count = length(GridpointsDrones_set)
for point in GridpointsDrones_set
    within_range = false
    for (cs_x, cs_y) in charging_stations
        x_min = cs_x - (half_range - 1)
        x_max = cs_x + half_range
        y_min = cs_y - (half_range - 1)
        y_max = cs_y + half_range
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
println("Transmission range capped to $(transmission_range_size)×$(transmission_range_size) per charging station (from $(transmission_range_size_orig)×$(transmission_range_size_orig) original)")
println("  Accessible points before filter: $original_count")
println("  After filter: $filtered_count")

GridpointsDronesDetecting_set = setdiff(GridpointsDrones_set, charging_stations)
GridpointsDronesDetecting = convert(Vector{Tuple{Int,Int}}, collect(GridpointsDronesDetecting_set))

println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")
println()

# Sanity checks
total_profit = sum(risk_pertime[1, :, :])
println("=== Sanity Checks ===")
println("Total profit at t=1 (downscaled): $(round(total_profit, digits=6))")
println("Total profit at t=1 (original):   $(round(sum(risk_pertime_orig[1, :, :]), digits=6))")
println("Profit match: $(isapprox(total_profit, sum(risk_pertime_orig[1, :, :]), atol=1e-10) ? "✓ YES" : "✗ NO")")
println()

# ---------------------------------------------------------------------------
# PSO runner (identical to the original test file)
# ---------------------------------------------------------------------------

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
# Run: BEST (all optimizations)
# ---------------------------------------------------------------------------

best_stats = run_pso("BEST"; shift_filter = true, swap_filter = true, incremental = true, cost_matrix = true, lazy_dead = true)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

println("="^70)
println("==== SUMMARY (3×3 DOWNSCALED) ====")
println("="^70)

total_split_best = best_stats.split_sparse_stats.calls + best_stats.split_sparse_profit_stats.calls + best_stats.split_dense_stats.calls
println("  Grid:           $(N_ds)×$(M_ds) ($(KERNEL_SIZE)×$(KERNEL_SIZE) downscaled from $(N_orig)×$(M_orig))")
println("  Customers:      $(length(GridpointsDronesDetecting))")
println("  Battery time:   $(max_battery_time) (from $(orig_max_battery_time))")
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
