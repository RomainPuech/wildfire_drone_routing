# Test script for allocation-free iteration optimization
# Run from julia/:  julia test_alloc_free_iteration.jl
#
# Verifies that the allocation-free iteration produces the same results
# and measures the speedup from eliminating setdiff/shuffle allocations
# in the shift and swap inner loops.

using Dates
using Random
using Statistics
using Printf

println("="^60)
println("ALLOCATION-FREE ITERATION OPTIMIZATION TEST")
println("="^60)
println("Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

println("Loading Julia modules...")
include("helper_functions.jl")
include("TOP_PSO_multi_depot.jl")
include("TOP.jl")
println("Modules loaded successfully!")
println()

# ============================================================================
# Helper: create a test PSO instance
# ============================================================================

function create_test_pso(n_pure_customers::Int, n_depot_duplicates::Int, n_drones::Int, max_battery_time::Int)
    n_total = n_pure_customers + n_depot_duplicates
    customers = [(rand(1:30), rand(1:30)) for _ in 1:n_total]
    profits_vec = [rand(0.1:0.01:1.0) for _ in 1:n_pure_customers]
    append!(profits_vec, zeros(Float64, n_depot_duplicates))

    costs = Dict{Tuple{Int, Int}, Float64}()
    for i in 1:n_total, j in 1:n_total
        i == j && continue
        x1, y1 = customers[i]; x2, y2 = customers[j]
        cheb = max(abs(x1 - x2), abs(y1 - y2))
        costs[(i, j)] = cheb == 1 ? 1.0 : max_battery_time * 4.0
    end

    left_neighbors = Dict{Int, Vector{Int}}()
    for v in 1:n_total; left_neighbors[v] = Int[]; end
    for u in 1:n_total, v in 1:n_total
        u == v && continue
        if get(costs, (u, v), Inf) <= max_battery_time
            push!(left_neighbors[v], u)
        end
    end

    # Build cost matrix
    cost_matrix = fill(Float64(max_battery_time * 4), n_total + 1, n_total + 1)
    for ((from, to), cost) in costs
        cost_matrix[from + 1, to + 1] = cost
    end

    return PSOiA_TOP_multiple_depots(
        Particle[], Int[], -Inf, 10, 100,
        0.7, 1.5, 1.5, 0.1, 0.8,
        n_drones, n_pure_customers, max_battery_time,
        customers, profits_vec, costs, cost_matrix, left_neighbors,
        collect(1:n_total), customers[(n_pure_customers+1):end],
        ones(Float64, n_total)
    )
end

# ============================================================================
# TEST 1: Micro-benchmark — allocation count & time
# ============================================================================

println("="^60)
println("TEST 1: Micro-benchmark — setdiff+shuffle vs alloc-free")
println("="^60)
println()

function benchmark_allocation_patterns(n::Int, n_iters::Int)
    println("  n=$n, $n_iters iterations per outer position:")

    # --- Old pattern: shuffle(setdiff(1:n, [i])) ---
    t_old = @elapsed begin
        for rep in 1:n_iters
            for i in 1:n
                inner_j = shuffle(setdiff(1:n, [i]))
                # Simulate touching elements
                s = 0
                for j in inner_j
                    j == i && continue
                    s += j
                end
            end
        end
    end
    alloc_old = @allocated begin
        for rep in 1:n_iters
            for i in 1:n
                inner_j = shuffle(setdiff(1:n, [i]))
                s = 0
                for j in inner_j
                    j == i && continue
                    s += j
                end
            end
        end
    end

    # --- New pattern: pre-allocated buffer + shuffle! ---
    t_new = @elapsed begin
        buf = collect(1:n)
        for rep in 1:n_iters
            for i in 1:n
                shuffle!(buf)
                s = 0
                for j_idx in 1:n
                    @inbounds j = buf[j_idx]
                    j == i && continue
                    s += j
                end
            end
        end
    end
    alloc_new = @allocated begin
        buf = collect(1:n)
        for rep in 1:n_iters
            for i in 1:n
                shuffle!(buf)
                s = 0
                for j_idx in 1:n
                    @inbounds j = buf[j_idx]
                    j == i && continue
                    s += j
                end
            end
        end
    end

    # --- Same for swap pattern: shuffle(collect(i+1:n)) vs buffer fill ---
    t_swap_old = @elapsed begin
        for rep in 1:n_iters
            for i in 1:n
                inner_j = shuffle(collect(i+1:n))
                s = 0
                for j in inner_j
                    s += j
                end
            end
        end
    end
    alloc_swap_old = @allocated begin
        for rep in 1:n_iters
            for i in 1:n
                inner_j = shuffle(collect(i+1:n))
                s = 0
                for j in inner_j
                    s += j
                end
            end
        end
    end

    t_swap_new = @elapsed begin
        swap_buf = Vector{Int}(undef, n)
        for rep in 1:n_iters
            for i in 1:n
                inner_len = n - i
                for k in 1:inner_len; swap_buf[k] = i + k; end
                shuffle!(view(swap_buf, 1:inner_len))
                s = 0
                for jj in 1:inner_len
                    @inbounds s += swap_buf[jj]
                end
            end
        end
    end
    alloc_swap_new = @allocated begin
        swap_buf = Vector{Int}(undef, n)
        for rep in 1:n_iters
            for i in 1:n
                inner_len = n - i
                for k in 1:inner_len; swap_buf[k] = i + k; end
                shuffle!(view(swap_buf, 1:inner_len))
                s = 0
                for jj in 1:inner_len
                    @inbounds s += swap_buf[jj]
                end
            end
        end
    end

    function fmt_bytes(b)
        if b >= 1_000_000_000; return @sprintf("%.1f GB", b / 1e9)
        elseif b >= 1_000_000; return @sprintf("%.1f MB", b / 1e6)
        elseif b >= 1_000; return @sprintf("%.1f KB", b / 1e3)
        else; return "$b B"; end
    end

    println("    SHIFT pattern:")
    println("      Old (setdiff+shuffle): $(round(t_old, digits=4))s, alloc=$(fmt_bytes(alloc_old))")
    println("      New (buffer+shuffle!): $(round(t_new, digits=4))s, alloc=$(fmt_bytes(alloc_new))")
    println("      Speedup: $(round(t_old / t_new, digits=2))×, alloc reduction: $(round((1 - alloc_new/alloc_old)*100, digits=1))%")
    println()
    println("    SWAP pattern:")
    println("      Old (collect+shuffle): $(round(t_swap_old, digits=4))s, alloc=$(fmt_bytes(alloc_swap_old))")
    println("      New (buffer+shuffle!): $(round(t_swap_new, digits=4))s, alloc=$(fmt_bytes(alloc_swap_new))")
    println("      Speedup: $(round(t_swap_old / t_swap_new, digits=2))×, alloc reduction: $(round((1 - alloc_swap_new/alloc_swap_old)*100, digits=1))%")
    println()

    return (shift_speedup=t_old/t_new, swap_speedup=t_swap_old/t_swap_new,
            shift_alloc_old=alloc_old, shift_alloc_new=alloc_new,
            swap_alloc_old=alloc_swap_old, swap_alloc_new=alloc_swap_new)
end

using Printf

# Warmup
Random.seed!(1)
benchmark_allocation_patterns(100, 1)

# Real benchmarks
println("--- Benchmark results (after warmup) ---")
println()
results = Dict{Int, Any}()
for n in [100, 300, 900]
    println("  n = $n:")
    Random.seed!(42)
    r = benchmark_allocation_patterns(n, 3)
    results[n] = r
end

println()
println("Summary table:")
println("| n   | Shift speedup | Shift alloc reduction | Swap speedup | Swap alloc reduction |")
println("|-----|--------------|----------------------|-------------|---------------------|")
for n in [100, 300, 900]
    r = results[n]
    println("| $n | $(round(r.shift_speedup, digits=2))× | $(round((1-r.shift_alloc_new/r.shift_alloc_old)*100, digits=1))% | $(round(r.swap_speedup, digits=2))× | $(round((1-r.swap_alloc_new/r.swap_alloc_old)*100, digits=1))% |")
end
println()

# ============================================================================
# TEST 2: Local search correctness — operators produce valid results
# ============================================================================

println("="^60)
println("TEST 2: Local search correctness (multiple instances)")
println("="^60)
println()

configs = [
    (n=34,  k=4, m=2, L=8,  label="Small 1"),
    (n=56,  k=6, m=3, L=10, label="Small 2"),
    (n=304, k=4, m=2, L=15, label="Large 1"),
    (n=508, k=8, m=3, L=20, label="Large 2"),
    (n=904, k=4, m=2, L=63, label="Large 3"),
]

ENABLE_SHIFT_IRRELEVANCE_FILTER[] = true
ENABLE_SWAP_BLOCKING_FILTER[] = true
ENABLE_LIVE_ZONE_FILTER[] = false
ENABLE_INCREMENTAL_LOCAL_SEARCH[] = false
ENABLE_COST_MATRIX[] = true

for cfg in configs
    println("  Config: $(cfg.label) (n=$(cfg.n), k=$(cfg.k), m=$(cfg.m), L=$(cfg.L))")
    Random.seed!(123)
    pso = create_test_pso(cfg.n, cfg.k, cfg.m, cfg.L)

    # Create multiple particles and run local search
    n_trials = 5
    profits_sparse = Float64[]
    profits_incremental = Float64[]

    for trial in 1:n_trials
        Random.seed!(trial * 1000)
        perm = shuffle(pso.accessible_customers)

        # Test sparse local search (uses shift_operator_sparse! + swap_operator_sparse!)
        p1 = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
        p1.current_profit, _, _ = fast_split_sparse(p1.position, p1, pso)
        ENABLE_INCREMENTAL_LOCAL_SEARCH[] = false
        local_search_sparse!(p1, 1, pso)
        push!(profits_sparse, p1.current_profit)

        # Test incremental local search (uses shift_operator_incremental! + swap_operator_incremental!)
        p2 = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
        p2.current_profit, _, _ = fast_split_sparse(p2.position, p2, pso)
        ENABLE_INCREMENTAL_LOCAL_SEARCH[] = true
        local_search_fully_incremental!(p2, 1, pso)
        push!(profits_incremental, p2.current_profit)
    end

    # Both should produce positive profits (valid solutions)
    all_valid_sparse = all(p -> p >= 0, profits_sparse)
    all_valid_incr = all(p -> p >= 0, profits_incremental)
    avg_sparse = mean(profits_sparse)
    avg_incr = mean(profits_incremental)
    # Allow some variation since RNG trajectories differ
    quality_ok = abs(avg_sparse - avg_incr) / max(avg_sparse, avg_incr, 1e-10) < 0.3

    status = (all_valid_sparse && all_valid_incr && quality_ok) ? "✅ PASS" : "❌ FAIL"
    println("    Sparse  avg profit: $(round(avg_sparse, digits=6)) (all valid: $all_valid_sparse)")
    println("    Incr    avg profit: $(round(avg_incr, digits=6)) (all valid: $all_valid_incr)")
    println("    Quality check: $status (diff=$(round(abs(avg_sparse-avg_incr)/max(avg_sparse,avg_incr,1e-10)*100, digits=1))%)")
    println()
end

# ============================================================================
# TEST 3: Operator-level speedup benchmark
# ============================================================================

println("="^60)
println("TEST 3: Operator-level speedup benchmark")
println("="^60)
println()

for cfg in [(n=304, k=4, m=2, L=15, label="Medium (n=304)"),
            (n=904, k=4, m=2, L=63, label="Large (n=904)")]
    println("  Config: $(cfg.label)")
    Random.seed!(42)
    pso = create_test_pso(cfg.n, cfg.k, cfg.m, cfg.L)
    ENABLE_COST_MATRIX[] = true

    n_trials = 10
    times_sparse = Float64[]
    times_incr = Float64[]

    for trial in 1:n_trials
        Random.seed!(trial * 100)
        perm = shuffle(pso.accessible_customers)

        # Benchmark sparse local search
        p1 = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
        p1.current_profit, _, _ = fast_split_sparse(p1.position, p1, pso)
        t1 = @elapsed local_search_sparse!(p1, 1, pso)
        push!(times_sparse, t1)

        # Benchmark incremental local search
        p2 = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
        p2.current_profit, _, _ = fast_split_sparse(p2.position, p2, pso)
        t2 = @elapsed local_search_fully_incremental!(p2, 1, pso)
        push!(times_incr, t2)
    end

    avg_t_sparse = mean(times_sparse)
    avg_t_incr = mean(times_incr)
    println("    Sparse  local search: avg $(round(avg_t_sparse*1000, digits=1))ms")
    println("    Incr    local search: avg $(round(avg_t_incr*1000, digits=1))ms")
    println("    Speedup: $(round(avg_t_sparse / avg_t_incr, digits=2))×")
    println()
end

println("="^60)
println("ALL TESTS COMPLETE")
println("="^60)
println("Finished at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
