# Test script for incremental swap AND shift correctness and speedup
# Run from julia/:  julia test_incremental_swap.jl
#
# Tests that incremental evaluations produce the exact same profit as full
# fast_split_sparse for every trial swap and shift, and benchmarks speedup.

using Dates
using Random
using Statistics

println("="^60)
println("INCREMENTAL SWAP & SHIFT CORRECTNESS + SPEEDUP TEST")
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

countmap_simple(v) = begin d = Dict{Int,Int}(); for x in v; d[x] = get(d, x, 0) + 1; end; d; end

# ============================================================================
# SWAP CORRECTNESS: exhaustive (small) and sampled (large)
# ============================================================================

function test_swap_correctness(pso, permutation; exhaustive::Bool=false, n_samples::Int=5000)
    label = exhaustive ? "EXHAUSTIVE" : "SAMPLED"
    println("  Swap correctness ($label)...")

    particle = Particle(copy(permutation), copy(permutation), 0.0, 0.0,
                        compute_node_to_position(permutation))
    initial_profit, _, _ = fast_split_sparse(particle.position, particle, pso)
    particle.current_profit = initial_profit
    cache = init_tour_cache(particle, pso)

    n = length(permutation)
    violations = 0; total = 0; dp_skipped = 0; affected_counts = Int[]
    pos = particle.position

    if exhaustive
        pairs = [(i, j) for i in 1:n-1 for j in i+1:n
                 if pos[i] <= pso.n_pure_customers && pos[j] <= pso.n_pure_customers]
    else
        Random.seed!(42)
        pairs = Tuple{Int,Int}[]
        while length(pairs) < n_samples
            i = rand(1:n-1); j = rand(i+1:n)
            if pos[i] <= pso.n_pure_customers && pos[j] <= pso.n_pure_customers
                push!(pairs, (i, j))
            end
        end
    end

    for (i, j) in pairs
        total += 1
        pos[i], pos[j] = pos[j], pos[i]
        full_profit, _, _ = fast_split_sparse(pos, pso)
        pos[i], pos[j] = pos[j], pos[i]

        pos[i], pos[j] = pos[j], pos[i]
        affected = find_affected_tour_indices(cache, i, j)
        push!(affected_counts, length(affected))
        old_vals = [(cache.P_sparse[t], cache.tour_lengths_sparse[t], cache.succ_sparse[t]) for t in affected]
        for t in affected; recompute_single_tour!(cache, t, pos, pso); end
        changed = any(idx -> cache.P_sparse[affected[idx]] != old_vals[idx][1] ||
                             cache.tour_lengths_sparse[affected[idx]] != old_vals[idx][2] ||
                             cache.succ_sparse[affected[idx]] != old_vals[idx][3], eachindex(affected))
        if !changed
            dp_skipped += 1; incr_profit = initial_profit
        else
            Γ = sparse_dp_phase2(cache.P_sparse, cache.succ_sparse, cache.sorted_depot_positions, pso.n_drones, n)
            incr_profit = lookup_Γ_sparse(1, pso.n_drones, cache.sorted_depot_positions, Γ)
        end
        pos[i], pos[j] = pos[j], pos[i]
        for (idx, t) in enumerate(affected)
            cache.P_sparse[t] = old_vals[idx][1]
            cache.tour_lengths_sparse[t] = old_vals[idx][2]
            cache.succ_sparse[t] = old_vals[idx][3]
        end
        if abs(full_profit - incr_profit) > 1e-9
            violations += 1
            println("    ❌ swap($i,$j): full=$full_profit incr=$incr_profit")
        end
    end

    dp_pct = round(100.0 * dp_skipped / max(total, 1), digits=1)
    println("    Tested: $total | Violations: $violations | DP skip: $dp_skipped ($dp_pct%)")
    if !isempty(affected_counts)
        println("    Affected: avg=$(round(mean(affected_counts), digits=2)) max=$(maximum(affected_counts)) dist=$(sort(collect(countmap_simple(affected_counts))))")
    end
    passed = violations == 0
    println("    $(passed ? "✅" : "❌") Swap $(label)")
    return passed
end

# ============================================================================
# SHIFT CORRECTNESS: exhaustive (small) and sampled (large)
# ============================================================================

function test_shift_correctness(pso, permutation; exhaustive::Bool=false, n_samples::Int=5000)
    label = exhaustive ? "EXHAUSTIVE" : "SAMPLED"
    println("  Shift correctness ($label)...")

    particle = Particle(copy(permutation), copy(permutation), 0.0, 0.0,
                        compute_node_to_position(permutation))
    initial_profit, _, _ = fast_split_sparse(particle.position, particle, pso)
    particle.current_profit = initial_profit
    cache = init_tour_cache(particle, pso)

    n = length(permutation)
    k = length(cache.sorted_depot_positions)
    violations = 0; total = 0; dp_skipped = 0; affected_counts = Int[]
    pos = particle.position

    if exhaustive
        pairs = [(i, j) for i in 1:n for j in 1:n if i != j]
    else
        Random.seed!(77)
        pairs = Tuple{Int,Int}[]
        while length(pairs) < n_samples
            i = rand(1:n); j = rand(1:n)
            if i != j; push!(pairs, (i, j)); end
        end
    end

    for (i, j) in pairs
        total += 1
        lo = min(i, j); hi = max(i, j)

        # Full split (ground truth)
        new_pos = move_element(pos, i, j)
        full_profit, _, _ = fast_split_sparse(new_pos, pso)

        # Incremental evaluation
        # Step 1: find affected tours (breakpoint check)
        bp1 = i
        bp2 = j
        any_affected = false
        affected_mask = falses(k)
        for t in 1:k
            d = cache.sorted_depot_positions[t]
            succ_pos = d + cache.tour_lengths_sparse[t]
            if (d <= bp1 <= succ_pos) || (d < bp2 <= succ_pos)
                affected_mask[t] = true
                any_affected = true
            end
        end
        push!(affected_counts, count(affected_mask))

        if !any_affected
            dp_skipped += 1
            incr_profit = initial_profit
        else
            # Step 2: compute new depot positions
            new_dep_unsorted = compute_shifted_depot_positions(cache.sorted_depot_positions, i, j)
            sp = sortperm(new_dep_unsorted)
            new_sorted_dep = new_dep_unsorted[sp]

            # Step 3: shift in-place
            shift_in_place!(pos, i, j)

            # Step 4: build new arrays
            new_P = Vector{Float64}(undef, k)
            new_len = Vector{Int}(undef, k)
            new_succ = Vector{Int}(undef, k)
            for new_idx in 1:k
                old_idx = sp[new_idx]
                if affected_mask[old_idx]
                    p, l, s = compute_tour_at(new_sorted_dep[new_idx], pos, pso)
                    new_P[new_idx] = p; new_len[new_idx] = l; new_succ[new_idx] = s
                else
                    new_P[new_idx] = cache.P_sparse[old_idx]
                    old_len = cache.tour_lengths_sparse[old_idx]
                    new_len[new_idx] = old_len
                    succ_val = new_sorted_dep[new_idx] + old_len
                    new_succ[new_idx] = (succ_val <= n) ? succ_val : 0
                end
            end

            # Step 5: check cache change
            cache_changed = (new_sorted_dep != cache.sorted_depot_positions)
            if !cache_changed
                for t in 1:k
                    if new_P[t] != cache.P_sparse[t] || new_len[t] != cache.tour_lengths_sparse[t] || new_succ[t] != cache.succ_sparse[t]
                        cache_changed = true; break
                    end
                end
            end

            if !cache_changed
                dp_skipped += 1
                incr_profit = initial_profit
            else
                Γ = sparse_dp_phase2(new_P, new_succ, new_sorted_dep, pso.n_drones, n)
                incr_profit = lookup_Γ_sparse(1, pso.n_drones, new_sorted_dep, Γ)
            end

            # Revert
            revert_shift_in_place!(pos, i, j)
        end

        if abs(full_profit - incr_profit) > 1e-9
            violations += 1
            println("    ❌ shift($i,$j): full=$(round(full_profit,digits=8)) incr=$(round(incr_profit,digits=8)) diff=$(abs(full_profit-incr_profit))")
        end
    end

    dp_pct = round(100.0 * dp_skipped / max(total, 1), digits=1)
    println("    Tested: $total | Violations: $violations | DP skip: $dp_skipped ($dp_pct%)")
    if !isempty(affected_counts)
        println("    Affected: avg=$(round(mean(affected_counts), digits=2)) max=$(maximum(affected_counts)) dist=$(sort(collect(countmap_simple(affected_counts))))")
    end
    passed = violations == 0
    println("    $(passed ? "✅" : "❌") Shift $(label)")
    return passed
end

# ============================================================================
# BENCHMARK: per-evaluation speedup
# ============================================================================

function benchmark_per_eval(pso, permutation, n_trials)
    println("  Per-evaluation speedup benchmark ($n_trials evals each)...")

    particle = Particle(copy(permutation), copy(permutation), 0.0, 0.0,
                        compute_node_to_position(permutation))
    initial_profit, _, _ = fast_split_sparse(particle.position, particle, pso)
    particle.current_profit = initial_profit
    cache = init_tour_cache(particle, pso)

    n = length(permutation); k = length(cache.sorted_depot_positions)
    pos = particle.position

    # --- SWAP benchmark ---
    Random.seed!(2026)
    swap_pairs = Tuple{Int,Int}[]
    while length(swap_pairs) < n_trials
        i = rand(1:n-1); j = rand(i+1:n)
        if pos[i] <= pso.n_pure_customers && pos[j] <= pso.n_pure_customers
            push!(swap_pairs, (i, j))
        end
    end

    # Warmup
    for (i, j) in swap_pairs[1:min(50, length(swap_pairs))]
        pos[i], pos[j] = pos[j], pos[i]; fast_split_sparse(pos, pso); pos[i], pos[j] = pos[j], pos[i]
    end

    t_swap_full = @elapsed for (i, j) in swap_pairs
        pos[i], pos[j] = pos[j], pos[i]; fast_split_sparse(pos, pso); pos[i], pos[j] = pos[j], pos[i]
    end

    swap_dp_skip = 0
    t_swap_incr = @elapsed for (i, j) in swap_pairs
        pos[i], pos[j] = pos[j], pos[i]
        aff = find_affected_tour_indices(cache, i, j)
        old = [(cache.P_sparse[t], cache.tour_lengths_sparse[t], cache.succ_sparse[t]) for t in aff]
        for t in aff; recompute_single_tour!(cache, t, pos, pso); end
        changed = any(idx -> cache.P_sparse[aff[idx]] != old[idx][1] ||
                            cache.tour_lengths_sparse[aff[idx]] != old[idx][2] ||
                            cache.succ_sparse[aff[idx]] != old[idx][3], eachindex(aff))
        if changed
            sparse_dp_phase2(cache.P_sparse, cache.succ_sparse, cache.sorted_depot_positions, pso.n_drones, n)
        else
            swap_dp_skip += 1
        end
        pos[i], pos[j] = pos[j], pos[i]
        for (idx, t) in enumerate(aff)
            cache.P_sparse[t] = old[idx][1]; cache.tour_lengths_sparse[t] = old[idx][2]; cache.succ_sparse[t] = old[idx][3]
        end
    end

    println("    SWAP: full=$(round(t_swap_full,digits=4))s incr=$(round(t_swap_incr,digits=4))s speedup=$(round(t_swap_full/max(t_swap_incr,1e-12),digits=1))× dp_skip=$(swap_dp_skip)/$n_trials")

    # --- SHIFT benchmark ---
    Random.seed!(3030)
    shift_pairs = Tuple{Int,Int}[]
    while length(shift_pairs) < n_trials
        i = rand(1:n); j = rand(1:n)
        if i != j; push!(shift_pairs, (i, j)); end
    end

    # Warmup
    for (i, j) in shift_pairs[1:min(50, length(shift_pairs))]
        np = move_element(pos, i, j); fast_split_sparse(np, pso)
    end

    t_shift_full = @elapsed for (i, j) in shift_pairs
        np = move_element(pos, i, j); fast_split_sparse(np, pso)
    end

    shift_dp_skip = 0
    t_shift_incr = @elapsed for (i, j) in shift_pairs
        bp1 = i; bp2 = j
        any_aff = false; aff_mask = falses(k)
        for t in 1:k
            d = cache.sorted_depot_positions[t]; sp = d + cache.tour_lengths_sparse[t]
            if (d <= bp1 <= sp) || (d < bp2 <= sp); aff_mask[t] = true; any_aff = true; end
        end
        if !any_aff
            shift_dp_skip += 1; continue
        end
        new_dep = compute_shifted_depot_positions(cache.sorted_depot_positions, i, j)
        spp = sortperm(new_dep); nsd = new_dep[spp]
        shift_in_place!(pos, i, j)
        new_P = Vector{Float64}(undef, k); new_len = Vector{Int}(undef, k); new_succ = Vector{Int}(undef, k)
        for ni in 1:k
            oi = spp[ni]
            if aff_mask[oi]
                p, l, s = compute_tour_at(nsd[ni], pos, pso)
                new_P[ni] = p; new_len[ni] = l; new_succ[ni] = s
            else
                new_P[ni] = cache.P_sparse[oi]; ol = cache.tour_lengths_sparse[oi]; new_len[ni] = ol
                sv = nsd[ni] + ol; new_succ[ni] = (sv <= n) ? sv : 0
            end
        end
        cc = (nsd != cache.sorted_depot_positions)
        if !cc
            for t in 1:k
                if new_P[t] != cache.P_sparse[t] || new_len[t] != cache.tour_lengths_sparse[t] || new_succ[t] != cache.succ_sparse[t]
                    cc = true; break
                end
            end
        end
        if cc
            sparse_dp_phase2(new_P, new_succ, nsd, pso.n_drones, n)
        else
            shift_dp_skip += 1
        end
        revert_shift_in_place!(pos, i, j)
    end

    println("    SHIFT: full=$(round(t_shift_full,digits=4))s incr=$(round(t_shift_incr,digits=4))s speedup=$(round(t_shift_full/max(t_shift_incr,1e-12),digits=1))× dp_skip=$(shift_dp_skip)/$n_trials")
end

# ============================================================================
# MAIN
# ============================================================================

function run_all_tests()
    all_passed = true

    # Small instances: exhaustive
    small_configs = [
        (n_pure_customers=30,  n_depot_duplicates=4, n_drones=2, max_battery_time=8),
        (n_pure_customers=50,  n_depot_duplicates=6, n_drones=3, max_battery_time=10),
        (n_pure_customers=60,  n_depot_duplicates=4, n_drones=2, max_battery_time=15),
    ]

    for (idx, c) in enumerate(small_configs)
        println("="^60)
        nt = c.n_pure_customers + c.n_depot_duplicates
        println("SMALL CONFIG $idx (EXHAUSTIVE): n=$nt k=$(c.n_depot_duplicates) m=$(c.n_drones) L=$(c.max_battery_time)")
        println("="^60)

        Random.seed!(idx * 1000)
        pso = create_test_pso(c.n_pure_customers, c.n_depot_duplicates, c.n_drones, c.max_battery_time)
        dn = collect((c.n_pure_customers+1):nt); cn = collect(1:c.n_pure_customers)
        third = div(c.n_pure_customers, 3)
        perm = vcat(shuffle(cn[1:third]), dn[1:div(length(dn),2)],
                    shuffle(cn[third+1:2*third]), dn[div(length(dn),2)+1:end],
                    shuffle(cn[2*third+1:end]))

        p1 = test_swap_correctness(pso, perm; exhaustive=true)
        p2 = test_shift_correctness(pso, perm; exhaustive=true)
        if !p1 || !p2; all_passed = false; end
        println()
    end

    # Large instances: sampled + benchmark
    large_configs = [
        (n_pure_customers=300, n_depot_duplicates=4, n_drones=2, max_battery_time=15, n_samples=5000),
        (n_pure_customers=500, n_depot_duplicates=8, n_drones=3, max_battery_time=20, n_samples=5000),
        (n_pure_customers=900, n_depot_duplicates=4, n_drones=2, max_battery_time=63, n_samples=5000),
    ]

    for (idx, c) in enumerate(large_configs)
        println("="^60)
        nt = c.n_pure_customers + c.n_depot_duplicates
        println("LARGE CONFIG $idx: n=$nt k=$(c.n_depot_duplicates) m=$(c.n_drones) L=$(c.max_battery_time)")
        println("="^60)

        Random.seed!((idx + 10) * 1000)
        pso = create_test_pso(c.n_pure_customers, c.n_depot_duplicates, c.n_drones, c.max_battery_time)
        dn = collect((c.n_pure_customers+1):nt); cn = collect(1:c.n_pure_customers)
        third = div(c.n_pure_customers, 3)
        perm = vcat(shuffle(cn[1:third]), dn[1:div(length(dn),2)],
                    shuffle(cn[third+1:2*third]), dn[div(length(dn),2)+1:end],
                    shuffle(cn[2*third+1:end]))

        p1 = test_swap_correctness(pso, perm; n_samples=c.n_samples)
        p2 = test_shift_correctness(pso, perm; n_samples=c.n_samples)
        if !p1 || !p2; all_passed = false; end

        benchmark_per_eval(pso, perm, min(c.n_samples, 5000))
        println()
    end

    # ====================================================================
    # LIVE ZONE CORRECTNESS: verify no improving move is wrongly skipped
    # ====================================================================
    println()
    println("="^60)
    println("LIVE ZONE FILTER CORRECTNESS TEST")
    println("="^60)
    println()

    for (idx, c) in enumerate(large_configs)
        nt = c.n_pure_customers + c.n_depot_duplicates
        println("LIVE ZONE TEST $idx: n=$nt k=$(c.n_depot_duplicates) m=$(c.n_drones) L=$(c.max_battery_time)")

        Random.seed!((idx + 100) * 1000)
        pso = create_test_pso(c.n_pure_customers, c.n_depot_duplicates, c.n_drones, c.max_battery_time)
        dn = collect((c.n_pure_customers+1):nt); cn = collect(1:c.n_pure_customers)
        third = div(c.n_pure_customers, 3)
        perm = vcat(shuffle(cn[1:third]), dn[1:div(length(dn),2)],
                    shuffle(cn[third+1:2*third]), dn[div(length(dn),2)+1:end],
                    shuffle(cn[2*third+1:end]))

        n = length(perm)
        particle = Particle(copy(perm), copy(perm), 0.0, 0.0,
                            compute_node_to_position(perm))
        base_profit, _, _ = fast_split_sparse(perm, particle, pso)
        particle.current_profit = base_profit

        # Compute live zone data (using safe dead positions with +1 extension)
        _, _, tl_sparse, sdp = compute_saturated_tours_sparse(perm, particle.node_to_position, pso)
        dead_pos = compute_safe_dead_positions(n, sdp, tl_sparse)
        dbs, dbe = compute_dead_block_boundaries(dead_pos)
        live_sorted = sort([p for p in 1:n if !dead_pos[p]])

        n_live = length(live_sorted)
        n_dead = n - n_live
        println("  Live positions: $n_live / $n ($(round(100.0*n_live/n, digits=1))%)")

        # Test swap: for sampled dead-dead pairs, verify profit is unchanged
        swap_violations = 0
        swap_tested = 0
        dead_list = [p for p in 1:n if dead_pos[p]]
        for _ in 1:min(5000, length(dead_list)^2)
            i = rand(dead_list); j = rand(dead_list)
            i >= j && continue
            pos = copy(perm)
            pos[i], pos[j] = pos[j], pos[i]
            new_profit, _, _ = fast_split_sparse(pos, pso)
            if new_profit != base_profit
                swap_violations += 1
            end
            swap_tested += 1
        end
        println("  Swap dead-dead test: $swap_tested tested, $swap_violations violations " *
                (swap_violations == 0 ? "✅" : "❌"))

        # Test shift: for sampled within-dead-block pairs, verify profit unchanged
        shift_violations = 0
        shift_tested = 0
        for _ in 1:min(5000, n^2)
            i = rand(dead_list)
            bs = dbs[i]; be = dbe[i]
            bs == 0 && continue  # not in a dead block (shouldn't happen)
            # Pick j within same dead block, different from i
            j = rand(bs:be)
            j == i && continue
            pos = move_element(perm, i, j)
            new_profit, _, _ = fast_split_sparse(pos, pso)
            if new_profit != base_profit
                shift_violations += 1
            end
            shift_tested += 1
        end
        println("  Shift dead-block test: $shift_tested tested, $shift_violations violations " *
                (shift_violations == 0 ? "✅" : "❌"))

        if swap_violations > 0 || shift_violations > 0
            all_passed = false
        end

        # Benchmark: compare iteration counts and timing with live zone on vs off
        ENABLE_LIVE_ZONE_FILTER[] = false
        cache_no_lz = init_tour_cache(particle, pso)
        t0 = time()
        INCREMENTAL_SWAP_STATS[] = (candidates=0, skipped_blocking=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
        INCREMENTAL_SHIFT_STATS[] = (candidates=0, skipped_filter=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
        _, _ = fast_split_sparse(particle.position, particle, pso)
        cache_off = init_tour_cache(particle, pso)
        _, _, tour_intervals_off = fast_split_sparse(particle.position, particle, pso)
        swap_counters_off = Ref((0, 0))
        swap_operator_incremental!(particle, pso, tour_intervals_off, cache_off, swap_counters_off)
        cands_swap_off = INCREMENTAL_SWAP_STATS[].candidates
        shift_operator_incremental!(particle, pso, cache_off, tour_intervals_off)
        cands_shift_off = INCREMENTAL_SHIFT_STATS[].candidates
        time_off = time() - t0

        ENABLE_LIVE_ZONE_FILTER[] = true
        t1 = time()
        INCREMENTAL_SWAP_STATS[] = (candidates=0, skipped_blocking=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
        INCREMENTAL_SHIFT_STATS[] = (candidates=0, skipped_filter=0, skipped_dp=0, evaluated=0, accepted=0, time=0.0, calls=0)
        _, _, tour_intervals_on = fast_split_sparse(particle.position, particle, pso)
        cache_on = init_tour_cache(particle, pso)
        swap_counters_on = Ref((0, 0))
        swap_operator_incremental!(particle, pso, tour_intervals_on, cache_on, swap_counters_on)
        cands_swap_on = INCREMENTAL_SWAP_STATS[].candidates
        shift_operator_incremental!(particle, pso, cache_on, tour_intervals_on)
        cands_shift_on = INCREMENTAL_SHIFT_STATS[].candidates
        time_on = time() - t1
        ENABLE_LIVE_ZONE_FILTER[] = false

        swap_reduction = cands_swap_off > 0 ? round(100.0*(1.0 - cands_swap_on/cands_swap_off), digits=1) : 0.0
        shift_reduction = cands_shift_off > 0 ? round(100.0*(1.0 - cands_shift_on/cands_shift_off), digits=1) : 0.0
        println("  Swap candidates:  OFF=$cands_swap_off  ON=$cands_swap_on  ($(swap_reduction)% reduction)")
        println("  Shift candidates: OFF=$cands_shift_off ON=$cands_shift_on ($(shift_reduction)% reduction)")
        println("  Time: OFF=$(round(time_off, digits=3))s  ON=$(round(time_on, digits=3))s  speedup=$(round(time_off/max(time_on,0.001), digits=2))×")
        println()
    end

    println("="^60)
    println(all_passed ? "✅ ALL CORRECTNESS TESTS PASSED" : "❌ SOME TESTS FAILED")
    println("="^60)
    println("Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    return all_passed
end

run_all_tests()
