#
# Incremental swap + shift benchmark on AugustComplexFire (real instance)
# Run from julia/:  julia test_incremental_swap_august_fire.jl
#
# Tests:
#   1. Per-swap correctness (incremental == full split)
#   2. Per-shift correctness (incremental == full split)
#   3. Per-evaluation speedup (swap + shift)
#   4. Full local search comparison (original vs fully incremental)
#

using Dates
using Random
using Statistics

println("="^60)
println("INCREMENTAL SWAP+SHIFT BENCHMARK – AugustComplexFire")
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
# Problem instance
# ---------------------------------------------------------------------------

burnmap_filename = "../MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
mask_filename = "../MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

const _ABS_BURNMAP = "/Users/romain/Desktop/wildfire_drone_routing/MiniTractDataset/AugustComplexFire/static_risk_whp_rescaled_103x112_63substeps.npy"
const _ABS_MASK = "/Users/romain/Desktop/wildfire_drone_routing/MiniTractDataset/AugustComplexFire/mask_rescaled_103x112_63substeps.npy"

if !isfile(burnmap_filename) && isfile(_ABS_BURNMAP); burnmap_filename = _ABS_BURNMAP; end
if !isfile(mask_filename) && isfile(_ABS_MASK); mask_filename = _ABS_MASK; end

charging_stations = [(28, 36)]
ground_stations = [(8, 26), (9, 26), (8, 27), (9, 27), (8, 28), (9, 28), (8, 29), (9, 29)]
n_drones = 2; max_battery_time = 63; t = 0
initial_drone_positions = Vector{Tuple{Int,Int}}()

println("Burnmap: $burnmap_filename")
println("Mask: $mask_filename")
println("n_drones=$n_drones, max_battery_time=$max_battery_time")
println()

# ---------------------------------------------------------------------------
# Build inputs
# ---------------------------------------------------------------------------

t_idx = t + 1
risk_pertime = load_burn_map(burnmap_filename)
risk_pertime = risk_pertime[t_idx:end, :, :]
for cs in charging_stations; risk_pertime[:, cs[1], cs[2]] .= 0; end
for gs in ground_stations; risk_pertime[:, gs[1], gs[2]] .= 0; end
_, N, M = size(risk_pertime)

if isfile(mask_filename)
    mask = load_mask(mask_filename)
    I = [(x, y) for x in 1:N for y in 1:M if mask[x,y] == 1]
    blocked = Set([(x, y) for x in 1:N for y in 1:M if mask[x,y] != 1])
    GridpointsDrones_set, _ = get_drone_gridpoints_BFS(charging_stations, floor(max_battery_time/2), I, N, M)
else
    I = [(x, y) for x in 1:N for y in 1:M]
    blocked = Set{Tuple{Int,Int}}()
    GridpointsDrones_set = get_drone_gridpoints(charging_stations, floor(max_battery_time/2), I)
end

half_range = 15
filtered = Set{Tuple{Int,Int}}()
for pt in GridpointsDrones_set
    for (cx, cy) in charging_stations
        if cx-(half_range-1) <= pt[1] <= cx+half_range && cy-(half_range-1) <= pt[2] <= cy+half_range
            push!(filtered, pt); break
        end
    end
end
GridpointsDrones_set = filtered
GridpointsDronesDetecting = collect(setdiff(GridpointsDrones_set, charging_stations))
println("GridpointsDronesDetecting: $(length(GridpointsDronesDetecting)) points")

# ---------------------------------------------------------------------------
# Build PSO instance
# ---------------------------------------------------------------------------

all_gp = GridpointsDronesDetecting
n_pure_customers = length(all_gp)
depot_pos_list = Tuple{Int,Int}[]
for cs in charging_stations; for _ in 1:n_drones; push!(depot_pos_list, cs); end; end
n_depot_dup = length(depot_pos_list)
n_total = n_pure_customers + n_depot_dup

customers = Vector{Tuple{Int,Int}}(undef, n_total)
for i in 1:n_pure_customers; customers[i] = all_gp[i]; end
for i in 1:n_depot_dup; customers[n_pure_customers + i] = depot_pos_list[i]; end

profits_vec = zeros(Float64, n_total)
for i in 1:n_pure_customers; x, y = customers[i]; profits_vec[i] = sum(risk_pertime[:, x, y]); end

L = max_battery_time
costs = Dict{Tuple{Int,Int}, Float64}()
for i in 1:n_total, j in 1:n_total
    i == j && continue
    x1, y1 = customers[i]; x2, y2 = customers[j]
    cheb = max(abs(x1-x2), abs(y1-y2))
    if cheb <= 1; costs[(i,j)] = Float64(cheb); end
end
left_neighbors = Dict{Int, Vector{Int}}()
for v in 1:n_total; left_neighbors[v] = Int[]; end
for ((u, v), c) in costs; if c <= L; push!(left_neighbors[v], u); end; end
closest_depot_distance = zeros(Float64, n_total)
for i in 1:n_total
    x, y = customers[i]
    closest_depot_distance[i] = Float64(minimum(max(abs(x-dx), abs(y-dy)) for (dx, dy) in charging_stations))
end

# Build cost matrix
cost_matrix = fill(Float64(max_battery_time * 4), n_total + 1, n_total + 1)
for ((from, to), cost) in costs
    cost_matrix[from + 1, to + 1] = cost
end

pso = PSOiA_TOP_multiple_depots(
    Particle[], Int[], -Inf, 10, 300,
    0.7, 1.5, 1.5, 0.1, 0.8,
    n_drones, n_pure_customers, max_battery_time,
    customers, profits_vec, costs, cost_matrix, left_neighbors,
    collect(1:n_total), [customers[n_pure_customers+i] for i in 1:n_depot_dup],
    closest_depot_distance
)

println("n=$n_total, n_pure_customers=$n_pure_customers, k=$n_depot_dup")
println()

# ---------------------------------------------------------------------------
# Create realistic starting permutation
# ---------------------------------------------------------------------------

println("Creating starting permutation...")
Random.seed!(42)
permutation = shuffle(collect(1:n_total))
p0 = Particle(copy(permutation), copy(permutation), 0.0, 0.0, compute_node_to_position(permutation))
p0.current_profit, _, _ = fast_split_sparse(p0.position, p0, pso)
println("  Random profit: $(round(p0.current_profit, digits=4))")
reset_boundary_stats!()
local_search_sparse!(p0, 1, pso)
permutation = copy(p0.position)
println("  After local search: $(round(p0.current_profit, digits=4))")

cache0 = init_tour_cache(p0, pso)
println("  k=$(length(cache0.sorted_depot_positions)), tours=$(cache0.tour_lengths_sparse)")
println()

# ---------------------------------------------------------------------------
# TEST 1: Per-swap correctness
# ---------------------------------------------------------------------------

function test_swap_real(pso, perm, n_samples=10000)
    println("="^60)
    println("TEST 1: Per-swap correctness ($n_samples samples)")
    println("="^60)
    p = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
    profit, _, _ = fast_split_sparse(p.position, p, pso)
    p.current_profit = profit
    cache = init_tour_cache(p, pso)
    n = length(perm); violations = 0; tested = 0; dp_skip = 0; depot_fallback = 0
    pos = p.position
    Random.seed!(2026)
    for _ in 1:n_samples
        i = rand(1:n-1); j = rand(i+1:n)
        tested += 1

        # Ground truth
        pos[i], pos[j] = pos[j], pos[i]
        fp, _, _ = fast_split_sparse(pos, pso)
        pos[i], pos[j] = pos[j], pos[i]

        is_depot_i = pos[i] > pso.n_pure_customers
        is_depot_j = pos[j] > pso.n_pure_customers

        if is_depot_i || is_depot_j
            # Depot swap: test via full split fallback (matching operator behavior)
            depot_fallback += 1
            pos[i], pos[j] = pos[j], pos[i]
            ip, _, _ = fast_split_sparse(pos, pso)
            pos[i], pos[j] = pos[j], pos[i]
        else
            # Customer-customer swap: test incremental path
            pos[i], pos[j] = pos[j], pos[i]
            aff = find_affected_tour_indices(cache, i, j)
            old = [(cache.P_sparse[t], cache.tour_lengths_sparse[t], cache.succ_sparse[t]) for t in aff]
            for t in aff; recompute_single_tour!(cache, t, pos, pso); end
            chg = any(idx -> cache.P_sparse[aff[idx]] != old[idx][1] || cache.tour_lengths_sparse[aff[idx]] != old[idx][2] || cache.succ_sparse[aff[idx]] != old[idx][3], eachindex(aff))
            ip = chg ? lookup_Γ_sparse(1, pso.n_drones, cache.sorted_depot_positions, sparse_dp_phase2(cache.P_sparse, cache.succ_sparse, cache.sorted_depot_positions, pso.n_drones, n)) : profit
            if !chg; dp_skip += 1; end
            pos[i], pos[j] = pos[j], pos[i]
            for (idx, t) in enumerate(aff); cache.P_sparse[t] = old[idx][1]; cache.tour_lengths_sparse[t] = old[idx][2]; cache.succ_sparse[t] = old[idx][3]; end
        end

        if abs(fp - ip) > 1e-9; violations += 1; println("  ❌ swap($i,$j) diff=$(abs(fp-ip))"); end
    end
    println("  Tested: $tested, Violations: $violations, DP skip: $dp_skip ($(round(100*dp_skip/max(tested,1), digits=1))%), depot_fallback: $depot_fallback")
    passed = violations == 0
    println("  $(passed ? "✅ PASSED" : "❌ FAILED")")
    println()
    return passed
end

# ---------------------------------------------------------------------------
# TEST 2: Per-shift correctness
# ---------------------------------------------------------------------------

function test_shift_real(pso, perm, n_samples=10000)
    println("="^60)
    println("TEST 2: Per-shift correctness ($n_samples samples)")
    println("="^60)
    p = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
    profit, _, _ = fast_split_sparse(p.position, p, pso)
    p.current_profit = profit
    cache = init_tour_cache(p, pso)
    n = length(perm); k = length(cache.sorted_depot_positions)
    violations = 0; tested = 0; dp_skip = 0; aff_counts = Int[]; pos = p.position
    Random.seed!(3030)
    for _ in 1:n_samples
        i = rand(1:n); j = rand(1:n); i == j && continue
        tested += 1

        # Ground truth
        np = move_element(pos, i, j)
        fp, _, _ = fast_split_sparse(np, pso)

        # Incremental (breakpoint check)
        bp1 = i; bp2 = j
        any_aff = false; aff_mask = falses(k)
        for t in 1:k
            d = cache.sorted_depot_positions[t]; sp = d + cache.tour_lengths_sparse[t]
            if (d <= bp1 <= sp) || (d < bp2 <= sp); aff_mask[t] = true; any_aff = true; end
        end
        push!(aff_counts, count(aff_mask))

        if !any_aff
            dp_skip += 1; ip = profit
        else
            nd = compute_shifted_depot_positions(cache.sorted_depot_positions, i, j)
            spp = sortperm(nd); nsd = nd[spp]
            shift_in_place!(pos, i, j)
            nP = Vector{Float64}(undef, k); nL = Vector{Int}(undef, k); nS = Vector{Int}(undef, k)
            for ni in 1:k
                oi = spp[ni]
                if aff_mask[oi]
                    pp, ll, ss = compute_tour_at(nsd[ni], pos, pso)
                    nP[ni] = pp; nL[ni] = ll; nS[ni] = ss
                else
                    nP[ni] = cache.P_sparse[oi]; ol = cache.tour_lengths_sparse[oi]; nL[ni] = ol
                    sv = nsd[ni] + ol; nS[ni] = (sv <= n) ? sv : 0
                end
            end
            cc = (nsd != cache.sorted_depot_positions)
            if !cc
                for t in 1:k
                    if nP[t] != cache.P_sparse[t] || nL[t] != cache.tour_lengths_sparse[t] || nS[t] != cache.succ_sparse[t]; cc = true; break; end
                end
            end
            if !cc; dp_skip += 1; ip = profit
            else
                Γ = sparse_dp_phase2(nP, nS, nsd, pso.n_drones, n)
                ip = lookup_Γ_sparse(1, pso.n_drones, nsd, Γ)
            end
            revert_shift_in_place!(pos, i, j)
        end

        if abs(fp - ip) > 1e-9; violations += 1; println("  ❌ shift($i,$j) full=$fp incr=$ip diff=$(abs(fp-ip))"); end
    end
    println("  Tested: $tested, Violations: $violations, DP skip: $dp_skip ($(round(100*dp_skip/max(tested,1), digits=1))%)")
    if !isempty(aff_counts)
        println("  Affected tours: avg=$(round(mean(aff_counts),digits=2)) max=$(maximum(aff_counts))")
    end
    passed = violations == 0
    println("  $(passed ? "✅ PASSED" : "❌ FAILED")")
    println()
    return passed
end

# ---------------------------------------------------------------------------
# TEST 3: Per-evaluation speedup
# ---------------------------------------------------------------------------

function benchmark_eval_real(pso, perm, n_trials=10000)
    println("="^60)
    println("TEST 3: Per-evaluation speedup ($n_trials evals each)")
    println("="^60)
    p = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
    pr, _, _ = fast_split_sparse(p.position, p, pso)
    p.current_profit = pr; cache = init_tour_cache(p, pso)
    n = length(perm); k = length(cache.sorted_depot_positions); pos = p.position

    # --- SWAP ---
    Random.seed!(7777)
    sp = Tuple{Int,Int}[]
    while length(sp) < n_trials; i = rand(1:n-1); j = rand(i+1:n)
        if perm[i] <= pso.n_pure_customers && perm[j] <= pso.n_pure_customers; push!(sp, (i,j)); end
    end
    # Warmup
    for (i,j) in sp[1:min(200,length(sp))]; pos[i],pos[j]=pos[j],pos[i]; fast_split_sparse(pos,pso); pos[i],pos[j]=pos[j],pos[i]; end
    for (i,j) in sp[1:min(200,length(sp))]
        pos[i],pos[j]=pos[j],pos[i]; aff=find_affected_tour_indices(cache,i,j)
        old=[(cache.P_sparse[t],cache.tour_lengths_sparse[t],cache.succ_sparse[t]) for t in aff]
        for t in aff; recompute_single_tour!(cache,t,pos,pso); end
        sparse_dp_phase2(cache.P_sparse,cache.succ_sparse,cache.sorted_depot_positions,pso.n_drones,n)
        pos[i],pos[j]=pos[j],pos[i]
        for (idx,t) in enumerate(aff); cache.P_sparse[t]=old[idx][1]; cache.tour_lengths_sparse[t]=old[idx][2]; cache.succ_sparse[t]=old[idx][3]; end
    end
    tf_swap = @elapsed for (i,j) in sp; pos[i],pos[j]=pos[j],pos[i]; fast_split_sparse(pos,pso); pos[i],pos[j]=pos[j],pos[i]; end
    dps = 0
    ti_swap = @elapsed for (i,j) in sp
        pos[i],pos[j]=pos[j],pos[i]; aff=find_affected_tour_indices(cache,i,j)
        old=[(cache.P_sparse[t],cache.tour_lengths_sparse[t],cache.succ_sparse[t]) for t in aff]
        for t in aff; recompute_single_tour!(cache,t,pos,pso); end
        chg=any(idx->cache.P_sparse[aff[idx]]!=old[idx][1]||cache.tour_lengths_sparse[aff[idx]]!=old[idx][2]||cache.succ_sparse[aff[idx]]!=old[idx][3],eachindex(aff))
        if chg; sparse_dp_phase2(cache.P_sparse,cache.succ_sparse,cache.sorted_depot_positions,pso.n_drones,n); else; dps+=1; end
        pos[i],pos[j]=pos[j],pos[i]
        for (idx,t) in enumerate(aff); cache.P_sparse[t]=old[idx][1]; cache.tour_lengths_sparse[t]=old[idx][2]; cache.succ_sparse[t]=old[idx][3]; end
    end
    println("  SWAP: full=$(round(tf_swap,digits=4))s incr=$(round(ti_swap,digits=4))s speedup=$(round(tf_swap/max(ti_swap,1e-12),digits=1))× dp_skip=$dps/$n_trials")

    # --- SHIFT ---
    Random.seed!(8888)
    shp = Tuple{Int,Int}[]
    while length(shp) < n_trials; i=rand(1:n); j=rand(1:n); i!=j && push!(shp,(i,j)); end
    # Warmup
    for (i,j) in shp[1:min(200,length(shp))]; np=move_element(pos,i,j); fast_split_sparse(np,pso); end
    for (i,j) in shp[1:min(200,length(shp))]
        lo=min(i,j); hi=max(i,j); nd=compute_shifted_depot_positions(cache.sorted_depot_positions,i,j)
        spp=sortperm(nd); nsd=nd[spp]; shift_in_place!(pos,i,j)
        for ni in 1:k; oi=spp[ni]; compute_tour_at(nsd[ni],pos,pso); end
        revert_shift_in_place!(pos,i,j)
    end
    tf_shift = @elapsed for (i,j) in shp; np=move_element(pos,i,j); fast_split_sparse(np,pso); end
    dps2 = 0
    ti_shift = @elapsed for (i,j) in shp
        bp1=i; bp2=j; any_a=false; am=falses(k)
        for t in 1:k; d=cache.sorted_depot_positions[t]; sp2=d+cache.tour_lengths_sparse[t]
            if (d<=bp1<=sp2)||(d<bp2<=sp2); am[t]=true; any_a=true; end
        end
        if !any_a; dps2+=1; continue; end
        nd=compute_shifted_depot_positions(cache.sorted_depot_positions,i,j); spp=sortperm(nd); nsd=nd[spp]
        shift_in_place!(pos,i,j)
        nP=Vector{Float64}(undef,k); nL=Vector{Int}(undef,k); nS=Vector{Int}(undef,k)
        for ni in 1:k; oi=spp[ni]
            if am[oi]; pp,ll,ss=compute_tour_at(nsd[ni],pos,pso); nP[ni]=pp; nL[ni]=ll; nS[ni]=ss
            else; nP[ni]=cache.P_sparse[oi]; ol=cache.tour_lengths_sparse[oi]; nL[ni]=ol
                sv=nsd[ni]+ol; nS[ni]=(sv<=n) ? sv : 0; end
        end
        cc=(nsd!=cache.sorted_depot_positions)
        if !cc; for t in 1:k
            if nP[t]!=cache.P_sparse[t]||nL[t]!=cache.tour_lengths_sparse[t]||nS[t]!=cache.succ_sparse[t]; cc=true; break; end
        end; end
        if cc; sparse_dp_phase2(nP,nS,nsd,pso.n_drones,n); else; dps2+=1; end
        revert_shift_in_place!(pos,i,j)
    end
    println("  SHIFT: full=$(round(tf_shift,digits=4))s incr=$(round(ti_shift,digits=4))s speedup=$(round(tf_shift/max(ti_shift,1e-12),digits=1))× dp_skip=$dps2/$n_trials")
    println()
end

# ---------------------------------------------------------------------------
# TEST 4: Full local search comparison
# ---------------------------------------------------------------------------

function benchmark_local_search_real(pso, perm, n_trials=5)
    println("="^60)
    println("TEST 4: Local search comparison ($n_trials trials)")
    println("="^60)
    to = Float64[]; ti = Float64[]; po = Float64[]; pi_arr = Float64[]
    for trial in 1:n_trials
        Random.seed!(trial*100)
        p1 = Particle(copy(perm),copy(perm),0.0,0.0,compute_node_to_position(perm))
        p1.current_profit = fast_split_sparse_profit(p1.position,p1,pso)
        reset_boundary_stats!()
        t1 = @elapsed local_search_sparse!(p1,1,pso)
        push!(to, t1); push!(po, p1.current_profit)

        Random.seed!(trial*100)
        p2 = Particle(copy(perm),copy(perm),0.0,0.0,compute_node_to_position(perm))
        p2.current_profit = fast_split_sparse_profit(p2.position,p2,pso)
        reset_boundary_stats!()
        t2 = @elapsed local_search_fully_incremental!(p2,1,pso)
        push!(ti, t2); push!(pi_arr, p2.current_profit)

        rd = abs(p1.current_profit-p2.current_profit)/max(abs(p1.current_profit),1e-12)*100
        println("  Trial $trial: orig=$(round(t1,digits=3))s P=$(round(p1.current_profit,digits=4)) | incr=$(round(t2,digits=3))s P=$(round(p2.current_profit,digits=4)) | diff=$(round(rd,digits=2))%")
    end
    ao = mean(to); ai = mean(ti)
    println()
    println("  Avg time (original):         $(round(ao,digits=3))s")
    println("  Avg time (fully incremental): $(round(ai,digits=3))s")
    println("  Speedup:                      $(round(ao/max(ai,1e-12),digits=2))×")
    println("  Avg profit (original):        $(round(mean(po),digits=4))")
    println("  Avg profit (incremental):     $(round(mean(pi_arr),digits=4))")
    println()
end

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

passed1 = test_swap_real(pso, permutation, 10000)
passed2 = test_shift_real(pso, permutation, 10000)
benchmark_eval_real(pso, permutation, 10000)
benchmark_local_search_real(pso, permutation, 5)

# ---------------------------------------------------------------------------
# TEST 5: Live zone filter correctness + speedup on real instance
# ---------------------------------------------------------------------------

function test_live_zone_real(pso, perm)
    println("="^60)
    println("TEST 5: Live zone filter correctness + speedup (real instance)")
    println("="^60)

    n = length(perm)
    particle = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
    base_profit, _, _ = fast_split_sparse(perm, particle, pso)
    particle.current_profit = base_profit

    _, _, tl_sparse, sdp = compute_saturated_tours_sparse(perm, particle.node_to_position, pso)
    dead_pos = compute_safe_dead_positions(n, sdp, tl_sparse)
    dbs, dbe = compute_dead_block_boundaries(dead_pos)
    live_sorted = sort([p for p in 1:n if !dead_pos[p]])

    n_live = length(live_sorted)
    n_dead = n - n_live
    println("  Live positions: $n_live / $n ($(round(100.0*n_live/n, digits=1))%)")
    println("  Dead positions: $n_dead / $n ($(round(100.0*n_dead/n, digits=1))%)")

    # 5a. Swap dead-dead correctness: sample 5000 dead-dead pairs
    dead_list = [p for p in 1:n if dead_pos[p]]
    swap_violations = 0; swap_tested = 0
    for _ in 1:5000
        length(dead_list) < 2 && break
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
    println("  Swap dead-dead correctness: $swap_tested tested, $swap_violations violations " *
            (swap_violations == 0 ? "✅" : "❌"))

    # 5b. Shift within-dead-block correctness: sample 5000 pairs
    shift_violations = 0; shift_tested = 0
    for _ in 1:5000
        length(dead_list) < 2 && break
        i = rand(dead_list)
        bs = dbs[i]; be = dbe[i]
        be - bs < 1 && continue
        j = rand(bs:be)
        j == i && continue
        pos = move_element(perm, i, j)
        new_profit, _, _ = fast_split_sparse(pos, pso)
        if new_profit != base_profit
            shift_violations += 1
        end
        shift_tested += 1
    end
    println("  Shift dead-block correctness: $shift_tested tested, $shift_violations violations " *
            (shift_violations == 0 ? "✅" : "❌"))

    # 5c. Speedup benchmark: full local search with live zone on vs off
    println()
    println("  Local search speedup (live zone filter):")
    time_off_arr = Float64[]; time_on_arr = Float64[]
    profit_off_arr = Float64[]; profit_on_arr = Float64[]
    for trial in 1:3
        Random.seed!(trial * 200)
        p_off = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
        p_off.current_profit = fast_split_sparse_profit(p_off.position, p_off, pso)
        ENABLE_LIVE_ZONE_FILTER[] = false
        reset_boundary_stats!()
        t_off = @elapsed local_search_fully_incremental!(p_off, 1, pso)

        Random.seed!(trial * 200)
        p_on = Particle(copy(perm), copy(perm), 0.0, 0.0, compute_node_to_position(perm))
        p_on.current_profit = fast_split_sparse_profit(p_on.position, p_on, pso)
        ENABLE_LIVE_ZONE_FILTER[] = true
        reset_boundary_stats!()
        t_on = @elapsed local_search_fully_incremental!(p_on, 1, pso)

        push!(time_off_arr, t_off); push!(time_on_arr, t_on)
        push!(profit_off_arr, p_off.current_profit); push!(profit_on_arr, p_on.current_profit)
        println("    Trial $trial: OFF=$(round(t_off,digits=3))s P=$(round(p_off.current_profit,digits=4)) | ON=$(round(t_on,digits=3))s P=$(round(p_on.current_profit,digits=4))")
    end
    ENABLE_LIVE_ZONE_FILTER[] = false
    avg_off = mean(time_off_arr); avg_on = mean(time_on_arr)
    println("    Avg: OFF=$(round(avg_off,digits=3))s ON=$(round(avg_on,digits=3))s speedup=$(round(avg_off/max(avg_on,1e-12),digits=2))×")
    println("    Avg profit: OFF=$(round(mean(profit_off_arr),digits=4)) ON=$(round(mean(profit_on_arr),digits=4))")
    println()

    return swap_violations == 0 && shift_violations == 0
end

passed_lz = test_live_zone_real(pso, permutation)

println("="^60)
println("SUMMARY")
println("="^60)
println("  Per-swap correctness:  $(passed1 ? "✅" : "❌")")
println("  Per-shift correctness: $(passed2 ? "✅" : "❌")")
println("  Live zone correctness: $(passed_lz ? "✅" : "❌")")
all_passed = passed1 && passed2 && passed_lz
println(all_passed ? "\nOverall: ✅ CORRECTNESS VERIFIED" : "\nOverall: ❌ ISSUES DETECTED")
println("\nCompleted at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
