# Incremental Tour Updates for Local Search in the Giant-Tour Split

## Abstract

We describe an incremental evaluation strategy for the swap and shift local search operators used in our PSO-based solver for the Team Orienteering Problem (TOP). Instead of rerunning the full split procedure (Phase 1 + Phase 2 DP) for every trial move, we maintain a **Tour Cache** and exploit the structure of each operator to:

1. **Identify affected tours** via a lightweight breakpoint check (\(O(k)\)).
2. **Recompute only the affected tours** (\(O(L_{\text{affected}})\) instead of \(O(L)\)).
3. **Skip the DP entirely** when no tour actually changed.

On the AugustComplexFire instance (\(n = 900\), \(k = 2\)), this yields a **4.3× full local search speedup** with identical solution quality.

---

## 1. Background

### 1.1 Split Procedure Recap

The split procedure converts a giant-tour permutation \(\pi\) into feasible drone routes in three phases:

- **Phase 1 (Saturated Tours):** For each depot at position \(d\), greedily extend a tour \(\pi_d, \pi_{d+1}, \ldots\) until the battery constraint is violated. Record profit \(P[t]\), length \(\ell[t]\), and successor position \(\text{succ}[t]\).
- **Phase 2 (DP):** Select up to \(m\) non-overlapping tours to maximize total profit.
- **Phase 3 (Backtracking):** Reconstruct the chosen routes (only needed for accepted moves).

### 1.2 Local Search Operators

The local search alternates two operators until no improvement is found:

- **Swap(i, j):** Exchange the nodes at positions \(i\) and \(j\) in the permutation.
- **Shift(i, j):** Remove the node at position \(i\) and insert it at position \(j\) (using `move_element` semantics: the element lands at position \(j{-}1\) when \(i < j\), or position \(j\) when \(i > j\)).

Each trial move currently requires a full split evaluation. With \(O(n^2)\) candidate moves per operator call, this dominates the PSO runtime.

---

## 2. Tour Cache

We cache the Phase 1 output across evaluations:

```julia
mutable struct TourCache
    sorted_depot_positions::Vector{Int}   # Sorted depot positions in permutation
    P_sparse::Vector{Float64}             # Profit of each saturated tour
    succ_sparse::Vector{Int}              # Successor position of each tour
    tour_lengths_sparse::Vector{Int}      # Length of each tour
end
```

The cache is initialized once from a full split, then maintained incrementally as moves are accepted. For rejected (trial) moves, we save and restore the affected entries.

---

## 3. Incremental Swap

### 3.1 Key Observation

A swap(i, j) exchanges two nodes in the permutation. This can only affect a tour if position \(i\) or \(j\) falls within the tour's **influence range** \((d, d + \ell]\), where \(d\) is the depot position and \(\ell\) is the tour length.

### 3.2 Algorithm

```
SWAP_INCREMENTAL(cache, pos, i, j):
  1. If either node is a depot → fall back to full split (rare: k/n fraction)
  2. Perform trial swap: pos[i], pos[j] ← pos[j], pos[i]
  3. Find affected tours: { t : i ∈ (d_t, d_t + ℓ_t] or j ∈ (d_t, d_t + ℓ_t] }
  4. Save old values for affected tours
  5. Recompute affected tours via greedy walk from their depot
  6. If no tour actually changed (P, ℓ, succ all identical) → skip DP, reject
  7. Otherwise, run Phase 2 DP on the updated cache
  8. Accept if profit improved; otherwise revert swap and cache
```

### 3.3 Depot Swap Fallback

When a depot node is involved in a swap, its position changes, invalidating the cache's `sorted_depot_positions`. Since depot swaps are rare (only \(k/n\) of pairs involve a depot), we fall back to a full split and rebuild the cache. This maintains correctness without measurable performance impact.

### 3.4 Complexity

| Component | Full split | Incremental |
|---|---|---|
| Phase 1 | \(O(L)\) | \(O(L_{\text{affected}})\) |
| Phase 2 (DP) | \(O(k \cdot m \cdot \log k)\) | Skipped ~81% of the time |
| Phase 3 | \(O(k)\) | Skipped for rejected moves |

---

## 4. Incremental Shift

### 4.1 Key Observation: Breakpoint Invariance

A shift(i, j) rearranges positions \([\min(i,j), \max(i,j)]\), but the **relative order** of nodes within the shifted block is preserved — they simply slide by one position. Only two "connections" in the node sequence break:

1. **Removal breakpoint at position \(i\):** The edge from \(\pi_{i-1}\) to \(\pi_i\) is severed.
2. **Insertion breakpoint at position \(j\):** The moved element is spliced in at the boundary between the changed and unchanged zones.

All consecutive node pairs *inside* the shifted block are preserved. Therefore, a tour whose range falls entirely within this block sees the exact same greedy walk and produces identical profit, length, and successor.

### 4.2 Affected Tour Condition

A tour at depot \(d\) with successor at \(d + \ell\) is affected if and only if one of the two breakpoint positions falls in its influence range:

\[
\boxed{i \in [d,\; d + \ell] \quad \text{or} \quad j \in (d,\; d + \ell]}
\]

The first condition uses \(\leq\) on the left (not strict \(<\)) because when \(d = i\), the depot itself is being relocated — this always affects the tour.

This is **much tighter** than the naïve range-overlap check \([d, d+\ell] \cap [\min(i,j), \max(i,j)] \neq \emptyset\), which would flag every tour that even partially overlaps the shift range.

### 4.3 Depot Position Updates

Unlike swap, a shift changes the absolute positions of depots within the shifted range:

| Case | Depot at \(d\) | New position |
|---|---|---|
| \(i < j\), \(d = i\) | \(j - 1\) |
| \(i < j\), \(i < d < j\) | \(d - 1\) |
| \(i > j\), \(d = i\) | \(j\) |
| \(i > j\), \(j \leq d < i\) | \(d + 1\) |
| Otherwise | \(d\) (unchanged) |

### 4.4 Successor Adjustment for Unaffected Tours

A subtle but critical detail: even when a tour's *node sequence* is invariant (breakpoints miss it), its successor's *absolute position* shifts along with the depot. For unaffected tours, we compute:

\[
\text{succ}_{\text{new}} = d_{\text{new}} + \ell_{\text{old}}
\]

rather than copying the old successor value. Failing to do this corrupts the DP's overlap checks.

### 4.5 Algorithm

```
SHIFT_INCREMENTAL(cache, pos, i, j):
  1. Compute breakpoints: bp1 = i, bp2 = j
  2. Find affected tours: { t : bp1 ∈ [d_t, d_t+ℓ_t] or bp2 ∈ (d_t, d_t+ℓ_t] }
  3. If no tour affected → profit unchanged, skip everything
  4. Compute new depot positions via shift rules (§4.3)
  5. Perform shift in-place: O(|i-j|), no allocation
  6. For each depot (sorted by new position):
     - If affected: recompute tour via greedy walk
     - If unaffected: copy old P and ℓ; set succ = d_new + ℓ_old
  7. Check if DP-relevant state changed (depot positions, P, ℓ, succ)
  8. If unchanged → skip DP, reject
  9. Otherwise, run Phase 2 DP and accept/reject
  10. On rejection: revert shift in-place
```

### 4.6 In-Place Shift and Revert

To avoid the \(O(n)\) allocation of `move_element` for every trial, we use an in-place shift with \(O(|i-j|)\) element moves and a matching revert:

```julia
function shift_in_place!(pos, i, j)    # Matches move_element semantics exactly
    saved = pos[i]
    if i < j
        for p in i:j-2; pos[p] = pos[p+1]; end
        pos[j-1] = saved
    else
        for p in i:-1:j+1; pos[p] = pos[p-1]; end
        pos[j] = saved
    end
end
```

---

## 5. DP Skip Optimization

Both operators benefit from a **DP skip**: after recomputing affected tours, we check whether any cached value actually changed. If the new \((P, \ell, \text{succ})\) values are identical to the old ones (and depot positions haven't moved), the split profit is guaranteed unchanged and we skip the \(O(k \cdot m \cdot \log k)\) DP entirely.

This is especially effective for swap, where ~81% of moves don't change any tour on the AugustComplexFire instance.

---

## 6. Combined Local Search

The `local_search_fully_incremental!` function integrates both operators:

```
1. Initialize TourCache from full split
2. Repeat until no improvement:
   a. Run swap_operator_incremental! (with blocking filter + incremental eval)
   b. Run shift_operator_incremental! (with irrelevance filter + breakpoint check)
3. Return improved particle
```

The cache is carried across operator calls within the same local search invocation, so accepted moves update the cache in-place without requiring a full split.

---

## 7. Correctness

### 7.1 Theoretical Argument

**Swap:** The only tours whose greedy walk can change are those whose influence range \((d, d+\ell]\) contains position \(i\) or \(j\). For all other tours, every intermediate edge cost and return distance is unchanged.

**Shift:** The node-pair relationships within the shifted block \([\min(i,j), \max(i,j)]\) are preserved by the slide. Only the two breakpoint positions (removal at \(i\), insertion boundary at \(j\)) introduce new node adjacencies. Tours whose range doesn't include either breakpoint see identical greedy walks.

### 7.2 Empirical Verification

| Test | Scope | Shifts/Swaps tested | Violations |
|---|---|---|---|
| Exhaustive (n=34, k=4) | All n(n-1) shifts | 1,122 | **0** |
| Exhaustive (n=56, k=6) | All n(n-1) shifts | 3,080 | **0** |
| Exhaustive (n=64, k=4) | All n(n-1) shifts | 4,032 | **0** |
| Sampled (n=304, k=4) | 5,000 random | 5,000 | **0** |
| Sampled (n=508, k=8) | 5,000 random | 5,000 | **0** |
| Sampled (n=904, k=4) | 5,000 random | 5,000 | **0** |
| AugustComplexFire (n=900) | 10,000 swap + 10,000 shift | 20,000 | **0** |
| Full local search (×5) | Complete runs | — | **Identical profits** |

Each test compares the incremental profit against a fresh `fast_split_sparse` on the modified permutation, with tolerance \(10^{-9}\).

---

## 8. Performance Results

### 8.1 Per-Evaluation Speedup (AugustComplexFire, n=900, k=2)

| Operator | Full split time | Incremental time | Speedup | DP skip rate |
|---|---|---|---|---|
| Swap | 0.020s (10k evals) | 0.0004s | **46×** | 81% |
| Shift | 0.025s (10k evals) | 0.0011s | **23×** | 75% |

### 8.2 Breakpoint Check vs Range Overlap (Shift)

| Metric | Range overlap | Breakpoint check | Improvement |
|---|---|---|---|
| Avg affected tours | 1.08 / 2 | 0.26 / 2 | −76% |
| DP skip rate | 28% | 75% | +47pp |
| Per-eval speedup | 6.8× | 23× | 3.4× better |

### 8.3 Full Local Search (AugustComplexFire)

| Metric | Original | Fully incremental |
|---|---|---|
| Avg time | 0.576s | **0.135s** |
| Speedup | — | **4.3×** |
| Avg profit | 2.7001 | 2.7001 |
| Profit difference | — | **0.0%** |

---

## 9. Implementation Reference

All functions are in `julia/TOP_PSO_multi_depot.jl`:

| Function | Purpose |
|---|---|
| `TourCache` (struct) | Cached Phase 1 arrays |
| `init_tour_cache` | Initialize cache from full split |
| `find_affected_tour_indices` | Swap: find tours covering positions i or j |
| `recompute_single_tour!` | Recompute one saturated tour in-place |
| `swap_operator_incremental!` | Full incremental swap with depot fallback |
| `compute_shifted_depot_positions` | New depot positions after shift(i,j) |
| `compute_tour_at` | Compute single tour at arbitrary depot position |
| `shift_in_place!` / `revert_shift_in_place!` | O(|i-j|) in-place shift and undo |
| `shift_operator_incremental!` | Full incremental shift with breakpoint check |
| `local_search_fully_incremental!` | Combined swap+shift local search |

Test files:
- `julia/test_incremental_swap.jl` — Exhaustive + sampled correctness and benchmarks on synthetic instances.
- `julia/test_incremental_swap_august_fire.jl` — Correctness and benchmarks on the AugustComplexFire instance.
