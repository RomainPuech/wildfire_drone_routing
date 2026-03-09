# Live Zone Optimization for Shift/Swap Candidate Iteration

> **Status: NOT RECOMMENDED.** While provably correct and effective at reducing candidate counts (74.5% swap reduction, 26.4% shift reduction), this optimization **hurts final solution quality** in practice. Skipping dead-zone candidate pairs removes diversification that helps the PSO find better solutions over many iterations. See [Verdict](#verdict) for details.

## Overview

The **live zone optimization** reduces the number of shift/swap candidate pairs evaluated during local search by restricting iteration to positions that can actually affect the split profit. Positions deep inside the "dead zone" (not covered or read by any saturated tour) are provably irrelevant — swapping or shifting them has no effect on the split result.

In the AugustComplexFire instance (n=900, k=2, L=63), only ~13% of positions are "live" (covered by tours plus their boundary). The remaining ~87% are dead. This optimization eliminates dead–dead candidate pairs entirely from the iteration, reducing swap candidates by **74.5%** and shift candidates by **26.4%**.

---

## Key Concepts

### Dead Positions (Original)

A position `p` in the permutation is **dead** if no saturated tour covers it:

```
dead_positions[p] = true  iff  p ∉ [d, d + tour_length - 1]  for all depots d
```

### Safe Dead Positions (Extended by +1)

The original dead position definition is **insufficient** for filtering because the greedy tour construction **reads** the first position after the tour (the "boundary") to decide whether to extend. Changing the node at the boundary can alter the tour's extension decision.

The **safe dead positions** extend each tour's coverage by +1:

```
covered range = [d, d + tour_length]    (instead of [d, d + tour_length - 1])
```

This includes the boundary position that the greedy construction reads. A position that is safe-dead is guaranteed to not be read by any tour's greedy computation.

### Dead Blocks

A **dead block** is a maximal contiguous range of safe-dead positions. For each dead position, we precompute `block_start[p]` and `block_end[p]`.

---

## Correctness Proof

### Swap(i, j) where both i and j are safe-dead

The swap only modifies `pos[i]` and `pos[j]`. The split procedure computes tours by scanning from depot positions. Each tour reads positions `[d, d + tour_length]` (including the boundary). If both `i` and `j` are outside all such ranges, no tour reads either position. All tours produce the same profit. **The split result is unchanged.** ∎

### Shift(i, j) where the entire range [min(i,j), max(i,j)] is safe-dead

The shift rearranges positions in `[min(i,j), max(i,j)]`:
1. No depot is in this range (depot positions are always live — they start their own tour).
2. No tour reads any position in this range (all positions are safe-dead).
3. All positions outside the range are unchanged.
4. All tours compute the same greedy path and produce the same profit.

**The split result is unchanged.** ∎

---

## Implementation

### Toggle

```julia
const ENABLE_LIVE_ZONE_FILTER = Ref(false)
```

### Helper Functions

**`compute_safe_dead_positions(n, sorted_depot_positions, tour_lengths_sparse)`**
- Returns a `BitVector` where `true` = position is not read by any tour's greedy construction.
- Extends each tour's coverage by +1 compared to `compute_dead_positions`.

**`compute_dead_block_boundaries(dead_positions)`**
- Returns `(block_start, block_end)` arrays.
- `block_start[p]` = first position in p's contiguous dead block (0 if p is live).
- `block_end[p]` = last position in p's contiguous dead block (0 if p is live).

### Swap Operator Modification

When `ENABLE_LIVE_ZONE_FILTER[]` is true and position `i` is a safe-dead customer:
- Compute sorted live positions via binary search.
- Only iterate inner `j` over live positions > i.
- Skip entirely if no live positions exist after `i`.

**Effect:** Eliminates all dead–dead swap pairs from iteration. For dead `i`, the inner loop shrinks from O(n) to O(|live zone|).

### Shift Operator Modification

When `ENABLE_LIVE_ZONE_FILTER[]` is true and position `i` is safe-dead:
- Compute `i`'s dead block `[bs, be]`.
- Only iterate inner `j` over positions outside the dead block.
- All `j` within the same dead block produce a range entirely within the dead zone → safe to skip.

**Effect:** Eliminates within-dead-block shift pairs. For dead `i`, the inner loop shrinks from O(n) to O(n - block_size).

### Precomputation Cost

Per operator call: O(n) to compute safe dead positions and dead block boundaries. This is negligible compared to the O(n²) iteration savings.

---

## Operators Modified

| Operator | File | Change |
|---|---|---|
| `swap_operator_sparse!` | TOP_PSO_multi_depot.jl | Dead i → inner loop over live j only |
| `swap_operator_incremental!` | TOP_PSO_multi_depot.jl | Same |
| `shift_operator_sparse!` | TOP_PSO_multi_depot.jl | Dead i → inner loop outside dead block |
| `shift_operator_incremental!` | TOP_PSO_multi_depot.jl | Same |

---

## Performance Results

### Synthetic Instances (test_incremental_swap.jl)

| Instance (n) | Live % | Swap candidates OFF→ON | Swap reduction | Shift candidates OFF→ON | Shift reduction |
|---|---|---|---|---|---|
| 304 | 2.0% | 2862 → 85 | 97.0% | 1723 → 98 | 94.3% |
| 508 | 2.0% | 2283 → 1 | ~100% | 12670 → 9834 | 22.4% |
| 904 | 0.7% | 4345 → 1 | ~100% | 35409 → 19784 | 44.1% |

All dead-zone correctness tests pass with **0 violations**.

### Real Instance (test_incremental_swap_august_fire.jl)

- **Instance:** AugustComplexFire, n=900, k=2, L=63
- **Live zone:** 119/900 positions (13.2%)
- **Dead-dead swap correctness:** 0 violations / 2491 tested ✅
- **Within-dead-block shift correctness:** 0 violations / 4981 tested ✅
- **Local search speedup:** 1.25× (0.133s → 0.107s, identical profit 2.7001)

### Full PSO Benchmark (test_pso_august_complex_fire.jl)

| Config | Profit | Elapsed | Split calls |
|---|---|---|---|
| CM_INCR (baseline) | 0.047085 | 60.17s | 1,219,370 |
| **LZ_INCR** | **0.046650** | **60.13s** | **1,342,921** |
| CM_INCR_LINF | 0.046895 | 60.83s | 718,500 |
| LZ_INCR_LINF | 0.045887 | 60.91s | 582,867 |

**Candidate reduction (LZ_INCR vs CM_INCR):**
- Swap candidates: 270M → 69M (**74.5% reduction**)
- Shift candidates: 441M → 325M (**26.4% reduction**)

**Wall-clock speedup:** ~1.0× (time-limited at 60s — the algorithm does more useful work per unit time).

---

## Analysis

The live zone filter is most effective at reducing candidate counts when:
1. **The dead zone is large** (typical for grid instances with sparse depots where n >> k×L).
2. **The swap operator dominates** — swap has n² pairs, and dead outer positions with only live inner targets shrink the inner loop dramatically.
3. **Dead blocks are large** — the shift inner loop shrinks by the block size.

The per-local-search-call speedup is **1.25×** (0.133s → 0.107s on AugustComplexFire).

---

## Verdict

**Not recommended for production use.**

Despite being provably correct and delivering significant candidate reduction, the live zone filter **degrades final solution quality** in the full PSO benchmark:

| Config | Profit | Δ vs baseline |
|---|---|---|
| CM_INCR (no LZ) | **0.047085** | — |
| LZ_INCR (with LZ) | 0.046650 | **−0.00044** |
| CM_INCR_LINF (no LZ) | **0.046895** | — |
| LZ_INCR_LINF (with LZ) | 0.045887 | **−0.00101** |

**Why it hurts:** Dead-zone swaps and shifts don't directly improve profit, but they shuffle nodes in the permutation. This shuffling provides **diversification** — it changes the relative order of nodes, which can create new improving opportunities in subsequent iterations. By filtering out these "useless" moves, the PSO loses exploration breadth and converges to slightly worse solutions.

The optimization is **orthogonal** to and **stacks with** the incremental tour updates, cost matrix, and boundary optimization filters — but the net effect on solution quality is negative, so `ENABLE_LIVE_ZONE_FILTER` should remain `false`.
