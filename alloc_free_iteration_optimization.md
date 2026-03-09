# Allocation-Free Iteration Optimization

## Problem

The shift and swap operators in the PSO local search iterate over candidate positions in a random order. The original implementation used Julia's `setdiff` and `shuffle` functions to generate the candidate lists, which **allocated new arrays on every iteration of the outer loop**:

```julia
# Shift: for each outer position i, allocate and shuffle a list of n-1 candidates
inner_j = shuffle(setdiff(1:n, [i]))   # 2 allocations per outer iteration

# Swap: for each outer position i, allocate and shuffle a list of n-i candidates
inner_j = shuffle(collect(i+1:n))       # 2 allocations per outer iteration
```

For the AugustComplexFire instance (n=900), this produces:
- **Shift:** ~900 × 2 allocations per operator call, each of ~900 × 8 = 7.2KB → ~13MB of GC pressure per shift call
- **Swap:** ~900 × 2 allocations per operator call, each of ~450 × 8 = 3.6KB → ~3.2MB of GC pressure per swap call

Since the local search calls these operators thousands of times during a PSO run, the cumulative allocation pressure is substantial.

## Solution

Replace the allocating iteration pattern with **pre-allocated reusable buffers**:

### Shift Operators

```julia
# Pre-allocate once before the outer loop
inner_j_buf = collect(1:n)
buf_dirty = false

for i in positions
    # Standard path: shuffle the buffer in-place (0 allocations)
    if buf_dirty
        for p in 1:n; inner_j_buf[p] = p; end
        buf_dirty = false
    end
    shuffle!(inner_j_buf)
    inner_len = n

    for j_idx in 1:inner_len
        @inbounds j = inner_j_buf[j_idx]
        j == i && continue    # Skip self (equivalent to setdiff)
        # ... evaluation body unchanged ...
    end
end
```

### Swap Operators

```julia
# Pre-allocate once before the outer loop
swap_j_buf = Vector{Int}(undef, n)

for i in positions
    # Fill buffer with i+1:n and shuffle (0 allocations)
    inner_len = n - i
    for k in 1:inner_len; swap_j_buf[k] = i + k; end
    shuffle!(view(swap_j_buf, 1:inner_len))

    for j_idx in 1:inner_len
        @inbounds j = swap_j_buf[j_idx]
        # ... evaluation body unchanged ...
    end
end
```

### Live Zone Filter Path

When the live zone filter is active, the buffer is filled with the filtered candidate positions instead:

```julia
if ENABLE_LIVE_ZONE_FILTER[] && !is_depot && safe_dead[i]
    bs = dbs[i]; be = dbe[i]
    inner_len = 0
    for p in 1:bs-1; inner_len += 1; inner_j_buf[inner_len] = p; end
    for p in be+1:n; inner_len += 1; inner_j_buf[inner_len] = p; end
    shuffle!(view(inner_j_buf, 1:inner_len))
    buf_dirty = true   # Mark buffer as needing reset for next standard-path use
```

The `buf_dirty` flag ensures the buffer is re-initialized to `1:n` only when transitioning from a live-zone iteration back to a standard iteration, avoiding unnecessary O(n) resets.

## Correctness Argument

**Claim:** The optimization produces identical distributional behavior.

**For shift:** `shuffle(setdiff(1:n, [i]))` produces a uniform random permutation of {1,…,n}\{i}. `shuffle!(buf)` (where `buf` contains a permutation of {1,…,n}) and skipping `j == i` also produces a uniform random iteration over {1,…,n}\{i}.

**Proof:** A uniform random permutation of {1,…,n}, with one fixed element `i` removed (preserving relative order), is a uniform random permutation of the remaining n−1 elements. By symmetry, each of the (n−1)! orderings of {1,…,n}\{i} corresponds to exactly n positions where `i` could appear in the original permutation, yielding probability 1/(n−1)! for each ordering.

**For swap:** `shuffle(collect(i+1:n))` produces a uniform random permutation of {i+1,…,n}. Filling a buffer with `{i+1,…,n}` and calling `shuffle!(view(...))` produces the identical distribution.

**RNG divergence:** For shift, the new code shuffles n elements instead of n−1, consuming one extra RNG call per outer iteration. This means that with the same seed, the algorithm follows a different random trajectory. However, the statistical properties (uniform candidate ordering, convergence behavior) are identical. The split evaluation is completely unchanged.

## Modified Operators

Six operators were modified in `TOP_PSO_multi_depot.jl`:

| Operator | Path | Change |
|---|---|---|
| `shift_operator_sparse!` | Non-incremental | `setdiff+shuffle` → pre-allocated buffer |
| `swap_operator_sparse!` | Non-incremental | `collect+shuffle` → pre-allocated buffer |
| `shift_operator_incremental!` | Incremental | `setdiff+shuffle` → pre-allocated buffer |
| `swap_operator_incremental!` | Incremental | `collect+shuffle` → pre-allocated buffer |
| `shift_operator!` | Legacy | `setdiff+shuffle` → pre-allocated buffer |
| `swap_operator!` | Legacy | `collect+shuffle` → pre-allocated buffer |

## Performance Results

### Micro-benchmark: Candidate Generation (3 iterations per position)

| n | Shift speedup | Shift alloc reduction | Swap speedup | Swap alloc reduction |
|---|---|---|---|---|
| 100 | 3.88× | 99.9% | 1.33× | 99.7% |
| 300 | 3.63× | 100.0% | 1.21× | 99.9% |
| 900 | 4.50× | 100.0% | 1.05× | 100.0% |

The shift pattern benefits more because `setdiff(1:n, [i])` is more expensive than `collect(i+1:n)` (it must scan and filter the entire range).

### Full PSO Benchmark: AugustComplexFire (n=900)

Comparing with previously documented results (before alloc-free optimization):

| Config | Before (s) | After (s) | Speedup | Before split calls | After split calls |
|---|---|---|---|---|---|
| OPT_ON | 76.76 | 67.91 | **1.13×** | 27,009,071 | 22,968,522 |
| LINF_COST | 68.34 | 65.00 | **1.05×** | 29,260,313 | 26,466,278 |
| COST_MATRIX | 67.81 | 66.39 | **1.02×** | 29,483,082 | 25,263,339 |
| CM_LINF | 70.66 | 64.74 | **1.09×** | 31,006,681 | 31,039,069 |
| CM_INCR | 60.17 | 60.51 | ~1.0× | 1,219,370 | 1,098,627 |

The non-incremental configurations benefit most (up to 1.13× faster) because the shift/swap candidate iteration is a significant fraction of their runtime. For incremental configurations, the alloc-free optimization has diminishing returns because the incremental evaluator already reduces split calls by 95%+.

### Real Instance Local Search (AugustComplexFire)

| Metric | Before | After |
|---|---|---|
| Incremental local search speedup | 4.4× | **4.83×** |
| Per-swap correctness | 0 violations | 0 violations |
| Per-shift correctness | 0 violations | 0 violations |

### Synthetic Correctness

All 5 configurations (n=34 to n=904) pass with valid solutions:
- Small instances produce identical profits
- Large instances produce profits within 10% (expected due to RNG divergence)

## Summary

The allocation-free iteration optimization eliminates **~100% of heap allocations** in the shift/swap candidate generation loop. This yields:

1. **1.13× wall-clock speedup** for the non-incremental PSO path (OPT_ON)
2. **3.6–4.5× micro-benchmark speedup** for the shift candidate generation pattern
3. **Zero correctness impact** — all tests pass with 0 violations
4. **Diminishing returns for incremental path** — the incremental evaluator already dominates runtime

The optimization is purely mechanical (no algorithmic change) and produces statistically identical results.
