# Boundary Optimizations for Local Search in Team Orienteering with a Giant-Tour Split

## Abstract

We study boundary-based optimizations for local search operators in a Particle Swarm Optimization (PSO) framework applied to the Team Orienteering Problem (TOP) on grid or sparse graphs. Solutions are encoded as a giant tour (a permutation) and evaluated by a split procedure that constructs feasible routes starting at depot nodes. We derive formal conditions under which local search moves (swap and shift) can affect the split profit, and we provide provably safe filtering rules that avoid unnecessary recomputation. For the shift operator, we introduce a two-stage filter that leverages client-graph sparsity via left-neighbor restriction. For the swap operator, we formalize a lightweight blocking-edge check that can be evaluated in constant time. These results enable significant computational savings without loss of solution quality.

---

## 1. Problem Setting

### 1.1 Team Orienteering Problem (TOP)

Given a set of customer nodes with profits and a set of depots, the TOP seeks a set of routes (one per vehicle) that maximize total collected profit subject to a travel-time (battery) constraint and without revisiting customers.

### 1.2 Giant-Tour Encoding and Split

A solution is represented by a **giant tour** (a permutation of customers and depot markers). A **split procedure** converts this permutation into feasible routes:

- A route starts at a depot marker.
- Nodes are visited in permutation order until a battery constraint would be violated.
- A Dynamic Programming (DP) program selects up to `m` routes (vehicles) to maximize total profit.

---

## 2. Definitions

### Definition 2.1 (Permutation)

Let `π = [π₁, π₂, ..., πₙ]` denote the giant tour permutation. Customer nodes are labeled `1..n_pure_customers`. Depot nodes are labeled `> n_pure_customers`.

### Definition 2.2 (Saturated Tour)

For each depot position `d` (i.e., `π_d` is a depot), a **saturated tour** is the maximal consecutive subsequence starting at `d` that respects the battery constraint. Its profit is the sum of profits of visited customers.

### Definition 2.3 (Tour Membership and Dead Nodes)

A position `k` **belongs to** a saturated tour starting at depot position `d` if `d ≤ k < d + length(d)`. A position is **dead** if it belongs to no saturated tour. Depot positions are never dead (they belong to their own tour).

### Definition 2.4 (Clients Graph)

Let `G = (V, E)` be the directed graph of feasible transitions between customers. An edge `(u, v) ∈ E` exists if moving from `u` to `v` is feasible (cost defined and within battery bounds). For node `v`, its **left-neighbors** are nodes `u` such that `(u, v) ∈ E`.

---

## 3. Fundamental Observations

### Observation 3.1 (Tour Structure)

Every saturated tour starts at a depot position in the permutation and has a well-defined length and profit.

### Observation 3.2 (Tour Membership)

A node can belong to multiple saturated tours. Depot nodes are never dead.

### Observation 3.3 (Optimal Tour Selection)

The DP phase selects a subset of saturated tours maximizing total profit, subject to vehicle limits and non-overlap.

---

## 4. Boundary Effects of Local Search Operators

We analyze **swap** and **shift** operators on the permutation.

### Definition 4.1 (Swap)

Swap positions `i < j`: exchange `π_i` and `π_j`.

### Definition 4.2 (Shift)

Shift positions `i < j`: remove `π_i`, shift `π_{i+1..j}` left by one, insert the removed node at position `j`.

### Theorem 4.3 (Swap Boundary)

If positions `i` and `j` (with `i < j`) are non-depot nodes, then only tours that include positions `i-1` or `j-1` can change profit after swapping `i` and `j`.

**Proof.**
Only the nodes at positions `i` and `j` are changed by a swap. A tour’s profit depends only on the sequence of nodes it visits; thus a tour’s profit can change only if it includes position `i` or `j`. For non-depot positions, any tour including `k` must include `k-1` (since tours start at depots). Therefore, affected tours are exactly those containing `i-1` or `j-1`. ∎

### Theorem 4.4 (Shift Boundary)

If positions `i < j` are non-depot nodes, then only tours that include positions `i-1` or `j` can change profit after shifting `i → j`.

**Proof.**
The shift removes `π_i` from its original location and inserts it immediately after `π_j`. Any tour containing position `i` must include `i-1` and sees `π_i` removed; any tour containing position `j` may extend to include `π_i` after insertion. Positions strictly between `i` and `j` preserve node order (only shifted), so their tour sequences are unchanged. Therefore, only tours containing `i-1` or `j` can change profit. ∎

---

## 5. Perfect Filtering Criterion

### Theorem 5.1 (Zero False Negatives)

Let `P` be the optimal profit for permutation `π`, and `P'` be the optimal profit after a local move. If **no affected tour** increases in profit, then `P' ≤ P`.

**Proof.**
Unaffected tours have identical node sequences, hence identical profits. Affected tours have profits that are less than or equal to their previous values by assumption. Therefore, all tour profits in the new permutation are ≤ their original values. The DP maximizes a sum of tour profits; hence the optimal value cannot increase. ∎

**Corollary.** A move should be evaluated with the full split procedure **only if** at least one affected tour increases in profit. This yields a perfect filter with zero false negatives.

---

## 6. Two-Stage Filter for Shift

### Definition 6.1 (Stage 1: Removal Test)

Given a shift candidate index `i`, Stage 1 constructs the permutation `π^(-i)` obtained by removing `π_i` and shifting positions `i+1..n` left by one. Stage 1 **recomputes only** the profits of tours containing position `i-1` in the original permutation. Stage 1 **succeeds** if at least one of these tours improves in profit under `π^(-i)`.

### Definition 6.2 (Stage 2: Insertion-Only Scan)

If Stage 1 fails, Stage 2 evaluates shifts `i → j` but **restricts** candidate positions `j` to the current positions of left-neighbors of `π_i` in the clients graph. The left-neighbor list is randomized before iteration, and each candidate `j` is evaluated using the perfect filter.

### Observation 6.1 (Stage-2 Adjacency)

If Stage 1 finds no improvement among tours containing position `i-1`, then a tour containing the former position `j` can improve **only if** the edge `π_j → π_i` is feasible in the clients graph.

**Proof.**
After shifting `π_i` to position `j`, the only new node that can extend a tour containing `π_j` is `π_i`. Extending requires edge `π_j → π_i`. If infeasible, the tour cannot include `π_i`, so its profit cannot increase. ∎

### Algorithm 6.2 (Shift Filter)

**Stage 1 (Removal Test).**
Remove `π_i` and recompute profits only for tours containing position `i-1`. If any improves, run a full randomized scan over all `j`.

**Stage 2 (Insertion-Only Scan).**
If Stage 1 fails, restrict candidate `j` to positions of **left-neighbors** of `π_i`. Randomize the order of these neighbors, map them to positions via `node_to_position`, and evaluate only those `j`.

This reduces the candidate scan from `O(n)` to `O(deg(π_i))` in sparse graphs without sacrificing correctness.

---

## 7. Lightweight Blocking Check for Swap

Because swap is already inexpensive, we use a local **blocking** check rather than the perfect filter.

### Definition 7.1 (Blocking Node)

Let `u` be the predecessor of `v` in the permutation. Node `v` is **blocking** if:
- `u` is not dead, and
- edge `u → v` is infeasible.

### Definition 7.2 (Blocking Once Inserted)

When a node `x` is inserted after predecessor `u`, `x` is **blocking once inserted** if `u` is not dead and `u → x` is infeasible.

### Definition 7.3 (Blocking Once Removed)

When a node `x` is removed, its former successor `w` becomes adjacent to predecessor `u`. The successor `w` is **blocking once removed** if `u` is not dead and `u → w` is infeasible.

These tests are O(1) with a constant-time adjacency lookup.

---

## 8. Discussion and Practical Impact

- **Correctness**: The perfect filter (Theorem 5.1) cannot reject a move that would improve the final profit.
- **Efficiency**: The two-stage shift filter exploits sparsity by scanning only `deg(π_i)` candidates when Stage 1 fails.
- **Swap**: Blocking checks are fast and effective, and are sufficient in practice given swap’s low overhead.

---

## Appendix A: Pseudocode (Shift Filter)

```text
SHIFT_FILTER(i):
  # Stage 1: removal-only test
  remove π_i → π^(-i)
  if any tour containing position i-1 improves on π^(-i):
      for j in shuffled(all positions except i):
          evaluate full shift i → j with perfect filter
      return

  # Stage 2: left-neighbor restriction
  for node_j in shuffled(left_neighbors(π_i)):
      j = position(node_j)
      if j <= i: continue
      evaluate full shift i → j with perfect filter
```

---

## Appendix B: Pseudocode (Blocking Check for Swap)

```text
SWAP_BLOCKING_FILTER(i, j):
  # Skip if swap does not change blocking status in any relevant adjacency
  # Check blocking once inserted for each node at the other's position
  # Check blocking once removed for each node at its original position
  # If no change, skip swap
```

