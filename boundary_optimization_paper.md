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

Shift positions `i < j`: remove `π_i`, shift `π_{i+1..j-1}` left by one, insert the removed node **before** the original position `j` (i.e., at position `j-1` in the new permutation).

### Theorem 4.3 (Swap Boundary)

If positions `i` and `j` (with `i < j`) are non-depot nodes, then only tours that include positions `i-1` or `j-1` can change profit after swapping `i` and `j`.

**Proof.**
Only the nodes at positions `i` and `j` are changed by a swap. A tour’s profit depends only on the sequence of nodes it visits; thus a tour’s profit can change only if it includes position `i` or `j`. For non-depot positions, any tour including `k` must include `k-1` (since tours start at depots). Therefore, affected tours are exactly those containing `i-1` or `j-1`. ∎

### Theorem 4.4 (Shift Boundary)

If positions `i < j` are non-depot nodes, then only tours that include positions `i-1` or `j-1` can change profit after shifting `i → j`.

**Proof.**
The shift removes `π_i` from its original location and inserts it immediately after `π_{j-1}` (before the original `π_j`). Any tour containing position `i` must include `i-1` and sees `π_i` removed; any tour containing position `j-1` may extend to include `π_i` after insertion. Positions strictly between `i` and `j-1` preserve node order (only shifted), so their tour sequences are unchanged. Therefore, only tours containing `i-1` or `j-1` can change profit. ∎

---

## 5. Irrelevance-Based Shift Filter

We replace profit-only checks with a conservative **irrelevance** criterion that guarantees no false negatives.

### Definition 5.1 (irrelevant_once_removed)

For a non-depot node at position `i`, define `irrelevant_once_removed(i)` as follows:

1. If `node_i` is **blocking or dead**:
   - If `is_blocking_once_removed(i)` is true, removal does **not** change any tour containing position `i-1` → **irrelevant**.
   - If `is_blocking_once_removed(i)` is false, removal can extend a tour → **relevant**.
2. If `node_i` is **not blocking/dead**, it belongs to a tour → **relevant**.

### Definition 5.2 (irrelevant_once_inserted_at_position_j)

Assume `irrelevant_once_removed(i)` is true. Let the shift insert `node_i` **before** position `j`. Then:

- If the node at position `j-1` is **dead**, insertion at `j` does **not** change any tour → **irrelevant**.
- If the node at position `j-1` is **not dead**, insertion **may** change a tour → **relevant**.

Because `irrelevant_once_removed(i)` guarantees no tours change under removal, “deadness” of the node at `j-1` is invariant before/after removal (for the same node identity).

### Theorem 5.3 (Safe Skip Criterion for Shift)

If both `irrelevant_once_removed(i)` and `irrelevant_once_inserted_at_position_j(i, j)` are true, then shifting `i → j` does not change any tour and cannot increase the optimal profit.

**Proof.**
By Observation 4.4, only tours containing positions `i-1` or `j-1` can change. The first predicate guarantees no tour containing `i-1` changes under removal; the second guarantees no tour containing `j-1` changes under insertion. Therefore, no tour changes at all, and the DP optimum cannot increase. ∎

---

## 6. Candidate Reduction (Optional)

### Observation 6.1 (Adjacency-Based Candidate Reduction)

If `irrelevant_once_removed(i)` is true, then insertion at `j` can only matter when the predecessor `π_{j-1}` is connected to `π_i` in the clients graph (i.e., `π_{j-1} ∈ left_neighbors(π_i)`).

**Justification.**
Insertion can only affect a tour if the tour reaches `π_{j-1}` and can extend to `π_i`, which requires the edge `π_{j-1} → π_i`.

### Practical Rule

When `irrelevant_once_removed(i)` holds, restrict candidate positions to
```
j = position(node) + 1  for node ∈ left_neighbors(π_i)
```
This reduces the scan from `O(n)` to `O(deg(π_i))` in sparse graphs.

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
SHIFT_FILTER(i, j):
  if irrelevant_once_removed(i) && irrelevant_once_inserted_at_position_j(i, j):
      skip move
  else:
      evaluate full shift i → j

OPTIONAL CANDIDATE REDUCTION:
  if irrelevant_once_removed(i):
      for node in shuffled(left_neighbors(π_i)):
          j = position(node) + 1
          evaluate SHIFT_FILTER(i, j)
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

---

## Appendix C: Prior “Perfect Filter” (Deprecated)

This appendix records the earlier **profit-only filter** for historical reference. It is **incorrect** because it can miss improvements caused by changes in tour overlap structure (i.e., different combinations become feasible even if no individual tour profit increases).

**Prior criterion (deprecated):**
> Evaluate a shift only if at least one affected tour increases in profit.

**Why it fails:**  
Tour profits can stay the same or decrease while **tour lengths/overlaps change**, enabling the DP to select a different combination with higher total profit.
