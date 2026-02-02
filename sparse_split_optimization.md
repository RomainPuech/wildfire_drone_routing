# Sparse Split Optimization for Team Orienteering with Giant Tours

## Abstract

We present a sparse reformulation of the split procedure used to evaluate giant-tour solutions for the Team Orienteering Problem (TOP). The baseline split computes saturated tours and a dynamic program (DP) over all positions in the permutation, yielding time complexity \(O(n \times m)\) for \(n\) nodes and \(m\) vehicles. For grid and sparse graphs, depot nodes are rare and tour starts are limited to depot positions. We exploit this structure to compute saturated tours and DP values only at depot positions, reducing the complexity to \(O(k \log k + L + k \times m \times \log k)\), where \(k \ll n\) is the number of depot positions and \(L\) is the total length of saturated tours. We formalize the method, provide proofs of correctness, and summarize the resulting complexity improvements.

---

## 1. Problem Setting and Notation

We consider a giant-tour permutation \(\pi = [\pi_1, \ldots, \pi_n]\) that interleaves customer nodes and depot markers. Customers are labeled \(1..n_{\text{pure}}\), depots are labeled \(> n_{\text{pure}}\). A saturated tour starting at depot position \(d\) is the maximal consecutive subsequence \(\pi_d, \pi_{d+1}, \ldots\) that respects the battery constraint.

We denote:

- \(n\): length of the permutation (customers + depots).
- \(m\): number of vehicles.
- \(k\): number of depot positions in the permutation (typically \(k \ll n\)).
- \(L\): total length of all saturated tours computed from depot positions.

---

## 2. Baseline Split Procedure (Dense Form)

The baseline split operates in three phases:

1. **Phase 1 (Saturated Tours)**: For each position \(i = 1..n\), if \(\pi_i\) is a depot, compute the maximal feasible tour starting at \(i\), recording its profit \(P[i]\), successor position \(\text{succ}[i]\), and tour length \(\ell[i]\).
2. **Phase 2 (DP)**: Compute a DP table \(\Gamma[i, j]\) giving the maximum profit from position \(i\) using \(j\) vehicles.
3. **Phase 3 (Backtracking)**: Reconstruct the optimal set of tours from the DP table.

The dense complexity is \(O(n)\) for Phase 1, \(O(n \times m)\) for Phase 2, and \(O(n)\) for Phase 3.

---

## 3. Key Insights for Sparsification

### Observation 3.1 (Tour Starts Are Sparse)
Only depot positions can start tours. Non-depot positions cannot start saturated tours, yet the dense algorithm evaluates all positions.

### Observation 3.2 (DP Values Propagate Through Non-Depots)
At non-depot positions, the saturated tour profit is zero and the successor is \(i+1\). The DP recurrence becomes:
\[
\\Gamma[i, j] = \max(\\Gamma[i+1, j], \\Gamma[i+1, j-1]) = \\Gamma[i+1, j]
\]
since \(\Gamma[i+1, j] \ge \Gamma[i+1, j-1]\). Thus, DP values are constant between consecutive depot positions.

These observations allow us to compute both Phase 1 and Phase 2 only at depot positions.

---

## 4. Data Structures

### Definition 4.1 (Depot Positions)

Let \(\mathcal{D} = [d_1 < d_2 < \cdots < d_k]\) be the sorted list of depot positions in the permutation.

We compute \(\mathcal{D}\) from the node-to-position mapping by collecting all depot node positions and sorting them. The cost is \(O(k)\) to collect and \(O(k \log k)\) to sort.

### Definition 4.2 (Sparse Tour Arrays)

For each depot index \(t = 1..k\) corresponding to position \(d_t\), we store:

- \(P_{\text{sparse}}[t]\): profit of saturated tour starting at \(d_t\)
- \(\text{succ}_{\text{sparse}}[t]\): next position after the tour (0 if it reaches the end)
- \(\ell_{\text{sparse}}[t]\): tour length

These arrays are indexed by depot index, not permutation position.

---

## 5. Phase 1 Optimization: Sparse Saturated Tours

We compute saturated tours only for depot positions, producing \(P_{\text{sparse}}\), \(\text{succ}_{\text{sparse}}\), and \(\ell_{\text{sparse}}\). A concise pseudocode version is provided in Appendix A.

### Complexity

Collecting and sorting depot positions is \(O(k \log k)\). Total tour construction work is \(O(L)\), where \(L\) is the sum of saturated tour lengths. Thus Phase 1 is \(O(k \log k + L)\), replacing the dense \(O(n)\).

---

## 6. Phase 2 Optimization: Sparse Dynamic Programming

We compute the DP only at depot indices. Let \(\Gamma_{\text{sparse}}[t, j]\) denote the maximum profit using \(j\) vehicles starting from depot index \(t\).

### Theorem 6.1 (Sparse DP Correctness)

Let \(d_t\) be the \(t\)-th depot position. For any position \(i\) between depots \(d_t\) and \(d_{t+1}\), the dense DP satisfies:
\[
\\Gamma[i, j] = \\Gamma[d_t, j].
\]

**Proof.**
By Observation 3.2, DP values propagate unchanged across non-depot positions, so all positions between depots share the same value as the next depot position. ∎

### Sparse DP Recurrence
The helper `lookup_Γ_sparse` performs a binary search for the first depot position at or after a given successor position. A compact recurrence is given in Appendix A.

### Complexity

The DP has \(O(k \times m)\) entries. Each uses an \(O(\log k)\) lookup. Thus Phase 2 runs in \(O(k \times m \times \log k)\), replacing \(O(n \times m)\).

---

## 7. Phase 3 Optimization: Sparse Backtracking

Backtracking proceeds depot-by-depot using the sparse DP table and the \(\text{succ}_{\text{sparse}}\) pointers. A concise pseudocode version is in Appendix A.

### Complexity

Each iteration advances to the next depot index; each step uses a binary search. Total complexity is \(O(k \log k)\), replacing \(O(n)\).

---

## 8. Complete Sparse Split

The full sparse split combines the three optimized phases: sparse saturated tours, sparse DP, and sparse backtracking. The overall structure is summarized in Appendix A.

---

## 9. Complexity Summary

Let \(k\) be the number of depot positions and \(L\) be the total length of saturated tours.

| Phase | Dense | Sparse |
|------|-------|--------|
| Phase 1 (tours) | \(O(n)\) | \(O(k \log k + L)\) |
| Phase 2 (DP) | \(O(n \times m)\) | \(O(k \times m \times \log k)\) |
| Phase 3 (backtracking) | \(O(n)\) | \(O(k \log k)\) |
| **Total** | \(O(n \times m)\) | \(O(k \log k + L + k \times m \times \log k)\) |

When \(k \ll n\) (typical for grid-based monitoring), the sparse split yields substantial speedups.

---

## 10. Practical Notes

1. **Node-to-position mapping** should be stored per particle to access depot positions in \(O(k)\).
2. **Binary search** is used to map successor positions to depot indices.
3. **Overloads** of the sparse split can compute `node_to_position` on-the-fly when needed for non-particle permutations.

---

## Appendix A: Pseudocode Summary

```text
SPARSE_SPLIT(π):
  D ← sorted depot positions
  Phase 1: compute P_sparse, succ_sparse, len_sparse at depot positions only
  Phase 2: compute Γ_sparse over depot indices using binary-search lookup
  Phase 3: backtrack depot-by-depot using Γ_sparse and succ_sparse
  return optimal profit and routes

LOOKUP_GAMMA(position, j):
  idx ← first depot position ≥ position (binary search)
  if idx out of range or j < 0: return 0
  return Γ_sparse[idx, j+1]
```

---

