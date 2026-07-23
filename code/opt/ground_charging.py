"""Sensor / charging-station placement MILPs.

Port of ``julia/ground_charging_opt.jl``.  All coordinates are **0-based**.

Provides
--------
sensor_maxcov_strategy  – basic Max-Coverage placement
max_coverage_kernel     – Gaussian-kernel coverage placement
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pyomo.environ as pyo

from . import get_solver
from .helpers import linf_distance, load_burn_map


# ======================================================================
# SENSOR_MAXCOV_STRATEGY
# ======================================================================

def sensor_maxcov_strategy(
    risk_pertime_file: str,
    n_grounds: int,
    n_charging: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Max-Coverage sensor + charging-station placement.

    Returns ``(ground_indices, charging_indices)`` – both lists of
    0-based ``(row, col)`` tuples.
    """
    t0 = time.time()

    risk_pertime = load_burn_map(risk_pertime_file)
    T, N, M = risk_pertime.shape
    avg_risk = risk_pertime.mean(axis=0)  # (N, M)

    # Feasible locations: cells with positive average risk
    I_prime = [(i, j) for i in range(N) for j in range(M) if avg_risk[i, j] > 0.0]
    I_second = list(I_prime)
    nP = len(I_prime)
    nS = len(I_second)
    if nP == 0:
        return [], []

    idx_p: Dict[Tuple[int, int], int] = {pt: k for k, pt in enumerate(I_prime)}

    # ---- build model ----
    m = pyo.ConcreteModel()
    rng_p = range(nP)
    rng_s = range(nS)
    m.x = pyo.Var(rng_p, domain=pyo.Binary)
    m.y = pyo.Var(rng_s, domain=pyo.Binary)

    # Objective: maximise risk-weighted placement
    m.obj = pyo.Objective(
        expr=sum(avg_risk[I_prime[k]] * m.x[k] for k in rng_p)
        + sum(avg_risk[I_second[k]] * m.y[k] for k in rng_s),
        sense=pyo.maximize,
    )

    # Mutual exclusion at each point
    m.excl = pyo.ConstraintList()
    for k in rng_p:
        m.excl.add(m.x[k] + m.y[k] <= 1)

    # Capacity
    m.cap_x = pyo.Constraint(expr=sum(m.x[k] for k in rng_p) <= n_grounds)
    m.cap_y = pyo.Constraint(expr=sum(m.y[k] for k in rng_s) <= n_charging)

    # Spatial exclusion (L-inf ≤ 10): bucket-accelerated
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for k, (px, py) in enumerate(I_second):
        buckets[(px // 11, py // 11)].append(k)

    m.sp_yy = pyo.ConstraintList()
    m.sp_yx = pyo.ConstraintList()
    for a_idx, a_pt in enumerate(I_second):
        bx, by = a_pt[0] // 11, a_pt[1] // 11
        for dbx in (-1, 0, 1):
            for dby in (-1, 0, 1):
                for b_idx in buckets.get((bx + dbx, by + dby), []):
                    if a_idx >= b_idx:
                        continue
                    if linf_distance(a_pt, I_second[b_idx]) > 10:
                        continue
                    m.sp_yy.add(m.y[a_idx] + m.y[b_idx] <= 1)
                    # Both directions for the y-x constraint
                    m.sp_yx.add(m.y[a_idx] + m.x[b_idx] <= 1)
                    if a_idx in idx_p:
                        m.sp_yx.add(m.y[b_idx] + m.x[a_idx] <= 1)

    # ---- solve ----
    solver = get_solver()
    result = solver.solve(m, tee=False)

    sel_x = [I_prime[k] for k in rng_p if pyo.value(m.x[k]) > 0.5]
    sel_y = [I_second[k] for k in rng_s if pyo.value(m.y[k]) > 0.5]

    print(f"SENSOR_MAXCOV_STRATEGY solved in {time.time() - t0:.2f}s  "
          f"({len(sel_x)} ground, {len(sel_y)} charging)")
    return sel_x, sel_y


# ======================================================================
# Max_Coverage_Kernel  (Gaussian coverage)
# ======================================================================

def max_coverage_kernel(
    static_map_file: str,
    n_grounds: int,
    n_charging: int,
    n_drones: int,
    kernel: Dict[Tuple[int, int], float],
    kernel_size_x: int,
    kernel_size_y: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Gaussian-kernel Max-Coverage placement (0-based output).

    *kernel* maps ``(dx, dy) -> coverage_value``.
    """
    t0 = time.time()

    static_map = load_burn_map(static_map_file)
    T, N, M = static_map.shape

    if T != 1:
        avg_risk = static_map[:10].mean(axis=0)
    else:
        avg_risk = static_map[0]

    nI = N * M

    def flat(x: int, y: int) -> int:
        return x * M + y

    def pt(idx: int) -> Tuple[int, int]:
        return divmod(idx, M)

    # ---- build model ----
    m = pyo.ConcreteModel()
    rng = range(nI)

    m.xg = pyo.Var(rng, domain=pyo.Binary)
    m.xc = pyo.Var(rng, domain=pyo.Binary)
    m.nc = pyo.Var(rng, domain=pyo.NonNegativeIntegers)
    m.theta = pyo.Var(rng, bounds=(0.0, 1.0))

    # Objective
    m.obj = pyo.Objective(
        expr=sum(avg_risk[pt(k)] * m.theta[k] for k in rng),
        sense=pyo.maximize,
    )

    # Placement constraints
    m.cons = pyo.ConstraintList()
    for k in rng:
        m.cons.add(m.xg[k] + m.xc[k] <= 1)

    m.cap_g = pyo.Constraint(expr=sum(m.xg[k] for k in rng) == n_grounds)
    m.cap_c = pyo.Constraint(expr=sum(m.xc[k] for k in rng) == n_charging)
    m.cap_d = pyo.Constraint(expr=sum(m.nc[k] for k in rng) == n_drones)

    # Linking: nc[i] <= n_drones * xc[i]
    for k in rng:
        m.cons.add(m.nc[k] <= n_drones * m.xc[k])

    # Coverage constraints
    m.cov = pyo.ConstraintList()
    for ip in range(N):
        for jp in range(M):
            idx = flat(ip, jp)
            expr = m.xg[idx]
            dx_lo = max(-ip, -kernel_size_x)
            dx_hi = min(N - 1 - ip, kernel_size_x)
            dy_lo = max(-jp, -kernel_size_y)
            dy_hi = min(M - 1 - jp, kernel_size_y)
            for dx in range(dx_lo, dx_hi + 1):
                for dy in range(dy_lo, dy_hi + 1):
                    key = (-dx, -dy)
                    kv = kernel.get(key)
                    if kv is not None and kv != 0.0:
                        nbr = flat(ip + dx, jp + dy)
                        expr += kv * m.xc[nbr]
            m.cov.add(m.theta[idx] <= expr)

    # ---- solve ----
    solver = get_solver()
    result = solver.solve(m, tee=False)

    sel_x = [pt(k) for k in rng if pyo.value(m.xg[k]) > 0.5]
    sel_y = [pt(k) for k in rng if pyo.value(m.xc[k]) > 0.5]

    print(f"Max_Coverage_Kernel solved in {time.time() - t0:.2f}s  "
          f"({len(sel_x)} ground, {len(sel_y)} charging)")
    return sel_x, sel_y
