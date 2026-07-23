"""Drone routing MILP with rolling-horizon re-solve.

Port of ``julia/drone_routing_opt.jl``.  All coordinates are **0-based**.

Provides
--------
RoutingModel           – container for the Pyomo model + metadata
create_routing_model   – build the structural MILP (called once)
solve_init_routing     – add init constraints, solve, extract plan
solve_next_move_routing – add warm-start constraints, solve, extract plan
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo

from . import get_solver
from .helpers import (
    closest_distances,
    get_drone_gridpoints,
    linf_neighbors_and_point,
    load_burn_map,
)


# ======================================================================
# Data container
# ======================================================================

class RoutingModel:
    """Stores the Pyomo ``ConcreteModel`` and associated grid metadata."""

    __slots__ = (
        "model",
        "grid_points",
        "charging_stations",
        "ground_stations",
        "detecting_points",
        "grid_to_idx",
        "charging_map",
        "neighbors_map",
        "closest_dist",
        "T",
        "n_drones",
        "max_battery_time",
        "risk_pertime_file",
        "nG",
        "nC",
        "nD",
    )

    def __init__(self) -> None:
        self.model: Optional[pyo.ConcreteModel] = None
        self.grid_points: List[Tuple[int, int]] = []
        self.charging_stations: List[Tuple[int, int]] = []
        self.ground_stations: List[Tuple[int, int]] = []
        self.detecting_points: List[Tuple[int, int]] = []
        self.grid_to_idx: Dict[Tuple[int, int], int] = {}
        self.charging_map: Dict[int, int] = {}
        self.neighbors_map: Dict[int, List[int]] = {}
        self.closest_dist: List[int] = []
        self.T: int = 0
        self.n_drones: int = 0
        self.max_battery_time: int = 0
        self.risk_pertime_file: str = ""
        self.nG: int = 0
        self.nC: int = 0
        self.nD: int = 0


# ======================================================================
# Model construction
# ======================================================================

def create_routing_model(
    risk_pertime_file: str,
    n_drones: int,
    charging_stations: List[Tuple[int, int]],
    ground_stations: List[Tuple[int, int]],
    optimization_horizon: int,
    max_battery_time: int,
) -> RoutingModel:
    """Build the structural routing MILP.  Coordinates are **0-based**."""
    t0 = time.time()
    print("Creating Python routing model")

    risk_pertime = load_burn_map(risk_pertime_file)
    H, N, M = risk_pertime.shape
    T = optimization_horizon

    if H == 1:
        risk_pertime = np.tile(risk_pertime, (100, 1, 1))
        H = 100

    all_points = [(x, y) for x in range(N) for y in range(M)]
    cs_set = set(map(tuple, charging_stations))
    gs_set = set(map(tuple, ground_stations))

    # Reachable grid points
    gp_set = get_drone_gridpoints(charging_stations, max_battery_time // 2, all_points)
    grid_points = sorted(gp_set)
    gp_set_frozen = set(grid_points)

    detecting_points = sorted(gp_set - cs_set - gs_set)

    grid_to_idx = {pt: i for i, pt in enumerate(grid_points)}
    charging_map = {j: grid_to_idx[cs] for j, cs in enumerate(charging_stations)}

    neighbors_map: Dict[int, List[int]] = {}
    for i, pt in enumerate(grid_points):
        nbrs = linf_neighbors_and_point(pt, gp_set_frozen)
        neighbors_map[i] = [grid_to_idx[p] for p in nbrs if p in grid_to_idx]

    c_dist = closest_distances(charging_stations, grid_points)

    nG = len(grid_points)
    nC = len(charging_stations)
    nD = len(detecting_points)

    # ------------------------------------------------------------------
    # Pyomo model
    # ------------------------------------------------------------------
    mdl = pyo.ConcreteModel()

    mdl.a = pyo.Var(range(nG), range(T), range(n_drones), domain=pyo.Binary)
    mdl.c = pyo.Var(range(nC), range(T), range(n_drones), domain=pyo.Binary)
    mdl.b = pyo.Var(
        range(T), range(n_drones), domain=pyo.Integers, bounds=(0, max_battery_time)
    )
    mdl.theta = pyo.Var(range(T), range(nD), domain=pyo.Binary)

    # ---- structural constraints ----
    mdl.sc = pyo.ConstraintList()

    # 1) Each drone either charges or flies at every time step
    for t in range(T):
        for s in range(n_drones):
            mdl.sc.add(
                sum(mdl.a[i, t, s] for i in range(nG))
                + sum(mdl.c[j, t, s] for j in range(nC))
                == 1
            )

    # 2a) Movement at charging-station locations
    for j, cs_pt in enumerate(charging_stations):
        gi = grid_to_idx[cs_pt]
        for t in range(T - 1):
            for s in range(n_drones):
                mdl.sc.add(
                    mdl.c[j, t + 1, s] + mdl.a[gi, t + 1, s]
                    <= sum(mdl.a[k, t, s] for k in neighbors_map[gi])
                    + mdl.c[j, t, s]
                )

    # 2b) Movement at non-charging locations
    for gi in range(nG):
        if grid_points[gi] in cs_set:
            continue
        for t in range(T - 1):
            for s in range(n_drones):
                mdl.sc.add(
                    mdl.a[gi, t + 1, s]
                    <= sum(mdl.a[k, t, s] for k in neighbors_map[gi])
                )

    # 3) Battery dynamics
    for t in range(T):
        for s in range(n_drones):
            mdl.sc.add(
                mdl.b[t, s]
                >= max_battery_time * sum(mdl.c[j, t, s] for j in range(nC))
            )

    for t in range(T - 1):
        for s in range(n_drones):
            mdl.sc.add(
                mdl.b[t + 1, s]
                <= mdl.b[t, s]
                - 1
                + (max_battery_time + 1) * sum(mdl.c[j, t + 1, s] for j in range(nC))
            )

    # 4) No-suicide constraint
    for s in range(n_drones):
        for gi in range(nG):
            mdl.sc.add(mdl.b[T - 1, s] >= mdl.a[gi, T - 1, s] * c_dist[gi])

    # 5) Coverage constraints
    for t in range(T):
        for k in range(nD):
            dk_gi = grid_to_idx[detecting_points[k]]
            for s in range(n_drones):
                mdl.sc.add(mdl.theta[t, k] >= mdl.a[dk_gi, t, s])

    for k in range(nD):
        dk_gi = grid_to_idx[detecting_points[k]]
        mdl.sc.add(
            mdl.theta[0, k] <= sum(mdl.a[dk_gi, 0, s] for s in range(n_drones))
        )

    for t in range(1, T):
        for k in range(nD):
            dk_gi = grid_to_idx[detecting_points[k]]
            mdl.sc.add(
                mdl.theta[t, k]
                <= sum(mdl.a[dk_gi, t, s] for s in range(n_drones))
                + mdl.theta[t - 1, k]
            )
            mdl.sc.add(mdl.theta[t, k] >= mdl.theta[t - 1, k])

    # 6) Initial objective (offset = 0)
    _set_objective(mdl, risk_pertime, detecting_points, grid_to_idx, T, nD, offset=0)

    # Placeholder for dynamic constraints
    mdl.dyn = pyo.ConstraintList()

    # ------------------------------------------------------------------
    rm = RoutingModel()
    rm.model = mdl
    rm.grid_points = grid_points
    rm.charging_stations = list(charging_stations)
    rm.ground_stations = list(ground_stations)
    rm.detecting_points = detecting_points
    rm.grid_to_idx = grid_to_idx
    rm.charging_map = charging_map
    rm.neighbors_map = neighbors_map
    rm.closest_dist = c_dist
    rm.T = T
    rm.n_drones = n_drones
    rm.max_battery_time = max_battery_time
    rm.risk_pertime_file = risk_pertime_file
    rm.nG = nG
    rm.nC = nC
    rm.nD = nD

    print(f"Routing model created in {time.time() - t0:.2f}s  "
          f"(nG={nG}, nC={nC}, nD={nD}, T={T})")
    return rm


# ======================================================================
# Internal helpers
# ======================================================================

def _set_objective(
    mdl: pyo.ConcreteModel,
    risk_pertime: np.ndarray,
    detecting_points: List[Tuple[int, int]],
    grid_to_idx: Dict[Tuple[int, int], int],
    T: int,
    nD: int,
    offset: int = 0,
) -> None:
    """(Re-)set the objective on *mdl*."""
    if hasattr(mdl, "obj"):
        mdl.del_component("obj")

    H = risk_pertime.shape[0]
    T_eff = min(T, H - offset)
    if T_eff < 1:
        T_eff = 1

    obj_expr = sum(
        float(risk_pertime[offset, dp[0], dp[1]]) * mdl.theta[0, k]
        for k, dp in enumerate(detecting_points)
    )
    for t in range(1, T_eff):
        obj_expr += sum(
            float(risk_pertime[t + offset, dp[0], dp[1]])
            * (mdl.theta[t, k] - mdl.theta[t - 1, k])
            for k, dp in enumerate(detecting_points)
        )

    mdl.obj = pyo.Objective(expr=obj_expr, sense=pyo.maximize)


def _clear_dyn(mdl: pyo.ConcreteModel) -> None:
    """Remove all dynamic constraints."""
    if hasattr(mdl, "dyn"):
        mdl.del_component("dyn")
    mdl.dyn = pyo.ConstraintList()


def _extract_plan(
    rm: RoutingModel,
    steps: int,
) -> List[List[Tuple[str, Tuple[int, int]]]]:
    """Read solution values and produce a movement plan (0-based coords)."""
    mdl = rm.model
    plan: List[List[Tuple[str, Tuple[int, int]]]] = []
    for t in range(steps):
        step_plan: List[Tuple[str, Tuple[int, int]]] = [
            ("stay", (0, 0)) for _ in range(rm.n_drones)
        ]
        for s in range(rm.n_drones):
            for i in range(rm.nG):
                if pyo.value(mdl.a[i, t, s]) >= 0.9:
                    step_plan[s] = ("fly", rm.grid_points[i])
            for j in range(rm.nC):
                if pyo.value(mdl.c[j, t, s]) >= 0.9:
                    step_plan[s] = ("charge", rm.charging_stations[j])
        plan.append(step_plan)
    return plan


# ======================================================================
# Solve helpers
# ======================================================================

def solve_init_routing(
    rm: RoutingModel,
    reevaluation_step: int,
) -> Optional[List[List[Tuple[str, Tuple[int, int]]]]]:
    """Add initial constraints, solve, return movement plan (0-based)."""
    mdl = rm.model
    _clear_dyn(mdl)

    # All drones start from a charging station (charge or fly there)
    for s in range(rm.n_drones):
        mdl.dyn.add(
            sum(mdl.c[j, 0, s] for j in range(rm.nC))
            + sum(mdl.a[rm.charging_map[j], 0, s] for j in range(rm.nC))
            == 1
        )

    # Full battery at t=0
    for s in range(rm.n_drones):
        mdl.dyn.add(
            mdl.b[0, s]
            == rm.max_battery_time - sum(mdl.a[i, 0, s] for i in range(rm.nG))
        )

    # Capacity per charging station
    capacity = 30
    for j in range(rm.nC):
        gi = rm.charging_map[j]
        mdl.dyn.add(
            sum(mdl.c[j, 0, s] for s in range(rm.n_drones))
            + sum(mdl.a[gi, 0, s] for s in range(rm.n_drones))
            <= capacity
        )

    # Make sure objective is at offset 0
    risk_pertime = load_burn_map(rm.risk_pertime_file)
    if risk_pertime.shape[0] == 1:
        risk_pertime = np.tile(risk_pertime, (100, 1, 1))
    _set_objective(mdl, risk_pertime, rm.detecting_points, rm.grid_to_idx, rm.T, rm.nD, offset=0)

    # Solve
    solver = get_solver()
    result = solver.solve(mdl, tee=False)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"Init routing: no optimal solution ({result.solver.termination_condition})")
        return None

    return _extract_plan(rm, reevaluation_step)


def solve_next_move_routing(
    rm: RoutingModel,
    reevaluation_step: int,
    drone_locations: List[Tuple[int, int]],
    drone_states: List[str],
    battery_levels: List[int],
    offset: int = 0,
) -> Optional[List[List[Tuple[str, Tuple[int, int]]]]]:
    """Fix drone starting state, re-solve, return movement plan (0-based)."""
    mdl = rm.model
    _clear_dyn(mdl)

    # Fix starting positions
    for s, state in enumerate(drone_states):
        loc = tuple(drone_locations[s])
        if state == "charge":
            for j, cs in enumerate(rm.charging_stations):
                if tuple(cs) == loc:
                    mdl.dyn.add(mdl.c[j, 0, s] == 1)
                    break
        elif state == "fly":
            loc_idx = rm.grid_to_idx[loc]
            mdl.dyn.add(mdl.a[loc_idx, 0, s] == 1)

    # Fix starting battery
    for s in range(rm.n_drones):
        if drone_states[s] != "charge":
            mdl.dyn.add(mdl.b[0, s] == int(battery_levels[s]))
        else:
            mdl.dyn.add(mdl.b[0, s] == rm.max_battery_time)

    # Update objective with offset
    risk_pertime = load_burn_map(rm.risk_pertime_file)
    if risk_pertime.shape[0] == 1:
        risk_pertime = np.tile(risk_pertime, (100, 1, 1))
    _set_objective(mdl, risk_pertime, rm.detecting_points, rm.grid_to_idx, rm.T, rm.nD, offset=offset)

    # Solve
    solver = get_solver()
    result = solver.solve(mdl, tee=False)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"Next-move routing: no optimal solution ({result.solver.termination_condition})")
        return None

    return _extract_plan(rm, reevaluation_step)
