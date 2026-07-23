"""Python MILP optimization models for wildfire drone routing.

Drop-in replacement for the Julia/JuMP models in ``julia/``.
Default solver: HiGHS (``highspy``).  Set ``WFDRONE_OPT_SOLVER=gurobi`` for Gurobi.
"""

import os


def get_solver():
    """Return a configured Pyomo solver (HiGHS by default, Gurobi if requested)."""
    import pyomo.environ as pyo

    solver_name = os.environ.get("WFDRONE_OPT_SOLVER", "highs").lower()
    if solver_name == "gurobi":
        solver = pyo.SolverFactory("gurobi")
        solver.options["OutputFlag"] = 0
    else:
        solver = pyo.SolverFactory("appsi_highs")
    return solver


from .ground_charging import sensor_maxcov_strategy, max_coverage_kernel  # noqa: E402,F401
from .drone_routing import (  # noqa: E402,F401
    RoutingModel,
    create_routing_model,
    solve_init_routing,
    solve_next_move_routing,
)
