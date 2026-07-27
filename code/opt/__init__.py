"""Python MILP optimization models for wildfire drone routing.

Drop-in replacement for the Julia/JuMP models in ``julia/``.
Default solver: HiGHS (``highspy``).  Alternatives via ``WFDRONE_OPT_SOLVER``:
``highs`` (default), ``scip`` (``pyscipopt`` / SCIP), ``gurobi``.
"""

import os


def _solver_available(solver) -> bool:
    """Return True if *solver* can be used (handles Pyomo ApplicationError)."""
    try:
        avail = solver.available(exception_flag=False)
    except TypeError:
        try:
            avail = solver.available()
        except Exception:
            return False
    except Exception:
        return False
    return bool(avail)


def get_solver():
    """Return a configured Pyomo solver.

    Selection via ``WFDRONE_OPT_SOLVER``:

    - ``highs`` (default): ``appsi_highs`` / ``highspy``
    - ``scip``: prefer ``scip_direct`` (``pyscipopt``), else classic ``scip`` AMPL
    - ``gurobi``: Gurobi (license required)
    """
    import pyomo.environ as pyo

    solver_name = os.environ.get("WFDRONE_OPT_SOLVER", "highs").lower()
    if solver_name == "gurobi":
        solver = pyo.SolverFactory("gurobi")
        solver.options["OutputFlag"] = 0
        return solver

    if solver_name == "scip":
        # Prefer the Python/direct interface (mirrors highspy for HiGHS).
        direct = pyo.SolverFactory("scip_direct")
        if _solver_available(direct):
            # Quiet by default; tee=False on solve() still applies.
            try:
                direct.options["display/verblevel"] = 0
            except Exception:
                pass
            return direct

        ampl = pyo.SolverFactory("scip")
        if _solver_available(ampl):
            try:
                ampl.options["display/verblevel"] = 0
            except Exception:
                pass
            return ampl

        raise RuntimeError(
            "WFDRONE_OPT_SOLVER=scip requested, but neither PySCIPOpt "
            "(pip install pyscipopt) nor a SCIP executable was found."
        )

    if solver_name not in ("highs", "appsi_highs", ""):
        raise ValueError(
            f"Unknown WFDRONE_OPT_SOLVER={solver_name!r}. "
            "Supported: highs (default), scip, gurobi."
        )

    return pyo.SolverFactory("appsi_highs")


from .ground_charging import sensor_maxcov_strategy, max_coverage_kernel  # noqa: E402,F401
from .drone_routing import (  # noqa: E402,F401
    RoutingModel,
    create_routing_model,
    solve_init_routing,
    solve_next_move_routing,
)
