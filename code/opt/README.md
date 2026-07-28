# Python optimization backends (optional)

Pyomo ports of the paper MILPs in `julia/ground_charging_opt.jl` and `julia/drone_routing_opt.jl`.

Enable with `WFDRONE_OPT_BACKEND=python` (default backend is `julia` + Gurobi).

Within the Python backend, select the MILP solver via `WFDRONE_OPT_SOLVER`:

- Default: HiGHS (`WFDRONE_OPT_SOLVER=highs`)
- Optional open-source: SCIP (`WFDRONE_OPT_SOLVER=scip`, via `pyscipopt` / SCIP)
- Optional commercial: Gurobi (`WFDRONE_OPT_SOLVER=gurobi`)
