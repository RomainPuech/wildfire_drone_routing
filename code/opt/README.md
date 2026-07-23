# Python optimization backends

Pyomo ports of the paper MILPs in `julia/ground_charging_opt.jl` and `julia/drone_routing_opt.jl`.

- Default solver: HiGHS (`WFDRONE_OPT_SOLVER=highs`)
- Optional: Gurobi (`WFDRONE_OPT_SOLVER=gurobi`)
- Selected via `WFDRONE_OPT_BACKEND=python|julia` (default `python`)

**Not ported:** `julia/TOP.jl` / PSO team-orienteering helpers remain Julia-only.
