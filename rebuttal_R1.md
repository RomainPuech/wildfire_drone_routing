# Rebuttal to Reviewer 8rSt (R1)

Coordination: tracking issue [#81](https://github.com/RomainPuech/wildfire_drone_routing/issues/81); child PRs merged into `r1/rebuttal` (#82–#87).

## Answers to reviewer questions

### Q1. Is the dynamic ground-truth risk map built from the test fires? Could Table 3 be re-reported under an independent map?

**Yes — and that is intentional.** The dynamic map is an *oracle / ground-truth* empirical burn field constructed by aggregating the layout’s own simulated fire-spread scenarios. We call it a ground-truth map **by design**: Table 3 measures strategy value under near-perfect risk knowledge (an upper bound), not the performance of an independent ML risk predictor.

The independent-map stress test is already Table 2 (USFS Burn Probability), where Max-Coverage’s advantage shrinks and Uniform Coverage is often competitive — exactly the comparison the reviewer asks for. We clarify this in §Risk maps, §Results, and Limitations (PR [#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)). Leave-one-out reconstruction of Table 3 is unnecessary for the oracle claim and is out of scope for this revision.

### Q2. `load_dataset`-compatible release, public 474-scenario split with per-cell \(n\), and open-source solver path?

**Yes (local artifacts + dual solver path):**

| Request | Delivery |
|---------|----------|
| HF packaging | Local `hf_release/` with parquet indices, configs `default` / `tables23`, dataset card, `UPLOAD_INSTRUCTIONS.md` for MasterYoda293 ([#85](https://github.com/RomainPuech/wildfire_drone_routing/pull/85)). We do not push to Hugging Face from this PR. |
| 474 / 12-layout split | `splits/tables23_layouts.txt`, `splits/tables23_scenarios.csv`, `SELECTION_RULE.md`, `reproduce_tables23.py`. Recovered from the authoritative experiment CSV (`combined_benchmark_resultsKMbm_parallel.csv`, 474 rows / 12 layouts). Per-layout \(n\) and fire bins documented; see `MANIFEST_NOTES.md` for residual metadata gaps on 10 scenarios. |
| Open-source solver | Default path is **Python + HiGHS** (`WFDRONE_OPT_BACKEND=python`); optional **SCIP** via `WFDRONE_OPT_SOLVER=scip` (`pyscipopt`). Julia + Gurobi retained (`WFDRONE_OPT_BACKEND=julia`) ([#87](https://github.com/RomainPuech/wildfire_drone_routing/pull/87)). |

### Q3. Will the framework include a learning-based (RL) baseline?

The benchmark is designed to **consume** risk maps (including ML-produced maps) and evaluate **decision** policies. Shipped baselines are OR/heuristic by design. A learning-based routing baseline is planned as future work; BP vs oracle GT already stress-tests map quality for ML researchers. We do not overclaim that the current release benchmarks ML detectors or RL policies ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### Q4. Why is detection purely geometric?

Detection uses a **certain-detection radius** (300 m in experiments): once a fire cell enters this radius, detection is treated as certain and instantaneous. This models the conservative *inner core* of a real sensing footprint and sits well below nominal hardware ranges, so the assumption is conservative rather than optimistic. Headline detection rates are driven primarily by spatial reachability/coverage. Probabilistic sensing (missed detections, false alarms, smoke/visual recognition) is left to future work. Framing adapted from our extended Nature manuscript discussion (certain-detection radius); see Limitations ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

---

## Point-by-point on weaknesses

### W1 — HF packaging
Addressed via `hf_release/` (structured parquet + configs). MasterYoda293 uploads per `UPLOAD_INSTRUCTIONS.md`. Composite license guidance in `NOTICE` / `docs/HF_LICENSE.md` ([#85](https://github.com/RomainPuech/wildfire_drone_routing/pull/85), [#86](https://github.com/RomainPuech/wildfire_drone_routing/pull/86)).

### W2 — Reproduce Tables 2/3
Published exact 12-layout / 474-scenario split + reproduction script + manifest notes ([#85](https://github.com/RomainPuech/wildfire_drone_routing/pull/85)).

### W3 — Risk units / JPG masks
Documented in [`DATASET.md`](DATASET.md) (dtype/units, normalization path, JPG `/255` + threshold ≥ 0.5, NPY preferred) ([#82](https://github.com/RomainPuech/wildfire_drone_routing/pull/82)).

### W4 — Inconsistencies + configuration-specific drone claim
- Five strategies (2 + 3), not six ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83)).
- Dual stack: Python benchmarking library with Python+HiGHS default and Julia+Gurobi ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83), [#87](https://github.com/RomainPuech/wildfire_drone_routing/pull/87)).
- Duplicate bibliography entries removed ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83)).
- Drone vs ground-sensor claim qualified to the reported configuration (8 / 2 / 2, 300 m) ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83)).

### W5 — No ML baseline
Clarified positioning + future work; no RL implemented ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### W6 — Geometric detection
Certain-detection radius justification in Problem Formulation + Limitations ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### W7 — GT map / leakage concern
**Not leakage: oracle by design.** Emphasized throughout paper and above. Independent evidence: Table 2 (BP) ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### License / checklist
Sim2Real-Fire listed as **Apache-2.0** (not MIT); root `NOTICE` + HF guidance ([#86](https://github.com/RomainPuech/wildfire_drone_routing/pull/86)).

---

## Merged PRs

| PR | Issue | Topic |
|----|-------|--------|
| [#82](https://github.com/RomainPuech/wildfire_drone_routing/pull/82) | #75 | DATASET.md encodings |
| [#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83) | #76 | Paper claim consistency |
| [#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84) | #78 | Limitations W5–W7 |
| [#85](https://github.com/RomainPuech/wildfire_drone_routing/pull/85) | #74 | HF release + Tables 2/3 split |
| [#86](https://github.com/RomainPuech/wildfire_drone_routing/pull/86) | #79 | NOTICE + checklist license |
| [#87](https://github.com/RomainPuech/wildfire_drone_routing/pull/87) | #77 | Python MILPs + HiGHS |

## Author follow-ups outside this branch

1. MasterYoda293: upload `hf_release/` per `UPLOAD_INSTRUCTIONS.md` and fix the HF license tag (composite, not pure MIT).
2. Optional: fill fire size/speed bins for the 10 scenarios missing from `scenario_summary.csv`.
