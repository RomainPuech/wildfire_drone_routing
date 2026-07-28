# Rebuttal to Reviewer 8rSt (R1)

Coordination: tracking issue [#81](https://github.com/RomainPuech/wildfire_drone_routing/issues/81); child PRs merged into `r1/rebuttal` (#82–#87).

## Answers to reviewer questions

### Q1. Is the dynamic ground-truth risk map built from the test fires? Could Table 3 be re-reported under an independent map?

**Yes — and that is intentional.** The dynamic map is an *oracle / ground-truth* empirical burn field constructed by aggregating the layout’s own simulated fire-spread scenarios. We call it a ground-truth map **by design**: Table 3 measures strategy value under near-perfect risk knowledge (an upper bound), not the performance of an independent ML risk predictor.

The independent-map stress test is already Table 2 (USFS Burn Probability), where Max-Coverage’s advantage shrinks and Uniform Coverage is often competitive — exactly the comparison the reviewer asks for. We clarify this in §Risk maps, §Results, and Limitations (PR [#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)). Leave-one-out reconstruction of Table 3 is unnecessary for the oracle claim and is out of scope for this revision.

### Q2. `load_dataset`-compatible release, public 471-scenario split with per-cell \(n\), and open-source solver path?

**Yes:**

| Request | Delivery |
|---------|----------|
| HF packaging | Live anonymous Hugging Face release: [anonymous-submission-neurips26-2831](https://huggingface.co/datasets/anonymoussubmission2/anonymous-submission-neurips26-2831) with Parquet configs `default` / `tables23`, `NOTICE`, and license tag `cc-by-4.0`. Companion packaging also lives under `hf_release/` / `splits/` in the code repository. |
| 471 / 12-layout split | Published on HF (`tables23` config) and in-repo as `splits/tables23_layouts.txt`, `splits/tables23_scenarios.csv`, `SELECTION_RULE.md`, with `reproduce_tables23.py`. Per-layout \(n\) and fire bins documented. |
| Open-source solver | Default remains **Julia + Gurobi**. Optional open-source path: **Python + HiGHS** (`WFDRONE_OPT_BACKEND=python`) or **SCIP** (`WFDRONE_OPT_SOLVER=scip`, `pyscipopt`) ([#87](https://github.com/RomainPuech/wildfire_drone_routing/pull/87)). |

### Q3. Will the framework include a learning-based (RL) baseline?

The benchmark is designed to **consume** risk maps (including ML-produced maps) and evaluate **decision** policies. Shipped baselines are OR/heuristic by design. A learning-based routing baseline is planned as future work; BP vs oracle GT already stress-tests map quality for ML researchers. We do not overclaim that the current release benchmarks ML detectors or RL policies ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### Q4. Why is detection purely geometric?

Detection uses a **certain-detection radius** (300 m in experiments): once a fire cell enters this radius, detection is treated as certain and instantaneous. This models the conservative *inner core* of a real sensing footprint and sits well below nominal hardware ranges, so the assumption is conservative rather than optimistic. Headline detection rates are driven primarily by spatial reachability/coverage. Probabilistic sensing (missed detections, false alarms, smoke/visual recognition) is left to future work. Framing adapted from our extended Nature manuscript discussion (certain-detection radius); see Limitations ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

---

## Point-by-point on weaknesses

### W1 — HF packaging
Addressed on the live HF dataset ([anonymous-submission-neurips26-2831](https://huggingface.co/datasets/anonymoussubmission2/anonymous-submission-neurips26-2831)): structured Parquet + configs, `NOTICE`, license `cc-by-4.0`. Composite-license guidance also in `NOTICE` / `docs/HF_LICENSE.md`.

### W2 — Reproduce Tables 2/3
Published exact 12-layout / 471-scenario split on HF and in `splits/`, plus `reproduce_tables23.py` and `MANIFEST_NOTES.md`.

### W3 — Risk units / JPG masks
Documented in [`DATASET.md`](DATASET.md) and on the HF dataset card (BP as FSim × 10 000 → divide by 10 000; JPG `/255` + threshold ≥ 0.5; NPY preferred).

### W4 — Inconsistencies + configuration-specific drone claim
- Five strategies (2 + 3), not six ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83)).
- Dual stack: Python benchmarking library with Julia+Gurobi default and optional Python+HiGHS/SCIP ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83), [#87](https://github.com/RomainPuech/wildfire_drone_routing/pull/87)).
- Duplicate bibliography entries removed ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83)).
- Drone vs ground-sensor claim qualified to the reported configuration (8 / 2 / 2, 300 m) ([#83](https://github.com/RomainPuech/wildfire_drone_routing/pull/83)).

### W5 — No ML baseline
Clarified positioning + future work; no RL implemented ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### W6 — Geometric detection
Certain-detection radius justification in Problem Formulation + Limitations ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### W7 — GT map / leakage concern
**Not leakage: oracle by design.** Emphasized throughout paper and above. Independent evidence: Table 2 (BP) ([#84](https://github.com/RomainPuech/wildfire_drone_routing/pull/84)).

### License / checklist
Sim2Real-Fire listed as **Apache-2.0** (not MIT); root `NOTICE` + HF guidance ([#86](https://github.com/RomainPuech/wildfire_drone_routing/pull/86)). HF tag set to `cc-by-4.0`.

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
