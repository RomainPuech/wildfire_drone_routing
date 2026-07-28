We thank Reviewer 8rSt for the careful assessment of reproducibility, dataset packaging, and claim clarity. We address each weakness and question below. The corresponding code, split files, documentation, and paper edits are on branch `r1/rebuttal` of the public repository (https://github.com/RomainPuech/wildfire_drone_routing). The structured dataset release is live at https://huggingface.co/datasets/anonymoussubmission2/anonymous-submission-neurips26-2831.

---

### W1: HuggingFace packaging / `load_dataset`

**Addressed.** We have published a `load_dataset`-compatible release on the anonymous Hugging Face dataset ([anonymous-submission-neurips26-2831](https://huggingface.co/datasets/anonymoussubmission2/anonymous-submission-neurips26-2831)) with Parquet indices, configs `default` and `tables23`, `NOTICE`, and license tag `cc-by-4.0`. Companion packaging also lives under `hf_release/` / `splits/` in the code repository.

### W2: Reproducing Tables 2/3 (471 scenarios / 12 layouts)

**Addressed.** We release the exact evaluation subset used for the main tables:
- HF `tables23` config and `splits/tables23_layouts.txt` (12 layouts)
- `splits/tables23_scenarios.csv` (471 scenarios, with per-layout sample sizes and fire attributes where available)
- `splits/SELECTION_RULE.md` and `splits/MANIFEST_NOTES.md`
- `reproduce_tables23.py` to re-run / verify the subset

Manifest inconsistencies in the prior zip (folder counts vs. `scenario_summary.csv`, missing risk files / frames) are documented in `MANIFEST_NOTES.md`.

### W3: Risk-map units and JPG fire masks

**Addressed.** We document encodings in `DATASET.md` and on the HF dataset card:
- on-disk encodings for `static_risk*.npy` / BP / WHP / `burn_map.npy` (USFS FSim BP × 10 000 → divide by 10 000 for \(r_{it}\in[0,1]\));
- JPG decoding (`/255`, threshold \(\ge 0.5\)), JPEG caveats, and that preprocessed NPY masks are canonical when both exist.

### W4: Inconsistencies and configuration-specific drone claim

**Addressed.**
1. Strategy count corrected to **five** (2 placement + 3 routing).
2. Stack clarified: open-source **Python** benchmarking library with **dual optimization backends**—default **Julia + Gurobi**, optional **Python + HiGHS / SCIP** (`WFDRONE_OPT_BACKEND=python`, `WFDRONE_OPT_SOLVER=highs|scip`). Placement and MaxCov/Uniform routing MILPs are reimplemented in Python.
3. Duplicate bibliography entries removed.
4. The claim that drones outperform ground sensors is now explicitly **configuration-specific** (8 ground sensors, 2 drones, 2 charging stations, 300 m sensing radius) in the abstract, contributions, results, and conclusion.

### W5: No learning-based baseline

**Clarified (text).** WFDroneBench is designed to *consume* risk maps—including ML-produced maps—and to evaluate deployment/routing policies under a common interface. The shipped baselines are optimization-based and heuristic by design. We revise the paper so it does not imply that learning-based routing or detection models are already benchmarked; adding a learned/RL baseline is listed as future work. The BP vs. oracle ground-truth comparison already stresses risk-map quality for ML researchers.

### W6: Geometric detection

**Clarified (text).** Detection uses a **certain-detection radius** (300 m in our experiments): once a fire enters this radius of any device, detection is treated as certain and instantaneous. This models the conservative inner core of a real sensing footprint and is set well below nominal hardware ranges, so the assumption is conservative rather than optimistic. Empirically, detection rates are driven primarily by spatial reachability/coverage. Probabilistic sensing (distance-dependent detection, false alarms, smoke/visual recognition) is left to future work and is now stated in Limitations.

### W7: Ground-truth risk map and possible leakage

**Clarified — this is by design, not leakage.** The dynamic “ground-truth” burn map is an **oracle** built by aggregating the layout’s own simulated fire-spread scenarios. That construction is precisely why we call it a ground-truth map: Table 3 reports strategy performance under near-perfect risk knowledge (an upper bound). It is **not** claimed to be an independent ML risk predictor.

The independent-map evaluation is Table 2 (USDA Burn Probability), where Max-Coverage’s advantage largely disappears—exactly the stress test requested. We make this bracketing explicit in §Risk maps, §Results, and Limitations. Held-out reconstruction of the oracle map is not needed for the intended claim.

### License / checklist

**Addressed.** Sim2Real-Fire is correctly listed as **Apache-2.0** (not MIT) in the checklist and appendix. We added `NOTICE` / composite-license guidance for the derived dataset (USFS BP/WHP CC-BY, FPA FOD public/CC0, code MIT). The Hugging Face license tag is `cc-by-4.0`.

---

### Responses to the reviewer’s questions

**Q1. Is the dynamic ground-truth risk map built from the test fires? Can Table 3 be re-reported under an independent map?**  
Yes, the oracle map uses the same scenario pool **by design** (see W7). Table 2 already provides the independent BP evaluation. We therefore treat Table 3 as an upper bound under perfect risk information rather than re-running leave-one-out oracle maps in this revision.

**Q2. Can you provide a `load_dataset`-compatible release, the 471-scenario split with per-cell \(n\), and an open-source solver path?**  
Yes: the live HF release above, public `splits/` with per-layout \(n\), and an optional open-source **Python + HiGHS / SCIP** path (`WFDRONE_OPT_BACKEND=python`; default remains Julia + Gurobi). See W1, W2, and W4.

**Q3. Will you include a learning-based baseline?**  
Not in this revision. The benchmark targets risk-map *consumption* and decision policies; OR/heuristic baselines are intentional. A learned/RL baseline is planned as future work (W5).

**Q4. Why is detection purely geometric?**  
We use a conservative certain-detection radius abstraction focused on coverage/reachability; probabilistic perception is out of scope for the current benchmark interface and is now discussed in Limitations (W6).

We believe these changes resolve the reproducibility blockers and the claim/positioning issues raised in the review.
