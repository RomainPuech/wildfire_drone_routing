# Tables 2 & 3 Scenario Selection Rule

## Overview

Tables 2 and 3 in the paper report benchmark results on a subset of 474
fire-spread scenarios drawn from 12 layouts out of the full 7 746-scenario
dataset.

## Selection procedure

1. **Layout eligibility** — The paper states that experiments use layouts whose
   fire scenarios achieve ≥ 80 % *historical match* with the USFS Burn
   Probability risk map.  The `historical_match` flag in
   `scenario_summary.csv` was computed per-scenario; layouts where ≥ 80 % of
   scenarios carry `historical_match = True` qualified.

2. **Scenario selection** — Within each qualifying layout, *all* scenarios
   with available benchmark results were included (no additional subsampling).
   The authoritative run list comes from the combined benchmark output
   `combined_benchmark_resultsKMbm_parallel.csv`.

3. **Resulting split** — 12 layouts, 474 unique (layout, scenario) pairs.

## Authoritative source files

| File | Purpose |
|------|---------|
| `combined_benchmark_resultsKMbm_parallel.csv` | Ground-truth list of (layout, scenario) pairs that were benchmarked |
| `agg_by_layout_new.csv` | Per-layout sample sizes (column `n`) |
| `scenario_summary.csv` | Per-scenario metadata incl. `historical_match`, size/speed bins |

## The 12 layouts

```
0016  (n = 249)
0024  (n =  30)
0025  (n =  11)
0106  (n =  69)
0111  (n = 108)
0264  (n =   1)
0265  (n =   1)
0319  (n =   1)
0320  (n =   1)
0321  (n =   1)
0323  (n =   1)
0337  (n =   1)
```

Total: 474 scenarios.

## Notes

- Layouts 0264, 0265, 0319, 0320, 0321, 0323, 0337 each contribute a single
  scenario.  These layouts are present in `config_s2r.json` but absent from the
  49-layout `scenario_summary.csv`, so their size/speed bin flags are not
  available.
- Layout 0016 alone accounts for 249 / 474 ≈ 52.5 % of the split.
- 10 of the 474 scenarios are missing from `scenario_summary.csv` (they belong
  to layouts added after that file was generated).
