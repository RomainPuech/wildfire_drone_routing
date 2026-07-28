# Tables 2 & 3 Scenario Selection Rule

## Overview

Tables 2 and 3 in the paper report benchmark results on a subset of **471**
fire-spread scenarios drawn from **12** layouts out of the full 7 746-scenario
dataset.

## Selection procedure

1. **Layout eligibility** — Experiments use layouts whose fire scenarios
   achieve ≥ 80 % *historical match* with the USFS Burn Probability risk map.
   The `historical_match` flag in `scenario_summary.csv` was computed
   per-scenario; layouts where ≥ 80 % of scenarios carry
   `historical_match = True` qualified.

2. **Scenario selection** — Within each qualifying layout, scenarios included
   in the published Tables 2/3 evaluation split are listed in
   `splits/tables23_scenarios.csv` (and the `tables23` parquet config).

3. **Resulting split** — 12 layouts, 471 unique `(layout_id, scenario_id)` pairs.

## The 12 layouts

```
0016  (n = 248)
0024  (n =  29)
0025  (n =  10)
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

Total: 471 scenarios.

## Notes

- Layouts 0264, 0265, 0319, 0320, 0321, 0323, 0337 each contribute a single
  scenario and are present in `config_s2r.json` (extras beyond the 49-layout
  summary index).
- Layout 0016 alone accounts for 248 / 471 ≈ 52.7 % of the split.
