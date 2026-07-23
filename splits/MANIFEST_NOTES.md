# Dataset Manifest & Consistency Notes

This document records discrepancies between the released dataset artifact
(DroneBench.zip on HuggingFace) and the paper's reported numbers, as flagged
by Reviewer 8rSt.

## 1. Layout / folder count

| Source | Count | Details |
|--------|------:|---------|
| Paper (§ 4) | 49 layouts | "49 unique layout IDs" |
| `scenario_summary.csv` | 49 layouts | Matches the paper |
| `config_s2r.json` | 56 layouts | 7 extra layouts (0264, 0265, 0319, 0320, 0321, 0323, 0337) not in `scenario_summary.csv` |
| Zip top-level folders | 78 folders | Includes per-layout benchmark result folders and other auxiliary dirs |
| `WideDataset/` on disk | 1 extracted folder (`0321_03136`) | Only a single scenario folder is present locally |

**Explanation**: The zip contains one top-level directory per layout *plus*
per-layout benchmark-result directories and metadata files, inflating the
apparent folder count from 49 to 78.  The 7 extra layouts in `config_s2r.json`
(each contributing 1 scenario) were added after `scenario_summary.csv` was
frozen; their metadata (size/speed bins) is not recorded.

## 2. Scenario count

| Source | Count |
|--------|------:|
| Paper total | 7 746 |
| `scenario_summary.csv` rows | 7 746 |
| `config_s2r.json` scenario keys | 7 753 |
| Tables 2/3 subset | 474 (12 layouts) |

The 7-scenario difference (7 753 vs 7 746) corresponds to the 7 single-scenario
layouts absent from `scenario_summary.csv`.

## 3. `static_risk.npy` availability

The reviewer noted only 63 of 78 folders contain `static_risk.npy`.  Layout
folders that are purely benchmark-result containers (no fire-spread data) lack
this file.  All 49 data-carrying layouts should contain it; the remainder are
result/metadata directories.

## 4. Fire-spread frame coverage

The reviewer reports ~4 716 / 7 746 scenarios have fire-spread frame data
(JPEG mask sequences).  The remaining scenarios either:
- Had fires too small to generate meaningful frame sequences, or
- Were processed with a different output pipeline version.

This is a known gap and does not affect Tables 2/3 (all 474 scenarios have
benchmark results).

## 5. `static_risk.npy` encoding

Values are `int16` in the range ~11–385, *not* [0, 1] probabilities.  These
are raw USFS Burn Probability grid values (× 1 000).  To recover [0, 1]:

```python
import numpy as np
risk = np.load("static_risk.npy").astype(np.float32) / 1000.0
```

This normalization step was applied internally but not documented in the
released artifact.

## 6. Licensing

The dataset combines data from multiple sources:

| Source | License |
|--------|---------|
| Sim2Real-Fire | Apache-2.0 |
| USFS Burn Probability | CC-BY-4.0 (US Gov) |
| FPA-FOD | Public domain (US Gov) |

The HuggingFace repo was tagged `license: mit`, which is incorrect for the
*data*.  The code can remain MIT; the data should be CC-BY-4.0 (the most
restrictive upstream license).  A `NOTICE` file with attribution is required.
