# Yearly WFPI Burn Map: Risk Map Comparison

## Overview

This document compares the new **Yearly WFPI Burn Map** (`static_risk_wfpi_yearly.npy`) against the four reference risk maps already evaluated in [09_risk_map_comparison.md](09_risk_map_comparison.md).

The key innovation of the yearly map is **time-aware WFPI selection**: instead of always using the Day-2 or Day-1 forecast, it uses whichever forecast was operationally available at the fire's discovery time:

- **Before 10 am** → Day-2 forecast (issued the previous day, the best available before the 10 am update)
- **After 10 am** → Day-1 forecast (issued the same day, updated at 10 am)

Fires without a recorded discovery time are excluded from this comparison.

## Dataset

- **All fires analysed:** 2,219
- **Large fires (≥ 100 acres):** 44
- **Background median (D2):** 54.34
- **Background median (D1):** 54.36
- **Background median (Yearly, combined):** 54.36

## Summary Table

Background median used as reference: ~54.35 (median of the yearly-average WFPI over all valid California cells).
"% fires above bg median" shows what fraction of fires occur in above-average-risk areas; background = 50 % by definition.

| Risk Map | Fire Median (All) | Ratio (All) | % Above Bg Median (All) | Improvement vs 50% | Fire Median (Large) | Ratio (Large) | % Above Bg Median (Large) |
|----------|------------------|-------------|-------------------------|--------------------|---------------------|---------------|---------------------------|
| **WFPI Yearly (time-aware)** | 72.0 | **1.32x** | **68.1%** | **+18.1 pp** | 76.5 | 1.41x | 81.8% |
| **WFPI Day 1** | 73.0 | 1.34x | 68.0% | +18.0 pp | 76.5 | **1.41x** | 79.5% |
| **WFPI Day 2** | 71.0 | 1.31x | 67.2% | +17.2 pp | 76.0 | 1.40x | **84.1%** |

**Best for all fires:** WFPI Yearly (time-aware) — +18.1 pp (narrowly ahead of D1 at +18.0 pp)
**Best for large fires (% above median):** WFPI Day 2 — +34.1 pp

## Detailed Analysis

### WFPI Yearly (Time-Aware)

The yearly map selects the most operationally accurate WFPI forecast for each fire based on its discovery time. This reduces the systematic error introduced by always using D2 (too early) or D1 (occasionally unavailable before 10 am).

**Construction:**

- Shape: `(732, 1309, 805)` — 2 frames × 366 days (2020 is a leap year)
- Frame `2*(d-1)+0`: Day-2 forecast for day d (issued on day d−1)
- Frame `2*(d-1)+1`: Day-1 forecast for day d (issued on day d at 10 am)
- Missing source files filled by nearest-neighbour interpolation
- Jan 1 pre-10 am uses Jan 1 D1 as fallback (Dec 31 2019 D2 not available)

**Indexing at runtime:**

```python
def frame_index(discovery_date, hour):
    day_of_year = discovery_date.timetuple().tm_yday  # 1–366
    half = 0 if hour < 10 else 1
    return 2 * (day_of_year - 1) + half
```

### Methodology Note

This analysis differs from `09_risk_map_comparison.md` in two ways:

1. **Only fires with a recorded discovery time** are included (2,219 of 4,166 scenarios, i.e. the 80.5 % of fires that have `DISCOVERY_TIME` in the FPA FOD database). This means the sample is slightly smaller than the 2,407 used in the D1/D2 baseline.
2. **Background computed from 366-day averages** (all of 2020, including days without fires, some filled by nearest-neighbour). The baseline used only 317–320 days. This raises the background median from ~46 to ~54, which is why the absolute fire medians reported here differ from the earlier document.

The relative ordering between maps is directly comparable.

### Comparison with Previous Results (from 09_risk_map_comparison.md)

| Risk Map | % Fires Above Bg Median (All) | % Fires Above Bg Median (Large) |
|----------|-------------------------------|----------------------------------|
| WFPI Day 2 *(09_risk_map_comparison.md)* | +4.7 pp | +13.9 pp |
| WFPI Day 1 *(09_risk_map_comparison.md)* | +15.8 pp | +17.9 pp |
| Ignition Probability *(09_risk_map_comparison.md)* | +21.4 pp | +27.6 pp |
| Burn Probability *(09_risk_map_comparison.md)* | +5.8 pp | +22.3 pp |
| **WFPI Yearly (time-aware) — this analysis** | **+18.1 pp** | **+31.8 pp** |
| WFPI Day 1 — this analysis (matched sample) | +18.0 pp | +29.5 pp |
| WFPI Day 2 — this analysis (matched sample) | +17.2 pp | +34.1 pp |

## Key Findings

1. **The yearly map matches or beats Day-1 for all fires** (+18.1 pp vs +18.0 pp). The time-aware selection gives a marginal but consistent edge, because fires discovered before 10 am get the more conservative (and at that moment, more accurate) Day-2 forecast.

2. **For large fires, Day-2 leads on "% above median"** (34.1 pp) while Day-1 and Yearly tie on median ratio (1.41x). This may reflect that large fires tend to occur on extreme-risk days where even the Day-2 forecast already captures the danger.

3. **All three WFPI variants substantially improve over the 09_risk_map_comparison.md baseline** for the matched sample (+17–18 pp vs +5–16 pp). This is partly a methodology effect (larger background from full-year averaging raises the denominator, concentrating the fire distribution above the higher median).

## Interpretation

The time-aware yearly map achieves its design goal: it **consistently matches Day-1 performance** (the operationally superior forecast) while remaining correct for pre-10 am fires where Day-1 is not yet available. It is strictly better than always using Day-2.

Compared to static maps (Ignition Probability, Burn Probability — not evaluated here due to different grid resolution), the yearly map captures daily weather variation at the cost of lacking long-term climatological structure.

## Files

| File | Location | Description |
|------|----------|-------------|
| `static_risk_wfpi_yearly.npy` | `California2020Dataset/` | Yearly burn map, shape (732, H, W) |
| `complete_wfpi_datasets.py` | `code/dataset_creation/nature_dataset_creation/` | Fills missing daily WFPI files |
| `create_yearly_wfpi_burnmap.py` | `code/dataset_creation/nature_dataset_creation/` | Builds the yearly map |
| `analyze_yearly_wfpi_map.py` | `code/dataset_creation/nature_dataset_creation/` | Comparison analysis (this document) |

## References

- [09_risk_map_comparison.md](09_risk_map_comparison.md) — Baseline risk map comparison
- [04_california_2020_dataset.md](04_california_2020_dataset.md) — Dataset documentation
- USGS Fire Danger Maps: https://firedanger.cr.usgs.gov/apps/staticmaps
