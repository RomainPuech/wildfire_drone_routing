# Large Fires Analysis: WFPI Correlation Comparison

## Overview

This document compares the fire risk analysis results between:
- **All fires** (no size filter)
- **Large fires only** (>= 100 acres)

The analysis examines how filtering for large fires affects the correlation between WFPI values and fire occurrence for both Day 1 and Day 2 forecasts.

## Fire Size Threshold

**Large Fire Definition:** Fires with `FIRE_SIZE >= 100 acres`

**Rationale:**
- Represents 2.6% of all CA 2020 fires (263 out of 10,198)
- These fires require significant response resources
- More likely to be influenced by environmental conditions (WFPI) rather than human factors
- Better represents fires that pose substantial threat

## Dataset Statistics

### All Fires Datasets

| Metric | Day 2 | Day 1 |
|--------|-------|-------|
| Successful fires | 2,418 | 2,325 |
| Failed fires | 2,260 | 2,354 |
| Background median WFPI | 46.00 | 46.00 |

### Large Fires Only Datasets

| Metric | Day 2 | Day 1 |
|--------|-------|-------|
| Successful fires | 146 | 137 |
| Failed fires | 67 | 76 |
| Background median WFPI | 59.00 | 60.00 |

**Note:** The background median WFPI is higher for large fires datasets because they aggregate across fewer dates (103 unique dates vs 350-351 for all fires), and these dates may have higher overall WFPI values.

## Fire Occurrence Analysis

### Day 2 Forecast

#### All Fires (Day 2)
- **Fire locations median WFPI:** 48.00
- **Background median WFPI:** 46.00
- **Median difference:** +2.00 (4.3% higher)
- **Fires above median:** 53.1% (vs 48.4% background)
- **Difference:** +4.7 percentage points

#### Large Fires Only (Day 2)
- **Fire locations median WFPI:** 75.00
- **Background median WFPI:** 59.00
- **Median difference:** +16.00 (27.1% higher)
- **Fires above median:** 63.0% (vs 49.1% background)
- **Difference:** +13.9 percentage points

**Key Finding:** Large fires show **8x stronger correlation** with WFPI than all fires (+16.00 vs +2.00 median difference).

### Day 1 Forecast

#### All Fires (Day 1)
- **Fire locations median WFPI:** 57.00
- **Background median WFPI:** 46.00
- **Median difference:** +11.00 (23.9% higher)
- **Fires above median:** 65.0% (vs 49.2% background)
- **Difference:** +15.8 percentage points

#### Large Fires Only (Day 1)
- **Fire locations median WFPI:** 72.00
- **Background median WFPI:** 60.00
- **Median difference:** +12.00 (20.0% higher)
- **Fires above median:** 67.2% (vs 49.3% background)
- **Difference:** +17.9 percentage points

**Key Finding:** Large fires show similar correlation strength to all fires for Day 1 (+12.00 vs +11.00), but with higher absolute WFPI values.

## Comparison Summary

### Median WFPI Difference

| Dataset | Day 2 | Day 1 |
|---------|-------|-------|
| All fires | +2.00 | +11.00 |
| Large fires only | **+16.00** | **+12.00** |
| **Improvement** | **8x stronger** | **1.1x stronger** |

### Percentage Above Median

| Dataset | Day 2 | Day 1 |
|---------|-------|-------|
| All fires | +4.7 pp | +15.8 pp |
| Large fires only | **+13.9 pp** | **+17.9 pp** |
| **Improvement** | **3x stronger** | **1.1x stronger** |

## Key Findings

### 1. Large Fires Show Much Stronger Correlation (Day 2)

For Day 2 forecasts, filtering for large fires dramatically improves the correlation:
- **Median difference:** +16.00 vs +2.00 (8x improvement)
- **Above median:** +13.9 pp vs +4.7 pp (3x improvement)

This suggests that **large fires are much more predictable** using Day 2 forecasts, as they are more influenced by environmental conditions rather than random ignition events.

### 2. Day 1 Correlation Remains Strong for Large Fires

For Day 1 forecasts, large fires show similar correlation strength:
- **Median difference:** +12.00 vs +11.00 (similar)
- **Above median:** +17.9 pp vs +15.8 pp (slightly stronger)

Day 1 forecasts already show strong correlation for all fires, so the improvement for large fires is more modest.

### 3. Higher Absolute WFPI Values for Large Fires

Large fires occur in areas with **significantly higher WFPI values**:
- **Day 2:** Median 75.00 vs 48.00 for all fires (+27 points)
- **Day 1:** Median 72.00 vs 57.00 for all fires (+15 points)

This indicates that large fires are more likely to occur in high-risk areas, as expected.

### 4. Better Risk Stratification for Large Fires

Large fires show clearer separation from background:
- **Day 2:** 63.0% of large fires above median vs 53.1% for all fires
- **Day 1:** 67.2% of large fires above median vs 65.0% for all fires

## Interpretation

### Why Large Fires Show Stronger Correlation

1. **Environmental Dependence:** Large fires are more dependent on environmental conditions (weather, fuel moisture, wind) that WFPI captures, rather than random human ignition events.

2. **Less Noise from Small Fires:** Small fires (< 100 acres) are often caused by:
   - Accidental ignitions (cigarettes, equipment)
   - Contained quickly before environmental factors dominate
   - Less influenced by WFPI-predicted conditions

3. **Better Signal-to-Noise Ratio:** Filtering for large fires removes the "noise" of small, random ignitions, revealing the stronger underlying correlation with environmental risk factors.

### Implications

1. **For Predictive Modeling:**
   - **Large fires:** Use Day 2 forecasts for best correlation (+16.00 median difference)
   - **All fires:** Use Day 1 forecasts for best correlation (+11.00 median difference)

2. **For Operational Planning:**
   - Large fires are more predictable using WFPI, making resource allocation more effective
   - Day 2 forecasts are particularly useful for large fire prediction
   - Focus on areas with WFPI > 60-75 for large fire risk

3. **For Dataset Selection:**
   - **Large fires dataset:** Better for understanding environmental risk factors
   - **All fires dataset:** Better for comprehensive fire occurrence patterns

## Statistical Summary

### Day 2 Forecast

| Metric | All Fires | Large Fires Only | Improvement |
|--------|-----------|------------------|-------------|
| Fire median | 48.00 | 75.00 | +27.00 |
| Background median | 46.00 | 59.00 | +13.00 |
| **Median difference** | **+2.00** | **+16.00** | **8x** |
| % above median | 53.1% | 63.0% | +9.9 pp |

### Day 1 Forecast

| Metric | All Fires | Large Fires Only | Improvement |
|--------|-----------|------------------|-------------|
| Fire median | 57.00 | 72.00 | +15.00 |
| Background median | 46.00 | 60.00 | +14.00 |
| **Median difference** | **+11.00** | **+12.00** | **1.1x** |
| % above median | 65.0% | 67.2% | +2.2 pp |

## Visualizations

A comparison plot is available at:
- `documentation/large_fires_analysis.png`

The plot shows:
1. Day 2 forecast histograms (all fires vs large fires)
2. Day 1 forecast histograms (all fires vs large fires)
3. Normalized distributions for both forecast types

## Conclusion

**Key Takeaways:**

1. **Large fires show dramatically stronger correlation with WFPI** for Day 2 forecasts (8x improvement)
2. **Day 1 forecasts maintain strong correlation** for both all fires and large fires
3. **Large fires occur in significantly higher WFPI areas** (median 72-75 vs 48-57 for all fires)
4. **Filtering for large fires improves predictive power**, especially for Day 2 forecasts

**Recommendations:**

- For **large fire prediction**, use **Day 2 forecasts** with large fires dataset (strongest correlation: +16.00 median difference)
- For **comprehensive fire risk assessment**, use **Day 1 forecasts** with all fires dataset (strong correlation: +11.00 median difference)
- **Large fires datasets** are better suited for understanding environmental risk factors and resource allocation
- **All fires datasets** are better for comprehensive fire occurrence patterns and small fire prediction

Both approaches are valid and serve different purposes in wildfire research and operational planning.

## References

- USGS Fire Danger Maps: https://firedanger.cr.usgs.gov/apps/staticmaps
- Wind-enhanced Fire Potential Index (WFPI) documentation
- California 2020 Wildfire Dataset documentation (see `04_california_2020_dataset.md`)
- WFPI Fire Risk Analysis - Day 2 (see `05_wfpi_fire_risk_analysis.md`)
- WFPI Day 1 vs Day 2 Comparison (see `06_wfpi_day1_vs_day2_comparison.md`)
