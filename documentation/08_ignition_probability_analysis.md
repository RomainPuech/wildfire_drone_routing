# Ignition Probability Map Analysis

## Overview

This document analyzes the correlation between the Pyrologix Wildfire Ignition Probability Map and actual fire occurrences in California 2020. The analysis compares results for all fires and large fires only (>= 100 acres).

## Ignition Probability Map

**Source:** Pyrologix Wildfire Ignition Probability by Total (Human + Natural) Cause for the Western United States

**Key Properties:**
- **Resolution:** 120 meters
- **CRS:** EPSG:5070 (Albers Equal Area Conic)
- **Value Range:** 0.0 to ~0.000435 (probability values)
- **Coverage:** Western United States
- **Model Type:** Random forest algorithm trained on 2006-2020 fire data
- **Purpose:** Predicts probability of wildfire ignition with growth potential (> 100 ha for western US)

**Note:** This is a **static map** (not time-varying like WFPI), representing long-term spatial patterns of ignition risk based on topographic, climatic, vegetative, and human development features.

## Dataset Statistics

### All Fires Dataset

| Metric | Value |
|--------|-------|
| Successful fires | 4,547 |
| Failed fires | 131 |
| Background median ignition probability | 0.000219 |
| Fire locations median ignition probability | 0.000266 |

### Large Fires Only Dataset (>= 100 acres)

| Metric | Value |
|--------|-------|
| Successful fires | 210 |
| Failed fires | 3 |
| Background median ignition probability | 0.000219 |
| Fire locations median ignition probability | 0.000273 |

## Fire Occurrence Analysis

### All Fires

**Background Statistics:**
- Valid cells: 26,935,429
- Median: 0.000219
- Mean: 0.000211
- Std: 0.000088
- Range: [0.000000, 0.000416]

**Fire Locations Statistics:**
- Total fires analyzed: 4,547
- Median: 0.000266
- Mean: 0.000252
- Range: [0.000000, 0.000405]

**Comparison:**
- **Median difference:** +0.000047 (21.5% higher)
- **Ratio:** 1.21x
- **Fires above median:** 71.4% (vs 50.0% background)
- **Difference:** +21.4 percentage points

### Large Fires Only (>= 100 acres)

**Background Statistics:**
- Valid cells: 26,935,429
- Median: 0.000219
- Mean: 0.000211
- Std: 0.000088
- Range: [0.000000, 0.000416]

**Fire Locations Statistics:**
- Total fires analyzed: 210
- Median: 0.000273
- Mean: 0.000266
- Range: [0.000114, 0.000405]

**Comparison:**
- **Median difference:** +0.000054 (24.7% higher)
- **Ratio:** 1.25x
- **Fires above median:** 77.6% (vs 50.0% background)
- **Difference:** +27.6 percentage points

## Key Findings

### 1. Strong Positive Correlation

Both all fires and large fires show **strong positive correlation** with ignition probability:
- **All fires:** 1.21x higher median (71.4% above median)
- **Large fires:** 1.25x higher median (77.6% above median)

This confirms that the ignition probability map effectively captures spatial patterns of fire risk.

### 2. Large Fires Show Slightly Stronger Correlation

Large fires show a **slightly stronger correlation** than all fires:
- **Median ratio:** 1.25x vs 1.21x (3% improvement)
- **Above median:** 77.6% vs 71.4% (+6.2 percentage points)

This is consistent with the map being trained on fires with growth potential (> 100 ha), making it more predictive for large fires.

### 3. Consistent Background Distribution

The background distribution is identical for both datasets (same map, same mask), confirming that the difference in fire statistics reflects the filtering by fire size.

### 4. Higher Absolute Values for Large Fires

Large fires occur in areas with **slightly higher ignition probability**:
- **All fires median:** 0.000266
- **Large fires median:** 0.000273 (+2.6%)

This suggests that large fires are more likely to occur in areas with higher baseline ignition risk.

## Comparison with WFPI Analysis

### All Fires

| Metric | WFPI Day 2 | WFPI Day 1 | Ignition Probability |
|--------|------------|------------|---------------------|
| Median difference | +2.00 (4.3%) | +11.00 (23.9%) | +0.000047 (21.5%) |
| Ratio | 1.04x | 1.24x | 1.21x |
| % above median | +4.7 pp | +15.8 pp | +21.4 pp |

**Key Insight:** Ignition probability shows **stronger correlation** than WFPI Day 2 (+21.4 pp vs +4.7 pp) and **similar strength** to WFPI Day 1 (+21.4 pp vs +15.8 pp).

### Large Fires Only

| Metric | WFPI Day 2 | WFPI Day 1 | Ignition Probability |
|--------|------------|------------|---------------------|
| Median difference | +16.00 (27.1%) | +12.00 (20.0%) | +0.000054 (24.7%) |
| Ratio | 1.27x | 1.20x | 1.25x |
| % above median | +13.9 pp | +17.9 pp | +27.6 pp |

**Key Insight:** For large fires, ignition probability shows **stronger correlation** than both WFPI Day 2 (+27.6 pp vs +13.9 pp) and WFPI Day 1 (+27.6 pp vs +17.9 pp).

## Interpretation

### Why Ignition Probability Shows Strong Correlation

1. **Purpose-Built for Fire Prediction:** The map is specifically designed to predict fire ignitions using machine learning trained on historical fire data (2006-2020).

2. **Comprehensive Feature Set:** The model incorporates:
   - Spatial trends of observed fire ignitions
   - Topographic features
   - Climatic variables
   - Vegetative characteristics
   - Human development patterns

3. **Long-Term Patterns:** As a static map, it captures long-term spatial patterns of ignition risk, which are more stable than day-to-day weather conditions (WFPI).

4. **Trained on Large Fires:** The model is trained on fires with growth potential (> 100 ha), making it particularly effective for large fire prediction.

### Comparison with WFPI

**WFPI Advantages:**
- **Time-varying:** Captures daily weather conditions and fire danger
- **Operational:** Updated daily for real-time fire danger assessment
- **Weather-driven:** Reflects current conditions (wind, temperature, moisture)

**Ignition Probability Advantages:**
- **Stronger correlation:** Better predictive power for fire occurrence
- **Stable:** Long-term patterns are more consistent
- **Purpose-built:** Specifically designed for fire prediction
- **Comprehensive:** Incorporates multiple risk factors beyond weather

**Complementary Use:**
- **Ignition Probability:** Best for long-term risk assessment and resource allocation
- **WFPI:** Best for daily operational decisions and short-term fire danger

## Statistical Summary

### All Fires

| Metric | Background | Fire Locations | Difference |
|--------|-----------|----------------|------------|
| Median | 0.000219 | 0.000266 | +0.000047 (21.5%) |
| Mean | 0.000211 | 0.000252 | +0.000041 (19.4%) |
| % above median | 50.0% | 71.4% | +21.4 pp |

### Large Fires Only

| Metric | Background | Fire Locations | Difference |
|--------|-----------|----------------|------------|
| Median | 0.000219 | 0.000273 | +0.000054 (24.7%) |
| Mean | 0.000211 | 0.000266 | +0.000055 (26.1%) |
| % above median | 50.0% | 77.6% | +27.6 pp |

## Visualizations

A comparison plot is available at:
- `documentation/ignition_prob_analysis.png`

The plot shows:
1. All fires histograms (background vs fire locations)
2. Large fires histograms (background vs fire locations)
3. Normalized distributions for both datasets

## Conclusion

**Key Takeaways:**

1. **Ignition probability map shows strong correlation** with fire occurrence (1.21-1.25x median ratio, 71-78% above median)

2. **Large fires show slightly stronger correlation** than all fires (1.25x vs 1.21x, 77.6% vs 71.4% above median)

3. **Ignition probability outperforms WFPI Day 2** for both all fires and large fires

4. **Ignition probability is comparable to WFPI Day 1** for all fires and **outperforms it** for large fires

5. **The map is effective for long-term risk assessment** and resource allocation, complementing WFPI's operational use

**Recommendations:**

- **For long-term risk assessment:** Use ignition probability map (stronger correlation, stable patterns)
- **For daily operational decisions:** Use WFPI Day 1 forecasts (time-varying, weather-driven)
- **For large fire prediction:** Use ignition probability map (1.25x ratio, 77.6% above median)
- **For comprehensive analysis:** Use both maps together (ignition probability for baseline risk, WFPI for daily conditions)

Both approaches are valuable and serve complementary purposes in wildfire research and operational planning.

## References

- Pyrologix Data Release: DOI: 10.17605/OSF.IO/CFGH9
- Vibrant Planet Data Commons: www.vpdatacommons.org
- California 2020 Wildfire Dataset documentation (see `04_california_2020_dataset.md`)
- WFPI Fire Risk Analysis (see `05_wfpi_fire_risk_analysis.md`)
- WFPI Day 1 vs Day 2 Comparison (see `06_wfpi_day1_vs_day2_comparison.md`)
- Large Fires Analysis (see `07_large_fires_analysis.md`)
