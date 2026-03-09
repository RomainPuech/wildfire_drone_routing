# Risk Map Comparison: Which is Most Accurate?

NOTE: ALL BUT WFPI MIGHT HAVE DATA LEAKAGE AS THEY COULD BE TRAINED ON 2020 DATA! PYROLOGIX IS FOR SURE CONTAMINATED, AND UNSURE FOR BP/WHP


## Overview

This document compares four different wildfire risk maps to determine which is most accurate for predicting fire occurrence in California 2020:

1. **WFPI Day 2 Forecast** - Wind-enhanced Fire Potential Index (day before forecast)
2. **WFPI Day 1 Forecast** - Wind-enhanced Fire Potential Index (same-day forecast)
3. **Ignition Probability (Pyrologix)** - Machine learning model trained on 2006-2020 fire data
4. **Burn Probability (FSim)** - Fire simulation model estimating probability of burning

## Summary Table: All Fires

| Risk Map | Median Ratio | % Above Median | Improvement vs Background |
|----------|--------------|----------------|---------------------------|
| **WFPI Day 2** | 1.04x | +4.7 pp | Weak |
| **WFPI Day 1** | 1.24x | +15.8 pp | Moderate |
| **Ignition Probability** | 1.21x | +21.4 pp | **Strong** |
| **Burn Probability (FSim)** | 1.34x | +5.8 pp | Moderate |

**Winner for All Fires: Ignition Probability** (1.21x ratio, +21.4 pp)

## Summary Table: Large Fires Only (>= 100 acres)

| Risk Map | Median Ratio | % Above Median | Improvement vs Background |
|----------|--------------|----------------|---------------------------|
| **WFPI Day 2** | 1.27x | +13.9 pp | Moderate |
| **WFPI Day 1** | 1.20x | +17.9 pp | Moderate |
| **Ignition Probability** | 1.25x | +27.6 pp | **Strong** |
| **Burn Probability (FSim)** | 2.21x | +22.3 pp | **Very Strong** |

**Winner for Large Fires: Burn Probability (FSim)** (2.21x ratio, +22.3 pp)

## Detailed Analysis

### 1. WFPI Day 2 Forecast

**All Fires:**
- Fire locations median: 48.00 vs background median: 46.00
- Ratio: 1.04x (4.3% higher)
- 53.1% of fires above median (vs 48.4% background)
- **Difference: +4.7 percentage points**

**Large Fires:**
- Fire locations median: 75.00 vs background median: 59.00
- Ratio: 1.27x (27.1% higher)
- 63.0% of fires above median (vs 49.1% background)
- **Difference: +13.9 percentage points**

**Strengths:**
- Time-varying, captures daily weather conditions
- Operational, updated daily
- Better for large fires than all fires

**Weaknesses:**
- Weak correlation for all fires (+4.7 pp)
- Day 2 forecast less accurate than Day 1

### 2. WFPI Day 1 Forecast

**All Fires:**
- Fire locations median: 57.00 vs background median: 46.00
- Ratio: 1.24x (23.9% higher)
- 65.0% of fires above median (vs 49.2% background)
- **Difference: +15.8 percentage points**

**Large Fires:**
- Fire locations median: 72.00 vs background median: 60.00
- Ratio: 1.20x (20.0% higher)
- 67.2% of fires above median (vs 49.3% background)
- **Difference: +17.9 percentage points**

**Strengths:**
- Strong correlation for all fires (+15.8 pp)
- Time-varying, captures current conditions
- Operational, updated daily
- Better than Day 2 forecast

**Weaknesses:**
- Slightly weaker than Ignition Probability for all fires
- Weaker than Burn Probability for large fires

### 3. Ignition Probability (Pyrologix)

**All Fires:**
- Fire locations median: 0.000266 vs background median: 0.000219
- Ratio: 1.21x (21.5% higher)
- 71.4% of fires above median (vs 50.0% background)
- **Difference: +21.4 percentage points**

**Large Fires:**
- Fire locations median: 0.000273 vs background median: 0.000219
- Ratio: 1.25x (24.7% higher)
- 77.6% of fires above median (vs 50.0% background)
- **Difference: +27.6 percentage points**

**Strengths:**
- **Best correlation for all fires** (+21.4 pp)
- **Best correlation for large fires** (+27.6 pp)
- Purpose-built for fire prediction
- Stable long-term patterns
- Comprehensive feature set (topography, climate, vegetation, human development)

**Weaknesses:**
- Static map (not time-varying)
- Not suitable for daily operational decisions

### 4. Burn Probability (FSim)

**All Fires:**
- Fire locations median: 0.003492 vs background median: 0.002598
- Ratio: 1.34x (34.4% higher)
- 55.5% of fires above median (vs 49.7% background)
- **Difference: +5.8 percentage points**

**Large Fires:**
- Fire locations median: 0.005746 vs background median: 0.002598
- Ratio: 2.21x (121.3% higher)
- 72.0% of fires above median (vs 49.7% background)
- **Difference: +22.3 percentage points**

**Strengths:**
- **Best ratio for large fires** (2.21x)
- Strong correlation for large fires (+22.3 pp)
- Based on fire simulation model
- Trained on fires with growth potential

**Weaknesses:**
- Weak correlation for all fires (+5.8 pp)
- Static map (not time-varying)
- Lower resolution (270m vs 120m for Ignition Probability)

## Key Findings

### 1. Best Overall: Ignition Probability (Pyrologix)

**For all fires:** Ignition Probability shows the **strongest correlation** (+21.4 pp), outperforming all other maps.

**For large fires:** Ignition Probability also shows the **strongest correlation** (+27.6 pp), though Burn Probability has a higher ratio (2.21x vs 1.25x).

**Why it's best:**
- Purpose-built for fire prediction using machine learning
- Trained on comprehensive historical fire data (2006-2020)
- Incorporates multiple risk factors (topography, climate, vegetation, human development)
- Shows consistent strong correlation for both all fires and large fires

### 2. Best for Large Fires: Burn Probability (FSim)

**For large fires:** Burn Probability shows the **highest ratio** (2.21x), meaning large fires occur in areas with more than double the background burn probability.

**Why it's good for large fires:**
- Trained specifically on fires with growth potential (> 100 ha)
- Based on fire simulation model that captures fire spread dynamics
- Strong correlation (+22.3 pp) for large fires

**Why it's not best overall:**
- Weak correlation for all fires (+5.8 pp)
- Lower resolution (270m)

### 3. Best for Operational Use: WFPI Day 1

**For operational decisions:** WFPI Day 1 is the best choice because:
- Time-varying, captures daily weather conditions
- Updated daily for real-time fire danger assessment
- Strong correlation (+15.8 pp for all fires, +17.9 pp for large fires)
- Operational tool used by fire management agencies

### 4. Weakest: WFPI Day 2

**For all fires:** WFPI Day 2 shows the **weakest correlation** (+4.7 pp), making it less useful for predicting fire occurrence.

**Why it's weak:**
- Day 2 forecast less accurate than Day 1
- Weather conditions change rapidly
- Less predictive power for fire occurrence

## Comparison Matrix

### All Fires

| Metric | WFPI Day 2 | WFPI Day 1 | Ignition Prob | Burn Prob |
|--------|------------|------------|---------------|-----------|
| Median Ratio | 1.04x | 1.24x | **1.21x** | 1.34x |
| % Above Median | +4.7 pp | +15.8 pp | **+21.4 pp** | +5.8 pp |
| **Rank** | 4th | 2nd | **1st** | 3rd |

### Large Fires Only

| Metric | WFPI Day 2 | WFPI Day 1 | Ignition Prob | Burn Prob |
|--------|------------|------------|---------------|-----------|
| Median Ratio | 1.27x | 1.20x | 1.25x | **2.21x** |
| % Above Median | +13.9 pp | +17.9 pp | **+27.6 pp** | +22.3 pp |
| **Rank** | 4th | 3rd | **1st** | 2nd |

## Interpretation

### Why Ignition Probability is Most Accurate

1. **Purpose-Built:** Specifically designed to predict fire ignitions using machine learning
2. **Comprehensive Training:** Trained on 15 years of historical fire data (2006-2020)
3. **Multi-Factor Model:** Incorporates topography, climate, vegetation, and human development
4. **Consistent Performance:** Strong correlation for both all fires and large fires
5. **High Resolution:** 120m resolution provides detailed spatial information

### Why Burn Probability is Best for Large Fires (Ratio)

1. **Growth Potential Focus:** Trained on fires with growth potential (> 100 ha)
2. **Fire Simulation:** Based on FSim model that captures fire spread dynamics
3. **High Ratio:** 2.21x ratio means large fires occur in areas with more than double the background risk
4. **Strong Correlation:** +22.3 pp for large fires

### Why WFPI Day 1 is Best for Operations

1. **Time-Varying:** Captures daily weather conditions
2. **Operational:** Updated daily for real-time fire danger assessment
3. **Weather-Driven:** Reflects current conditions (wind, temperature, moisture)
4. **Strong Correlation:** +15.8 pp for all fires, +17.9 pp for large fires

## Recommendations

### For Long-Term Risk Assessment
**Use: Ignition Probability (Pyrologix)**
- Best overall correlation (+21.4 pp for all fires, +27.6 pp for large fires)
- Stable long-term patterns
- Purpose-built for fire prediction

### For Large Fire Prediction
**Use: Burn Probability (FSim) or Ignition Probability**
- Burn Probability: Highest ratio (2.21x) for large fires
- Ignition Probability: Strongest correlation (+27.6 pp) for large fires
- Both are effective, choose based on specific needs

### For Daily Operational Decisions
**Use: WFPI Day 1 Forecast**
- Time-varying, captures current conditions
- Updated daily
- Strong correlation (+15.8 pp for all fires, +17.9 pp for large fires)
- Operational tool used by fire management agencies

### For Comprehensive Analysis
**Use: Multiple Maps Together**
- **Ignition Probability:** Baseline long-term risk
- **WFPI Day 1:** Daily fire danger conditions
- **Burn Probability:** Large fire growth potential
- **WFPI Day 2:** Less useful, can be skipped

## Conclusion

**Most Accurate Risk Map: Ignition Probability (Pyrologix)**

The Ignition Probability map shows the **strongest correlation** with fire occurrence for both all fires (+21.4 pp) and large fires (+27.6 pp). It is purpose-built for fire prediction using machine learning trained on comprehensive historical data, making it the most accurate risk map overall.

**Best for Large Fires (Ratio): Burn Probability (FSim)**

The Burn Probability map shows the **highest ratio** (2.21x) for large fires, meaning large fires occur in areas with more than double the background burn probability. It is particularly effective for predicting large fire occurrence.

**Best for Operations: WFPI Day 1**

The WFPI Day 1 forecast is the best choice for daily operational decisions because it is time-varying, updated daily, and shows strong correlation with fire occurrence.

**All maps serve different purposes:**
- **Ignition Probability:** Long-term risk assessment (most accurate)
- **Burn Probability:** Large fire prediction (highest ratio)
- **WFPI Day 1:** Daily operational decisions (time-varying)
- **WFPI Day 2:** Less useful, can be skipped

## References

- WFPI Fire Risk Analysis (see `05_wfpi_fire_risk_analysis.md`)
- WFPI Day 1 vs Day 2 Comparison (see `06_wfpi_day1_vs_day2_comparison.md`)
- Large Fires Analysis (see `07_large_fires_analysis.md`)
- Ignition Probability Analysis (see `08_ignition_probability_analysis.md`)
- Pyrologix Data Release: DOI: 10.17605/OSF.IO/CFGH9
- FSim Data Release: https://doi.org/10.2737/RDS-2016-0034-2
