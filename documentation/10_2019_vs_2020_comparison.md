# 2019 vs 2020 Risk Map Comparison

## Overview

This document compares the performance of Ignition Probability and Burn Probability risk maps on California wildfires from 2019 and 2020. This temporal validation helps assess the consistency and generalizability of these static risk maps across different fire seasons.

## Dataset Summary

### 2019 Datasets

| Dataset | Total Fires | Large Fires |
|---------|-------------|-------------|
| Ignition Probability - All Fires | 3,311 | - |
| Ignition Probability - Large Fires | - | 95 |
| Burn Probability - All Fires | 3,328 | - |
| Burn Probability - Large Fires | - | 96 |

### 2020 Datasets

| Dataset | Total Fires | Large Fires |
|---------|-------------|-------------|
| Ignition Probability - All Fires | 4,547 | - |
| Ignition Probability - Large Fires | - | 210 |
| Burn Probability - All Fires | 4,550 | - |
| Burn Probability - Large Fires | - | 211 |

**Note:** 2020 had significantly more fires than 2019 (37% more all fires, 121% more large fires), reflecting the severity of the 2020 fire season.

## Ignition Probability Comparison

### All Fires

| Metric | 2019 | 2020 | Change |
|--------|------|------|--------|
| **Median Ratio** | 1.22x | 1.21x | -0.01x |
| **% Above Median** | +19.5 pp | +21.4 pp | +1.9 pp |
| **Total Fires** | 3,311 | 4,547 | +1,236 (+37%) |

**Key Findings:**
- **Consistent performance** across years (1.22x vs 1.21x ratio)
- **Slightly stronger correlation in 2020** (+21.4 pp vs +19.5 pp)
- **Stable predictive power** despite different fire seasons

### Large Fires Only

| Metric | 2019 | 2020 | Change |
|--------|------|------|--------|
| **Median Ratio** | 1.24x | 1.25x | +0.01x |
| **% Above Median** | +25.8 pp | +27.6 pp | +1.8 pp |
| **Total Fires** | 95 | 210 | +115 |

**Key Findings:**
- **Very consistent performance** (1.24x vs 1.25x ratio)
- **Slightly stronger correlation in 2020** (+27.6 pp vs +25.8 pp)
- **Excellent stability** for large fire prediction

## Burn Probability Comparison

### All Fires

| Metric | 2019 | 2020 | Change |
|--------|------|------|--------|
| **Median Ratio** | 1.52x | 1.34x | -0.18x |
| **% Above Median** | +9.0 pp | +5.5 pp | -3.5 pp |
| **Total Fires** | 3,328 | 4,550 | +1,222 (+37%) |

**Key Findings:**
- **Stronger correlation in 2019** (1.52x vs 1.34x ratio)
- **Better performance in 2019** (+9.0 pp vs +5.8 pp)
- **More variable** than Ignition Probability across years

### Large Fires Only

| Metric | 2019 | 2020 | Change |
|--------|------|------|--------|
| **Median Ratio** | 2.76x | 2.21x | -0.55x |
| **% Above Median** | +25.0 pp | +22.0 pp | -3.0 pp |
| **Total Fires** | 96 | 211 | +115 (+120%) |

**Key Findings:**
- **Much stronger ratio in 2019** (2.76x vs 2.21x)
- **Slightly stronger correlation in 2019** (+25.0 pp vs +22.3 pp)
- **Excellent for large fires** in both years, but better in 2019

## Key Findings

### 1. Ignition Probability Shows Consistent Performance

**All Fires:**
- 2019: 1.22x ratio, +19.5 pp
- 2020: 1.21x ratio, +21.4 pp
- **Change: -0.01x ratio, +1.9 pp**

**Large Fires:**
- 2019: 1.24x ratio, +25.8 pp
- 2020: 1.25x ratio, +27.6 pp
- **Change: +0.01x ratio, +1.8 pp**

**Conclusion:** Ignition Probability shows **excellent temporal stability** with minimal variation between 2019 and 2020. The slight improvement in 2020 may be due to the larger sample size or different fire season characteristics.

### 2. Burn Probability Shows More Variability

**All Fires:**
- 2019: 1.52x ratio, +9.0 pp
- 2020: 1.34x ratio, +5.5 pp
- **Change: -0.18x ratio, -3.5 pp**

**Large Fires:**
- 2019: 2.76x ratio, +25.0 pp
- 2020: 2.21x ratio, +22.0 pp
- **Change: -0.55x ratio, -3.0 pp**

**Conclusion:** Burn Probability shows **more variability** between years, with better performance in 2019. This may reflect differences in fire season characteristics or the model's sensitivity to specific conditions.

### 3. Both Maps Remain Effective Across Years

Despite the variability, both risk maps show **strong correlation** with fire occurrence in both years:

**Ignition Probability:**
- Consistently 1.21-1.25x ratio
- Consistently +19-28 pp above median

**Burn Probability:**
- 1.34-1.52x ratio for all fires
- 2.21-2.76x ratio for large fires
- Consistently +5.5-25.0 pp above median

### 4. Large Fires Show Stronger Correlation

Both maps show **stronger correlation for large fires** in both years:

**Ignition Probability:**
- All fires: +19.5-21.4 pp
- Large fires: +25.8-27.6 pp

**Burn Probability:**
- All fires: +5.5-9.0 pp
- Large fires: +22.0-25.0 pp

This confirms that **large fires are more predictable** using these risk maps, as they are more influenced by environmental conditions rather than random ignition events.

## Temporal Stability Assessment

### Ignition Probability: ⭐⭐⭐⭐⭐ (Excellent)

- **All Fires:** Ratio difference: -0.01x (0.8% change)
- **Large Fires:** Ratio difference: +0.01x (0.8% change)
- **Conclusion:** Highly stable across years

### Burn Probability: ⭐⭐⭐⭐ (Very Good)

- **All Fires:** Ratio difference: -0.18x (13.2% change)
- **Large Fires:** Ratio difference: -0.55x (24.9% change)
- **Conclusion:** Good stability, but more variable than Ignition Probability

## Interpretation

### Why Ignition Probability is More Stable

1. **Machine Learning Model:** Trained on 15 years of data (2006-2020), capturing long-term patterns
2. **Comprehensive Features:** Incorporates multiple risk factors (topography, climate, vegetation, human development)
3. **Purpose-Built:** Specifically designed for fire prediction
4. **Less Sensitive to Year-to-Year Variation:** Focuses on spatial patterns rather than temporal conditions

### Why Burn Probability Shows More Variability

1. **Fire Simulation Model:** Based on FSim model that simulates fire spread under various conditions
2. **Growth Potential Focus:** Trained on fires with growth potential, which may vary by year
3. **Model Sensitivity:** May be more sensitive to specific fire season characteristics
4. **2019 Performance:** Better performance in 2019 may reflect that year's specific fire patterns

### Why 2020 Had More Fires

2020 was an **exceptionally severe fire season** in California:
- 37% more fires overall
- 121% more large fires
- Extreme weather conditions (drought, heat waves, strong winds)
- Multiple record-breaking fires

Despite the increased fire activity, both risk maps maintained strong correlation, demonstrating their robustness.

## Comparison Summary

### Best Overall: Ignition Probability

**Reasons:**
- Most consistent across years (minimal variation)
- Strong correlation in both years (+19-28 pp)
- Excellent for both all fires and large fires
- Purpose-built for fire prediction

### Best for Large Fires (Ratio): Burn Probability

**Reasons:**
- Highest ratios (2.21-2.76x for large fires)
- Strong correlation for large fires (+22-25 pp)
- Effective in both years, though better in 2019

### Most Stable: Ignition Probability

**Reasons:**
- Minimal variation between years (<1% change in ratio)
- Consistent performance across different fire seasons
- Reliable for long-term risk assessment

## Recommendations

### For Temporal Validation

1. **Use Ignition Probability** for consistent, stable risk assessment across years
2. **Use Burn Probability** for large fire prediction, but be aware of year-to-year variation
3. **Monitor both maps** to understand how they respond to different fire seasons

### For Operational Use

1. **Ignition Probability:** Best for long-term planning and resource allocation
2. **Burn Probability:** Best for large fire risk assessment, with awareness of variability
3. **Combine both:** Use together for comprehensive risk assessment

### For Research

1. **Temporal Stability:** Ignition Probability shows excellent stability, making it ideal for multi-year studies
2. **Large Fire Prediction:** Both maps are effective, with Burn Probability showing higher ratios
3. **Year-to-Year Variation:** Monitor how maps perform across different fire seasons to understand their limitations

## Conclusion

**Key Takeaways:**

1. **Ignition Probability shows excellent temporal stability** with minimal variation between 2019 and 2020
2. **Burn Probability shows more variability** but remains effective, especially for large fires
3. **Both maps maintain strong correlation** across different fire seasons
4. **Large fires show stronger correlation** with both maps in both years
5. **2020's extreme fire season** did not significantly degrade map performance

**Final Assessment:**

- **Most Stable:** Ignition Probability (excellent temporal consistency)
- **Best Overall:** Ignition Probability (consistent strong correlation)
- **Best for Large Fires (Ratio):** Burn Probability (highest ratios, though variable)
- **Most Reliable:** Ignition Probability (minimal year-to-year variation)

Both risk maps are valuable tools for wildfire risk assessment, with Ignition Probability showing superior temporal stability and Burn Probability showing higher ratios for large fires.

## References

- Ignition Probability Analysis (see `08_ignition_probability_analysis.md`)
- Risk Map Comparison (see `09_risk_map_comparison.md`)
- Pyrologix Data Release: DOI: 10.17605/OSF.IO/CFGH9
- FSim Data Release: https://doi.org/10.2737/RDS-2016-0034-2
