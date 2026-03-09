# WFPI Fire Risk Analysis

## Overview

This document presents an analysis of wildfire ignition occurrences with respect to the Wildland Fire Potential Index (WFPI) values in the California 2020 dataset. The analysis investigates whether fires occur more frequently in high-risk or low-risk areas as indicated by WFPI values.

## Dataset Characteristics

### WFPI Data Processing

- **Source:** USGS Wind-enhanced Fire Potential Index (WFPI) Day 2 Forecast maps
- **Date Range:** 2020 (Day 2 forecasts from day before fire discovery)
- **Data Format:** GeoTIFF raster files (uint8)
- **Value Range:** 0-254 (valid operational values)
- **Special Values:** Values ≥ 249 represent special categories:
  - Outside US boundaries
  - Deserts with no ignition risk
  - Other non-operational areas
- **NoData Value:** 255 (masked out during processing)

### Masking Strategy

The dataset applies the following masking rules:
1. **Invalid WFPI data:** Values < 0
2. **NoData values:** Values = 255
3. **Special values:** Values ≥ 249 (non-operational areas)
4. **Urban areas:** Excluded using US Census Urban Area Criteria 2025
5. **California boundary:** Cropped to California state boundary
6. **Connected components:** Only the largest connected component is retained (mainland California)

## WFPI Statistics

### Overall Dataset Statistics

After applying all masking criteria:

- **Valid WFPI cells:** 175,646
- **Value range:** 0-119 (after masking special values ≥ 249)
- **Mean WFPI:** 47.16
- **Median WFPI:** **46.00**
- **Standard deviation:** 17.87

### Value Distribution

The WFPI values in the dataset follow a relatively normal distribution centered around 46, with most values falling in the range of 20-70.

## Fire Occurrence Analysis

### Analysis Methodology

We analyzed **2,407 fires** from the California 2020 dataset, comparing the WFPI values at fire ignition points against the background WFPI distribution across all valid operational areas.

### Key Findings

#### 1. Median Comparison

- **Background median WFPI:** 46.00
- **Fire locations median WFPI:** 48.00
- **Difference:** +2.00 (4.3% higher)

**Conclusion:** Fires occur slightly more frequently in areas with higher WFPI values, indicating a positive correlation between WFPI and fire ignition risk.

#### 2. Risk Category Analysis

Using the median (46.00) as a reference point, we defined risk categories:
- **Low risk:** WFPI < 23.00
- **Medium risk:** WFPI 23.00 - 69.00
- **High risk:** WFPI > 69.00

**Fire Distribution:**
- Low risk: 84 fires (3.5%)
- Medium risk: 1,986 fires (82.5%)
- High risk: 337 fires (14.0%)

**Background Distribution:**
- Low risk: 13,093 cells (7.5%)
- Medium risk: 139,691 cells (79.5%)
- High risk: 22,862 cells (13.0%)

**Key Observations:**
- Most fires (82.5%) occur in medium-risk areas, matching the background distribution (79.5%)
- Fires are **underrepresented** in low-risk areas (3.5% vs 7.5% background)
- Fires are **slightly overrepresented** in high-risk areas (14.0% vs 13.0% background)

#### 3. Relative to Median Analysis

- **Fires in areas with WFPI > median:** 1,278 fires (53.1%)
- **Fires in areas with WFPI ≤ median:** 1,129 fires (46.9%)
- **Background above median:** 48.4%

**Conclusion:** There is a **4.7 percentage point** increase in fire occurrence in areas above the median WFPI value, confirming that higher WFPI values correspond to higher ignition risk.

### Statistical Summary

| Metric | Background | Fire Locations | Difference |
|--------|-------------|----------------|------------|
| Mean | 47.16 | 49.24 | +2.08 |
| Median | 46.00 | 48.00 | +2.00 |
| Min | 0.00 | 9.00 | - |
| Max | 119.00 | 115.00 | - |
| Std Dev | 17.87 | 19.45 | +1.58 |

## Interpretation

### Positive Correlation Confirmed

The analysis confirms that **higher WFPI values correspond to higher fire ignition risk**:

1. **Median shift:** Fire locations have a median WFPI 2 points higher than background
2. **Above-median preference:** 53.1% of fires occur in above-median WFPI areas vs 48.4% of background
3. **High-risk overrepresentation:** 14.0% of fires in high-risk areas vs 13.0% of background
4. **Low-risk underrepresentation:** 3.5% of fires in low-risk areas vs 7.5% of background

### Moderate Effect Size

While the correlation is positive and statistically meaningful, the effect size is **moderate**:
- The median difference is relatively small (2.0 points, ~4.3%)
- Most fires still occur in medium-risk areas (82.5%)
- The distribution of fires largely mirrors the background distribution

This suggests that:
- WFPI is a useful indicator of fire risk, but not the sole determinant
- Other factors (human activity, ignition sources, weather conditions at time of ignition) also play significant roles
- The WFPI Day 2 forecast provides a baseline risk assessment, but actual ignition depends on additional factors

## Visualizations

An analysis plot is available at:
- `California2020Dataset/wfpi_fire_analysis.png`

This plot shows:
1. Overlaid histograms of background WFPI vs fire location WFPI
2. Normalized percentage comparison by WFPI bins
3. Median lines for both distributions

## Implications for Drone Routing

### Risk-Based Routing

The positive correlation between WFPI and fire occurrence suggests:

1. **High-risk area prioritization:** Drones should be prepared to respond more frequently in areas with WFPI > 46 (median)
2. **Resource allocation:** Areas with WFPI > 69 (high-risk) may require more standby resources
3. **Risk awareness:** While WFPI is predictive, the moderate effect size indicates that drones should maintain readiness across all risk levels

### Dataset Validity

The analysis validates the dataset construction:
- WFPI values are correctly interpreted (higher = higher risk)
- Special values (≥ 249) are properly masked
- The dataset provides a realistic representation of fire risk patterns

## Limitations

1. **Temporal mismatch:** WFPI is a Day 2 forecast from the day before the fire, which may not capture all conditions at ignition time
2. **Other risk factors:** WFPI doesn't account for human activity, ignition sources, or local weather variations
3. **Sample size:** Analysis based on 2,407 fires may not capture all patterns
4. **Spatial resolution:** 1km resolution may miss local variations in risk

## Conclusion

The analysis confirms that **WFPI is a valid indicator of fire ignition risk**, with higher WFPI values corresponding to increased fire occurrence. However, the effect size is moderate, indicating that WFPI should be used in conjunction with other factors for comprehensive fire risk assessment and drone routing optimization.

The California 2020 dataset provides a solid foundation for training and evaluating wildfire drone routing algorithms, with WFPI values that accurately reflect fire risk patterns.

## References

- USGS Fire Danger Maps: https://firedanger.cr.usgs.gov/apps/staticmaps
- Wind-enhanced Fire Potential Index (WFPI) documentation
- California 2020 Wildfire Dataset documentation (see `04_california_2020_dataset.md`)
