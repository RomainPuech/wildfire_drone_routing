# Risk Map Comparison: California 2021 USFS Dataset

> **Data leakage note:** Pyrologix was trained on fires from **2006–2020 only**. The California
> 2021 dataset contains exclusively 2021 fires, so Pyrologix is **strictly out-of-sample** here
> — there is **no data leakage**. Burn Probability (FSim) is a simulation model, not trained on
> fire occurrence data, so it is also leakage-free. The WFPI-derived maps (Yearly, D1, D2,
> Averaged, Burn-at-least-once) are likewise leakage-free.

## Overview

This document evaluates **seven burn map variants** on the **California 2021 USFS wildfire
dataset** (931 fires with recorded discovery times, out of 932 in the dataset).
Fires come from the USFS ignition-point database, filtered with the 2020 Day-1
union-of-burnable validity mask, and excluding fires on dates with missing 2021 WFPI zips.

Two methodologies are used:

**Methodology A — Annual background (global):** the risk-map value at each fire's ignition
cell is compared to a single background median computed over all valid cells and all days
(or the static map). This is the standard methodology from doc 09 and doc 12.

**Methodology B — Per-day background (WFPI maps only):** for each fire, the background
median is computed from that fire's *specific* daily WFPI map. This removes seasonal
bias (a summer fire is no longer compared to a winter-diluted background) and directly
answers: *was this ignition in a relatively high-risk area on its own day?*

The key metric is **improvement vs background** = (% fires above bg median) − 50%.
A positive value means fires cluster in higher-risk areas; negative means anti-correlation.

### Maps Evaluated

| # | Map | Type | Notes |
|---|-----|------|-------|
| 1 | WFPI Yearly 2021 (time-aware) | Time-varying | D2 before 10 am, D1 from 10 am — same logic as 2020 yearly map |
| 2 | WFPI Day 1 2021 | Time-varying | Per-fire same-day forecast |
| 3 | WFPI Day 2 2021 | Time-varying | Per-fire day-before forecast |
| 4 | WFPI 2021 Averaged | Static | Mean over year, values ≥249 excluded |
| 5 | WFPI 2021 Burn-at-least-once | Static | P(≥1 high-WFPI day) remapped to 0–248; saturated for most cells |
| 6 | Ignition Probability (Pyrologix) | Static | ML model, 2006–2020 training; resampled to 1309×805 |
| 7 | Burn Probability (FSim/BP) | Static | Fire-simulation model; resampled to 1309×805 |

## Dataset

- **Fires analysed:** 931 (all with valid discovery date + time in config)
- **Large fires (≥ 100 acres, size class D–K):** 33
- **Grid:** 1309 × 805, 1 km, WFPI Lambert Azimuthal Equal-Area CRS
- **Valid cells (mask == 1):** 249,255
- **Mask:** `mask_union_burnable_no_snow_excluded_day1.npy` (2020 D1 union-of-burnable)

## Summary Table

- **Bg Median:** median map value over all valid California cells
- **% Above Bg:** fraction of fire cells with value > bg median (50% = random)
- **Improvement vs bg:** (% Above Bg) − 50% — positive means correlated with fires
- **Ratio:** fire median / background median

| Rank | Risk Map | Bg Median | Fire Med. (All) | Ratio | % Above (All) | Improvement (All) | % Above (Large) | Improvement (Large) |
|------|----------|-----------|-----------------|-------|---------------|-------------------|-----------------|---------------------|
| 1 | **Ignition Probability (Pyrologix)** | 161.068 | 174.25 | 1.08x | 71.0% | +21.0 pp | 63.6% | +13.6 pp |
| 2 | **Burn Probability (FSim/BP)** | 15.434 | 21.36 | 1.38x | 59.3% | +9.3 pp | 60.6% | +10.6 pp |
| 3 | **WFPI 2021 Averaged (excl. ≥249)** | 46.592 | 43.84 | 0.94x | 46.5% | -3.5 pp | 51.5% | +1.5 pp |
| 4 | **WFPI Yearly 2021 (time-aware)** | 63.637 | 47.00 | 0.74x | 34.6% | -15.4 pp | 39.4% | -10.6 pp |
| 5 | **WFPI Day 2 2021** | 63.753 | 48.00 | 0.75x | 34.5% | -15.5 pp | 42.4% | -7.6 pp |
| 6 | **WFPI Day 1 2021** | 63.529 | 48.00 | 0.76x | 33.6% | -16.4 pp | 39.4% | -10.6 pp |
| 7 | **WFPI 2021 Burn-at-least-once** | 248.000 | 248.00 | 1.00x | 0.0% | -50.0 pp | 0.0% | -50.0 pp |

**Best for all fires (annual bg):** Ignition Probability (Pyrologix) (71.0% fires above bg, +21.0 pp)
**Best for large fires (annual bg):** Ignition Probability (Pyrologix) (63.6% fires above bg, +13.6 pp)

## Per-Day Background: WFPI Maps (Corrected Methodology)

For time-varying maps, comparing each fire to a single annual background median is
misleading: a fire in summer is compared to a background that mixes summer and winter days.
The correct approach is to compare each fire to the **background of its own day**:

1. Load the daily WFPI map for each fire's discovery date.
2. Compute the median over all 249,255 valid California cells for that day.
3. Check whether the fire's value exceeds that day's median.
4. Average the binary above/below flags across all fires.

This removes the seasonal bias and measures whether each fire was in a relatively
**high-risk area on its specific day** — regardless of the season.

| Map | % Above Day Bg (All) | Improvement (All) | % Above Day Bg (Large) | Improvement (Large) | N fires |
|-----|----------------------|-------------------|------------------------|---------------------|---------|
| **WFPI Yearly 2021 (time-aware)** | 36.6% | -13.4 pp | 39.4% | -10.6 pp | 931 |
| **WFPI Day 1 2021** | 36.7% | -13.3 pp | 39.4% | -10.6 pp | 931 |
| **WFPI Day 2 2021** | 35.2% | -14.8 pp | 36.4% | -13.6 pp | 931 |

**Best WFPI map (per-day bg, all fires):** WFPI Day 1 2021 (36.7%, -13.3 pp)
**Best WFPI map (per-day bg, large fires):** WFPI Day 1 2021 (39.4%, -10.6 pp)

## Rankings

### All Fires — ranked by improvement vs 50% background

| Rank | Map | % Fires Above Bg Median | Improvement vs 50% | Assessment |
|------|-----|------------------------|---------------------|------------|
| 1 | Ignition Probability (Pyrologix) | 71.0% | +21.0 pp | Strong positive correlation |
| 2 | Burn Probability (FSim/BP) | 59.3% | +9.3 pp | Moderate positive correlation |
| 3 | WFPI 2021 Averaged (excl. ≥249) | 46.5% | -3.5 pp | Near-random / no signal |
| 4 | WFPI Yearly 2021 (time-aware) | 34.6% | -15.4 pp | Strong anti-correlation or saturated |
| 5 | WFPI Day 2 2021 | 34.5% | -15.5 pp | Strong anti-correlation or saturated |
| 6 | WFPI Day 1 2021 | 33.6% | -16.4 pp | Strong anti-correlation or saturated |
| 7 | WFPI 2021 Burn-at-least-once | 0.0% | -50.0 pp | Strong anti-correlation or saturated |

### Large Fires — ranked by improvement vs 50% background

| Rank | Map | % Fires Above Bg Median | Improvement vs 50% |
|------|-----|------------------------|---------------------|
| 1 | Ignition Probability (Pyrologix) | 63.6% | +13.6 pp |
| 2 | Burn Probability (FSim/BP) | 60.6% | +10.6 pp |
| 3 | WFPI 2021 Averaged (excl. ≥249) | 51.5% | +1.5 pp |
| 4 | WFPI Day 2 2021 | 42.4% | -7.6 pp |
| 5 | WFPI Yearly 2021 (time-aware) | 39.4% | -10.6 pp |
| 6 | WFPI Day 1 2021 | 39.4% | -10.6 pp |
| 7 | WFPI 2021 Burn-at-least-once | 0.0% | -50.0 pp |

## Detailed Results

### Ignition Probability (Pyrologix)

- **Background median:** 161.068
- **All fires** (931): fire median = 174.25, ratio = 1.08x, 71.0% above bg median (improvement: +21.0 pp)
- **Large fires** (33): fire median = 168.60, ratio = 1.05x, 63.6% above bg median (improvement: +13.6 pp)

### Burn Probability (FSim/BP)

- **Background median:** 15.434
- **All fires** (931): fire median = 21.36, ratio = 1.38x, 59.3% above bg median (improvement: +9.3 pp)
- **Large fires** (33): fire median = 23.38, ratio = 1.51x, 60.6% above bg median (improvement: +10.6 pp)

### WFPI 2021 Averaged (excl. ≥249)

- **Background median:** 46.592
- **All fires** (931): fire median = 43.84, ratio = 0.94x, 46.5% above bg median (improvement: -3.5 pp)
- **Large fires** (33): fire median = 48.28, ratio = 1.04x, 51.5% above bg median (improvement: +1.5 pp)

### WFPI Yearly 2021 (time-aware)

- **Background median:** 63.637
- **All fires** (931): fire median = 47.00, ratio = 0.74x, 34.6% above bg median (improvement: -15.4 pp)
- **Large fires** (33): fire median = 56.00, ratio = 0.88x, 39.4% above bg median (improvement: -10.6 pp)

### WFPI Day 2 2021

- **Background median:** 63.753
- **All fires** (931): fire median = 48.00, ratio = 0.75x, 34.5% above bg median (improvement: -15.5 pp)
- **Large fires** (33): fire median = 49.00, ratio = 0.77x, 42.4% above bg median (improvement: -7.6 pp)

### WFPI Day 1 2021

- **Background median:** 63.529
- **All fires** (931): fire median = 48.00, ratio = 0.76x, 33.6% above bg median (improvement: -16.4 pp)
- **Large fires** (33): fire median = 56.00, ratio = 0.88x, 39.4% above bg median (improvement: -10.6 pp)

### WFPI 2021 Burn-at-least-once

- **Background median:** 248.000
- **All fires** (931): fire median = 248.00, ratio = 1.00x, 0.0% above bg median (improvement: -50.0 pp)
- **Large fires** (33): fire median = 248.00, ratio = 1.00x, 0.0% above bg median (improvement: -50.0 pp)

## Key Findings

### 1. Overall winner: Ignition Probability (Pyrologix)

Pyrologix places 71.0% of fires above the background median — an improvement of +21.0 pp over the
50% baseline. This is consistent with the 2020 result (+21.4 pp in doc 09) and confirms that
Pyrologix captures long-term ignition risk very well.

Pyrologix was trained on historical fire data from **2006–2020 only**. The 2021 USFS dataset
contains exclusively 2021 fires, so **this evaluation is fully out-of-sample — no data leakage**.
Pyrologix is the **recommended burn map for the California 2021 dataset**.

### 2. Runner-up: Burn Probability (FSim/BP)

Burn Probability achieves +9.3 pp improvement for all fires and +10.6 pp for large fires.
Its ratio for large fires (1.51x) indicates large fires occur in areas with significantly
above-average burn probability. FSim BP is a fire-simulation model (not trained on occurrence
data), so it is also leakage-free for 2021.

### 3. WFPI time-varying maps are anti-correlated with 2021 fires

**Annual-background method:** All three WFPI variants (Yearly −15.4 pp, D1 −16.4 pp, D2 −15.5 pp)
show **negative improvements**: fires occur in areas of *below-average* WFPI. This is the opposite of the 2020 result (+17–18 pp).

**Per-day-background method (corrected):** Even when each fire is compared to the background median
of its *own* day, the anti-correlation persists — Yearly −13.4 pp, D1 −13.3 pp, D2 −14.8 pp.
The improvement vs the annual-bg method is small (2–3 pp), confirming that **the anti-correlation
is not a seasonal artifact** from mixing winter and summer days. Fires genuinely occur in areas of
below-median WFPI even relative to their contemporaneous spatial distribution.

Possible explanations:

- **2021 was a severe drought year.** Many large fires (Dixie, Caldor, Monument) burned in
  northern California forests where WFPI (wind-enhanced) tends to be lower than the
  statewide average. Drought-driven fires in low-wind forest environments are poorly
  captured by a wind-centric index.
- **USFS dataset composition.** The USFS ignition points include all 932 California
  wildfires, many of which are small (size class A–C). In 2020 only FPA-FOD records with
  a recorded FOD_ID and discovery time were used (~2,219 fires). The USFS small fires may
  be concentrated in lower-WFPI areas (e.g. residential interface, roadsides), dragging
  the distribution below the daily WFPI median.
- **Fundamental limit of WFPI as a location predictor.** WFPI is a *danger* index: high values
  mean high fire propagation potential *if ignited*. It is not trained to predict *where*
  ignitions occur. Ignitions are driven by human activity, lightning, and fuel availability
  — which can be spatially anti-correlated with wind-driven danger zones.

### 4. WFPI Averaged map is near-random

The year-averaged WFPI (improvement -3.5 pp all fires; +1.5 pp large fires)
is nearly indistinguishable from a random placement. This suggests that the mean WFPI over
the year does not add information beyond what the mask already captures — the valid cells
have similar mean WFPI regardless of whether they are fire-prone.

### 5. Burn-at-least-once map is unusable

The background median is saturated at 248.0 (the maximum value). Over 365 days of WFPI,
virtually every valid California cell reaches a WFPI < 249 at least once, giving all
cells a burn-at-least-once probability ≈ 1 remapped to 248. The map provides no
discriminative power and should not be used as a risk map.

### 6. WFPI Yearly vs D1 vs D2 (time-aware selection adds marginal value)

Annual-bg method: Yearly −15.4 pp, D1 −16.4 pp, D2 −15.5 pp.
Per-day-bg method: Yearly −13.4 pp, D1 −13.3 pp, D2 −14.8 pp.

In both methodologies, Yearly and D1 are nearly identical and marginally outperform D2 —
consistent with the 2020 pattern where time-aware selection gave a small edge. The gap
between D1 and D2 is narrow (1–2 pp), so the choice of forecast is not the driver of
the anti-correlation signal.

### 7. Large-fire pattern

Large fires (≥ 100 acres, n = 33) show stronger positive correlation with
Pyrologix (+13.6 pp) and Burn Probability (+10.6 pp) than all fires.
The WFPI maps remain anti-correlated even for large fires, though less severely.
This suggests that large fires in 2021 occurred in areas of historically high fire
probability but not necessarily high wind-driven fire danger on the specific day.

## Comparison with 2020 Results

All 2020 results use improvement vs background (pp above 50% baseline).

| Map | Improvement (All, 2020) | Improvement (All, 2021 annual bg) | Improvement (All, 2021 per-day bg) | Improvement (Large, 2020) | Improvement (Large, 2021 per-day bg) |
|-----|------------------------|----------------------------------|-----------------------------------|--------------------------|---------------------------------------|
| WFPI Yearly (time-aware) | +18.1 pp | -15.4 pp | -13.4 pp | +31.8 pp | -10.6 pp |
| WFPI Day 1 | +18.0 pp | -16.4 pp | -13.3 pp | +29.5 pp | -10.6 pp |
| WFPI Day 2 | +17.2 pp | -15.5 pp | -14.8 pp | +34.1 pp | -13.6 pp |
| Ignition Probability (Pyrologix) | +21.4 pp | +21.0 pp | N/A (static) | +27.6 pp | N/A (static) |
| Burn Probability (FSim) | +5.8 pp | +9.3 pp | N/A (static) | +22.3 pp | N/A (static) |

The most striking change from 2020 to 2021 is the **reversal of the WFPI signal**:
strongly positive in 2020 (+17–18 pp) but negative in 2021 even with the per-day
background correction (−13 to −15 pp). Pyrologix and Burn Probability maintain a
positive signal in both years, making them more robust across dataset compositions
and fire regimes.

## Decision: Recommended Burn Map for California 2021 Dataset

**→ Use Ignition Probability (Pyrologix) for both sensor/charging placement AND drone routing.**

Pyrologix is the strongest predictor of 2021 fire locations (+21.0 pp all fires, +13.6 pp large
fires), is stable relative to its 2020 performance (+21.4 pp), and has **no data leakage** for
2021 (trained on 2006–2020 data only, evaluated here on entirely held-out 2021 fires).

The benchmark script (`run_benchmark_california2021_yearly.py`) uses Pyrologix for both steps:
- **Sensor/charging placement** — pooled to operational scale, tiled to `operational_substeps` frames.
- **Drone routing** — single-frame `(1, N, M)` passed with `burnmap_type="static"` so the routing
  strategy tiles it 200× internally; one routing cache per cluster (`log_key="pyrologix"`), valid for
  any scenario regardless of offset (log computed for the full `MAX_ROUTING_DATA_STEPS × substeps` length).

The WFPI Yearly map is **not loaded at benchmark runtime** — fire spread is pre-computed into the
per-scenario `.npy` files during dataset creation (`create_california_2021_dataset.py`).

## Full Map Ranking

| Priority | Map | Signal (all fires) | Signal (large fires) | Leakage | Notes |
|----------|----|---------------------|----------------------|---------|-------|
| **1st ✓** | **Ignition Probability (Pyrologix)** | **+21.0 pp** | **+13.6 pp** | None | Recommended; sensor placement + drone routing; `California2021Dataset/static_risk_pyrologix.npy` |
| 2nd | Burn Probability (FSim/BP) | +9.3 pp | +10.6 pp | None | Good runner-up |
| 3rd | WFPI Yearly 2021 (time-aware) | −13.4 pp (per-day) | −10.6 pp (per-day) | None | Pre-computed into scenario files; not loaded at benchmark runtime |
| 4th | WFPI Day 1 2021 | −13.3 pp | −10.6 pp | None | Fallback if time-aware map unavailable |
| 5th | WFPI Day 2 2021 | −14.8 pp | −13.6 pp | None | Marginally worse than D1 |
| 6th | WFPI 2021 Averaged | −3.5 pp | +1.5 pp | None | Near-random; use only for visualisation |
| — | WFPI 2021 Burn-at-least-once | −50.0 pp | −50.0 pp | None | Saturated; not useful |

## Files

| File | Location | Description |
|------|----------|-------------|
| `analyze_2021_risk_maps.py` | `code/dataset_creation/nature_dataset_creation/` | This analysis script |
| `run_benchmark_california2021_yearly.py` | project root | Benchmark entry point — uses Pyrologix for placement + routing |
| `static_risk_pyrologix.npy` | `California2021Dataset/` | **Pyrologix copy (self-contained)** — used for sensor placement and drone routing |
| `static_risk_wfpi_yearly.npy` | `California2021Dataset/` | 2021 yearly burn map (730, H, W) — used only during dataset creation (pre-computed into scenarios) |
| `static_risk_wfpi_avg.npy` | `California2021Dataset/` | 2021 WFPI average (values ≥249 excluded) |
| `static_risk_wfpi_burn_at_least_once.npy` | `California2021Dataset/` | P(burn ≥1 time), rescaled 0–248 |
| `static_risk_pyrologix_resampled.npy` | `California2020Dataset/` | Pyrologix original source (resampled 1309×805) |
| `static_risk_burn_prob_resampled.npy` | `California2020Dataset/` | FSim burn prob (resampled 1309×805) |
| `wfpi_YYYYMMDD.npy` | `California2021Dataset/` | 2021 D2 daily maps (365 files) |
| `wfpi_day1_YYYYMMDD.npy` | `California2021Dataset_Day1/` | 2021 D1 daily maps (365 files) |

## References

- [09_risk_map_comparison.md](09_risk_map_comparison.md) — 2020 baseline (static maps)
- [12_yearly_wfpi_map_comparison.md](12_yearly_wfpi_map_comparison.md) — 2020 WFPI yearly map
- [14_usfs_california_dataset_creation.md](14_usfs_california_dataset_creation.md) — 2021 dataset pipeline
- Pyrologix DOI: 10.17605/OSF.IO/CFGH9
- FSim/BP DOI: 10.2737/RDS-2016-0034-2
