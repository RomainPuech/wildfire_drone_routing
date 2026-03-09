# Test Scenarios Explanation

## Kernel Coverage Analysis

Based on the kernel coverage analysis, we found that:

- **Distance 0 (center)**: 100% coverage with 1 drone (already capped)
- **Distance 1**: 85-92% coverage with 1 drone → **capped at 100% with 2 drones**
- **Distance 2**: 67-78% coverage with 1 drone → **capped at 100% with 2 drones**
- **Distance 3**: 50-77% coverage with 1 drone → can benefit from 2-3 drones
- **Distance 4**: 33-64% coverage with 1 drone → can benefit from 2-4 drones
- **Distance 5+**: Lower coverage → can benefit from multiple drones

## Key Insight

**Concentration doesn't help for nearby cells (distance 0-2)** because they're already well-covered or quickly capped. **Concentration helps for distance 3+ cells** where multiple drones can meaningfully improve coverage.

## Original Test Scenarios

These scenarios may not show concentration because high-risk areas are too close to stations:

1. **Single Hotspot**: Small 3x3 hotspot in corner - nearby cells already well-covered
2. **Two Distant Hotspots**: Two 4x4 hotspots - but cells at distance 0-2 are already capped
3. **Large Center Hotspot**: 11x11 area - but if station is at center, distance 0-2 cells are capped

## New Test Scenarios (Designed with Kernel Behavior in Mind)

### Scenario 4: Large Extended Area
- **Purpose**: Large 12x12 high-risk area where cells at distance 3-5 from station center are high-risk
- **Why it helps**: Station placed at edge of area will have high-risk cells at distance 3-5 that benefit from multiple drones
- **Expected**: Drones should concentrate to cover distance 3-5 cells better

### Scenario 5: High-Risk Ring at Distance 3-4
- **Purpose**: High-risk cells are **specifically** at distance 3-4 from center
- **Why it helps**: These cells have 50-60% coverage with 1 drone, can reach 100% with 2-3 drones
- **Expected**: Strong concentration to maximize coverage of the ring

### Scenario 6: Gradient Peak at Distance 3-4
- **Purpose**: Risk increases with distance, peaking at distance 3-4
- **Why it helps**: Medium-distance cells (distance 3-4) have highest risk and benefit from multiple drones
- **Expected**: Drones concentrate to cover the peak risk area

### Scenario 7: Two Large Overlapping Areas
- **Purpose**: Two large overlapping high-risk areas creating a very large region
- **Why it helps**: A single station can't cover all cells well; multiple drones help cover distance 3+ cells
- **Expected**: Drones concentrate at stations covering the overlap area

### Scenario 8: High-Risk Corridor
- **Purpose**: Long, narrow diagonal corridor of high-risk cells
- **Why it helps**: Stations placed along corridor will have high-risk cells at various distances, including distance 3+
- **Expected**: Drones concentrate along the corridor

### Scenarios with Larger Kernel (size 10)
- **Purpose**: Test with kernel size 10 where distance 3+ cells have more room for improvement
- **Why it helps**: Larger kernel means distance 3-4 cells have lower base coverage, so multiple drones provide more benefit
- **Expected**: Better concentration behavior with larger kernel

## What We're Testing

1. **Does concentration occur?** (max_allocation > 1)
2. **Does it improve objective?** (allocation objective > original objective)
3. **Are drones placed at high-risk areas?** (stations in high-risk areas get more drones)
4. **Do distance 3+ cells get better coverage?** (cells that benefit from multiple drones)

## Expected Results

- **Original scenarios**: May not show concentration (distance 0-2 cells already capped)
- **New scenarios**: Should show concentration, especially:
  - Scenario 5 (High-Risk Ring) - strongest expected concentration
  - Scenario 6 (Gradient Peak) - should show concentration
  - Scenarios with kernel size 10 - better concentration than kernel size 6-8

## Interpretation

If concentration still doesn't occur in the new scenarios, it suggests:
1. The formulation may need adjustment (e.g., different objective weighting)
2. The solver may need better parameters (time limit, optimality gap, MIP focus)
3. The problem structure may inherently favor spreading (mathematically optimal but not desired)

If concentration occurs in new scenarios but not original:
- This confirms the kernel coverage analysis
- The formulation is working correctly
- We need to adjust expectations or modify the kernel/coverage model
