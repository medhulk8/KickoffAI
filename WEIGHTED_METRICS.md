# Recency Weighting Feature

## Overview

The recency weighting feature improves prediction accuracy by giving more importance to recent matches when calculating team form. Older matches are exponentially discounted, ensuring that a team's current form matters more than their performance weeks ago.

## Implementation

### Formula

```
weight = e^(-λ * days_ago)
```

Where:
- `λ` (lambda) = decay rate (default: 0.05)
- `days_ago` = number of days since the match
- `e` = Euler's number (2.718...)

### Decay Rate Options

- `λ = 0.05` → ~14 day half-life (default)
- `λ = 0.10` → ~7 day half-life (faster decay)

## New Metrics

### 1. Weighted Points Per Game (PPG)
Traditional PPG treats all matches equally. Weighted PPG gives more weight to recent matches.

**Example:**
```
Liverpool last 5 matches: D-W-D-L-W
Traditional PPG: 1.6
Weighted PPG: 1.95 (recent wins weighted more heavily)
```

### 2. Momentum Score (0-3 scale)
Compares recent 2 matches vs next 3 matches to identify form trends.

- **3.0** = Won both recent matches (strong momentum)
- **2.0** = Won 1, drew 1 (positive momentum)
- **1.0** = Drew both or mixed results
- **0.0** = Lost both recent matches (negative momentum)

### 3. Form Comparison
Classifies overall form advantage between teams:

- **even**: PPG difference < 0.3
- **home_slight** / **away_slight**: PPG difference 0.3-0.8
- **home** / **away**: PPG difference > 0.8

## Files Modified

### Created
- **[src/data/weighted_stats.py](src/data/weighted_stats.py)**: Core implementation

### Modified
- **[src/workflows/prediction_workflow.py](src/workflows/prediction_workflow.py)**:
  - Lines 91-98: Added `PredictionState` fields
  - Lines 609-627: Calculate weighted form in `stats_collector_node`
  - Lines 215-245: Updated prompt with weighted metrics
- **[src/evaluation/batch_evaluator.py](src/evaluation/batch_evaluator.py)**:
  - Lines 147-157: Extract weighted form data for analysis

## Usage

### Testing Manually

```python
from src.data.weighted_stats import WeightedStatsCalculator

calc = WeightedStatsCalculator("data/processed/asil.db", decay_rate=0.05)

# Get weighted form for a team
form = calc.get_weighted_form("Liverpool", last_n=5, before_date="2024-05-19")

print(f"Weighted PPG: {form['weighted_points_per_game']}")
print(f"Momentum: {form['momentum_score']}/3")
print(f"Form string: {form['form_string']}")
```

### In Predictions

Weighted metrics are automatically calculated and included in:
1. The LLM prompt (STEP 1 - Recent Form Analysis)
2. Evaluation results CSV
3. Workflow state for downstream nodes

## Expected Impact

**Goal**: Improve accuracy from 56.7% to 60-65%

**Hypothesis**: Recent matches are more predictive of future performance than older matches, especially in football where:
- Form changes mid-season
- Manager changes affect recent results
- Injuries impact current but not historical form
- Momentum matters (winning/losing streaks)

## Evaluation

Run the evaluation script to measure impact:

```bash
python evaluate_weighted_metrics.py
```

This will:
1. Run predictions on 20 matches with weighted metrics
2. Compare accuracy to previous baseline (56.7%)
3. Report Brier score improvements
4. Export detailed results to CSV

## Next Steps (Phase 2-4)

1. **Enhanced Draw Detection**: Use weighted metrics to better predict draws
2. **Improved Confidence Calibration**: Incorporate momentum into confidence scoring
3. **Ensemble Improvements**: Weight models based on recent accuracy
4. **Additional Features**:
   - Expected Goals (xG) data
   - Player-level statistics
   - Weather conditions
   - Referee statistics

## Technical Notes

### Why Exponential Decay?

Linear decay (`weight = 1 - days_ago / max_days`) is simpler but:
- Treats differences between day 0 and day 7 same as day 21 and day 28
- No smooth gradient

Exponential decay:
- Natural diminishing returns (day 0→7 matters more than day 21→28)
- Smooth, continuous weighting
- Well-established in time-series analysis

### Database Queries

Weighted stats query matches before a specific date to ensure backtesting integrity:

```sql
SELECT date, goals_for, goals_against, result
FROM matches
WHERE (home_team = ? OR away_team = ?)
  AND home_goals IS NOT NULL
  AND date < ?  -- Critical for backtesting!
ORDER BY date DESC
LIMIT ?
```

## Testing

Verified with:
- ✓ Direct WeightedStatsCalculator tests
- ✓ Single match prediction (Liverpool vs Wolverhampton)
- ✓ Weighted data flows through full workflow
- ✓ Data exported to CSV correctly
- ⏳ 20-match batch evaluation (in progress)

## References

- **File**: [src/data/weighted_stats.py](src/data/weighted_stats.py)
- **Test**: [evaluate_weighted_metrics.py](evaluate_weighted_metrics.py)
- **Documentation**: This file

---

*Last updated: 2026-02-06*
*Feature status: ✓ Implemented and tested*
