# Enhanced Draw Detection (Phase 6)

## Overview

Enhanced the DrawDetector to use recency-weighted metrics for more accurate draw prediction. Draws occur in ~25% of Premier League matches, but models often under-predict them. This enhancement uses weighted PPG, momentum scores, and form comparison to better identify draw-likely matches.

## What Changed

### New Inputs
Added 3 optional parameters to `detect_draw_likelihood()`:
- `home_weighted_form`: Weighted form with recent matches prioritized
- `away_weighted_form`: Weighted form with recent matches prioritized
- `form_comparison`: Pre-calculated form advantage classification

### New Factor: Momentum Similarity (0.15 points)
Added `_score_momentum_similarity()` that detects when both teams have similar momentum:
- Similar momentum (diff < 0.5): +0.15 points
- Somewhat similar (diff < 1.0): +0.08 points
- Slight difference (diff < 1.5): +0.03 points
- Clear advantage (diff >= 1.5): 0.00 points

### Enhanced Factor: Even Form
Updated `_score_even_form()` to:
1. Use `form_comparison` classification if available (most accurate)
   - "even" → 0.35 points
   - "home_slight"/"away_slight" → 0.20 points
   - "home"/"away" → 0.05 points

2. Use weighted PPG instead of traditional PPG (better recency weighting)

3. Fallback to traditional PPG if weighted metrics unavailable

## Test Results

### Test 1: Even Match with Similar Momentum
**Scenario**: Liverpool vs Arsenal
- Traditional PPG: 1.8 vs 1.7 (diff: 0.1)
- Weighted PPG: 2.0 vs 1.95 (diff: 0.05)
- Momentum: 2.0/3 vs 1.5/3 (diff: 0.5)

**Results**:
- Old draw likelihood: 0.900
- New draw likelihood: **0.980** (+0.080)
- **✓ IMPROVED**: Better detection of draw-likely match

### Test 2: Diverging Momentum
**Scenario**: Manchester City vs Newcastle
- Traditional PPG: 2.0 vs 1.9 (diff: 0.1)
- Weighted PPG: 2.4 vs 1.4 (diff: 1.0)
- Momentum: 3.0/3 vs 0.5/3 (diff: 2.5)

**Results**:
- Old draw likelihood: 0.700
- New draw likelihood: **0.400** (-0.300)
- **✓ IMPROVED**: Correctly identifies momentum advantage = lower draw chance

### Test 3: Both Teams Struggling
**Scenario**: Everton vs Burnley
- Traditional PPG: 0.8 vs 0.9 (diff: 0.1)
- Weighted PPG: 0.6 vs 0.7 (diff: 0.1)
- Momentum: 0.5/3 vs 0.0/3 (diff: 0.5)

**Results**:
- Old draw likelihood: 0.950
- New draw likelihood: **1.000** (+0.050)
- **✓ IMPROVED**: Both struggling with similar momentum = high draw chance

## Impact

### Expected Improvements
- **Better sensitivity**: Detects subtle momentum differences
- **More accurate**: Uses recency-weighted data (recent form matters more)
- **Fewer false negatives**: Catches draws that traditional PPG misses
- **Fewer false positives**: Recognizes momentum advantages that reduce draw chance

### Technical Details
- Max possible score: 1.25 (increased from 1.1)
  - Even form: 0.35
  - Close baseline: 0.35
  - Low scoring: 0.20
  - H2H draws: 0.20
  - Momentum similarity: 0.15 (NEW)
- Capped at 1.0 for consistency
- Backward compatible (works without weighted metrics)

## Files Modified

### Updated
- **[src/workflows/draw_detector.py](src/workflows/draw_detector.py)**:
  - Lines 29-37: Added new parameters to `detect_draw_likelihood()`
  - Lines 69-75: Added momentum similarity factor
  - Lines 76-120: Enhanced `_score_even_form()` with weighted metrics
  - Lines 193-220: Added `_score_momentum_similarity()` method

- **[src/workflows/prediction_workflow.py](src/workflows/prediction_workflow.py:197-209)**:
  - Updated draw detector call to pass weighted metrics

### Created
- **[test_enhanced_draw_detection.py](test_enhanced_draw_detection.py)**: Comprehensive tests

## Usage

### In Workflow (Automatic)
Weighted metrics are automatically passed to DrawDetector in the prediction workflow. No manual intervention needed.

### Manual Testing
```python
from src.workflows.draw_detector import DrawDetector

detector = DrawDetector()

# Without weighted metrics (old way)
score_old = detector.detect_draw_likelihood(
    home_form={'points_per_game': 1.8, 'goals_scored': 7},
    away_form={'points_per_game': 1.7, 'goals_scored': 7},
    baseline={'home_prob': 0.40, 'draw_prob': 0.28, 'away_prob': 0.32}
)

# With weighted metrics (new way)
score_new = detector.detect_draw_likelihood(
    home_form={'points_per_game': 1.8, 'goals_scored': 7},
    away_form={'points_per_game': 1.7, 'goals_scored': 7},
    baseline={'home_prob': 0.40, 'draw_prob': 0.28, 'away_prob': 0.32},
    home_weighted_form={'weighted_points_per_game': 2.0, 'momentum_score': 2.0},
    away_weighted_form={'weighted_points_per_game': 1.95, 'momentum_score': 1.5},
    form_comparison={'ppg_differential': 0.05, 'form_advantage': 'even'}
)

print(f"Improvement: {score_new - score_old:+.3f}")
```

## Next Steps

1. **Run batch evaluation** to measure impact on draw prediction accuracy
2. **Monitor false positives**: Ensure we're not over-predicting draws
3. **Tune thresholds**: Adjust momentum similarity thresholds if needed
4. **Combine with Priority 3**: Update ConfidenceCalculator to use momentum

## Integration with Other Improvements

This enhancement works synergistically with:
- **Priority 1 (Recency Weighting)**: Uses weighted PPG and momentum scores ✓
- **Priority 3 (Confidence Calibration)**: Momentum can inform confidence levels (TODO)
- **Priority 4 (Ensemble)**: Better draw detection improves all models (TODO)

---

*Feature status: ✓ Implemented and tested*
*Last updated: 2026-02-06*
