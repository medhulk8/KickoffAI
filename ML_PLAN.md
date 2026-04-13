# KickoffAI — ML Model Implementation Plan

## Goal
Replace the LLM predictor node in the LangGraph workflow with a trained ML model.
Keep everything else (stats collector, draw detector, confidence calculator, logger, Streamlit app) intact.

---

## Core Decisions

- **Model family**: Start with multinomial logistic regression, then LightGBM as challenger
- **No neural nets** — dataset too small
- **Two-stage model** (Draw vs Not-Draw → Home vs Away) is a challenger, not the default
- **Season weighting**: Start equal, test recency decay later as an ablation
- **Draw detector**: Keep as rule-based, use draw-likelihood score as a feature — do not replace with learned classifier yet
- **H2H**: Keep H2H draw rate only, exclude broader win/loss history
- **Calibration**: Test temperature scaling after base models are working, only keep if it improves out-of-time metrics

---

## Initial Feature Set (10–12 features, expand later via ablations)

1. Bookmaker home probability
2. Bookmaker draw probability
3. Bookmaker away probability
4. Weighted PPG — home team
5. Weighted PPG — away team
6. Weighted goals per game — home team
7. Weighted goals per game — away team
8. Home defensive solidity
9. Away defensive solidity
10. Draw-likelihood score
11. H2H draw rate

---

## LightGBM Starting Settings (conservative, anti-overfit)

- `max_depth`: 3
- `num_leaves`: 7–15
- `min_child_samples`: 40–80
- `learning_rate`: 0.02–0.05
- `feature_fraction`: 0.7–0.9
- `bagging_fraction`: 0.7–0.9
- `bagging_freq`: 1
- Early stopping on validation log loss

---

## Evaluation Metrics

- Accuracy (overall + per class)
- Log loss
- Brier score
- Per-class precision / recall / F1 (especially Draw)
- Draw recall specifically
- Calibration diagnostics
- High-confidence accuracy

---

## Validation Strategy

- Strict time-based splits only — no random shuffling
- Expanding-window or rolling-window cross-validation
- Final holdout = most recent unseen block (last season or last chunk)
- All features must be computed using only data strictly before match date

---

## Implementation Order

1. **Audit feature engineering for leakage**
   - Verify `before_date` param is used correctly everywhere
   - Confirm predicted match is excluded from all aggregates
   - Check: advanced_stats.py, weighted_stats.py, draw_detector.py, h2h queries

2. **Build supervised training table**
   - One row per match
   - Features computed point-in-time (strictly before match date)
   - Label = H / D / A
   - Reproducible dataset builder script

3. **Set up time-based backtesting framework**
   - Rolling/expanding splits
   - Metrics function covering all metrics above
   - One untouched final holdout

4. **Implement baselines**
   - Bookmaker-only baseline
   - Existing rule-based draw detector system

5. **Train multinomial logistic regression**
   - Core 10–12 features only
   - Regularized (C tuned via CV)
   - First true ML benchmark

6. **Expand to more seasons** (target 8–10 total)
   - Only after pipeline validated and first model running cleanly
   - Download from football-data.co.uk (same format as existing CSVs)

7. **Train LightGBM challenger**
   - Same feature set as logistic regression first
   - Conservative settings above

8. **Feature ablations**
   - With/without bookmaker odds
   - With/without draw score
   - With/without H2H draw rate
   - With/without tactical KG features
   - Add features only if ablations justify them

9. **Test light draw post-processing**
   - Small draw floor in high-draw-score contexts
   - Threshold tuning for Draw prediction
   - Only after base model behavior understood

10. **Test two-stage challenger**
    - Stage 1: Draw vs Not-Draw
    - Stage 2: Home vs Away given Not-Draw
    - Combine: P(home) = (1 - P(draw)) * P(home|not draw)
    - Judge only by backtest evidence

11. **Test season-level sample weighting**
    - Current season = 1.0, previous = 0.8–0.9, older taper gradually
    - Ablation only, not the default

12. **Wire winning model into existing app**
    - Remove LLM predictor node from LangGraph workflow
    - Replace with trained model inference node
    - Keep: stats collector, draw detector, confidence calculator, logger, Streamlit

---

## Leakage Audit Checklist (Step 1)

- [ ] `advanced_stats.py` — `before_date` param passed and uses `date < ?` (not `<=`)
- [ ] `weighted_stats.py` — same check
- [ ] `draw_detector.py` — check all stat calls pass `before_date`
- [ ] H2H queries — same check
- [ ] Training table builder — verify date filter applied before every feature call
- [ ] No features derived from match result or post-match stats

---

## Realistic Ceiling

Bookmaker odds are the strongest signal. The model will likely function mostly as a calibrated correction layer on top of market probabilities. Expected gains:
- Similar or slightly better overall accuracy (~56–60%)
- Better draw handling (currently 33.3% with LLM, target higher)
- Better calibrated probabilities (lower Brier / log loss)
- Better high-confidence accuracy
- Deterministic, fast, reproducible

Do not expect large raw accuracy jumps over bookmakers.
