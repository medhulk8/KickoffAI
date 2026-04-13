# KickoffAI — CLAUDE.md

## Current Status
**Phase: PRODUCTION MODEL DEPLOYED. ML fully replaces LLM in LangGraph workflow.**
LR model trained on all 3 seasons (1125 matches, full 2023-24 fixed). Validated end-to-end. Ready for live predictions.

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with probabilities.
Goal: replace LLM predictor node in LangGraph with trained ML model. Keep everything else.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- LangGraph workflow (`src/workflows/prediction_workflow.py`)
- ML: `src/ml/` — build_dataset.py, backtest.py, ablations.py
- Data: 3 seasons PL 2021–24, 1125 training rows, source: football-data.co.uk

## Locked Architecture Decisions
- Replace only the LLM predictor node in LangGraph — keep stats collector, draw detector, confidence calculator, logger, Streamlit intact
- Champion model: **Logistic Regression** (LightGBM lost on all metrics)
- Strict time-based splits only — no shuffling
- Equal season weighting; recency decay as later ablation
- Two-stage model is a future challenger, not current priority
- Confidence threshold: **0.65** (79.9% accuracy on 51% of matches in Fold 2)

## Champion Feature Set (8 features — locked after ablations)
- bm_home_prob, bm_draw_prob, bm_away_prob
- h2h_draw_rate
- home_weighted_ppg, away_weighted_ppg
- home_weighted_goals, away_weighted_goals

**Dropped:** draw_likelihood (noisy/redundant), def_solidity (zero contribution)

## Champion Model Results (Fold 2, n=378 — full 2023-24)
- Accuracy: 65.9% (vs 59.0% bookmaker baseline)
- Draw Recall: 56.1% (vs 0% bookmaker)
- Log Loss: 0.7737 (vs 0.9112 bookmaker)
- Brier: 0.1534 (vs 0.1782 bookmaker)
- High-conf accuracy @ 0.65: 79.9% on 51% of matches

## Next Session Priority
When season 4 data is available:
1. Download new CSV from football-data.co.uk, run `src/data/load_data.py`
2. Rebuild dataset: `src/ml/build_dataset.py`
3. Rerun backtests: `src/ml/backtest.py` (add new fold)
4. Retrain from scratch: `src/ml/train_final.py` (all seasons)
5. Sanity check: schema, parity, distribution checks

Otherwise next focus: test the Streamlit app with live match predictions.

## Collaboration Workflow
At every major implementation step, frame a prompt for ChatGPT asking for clarifications or review. Claude implements, GPT reviews/advises, user relays responses.
**Always end GPT prompts with "Answer compactly." — user has limited tokens.**

## Session Log
**2026-04-14** — Planned full ML replacement of LLM predictor. Agreed plan with GPT + Claude review. Decisions locked. Plan saved to `ML_PLAN.md`.
**2026-04-14** — Leakage audit passed. Built dataset builder. 941 rows, 11 features. Pushed.
**2026-04-14** — Built backtesting framework. LR: 65% acc, 60% draw recall, LL=0.745. LightGBM: 59.8% — loses on all metrics, dropped. Confidence sweep: 0.65 threshold chosen.
**2026-04-14** — Feature ablations complete. h2h_draw_rate is critical. draw_likelihood and def_solidity dropped. Champion: 8 features (bm probs + h2h_draw_rate + weighted form). Results stable across both folds.
**2026-04-14** — Trained champion model on 2122+2223, holdout 2324: 65% acc, 63% draw recall, LL=0.748. Saved to models/lr_champion.pkl + model_metadata.json. Built MLPredictor inference wrapper with validation. Wired ml_predictor_node into LangGraph (llm_predictor kept but unwired). Smoke test passed.
**2026-04-14** — Fixed h2h_draw_rate with Laplace smoothing (α=1), capped max output prob at 92%. End-to-end validation passed on 3 known matches. Retrained production model on all 3 seasons (941 matches). Schema/behavior/validation checks all pass.
**2026-04-14** — Fixed data bug: 2023-24 DB had 184/380 matches missing odds (loaded from partial CSV). Patched asil.db using AvgH/D/A from full E0_2324.csv. Rebuilt dataset (1125 rows, +184). Retrained. Fold 2 metrics: 65.9% acc, LL=0.7737, Brier=0.1534, high-conf 0.65→79.9% acc.
