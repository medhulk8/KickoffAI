# KickoffAI — CLAUDE.md

## Current Status
**Phase: ML model trained and ablated. Champion model identified. Next: wire into app.**
LR model beats bookmakers (65% vs 57.7%, 63% draw recall vs 0%). Feature set locked.

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with probabilities.
Goal: replace LLM predictor node in LangGraph with trained ML model. Keep everything else.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- LangGraph workflow (`src/workflows/prediction_workflow.py`)
- ML: `src/ml/` — build_dataset.py, backtest.py, ablations.py
- Data: 3 seasons PL 2021–24, 941 training rows, source: football-data.co.uk

## Locked Architecture Decisions
- Replace only the LLM predictor node in LangGraph — keep stats collector, draw detector, confidence calculator, logger, Streamlit intact
- Champion model: **Logistic Regression** (LightGBM lost on all metrics)
- Strict time-based splits only — no shuffling
- Equal season weighting; recency decay as later ablation
- Two-stage model is a future challenger, not current priority
- Confidence threshold: **0.65** (76.8% accuracy on 58% of matches in Fold 2)

## Champion Feature Set (8 features — locked after ablations)
- bm_home_prob, bm_draw_prob, bm_away_prob
- h2h_draw_rate
- home_weighted_ppg, away_weighted_ppg
- home_weighted_goals, away_weighted_goals

**Dropped:** draw_likelihood (noisy/redundant), def_solidity (zero contribution)

## Champion Model Results (Fold 2, n=194)
- Accuracy: 65.0% (vs 57.7% bookmaker baseline)
- Draw Recall: 63% (vs 0% bookmaker)
- Log Loss: 0.748 (vs 0.932 bookmaker)
- Brier: 0.151 (vs 0.183 bookmaker)
- High-conf accuracy @ 0.65: 76.8% on 58% of matches

## Next Session Priority
Validate end-to-end app path on one known historical match:
- Run the Streamlit app or workflow on a 2023-24 match and confirm ML prediction fires correctly
- Check logs for feature values and output probs
- Then: retrain on all 3 seasons once integration is confirmed stable (`src/ml/train_final.py` — change TRAIN_SEASONS to include "2324")

## Collaboration Workflow
At every major implementation step, frame a prompt for ChatGPT asking for clarifications or review. Claude implements, GPT reviews/advises, user relays responses.
**Always end GPT prompts with "Answer compactly." — user has limited tokens.**

## Session Log
**2026-04-14** — Planned full ML replacement of LLM predictor. Agreed plan with GPT + Claude review. Decisions locked. Plan saved to `ML_PLAN.md`.
**2026-04-14** — Leakage audit passed. Built dataset builder. 941 rows, 11 features. Pushed.
**2026-04-14** — Built backtesting framework. LR: 65% acc, 60% draw recall, LL=0.745. LightGBM: 59.8% — loses on all metrics, dropped. Confidence sweep: 0.65 threshold chosen.
**2026-04-14** — Feature ablations complete. h2h_draw_rate is critical. draw_likelihood and def_solidity dropped. Champion: 8 features (bm probs + h2h_draw_rate + weighted form). Results stable across both folds.
**2026-04-14** — Trained champion model on 2122+2223, holdout 2324: 65% acc, 63% draw recall, LL=0.748. Saved to models/lr_champion.pkl + model_metadata.json. Built MLPredictor inference wrapper with validation. Wired ml_predictor_node into LangGraph (llm_predictor kept but unwired). Smoke test passed.
