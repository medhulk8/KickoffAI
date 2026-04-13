# KickoffAI — CLAUDE.md

## Current Status
**Phase: Pre-implementation — ML model planned, not yet built.**
LLM-based prediction system is working. Planning phase complete. Ready to build trained ML model to replace LLM.

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with probabilities.
Currently uses LangGraph workflow + local LLM (llama3.1:8b via Ollama).
Goal: replace LLM predictor node with a trained ML model. Keep everything else.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- LangGraph workflow (`src/workflows/prediction_workflow.py`)
- Data: 3 seasons PL 2021–24, ~1140 matches, source: football-data.co.uk

## Locked Architecture Decisions
- Replace only the LLM predictor node in LangGraph — keep stats collector, draw detector, confidence calculator, logger, Streamlit intact
- Start with logistic regression, then LightGBM — no neural nets
- Keep rule-based draw detector as-is; use draw-likelihood score as a feature
- Strict time-based splits only — no shuffling
- Start with 3 seasons; expand to 8–10 only after pipeline validated
- Equal season weighting to start; recency decay as later ablation
- Two-stage model (Draw vs Not-Draw → Home vs Away) is challenger only

## Initial Feature Set (10–12)
bookmaker H/D/A probs, weighted PPG home/away, weighted goals/game home/away, home/away defensive solidity, draw-likelihood score, H2H draw rate

## Next Session Priority
Build backtesting framework (`src/ml/backtest.py`) — time-based splits, metrics (accuracy, log loss, Brier, per-class F1, draw recall). Then train logistic regression baseline.
Awaiting GPT answers on: class imbalance strategy, split design (2+1 seasons vs rolling), draw_likelihood feature skew concern, defensive solidity variance concern.

## Collaboration Workflow
At every major implementation step, a prompt is framed for ChatGPT asking for clarifications or review before proceeding. Claude implements, GPT reviews/advises, both work together. User relays responses between them.

## Session Log
**2026-04-14** — Planned full ML replacement of LLM predictor. Agreed plan with GPT + Claude review. Decisions locked. Plan saved to `ML_PLAN.md`. No code written yet.
**2026-04-14** — Leakage audit passed (all stat calls use strict `date < ?`; MCP get_team_form bypassed). Built `src/ml/build_dataset.py`. Generated `data/processed/training_dataset.csv` (941 rows, 11 features + label). Distribution: H=432, D=209, A=300. Pushed to GitHub.
