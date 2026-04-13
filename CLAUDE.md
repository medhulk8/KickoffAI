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
**Start here:** Audit feature engineering for leakage.
- Check `before_date` param is used correctly in `advanced_stats.py`, `weighted_stats.py`, draw detector, H2H queries
- Confirm `date < ?` (strict less-than, not `<=`)
- Confirm predicted match excluded from all aggregates
Then: build supervised training table (one row per match, point-in-time features, label H/D/A).
Full plan in `ML_PLAN.md`.

## Session Log
**2026-04-14** — Planned full ML replacement of LLM predictor. Agreed plan with GPT + Claude review. Decisions locked. Plan saved to `ML_PLAN.md`. No code written yet.
