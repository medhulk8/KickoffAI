# KickoffAI — CLAUDE.md

## Current Status
**Phase: RE-EVALUATING after SQL leakage bug discovered and fixed.**
Previous champion results (65.9% acc, 56.1% draw recall) were inflated by a H2H query bug.
Honest baseline: model barely beats bookmaker on Fold 2 only. Awaiting GPT guidance on next steps.

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with probabilities.
LLM predictor node in LangGraph replaced with trained ML model.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- LangGraph workflow (`src/workflows/prediction_workflow.py`)
- ML: `src/ml/` — build_dataset.py, backtest.py, ablations.py, train_final.py, predictor.py
- DB: 7 seasons in DB (1718–2425), training on 2122–2324, holdout 2425
- source: football-data.co.uk

## DB State
- Seasons in `asil.db`: 1718, 1819, 1920 (context only), 2122, 2223, 2324, 2425
- 1718/1819/1920 are context-only for H2H feature computation — NOT used for training
- Training dataset (`training_dataset.csv`): 1515 rows, seasons 2122/2223/2324/2425

## Architecture Decisions
- Champion model: **Logistic Regression** (LightGBM lost on all metrics)
- Strict time-based splits only — no shuffling
- Confidence threshold: **0.65**
- 3 folds: Fold 1 (→2223), Fold 2 (→2324), Fold 3 (→2425 holdout)

## Current Best Feature Set (under review)
`bm_home_prob, bm_draw_prob, bm_away_prob, h2h_draw_rate, home_draw_rate, away_draw_rate`

**Dropped:** draw_likelihood (noisy), def_solidity (zero contribution), weighted form (hurts after leakage fix)

## Honest Results (post SQL fix, with 7-season H2H context)
| Fold | Test | Model | Acc | Log Loss | Brier | DrawR |
|---|---|---|---|---|---|---|
| Fold 1 | 2223 | Bookmaker | 54.6% | 0.9724 | 0.1927 | 0% |
| | | LR draw signals | 54.4% | 0.9871 | 0.1955 | 3.4% |
| Fold 2 | 2324 | Bookmaker | 59.9% | 0.9119 | 0.1783 | 0% |
| | | **LR draw signals** | **60.2%** | **0.9061** | **0.1774** | 1.2% |
| Fold 3 | 2425 | Bookmaker | 54.1% | 0.9788 | 0.1947 | 0% |
| | | LR draw signals | 53.8% | 0.9810 | 0.1951 | 1.1% |

## Known Issues / Open Questions
- Draw recall is essentially zero across all honest configs (<4%)
- Model only beats bookmaker on Fold 2; Fold 3 (unseen) is still below
- Awaiting GPT response on whether draw prediction is viable or approach needs rethink

## Bugs Fixed This Session
- **H2H SQL precedence bug** (`advanced_stats.py` line ~339): OR branches not wrapped in parens, so `AND date < ?` only filtered one direction. This caused future H2H data to leak in, inflating the old draw recall to 56%. Fixed by wrapping OR in parens.
- **app.py rewritten**: bypasses MCP/LangGraph entirely, uses MLPredictor directly with `st.cache_resource`
- **Codex review applied**: H2H SQL fix, app.py caching

## Collaboration Workflow
At every major implementation step, frame a prompt for ChatGPT. Claude implements, GPT reviews.
**Always end GPT prompts with "Answer compactly." — user has limited tokens.**

## Session Log
**2026-04-14** — Planned ML replacement. Decisions locked. Plan in `ML_PLAN.md`.
**2026-04-14** — Built dataset, backtest, ablations. LR champion chosen over LightGBM.
**2026-04-14** — Trained production model (2122+2223+2324). MLPredictor wired into LangGraph.
**2026-04-14** — Fixed data bug: 2023-24 had 184 missing odds rows. Patched, rebuilt, retrained.
**2026-04-14** — Added 2024-25 season (E0_2425.csv). Fold 3 out-of-sample: 57.5% acc (inflated).
**2026-04-14** — Codex review: fixed H2H SQL precedence bug, rewrote app.py with caching.
**2026-04-14** — SQL fix revealed old results were leaked. Honest Fold 2: 57.7% acc, 2.4% draw recall.
**2026-04-14** — Added 7-season H2H context (1718/1819/1920 as context-only). Added home_draw_rate/away_draw_rate features. H2H coverage for 2122 improved from 48% to 16% at Laplace default. Best config (bm + all draw signals): Fold 2 = 60.2% acc, LL=0.9061 — marginally beats bookmaker. Draw recall still <4%.
