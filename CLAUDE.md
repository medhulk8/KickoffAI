# KickoffAI — CLAUDE.md

## Current Status
**Phase: PATH A — PRODUCTION MODEL DEPLOYED. Bookmaker probability calibration + confidence layer.**
SQL leakage bug fixed. No meaningful draw edge found after exhaustive honest testing.
Production model: bookmaker-only LR, trained on 2122+2223+2324 (1136 matches). Confidence threshold 0.65.

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

## Production Feature Set (locked — Path A)
`bm_home_prob, bm_draw_prob, bm_away_prob`

**Conclusion:** No draw edge found after exhaustive honest testing. H2H, team draw rates, weighted form, def_solidity, draw_likelihood — none produced meaningful draw recall (<4%) without leakage. Bookmaker-only is the cleanest honest model.

## Honest Results (post SQL fix)
| Fold | Test | Bookmaker baseline | LR bookmaker-only |
|---|---|---|---|
| Fold 1 | 2223 | 54.6% / LL=0.9724 / DrawR=0% | ~same |
| Fold 2 | 2324 | 59.9% / LL=0.9119 | 57.7% / LL=0.8301 |
| **Fold 3** | **2425 (canonical unseen)** | **54.1% / LL=0.9722** | **54.1% / LL=0.9788** |

**Conclusion:** LR matches bookmaker baseline out-of-sample. No edge. Draw recall = 0% across all honest evals.

## Confidence Thresholds (Fold 3, 2425)
| Threshold | Coverage | Accuracy |
|---|---|---|
| 0.65 (default) | 29% | 69.4% |
| 0.70 (strict) | 20% | 73.3% |

## Model Positioning
Path A: bookmaker probability wrapper and confidence filter.
- Does **not** show a reliable out-of-sample edge over bookmaker baseline
- Draw prediction unresolved — draw recall = 0% in all honest post-leakage evals
- Value: structured probability API with calibrated confidence tiers
- **V1 modeling closed.**

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
**2026-04-14** — Path A conclusion: no meaningful draw signal found. Retrained production on bookmaker-only (3 features, 1136 matches). Updated metadata with honest evaluation notes and leakage bug context. Deployed.
**2026-04-14** — Fold 3 canonical eval: 54.1% acc / LL=0.9788. LR matches bookmaker baseline on unseen season. Confidence sweep: 0.65→69.4% on 29%, 0.70→73.3% on 20%. Metadata updated. V1 modeling closed.
