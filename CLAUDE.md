# KickoffAI — CLAUDE.md

## Current Status
**V3 independent H/D/A model deployed** — no bookmaker odds, pure match-fact features.
Training: Last 5 seasons [1920, 2122, 2223, 2324, 2425]. Holdout 2526: **48.4% acc / LL=1.0258** (bookmaker: 48.6% / LL=1.0255 — essentially matched with no odds input).

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with calibrated probabilities. Pure match-statistics model, no bookmaker odds used.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- ML: `src/ml/` — build_dataset_v3.py, backtest_v3.py, train_v3_final.py, v3_predictor.py, elo.py, backtest_rolling.py, backtest_lineup.py, train_v3_lineup.py, injury_extractor.py, injury_adjuster.py, live_features.py
- Data: `src/data/` — fetch_fpl_lineups.py (FPL GitHub lineup/xG), fetch_fpl_xg.py
- Models: `models/lr_v3_final.pkl` (V3, 24 features, Last 5 seasons), `models/lr_v3_lineup.pkl` (25 features, 3 seasons)
- DB: 8 seasons (1718–2526). **2526 is permanent holdout — never train on it.**
- Processed: `data/processed/training_dataset_v3.csv` (2957 rows), `data/processed/lineup_features.csv` (1431 rows)

## Architecture
- **Model:** CalibratedClassifierCV(Pipeline(StandardScaler + LR), cv=5, sigmoid)
- **Training window:** Last 5 seasons [1920, 2122, 2223, 2324, 2425] — concept drift confirmed by rolling-origin backtest
- **Holdout:** 2526 — permanent, never touched during training
- **Routing:** V3Predictor routes to lineup model when `xi_strength_diff` kwarg provided, base otherwise
- Strict time-based splits only, no shuffling

## V3 Feature Set (24 features)
Per team (home + away ×2):
- `sot_l5`, `sot_conceded_l5` — venue-specific rolling
- `conversion` — Laplace-smoothed goals/SOT
- `clean_sheet_l5` — venue-specific rolling
- `pts_momentum`, `goals_momentum`, `sot_momentum` — l3 minus l5 trajectory
- `days_rest` — capped at 30

Shared:
- `elo_diff` — home_elo − away_elo (K=20, home_adv=100)
- `ppg_diff`, `ppg_mean`, `gd_pg_diff`, `gd_pg_mean`, `rank_diff`, `rank_mean`
- `matchweek`

Lineup model adds: `xi_strength_diff` (home − away season-to-date minutes share of starting XI)

## Results

### 2526 holdout — production model
| Model | Acc | LL | Brier |
|---|---|---|---|
| Bookmaker (B365) | 48.6% | 1.0255 | — |
| **LR V3 Last-5 (production)** | **48.4%** | **1.0258** | **0.6169** |
| LR V3 + xi_strength_diff (lineup) | 49.3% | — | — |

### Rolling-origin summary (3 holdouts: 2324, 2425, 2526)
| Training window | Avg Acc | Avg LL |
|---|---|---|
| **Last 5 seasons** | **53.5%** | **0.9830** |
| Full history | 53.6% | 0.9844 |
| Last 3 seasons | 53.7% | 0.9874 |

Draw recall = 0% across all models (structural — features describe strength, not balance).

## Closed Experiments (Negative Results)
- **xG features** — redundant with SOT (signal already captured). Closed.
- **Dixon-Coles** — worse than LR on all holdouts. rho≈-0.02, zero draw discrimination.
- **Draw detection** — P(D) on actual draws (0.230) = P(D) on non-draws (0.227). Closeness features collinear, degrade acc. Closed.
- **Lineup → 2526** — +1% on 2425 validation did not replicate on 2526 (-0.7%). More seasons needed.
- **V2 Elo/Dixon-Coles** — no gain vs bookmaker odds. Bookmaker is effective ceiling without genuinely new data.

## Experimental Modules (Deployed but Unvalidated)
- **Lineup model** (`lr_v3_lineup.pkl`): activates when `xi_strength_diff` provided in UI. Unvalidated on 2526.
- **Injury layer** (`injury_extractor.py` + `injury_adjuster.py`): Tavily → Ollama llama3.1:8b → post-hoc ±6% prob shift. Heuristic only. Requires: Ollama running locally, TAVILY_API_KEY.
- **O/U 2.5 ranker** (`models/ou_ranker.pkl`): top-20% lift 1.27x but AUC CI crosses zero. Paused pending live data.

## Next Steps (all testable on historical data, 2526 as holdout)

### Phase 1 — Feature Engineering
**1A. Opponent-quality weighted form**
- Weight each team's last-5 SOT/goals by opponent PPG rank
- Hypothesis: 3 shots vs Man City ≠ 3 shots vs Luton
- New features: `home_sot_l5_quality`, `away_sot_l5_quality`

**1B. Exponential-weighted form**
- Replace linear l3−l5 momentum with exponential decay (λ=0.5) over last 5 matches
- More stable, less single-game noise sensitivity
- Replaces current `pts_momentum`, `goals_momentum`, `sot_momentum`

**1C. Lineup features — fix cold-start**
- Current `xi_strength` = season-to-date minutes share → unreliable GW1–5
- Fix: use last-10-matches minutes share instead
- Also test: binary `star_absent` flag (top-2 starters by minutes not in XI)
- Rerun 2425 and 2526 holdout

### Phase 2 — Model Architecture
**2A. Soft ensemble**
- Blend base + lineup: `probs = α * base + (1−α) * lineup`
- α tuned on 2425 validation; α=1.0 when no lineup data
- Better than hard routing

**2B. Venue-split Elo**
- Track separate home-Elo and away-Elo per team
- Some teams massively overperform at home — current single Elo averages this out

**2C. Calibration: isotonic vs sigmoid**
- Test isotonic calibration — may fit football's non-monotonic probability shape better

### Phase 3 — Live (only after Phase 1–2 validate)
- Auto-fetch current GW lineups from FPL API
- Automate rolling feature update post-matchday

## Collaboration Workflow
At every major implementation step, frame a prompt for ChatGPT. Claude implements, GPT reviews.
**Always end GPT prompts with "Answer compactly." — user has limited tokens.**
