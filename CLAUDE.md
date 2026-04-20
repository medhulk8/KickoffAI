# KickoffAI — CLAUDE.md

## Current Status
**V4 model deployed** — 26 features, EWM form + opponent quality. 2526 holdout: **49.7% acc / LL=1.0248** (+1.3% acc vs V3, vs bookmaker 48.6%).
Training: Last 5 seasons [1920, 2122, 2223, 2324, 2425]. 2526 is permanent holdout.

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with calibrated probabilities. Pure match-statistics model, no bookmaker odds used.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- ML: `src/ml/` — build_dataset_v4.py, backtest_v4.py, train_v4_final.py, v4_predictor.py, elo.py, backtest_rolling.py, backtest_lineup.py, train_v3_lineup.py, injury_extractor.py, injury_adjuster.py, live_features.py
- Data: `src/data/` — fetch_fpl_lineups.py (FPL GitHub lineup/xG), fetch_fpl_xg.py
- Models: `models/lr_v4_final.pkl` (V4, 26 features, Last 5 seasons), `models/lr_v3_lineup.pkl` (25 features, 3 seasons, experimental)
- DB: 8 seasons (1718–2526). **2526 is permanent holdout — never train on it.**
- Processed: `data/processed/training_dataset_v4.csv` (2957 rows), `data/processed/lineup_features.csv` (1431 rows)

## Architecture
- **Model:** CalibratedClassifierCV(Pipeline(StandardScaler + LR), cv=5, sigmoid)
- **Training window:** Last 5 seasons [1920, 2122, 2223, 2324, 2425] — concept drift confirmed by rolling-origin backtest
- **Holdout:** 2526 — permanent, never touched during training
- **Routing:** V4Predictor is default; V3Predictor + lineup model activates when `xi_strength_diff` passed in app
- Strict time-based splits only, no shuffling

## V4 Feature Set (26 features)
Per team (home + away ×2):
- `sot_l5`, `sot_conceded_l5` — venue-specific rolling
- `conversion` — Laplace-smoothed goals/SOT
- `clean_sheet_l5` — venue-specific rolling
- `pts_ewm`, `goals_ewm`, `sot_ewm` — EWMA(span=7), replaces l3-l5 linear momentum
- `days_rest` — capped at 30
- `opp_ppg_l5` — rolling avg PPG of last 5 opponents (schedule difficulty)

Shared:
- `elo_diff` — home_elo − away_elo (K=20, home_adv=100)
- `ppg_diff`, `ppg_mean`, `gd_pg_diff`, `gd_pg_mean`, `rank_diff`, `rank_mean`
- `matchweek`

Lineup model (V3, experimental): adds `xi_strength_diff` (home − away season-to-date minutes share of starting XI)

## Results

### 2526 holdout — production model
| Model | Acc | LL | Brier |
|---|---|---|---|
| Bookmaker (B365) | 48.6% | 1.0255 | — |
| LR V3 Last-5 (superseded) | 48.4% | 1.0258 | 0.6169 |
| **LR V4 EWM + opp_ppg (production)** | **49.7%** | **1.0248** | **0.6165** |
| LR V3 + xi_strength_diff (lineup, experimental) | 49.3% | — | — |

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

### Phase 1 — Feature Engineering ✓ DONE
**1A. Opponent quality (opp_ppg_l5)** — deployed in V4 ✓
**1B. EWM form (span=7)** — deployed in V4, replaces linear momentum ✓
- Key finding: neither alone moves accuracy; combination gives +1.3% (interaction effect)

### Remaining Phase 1
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
