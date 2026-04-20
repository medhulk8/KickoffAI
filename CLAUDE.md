# KickoffAI — CLAUDE.md

## Current Status
**V4 model deployed** — 26 features, EWM form + opponent quality. 2526 holdout: **49.7% acc / LL=1.0248** (+1.3% acc vs V3, vs bookmaker 48.6%).
Training: Last 5 seasons [1920, 2122, 2223, 2324, 2425]. 2526 is permanent holdout.

## What This Project Does
Predicts Premier League match outcomes (H/D/A) with calibrated probabilities. Pure match-statistics model, no bookmaker odds used.

## Stack
- Python, SQLite (`data/processed/asil.db`), Streamlit (`app.py`)
- ML: `src/ml/` — build_dataset_v4.py, train_v4_final.py, v4_predictor.py, elo.py, v3_predictor.py, train_v3_lineup.py, injury_extractor.py, injury_adjuster.py, live_features.py, ou_ranker.py
- Data: `src/data/` — fetch_fpl_lineups.py, load_data.py, schema.sql, verify_database.py, update_results.py, fetch_fixtures.py
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
- **xG features** — redundant with SOT. Closed.
- **Dixon-Coles** — worse than LR. rho≈-0.02, zero draw discrimination. Closed.
- **Draw detection** — P(D) on draws (0.230) = P(D) on non-draws (0.227). Features don't separate balanced matches. Closed.
- **Lineup → 2526** — +1% on 2425 did not replicate. Lineup helps LL but hurts accuracy on 2526 subset.
- **Lineup cold-start fix (Phase 1C)** — xi_strength_l10 (rolling-10) worse than season-to-date. star_absent_diff redundant with xi_s2d. Closed.
- **Isotonic calibration (Phase 2C)** — worse than sigmoid on both acc and LL. Closed.
- **Venue-split Elo (Phase 2B)** — 0.936 correlation with unified elo_diff. Noisier (19 games/rating vs 38). LL degrades. Closed.
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

### Phase 1–2 Status — All completed
- Phase 1A+B (EWM + opp_ppg): ✅ Deployed as V4 (+1.3% acc)
- Phase 1C (lineup cold-start): ❌ Negative — xi_l10 worse than s2d, star_absent redundant
- Phase 2A (soft ensemble): ⏭ Skipped — lineup doesn't improve accuracy
- Phase 2B (venue-split Elo): ❌ Negative — 0.936 corr with unified Elo, noisier, LL degrades
- Phase 2C (isotonic calibration): ❌ Negative — worse than sigmoid on both metrics

### Ceiling Assessment — Free Historical Data Exhausted
V4 (49.7%) already beats bookmaker (48.6%) on 2526. GPT assessment: with free team-level match stats, we are likely close to the ceiling. We've already implemented the most plausible "historical-only" improvement (recency-aware training — rolling-origin backtest confirmed Last 5 is optimal). Further meaningful gains require genuinely new information classes:

**Paths requiring new data (in priority order):**
1. **Historical lineup/availability data** (paid) — API-Football, SportsDataIO. Strongest new signal class.
2. **Richer event data** (paid) — shot location buckets, big chances, set-piece vs open-play xG, pressing proxies. Not just match-level xG averages.
3. **Odds movement data** — pre-match line movement. Real signal but changes project goal (no longer independent).

**What to do next without new data — Live Pipeline:**

### Phase 3 — Live Data Pipeline (current)
Free data source: **football-data.co.uk** (E0.csv per season). Provides same match stats the model was trained on (results, SOT). Lag: ~1 week behind real-time.

**3A. Auto-update DB** (`src/data/update_results.py`) ✅ Built
- Fetches latest E0.csv for current season from football-data.co.uk
- Finds matches not yet in asil.db (by date + home_team + away_team)
- Inserts new rows into `matches` table using same column mapping as load_data.py
- Run manually or via cron before predicting

**3B. Upcoming fixtures** (`src/data/fetch_fixtures.py`) ✅ Built
- Hits FPL API (`bootstrap-static` + `fixtures`) for upcoming GW
- Maps FPL team names to canonical DB names
- Returns list of {home_team, away_team, date} for the next unfinished GW

**3C. App UI** — "Load Upcoming Fixtures" button in `app.py` ✅ Built
- Calls fetch_fixtures.py, auto-populates home/away dropdowns for next GW
- User selects fixture, hits Predict

**Live prediction workflow:**
1. Run `python src/data/update_results.py` (pulls latest results ~weekly)
2. Open app → Load Upcoming Fixtures → select match → Predict
3. V4Predictor queries asil.db for rolling features — works as long as DB is ≤1 GW stale

**Seasonal retraining (each August):**
- Run `python src/ml/train_v4_final.py` (auto-rebuilds dataset, saves new pkl)
- Training window slides forward: drop 1718, add 2526 → [2122,2223,2324,2425,2526]

## Collaboration Workflow
At every major implementation step, frame a prompt for ChatGPT. Claude implements, GPT reviews.
**Always end GPT prompts with "Answer compactly." — user has limited tokens.**
