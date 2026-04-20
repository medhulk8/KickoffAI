# KickoffAI

**Premier League match outcome prediction using calibrated logistic regression and a leakage-safe feature pipeline.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/data-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

KickoffAI predicts the outcome of Premier League matches — Home win, Draw, or Away win — as calibrated probabilities. It uses only free, publicly available match statistics (no bookmaker odds, no paid data), yet outperforms the Bet365 implied baseline on a fully withheld 2025-26 season holdout.

The model is a calibrated logistic regression trained on 26 hand-engineered pre-match features derived from a SQLite database of 2,957 historical matches spanning eight seasons (2017–2026).

---

## Results

| Model | Accuracy | Log-Loss |
|---|---|---|
| Bet365 bookmaker (implied) | 48.6% | 1.0255 |
| **KickoffAI V4 (production)** | **49.7%** | **1.0248** |

Evaluated on a **permanent 2025-26 holdout** (318 matches) never seen during any training or tuning decision. Rolling-origin backtests across three held-out seasons confirm the Last-5-season training window as optimal.

> Draw recall is structurally 0% — a known limitation. The 26 features describe team strength, not match balance.

---

## Feature Pipeline

The model computes 26 pre-match signals per fixture, all derived without any information leakage (strict pre-match database snapshots):

**Per team (home + away):**
| Feature | Description |
|---|---|
| `sot_l5` | Venue-specific shots on target rolling avg (last 5) |
| `sot_conceded_l5` | Venue-specific shots conceded rolling avg |
| `conversion` | Laplace-smoothed goals per shot (last 5) |
| `clean_sheet_l5` | Venue-specific clean sheet rate |
| `pts_ewm` | EWMA points (span=7, α≈0.25) |
| `goals_ewm` | EWMA goals scored |
| `sot_ewm` | EWMA shots on target |
| `days_rest` | Days since last match, capped at 30 |
| `opp_ppg_l5` | Rolling avg PPG of last 5 opponents faced |

**Shared:**
| Feature | Description |
|---|---|
| `elo_diff` | Unified Elo rating gap (K=20, home advantage=100) |
| `ppg_diff / ppg_mean` | Season PPG differential and mean |
| `gd_pg_diff / gd_pg_mean` | Goal difference per game differential and mean |
| `rank_diff / rank_mean` | League rank differential and mean |
| `matchweek` | Current matchweek |

EWMA (span=7) replaced linear momentum after a span sensitivity test showed a synergistic interaction with opponent quality — neither feature improved accuracy in isolation, but together they gave the full +1.3pp gain over V3.

---

## Architecture

```
football-data.co.uk (E0.csv)
        │
        ▼
  update_results.py  ──────────────────────────────────────────────┐
  (weekly refresh)                                                  │
                                                                    ▼
                                                            data/processed/
                                                               asil.db
                                                           (2,957 matches,
                                                            8 seasons)
                                                                    │
                     FPL API                                        │
                        │                                           │
                        ▼                                           ▼
              fetch_fixtures.py                            V4Predictor
              (next GW fixtures)                   (rolling features from DB)
                        │                                           │
                        └──────────────────┬────────────────────────┘
                                           │
                                           ▼
                                       app.py
                               (Streamlit UI — predict
                                upcoming or custom fixtures)
```

---

## Model

```
CalibratedClassifierCV(
    Pipeline([
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000)
    ]),
    cv=5, method="sigmoid"
)
```

- **Training window:** Last 5 seasons [2019-20, 2021-22, 2022-23, 2023-24, 2024-25]
- **Holdout:** 2025-26 season — permanent, never touched during any experiment
- **Calibration:** Sigmoid (5-fold cross-fit). Isotonic calibration tested and rejected.
- **Classes:** H (home win), D (draw), A (away win)

---

## Experiments Closed

Six modeling directions were tested and closed with ablation results:

| Experiment | Outcome |
|---|---|
| xG features | Redundant with SOT. No gain. |
| Dixon-Coles model | ρ ≈ −0.02, zero draw discrimination. Worse than LR. |
| Lineup XI strength | +1% on 2024-25 did not replicate on 2025-26 holdout. |
| Lineup cold-start fix | Rolling-10 xi_strength worse than season-to-date. |
| Venue-split Elo | 0.936 correlation with unified Elo. Noisier, LL degrades. |
| Isotonic calibration | Worse than sigmoid on both accuracy and log-loss. |

---

## Quick Start

```bash
git clone https://github.com/medhulk8/KickoffAI.git
cd KickoffAI

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch the prediction app
streamlit run app.py
```

**To refresh match data before predicting:**

```bash
python src/data/update_results.py
```

This fetches the latest completed matches from football-data.co.uk (~1 week lag for shots-on-target data) and inserts any new rows into the database.

---

## Project Structure

```
KickoffAI/
├── app.py                          # Streamlit prediction UI
├── requirements.txt
│
├── data/
│   ├── processed/
│   │   ├── asil.db                 # Main match database (8 seasons)
│   │   ├── training_dataset_v4.csv # 2,957-row training set
│   │   └── lineup_features.csv     # FPL-derived lineup data (experimental)
│   └── raw/                        # Season CSVs from football-data.co.uk
│
├── models/
│   ├── lr_v4_final.pkl             # Production model (26 features)
│   ├── lr_v4_meta.json             # Holdout evaluation metadata
│   └── lr_v3_lineup.pkl            # Experimental lineup model (25 features)
│
└── src/
    ├── data/
    │   ├── update_results.py       # Pull latest results from football-data.co.uk
    │   ├── fetch_fixtures.py       # Upcoming GW fixtures from FPL API
    │   ├── fetch_fpl_lineups.py    # Build lineup_features.csv from FPL GitHub
    │   ├── load_data.py            # Initial DB population
    │   └── schema.sql              # SQLite schema
    │
    └── ml/
        ├── build_dataset_v4.py     # Feature engineering pipeline
        ├── train_v4_final.py       # Model training
        ├── v4_predictor.py         # Live inference (V4)
        ├── v3_predictor.py         # Live inference (V3 + lineup)
        ├── elo.py                  # Elo rating calculator
        ├── injury_extractor.py     # Tavily + Ollama injury parser (optional)
        ├── injury_adjuster.py      # Post-hoc probability adjustment (optional)
        └── ou_ranker.py            # O/U 2.5 ranker (paused)
```

---

## Live Prediction Workflow

1. **Update the database** — run `update_results.py` to pull the latest completed matches (≈ weekly)
2. **Open the app** — `streamlit run app.py`
3. **Load upcoming fixtures** — click "Load Upcoming GW Fixtures" to auto-populate from the FPL API
4. **Select a match and predict** — V4Predictor queries rolling features from the DB and returns H/D/A probabilities

Optionally, enable the injury layer (requires Ollama running locally and a Tavily API key), which applies a heuristic ±6% probability shift based on parsed injury news.

---

## Seasonal Retraining

Each August, slide the training window forward and retrain:

```bash
# Deletes and rebuilds training_dataset_v4.csv, saves new model pkl
python src/ml/train_v4_final.py
```

The window shifts from [1920, 2122, 2223, 2324, 2425] to [2122, 2223, 2324, 2425, 2526], keeping exactly 5 seasons to balance recency and sample size (confirmed optimal by rolling-origin backtest).

---

## Data Sources

| Source | What it provides | Lag |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | Results, goals, shots, corners, odds | ~1 week |
| [FPL API](https://fantasy.premierleague.com/api/) | Upcoming fixtures, team names | Real-time |
| [FPL GitHub (vaastav)](https://github.com/vaastav/Fantasy-Premier-League) | Historical GW data for lineup features | End of season |

---

## License

MIT — see [LICENSE](LICENSE).
