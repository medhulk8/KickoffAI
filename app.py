"""
KickoffAI - Football Match Prediction App
Interactive Streamlit interface for match predictions
"""

import streamlit as st
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT_PATH = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

import csv
from src.ml.v3_predictor import V3Predictor
from src.ml.injury_extractor import InjuryExtractor
from src.ml.injury_adjuster import adjust_probabilities
from src.ml.ou_ranker import OUranker
from src.ml.live_features import compute_ou_features

LIVE_LOG_PATH = PROJECT_ROOT_PATH / "data" / "live_predictions.csv"
_LOG_HEADER = [
    "logged_at", "match_date", "home_team", "away_team",
    "bm_home_prob", "bm_draw_prob", "bm_away_prob",
    "ml_home_prob", "ml_draw_prob", "ml_away_prob",
    "confidence",
    "ou_score", "ou_bucket",
    "bm_over25_prob",
    "home_disruption", "away_disruption",
    "actual_result", "actual_over25",
    "notes",
]

def log_prediction(row: dict):
    """Append a prediction row to the live log CSV."""
    path = LIVE_LOG_PATH
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _LOG_HEADER})

# Set page config
st.set_page_config(
    page_title="KickoffAI - Football Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚽ KickoffAI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Football Match Predictions</div>', unsafe_allow_html=True)

# Database path
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"

@st.cache_resource
def get_v3_predictor():
    return V3Predictor()

@st.cache_resource
def get_injury_extractor():
    return InjuryExtractor()

@st.cache_resource
def get_ou_ranker():
    try:
        return OUranker()
    except Exception:
        return None

@st.cache_data
def get_seasons():
    """Get available seasons from database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT
                CASE
                    WHEN CAST(SUBSTR(date, 6, 2) AS INTEGER) >= 8
                    THEN SUBSTR(date, 1, 4) || '-' || CAST(CAST(SUBSTR(date, 1, 4) AS INTEGER) + 1 AS TEXT)
                    ELSE CAST(CAST(SUBSTR(date, 1, 4) AS INTEGER) - 1 AS TEXT) || '-' || SUBSTR(date, 1, 4)
                END as season
            FROM matches
            WHERE home_goals IS NOT NULL
            ORDER BY season DESC
        """)

        seasons = [row[0] for row in cursor.fetchall()]
        conn.close()
        return seasons
    except Exception as e:
        st.error(f"Error loading seasons: {e}")
        return []

@st.cache_data
def get_teams_in_season(season: str):
    """Get teams that played in a specific season"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Parse season (e.g., "2023-2024" -> start_year=2023, end_year=2024)
        start_year, end_year = season.split('-')

        cursor.execute("""
            SELECT DISTINCT home_team FROM matches
            WHERE (
                (CAST(SUBSTR(date, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(date, 6, 2) AS INTEGER) >= 8)
                OR
                (CAST(SUBSTR(date, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(date, 6, 2) AS INTEGER) < 8)
            )
            UNION
            SELECT DISTINCT away_team FROM matches
            WHERE (
                (CAST(SUBSTR(date, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(date, 6, 2) AS INTEGER) >= 8)
                OR
                (CAST(SUBSTR(date, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(date, 6, 2) AS INTEGER) < 8)
            )
            ORDER BY 1
        """, (int(start_year), int(end_year), int(start_year), int(end_year)))

        teams = [row[0] for row in cursor.fetchall()]
        conn.close()
        return teams
    except Exception as e:
        st.error(f"Error loading teams: {e}")
        return []

@st.cache_data
def get_matches_between_teams(season: str, team1: str, team2: str):
    """Get matches between two specific teams in a season"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Parse season
        start_year, end_year = season.split('-')

        cursor.execute("""
            SELECT match_id, home_team, away_team, date, home_goals, away_goals
            FROM matches
            WHERE (
                (home_team = ? AND away_team = ?)
                OR
                (home_team = ? AND away_team = ?)
            )
            AND (
                (CAST(SUBSTR(date, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(date, 6, 2) AS INTEGER) >= 8)
                OR
                (CAST(SUBSTR(date, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(date, 6, 2) AS INTEGER) < 8)
            )
            AND home_goals IS NOT NULL
            ORDER BY date
        """, (team1, team2, team2, team1, int(start_year), int(end_year)))

        matches = cursor.fetchall()
        conn.close()
        return matches
    except Exception as e:
        st.error(f"Error loading matches: {e}")
        return []

def _predict_historical(match_id: int, use_injury_adjustment: bool = False):
    """Run V3 prediction for a historical match (features computed as-of match date)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,)).fetchone()
    conn.close()
    if row is None:
        return None

    match = dict(row)
    home = match["home_team"]
    away = match["away_team"]
    date = match["date"][:10]

    pred = get_v3_predictor()
    result = pred.predict(home, away, date)

    home_prob = result["home_prob"]
    draw_prob = result["draw_prob"]
    away_prob = result["away_prob"]
    injury_info = None
    injury_adjustment = None

    if use_injury_adjustment:
        try:
            injury_info = get_injury_extractor().extract(home, away, date)
            adj = adjust_probabilities(
                home_prob, draw_prob, away_prob,
                injury_info["home_disruption"], injury_info["away_disruption"],
            )
            if adj["adjusted"]:
                home_prob, draw_prob, away_prob = adj["home_prob"], adj["draw_prob"], adj["away_prob"]
            injury_adjustment = adj
        except Exception as e:
            injury_info = {"summary": f"Injury fetch failed: {e}", "home_disruption": 0.0, "away_disruption": 0.0}

    actual = match.get("result")
    prediction_label = result["prediction"]

    return {
        "match":      {"home_team": home, "away_team": away, "date": date},
        "home_prob":  home_prob, "draw_prob": draw_prob, "away_prob": away_prob,
        "confidence": result["confidence"],
        "features":   result["features"],
        "injury_info":       injury_info,
        "injury_adjustment": injury_adjustment,
        "evaluation": {"outcome": actual, "correct": prediction_label == actual} if actual else None,
    }


def _display_probs(home_prob, draw_prob, away_prob, confidence):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="prediction-card">
            <h3 style="text-align:center;color:#ff7f0e">🏠 Home Win</h3>
            <h1 style="text-align:center;color:#1f77b4">{home_prob*100:.1f}%</h1>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="prediction-card">
            <h3 style="text-align:center;color:#ff7f0e">🤝 Draw</h3>
            <h1 style="text-align:center;color:#1f77b4">{draw_prob*100:.1f}%</h1>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="prediction-card">
            <h3 style="text-align:center;color:#ff7f0e">✈️ Away Win</h3>
            <h1 style="text-align:center;color:#1f77b4">{away_prob*100:.1f}%</h1>
        </div>""", unsafe_allow_html=True)
    conf_class = f"confidence-{confidence}"
    st.markdown(f"""<div style="text-align:center;margin:1rem 0;">
        <h3>Confidence: <span class="{conf_class}">{confidence.upper()}</span></h3>
    </div>""", unsafe_allow_html=True)


def _display_injury(injury_info, injury_adjustment, home, away):
    if not injury_info:
        return
    st.markdown("### 🏥 Injury / Suspension Context")
    if injury_info.get("summary"):
        st.info(injury_info["summary"])
    ic1, ic2 = st.columns(2)
    for col, team, key in [(ic1, home, "home"), (ic2, away, "away")]:
        with col:
            dis = injury_info.get(f"{key}_disruption", 0.0)
            st.markdown(f"**{team}** — disruption: {dis:.0%}")
            out = injury_info.get(f"{key}_confirmed_out", [])
            dbt = injury_info.get(f"{key}_doubtful", [])
            if out: st.markdown("Out: " + ", ".join(out))
            if dbt: st.markdown("Doubtful: " + ", ".join(dbt))
            if not out and not dbt: st.markdown("No notable absences reported.")
    if injury_adjustment and injury_adjustment.get("adjusted"):
        st.caption(f"Heuristic adjustment: {injury_adjustment['note']}  (max ±6% — not trained)")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Prediction Mode",
        ["Live / Custom Match", "Historical Match (Backtest)"]
    )

    st.markdown("---")
    st.markdown("### 🏥 Injury Adjustment")
    use_injury = st.toggle(
        "Fetch live injury news",
        value=False,
        help=(
            "Searches Tavily for injury/suspension news, extracts disruption scores "
            "with a local LLM, and applies a small heuristic adjustment (max ±6%). "
            "Requires Ollama (llama3.1:8b) running locally and a TAVILY_API_KEY. "
            "Adds ~10-20s to prediction time."
        ),
    )
    if use_injury:
        st.caption("Heuristic only — not trained on historical lineup data.")

    st.markdown("---")

    st.markdown("### 📊 System Info")
    st.info("""
    **V3 Independent Model:**
    - 53.6% Accuracy (2425 holdout)
    - Trained on 7 seasons, 2639 matches
    - No bookmaker odds — match-facts only

    **Features (24):**
    SOT, conversion, form, ELO, rank,
    PPG, GD/game, momentum, rest days

    **Tech Stack:**
    - Calibrated Logistic Regression
    - Optional injury layer (Tavily + Ollama)
    """)

    st.markdown("### ℹ️ About Validation")
    st.success("""
    **Why historical matches?**

    This is standard ML practice called "backtesting". The AI doesn't see the actual results when predicting - it only uses patterns from older matches.

    For future matches, the system works identically, just without immediate verification. This validates the methodology is sound.
    """)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# ──────────────────────────────────────────────────────────────────────────────
# MODE: Live / Custom Match
# ──────────────────────────────────────────────────────────────────────────────
if mode == "Live / Custom Match":
    st.markdown("## 🔮 Live Match Prediction")
    st.caption(
        "Enter a fixture. Rolling stats and ELO are computed from the DB automatically. "
        "No bookmaker odds needed for H/D/A."
    )

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("🏠 Home Team", placeholder="e.g. Arsenal")
    with col2:
        away_team = st.text_input("✈️ Away Team", placeholder="e.g. Chelsea")

    match_date = st.date_input("📅 Match Date", value=datetime.today())

    st.markdown("#### Lineup (optional — activates lineup-aware model)")
    use_lineup = st.checkbox("Lineups confirmed", value=False,
                             help="When starting XIs are known, the lineup-aware model (3-season, 25 features) is used instead of the default 7-season model.")
    if use_lineup:
        st.caption(
            "xi_strength_diff = home XI strength − away XI strength. "
            "Each team's XI strength is the mean season-to-date minutes share of the 11 starters. "
            "Positive → home lineup stronger, negative → away lineup stronger."
        )
        xi_diff = st.slider("XI Strength Diff (home − away)", min_value=-1.0, max_value=1.0,
                            value=0.0, step=0.01, format="%.2f")
    else:
        xi_diff = None

    st.markdown("#### O/U 2.5 Odds (optional, for O/U ranking only)")
    use_ou = st.checkbox("Include Over/Under 2.5 prediction", value=True)
    ou_col1, ou_col2 = st.columns(2)
    with ou_col1:
        over_odds  = st.number_input("Over 2.5 Odds",  min_value=1.01, max_value=20.0, value=1.85, step=0.05, disabled=not use_ou)
    with ou_col2:
        under_odds = st.number_input("Under 2.5 Odds", min_value=1.01, max_value=20.0, value=1.95, step=0.05, disabled=not use_ou)

    st.markdown("---")

    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True,
                 disabled=not (home_team.strip() and away_team.strip())):
        spinner_msg = "Running analysis + fetching injury news..." if use_injury else "Running analysis..."
        with st.spinner(spinner_msg):
            h = home_team.strip()
            a = away_team.strip()
            date_str = match_date.strftime("%Y-%m-%d")

            # ── H/D/A — V3 model (lineup-aware when xi_diff provided) ─────────
            try:
                hda = get_v3_predictor().predict(h, a, date_str, xi_strength_diff=xi_diff)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            home_prob = hda["home_prob"]
            draw_prob = hda["draw_prob"]
            away_prob = hda["away_prob"]

            # ── Injury adjustment (optional) ──────────────────────────────────
            injury_info = None
            injury_adjustment = None
            if use_injury:
                try:
                    injury_info = get_injury_extractor().extract(h, a, date_str)
                    adj = adjust_probabilities(
                        home_prob, draw_prob, away_prob,
                        injury_info["home_disruption"], injury_info["away_disruption"],
                    )
                    if adj["adjusted"]:
                        home_prob, draw_prob, away_prob = adj["home_prob"], adj["draw_prob"], adj["away_prob"]
                    injury_adjustment = adj
                except Exception as e:
                    injury_info = {"summary": f"Injury fetch failed: {e}",
                                   "home_disruption": 0.0, "away_disruption": 0.0}

            # ── O/U prediction (optional) ──────────────────────────────────────
            ou_result = None
            ou_features = None
            if use_ou:
                bm_over_prob = (1 / over_odds) / (1 / over_odds + 1 / under_odds)
                ranker = get_ou_ranker()
                if ranker:
                    try:
                        ou_features = compute_ou_features(h, a, bm_over_prob, str(DB_PATH))
                        model_feats = {k: v for k, v in ou_features.items() if not k.startswith("_")}
                        ou_result = ranker.rank(model_feats)
                    except Exception as e:
                        ou_result = {"error": str(e)}
                else:
                    ou_result = {"error": "O/U model not loaded"}

            # ── Display ────────────────────────────────────────────────────────
            st.markdown("### 📊 Match")
            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.metric("Home", h)
            with mc2: st.metric("Away", a)
            with mc3: st.metric("Date", date_str)

            st.markdown("### 🎯 H/D/A Prediction")
            _display_probs(home_prob, draw_prob, away_prob, hda["confidence"])
            model_label = "v3-lineup (3-season)" if hda["model_version"] == "v3_lineup" else "v3-independent (7-season)"
            st.caption(
                f"Model: {model_label} — ELO diff: {hda['features']['elo_diff']:+.0f} | "
                f"PPG diff: {hda['features']['ppg_diff']:+.2f} | Rank diff: {hda['features']['rank_diff']:+.0f}"
            )

            # Key features expander
            with st.expander("Feature breakdown", expanded=False):
                import pandas as pd
                feats = hda["features"]
                rows = [
                    ("ELO diff (home − away)", feats["elo_diff"]),
                    ("PPG diff (home − away)", feats["ppg_diff"]),
                    ("PPG mean (match quality)", feats["ppg_mean"]),
                    ("Rank diff (home − away)", feats["rank_diff"]),
                    ("Rank mean (match quality)", feats["rank_mean"]),
                    ("Home SOT L5", feats["home_sot_l5"]),
                    ("Away SOT L5", feats["away_sot_l5"]),
                    ("Home pts momentum", feats["home_pts_momentum"]),
                    ("Away pts momentum", feats["away_pts_momentum"]),
                    ("GD/game diff", feats["gd_pg_diff"]),
                ]
                if "xi_strength_diff" in feats:
                    rows.insert(0, ("XI strength diff (lineup)", feats["xi_strength_diff"]))
                st.dataframe(pd.DataFrame(rows, columns=["Feature", "Value"]), use_container_width=True, hide_index=True)

            # O/U result
            if ou_result:
                st.markdown("### ⚽ O/U 2.5 Ranking")
                if "error" in ou_result:
                    st.warning(f"O/U ranker: {ou_result['error']}")
                else:
                    BUCKET_LABEL = {"top_10pct": "Top 10% — Likely Over", "top_20pct": "Top 20% — Likely Over", "unranked": "No Signal"}
                    BUCKET_COLOR = {"top_10pct": "#28a745", "top_20pct": "#17a2b8", "unranked": "#6c757d"}
                    bucket = ou_result["bucket"]
                    oc1, oc2, oc3 = st.columns(3)
                    with oc1: st.metric("O/U Score", f"{ou_result['score']:.3f}")
                    with oc2:
                        st.markdown(f"""<div style="text-align:center;padding:0.5rem;border-radius:8px;background:{BUCKET_COLOR.get(bucket,'#6c757d')};color:white">
                            <strong>{BUCKET_LABEL.get(bucket, bucket)}</strong></div>""", unsafe_allow_html=True)
                    with oc3: st.metric("Expected Over Rate", f"{ou_result['over_rate_expected']:.0%}")
                    if ou_features and ou_features.get("_missing"):
                        st.warning("One or both teams not found in DB — O/U used league-average defaults.")
                    else:
                        st.caption(f"Based on last {ou_features['_home_n']} {h} + {ou_features['_away_n']} {a} matches.")

            _display_injury(injury_info, injury_adjustment, h, a)

            # Log
            log_row = {
                "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "match_date": date_str, "home_team": h, "away_team": a,
                "bm_home_prob": "", "bm_draw_prob": "", "bm_away_prob": "",
                "ml_home_prob": home_prob, "ml_draw_prob": draw_prob, "ml_away_prob": away_prob,
                "confidence": hda["confidence"],
                "ou_score":  ou_result.get("score", "") if (ou_result and "error" not in ou_result) else "",
                "ou_bucket": ou_result.get("bucket", "") if (ou_result and "error" not in ou_result) else "",
                "bm_over25_prob": ou_features["bm_over25_prob"] if ou_features else "",
                "home_disruption": injury_info.get("home_disruption", "") if injury_info else "",
                "away_disruption": injury_info.get("away_disruption", "") if injury_info else "",
                "actual_result": "", "actual_over25": "", "notes": "",
            }
            try:
                log_prediction(log_row)
                st.success("Logged to `data/live_predictions.csv`.")
            except Exception as e:
                st.warning(f"Could not log prediction: {e}")

    else:
        st.info("👆 Enter both team names and click Generate Prediction.")

    # Show prediction log if it exists
    if LIVE_LOG_PATH.exists() and LIVE_LOG_PATH.stat().st_size > 0:
        with st.expander("📋 Prediction Log (live_predictions.csv)", expanded=False):
            import pandas as pd
            log_df = pd.read_csv(LIVE_LOG_PATH)
            st.dataframe(log_df, use_container_width=True)
            st.caption("Fill in `actual_result` (H/D/A) and `actual_over25` (0/1) manually in the CSV after each match.")


# ──────────────────────────────────────────────────────────────────────────────
# MODE: Historical / Backtest
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("## 🔍 Historical Match Backtest")
    st.info("Select a past match to run the model on it and compare against the known result.")

    # Cascading filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        seasons = get_seasons()
        if seasons:
            selected_season = st.selectbox("📅 Season", seasons)
        else:
            st.error("No seasons found")
            selected_season = None

    with col2:
        if selected_season:
            teams = get_teams_in_season(selected_season)
            if teams:
                team1 = st.selectbox("🏠 Team 1", teams)
            else:
                st.warning("No teams found")
                team1 = None
        else:
            team1 = None

    with col3:
        if selected_season and team1:
            available_teams = [t for t in teams if t != team1]
            if available_teams:
                team2 = st.selectbox("✈️ Team 2", available_teams)
            else:
                st.warning("No other teams found")
                team2 = None
        else:
            team2 = None

    with col4:
        if selected_season and team1 and team2:
            matches = get_matches_between_teams(selected_season, team1, team2)
            if matches:
                match_options = {}
                for match in matches:
                    match_id, home, away, date, home_goals, away_goals = match
                    label = f"{date} ({home} vs {away})"
                    match_options[label] = match_id
                selected_match = st.selectbox("📆 Match Date", list(match_options.keys()))
            else:
                st.warning("No matches found between these teams")
                selected_match = None
        else:
            selected_match = None

    st.markdown("---")

    if selected_season and team1 and team2 and selected_match:
        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
            spinner_msg = "Running analysis + fetching injury news..." if use_injury else "Running AI analysis..."
            with st.spinner(spinner_msg):
                match_id = match_options[selected_match]
                result = _predict_historical(match_id, use_injury_adjustment=use_injury)
                if result:
                    m = result["match"]
                    st.markdown("### 📊 Match")
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1: st.metric("Home", m["home_team"])
                    with mc2: st.metric("Away", m["away_team"])
                    with mc3: st.metric("Date", m["date"])

                    st.markdown("### 🎯 Prediction")
                    _display_probs(result["home_prob"], result["draw_prob"], result["away_prob"], result["confidence"])

                    _display_injury(result.get("injury_info"), result.get("injury_adjustment"),
                                    m["home_team"], m["away_team"])

                    if result.get("evaluation"):
                        ev = result["evaluation"]
                        st.markdown("### ✅ Actual Result")
                        outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
                        ec1, ec2 = st.columns(2)
                        with ec1: st.metric("Actual Outcome", outcome_map.get(ev["outcome"], ev["outcome"]))
                        with ec2:
                            emoji = "✅" if ev["correct"] else "❌"
                            st.metric("Prediction", f"{emoji} {'Correct' if ev['correct'] else 'Incorrect'}")
    else:
        st.info("👆 Select a season, two teams, and a match date to generate a prediction")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>KickoffAI</strong> - Football Match Prediction Engine</p>
    <p>Disclaimer: Predictions are for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
