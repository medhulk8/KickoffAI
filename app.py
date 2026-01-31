"""
KickoffAI - Football Match Prediction App
Interactive Streamlit interface for match predictions
"""

import streamlit as st
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime

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

async def predict_match_async(match_id: int):
    """Run prediction for a match"""
    try:
        # Import here to avoid issues
        from src.agent.agent import connect_to_mcp
        from src.kg import DynamicKnowledgeGraph
        from src.rag import WebSearchRAG
        from src.workflows.prediction_workflow import build_prediction_graph

        # Connect to MCP
        async with connect_to_mcp() as mcp_client:
            # Initialize components
            kg = DynamicKnowledgeGraph(
                db_path=str(DB_PATH),
                web_rag=None,  # No web search
                ollama_model="llama3.1:8b"
            )

            # Build workflow
            workflow = build_prediction_graph(
                mcp_client=mcp_client,
                db_path=DB_PATH,
                kg=kg,
                web_rag=None,
                ollama_model="llama3.1:8b",
                use_ensemble=False
            )

            # Run prediction
            result = await workflow.ainvoke({
                "match_id": match_id,
                "verbose": False,
                "skip_web_search": True
            })

            return result

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def display_prediction_results(result):
    """Display prediction results in a nice format"""
    if not result or result.get("error"):
        st.error("Failed to generate prediction")
        return

    match = result.get("match", {})
    prediction = result.get("prediction", {})
    evaluation = result.get("evaluation")

    # Match Info
    st.markdown("### 📊 Match Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Home Team", match.get("home_team", "N/A"))
    with col2:
        st.metric("Away Team", match.get("away_team", "N/A"))
    with col3:
        st.metric("Date", match.get("date", "N/A"))

    # Predictions
    st.markdown("### 🎯 AI Predictions")

    home_prob = prediction.get("home_prob", 0) * 100
    draw_prob = prediction.get("draw_prob", 0) * 100
    away_prob = prediction.get("away_prob", 0) * 100
    confidence = prediction.get("confidence", "medium")

    # Create three columns for probabilities
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="text-align: center; color: #ff7f0e">🏠 Home Win</h3>
            <h1 style="text-align: center; color: #1f77b4;">{home_prob:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="text-align: center; color: #ff7f0e">🤝 Draw</h3>
            <h1 style="text-align: center; color: #1f77b4;">{draw_prob:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="text-align: center; color: #ff7f0e">✈️ Away Win</h3>
            <h1 style="text-align: center; color: #1f77b4;">{away_prob:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    # Confidence
    confidence_class = f"confidence-{confidence}"
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>Confidence Level: <span class="{confidence_class}">{confidence.upper()}</span></h3>
    </div>
    """, unsafe_allow_html=True)

    # Reasoning
    if prediction.get("reasoning"):
        st.markdown("### 🧠 Detailed Analysis")
        # Use markdown for better multi-paragraph formatting
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1f77b4; color: #1f77b4;">
            {prediction["reasoning"].replace(chr(10), '<br><br>')}
        </div>
        """, unsafe_allow_html=True)

    # Actual Result (if available)
    if evaluation:
        st.markdown("### ✅ Actual Result")

        actual = evaluation.get("outcome", "N/A")
        llm_correct = evaluation.get("llm_correct", False)

        outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        actual_outcome = outcome_map.get(actual, actual)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Actual Outcome", actual_outcome)
        with col2:
            result_emoji = "✅" if llm_correct else "❌"
            st.metric("Prediction Result", f"{result_emoji} {'Correct' if llm_correct else 'Incorrect'}")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Prediction Mode",
        ["Historical Match (Test)", "Custom Match (Coming Soon)"]
    )

    st.markdown("---")

    st.markdown("### 📊 System Info")
    st.info("""
    **Performance:**
    - 56.7% Overall Accuracy
    - 66.7% High-Confidence
    - ~10-15s per prediction

    **Tech Stack:**
    - LangGraph Workflow
    - Ollama LLM (llama3.1:8b)
    - Dynamic Knowledge Graph
    - Advanced Statistics
    """)

    st.markdown("### ℹ️ About Validation")
    st.success("""
    **Why historical matches?**

    This is standard ML practice called "backtesting". The AI doesn't see the actual results when predicting - it only uses patterns from older matches.

    For future matches, the system works identically, just without immediate verification. This validates the methodology is sound.
    """)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Main content
if mode == "Historical Match (Test)":
    st.markdown("## 🔍 Select a Historical Match")

    st.info("""
    **Why Historical Matches?** This is standard practice in ML/AI systems. The system doesn't know
    the actual results when making predictions - it only uses historical statistics and patterns.
    This is called "backtesting" and validates that the system works. For future matches, it would
    work exactly the same way, just without being able to verify accuracy immediately.
    """)

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
            # Filter out team1 from the list
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

    # Generate prediction button
    if selected_season and team1 and team2 and selected_match:
        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
            with st.spinner("Running AI analysis..."):
                match_id = match_options[selected_match]

                # Run async prediction
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(predict_match_async(match_id))
                loop.close()

                if result:
                    display_prediction_results(result)
    else:
        st.info("👆 Select a season, two teams, and a match date to generate a prediction")

else:
    st.info("Custom match prediction coming soon! For now, use historical matches to test the system.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>KickoffAI</strong> - Football Match Prediction Engine</p>
    <p>Disclaimer: Predictions are for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
