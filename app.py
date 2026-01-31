"""
ASIL - Football Match Prediction App
Interactive Streamlit interface for match predictions
"""

import streamlit as st
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="ASIL - Football Predictions",
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
st.markdown('<div class="main-header">⚽ ASIL Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Football Match Predictions</div>', unsafe_allow_html=True)

# Database path
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"

@st.cache_data
def get_teams():
    """Get list of all teams from database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Get unique teams
        cursor.execute("""
            SELECT DISTINCT home_team FROM matches
            UNION
            SELECT DISTINCT away_team FROM matches
            ORDER BY 1
        """)

        teams = [row[0] for row in cursor.fetchall()]
        conn.close()
        return teams
    except Exception as e:
        st.error(f"Error loading teams: {e}")
        return []

@st.cache_data
def get_match_ids():
    """Get available match IDs for testing"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT match_id, home_team, away_team, date, home_goals, away_goals
            FROM matches
            WHERE home_goals IS NOT NULL
            ORDER BY date DESC
            LIMIT 100
        """)

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
            <h3 style="text-align: center;">🏠 Home Win</h3>
            <h1 style="text-align: center; color: #1f77b4;">{home_prob:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="text-align: center;">🤝 Draw</h3>
            <h1 style="text-align: center; color: #ff7f0e;">{draw_prob:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="text-align: center;">✈️ Away Win</h3>
            <h1 style="text-align: center; color: #2ca02c;">{away_prob:.1f}%</h1>
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
        st.markdown("### 🧠 Analysis")
        st.info(prediction["reasoning"])

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
        ["Historical Match (Test)", "Custom Match (Coming Soon)"],
        disabled=[False, True]
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

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Main content
if mode == "Historical Match (Test)":
    st.markdown("## 🔍 Select a Historical Match")

    matches = get_match_ids()

    if matches:
        # Create match options
        match_options = {}
        for match in matches:
            match_id, home, away, date, home_goals, away_goals = match
            label = f"{home} vs {away} ({date}) - Result: {home_goals}-{away_goals}"
            match_options[label] = match_id

        selected_match = st.selectbox(
            "Choose a match to predict:",
            options=list(match_options.keys())
        )

        if st.button("🎯 Generate Prediction", type="primary"):
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
        st.warning("No matches found in database. Please check your data setup.")

else:
    st.info("Custom match prediction coming soon! For now, use historical matches to test the system.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>ASIL</strong> - Advanced Sports Intelligence & Learning</p>
    <p>Disclaimer: Predictions are for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
