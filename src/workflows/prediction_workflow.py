"""
LangGraph Prediction Workflow

This module defines a graph-based workflow for football match predictions
using LangGraph's StateGraph. The workflow orchestrates 8 nodes:

1. match_selector   - Load match details and initialize state
2. stats_collector  - Gather team form and baseline probabilities
3. kg_query         - Query knowledge graph for tactical insights
4. web_search       - Execute web searches for current context (conditional)
5. llm_predictor    - Generate LLM prediction with Ollama
6. critique         - Self-critique the prediction
7. logger           - Save prediction to database
8. evaluator        - Evaluate prediction if match completed

The graph structure enables:
- Conditional branching (skip web search if not needed)
- Cycle handling (re-prediction if confidence too low)
- State persistence across nodes
- Clear visualization of workflow

Usage:
    from src.workflows import create_prediction_workflow, PredictionState

    components = create_workflow_components(mcp_client, db_path, kg, web_rag)
    workflow = create_prediction_workflow(components)
    result = await workflow.ainvoke({"match_id": 1})
"""

from typing import TypedDict, Literal, Optional, Any, Dict, List, Callable
from typing_extensions import Annotated
from pathlib import Path
from datetime import datetime
import re
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END

# Local imports
from .confidence_calculator import ConfidenceCalculator
from .draw_detector import DrawDetector
from .ensemble_predictor import EnsemblePredictor
from src.data.advanced_stats import AdvancedStatsCalculator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class PredictionState(TypedDict, total=False):
    """
    Shared state across all workflow nodes.

    This TypedDict defines the structure of data that flows through
    the prediction workflow graph. Each node can read from and write
    to this state.

    Attributes:
        match_id: Unique identifier for the match
        match: Match details from database
        home_form: Home team's recent form statistics
        away_form: Away team's recent form statistics
        baseline: Bookmaker baseline probabilities
        kg_insights: Tactical insights from knowledge graph
        web_context: Retrieved web search context
        prediction: LLM prediction with probabilities
        critique: Self-critique results
        evaluation: Prediction evaluation results
        prediction_id: Database ID of logged prediction
        skip_web_search: Flag to skip web search step
        confidence_level: Current prediction confidence
        iteration_count: Number of prediction iterations (for cycle limiting)
        verbose: Enable verbose output
        error: Error message if any step fails
    """
    # Input
    match_id: int
    verbose: bool

    # Match data (populated by match_selector and stats_collector nodes)
    match: Dict[str, Any]
    home_form: Dict[str, Any]
    away_form: Dict[str, Any]
    baseline: Dict[str, Any]

    # Advanced stats (populated by stats_collector node)
    home_advanced_stats: Dict[str, Any]
    away_advanced_stats: Dict[str, Any]
    h2h_stats: Dict[str, Any]

    # Weighted form stats (populated by stats_collector node)
    home_weighted_form: Dict[str, Any]
    away_weighted_form: Dict[str, Any]
    form_comparison: Dict[str, Any]

    # Context gathering (populated by respective nodes)
    kg_insights: Dict[str, Any]
    web_context: Dict[str, Any]

    # Prediction (populated by llm_predictor node)
    prediction: Dict[str, Any]
    critique: Dict[str, Any]

    # Evaluation (populated by evaluator node)
    evaluation: Dict[str, Any]
    prediction_id: int

    # Workflow control
    skip_web_search: bool
    confidence_level: str
    iteration_count: int

    # Error handling
    error: Optional[str]


# ============================================================================
# Helper Functions
# ============================================================================

def _build_prediction_prompt(state: PredictionState) -> str:
    """
    Build structured 6-step Chain-of-Thought prompt from state.

    Forces systematic reasoning through:
    1. Recent form analysis
    2. Advanced statistics analysis
    3. Tactical matchup analysis
    4. Head-to-head history
    5. Current context & news
    6. Baseline check & final synthesis
    """
    match = state["match"]
    home_form = state["home_form"]
    away_form = state["away_form"]
    baseline = state["baseline"]
    kg = state.get("kg_insights", {})
    web = state.get("web_context", {})

    # Get advanced stats (with fallback to empty dicts)
    home_adv = state.get("home_advanced_stats", {})
    away_adv = state.get("away_advanced_stats", {})
    h2h = state.get("h2h_stats", {})

    # Get weighted form data (recency-weighted metrics)
    home_weighted = state.get("home_weighted_form", {})
    away_weighted = state.get("away_weighted_form", {})
    form_comp = state.get("form_comparison", {})

    # Extract nested stats with safe defaults
    home_att = home_adv.get('attacking', {})
    home_def = home_adv.get('defensive', {})
    home_eff = home_adv.get('efficiency', {})

    away_att = away_adv.get('attacking', {})
    away_def = away_adv.get('defensive', {})
    away_eff = away_adv.get('efficiency', {})

    # Format KG insights
    home_styles = kg.get("home_styles", []) if kg else []
    away_styles = kg.get("away_styles", []) if kg else []
    matchup_summary = kg.get('matchup_summary', 'No clear tactical advantage') if kg else 'No data'
    kg_confidence = kg.get('confidence', 'none') if kg else 'none'

    # Format web context (truncate for prompt size)
    web_text = web.get("all_content", "No web context available.")[:800] if web else "Web search skipped."

    # Format form strings with safe defaults
    home_ppg = home_form.get('points_per_game', 0) or 0
    away_ppg = away_form.get('points_per_game', 0) or 0
    home_wins = home_form.get('wins', 0) or 0
    home_draws = home_form.get('draws', 0) or 0
    home_losses = home_form.get('losses', 0) or 0
    away_wins = away_form.get('wins', 0) or 0
    away_draws = away_form.get('draws', 0) or 0
    away_losses = away_form.get('losses', 0) or 0
    home_gs = home_form.get('goals_scored', 0) or 0
    home_gc = home_form.get('goals_conceded', 0) or 0
    away_gs = away_form.get('goals_scored', 0) or 0
    away_gc = away_form.get('goals_conceded', 0) or 0

    # Format H2H with safe defaults
    h2h_played = h2h.get('matches_played', 0) or 0
    h2h_t1_wins = h2h.get('team1_wins', 0) or 0
    h2h_t2_wins = h2h.get('team2_wins', 0) or 0
    h2h_draws = h2h.get('draws', 0) or 0
    h2h_results = h2h.get('recent_results', [])
    h2h_t1_goals = h2h.get('avg_goals_team1', 0) or 0
    h2h_t2_goals = h2h.get('avg_goals_team2', 0) or 0
    h2h_dominance = h2h.get('dominance', 'no_history') or 'no_history'

    # Draw detection - check if this match has high draw likelihood
    # Phase 6: Now uses weighted metrics for better accuracy
    draw_detector = DrawDetector()
    draw_likelihood = draw_detector.detect_draw_likelihood(
        home_form=home_form,
        away_form=away_form,
        baseline=baseline,
        h2h_stats=h2h,
        advanced_stats_home=home_adv,
        advanced_stats_away=away_adv,
        home_weighted_form=home_weighted,
        away_weighted_form=away_weighted,
        form_comparison=form_comp
    )
    draw_warning = draw_detector.get_draw_warning(draw_likelihood)
    draw_warning_text = f"\n{draw_warning}\n" if draw_warning else ""

    return f"""You are an expert football analyst.{draw_warning_text} Predict the match outcome using this structured 6-step process.

MATCH: {match['home_team']} vs {match['away_team']}
DATE: {match.get('date', 'Unknown')}
VENUE: {match['home_team']} (Home)

================================================================================
ANALYZE EACH STEP SYSTEMATICALLY:
================================================================================

STEP 1 - RECENT FORM ANALYSIS (with Recency Weighting)
-------------------------------------------------------
Home Team ({match['home_team']}) - Last 5 Matches:
- Record: {home_wins}W-{home_draws}D-{home_losses}L
- Traditional PPG: {home_ppg:.2f}
- Weighted PPG: {home_weighted.get('weighted_points_per_game', 0):.2f} (recent matches weighted higher)
- Goals: {home_gs} scored, {home_gc} conceded
- Weighted Goals/Game: {home_weighted.get('weighted_goals_per_game', 0):.2f} scored, {home_weighted.get('weighted_goals_conceded', 0):.2f} conceded
- Momentum Score: {home_weighted.get('momentum_score', 0):.1f}/3.0 (based on recent 2 matches)
- Form string: {home_form.get('form_string', 'N/A')}

Away Team ({match['away_team']}) - Last 5 Matches:
- Record: {away_wins}W-{away_draws}D-{away_losses}L
- Traditional PPG: {away_ppg:.2f}
- Weighted PPG: {away_weighted.get('weighted_points_per_game', 0):.2f} (recent matches weighted higher)
- Goals: {away_gs} scored, {away_gc} conceded
- Weighted Goals/Game: {away_weighted.get('weighted_goals_per_game', 0):.2f} scored, {away_weighted.get('weighted_goals_conceded', 0):.2f} conceded
- Momentum Score: {away_weighted.get('momentum_score', 0):.1f}/3.0 (based on recent 2 matches)
- Form string: {away_form.get('form_string', 'N/A')}

Form Comparison:
- Weighted PPG Differential: {form_comp.get('ppg_differential', 0):+.2f} (positive favors home)
- Goal Differential: {form_comp.get('goal_differential', 0):+.2f} (positive favors home)
- Momentum Advantage: {form_comp.get('momentum_advantage', 'even').upper()}
- Overall Form Advantage: {form_comp.get('form_advantage', 'even').upper()}

STEP 2 - ADVANCED STATISTICS
-----------------------------
{match['home_team']} Stats:
- Attacking: {home_att.get('avg_shots', 0):.1f} shots/game, {home_att.get('avg_goals', 0):.2f} goals/game
- Defensive: {home_def.get('avg_goals_conceded', 0):.2f} conceded/game, {home_def.get('clean_sheet_rate', 0):.0%} clean sheets
- Efficiency: {home_eff.get('conversion_rate', 0):.1f}% conversion, {home_eff.get('attacking_threat', 0):.1f}/10 threat

{match['away_team']} Stats:
- Attacking: {away_att.get('avg_shots', 0):.1f} shots/game, {away_att.get('avg_goals', 0):.2f} goals/game
- Defensive: {away_def.get('avg_goals_conceded', 0):.2f} conceded/game, {away_def.get('clean_sheet_rate', 0):.0%} clean sheets
- Efficiency: {away_eff.get('conversion_rate', 0):.1f}% conversion, {away_eff.get('attacking_threat', 0):.1f}/10 threat

STEP 3 - TACTICAL MATCHUP
--------------------------
- {match['home_team']} style: {', '.join(home_styles) if home_styles else 'Unknown'}
- {match['away_team']} style: {', '.join(away_styles) if away_styles else 'Unknown'}
- Matchup analysis: {matchup_summary}
- Tactical confidence: {kg_confidence.upper()}

STEP 4 - HEAD-TO-HEAD HISTORY
------------------------------
Recent H2H ({h2h_played} meetings):
- {match['home_team']} wins: {h2h_t1_wins}
- {match['away_team']} wins: {h2h_t2_wins}
- Draws: {h2h_draws}
- Recent results: {', '.join(h2h_results[:3]) if h2h_results else 'No recent meetings'}
- Average goals: {match['home_team']} {h2h_t1_goals:.1f}, {match['away_team']} {h2h_t2_goals:.1f}
- Pattern: {h2h_dominance.upper()}

STEP 5 - CURRENT CONTEXT
-------------------------
{web_text}

STEP 6 - BASELINE COMPARISON
-----------------------------
Bookmaker probabilities:
- Home win: {baseline['home_prob']:.1%}
- Draw: {baseline['draw_prob']:.1%}
- Away win: {baseline['away_prob']:.1%}

Consider: Does your analysis support or contradict the bookmakers?

DRAW CONSIDERATION:
Draws occur in ~25% of Premier League matches. Consider draws seriously when:
• Teams evenly matched (form difference < 0.5 PPG)
• Both defensive/cautious (low goals per game)
• Close baseline probabilities (no clear favorite >55%)
• Historical draws between these teams

================================================================================
FINAL PREDICTION
================================================================================
⚠️ UNCERTAINTY REMINDER: Avoid overconfidence. Draws happen 1 in 4 matches.
If no team dominates across ALL factors, draw probability should be 25-35%.

Based on your 6-step analysis, provide your prediction with DETAILED analysis.

Respond EXACTLY in this format:
PROBABILITIES: H=XX%, D=XX%, A=XX%

REASONING:
Write a comprehensive 4-6 paragraph analysis covering:

**Form Analysis:** Detail recent performance trends for both teams. Include win/loss records, scoring patterns, defensive solidity, and momentum. Explain which team has better current form and why this matters.

**Statistical Edge:** Compare key stats - shots, conversion rates, defensive records, clean sheets. Identify which team has statistical advantages and quantify the gap. Explain how these numbers translate to match outcome.

**Tactical Matchup:** Analyze the clash of playing styles. How will each team's tactical approach work against the opponent? What are the key battles (e.g., possession vs counter-attack, high press vs deep block)? Which style is likely to dominate and why?

**Historical Context:** Review head-to-head patterns and historical dominance. Are there psychological factors? Does one team consistently perform better in this fixture? How reliable is this historical trend?

**Critical Factors:** Identify 2-3 decisive factors that will determine the outcome. These could be: key player matchups, home advantage effectiveness, defensive vulnerabilities, injury impacts, or current context from news.

**Final Synthesis:** Bring it all together - why are you picking this specific probability distribution? What's the single most important reason for your prediction? Address any conflicting signals in the data.

CONFIDENCE: HIGH/MEDIUM/LOW"""


def _parse_prediction_response(response_text: str, baseline: dict) -> dict:
    """
    Parse LLM response to extract probabilities and reasoning.

    Uses multiple parsing strategies in order of preference:
    1. Exact format: "H=X%, D=Y%, A=Z%"
    2. Keyword format: "Home: X%, Draw: Y%, Away: Z%"
    3. Labeled format: "Home win: X%", "Draw: Y%", "Away win: Z%"
    4. Bullet format: "- Home: X%"
    5. Any 3 consecutive percentages
    6. Fallback to baseline
    """
    home_prob = None
    draw_prob = None
    away_prob = None
    parse_method = None

    # Strategy 1: Exact format "H=X%, D=Y%, A=Z%"
    match1 = re.search(
        r'H\s*=\s*(\d+(?:\.\d+)?)\s*%.*?D\s*=\s*(\d+(?:\.\d+)?)\s*%.*?A\s*=\s*(\d+(?:\.\d+)?)\s*%',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if match1:
        home_prob = float(match1.group(1)) / 100
        draw_prob = float(match1.group(2)) / 100
        away_prob = float(match1.group(3)) / 100
        parse_method = "exact_format"

    # Strategy 2: Keyword format "Home: X%, Draw: Y%, Away: Z%"
    if home_prob is None:
        match2 = re.search(
            r'Home[:\s]+(\d+(?:\.\d+)?)\s*%.*?Draw[:\s]+(\d+(?:\.\d+)?)\s*%.*?Away[:\s]+(\d+(?:\.\d+)?)\s*%',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if match2:
            home_prob = float(match2.group(1)) / 100
            draw_prob = float(match2.group(2)) / 100
            away_prob = float(match2.group(3)) / 100
            parse_method = "keyword_format"

    # Strategy 3: Labeled format with "win" keyword
    if home_prob is None:
        home_match = re.search(r'Home\s*(?:win)?[:\s]+(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)
        draw_match = re.search(r'Draw[:\s]+(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)
        away_match = re.search(r'Away\s*(?:win)?[:\s]+(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)

        if home_match and draw_match and away_match:
            home_prob = float(home_match.group(1)) / 100
            draw_prob = float(draw_match.group(1)) / 100
            away_prob = float(away_match.group(1)) / 100
            parse_method = "labeled_format"

    # Strategy 4: Bullet format "- Home: X%"
    if home_prob is None:
        bullet_pattern = re.findall(
            r'[-*•]\s*(?:Home|Draw|Away)[^:]*:\s*(\d+(?:\.\d+)?)\s*%',
            response_text,
            re.IGNORECASE
        )
        if len(bullet_pattern) >= 3:
            home_prob = float(bullet_pattern[0]) / 100
            draw_prob = float(bullet_pattern[1]) / 100
            away_prob = float(bullet_pattern[2]) / 100
            parse_method = "bullet_format"

    # Strategy 5: Any 3 consecutive percentages
    if home_prob is None:
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', response_text)
        if len(percentages) >= 3:
            valid_pcts = [float(p) for p in percentages if 0 <= float(p) <= 100]
            if len(valid_pcts) >= 3:
                home_prob = valid_pcts[0] / 100
                draw_prob = valid_pcts[1] / 100
                away_prob = valid_pcts[2] / 100
                parse_method = "any_percentages"

    # Strategy 6: Fallback to baseline
    if home_prob is None:
        home_prob = baseline['home_prob']
        draw_prob = baseline['draw_prob']
        away_prob = baseline['away_prob']
        parse_method = "baseline_fallback"
        logger.warning("Could not parse LLM response, using baseline probabilities")

    # Validate and normalize probabilities
    home_prob = max(0.01, min(0.98, home_prob))
    draw_prob = max(0.01, min(0.98, draw_prob))
    away_prob = max(0.01, min(0.98, away_prob))

    # Normalize to sum to 1.0
    total = home_prob + draw_prob + away_prob
    if total > 0:
        home_prob /= total
        draw_prob /= total
        away_prob /= total

    # Extract reasoning (multi-paragraph support)
    reasoning = ""
    reasoning_patterns = [
        r'REASONING:?\s*(.+?)(?:\n\s*CONFIDENCE|$)',
        r'Analysis:?\s*(.+?)(?:\n\s*CONFIDENCE|PROBABILITIES|$)',
        r'(?:My |The )?(?:analysis|reasoning|prediction)[:\s]+(.+?)(?:\n\s*CONFIDENCE|PROBABILITIES|$)',
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            # Clean up any leading newlines or excessive whitespace
            reasoning = re.sub(r'\n{3,}', '\n\n', reasoning)
            break

    if not reasoning:
        # Fallback: extract everything between PROBABILITIES and CONFIDENCE
        fallback = re.search(r'PROBABILITIES:.*?\n(.+?)CONFIDENCE:', response_text, re.DOTALL | re.IGNORECASE)
        if fallback:
            reasoning = fallback.group(1).strip()
        else:
            reasoning = response_text[:1000].strip()

    # Extract confidence
    conf_match = re.search(r'CONFIDENCE:?\s*(HIGH|MEDIUM|LOW)', response_text, re.IGNORECASE)
    confidence = conf_match.group(1).lower() if conf_match else "medium"

    return {
        "home_prob": round(home_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_prob": round(away_prob, 4),
        "reasoning": reasoning,
        "confidence": confidence,
        "parse_method": parse_method,
        "raw_response": response_text
    }


# ============================================================================
# Workflow Components Factory
# ============================================================================

def create_workflow_components(
    mcp_client,
    db_path: Path,
    kg=None,
    web_rag=None,
    ollama_model: str = "llama3.1:8b",
    use_ensemble: bool = False
) -> Dict[str, Callable]:
    """
    Factory function to create workflow nodes with dependencies injected.

    This function creates all 8 workflow nodes using closures to capture
    the dependencies (mcp_client, kg, web_rag, ollama). This allows nodes
    to be pure functions that only take state as input.

    Args:
        mcp_client: Connected MCP client for database access
        db_path: Path to SQLite database
        kg: Knowledge graph instance (FootballKnowledgeGraph or DynamicKnowledgeGraph)
        web_rag: WebSearchRAG instance for web searches
        ollama_model: LLM model name for Ollama
        use_ensemble: If True, use ensemble prediction with multiple models

    Returns:
        Dict mapping node names to node functions ready for StateGraph
    """
    import ollama

    # Initialize ensemble predictor if enabled
    ensemble_predictor = EnsemblePredictor() if use_ensemble else None

    # =========================================================================
    # Node 1: match_selector_node
    # =========================================================================
    async def match_selector_node(state: PredictionState) -> PredictionState:
        """
        Initialize workflow with match details.

        Actions:
        - Call MCP get_match to load match data
        - Initialize workflow control flags
        """
        verbose = state.get("verbose", True)
        match_id = state["match_id"]

        if verbose:
            print(f"\n📍 [MatchSelector] Loading match {match_id}...")

        try:
            match = await mcp_client.call_tool("get_match", {
                "match_id": match_id
            })

            if "error" in match:
                logger.error(f"Match not found: {match['error']}")
                return {
                    "error": f"Match not found: {match['error']}",
                    "skip_web_search": True,
                    "iteration_count": 0
                }

            if verbose:
                print(f"   ✓ {match['home_team']} vs {match['away_team']} ({match.get('date', 'Unknown')})")

            return {
                "match": match,
                "skip_web_search": state.get("skip_web_search", False),  # Preserve from initial state
                "iteration_count": 0
            }

        except Exception as e:
            logger.error(f"Error loading match: {e}")
            return {"error": str(e), "skip_web_search": True, "iteration_count": 0}

    # =========================================================================
    # Node 2: stats_collector_node
    # =========================================================================
    async def stats_collector_node(state: PredictionState) -> PredictionState:
        """
        Gather statistics: team form, baseline probabilities, advanced stats, and H2H.

        Actions:
        - Call get_team_form for home team (5 matches)
        - Call get_team_form for away team (5 matches)
        - Call get_baseline_probs for bookmaker odds
        - Calculate advanced stats for both teams
        - Calculate head-to-head statistics
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {}

        if verbose:
            print("\n📊 [StatsCollector] Gathering team statistics...")

        try:
            match = state["match"]

            # Get home team form
            home_form = await mcp_client.call_tool("get_team_form", {
                "team_name": match["home_team"],
                "window": 5
            })

            # Get away team form
            away_form = await mcp_client.call_tool("get_team_form", {
                "team_name": match["away_team"],
                "window": 5
            })

            # Get baseline probabilities
            baseline = await mcp_client.call_tool("get_baseline_probs", {
                "match_id": state["match_id"]
            })

            if verbose:
                home_str = home_form.get('form_string', 'N/A')
                away_str = away_form.get('form_string', 'N/A')
                print(f"   Home form: {home_str}")
                print(f"   Away form: {away_str}")
                print(f"   Baseline: H={baseline['home_prob']:.1%}, D={baseline['draw_prob']:.1%}, A={baseline['away_prob']:.1%}")

            # Calculate advanced statistics
            if verbose:
                print("   Calculating advanced stats...")

            stats_calc = AdvancedStatsCalculator(str(db_path))

            # Get advanced stats for both teams (before this match date)
            match_date = match.get('date')
            home_advanced = stats_calc.get_team_advanced_stats(
                match["home_team"],
                last_n=5,
                before_date=match_date
            )
            away_advanced = stats_calc.get_team_advanced_stats(
                match["away_team"],
                last_n=5,
                before_date=match_date
            )

            # Get head-to-head statistics
            h2h_stats = stats_calc.get_head_to_head_stats(
                match["home_team"],
                match["away_team"],
                last_n=5,
                before_date=match_date
            )

            # Calculate weighted form (recency bias)
            from src.data.weighted_stats import WeightedStatsCalculator
            weighted_calc = WeightedStatsCalculator(str(db_path), decay_rate=0.05)

            home_weighted_form = weighted_calc.get_weighted_form(
                match["home_team"],
                last_n=5,
                before_date=match_date
            )
            away_weighted_form = weighted_calc.get_weighted_form(
                match["away_team"],
                last_n=5,
                before_date=match_date
            )

            form_comparison = weighted_calc.compare_form_differential(
                home_weighted_form,
                away_weighted_form
            )

            if verbose:
                home_threat = home_advanced.get('efficiency', {}).get('attacking_threat', 0)
                away_threat = away_advanced.get('efficiency', {}).get('attacking_threat', 0)
                print(f"   Home attacking threat: {home_threat:.1f}/10")
                print(f"   Away attacking threat: {away_threat:.1f}/10")
                print(f"   H2H: {h2h_stats.get('matches_played', 0)} meetings, dominance: {h2h_stats.get('dominance', 'N/A')}")
                print(f"   Weighted form: Home {home_weighted_form['weighted_points_per_game']:.2f} vs Away {away_weighted_form['weighted_points_per_game']:.2f} PPG")
                print(f"   Momentum: Home {home_weighted_form['momentum_score']:.1f}/3 vs Away {away_weighted_form['momentum_score']:.1f}/3")

            return {
                "home_form": home_form,
                "away_form": away_form,
                "baseline": baseline,
                "home_advanced_stats": home_advanced,
                "away_advanced_stats": away_advanced,
                "h2h_stats": h2h_stats,
                "home_weighted_form": home_weighted_form,
                "away_weighted_form": away_weighted_form,
                "form_comparison": form_comparison,
                "skip_web_search": state.get("skip_web_search", False)  # Preserve this flag
            }

        except Exception as e:
            logger.error(f"Error collecting stats: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Node 3: kg_query_node
    # =========================================================================
    async def kg_query_node(state: PredictionState) -> PredictionState:
        """
        Query knowledge graph for tactical insights.

        Actions:
        - Get team tactical styles
        - Analyze tactical matchup
        - Return advantages/disadvantages
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {"kg_insights": {"error": "Skipped due to previous error"}}

        if verbose:
            print("\n🧠 [KGQuery] Querying knowledge graph...")

        try:
            if kg is None:
                if verbose:
                    print("   ⚠️ No knowledge graph configured")
                return {"kg_insights": {"error": "No KG configured"}}

            match = state["match"]
            kg_insights = kg.get_tactical_matchup(
                match["home_team"],
                match["away_team"]
            )

            if verbose:
                home_styles = kg_insights.get('home_styles', [])
                away_styles = kg_insights.get('away_styles', [])
                print(f"   Home styles: {', '.join(home_styles) if home_styles else 'Unknown'}")
                print(f"   Away styles: {', '.join(away_styles) if away_styles else 'Unknown'}")
                if kg_insights.get('matchup_summary'):
                    print(f"   Matchup: {kg_insights['matchup_summary'][:100]}...")

            return {"kg_insights": kg_insights}

        except Exception as e:
            logger.warning(f"KG query failed: {e}")
            if verbose:
                print(f"   ⚠️ KG query failed: {e}")
            return {"kg_insights": {"error": str(e)}}

    # =========================================================================
    # Node 4: web_search_node
    # =========================================================================
    async def web_search_node(state: PredictionState) -> PredictionState:
        """
        Execute web searches for current information.

        Actions:
        - Generate search queries based on match + KG
        - Execute searches (max 5)
        - Format results for LLM context

        Note: Only runs if confidence router decides to (not skipped)
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {"web_context": {"all_content": "Skipped due to error."}}

        if verbose:
            print("\n🔍 [WebSearch] Searching for current context...")

        try:
            if web_rag is None:
                if verbose:
                    print("   ⚠️ Web search not configured")
                return {"web_context": {"all_content": "Web search not configured."}}

            match = state["match"]

            # Generate queries incorporating KG insights
            queries = web_rag.generate_match_queries(
                match["home_team"],
                match["away_team"],
                match.get("date")
            )

            # Add KG-informed queries if available
            kg_insights = state.get("kg_insights", {})
            if kg_insights.get("home_styles"):
                style = kg_insights["home_styles"][0] if kg_insights["home_styles"] else None
                if style:
                    queries.append(f"{match['home_team']} {style} tactics analysis")

            # Execute searches
            results = web_rag.execute_searches(queries, max_searches=5)

            if verbose:
                print(f"   Executed {results['queries_executed']} searches")
                print(f"   Cache hits: {results['cached_hits']}")

            return {"web_context": results}

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            if verbose:
                print(f"   ⚠️ Web search failed: {e}")
            return {"web_context": {"error": str(e), "all_content": f"Web search error: {e}"}}

    # =========================================================================
    # Node 5: llm_predictor_node
    # =========================================================================
    async def llm_predictor_node(state: PredictionState) -> PredictionState:
        """
        Generate prediction using LLM (single or ensemble).

        Actions:
        - Build context from stats + KG + web
        - Call Ollama (single model) or EnsemblePredictor (multiple models)
        - Parse response to extract probabilities
        - Calculate confidence using objective ConfidenceCalculator (not LLM self-assessment)
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {}

        mode = "Ensemble" if use_ensemble else "Single"
        if verbose:
            print(f"\n🤖 [LLMPredictor] Generating prediction ({mode} mode)...")

        try:
            # Build prompt from all gathered context
            prompt = _build_prediction_prompt(state)

            # Use ensemble or single model
            if use_ensemble and ensemble_predictor:
                # Ensemble prediction with multiple models
                ensemble_result = ensemble_predictor.predict_ensemble(prompt, verbose=verbose)

                prediction = {
                    'home_prob': ensemble_result['home_prob'],
                    'draw_prob': ensemble_result['draw_prob'],
                    'away_prob': ensemble_result['away_prob'],
                    'reasoning': ensemble_result['reasoning'],
                    'parse_method': 'ensemble',
                    'ensemble_info': {
                        'num_models': ensemble_result['num_models'],
                        'ensemble_confidence': ensemble_result['ensemble_confidence'],
                        'individual_predictions': ensemble_result['individual_predictions']
                    }
                }
            else:
                # Single model prediction (original behavior)
                response = ollama.chat(
                    model=ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3, "num_predict": 2000}  # Increased for detailed analysis
                )
                response_text = response["message"]["content"]
                prediction = _parse_prediction_response(response_text, state["baseline"])

            # Calculate objective confidence using ConfidenceCalculator
            # (instead of trusting LLM self-assessment)
            confidence_calc = ConfidenceCalculator()
            context = {
                'baseline': state.get('baseline', {}),
                'kg_insights': state.get('kg_insights', {}),
                'web_context': state.get('web_context', {}),
                'home_form': state.get('home_form', {}),
                'away_form': state.get('away_form', {})
            }
            calculated_confidence = confidence_calc.calculate_confidence(prediction, context)

            # Store both LLM's claimed confidence and our calculated confidence
            prediction['llm_claimed_confidence'] = prediction.get('confidence', 'unknown')
            prediction['confidence'] = calculated_confidence

            if verbose:
                print(f"   Prediction: H={prediction['home_prob']:.1%}, D={prediction['draw_prob']:.1%}, A={prediction['away_prob']:.1%}")
                if use_ensemble and 'ensemble_info' in prediction:
                    print(f"   Ensemble Agreement: {prediction['ensemble_info']['ensemble_confidence']:.1%}")
                print(f"   Calculated Confidence: {calculated_confidence}")
                print(f"   Parse method: {prediction['parse_method']}")

            return {
                "prediction": prediction,
                "confidence_level": calculated_confidence,
                "iteration_count": state.get("iteration_count", 0) + 1
            }

        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            if verbose:
                print(f"   ⚠️ LLM error: {e}")

            # Fallback to baseline
            baseline = state["baseline"]
            return {
                "prediction": {
                    "home_prob": baseline["home_prob"],
                    "draw_prob": baseline["draw_prob"],
                    "away_prob": baseline["away_prob"],
                    "reasoning": f"LLM error: {e}. Using baseline.",
                    "confidence": "low",
                    "llm_claimed_confidence": "error",
                    "parse_method": "error_fallback"
                },
                "confidence_level": "low",
                "iteration_count": state.get("iteration_count", 0) + 1
            }

    # =========================================================================
    # Node 6: critique_node
    # =========================================================================
    async def critique_node(state: PredictionState) -> PredictionState:
        """
        LLM self-critique of the prediction.

        Actions:
        - Prompt LLM to critique its own prediction
        - Check for warnings or concerns
        - Return critique text
        """
        verbose = state.get("verbose", True)

        if state.get("error") or not state.get("prediction"):
            return {"critique": {"text": "No prediction to critique"}}

        if verbose:
            print("\n🔎 [Critique] Self-critiquing prediction...")

        try:
            pred = state["prediction"]
            match = state["match"]

            prompt = f"""Critique this football prediction for {match['home_team']} vs {match['away_team']}:
- Home win: {pred['home_prob']:.1%}
- Draw: {pred['draw_prob']:.1%}
- Away win: {pred['away_prob']:.1%}

Reasoning given: {pred.get('reasoning', 'N/A')[:300]}

In 1-2 sentences, identify any concerns or potential biases in this prediction."""

            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5, "num_predict": 150}
            )

            critique_text = response["message"]["content"]

            if verbose:
                print(f"   Critique: {critique_text[:100]}...")

            return {"critique": {"text": critique_text}}

        except Exception as e:
            logger.warning(f"Critique failed: {e}")
            if verbose:
                print(f"   ⚠️ Critique error: {e}")
            return {"critique": {"text": f"Critique unavailable: {e}"}}

    # =========================================================================
    # Node 7: logger_node
    # =========================================================================
    async def logger_node(state: PredictionState) -> PredictionState:
        """
        Save prediction to database via MCP.

        Actions:
        - Call log_prediction MCP tool
        - Return prediction_id
        """
        verbose = state.get("verbose", True)

        if state.get("error") or not state.get("prediction"):
            return {}

        if verbose:
            print("\n💾 [Logger] Saving prediction to database...")

        try:
            pred = state["prediction"]
            baseline = state["baseline"]

            result = await mcp_client.call_tool("log_prediction", {
                "match_id": state["match_id"],
                "baseline_home_prob": baseline["home_prob"],
                "baseline_draw_prob": baseline["draw_prob"],
                "baseline_away_prob": baseline["away_prob"],
                "llm_home_prob": pred["home_prob"],
                "llm_draw_prob": pred["draw_prob"],
                "llm_away_prob": pred["away_prob"],
                "rationale_text": pred.get("reasoning", "")[:1000],
                "timestamp": datetime.now().isoformat()
            })

            prediction_id = result.get("prediction_id")

            if verbose:
                print(f"   ✓ Logged as prediction #{prediction_id}")

            return {"prediction_id": prediction_id}

        except Exception as e:
            logger.error(f"Logging failed: {e}")
            if verbose:
                print(f"   ⚠️ Logging error: {e}")
            return {}

    # =========================================================================
    # Node 8: evaluator_node
    # =========================================================================
    async def evaluator_node(state: PredictionState) -> PredictionState:
        """
        Evaluate prediction if match is completed.

        Actions:
        - Check if match has result (home_goals not null)
        - If yes, call evaluate_prediction MCP tool
        - If no, return pending status
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {"evaluation": {"status": "error"}}

        if verbose:
            print("\n📈 [Evaluator] Checking match result...")

        try:
            match = state["match"]

            # Check if match has a result
            if match.get("home_goals") is None:
                if verbose:
                    print("   ⏳ Match not yet played - evaluation pending")
                return {"evaluation": {"status": "pending", "message": "Match not yet completed"}}

            # Match has result, evaluate
            result = await mcp_client.call_tool("evaluate_prediction", {
                "match_id": state["match_id"]
            })

            if verbose:
                actual = f"{match['home_goals']}-{match['away_goals']}"
                pred = state["prediction"]
                print(f"   Actual result: {actual}")
                if result.get("llm_brier"):
                    print(f"   LLM Brier: {result['llm_brier']:.4f}")
                    print(f"   Baseline Brier: {result.get('baseline_brier', 'N/A')}")

            return {"evaluation": result}

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            if verbose:
                print(f"   ⚠️ Evaluation error: {e}")
            return {"evaluation": {"status": "error", "error": str(e)}}

    # =========================================================================
    # Return all nodes
    # =========================================================================
    return {
        "match_selector": match_selector_node,
        "stats_collector": stats_collector_node,
        "kg_query": kg_query_node,
        "web_search": web_search_node,
        "llm_predictor": llm_predictor_node,
        "critique": critique_node,
        "logger": logger_node,
        "evaluator": evaluator_node
    }


# ============================================================================
# Routing Functions
# ============================================================================

def should_skip_web_search(state: PredictionState) -> Literal["web_search", "llm_predictor"]:
    """Decide whether to skip web search."""
    if state.get("skip_web_search", False):
        return "llm_predictor"
    if state.get("error"):
        return "llm_predictor"
    return "web_search"


def should_retry_prediction(state: PredictionState) -> Literal["llm_predictor", "critique"]:
    """Decide whether to retry prediction based on confidence."""
    # Limit iterations to prevent infinite loops
    if state.get("iteration_count", 0) >= 2:
        return "critique"
    if state.get("confidence_level") == "low":
        return "llm_predictor"  # Retry
    return "critique"


# ============================================================================
# Workflow Builder
# ============================================================================

def create_prediction_workflow(
    components: Dict[str, Callable],
    skip_web_search: bool = True
) -> StateGraph:
    """
    Create the prediction workflow graph.

    Args:
        components: Dict of node functions from create_workflow_components()
        skip_web_search: If True, always skip web search node

    Returns:
        Compiled LangGraph workflow

    Graph structure:
        match_selector -> stats_collector -> kg_query -> [web_search] -> llm_predictor
        -> critique -> logger -> evaluator -> END
    """
    # Create workflow graph
    workflow = StateGraph(PredictionState)

    # Add all 8 nodes
    workflow.add_node("match_selector", components["match_selector"])
    workflow.add_node("stats_collector", components["stats_collector"])
    workflow.add_node("kg_query", components["kg_query"])
    workflow.add_node("web_search", components["web_search"])
    workflow.add_node("llm_predictor", components["llm_predictor"])
    workflow.add_node("critique", components["critique"])
    workflow.add_node("logger", components["logger"])
    workflow.add_node("evaluator", components["evaluator"])

    # Set entry point
    workflow.set_entry_point("match_selector")

    # Linear flow: match_selector -> stats_collector -> kg_query
    workflow.add_edge("match_selector", "stats_collector")
    workflow.add_edge("stats_collector", "kg_query")

    # Conditional: skip web search?
    if skip_web_search:
        workflow.add_edge("kg_query", "llm_predictor")
    else:
        workflow.add_conditional_edges(
            "kg_query",
            should_skip_web_search,
            {
                "web_search": "web_search",
                "llm_predictor": "llm_predictor"
            }
        )
        workflow.add_edge("web_search", "llm_predictor")

    # Continue linear flow
    workflow.add_edge("llm_predictor", "critique")
    workflow.add_edge("critique", "logger")
    workflow.add_edge("logger", "evaluator")
    workflow.add_edge("evaluator", END)

    # Compile and return
    return workflow.compile()


# ============================================================================
# Convenience Function
# ============================================================================

def build_workflow(
    mcp_client,
    db_path: Path,
    kg=None,
    web_rag=None,
    ollama_model: str = "llama3.1:8b",
    skip_web_search: bool = True
):
    """
    Convenience function to build workflow in one call.

    Args:
        mcp_client: Connected MCP client
        db_path: Path to SQLite database
        kg: Knowledge graph instance
        web_rag: WebSearchRAG instance
        ollama_model: Ollama model name
        skip_web_search: Whether to skip web search

    Returns:
        Compiled LangGraph workflow
    """
    components = create_workflow_components(
        mcp_client=mcp_client,
        db_path=db_path,
        kg=kg,
        web_rag=web_rag,
        ollama_model=ollama_model
    )
    return create_prediction_workflow(components, skip_web_search=skip_web_search)


# ============================================================================
# Advanced Graph Builder with Conditional Routing
# ============================================================================

def build_prediction_graph(
    mcp_client,
    db_path: Path,
    kg=None,
    web_rag=None,
    ollama_model: str = "llama3.1:8b",
    use_ensemble: bool = False
):
    """
    Build and compile the complete prediction workflow graph with advanced routing.

    This version includes:
    - Conditional web search based on baseline confidence
    - Retry loop for low-confidence predictions
    - Iteration limiting to prevent infinite loops
    - Optional ensemble prediction using multiple models

    Args:
        mcp_client: Connected MCP client for database access
        db_path: Path to SQLite database
        kg: Knowledge graph instance
        web_rag: WebSearchRAG instance
        ollama_model: Ollama model name
        use_ensemble: If True, use ensemble prediction with multiple models

    Returns:
        Compiled LangGraph workflow ready to invoke
    """
    # Create nodes with dependencies
    nodes = create_workflow_components(mcp_client, db_path, kg, web_rag, ollama_model, use_ensemble)

    # Create graph
    graph = StateGraph(PredictionState)

    # Add all 8 nodes to graph
    graph.add_node("match_selector", nodes['match_selector'])
    graph.add_node("stats_collector", nodes['stats_collector'])
    graph.add_node("kg_query", nodes['kg_query'])
    graph.add_node("web_search", nodes['web_search'])
    graph.add_node("llm_predictor", nodes['llm_predictor'])
    graph.add_node("critique", nodes['critique'])
    graph.add_node("logger", nodes['logger'])
    graph.add_node("evaluator", nodes['evaluator'])

    # Set entry point
    graph.set_entry_point("match_selector")

    # =========================================================================
    # Wire the graph with edges
    # =========================================================================

    # Step 1: Match selector → Stats collector (always)
    graph.add_edge("match_selector", "stats_collector")

    # Step 2: Stats collector → KG query (always)
    graph.add_edge("stats_collector", "kg_query")

    # Step 3: KG query → Conditional routing (check baseline confidence)
    def route_after_kg(state: PredictionState) -> str:
        """
        Decision: Should we do web search?

        DISABLED BY DEFAULT - Web search was found to degrade accuracy by -35.6%
        Based on Phase 5 analysis, we always skip web search now.

        The flag can be overridden by setting skip_web_search=False in initial state.
        """
        # Always skip web search unless explicitly requested
        if not state.get("skip_web_search", True):  # Default to True (skip)
            verbose = state.get("verbose", True)
            if verbose:
                print(f"   🔍 Web search explicitly enabled")
            return "do_web"

        # Skip web search (default behavior)
        verbose = state.get("verbose", True)
        if verbose:
            print(f"   ⚡ Web search disabled (improves accuracy)")
        return "skip_web"

    graph.add_conditional_edges(
        "kg_query",
        route_after_kg,
        {
            "skip_web": "llm_predictor",  # Skip directly to prediction
            "do_web": "web_search"        # Do web search first
        }
    )

    # Step 4: Web search → LLM predictor (always, if we did web search)
    graph.add_edge("web_search", "llm_predictor")

    # Step 5: LLM predictor → Conditional routing (check prediction confidence)
    def route_after_prediction(state: PredictionState) -> str:
        """
        Decision: Is LLM confident enough?

        - If high confidence OR we've already retried twice, proceed to critique
        - If low confidence AND haven't retried max times, loop back to web search
        """
        confidence = state.get('confidence_level', 'medium')
        iteration_count = state.get('iteration_count', 0)
        verbose = state.get("verbose", True)

        # Always proceed if we've hit max iterations or have high confidence
        if confidence == 'high' or iteration_count >= 2:
            if verbose:
                print(f"   ✓ Confidence: {confidence} (iteration {iteration_count}) - proceeding")
            return "proceed"
        elif confidence == 'low' and iteration_count < 2:
            if verbose:
                print(f"   ⚠️ Low confidence ({confidence}) - retry {iteration_count + 1}/2")
            return "retry"
        else:
            # Medium confidence - proceed
            if verbose:
                print(f"   ✓ Confidence: {confidence} - proceeding")
            return "proceed"

    graph.add_conditional_edges(
        "llm_predictor",
        route_after_prediction,
        {
            "proceed": "critique",     # Move forward
            "retry": "web_search"      # Loop back (with updated iteration_count)
        }
    )

    # Step 6: Critique → Logger (always)
    graph.add_edge("critique", "logger")

    # Step 7: Logger → Evaluator (always)
    graph.add_edge("logger", "evaluator")

    # Step 8: Evaluator → END (always)
    graph.add_edge("evaluator", END)

    # Compile the graph
    workflow = graph.compile()

    return workflow


def print_workflow_structure():
    """Print the workflow graph structure."""
    print("\n" + "=" * 70)
    print(" LANGGRAPH WORKFLOW STRUCTURE")
    print("=" * 70)
    print("""
    ┌─────────────────┐
    │  match_selector │  ← Entry point
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ stats_collector │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │    kg_query     │
    └────────┬────────┘
             │
             ├─── baseline > 75%? ───► skip to llm_predictor
             │
    ┌────────▼────────┐
    │   web_search    │ ◄────────────┐
    └────────┬────────┘              │
             │                       │
    ┌────────▼────────┐              │
    │  llm_predictor  │              │
    └────────┬────────┘              │
             │                       │
             ├─── low confidence? ───┘ (max 2 retries)
             │
    ┌────────▼────────┐
    │    critique     │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │     logger      │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │    evaluator    │
    └────────┬────────┘
             │
           [END]
    """)


# ============================================================================
# Test
# ============================================================================

async def test_workflow():
    """Test the prediction workflow structure."""
    import sys

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print(" LANGGRAPH PREDICTION WORKFLOW TEST")
    print("=" * 70)

    # Check imports
    print("\n1. Checking imports...")
    try:
        from langgraph.graph import StateGraph, END
        print("   ✓ langgraph imported")
    except ImportError as e:
        print(f"   ✗ langgraph import failed: {e}")
        return

    try:
        import ollama
        print("   ✓ ollama imported")
    except ImportError as e:
        print(f"   ✗ ollama import failed: {e}")

    # Test state definition
    print("\n2. Testing PredictionState...")
    test_state: PredictionState = {
        "match_id": 1,
        "verbose": True,
        "skip_web_search": False,
        "iteration_count": 0
    }
    print(f"   ✓ State created with keys: {list(test_state.keys())}")

    # Test helper functions
    print("\n3. Testing helper functions...")
    test_baseline = {"home_prob": 0.4, "draw_prob": 0.3, "away_prob": 0.3}

    # Test parse with exact format
    test_response = "PROBABILITIES: H=45%, D=30%, A=25%\nREASONING: Test\nCONFIDENCE: HIGH"
    parsed = _parse_prediction_response(test_response, test_baseline)
    print(f"   ✓ Parsed exact format: {parsed['parse_method']}")
    assert parsed['parse_method'] == 'exact_format'

    # Test parse with keyword format
    test_response2 = "Home: 50%, Draw: 25%, Away: 25%"
    parsed2 = _parse_prediction_response(test_response2, test_baseline)
    print(f"   ✓ Parsed keyword format: {parsed2['parse_method']}")

    # List all 8 nodes
    print("\n4. Workflow nodes defined:")
    nodes = [
        "match_selector   - Initialize state with match details",
        "stats_collector  - Gather team form and baseline probabilities",
        "kg_query         - Query knowledge graph for tactical insights",
        "web_search       - Execute web searches (conditional)",
        "llm_predictor    - Generate LLM prediction",
        "critique         - Self-critique the prediction",
        "logger           - Save prediction to database",
        "evaluator        - Evaluate if match completed"
    ]
    for node in nodes:
        print(f"   ✓ {node}")

    # Print workflow structure
    print_workflow_structure()

    print("\n" + "=" * 70)
    print(" WORKFLOW STRUCTURE VERIFIED")
    print("=" * 70)
    print("\nTo run full workflow, use:")
    print("  workflow = build_prediction_graph(mcp_client, db_path, kg, web_rag)")
    print("  result = await workflow.ainvoke({'match_id': 1, 'verbose': True})")


async def test_full_workflow():
    """
    Test the complete workflow with a sample match.

    Requires:
    - TAVILY_API_KEY environment variable
    - Ollama running with llama3.1:8b
    - MCP server running
    """
    import sys
    import os

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print(" FULL LANGGRAPH WORKFLOW TEST")
    print("=" * 70)

    # Check Tavily key
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("\n⚠️  Set TAVILY_API_KEY environment variable to run full test")
        print("   export TAVILY_API_KEY=your_key_here")
        return

    # Check Ollama
    print("\n1. Checking Ollama...")
    try:
        import ollama
        _ = ollama.list()  # Verify connection
        print("   ✓ Ollama connected")
    except Exception as e:
        print(f"   ✗ Ollama error: {e}")
        print("   Run: ollama serve")
        return

    # Setup paths
    db_path = PROJECT_ROOT / "data" / "processed" / "asil.db"
    print(f"\n2. Database: {db_path}")
    if not db_path.exists():
        print(f"   ✗ Database not found")
        return
    print("   ✓ Database exists")

    # Initialize components
    print("\n3. Initializing components...")
    try:
        from src.kg import DynamicKnowledgeGraph
        from src.rag import WebSearchRAG

        web_rag = WebSearchRAG(tavily_api_key=tavily_key)
        print("   ✓ WebSearchRAG initialized")

        kg = DynamicKnowledgeGraph(
            db_path=str(db_path),
            web_rag=web_rag,
            ollama_model="llama3.1:8b"
        )
        print("   ✓ DynamicKnowledgeGraph initialized")
    except Exception as e:
        print(f"   ✗ Component error: {e}")
        return

    # Initialize MCP client
    print("\n4. Connecting to MCP...")
    try:
        # Try different import paths
        try:
            from src.mcp.client import MCPClient
        except ImportError:
            try:
                from mcp.client import MCPClient
            except ImportError:
                # Fallback: look for it in the project
                import importlib.util
                mcp_path = PROJECT_ROOT / "src" / "mcp" / "client.py"
                if mcp_path.exists():
                    spec = importlib.util.spec_from_file_location("mcp_client", mcp_path)
                    mcp_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mcp_module)
                    MCPClient = mcp_module.MCPClient
                else:
                    raise ImportError("MCPClient not found")

        mcp_client = MCPClient(str(db_path))
        await mcp_client.connect()
        print("   ✓ MCP connected")
    except Exception as e:
        print(f"   ✗ MCP error: {e}")
        print("   Note: Full test requires MCP server")
        return

    # Build workflow
    print("\n5. Building workflow...")
    try:
        workflow = build_prediction_graph(
            mcp_client=mcp_client,
            db_path=db_path,
            kg=kg,
            web_rag=web_rag,
            ollama_model="llama3.1:8b"
        )
        print("   ✓ Workflow compiled")
    except Exception as e:
        print(f"   ✗ Workflow error: {e}")
        return

    # Run prediction
    print("\n" + "=" * 70)
    print(" RUNNING PREDICTION FOR MATCH ID=1")
    print("=" * 70)

    try:
        result = await workflow.ainvoke({
            "match_id": 1,
            "verbose": True
        })

        print("\n" + "=" * 70)
        print(" WORKFLOW COMPLETE")
        print("=" * 70)

        if result.get("prediction"):
            pred = result["prediction"]
            print(f"\nPrediction: H={pred['home_prob']:.1%}, "
                  f"D={pred['draw_prob']:.1%}, "
                  f"A={pred['away_prob']:.1%}")
            print(f"Confidence: {pred.get('confidence', 'N/A')}")
            print(f"Parse method: {pred.get('parse_method', 'N/A')}")

        if result.get("evaluation"):
            eval_result = result["evaluation"]
            if eval_result.get("status") == "pending":
                print("\nEvaluation: Match not yet played")
            elif eval_result.get("llm_brier"):
                print(f"\nEvaluation:")
                print(f"  LLM Brier: {eval_result['llm_brier']:.4f}")
                print(f"  Baseline Brier: {eval_result.get('baseline_brier', 'N/A')}")

    except Exception as e:
        print(f"\n✗ Workflow execution error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await mcp_client.disconnect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_workflow())
