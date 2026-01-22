"""
Error Classifier for ASIL Football Prediction Project

This module implements an error taxonomy for analyzing failed predictions.
Understanding WHY predictions fail is crucial for improving the model.

Error Categories:
- DATA_MISSING: Key statistics or context unavailable
- CONTEXT_MISSED: Information existed but wasn't retrieved/used
- FORM_MISJUDGED: Over/under-weighted recent performance
- TACTICAL_MISREAD: Misinterpreted team style matchups
- CALIBRATION_ERROR: Predicted correct outcome but probability was off
- SHOCK_EVENT: Unpredictable event (red card, injury, referee error)

Usage:
    from src.agent.error_classifier import classify_error, generate_critique

    error_tags = classify_error(match_id=10, prediction_id=1)
    critique = generate_critique(match_id=10, error_tags=error_tags)
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================================
# Error Taxonomy
# ============================================================================

class ErrorTag(Enum):
    """Error categories for prediction failures."""
    DATA_MISSING = "DATA_MISSING"
    CONTEXT_MISSED = "CONTEXT_MISSED"
    FORM_MISJUDGED = "FORM_MISJUDGED"
    TACTICAL_MISREAD = "TACTICAL_MISREAD"
    CALIBRATION_ERROR = "CALIBRATION_ERROR"
    SHOCK_EVENT = "SHOCK_EVENT"


# Error tag descriptions for critique generation
ERROR_DESCRIPTIONS = {
    ErrorTag.DATA_MISSING: "Key statistics or context were unavailable at prediction time",
    ErrorTag.CONTEXT_MISSED: "Relevant information existed but wasn't properly utilized",
    ErrorTag.FORM_MISJUDGED: "Recent team performance was over or under-weighted",
    ErrorTag.TACTICAL_MISREAD: "Team style or tactical matchup was misinterpreted",
    ErrorTag.CALIBRATION_ERROR: "Probability calibration was poor despite correct outcome direction",
    ErrorTag.SHOCK_EVENT: "Match outcome was influenced by unpredictable events",
}


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp" / "sports-lab" / "dist" / "index.js"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"
FALLBACK_DB_PATH = PROJECT_ROOT / "data" / "matches.db"


# Classification thresholds
BRIER_EXCELLENT = 0.15  # Very good prediction
BRIER_GOOD = 0.25       # Acceptable calibration
BRIER_POOR = 0.40       # Poor calibration
GOAL_DIFF_SHOCK = 3     # Goal difference suggesting shock result
FORM_DIFF_SIGNIFICANT = 1.0  # PPG difference considered significant


# ============================================================================
# Database Utilities
# ============================================================================

def get_db_path() -> Path:
    """Get path to database file."""
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    elif FALLBACK_DB_PATH.exists():
        return FALLBACK_DB_PATH
    else:
        raise FileNotFoundError("Database not found")


def get_match_details(match_id: int, db_path: Optional[Path] = None) -> dict:
    """Get match details from database."""
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"Match {match_id} not found")

    return dict(row)


def get_prediction_details(prediction_id: int, db_path: Optional[Path] = None) -> dict:
    """Get prediction details from database."""
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"Prediction {prediction_id} not found")

    return dict(row)


def get_evaluation_details(match_id: int, db_path: Optional[Path] = None) -> Optional[dict]:
    """Get most recent evaluation for a match."""
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM evaluations WHERE match_id = ? ORDER BY evaluation_id DESC LIMIT 1",
        (match_id,)
    )
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


# ============================================================================
# MCP Client
# ============================================================================

class MCPClient:
    """Wrapper for MCP client session."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool and return parsed JSON response."""
        result = await self.session.call_tool(name, arguments)
        if result.content and len(result.content) > 0:
            content = result.content[0]
            if hasattr(content, 'text'):
                return json.loads(content.text)
        raise RuntimeError(f"Tool '{name}' returned empty response")


@asynccontextmanager
async def connect_to_mcp():
    """Context manager for MCP server connection."""
    server_params = StdioServerParameters(
        command="node",
        args=[str(MCP_SERVER_PATH)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPClient(session)


# ============================================================================
# Error Classification
# ============================================================================

@dataclass
class ClassificationContext:
    """Context data for error classification."""
    match: dict
    prediction: dict
    evaluation: Optional[dict]
    home_form: Optional[dict]
    away_form: Optional[dict]
    was_correct: bool
    brier_score: float
    predicted_outcome: str
    actual_outcome: str


def get_predicted_outcome(home_prob: float, draw_prob: float, away_prob: float) -> str:
    """Determine predicted outcome from probabilities."""
    if home_prob >= draw_prob and home_prob >= away_prob:
        return "H"
    elif away_prob >= home_prob and away_prob >= draw_prob:
        return "A"
    return "D"


async def gather_classification_context(
    match_id: int,
    prediction_id: int,
    mcp_client: MCPClient,
    db_path: Optional[Path] = None
) -> ClassificationContext:
    """Gather all context needed for error classification."""

    # Get basic data from database
    match = get_match_details(match_id, db_path)
    prediction = get_prediction_details(prediction_id, db_path)
    evaluation = get_evaluation_details(match_id, db_path)

    # Get team form via MCP
    try:
        home_form = await mcp_client.call_tool("get_team_form", {
            "team_name": match["home_team"],
            "window": 5
        })
        if "error" in home_form:
            home_form = None
    except Exception:
        home_form = None

    try:
        away_form = await mcp_client.call_tool("get_team_form", {
            "team_name": match["away_team"],
            "window": 5
        })
        if "error" in away_form:
            away_form = None
    except Exception:
        away_form = None

    # Determine outcomes
    predicted_outcome = get_predicted_outcome(
        prediction["llm_home_prob"],
        prediction["llm_draw_prob"],
        prediction["llm_away_prob"]
    )
    actual_outcome = match["result"]
    was_correct = predicted_outcome == actual_outcome

    # Calculate Brier score if not in evaluation
    if evaluation:
        brier_score = evaluation["llm_brier_score"]
    else:
        # Calculate manually
        actual_h = 1 if actual_outcome == "H" else 0
        actual_d = 1 if actual_outcome == "D" else 0
        actual_a = 1 if actual_outcome == "A" else 0
        brier_score = (
            (prediction["llm_home_prob"] - actual_h) ** 2 +
            (prediction["llm_draw_prob"] - actual_d) ** 2 +
            (prediction["llm_away_prob"] - actual_a) ** 2
        )

    return ClassificationContext(
        match=match,
        prediction=prediction,
        evaluation=evaluation,
        home_form=home_form,
        away_form=away_form,
        was_correct=was_correct,
        brier_score=brier_score,
        predicted_outcome=predicted_outcome,
        actual_outcome=actual_outcome,
    )


def apply_classification_rules(ctx: ClassificationContext) -> list[str]:
    """
    Apply heuristic rules to classify prediction errors.

    Rules are applied in order of specificity. A prediction can have
    multiple error tags if multiple issues contributed to the failure.

    Args:
        ctx: Classification context with all relevant data

    Returns:
        List of error tag strings (may be empty if prediction was good)
    """
    tags = []

    # Rule 0: Excellent prediction - no errors
    if ctx.was_correct and ctx.brier_score < BRIER_EXCELLENT:
        return []  # No errors to classify

    # Rule 1: Check for shock events (large goal difference)
    goal_diff = abs(ctx.match["home_goals"] - ctx.match["away_goals"])
    if goal_diff >= GOAL_DIFF_SHOCK:
        tags.append(ErrorTag.SHOCK_EVENT.value)

    # Rule 2: Check for unexpected result (underdog won by large margin)
    # If the predicted loser won convincingly, it's a shock
    predicted_winner_prob = max(
        ctx.prediction["llm_home_prob"],
        ctx.prediction["llm_away_prob"]
    )
    if not ctx.was_correct and predicted_winner_prob > 0.5 and goal_diff >= 2:
        if ErrorTag.SHOCK_EVENT.value not in tags:
            tags.append(ErrorTag.SHOCK_EVENT.value)

    # Rule 3: Calibration error - correct direction but bad probability
    if ctx.was_correct and ctx.brier_score > BRIER_GOOD:
        tags.append(ErrorTag.CALIBRATION_ERROR.value)

    # Rule 4: Severe calibration error - very high Brier score
    if ctx.brier_score > BRIER_POOR:
        if ErrorTag.CALIBRATION_ERROR.value not in tags:
            tags.append(ErrorTag.CALIBRATION_ERROR.value)

    # Rule 5: Form misjudgment - significant form difference and wrong prediction
    if ctx.home_form and ctx.away_form and not ctx.was_correct:
        home_ppg = ctx.home_form.get("points_per_game", 0)
        away_ppg = ctx.away_form.get("points_per_game", 0)
        form_diff = abs(home_ppg - away_ppg)

        if form_diff > FORM_DIFF_SIGNIFICANT:
            # Team in better form lost, or worse form won
            better_form_team = "home" if home_ppg > away_ppg else "away"
            actual_winner = "home" if ctx.actual_outcome == "H" else ("away" if ctx.actual_outcome == "A" else "draw")

            # If the team with worse form won, form was misjudged
            if actual_winner != better_form_team and actual_winner != "draw":
                tags.append(ErrorTag.FORM_MISJUDGED.value)

    # Rule 6: Data missing - no form data available
    if (ctx.home_form is None or ctx.away_form is None) and not ctx.was_correct:
        tags.append(ErrorTag.DATA_MISSING.value)

    # Rule 7: Default for wrong predictions without other tags
    if not ctx.was_correct and len(tags) == 0:
        tags.append(ErrorTag.TACTICAL_MISREAD.value)

    # Limit to 2 most relevant tags
    return tags[:2]


async def classify_error(
    match_id: int,
    prediction_id: int,
    db_path: Optional[Path] = None,
    mcp_client: Optional[MCPClient] = None
) -> list[str]:
    """
    Classify prediction errors using heuristic rules.

    Analyzes a prediction and its outcome to determine what went wrong.
    Returns a list of error tags that describe the failure modes.

    Args:
        match_id: ID of the match
        prediction_id: ID of the prediction to analyze
        db_path: Optional path to database
        mcp_client: Optional MCP client (will create one if not provided)

    Returns:
        List of error tag strings (empty if prediction was good)
    """
    if mcp_client:
        ctx = await gather_classification_context(match_id, prediction_id, mcp_client, db_path)
        return apply_classification_rules(ctx)
    else:
        async with connect_to_mcp() as client:
            ctx = await gather_classification_context(match_id, prediction_id, client, db_path)
            return apply_classification_rules(ctx)


# ============================================================================
# Database Updates
# ============================================================================

def add_error_tags_to_evaluation(
    evaluation_id: int,
    error_tags: list[str],
    db_path: Optional[Path] = None
) -> bool:
    """
    Update evaluation record with error tags.

    Args:
        evaluation_id: ID of the evaluation to update
        error_tags: List of error tag strings
        db_path: Optional path to database

    Returns:
        True if update was successful
    """
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()
        tags_json = json.dumps(error_tags)
        cursor.execute(
            "UPDATE evaluations SET error_tags = ? WHERE evaluation_id = ?",
            (tags_json, evaluation_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def add_critique_to_evaluation(
    evaluation_id: int,
    critique_text: str,
    db_path: Optional[Path] = None
) -> bool:
    """
    Update evaluation record with critique text.

    Args:
        evaluation_id: ID of the evaluation to update
        critique_text: Generated critique text
        db_path: Optional path to database

    Returns:
        True if update was successful
    """
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE evaluations SET critique_text = ? WHERE evaluation_id = ?",
            (critique_text, evaluation_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================================================
# Critique Generation
# ============================================================================

async def generate_critique(
    match_id: int,
    prediction_id: int,
    error_tags: list[str],
    db_path: Optional[Path] = None,
    mcp_client: Optional[MCPClient] = None
) -> str:
    """
    Generate human-readable critique text based on error tags.

    Uses template-based generation to create specific, actionable
    critique for each error type. Future versions could use LLM
    for more nuanced critique generation.

    Args:
        match_id: ID of the match
        prediction_id: ID of the prediction
        error_tags: List of error tags from classification
        db_path: Optional path to database
        mcp_client: Optional MCP client

    Returns:
        Human-readable critique string
    """
    # Get context data
    match = get_match_details(match_id, db_path)
    prediction = get_prediction_details(prediction_id, db_path)

    home_team = match["home_team"]
    away_team = match["away_team"]
    home_goals = match["home_goals"]
    away_goals = match["away_goals"]
    actual_result = match["result"]

    predicted_outcome = get_predicted_outcome(
        prediction["llm_home_prob"],
        prediction["llm_draw_prob"],
        prediction["llm_away_prob"]
    )

    # Get the winning probability
    if predicted_outcome == "H":
        pred_prob = prediction["llm_home_prob"]
    elif predicted_outcome == "A":
        pred_prob = prediction["llm_away_prob"]
    else:
        pred_prob = prediction["llm_draw_prob"]

    # Build critique parts
    critique_parts = []

    # Opening context
    result_map = {"H": f"{home_team} win", "D": "draw", "A": f"{away_team} win"}
    critique_parts.append(
        f"Match: {home_team} {home_goals}-{away_goals} {away_team}. "
        f"Predicted {result_map[predicted_outcome]} ({pred_prob:.0%}), "
        f"actual result was {result_map[actual_result]}."
    )

    # Tag-specific critiques
    if not error_tags:
        critique_parts.append("Prediction was accurate with good calibration.")
    else:
        for tag in error_tags:
            if tag == ErrorTag.SHOCK_EVENT.value:
                goal_diff = abs(home_goals - away_goals)
                if goal_diff >= 3:
                    critique_parts.append(
                        f"SHOCK: Unexpected {goal_diff}-goal margin suggests "
                        f"unpredictable match events (possible red card, injury, or tactical collapse)."
                    )
                else:
                    critique_parts.append(
                        "SHOCK: Result defied pre-match expectations significantly, "
                        "suggesting in-match events influenced the outcome."
                    )

            elif tag == ErrorTag.CALIBRATION_ERROR.value:
                if predicted_outcome == actual_result:
                    critique_parts.append(
                        f"CALIBRATION: Correctly predicted {result_map[actual_result]} but "
                        f"probability ({pred_prob:.0%}) was poorly calibrated. "
                        "Consider adjusting confidence levels."
                    )
                else:
                    critique_parts.append(
                        f"CALIBRATION: Assigned {pred_prob:.0%} to {result_map[predicted_outcome]}, "
                        "which was significantly overconfident given the actual outcome."
                    )

            elif tag == ErrorTag.FORM_MISJUDGED.value:
                critique_parts.append(
                    "FORM: Recent team performance was given inappropriate weight. "
                    "Form-based adjustments may have pushed probability in wrong direction. "
                    "Consider that form against weaker/stronger opponents varies in relevance."
                )

            elif tag == ErrorTag.TACTICAL_MISREAD.value:
                critique_parts.append(
                    "TACTICAL: The matchup dynamics were not correctly assessed. "
                    "Team styles, head-to-head record, or tactical setup may have been overlooked. "
                    "Consider deeper analysis of playing styles and manager tendencies."
                )

            elif tag == ErrorTag.DATA_MISSING.value:
                critique_parts.append(
                    "DATA: Key context was unavailable at prediction time. "
                    "Missing form data, injury reports, or other factors limited analysis quality."
                )

            elif tag == ErrorTag.CONTEXT_MISSED.value:
                critique_parts.append(
                    "CONTEXT: Relevant information existed but wasn't properly utilized. "
                    "Review data retrieval and ensure all available signals are considered."
                )

    # Closing recommendation
    if error_tags:
        critique_parts.append(
            f"Tags: [{', '.join(error_tags)}]. Review these factors for similar future matchups."
        )

    return " ".join(critique_parts)


# ============================================================================
# Test
# ============================================================================

async def run_test():
    """Test error classification for a match."""
    print("=" * 70)
    print(" ERROR CLASSIFIER TEST")
    print("=" * 70)

    # First, we need to make a prediction for a match and evaluate it
    # Let's use match_id = 10 and create a prediction if needed

    async with connect_to_mcp() as client:
        # Get match details
        match = await client.call_tool("get_match", {"match_id": 10})
        print(f"\nMatch 10: {match['home_team']} vs {match['away_team']}")
        print(f"Date: {match['date']}")
        print(f"Score: {match['home_goals']}-{match['away_goals']} (Result: {match['result']})")

        # Get baseline probabilities
        baseline = await client.call_tool("get_baseline_probs", {"match_id": 10})
        print(f"\nBaseline probs: H={baseline['home_prob']:.1%}, D={baseline['draw_prob']:.1%}, A={baseline['away_prob']:.1%}")

        # Log a test prediction (simulate LLM prediction)
        # Make it somewhat wrong to test error classification
        import random
        noise = random.uniform(-0.1, 0.1)
        llm_home = max(0.05, min(0.90, baseline['home_prob'] + noise))
        llm_draw = max(0.05, min(0.90, baseline['draw_prob'] - noise/2))
        llm_away = max(0.05, 1.0 - llm_home - llm_draw)

        # Normalize
        total = llm_home + llm_draw + llm_away
        llm_home = round(llm_home / total, 4)
        llm_draw = round(llm_draw / total, 4)
        llm_away = round(1.0 - llm_home - llm_draw, 4)

        print(f"LLM probs:      H={llm_home:.1%}, D={llm_draw:.1%}, A={llm_away:.1%}")

        from datetime import datetime
        log_result = await client.call_tool("log_prediction", {
            "match_id": 10,
            "baseline_home_prob": baseline['home_prob'],
            "baseline_draw_prob": baseline['draw_prob'],
            "baseline_away_prob": baseline['away_prob'],
            "llm_home_prob": llm_home,
            "llm_draw_prob": llm_draw,
            "llm_away_prob": llm_away,
            "rationale_text": "Test prediction for error classification",
            "timestamp": datetime.now().isoformat(),
        })
        prediction_id = log_result['prediction_id']
        print(f"\nLogged prediction ID: {prediction_id}")

        # Evaluate the prediction
        eval_result = await client.call_tool("evaluate_prediction", {"match_id": 10})
        print(f"\nEvaluation:")
        print(f"  Actual outcome: {eval_result['outcome']}")
        print(f"  LLM predicted:  {eval_result['llm_predicted']}")
        print(f"  LLM correct:    {eval_result['llm_correct']}")
        print(f"  LLM Brier:      {eval_result['llm_brier_score']:.4f}")

        # Get team form for context
        print("\n" + "-" * 70)
        print(" TEAM FORM CONTEXT")
        print("-" * 70)

        home_form = await client.call_tool("get_team_form", {
            "team_name": match['home_team'],
            "window": 5
        })
        away_form = await client.call_tool("get_team_form", {
            "team_name": match['away_team'],
            "window": 5
        })

        print(f"\n{match['home_team']}: {home_form['form_string']} ({home_form['points_per_game']:.2f} PPG)")
        print(f"{match['away_team']}: {away_form['form_string']} ({away_form['points_per_game']:.2f} PPG)")

        # Classify errors
        print("\n" + "-" * 70)
        print(" ERROR CLASSIFICATION")
        print("-" * 70)

        error_tags = await classify_error(10, prediction_id, mcp_client=client)
        print(f"\nError tags: {error_tags}")

        # Generate critique
        print("\n" + "-" * 70)
        print(" GENERATED CRITIQUE")
        print("-" * 70)

        critique = await generate_critique(10, prediction_id, error_tags, mcp_client=client)
        print(f"\n{critique}")

        # Update database
        print("\n" + "-" * 70)
        print(" DATABASE UPDATE")
        print("-" * 70)

        evaluation = get_evaluation_details(10)
        if evaluation:
            success1 = add_error_tags_to_evaluation(evaluation['evaluation_id'], error_tags)
            success2 = add_critique_to_evaluation(evaluation['evaluation_id'], critique)
            print(f"\nUpdated evaluation {evaluation['evaluation_id']}:")
            print(f"  Error tags saved: {success1}")
            print(f"  Critique saved:   {success2}")
        else:
            print("\nNo evaluation found to update")

    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_test())
