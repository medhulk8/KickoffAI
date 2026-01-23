"""
Prompt Chain for ASIL Football Prediction Project

Implements a 3-step reasoning process for match prediction:
1. Build Context: Gather all relevant information
2. Make Prediction: Generate prediction with reasoning
3. Self-Critique: Validate prediction before logging

This module prepares the structure for LLM integration (Phase 3),
but currently uses template-based reasoning as a placeholder.

The chain-of-thought approach mirrors how an LLM would reason:
- First, gather and summarize all available data
- Then, reason through the prediction step by step
- Finally, self-critique to catch potential errors

Usage:
    from src.agent.prompt_chain import run_full_chain

    async with connect_to_mcp() as client:
        result = await run_full_chain(match_id=1, mcp_client=client)
        print(f"Prediction: {result['prediction']}")
"""

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.enhanced_baseline import get_enhanced_baseline, MCPClient as EnhancedMCPClient
from src.agent.error_classifier import classify_error, generate_critique


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp" / "sports-lab" / "dist" / "index.js"


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
# Step 1: Build Context
# ============================================================================

async def step1_build_context(
    match_id: int,
    mcp_client: MCPClient,
    db_path: Optional[Path] = None
) -> dict[str, Any]:
    """
    Gather all relevant information for the match.

    This step collects comprehensive context that would be used by an LLM
    to reason about the prediction. It includes:
    - Basic match information
    - Both teams' recent form
    - League-wide statistics
    - Enhanced baseline probabilities

    Args:
        match_id: ID of the match to analyze
        mcp_client: Connected MCP client
        db_path: Optional database path

    Returns:
        Dictionary containing:
        - match: Basic match info
        - home_form: Home team's recent form
        - away_form: Away team's recent form
        - league_stats: League statistics for the season
        - enhanced_baseline: Form-adjusted probabilities
        - context_summary: Human-readable summary string
    """
    context = {
        "match_id": match_id,
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Get match basic info
    match = await mcp_client.call_tool("get_match", {"match_id": match_id})
    if "error" in match:
        raise RuntimeError(f"Failed to get match: {match['error']}")
    context["match"] = match

    home_team = match["home_team"]
    away_team = match["away_team"]
    season = match["season"]

    # 2. Get home team form
    home_form = await mcp_client.call_tool("get_team_form", {
        "team_name": home_team,
        "window": 5
    })
    context["home_form"] = home_form if "error" not in home_form else None

    # 3. Get away team form
    away_form = await mcp_client.call_tool("get_team_form", {
        "team_name": away_team,
        "window": 5
    })
    context["away_form"] = away_form if "error" not in away_form else None

    # 4. Get league statistics
    league_stats = await mcp_client.call_tool("get_league_stats", {
        "season": season
    })
    context["league_stats"] = league_stats if "error" not in league_stats else None

    # 5. Get enhanced baseline
    # Create an EnhancedMCPClient wrapper for the enhanced_baseline module
    enhanced_client = EnhancedMCPClient(mcp_client.session)
    try:
        enhanced = await get_enhanced_baseline(match_id, enhanced_client)
        context["enhanced_baseline"] = enhanced
    except Exception as e:
        context["enhanced_baseline"] = None
        context["enhanced_baseline_error"] = str(e)

    # 6. Generate context summary
    context["context_summary"] = _generate_context_summary(context)

    return context


def _generate_context_summary(context: dict) -> str:
    """Generate a human-readable summary of the match context."""
    match = context["match"]
    home_form = context.get("home_form")
    away_form = context.get("away_form")
    enhanced = context.get("enhanced_baseline")

    parts = []

    # Match header
    parts.append(f"{match['home_team']} (home) vs {match['away_team']} (away)")
    parts.append(f"Date: {match['date']}, Season: {match['season']}")

    # Form comparison
    if home_form and away_form:
        home_ppg = home_form.get("points_per_game", 0)
        away_ppg = away_form.get("points_per_game", 0)
        ppg_diff = home_ppg - away_ppg

        parts.append(
            f"\nForm (last 5): {match['home_team']} {home_form.get('form_string', '?')} "
            f"({home_ppg:.2f} PPG) vs {match['away_team']} {away_form.get('form_string', '?')} "
            f"({away_ppg:.2f} PPG)"
        )
        parts.append(f"PPG difference: {ppg_diff:+.2f} (positive favors home)")

    # Baseline probabilities
    if enhanced:
        parts.append(
            f"\nBookmaker odds: H={enhanced['baseline_home']:.1%}, "
            f"D={enhanced['baseline_draw']:.1%}, A={enhanced['baseline_away']:.1%}"
        )
        parts.append(
            f"Enhanced baseline: H={enhanced['home_prob']:.1%}, "
            f"D={enhanced['draw_prob']:.1%}, A={enhanced['away_prob']:.1%}"
        )

        adj = enhanced.get("adjustments", {})
        if adj:
            parts.append(
                f"Adjustments: Form={adj.get('form_adjustment', 0)*100:+.1f}%, "
                f"Home adv={adj.get('home_advantage', 0)*100:+.1f}%"
            )

    # Actual result (if match completed)
    if match.get("home_goals") is not None:
        parts.append(
            f"\nActual result: {match['home_team']} {match['home_goals']}-{match['away_goals']} "
            f"{match['away_team']} ({match['result']})"
        )

    return "\n".join(parts)


# ============================================================================
# Step 2: Make Prediction
# ============================================================================

def step2_make_prediction(
    context: dict[str, Any],
    reasoning_mode: str = "template"
) -> dict[str, Any]:
    """
    Generate prediction with reasoning.

    This step produces a prediction along with chain-of-thought reasoning.
    Currently uses template-based reasoning; future versions will use LLM.

    Args:
        context: Context dictionary from step1
        reasoning_mode: "template" (current) or "llm" (future)

    Returns:
        Dictionary containing:
        - home_prob, draw_prob, away_prob: Predicted probabilities
        - reasoning: Chain-of-thought explanation
        - confidence: "high", "medium", or "low"
    """
    if reasoning_mode != "template":
        raise NotImplementedError(f"Reasoning mode '{reasoning_mode}' not yet implemented")

    enhanced = context.get("enhanced_baseline")
    match = context["match"]
    home_form = context.get("home_form")
    away_form = context.get("away_form")

    if not enhanced:
        # Fallback to basic prediction
        return {
            "home_prob": 0.33,
            "draw_prob": 0.34,
            "away_prob": 0.33,
            "reasoning": "Insufficient data for prediction. Using uniform probabilities.",
            "confidence": "low",
        }

    # Start with enhanced baseline
    home_prob = enhanced["home_prob"]
    draw_prob = enhanced["draw_prob"]
    away_prob = enhanced["away_prob"]

    # Add small random adjustment to simulate LLM reasoning (-3% to +3%)
    adjustment = random.uniform(-0.03, 0.03)
    home_prob = max(0.05, min(0.90, home_prob + adjustment))
    draw_prob = max(0.05, min(0.90, draw_prob - adjustment / 2))
    away_prob = 1.0 - home_prob - draw_prob
    away_prob = max(0.05, min(0.90, away_prob))

    # Normalize
    total = home_prob + draw_prob + away_prob
    home_prob = round(home_prob / total, 4)
    draw_prob = round(draw_prob / total, 4)
    away_prob = round(1.0 - home_prob - draw_prob, 4)

    # Generate reasoning
    reasoning = _generate_reasoning(context, home_prob, draw_prob, away_prob, adjustment)

    # Determine confidence
    confidence = _determine_confidence(context, home_prob, draw_prob, away_prob)

    return {
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def _generate_reasoning(
    context: dict,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    adjustment: float
) -> str:
    """Generate template-based reasoning for the prediction."""
    match = context["match"]
    home_form = context.get("home_form")
    away_form = context.get("away_form")
    enhanced = context.get("enhanced_baseline")

    home_team = match["home_team"]
    away_team = match["away_team"]

    parts = []

    # Opening assessment
    favorite = "home" if home_prob > away_prob else ("away" if away_prob > home_prob else "neither")

    if favorite == "home":
        parts.append(f"Analysis favors {home_team} (home) at {home_prob:.1%}.")
    elif favorite == "away":
        parts.append(f"Analysis favors {away_team} (away) at {away_prob:.1%}.")
    else:
        parts.append(f"Match appears evenly balanced with draw at {draw_prob:.1%}.")

    # Form-based reasoning
    if home_form and away_form:
        home_ppg = home_form.get("points_per_game", 0)
        away_ppg = away_form.get("points_per_game", 0)
        ppg_diff = home_ppg - away_ppg

        if abs(ppg_diff) > 1.0:
            better_team = home_team if ppg_diff > 0 else away_team
            parts.append(
                f"{better_team}'s superior recent form ({abs(ppg_diff):.2f} PPG advantage) "
                f"is a significant factor."
            )
        elif abs(ppg_diff) > 0.3:
            parts.append(f"Form difference ({ppg_diff:+.2f} PPG) provides slight edge.")
        else:
            parts.append("Both teams show similar recent form.")

        # Specific form details
        home_str = home_form.get("form_string", "")
        away_str = away_form.get("form_string", "")

        if "WWW" in home_str:
            parts.append(f"{home_team} on a winning streak.")
        elif "LLL" in home_str:
            parts.append(f"{home_team} struggling with 3+ consecutive losses.")

        if "WWW" in away_str:
            parts.append(f"{away_team} on a winning streak.")
        elif "LLL" in away_str:
            parts.append(f"{away_team} struggling with 3+ consecutive losses.")

    # Home advantage consideration
    if enhanced:
        home_adv = enhanced.get("adjustments", {}).get("home_advantage", 0)
        if home_adv > 0.03:
            parts.append(f"Home advantage adjustment of {home_adv:.1%} applied.")

    # Adjustment explanation
    if abs(adjustment) > 0.01:
        direction = "increased" if adjustment > 0 else "decreased"
        parts.append(
            f"Reasoning-based adjustment {direction} home probability by {abs(adjustment):.1%}."
        )

    # Closing
    parts.append(f"Final prediction: H={home_prob:.1%}, D={draw_prob:.1%}, A={away_prob:.1%}.")

    return " ".join(parts)


def _determine_confidence(
    context: dict,
    home_prob: float,
    draw_prob: float,
    away_prob: float
) -> str:
    """Determine prediction confidence level."""
    home_form = context.get("home_form")
    away_form = context.get("away_form")

    # High confidence if clear favorite and good form data
    max_prob = max(home_prob, draw_prob, away_prob)

    if max_prob > 0.55 and home_form and away_form:
        return "high"
    elif max_prob > 0.40 and (home_form or away_form):
        return "medium"
    else:
        return "low"


# ============================================================================
# Step 3: Self-Critique
# ============================================================================

def step3_self_critique(
    prediction: dict[str, Any],
    context: dict[str, Any]
) -> dict[str, Any]:
    """
    Critique the prediction before logging.

    This step validates the prediction and identifies potential issues.
    It helps catch obvious errors before committing the prediction.

    Args:
        prediction: Prediction dictionary from step2
        context: Context dictionary from step1

    Returns:
        Dictionary containing:
        - critique: Main critique text
        - warnings: List of warning strings
        - recommended_confidence: Adjusted confidence level
    """
    warnings = []
    critique_parts = []

    enhanced = context.get("enhanced_baseline")
    home_form = context.get("home_form")
    away_form = context.get("away_form")

    # Check 1: Deviation from market consensus
    if enhanced:
        baseline_home = enhanced["baseline_home"]
        pred_home = prediction["home_prob"]
        deviation = abs(pred_home - baseline_home)

        if deviation > 0.10:
            warnings.append(
                f"Large deviation from market: {deviation:.1%} difference from bookmaker odds"
            )
            critique_parts.append(
                f"Warning: Prediction deviates significantly from market consensus "
                f"({pred_home:.1%} vs market {baseline_home:.1%}). Verify reasoning."
            )

    # Check 2: Form difference analysis
    if home_form and away_form:
        home_ppg = home_form.get("points_per_game", 0)
        away_ppg = away_form.get("points_per_game", 0)
        form_diff = abs(home_ppg - away_ppg)

        if form_diff > 2.0:
            warnings.append(
                f"Extreme form difference: {form_diff:.2f} PPG"
            )
            critique_parts.append(
                "Caution: Large form difference may be influenced by opponent strength. "
                "Consider quality of recent opponents."
            )

        # Check for form volatility
        home_str = home_form.get("form_string", "")
        away_str = away_form.get("form_string", "")

        if ("W" in home_str and "L" in home_str) or ("W" in away_str and "L" in away_str):
            if "WL" in home_str or "LW" in home_str or "WL" in away_str or "LW" in away_str:
                warnings.append("Inconsistent form detected - results may be unpredictable")

    # Check 3: Draw probability sanity
    if prediction["draw_prob"] > 0.40:
        warnings.append("High draw probability - typically draws are around 25%")
        critique_parts.append(
            "Note: Draw probability above 40% is unusual. Most leagues see ~25% draws."
        )

    # Check 4: Extreme confidence
    max_prob = max(prediction["home_prob"], prediction["draw_prob"], prediction["away_prob"])
    if max_prob > 0.80:
        warnings.append(f"Very high confidence ({max_prob:.1%}) - upsets do happen")
        critique_parts.append(
            f"Caution: {max_prob:.1%} confidence is very high. "
            "Even strong favorites occasionally lose."
        )

    # Generate main critique
    if not warnings:
        critique = (
            f"Prediction aligns with available evidence. "
            f"Confidence: {prediction['confidence']}. "
            f"No significant concerns identified."
        )
    else:
        critique = " ".join(critique_parts)

    # Adjust confidence based on warnings
    recommended_confidence = prediction["confidence"]
    if len(warnings) >= 2:
        recommended_confidence = "low"
    elif len(warnings) == 1 and prediction["confidence"] == "high":
        recommended_confidence = "medium"

    return {
        "critique": critique,
        "warnings": warnings,
        "recommended_confidence": recommended_confidence,
    }


# ============================================================================
# Full Chain Orchestration
# ============================================================================

async def run_full_chain(
    match_id: int,
    mcp_client: MCPClient,
    db_path: Optional[Path] = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run the complete prediction chain.

    Orchestrates all 3 steps:
    1. Build context
    2. Make prediction
    3. Self-critique
    4. Log prediction to database
    5. Evaluate (if match completed)
    6. Classify errors (if prediction was wrong)

    Args:
        match_id: ID of the match to predict
        mcp_client: Connected MCP client
        db_path: Optional database path
        verbose: Whether to print detailed output

    Returns:
        Dictionary containing all chain results
    """
    result = {
        "match_id": match_id,
        "success": False,
        "timestamp": datetime.now().isoformat(),
    }

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"\n{'='*70}")
    log(f" PREDICTION CHAIN - Match {match_id}")
    log(f"{'='*70}")

    # -------------------------------------------------------------------------
    # Step 1: Build Context
    # -------------------------------------------------------------------------
    log(f"\n[Step 1] Building context...")

    try:
        context = await step1_build_context(match_id, mcp_client, db_path)
        result["context"] = context
        log(f"\n{context['context_summary']}")
    except Exception as e:
        log(f"  Error: {e}")
        result["error"] = str(e)
        return result

    # -------------------------------------------------------------------------
    # Step 2: Make Prediction
    # -------------------------------------------------------------------------
    log(f"\n[Step 2] Making prediction...")

    prediction = step2_make_prediction(context)
    result["prediction"] = prediction

    log(f"\n  Probabilities: H={prediction['home_prob']:.1%}, "
        f"D={prediction['draw_prob']:.1%}, A={prediction['away_prob']:.1%}")
    log(f"  Confidence: {prediction['confidence']}")
    log(f"\n  Reasoning: {prediction['reasoning']}")

    # -------------------------------------------------------------------------
    # Step 3: Self-Critique
    # -------------------------------------------------------------------------
    log(f"\n[Step 3] Self-critique...")

    critique = step3_self_critique(prediction, context)
    result["critique"] = critique

    log(f"\n  {critique['critique']}")
    if critique["warnings"]:
        log(f"  Warnings:")
        for w in critique["warnings"]:
            log(f"    - {w}")
    log(f"  Recommended confidence: {critique['recommended_confidence']}")

    # -------------------------------------------------------------------------
    # Step 4: Log Prediction
    # -------------------------------------------------------------------------
    log(f"\n[Step 4] Logging prediction...")

    enhanced = context.get("enhanced_baseline", {})
    baseline_home = enhanced.get("baseline_home", prediction["home_prob"])
    baseline_draw = enhanced.get("baseline_draw", prediction["draw_prob"])
    baseline_away = enhanced.get("baseline_away", prediction["away_prob"])

    try:
        log_result = await mcp_client.call_tool("log_prediction", {
            "match_id": match_id,
            "baseline_home_prob": baseline_home,
            "baseline_draw_prob": baseline_draw,
            "baseline_away_prob": baseline_away,
            "llm_home_prob": prediction["home_prob"],
            "llm_draw_prob": prediction["draw_prob"],
            "llm_away_prob": prediction["away_prob"],
            "rationale_text": prediction["reasoning"],
            "timestamp": datetime.now().isoformat(),
        })
        result["prediction_id"] = log_result["prediction_id"]
        log(f"  Logged prediction ID: {log_result['prediction_id']}")
    except Exception as e:
        log(f"  Error logging: {e}")
        result["log_error"] = str(e)

    # -------------------------------------------------------------------------
    # Step 5: Evaluate (if match completed)
    # -------------------------------------------------------------------------
    match = context["match"]
    if match.get("home_goals") is not None:
        log(f"\n[Step 5] Evaluating prediction...")

        try:
            eval_result = await mcp_client.call_tool("evaluate_prediction", {
                "match_id": match_id
            })
            result["evaluation"] = eval_result

            log(f"  Actual: {eval_result['outcome']}")
            log(f"  Predicted: {eval_result['llm_predicted']}")
            log(f"  Correct: {'Yes ✓' if eval_result['llm_correct'] else 'No ✗'}")
            log(f"  Brier: {eval_result['llm_brier_score']:.4f}")

            # -------------------------------------------------------------------------
            # Step 6: Error Classification (if wrong)
            # -------------------------------------------------------------------------
            if not eval_result["llm_correct"] and "prediction_id" in result:
                log(f"\n[Step 6] Classifying error...")

                error_tags = await classify_error(
                    match_id,
                    result["prediction_id"],
                    mcp_client=mcp_client
                )
                result["error_tags"] = error_tags
                log(f"  Error tags: {error_tags}")

                error_critique = await generate_critique(
                    match_id,
                    result["prediction_id"],
                    error_tags,
                    mcp_client=mcp_client
                )
                result["error_critique"] = error_critique
                log(f"  Error critique: {error_critique[:200]}...")

        except Exception as e:
            log(f"  Evaluation error: {e}")
            result["eval_error"] = str(e)
    else:
        log(f"\n[Step 5] Match not yet played - skipping evaluation")

    result["success"] = True

    log(f"\n{'='*70}")
    log(f" CHAIN COMPLETE")
    log(f"{'='*70}")

    return result


# ============================================================================
# Test
# ============================================================================

async def run_test():
    """Test the prediction chain on 3 different matches."""
    print("=" * 70)
    print(" PROMPT CHAIN TEST - 3 MATCHES")
    print("=" * 70)

    # Match selections:
    # Match 1: First match (Brentford upset Arsenal)
    # Match 50: Mid-season match
    # Match 100: Another sample

    test_matches = [
        (1, "Opening day upset"),
        (50, "Mid-table clash"),
        (100, "Sample match"),
    ]

    async with connect_to_mcp() as client:
        results = []

        for match_id, description in test_matches:
            print(f"\n\n{'#'*70}")
            print(f" TEST: Match {match_id} - {description}")
            print(f"{'#'*70}")

            result = await run_full_chain(match_id, client, verbose=True)
            results.append(result)

            # Brief pause between matches
            await asyncio.sleep(0.5)

        # Summary
        print(f"\n\n{'='*70}")
        print(" SUMMARY")
        print("=" * 70)

        for i, (match_id, description) in enumerate(test_matches):
            result = results[i]
            print(f"\nMatch {match_id} ({description}):")

            if "context" in result:
                match = result["context"]["match"]
                print(f"  {match['home_team']} vs {match['away_team']}")
                print(f"  Score: {match['home_goals']}-{match['away_goals']} ({match['result']})")

            if "prediction" in result:
                pred = result["prediction"]
                print(f"  Prediction: H={pred['home_prob']:.1%}, D={pred['draw_prob']:.1%}, A={pred['away_prob']:.1%}")
                print(f"  Confidence: {pred['confidence']}")

            if "evaluation" in result:
                eval_r = result["evaluation"]
                print(f"  Predicted: {eval_r['llm_predicted']}, Actual: {eval_r['outcome']}")
                print(f"  Correct: {'Yes ✓' if eval_r['llm_correct'] else 'No ✗'}")
                print(f"  Brier: {eval_r['llm_brier_score']:.4f}")

            if "error_tags" in result:
                print(f"  Error tags: {result['error_tags']}")


if __name__ == "__main__":
    asyncio.run(run_test())
