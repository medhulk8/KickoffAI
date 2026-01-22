"""
Enhanced Baseline Model for ASIL Football Prediction Project

This module provides an enhanced baseline that adjusts bookmaker probabilities
based on recent team form and league-wide statistics.

The key insight is that bookmaker odds are generally well-calibrated, but they
may not fully account for recent form changes. By incorporating team form data,
we can make small adjustments to capture momentum effects.

Adjustment Logic:
1. Form Adjustment: Teams in better recent form get a probability boost
   - Calculate points-per-game difference between home and away teams
   - Apply a 5% adjustment per point difference (conservative multiplier)

2. Home Advantage: Use league-wide home advantage statistics
   - Premier League typically shows ~14% more home wins than away wins
   - Apply 50% of this as an adjustment (bookmakers already price some in)

3. Normalization: Ensure probabilities sum to 1.0 and are in valid range

Usage:
    from src.data.enhanced_baseline import get_enhanced_baseline

    async with connect_to_mcp() as client:
        result = await get_enhanced_baseline(match_id=1, mcp_client=client)
        print(f"Enhanced home prob: {result['home_prob']:.1%}")
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp" / "sports-lab" / "dist" / "index.js"

# Adjustment parameters (conservative values to avoid over-fitting)
FORM_WEIGHT = 0.05  # 5% probability adjustment per point-per-game difference
HOME_ADVANTAGE_WEIGHT = 0.5  # Use 50% of league home advantage (rest is priced in)
MIN_PROB = 0.05  # Minimum probability (never say something is impossible)
MAX_PROB = 0.95  # Maximum probability (never say something is certain)


# ============================================================================
# MCP Client Wrapper
# ============================================================================

class MCPClient:
    """Wrapper for MCP client session with tool calling."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
# Enhanced Baseline Functions
# ============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a specified range."""
    return max(min_val, min(max_val, value))


def normalize_probabilities(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """
    Normalize probabilities to sum to 1.0 and clamp to valid range.

    First clamps each probability to [MIN_PROB, MAX_PROB], then normalizes
    so the total sums to 1.0.

    Args:
        home: Raw home win probability
        draw: Raw draw probability
        away: Raw away win probability

    Returns:
        Tuple of (home, draw, away) probabilities summing to 1.0
    """
    # Clamp to valid range
    home = clamp(home, MIN_PROB, MAX_PROB)
    draw = clamp(draw, MIN_PROB, MAX_PROB)
    away = clamp(away, MIN_PROB, MAX_PROB)

    # Normalize to sum to 1.0
    total = home + draw + away
    return (
        round(home / total, 4),
        round(draw / total, 4),
        round(away / total, 4)
    )


async def get_enhanced_baseline(
    match_id: int,
    mcp_client: MCPClient,
    db_path: Optional[Path] = None  # Not used but kept for API compatibility
) -> dict[str, Any]:
    """
    Calculate enhanced baseline probabilities using team form and league stats.

    This function improves upon simple bookmaker odds by incorporating:
    1. Recent team form (points per game over last 5 matches)
    2. League-wide home advantage statistics

    The adjustments are intentionally conservative to avoid over-fitting.
    Bookmaker odds are generally well-calibrated, so we only make small
    corrections based on recent form changes.

    Args:
        match_id: Unique identifier for the match
        mcp_client: Connected MCP client for tool calls
        db_path: Optional database path (not used, kept for API compatibility)

    Returns:
        Dictionary containing:
        - home_prob, draw_prob, away_prob: Enhanced probabilities
        - baseline_home, baseline_draw, baseline_away: Original bookmaker odds
        - adjustments: Details of form and home advantage adjustments
        - explanation: Human-readable explanation of the adjustments

    Raises:
        RuntimeError: If required data cannot be retrieved
    """
    # -------------------------------------------------------------------------
    # Step 1: Get match details
    # -------------------------------------------------------------------------
    match = await mcp_client.call_tool("get_match", {"match_id": match_id})

    if "error" in match:
        raise RuntimeError(f"Failed to get match: {match['error']}")

    home_team = match["home_team"]
    away_team = match["away_team"]
    season = match["season"]

    # -------------------------------------------------------------------------
    # Step 2: Get simple baseline (bookmaker odds)
    # -------------------------------------------------------------------------
    baseline = await mcp_client.call_tool("get_baseline_probs", {"match_id": match_id})

    if "error" in baseline or baseline.get("source") == "none":
        raise RuntimeError(f"Failed to get baseline probs: {baseline.get('error', 'No odds available')}")

    baseline_home = baseline["home_prob"]
    baseline_draw = baseline["draw_prob"]
    baseline_away = baseline["away_prob"]

    # -------------------------------------------------------------------------
    # Step 3: Get team form data
    # -------------------------------------------------------------------------
    home_form = await mcp_client.call_tool("get_team_form", {
        "team_name": home_team,
        "window": 5
    })

    away_form = await mcp_client.call_tool("get_team_form", {
        "team_name": away_team,
        "window": 5
    })

    if "error" in home_form:
        raise RuntimeError(f"Failed to get home team form: {home_form['error']}")
    if "error" in away_form:
        raise RuntimeError(f"Failed to get away team form: {away_form['error']}")

    # -------------------------------------------------------------------------
    # Step 4: Get league statistics
    # -------------------------------------------------------------------------
    league_stats = await mcp_client.call_tool("get_league_stats", {
        "season": season
    })

    if "error" in league_stats:
        # Fallback to all seasons if specific season fails
        league_stats = await mcp_client.call_tool("get_league_stats", {})

    # -------------------------------------------------------------------------
    # Step 5: Calculate adjustments
    # -------------------------------------------------------------------------

    # Form adjustment: Better form = higher probability
    # Points per game is typically 0-3, difference ranges from -3 to +3
    home_ppg = home_form["points_per_game"]
    away_ppg = away_form["points_per_game"]
    form_diff = home_ppg - away_ppg  # Positive = home team in better form

    # Convert form difference to probability adjustment
    # A 1-point PPG difference (significant) yields 5% adjustment
    form_adjustment = form_diff * FORM_WEIGHT

    # Home advantage adjustment
    # League home_advantage is (home_win% - away_win%), typically ~14%
    # We use 50% of this since bookmakers already price some home advantage
    league_home_adv = league_stats.get("home_advantage", 14.0) / 100  # Convert to decimal
    home_advantage_adj = league_home_adv * HOME_ADVANTAGE_WEIGHT

    # -------------------------------------------------------------------------
    # Step 6: Apply adjustments
    # -------------------------------------------------------------------------

    # Adjust probabilities
    # Form adjustment affects home/away, home advantage adds to home
    enhanced_home = baseline_home + form_adjustment + home_advantage_adj
    enhanced_away = baseline_away - form_adjustment - home_advantage_adj

    # Draw probability absorbs the remaining
    enhanced_draw = 1.0 - enhanced_home - enhanced_away

    # -------------------------------------------------------------------------
    # Step 7: Normalize and clamp
    # -------------------------------------------------------------------------
    enhanced_home, enhanced_draw, enhanced_away = normalize_probabilities(
        enhanced_home, enhanced_draw, enhanced_away
    )

    # -------------------------------------------------------------------------
    # Step 8: Generate explanation
    # -------------------------------------------------------------------------
    explanation_parts = []

    # Form explanation
    if abs(form_diff) > 0.3:
        better_team = home_team if form_diff > 0 else away_team
        explanation_parts.append(
            f"{better_team} in better recent form "
            f"(PPG: {home_ppg:.2f} vs {away_ppg:.2f}, diff: {form_diff:+.2f})"
        )

    # Home advantage explanation
    explanation_parts.append(
        f"Home advantage adjustment: {home_advantage_adj:+.1%} "
        f"(league avg: {league_home_adv:.1%})"
    )

    # Net effect
    home_change = enhanced_home - baseline_home
    if abs(home_change) > 0.01:
        direction = "increased" if home_change > 0 else "decreased"
        explanation_parts.append(
            f"Net effect: {home_team} win probability {direction} by {abs(home_change):.1%}"
        )

    explanation = ". ".join(explanation_parts) + "."

    # -------------------------------------------------------------------------
    # Return result
    # -------------------------------------------------------------------------
    return {
        "home_prob": enhanced_home,
        "draw_prob": enhanced_draw,
        "away_prob": enhanced_away,
        "baseline_home": baseline_home,
        "baseline_draw": baseline_draw,
        "baseline_away": baseline_away,
        "home_team": home_team,
        "away_team": away_team,
        "adjustments": {
            "form_diff": round(form_diff, 3),
            "form_adjustment": round(form_adjustment, 4),
            "home_advantage": round(home_advantage_adj, 4),
            "total_home_adjustment": round(form_adjustment + home_advantage_adj, 4),
        },
        "form_data": {
            "home_ppg": home_ppg,
            "home_form": home_form["form_string"],
            "away_ppg": away_ppg,
            "away_form": away_form["form_string"],
        },
        "explanation": explanation,
    }


async def compare_baselines(
    match_id: int,
    mcp_client: MCPClient,
    db_path: Optional[Path] = None
) -> dict[str, Any]:
    """
    Compare simple baseline (bookmaker odds) with enhanced baseline.

    Runs both models and shows a side-by-side comparison to illustrate
    the effect of form and home advantage adjustments.

    Args:
        match_id: Unique identifier for the match
        mcp_client: Connected MCP client for tool calls
        db_path: Optional database path

    Returns:
        Dictionary with both predictions and comparison metrics
    """
    # Get simple baseline
    simple = await mcp_client.call_tool("get_baseline_probs", {"match_id": match_id})

    # Get enhanced baseline
    enhanced = await get_enhanced_baseline(match_id, mcp_client, db_path)

    # Calculate differences
    home_diff = enhanced["home_prob"] - simple["home_prob"]
    draw_diff = enhanced["draw_prob"] - simple["draw_prob"]
    away_diff = enhanced["away_prob"] - simple["away_prob"]

    return {
        "match_id": match_id,
        "home_team": enhanced["home_team"],
        "away_team": enhanced["away_team"],
        "simple_baseline": {
            "home_prob": simple["home_prob"],
            "draw_prob": simple["draw_prob"],
            "away_prob": simple["away_prob"],
            "source": simple["source"],
        },
        "enhanced_baseline": {
            "home_prob": enhanced["home_prob"],
            "draw_prob": enhanced["draw_prob"],
            "away_prob": enhanced["away_prob"],
        },
        "differences": {
            "home_diff": round(home_diff, 4),
            "draw_diff": round(draw_diff, 4),
            "away_diff": round(away_diff, 4),
        },
        "adjustments": enhanced["adjustments"],
        "form_data": enhanced["form_data"],
        "explanation": enhanced["explanation"],
    }


# ============================================================================
# Test
# ============================================================================

async def run_test():
    """Test enhanced baseline for match_id = 1."""
    print("=" * 70)
    print(" ENHANCED BASELINE MODEL TEST")
    print("=" * 70)

    async with connect_to_mcp() as client:
        # Get match details first
        match = await client.call_tool("get_match", {"match_id": 1})
        print(f"\nMatch: {match['home_team']} vs {match['away_team']}")
        print(f"Date: {match['date']} (Season {match['season']})")
        print(f"Actual Result: {match['home_goals']}-{match['away_goals']} ({match['result']})")

        # Run comparison
        print("\n" + "-" * 70)
        print(" BOOKMAKER ODDS (Simple Baseline)")
        print("-" * 70)

        baseline = await client.call_tool("get_baseline_probs", {"match_id": 1})
        print(f"Source: {baseline['source']}")
        print(f"  Home Win:  {baseline['home_prob']:.1%}")
        print(f"  Draw:      {baseline['draw_prob']:.1%}")
        print(f"  Away Win:  {baseline['away_prob']:.1%}")

        print("\n" + "-" * 70)
        print(" TEAM FORM (Last 5 matches)")
        print("-" * 70)

        home_form = await client.call_tool("get_team_form", {
            "team_name": match['home_team'],
            "window": 5
        })
        away_form = await client.call_tool("get_team_form", {
            "team_name": match['away_team'],
            "window": 5
        })

        print(f"\n{match['home_team']} (Home):")
        print(f"  Form: {home_form['form_string']}")
        print(f"  W-D-L: {home_form['wins']}-{home_form['draws']}-{home_form['losses']}")
        print(f"  Points: {home_form['points']} ({home_form['points_per_game']:.2f} PPG)")
        print(f"  Goals: {home_form['goals_scored']} scored, {home_form['goals_conceded']} conceded")

        print(f"\n{match['away_team']} (Away):")
        print(f"  Form: {away_form['form_string']}")
        print(f"  W-D-L: {away_form['wins']}-{away_form['draws']}-{away_form['losses']}")
        print(f"  Points: {away_form['points']} ({away_form['points_per_game']:.2f} PPG)")
        print(f"  Goals: {away_form['goals_scored']} scored, {away_form['goals_conceded']} conceded")

        print("\n" + "-" * 70)
        print(" ENHANCED BASELINE CALCULATION")
        print("-" * 70)

        enhanced = await get_enhanced_baseline(1, client)

        print(f"\nAdjustments:")
        adj = enhanced['adjustments']
        print(f"  Form difference (PPG):    {adj['form_diff']:+.3f}")
        print(f"  Form adjustment:          {adj['form_adjustment']:+.4f} ({adj['form_adjustment']*100:+.2f}%)")
        print(f"  Home advantage:           {adj['home_advantage']:+.4f} ({adj['home_advantage']*100:+.2f}%)")
        print(f"  Total home adjustment:    {adj['total_home_adjustment']:+.4f} ({adj['total_home_adjustment']*100:+.2f}%)")

        print("\n" + "-" * 70)
        print(" FINAL PROBABILITIES COMPARISON")
        print("-" * 70)

        print(f"\n{'Outcome':<12} {'Bookmaker':>12} {'Enhanced':>12} {'Change':>12}")
        print("-" * 50)
        print(f"{'Home Win':<12} {enhanced['baseline_home']:>11.1%} {enhanced['home_prob']:>11.1%} {enhanced['home_prob']-enhanced['baseline_home']:>+11.1%}")
        print(f"{'Draw':<12} {enhanced['baseline_draw']:>11.1%} {enhanced['draw_prob']:>11.1%} {enhanced['draw_prob']-enhanced['baseline_draw']:>+11.1%}")
        print(f"{'Away Win':<12} {enhanced['baseline_away']:>11.1%} {enhanced['away_prob']:>11.1%} {enhanced['away_prob']-enhanced['baseline_away']:>+11.1%}")

        print("\n" + "-" * 70)
        print(" EXPLANATION")
        print("-" * 70)
        print(f"\n{enhanced['explanation']}")

        # Check prediction vs actual
        print("\n" + "-" * 70)
        print(" PREDICTION VS ACTUAL")
        print("-" * 70)

        probs = [
            ('H', enhanced['home_prob']),
            ('D', enhanced['draw_prob']),
            ('A', enhanced['away_prob'])
        ]
        predicted = max(probs, key=lambda x: x[1])[0]
        actual = match['result']

        print(f"\nEnhanced model predicted: {predicted}")
        print(f"Actual outcome: {actual}")
        print(f"Correct: {'Yes ✓' if predicted == actual else 'No ✗'}")

    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_test())
