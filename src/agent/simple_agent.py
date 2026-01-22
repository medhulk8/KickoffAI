"""
Simple Agent for ASIL Football Prediction Project

This agent connects to the Sports Lab MCP server and runs a basic
prediction workflow:
1. Get match details
2. Get baseline probabilities
3. Generate LLM prediction (placeholder: adds noise to baseline)
4. Log prediction to database
5. Evaluate prediction against actual outcome

Usage:
    python src/agent/simple_agent.py
"""

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp" / "sports-lab" / "dist" / "index.js"


# ============================================================================
# MCP Client Wrapper
# ============================================================================

class SportsLabClient:
    """
    Client wrapper for the Sports Lab MCP server.

    Provides a clean interface for calling MCP tools and parsing responses.
    """

    def __init__(self, session: ClientSession):
        self.session = session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call an MCP tool and return the parsed JSON response.

        Args:
            name: Tool name (e.g., "get_match", "log_prediction")
            arguments: Tool arguments as a dictionary

        Returns:
            Parsed JSON response from the tool

        Raises:
            RuntimeError: If the tool call fails
        """
        result = await self.session.call_tool(name, arguments)

        # Extract text content from response
        if result.content and len(result.content) > 0:
            content = result.content[0]
            if hasattr(content, 'text'):
                return json.loads(content.text)

        raise RuntimeError(f"Tool '{name}' returned empty or invalid response")


@asynccontextmanager
async def connect_to_server():
    """
    Context manager that connects to the Sports Lab MCP server.

    Yields:
        SportsLabClient instance for calling tools
    """
    server_params = StdioServerParameters(
        command="node",
        args=[str(MCP_SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield SportsLabClient(session)


# ============================================================================
# Prediction Workflow
# ============================================================================

async def predict_match(client: SportsLabClient, match_id: int) -> dict[str, Any]:
    """
    Run the full prediction workflow for a match.

    Steps:
    1. Get match details from database
    2. Get baseline probabilities (market odds)
    3. Generate LLM prediction (placeholder: adds noise to baseline)
    4. Log prediction to database
    5. Evaluate prediction against actual outcome

    Args:
        client: Connected SportsLabClient instance
        match_id: ID of the match to predict

    Returns:
        Dictionary containing all prediction and evaluation data
    """
    print("\n" + "=" * 60)
    print(f" PREDICTING MATCH {match_id}")
    print("=" * 60)

    result = {
        "match_id": match_id,
        "success": False,
    }

    # -------------------------------------------------------------------------
    # Step 1: Get match details
    # -------------------------------------------------------------------------
    print("\n[Step 1] Getting match details...")

    match = await client.call_tool("get_match", {"match_id": match_id})

    if "error" in match:
        print(f"  ✗ Error: {match['error']}")
        result["error"] = match["error"]
        return result

    print(f"  ✓ {match['home_team']} vs {match['away_team']}")
    print(f"    Date: {match['date']} (Season {match['season']})")
    print(f"    Score: {match['home_goals']}-{match['away_goals']} ({match['result']})")

    result["match"] = match

    # -------------------------------------------------------------------------
    # Step 2: Get baseline probabilities
    # -------------------------------------------------------------------------
    print("\n[Step 2] Getting baseline probabilities...")

    baseline = await client.call_tool("get_baseline_probs", {"match_id": match_id})

    if "error" in baseline:
        print(f"  ✗ Error: {baseline['error']}")
        result["error"] = baseline["error"]
        return result

    if baseline["source"] == "none":
        print("  ✗ No baseline probabilities available for this match")
        result["error"] = "No baseline probabilities available"
        return result

    print(f"  ✓ Source: {baseline['source']}")
    print(f"    Home Win:  {baseline['home_prob']:.1%}")
    print(f"    Draw:      {baseline['draw_prob']:.1%}")
    print(f"    Away Win:  {baseline['away_prob']:.1%}")

    result["baseline"] = baseline

    # -------------------------------------------------------------------------
    # Step 3: Generate LLM prediction
    # -------------------------------------------------------------------------
    print("\n[Step 3] Generating LLM prediction...")

    # Placeholder: Add small random noise to baseline
    # In a real implementation, this would call an actual LLM
    noise = random.uniform(-0.05, 0.05)

    llm_home = max(0.05, min(0.90, baseline['home_prob'] + noise))
    llm_draw = max(0.05, min(0.90, baseline['draw_prob'] - noise / 2))
    llm_away = 1.0 - llm_home - llm_draw

    # Ensure away prob is in valid range
    llm_away = max(0.05, min(0.90, llm_away))

    # Renormalize to sum to 1.0
    total = llm_home + llm_draw + llm_away
    llm_home = round(llm_home / total, 4)
    llm_draw = round(llm_draw / total, 4)
    llm_away = round(1.0 - llm_home - llm_draw, 4)

    llm_prediction = {
        "home_prob": llm_home,
        "draw_prob": llm_draw,
        "away_prob": llm_away,
    }

    print(f"  ✓ LLM Prediction (placeholder with noise={noise:.3f}):")
    print(f"    Home Win:  {llm_home:.1%} (baseline: {baseline['home_prob']:.1%})")
    print(f"    Draw:      {llm_draw:.1%} (baseline: {baseline['draw_prob']:.1%})")
    print(f"    Away Win:  {llm_away:.1%} (baseline: {baseline['away_prob']:.1%})")

    result["llm_prediction"] = llm_prediction

    # -------------------------------------------------------------------------
    # Step 4: Log prediction to database
    # -------------------------------------------------------------------------
    print("\n[Step 4] Logging prediction to database...")

    rationale = (
        f"Simple test prediction (Phase 1). "
        f"Added random noise of {noise:.3f} to baseline probabilities. "
        f"Match: {match['home_team']} vs {match['away_team']} on {match['date']}."
    )

    log_result = await client.call_tool("log_prediction", {
        "match_id": match_id,
        "baseline_home_prob": baseline['home_prob'],
        "baseline_draw_prob": baseline['draw_prob'],
        "baseline_away_prob": baseline['away_prob'],
        "llm_home_prob": llm_home,
        "llm_draw_prob": llm_draw,
        "llm_away_prob": llm_away,
        "rationale_text": rationale,
        "timestamp": datetime.now().isoformat(),
    })

    if not log_result.get("success"):
        print(f"  ✗ Error: {log_result.get('error', 'Unknown error')}")
        result["error"] = log_result.get("error")
        return result

    print(f"  ✓ Logged prediction ID: {log_result['prediction_id']}")
    result["prediction_id"] = log_result["prediction_id"]

    # -------------------------------------------------------------------------
    # Step 5: Evaluate prediction
    # -------------------------------------------------------------------------
    print("\n[Step 5] Evaluating prediction...")

    if match.get('home_goals') is not None:  # Match has been played
        eval_result = await client.call_tool("evaluate_prediction", {"match_id": match_id})

        if "error" in eval_result:
            print(f"  ✗ Error: {eval_result['error']}")
        else:
            print(f"  ✓ Evaluation complete:")
            print(f"    Actual Outcome: {eval_result['outcome']}")
            print()
            print(f"    Baseline Model:")
            print(f"      Predicted: {eval_result['baseline_predicted']}")
            print(f"      Correct:   {'Yes ✓' if eval_result['baseline_correct'] else 'No ✗'}")
            print(f"      Brier:     {eval_result['baseline_brier_score']:.4f}")
            print()
            print(f"    LLM Model:")
            print(f"      Predicted: {eval_result['llm_predicted']}")
            print(f"      Correct:   {'Yes ✓' if eval_result['llm_correct'] else 'No ✗'}")
            print(f"      Brier:     {eval_result['llm_brier_score']:.4f}")

            # Compare models
            brier_diff = eval_result['baseline_brier_score'] - eval_result['llm_brier_score']
            if brier_diff > 0.001:
                print(f"\n    → LLM outperformed baseline by {brier_diff:.4f} Brier")
            elif brier_diff < -0.001:
                print(f"\n    → Baseline outperformed LLM by {-brier_diff:.4f} Brier")
            else:
                print(f"\n    → Models performed similarly")

            result["evaluation"] = eval_result
    else:
        print("  ⏳ Match not yet played - evaluation pending")

    result["success"] = True

    print("\n" + "=" * 60)
    print(" PREDICTION COMPLETE")
    print("=" * 60)

    return result


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for the simple agent."""
    print("=" * 60)
    print(" ASIL SIMPLE AGENT - PHASE 1")
    print("=" * 60)
    print(f"\nMCP Server: {MCP_SERVER_PATH}")

    # Check server exists
    if not MCP_SERVER_PATH.exists():
        print(f"\n✗ Error: MCP server not found at {MCP_SERVER_PATH}")
        print("  Run 'npm run build' in src/mcp/sports-lab/ first")
        return

    print("\nConnecting to Sports Lab MCP server...")

    try:
        async with connect_to_server() as client:
            print("✓ Connected to MCP server")

            # Run prediction for match_id = 1
            result = await predict_match(client, match_id=1)

            # Summary
            print("\n" + "=" * 60)
            print(" RESULT SUMMARY")
            print("=" * 60)
            print(f"\nMatch ID: {result['match_id']}")
            print(f"Success: {result['success']}")

            if result['success']:
                print(f"Prediction ID: {result.get('prediction_id')}")
                if 'evaluation' in result:
                    eval_data = result['evaluation']
                    print(f"LLM Correct: {eval_data['llm_correct']}")
                    print(f"Brier Score: {eval_data['llm_brier_score']:.4f}")
            else:
                print(f"Error: {result.get('error')}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
