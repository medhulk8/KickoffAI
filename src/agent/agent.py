"""
KickoffAI Prediction Agent

Main agent class that orchestrates football match predictions using
the prompt-chaining approach. This agent replaces the simple_agent
with a more sophisticated reasoning pipeline.

Features:
- Single match prediction with detailed reasoning
- Batch prediction for multiple matches
- Error analysis across predictions
- Integration with MCP tools for data access

Usage:
    python src/agent/agent.py

Or programmatically:
    from src.agent.agent import Agent, connect_to_mcp

    async with connect_to_mcp() as client:
        agent = Agent(client)
        result = await agent.predict_match(match_id=1)
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp" / "sports-lab" / "dist" / "index.js"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"


# ============================================================================
# MCP Client Setup
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
# Import prompt chain functions
# ============================================================================

# Import after defining MCPClient to avoid circular imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.prompt_chain import (
    step1_build_context,
    step2_make_prediction,
    step3_self_critique,
    run_full_chain,
)
from src.agent.error_classifier import classify_error, generate_critique


# ============================================================================
# Agent Class
# ============================================================================

class Agent:
    """
    KickoffAI Prediction Agent using prompt-chaining approach.

    This agent coordinates the full prediction workflow:
    1. Gather context (match info, team form, league stats)
    2. Make prediction with reasoning
    3. Self-critique the prediction
    4. Log and evaluate results
    5. Classify errors for learning

    Attributes:
        mcp_client: Connected MCP client for tool access
        db_path: Path to SQLite database
    """

    def __init__(self, mcp_client: MCPClient, db_path: Optional[Path] = None):
        """
        Initialize the agent.

        Args:
            mcp_client: Connected MCP client instance
            db_path: Optional path to database (uses default if not provided)
        """
        self.mcp_client = mcp_client
        self.db_path = db_path or DEFAULT_DB_PATH
        self.prediction_history = []

    async def predict_match(
        self,
        match_id: int,
        verbose: bool = True
    ) -> dict[str, Any]:
        """
        Run full prediction workflow using prompt chaining.

        This method orchestrates the complete prediction pipeline:
        - Builds context from available data
        - Generates prediction with reasoning
        - Self-critiques the prediction
        - Logs prediction to database
        - Evaluates against actual result (if available)
        - Classifies errors (if prediction was wrong)

        Args:
            match_id: ID of the match to predict
            verbose: If True, print detailed output at each step

        Returns:
            Dictionary containing:
            - context: All gathered context data
            - prediction: Predicted probabilities and reasoning
            - critique: Self-critique results
            - prediction_id: ID of logged prediction
            - evaluation: Evaluation results (if match completed)
            - error_tags: Error classification (if prediction wrong)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f" AGENT: PREDICTING MATCH {match_id}")
            print(f"{'='*60}")

        try:
            result = await run_full_chain(
                match_id,
                self.mcp_client,
                self.db_path,
                verbose=verbose
            )

            # Store in history
            self.prediction_history.append({
                "match_id": match_id,
                "timestamp": datetime.now().isoformat(),
                "result": result,
            })

            if verbose:
                self._print_result_summary(result)

            return result

        except Exception as e:
            logger.error(f"Error predicting match {match_id}: {e}")
            error_result = {
                "match_id": match_id,
                "success": False,
                "error": str(e),
            }
            self.prediction_history.append({
                "match_id": match_id,
                "timestamp": datetime.now().isoformat(),
                "result": error_result,
            })
            return error_result

    def _print_result_summary(self, result: dict) -> None:
        """Print a brief summary of the prediction result."""
        if not result.get("success"):
            print(f"\n  ✗ Prediction failed: {result.get('error', 'Unknown error')}")
            return

        print(f"\n{'─'*60}")
        print(" RESULT SUMMARY")
        print(f"{'─'*60}")

        if "prediction" in result:
            pred = result["prediction"]
            print(f"  Prediction: H={pred['home_prob']:.1%}, "
                  f"D={pred['draw_prob']:.1%}, A={pred['away_prob']:.1%}")
            print(f"  Confidence: {pred['confidence']}")

        if "evaluation" in result:
            eval_r = result["evaluation"]
            status = "✓" if eval_r["llm_correct"] else "✗"
            print(f"  Outcome: {eval_r['outcome']} | "
                  f"Predicted: {eval_r['llm_predicted']} {status}")
            print(f"  Brier Score: {eval_r['llm_brier_score']:.4f}")

        if "error_tags" in result:
            print(f"  Error Tags: {result['error_tags']}")

    async def predict_batch(
        self,
        match_ids: list[int],
        verbose: bool = False
    ) -> tuple[list[dict], dict]:
        """
        Predict multiple matches in batch.

        Runs predictions for all specified matches and calculates
        aggregate statistics.

        Args:
            match_ids: List of match IDs to predict
            verbose: If True, print detailed output for each match

        Returns:
            Tuple of (results_list, summary_dict):
            - results_list: Individual prediction results for each match
            - summary_dict: Aggregate statistics (accuracy, avg Brier, etc.)
        """
        print(f"\n{'='*60}")
        print(f" BATCH PREDICTION: {len(match_ids)} MATCHES")
        print(f"{'='*60}")

        results = []
        for i, match_id in enumerate(match_ids):
            if not verbose:
                print(f"  Predicting match {match_id}... ({i+1}/{len(match_ids)})", end="")

            try:
                result = await self.predict_match(match_id, verbose=verbose)
                results.append(result)

                if not verbose:
                    if result.get("success"):
                        eval_r = result.get("evaluation", {})
                        if eval_r:
                            status = "✓" if eval_r.get("llm_correct") else "✗"
                            print(f" {status}")
                        else:
                            print(" (pending)")
                    else:
                        print(" ✗ (error)")

            except Exception as e:
                logger.error(f"Error predicting match {match_id}: {e}")
                results.append({
                    "match_id": match_id,
                    "success": False,
                    "error": str(e)
                })
                if not verbose:
                    print(" ✗ (error)")

        # Calculate summary statistics
        summary = self._calculate_batch_summary(results)

        # Print summary
        print(f"\n{'─'*60}")
        print(" BATCH SUMMARY")
        print(f"{'─'*60}")
        print(f"  Total predictions: {summary['total_predictions']}")
        print(f"  Successfully evaluated: {summary['evaluated']}")
        print(f"  Correct predictions: {summary['correct']}")
        print(f"  Accuracy: {summary['accuracy']:.1%}")
        print(f"  Average Brier Score: {summary['avg_brier']:.4f}")

        if summary.get("error_distribution"):
            print(f"\n  Error Distribution:")
            for tag, count in summary["error_distribution"].items():
                print(f"    {tag}: {count}")

        return results, summary

    def _calculate_batch_summary(self, results: list[dict]) -> dict:
        """Calculate summary statistics from batch results."""
        total = len(results)
        successful = [r for r in results if r.get("success")]
        evaluated = [r for r in successful if r.get("evaluation")]
        correct = [r for r in evaluated if r["evaluation"].get("llm_correct")]

        # Calculate average Brier
        brier_scores = [
            r["evaluation"]["llm_brier_score"]
            for r in evaluated
            if "llm_brier_score" in r.get("evaluation", {})
        ]
        avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0

        # Collect error tags
        error_distribution = {}
        for r in results:
            if "error_tags" in r:
                for tag in r["error_tags"]:
                    error_distribution[tag] = error_distribution.get(tag, 0) + 1

        return {
            "total_predictions": total,
            "successful": len(successful),
            "evaluated": len(evaluated),
            "correct": len(correct),
            "accuracy": len(correct) / len(evaluated) if evaluated else 0,
            "avg_brier": avg_brier,
            "error_distribution": error_distribution,
        }

    async def analyze_errors(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        team: Optional[str] = None
    ) -> dict:
        """
        Analyze error patterns across predictions.

        Uses the aggregate_evaluations MCP tool to gather statistics
        across all predictions, optionally filtered by date or team.

        Args:
            start_date: Filter evaluations after this date (YYYY-MM-DD)
            end_date: Filter evaluations before this date (YYYY-MM-DD)
            team: Filter for matches involving this team

        Returns:
            Dictionary with aggregate metrics and error analysis
        """
        print(f"\n{'='*60}")
        print(" ERROR ANALYSIS")
        print(f"{'='*60}")

        # Build filter arguments
        filters = {}
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        if team:
            filters["team"] = team

        try:
            result = await self.mcp_client.call_tool("aggregate_evaluations", filters)

            if "error" in result:
                print(f"\n  No evaluations found: {result['error']}")
                return result

            print(f"\n  Filters applied: {result.get('filters_applied', 'None')}")
            print(f"\n  Total predictions analyzed: {result['total_predictions']}")
            print(f"\n  Accuracy Comparison:")
            print(f"    Baseline: {result['baseline_accuracy']:.1f}% "
                  f"({result['baseline_correct']}/{result['total_predictions']})")
            print(f"    LLM:      {result['llm_accuracy']:.1f}% "
                  f"({result['llm_correct']}/{result['total_predictions']})")

            print(f"\n  Brier Score Comparison:")
            print(f"    Baseline: {result['avg_baseline_brier']:.4f}")
            print(f"    LLM:      {result['avg_llm_brier']:.4f}")
            print(f"    Improvement: {result['brier_improvement']:+.4f}")

            # Determine which is better
            if result['brier_improvement'] > 0:
                print(f"\n  → LLM outperforms baseline by {result['brier_improvement']:.4f} Brier")
            elif result['brier_improvement'] < 0:
                print(f"\n  → Baseline outperforms LLM by {-result['brier_improvement']:.4f} Brier")
            else:
                print(f"\n  → Models perform equally")

            return result

        except Exception as e:
            logger.error(f"Error in error analysis: {e}")
            return {"error": str(e)}

    async def get_prediction_history(self) -> list[dict]:
        """Get the history of predictions made in this session."""
        return self.prediction_history


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """
    Run agent demo.

    Demonstrates:
    1. Individual match predictions (verbose)
    2. Batch predictions (quiet mode)
    3. Error analysis across all predictions
    """
    print("=" * 70)
    print(" KickoffAI PREDICTION AGENT - DEMO")
    print("=" * 70)
    print(f"\nMCP Server: {MCP_SERVER_PATH}")
    print(f"Database: {DEFAULT_DB_PATH}")

    async with connect_to_mcp() as client:
        # Create agent
        agent = Agent(client)
        print("\n✓ Agent initialized")

        # =====================================================================
        # Part 1: Individual Match Predictions (Verbose)
        # =====================================================================
        print(f"\n\n{'#'*70}")
        print(" PART 1: INDIVIDUAL PREDICTIONS (VERBOSE)")
        print(f"{'#'*70}")

        individual_matches = [1, 25, 50]

        for match_id in individual_matches:
            await agent.predict_match(match_id, verbose=True)
            await asyncio.sleep(0.3)  # Brief pause between predictions

        # =====================================================================
        # Part 2: Batch Prediction (Quiet Mode)
        # =====================================================================
        print(f"\n\n{'#'*70}")
        print(" PART 2: BATCH PREDICTION (QUIET MODE)")
        print(f"{'#'*70}")

        batch_matches = list(range(100, 111))  # Matches 100-110
        results, summary = await agent.predict_batch(batch_matches, verbose=False)

        # =====================================================================
        # Part 3: Error Analysis
        # =====================================================================
        print(f"\n\n{'#'*70}")
        print(" PART 3: ERROR ANALYSIS")
        print(f"{'#'*70}")

        await agent.analyze_errors()

        # =====================================================================
        # Final Summary
        # =====================================================================
        print(f"\n\n{'='*70}")
        print(" DEMO COMPLETE")
        print(f"{'='*70}")

        history = await agent.get_prediction_history()
        total_predictions = len(history)
        successful = sum(1 for h in history if h["result"].get("success"))

        print(f"\nSession Statistics:")
        print(f"  Total predictions made: {total_predictions}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {total_predictions - successful}")


if __name__ == "__main__":
    asyncio.run(main())
