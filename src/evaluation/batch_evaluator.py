"""
Batch Evaluator for LangGraph Prediction Workflow

This module runs the prediction workflow on multiple matches and provides
comprehensive analysis of results, comparing LLM predictions against baseline.

Features:
- Batch processing with progress tracking
- Comprehensive metrics extraction
- Statistical analysis with pandas
- Accuracy and Brier score comparisons
- Workflow efficiency metrics
- CSV export for further analysis

Usage:
    from src.evaluation import BatchEvaluator
    from src.workflows import build_prediction_graph

    workflow = build_prediction_graph(mcp_client, db_path, kg, web_rag)
    evaluator = BatchEvaluator(workflow, db_path)
    results = await evaluator.run_batch([1, 2, 3, 4, 5])
    analysis = evaluator.analyze_results()
    evaluator.print_analysis(analysis)
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """
    Batch evaluator for running LangGraph workflows on multiple matches.

    Attributes:
        workflow: Compiled LangGraph workflow
        db_path: Path to SQLite database
        results: List of result dictionaries from each match
    """

    def __init__(self, workflow, db_path: Path):
        """
        Initialize batch evaluator.

        Args:
            workflow: Compiled LangGraph workflow from build_prediction_graph()
            db_path: Path to SQLite database
        """
        self.workflow = workflow
        self.db_path = Path(db_path)
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def run_batch(
        self,
        match_ids: List[int],
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run workflow on multiple matches.

        Args:
            match_ids: List of match IDs to predict
            verbose: Print detailed progress for each match

        Returns:
            List of result dicts, one per match
        """
        self.start_time = datetime.now()
        self.results = []

        print(f"\n{'=' * 70}")
        print(f" BATCH EVALUATION: {len(match_ids)} matches")
        print(f"{'=' * 70}")
        print(f" Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}\n")

        for i, match_id in enumerate(match_ids, 1):
            print(f"[{i:3d}/{len(match_ids)}] Processing match {match_id}...", end=" ", flush=True)

            try:
                # Run workflow
                result = await self.workflow.ainvoke({
                    "match_id": match_id,
                    "verbose": False,  # Suppress individual match output
                    "skip_web_search": True  # Web search disabled by default (improves accuracy)
                })

                # Extract key metrics
                metrics = self._extract_metrics(result)
                self.results.append(metrics)

                # Print summary
                if metrics.get('actual_outcome'):
                    outcome_symbol = "✓" if metrics.get('llm_correct') else "✗"
                    brier = metrics.get('llm_brier', 0)
                    print(f"{outcome_symbol} (Brier: {brier:.3f})")
                else:
                    print("⏳ Pending (future match)")

                if verbose:
                    self._print_match_details(result)

            except Exception as e:
                logger.error(f"Error processing match {match_id}: {e}")
                print(f"✗ Error: {str(e)[:50]}")
                self.results.append({
                    "match_id": match_id,
                    "error": str(e)
                })

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        print(f"\n{'=' * 70}")
        print(f" BATCH COMPLETE")
        print(f"{'=' * 70}")
        print(f" Total matches: {len(self.results)}")
        print(f" Successful: {len([r for r in self.results if 'error' not in r])}")
        print(f" Errors: {len([r for r in self.results if 'error' in r])}")
        print(f" Duration: {duration:.1f} seconds ({duration/len(match_ids):.1f}s per match)")
        print(f"{'=' * 70}\n")

        return self.results

    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from workflow result.

        Args:
            result: Full workflow result dictionary

        Returns:
            Dictionary with extracted metrics
        """
        match = result.get('match', {})
        baseline = result.get('baseline', {})
        prediction = result.get('prediction', {})
        evaluation = result.get('evaluation', {})
        web_context = result.get('web_context', {})

        # Compute web searches performed first (used to infer if skipped)
        web_searches_performed = len(web_context.get('results', {})) if isinstance(web_context, dict) else 0

        # Extract weighted form data
        home_weighted = result.get('home_weighted_form', {})
        away_weighted = result.get('away_weighted_form', {})
        form_comp = result.get('form_comparison', {})

        metrics = {
            "match_id": result.get('match_id'),
            "home_team": match.get('home_team', 'Unknown'),
            "away_team": match.get('away_team', 'Unknown'),
            "date": match.get('date', 'Unknown'),

            # Actual result (if available)
            "home_goals": match.get('home_goals'),
            "away_goals": match.get('away_goals'),

            # Baseline probabilities
            "baseline_home": baseline.get('home_prob', 0),
            "baseline_draw": baseline.get('draw_prob', 0),
            "baseline_away": baseline.get('away_prob', 0),

            # LLM prediction
            "llm_home": prediction.get('home_prob', 0),
            "llm_draw": prediction.get('draw_prob', 0),
            "llm_away": prediction.get('away_prob', 0),
            "llm_confidence": result.get('confidence_level', 'unknown'),
            "parse_method": prediction.get('parse_method', 'unknown'),

            # Workflow metadata (infer skipped from 0 searches performed)
            "skipped_web_search": web_searches_performed == 0,
            "iteration_count": result.get('iteration_count', 0),
            "web_searches_performed": web_searches_performed,

            # KG insights
            "has_kg_insights": bool(result.get('kg_insights') and not result.get('kg_insights', {}).get('error')),

            # Weighted form data (recency-weighted metrics)
            "home_weighted_form": home_weighted,
            "away_weighted_form": away_weighted,
            "form_comparison": form_comp,
        }

        # Evaluation metrics (if match completed)
        if evaluation and evaluation.get('status') != 'pending':
            metrics.update({
                "actual_outcome": evaluation.get('outcome'),
                "baseline_correct": evaluation.get('baseline_correct'),
                "llm_correct": evaluation.get('llm_correct'),
                "baseline_brier": evaluation.get('baseline_brier') or evaluation.get('baseline_brier_score'),
                "llm_brier": evaluation.get('llm_brier') or evaluation.get('llm_brier_score'),
            })

        return metrics

    def _print_match_details(self, result: Dict[str, Any]):
        """Print detailed info for one match."""
        match = result.get('match', {})
        baseline = result.get('baseline', {})
        prediction = result.get('prediction', {})
        evaluation = result.get('evaluation', {})

        print(f"\n  Match: {match.get('home_team', '?')} vs {match.get('away_team', '?')}")
        print(f"  Date: {match.get('date', '?')}")
        print(f"  Baseline: H={baseline.get('home_prob', 0):.0%} "
              f"D={baseline.get('draw_prob', 0):.0%} "
              f"A={baseline.get('away_prob', 0):.0%}")
        print(f"  LLM: H={prediction.get('home_prob', 0):.0%} "
              f"D={prediction.get('draw_prob', 0):.0%} "
              f"A={prediction.get('away_prob', 0):.0%}")
        print(f"  Confidence: {result.get('confidence_level', '?')}")
        print(f"  Workflow: Skipped web={result.get('skip_web_search')}, "
              f"Iterations={result.get('iteration_count')}")

        if evaluation and evaluation.get('status') != 'pending':
            brier = evaluation.get('llm_brier') or evaluation.get('llm_brier_score', 0)
            print(f"  Result: {evaluation.get('outcome', '?')} (Brier: {brier:.3f})")

    def analyze_results(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of batch results.

        Returns:
            Dictionary with statistics and insights
        """
        try:
            import pandas as pd
        except ImportError:
            return {"error": "pandas not installed. Run: pip install pandas"}

        if not self.results:
            return {"error": "No results to analyze. Run run_batch() first."}

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame([r for r in self.results if 'error' not in r])

        if len(df) == 0:
            return {"error": "All matches failed. No data to analyze."}

        # Filter only evaluated matches (completed games)
        df_eval = df[df['actual_outcome'].notna()].copy() if 'actual_outcome' in df.columns else pd.DataFrame()

        analysis = {
            "total_matches": len(df),
            "evaluated_matches": len(df_eval),
            "future_matches": len(df) - len(df_eval),
            "error_count": len([r for r in self.results if 'error' in r]),
        }

        if len(df_eval) == 0:
            analysis["warning"] = "No completed matches to evaluate accuracy"
            return analysis

        # Accuracy
        analysis['baseline_accuracy'] = float(df_eval['baseline_correct'].mean()) if 'baseline_correct' in df_eval.columns else 0
        analysis['llm_accuracy'] = float(df_eval['llm_correct'].mean()) if 'llm_correct' in df_eval.columns else 0
        analysis['accuracy_improvement'] = analysis['llm_accuracy'] - analysis['baseline_accuracy']

        # Brier scores (lower is better)
        if 'baseline_brier' in df_eval.columns and 'llm_brier' in df_eval.columns:
            analysis['baseline_avg_brier'] = float(df_eval['baseline_brier'].mean())
            analysis['llm_avg_brier'] = float(df_eval['llm_brier'].mean())
            analysis['brier_improvement'] = analysis['baseline_avg_brier'] - analysis['llm_avg_brier']

            # Brier score std dev
            analysis['baseline_brier_std'] = float(df_eval['baseline_brier'].std())
            analysis['llm_brier_std'] = float(df_eval['llm_brier'].std())

        # Workflow efficiency
        analysis['web_search_skip_rate'] = float(df['skipped_web_search'].mean()) if 'skipped_web_search' in df.columns else 0
        analysis['avg_iterations'] = float(df['iteration_count'].mean()) if 'iteration_count' in df.columns else 0
        analysis['avg_web_searches'] = float(df['web_searches_performed'].mean()) if 'web_searches_performed' in df.columns else 0

        # KG usage
        if 'has_kg_insights' in df.columns:
            analysis['kg_usage_rate'] = float(df['has_kg_insights'].mean())

        # Confidence analysis
        if 'llm_confidence' in df.columns:
            confidence_counts = df['llm_confidence'].value_counts()
            analysis['confidence_distribution'] = confidence_counts.to_dict()

            # Accuracy by confidence level
            if len(df_eval) > 0 and 'llm_confidence' in df_eval.columns and 'llm_correct' in df_eval.columns:
                confidence_accuracy = df_eval.groupby('llm_confidence')['llm_correct'].mean()
                analysis['accuracy_by_confidence'] = confidence_accuracy.to_dict()

        # Parse method distribution
        if 'parse_method' in df.columns:
            parse_counts = df['parse_method'].value_counts()
            analysis['parse_method_distribution'] = parse_counts.to_dict()

        # Outcome distribution (actual results)
        if 'actual_outcome' in df_eval.columns:
            outcome_dist = df_eval['actual_outcome'].value_counts()
            analysis['outcome_distribution'] = {
                'home_wins': int(outcome_dist.get('H', 0)),
                'draws': int(outcome_dist.get('D', 0)),
                'away_wins': int(outcome_dist.get('A', 0))
            }

        # Find best and worst predictions
        if 'llm_brier' in df_eval.columns and len(df_eval) > 0:
            df_eval_sorted = df_eval.sort_values('llm_brier')

            best_cols = ['match_id', 'home_team', 'away_team', 'llm_brier', 'actual_outcome']
            available_cols = [c for c in best_cols if c in df_eval_sorted.columns]

            analysis['best_predictions'] = df_eval_sorted.head(5)[available_cols].to_dict('records')
            analysis['worst_predictions'] = df_eval_sorted.tail(5)[available_cols].to_dict('records')

        # LLM vs Baseline comparison per match
        if 'llm_brier' in df_eval.columns and 'baseline_brier' in df_eval.columns:
            df_eval['llm_better'] = df_eval['llm_brier'] < df_eval['baseline_brier']
            analysis['llm_beats_baseline_rate'] = float(df_eval['llm_better'].mean())
            analysis['llm_wins'] = int(df_eval['llm_better'].sum())
            analysis['baseline_wins'] = int((~df_eval['llm_better']).sum())

        # Timing
        if self.start_time and self.end_time:
            analysis['duration_seconds'] = (self.end_time - self.start_time).total_seconds()
            analysis['seconds_per_match'] = analysis['duration_seconds'] / len(df) if len(df) > 0 else 0

        return analysis

    def print_analysis(self, analysis: Dict[str, Any]):
        """
        Pretty print analysis results.

        Args:
            analysis: Dictionary from analyze_results()
        """
        print("\n" + "=" * 70)
        print(" BATCH EVALUATION ANALYSIS")
        print("=" * 70)

        if analysis.get('error'):
            print(f"\n⚠️  Error: {analysis['error']}")
            return

        # Dataset summary
        print(f"\n📊 DATASET:")
        print(f"   Total matches processed: {analysis['total_matches']}")
        print(f"   Evaluated (completed):   {analysis['evaluated_matches']}")
        print(f"   Future matches:          {analysis['future_matches']}")
        if analysis.get('error_count', 0) > 0:
            print(f"   Errors:                  {analysis['error_count']}")

        if analysis.get('warning'):
            print(f"\n⚠️  {analysis['warning']}")
            return

        # Accuracy comparison
        print(f"\n🎯 ACCURACY (correct outcome):")
        baseline_acc = analysis.get('baseline_accuracy', 0)
        llm_acc = analysis.get('llm_accuracy', 0)
        improvement = analysis.get('accuracy_improvement', 0)

        print(f"   Baseline: {baseline_acc:.1%}")
        print(f"   LLM:      {llm_acc:.1%}")
        arrow = "📈" if improvement > 0 else ("📉" if improvement < 0 else "➡️")
        print(f"   Change:   {arrow} {improvement:+.1%}")

        # Brier score comparison
        if 'baseline_avg_brier' in analysis:
            print(f"\n📉 BRIER SCORE (lower is better):")
            baseline_brier = analysis.get('baseline_avg_brier', 0)
            llm_brier = analysis.get('llm_avg_brier', 0)
            brier_imp = analysis.get('brier_improvement', 0)

            print(f"   Baseline: {baseline_brier:.4f} (±{analysis.get('baseline_brier_std', 0):.4f})")
            print(f"   LLM:      {llm_brier:.4f} (±{analysis.get('llm_brier_std', 0):.4f})")
            arrow = "📈" if brier_imp > 0 else ("📉" if brier_imp < 0 else "➡️")
            print(f"   Change:   {arrow} {brier_imp:+.4f}")

        # Head-to-head
        if 'llm_beats_baseline_rate' in analysis:
            print(f"\n🏆 HEAD-TO-HEAD (per match):")
            print(f"   LLM wins:      {analysis.get('llm_wins', 0)} matches")
            print(f"   Baseline wins: {analysis.get('baseline_wins', 0)} matches")
            print(f"   LLM win rate:  {analysis.get('llm_beats_baseline_rate', 0):.1%}")

        # Workflow efficiency
        print(f"\n⚡ WORKFLOW EFFICIENCY:")
        print(f"   Web search skip rate: {analysis.get('web_search_skip_rate', 0):.1%}")
        print(f"   Avg iterations/match: {analysis.get('avg_iterations', 0):.2f}")
        print(f"   Avg web searches:     {analysis.get('avg_web_searches', 0):.1f}")
        if 'kg_usage_rate' in analysis:
            print(f"   KG insights used:     {analysis.get('kg_usage_rate', 0):.1%}")

        # Confidence distribution
        if 'confidence_distribution' in analysis:
            print(f"\n🎲 CONFIDENCE DISTRIBUTION:")
            for conf, count in analysis['confidence_distribution'].items():
                pct = count / analysis['total_matches'] * 100
                print(f"   {conf:8s}: {count:3d} ({pct:.0f}%)")

        # Accuracy by confidence
        if 'accuracy_by_confidence' in analysis:
            print(f"\n🎯 ACCURACY BY CONFIDENCE:")
            for conf, acc in sorted(analysis['accuracy_by_confidence'].items()):
                print(f"   {conf:8s}: {acc:.1%}")

        # Parse method distribution
        if 'parse_method_distribution' in analysis:
            print(f"\n🔧 PARSE METHOD DISTRIBUTION:")
            for method, count in analysis['parse_method_distribution'].items():
                pct = count / analysis['total_matches'] * 100
                print(f"   {method:20s}: {count:3d} ({pct:.0f}%)")

        # Outcome distribution
        if 'outcome_distribution' in analysis:
            print(f"\n⚽ ACTUAL OUTCOME DISTRIBUTION:")
            out_dist = analysis['outcome_distribution']
            total_eval = analysis['evaluated_matches']
            print(f"   Home wins: {out_dist['home_wins']:3d} ({out_dist['home_wins']/total_eval*100:.0f}%)")
            print(f"   Draws:     {out_dist['draws']:3d} ({out_dist['draws']/total_eval*100:.0f}%)")
            print(f"   Away wins: {out_dist['away_wins']:3d} ({out_dist['away_wins']/total_eval*100:.0f}%)")

        # Best predictions
        if 'best_predictions' in analysis and analysis['best_predictions']:
            print(f"\n✅ BEST 5 PREDICTIONS (lowest Brier):")
            for pred in analysis['best_predictions']:
                print(f"   Match {pred.get('match_id', '?')}: "
                      f"{pred.get('home_team', '?')} vs {pred.get('away_team', '?')}")
                print(f"      Brier: {pred.get('llm_brier', 0):.4f}, "
                      f"Outcome: {pred.get('actual_outcome', '?')}")

        # Worst predictions
        if 'worst_predictions' in analysis and analysis['worst_predictions']:
            print(f"\n❌ WORST 5 PREDICTIONS (highest Brier):")
            for pred in analysis['worst_predictions']:
                print(f"   Match {pred.get('match_id', '?')}: "
                      f"{pred.get('home_team', '?')} vs {pred.get('away_team', '?')}")
                print(f"      Brier: {pred.get('llm_brier', 0):.4f}, "
                      f"Outcome: {pred.get('actual_outcome', '?')}")

        # Timing
        if 'duration_seconds' in analysis:
            print(f"\n⏱️  TIMING:")
            print(f"   Total duration:    {analysis['duration_seconds']:.1f}s")
            print(f"   Per match average: {analysis.get('seconds_per_match', 0):.1f}s")

        print("\n" + "=" * 70)

    def export_results(self, filepath: str):
        """
        Export results to CSV for further analysis.

        Args:
            filepath: Path to output CSV file
        """
        try:
            import pandas as pd
        except ImportError:
            print("⚠️  pandas not installed. Run: pip install pandas")
            return

        if not self.results:
            print("⚠️  No results to export. Run run_batch() first.")
            return

        df = pd.DataFrame(self.results)

        # Create output directory if needed
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"\n✓ Results exported to {filepath}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get quick summary statistics without full analysis.

        Returns:
            Dictionary with key stats
        """
        if not self.results:
            return {"error": "No results available"}

        successful = [r for r in self.results if 'error' not in r]
        evaluated = [r for r in successful if r.get('actual_outcome')]

        stats = {
            "total": len(self.results),
            "successful": len(successful),
            "evaluated": len(evaluated),
            "errors": len(self.results) - len(successful),
        }

        if evaluated:
            correct = sum(1 for r in evaluated if r.get('llm_correct'))
            stats["accuracy"] = correct / len(evaluated)
            stats["avg_brier"] = sum(r.get('llm_brier', 0) for r in evaluated) / len(evaluated)

        return stats


# ============================================================================
# Standalone Runner
# ============================================================================

async def run_batch_evaluation(
    match_ids: Optional[List[int]] = None,
    num_matches: int = 30,
    verbose: bool = False,
    use_ensemble: bool = False
):
    """
    Run batch evaluation on sample matches.

    This is a standalone function that:
    1. Initializes all components (MCP, KG, WebRAG)
    2. Builds the LangGraph workflow
    3. Runs batch evaluation
    4. Analyzes and exports results

    Args:
        match_ids: Optional list of specific match IDs to evaluate
        num_matches: Number of matches to sample if match_ids not provided
        verbose: Print detailed output for each match
        use_ensemble: If True, use ensemble prediction with multiple models
    """
    import os
    import sys
    import random
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    mode = "Ensemble" if use_ensemble else "Single Model"
    print(f" KickoffAI BATCH EVALUATION (LangGraph Workflow - {mode})")
    print("=" * 70)

    # Check Tavily API key (optional - web search disabled by default)
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("\n⚠️  TAVILY_API_KEY not set - web search disabled (recommended)")
        print("   This is fine! Web search degrades accuracy by -35.6%")

    # Check paths
    db_path = PROJECT_ROOT / "data" / "processed" / "asil.db"
    if not db_path.exists():
        print(f"\n⚠️  Database not found: {db_path}")
        return None, None

    print(f"\n✓ Database: {db_path}")
    if tavily_key:
        print(f"✓ Tavily API key: {tavily_key[:10]}...")
    else:
        print(f"✓ Web search: Disabled (improves accuracy)")

    # Import MCP connection from existing agent
    print("\n1. Connecting to MCP server...")
    try:
        from src.agent.agent import connect_to_mcp
        print("   ✓ MCP module imported")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return None, None

    # Use the MCP context manager
    async with connect_to_mcp() as mcp_client:
        print("   ✓ MCP server connected")

        # Initialize components
        print("\n2. Initializing components...")
        try:
            from src.kg import DynamicKnowledgeGraph
            from src.rag import WebSearchRAG

            # Initialize WebSearchRAG only if API key is provided
            web_rag = None
            if tavily_key:
                web_rag = WebSearchRAG(tavily_api_key=tavily_key)
                print("   ✓ WebSearchRAG initialized")
            else:
                print("   ⚠️ WebSearchRAG skipped (no API key)")

            kg = DynamicKnowledgeGraph(
                db_path=str(db_path),
                web_rag=web_rag,
                ollama_model="llama3.1:8b"
            )
            print("   ✓ DynamicKnowledgeGraph initialized")
        except Exception as e:
            print(f"   ✗ Component error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        # Build workflow
        print("\n3. Building LangGraph workflow...")
        try:
            from src.workflows.prediction_workflow import build_prediction_graph

            workflow = build_prediction_graph(
                mcp_client=mcp_client,
                db_path=db_path,
                kg=kg,
                web_rag=web_rag,
                ollama_model="llama3.1:8b",
                use_ensemble=use_ensemble
            )
            print("   ✓ Workflow compiled")
        except Exception as e:
            print(f"   ✗ Workflow error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        # Create evaluator
        evaluator = BatchEvaluator(workflow, db_path)

        # Select matches to evaluate
        print("\n4. Selecting matches...")

        if match_ids is None:
            match_ids = []

            # Early season (1-50)
            early = list(range(1, 51))
            match_ids.extend(random.sample(early, min(num_matches // 3, len(early))))

            # Mid season (100-200)
            mid = list(range(100, 201))
            match_ids.extend(random.sample(mid, min(num_matches // 3, len(mid))))

            # Late season (250-380)
            late = list(range(250, 381))
            match_ids.extend(random.sample(late, min(num_matches // 3, len(late))))

            # Remove duplicates and sort
            match_ids = sorted(set(match_ids))

        print(f"   Selected {len(match_ids)} matches")
        if len(match_ids) > 10:
            print(f"   Sample: {match_ids[:5]}...{match_ids[-5:]}")
        else:
            print(f"   IDs: {match_ids}")

        # Run batch evaluation
        try:
            results = await evaluator.run_batch(match_ids, verbose=verbose)

            # Analyze results
            analysis = evaluator.analyze_results()
            evaluator.print_analysis(analysis)

            # Export results
            output_path = PROJECT_ROOT / "data" / "evaluation_results.csv"
            evaluator.export_results(str(output_path))

            return evaluator, analysis

        except Exception as e:
            print(f"\n✗ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return evaluator, None

    print("\n✓ MCP disconnected")


# ============================================================================
# Quick Test (no MCP required)
# ============================================================================

def test_evaluator_structure():
    """Test the evaluator class structure without running actual predictions."""
    print("=" * 70)
    print(" BATCH EVALUATOR STRUCTURE TEST")
    print("=" * 70)

    # Test with mock results
    mock_results = [
        {
            "match_id": 1,
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "date": "2024-01-15",
            "baseline_home": 0.45,
            "baseline_draw": 0.28,
            "baseline_away": 0.27,
            "llm_home": 0.50,
            "llm_draw": 0.25,
            "llm_away": 0.25,
            "llm_confidence": "high",
            "actual_outcome": "H",
            "baseline_correct": True,
            "llm_correct": True,
            "baseline_brier": 0.35,
            "llm_brier": 0.30,
            "skipped_web_search": False,
            "iteration_count": 1,
            "web_searches_performed": 3,
            "has_kg_insights": True,
            "parse_method": "exact_format"
        },
        {
            "match_id": 2,
            "home_team": "Man City",
            "away_team": "Chelsea",
            "date": "2024-01-16",
            "baseline_home": 0.60,
            "baseline_draw": 0.22,
            "baseline_away": 0.18,
            "llm_home": 0.55,
            "llm_draw": 0.25,
            "llm_away": 0.20,
            "llm_confidence": "medium",
            "actual_outcome": "D",
            "baseline_correct": False,
            "llm_correct": False,
            "baseline_brier": 0.65,
            "llm_brier": 0.58,
            "skipped_web_search": True,
            "iteration_count": 1,
            "web_searches_performed": 0,
            "has_kg_insights": True,
            "parse_method": "exact_format"
        },
        {
            "match_id": 3,
            "home_team": "Tottenham",
            "away_team": "Newcastle",
            "date": "2024-01-17",
            "baseline_home": 0.40,
            "baseline_draw": 0.30,
            "baseline_away": 0.30,
            "llm_home": 0.35,
            "llm_draw": 0.35,
            "llm_away": 0.30,
            "llm_confidence": "low",
            "actual_outcome": "A",
            "baseline_correct": False,
            "llm_correct": False,
            "baseline_brier": 0.70,
            "llm_brier": 0.68,
            "skipped_web_search": False,
            "iteration_count": 2,
            "web_searches_performed": 5,
            "has_kg_insights": False,
            "parse_method": "keyword_format"
        }
    ]

    # Create evaluator with mock workflow
    evaluator = BatchEvaluator(workflow=None, db_path="test.db")
    evaluator.results = mock_results

    print("\n✓ BatchEvaluator created with mock data")

    # Test analysis
    print("\n1. Testing analyze_results()...")
    analysis = evaluator.analyze_results()

    if 'error' in analysis:
        print(f"   ⚠️ {analysis['error']}")
    else:
        print(f"   ✓ Analysis completed")
        print(f"   Total matches: {analysis['total_matches']}")
        print(f"   Evaluated: {analysis['evaluated_matches']}")
        print(f"   Baseline accuracy: {analysis.get('baseline_accuracy', 0):.1%}")
        print(f"   LLM accuracy: {analysis.get('llm_accuracy', 0):.1%}")

    # Test print_analysis
    print("\n2. Testing print_analysis()...")
    evaluator.print_analysis(analysis)

    # Test export
    print("\n3. Testing export_results()...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        evaluator.export_results(f.name)
        print(f"   ✓ Exported to {f.name}")

    print("\n" + "=" * 70)
    print(" STRUCTURE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run structure test without MCP
        test_evaluator_structure()
    else:
        # Run full batch evaluation
        asyncio.run(run_batch_evaluation())


