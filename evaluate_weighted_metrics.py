"""
Evaluate the impact of weighted metrics (recency weighting) on prediction accuracy.

This script runs batch evaluation and reports:
1. Overall accuracy with weighted metrics
2. Comparison to previous baseline (56.7%)
3. Key metrics showing the weighted data is being used
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.batch_evaluator import run_batch_evaluation


async def evaluate_weighted_impact():
    """Evaluate weighted metrics on a batch of matches."""

    print("=" * 80)
    print("WEIGHTED METRICS EVALUATION")
    print("=" * 80)
    print("\nThis evaluation tests the recency weighting feature that gives more")
    print("weight to recent matches using exponential decay.")
    print("\nPrevious accuracy (without weighting): 56.7%")
    print("Target accuracy (with weighting): 60-65%\n")

    # Run evaluation on 20 matches
    print("Running evaluation on 20 matches...")
    print("=" * 80 + "\n")

    evaluator, analysis = await run_batch_evaluation(
        num_matches=20,
        verbose=False,
        use_ensemble=False
    )

    if evaluator and analysis:
        print("\n" + "=" * 80)
        print("WEIGHTED METRICS IMPACT SUMMARY")
        print("=" * 80)

        # Check if weighted data is present in results
        if evaluator.results and len(evaluator.results) > 0:
            sample = evaluator.results[0]
            if 'home_weighted_form' in sample and sample['home_weighted_form']:
                print("\n✓ Weighted metrics successfully integrated")
                home_wf = sample['home_weighted_form']
                if home_wf and 'weighted_points_per_game' in home_wf:
                    print(f"  Sample: Home team weighted PPG = {home_wf['weighted_points_per_game']:.2f}")
                else:
                    print("  ⚠️  Weighted form present but empty")
            else:
                print("\n✗ WARNING: Weighted metrics not found in results!")

        # Extract key metrics
        llm_accuracy = analysis.get('llm_accuracy', 0)
        baseline_accuracy = analysis.get('baseline_accuracy', 0)
        accuracy_change = llm_accuracy - baseline_accuracy

        previous_llm = 56.7  # Previous LLM accuracy without weighting
        improvement = llm_accuracy - previous_llm

        print(f"\n📊 ACCURACY RESULTS:")
        print(f"  LLM (with weighting):  {llm_accuracy:.1f}%")
        print(f"  Baseline:              {baseline_accuracy:.1f}%")
        print(f"  vs Baseline:           {accuracy_change:+.1f}%")
        print(f"\n  Previous LLM:          {previous_llm:.1f}%")
        print(f"  Improvement:           {improvement:+.1f}%")

        if improvement > 0:
            print(f"\n✓ SUCCESS! Weighted metrics improved accuracy by {improvement:.1f}%")
        elif improvement < 0:
            print(f"\n⚠️  WARNING: Accuracy decreased by {abs(improvement):.1f}%")
        else:
            print(f"\n➡️  No change in accuracy (may need larger sample)")

        # Brier score comparison
        llm_brier = analysis.get('llm_brier_mean', 0)
        baseline_brier = analysis.get('baseline_brier_mean', 0)
        print(f"\n📉 BRIER SCORE (lower is better):")
        print(f"  LLM:      {llm_brier:.4f}")
        print(f"  Baseline: {baseline_brier:.4f}")
        print(f"  Change:   {llm_brier - baseline_brier:+.4f}")

        print("\n" + "=" * 80)
        print("\nFull analysis above includes detailed breakdowns.")
        print("Results exported to: data/evaluation_results.csv")
    else:
        print("\n✗ Evaluation failed!")


if __name__ == "__main__":
    asyncio.run(evaluate_weighted_impact())
