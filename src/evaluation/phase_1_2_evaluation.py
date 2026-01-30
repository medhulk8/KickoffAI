"""
Phase 1+2 Comprehensive Evaluation

Tests combined improvements:
- Phase 1: Confidence calc, injury searches
- Phase 2: Advanced stats, 6-step CoT, H2H history
"""

import asyncio
from src.evaluation.batch_evaluator import run_batch_evaluation


async def phase_1_2_evaluation():
    """
    Comprehensive evaluation of Phase 1+2 improvements

    Runs 50 matches and analyzes:
    - Overall performance vs baseline
    - Confidence distribution and calibration
    - Impact of advanced stats
    - H2H history utilization
    - Token efficiency
    """

    print("\n" + "="*70)
    print("PHASE 1+2 COMPREHENSIVE EVALUATION (50 matches)")
    print("="*70)
    print("\nThis will take ~30-40 minutes...")
    print("Improvements being tested:")
    print("  ✓ Automatic confidence calculation")
    print("  ✓ Injury/suspension searches")
    print("  ✓ Advanced statistics (shots, corners, clean sheets)")
    print("  ✓ 6-step Chain-of-Thought reasoning")
    print("  ✓ Head-to-head history analysis")
    print("\n" + "="*70 + "\n")

    # Run on 50 random matches
    evaluator, analysis = await run_batch_evaluation(
        num_matches=50,
        verbose=False  # Don't print each match (too much output)
    )

    if evaluator is None or analysis is None:
        print("\n⚠️ Evaluation failed to complete")
        return None, None

    # Standard analysis is already printed by run_batch_evaluation
    # Add Phase 1+2 specific analysis here

    results = evaluator.results

    print("\n" + "="*70)
    print("PHASE 1+2 SPECIFIC INSIGHTS")
    print("="*70)

    # Analyze confidence-based performance
    print("\n📊 PERFORMANCE BY CONFIDENCE LEVEL:")

    by_confidence = {'high': [], 'medium': [], 'low': []}
    for r in results:
        if 'actual_outcome' in r:  # Only evaluated matches
            conf = r.get('llm_confidence', 'medium')
            if conf:
                conf = conf.lower()
            else:
                conf = 'medium'
            if conf in by_confidence:
                by_confidence[conf].append(r)

    for conf_level in ['high', 'medium', 'low']:
        matches = by_confidence[conf_level]
        if matches:
            correct = sum(1 for m in matches if m.get('llm_correct'))
            accuracy = correct / len(matches)
            avg_brier = sum(m.get('llm_brier', 0) for m in matches) / len(matches)

            print(f"\n  {conf_level.upper()} Confidence ({len(matches)} matches):")
            print(f"    Accuracy: {accuracy:.1%}")
            print(f"    Avg Brier: {avg_brier:.3f}")
            print(f"    Prediction quality: {'✓ Excellent' if accuracy > 0.70 else '⚠ Needs improvement' if accuracy > 0.50 else '✗ Poor'}")

    # Check if retry loops triggered
    retried = [r for r in results if r.get('iteration_count', 1) > 1]
    if retried:
        print(f"\n🔄 RETRY LOOPS:")
        print(f"  Triggered: {len(retried)} matches ({len(retried)/len(results):.1%})")
        print(f"  These were uncertain predictions that got more context")

        # Check if retried predictions improved
        retry_accuracy = sum(1 for r in retried if r.get('llm_correct', False)) / len(retried) if retried else 0
        print(f"  Retry accuracy: {retry_accuracy:.1%}")
    else:
        print(f"\n🔄 RETRY LOOPS:")
        print(f"  No retries triggered (all predictions confident enough)")

    # Token efficiency
    total_searches = sum(r.get('web_searches_performed', 0) for r in results)
    avg_searches = total_searches / len(results) if results else 0
    skipped = sum(1 for r in results if r.get('skipped_web_search', False))

    print(f"\n💰 TOKEN EFFICIENCY:")
    print(f"  Total searches: {total_searches}")
    print(f"  Avg per match: {avg_searches:.1f}")
    print(f"  Skipped web search: {skipped}/{len(results)} ({skipped/len(results)*100:.1f}%)")
    print(f"  Estimated Tavily tokens used: ~{int(total_searches * 0.7)} (with caching)")

    # Calculate ROI
    baseline_acc = analysis.get('baseline_accuracy', 0)
    llm_acc = analysis.get('llm_accuracy', 0)
    improvement = llm_acc - baseline_acc

    print(f"\n🎯 OVERALL IMPROVEMENT:")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"  Phase 1+2 accuracy: {llm_acc:.1%}")
    print(f"  Absolute gain: {improvement*100:+.1f}%")

    if improvement > 0:
        print(f"  ✓ Successfully beating baseline!")
    elif improvement > -0.02:
        print(f"  ≈ At parity with baseline (acceptable)")
    else:
        print(f"  ⚠ Underperforming baseline")

    baseline_brier = analysis.get('baseline_avg_brier', 0)
    llm_brier = analysis.get('llm_avg_brier', 0)
    brier_improvement = baseline_brier - llm_brier
    brier_improvement_pct = (brier_improvement / baseline_brier * 100) if baseline_brier > 0 else 0

    print(f"\n📉 BRIER SCORE IMPROVEMENT:")
    print(f"  Baseline: {baseline_brier:.3f}")
    print(f"  Phase 1+2: {llm_brier:.3f}")
    print(f"  Improvement: {brier_improvement:+.3f} ({brier_improvement_pct:+.1f}%)")

    if brier_improvement_pct > 10:
        print(f"  ✓ Excellent calibration improvement!")
    elif brier_improvement_pct > 5:
        print(f"  ✓ Good calibration improvement")
    elif brier_improvement_pct > 0:
        print(f"  ✓ Slight calibration improvement")
    else:
        print(f"  ⚠ Calibration needs work")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return evaluator, analysis


if __name__ == "__main__":
    asyncio.run(phase_1_2_evaluation())
