"""
Phase 1 Improvement Test

Test Phase 1 improvements on 10 matches:
1. ConfidenceCalculator (objective confidence)
2. Smart query generation (injury searches)
3. Retry loops for low confidence
4. KG-aware query optimization
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def test_phase1_improvements():
    """
    Test Phase 1 improvements on 10 matches.

    Focus on verifying:
    1. Confidence calculation working
    2. Retry loops triggering
    3. Injury searches executing
    4. Query generation conditional logic
    """
    from src.evaluation.batch_evaluator import run_batch_evaluation

    print("\n" + "=" * 70)
    print(" PHASE 1 IMPROVEMENT TEST (10 matches)")
    print("=" * 70)

    # Select 10 specific matches that should show improvements:
    # - Mix of close matches (should trigger retry)
    # - Mix of dominant favorites (should skip retry)
    # - Include matches where injuries might matter
    test_matches = [
        1,    # Brentford vs Arsenal (upset - should retry)
        11,   # Liverpool vs Burnley (dominant - high confidence)
        21,   # Man City vs Arsenal (big match - injuries matter)
        50,   # Random mid-season match
        100,  # Another test match
        150,  # Late season
        200,  # Different season
        250,  # Another sample
        300,  # Late match
        350   # End of season
    ]

    print(f"\nSelected {len(test_matches)} test matches")
    print(f"IDs: {test_matches}")

    # Run evaluation
    evaluator, analysis = await run_batch_evaluation(
        match_ids=test_matches,
        verbose=True  # Show details for each match
    )

    if evaluator is None or analysis is None:
        print("\n[ERROR] Evaluation failed - check setup")
        return None, None

    # Analyze Phase 1 specific metrics
    print("\n" + "=" * 70)
    print(" PHASE 1 SPECIFIC ANALYSIS")
    print("=" * 70)

    results = evaluator.results

    # Metric 1: Confidence distribution
    confidence_dist = {}
    for r in results:
        conf = r.get('confidence_level', 'unknown')
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

    print("\n📊 CONFIDENCE DISTRIBUTION (should have some LOW now):")
    for conf, count in sorted(confidence_dist.items()):
        pct = count / len(results) * 100
        print(f"   {conf.upper()}: {count} matches ({pct:.1f}%)")

    has_low = 'low' in confidence_dist
    has_variety = len(confidence_dist) > 1
    print(f"\n   Low confidence present: {'✓ YES' if has_low else '✗ NO'}")
    print(f"   Confidence variety: {'✓ YES' if has_variety else '✗ NO (all same)'}")

    # Metric 2: Retry loops triggered
    retry_count = sum(1 for r in results if r.get('iteration_count', 1) > 1)
    retry_rate = retry_count / len(results) if results else 0

    print(f"\n🔄 RETRY LOOPS:")
    print(f"   Triggered: {retry_count}/{len(results)} matches ({retry_rate:.1%})")
    print(f"   Expected: ~10-20% (should see at least 1-2 retries)")

    if retry_count > 0:
        print(f"   ✓ Retry mechanism working!")
        retried_matches = [r for r in results if r.get('iteration_count', 1) > 1]
        print(f"   Retried matches:")
        for r in retried_matches:
            match = r.get('match', {})
            print(f"      - Match {r.get('match_id')}: {match.get('home_team', '?')} vs {match.get('away_team', '?')}")
            print(f"        Iterations: {r.get('iteration_count')}")
            print(f"        Final confidence: {r.get('confidence_level')}")
    else:
        print(f"   ⚠️  No retries triggered")
        print(f"   This may be OK if all predictions had medium/high confidence")

    # Metric 3: Query patterns
    total_searches = sum(r.get('web_searches_performed', 0) for r in results if r.get('web_searches_performed'))
    matches_with_searches = sum(1 for r in results if r.get('web_searches_performed', 0) > 0)
    avg_searches = total_searches / matches_with_searches if matches_with_searches > 0 else 0

    print(f"\n🔍 WEB SEARCH PATTERNS:")
    print(f"   Total searches: {total_searches}")
    print(f"   Matches with web search: {matches_with_searches}/{len(results)}")
    print(f"   Average per match (when searched): {avg_searches:.1f}")
    print(f"   Expected: 5-6 queries (with KG) or 7 (without KG)")

    # Check if injury searches are happening
    if 5 <= avg_searches <= 7:
        print(f"   ✓ Query count looks correct (includes injury searches)")
    elif avg_searches > 0:
        print(f"   ? Query count {avg_searches:.1f} - may need verification")

    # Metric 4: Web search skip rate (high confidence baseline skips)
    skipped_count = sum(1 for r in results if r.get('web_searches_performed', 0) == 0)
    skip_rate = skipped_count / len(results) if results else 0

    print(f"\n⚡ WEB SEARCH SKIP RATE:")
    print(f"   Skipped: {skipped_count}/{len(results)} matches ({skip_rate:.1%})")
    print(f"   Expected: ~10-20% (high-confidence baseline predictions)")

    if skipped_count > 0:
        print(f"   ✓ Conditional skip working!")
        skipped_matches = [r for r in results if r.get('web_searches_performed', 0) == 0]
        print(f"   Skipped matches (high baseline confidence):")
        for r in skipped_matches[:3]:  # Show first 3
            match = r.get('match', {})
            baseline = r.get('baseline', {})
            max_baseline = max(
                baseline.get('home_prob', 0),
                baseline.get('draw_prob', 0),
                baseline.get('away_prob', 0)
            )
            print(f"      - Match {r.get('match_id')}: {match.get('home_team', '?')} vs {match.get('away_team', '?')}")
            print(f"        Baseline max prob: {max_baseline:.1%}")

    # Metric 5: Accuracy comparison
    print(f"\n🎯 ACCURACY:")
    baseline_acc = analysis.get('baseline_accuracy', 0)
    llm_acc = analysis.get('llm_accuracy', 0)
    diff = llm_acc - baseline_acc

    print(f"   Baseline: {baseline_acc:.1%}")
    print(f"   LLM: {llm_acc:.1%}")
    print(f"   Difference: {diff*100:+.1f}%")

    if llm_acc > baseline_acc:
        print(f"   ✓ Beating baseline!")
    elif llm_acc >= baseline_acc - 0.05:
        print(f"   ~ Within 5% of baseline")
    else:
        print(f"   ✗ Underperforming baseline")

    # Metric 6: High confidence accuracy (should be high)
    high_conf_results = [r for r in results if r.get('confidence_level') == 'high']
    if high_conf_results:
        high_conf_correct = sum(
            1 for r in high_conf_results
            if r.get('evaluation', {}).get('llm_correct', False)
        )
        high_conf_acc = high_conf_correct / len(high_conf_results)
        print(f"\n📈 HIGH CONFIDENCE ACCURACY:")
        print(f"   {high_conf_correct}/{len(high_conf_results)} correct ({high_conf_acc:.1%})")
        if high_conf_acc >= 0.70:
            print(f"   ✓ High confidence predictions are reliable!")
        else:
            print(f"   ⚠️ High confidence accuracy lower than expected")

    # Summary
    print("\n" + "=" * 70)
    print(" PHASE 1 TEST SUMMARY")
    print("=" * 70)

    checks_passed = 0
    total_checks = 5

    # Check 1: Confidence variety
    if has_variety:
        checks_passed += 1
        print("   [✓] Confidence calculator producing variety")
    else:
        print("   [✗] Confidence calculator not producing variety")

    # Check 2: Query count correct
    if 4 <= avg_searches <= 8:
        checks_passed += 1
        print("   [✓] Query generation working (correct count)")
    else:
        print("   [✗] Query count unexpected")

    # Check 3: Some web search skips (conditional logic)
    if skip_rate > 0:
        checks_passed += 1
        print("   [✓] Conditional web search skip working")
    else:
        print("   [~] No web search skips (may be OK if no high-conf baseline)")

    # Check 4: Accuracy reasonable
    if llm_acc >= 0.50:
        checks_passed += 1
        print("   [✓] Accuracy reasonable (>=50%)")
    else:
        print("   [✗] Accuracy too low")

    # Check 5: Parsing working
    parse_success = sum(1 for r in results if r.get('prediction', {}).get('parse_method') != 'error_fallback')
    if parse_success == len(results):
        checks_passed += 1
        print("   [✓] All predictions parsed successfully")
    else:
        print(f"   [~] {len(results) - parse_success} parse failures")

    print(f"\n   Total: {checks_passed}/{total_checks} checks passed")

    print("\n" + "=" * 70)
    print(" PHASE 1 TEST COMPLETE")
    print("=" * 70)

    return evaluator, analysis


if __name__ == "__main__":
    asyncio.run(test_phase1_improvements())
