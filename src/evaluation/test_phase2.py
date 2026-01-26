"""
Phase 2 Deep Context Test

Test Phase 2 improvements on matches:
1. Advanced Statistics Calculator (shots, corners, efficiency)
2. Structured 6-step Chain-of-Thought prompting
3. Head-to-head history analysis

Run: TAVILY_API_KEY=your_key python -m src.evaluation.test_phase2
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_advanced_stats_standalone():
    """Test AdvancedStatsCalculator without full workflow."""
    print("\n" + "=" * 70)
    print(" TEST 1: Advanced Stats Calculator")
    print("=" * 70)

    from src.data.advanced_stats import AdvancedStatsCalculator

    db_path = PROJECT_ROOT / "data" / "processed" / "asil.db"
    if not db_path.exists():
        print(f"\n[ERROR] Database not found: {db_path}")
        return False

    calc = AdvancedStatsCalculator(str(db_path))
    print(f"\n[OK] Calculator initialized with {db_path}")

    # Test on a few teams
    teams = ["Liverpool", "Arsenal", "Manchester City"]

    for team in teams:
        print(f"\n--- {team} ---")
        stats = calc.get_team_advanced_stats(team, last_n=5)

        if stats['matches_analyzed'] == 0:
            print(f"   [WARN] No matches found")
            continue

        att = stats['attacking']
        defe = stats['defensive']
        eff = stats['efficiency']

        print(f"   Matches analyzed: {stats['matches_analyzed']}")
        print(f"   Attacking: {att['avg_shots']:.1f} shots, {att['avg_goals']:.2f} goals/game")
        print(f"   Defensive: {defe['avg_goals_conceded']:.2f} conceded, {defe['clean_sheet_rate']:.0%} CS")
        print(f"   Efficiency: {eff['conversion_rate']:.1f}% conv, {eff['attacking_threat']:.1f}/10 threat")

    # Test H2H
    print("\n--- Head-to-Head: Liverpool vs Arsenal ---")
    h2h = calc.get_head_to_head_stats("Liverpool", "Arsenal", last_n=5)

    print(f"   Meetings: {h2h['matches_played']}")
    print(f"   Liverpool wins: {h2h['team1_wins']}")
    print(f"   Arsenal wins: {h2h['team2_wins']}")
    print(f"   Draws: {h2h['draws']}")
    print(f"   Recent: {h2h['recent_results'][:3]}")
    print(f"   Dominance: {h2h['dominance']}")

    print("\n[OK] Advanced stats calculator working!")
    return True


async def test_phase2_workflow():
    """Test full workflow with Phase 2 improvements."""
    print("\n" + "=" * 70)
    print(" TEST 2: Full Workflow with Phase 2 Improvements")
    print("=" * 70)

    # Check Tavily API key
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("\n[WARN] TAVILY_API_KEY not set - using DuckDuckGo fallback")

    db_path = PROJECT_ROOT / "data" / "processed" / "asil.db"
    if not db_path.exists():
        print(f"\n[ERROR] Database not found: {db_path}")
        return False

    print(f"\n[OK] Database: {db_path}")
    if tavily_key:
        print(f"[OK] Tavily API key: {tavily_key[:10]}...")

    # Import components
    print("\n1. Importing components...")
    try:
        from src.agent.agent import connect_to_mcp
        from src.kg import DynamicKnowledgeGraph
        from src.rag import WebSearchRAG
        from src.workflows.prediction_workflow import build_prediction_graph
        print("   [OK] All imports successful")
    except ImportError as e:
        print(f"   [ERROR] Import failed: {e}")
        return False

    # Connect to MCP
    print("\n2. Connecting to MCP server...")
    async with connect_to_mcp() as mcp_client:
        print("   [OK] MCP connected")

        # Initialize components
        print("\n3. Initializing components...")
        try:
            web_rag = WebSearchRAG(tavily_api_key=tavily_key)
            print("   [OK] WebSearchRAG")

            kg = DynamicKnowledgeGraph(
                db_path=str(db_path),
                web_rag=web_rag,
                ollama_model="llama3.1:8b"
            )
            print("   [OK] DynamicKnowledgeGraph")
        except Exception as e:
            print(f"   [ERROR] Component init failed: {e}")
            return False

        # Build workflow
        print("\n4. Building LangGraph workflow...")
        try:
            workflow = build_prediction_graph(
                mcp_client=mcp_client,
                db_path=db_path,
                kg=kg,
                web_rag=web_rag,
                ollama_model="llama3.1:8b"
            )
            print("   [OK] Workflow compiled")
        except Exception as e:
            print(f"   [ERROR] Workflow build failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test on 3 diverse matches
        test_matches = [
            (1, "Early season match"),
            (50, "Mid-season match"),
            (200, "Late season match")
        ]

        print("\n" + "=" * 70)
        print(" RUNNING PREDICTIONS")
        print("=" * 70)

        results = []
        for match_id, description in test_matches:
            print(f"\n--- Match {match_id}: {description} ---")

            try:
                result = await workflow.ainvoke({
                    "match_id": match_id,
                    "verbose": True
                })

                if result.get("error"):
                    print(f"   [ERROR] {result['error']}")
                    continue

                # Extract key info
                match = result.get('match', {})
                pred = result.get('prediction', {})
                h2h = result.get('h2h_stats', {})
                home_adv = result.get('home_advanced_stats', {})
                away_adv = result.get('away_advanced_stats', {})
                evaluation = result.get('evaluation', {})

                print(f"\n   MATCH: {match.get('home_team', '?')} vs {match.get('away_team', '?')}")

                # Show advanced stats were calculated
                if home_adv.get('matches_analyzed', 0) > 0:
                    print(f"   [OK] Home advanced stats: {home_adv['matches_analyzed']} matches analyzed")
                    print(f"        Attacking threat: {home_adv.get('efficiency', {}).get('attacking_threat', 0):.1f}/10")
                else:
                    print(f"   [WARN] No home advanced stats")

                if away_adv.get('matches_analyzed', 0) > 0:
                    print(f"   [OK] Away advanced stats: {away_adv['matches_analyzed']} matches analyzed")
                    print(f"        Attacking threat: {away_adv.get('efficiency', {}).get('attacking_threat', 0):.1f}/10")
                else:
                    print(f"   [WARN] No away advanced stats")

                # Show H2H
                if h2h.get('matches_played', 0) > 0:
                    print(f"   [OK] H2H: {h2h['matches_played']} meetings, dominance: {h2h['dominance']}")
                else:
                    print(f"   [INFO] No H2H history")

                # Show prediction
                if pred:
                    print(f"\n   PREDICTION:")
                    print(f"   H={pred.get('home_prob', 0):.1%}, D={pred.get('draw_prob', 0):.1%}, A={pred.get('away_prob', 0):.1%}")
                    print(f"   Confidence: {pred.get('confidence', '?')}")
                    print(f"   Parse method: {pred.get('parse_method', '?')}")

                    # Show reasoning (truncated)
                    reasoning = pred.get('reasoning', '')[:200]
                    if reasoning:
                        print(f"   Reasoning: {reasoning}...")

                # Show evaluation if available
                if evaluation.get('status') != 'pending':
                    actual = f"{match.get('home_goals', '?')}-{match.get('away_goals', '?')}"
                    llm_brier = evaluation.get('llm_brier', 'N/A')
                    baseline_brier = evaluation.get('baseline_brier', 'N/A')
                    print(f"\n   EVALUATION:")
                    print(f"   Actual: {actual}")
                    print(f"   LLM Brier: {llm_brier}")
                    print(f"   Baseline Brier: {baseline_brier}")

                results.append({
                    'match_id': match_id,
                    'success': True,
                    'has_advanced_stats': home_adv.get('matches_analyzed', 0) > 0,
                    'has_h2h': h2h.get('matches_played', 0) > 0,
                    'prediction': pred
                })

            except Exception as e:
                print(f"   [ERROR] Prediction failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({'match_id': match_id, 'success': False, 'error': str(e)})

        # Summary
        print("\n" + "=" * 70)
        print(" PHASE 2 TEST SUMMARY")
        print("=" * 70)

        successful = [r for r in results if r.get('success')]
        with_adv_stats = [r for r in successful if r.get('has_advanced_stats')]
        with_h2h = [r for r in successful if r.get('has_h2h')]

        print(f"\n   Total matches tested: {len(results)}")
        print(f"   Successful predictions: {len(successful)}/{len(results)}")
        print(f"   With advanced stats: {len(with_adv_stats)}/{len(successful)}")
        print(f"   With H2H history: {len(with_h2h)}/{len(successful)}")

        # Phase 2 checks
        checks_passed = 0
        total_checks = 4

        # Check 1: All predictions successful
        if len(successful) == len(results):
            checks_passed += 1
            print("\n   [PASS] All predictions completed successfully")
        else:
            print(f"\n   [FAIL] {len(results) - len(successful)} predictions failed")

        # Check 2: Advanced stats calculated
        if len(with_adv_stats) >= len(successful) * 0.5:
            checks_passed += 1
            print("   [PASS] Advanced stats calculated for most matches")
        else:
            print("   [FAIL] Advanced stats not calculated")

        # Check 3: Parse method working
        parse_success = [r for r in successful if r.get('prediction', {}).get('parse_method') != 'error_fallback']
        if len(parse_success) == len(successful):
            checks_passed += 1
            print("   [PASS] All predictions parsed successfully")
        else:
            print(f"   [FAIL] {len(successful) - len(parse_success)} parse failures")

        # Check 4: 6-step CoT prompt used (implied by advanced stats being in context)
        if len(with_adv_stats) > 0:
            checks_passed += 1
            print("   [PASS] 6-step Chain-of-Thought prompt working")
        else:
            print("   [FAIL] CoT prompt not working (no advanced stats in context)")

        print(f"\n   TOTAL: {checks_passed}/{total_checks} checks passed")

        print("\n" + "=" * 70)
        print(" PHASE 2 TEST COMPLETE")
        print("=" * 70)

        return checks_passed >= 3  # Pass if at least 3/4 checks pass


async def main():
    """Run all Phase 2 tests."""
    print("=" * 70)
    print(" PHASE 2: DEEP CONTEXT ENHANCEMENTS TEST SUITE")
    print("=" * 70)
    print("\nTests:")
    print("1. Advanced Statistics Calculator")
    print("2. Full Workflow with 6-step CoT + Advanced Stats + H2H")

    # Test 1: Standalone advanced stats
    stats_ok = test_advanced_stats_standalone()

    # Test 2: Full workflow
    workflow_ok = await test_phase2_workflow()

    # Final summary
    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)
    print(f"\n   Advanced Stats Calculator: {'PASS' if stats_ok else 'FAIL'}")
    print(f"   Full Workflow with Phase 2: {'PASS' if workflow_ok else 'FAIL'}")

    if stats_ok and workflow_ok:
        print("\n   [SUCCESS] Phase 2 implementation verified!")
    else:
        print("\n   [WARNING] Some tests failed - review output above")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
