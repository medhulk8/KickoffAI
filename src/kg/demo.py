"""
Demo script showcasing Football Knowledge Graph capabilities
"""

from football_knowledge_graph import FootballKnowledgeGraph


def main():
    """Run demonstrations of the knowledge graph capabilities."""
    kg = FootballKnowledgeGraph()

    print("\n🎯 FOOTBALL KNOWLEDGE GRAPH DEMO\n")

    # Demo 1: Team Style Query
    print("1️⃣  TEAM STYLE QUERIES")
    print("-" * 50)
    teams = ["Liverpool", "Man City", "Burnley", "Leicester"]
    for team in teams:
        styles = kg.get_team_style(team)
        print(f"   {team:20s} → {', '.join(styles)}")

    # Demo 2: Tactical Counter Analysis
    print("\n2️⃣  TACTICAL COUNTER RELATIONSHIPS")
    print("-" * 50)
    styles_to_check = ["HIGH_PRESS", "POSSESSION", "LOW_BLOCK"]
    for style in styles_to_check:
        counters = kg.find_counter_styles(style)
        effective = kg.find_effective_against(style)
        print(f"   {style}")
        print(f"      Countered by: {', '.join(counters) if counters else 'None'}")
        print(f"      Effective vs: {', '.join(effective) if effective else 'None'}")

    # Demo 3: Match Predictions
    print("\n3️⃣  TACTICAL MATCHUP ANALYSIS")
    print("-" * 50)

    matchups = [
        ("Liverpool", "Newcastle"),
        ("Man City", "Leicester"),
        ("Arsenal", "Wolves"),
    ]

    for home, away in matchups:
        print(f"\n   {home} vs {away}")
        matchup = kg.get_tactical_matchup(home, away)

        # Calculate advantage score
        home_adv_count = len(matchup['advantages']['home'])
        away_adv_count = len(matchup['advantages']['away'])

        if home_adv_count > away_adv_count:
            prediction = f"FAVOR: {home} (Tactical Edge: {home_adv_count} vs {away_adv_count})"
        elif away_adv_count > home_adv_count:
            prediction = f"FAVOR: {away} (Tactical Edge: {away_adv_count} vs {home_adv_count})"
        else:
            prediction = "EVEN MATCHUP"

        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {matchup['confidence'].upper()}")

    # Demo 4: Full Visualization
    print("\n4️⃣  DETAILED MATCHUP VISUALIZATION")
    print("-" * 50)
    kg.visualize_matchup("Chelsea", "West Ham")

    # Demo 5: Knowledge Graph Statistics
    print("\n5️⃣  KNOWLEDGE GRAPH STATISTICS")
    print("-" * 50)
    print(f"   Total Nodes: {kg.graph.number_of_nodes()}")
    print(f"   Total Edges: {kg.graph.number_of_edges()}")

    # Count node types
    teams = sum(1 for n in kg.graph.nodes() if kg.graph.nodes[n].get('node_type') == 'TEAM')
    styles = sum(1 for n in kg.graph.nodes() if kg.graph.nodes[n].get('node_type') == 'STYLE')
    concepts = sum(1 for n in kg.graph.nodes() if kg.graph.nodes[n].get('node_type') == 'CONCEPT')

    print(f"   Teams: {teams}")
    print(f"   Tactical Styles: {styles}")
    print(f"   Concepts: {concepts}")

    # Count edge types
    has_style = sum(1 for _, _, d in kg.graph.edges(data=True) if d.get('relation') == 'HAS_STYLE')
    countered = sum(1 for _, _, d in kg.graph.edges(data=True) if d.get('relation') == 'COUNTERED_BY')
    effective = sum(1 for _, _, d in kg.graph.edges(data=True) if d.get('relation') == 'EFFECTIVE_AGAINST')

    print(f"\n   Relationship Counts:")
    print(f"   HAS_STYLE edges: {has_style}")
    print(f"   COUNTERED_BY edges: {countered}")
    print(f"   EFFECTIVE_AGAINST edges: {effective}")

    print("\n" + "=" * 50)
    print("✅ DEMO COMPLETE!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
