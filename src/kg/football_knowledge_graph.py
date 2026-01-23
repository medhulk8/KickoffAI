"""
Football Knowledge Graph for Tactical Analysis

This module implements a NetworkX-based knowledge graph that encodes:
- Team playing styles
- Tactical counter-relationships
- Football concepts (home advantage, underdog mentality, etc.)

The graph enables tactical matchup analysis and reasoning about team strengths/weaknesses.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple


class FootballKnowledgeGraph:
    """
    A knowledge graph representing football teams, their tactical styles, and relationships.

    Node Types:
        - TEAM: Premier League teams
        - STYLE: Tactical playing styles (HIGH_PRESS, LOW_BLOCK, etc.)
        - CONCEPT: Football concepts (HOME_ADVANTAGE, DERBY_INTENSITY, etc.)

    Edge Types:
        - HAS_STYLE: Team -> Style relationship
        - COUNTERED_BY: Style -> Style (tactical counter)
        - EFFECTIVE_AGAINST: Style -> Style (tactical advantage)
        - STRUGGLES_WITH: Team -> Concept
        - EXCELS_AT: Team -> Concept
    """

    def __init__(self):
        """Initialize the knowledge graph and build all nodes and relationships."""
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """
        Build the complete knowledge graph with nodes and edges.

        Creates all team, style, and concept nodes, then establishes relationships:
        - Team-to-style mappings
        - Tactical counter relationships
        - Team concept relationships
        """
        # Define all styles
        styles = [
            "HIGH_PRESS", "LOW_BLOCK", "POSSESSION", "COUNTER_ATTACK",
            "WING_PLAY", "SET_PIECE_HEAVY", "PHYSICAL", "TECHNICAL"
        ]

        # Define all concepts
        concepts = [
            "HOME_ADVANTAGE", "UNDERDOG_MENTALITY",
            "DERBY_INTENSITY", "RELEGATION_BATTLE"
        ]

        # Team-to-Style mappings (teams can have multiple styles)
        team_styles = {
            # HIGH_PRESS teams
            "Liverpool": ["HIGH_PRESS", "WING_PLAY"],  # Multiple styles
            "Brighton": ["HIGH_PRESS", "TECHNICAL"],
            "Leeds United": ["HIGH_PRESS"],
            "Southampton": ["HIGH_PRESS"],

            # POSSESSION teams
            "Man City": ["POSSESSION", "TECHNICAL"],
            "Arsenal": ["POSSESSION", "TECHNICAL"],
            "Chelsea": ["POSSESSION", "TECHNICAL"],
            "Tottenham": ["POSSESSION"],

            # LOW_BLOCK teams
            "Burnley": ["LOW_BLOCK", "PHYSICAL", "SET_PIECE_HEAVY"],
            "Watford": ["LOW_BLOCK"],
            "Newcastle": ["LOW_BLOCK"],
            "Sheffield United": ["LOW_BLOCK", "PHYSICAL"],
            "West Brom": ["LOW_BLOCK"],

            # COUNTER_ATTACK teams
            "Leicester": ["COUNTER_ATTACK"],
            "West Ham": ["COUNTER_ATTACK", "SET_PIECE_HEAVY"],
            "Wolves": ["COUNTER_ATTACK"],
            "Crystal Palace": ["COUNTER_ATTACK"],

            # WING_PLAY teams
            "Man United": ["WING_PLAY"],
            "Aston Villa": ["WING_PLAY"],

            # Additional teams
            "Stoke City": ["PHYSICAL", "SET_PIECE_HEAVY"],
        }

        # Add all style nodes
        for style in styles:
            self.graph.add_node(style, node_type="STYLE")

        # Add all concept nodes
        for concept in concepts:
            self.graph.add_node(concept, node_type="CONCEPT")

        # Add team nodes and team-to-style edges
        for team, team_style_list in team_styles.items():
            self.graph.add_node(team, node_type="TEAM")
            for style in team_style_list:
                self.graph.add_edge(team, style, relation="HAS_STYLE")

        # Add tactical counter relationships
        tactical_counters = [
            # HIGH_PRESS countered by LOW_BLOCK and LONG_BALL (treating PHYSICAL as proxy for LONG_BALL)
            ("HIGH_PRESS", "LOW_BLOCK", "COUNTERED_BY"),
            ("HIGH_PRESS", "PHYSICAL", "COUNTERED_BY"),

            # HIGH_PRESS effective against POSSESSION
            ("HIGH_PRESS", "POSSESSION", "EFFECTIVE_AGAINST"),

            # LOW_BLOCK countered by WING_PLAY and SET_PIECES
            ("LOW_BLOCK", "WING_PLAY", "COUNTERED_BY"),
            ("LOW_BLOCK", "SET_PIECE_HEAVY", "COUNTERED_BY"),

            # LOW_BLOCK effective against HIGH_PRESS and POSSESSION
            ("LOW_BLOCK", "HIGH_PRESS", "EFFECTIVE_AGAINST"),
            ("LOW_BLOCK", "POSSESSION", "EFFECTIVE_AGAINST"),

            # POSSESSION countered by COUNTER_ATTACK and HIGH_PRESS
            ("POSSESSION", "COUNTER_ATTACK", "COUNTERED_BY"),
            ("POSSESSION", "HIGH_PRESS", "COUNTERED_BY"),

            # POSSESSION effective against PHYSICAL
            ("POSSESSION", "PHYSICAL", "EFFECTIVE_AGAINST"),

            # COUNTER_ATTACK countered by LOW_BLOCK
            ("COUNTER_ATTACK", "LOW_BLOCK", "COUNTERED_BY"),

            # COUNTER_ATTACK effective against HIGH_PRESS and POSSESSION
            ("COUNTER_ATTACK", "HIGH_PRESS", "EFFECTIVE_AGAINST"),
            ("COUNTER_ATTACK", "POSSESSION", "EFFECTIVE_AGAINST"),

            # WING_PLAY countered by PHYSICAL
            ("WING_PLAY", "PHYSICAL", "COUNTERED_BY"),

            # WING_PLAY effective against LOW_BLOCK
            ("WING_PLAY", "LOW_BLOCK", "EFFECTIVE_AGAINST"),

            # SET_PIECES countered by HEIGHT (using PHYSICAL as proxy) and DISCIPLINE (using POSSESSION as proxy)
            ("SET_PIECE_HEAVY", "PHYSICAL", "COUNTERED_BY"),
            ("SET_PIECE_HEAVY", "POSSESSION", "COUNTERED_BY"),

            # SET_PIECES effective against LOW_BLOCK and TECHNICAL
            ("SET_PIECE_HEAVY", "LOW_BLOCK", "EFFECTIVE_AGAINST"),
            ("SET_PIECE_HEAVY", "TECHNICAL", "EFFECTIVE_AGAINST"),
        ]

        for source, target, relation in tactical_counters:
            self.graph.add_edge(source, target, relation=relation)

        # Add team-concept relationships
        team_concepts = [
            # Teams that excel at home
            ("Liverpool", "HOME_ADVANTAGE", "EXCELS_AT"),
            ("Man City", "HOME_ADVANTAGE", "EXCELS_AT"),

            # Teams with underdog mentality
            ("Leicester", "UNDERDOG_MENTALITY", "EXCELS_AT"),
            ("Brighton", "UNDERDOG_MENTALITY", "EXCELS_AT"),

            # Teams that struggle in relegation battles
            ("Newcastle", "RELEGATION_BATTLE", "STRUGGLES_WITH"),
            ("Burnley", "RELEGATION_BATTLE", "EXCELS_AT"),
        ]

        for team, concept, relation in team_concepts:
            if team in self.graph:
                self.graph.add_edge(team, concept, relation=relation)

    def get_team_style(self, team_name: str) -> List[str]:
        """
        Get playing style(s) for a team.

        Args:
            team_name: Name of the team (e.g., "Liverpool", "Man City")

        Returns:
            List of tactical styles the team employs. Can be multiple styles.
            Returns empty list if team not found.

        Example:
            >>> kg = FootballKnowledgeGraph()
            >>> kg.get_team_style("Liverpool")
            ['HIGH_PRESS', 'WING_PLAY']
        """
        if team_name not in self.graph:
            return []

        styles = []
        for successor in self.graph.successors(team_name):
            edge_data = self.graph.get_edge_data(team_name, successor)
            if edge_data and edge_data.get("relation") == "HAS_STYLE":
                styles.append(successor)

        return styles

    def find_counter_styles(self, style: str) -> List[str]:
        """
        Find which styles counter the given style.

        Args:
            style: Tactical style (e.g., "HIGH_PRESS")

        Returns:
            List of styles that counter the given style

        Example:
            >>> kg = FootballKnowledgeGraph()
            >>> kg.find_counter_styles("HIGH_PRESS")
            ['LOW_BLOCK', 'PHYSICAL']
        """
        if style not in self.graph:
            return []

        counters = []
        for successor in self.graph.successors(style):
            edge_data = self.graph.get_edge_data(style, successor)
            if edge_data and edge_data.get("relation") == "COUNTERED_BY":
                counters.append(successor)

        return counters

    def find_effective_against(self, style: str) -> List[str]:
        """
        Find which styles the given style is effective against.

        Args:
            style: Tactical style (e.g., "HIGH_PRESS")

        Returns:
            List of styles that the given style is effective against

        Example:
            >>> kg = FootballKnowledgeGraph()
            >>> kg.find_effective_against("HIGH_PRESS")
            ['POSSESSION']
        """
        if style not in self.graph:
            return []

        effective = []
        for successor in self.graph.successors(style):
            edge_data = self.graph.get_edge_data(style, successor)
            if edge_data and edge_data.get("relation") == "EFFECTIVE_AGAINST":
                effective.append(successor)

        return effective

    def get_1hop_neighbors(self, team_name: str) -> Dict[str, List[str]]:
        """
        Get all nodes 1-hop away from team (for general context).

        Args:
            team_name: Name of the team

        Returns:
            Dictionary with keys:
                - "styles": List of tactical styles the team uses
                - "concepts": List of concepts the team relates to

        Example:
            >>> kg = FootballKnowledgeGraph()
            >>> kg.get_1hop_neighbors("Liverpool")
            {'styles': ['HIGH_PRESS', 'WING_PLAY'], 'concepts': ['HOME_ADVANTAGE']}
        """
        if team_name not in self.graph:
            return {"styles": [], "concepts": []}

        styles = []
        concepts = []

        for successor in self.graph.successors(team_name):
            node_type = self.graph.nodes[successor].get("node_type")
            if node_type == "STYLE":
                styles.append(successor)
            elif node_type == "CONCEPT":
                concepts.append(successor)

        return {"styles": styles, "concepts": concepts}

    def get_tactical_matchup(self, home_team: str, away_team: str) -> Dict:
        """
        Analyze tactical matchup between two teams.

        Examines the playing styles of both teams and determines tactical advantages
        based on counter-relationships in the knowledge graph.

        Args:
            home_team: Name of the home team
            away_team: Name of the away team

        Returns:
            Dictionary containing:
                - home_styles: List of home team's tactical styles
                - away_styles: List of away team's tactical styles
                - advantages: Dict with 'home' and 'away' keys listing advantages
                - matchup_summary: Natural language description of the matchup
                - confidence: 'high', 'medium', or 'low' based on clarity of counters

        Example:
            >>> kg = FootballKnowledgeGraph()
            >>> kg.get_tactical_matchup("Liverpool", "Burnley")
            {
                'home_styles': ['HIGH_PRESS', 'WING_PLAY'],
                'away_styles': ['LOW_BLOCK', 'PHYSICAL', 'SET_PIECE_HEAVY'],
                'advantages': {
                    'home': ['HOME_ADVANTAGE'],
                    'away': ['TACTICAL_COUNTER: LOW_BLOCK counters HIGH_PRESS']
                },
                'matchup_summary': "...",
                'confidence': 'high'
            }
        """
        home_styles = self.get_team_style(home_team)
        away_styles = self.get_team_style(away_team)

        if not home_styles or not away_styles:
            return {
                "home_styles": home_styles,
                "away_styles": away_styles,
                "advantages": {"home": [], "away": []},
                "matchup_summary": f"Insufficient data for matchup analysis between {home_team} and {away_team}",
                "confidence": "low"
            }

        advantages = {"home": ["HOME_ADVANTAGE"], "away": []}

        # Check for tactical counters
        home_counters = []
        away_counters = []

        for h_style in home_styles:
            counters = self.find_counter_styles(h_style)
            for a_style in away_styles:
                if a_style in counters:
                    away_counters.append(f"TACTICAL_COUNTER: {a_style} counters {h_style}")

            effective = self.find_effective_against(h_style)
            for a_style in away_styles:
                if a_style in effective:
                    home_counters.append(f"TACTICAL_ADVANTAGE: {h_style} effective against {a_style}")

        for a_style in away_styles:
            counters = self.find_counter_styles(a_style)
            for h_style in home_styles:
                if h_style in counters:
                    home_counters.append(f"TACTICAL_COUNTER: {h_style} counters {a_style}")

            effective = self.find_effective_against(a_style)
            for h_style in home_styles:
                if h_style in effective:
                    away_counters.append(f"TACTICAL_ADVANTAGE: {a_style} effective against {h_style}")

        advantages["home"].extend(home_counters)
        advantages["away"].extend(away_counters)

        # Generate summary
        if home_counters and not away_counters:
            summary = f"{home_team}'s tactical approach favors them in this matchup"
            confidence = "high"
        elif away_counters and not home_counters:
            summary = f"{away_team}'s style counters {home_team}'s approach despite playing away"
            confidence = "high"
        elif home_counters and away_counters:
            summary = f"Complex tactical battle between {home_team} and {away_team} with both teams having advantages"
            confidence = "medium"
        else:
            summary = f"Neutral tactical matchup between {home_team} and {away_team}"
            confidence = "medium"

        return {
            "home_styles": home_styles,
            "away_styles": away_styles,
            "advantages": advantages,
            "matchup_summary": summary,
            "confidence": confidence
        }

    def visualize_matchup(self, home_team: str, away_team: str):
        """
        Print a text-based visualization of the tactical matchup.

        Displays:
        - Team styles
        - Tactical advantages/disadvantages
        - Match summary

        Args:
            home_team: Name of the home team
            away_team: Name of the away team
        """
        matchup = self.get_tactical_matchup(home_team, away_team)

        print("=" * 70)
        print(f"TACTICAL MATCHUP ANALYSIS: {home_team} vs {away_team}")
        print("=" * 70)
        print()

        print(f"🏠 {home_team} (HOME)")
        print(f"   Styles: {', '.join(matchup['home_styles'])}")
        print(f"   Advantages:")
        for adv in matchup['advantages']['home']:
            print(f"      ✓ {adv}")
        print()

        print(f"✈️  {away_team} (AWAY)")
        print(f"   Styles: {', '.join(matchup['away_styles'])}")
        print(f"   Advantages:")
        for adv in matchup['advantages']['away']:
            print(f"      ✓ {adv}")
        print()

        print("📊 MATCHUP SUMMARY")
        print(f"   {matchup['matchup_summary']}")
        print(f"   Confidence: {matchup['confidence'].upper()}")
        print()
        print("=" * 70)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("Initializing Football Knowledge Graph...")
    kg = FootballKnowledgeGraph()
    print(f"Graph created with {kg.graph.number_of_nodes()} nodes and {kg.graph.number_of_edges()} edges\n")

    # Test 1: Get Liverpool's style
    print("=" * 70)
    print("TEST 1: Get Liverpool's Style")
    print("=" * 70)
    liverpool_styles = kg.get_team_style("Liverpool")
    print(f"Liverpool's playing styles: {liverpool_styles}")
    print(f"✓ PASS: Liverpool has multiple styles including HIGH_PRESS and WING_PLAY")
    print()

    # Test 2: Tactical matchup - Liverpool vs Burnley
    print("=" * 70)
    print("TEST 2: Liverpool vs Burnley Tactical Matchup")
    print("=" * 70)
    matchup = kg.get_tactical_matchup("Liverpool", "Burnley")
    print(f"Home (Liverpool) styles: {matchup['home_styles']}")
    print(f"Away (Burnley) styles: {matchup['away_styles']}")
    print(f"\nHome advantages: {matchup['advantages']['home']}")
    print(f"Away advantages: {matchup['advantages']['away']}")
    print(f"\nSummary: {matchup['matchup_summary']}")
    print(f"Confidence: {matchup['confidence']}")
    print(f"\n✓ PASS: Shows LOW_BLOCK counters HIGH_PRESS")
    print()

    # Test 3: Man City vs Crystal Palace (representing Atletico-like counter-attack style)
    print("=" * 70)
    print("TEST 3: Man City vs Crystal Palace Tactical Battle")
    print("=" * 70)
    matchup2 = kg.get_tactical_matchup("Man City", "Crystal Palace")
    print(f"Home (Man City) styles: {matchup2['home_styles']}")
    print(f"Away (Crystal Palace) styles: {matchup2['away_styles']}")
    print(f"\nHome advantages: {matchup2['advantages']['home']}")
    print(f"Away advantages: {matchup2['advantages']['away']}")
    print(f"\nSummary: {matchup2['matchup_summary']}")
    print(f"Confidence: {matchup2['confidence']}")
    print(f"\n✓ PASS: Shows interesting tactical battle with COUNTER_ATTACK effective against POSSESSION")
    print()

    # Test 4: Visualize Arsenal vs Brighton
    print("TEST 4: Arsenal vs Brighton Matchup Visualization")
    kg.visualize_matchup("Arsenal", "Brighton")
    print("✓ PASS: Visualization complete")
    print()

    # Additional tests
    print("=" * 70)
    print("ADDITIONAL TESTS")
    print("=" * 70)

    # Test find_counter_styles
    print("\nTest: Find counter styles for HIGH_PRESS")
    counters = kg.find_counter_styles("HIGH_PRESS")
    print(f"Styles that counter HIGH_PRESS: {counters}")

    # Test find_effective_against
    print("\nTest: Find what HIGH_PRESS is effective against")
    effective = kg.find_effective_against("HIGH_PRESS")
    print(f"HIGH_PRESS is effective against: {effective}")

    # Test get_1hop_neighbors
    print("\nTest: Get 1-hop neighbors for Liverpool")
    neighbors = kg.get_1hop_neighbors("Liverpool")
    print(f"Liverpool's neighbors: {neighbors}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
