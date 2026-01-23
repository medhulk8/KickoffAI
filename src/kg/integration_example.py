"""
Integration Examples: Football Knowledge Graph with ASIL Components

This module demonstrates how to integrate the Football Knowledge Graph
with the existing ASIL project components (MCP, Agent, Baseline Model).
"""

from football_knowledge_graph import FootballKnowledgeGraph
import json


def example_1_tactical_features_for_ml():
    """
    Example 1: Generate tactical features for machine learning models.

    Use case: Augment baseline model with tactical knowledge.
    """
    print("=" * 70)
    print("EXAMPLE 1: Tactical Features for ML Models")
    print("=" * 70)

    kg = FootballKnowledgeGraph()

    # Sample match: Liverpool vs Burnley
    home_team = "Liverpool"
    away_team = "Burnley"

    matchup = kg.get_tactical_matchup(home_team, away_team)

    # Generate tactical features
    tactical_features = {
        # Basic style encoding (one-hot or multi-hot)
        'home_has_high_press': 'HIGH_PRESS' in matchup['home_styles'],
        'home_has_possession': 'POSSESSION' in matchup['home_styles'],
        'home_has_wing_play': 'WING_PLAY' in matchup['home_styles'],
        'away_has_low_block': 'LOW_BLOCK' in matchup['away_styles'],
        'away_has_counter': 'COUNTER_ATTACK' in matchup['away_styles'],

        # Tactical advantage scores
        'home_tactical_advantage_count': len(matchup['advantages']['home']),
        'away_tactical_advantage_count': len(matchup['advantages']['away']),
        'net_tactical_advantage': len(matchup['advantages']['home']) - len(matchup['advantages']['away']),

        # Confidence as numeric weight
        'tactical_confidence': {'high': 1.0, 'medium': 0.6, 'low': 0.3}[matchup['confidence']],

        # Style complexity (number of styles)
        'home_style_count': len(matchup['home_styles']),
        'away_style_count': len(matchup['away_styles']),
    }

    print(f"\nMatch: {home_team} vs {away_team}")
    print(f"\nTactical Features Generated:")
    for key, value in tactical_features.items():
        print(f"  {key:35s} = {value}")

    print(f"\n💡 Use Case:")
    print(f"  Add these features to your ML feature vector:")
    print(f"  X = [statistical_features, tactical_features]")
    print()

    return tactical_features


def example_2_agent_context():
    """
    Example 2: Provide tactical context to agent for reasoning.

    Use case: Enhance agent's decision-making with tactical knowledge.
    """
    print("=" * 70)
    print("EXAMPLE 2: Tactical Context for Agent Reasoning")
    print("=" * 70)

    kg = FootballKnowledgeGraph()

    # Agent is analyzing Man City vs Leicester
    home_team = "Man City"
    away_team = "Leicester"

    matchup = kg.get_tactical_matchup(home_team, away_team)
    home_context = kg.get_1hop_neighbors(home_team)
    away_context = kg.get_1hop_neighbors(away_team)

    # Create rich context for agent
    agent_context = {
        'match': f"{home_team} vs {away_team}",
        'home_team_profile': {
            'name': home_team,
            'styles': home_context['styles'],
            'concepts': home_context['concepts'],
        },
        'away_team_profile': {
            'name': away_team,
            'styles': away_context['styles'],
            'concepts': away_context['concepts'],
        },
        'tactical_analysis': {
            'summary': matchup['matchup_summary'],
            'confidence': matchup['confidence'],
            'home_advantages': matchup['advantages']['home'],
            'away_advantages': matchup['advantages']['away'],
        },
        'reasoning_hints': []
    }

    # Add reasoning hints based on tactical analysis
    if len(matchup['advantages']['away']) > len(matchup['advantages']['home']):
        agent_context['reasoning_hints'].append(
            f"{away_team}'s tactical style may offset home advantage"
        )

    if matchup['confidence'] == 'high':
        agent_context['reasoning_hints'].append(
            "High tactical clarity - tactical factors strongly favor one side"
        )

    print(f"\nAgent Context Generated:")
    print(json.dumps(agent_context, indent=2))

    print(f"\n💡 Use Case:")
    print(f"  Pass this context to agent's chain-of-thought reasoning:")
    print(f"  agent.reason(match_data, tactical_context=agent_context)")
    print()

    return agent_context


def example_3_mcp_tool_integration():
    """
    Example 3: Expose knowledge graph as MCP tool.

    Use case: Allow MCP server to query tactical knowledge.
    """
    print("=" * 70)
    print("EXAMPLE 3: MCP Tool Integration")
    print("=" * 70)

    # Simulate MCP tool definitions
    mcp_tools = [
        {
            'name': 'get_team_tactical_style',
            'description': 'Get the tactical playing style(s) of a football team',
            'parameters': {
                'team_name': {
                    'type': 'string',
                    'description': 'Name of the team (e.g., Liverpool, Man City)'
                }
            }
        },
        {
            'name': 'analyze_tactical_matchup',
            'description': 'Analyze the tactical matchup between two teams',
            'parameters': {
                'home_team': {
                    'type': 'string',
                    'description': 'Name of the home team'
                },
                'away_team': {
                    'type': 'string',
                    'description': 'Name of the away team'
                }
            }
        },
        {
            'name': 'find_tactical_counters',
            'description': 'Find which tactical styles counter a given style',
            'parameters': {
                'style': {
                    'type': 'string',
                    'description': 'Tactical style (e.g., HIGH_PRESS, LOW_BLOCK)',
                    'enum': ['HIGH_PRESS', 'LOW_BLOCK', 'POSSESSION', 'COUNTER_ATTACK',
                             'WING_PLAY', 'SET_PIECE_HEAVY', 'PHYSICAL', 'TECHNICAL']
                }
            }
        }
    ]

    print("\nMCP Tool Definitions:")
    for tool in mcp_tools:
        print(f"\n  Tool: {tool['name']}")
        print(f"  Description: {tool['description']}")
        print(f"  Parameters: {list(tool['parameters'].keys())}")

    # Example MCP tool execution
    print("\n" + "-" * 70)
    print("Example MCP Tool Execution:")
    print("-" * 70)

    kg = FootballKnowledgeGraph()

    # Simulate tool call 1
    print("\nTool Call: get_team_tactical_style(team_name='Liverpool')")
    result = kg.get_team_style("Liverpool")
    print(f"Result: {result}")

    # Simulate tool call 2
    print("\nTool Call: analyze_tactical_matchup(home_team='Arsenal', away_team='Wolves')")
    result = kg.get_tactical_matchup("Arsenal", "Wolves")
    print(f"Result Summary: {result['matchup_summary']}")
    print(f"Confidence: {result['confidence']}")

    print(f"\n💡 Use Case:")
    print(f"  Add these tools to your MCP server:")
    print(f"  server.add_tool('get_team_tactical_style', kg.get_team_style)")
    print(f"  server.add_tool('analyze_tactical_matchup', kg.get_tactical_matchup)")
    print()


def example_4_match_prediction_pipeline():
    """
    Example 4: Complete match prediction pipeline with tactical knowledge.

    Use case: End-to-end prediction incorporating tactical analysis.
    """
    print("=" * 70)
    print("EXAMPLE 4: Match Prediction Pipeline")
    print("=" * 70)

    kg = FootballKnowledgeGraph()

    # Sample matches to predict
    matches = [
        ("Liverpool", "Newcastle"),
        ("Man City", "Leicester"),
        ("Burnley", "Arsenal"),
        ("Chelsea", "West Ham"),
    ]

    print("\nPrediction Pipeline Results:")
    print("-" * 70)

    predictions = []

    for home, away in matches:
        # Step 1: Get tactical matchup
        matchup = kg.get_tactical_matchup(home, away)

        # Step 2: Calculate tactical score
        home_adv = len(matchup['advantages']['home'])
        away_adv = len(matchup['advantages']['away'])
        tactical_score = home_adv - away_adv

        # Step 3: Make prediction
        if tactical_score > 2:
            prediction = f"{home} (Strong)"
        elif tactical_score > 0:
            prediction = f"{home} (Slight)"
        elif tactical_score < -2:
            prediction = f"{away} (Strong)"
        elif tactical_score < 0:
            prediction = f"{away} (Slight)"
        else:
            prediction = "Draw/Even"

        # Step 4: Confidence adjustment
        confidence = matchup['confidence']
        confidence_emoji = {'high': '🔥', 'medium': '⚡', 'low': '❓'}[confidence]

        predictions.append({
            'match': f"{home} vs {away}",
            'prediction': prediction,
            'tactical_score': tactical_score,
            'confidence': confidence,
            'confidence_emoji': confidence_emoji
        })

        print(f"\n{home} vs {away}")
        print(f"  Prediction: {prediction} {confidence_emoji}")
        print(f"  Tactical Score: {tactical_score:+d} ({home_adv} vs {away_adv})")
        print(f"  Confidence: {confidence.upper()}")
        print(f"  Reasoning: {matchup['matchup_summary']}")

    print(f"\n💡 Use Case:")
    print(f"  Combine with baseline model:")
    print(f"  final_prediction = 0.7 * statistical_pred + 0.3 * tactical_pred")
    print()

    return predictions


def example_5_graph_embeddings():
    """
    Example 5: Generate graph embeddings for neural network input.

    Use case: Create vector representations of teams for deep learning.
    """
    print("=" * 70)
    print("EXAMPLE 5: Graph Embeddings for Neural Networks")
    print("=" * 70)

    kg = FootballKnowledgeGraph()

    # Define style embedding dimensions
    style_to_idx = {
        'HIGH_PRESS': 0,
        'LOW_BLOCK': 1,
        'POSSESSION': 2,
        'COUNTER_ATTACK': 3,
        'WING_PLAY': 4,
        'SET_PIECE_HEAVY': 5,
        'PHYSICAL': 6,
        'TECHNICAL': 7
    }

    def get_team_embedding(team_name):
        """Generate one-hot style embedding for a team."""
        embedding = [0] * 8  # 8 styles
        styles = kg.get_team_style(team_name)
        for style in styles:
            if style in style_to_idx:
                embedding[style_to_idx[style]] = 1
        return embedding

    # Generate embeddings for sample teams
    teams = ["Liverpool", "Man City", "Burnley", "Leicester"]

    print("\nTeam Style Embeddings:")
    print("-" * 70)

    embeddings = {}
    for team in teams:
        embedding = get_team_embedding(team)
        embeddings[team] = embedding
        styles = kg.get_team_style(team)

        print(f"\n{team}:")
        print(f"  Styles: {', '.join(styles)}")
        print(f"  Embedding: {embedding}")

    # Show how to use in match prediction
    print("\n" + "-" * 70)
    print("Match Embedding Example:")
    print("-" * 70)

    home = "Liverpool"
    away = "Burnley"
    home_emb = get_team_embedding(home)
    away_emb = get_team_embedding(away)

    print(f"\n{home} vs {away}")
    print(f"  Home embedding: {home_emb}")
    print(f"  Away embedding: {away_emb}")
    print(f"  Combined:       {home_emb + away_emb}  (concat for NN input)")

    print(f"\n💡 Use Case:")
    print(f"  Feed to neural network:")
    print(f"  input_vector = [home_embedding, away_embedding, statistical_features]")
    print(f"  prediction = neural_net(input_vector)")
    print()

    return embeddings


def run_all_examples():
    """Run all integration examples."""
    print("\n")
    print("=" * 70)
    print("FOOTBALL KNOWLEDGE GRAPH - INTEGRATION EXAMPLES")
    print("=" * 70)
    print()

    example_1_tactical_features_for_ml()
    example_2_agent_context()
    example_3_mcp_tool_integration()
    example_4_match_prediction_pipeline()
    example_5_graph_embeddings()

    print("=" * 70)
    print("ALL INTEGRATION EXAMPLES COMPLETED!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run_all_examples()
