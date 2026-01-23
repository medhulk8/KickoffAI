# Football Knowledge Graph

A NetworkX-based knowledge graph for tactical football analysis, enabling reasoning about team playing styles, tactical matchups, and match predictions.

## Overview

The Football Knowledge Graph encodes:
- **20+ Premier League teams** with their tactical characteristics
- **8 tactical styles** (HIGH_PRESS, LOW_BLOCK, POSSESSION, etc.)
- **Tactical counter-relationships** (what beats what)
- **Football concepts** (HOME_ADVANTAGE, UNDERDOG_MENTALITY, etc.)

## Architecture

### Node Types

1. **TEAM Nodes**: Premier League teams (Liverpool, Man City, etc.)
2. **STYLE Nodes**: Tactical playing styles
   - HIGH_PRESS
   - LOW_BLOCK
   - POSSESSION
   - COUNTER_ATTACK
   - WING_PLAY
   - SET_PIECE_HEAVY
   - PHYSICAL
   - TECHNICAL

3. **CONCEPT Nodes**: Football concepts
   - HOME_ADVANTAGE
   - UNDERDOG_MENTALITY
   - DERBY_INTENSITY
   - RELEGATION_BATTLE

### Edge Types (Relationships)

- `HAS_STYLE`: Team → Style (e.g., Liverpool → HIGH_PRESS)
- `COUNTERED_BY`: Style → Style (e.g., HIGH_PRESS → LOW_BLOCK)
- `EFFECTIVE_AGAINST`: Style → Style (e.g., HIGH_PRESS → POSSESSION)
- `EXCELS_AT`: Team → Concept (e.g., Liverpool → HOME_ADVANTAGE)
- `STRUGGLES_WITH`: Team → Concept

## Installation

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies
pip install networkx
```

## Usage

### Basic Usage

```python
from src.kg import FootballKnowledgeGraph

# Initialize the knowledge graph
kg = FootballKnowledgeGraph()

# Get a team's playing style
styles = kg.get_team_style("Liverpool")
print(styles)  # ['HIGH_PRESS', 'WING_PLAY']

# Find tactical counters
counters = kg.find_counter_styles("HIGH_PRESS")
print(counters)  # ['LOW_BLOCK', 'PHYSICAL']

# Analyze a tactical matchup
matchup = kg.get_tactical_matchup("Liverpool", "Burnley")
print(matchup['matchup_summary'])
```

### Tactical Matchup Analysis

```python
# Detailed matchup analysis
matchup = kg.get_tactical_matchup("Man City", "Leicester")

# Access matchup details
print(f"Home styles: {matchup['home_styles']}")
print(f"Away styles: {matchup['away_styles']}")
print(f"Home advantages: {matchup['advantages']['home']}")
print(f"Away advantages: {matchup['advantages']['away']}")
print(f"Confidence: {matchup['confidence']}")
```

### Visualization

```python
# Pretty-print tactical matchup
kg.visualize_matchup("Arsenal", "Brighton")
```

Output:
```
======================================================================
TACTICAL MATCHUP ANALYSIS: Arsenal vs Brighton
======================================================================

🏠 Arsenal (HOME)
   Styles: POSSESSION, TECHNICAL
   Advantages:
      ✓ HOME_ADVANTAGE

✈️  Brighton (AWAY)
   Styles: HIGH_PRESS, TECHNICAL
   Advantages:
      ✓ TACTICAL_COUNTER: HIGH_PRESS counters POSSESSION
      ✓ TACTICAL_ADVANTAGE: HIGH_PRESS effective against POSSESSION
...
```

## API Reference

### `FootballKnowledgeGraph`

#### `get_team_style(team_name: str) -> List[str]`
Returns list of tactical styles for a team.

**Example:**
```python
kg.get_team_style("Liverpool")  # ['HIGH_PRESS', 'WING_PLAY']
```

#### `find_counter_styles(style: str) -> List[str]`
Returns styles that counter the given style.

**Example:**
```python
kg.find_counter_styles("HIGH_PRESS")  # ['LOW_BLOCK', 'PHYSICAL']
```

#### `find_effective_against(style: str) -> List[str]`
Returns styles that the given style is effective against.

**Example:**
```python
kg.find_effective_against("HIGH_PRESS")  # ['POSSESSION']
```

#### `get_tactical_matchup(home_team: str, away_team: str) -> Dict`
Analyzes tactical matchup between two teams.

**Returns:**
```python
{
    'home_styles': List[str],
    'away_styles': List[str],
    'advantages': {
        'home': List[str],
        'away': List[str]
    },
    'matchup_summary': str,
    'confidence': str  # 'high', 'medium', or 'low'
}
```

#### `get_1hop_neighbors(team_name: str) -> Dict[str, List[str]]`
Returns all nodes 1-hop away from the team.

**Returns:**
```python
{
    'styles': List[str],
    'concepts': List[str]
}
```

#### `visualize_matchup(home_team: str, away_team: str)`
Prints a formatted tactical matchup analysis.

## Tactical Knowledge

### Counter Relationships

The knowledge graph encodes these key tactical counters:

| Style | Countered By | Effective Against |
|-------|-------------|-------------------|
| HIGH_PRESS | LOW_BLOCK, PHYSICAL | POSSESSION |
| LOW_BLOCK | WING_PLAY, SET_PIECES | HIGH_PRESS, POSSESSION |
| POSSESSION | COUNTER_ATTACK, HIGH_PRESS | PHYSICAL |
| COUNTER_ATTACK | LOW_BLOCK | HIGH_PRESS, POSSESSION |
| WING_PLAY | PHYSICAL | LOW_BLOCK |
| SET_PIECES | PHYSICAL, POSSESSION | LOW_BLOCK, TECHNICAL |

### Team Styles

Teams can have **multiple styles**. Examples:

- **Liverpool**: HIGH_PRESS + WING_PLAY (hybrid)
- **Man City**: POSSESSION + TECHNICAL
- **Burnley**: LOW_BLOCK + PHYSICAL + SET_PIECE_HEAVY
- **Leicester**: COUNTER_ATTACK

## Testing

Run the test suite:

```bash
python src/kg/football_knowledge_graph.py
```

Run the interactive demo:

```bash
python src/kg/demo.py
```

## Example: Match Prediction

```python
kg = FootballKnowledgeGraph()

# Analyze Liverpool vs Burnley
matchup = kg.get_tactical_matchup("Liverpool", "Burnley")

# Count advantages
home_advantages = len(matchup['advantages']['home'])
away_advantages = len(matchup['advantages']['away'])

if home_advantages > away_advantages:
    print(f"Prediction: Liverpool favored")
    print(f"Tactical edge: {home_advantages} vs {away_advantages}")
else:
    print(f"Prediction: Burnley has tactical counters")

print(f"Confidence: {matchup['confidence']}")
```

## Graph Statistics

- **Nodes**: 32 (20 teams, 8 styles, 4 concepts)
- **Edges**: 55+ relationships
- **Team-Style mappings**: 30 (teams can have multiple styles)
- **Tactical counters**: 10 COUNTERED_BY relationships
- **Tactical advantages**: 9 EFFECTIVE_AGAINST relationships

## Integration with ASIL Project

This knowledge graph can be integrated with:

1. **MCP Server**: Provide tactical context for match predictions
2. **Agent System**: Use graph reasoning in decision-making
3. **Baseline Model**: Augment statistical features with tactical insights
4. **Analytics Pipeline**: Generate tactical feature embeddings

### Example Integration

```python
from src.kg import FootballKnowledgeGraph
from src.agent import Agent

kg = FootballKnowledgeGraph()
agent = Agent()

# Get tactical context
matchup = kg.get_tactical_matchup("Liverpool", "Burnley")

# Use in agent reasoning
tactical_features = {
    'home_styles': matchup['home_styles'],
    'tactical_advantage_score': len(matchup['advantages']['home']) - len(matchup['advantages']['away'])
}

# Feed to agent
prediction = agent.predict_with_context(match_data, tactical_features)
```

## Future Enhancements

Potential extensions:

1. **Dynamic Style Updates**: Learn team styles from recent match data
2. **Manager Tactics**: Encode manager-specific tactical preferences
3. **Formation Analysis**: Add 4-3-3, 3-5-2 formations to the graph
4. **Player Nodes**: Individual player tactical attributes
5. **Temporal Graph**: Track tactical evolution over seasons
6. **Injury Impact**: Model how key player absences affect style
7. **Graph Neural Networks**: Use GNN embeddings for predictions

## References

Tactical concepts based on:
- Premier League tactical analysis
- Statistical football research
- Historical team playing styles (2017-2023)

## License

Part of the ASIL (Agent-based Statistical Inference Learning) project.
