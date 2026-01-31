# Data Directory

This directory contains all data files for the KickoffAI project.

## Structure

```
data/
├── README.md                    # This file
├── evaluation_results.csv       # Latest evaluation results
├── phase_5_insights.txt         # Analysis insights
├── phase_6_analysis.json        # Search strategy analysis
├── matches.db                   # Legacy match database
├── cache/                       # Search cache (auto-generated)
│   └── search_cache.db
├── processed/                   # Processed data files
│   └── asil.db                  # Main SQLite database
└── raw/                         # Raw data files
    └── *.csv                    # Original CSV files
```

## File Descriptions

### Main Database
- **`processed/asil.db`** - Main SQLite database containing:
  - Match results (teams, scores, dates)
  - Team statistics (form, goals, clean sheets)
  - Head-to-head history
  - Baseline probabilities

### Evaluation Results
- **`evaluation_results.csv`** - Latest batch evaluation results with:
  - Match details (teams, date, actual outcome)
  - Baseline predictions
  - LLM predictions
  - Confidence levels
  - Brier scores
  - Search statistics

### Analysis Files
- **`phase_5_insights.txt`** - Granular analysis output
- **`phase_6_analysis.json`** - Search strategy comparison

### Cache
- **`cache/search_cache.db`** - Persistent web search cache
  - Reduces API calls
  - 24-hour TTL by default
  - Auto-managed by SmartWebSearch

## Data Sources

- **Premier League Match Data** - Historical results from 2021-2022 season
- **Bookmaker Odds** - Baseline probabilities
- **Web Search** - Current context via Tavily API

## Database Schema

### Main Tables (asil.db)

**matches**
- `id` - Match ID
- `home_team` - Home team name
- `away_team` - Away team name
- `date` - Match date
- `home_goals` - Home team goals
- `away_goals` - Away team goals

**team_stats**
- `team_id` - Team identifier
- `match_id` - Related match
- `goals_scored` - Goals in last 5 matches
- `goals_conceded` - Goals conceded
- `points` - Points earned
- `form_string` - W/D/L sequence

## Usage

### Loading Data
```python
from src.data.load_data import load_match_data

# Load specific match
match = load_match_data(match_id=100)

# Load team statistics
stats = load_team_stats(team_name="Liverpool")
```

### Querying Database
```python
import sqlite3

conn = sqlite3.connect('data/processed/asil.db')
cursor = conn.cursor()

# Get all matches for a team
cursor.execute("""
    SELECT * FROM matches
    WHERE home_team = ? OR away_team = ?
""", ("Liverpool", "Liverpool"))
```

## Data Privacy

⚠️ **Note:** Do not commit large database files to Git. The `.gitignore` is configured to:
- Include: Schema and small sample data
- Exclude: Full databases, cache files, large CSVs

## Maintenance

### Rebuilding Database
```bash
python -m src.data.load_data --rebuild
```

### Clearing Cache
```bash
rm -rf data/cache/search_cache.db
```

### Backup
```bash
# Backup main database
cp data/processed/asil.db data/processed/asil_backup_$(date +%Y%m%d).db
```

## File Sizes (Approximate)

- `asil.db`: ~300 KB
- `evaluation_results.csv`: ~15 KB
- `search_cache.db`: Variable (grows with usage)

## Updates

- **2026-01-31:** Latest evaluation results (99 matches)
- **2026-01-24:** Knowledge graph integration
- **2026-01-23:** Initial database creation
