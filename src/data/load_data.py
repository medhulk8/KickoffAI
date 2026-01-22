"""
Data loading and processing module for ASIL football prediction project.

This module handles:
- Loading match data from football-data.co.uk CSV files
- Converting betting odds to probabilities
- Standardizing team names
- Populating the SQLite database

Usage:
    python src/data/load_data.py
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DB_PATH = PROJECT_ROOT / "data" / "matches.db"

# Team name standardization mapping
TEAM_NAME_MAPPING = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton",
    "Brighton": "Brighton & Hove Albion",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
}


def odds_to_prob(
    home_odds: Optional[float],
    draw_odds: Optional[float],
    away_odds: Optional[float]
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Convert decimal betting odds to normalized probabilities.

    Betting odds represent implied probabilities, but bookmakers include
    a margin (overround) so raw implied probabilities sum to > 1.0.
    This function normalizes them to sum to exactly 1.0.

    Formula:
        raw_prob = 1 / odds
        normalized_prob = raw_prob / sum(all_raw_probs)

    Args:
        home_odds: Decimal odds for home win (e.g., 2.5 means $2.50 return per $1)
        draw_odds: Decimal odds for draw
        away_odds: Decimal odds for away win

    Returns:
        Tuple of (home_prob, draw_prob, away_prob), normalized to sum to 1.0.
        Returns (None, None, None) if any odds are missing or invalid.

    Example:
        >>> odds_to_prob(2.0, 3.5, 4.0)
        (0.4667, 0.2667, 0.2333)  # approximately
    """
    # Check for missing or invalid odds
    if home_odds is None or draw_odds is None or away_odds is None:
        return (None, None, None)

    try:
        home_odds = float(home_odds)
        draw_odds = float(draw_odds)
        away_odds = float(away_odds)
    except (ValueError, TypeError):
        return (None, None, None)

    # Odds must be > 1.0 for valid decimal odds
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return (None, None, None)

    # Convert to raw implied probabilities
    home_raw = 1.0 / home_odds
    draw_raw = 1.0 / draw_odds
    away_raw = 1.0 / away_odds

    # Normalize to sum to 1.0 (removes bookmaker margin)
    total = home_raw + draw_raw + away_raw

    if total <= 0:
        return (None, None, None)

    return (
        round(home_raw / total, 4),
        round(draw_raw / total, 4),
        round(away_raw / total, 4)
    )


def clean_team_name(name: str) -> str:
    """
    Standardize team names to a consistent format.

    Football data sources use inconsistent team names (e.g., "Man United"
    vs "Manchester United"). This function maps common variants to their
    full, standardized names for consistency in the database.

    Args:
        name: Raw team name from data source

    Returns:
        Standardized team name, with whitespace stripped

    Example:
        >>> clean_team_name("Man United")
        'Manchester United'
        >>> clean_team_name("  Arsenal  ")
        'Arsenal'
    """
    if not name:
        return name

    # Strip whitespace
    name = name.strip()

    # Apply mapping if exists, otherwise return cleaned name
    return TEAM_NAME_MAPPING.get(name, name)


def load_csv_to_dataframe(csv_path: Path, season: str) -> pd.DataFrame:
    """
    Load a football-data.co.uk CSV file into a pandas DataFrame.

    The CSV files from football-data.co.uk contain match results and
    betting odds for a single season. This function loads the data
    and adds a season identifier column.

    Args:
        csv_path: Path to the CSV file
        season: Season identifier (e.g., '2122' for 2021-22)

    Returns:
        DataFrame with all CSV columns plus a 'season' column

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    logger.info(f"Loading CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df['season'] = season

    logger.info(f"Loaded {len(df)} matches from {csv_path.name}")
    return df


def process_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw match data into the format needed for the database.

    This function:
    1. Extracts relevant columns (date, teams, goals, shots, corners)
    2. Converts betting odds to probabilities for multiple bookmakers
    3. Calculates average market probability across bookmakers
    4. Standardizes team names
    5. Converts date format to ISO standard (YYYY-MM-DD)

    Bookmakers used (when available):
    - B365: Bet365
    - PS/P: Pinnacle Sports
    - BW: Betway
    - IW: Interwetten
    - WH: William Hill
    - VC: VC Bet

    Args:
        df: Raw DataFrame from load_csv_to_dataframe()

    Returns:
        Processed DataFrame with columns matching the matches table schema
    """
    logger.info("Processing match data...")

    processed = pd.DataFrame()

    # Basic match info
    processed['season'] = df['season']
    processed['home_team'] = df['HomeTeam'].apply(clean_team_name)
    processed['away_team'] = df['AwayTeam'].apply(clean_team_name)
    processed['home_goals'] = df['FTHG'].astype(int)
    processed['away_goals'] = df['FTAG'].astype(int)
    processed['result'] = df['FTR']

    # Match statistics (may have missing values)
    processed['home_shots'] = pd.to_numeric(df.get('HS'), errors='coerce')
    processed['away_shots'] = pd.to_numeric(df.get('AS'), errors='coerce')
    processed['home_corners'] = pd.to_numeric(df.get('HC'), errors='coerce')
    processed['away_corners'] = pd.to_numeric(df.get('AC'), errors='coerce')

    # Convert date to ISO format (YYYY-MM-DD)
    # football-data.co.uk uses DD/MM/YYYY format
    processed['date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')

    # Convert Bet365 odds to probabilities
    b365_probs = df.apply(
        lambda row: odds_to_prob(row.get('B365H'), row.get('B365D'), row.get('B365A')),
        axis=1
    )
    processed['b365_home_prob'] = b365_probs.apply(lambda x: x[0])
    processed['b365_draw_prob'] = b365_probs.apply(lambda x: x[1])
    processed['b365_away_prob'] = b365_probs.apply(lambda x: x[2])

    # Calculate average market probability from multiple bookmakers
    # Use columns that exist in the dataframe
    bookmaker_odds_cols = [
        ('B365H', 'B365D', 'B365A'),  # Bet365
        ('PSH', 'PSD', 'PSA'),         # Pinnacle
        ('BWH', 'BWD', 'BWA'),         # Betway
        ('IWH', 'IWD', 'IWA'),         # Interwetten
        ('WHH', 'WHD', 'WHA'),         # William Hill
        ('VCH', 'VCD', 'VCA'),         # VC Bet
    ]

    def calculate_avg_probs(row):
        """Calculate average probabilities across all available bookmakers."""
        home_probs = []
        draw_probs = []
        away_probs = []

        for h_col, d_col, a_col in bookmaker_odds_cols:
            if h_col in row.index and d_col in row.index and a_col in row.index:
                probs = odds_to_prob(row.get(h_col), row.get(d_col), row.get(a_col))
                if probs[0] is not None:
                    home_probs.append(probs[0])
                    draw_probs.append(probs[1])
                    away_probs.append(probs[2])

        if home_probs:
            return (
                round(sum(home_probs) / len(home_probs), 4),
                round(sum(draw_probs) / len(draw_probs), 4),
                round(sum(away_probs) / len(away_probs), 4)
            )
        return (None, None, None)

    avg_probs = df.apply(calculate_avg_probs, axis=1)
    processed['avg_home_prob'] = avg_probs.apply(lambda x: x[0])
    processed['avg_draw_prob'] = avg_probs.apply(lambda x: x[1])
    processed['avg_away_prob'] = avg_probs.apply(lambda x: x[2])

    logger.info(f"Processed {len(processed)} matches")
    return processed


def create_database(db_path: Path) -> sqlite3.Connection:
    """
    Create the SQLite database by executing the schema.sql file.

    If the database file already exists, this will fail on table creation
    (tables already exist). Use this for initial setup only.

    Args:
        db_path: Path where the database file should be created

    Returns:
        Open SQLite connection to the new database

    Raises:
        FileNotFoundError: If schema.sql doesn't exist
        sqlite3.Error: If database creation fails
    """
    logger.info(f"Creating database at: {db_path}")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Read schema
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    schema_sql = SCHEMA_PATH.read_text()

    # Create database and execute schema
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)
    conn.commit()

    logger.info("Database schema created successfully")
    return conn


def insert_matches(conn: sqlite3.Connection, matches_df: pd.DataFrame) -> int:
    """
    Insert processed matches into the database.

    This function:
    1. Populates the teams table with unique team names
    2. Inserts matches, skipping any that already exist (based on
       date + home_team + away_team combination)

    Duplicate detection prevents re-inserting the same match if the
    script is run multiple times.

    Args:
        conn: SQLite database connection
        matches_df: Processed DataFrame from process_matches()

    Returns:
        Number of matches successfully inserted
    """
    cursor = conn.cursor()
    inserted_count = 0

    # Collect all unique team names
    all_teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())

    # Insert teams (ignore if already exists)
    for team in all_teams:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO teams (team_name) VALUES (?)",
                (team,)
            )
        except sqlite3.Error as e:
            logger.warning(f"Failed to insert team {team}: {e}")

    logger.info(f"Processed {len(all_teams)} unique teams")

    # Insert matches
    for _, row in matches_df.iterrows():
        # Check if match already exists (same date, home team, away team)
        cursor.execute(
            """
            SELECT match_id FROM matches
            WHERE date = ? AND home_team = ? AND away_team = ?
            """,
            (row['date'], row['home_team'], row['away_team'])
        )

        if cursor.fetchone() is not None:
            # Match already exists, skip
            continue

        try:
            cursor.execute(
                """
                INSERT INTO matches (
                    date, season, home_team, away_team, home_goals, away_goals,
                    result, home_shots, away_shots, home_corners, away_corners,
                    b365_home_prob, b365_draw_prob, b365_away_prob,
                    avg_home_prob, avg_draw_prob, avg_away_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row['date'],
                    row['season'],
                    row['home_team'],
                    row['away_team'],
                    row['home_goals'],
                    row['away_goals'],
                    row['result'],
                    row['home_shots'] if pd.notna(row['home_shots']) else None,
                    row['away_shots'] if pd.notna(row['away_shots']) else None,
                    row['home_corners'] if pd.notna(row['home_corners']) else None,
                    row['away_corners'] if pd.notna(row['away_corners']) else None,
                    row['b365_home_prob'],
                    row['b365_draw_prob'],
                    row['b365_away_prob'],
                    row['avg_home_prob'],
                    row['avg_draw_prob'],
                    row['avg_away_prob'],
                )
            )
            inserted_count += 1
        except sqlite3.Error as e:
            logger.error(f"Failed to insert match {row['home_team']} vs {row['away_team']}: {e}")

    conn.commit()
    logger.info(f"Inserted {inserted_count} matches into database")
    return inserted_count


def main():
    """
    Main pipeline: load all CSV files and populate the database.

    This function orchestrates the full ETL process:
    1. Creates the database (if it doesn't exist)
    2. Loads each season's CSV file
    3. Processes and inserts the data
    4. Prints summary statistics
    """
    logger.info("=" * 60)
    logger.info("Starting data loading pipeline")
    logger.info("=" * 60)

    # Define seasons and their CSV files
    seasons = [
        ('2122', DATA_RAW_DIR / 'E0_2122.csv'),
        ('2223', DATA_RAW_DIR / 'E0_2223.csv'),
        ('2324', DATA_RAW_DIR / 'E0_2324.csv'),
    ]

    # Check if database exists - if so, delete and recreate
    if DB_PATH.exists():
        logger.info(f"Removing existing database: {DB_PATH}")
        DB_PATH.unlink()

    # Create fresh database
    conn = create_database(DB_PATH)

    total_inserted = 0

    # Process each season
    for season, csv_path in seasons:
        try:
            # Load CSV
            df = load_csv_to_dataframe(csv_path, season)

            # Process data
            processed_df = process_matches(df)

            # Insert into database
            inserted = insert_matches(conn, processed_df)
            total_inserted += inserted

        except FileNotFoundError as e:
            logger.error(f"Skipping season {season}: {e}")
        except Exception as e:
            logger.error(f"Error processing season {season}: {e}")
            raise

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    cursor = conn.cursor()

    # Total matches
    cursor.execute("SELECT COUNT(*) FROM matches")
    total_matches = cursor.fetchone()[0]
    logger.info(f"Total matches in database: {total_matches}")

    # Matches per season
    cursor.execute("SELECT season, COUNT(*) FROM matches GROUP BY season ORDER BY season")
    for season, count in cursor.fetchall():
        logger.info(f"  Season {season}: {count} matches")

    # Total teams
    cursor.execute("SELECT COUNT(*) FROM teams")
    total_teams = cursor.fetchone()[0]
    logger.info(f"Total teams in database: {total_teams}")

    # Results distribution
    cursor.execute("""
        SELECT result, COUNT(*), ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM matches), 1)
        FROM matches GROUP BY result ORDER BY result
    """)
    logger.info("Results distribution:")
    for result, count, pct in cursor.fetchall():
        result_name = {'H': 'Home wins', 'D': 'Draws', 'A': 'Away wins'}[result]
        logger.info(f"  {result_name}: {count} ({pct}%)")

    conn.close()
    logger.info("=" * 60)
    logger.info("Data loading complete!")
    logger.info("=" * 60)

    return total_inserted


if __name__ == "__main__":
    main()
