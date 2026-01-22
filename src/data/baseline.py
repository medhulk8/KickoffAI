"""
Baseline prediction model for ASIL football prediction project.

This module provides market-derived baseline probabilities for match outcomes.
These probabilities serve as a benchmark against which LLM predictions are compared.

The baseline uses betting market odds converted to probabilities:
- Primary: Average market probabilities (avg across multiple bookmakers)
- Fallback: Bet365 probabilities (if avg not available)

Usage:
    from src.data.baseline import get_baseline_prediction

    result = get_baseline_prediction(match_id=1)
    print(f"Home win probability: {result['home_prob']:.1%}")
"""

import sqlite3
from pathlib import Path
from typing import TypedDict, Optional


class BaselinePrediction(TypedDict):
    """Type definition for baseline prediction result."""
    home_prob: Optional[float]
    draw_prob: Optional[float]
    away_prob: Optional[float]
    source: str  # "avg_market", "b365_fallback", or "none"


# Default database paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"
FALLBACK_DB_PATH = PROJECT_ROOT / "data" / "matches.db"


def get_database_path() -> Path:
    """
    Get the path to the database file.

    Tries the primary path first (data/processed/asil.db),
    then falls back to data/matches.db.

    Returns:
        Path to the existing database file

    Raises:
        FileNotFoundError: If no database file exists
    """
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    elif FALLBACK_DB_PATH.exists():
        return FALLBACK_DB_PATH
    else:
        raise FileNotFoundError(
            f"Database not found at:\n"
            f"  - {DEFAULT_DB_PATH}\n"
            f"  - {FALLBACK_DB_PATH}\n"
            f"Run load_data.py first to create the database."
        )


def get_baseline_prediction(
    match_id: int,
    db_path: Optional[Path] = None
) -> BaselinePrediction:
    """
    Get baseline probability prediction for a match.

    Retrieves market-derived probabilities from the database. These probabilities
    represent the consensus betting market view of the match outcome.

    The function first tries to use average market probabilities (computed from
    multiple bookmakers). If those are not available, it falls back to Bet365
    probabilities.

    Args:
        match_id: Unique identifier for the match in the database
        db_path: Optional path to database file. If not provided, uses default paths.

    Returns:
        Dictionary containing:
            - home_prob: Probability of home team winning (0.0 to 1.0)
            - draw_prob: Probability of draw (0.0 to 1.0)
            - away_prob: Probability of away team winning (0.0 to 1.0)
            - source: Data source ("avg_market", "b365_fallback", or "none")

    Raises:
        FileNotFoundError: If database file doesn't exist
        ValueError: If match_id is not found in database
        sqlite3.Error: If database query fails

    Example:
        >>> result = get_baseline_prediction(match_id=1)
        >>> print(f"Home: {result['home_prob']:.1%}")
        Home: 23.9%
        >>> print(f"Source: {result['source']}")
        Source: avg_market
    """
    # Validate input
    if not isinstance(match_id, int) or match_id < 1:
        raise ValueError(f"match_id must be a positive integer, got: {match_id}")

    # Get database path
    if db_path is None:
        db_path = get_database_path()
    elif not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")

    # Query database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                avg_home_prob, avg_draw_prob, avg_away_prob,
                b365_home_prob, b365_draw_prob, b365_away_prob
            FROM matches
            WHERE match_id = ?
            """,
            (match_id,)
        )

        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Match with ID {match_id} not found in database")

        # Try average market probabilities first
        if row['avg_home_prob'] is not None:
            return BaselinePrediction(
                home_prob=row['avg_home_prob'],
                draw_prob=row['avg_draw_prob'],
                away_prob=row['avg_away_prob'],
                source="avg_market"
            )

        # Fallback to Bet365 probabilities
        elif row['b365_home_prob'] is not None:
            return BaselinePrediction(
                home_prob=row['b365_home_prob'],
                draw_prob=row['b365_draw_prob'],
                away_prob=row['b365_away_prob'],
                source="b365_fallback"
            )

        # No probabilities available
        else:
            return BaselinePrediction(
                home_prob=None,
                draw_prob=None,
                away_prob=None,
                source="none"
            )

    finally:
        conn.close()


def get_match_info(match_id: int, db_path: Optional[Path] = None) -> dict:
    """
    Get basic match information for context.

    Args:
        match_id: Unique identifier for the match
        db_path: Optional path to database file

    Returns:
        Dictionary with match details (date, teams, score, result)
    """
    if db_path is None:
        db_path = get_database_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT date, season, home_team, away_team,
                   home_goals, away_goals, result
            FROM matches
            WHERE match_id = ?
            """,
            (match_id,)
        )

        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Match with ID {match_id} not found in database")

        return dict(row)

    finally:
        conn.close()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" BASELINE MODEL TEST")
    print("=" * 60)

    # Test match_id = 1
    match_id = 1

    try:
        # Get match info
        match_info = get_match_info(match_id)
        print(f"\nMatch {match_id}:")
        print(f"  Date:   {match_info['date']}")
        print(f"  Season: {match_info['season']}")
        print(f"  Match:  {match_info['home_team']} vs {match_info['away_team']}")
        print(f"  Score:  {match_info['home_goals']}-{match_info['away_goals']}")
        print(f"  Result: {match_info['result']}")

        # Get baseline prediction
        prediction = get_baseline_prediction(match_id)

        print(f"\nBaseline Prediction:")
        print(f"  Source: {prediction['source']}")
        print(f"  Home Win:  {prediction['home_prob']:.1%} ({prediction['home_prob']:.4f})")
        print(f"  Draw:      {prediction['draw_prob']:.1%} ({prediction['draw_prob']:.4f})")
        print(f"  Away Win:  {prediction['away_prob']:.1%} ({prediction['away_prob']:.4f})")

        # Verify probabilities sum to 1
        total = prediction['home_prob'] + prediction['draw_prob'] + prediction['away_prob']
        print(f"  Sum:       {total:.4f}")

        # Show predicted outcome
        probs = [
            ('H', prediction['home_prob']),
            ('D', prediction['draw_prob']),
            ('A', prediction['away_prob'])
        ]
        predicted = max(probs, key=lambda x: x[1])[0]
        actual = match_info['result']

        print(f"\nPredicted Outcome: {predicted}")
        print(f"Actual Outcome:    {actual}")
        print(f"Correct:           {'Yes ✓' if predicted == actual else 'No ✗'}")

        # Test a few more matches
        print("\n" + "=" * 60)
        print(" ADDITIONAL TESTS")
        print("=" * 60)

        for test_id in [100, 500, 1000]:
            try:
                info = get_match_info(test_id)
                pred = get_baseline_prediction(test_id)
                probs = [('H', pred['home_prob']), ('D', pred['draw_prob']), ('A', pred['away_prob'])]
                predicted = max(probs, key=lambda x: x[1])[0]
                correct = "✓" if predicted == info['result'] else "✗"
                print(f"\nMatch {test_id}: {info['home_team']} vs {info['away_team']}")
                print(f"  Probs: H={pred['home_prob']:.1%}, D={pred['draw_prob']:.1%}, A={pred['away_prob']:.1%}")
                print(f"  Predicted: {predicted}, Actual: {info['result']} {correct}")
            except ValueError as e:
                print(f"\nMatch {test_id}: {e}")

        print("\n" + "=" * 60)
        print(" TEST COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise
