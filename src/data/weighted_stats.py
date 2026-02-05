"""
Weighted Statistics Calculator with Recency Bias

Applies exponential decay to give more weight to recent matches.
A match from yesterday matters more than a match from 5 weeks ago.

Formula: weight = e^(-λ * days_ago)
where λ (lambda) controls decay rate.
"""

import sqlite3
from typing import Dict, Optional
from datetime import datetime
import math


class WeightedStatsCalculator:
    """
    Calculate team statistics with recency weighting.

    Recent matches receive higher weights using exponential decay.
    """

    def __init__(self, db_path: str, decay_rate: float = 0.05):
        """
        Initialize calculator.

        Args:
            db_path: Path to SQLite database
            decay_rate: Lambda for exponential decay (higher = faster decay)
                       0.05 = ~14 day half-life
                       0.10 = ~7 day half-life
        """
        self.db_path = db_path
        self.decay_rate = decay_rate

    def _calculate_weight(self, days_ago: int) -> float:
        """
        Calculate exponential decay weight for a match.

        Args:
            days_ago: Number of days since the match

        Returns:
            Weight between 0 and 1 (recent matches closer to 1)
        """
        return math.exp(-self.decay_rate * days_ago)

    def get_weighted_form(
        self,
        team_name: str,
        last_n: int = 5,
        before_date: Optional[str] = None
    ) -> Dict:
        """
        Calculate weighted form statistics.

        Args:
            team_name: Name of the team
            last_n: Number of recent matches to consider
            before_date: Only consider matches before this date (ISO format)

        Returns:
            Dict with weighted statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get reference date for weighting
        if before_date:
            ref_date = datetime.fromisoformat(before_date)
        else:
            ref_date = datetime.now()

        # Query recent matches
        query = """
            SELECT
                date,
                CASE
                    WHEN home_team = ? THEN home_goals
                    ELSE away_goals
                END as goals_for,
                CASE
                    WHEN home_team = ? THEN away_goals
                    ELSE home_goals
                END as goals_against,
                CASE
                    WHEN home_team = ? AND home_goals > away_goals THEN 'W'
                    WHEN away_team = ? AND away_goals > home_goals THEN 'W'
                    WHEN home_goals = away_goals THEN 'D'
                    ELSE 'L'
                END as result,
                CASE WHEN home_team = ? THEN 1 ELSE 0 END as is_home
            FROM matches
            WHERE (home_team = ? OR away_team = ?)
                AND home_goals IS NOT NULL
                {}
            ORDER BY date DESC
            LIMIT ?
        """.format("AND date < ?" if before_date else "")

        params = [team_name] * 7
        if before_date:
            params.append(before_date)
        params.append(last_n)

        cursor.execute(query, params)
        matches = cursor.fetchall()
        conn.close()

        if not matches:
            return self._empty_form()

        # Calculate weighted statistics
        total_weight = 0
        weighted_points = 0
        weighted_goals_for = 0
        weighted_goals_against = 0
        form_string = []

        wins = 0
        draws = 0
        losses = 0

        for match in matches:
            match_date, goals_for, goals_against, result, is_home = match

            # Calculate days ago and weight
            match_datetime = datetime.fromisoformat(match_date)
            days_ago = (ref_date - match_datetime).days
            weight = self._calculate_weight(days_ago)

            total_weight += weight

            # Weighted points
            if result == 'W':
                weighted_points += 3 * weight
                wins += 1
            elif result == 'D':
                weighted_points += 1 * weight
                draws += 1
            else:
                losses += 1

            # Weighted goals
            weighted_goals_for += goals_for * weight
            weighted_goals_against += goals_against * weight

            # Form string (recent first)
            form_string.append(result)

        # Normalize by total weight
        if total_weight > 0:
            avg_weighted_points = weighted_points / total_weight
            avg_weighted_goals_for = weighted_goals_for / total_weight
            avg_weighted_goals_against = weighted_goals_against / total_weight
        else:
            avg_weighted_points = 0
            avg_weighted_goals_for = 0
            avg_weighted_goals_against = 0

        # Calculate momentum (recent form vs older form)
        # Compare most recent 2 matches vs next 3
        recent_weight = sum(self._calculate_weight(i * 7) for i in range(2))
        older_weight = sum(self._calculate_weight((i + 2) * 7) for i in range(3))

        momentum_score = 0
        if len(matches) >= 2:
            recent_results = form_string[:2]
            recent_pts = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in recent_results)
            momentum_score = recent_pts / 2  # 0-3 scale

        return {
            'matches_played': len(matches),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': int(weighted_goals_for),  # For display
            'goals_conceded': int(weighted_goals_against),  # For display
            'form_string': '-'.join(form_string),

            # New weighted metrics
            'weighted_points_per_game': round(avg_weighted_points, 2),
            'weighted_goals_per_game': round(avg_weighted_goals_for, 2),
            'weighted_goals_conceded': round(avg_weighted_goals_against, 2),
            'weighted_goal_difference': round(avg_weighted_goals_for - avg_weighted_goals_against, 2),
            'momentum_score': round(momentum_score, 2),  # 0-3 scale

            # Traditional metrics for comparison
            'points_per_game': round(
                (wins * 3 + draws) / len(matches), 2
            ) if len(matches) > 0 else 0,
        }

    def _empty_form(self) -> Dict:
        """Return empty form statistics."""
        return {
            'matches_played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'form_string': 'N/A',
            'weighted_points_per_game': 0,
            'weighted_goals_per_game': 0,
            'weighted_goals_conceded': 0,
            'weighted_goal_difference': 0,
            'momentum_score': 0,
            'points_per_game': 0,
        }

    def compare_form_differential(
        self,
        home_form: Dict,
        away_form: Dict
    ) -> Dict:
        """
        Compare weighted form between two teams.

        Args:
            home_form: Weighted form for home team
            away_form: Weighted form for away team

        Returns:
            Dict with comparison metrics
        """
        ppg_diff = home_form['weighted_points_per_game'] - away_form['weighted_points_per_game']
        goal_diff = home_form['weighted_goal_difference'] - away_form['weighted_goal_difference']
        momentum_diff = home_form['momentum_score'] - away_form['momentum_score']

        # Classify form advantage
        if abs(ppg_diff) < 0.3:
            form_advantage = "even"
        elif ppg_diff > 0:
            form_advantage = "home" if ppg_diff > 0.8 else "home_slight"
        else:
            form_advantage = "away" if ppg_diff < -0.8 else "away_slight"

        # Momentum classification
        if abs(momentum_diff) < 0.5:
            momentum_advantage = "even"
        elif momentum_diff > 0:
            momentum_advantage = "home"
        else:
            momentum_advantage = "away"

        return {
            'ppg_differential': round(ppg_diff, 2),
            'goal_differential': round(goal_diff, 2),
            'momentum_differential': round(momentum_diff, 2),
            'form_advantage': form_advantage,
            'momentum_advantage': momentum_advantage,
            'home_form_strength': round(home_form['weighted_points_per_game'], 2),
            'away_form_strength': round(away_form['weighted_points_per_game'], 2),
        }


if __name__ == "__main__":
    # Test the weighted calculator
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"

    calc = WeightedStatsCalculator(str(DB_PATH), decay_rate=0.05)

    # Test with Liverpool
    form = calc.get_weighted_form("Liverpool", last_n=5)

    print("Weighted Form for Liverpool:")
    print(f"  Form string: {form['form_string']}")
    print(f"  Traditional PPG: {form['points_per_game']}")
    print(f"  Weighted PPG: {form['weighted_points_per_game']}")
    print(f"  Weighted Goals/Game: {form['weighted_goals_per_game']}")
    print(f"  Momentum Score: {form['momentum_score']}/3")
