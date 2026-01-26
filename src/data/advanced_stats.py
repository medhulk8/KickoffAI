"""
Advanced Statistics Calculator for ASIL Predictions

Calculate advanced team statistics from match data including:
- Attacking metrics (shots, corners, goals, efficiency)
- Defensive metrics (shots conceded, clean sheets)
- Head-to-head history analysis

Works with the available database columns:
- home_shots, away_shots
- home_corners, away_corners
- home_goals, away_goals
- result, date
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AdvancedStatsCalculator:
    """
    Calculate advanced team statistics from match data.

    Provides attacking, defensive, and efficiency metrics
    based on historical match data.
    """

    def __init__(self, db_path: str):
        """
        Initialize the calculator with database path.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def get_team_advanced_stats(
        self,
        team_name: str,
        last_n: int = 5,
        before_date: Optional[str] = None,
        home_only: bool = False,
        away_only: bool = False
    ) -> Dict:
        """
        Calculate comprehensive advanced statistics for a team.

        Args:
            team_name: Name of the team
            last_n: Number of recent matches to analyze
            before_date: Only consider matches before this date (YYYY-MM-DD)
            home_only: Only consider home matches
            away_only: Only consider away matches

        Returns:
            dict with:
            - matches_analyzed: Number of matches used
            - attacking: shots, corners, goals, efficiency metrics
            - defensive: shots conceded, clean sheets, solidity
            - efficiency: conversion rate, threat level
        """
        conn = sqlite3.connect(self.db_path)

        # Build query based on filters
        if home_only:
            where_clause = "WHERE home_team = ?"
            params = [team_name]
        elif away_only:
            where_clause = "WHERE away_team = ?"
            params = [team_name]
        else:
            where_clause = "WHERE (home_team = ? OR away_team = ?)"
            params = [team_name, team_name]

        # Add date filter if provided
        if before_date:
            where_clause += " AND date < ?"
            params.append(before_date)

        params.append(last_n)

        query = f"""
            SELECT
                home_team, away_team, result,
                home_goals, away_goals,
                home_shots, away_shots,
                home_corners, away_corners,
                date
            FROM matches
            {where_clause}
            ORDER BY date DESC
            LIMIT ?
        """

        cursor = conn.execute(query, params)
        matches = cursor.fetchall()
        conn.close()

        if not matches:
            return self._empty_stats()

        # Calculate all statistics
        stats = {
            'matches_analyzed': len(matches),
            'attacking': self._calc_attacking(matches, team_name),
            'defensive': self._calc_defensive(matches, team_name),
            'efficiency': self._calc_efficiency(matches, team_name),
            'form': self._calc_form(matches, team_name)
        }

        return stats

    def _calc_attacking(self, matches: List[Tuple], team: str) -> Dict:
        """
        Calculate attacking metrics: shots, corners, goals.

        Args:
            matches: List of match tuples from database
            team: Team name to analyze

        Returns:
            dict with attacking metrics
        """
        shots, corners, goals = 0, 0, 0

        for m in matches:
            home_team, away_team, result, hg, ag, hs, aws, hc, ac, date = m
            is_home = home_team == team

            if is_home:
                shots += hs or 0
                corners += hc or 0
                goals += hg or 0
            else:
                shots += aws or 0
                corners += ac or 0
                goals += ag or 0

        n = len(matches)
        avg_shots = shots / n if n > 0 else 0
        avg_goals = goals / n if n > 0 else 0

        return {
            'avg_shots': round(avg_shots, 1),
            'avg_corners': round(corners / n, 1) if n > 0 else 0,
            'avg_goals': round(avg_goals, 2),
            'total_goals': goals,
            'total_shots': shots,
            # Estimate shot accuracy as goals/shots (proxy for shots on target)
            'shot_accuracy': round((goals / shots * 100) if shots > 0 else 0, 1),
            'goals_per_shot': round(goals / shots, 3) if shots > 0 else 0
        }

    def _calc_defensive(self, matches: List[Tuple], team: str) -> Dict:
        """
        Calculate defensive metrics: shots conceded, clean sheets.

        Args:
            matches: List of match tuples from database
            team: Team name to analyze

        Returns:
            dict with defensive metrics
        """
        shots_conceded, goals_conceded, clean_sheets = 0, 0, 0

        for m in matches:
            home_team, away_team, result, hg, ag, hs, aws, hc, ac, date = m
            is_home = home_team == team

            if is_home:
                shots_conceded += aws or 0
                gc = ag or 0
            else:
                shots_conceded += hs or 0
                gc = hg or 0

            goals_conceded += gc
            if gc == 0:
                clean_sheets += 1

        n = len(matches)
        avg_goals_conceded = goals_conceded / n if n > 0 else 0

        return {
            'avg_shots_conceded': round(shots_conceded / n, 1) if n > 0 else 0,
            'avg_goals_conceded': round(avg_goals_conceded, 2),
            'total_goals_conceded': goals_conceded,
            'clean_sheets': clean_sheets,
            'clean_sheet_rate': round(clean_sheets / n, 2) if n > 0 else 0,
            # Defensive solidity: ability to prevent goals from shots
            'defensive_solidity': round(1 - (goals_conceded / (shots_conceded or 1)), 3)
        }

    def _calc_efficiency(self, matches: List[Tuple], team: str) -> Dict:
        """
        Calculate efficiency metrics: conversion, threat level.

        Args:
            matches: List of match tuples from database
            team: Team name to analyze

        Returns:
            dict with efficiency metrics
        """
        shots, corners, goals = 0, 0, 0
        opp_shots = 0

        for m in matches:
            home_team, away_team, result, hg, ag, hs, aws, hc, ac, date = m
            is_home = home_team == team

            if is_home:
                shots += hs or 0
                corners += hc or 0
                goals += hg or 0
                opp_shots += aws or 0
            else:
                shots += aws or 0
                corners += ac or 0
                goals += ag or 0
                opp_shots += hs or 0

        n = len(matches)

        # Shot dominance: ratio of our shots to opponent shots
        shot_ratio = shots / opp_shots if opp_shots > 0 else 1.0

        return {
            'conversion_rate': round((goals / shots * 100) if shots > 0 else 0, 1),
            'corners_per_game': round(corners / n, 1) if n > 0 else 0,
            'shot_dominance': round(shot_ratio, 2),
            # Attacking threat: normalized score (0-10)
            'attacking_threat': round(min(10, (shots / n) * 0.7 + (goals / n) * 2), 1) if n > 0 else 0
        }

    def _calc_form(self, matches: List[Tuple], team: str) -> Dict:
        """
        Calculate form metrics from results.

        Args:
            matches: List of match tuples from database
            team: Team name to analyze

        Returns:
            dict with form metrics
        """
        wins, draws, losses = 0, 0, 0
        points = 0

        for m in matches:
            home_team, away_team, result, hg, ag, hs, aws, hc, ac, date = m
            is_home = home_team == team

            # Determine result from team's perspective
            if is_home:
                if result == 'H':
                    wins += 1
                    points += 3
                elif result == 'D':
                    draws += 1
                    points += 1
                else:
                    losses += 1
            else:
                if result == 'A':
                    wins += 1
                    points += 3
                elif result == 'D':
                    draws += 1
                    points += 1
                else:
                    losses += 1

        n = len(matches)

        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points': points,
            'points_per_game': round(points / n, 2) if n > 0 else 0,
            'win_rate': round(wins / n, 2) if n > 0 else 0
        }

    def _empty_stats(self) -> Dict:
        """Return empty stats when no matches found."""
        return {
            'matches_analyzed': 0,
            'attacking': {
                'avg_shots': 0, 'avg_goals': 0, 'shot_accuracy': 0,
                'avg_corners': 0, 'total_goals': 0, 'total_shots': 0,
                'goals_per_shot': 0
            },
            'defensive': {
                'avg_shots_conceded': 0, 'avg_goals_conceded': 0,
                'clean_sheet_rate': 0, 'defensive_solidity': 0,
                'total_goals_conceded': 0, 'clean_sheets': 0
            },
            'efficiency': {
                'conversion_rate': 0, 'corners_per_game': 0,
                'shot_dominance': 1.0, 'attacking_threat': 0
            },
            'form': {
                'wins': 0, 'draws': 0, 'losses': 0,
                'points': 0, 'points_per_game': 0, 'win_rate': 0
            }
        }

    def get_head_to_head_stats(
        self,
        team1: str,
        team2: str,
        last_n: int = 5,
        before_date: Optional[str] = None
    ) -> Dict:
        """
        Get head-to-head statistics between two teams.

        Args:
            team1: First team (typically home team)
            team2: Second team (typically away team)
            last_n: Number of recent meetings to analyze
            before_date: Only consider matches before this date

        Returns:
            dict with:
            - matches_played: Total meetings analyzed
            - team1_wins, team2_wins, draws: Win counts
            - recent_results: List of recent results from team1's perspective
            - avg_goals: Average goals per team
            - dominance: Which team historically dominates
            - last_winner: Winner of most recent meeting
        """
        conn = sqlite3.connect(self.db_path)

        # Build query
        where_clause = """
            WHERE (home_team = ? AND away_team = ?)
               OR (home_team = ? AND away_team = ?)
        """
        params = [team1, team2, team2, team1]

        if before_date:
            where_clause += " AND date < ?"
            params.append(before_date)

        params.append(last_n)

        query = f"""
            SELECT
                home_team, away_team, result,
                home_goals, away_goals,
                date
            FROM matches
            {where_clause}
            ORDER BY date DESC
            LIMIT ?
        """

        cursor = conn.execute(query, params)
        matches = cursor.fetchall()
        conn.close()

        if not matches:
            return {
                'matches_played': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'recent_results': [],
                'avg_goals_team1': 0,
                'avg_goals_team2': 0,
                'total_goals_team1': 0,
                'total_goals_team2': 0,
                'last_winner': None,
                'dominance': 'no_history',
                'high_scoring': False
            }

        team1_wins, team2_wins, draws = 0, 0, 0
        team1_goals, team2_goals = 0, 0
        recent_results = []

        for match in matches:
            home, away, result, hg, ag, date = match

            # Determine outcome from team1's perspective
            if home == team1:
                team1_goals += hg
                team2_goals += ag
                if result == 'H':
                    team1_wins += 1
                    recent_results.append(f"W ({hg}-{ag})")
                elif result == 'D':
                    draws += 1
                    recent_results.append(f"D ({hg}-{ag})")
                else:
                    team2_wins += 1
                    recent_results.append(f"L ({hg}-{ag})")
            else:
                # team1 was away
                team1_goals += ag
                team2_goals += hg
                if result == 'A':
                    team1_wins += 1
                    recent_results.append(f"W ({ag}-{hg})")
                elif result == 'D':
                    draws += 1
                    recent_results.append(f"D ({ag}-{hg})")
                else:
                    team2_wins += 1
                    recent_results.append(f"L ({ag}-{hg})")

        n = len(matches)

        # Determine last winner
        if recent_results:
            first_result = recent_results[0]
            if first_result.startswith('W'):
                last_winner = 'team1'
            elif first_result.startswith('L'):
                last_winner = 'team2'
            else:
                last_winner = 'draw'
        else:
            last_winner = None

        # Determine dominance
        if team1_wins > team2_wins + 1:
            dominance = 'team1'
        elif team2_wins > team1_wins + 1:
            dominance = 'team2'
        else:
            dominance = 'balanced'

        # Check if historically high-scoring fixture
        avg_total_goals = (team1_goals + team2_goals) / n if n > 0 else 0
        high_scoring = avg_total_goals >= 3.0

        return {
            'matches_played': n,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'draws': draws,
            'recent_results': recent_results,
            'avg_goals_team1': round(team1_goals / n, 1) if n > 0 else 0,
            'avg_goals_team2': round(team2_goals / n, 1) if n > 0 else 0,
            'total_goals_team1': team1_goals,
            'total_goals_team2': team2_goals,
            'last_winner': last_winner,
            'dominance': dominance,
            'high_scoring': high_scoring,
            'avg_total_goals': round(avg_total_goals, 1)
        }

    def get_home_away_split(
        self,
        team_name: str,
        last_n: int = 10,
        before_date: Optional[str] = None
    ) -> Dict:
        """
        Get separate home and away statistics for a team.

        Args:
            team_name: Name of the team
            last_n: Number of matches per venue type
            before_date: Only consider matches before this date

        Returns:
            dict with home_stats and away_stats
        """
        home_stats = self.get_team_advanced_stats(
            team_name,
            last_n=last_n,
            before_date=before_date,
            home_only=True
        )

        away_stats = self.get_team_advanced_stats(
            team_name,
            last_n=last_n,
            before_date=before_date,
            away_only=True
        )

        # Calculate home advantage metrics
        home_ppg = home_stats.get('form', {}).get('points_per_game', 0)
        away_ppg = away_stats.get('form', {}).get('points_per_game', 0)

        return {
            'home': home_stats,
            'away': away_stats,
            'home_advantage': round(home_ppg - away_ppg, 2),
            'home_stronger': home_ppg > away_ppg + 0.3
        }

    def format_stats_for_prompt(
        self,
        team_name: str,
        stats: Dict,
        h2h: Optional[Dict] = None,
        opponent_name: Optional[str] = None
    ) -> str:
        """
        Format stats into a readable string for LLM prompt.

        Args:
            team_name: Name of the team
            stats: Advanced stats dict
            h2h: Optional head-to-head stats
            opponent_name: Optional opponent name for H2H context

        Returns:
            Formatted string for prompt
        """
        att = stats.get('attacking', {})
        defe = stats.get('defensive', {})
        eff = stats.get('efficiency', {})
        form = stats.get('form', {})

        text = f"""
{team_name} Advanced Statistics (last {stats.get('matches_analyzed', 0)} matches):

ATTACKING:
- Shots per game: {att.get('avg_shots', 0):.1f}
- Goals per game: {att.get('avg_goals', 0):.2f}
- Conversion rate: {eff.get('conversion_rate', 0):.1f}%
- Corners per game: {att.get('avg_corners', 0):.1f}

DEFENSIVE:
- Shots conceded per game: {defe.get('avg_shots_conceded', 0):.1f}
- Goals conceded per game: {defe.get('avg_goals_conceded', 0):.2f}
- Clean sheet rate: {defe.get('clean_sheet_rate', 0):.0%}

EFFICIENCY:
- Shot dominance: {eff.get('shot_dominance', 1):.2f}x opponent
- Attacking threat: {eff.get('attacking_threat', 0):.1f}/10

FORM:
- Record: {form.get('wins', 0)}W-{form.get('draws', 0)}D-{form.get('losses', 0)}L
- Points per game: {form.get('points_per_game', 0):.2f}
"""

        if h2h and opponent_name and h2h.get('matches_played', 0) > 0:
            text += f"""
HEAD-TO-HEAD vs {opponent_name} (last {h2h['matches_played']} meetings):
- {team_name} wins: {h2h['team1_wins']}
- {opponent_name} wins: {h2h['team2_wins']}
- Draws: {h2h['draws']}
- Recent: {', '.join(h2h['recent_results'][:3])}
- {team_name} avg goals: {h2h['avg_goals_team1']:.1f}
- Historical pattern: {h2h['dominance'].upper()}
"""

        return text.strip()


# ============================================================================
# Tests
# ============================================================================

def test_advanced_stats():
    """Test the advanced stats calculator."""
    from pathlib import Path

    print("=" * 70)
    print(" ADVANCED STATS CALCULATOR TEST")
    print("=" * 70)

    # Find database
    db_path = Path(__file__).parent.parent.parent / "data" / "processed" / "asil.db"
    if not db_path.exists():
        print(f"\n✗ Database not found: {db_path}")
        return

    print(f"\n✓ Database: {db_path}")

    # Initialize calculator
    calc = AdvancedStatsCalculator(str(db_path))
    print("✓ Calculator initialized")

    # Test 1: Get team stats
    print("\n" + "-" * 50)
    print(" TEST 1: Team Advanced Stats (Liverpool)")
    print("-" * 50)

    stats = calc.get_team_advanced_stats("Liverpool", last_n=5)

    print(f"\nMatches analyzed: {stats['matches_analyzed']}")
    print(f"\nATTACKING:")
    for k, v in stats['attacking'].items():
        print(f"  {k}: {v}")
    print(f"\nDEFENSIVE:")
    for k, v in stats['defensive'].items():
        print(f"  {k}: {v}")
    print(f"\nEFFICIENCY:")
    for k, v in stats['efficiency'].items():
        print(f"  {k}: {v}")
    print(f"\nFORM:")
    for k, v in stats['form'].items():
        print(f"  {k}: {v}")

    # Test 2: Head-to-head
    print("\n" + "-" * 50)
    print(" TEST 2: Head-to-Head (Liverpool vs Arsenal)")
    print("-" * 50)

    h2h = calc.get_head_to_head_stats("Liverpool", "Arsenal", last_n=5)

    print(f"\nMatches played: {h2h['matches_played']}")
    print(f"Liverpool wins: {h2h['team1_wins']}")
    print(f"Arsenal wins: {h2h['team2_wins']}")
    print(f"Draws: {h2h['draws']}")
    print(f"Recent results: {h2h['recent_results']}")
    print(f"Liverpool avg goals: {h2h['avg_goals_team1']}")
    print(f"Arsenal avg goals: {h2h['avg_goals_team2']}")
    print(f"Dominance: {h2h['dominance']}")
    print(f"High scoring: {h2h['high_scoring']}")

    # Test 3: Home/Away split
    print("\n" + "-" * 50)
    print(" TEST 3: Home/Away Split (Manchester City)")
    print("-" * 50)

    split = calc.get_home_away_split("Manchester City", last_n=5)

    print(f"\nHome form: {split['home']['form']}")
    print(f"Away form: {split['away']['form']}")
    print(f"Home advantage: {split['home_advantage']} PPG")
    print(f"Home stronger: {split['home_stronger']}")

    # Test 4: Format for prompt
    print("\n" + "-" * 50)
    print(" TEST 4: Formatted Stats for Prompt")
    print("-" * 50)

    formatted = calc.format_stats_for_prompt(
        "Liverpool",
        stats,
        h2h,
        "Arsenal"
    )
    print(formatted)

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    test_advanced_stats()
