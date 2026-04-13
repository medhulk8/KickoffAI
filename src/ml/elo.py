"""
Rolling Elo ratings for KickoffAI.

Computes pre-match Elo for every match using all available match history.
Early seasons act as warm-up — ratings are never used for training, only
the pre-match snapshot at the time of each match.

Parameters (fixed for V1 — do not tune until after honest eval):
    K = 20
    home_advantage = 100  (added to home team's effective rating)
    initial_elo = 1500    (all teams, including newly promoted sides)
"""

from __future__ import annotations


class EloCalculator:
    """
    Compute rolling Elo ratings from a chronological list of matches.

    Usage:
        calc = EloCalculator()
        elo_map = calc.compute(matches)  # matches: list of dicts
        elo_diff = elo_map[match_id]["elo_diff"]
    """

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 100.0,
        initial_elo: float = 1500.0,
    ):
        self.k = k
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo

    def compute(self, matches: list[dict]) -> dict[int, dict]:
        """
        Process all matches in chronological order.

        Args:
            matches: list of dicts, each with:
                match_id (int), date (str), home_team, away_team, result ('H'/'D'/'A' or None)
                Must be sorted by date ascending before calling.

        Returns:
            dict: match_id -> {home_elo, away_elo, elo_diff}
                  All values are pre-match (recorded before the result is applied).
        """
        ratings: dict[str, float] = {}
        elo_map: dict[int, dict] = {}

        for match in matches:
            home = match["home_team"]
            away = match["away_team"]
            mid  = match["match_id"]

            r_home = ratings.get(home, self.initial_elo)
            r_away = ratings.get(away, self.initial_elo)

            # Record PRE-match snapshot
            elo_map[mid] = {
                "home_elo": round(r_home, 2),
                "away_elo": round(r_away, 2),
                "elo_diff": round(r_home - r_away, 2),
            }

            # Update only if result is known
            result = match.get("result")
            if result not in ("H", "D", "A"):
                continue

            # Expected score for home team (home advantage baked in)
            e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - self.home_advantage) / 400.0))
            e_away = 1.0 - e_home

            if result == "H":
                s_home, s_away = 1.0, 0.0
            elif result == "D":
                s_home, s_away = 0.5, 0.5
            else:  # A
                s_home, s_away = 0.0, 1.0

            ratings[home] = r_home + self.k * (s_home - e_home)
            ratings[away] = r_away + self.k * (s_away - e_away)

        return elo_map

    def get_current_ratings(self, matches: list[dict]) -> dict[str, float]:
        """Return final ratings after processing all matches. Useful for inspection."""
        ratings: dict[str, float] = {}
        for match in matches:
            home = match["home_team"]
            away = match["away_team"]
            r_home = ratings.get(home, self.initial_elo)
            r_away = ratings.get(away, self.initial_elo)
            result = match.get("result")
            if result not in ("H", "D", "A"):
                continue
            e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - self.home_advantage) / 400.0))
            e_away = 1.0 - e_home
            if result == "H":
                s_home, s_away = 1.0, 0.0
            elif result == "D":
                s_home, s_away = 0.5, 0.5
            else:
                s_home, s_away = 0.0, 1.0
            ratings[home] = r_home + self.k * (s_home - e_home)
            ratings[away] = r_away + self.k * (s_away - e_away)
        return {team: round(r, 2) for team, r in sorted(ratings.items(), key=lambda x: -x[1])}


if __name__ == "__main__":
    import sqlite3
    from pathlib import Path

    DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "asil.db"
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT match_id, date, home_team, away_team, result FROM matches ORDER BY date ASC"
    ).fetchall()
    conn.close()

    matches = [dict(r) for r in rows]
    calc = EloCalculator()
    elo_map = calc.compute(matches)

    print(f"Computed Elo for {len(elo_map)} matches")
    print(f"\nTop 10 teams by final rating:")
    final = calc.get_current_ratings(matches)
    for i, (team, elo) in enumerate(list(final.items())[:10], 1):
        print(f"  {i:2}. {team:<25} {elo:.0f}")

    # Sample: elo_diff distribution
    import statistics
    diffs = [v["elo_diff"] for v in elo_map.values()]
    print(f"\nelo_diff  mean={statistics.mean(diffs):.1f}  "
          f"stdev={statistics.stdev(diffs):.1f}  "
          f"min={min(diffs):.1f}  max={max(diffs):.1f}")
