"""
Live feature computation for KickoffAI custom match predictions.

Computes rolling team stats from the DB for any team at the current moment
(i.e., using all available match history). Used to feed the O/U ranker
when the user inputs a future fixture not in the DB.

Rolling stats mirror the training methodology in build_ou_dataset.py:
  - Last 5 matches for each team (home + away combined)
  - goals_scored, goals_conceded, sot, sot_conceded per match
  - Shot conversion: (avg_goals_scored + 1) / (avg_sot + 2) — Laplace smoothed
  - elo_diff: computed from full match history using EloCalculator
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.elo import EloCalculator

DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"
LAST_N = 5


def _get_team_rolling(team: str, db_path: str, last_n: int = LAST_N) -> Optional[dict]:
    """
    Get rolling average stats for a team from their last N matches.

    Returns dict with: goals_scored, goals_conceded, sot, sot_conceded, conversion
    Returns None if the team is not found or has fewer than 1 match.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT home_team, away_team,
               home_goals, away_goals,
               home_shots_target, away_shots_target
        FROM matches
        WHERE (home_team = ? OR away_team = ?)
          AND home_goals IS NOT NULL
          AND home_shots_target IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (team, team, last_n),
    ).fetchall()
    conn.close()

    if not rows:
        return None

    gs_list, gc_list, sot_list, sotc_list = [], [], [], []
    for home_team, away_team, hg, ag, hs, as_ in rows:
        if home_team == team:
            gs_list.append(hg)
            gc_list.append(ag)
            sot_list.append(hs)
            sotc_list.append(as_)
        else:
            gs_list.append(ag)
            gc_list.append(hg)
            sot_list.append(as_)
            sotc_list.append(hs)

    avg_gs   = float(np.mean(gs_list))
    avg_gc   = float(np.mean(gc_list))
    avg_sot  = float(np.mean(sot_list))
    avg_sotc = float(np.mean(sotc_list))
    conversion = (avg_gs + 1) / (avg_sot + 2)  # Laplace smoothed

    return {
        "goals_scored":   round(avg_gs, 4),
        "goals_conceded": round(avg_gc, 4),
        "sot":            round(avg_sot, 4),
        "sot_conceded":   round(avg_sotc, 4),
        "conversion":     round(conversion, 4),
        "n_matches":      len(rows),
    }


def _get_elo_diff(home_team: str, away_team: str, db_path: str) -> float:
    """
    Compute current Elo ratings and return home_elo - away_elo.
    Uses all matches in DB. Returns 0.0 if teams not found.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT match_id, date, home_team, away_team, result
        FROM matches
        WHERE home_goals IS NOT NULL
        ORDER BY date ASC
        """
    ).fetchall()
    conn.close()

    matches = [
        {"match_id": r[0], "date": r[1], "home_team": r[2], "away_team": r[3], "result": r[4]}
        for r in rows
    ]
    calc = EloCalculator()
    current = calc.get_current_ratings(matches)

    home_elo = current.get(home_team, 1500)
    away_elo = current.get(away_team, 1500)
    return float(home_elo - away_elo)


def compute_ou_features(
    home_team: str,
    away_team: str,
    bm_over25_prob: float,
    db_path: Optional[str] = None,
) -> dict:
    """
    Compute the full O/U ranker feature dict for a live fixture.

    Args:
        home_team: Home team name (must match DB spelling)
        away_team: Away team name (must match DB spelling)
        bm_over25_prob: Bookmaker implied probability of over 2.5 goals (0-1)
        db_path: Path to asil.db; uses default if None

    Returns:
        Feature dict ready to pass into OURanker.rank(), plus metadata:
        {
          <all 15 ranker features>,
          "_home_n":  int,   # matches found for home team
          "_away_n":  int,   # matches found for away team
          "_missing": bool,  # True if either team not in DB
        }
    """
    path = db_path or str(DB_PATH)

    home_stats = _get_team_rolling(home_team, path)
    away_stats = _get_team_rolling(away_team, path)
    elo_diff   = _get_elo_diff(home_team, away_team, path)

    missing = home_stats is None or away_stats is None

    # Use league-average defaults for unknown teams
    default = {"goals_scored": 1.5, "goals_conceded": 1.5, "sot": 4.0, "sot_conceded": 4.0, "conversion": 0.375}
    h = home_stats or default
    a = away_stats or default

    features = {
        "home_goals_scored":   h["goals_scored"],
        "home_goals_conceded": h["goals_conceded"],
        "away_goals_scored":   a["goals_scored"],
        "away_goals_conceded": a["goals_conceded"],
        "home_sot":            h["sot"],
        "away_sot":            a["sot"],
        "home_sot_conceded":   h["sot_conceded"],
        "away_sot_conceded":   a["sot_conceded"],
        "home_conversion":     h["conversion"],
        "away_conversion":     a["conversion"],
        "total_attack":        round(h["goals_scored"] + a["goals_scored"], 4),
        "total_defence_loose": round(h["goals_conceded"] + a["goals_conceded"], 4),
        "total_sot":           round(h["sot"] + a["sot"], 4),
        "elo_diff":            round(elo_diff, 2),
        "bm_over25_prob":      round(bm_over25_prob, 4),
        # Metadata (not passed to model)
        "_home_n":  home_stats["n_matches"] if home_stats else 0,
        "_away_n":  away_stats["n_matches"] if away_stats else 0,
        "_missing": missing,
    }
    return features
