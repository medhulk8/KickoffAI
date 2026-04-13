"""
Build Over/Under 2.5 goals training dataset.

One row per match. All features computed point-in-time (strictly before match date).
Uses vectorized pandas rolling rather than per-match DB queries.

Label: over25 = 1 if (home_goals + away_goals) > 2.5 else 0

Output columns:
    match_id, date, season, home_team, away_team,
    # Rolling attack/defence (last 5 matches, each team)
    home_goals_scored, home_goals_conceded,
    away_goals_scored, away_goals_conceded,
    # Rolling shots on target (last 5)
    home_sot, away_sot,
    home_sot_conceded, away_sot_conceded,
    # Shot conversion (goals / sot, Laplace smoothed)
    home_conversion, away_conversion,
    # Combined environment proxies
    total_attack,        # home_goals_scored + away_goals_scored
    total_defence_loose, # home_goals_conceded + away_goals_conceded
    total_sot,           # home_sot + away_sot
    # Elo
    elo_diff,
    # Label
    over25
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.elo import EloCalculator

DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ou_dataset.csv"

LAST_N = 5  # rolling window size


def load_matches() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT match_id, date, season, home_team, away_team,
               home_goals, away_goals, result,
               home_shots_target, away_shots_target,
               avg_draw_prob,
               avg_over25_odds, avg_under25_odds
        FROM matches
        WHERE home_goals IS NOT NULL
          AND home_shots_target IS NOT NULL
        ORDER BY date ASC
    """, conn)
    conn.close()
    return df


def compute_team_rolling(matches_df: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
    """
    For each match, compute rolling pre-match stats for home and away teams.

    Strategy:
      1. Expand each match into two team-perspective rows.
      2. Sort by date, shift(1) before rolling to prevent leakage.
      3. Pivot back to match-level.
    """
    records = []
    for _, row in matches_df.iterrows():
        base = dict(match_id=row["match_id"], date=row["date"])
        records.append({**base,
            "team": row["home_team"], "role": "home",
            "goals_scored":    row["home_goals"],
            "goals_conceded":  row["away_goals"],
            "sot":             row["home_shots_target"],
            "sot_conceded":    row["away_shots_target"],
        })
        records.append({**base,
            "team": row["away_team"], "role": "away",
            "goals_scored":    row["away_goals"],
            "goals_conceded":  row["home_goals"],
            "sot":             row["away_shots_target"],
            "sot_conceded":    row["home_shots_target"],
        })

    tm = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)

    stat_cols = ["goals_scored", "goals_conceded", "sot", "sot_conceded"]

    rolled = {}
    for team, grp in tm.groupby("team"):
        grp = grp.sort_values("date").copy()
        for col in stat_cols:
            # shift(1) = exclude current match; rolling(last_n) = last N prior matches
            grp[f"roll_{col}"] = (
                grp[col].shift(1).rolling(last_n, min_periods=1).mean()
            )
        rolled[team] = grp

    tm_rolled = pd.concat(rolled.values(), ignore_index=True)

    # Pivot back to match-level
    home = tm_rolled[tm_rolled["role"] == "home"].set_index("match_id")
    away = tm_rolled[tm_rolled["role"] == "away"].set_index("match_id")

    result = matches_df[["match_id"]].copy().set_index("match_id")
    for col in stat_cols:
        result[f"home_{col}"] = home[f"roll_{col}"]
        result[f"away_{col}"] = away[f"roll_{col}"]

    return result.reset_index()


def build_ou_dataset(output_path: str = str(OUTPUT_PATH)) -> pd.DataFrame:
    print("Loading matches from DB...")
    matches = load_matches()
    print(f"  {len(matches)} matches with goals + shots on target")

    # Rolling team stats
    print("Computing rolling team stats...")
    rolling = compute_team_rolling(matches, last_n=LAST_N)

    # Elo
    print("Computing Elo ratings...")
    elo_matches = matches[["match_id", "date", "home_team", "away_team", "result"]].to_dict("records")
    elo_map = EloCalculator().compute(elo_matches)

    # Merge
    df = matches.merge(rolling, on="match_id")

    # Drop rows where rolling stats are NaN (very first match for a team)
    stat_roll_cols = [c for c in df.columns if c.startswith("home_") or c.startswith("away_")]
    df = df.dropna(subset=stat_roll_cols).copy()

    # Shot conversion (Laplace smoothed: goals+1 / sot+2 to avoid div/zero and extremes)
    df["home_conversion"] = (df["home_goals_scored"] + 1) / (df["home_sot"] + 2)
    df["away_conversion"] = (df["away_goals_scored"] + 1) / (df["away_sot"] + 2)

    # Combined environment features
    df["total_attack"]        = df["home_goals_scored"] + df["away_goals_scored"]
    df["total_defence_loose"] = df["home_goals_conceded"] + df["away_goals_conceded"]
    df["total_sot"]           = df["home_sot"] + df["away_sot"]

    # Elo diff
    df["elo_diff"] = df["match_id"].map(lambda mid: elo_map.get(mid, {}).get("elo_diff", 0.0))

    # Bookmaker O/U implied probability (normalized)
    def ou_prob(row):
        o, u = row["avg_over25_odds"], row["avg_under25_odds"]
        if pd.isna(o) or pd.isna(u) or o <= 1 or u <= 1:
            return None
        raw_over = 1.0 / o
        raw_under = 1.0 / u
        return round(raw_over / (raw_over + raw_under), 4)

    df["bm_over25_prob"] = df.apply(ou_prob, axis=1)

    # Label
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["over25"] = (df["total_goals"] > 2.5).astype(int)

    # Final column selection
    feature_cols = [
        "home_goals_scored", "home_goals_conceded",
        "away_goals_scored", "away_goals_conceded",
        "home_sot", "away_sot",
        "home_sot_conceded", "away_sot_conceded",
        "home_conversion", "away_conversion",
        "total_attack", "total_defence_loose", "total_sot",
        "elo_diff",
        "avg_draw_prob",
        "bm_over25_prob",
    ]
    out_cols = ["match_id", "date", "season", "home_team", "away_team",
                *feature_cols, "over25"]
    df = df[out_cols].copy()

    print(f"\nDataset: {len(df)} rows, {df['over25'].mean():.1%} over 2.5")
    print(f"Season distribution:\n{df.groupby('season')['over25'].agg(['count','mean']).round(3)}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return df


if __name__ == "__main__":
    build_ou_dataset()
