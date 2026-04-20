"""
Fetch the latest Premier League results from football-data.co.uk and insert
any new matches into asil.db.

Run this weekly (or before predicting) to keep the DB current.
Typical lag: ~1 week behind real-time for shots-on-target data.

Usage:
    python src/data/update_results.py
"""

from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"

CURRENT_SEASON = "2526"
FD_URL = f"https://www.football-data.co.uk/mmz4281/{CURRENT_SEASON}/E0.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TEAM_MAP = {
    "Man United":     "Manchester United",
    "Man City":       "Manchester City",
    "Nott'm Forest":  "Nottingham Forest",
    "Nottingham":     "Nottingham Forest",
    "Newcastle":      "Newcastle United",
    "Tottenham":      "Tottenham Hotspur",
    "Spurs":          "Tottenham Hotspur",
    "West Ham":       "West Ham United",
    "Wolves":         "Wolverhampton",
    "Brighton":       "Brighton & Hove Albion",
    "Leeds":          "Leeds United",
    "Leicester":      "Leicester City",
    "Norwich":        "Norwich City",
    "Sheffield Utd":  "Sheffield United",
    "Ipswich":        "Ipswich",
    "Luton":          "Luton",
}


def _norm(name: str) -> str:
    return TEAM_MAP.get(name.strip(), name.strip())


def _odds_to_prob(h, d, a):
    try:
        raw = [1/float(h), 1/float(d), 1/float(a)]
        total = sum(raw)
        return [round(v/total, 4) for v in raw]
    except (TypeError, ValueError, ZeroDivisionError):
        return [None, None, None]


def fetch_csv() -> pd.DataFrame:
    resp = requests.get(FD_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["season"]     = CURRENT_SEASON
    out["home_team"]  = df["HomeTeam"].map(_norm)
    out["away_team"]  = df["AwayTeam"].map(_norm)
    out["home_goals"] = df["FTHG"].astype(int)
    out["away_goals"] = df["FTAG"].astype(int)
    out["result"]     = df["FTR"]
    out["date"]       = pd.to_datetime(df["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    out["home_shots"] = pd.to_numeric(df.get("HS"), errors="coerce")
    out["away_shots"] = pd.to_numeric(df.get("AS"), errors="coerce")
    out["home_corners"] = pd.to_numeric(df.get("HC"), errors="coerce")
    out["away_corners"] = pd.to_numeric(df.get("AC"), errors="coerce")
    out["home_shots_target"] = pd.to_numeric(df.get("HST"), errors="coerce")
    out["away_shots_target"] = pd.to_numeric(df.get("AST"), errors="coerce")

    probs = df.apply(lambda r: _odds_to_prob(r.get("B365H"), r.get("B365D"), r.get("B365A")), axis=1)
    out["b365_home_prob"] = probs.apply(lambda x: x[0])
    out["b365_draw_prob"] = probs.apply(lambda x: x[1])
    out["b365_away_prob"] = probs.apply(lambda x: x[2])

    return out.reset_index(drop=True)


def update(dry_run: bool = False) -> int:
    print(f"Fetching {FD_URL} ...")
    raw = fetch_csv()
    print(f"  {len(raw)} completed matches in remote CSV")

    processed = process(raw)

    conn = sqlite3.connect(str(DB_PATH))
    existing = pd.read_sql(
        f"SELECT date, home_team, away_team FROM matches WHERE season='{CURRENT_SEASON}'",
        conn,
    )
    existing_keys = set(zip(existing["date"], existing["home_team"], existing["away_team"]))

    new_rows = processed[
        ~processed.apply(lambda r: (r["date"], r["home_team"], r["away_team"]) in existing_keys, axis=1)
    ]
    print(f"  {len(new_rows)} new matches to insert")

    if dry_run:
        print("  (dry-run — nothing written)")
        conn.close()
        return len(new_rows)

    inserted = 0
    for _, row in new_rows.iterrows():
        try:
            conn.execute("""
                INSERT INTO matches (
                    date, season, home_team, away_team, home_goals, away_goals, result,
                    home_shots, away_shots, home_corners, away_corners,
                    home_shots_target, away_shots_target,
                    b365_home_prob, b365_draw_prob, b365_away_prob
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row["date"], row["season"], row["home_team"], row["away_team"],
                int(row["home_goals"]), int(row["away_goals"]), row["result"],
                _int_or_none(row["home_shots"]), _int_or_none(row["away_shots"]),
                _int_or_none(row["home_corners"]), _int_or_none(row["away_corners"]),
                _int_or_none(row["home_shots_target"]), _int_or_none(row["away_shots_target"]),
                row["b365_home_prob"], row["b365_draw_prob"], row["b365_away_prob"],
            ))
            inserted += 1
        except Exception as e:
            print(f"  WARN: failed to insert {row['home_team']} vs {row['away_team']}: {e}")

    conn.commit()
    conn.close()
    print(f"  Inserted {inserted} new matches into asil.db")
    return inserted


def _int_or_none(val):
    try:
        return int(val) if pd.notna(val) else None
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted without writing")
    args = parser.parse_args()
    update(dry_run=args.dry_run)
