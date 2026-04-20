"""
Fetch match-level xG from FPL (Fantasy Premier League) GitHub data.

FPL xG data is available from 2022-23 onwards (seasons 2223, 2324, 2425).
Aggregates player-level expected_goals / expected_goals_conceded per fixture.

Source: https://github.com/vaastav/Fantasy-Premier-League

Output: data/processed/xg_data.csv
  season, date, home_team, away_team, home_xg, away_xg
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "xg_data.csv"

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# FPL season folder → our season label
SEASONS = {
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
}

# Max GW per season
MAX_GW = 38

HEADERS = {"User-Agent": "Mozilla/5.0"}

# FPL short team name → our DB team name
TEAM_NAME_MAP = {
    "Arsenal":        "Arsenal",
    "Aston Villa":    "Aston Villa",
    "Bournemouth":    "Bournemouth",
    "Brentford":      "Brentford",
    "Brighton":       "Brighton & Hove Albion",
    "Burnley":        "Burnley",
    "Chelsea":        "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton":        "Everton",
    "Fulham":         "Fulham",
    "Ipswich":        "Ipswich",
    "Leeds":          "Leeds United",
    "Leicester":      "Leicester City",
    "Liverpool":      "Liverpool",
    "Luton":          "Luton",
    "Man City":       "Manchester City",
    "Man Utd":        "Manchester United",
    "Newcastle":      "Newcastle United",
    "Nott'm Forest":  "Nottingham Forest",
    "Sheffield Utd":  "Sheffield United",
    "Southampton":    "Southampton",
    "Spurs":          "Tottenham Hotspur",
    "West Ham":       "West Ham United",
    "Wolves":         "Wolverhampton",
}


def normalise(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def fetch_gw(fpl_season: str, gw: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{fpl_season}/gws/gw{gw}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if "expected_goals" not in df.columns:
        return None
    return df


def fetch_season_xg(fpl_season: str, season_label: str) -> list[dict]:
    print(f"  Season {season_label} (FPL folder: {fpl_season})")
    all_rows = []

    for gw in range(1, MAX_GW + 1):
        df = fetch_gw(fpl_season, gw)
        if df is None:
            break

        df["expected_goals"]          = pd.to_numeric(df["expected_goals"],          errors="coerce").fillna(0.0)
        df["expected_goals_conceded"] = pd.to_numeric(df["expected_goals_conceded"], errors="coerce").fillna(0.0)

        # Aggregate per fixture per team (was_home determines home/away)
        agg = (
            df.groupby(["fixture", "team", "was_home", "kickoff_time"])
            [["expected_goals", "expected_goals_conceded"]]
            .sum()
            .reset_index()
        )

        # For each fixture, pivot home and away
        fixtures = agg["fixture"].unique()
        for fid in fixtures:
            fdf = agg[agg["fixture"] == fid]
            home_rows = fdf[fdf["was_home"] == True]
            away_rows = fdf[fdf["was_home"] == False]

            if home_rows.empty or away_rows.empty:
                continue

            home_team = normalise(home_rows.iloc[0]["team"])
            away_team = normalise(away_rows.iloc[0]["team"])
            date_str  = str(home_rows.iloc[0]["kickoff_time"])[:10]

            home_xg = round(float(home_rows["expected_goals"].sum()), 4)
            away_xg = round(float(away_rows["expected_goals"].sum()), 4)

            # Skip if both are zero — FPL didn't track xG in early 2022-23 GWs
            if home_xg == 0.0 and away_xg == 0.0:
                continue

            all_rows.append({
                "season":    season_label,
                "date":      date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_xg":   home_xg,
                "away_xg":   away_xg,
                "fixture_id": fid,
            })

        time.sleep(0.3)

    # Deduplicate (same fixture may appear across GW CSVs if replayed)
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["season"], r["fixture_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"    → {len(deduped)} matches")
    return deduped


def fetch_all() -> None:
    all_rows = []
    for fpl_season, season_label in SEASONS.items():
        rows = fetch_season_xg(fpl_season, season_label)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop(columns=["fixture_id"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Season counts:\n{df.groupby('season').size()}")
    print(f"\nSample:\n{df.head(3).to_string()}")


if __name__ == "__main__":
    fetch_all()
