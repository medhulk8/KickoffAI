"""
Fetch FPL GW data and compute match-level lineup features.

Features per team (home + away):
  xi_strength         — mean season-to-date minutes share of 11 starters (0-1)
  missing_regulars    — count of regular starters (≥2 of last 3) not in today's XI
  starter_continuity  — fraction of last match starters in today's XI

xi_strength_diff      — home_xi_strength - away_xi_strength

Lineup source: FPL GitHub GW CSVs
  - 2022-23 GW1-19: starts inferred from minutes >= 45 (starts col absent)
  - 2022-23 GW20+, 2023-24, 2024-25: starts col present

Output: data/processed/lineup_features.csv
"""

from __future__ import annotations

import io
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "lineup_features.csv"

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
HEADERS  = {"User-Agent": "Mozilla/5.0"}

# FPL season folders to process (in chronological order)
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# FPL team name → our DB team name
TEAM_MAP = {
    "Man City":       "Manchester City",
    "Man Utd":        "Manchester United",
    "Spurs":          "Tottenham Hotspur",
    "Nott'm Forest":  "Nottingham Forest",
    "Brighton":       "Brighton & Hove Albion",
    "Newcastle":      "Newcastle United",
    "West Ham":       "West Ham United",
    "Wolves":         "Wolverhampton",
    "Leicester":      "Leicester City",
    "Leeds":          "Leeds United",
    "Sheffield Utd":  "Sheffield United",
    "Ipswich":        "Ipswich",
    "Luton":          "Luton",
}


def norm_team(name: str) -> str:
    return TEAM_MAP.get(name, name)


def fetch_gw(season: str, gw: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{season}/gws/gw{gw}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def is_starter(row: pd.Series, has_starts_col: bool) -> bool:
    if has_starts_col:
        return int(row.get("starts", 0)) == 1
    return int(row.get("minutes", 0)) >= 45


def build_lineup_features() -> pd.DataFrame:
    # player_id → list of (date, team, fixture, is_home, started, minutes)
    player_history: dict[str, list] = defaultdict(list)
    # team → list of (date, fixture, is_home, set_of_starter_names)
    team_lineup_history: dict[str, list] = defaultdict(list)
    # team → dict player_name → cumulative minutes this season
    team_season_minutes: dict[str, dict] = defaultdict(lambda: defaultdict(float))

    output_rows: list[dict] = []

    for fpl_season in SEASONS:
        print(f"  Processing {fpl_season}...")
        # Reset season minutes at start of each season
        team_season_minutes = defaultdict(lambda: defaultdict(float))

        gw = 1
        while True:
            df = fetch_gw(fpl_season, gw)
            if df is None:
                break

            has_starts = "starts" in df.columns and df["starts"].sum() > 0

            df["team_norm"]   = df["team"].map(norm_team)
            df["minutes"]     = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
            df["goals_scored"] = pd.to_numeric(df.get("goals_scored", 0), errors="coerce").fillna(0)
            df["assists"]     = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0)
            if "starts" in df.columns:
                df["starts"] = pd.to_numeric(df["starts"], errors="coerce").fillna(0)

            if has_starts:
                df["started"] = df["starts"].astype(int) == 1
            else:
                df["started"] = df["minutes"].astype(int) >= 45
            df["date"]    = pd.to_datetime(df["kickoff_time"]).dt.strftime("%Y-%m-%d")

            # Process each fixture
            fixtures = df["fixture"].unique()
            for fid in fixtures:
                fdf = df[df["fixture"] == fid]
                home_df = fdf[fdf["was_home"] == True]
                away_df = fdf[fdf["was_home"] == False]

                if home_df.empty or away_df.empty:
                    continue

                home_team = home_df.iloc[0]["team_norm"]
                away_team = away_df.iloc[0]["team_norm"]
                date      = home_df.iloc[0]["date"]

                def xi_features(team: str, team_df: pd.DataFrame) -> dict:
                    starters = team_df[team_df["started"] == True]["name"].tolist()
                    if len(starters) < 8:
                        return None

                    # XI strength: mean season-to-date minutes share of starters
                    total_team_mins = sum(team_season_minutes[team].values()) or 1.0
                    xi_mins = [team_season_minutes[team].get(p, 0.0) for p in starters]
                    xi_strength = sum(xi_mins) / (total_team_mins * len(starters) / 11)
                    xi_strength = min(xi_strength, 2.0)  # cap outliers

                    # Missing regulars: ≥2 starts in last 3 home/away matches, not starting today
                    hist = team_lineup_history[team]
                    recent = [h["starters"] for h in hist[-3:]]
                    if recent:
                        regulars = set.intersection(*[set(s) for s in recent]) if len(recent) >= 2 else set()
                        # Players who appeared in ≥2 of last 3
                        from collections import Counter
                        counts = Counter(p for lineup in recent for p in lineup)
                        regulars = {p for p, c in counts.items() if c >= 2}
                        missing = len(regulars - set(starters))
                    else:
                        missing = 0

                    # Starter continuity: fraction of last match starters in today's XI
                    if hist:
                        last_starters = hist[-1]["starters"]
                        continuity = len(set(starters) & set(last_starters)) / max(len(last_starters), 1)
                    else:
                        continuity = 1.0

                    return {
                        "xi_strength":       round(xi_strength, 4),
                        "missing_regulars":  missing,
                        "starter_continuity": round(continuity, 4),
                    }

                h_feats = xi_features(home_team, home_df)
                a_feats = xi_features(away_team, away_df)

                if h_feats and a_feats:
                    output_rows.append({
                        "date":                    date,
                        "home_team":               home_team,
                        "away_team":               away_team,
                        "home_xi_strength":        h_feats["xi_strength"],
                        "home_missing_regulars":   h_feats["missing_regulars"],
                        "home_starter_continuity": h_feats["starter_continuity"],
                        "away_xi_strength":        a_feats["xi_strength"],
                        "away_missing_regulars":   a_feats["missing_regulars"],
                        "away_starter_continuity": a_feats["starter_continuity"],
                        "xi_strength_diff":        round(h_feats["xi_strength"] - a_feats["xi_strength"], 4),
                    })

                # Update histories AFTER computing features (leakage-safe)
                for team, team_df in [(home_team, home_df), (away_team, away_df)]:
                    is_home = (team == home_team)
                    starters = team_df[team_df["started"] == True]["name"].tolist()
                    team_lineup_history[team].append({
                        "date": date, "fixture": fid,
                        "is_home": is_home, "starters": starters
                    })
                    # Update season minutes
                    for _, prow in team_df.iterrows():
                        team_season_minutes[team][prow["name"]] += prow["minutes"]

            time.sleep(0.2)
            gw += 1

        print(f"    → {sum(1 for r in output_rows if r['date'] >= f'{fpl_season[:4]}-07-01')} new rows")

    df_out = pd.DataFrame(output_rows).drop_duplicates(subset=["date", "home_team", "away_team"])
    df_out = df_out.sort_values("date").reset_index(drop=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUTPUT_PATH}")
    return df_out


if __name__ == "__main__":
    build_lineup_features()
