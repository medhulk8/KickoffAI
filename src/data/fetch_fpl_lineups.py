"""
Fetch FPL GW data and compute match-level lineup features.

Features per team (home + away):
  xi_strength         — mean season-to-date minutes share of starting XI (0-1)
  xi_strength_l10     — same but using last-10-match rolling minutes (cold-start fix)
  missing_regulars    — count of regular starters (≥2 of last 3) not in today's XI
  starter_continuity  — fraction of last match starters in today's XI
  star_absent         — count of players who started ALL of last 5 matches but not today

Derived:
  xi_strength_diff     — home_xi_strength - away_xi_strength (season-to-date)
  xi_strength_diff_l10 — home_xi_strength_l10 - away_xi_strength_l10 (rolling window)

Output: data/processed/lineup_features.csv
"""

from __future__ import annotations

import io
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "lineup_features.csv"

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
HEADERS  = {"User-Agent": "Mozilla/5.0"}

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

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

ROLLING_WINDOW = 10   # matches for xi_strength_l10
STAR_WINDOW    = 5    # matches for star_absent (must start ALL of these)


def norm_team(name: str) -> str:
    return TEAM_MAP.get(name, name)


def fetch_gw(season: str, gw: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{season}/gws/gw{gw}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def build_lineup_features() -> pd.DataFrame:
    # team → list of {date, fixture, is_home, starters}
    team_lineup_history: dict[str, list] = defaultdict(list)
    # team → dict player → cumulative minutes this season (season-to-date xi_strength)
    team_season_minutes: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    # team → deque of dicts {player: minutes} for last ROLLING_WINDOW matches
    team_rolling_matches: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

    output_rows: list[dict] = []

    for fpl_season in SEASONS:
        print(f"  Processing {fpl_season}...")
        team_season_minutes = defaultdict(lambda: defaultdict(float))

        gw = 1
        while True:
            df = fetch_gw(fpl_season, gw)
            if df is None:
                break

            has_starts = "starts" in df.columns and df["starts"].sum() > 0

            df["team_norm"] = df["team"].map(norm_team)
            df["minutes"]   = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
            if "starts" in df.columns:
                df["starts"] = pd.to_numeric(df["starts"], errors="coerce").fillna(0)

            if has_starts:
                df["started"] = df["starts"].astype(int) == 1
            else:
                df["started"] = df["minutes"].astype(int) >= 45
            df["date"] = pd.to_datetime(df["kickoff_time"]).dt.strftime("%Y-%m-%d")

            fixtures = df["fixture"].unique()
            for fid in fixtures:
                fdf      = df[df["fixture"] == fid]
                home_df  = fdf[fdf["was_home"] == True]
                away_df  = fdf[fdf["was_home"] == False]

                if home_df.empty or away_df.empty:
                    continue

                home_team = home_df.iloc[0]["team_norm"]
                away_team = away_df.iloc[0]["team_norm"]
                date      = home_df.iloc[0]["date"]

                def xi_features(team: str, team_df: pd.DataFrame) -> dict | None:
                    starters = team_df[team_df["started"] == True]["name"].tolist()
                    if len(starters) < 8:
                        return None

                    # ── Season-to-date xi_strength (original) ──────────────────
                    total_season_mins = sum(team_season_minutes[team].values()) or 1.0
                    xi_mins_s2d = [team_season_minutes[team].get(p, 0.0) for p in starters]
                    xi_strength = sum(xi_mins_s2d) / (total_season_mins * len(starters) / 11)
                    xi_strength = min(xi_strength, 2.0)

                    # ── Rolling-10 xi_strength (cold-start fix) ────────────────
                    rolling = team_rolling_matches[team]
                    if rolling:
                        rolling_mins: dict[str, float] = defaultdict(float)
                        for match_mins in rolling:
                            for p, m in match_mins.items():
                                rolling_mins[p] += m
                        total_rolling = sum(rolling_mins.values()) or 1.0
                        xi_mins_l10 = [rolling_mins.get(p, 0.0) for p in starters]
                        xi_strength_l10 = sum(xi_mins_l10) / (total_rolling * len(starters) / 11)
                        xi_strength_l10 = min(xi_strength_l10, 2.0)
                    else:
                        xi_strength_l10 = xi_strength  # no history yet → fall back to s2d

                    # ── Star absent (iron regulars not starting today) ──────────
                    hist = team_lineup_history[team]
                    n_window = min(STAR_WINDOW, len(hist))
                    if n_window >= 2:
                        recent_sets = [set(h["starters"]) for h in hist[-n_window:]]
                        iron_regulars = set.intersection(*recent_sets)
                        star_absent = len(iron_regulars - set(starters))
                    else:
                        star_absent = 0

                    # ── Missing regulars (≥2 of last 3) ───────────────────────
                    recent3 = [h["starters"] for h in hist[-3:]]
                    if recent3:
                        counts   = Counter(p for lineup in recent3 for p in lineup)
                        regulars = {p for p, c in counts.items() if c >= 2}
                        missing  = len(regulars - set(starters))
                    else:
                        missing = 0

                    # ── Starter continuity ─────────────────────────────────────
                    if hist:
                        last_s     = hist[-1]["starters"]
                        continuity = len(set(starters) & set(last_s)) / max(len(last_s), 1)
                    else:
                        continuity = 1.0

                    return {
                        "xi_strength":       round(xi_strength, 4),
                        "xi_strength_l10":   round(xi_strength_l10, 4),
                        "missing_regulars":  missing,
                        "star_absent":       star_absent,
                        "starter_continuity": round(continuity, 4),
                    }

                h_feats = xi_features(home_team, home_df)
                a_feats = xi_features(away_team, away_df)

                if h_feats and a_feats:
                    output_rows.append({
                        "date":                      date,
                        "home_team":                 home_team,
                        "away_team":                 away_team,
                        "home_xi_strength":          h_feats["xi_strength"],
                        "home_xi_strength_l10":      h_feats["xi_strength_l10"],
                        "home_missing_regulars":     h_feats["missing_regulars"],
                        "home_star_absent":          h_feats["star_absent"],
                        "home_starter_continuity":   h_feats["starter_continuity"],
                        "away_xi_strength":          a_feats["xi_strength"],
                        "away_xi_strength_l10":      a_feats["xi_strength_l10"],
                        "away_missing_regulars":     a_feats["missing_regulars"],
                        "away_star_absent":          a_feats["star_absent"],
                        "away_starter_continuity":   a_feats["starter_continuity"],
                        "xi_strength_diff":          round(h_feats["xi_strength"]     - a_feats["xi_strength"],     4),
                        "xi_strength_diff_l10":      round(h_feats["xi_strength_l10"] - a_feats["xi_strength_l10"], 4),
                    })

                # Update histories AFTER computing features (leakage-safe)
                for team, team_df_u in [(home_team, home_df), (away_team, away_df)]:
                    is_home = (team == home_team)
                    starters_u = team_df_u[team_df_u["started"] == True]["name"].tolist()
                    team_lineup_history[team].append({
                        "date": date, "fixture": fid,
                        "is_home": is_home, "starters": starters_u,
                    })
                    match_player_mins = {}
                    for _, prow in team_df_u.iterrows():
                        m = float(prow["minutes"])
                        team_season_minutes[team][prow["name"]] += m
                        match_player_mins[prow["name"]] = m
                    team_rolling_matches[team].append(match_player_mins)

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
