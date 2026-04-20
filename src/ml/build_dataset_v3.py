"""
Build H/D/A training dataset V3 — independent model, no bookmaker odds.

Features (~18 total):
  Per team (home + away, x2):
    - sot_l5            rolling avg SOT last 5 (venue-specific if ≥3 venue matches, else overall)
    - sot_conceded_l5   rolling avg SOT conceded last 5 (venue fallback)
    - conversion        Laplace-smoothed shot conversion (goals+1)/(sot+2), overall last 5
    - clean_sheet_l5    clean sheet rate last 5 (venue fallback)
    - pts_momentum      points last 3 minus last 5 per match (overall — momentum signal)
    - goals_momentum    goals scored last 3 minus last 5 per match (overall)
    - days_rest         days since team's previous match
  Shared:
    - elo_diff          home_elo - away_elo
    - home_rank         home team's league rank BEFORE this match
    - away_rank
    - matchweek         avg matches played by both teams this season

Fallback rule (per-stat, per-team, per-match):
  If team has played fewer than 3 home (or away) matches this season so far,
  use overall last-5 instead of venue-specific last-5.

NO bookmaker odds anywhere.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.elo import EloCalculator

DB_PATH    = PROJECT_ROOT / "data" / "processed" / "asil.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"

VENUE_FALLBACK_MIN = 3   # min venue matches needed to use venue-specific rolling
LAST_N = 5
LAST_N_SHORT = 3         # for momentum


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────────────────────────────────────

def load_matches() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT match_id, date, season, home_team, away_team,
               home_goals, away_goals, result,
               home_shots_target, away_shots_target
        FROM matches
        WHERE home_goals IS NOT NULL
          AND home_shots_target IS NOT NULL
        ORDER BY date ASC
    """, conn)
    conn.close()
    df["season"] = df["season"].astype(str)
    df["date"]   = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling features with venue fallback
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(series: pd.Series, n: int) -> pd.Series:
    """Shift-then-roll: strictly pre-match average."""
    return series.shift(1).rolling(n, min_periods=1).mean()


def compute_rolling_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by match_id with columns:
      home_sot_l5, home_sot_conceded_l5, home_conversion,
      home_clean_sheet_l5, home_pts_momentum, home_goals_momentum, home_days_rest
      (and same for away_*)
    """
    # Expand each match into two team-perspective rows
    records = []
    for _, m in matches_df.iterrows():
        base = dict(match_id=m["match_id"], date=m["date"], season=m["season"])
        records.append({**base,
            "team": m["home_team"], "is_home": True,
            "goals_scored":  m["home_goals"],   "goals_conceded": m["away_goals"],
            "sot":           m["home_shots_target"], "sot_conceded":  m["away_shots_target"],
            "clean_sheet":   int(m["away_goals"] == 0),
            "points":        3 if m["result"] == "H" else (1 if m["result"] == "D" else 0),
        })
        records.append({**base,
            "team": m["away_team"], "is_home": False,
            "goals_scored":  m["away_goals"],   "goals_conceded": m["home_goals"],
            "sot":           m["away_shots_target"], "sot_conceded":  m["home_shots_target"],
            "clean_sheet":   int(m["home_goals"] == 0),
            "points":        3 if m["result"] == "A" else (1 if m["result"] == "D" else 0),
        })

    tm = pd.DataFrame(records)

    out_rows = {}   # match_id -> {home_*: ..., away_*: ...}
    for mid in matches_df["match_id"]:
        out_rows[mid] = {}

    for team, grp in tm.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # ── Overall rolling (all matches) ──────────────────────────────────
        for col in ["sot", "sot_conceded", "goals_scored", "clean_sheet", "points"]:
            grp[f"ovr_{col}_l5"] = _rolling_mean(grp[col], LAST_N)
            grp[f"ovr_{col}_l3"] = _rolling_mean(grp[col], LAST_N_SHORT)

        # ── Venue-specific rolling ─────────────────────────────────────────
        for is_home, label in [(True, "home"), (False, "away")]:
            venue_mask = grp["is_home"] == is_home
            vg = grp[venue_mask].copy()

            # Season venue count (how many venue matches played BEFORE current)
            vg["season_venue_count"] = vg.groupby("season").cumcount()  # 0-indexed

            for col in ["sot", "sot_conceded", "goals_scored", "clean_sheet"]:
                vg[f"ven_{col}_l5"] = _rolling_mean(vg[col], LAST_N)

            grp = grp.merge(
                vg[["match_id", "season_venue_count"] +
                   [f"ven_{col}_l5" for col in ["sot", "sot_conceded", "goals_scored", "clean_sheet"]]],
                on="match_id", how="left",
            )
            grp.rename(columns={"season_venue_count": f"{label}_season_count"}, inplace=True)
            for col in ["sot", "sot_conceded", "goals_scored", "clean_sheet"]:
                grp.rename(columns={f"ven_{col}_l5": f"{label}_{col}_l5"}, inplace=True)

        # ── Days rest ──────────────────────────────────────────────────────
        grp["days_rest"] = grp["date"].diff().dt.days.fillna(30).clip(upper=30)

        # ── Write back to match-level output dict ─────────────────────────
        for _, row in grp.iterrows():
            mid = row["match_id"]
            role = "home" if row["is_home"] else "away"

            # Determine which venue label to use for fallback
            venue_label = "home" if row["is_home"] else "away"
            use_venue = row.get(f"{venue_label}_season_count", 0) >= VENUE_FALLBACK_MIN

            for col in ["sot", "sot_conceded", "clean_sheet"]:
                val = (row[f"{venue_label}_{col}_l5"]
                       if use_venue and not pd.isna(row.get(f"{venue_label}_{col}_l5"))
                       else row[f"ovr_{col}_l5"])
                out_rows[mid][f"{role}_{col}_l5"] = round(float(val), 4)

            # Goals scored: needed for conversion, also for momentum — use overall
            gs_l5 = row["ovr_goals_scored_l5"]
            gs_l3 = row["ovr_goals_scored_l3"]
            sot_l5_val = out_rows[mid][f"{role}_sot_l5"]

            # Shot conversion: Laplace smoothed
            out_rows[mid][f"{role}_conversion"] = round(
                (gs_l5 + 1) / (sot_l5_val + 2), 4
            )

            # Momentum: l3 - l5 per match (positive = improving)
            pts_l5 = row["ovr_points_l5"]
            pts_l3 = row["ovr_points_l3"]
            sot_l3 = row["ovr_sot_l3"]
            out_rows[mid][f"{role}_pts_momentum"]   = round(float(pts_l3 - pts_l5), 4)
            out_rows[mid][f"{role}_goals_momentum"] = round(float(gs_l3  - gs_l5),  4)
            out_rows[mid][f"{role}_sot_momentum"]   = round(float(sot_l3 - sot_l5_val), 4)

            # Days rest
            out_rows[mid][f"{role}_days_rest"] = int(row["days_rest"])

    return pd.DataFrame.from_dict(out_rows, orient="index").rename_axis("match_id").reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 3. League positions (leakage-safe loop)
# ─────────────────────────────────────────────────────────────────────────────

def compute_league_positions(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match returns: home_pts, home_gd, home_rank, home_played,
                            away_pts, away_gd, away_rank, away_played, matchweek.
    Snapshot is taken BEFORE the match is processed.
    """
    results = {}

    for season, season_df in matches_df.groupby("season"):
        season_df = season_df.sort_values("date")
        table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0})

        for date, day_df in season_df.groupby("date"):
            # Snapshot BEFORE today's matches
            if table:
                ranked = sorted(
                    table.items(),
                    key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]),
                )
                rank_map = {team: i + 1 for i, (team, _) in enumerate(ranked)}
            else:
                rank_map = {}

            for _, m in day_df.iterrows():
                h, a = m["home_team"], m["away_team"]
                mw = round((table[h]["played"] + table[a]["played"]) / 2) + 1
                results[m["match_id"]] = {
                    "home_pts":    table[h]["pts"],
                    "home_gd":     table[h]["gd"],
                    "home_rank":   rank_map.get(h, 20),
                    "home_played": table[h]["played"],
                    "away_pts":    table[a]["pts"],
                    "away_gd":     table[a]["gd"],
                    "away_rank":   rank_map.get(a, 20),
                    "away_played": table[a]["played"],
                    "matchweek":   mw,
                }

            # Update table AFTER all matches on this date
            for _, m in day_df.iterrows():
                h, a = m["home_team"], m["away_team"]
                hg, ag = int(m["home_goals"]), int(m["away_goals"])
                h_pts = 3 if m["result"] == "H" else (1 if m["result"] == "D" else 0)
                a_pts = 3 if m["result"] == "A" else (1 if m["result"] == "D" else 0)
                for team, pts, gf, ga in [(h, h_pts, hg, ag), (a, a_pts, ag, hg)]:
                    table[team]["pts"]    += pts
                    table[team]["gf"]     += gf
                    table[team]["ga"]     += ga
                    table[team]["gd"]     += gf - ga
                    table[team]["played"] += 1

    return pd.DataFrame.from_dict(results, orient="index").rename_axis("match_id").reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Elo
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo(matches_df: pd.DataFrame) -> pd.DataFrame:
    records = matches_df[["match_id", "date", "home_team", "away_team", "result"]].to_dict("records")
    elo_map = EloCalculator().compute(records)
    rows = []
    for mid, vals in elo_map.items():
        rows.append({"match_id": mid, "elo_diff": vals["elo_diff"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Assemble
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "home_sot_l5", "home_sot_conceded_l5", "home_conversion",
    "home_clean_sheet_l5", "home_pts_momentum", "home_goals_momentum",
    "home_sot_momentum", "home_days_rest",
    "away_sot_l5", "away_sot_conceded_l5", "away_conversion",
    "away_clean_sheet_l5", "away_pts_momentum", "away_goals_momentum",
    "away_sot_momentum", "away_days_rest",
    "elo_diff",
    # Reparameterized: diff (direction) + mean (match quality floor)
    "ppg_diff", "ppg_mean",
    "gd_pg_diff", "gd_pg_mean",
    "rank_diff", "rank_mean",
    "matchweek",
]


def build_dataset_v3(output_path: str = str(OUTPUT_PATH)) -> pd.DataFrame:
    print("Loading matches...")
    matches = load_matches()
    print(f"  {len(matches)} matches across seasons: {sorted(matches['season'].unique())}")

    print("Computing rolling features...")
    rolling = compute_rolling_features(matches)

    print("Computing league positions...")
    positions = compute_league_positions(matches)

    print("Computing Elo...")
    elo = compute_elo(matches)

    # Merge
    df = matches.merge(rolling,   on="match_id")
    df = df.merge(positions,      on="match_id")
    df = df.merge(elo,            on="match_id")

    # Derived positional features (avoid div-by-zero for GW1)
    home_ppg   = (df["home_pts"] / df["home_played"].replace(0, np.nan)).fillna(0.0)
    away_ppg   = (df["away_pts"] / df["away_played"].replace(0, np.nan)).fillna(0.0)
    home_gd_pg = (df["home_gd"]  / df["home_played"].replace(0, np.nan)).fillna(0.0)
    away_gd_pg = (df["away_gd"]  / df["away_played"].replace(0, np.nan)).fillna(0.0)

    # Reparameterize: diff (direction) + mean (match quality floor)
    # mean captures "both teams good" → higher-quality, less draw-friendly matches
    df["ppg_diff"]    = (home_ppg   - away_ppg).round(4)
    df["ppg_mean"]    = ((home_ppg  + away_ppg)  / 2).round(4)
    df["gd_pg_diff"]  = (home_gd_pg - away_gd_pg).round(4)
    df["gd_pg_mean"]  = ((home_gd_pg + away_gd_pg) / 2).round(4)
    df["rank_diff"]   = (df["home_rank"].astype(float) - df["away_rank"]).round(4)
    df["rank_mean"]   = ((df["home_rank"] + df["away_rank"]) / 2).round(4)


    # Drop rows where rolling features are NaN (very first match for a team)
    df = df.dropna(subset=FEATURE_COLS).copy()

    out_cols = ["match_id", "date", "season", "home_team", "away_team",
                *FEATURE_COLS, "result"]
    df = df[out_cols]

    print(f"\nDataset: {len(df)} rows")
    print(f"Season distribution:\n{df.groupby('season')['result'].count()}")
    print(f"\nResult distribution:\n{df['result'].value_counts(normalize=True).round(3)}")
    print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return df


if __name__ == "__main__":
    build_dataset_v3()
