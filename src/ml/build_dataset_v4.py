"""
Build H/D/A training dataset V4 (26 features).

Extends V3 base (24 features) with:
  1A. Opponent quality: home_opp_ppg_l5, away_opp_ppg_l5
  1B. EWM form (span=7): pts_ewm, goals_ewm, sot_ewm replacing momentum cols

Standalone — no dependency on build_dataset_v3.
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

DB_PATH     = PROJECT_ROOT / "data" / "processed" / "asil.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v4.csv"

VENUE_FALLBACK_MIN = 3
LAST_N             = 5
LAST_N_SHORT       = 3


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


def compute_elo(matches_df: pd.DataFrame) -> pd.DataFrame:
    records = matches_df[["match_id", "date", "home_team", "away_team", "result"]].to_dict("records")
    elo_map = EloCalculator().compute(records)
    return pd.DataFrame([{"match_id": mid, "elo_diff": vals["elo_diff"]} for mid, vals in elo_map.items()])


# ─────────────────────────────────────────────────────────────────────────────
# League positions (leakage-safe) — extended to also return team PPG
# ─────────────────────────────────────────────────────────────────────────────

def compute_league_positions_v4(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as V3 but also returns home_ppg, away_ppg (pre-match) for each fixture.
    Used to compute opponent quality features without leakage.
    """
    results = {}

    for season, season_df in matches_df.groupby("season"):
        season_df = season_df.sort_values("date")
        table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0})

        for date, day_df in season_df.groupby("date"):
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
                h_played = table[h]["played"]
                a_played = table[a]["played"]
                mw = round((h_played + a_played) / 2) + 1

                h_ppg = table[h]["pts"] / h_played if h_played > 0 else 0.0
                a_ppg = table[a]["pts"] / a_played if a_played > 0 else 0.0

                results[m["match_id"]] = {
                    "home_pts":    table[h]["pts"],
                    "home_gd":     table[h]["gd"],
                    "home_rank":   rank_map.get(h, 20),
                    "home_played": h_played,
                    "home_ppg":    round(h_ppg, 4),
                    "away_pts":    table[a]["pts"],
                    "away_gd":     table[a]["gd"],
                    "away_rank":   rank_map.get(a, 20),
                    "away_played": a_played,
                    "away_ppg":    round(a_ppg, 4),
                    "matchweek":   mw,
                }

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
# Rolling features — V4 adds EWM and opponent quality
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(series: pd.Series, n: int) -> pd.Series:
    return series.shift(1).rolling(n, min_periods=1).mean()


def compute_rolling_features_v4(matches_df: pd.DataFrame, positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all V3 features plus:
      - {role}_opp_ppg_l5   : rolling mean PPG of last 5 opponents
      - {role}_pts_ewm       : EWMA(span=5) of points
      - {role}_goals_ewm     : EWMA(span=5) of goals scored
      - {role}_sot_ewm       : EWMA(span=5) of SOT
    """
    # Build opponent PPG lookup: match_id → {home_opp_ppg, away_opp_ppg}
    # home team's opponent = away team's pre-match PPG, and vice versa
    opp_ppg = positions_df.set_index("match_id")[["home_ppg", "away_ppg"]]
    # home team faces the away team → opp_ppg for home = away_ppg
    opp_ppg_map = {
        mid: {"home": row["away_ppg"], "away": row["home_ppg"]}
        for mid, row in opp_ppg.iterrows()
    }

    # Expand into team-perspective records
    records = []
    for _, m in matches_df.iterrows():
        base = dict(match_id=m["match_id"], date=m["date"], season=m["season"])
        h_opp = opp_ppg_map.get(m["match_id"], {}).get("home", 0.0)
        a_opp = opp_ppg_map.get(m["match_id"], {}).get("away", 0.0)

        records.append({**base,
            "team": m["home_team"], "is_home": True,
            "goals_scored":  m["home_goals"],   "goals_conceded": m["away_goals"],
            "sot":           m["home_shots_target"], "sot_conceded": m["away_shots_target"],
            "clean_sheet":   int(m["away_goals"] == 0),
            "points":        3 if m["result"] == "H" else (1 if m["result"] == "D" else 0),
            "opp_ppg":       h_opp,
        })
        records.append({**base,
            "team": m["away_team"], "is_home": False,
            "goals_scored":  m["away_goals"],   "goals_conceded": m["home_goals"],
            "sot":           m["away_shots_target"], "sot_conceded": m["home_shots_target"],
            "clean_sheet":   int(m["home_goals"] == 0),
            "points":        3 if m["result"] == "A" else (1 if m["result"] == "D" else 0),
            "opp_ppg":       a_opp,
        })

    tm = pd.DataFrame(records)
    out_rows = {mid: {} for mid in matches_df["match_id"]}

    for team, grp in tm.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # ── V3 overall rolling ─────────────────────────────────────────────
        for col in ["sot", "sot_conceded", "goals_scored", "clean_sheet", "points"]:
            grp[f"ovr_{col}_l5"] = _rolling_mean(grp[col], LAST_N)
            grp[f"ovr_{col}_l3"] = _rolling_mean(grp[col], LAST_N_SHORT)

        # ── V3 venue-specific rolling ──────────────────────────────────────
        for is_home, label in [(True, "home"), (False, "away")]:
            venue_mask = grp["is_home"] == is_home
            vg = grp[venue_mask].copy()
            vg["season_venue_count"] = vg.groupby("season").cumcount()
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

        # ── 1B: EWM form (span=7, α=0.25) — span sensitivity: 7 best on 2526
        for col in ["points", "goals_scored", "sot"]:
            grp[f"{col}_ewm"] = grp[col].shift(1).ewm(span=7, min_periods=1).mean()

        # ── 1A: Opponent quality rolling ───────────────────────────────────
        grp["opp_ppg_l5"] = _rolling_mean(grp["opp_ppg"], LAST_N)

        # ── Write to output ────────────────────────────────────────────────
        for _, row in grp.iterrows():
            mid  = row["match_id"]
            role = "home" if row["is_home"] else "away"
            venue_label = role

            use_venue = row.get(f"{venue_label}_season_count", 0) >= VENUE_FALLBACK_MIN

            # V3 features (venue fallback)
            for col in ["sot", "sot_conceded", "clean_sheet"]:
                val = (row[f"{venue_label}_{col}_l5"]
                       if use_venue and not pd.isna(row.get(f"{venue_label}_{col}_l5"))
                       else row[f"ovr_{col}_l5"])
                out_rows[mid][f"{role}_{col}_l5"] = round(float(val), 4)

            gs_l5 = row["ovr_goals_scored_l5"]
            gs_l3 = row["ovr_goals_scored_l3"]
            sot_l5_val = out_rows[mid][f"{role}_sot_l5"]

            out_rows[mid][f"{role}_conversion"] = round(
                (gs_l5 + 1) / (sot_l5_val + 2), 4
            )

            pts_l5 = row["ovr_points_l5"]
            pts_l3 = row["ovr_points_l3"]
            sot_l3 = row["ovr_sot_l3"]
            out_rows[mid][f"{role}_pts_momentum"]   = round(float(pts_l3 - pts_l5), 4)
            out_rows[mid][f"{role}_goals_momentum"] = round(float(gs_l3  - gs_l5),  4)
            out_rows[mid][f"{role}_sot_momentum"]   = round(float(sot_l3 - sot_l5_val), 4)
            out_rows[mid][f"{role}_days_rest"]       = int(row["days_rest"])

            # V4 new features
            out_rows[mid][f"{role}_pts_ewm"]   = round(float(row["points_ewm"]),      4)
            out_rows[mid][f"{role}_goals_ewm"] = round(float(row["goals_scored_ewm"]), 4)
            out_rows[mid][f"{role}_sot_ewm"]   = round(float(row["sot_ewm"]),          4)
            out_rows[mid][f"{role}_opp_ppg_l5"] = round(float(row["opp_ppg_l5"]),      4)

    return pd.DataFrame.from_dict(out_rows, orient="index").rename_axis("match_id").reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Assemble
# ─────────────────────────────────────────────────────────────────────────────

BASE_24_COLS = [
    "home_sot_l5", "home_sot_conceded_l5", "home_conversion",
    "home_clean_sheet_l5", "home_pts_momentum", "home_goals_momentum",
    "home_sot_momentum", "home_days_rest",
    "away_sot_l5", "away_sot_conceded_l5", "away_conversion",
    "away_clean_sheet_l5", "away_pts_momentum", "away_goals_momentum",
    "away_sot_momentum", "away_days_rest",
    "elo_diff",
    "ppg_diff", "ppg_mean",
    "gd_pg_diff", "gd_pg_mean",
    "rank_diff", "rank_mean",
    "matchweek",
]

# 1A: add opponent quality (2 new)
OPP_COLS = ["home_opp_ppg_l5", "away_opp_ppg_l5"]

# 1B: swap 6 momentum cols with 6 ewm cols (same count)
MOMENTUM_COLS = [
    "home_pts_momentum", "home_goals_momentum", "home_sot_momentum",
    "away_pts_momentum", "away_goals_momentum", "away_sot_momentum",
]
EWM_COLS = [
    "home_pts_ewm", "home_goals_ewm", "home_sot_ewm",
    "away_pts_ewm", "away_goals_ewm", "away_sot_ewm",
]

# Prebuilt feature sets for backtest
FEATURES_BASE   = BASE_24_COLS
FEATURES_OPP    = BASE_24_COLS + OPP_COLS                                         # 26
FEATURES_EWM    = [c for c in BASE_24_COLS if c not in MOMENTUM_COLS] + EWM_COLS  # 24, ewm replaces momentum
FEATURES_COMBO  = FEATURES_EWM + OPP_COLS                                         # 26


def build_dataset_v4(output_path: str = str(OUTPUT_PATH)) -> pd.DataFrame:
    print("Loading matches...")
    matches = load_matches()
    print(f"  {len(matches)} matches across seasons: {sorted(matches['season'].unique())}")

    print("Computing league positions (V4)...")
    positions = compute_league_positions_v4(matches)

    print("Computing rolling features (V4)...")
    rolling = compute_rolling_features_v4(matches, positions)

    print("Computing Elo...")
    elo = compute_elo(matches)

    df = matches.merge(rolling,    on="match_id")
    df = df.merge(positions,       on="match_id")
    df = df.merge(elo,             on="match_id")

    # Derived positional features
    home_ppg_s   = (df["home_pts"] / df["home_played"].replace(0, np.nan)).fillna(0.0)
    away_ppg_s   = (df["away_pts"] / df["away_played"].replace(0, np.nan)).fillna(0.0)
    home_gd_pg   = (df["home_gd"]  / df["home_played"].replace(0, np.nan)).fillna(0.0)
    away_gd_pg   = (df["away_gd"]  / df["away_played"].replace(0, np.nan)).fillna(0.0)

    df["ppg_diff"]   = (home_ppg_s - away_ppg_s).round(4)
    df["ppg_mean"]   = ((home_ppg_s + away_ppg_s) / 2).round(4)
    df["gd_pg_diff"] = (home_gd_pg  - away_gd_pg).round(4)
    df["gd_pg_mean"] = ((home_gd_pg  + away_gd_pg) / 2).round(4)
    df["rank_diff"]  = (df["home_rank"].astype(float) - df["away_rank"]).round(4)
    df["rank_mean"]  = ((df["home_rank"] + df["away_rank"]) / 2).round(4)

    all_feature_cols = FEATURES_COMBO  # superset includes all new cols
    df = df.dropna(subset=all_feature_cols).copy()

    out_cols = ["match_id", "date", "season", "home_team", "away_team",
                *BASE_24_COLS, *OPP_COLS, *EWM_COLS, "result"]
    df = df[out_cols]

    print(f"\nDataset V4: {len(df)} rows")
    print(f"Season distribution:\n{df.groupby('season')['result'].count()}")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return df


if __name__ == "__main__":
    build_dataset_v4()
