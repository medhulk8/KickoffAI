"""
Extend training_dataset_v3.csv with rolling xG features.

Reads xg_data.csv (scraped from understat), matches to training dataset
by home_team + away_team + date, then computes:
  home_xg_scored_l5    rolling avg home xG generated, last 5
  home_xg_conceded_l5  rolling avg home xG conceded, last 5
  away_xg_scored_l5
  away_xg_conceded_l5

Venue fallback rule same as other features:
  < 3 venue matches this season → use overall last-5 xG

Output: data/processed/training_dataset_v3_xg.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"
XG_PATH      = PROJECT_ROOT / "data" / "processed" / "xg_data.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3_xg.csv"

VENUE_FALLBACK_MIN = 3
LAST_N = 5

# Our DB team names that might differ from understat
# (extend if match rate is low after first run)
TEAM_NAME_MAP = {
    "Brighton & Hove Albion": ["Brighton"],
    "Wolverhampton":          ["Wolverhampton Wanderers", "Wolves"],
    "Leeds United":           ["Leeds"],
    "Leicester City":         ["Leicester"],
    "Huddersfield Town":      ["Huddersfield"],
    "Cardiff City":           ["Cardiff"],
    "Swansea City":           ["Swansea"],
    "Stoke City":             ["Stoke"],
    "Norwich City":           ["Norwich"],
    "Luton":                  ["Luton Town"],
    "West Ham United":        ["West Ham"],
    "Tottenham Hotspur":      ["Tottenham"],
}


def _build_alias_map(team_name_map: dict) -> dict:
    """Reverse map: understat alias → our DB name."""
    m = {}
    for db_name, aliases in team_name_map.items():
        for alias in aliases:
            m[alias] = db_name
    return m


def _rolling_mean(series: pd.Series, n: int) -> pd.Series:
    return series.shift(1).rolling(n, min_periods=1).mean()


def load_and_match_xg(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load xg_data.csv and match to dataset rows by home_team + away_team + date.
    Returns dataset_df with home_xg, away_xg columns added.
    """
    xg = pd.read_csv(XG_PATH)
    xg["date"] = pd.to_datetime(xg["date"]).dt.strftime("%Y-%m-%d")
    xg["season"] = xg["season"].astype(str)

    alias_map = _build_alias_map(TEAM_NAME_MAP)
    xg["home_team"] = xg["home_team"].map(lambda x: alias_map.get(x, x))
    xg["away_team"] = xg["away_team"].map(lambda x: alias_map.get(x, x))

    dataset_df["date_str"] = pd.to_datetime(dataset_df["date"]).dt.strftime("%Y-%m-%d")

    merged = dataset_df.merge(
        xg[["date", "home_team", "away_team", "home_xg", "away_xg"]].rename(columns={"date": "date_str"}),
        on=["date_str", "home_team", "away_team"],
        how="left",
    )

    match_rate = merged["home_xg"].notna().mean()
    print(f"xG match rate: {match_rate:.1%}  ({merged['home_xg'].notna().sum()}/{len(merged)} matches)")
    if match_rate < 0.90:
        unmatched = merged[merged["home_xg"].isna()][["date_str","home_team","away_team","season"]].head(10)
        print(f"Sample unmatched:\n{unmatched.to_string()}")

    merged.drop(columns=["date_str"], inplace=True)
    return merged


def compute_xg_rolling(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling xG features per team.
    Returns DataFrame indexed by match_id with xG rolling columns.
    """
    records = []
    for _, m in matches_df.iterrows():
        if pd.isna(m.get("home_xg")):
            continue
        base = dict(match_id=m["match_id"], date=pd.to_datetime(m["date"]), season=str(m["season"]))
        records.append({**base, "team": m["home_team"], "is_home": True,
                        "xg_scored": m["home_xg"], "xg_conceded": m["away_xg"]})
        records.append({**base, "team": m["away_team"], "is_home": False,
                        "xg_scored": m["away_xg"], "xg_conceded": m["home_xg"]})

    if not records:
        print("No xG data found — cannot compute rolling features")
        return pd.DataFrame()

    tm = pd.DataFrame(records)
    out = {}

    for team, grp in tm.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)

        grp["ovr_xgs_l5"] = _rolling_mean(grp["xg_scored"],   LAST_N)
        grp["ovr_xgc_l5"] = _rolling_mean(grp["xg_conceded"], LAST_N)

        for is_home, label in [(True, "home"), (False, "away")]:
            vg = grp[grp["is_home"] == is_home].copy()
            vg["season_venue_count"] = vg.groupby("season").cumcount()
            vg["ven_xgs_l5"] = _rolling_mean(vg["xg_scored"],   LAST_N)
            vg["ven_xgc_l5"] = _rolling_mean(vg["xg_conceded"], LAST_N)
            grp = grp.merge(
                vg[["match_id", "season_venue_count", "ven_xgs_l5", "ven_xgc_l5"]],
                on="match_id", how="left",
            )
            grp.rename(columns={
                "season_venue_count": f"{label}_season_count",
                "ven_xgs_l5": f"{label}_xgs_l5",
                "ven_xgc_l5": f"{label}_xgc_l5",
            }, inplace=True)

        for _, row in grp.iterrows():
            mid  = row["match_id"]
            role = "home" if row["is_home"] else "away"
            venue_label = role
            use_venue = row.get(f"{venue_label}_season_count", 0) >= VENUE_FALLBACK_MIN

            xgs = (row[f"{venue_label}_xgs_l5"]
                   if use_venue and not pd.isna(row.get(f"{venue_label}_xgs_l5"))
                   else row["ovr_xgs_l5"])
            xgc = (row[f"{venue_label}_xgc_l5"]
                   if use_venue and not pd.isna(row.get(f"{venue_label}_xgc_l5"))
                   else row["ovr_xgc_l5"])

            if mid not in out:
                out[mid] = {}
            out[mid][f"{role}_xg_scored_l5"]   = round(float(xgs), 4)
            out[mid][f"{role}_xg_conceded_l5"] = round(float(xgc), 4)

    return pd.DataFrame.from_dict(out, orient="index").rename_axis("match_id").reset_index()


XG_FEATURE_COLS = [
    "home_xg_scored_l5", "home_xg_conceded_l5",
    "away_xg_scored_l5", "away_xg_conceded_l5",
]


def build_xg_dataset():
    print("Loading base dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("Matching xG data...")
    df = load_and_match_xg(df)

    print("Computing rolling xG features...")
    xg_rolling = compute_xg_rolling(df)

    if xg_rolling.empty:
        print("Cannot build xG dataset — no data matched.")
        return

    df = df.merge(xg_rolling, on="match_id", how="left")
    before = len(df)
    df = df.dropna(subset=XG_FEATURE_COLS).copy()
    print(f"Dropped {before - len(df)} rows with missing xG rolling features ({len(df)} remain)")

    base_feature_cols = [
        "home_sot_l5", "home_sot_conceded_l5", "home_conversion",
        "home_clean_sheet_l5", "home_pts_momentum", "home_goals_momentum", "home_days_rest",
        "away_sot_l5", "away_sot_conceded_l5", "away_conversion",
        "away_clean_sheet_l5", "away_pts_momentum", "away_goals_momentum", "away_days_rest",
        "elo_diff", "home_rank", "away_rank", "matchweek",
    ]
    all_features = base_feature_cols + XG_FEATURE_COLS

    out_cols = ["match_id", "date", "season", "home_team", "away_team",
                *all_features, "result"]
    df = df[out_cols]

    print(f"\nFinal dataset: {len(df)} rows, {len(all_features)} features")
    print(f"Season distribution:\n{df.groupby('season')['result'].count()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    build_xg_dataset()
