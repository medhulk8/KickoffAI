"""
Span sensitivity test for EWM features.

Tests span=3,5,7 for the EWM form signal (pts/goals/sot EWMA).
Keeps all other features fixed (base 24 + opp_ppg_l5).
Train: [1920,2122,2223,2324,2425]  Test: 2526 holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.build_dataset_v4 import (
    load_matches, compute_league_positions_v4, compute_elo,
    FEATURES_COMBO, OPP_COLS,
    OUTPUT_PATH as V4_PATH,
)

TRAIN_SEASONS  = [1920, 2122, 2223, 2324, 2425]
HOLDOUT_SEASON = 2526
CLASSES = ["A", "D", "H"]

EWM_COLS = [
    "home_pts_ewm", "home_goals_ewm", "home_sot_ewm",
    "away_pts_ewm", "away_goals_ewm", "away_sot_ewm",
]
MOMENTUM_COLS = [
    "home_pts_momentum", "home_goals_momentum", "home_sot_momentum",
    "away_pts_momentum", "away_goals_momentum", "away_sot_momentum",
]
BASE_NO_MOMENTUM = [c for c in FEATURES_COMBO if c not in
                    ["home_pts_ewm","home_goals_ewm","home_sot_ewm",
                     "away_pts_ewm","away_goals_ewm","away_sot_ewm",
                     *MOMENTUM_COLS]]


def recompute_ewm(matches_df: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    Recompute the 6 EWM columns for a given span.
    Returns a DataFrame: match_id, home_pts_ewm, home_goals_ewm, home_sot_ewm,
                                   away_pts_ewm, away_goals_ewm, away_sot_ewm
    """
    records = []
    for _, m in matches_df.iterrows():
        base = dict(match_id=m["match_id"], date=m["date"])
        records.append({**base, "team": m["home_team"], "is_home": True,
                        "points": 3 if m["result"]=="H" else (1 if m["result"]=="D" else 0),
                        "goals_scored": m["home_goals"], "sot": m["home_shots_target"]})
        records.append({**base, "team": m["away_team"], "is_home": False,
                        "points": 3 if m["result"]=="A" else (1 if m["result"]=="D" else 0),
                        "goals_scored": m["away_goals"], "sot": m["away_shots_target"]})

    tm = pd.DataFrame(records)
    out = {mid: {} for mid in matches_df["match_id"]}

    for team, grp in tm.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for col, out_col in [("points","pts_ewm"), ("goals_scored","goals_ewm"), ("sot","sot_ewm")]:
            grp[out_col] = grp[col].shift(1).ewm(span=span, min_periods=1).mean()
        for _, row in grp.iterrows():
            mid  = row["match_id"]
            role = "home" if row["is_home"] else "away"
            out[mid][f"{role}_pts_ewm"]   = round(float(row["pts_ewm"]),   4)
            out[mid][f"{role}_goals_ewm"] = round(float(row["goals_ewm"]), 4)
            out[mid][f"{role}_sot_ewm"]   = round(float(row["sot_ewm"]),   4)

    return pd.DataFrame.from_dict(out, orient="index").rename_axis("match_id").reset_index()


def fit_eval(X_train, y_train, X_test, y_test, label: str) -> tuple[float, float]:
    model = CalibratedClassifierCV(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42))]),
        cv=5, method="sigmoid",
    )
    model.fit(X_train, y_train)
    p_raw = model.predict_proba(X_test)
    classes = list(model.classes_)
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(classes):
        p[:, CLASSES.index(c)] = p_raw[:, i]
    y_pred = [CLASSES[i] for i in np.argmax(p, axis=1)]
    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, p, labels=CLASSES)
    print(f"  {label:<30}  acc={acc:.3f}  ll={ll:.4f}")
    return acc, ll


def run():
    base_df = pd.read_csv(V4_PATH)
    base_df["season"] = base_df["season"].astype(int)

    matches = load_matches()

    print(f"Span sensitivity test — EWM + opp_ppg combo, 2526 holdout (n=318)\n")

    results = []
    for span in [3, 5, 7]:
        ewm_df = recompute_ewm(matches, span)
        df = base_df.drop(columns=EWM_COLS).merge(ewm_df, on="match_id")

        feats = BASE_NO_MOMENTUM + EWM_COLS + OPP_COLS

        train = df[df["season"].isin(TRAIN_SEASONS)]
        test  = df[df["season"] == HOLDOUT_SEASON]

        acc, ll = fit_eval(
            train[feats].to_numpy(), np.array(train["result"]),
            test[feats].to_numpy(),  np.array(test["result"]),
            f"span={span}",
        )
        results.append((span, acc, ll))

    best_span = min(results, key=lambda x: x[2])[0]
    print(f"\nBest span by LL: span={best_span}")


if __name__ == "__main__":
    run()
