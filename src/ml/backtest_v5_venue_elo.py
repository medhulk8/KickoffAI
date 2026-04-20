"""
Phase 2B: venue-split Elo ablation.

Tests whether replacing/augmenting the unified elo_diff with venue_elo_diff
(home_elo[home_team] - away_elo[away_team], trained separately) improves on V4.

Configs (all 2526 holdout, train [1920,2122,2223,2324,2425]):
  A. V4 26 features (unified elo_diff)
  B. elo_diff → venue_elo_diff (1-for-1 swap, still 26 features)
  C. V4 + venue_elo_diff as extra (27 features, both signals)
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
    load_matches, compute_league_positions_v4, compute_rolling_features_v4,
    FEATURES_COMBO, OUTPUT_PATH as V4_PATH,
)
from src.ml.elo import EloCalculator, VenueEloCalculator

TRAIN_SEASONS  = [1920, 2122, 2223, 2324, 2425]
HOLDOUT_SEASON = 2526
CLASSES        = ["A", "D", "H"]

V5_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v5.csv"


def build_v5_dataset() -> pd.DataFrame:
    """V4 dataset + venue_elo_diff column."""
    if Path(V4_PATH).exists():
        df = pd.read_csv(V4_PATH)
    else:
        from src.ml.build_dataset_v4 import build_dataset_v4
        df = build_dataset_v4()

    matches = load_matches()
    records = matches[["match_id", "date", "home_team", "away_team", "result"]].to_dict("records")

    venue_calc = VenueEloCalculator(k=20.0)
    venue_map  = venue_calc.compute(records)

    venue_rows = [{"match_id": mid, "venue_elo_diff": v["venue_elo_diff"]}
                  for mid, v in venue_map.items()]
    venue_df = pd.DataFrame(venue_rows)

    merged = df.merge(venue_df, on="match_id", how="left")
    merged.to_csv(V5_PATH, index=False)
    print(f"V5 dataset: {len(merged)} rows, venue_elo_diff added → {V5_PATH}")
    return merged


def fit_eval(X_tr, y_tr, X_te, y_te, label, base_acc=None, base_ll=None):
    model = CalibratedClassifierCV(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42))]),
        cv=5, method="sigmoid",
    )
    model.fit(X_tr, y_tr)
    p_raw = model.predict_proba(X_te)
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(model.classes_):
        p[:, CLASSES.index(c)] = p_raw[:, i]
    y_pred = [CLASSES[i] for i in np.argmax(p, axis=1)]
    acc = accuracy_score(y_te, y_pred)
    ll  = log_loss(y_te, p, labels=CLASSES)

    delta = ""
    if base_acc is not None:
        delta = f"  Δacc={acc-base_acc:+.3f}  Δll={ll-base_ll:+.4f}"
    print(f"  {label:<45}  acc={acc:.3f}  ll={ll:.4f}  n_feat={X_tr.shape[1]}{delta}")
    return acc, ll


def run():
    if not V5_PATH.exists():
        df = build_v5_dataset()
    else:
        df = pd.read_csv(V5_PATH)

    df["season"] = df["season"].astype(int)
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test  = df[df["season"] == HOLDOUT_SEASON]

    print(f"Train: {len(train)}  Test: {len(test)}\n")

    # Feature sets
    FEAT_V4    = FEATURES_COMBO                              # 26 (has elo_diff)
    FEAT_VENUE = [c if c != "elo_diff" else "venue_elo_diff"
                  for c in FEATURES_COMBO]                   # 26, swap elo_diff
    FEAT_BOTH  = FEATURES_COMBO + ["venue_elo_diff"]         # 27

    base_acc, base_ll = fit_eval(
        train[FEAT_V4].to_numpy(), np.array(train["result"]),
        test[FEAT_V4].to_numpy(),  np.array(test["result"]),
        "A. V4 unified elo_diff (baseline)",
    )

    fit_eval(
        train[FEAT_VENUE].to_numpy(), np.array(train["result"]),
        test[FEAT_VENUE].to_numpy(),  np.array(test["result"]),
        "B. venue_elo_diff replaces elo_diff",
        base_acc, base_ll,
    )

    fit_eval(
        train[FEAT_BOTH].to_numpy(), np.array(train["result"]),
        test[FEAT_BOTH].to_numpy(),  np.array(test["result"]),
        "C. both elo_diff + venue_elo_diff",
        base_acc, base_ll,
    )

    # Sanity: correlation between elo_diff and venue_elo_diff
    corr = df[["elo_diff", "venue_elo_diff"]].corr().iloc[0, 1]
    print(f"\nCorrelation elo_diff / venue_elo_diff: {corr:.3f}")

    # Variance: do they actually diverge meaningfully?
    print(f"\nStd elo_diff:       {df['elo_diff'].std():.1f}")
    print(f"Std venue_elo_diff: {df['venue_elo_diff'].std():.1f}")


if __name__ == "__main__":
    run()
