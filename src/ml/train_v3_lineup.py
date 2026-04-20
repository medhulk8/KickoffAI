"""
Train the lineup-aware production model.

Trains on all 3 seasons where FPL lineup data exists: 2223 + 2324 + 2425.
Features: V3 base 24 + xi_strength_diff = 25 features.

This is the "lineup-known" variant. Used when confirmed XI is available.
Default fallback is lr_v3_final.pkl (7-season, 24 features).

Output: models/lr_v3_lineup.pkl
"""

from __future__ import annotations

import json
import pickle
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

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"
LINEUP_PATH  = PROJECT_ROOT / "data" / "processed" / "lineup_features.csv"
MODEL_PATH   = PROJECT_ROOT / "models" / "lr_v3_lineup.pkl"
META_PATH    = PROJECT_ROOT / "models" / "lr_v3_lineup_meta.json"

LINEUP_SEASONS = [2223, 2324, 2425]

BASE_FEATURES = [
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

LINEUP_FEATURES = BASE_FEATURES + ["xi_strength_diff"]


def train():
    print("Loading base dataset...")
    base = pd.read_csv(DATASET_PATH)
    base["date"] = base["date"].str[:10]

    print("Loading lineup features...")
    lu = pd.read_csv(LINEUP_PATH)
    lu["date"] = lu["date"].str[:10]

    merged = base.merge(
        lu[["date", "home_team", "away_team", "xi_strength_diff"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    train_df = merged[merged["season"].isin(LINEUP_SEASONS)].dropna(subset=["xi_strength_diff"])
    print(f"Training rows: {len(train_df)} across seasons {sorted(train_df['season'].unique())}")
    print(f"Result distribution:\n{train_df['result'].value_counts().to_dict()}")

    X = train_df[LINEUP_FEATURES].values.astype(float)
    y = np.array(train_df["result"])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    clf = CalibratedClassifierCV(pipe, cv=5, method="sigmoid")
    clf.fit(X, y)

    # In-sample check
    p_raw = clf.predict_proba(X)
    classes = list(clf.classes_)
    y_pred = classes[np.argmax(p_raw, axis=1).tolist()] if False else [
        classes[i] for i in np.argmax(p_raw, axis=1)
    ]
    acc = accuracy_score(y, y_pred)
    ll  = log_loss(y, p_raw, labels=classes)
    print(f"In-sample (all 3 seasons): acc={acc:.3f}  ll={ll:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model → {MODEL_PATH}")

    meta = {
        "model":    "lr_v3_lineup",
        "features": LINEUP_FEATURES,
        "n_features": len(LINEUP_FEATURES),
        "seasons":  LINEUP_SEASONS,
        "n_train":  len(train_df),
        "note":     "Lineup-aware variant. Use when confirmed xi_strength_diff is known.",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {META_PATH}")

    print("\nFeature importances (abs coef, H class, scaled):")
    from sklearn.pipeline import Pipeline as _P
    pipe_raw = _P([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    pipe_raw.fit(X, y)
    lr_raw = pipe_raw.named_steps["lr"]
    coef_H = lr_raw.coef_[list(lr_raw.classes_).index("H")]
    for feat, c in sorted(zip(LINEUP_FEATURES, coef_H), key=lambda x: -abs(x[1]))[:8]:
        print(f"  {feat:<30}  {c:+.4f}")

    return clf


if __name__ == "__main__":
    train()
