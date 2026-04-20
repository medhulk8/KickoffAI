"""
Train V3 final model — last 5 seasons, with Platt calibration.

Rolling-origin backtest showed Last 5 seasons gives best LL (0.9830 avg)
vs full history (0.9844), confirming concept drift across older seasons.

Training window: [1920, 2122, 2223, 2324, 2425]
Holdout eval:   2526 (true unseen test — never touched during training)

Outputs:
  models/lr_v3_final.pkl   — CalibratedClassifierCV(LR, cv=5, method='sigmoid')
  models/lr_v3_meta.json   — feature list, training size, calibration info
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"
MODEL_PATH   = PROJECT_ROOT / "models" / "lr_v3_final.pkl"
META_PATH    = PROJECT_ROOT / "models" / "lr_v3_meta.json"

FEATURE_COLS = [
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

CLASSES = ["A", "D", "H"]


def multiclass_brier(y_enc, y_prob, n=3):
    return float(np.mean(np.sum((y_prob - np.eye(n)[y_enc]) ** 2, axis=1)))


def draw_recall(y_true, y_pred):
    mask = y_true == "D"
    return float((y_pred[mask] == "D").mean()) if mask.sum() > 0 else 0.0


def train():
    df = pd.read_csv(DATASET_PATH)
    le = LabelEncoder()
    le.fit(CLASSES)

    TRAIN_SEASONS = [1920, 2122, 2223, 2324, 2425]
    HOLDOUT_SEASON = 2526

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df  = df[df["season"] == HOLDOUT_SEASON]

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["result"].to_numpy().astype(str)
    X_test  = test_df[FEATURE_COLS].to_numpy()
    y_test  = test_df["result"].to_numpy().astype(str)

    print(f"Training on {len(train_df)} rows (seasons {TRAIN_SEASONS})")
    print(f"Holdout: {len(test_df)} rows ({HOLDOUT_SEASON})\n")

    # Base LR pipeline
    base_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42)),
    ])

    # Calibrated wrapper — CV=5 uses cross-validation within training set
    model = CalibratedClassifierCV(base_lr, cv=5, method="sigmoid")
    model.fit(X_train, y_train)

    # Evaluate on holdout
    p_raw  = model.predict_proba(X_test)
    # Align to CLASSES order
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(model.classes_):
        p[:, le.transform([c])[0]] = p_raw[:, i]

    y_pred = le.inverse_transform(p.argmax(axis=1))
    acc    = accuracy_score(y_test, y_pred)
    ll     = log_loss(y_test, p, labels=le.classes_)
    brier  = multiclass_brier(le.transform(y_test), p)
    dr     = draw_recall(y_test, y_pred)

    print(f"Holdout {HOLDOUT_SEASON} (calibrated LR, trained on {TRAIN_SEASONS}):")
    print(f"  acc={acc:.1%}  ll={ll:.4f}  brier={brier:.4f}  draw_recall={dr:.1%}")

    # Check calibration improvement vs uncalibrated
    base_lr_unc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42)),
    ])
    base_lr_unc.fit(X_train, y_train)
    p_unc_raw = base_lr_unc.predict_proba(X_test)
    p_unc = np.zeros((len(p_unc_raw), 3))
    for i, c in enumerate(base_lr_unc.named_steps["clf"].classes_):
        p_unc[:, le.transform([c])[0]] = p_unc_raw[:, i]
    print(f"\nUncalibrated LR (same training):")
    print(f"  acc={accuracy_score(y_test, base_lr_unc.predict(X_test)):.1%}  "
          f"ll={log_loss(y_test, p_unc, labels=le.classes_):.4f}  "
          f"brier={multiclass_brier(le.transform(y_test), p_unc):.4f}  "
          f"draw_recall={draw_recall(y_test, base_lr_unc.predict(X_test)):.1%}")

    # Production model: same 5-season window (2526 stays as held-out eval)
    print(f"\nProduction model = same 5-season window ({TRAIN_SEASONS}, {len(train_df)} rows)")
    prod_model = model  # already trained above

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(prod_model, f)
    print(f"Saved model → {MODEL_PATH}")

    meta = {
        "model_type": "CalibratedClassifierCV(LR, cv=5, sigmoid)",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "training_seasons": TRAIN_SEASONS,
        "training_rows": len(train_df),
        "holdout_eval": {
            "season": HOLDOUT_SEASON,
            "n": len(test_df),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier": round(brier, 4),
            "draw_recall": round(dr, 4),
        },
        "classes_order": CLASSES,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {META_PATH}")

    return prod_model


if __name__ == "__main__":
    train()
