"""
Train V4 production model — 26 features, Last 5 seasons, span=7 EWM.

New vs V3:
  - Replaces 6 linear momentum cols with 6 EWM cols (EWMA span=7)
  - Adds home_opp_ppg_l5, away_opp_ppg_l5 (schedule difficulty)

Training window: [1920, 2122, 2223, 2324, 2425]
Holdout eval:   2526 (permanent holdout — never touched during training)

Outputs:
  models/lr_v4_final.pkl   — CalibratedClassifierCV(Pipeline(Scaler+LR), cv=5, sigmoid)
  models/lr_v4_meta.json
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

from src.ml.build_dataset_v4 import FEATURES_COMBO, OUTPUT_PATH as DATASET_PATH

MODEL_PATH = PROJECT_ROOT / "models" / "lr_v4_final.pkl"
META_PATH  = PROJECT_ROOT / "models" / "lr_v4_meta.json"

CLASSES = ["A", "D", "H"]
TRAIN_SEASONS  = [1920, 2122, 2223, 2324, 2425]
HOLDOUT_SEASON = 2526


def multiclass_brier(y_enc, y_prob, n=3):
    return float(np.mean(np.sum((y_prob - np.eye(n)[y_enc]) ** 2, axis=1)))


def train():
    if not Path(DATASET_PATH).exists():
        print("V4 dataset not found — building...")
        from src.ml.build_dataset_v4 import build_dataset_v4
        build_dataset_v4()

    df = pd.read_csv(DATASET_PATH)
    df["season"] = df["season"].astype(int)

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df  = df[df["season"] == HOLDOUT_SEASON]

    X_train = train_df[FEATURES_COMBO].to_numpy()
    y_train = np.array(train_df["result"])
    X_test  = test_df[FEATURES_COMBO].to_numpy()
    y_test  = np.array(test_df["result"])

    print(f"Training on {len(train_df)} rows, {len(FEATURES_COMBO)} features")
    print(f"Holdout: {len(test_df)} rows ({HOLDOUT_SEASON})\n")

    model = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42))]),
        cv=5, method="sigmoid",
    )
    model.fit(X_train, y_train)

    p_raw = model.predict_proba(X_test)
    classes = list(model.classes_)
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(classes):
        p[:, CLASSES.index(c)] = p_raw[:, i]

    y_pred = [CLASSES[i] for i in np.argmax(p, axis=1)]
    acc   = accuracy_score(y_test, y_pred)
    ll    = log_loss(y_test, p, labels=CLASSES)
    brier = multiclass_brier(
        np.array([CLASSES.index(c) for c in y_test]), p
    )
    dr = float((np.array(y_pred)[np.array(y_test) == "D"] == "D").mean()) \
         if (np.array(y_test) == "D").sum() > 0 else 0.0

    print(f"Holdout {HOLDOUT_SEASON}:")
    print(f"  acc={acc:.1%}  ll={ll:.4f}  brier={brier:.4f}  draw_recall={dr:.1%}")
    print(f"\n  V3 baseline:  acc=48.4%  ll=1.0258")
    print(f"  V4 (this):    acc={acc:.1%}  ll={ll:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model → {MODEL_PATH}")

    meta = {
        "model_type": "CalibratedClassifierCV(Pipeline(StandardScaler+LR), cv=5, sigmoid)",
        "features": FEATURES_COMBO,
        "n_features": len(FEATURES_COMBO),
        "ewm_span": 7,
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
        "vs_v3": {
            "acc_delta": round(acc - 0.484, 4),
            "ll_delta": round(ll - 1.0258, 4),
        },
        "classes_order": CLASSES,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {META_PATH}")
    return model


if __name__ == "__main__":
    train()
