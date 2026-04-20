"""
Ablation backtest V4 — tests opponent quality (1A) and EWM form (1B).

Feature sets compared (all trained on [1920,2122,2223,2324,2425], tested on 2526):
  A. Base 24 — V3 production baseline
  B. Base + opp_ppg_l5 (1A: opponent quality, 26 features)
  C. EWM swap (1B: replace momentum cols with ewm, still 24 features)
  D. EWM + opp_ppg (1A+1B combined, 26 features)
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
    FEATURES_BASE, FEATURES_OPP, FEATURES_EWM, FEATURES_COMBO,
    OUTPUT_PATH as V4_DATASET_PATH,
)

TRAIN_SEASONS  = [1920, 2122, 2223, 2324, 2425]
HOLDOUT_SEASON = 2526
CLASSES = ["A", "D", "H"]


def fit_eval(X_train, y_train, X_test, y_test, label: str) -> tuple[float, float]:
    model = CalibratedClassifierCV(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42))]),
        cv=5, method="sigmoid",
    )
    model.fit(X_train, y_train)

    p_raw = model.predict_proba(X_test)
    classes = list(model.classes_)

    # Align to A/D/H order
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(classes):
        p[:, CLASSES.index(c)] = p_raw[:, i]

    y_pred = [CLASSES[i] for i in np.argmax(p, axis=1)]
    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, p, labels=CLASSES)
    dr  = float((np.array(y_pred)[np.array(y_test) == "D"] == "D").mean()) \
          if (np.array(y_test) == "D").sum() > 0 else 0.0

    delta_acc = ""
    delta_ll  = ""

    print(f"  {label:<40}  acc={acc:.3f}  ll={ll:.4f}  draw_recall={dr:.1%}  n_feat={X_train.shape[1]}")
    return acc, ll


def run():
    if not Path(V4_DATASET_PATH).exists():
        print("V4 dataset not found — building...")
        from src.ml.build_dataset_v4 import build_dataset_v4
        build_dataset_v4()

    df = pd.read_csv(V4_DATASET_PATH)
    df["season"] = df["season"].astype(int)

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df  = df[df["season"] == HOLDOUT_SEASON]

    print(f"Train: {len(train_df)} rows  |  Test (2526): {len(test_df)} rows\n")

    configs = [
        ("A. Base 24 (V3 production)",   FEATURES_BASE),
        ("B. Base + opp_ppg (1A)",        FEATURES_OPP),
        ("C. EWM replaces momentum (1B)", FEATURES_EWM),
        ("D. EWM + opp_ppg (1A+1B)",      FEATURES_COMBO),
    ]

    results = []
    base_acc, base_ll = None, None
    for label, feats in configs:
        acc, ll = fit_eval(
            train_df[feats].to_numpy(), np.array(train_df["result"]),
            test_df[feats].to_numpy(),  np.array(test_df["result"]),
            label,
        )
        if base_acc is None:
            base_acc, base_ll = acc, ll
        results.append((label, acc, ll, acc - base_acc, ll - base_ll))

    print("\n" + "="*70)
    print("SUMMARY vs V3 baseline (A)")
    print("="*70)
    for label, acc, ll, d_acc, d_ll in results:
        sign_acc = "+" if d_acc >= 0 else ""
        sign_ll  = "+" if d_ll  >= 0 else ""
        print(f"  {label:<40}  Δacc={sign_acc}{d_acc:+.3f}  Δll={sign_ll}{d_ll:+.4f}")


if __name__ == "__main__":
    run()
