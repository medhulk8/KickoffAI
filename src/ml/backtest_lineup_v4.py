"""
Phase 1C lineup backtest — cold-start fix and star_absent on top of V4.

Train: [2223, 2324, 2425] (lineup data covers these seasons)
Test:  2526 holdout

Ablation:
  A. V4 base 26 features (no lineup)
  B. V4 + xi_strength_diff (original season-to-date)
  C. V4 + xi_strength_diff_l10 (rolling-10, cold-start fix)
  D. V4 + star_absent_diff (home_star_absent - away_star_absent)
  E. V4 + xi_strength_diff_l10 + star_absent_diff (combined)
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

from src.ml.build_dataset_v4 import FEATURES_COMBO, OUTPUT_PATH as V4_DATASET_PATH

LINEUP_PATH    = PROJECT_ROOT / "data" / "processed" / "lineup_features.csv"
TRAIN_SEASONS  = [2223, 2324, 2425]
HOLDOUT_SEASON = 2526
CLASSES        = ["A", "D", "H"]


def fit_eval(X_train, y_train, X_test, y_test, label: str, base_acc=None, base_ll=None) -> tuple[float, float]:
    model = CalibratedClassifierCV(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42))]),
        cv=5, method="sigmoid",
    )
    model.fit(X_train, y_train)

    p_raw   = model.predict_proba(X_test)
    classes = list(model.classes_)
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(classes):
        p[:, CLASSES.index(c)] = p_raw[:, i]

    y_pred = [CLASSES[i] for i in np.argmax(p, axis=1)]
    acc    = accuracy_score(y_test, y_pred)
    ll     = log_loss(y_test, p, labels=CLASSES)
    dr     = float((np.array(y_pred)[np.array(y_test) == "D"] == "D").mean()) \
             if (np.array(y_test) == "D").sum() > 0 else 0.0

    d_acc = f"  Δacc={acc - base_acc:+.3f}  Δll={ll - base_ll:+.4f}" if base_acc is not None else ""
    print(f"  {label:<45}  acc={acc:.3f}  ll={ll:.4f}  draw_recall={dr:.1%}  n_feat={X_train.shape[1]}{d_acc}")
    return acc, ll


def run():
    base_df = pd.read_csv(V4_DATASET_PATH)
    base_df["season"] = base_df["season"].astype(int)

    lu = pd.read_csv(LINEUP_PATH)
    lu["date"] = lu["date"].str[:10]
    base_df["date"] = base_df["date"].str[:10]

    # Derive star_absent_diff = home_star_absent - away_star_absent
    lu["star_absent_diff"] = lu["home_star_absent"] - lu["away_star_absent"]

    merged = base_df.merge(
        lu[["date", "home_team", "away_team",
            "xi_strength_diff", "xi_strength_diff_l10", "star_absent_diff"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Restrict to seasons with lineup data
    exp = merged[merged["season"].isin(TRAIN_SEASONS + [HOLDOUT_SEASON])].copy()

    n_with = exp["xi_strength_diff"].notna().sum()
    print(f"Rows with lineup data: {n_with}/{len(exp)}")
    print(f"Per season:\n{exp.groupby('season')['xi_strength_diff'].apply(lambda x: x.notna().sum())}\n")

    # Drop rows missing ANY lineup feature (fair comparison for B/C/D/E)
    lineup_cols = ["xi_strength_diff", "xi_strength_diff_l10", "star_absent_diff"]
    exp_lu = exp.dropna(subset=lineup_cols)

    # Full data (A: no lineup constraint)
    train_all = exp[exp["season"].isin(TRAIN_SEASONS)]
    test_all  = exp[exp["season"] == HOLDOUT_SEASON]

    # Lineup-matched data (B/C/D/E)
    train_lu  = exp_lu[exp_lu["season"].isin(TRAIN_SEASONS)]
    test_lu   = exp_lu[exp_lu["season"] == HOLDOUT_SEASON]

    print(f"Train all: {len(train_all)}  Train lineup-matched: {len(train_lu)}")
    print(f"Test all:  {len(test_all)}   Test lineup-matched:  {len(test_lu)}\n")

    print("─" * 80)

    # A: V4 base, no lineup (full data)
    base_acc, base_ll = fit_eval(
        train_all[FEATURES_COMBO].to_numpy(), np.array(train_all["result"]),
        test_all[FEATURES_COMBO].to_numpy(),  np.array(test_all["result"]),
        "A. V4 26 features (no lineup)",
    )

    # B: V4 + xi_strength_diff (season-to-date, original)
    fit_eval(
        train_lu[FEATURES_COMBO + ["xi_strength_diff"]].to_numpy(), np.array(train_lu["result"]),
        test_lu[FEATURES_COMBO  + ["xi_strength_diff"]].to_numpy(), np.array(test_lu["result"]),
        "B. V4 + xi_strength_diff (s2d)",
        base_acc, base_ll,
    )

    # C: V4 + xi_strength_diff_l10 (rolling-10, cold-start fix)
    fit_eval(
        train_lu[FEATURES_COMBO + ["xi_strength_diff_l10"]].to_numpy(), np.array(train_lu["result"]),
        test_lu[FEATURES_COMBO  + ["xi_strength_diff_l10"]].to_numpy(), np.array(test_lu["result"]),
        "C. V4 + xi_strength_diff_l10 (rolling)",
        base_acc, base_ll,
    )

    # D: V4 + star_absent_diff
    fit_eval(
        train_lu[FEATURES_COMBO + ["star_absent_diff"]].to_numpy(), np.array(train_lu["result"]),
        test_lu[FEATURES_COMBO  + ["star_absent_diff"]].to_numpy(), np.array(test_lu["result"]),
        "D. V4 + star_absent_diff",
        base_acc, base_ll,
    )

    # E: V4 + xi_strength_diff_l10 + star_absent_diff
    fit_eval(
        train_lu[FEATURES_COMBO + ["xi_strength_diff_l10", "star_absent_diff"]].to_numpy(), np.array(train_lu["result"]),
        test_lu[FEATURES_COMBO  + ["xi_strength_diff_l10", "star_absent_diff"]].to_numpy(), np.array(test_lu["result"]),
        "E. V4 + xi_l10 + star_absent",
        base_acc, base_ll,
    )

    # B-fair: V4 base on same lineup-matched subset (controls for data subset difference)
    fit_eval(
        train_lu[FEATURES_COMBO].to_numpy(), np.array(train_lu["result"]),
        test_lu[FEATURES_COMBO].to_numpy(),  np.array(test_lu["result"]),
        "A-fair. V4 base (lineup-matched rows only)",
        base_acc, base_ll,
    )

    # Also check feature distributions to sanity-check cold-start fix
    print("\n── xi_strength_diff stats (test set) ──")
    print(test_lu[["xi_strength_diff", "xi_strength_diff_l10"]].describe().round(4))


if __name__ == "__main__":
    run()
