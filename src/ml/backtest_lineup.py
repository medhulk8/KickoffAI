"""
Lineup features experiment — clean train/test comparison.

Train: 2022-23 + 2023-24 (seasons where lineup data exists + overlap with base)
Test:  2024-25

Comparison:
  A. Base 24 features only
  B. Base + 4 lineup features (home/away xi_strength, home/away missing_regulars)
  C. Base + xi_strength_diff only (most parsimonious)

Lineup source: data/processed/lineup_features.csv (FPL GitHub GW CSVs)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"
LINEUP_PATH  = PROJECT_ROOT / "data" / "processed" / "lineup_features.csv"

TRAIN_SEASONS = [2223, 2324, 2425]
TEST_SEASON   = 2526

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

LINEUP_FEATURES_FULL = [
    "home_xi_strength", "home_missing_regulars",
    "away_xi_strength", "away_missing_regulars",
]

LINEUP_FEATURES_DIFF = ["xi_strength_diff"]


def fit_eval(X_train, y_train, X_test, y_test, label: str):
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf = CalibratedClassifierCV(lr, cv=5, method="sigmoid")
    clf.fit(X_train, y_train)

    p_raw = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    idx_H = classes.index("H")
    idx_D = classes.index("D")
    idx_A = classes.index("A")

    p_mat = np.column_stack([p_raw[:, idx_H], p_raw[:, idx_D], p_raw[:, idx_A]])
    y_pred = np.array(["H", "D", "A"])[np.argmax(p_mat, axis=1)]

    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, p_raw, labels=classes)
    dr  = float((y_pred[y_test == "D"] == "D").mean()) if (y_test == "D").sum() else 0.0

    print(f"  {label:<30}  acc={acc:.3f}  ll={ll:.4f}  draw_recall={dr:.1%}  n_train={len(y_train)}  n_test={len(y_test)}")
    return clf


def run():
    base = pd.read_csv(DATASET_PATH)
    lu   = pd.read_csv(LINEUP_PATH)
    lu["date"] = lu["date"].str[:10]
    base["date"] = base["date"].str[:10]

    merged = base.merge(
        lu[["date", "home_team", "away_team"] + LINEUP_FEATURES_FULL + LINEUP_FEATURES_DIFF],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Restrict to seasons where lineup data exists
    exp = merged[merged["season"].isin(TRAIN_SEASONS + [TEST_SEASON])].copy()

    n_with_lu = exp["home_xi_strength"].notna().sum()
    print(f"Rows with lineup data: {n_with_lu}/{len(exp)}")
    print(f"Per season:\n{exp.groupby('season')['home_xi_strength'].apply(lambda x: x.notna().sum())}\n")

    # Drop rows missing lineup features (needed for fair B vs C comparison)
    exp_lu = exp.dropna(subset=LINEUP_FEATURES_FULL + LINEUP_FEATURES_DIFF)

    train_all = exp[exp["season"].isin(TRAIN_SEASONS)]
    test_all  = exp[exp["season"] == TEST_SEASON]

    train_lu  = exp_lu[exp_lu["season"].isin(TRAIN_SEASONS)]
    test_lu   = exp_lu[exp_lu["season"] == TEST_SEASON]

    y_train_all = np.array(train_all["result"])
    y_test_all  = np.array(test_all["result"])
    y_train_lu  = np.array(train_lu["result"])
    y_test_lu   = np.array(test_lu["result"])

    print("=== Restricted to 2223+2324 train / 2425 test ===\n")

    # A: Base only (full data, no lineup constraint)
    fit_eval(
        train_all[BASE_FEATURES].values, y_train_all,
        test_all[BASE_FEATURES].values,  y_test_all,
        "A. Base only (all rows)",
    )

    # B: Base only (same subset as lineup-matched rows, for fair comparison)
    fit_eval(
        train_lu[BASE_FEATURES].values, y_train_lu,
        test_lu[BASE_FEATURES].values,  y_test_lu,
        "B. Base only (lineup-matched rows)",
    )

    # C: Base + 4 lineup features
    fit_eval(
        train_lu[BASE_FEATURES + LINEUP_FEATURES_FULL].values, y_train_lu,
        test_lu[BASE_FEATURES + LINEUP_FEATURES_FULL].values,  y_test_lu,
        "C. Base + 4 lineup features",
    )

    # D: Base + xi_strength_diff only
    fit_eval(
        train_lu[BASE_FEATURES + LINEUP_FEATURES_DIFF].values, y_train_lu,
        test_lu[BASE_FEATURES + LINEUP_FEATURES_DIFF].values,  y_test_lu,
        "D. Base + xi_strength_diff",
    )

    # Feature importance for C
    from sklearn.linear_model import LogisticRegression as LR
    all_feats = BASE_FEATURES + LINEUP_FEATURES_FULL
    lr_raw = LR(C=1.0, max_iter=500, random_state=42)
    lr_raw.fit(train_lu[all_feats].values, y_train_lu)
    coef_H = lr_raw.coef_[list(lr_raw.classes_).index("H")]
    print("\nTop feature importances (abs coef, H class):")
    for feat, c in sorted(zip(all_feats, coef_H), key=lambda x: -abs(x[1]))[:8]:
        print(f"  {feat:<30}  {c:+.4f}")


if __name__ == "__main__":
    run()
