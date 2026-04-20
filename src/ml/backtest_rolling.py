"""
Rolling-origin backtest + recency-aware training experiment.

Rolling origins (each test season is truly unseen):
  Train 1718-2223 → Test 2324
  Train 1718-2324 → Test 2425
  Train 1718-2425 → Test 2526

For each origin, compare:
  A. Full history (all seasons up to cutoff)
  B. Last 5 seasons
  C. Last 3 seasons
  D. Exponential time weights (λ=0.5, half-life ≈ 2 seasons)
  E. Exponential time weights (λ=1.0, half-life ≈ 1 season)
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

ALL_SEASONS = [1718, 1819, 1920, 2122, 2223, 2324, 2425, 2526]

# Rolling origin test seasons (each uses all prior seasons as train)
ORIGINS = [
    (2324, "Train →2223, Test 2324"),
    (2425, "Train →2324, Test 2425"),
    (2526, "Train →2425, Test 2526"),
]


def season_weight(season: int, test_season: int, lam: float) -> float:
    """Exponential decay: seasons further from test get lower weight."""
    age = ALL_SEASONS.index(test_season) - ALL_SEASONS.index(season)
    return float(np.exp(-lam * age))


def fit_eval(X_train, y_train, X_test, y_test, label: str, weights=None):
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf = CalibratedClassifierCV(lr, cv=5, method="sigmoid")
    clf.fit(X_train, y_train, sample_weight=weights)

    p_raw = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    y_pred = [classes[i] for i in np.argmax(p_raw, axis=1)]

    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, p_raw, labels=classes)
    print(f"    {label:<40}  acc={acc:.3f}  ll={ll:.4f}  n_train={len(y_train)}")
    return acc, ll


def run():
    df = pd.read_csv(DATASET_PATH)
    df["season"] = df["season"].astype(int)

    results = []

    for test_season, origin_label in ORIGINS:
        train_all = df[df["season"] < test_season]
        test_df   = df[df["season"] == test_season]

        if len(test_df) == 0:
            print(f"\n{origin_label} — no test data, skipping")
            continue

        X_test  = test_df[FEATURE_COLS].values.astype(float)
        y_test  = np.array(test_df["result"])

        print(f"\n{origin_label}  (n_test={len(test_df)})")
        print(f"  Result dist: {pd.Series(y_test).value_counts(normalize=True).round(3).to_dict()}")

        train_seasons = sorted(train_all["season"].unique())

        # A: Full history
        X_a = train_all[FEATURE_COLS].values.astype(float)
        y_a = np.array(train_all["result"])
        a_acc, a_ll = fit_eval(X_a, y_a, X_test, y_test, f"A. Full ({len(train_seasons)} seasons)")
        results.append((origin_label, "A_full", a_acc, a_ll))

        # B: Last 5 seasons
        last5 = sorted(train_seasons)[-5:]
        tr5 = train_all[train_all["season"].isin(last5)]
        X_b = tr5[FEATURE_COLS].values.astype(float)
        y_b = np.array(tr5["result"])
        b_acc, b_ll = fit_eval(X_b, y_b, X_test, y_test, f"B. Last 5 {last5}")
        results.append((origin_label, "B_last5", b_acc, b_ll))

        # C: Last 3 seasons
        last3 = sorted(train_seasons)[-3:]
        tr3 = train_all[train_all["season"].isin(last3)]
        X_c = tr3[FEATURE_COLS].values.astype(float)
        y_c = np.array(tr3["result"])
        c_acc, c_ll = fit_eval(X_c, y_c, X_test, y_test, f"C. Last 3 {last3}")
        results.append((origin_label, "C_last3", c_acc, c_ll))

        # D: Exponential weights λ=0.5
        w05 = np.array([season_weight(int(s), test_season, 0.5)
                        for s in train_all["season"]])
        d_acc, d_ll = fit_eval(X_a, y_a, X_test, y_test, "D. Exp weights λ=0.5", weights=w05)
        results.append((origin_label, "D_exp05", d_acc, d_ll))

        # E: Exponential weights λ=1.0
        w10 = np.array([season_weight(int(s), test_season, 1.0)
                        for s in train_all["season"]])
        e_acc, e_ll = fit_eval(X_a, y_a, X_test, y_test, "E. Exp weights λ=1.0", weights=w10)
        results.append((origin_label, "E_exp10", e_acc, e_ll))

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY — avg acc and avg ll across all 3 origins")
    print("="*70)
    res_df = pd.DataFrame(results, columns=["origin", "config", "acc", "ll"])
    summary = res_df.groupby("config")[["acc", "ll"]].mean().round(4)
    summary["acc_rank"] = summary["acc"].rank(ascending=False).astype(int)
    summary["ll_rank"]  = summary["ll"].rank(ascending=True).astype(int)
    print(summary.sort_values("acc", ascending=False).to_string())


if __name__ == "__main__":
    run()
