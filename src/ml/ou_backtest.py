"""
Over/Under 2.5 goals backtest for KickoffAI.

Three models tested across all folds:
  1. LR — core features (rolling goals + SOT + conversion)
  2. LR — core + combined environment features
  3. LightGBM — full feature set

Baseline: majority class (over) and calibrated rate.
Evaluation: accuracy, log loss, Brier, AUC.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ou_dataset.csv"

FOLDS = [
    {"name": "Fold 1", "train_seasons": ["1718", "1819", "1920", "2021", "2122"], "test_seasons": ["2223"]},
    {"name": "Fold 2", "train_seasons": ["1718", "1819", "1920", "2021", "2122", "2223"], "test_seasons": ["2324"]},
    {"name": "Fold 3", "train_seasons": ["1718", "1819", "1920", "2021", "2122", "2223", "2324"], "test_seasons": ["2425"]},
]

CORE_FEATURES = [
    "home_goals_scored", "home_goals_conceded",
    "away_goals_scored", "away_goals_conceded",
    "home_sot", "away_sot",
    "home_sot_conceded", "away_sot_conceded",
    "home_conversion", "away_conversion",
    "elo_diff",
]

COMBINED_FEATURES = CORE_FEATURES + [
    "total_attack",
    "total_defence_loose",
    "total_sot",
]

COMBINED_DRAW_FEATURES = COMBINED_FEATURES + ["avg_draw_prob"]

BM_ONLY_FEATURES = ["bm_over25_prob"]

BM_PLUS_FEATURES = COMBINED_FEATURES + ["bm_over25_prob"]


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=0.1, max_iter=1000, random_state=42)),
    ])


def metrics(y_true, y_prob, name="") -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "name":     name,
        "n":        len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier":    round(brier_score_loss(y_true, y_prob), 4),
        "auc":      round(roc_auc_score(y_true, y_prob), 4),
    }


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["season"] = df["season"].astype(str)
    return df


def get_fold(df, fold):
    train = df[df["season"].isin(fold["train_seasons"])].copy()
    test  = df[df["season"].isin(fold["test_seasons"])].copy()
    return train, test


def run_ou_backtest():
    df = load_data()
    print(f"Dataset: {len(df)} rows  |  over25 rate: {df['over25'].mean():.1%}\n")
    print(f"  {'Model':<35} {'Fold':>6}  {'Acc':>6} {'LL':>8} {'Brier':>7} {'AUC':>7}")
    print("  " + "-" * 70)

    try:
        import lightgbm as lgb
        has_lgbm = True
    except ImportError:
        has_lgbm = False
        print("  (LightGBM not available — skipping)")

    for fold in FOLDS:
        train, test = get_fold(df, fold)
        y_train = train["over25"].values
        y_test  = test["over25"].values
        fname   = fold["name"]

        # Drop rows missing bm_over25_prob for bookmaker tests
        test_bm   = test.dropna(subset=["bm_over25_prob"])
        train_bm  = train.dropna(subset=["bm_over25_prob"])
        y_test_bm = test_bm["over25"].values

        # Majority baseline
        majority_rate = y_train.mean()
        base_prob = np.full(len(y_test), majority_rate)
        m = metrics(y_test, base_prob)
        print(f"  {'Majority baseline':<35} {fname:>6}  {m['accuracy']:>6.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['auc']:>7.4f}")

        # Bookmaker O/U baseline (implied probability)
        p_bm = test_bm["bm_over25_prob"].values
        m = metrics(y_test_bm, p_bm)
        print(f"  {'Bookmaker O/U baseline':<35} {fname:>6}  {m['accuracy']:>6.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['auc']:>7.4f}")

        # LR core features
        lr1 = make_lr()
        lr1.fit(train[CORE_FEATURES], y_train)
        p = lr1.predict_proba(test[CORE_FEATURES])[:, 1]
        m = metrics(y_test, p)
        print(f"  {'LR core features':<35} {fname:>6}  {m['accuracy']:>6.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['auc']:>7.4f}")

        # LR core + combined + draw_prob
        lr2 = make_lr()
        lr2.fit(train[COMBINED_DRAW_FEATURES], y_train)
        p = lr2.predict_proba(test[COMBINED_DRAW_FEATURES])[:, 1]
        m = metrics(y_test, p)
        print(f"  {'LR + combined + draw_prob':<35} {fname:>6}  {m['accuracy']:>6.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['auc']:>7.4f}")

        # LR bookmaker O/U only
        lr_bm = make_lr()
        lr_bm.fit(train_bm[BM_ONLY_FEATURES], train_bm["over25"].values)
        p = lr_bm.predict_proba(test_bm[BM_ONLY_FEATURES])[:, 1]
        m = metrics(y_test_bm, p)
        print(f"  {'LR bookmaker O/U only':<35} {fname:>6}  {m['accuracy']:>6.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['auc']:>7.4f}")

        # LR bookmaker + shot features
        lr_bmp = make_lr()
        lr_bmp.fit(train_bm[BM_PLUS_FEATURES], train_bm["over25"].values)
        p = lr_bmp.predict_proba(test_bm[BM_PLUS_FEATURES])[:, 1]
        m = metrics(y_test_bm, p)
        print(f"  {'LR bm O/U + shot features':<35} {fname:>6}  {m['accuracy']:>6.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['auc']:>7.4f}")

        print()

    # Feature importance from LR on full training set (Folds 1+2+3)
    full_train = df[~df["season"].isin(["2425"])].copy()
    lr_final = make_lr()
    lr_final.fit(full_train[COMBINED_FEATURES], full_train["over25"].values)
    coefs = lr_final.named_steps["clf"].coef_[0]
    print("\nLR feature importance (|coef|, trained on all seasons except 2425):")
    for feat, coef in sorted(zip(COMBINED_FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<30}  {coef:+.4f}")


if __name__ == "__main__":
    run_ou_backtest()
