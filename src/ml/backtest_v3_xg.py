"""
Backtest V3 + xG — compares base V3 model against xG-augmented model.

Run after data/processed/training_dataset_v3_xg.csv exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_V3    = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"
DATASET_V3_XG = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3_xg.csv"

HOLDOUT = 2425
TRAIN_SEASONS = [1718, 1819, 1920, 2122, 2223, 2324]
CLASSES = ["A", "D", "H"]

BASE_FEATURES = [
    "home_sot_l5", "home_sot_conceded_l5", "home_conversion",
    "home_clean_sheet_l5", "home_pts_momentum", "home_goals_momentum", "home_days_rest",
    "away_sot_l5", "away_sot_conceded_l5", "away_conversion",
    "away_clean_sheet_l5", "away_pts_momentum", "away_goals_momentum", "away_days_rest",
    "elo_diff", "home_rank", "away_rank", "matchweek",
]

XG_FEATURES = BASE_FEATURES + [
    "home_xg_scored_l5", "home_xg_conceded_l5",
    "away_xg_scored_l5", "away_xg_conceded_l5",
]


def multiclass_brier(y_enc, y_prob, n=3):
    return float(np.mean(np.sum((y_prob - np.eye(n)[y_enc]) ** 2, axis=1)))

def draw_recall(y_true, y_pred):
    mask = y_true == "D"
    return float((y_pred[mask] == "D").mean()) if mask.sum() > 0 else 0.0

def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42)),
    ])

def eval_model(name, model, X_test, y_test, le):
    p_raw = model.predict_proba(X_test)
    p = np.zeros((len(p_raw), 3))
    for i, c in enumerate(model.named_steps["clf"].classes_):
        p[:, le.transform([c])[0]] = p_raw[:, i]
    y_pred = le.inverse_transform(p.argmax(axis=1))
    print(f"  {name:<40}  acc={accuracy_score(y_test, y_pred):>5.1%}  "
          f"ll={log_loss(y_test, p, labels=le.classes_):.4f}  "
          f"brier={multiclass_brier(le.transform(y_test), p):.4f}  "
          f"draw_recall={draw_recall(y_test, y_pred):.1%}")


def run():
    if not DATASET_V3_XG.exists():
        print(f"xG dataset not found at {DATASET_V3_XG}")
        print("Run: python3 src/ml/add_xg_features.py")
        return

    le = LabelEncoder(); le.fit(CLASSES)

    df_base = pd.read_csv(DATASET_V3)
    df_xg   = pd.read_csv(DATASET_V3_XG)

    train_b = df_base[df_base["season"].isin(TRAIN_SEASONS)]
    test_b  = df_base[df_base["season"] == HOLDOUT]
    train_x = df_xg[df_xg["season"].isin(TRAIN_SEASONS)]
    test_x  = df_xg[df_xg["season"] == HOLDOUT]

    print(f"Base train n={len(train_b)}, xG train n={len(train_x)}")
    print(f"Holdout n={len(test_b)} (base), n={len(test_x)} (xG)\n")
    print(f"{'Model':<40}  {'Acc':>5}  {'LL':>8}  {'Brier':>7}  {'DrawRecall':>10}")
    print("  " + "-"*72)

    lr_base = make_lr()
    lr_base.fit(train_b[BASE_FEATURES].values, train_b["result"].values)
    eval_model("LR — base features (no xG)", lr_base, test_b[BASE_FEATURES].values, test_b["result"].values, le)

    lr_xg = make_lr()
    lr_xg.fit(train_x[XG_FEATURES].values, train_x["result"].values)
    eval_model("LR — base + xG features", lr_xg, test_x[XG_FEATURES].values, test_x["result"].values, le)

    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, max_depth=3, num_leaves=8,
            min_child_samples=75, learning_rate=0.05, n_estimators=300,
            reg_alpha=1.0, reg_lambda=1.0, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=5, verbose=-1,
            class_weight="balanced", random_state=42,
        )
        lgbm.fit(train_x[XG_FEATURES].values, train_x["result"].values)
        p_raw = lgbm.predict_proba(test_x[XG_FEATURES].values)
        p = np.zeros((len(p_raw), 3))
        for i, c in enumerate(lgbm.classes_):
            p[:, le.transform([c])[0]] = p_raw[:, i]
        y_pred = le.inverse_transform(p.argmax(axis=1))
        y_test = test_x["result"].values
        print(f"  {'LightGBM — base + xG':<40}  acc={accuracy_score(y_test, y_pred):>5.1%}  "
              f"ll={log_loss(y_test, p, labels=le.classes_):.4f}  "
              f"brier={multiclass_brier(le.transform(y_test), p):.4f}  "
              f"draw_recall={draw_recall(y_test, y_pred):.1%}")
    except ImportError:
        print("  LightGBM not available")

    print("\nxG feature importance (LR):")
    coefs = lr_xg.named_steps["clf"].coef_
    mean_abs = np.abs(coefs).mean(axis=0)
    for feat, imp in sorted(zip(XG_FEATURES, mean_abs), key=lambda x: -x[1])[:10]:
        print(f"  {feat:<40}  {imp:.4f}")


if __name__ == "__main__":
    run()
