"""
Feature ablation study for KickoffAI LR model.

Tests which feature groups actually contribute signal.
All ablations evaluated on Fold 2 (most important — closest to deployment).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.backtest import (
    load_data, get_fold_data, FOLDS, CLASSES,
    compute_metrics, print_metrics
)

ALL_FEATURES = [
    "bm_home_prob", "bm_draw_prob", "bm_away_prob",
    "home_weighted_ppg", "away_weighted_ppg",
    "home_weighted_goals", "away_weighted_goals",
    "home_def_solidity", "away_def_solidity",
    "draw_likelihood", "h2h_draw_rate",
]

FEATURE_GROUPS = {
    "bookmaker_only": [
        "bm_home_prob", "bm_draw_prob", "bm_away_prob",
    ],
    "bookmaker + form": [
        "bm_home_prob", "bm_draw_prob", "bm_away_prob",
        "home_weighted_ppg", "away_weighted_ppg",
        "home_weighted_goals", "away_weighted_goals",
    ],
    "bookmaker + form + defense": [
        "bm_home_prob", "bm_draw_prob", "bm_away_prob",
        "home_weighted_ppg", "away_weighted_ppg",
        "home_weighted_goals", "away_weighted_goals",
        "home_def_solidity", "away_def_solidity",
    ],
    "all features (full model)": ALL_FEATURES,
    # Drop one at a time from full model
    "drop draw_likelihood": [f for f in ALL_FEATURES if f != "draw_likelihood"],
    "drop h2h_draw_rate":   [f for f in ALL_FEATURES if f != "h2h_draw_rate"],
    "drop def_solidity":    [f for f in ALL_FEATURES if f not in ("home_def_solidity", "away_def_solidity")],
    "drop draw signals":    [f for f in ALL_FEATURES if f not in ("draw_likelihood", "h2h_draw_rate")],
}


def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=0.1, max_iter=1000, random_state=42)),
    ])


def run_ablations():
    df = load_data()
    train, test = get_fold_data(df, FOLDS[1])  # Fold 2 only
    y_test = test["result"].values

    print(f"Fold 2: train={len(train)} | test={len(test)}\n")
    print(f"{'Feature set':<35} {'Acc':>7} {'LL':>8} {'Brier':>7} {'DrawR':>7} {'#feats':>7}")
    print("-" * 75)

    results = []
    for name, features in FEATURE_GROUPS.items():
        model = make_lr()
        model.fit(train[features], train["result"].values)
        preds  = model.predict(test[features])
        probas = model.predict_proba(test[features])

        # Align proba columns to CLASSES order [A, D, H]
        le = model.named_steps["clf"].classes_
        class_order = {c: i for i, c in enumerate(le)}
        aligned = np.zeros((len(probas), 3))
        for i, cls in enumerate(CLASSES):
            aligned[:, i] = probas[:, class_order[cls]]

        m = compute_metrics(y_test, preds, aligned, name=name)

        draw_r = m["per_class"]["D"]["recall"]
        print(f"  {name:<33} {m['accuracy']:>7.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {draw_r:>7.1%} {len(features):>7}")
        results.append({**m, "features": features, "n_features": len(features)})

    print("\nNote: all ablations on Fold 2 (train=2122+2223, test=2324)")
    return results


if __name__ == "__main__":
    run_ablations()
