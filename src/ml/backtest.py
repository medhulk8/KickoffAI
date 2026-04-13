"""
Backtesting framework for KickoffAI ML models.

Two expanding-window folds:
  Fold 1: Train 2021-22  → Test 2022-23
  Fold 2: Train 2021-22 + 2022-23 → Test 2023-24

Metrics: accuracy, log loss, Brier, per-class precision/recall/F1,
         draw recall, calibration summary, high-confidence accuracy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"

FEATURE_COLS = [
    "bm_home_prob", "bm_draw_prob", "bm_away_prob",
    "home_weighted_ppg", "away_weighted_ppg",
    "home_weighted_goals", "away_weighted_goals",
    "home_def_solidity", "away_def_solidity",
    "draw_likelihood", "h2h_draw_rate",
    "home_draw_rate", "away_draw_rate",
]
LABEL_COL = "result"
CLASSES = ["A", "D", "H"]  # alphabetical — sklearn default


# ============================================================================
# Fold definitions
# ============================================================================

FOLDS = [
    {
        "name": "Fold 1",
        "train_seasons": ["2122"],
        "test_seasons":  ["2223"],
    },
    {
        "name": "Fold 2",
        "train_seasons": ["2122", "2223"],
        "test_seasons":  ["2324"],
    },
    {
        "name": "Fold 3",
        "train_seasons": ["2122", "2223", "2324"],
        "test_seasons":  ["2425"],
    },
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["season"] = df["season"].astype(str)
    return df


def get_fold_data(df: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(fold["train_seasons"])].copy()
    test  = df[df["season"].isin(fold["test_seasons"])].copy()
    return train, test


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred_labels: np.ndarray,
    y_pred_proba: np.ndarray,
    name: str = "",
    high_conf_threshold: float = 0.55,
) -> dict:
    """
    Compute full metric suite.

    y_true: array of 'H'/'D'/'A' strings
    y_pred_labels: predicted class strings
    y_pred_proba: (n, 3) probability array — columns in CLASSES order [A, D, H]
    """
    # Encode labels to indices matching CLASSES order
    le = LabelEncoder()
    le.fit(CLASSES)
    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred_labels)

    # --- Accuracy ---
    acc = accuracy_score(y_true_enc, y_pred_enc)

    # --- Log loss ---
    ll = log_loss(y_true_enc, y_pred_proba, labels=list(range(len(CLASSES))))

    # --- Brier score (multi-class: mean over outcomes) ---
    n = len(y_true)
    brier = 0.0
    for i, cls in enumerate(CLASSES):
        actual_binary = (y_true_enc == i).astype(float)
        brier += brier_score_loss(actual_binary, y_pred_proba[:, i])
    brier /= len(CLASSES)

    # --- Per-class metrics ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, labels=[0, 1, 2], zero_division=0
    )
    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(precision[i], 3),
            "recall":    round(recall[i], 3),
            "f1":        round(f1[i], 3),
            "support":   int(support[i]),
        }

    # --- High-confidence accuracy ---
    max_prob = y_pred_proba.max(axis=1)
    high_conf_mask = max_prob >= high_conf_threshold
    if high_conf_mask.sum() > 0:
        hc_acc = accuracy_score(
            y_true_enc[high_conf_mask],
            y_pred_enc[high_conf_mask]
        )
        hc_n = int(high_conf_mask.sum())
    else:
        hc_acc = None
        hc_n = 0

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=[0, 1, 2])

    return {
        "name": name,
        "n": n,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "per_class": per_class,
        "draw_recall": round(recall[1], 3),  # index 1 = D
        "high_conf_accuracy": round(hc_acc, 3) if hc_acc is not None else None,
        "high_conf_n": hc_n,
        "confusion_matrix": cm,
    }


def print_metrics(m: dict):
    print(f"\n{'='*55}")
    print(f"  {m['name']}  (n={m['n']})")
    print(f"{'='*55}")
    print(f"  Accuracy:        {m['accuracy']:.1%}")
    print(f"  Log Loss:        {m['log_loss']:.4f}")
    print(f"  Brier Score:     {m['brier']:.4f}")
    print(f"  Draw Recall:     {m['draw_recall']:.1%}")
    if m["high_conf_n"] > 0:
        print(f"  High-Conf Acc:   {m['high_conf_accuracy']:.1%}  (n={m['high_conf_n']})")
    print(f"\n  Per-class:")
    for cls in CLASSES:
        pc = m["per_class"][cls]
        print(f"    {cls}: P={pc['precision']:.2f}  R={pc['recall']:.2f}  F1={pc['f1']:.2f}  n={pc['support']}")
    print(f"\n  Confusion matrix (rows=actual, cols=pred)  [A, D, H]:")
    for row in m["confusion_matrix"]:
        print(f"    {row}")


# ============================================================================
# Baselines
# ============================================================================

class BookmakerBaseline:
    """Predict the outcome with highest bookmaker probability."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = ["bm_away_prob", "bm_draw_prob", "bm_home_prob"]
        idx = X[cols].values.argmax(axis=1)
        return np.array(CLASSES)[idx]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[["bm_away_prob", "bm_draw_prob", "bm_home_prob"]].values
        return p / p.sum(axis=1, keepdims=True)


class HeuristicDrawBaseline:
    """
    Bookmaker baseline but overrides to Draw when draw_likelihood >= threshold.
    Probabilities from bookmaker, label overridden.
    """

    def __init__(self, draw_threshold: float = 0.7):
        self.draw_threshold = draw_threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        bm = BookmakerBaseline()
        labels = bm.predict(X).copy()
        draw_mask = X["draw_likelihood"].values >= self.draw_threshold
        labels[draw_mask] = "D"
        return labels

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[["bm_away_prob", "bm_draw_prob", "bm_home_prob"]].values
        return p / p.sum(axis=1, keepdims=True)


class HeuristicDrawProbBaseline:
    """
    Probability-adjusted heuristic: when draw_likelihood >= threshold,
    boost draw probability to max(bm_draw, draw_likelihood * 0.5) and
    renormalize. Label is argmax of adjusted probabilities.
    """

    def __init__(self, draw_threshold: float = 0.7):
        self.draw_threshold = draw_threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[["bm_away_prob", "bm_draw_prob", "bm_home_prob"]].values.copy()
        p = p / p.sum(axis=1, keepdims=True)
        dl = X["draw_likelihood"].values
        draw_mask = dl >= self.draw_threshold
        boosted = np.maximum(p[draw_mask, 1], dl[draw_mask] * 0.5)
        boosted = np.minimum(boosted, 0.55)
        non_draw = p[draw_mask, 0] + p[draw_mask, 2]
        scale = np.where(non_draw > 0, (1 - boosted) / non_draw, 0.5)
        p[draw_mask, 0] *= scale
        p[draw_mask, 2] *= scale
        p[draw_mask, 1] = boosted
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(CLASSES)[self.predict_proba(X).argmax(axis=1)]


# ============================================================================
# Runner
# ============================================================================

def run_backtest(models: dict[str, Any], df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Run all models across both folds.

    models: dict of name → model instance with predict() and predict_proba()
    Returns: dict of model_name → list of metric dicts (one per fold)
    """
    results = {name: [] for name in models}

    for fold in FOLDS:
        train, test = get_fold_data(df, fold)
        X_train = train[FEATURE_COLS]
        y_train = train[LABEL_COL].values
        X_test  = test[FEATURE_COLS]
        y_test  = test[LABEL_COL].values

        print(f"\n{fold['name']}: train={len(train)} | test={len(test)}")

        for model_name, model in models.items():
            # Fit if model has fit()
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)

            preds  = model.predict(X_test)
            probas = model.predict_proba(X_test)

            m = compute_metrics(
                y_test, preds, probas,
                name=f"{model_name} — {fold['name']}"
            )
            results[model_name].append(m)
            print_metrics(m)

    return results


# ============================================================================
# Confidence threshold sweep
# ============================================================================

def confidence_sweep(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: list[float] = [0.50, 0.55, 0.60, 0.65, 0.70],
) -> None:
    """Print accuracy and coverage at each confidence threshold."""
    le = LabelEncoder()
    le.fit(CLASSES)
    y_true_enc = le.transform(y_true)
    y_pred_labels = np.array(CLASSES)[y_pred_proba.argmax(axis=1)]
    y_pred_enc = le.transform(y_pred_labels)
    max_prob = y_pred_proba.max(axis=1)
    n = len(y_true)

    print(f"\n  Confidence threshold sweep:")
    print(f"  {'Threshold':>10}  {'Coverage':>10}  {'Accuracy':>10}")
    for t in thresholds:
        mask = max_prob >= t
        coverage = mask.sum() / n
        if mask.sum() > 0:
            acc = accuracy_score(y_true_enc[mask], y_pred_enc[mask])
            print(f"  {t:>10.2f}  {coverage:>10.1%}  {acc:>10.1%}  (n={mask.sum()})")
        else:
            print(f"  {t:>10.2f}  {coverage:>10.1%}  {'N/A':>10}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import lightgbm as lgb

    df = load_data()
    print(f"Loaded {len(df)} rows. Seasons: {sorted(df['season'].unique())}")

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=0.1,
            max_iter=1000,
            random_state=42,
        )),
    ])

    lgbm_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        max_depth=3,
        num_leaves=7,
        min_child_samples=60,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=300,
        random_state=42,
        verbose=-1,
    )

    models = {
        "Bookmaker Baseline":        BookmakerBaseline(),
        "Heuristic Draw (label)":    HeuristicDrawBaseline(draw_threshold=0.7),
        "Heuristic Draw (prob)":     HeuristicDrawProbBaseline(draw_threshold=0.7),
        "Logistic Regression":       lr_pipeline,
        "LightGBM":                  lgbm_model,
    }

    results = run_backtest(models, df)

    # Confidence sweep on Fold 2 (most recent, most important)
    print("\n\n=== CONFIDENCE SWEEP — Fold 2 ===")
    fold2_train, fold2_test = get_fold_data(df, FOLDS[1])
    X_f2 = fold2_test[FEATURE_COLS]
    y_f2 = fold2_test[LABEL_COL].values

    lr_pipeline.fit(fold2_train[FEATURE_COLS], fold2_train[LABEL_COL].values)
    print("\nLogistic Regression:")
    confidence_sweep(y_f2, lr_pipeline.predict_proba(X_f2))

    lgbm_model.fit(
        fold2_train[FEATURE_COLS], fold2_train[LABEL_COL].values,
        categorical_feature="auto"
    )
    print("\nLightGBM:")
    confidence_sweep(y_f2, lgbm_model.predict_proba(X_f2))
