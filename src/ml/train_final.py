"""
Train and save the champion LR model.

Trains on 2021-22 + 2022-23 (seasons 2122, 2223).
Holds out 2023-24 for integration validation.
After integration is stable, retrain on all 3 seasons for deployment.

Saves:
  models/lr_champion.pkl       — trained sklearn pipeline
  models/model_metadata.json   — feature schema, training info, label map
"""

import pickle
import json
from datetime import date
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml.backtest import load_data, FOLDS, get_fold_data, compute_metrics, print_metrics

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CHAMPION_FEATURES = [
    "bm_home_prob",
    "bm_draw_prob",
    "bm_away_prob",
    "h2h_draw_rate",
    "home_weighted_ppg",
    "away_weighted_ppg",
    "home_weighted_goals",
    "away_weighted_goals",
]

TRAIN_SEASONS = ["2122", "2223", "2324"]  # all 3 seasons — production model
HOLDOUT_SEASON = None  # no holdout for production retrain
CLASSES = ["A", "D", "H"]


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=0.1,
            max_iter=1000,
            random_state=42,
        )),
    ])


def train_and_save():
    df = load_data()

    train_df = df[df["season"].isin(TRAIN_SEASONS)]

    print(f"Training on seasons {TRAIN_SEASONS}: {len(train_df)} matches")
    print("No holdout — production model trained on all available data.")

    X_train = train_df[CHAMPION_FEATURES]
    y_train = train_df["result"].values

    model = build_model()
    model.fit(X_train, y_train)

    # Save model
    model_path = MODELS_DIR / "lr_champion.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "LogisticRegression",
        "sklearn_pipeline": True,
        "feature_version": "v1",
        "features": CHAMPION_FEATURES,   # ordered — must match at inference
        "n_features": len(CHAMPION_FEATURES),
        "classes": CLASSES,              # [A, D, H] — alphabetical
        "train_seasons": TRAIN_SEASONS,
        "holdout_season": None,
        "train_n": len(train_df),
        "training_date": str(date.today()),
        "hyperparams": {"C": 0.1, "solver": "lbfgs", "max_iter": 1000},
        "confidence_threshold": 0.65,
        "holdout_metrics": {
            "accuracy": 0.655,
            "log_loss": 0.770,
            "brier": 0.154,
            "draw_recall": 0.629,
            "note": "From pre-production holdout eval on 2023-24 (n=194)"
        },
        "notes": "Production model. Trained on all 3 seasons. Add season 4 data and retrain from scratch when available."
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    return model, metadata


if __name__ == "__main__":
    train_and_save()
