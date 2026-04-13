"""
Over/Under 2.5 percentile ranker for KickoffAI.

Ranks matches by likelihood of going over 2.5 goals.
Product slices (based on Fold 3 Fold 3 validation):
  Top 10%: 75.7% over rate (n=37, promising but fragile)
  Top 20%: 72.0% over rate (n=75, main product slice)

Design decisions:
  - Ranking-first: percentile buckets, not raw probability thresholds
  - Calibration is weaker than bookmaker; do not treat scores as probabilities
  - Trained on all seasons except 2425 (holdout)
  - Requires: bm_over25_prob + rolling shot/goal features (last 5 matches)

Honest status: ranking signal not yet statistically proven (bootstrap 95% CI
crosses zero at n=379). Top 20% lift observed once on one unseen season.
Validate with more live/unseen data before treating as production-grade.
"""

from __future__ import annotations

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OU_MODEL_PATH = MODELS_DIR / "ou_ranker.pkl"
OU_META_PATH  = MODELS_DIR / "ou_ranker_metadata.json"
DATASET_PATH  = PROJECT_ROOT / "data" / "processed" / "ou_dataset.csv"

FEATURES = [
    "home_goals_scored", "home_goals_conceded",
    "away_goals_scored", "away_goals_conceded",
    "home_sot", "away_sot",
    "home_sot_conceded", "away_sot_conceded",
    "home_conversion", "away_conversion",
    "total_attack", "total_defence_loose", "total_sot",
    "elo_diff",
    "bm_over25_prob",
]

TRAIN_SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]

# Percentile thresholds calibrated on Fold 3 (2425, n=379)
# Top 10% → score >= p90, Top 20% → score >= p80
TOP10_LIFT = 1.33  # 75.7% over rate vs 56.7% base
TOP20_LIFT = 1.27  # 72.0% over rate vs 56.7% base


# ============================================================================
# Training
# ============================================================================

def train_and_save() -> tuple:
    df = pd.read_csv(DATASET_PATH)
    df["season"] = df["season"].astype(str)
    train = df[df["season"].isin(TRAIN_SEASONS)].dropna(subset=["bm_over25_prob"])

    print(f"Training O/U ranker on seasons {TRAIN_SEASONS}")
    print(f"  {len(train)} matches  |  over25 rate: {train['over25'].mean():.1%}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=0.1, max_iter=1000, random_state=42)),
    ])
    model.fit(train[FEATURES], train["over25"].values)

    # Compute training score distribution for percentile anchoring
    train_scores = model.predict_proba(train[FEATURES])[:, 1]
    p80 = float(np.percentile(train_scores, 80))
    p90 = float(np.percentile(train_scores, 90))

    MODELS_DIR.mkdir(exist_ok=True)
    with open(OU_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_type": "LogisticRegression",
        "features": FEATURES,
        "n_features": len(FEATURES),
        "train_seasons": TRAIN_SEASONS,
        "train_n": len(train),
        "score_percentiles": {"p80": round(p80, 4), "p90": round(p90, 4)},
        "product_slices": {
            "top_20pct": {
                "threshold": round(p80, 4),
                "fold3_over_rate": 0.720,
                "fold3_base_rate": 0.567,
                "fold3_lift": TOP20_LIFT,
                "fold3_n": 75,
                "recommendation": "Main product slice",
            },
            "top_10pct": {
                "threshold": round(p90, 4),
                "fold3_over_rate": 0.757,
                "fold3_base_rate": 0.567,
                "fold3_lift": TOP10_LIFT,
                "fold3_n": 37,
                "recommendation": "Experimental elite slice — fragile at n=37",
            },
        },
        "validation": {
            "fold3_auc_model":      0.5797,
            "fold3_auc_bookmaker":  0.5685,
            "bootstrap_auc_diff_mean": +0.011,
            "bootstrap_auc_diff_ci":   [-0.013, 0.035],
            "bootstrap_p_positive":    0.819,
            "calibration": "Model calibration weaker than bookmaker in mid-range bins",
            "honest_status": (
                "Ranking signal not statistically proven (CI crosses zero, n=379). "
                "Top-20% lift observed on one unseen season (2425). "
                "Validate with more live data before treating as production-grade."
            ),
        },
        "notes": (
            "Ranking-first product: use percentile buckets, not raw score thresholds. "
            "Do not present scores as calibrated probabilities. "
            "Reopen after 2025-26 season data available."
        ),
    }

    with open(OU_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved:    {OU_MODEL_PATH}")
    print(f"Metadata saved: {OU_META_PATH}")
    print(f"Score p80={p80:.4f} (top-20% threshold)")
    print(f"Score p90={p90:.4f} (top-10% threshold)")
    return model, metadata


# ============================================================================
# Inference
# ============================================================================

class OUranker:
    """
    Ranks a match by over-2.5 likelihood.

    Usage:
        ranker = OURanker()
        result = ranker.rank(features_dict)
        # result["bucket"] -> "top_10pct" | "top_20pct" | "unranked"
    """

    def __init__(self):
        with open(OU_MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(OU_META_PATH) as f:
            self._meta = json.load(f)
        self._features = self._meta["features"]
        self._p80 = self._meta["score_percentiles"]["p80"]
        self._p90 = self._meta["score_percentiles"]["p90"]

    def rank(self, features: dict) -> dict:
        """
        Score and bucket a single match.

        Args:
            features: dict containing all keys in self.features.

        Returns:
            {
              "score":   float  — raw model probability (not a calibrated prob)
              "bucket":  str    — "top_10pct" | "top_20pct" | "unranked"
              "over_rate_expected": float — observed over rate for this bucket (Fold 3)
            }
        """
        missing = [f for f in self._features if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        x = pd.DataFrame([[features[f] for f in self._features]], columns=self._features)
        score = float(self._model.predict_proba(x)[0, 1])

        if score >= self._p90:
            bucket = "top_10pct"
            expected_rate = self._meta["product_slices"]["top_10pct"]["fold3_over_rate"]
        elif score >= self._p80:
            bucket = "top_20pct"
            expected_rate = self._meta["product_slices"]["top_20pct"]["fold3_over_rate"]
        else:
            bucket = "unranked"
            expected_rate = self._meta["product_slices"]["top_20pct"]["fold3_base_rate"]

        return {
            "score":              round(score, 4),
            "bucket":             bucket,
            "over_rate_expected": expected_rate,
        }

    @property
    def metadata(self) -> dict:
        return dict(self._meta)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    model, meta = train_and_save()

    # Quick smoke test using OURanker
    ranker = OUranker()
    print("\nSmoke test:")

    # High-attack match
    test1 = {f: 0.0 for f in FEATURES}
    test1.update({
        "home_goals_scored": 2.2, "home_goals_conceded": 1.5,
        "away_goals_scored": 2.0, "away_goals_conceded": 1.8,
        "home_sot": 5.5, "away_sot": 5.0,
        "home_sot_conceded": 4.5, "away_sot_conceded": 4.2,
        "home_conversion": 0.40, "away_conversion": 0.38,
        "total_attack": 4.2, "total_defence_loose": 3.3, "total_sot": 10.5,
        "elo_diff": 30.0, "bm_over25_prob": 0.68,
    })
    r = ranker.rank(test1)
    print(f"  High-attack match  → score={r['score']:.3f}  bucket={r['bucket']}")

    # Low-scoring match
    test2 = {f: 0.0 for f in FEATURES}
    test2.update({
        "home_goals_scored": 1.0, "home_goals_conceded": 0.8,
        "away_goals_scored": 0.9, "away_goals_conceded": 0.7,
        "home_sot": 2.8, "away_sot": 2.5,
        "home_sot_conceded": 2.2, "away_sot_conceded": 2.0,
        "home_conversion": 0.33, "away_conversion": 0.30,
        "total_attack": 1.9, "total_defence_loose": 1.5, "total_sot": 5.3,
        "elo_diff": -20.0, "bm_over25_prob": 0.38,
    })
    r = ranker.rank(test2)
    print(f"  Low-scoring match  → score={r['score']:.3f}  bucket={r['bucket']}")
