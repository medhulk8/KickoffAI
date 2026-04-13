"""
ML inference wrapper for KickoffAI champion model.

Loads the saved LR model and metadata, validates features at inference time,
and returns probabilities + prediction + confidence.

Used to replace the LLM predictor node in the LangGraph workflow.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "lr_champion.pkl"
META_PATH  = MODELS_DIR / "model_metadata.json"


class MLPredictor:
    """
    Loads the champion ML model and runs inference with feature validation.

    Usage:
        predictor = MLPredictor()
        result = predictor.predict(features_dict)
    """

    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(META_PATH) as f:
            self._meta = json.load(f)
        self._features = self._meta["features"]  # ordered list
        self._classes  = self._meta["classes"]   # [A, D, H]
        self._conf_threshold = self._meta["confidence_threshold"]

    def predict(self, features: dict) -> dict:
        """
        Run inference on a single match.

        Args:
            features: dict with at minimum all keys in self._features.
                      Extra keys are ignored.

        Returns:
            {
              "home_prob": float,
              "draw_prob": float,
              "away_prob": float,
              "prediction": "H" | "D" | "A",
              "confidence": "high" | "medium" | "low",
              "max_prob": float,
              "model_version": str,
            }
        """
        self._validate(features)

        # Build input row in exact feature order (DataFrame preserves names)
        import pandas as pd
        x = pd.DataFrame([[features[f] for f in self._features]], columns=self._features)

        probas = self._model.predict_proba(x)[0]

        # Map model output (alphabetical) to named probs
        le_classes = self._model.named_steps["clf"].classes_
        class_idx = {c: i for i, c in enumerate(le_classes)}

        home_prob = float(probas[class_idx["H"]])
        draw_prob = float(probas[class_idx["D"]])
        away_prob = float(probas[class_idx["A"]])

        # Normalize (safety)
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        max_prob = max(home_prob, draw_prob, away_prob)
        prediction = max({"H": home_prob, "D": draw_prob, "A": away_prob}, key=lambda k: {"H": home_prob, "D": draw_prob, "A": away_prob}[k])

        if max_prob >= self._conf_threshold:
            confidence = "high"
        elif max_prob >= 0.45:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "home_prob":     round(home_prob, 4),
            "draw_prob":     round(draw_prob, 4),
            "away_prob":     round(away_prob, 4),
            "prediction":    prediction,
            "confidence":    confidence,
            "max_prob":      round(max_prob, 4),
            "model_version": self._meta.get("feature_version", "v1"),
            "input_features": {f: features[f] for f in self._features},
        }

    def _validate(self, features: dict):
        missing = [f for f in self._features if f not in features]
        if missing:
            raise ValueError(f"Missing features for ML inference: {missing}")

        for f in self._features:
            v = features[f]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raise ValueError(f"Feature '{f}' is None or NaN")

    @property
    def feature_names(self) -> list[str]:
        return list(self._features)

    @property
    def metadata(self) -> dict:
        return dict(self._meta)


# ============================================================================
# Quick smoke test
# ============================================================================

if __name__ == "__main__":
    predictor = MLPredictor()
    print(f"Model loaded. Features: {predictor.feature_names}")
    print(f"Confidence threshold: {predictor.metadata['confidence_threshold']}")

    # Simulate a close match
    test_features = {
        "bm_home_prob": 0.38,
        "bm_draw_prob":  0.30,
        "bm_away_prob":  0.32,
        "h2h_draw_rate": 0.40,
        "home_weighted_ppg":   1.5,
        "away_weighted_ppg":   1.4,
        "home_weighted_goals": 1.2,
        "away_weighted_goals": 1.1,
    }

    result = predictor.predict(test_features)
    print(f"\nTest prediction (close match):")
    print(f"  Home: {result['home_prob']:.1%}  Draw: {result['draw_prob']:.1%}  Away: {result['away_prob']:.1%}")
    print(f"  Prediction: {result['prediction']}  |  Confidence: {result['confidence']}  |  Max prob: {result['max_prob']:.1%}")

    # Simulate a clear favourite
    test_features2 = {
        "bm_home_prob": 0.78,
        "bm_draw_prob":  0.14,
        "bm_away_prob":  0.08,
        "h2h_draw_rate": 0.10,
        "home_weighted_ppg":   2.4,
        "away_weighted_ppg":   0.8,
        "home_weighted_goals": 2.5,
        "away_weighted_goals": 0.9,
    }

    result2 = predictor.predict(test_features2)
    print(f"\nTest prediction (clear favourite):")
    print(f"  Home: {result2['home_prob']:.1%}  Draw: {result2['draw_prob']:.1%}  Away: {result2['away_prob']:.1%}")
    print(f"  Prediction: {result2['prediction']}  |  Confidence: {result2['confidence']}  |  Max prob: {result2['max_prob']:.1%}")
