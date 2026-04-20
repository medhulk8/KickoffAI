"""
Backtest V3 — independent model, no bookmaker odds.

Models compared on 2425 holdout (n=379):
  1. Majority class baseline
  2. Bookmaker baseline (from DB odds — comparison only, not a model feature)
  3. LR — 7 seasons training (1718-2324)
  4. LR — 3 seasons training (2122-2324)
  5. LightGBM — 7 seasons training

Primary metrics: log-loss, Brier score (multiclass)
Secondary: accuracy, draw recall
"""

from __future__ import annotations

import sqlite3
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

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset_v3.csv"
DB_PATH      = PROJECT_ROOT / "data" / "processed" / "asil.db"

HOLDOUT_SEASON = 2425
TRAIN_7 = [1718, 1819, 1920, 2122, 2223, 2324]
TRAIN_3 = [2122, 2223, 2324]

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

CLASSES = ["A", "D", "H"]   # alphabetical — sklearn LabelEncoder order


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def multiclass_brier(y_true_encoded: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    """Multiclass Brier score: mean over matches of sum of squared errors."""
    one_hot = np.eye(n_classes)[y_true_encoded]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def draw_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true == "D"
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] == "D").mean())


def compute_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray,
                    le: LabelEncoder) -> dict:
    y_pred     = le.inverse_transform(y_prob.argmax(axis=1))
    y_enc      = le.transform(y_true)
    acc        = accuracy_score(y_true, y_pred)
    ll         = log_loss(y_true, y_prob, labels=le.classes_)
    brier      = multiclass_brier(y_enc, y_prob)
    dr         = draw_recall(y_true, y_pred)
    return dict(name=name, accuracy=acc, log_loss=ll, brier=brier, draw_recall=dr,
                n=len(y_true))


def print_row(m: dict):
    print(f"  {m['name']:<35}  acc={m['accuracy']:>5.1%}  "
          f"ll={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
          f"draw_recall={m['draw_recall']:.1%}  n={m['n']}")


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=1.0, max_iter=2000, random_state=42,
        )),
    ])


class TwoStageModel:
    """
    Stage A: binary LR — draw vs not-draw.
    Stage B: binary LR — home vs away (trained only on non-draw matches).
    P(H) = P(not_D) * P(H | not_D)
    P(A) = P(not_D) * P(A | not_D)
    P(D) = P(D)
    """

    def __init__(self):
        self.stage_a = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42)),
        ])
        self.stage_b = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42)),
        ])

    def fit(self, X, y):
        # Stage A: draw=1 vs not-draw=0
        y_a = (y == "D").astype(int)
        self.stage_a.fit(X, y_a)

        # Stage B: home vs away, excluding draws
        mask = y != "D"
        X_b = X[mask]
        y_b = y[mask]
        self.stage_b.fit(X_b, y_b)
        self._stage_b_classes = self.stage_b.named_steps["clf"].classes_
        return self

    def predict_proba_hda(self, X) -> np.ndarray:
        """Returns array shape (n, 3) in [A, D, H] order."""
        p_draw = self.stage_a.predict_proba(X)[:, 1]          # P(D)
        p_not_draw = 1 - p_draw

        p_b = self.stage_b.predict_proba(X)                    # P(H|not_D), P(A|not_D)
        classes = list(self._stage_b_classes)
        p_home_given_nd = p_b[:, classes.index("H")]
        p_away_given_nd = p_b[:, classes.index("A")]

        p_home = p_not_draw * p_home_given_nd
        p_away = p_not_draw * p_away_given_nd

        out = np.stack([p_away, p_draw, p_home], axis=1)       # [A, D, H]
        # Normalise to sum to 1 (floating-point safety)
        out = out / out.sum(axis=1, keepdims=True)
        return out


def make_lgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="multiclass", num_class=3,
            max_depth=3, num_leaves=8,
            min_child_samples=75, learning_rate=0.05,
            n_estimators=300,
            reg_alpha=1.0, reg_lambda=1.0,
            feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=5,
            min_gain_to_split=0.01,
            verbose=-1, random_state=42,
            class_weight="balanced",
        )
    except ImportError:
        return None


def bookmaker_baseline(test_match_ids: np.ndarray, le: LabelEncoder):
    """Load bookmaker implied probs from DB for the holdout matches."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("""
        SELECT match_id, avg_home_prob, avg_draw_prob, avg_away_prob
        FROM matches
        WHERE match_id IN ({})
          AND avg_home_prob IS NOT NULL
    """.format(",".join("?" * len(test_match_ids))),
        test_match_ids.tolist()
    ).fetchall()
    conn.close()

    prob_map = {r[0]: (r[1], r[2], r[3]) for r in rows}

    probs, valid_ids = [], []
    for mid in test_match_ids:
        if mid in prob_map:
            h, d, a = prob_map[mid]
            total = h + d + a
            # Map to CLASSES order [A, D, H]
            idx_a = le.transform(["A"])[0]
            idx_d = le.transform(["D"])[0]
            idx_h = le.transform(["H"])[0]
            p = np.zeros(3)
            p[idx_a] = a / total
            p[idx_d] = d / total
            p[idx_h] = h / total
            probs.append(p)
            valid_ids.append(mid)

    return np.array(probs), np.array(valid_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_v3():
    df = pd.read_csv(DATASET_PATH)

    le = LabelEncoder()
    le.fit(CLASSES)

    train_7  = df[df["season"].isin(TRAIN_7)]
    train_3  = df[df["season"].isin(TRAIN_3)]
    test     = df[df["season"] == HOLDOUT_SEASON]

    X_test = test[FEATURE_COLS].values
    y_test = test["result"].values

    print(f"\nTraining seasons (7): {TRAIN_7}  n={len(train_7)}")
    print(f"Training seasons (3): {TRAIN_3}  n={len(train_3)}")
    print(f"Holdout season:       {HOLDOUT_SEASON}   n={len(test)}")
    print(f"Result distribution (holdout): {pd.Series(y_test).value_counts(normalize=True).round(3).to_dict()}")
    print(f"\n{'Model':<35}  {'Acc':>6}  {'LogLoss':>8}  {'Brier':>7}  {'DrawRecall':>10}  {'n':>5}")
    print("  " + "-" * 75)

    results = []

    # ── 1. Majority class baseline ─────────────────────────────────────────
    majority_class = pd.Series(train_7["result"]).mode()[0]
    majority_prob  = np.zeros((len(test), 3))
    majority_prob[:, le.transform([majority_class])[0]] = 1.0
    m = compute_metrics("Majority class (H)", y_test, majority_prob, le)
    print_row(m); results.append(m)

    # ── 2. Bookmaker baseline ──────────────────────────────────────────────
    bm_probs, bm_ids = bookmaker_baseline(test["match_id"].values, le)
    if len(bm_probs) > 0:
        bm_y = test[test["match_id"].isin(bm_ids)]["result"].values
        m = compute_metrics(f"Bookmaker baseline (n={len(bm_ids)})", bm_y, bm_probs, le)
        print_row(m); results.append(m)
    else:
        print("  Bookmaker baseline: no odds found in DB for holdout")

    print()

    # ── 3. LR — 7 seasons ─────────────────────────────────────────────────
    lr7 = make_lr()
    lr7.fit(train_7[FEATURE_COLS].values, train_7["result"].values)
    p = lr7.predict_proba(X_test)
    # Align to CLASSES order
    clf_classes = lr7.named_steps["clf"].classes_
    p_aligned   = np.zeros((len(p), 3))
    for i, c in enumerate(clf_classes):
        p_aligned[:, le.transform([c])[0]] = p[:, i]
    m = compute_metrics("LR — 7 seasons (1718-2324)", y_test, p_aligned, le)
    print_row(m); results.append(m)

    # ── 4. LR — 3 seasons ─────────────────────────────────────────────────
    lr3 = make_lr()
    lr3.fit(train_3[FEATURE_COLS].values, train_3["result"].values)
    p = lr3.predict_proba(X_test)
    clf_classes = lr3.named_steps["clf"].classes_
    p_aligned   = np.zeros((len(p), 3))
    for i, c in enumerate(clf_classes):
        p_aligned[:, le.transform([c])[0]] = p[:, i]
    m = compute_metrics("LR — 3 seasons (2122-2324)", y_test, p_aligned, le)
    print_row(m); results.append(m)

    # ── 5. LightGBM — 7 seasons ───────────────────────────────────────────
    lgbm = make_lgbm()
    if lgbm is not None:
        lgbm.fit(train_7[FEATURE_COLS].values, train_7["result"].values)
        p = lgbm.predict_proba(X_test)
        lgbm_classes = lgbm.classes_
        p_aligned    = np.zeros((len(p), 3))
        for i, c in enumerate(lgbm_classes):
            p_aligned[:, le.transform([c])[0]] = p[:, i]
        m = compute_metrics("LightGBM — 7 seasons", y_test, p_aligned, le)
        print_row(m); results.append(m)
    else:
        print("  LightGBM not available — skipping")

    # ── 6. Two-stage: draw vs not-draw → home vs away ─────────────────────
    ts = TwoStageModel()
    ts.fit(train_7[FEATURE_COLS].values, train_7["result"].values)
    p_aligned = ts.predict_proba_hda(X_test)
    m = compute_metrics("Two-stage (draw|not-draw, 7 seasons)", y_test, p_aligned, le)
    print_row(m); results.append(m)

    # ── Feature importance (LR 7-season) ──────────────────────────────────
    print("\nLR feature importance (|coef| averaged across classes):")
    coefs = lr7.named_steps["clf"].coef_   # shape (3, n_features)
    mean_abs = np.abs(coefs).mean(axis=0)
    for feat, imp in sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: -x[1]):
        print(f"  {feat:<35}  {imp:.4f}")

    return results


if __name__ == "__main__":
    run_backtest_v3()
