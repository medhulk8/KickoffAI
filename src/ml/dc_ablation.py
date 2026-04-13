"""
Dixon-Coles ablation for KickoffAI.

Compares across all 3 folds:
  1. Bookmaker baseline
  2. Pure Dixon-Coles goal model
  3. DC probs + bookmaker probs → LR calibration layer

Dixon-Coles is fit on training fold matches only (no leakage into test fold).
"""

import sqlite3
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml.backtest import (
    load_data, get_fold_data, FOLDS, CLASSES,
    compute_metrics, BookmakerBaseline,
)
from src.ml.dixon_coles import DixonColesModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"


def load_matches_with_goals() -> list[dict]:
    """Load all matches with goals data from DB, ordered by date."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT match_id, date, season, home_team, away_team, home_goals, away_goals, result
        FROM matches
        WHERE home_goals IS NOT NULL
        ORDER BY date ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dc_probas(dc_model: DixonColesModel, df) -> np.ndarray:
    """Get DC probability arrays for a DataFrame of matches. Returns (n,3) [A,D,H]."""
    probas = np.zeros((len(df), 3))
    for i, (_, row) in enumerate(df.iterrows()):
        probas[i] = dc_model.predict_proba(row["home_team"], row["away_team"])
    return probas


def align_probas(model, df, features) -> np.ndarray:
    """Get LR probas aligned to CLASSES=[A,D,H] order."""
    raw = model.predict_proba(df[features])
    le  = model.named_steps["clf"].classes_
    co  = {c: i for i, c in enumerate(le)}
    out = np.zeros((len(raw), 3))
    for i, cls in enumerate(CLASSES):
        out[:, i] = raw[:, co[cls]]
    return out


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=0.1, max_iter=1000, random_state=42)),
    ])


def run_dc_ablation():
    df        = load_data()          # CSV dataset (has team names + odds + result)
    all_goals = load_matches_with_goals()   # DB matches for DC fitting

    # Index goals matches by (home_team, away_team, date) for season filtering
    goals_by_season: dict[str, list[dict]] = {}
    for m in all_goals:
        s = str(m["season"])
        goals_by_season.setdefault(s, []).append(m)

    print(f"{'Model':<35} {'Fold':>6} {'Acc':>7} {'LL':>8} {'Brier':>7} {'DrawR':>7}")
    print("-" * 70)

    for fold in FOLDS:
        train_df, test_df = get_fold_data(df, fold)
        y_test = test_df["result"].values
        fname  = fold["name"]

        # Training matches with goals (for DC fitting)
        train_goals = [
            m for m in all_goals
            if str(m["season"]) in fold["train_seasons"]
        ]

        # ── 1. Bookmaker baseline ──────────────────────────────────────────
        bm = BookmakerBaseline()
        bm_pred   = bm.predict(test_df[["bm_away_prob", "bm_draw_prob", "bm_home_prob"]])
        bm_probas = bm.predict_proba(test_df[["bm_away_prob", "bm_draw_prob", "bm_home_prob"]])
        m = compute_metrics(y_test, bm_pred, bm_probas)
        print(f"  {'Bookmaker baseline':<33} {fname:>6} {m['accuracy']:>7.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['draw_recall']:>7.1%}")

        # ── 2. Pure DC model ───────────────────────────────────────────────
        dc = DixonColesModel(rho=-0.10, half_life_days=90, max_goals=6)
        dc.fit(train_goals)
        dc_probas = get_dc_probas(dc, test_df)
        dc_preds  = np.array(CLASSES)[dc_probas.argmax(axis=1)]
        m = compute_metrics(y_test, dc_preds, dc_probas)
        print(f"  {'DC goal model (pure)':<33} {fname:>6} {m['accuracy']:>7.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['draw_recall']:>7.1%}")

        # ── 3. DC + bookmaker calibration layer ───────────────────────────
        # Train LR on training fold: features = DC probs + bookmaker probs
        dc_train = get_dc_probas(dc, train_df)
        bm_train = train_df[["bm_home_prob", "bm_draw_prob", "bm_away_prob"]].values
        X_cal_train = np.hstack([dc_train, bm_train])

        bm_test  = test_df[["bm_home_prob", "bm_draw_prob", "bm_away_prob"]].values
        X_cal_test  = np.hstack([dc_probas, bm_test])

        cal = make_lr()
        cal.fit(X_cal_train, train_df["result"].values)
        cal_probas = align_probas(cal, None, None) if False else \
                     _align_raw(cal.predict_proba(X_cal_test), cal.named_steps["clf"].classes_)
        cal_preds  = np.array(CLASSES)[cal_probas.argmax(axis=1)]
        m = compute_metrics(y_test, cal_preds, cal_probas)
        print(f"  {'DC + bookmaker (LR layer)':<33} {fname:>6} {m['accuracy']:>7.1%} {m['log_loss']:>8.4f} {m['brier']:>7.4f} {m['draw_recall']:>7.1%}")

        print()

    # Show DC team rankings for Fold 3 model (all 3 training seasons)
    fold3_goals = [m for m in all_goals if str(m["season"]) in FOLDS[2]["train_seasons"]]
    dc_final = DixonColesModel(rho=-0.10, half_life_days=90)
    dc_final.fit(fold3_goals)
    print("\nTop 10 teams by DC strength (attack − defence) — Fold 3 model:")
    for team, att, dfe in dc_final.top_teams(10):
        print(f"  {team:<28}  att={att:+.3f}  def={dfe:+.3f}")


def _align_raw(raw: np.ndarray, le_classes) -> np.ndarray:
    co  = {c: i for i, c in enumerate(le_classes)}
    out = np.zeros((len(raw), 3))
    for i, cls in enumerate(CLASSES):
        out[:, i] = raw[:, co[cls]]
    return out


if __name__ == "__main__":
    run_dc_ablation()
