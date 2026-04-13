"""
Build supervised training dataset for ML model.

One row per match. All features computed point-in-time (strictly before match date).
No leakage: uses before_date param on every stat call.

Output columns:
    match_id, date, season, home_team, away_team,
    # Bookmaker probabilities
    bm_home_prob, bm_draw_prob, bm_away_prob,
    # Weighted form
    home_weighted_ppg, away_weighted_ppg,
    home_weighted_goals, away_weighted_goals,
    # Defensive
    home_def_solidity, away_def_solidity,
    # Draw signal
    draw_likelihood,
    h2h_draw_rate,
    # Label
    result  (H / D / A)
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.advanced_stats import AdvancedStatsCalculator
from src.data.weighted_stats import WeightedStatsCalculator
from src.ml.elo import EloCalculator

# Import directly to avoid pulling in Ollama/LangGraph from workflows __init__
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "draw_detector",
    pathlib.Path(__file__).parent.parent / "workflows" / "draw_detector.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DrawDetector = _mod.DrawDetector


DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"


def load_all_matches(db_path: str) -> list[dict]:
    """Load all matches from DB ordered by date ascending."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT
            match_id, date, season,
            home_team, away_team,
            result,
            avg_home_prob  AS home_baseline_prob,
            avg_draw_prob  AS draw_baseline_prob,
            avg_away_prob  AS away_baseline_prob
        FROM matches
        WHERE result IS NOT NULL
          AND avg_home_prob IS NOT NULL
        ORDER BY date ASC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def load_all_matches_for_elo(db_path: str) -> list[dict]:
    """Load ALL matches (including those without odds) for Elo warm-up."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT match_id, date, home_team, away_team, result
        FROM matches
        ORDER BY date ASC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def compute_h2h_draw_rate(h2h_stats: dict, alpha: float = 1.0) -> float:
    """
    Smoothed H2H draw rate using Laplace smoothing.
    Prevents extreme values (0.0 or 1.0) when sample is small.
    Formula: (draws + alpha) / (matches + 2*alpha)
    """
    n = h2h_stats.get("matches_played", 0)
    draws = h2h_stats.get("draws", 0)
    return round((draws + alpha) / (n + 2 * alpha), 4)


def build_row(
    match: dict,
    adv_calc: AdvancedStatsCalculator,
    wgt_calc: WeightedStatsCalculator,
    draw_detector: DrawDetector,
    elo_diff: float = 0.0,
) -> Optional[dict]:
    """
    Compute all features for one match.
    Returns None if insufficient data (skip match).
    """
    date = match["date"]
    home = match["home_team"]
    away = match["away_team"]

    # --- Weighted form (last 5 for PPG/goals, last 10 for draw rates) ---
    home_wform = wgt_calc.get_weighted_form(home, last_n=5, before_date=date)
    away_wform = wgt_calc.get_weighted_form(away, last_n=5, before_date=date)

    # Skip matches with no prior data for either team
    if home_wform["matches_played"] == 0 or away_wform["matches_played"] == 0:
        return None

    # Draw rate over last 10 matches (Laplace smoothed, alpha=1)
    home_wform10 = wgt_calc.get_weighted_form(home, last_n=10, before_date=date)
    away_wform10 = wgt_calc.get_weighted_form(away, last_n=10, before_date=date)
    home_n, home_draws = home_wform10["matches_played"], home_wform10["draws"]
    away_n, away_draws = away_wform10["matches_played"], away_wform10["draws"]
    home_draw_rate = round((home_draws + 1) / (home_n + 2), 4)
    away_draw_rate = round((away_draws + 1) / (away_n + 2), 4)

    # --- Advanced stats (for defensive solidity) ---
    home_adv = adv_calc.get_team_advanced_stats(home, last_n=5, before_date=date)
    away_adv = adv_calc.get_team_advanced_stats(away, last_n=5, before_date=date)

    # --- H2H ---
    h2h = adv_calc.get_head_to_head_stats(home, away, last_n=10, before_date=date)
    h2h_draw_rate = compute_h2h_draw_rate(h2h)

    # --- Bookmaker probs ---
    bm_home = match["home_baseline_prob"]
    bm_draw = match["draw_baseline_prob"]
    bm_away = match["away_baseline_prob"]

    # --- Draw likelihood score ---
    baseline_dict = {
        "home_prob": bm_home,
        "draw_prob": bm_draw,
        "away_prob": bm_away,
    }
    form_comparison = wgt_calc.compare_form_differential(home_wform, away_wform)

    draw_likelihood = draw_detector.detect_draw_likelihood(
        home_form=home_wform,
        away_form=away_wform,
        baseline=baseline_dict,
        h2h_stats=h2h,
        advanced_stats_home=home_adv.get("defensive", {}),
        advanced_stats_away=away_adv.get("defensive", {}),
        home_weighted_form=home_wform,
        away_weighted_form=away_wform,
        form_comparison=form_comparison,
    )

    return {
        "match_id": match["match_id"],
        "date": date,
        "season": match["season"],
        "home_team": home,
        "away_team": away,
        # Bookmaker probs
        "bm_home_prob": round(bm_home, 4),
        "bm_draw_prob": round(bm_draw, 4),
        "bm_away_prob": round(bm_away, 4),
        # Weighted form
        "home_weighted_ppg": home_wform["weighted_points_per_game"],
        "away_weighted_ppg": away_wform["weighted_points_per_game"],
        "home_weighted_goals": home_wform["weighted_goals_per_game"],
        "away_weighted_goals": away_wform["weighted_goals_per_game"],
        # Defensive
        "home_def_solidity": home_adv["defensive"]["defensive_solidity"],
        "away_def_solidity": away_adv["defensive"]["defensive_solidity"],
        # Draw signals
        "draw_likelihood": round(draw_likelihood, 4),
        "h2h_draw_rate": h2h_draw_rate,
        "home_draw_rate": home_draw_rate,
        "away_draw_rate": away_draw_rate,
        # Elo
        "elo_diff": round(elo_diff, 2),
        # Label
        "result": match["result"],
    }


def build_dataset(db_path: str = str(DB_PATH), output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build the full training dataset.

    Args:
        db_path: Path to SQLite database
        output_path: Optional CSV path to save dataset

    Returns:
        DataFrame with one row per match
    """
    print(f"Loading matches from {db_path}...")
    matches = load_all_matches(db_path)
    print(f"Total matches with result + odds: {len(matches)}")

    adv_calc = AdvancedStatsCalculator(db_path)
    wgt_calc = WeightedStatsCalculator(db_path)
    draw_detector = DrawDetector()

    # Precompute Elo using all matches (including those without odds) for proper warm-up
    all_matches_for_elo = load_all_matches_for_elo(db_path)
    elo_calc = EloCalculator()
    elo_map = elo_calc.compute(all_matches_for_elo)
    print(f"Elo computed for {len(elo_map)} matches")

    rows = []
    skipped = 0

    for i, match in enumerate(matches):
        elo_diff = elo_map.get(match["match_id"], {}).get("elo_diff", 0.0)
        row = build_row(match, adv_calc, wgt_calc, draw_detector, elo_diff=elo_diff)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(matches)} matches...")

    df = pd.DataFrame(rows)
    print(f"\nDataset built: {len(df)} rows ({skipped} skipped — no prior data)")
    print(f"Result distribution:\n{df['result'].value_counts()}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    output = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
    df = build_dataset(output_path=str(output))

    print("\nSample rows:")
    print(df.head(3).to_string())

    print("\nFeature stats:")
    feature_cols = [c for c in df.columns if c not in ["match_id", "date", "season", "home_team", "away_team", "result"]]
    print(df[feature_cols].describe().round(3).to_string())
