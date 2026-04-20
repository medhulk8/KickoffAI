"""
V3 live predictor — computes all 24 features from DB for any upcoming fixture.

Hybrid model routing:
  - Default (no lineup): 7-season calibrated LR, 24 features (lr_v3_final.pkl)
  - Lineup-known: 3-season calibrated LR + StandardScaler, 25 features (lr_v3_lineup.pkl)
    Activated by passing xi_strength_diff= to predict()

Usage:
    pred = V3Predictor()
    result = pred.predict("Arsenal", "Chelsea", "2025-05-10")
    result = pred.predict("Arsenal", "Chelsea", "2025-05-10", xi_strength_diff=0.15)
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.elo import EloCalculator

DB_PATH         = PROJECT_ROOT / "data" / "processed" / "asil.db"
MODEL_PATH      = PROJECT_ROOT / "models" / "lr_v3_final.pkl"
LINEUP_MODEL_PATH = PROJECT_ROOT / "models" / "lr_v3_lineup.pkl"

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

LINEUP_FEATURE_COLS = FEATURE_COLS + ["xi_strength_diff"]

VENUE_FALLBACK_MIN = 3
LAST_N  = 5
LAST_N3 = 3
CONFIDENCE_HIGH   = 0.65
CONFIDENCE_MEDIUM = 0.50


def _season_code(date_str: str) -> int:
    """Convert a date string to our 4-digit season code (e.g., '2024-09-01' → 2425)."""
    d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    if d.month >= 8:
        y1, y2 = d.year % 100, (d.year + 1) % 100
    else:
        y1, y2 = (d.year - 1) % 100, d.year % 100
    return y1 * 100 + y2


def _rolling_avg(vals: list[float], n: int) -> float:
    """Mean of last n values (or all if fewer than n)."""
    return float(np.mean(vals[-n:])) if vals else 0.0


def _team_stats(team: str, match_date: str, season: int, conn: sqlite3.Connection) -> dict:
    """
    Compute all rolling features for one team before match_date.
    Returns a dict keyed by the role-prefixed feature names (role omitted — caller adds it).
    """
    rows = conn.execute(
        """
        SELECT date, home_team, away_team,
               home_goals, away_goals,
               home_shots_target, away_shots_target,
               result, season
        FROM matches
        WHERE (home_team = ? OR away_team = ?)
          AND date < ?
          AND home_goals IS NOT NULL
        ORDER BY date ASC
        """,
        (team, team, match_date),
    ).fetchall()

    if not rows:
        # No history — return league-average defaults
        return _default_stats()

    # Build per-match arrays (chronological)
    all_sot, all_sotc, all_gs, all_gc, all_pts, all_cs = [], [], [], [], [], []
    all_dates = []
    home_sot, home_sotc, home_gs, home_gc, home_cs = [], [], [], [], []
    away_sot, away_sotc, away_gs, away_gc, away_cs = [], [], [], [], []
    home_season_count = 0
    away_season_count = 0

    for row in rows:
        date, ht, at, hg, ag, hs, as_, result, row_season = row
        is_home = (ht == team)
        gs  = hg if is_home else ag
        gc  = ag if is_home else hg
        sot = hs if is_home else as_
        sotc = as_ if is_home else hs
        cs  = int(gc == 0)

        if result == "H":
            pts = 3 if is_home else 0
        elif result == "A":
            pts = 0 if is_home else 3
        else:
            pts = 1

        all_sot.append(sot); all_sotc.append(sotc)
        all_gs.append(gs);   all_gc.append(gc)
        all_pts.append(pts); all_cs.append(cs)
        all_dates.append(date)

        if str(row_season) == str(season):
            if is_home:
                home_sot.append(sot); home_sotc.append(sotc)
                home_gs.append(gs);   home_gc.append(gc)
                home_cs.append(cs)
                home_season_count += 1
            else:
                away_sot.append(sot); away_sotc.append(sotc)
                away_gs.append(gs);   away_gc.append(gc)
                away_cs.append(cs)
                away_season_count += 1

    # Overall L5 and L3
    ovr_sot_l5  = _rolling_avg(all_sot,  LAST_N)
    ovr_sotc_l5 = _rolling_avg(all_sotc, LAST_N)
    ovr_gs_l5   = _rolling_avg(all_gs,   LAST_N)
    ovr_gc_l5   = _rolling_avg(all_gc,   LAST_N)
    ovr_pts_l5  = _rolling_avg(all_pts,  LAST_N)
    ovr_cs_l5   = _rolling_avg(all_cs,   LAST_N)

    ovr_sot_l3 = _rolling_avg(all_sot, LAST_N3)
    ovr_gs_l3  = _rolling_avg(all_gs,  LAST_N3)
    ovr_pts_l3 = _rolling_avg(all_pts, LAST_N3)

    # Venue-specific (with fallback)
    def venue_val(venue_list, ovr_val, venue_count):
        if venue_count >= VENUE_FALLBACK_MIN and venue_list:
            return _rolling_avg(venue_list, LAST_N)
        return ovr_val

    return {
        "sot_l5":        venue_val(home_sot,  ovr_sot_l5,  home_season_count),
        "sotc_l5":       venue_val(home_sotc, ovr_sotc_l5, home_season_count),
        "cs_l5":         venue_val(home_cs,   ovr_cs_l5,   home_season_count),
        "sot_l5_away":   venue_val(away_sot,  ovr_sot_l5,  away_season_count),
        "sotc_l5_away":  venue_val(away_sotc, ovr_sotc_l5, away_season_count),
        "cs_l5_away":    venue_val(away_cs,   ovr_cs_l5,   away_season_count),
        "gs_l5":         ovr_gs_l5,
        "gs_l3":         ovr_gs_l3,
        "pts_l5":        ovr_pts_l5,
        "pts_l3":        ovr_pts_l3,
        "sot_l3":        ovr_sot_l3,
        "days_rest":     _days_rest(all_dates, match_date),
        "home_season_count": home_season_count,
        "away_season_count": away_season_count,
    }


def _days_rest(all_dates: list[str], match_date: str) -> int:
    if not all_dates:
        return 30
    last = datetime.strptime(all_dates[-1][:10], "%Y-%m-%d")
    curr = datetime.strptime(match_date[:10], "%Y-%m-%d")
    return min(int((curr - last).days), 30)


def _default_stats() -> dict:
    return {
        "sot_l5": 4.0, "sotc_l5": 4.0, "cs_l5": 0.2,
        "sot_l5_away": 4.0, "sotc_l5_away": 4.0, "cs_l5_away": 0.2,
        "gs_l5": 1.3, "gs_l3": 1.3, "pts_l5": 1.2, "pts_l3": 1.2,
        "sot_l3": 4.0, "days_rest": 7,
        "home_season_count": 0, "away_season_count": 0,
    }


def _league_position(team: str, season: int, match_date: str, conn: sqlite3.Connection) -> dict:
    """Return {pts, gd, played, rank, ppg, gd_pg} for team before match_date."""
    rows = conn.execute(
        """
        SELECT home_team, away_team, home_goals, away_goals, result, date
        FROM matches
        WHERE season = ?
          AND date < ?
          AND home_goals IS NOT NULL
        ORDER BY date ASC
        """,
        (season, match_date),
    ).fetchall()

    table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0})
    for ht, at, hg, ag, result, date in rows:
        h_pts = 3 if result == "H" else (1 if result == "D" else 0)
        a_pts = 3 if result == "A" else (1 if result == "D" else 0)
        for t, pts, gf, ga in [(ht, h_pts, hg, ag), (at, a_pts, ag, hg)]:
            table[t]["pts"]    += pts
            table[t]["gf"]     += gf
            table[t]["ga"]     += ga
            table[t]["gd"]     += (gf - ga)
            table[t]["played"] += 1

    ranked = sorted(table.items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]))
    rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}

    info = table.get(team, {"pts": 0, "gd": 0, "played": 0})
    played = info["played"] or 1  # avoid div-by-zero
    return {
        "rank":   rank_map.get(team, 20),
        "pts":    info["pts"],
        "gd":     info["gd"],
        "played": info["played"],
        "ppg":    round(info["pts"] / played, 4),
        "gd_pg":  round(info["gd"]  / played, 4),
    }


def _elo_diff(home_team: str, away_team: str, match_date: str, conn: sqlite3.Connection) -> float:
    rows = conn.execute(
        """
        SELECT match_id, date, home_team, away_team, result
        FROM matches
        WHERE home_goals IS NOT NULL
          AND date < ?
        ORDER BY date ASC
        """,
        (match_date,),
    ).fetchall()
    matches = [
        {"match_id": r[0], "date": r[1], "home_team": r[2], "away_team": r[3], "result": r[4]}
        for r in rows
    ]
    ratings = EloCalculator().get_current_ratings(matches)
    return float(ratings.get(home_team, 1500) - ratings.get(away_team, 1500))


class V3Predictor:
    def __init__(self, db_path: str | None = None, model_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        mp = Path(model_path) if model_path else MODEL_PATH
        with open(mp, "rb") as f:
            self.model = pickle.load(f)
        if LINEUP_MODEL_PATH.exists():
            with open(LINEUP_MODEL_PATH, "rb") as f:
                self.lineup_model = pickle.load(f)
        else:
            self.lineup_model = None

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        xi_strength_diff: float | None = None,
    ) -> dict:
        """
        Returns:
            home_prob, draw_prob, away_prob (floats summing to 1.0)
            prediction  ("H" / "D" / "A")
            confidence  ("high" / "medium" / "low")
            features    (dict of all feature values, for transparency)
            model_version: "v3_lineup" when xi_strength_diff provided, else "v3_independent"

        xi_strength_diff: home_xi_strength - away_xi_strength (mean season-to-date
            minutes share of starting XI). When provided, routes to the 3-season
            lineup-aware model (lr_v3_lineup.pkl). Otherwise uses the 7-season base model.
        """
        feats = self._compute_features(home_team, away_team, match_date)

        use_lineup = (xi_strength_diff is not None) and (self.lineup_model is not None)
        if use_lineup:
            feats["xi_strength_diff"] = round(float(xi_strength_diff), 4)
            X = np.array([[feats[f] for f in LINEUP_FEATURE_COLS]])
            model = self.lineup_model
            model_version = "v3_lineup"
        else:
            X = np.array([[feats[f] for f in FEATURE_COLS]])
            model = self.model
            model_version = "v3_independent"

        p_raw = model.predict_proba(X)[0]
        classes = list(model.classes_)
        p = {"A": 0.0, "D": 0.0, "H": 0.0}
        for i, c in enumerate(classes):
            p[c] = float(p_raw[i])

        prediction = max(p, key=p.get)
        conf_val   = p[prediction]
        if conf_val >= CONFIDENCE_HIGH:
            confidence = "high"
        elif conf_val >= CONFIDENCE_MEDIUM:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "home_prob":  round(p["H"], 4),
            "draw_prob":  round(p["D"], 4),
            "away_prob":  round(p["A"], 4),
            "prediction": prediction,
            "confidence": confidence,
            "features":   feats,
            "model_version": model_version,
        }

    def _compute_features(self, home_team: str, away_team: str, match_date: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        try:
            season = _season_code(match_date)

            h = _team_stats(home_team, match_date, season, conn)
            a = _team_stats(away_team, match_date, season, conn)

            h_pos = _league_position(home_team, season, match_date, conn)
            a_pos = _league_position(away_team, season, match_date, conn)

            elo = _elo_diff(home_team, away_team, match_date, conn)

            # Venue fallback: use home-specific for home team, away-specific for away team
            h_sot  = h["sot_l5"]   if h["home_season_count"] >= VENUE_FALLBACK_MIN else h["sot_l5"]
            h_sotc = h["sotc_l5"]  if h["home_season_count"] >= VENUE_FALLBACK_MIN else h["sotc_l5"]
            h_cs   = h["cs_l5"]    if h["home_season_count"] >= VENUE_FALLBACK_MIN else h["cs_l5"]

            a_sot  = a["sot_l5_away"]  if a["away_season_count"] >= VENUE_FALLBACK_MIN else a["sot_l5"]
            a_sotc = a["sotc_l5_away"] if a["away_season_count"] >= VENUE_FALLBACK_MIN else a["sotc_l5"]
            a_cs   = a["cs_l5_away"]   if a["away_season_count"] >= VENUE_FALLBACK_MIN else a["cs_l5"]

            # matchweek
            matchweek = round((h_pos["played"] + a_pos["played"]) / 2) + 1

            return {
                "home_sot_l5":         round(h_sot, 4),
                "home_sot_conceded_l5":round(h_sotc, 4),
                "home_conversion":     round((h["gs_l5"] + 1) / (h_sot + 2), 4),
                "home_clean_sheet_l5": round(h_cs, 4),
                "home_pts_momentum":   round(h["pts_l3"] - h["pts_l5"], 4),
                "home_goals_momentum": round(h["gs_l3"] - h["gs_l5"], 4),
                "home_sot_momentum":   round(h["sot_l3"] - h_sot, 4),
                "home_days_rest":      h["days_rest"],
                "away_sot_l5":         round(a_sot, 4),
                "away_sot_conceded_l5":round(a_sotc, 4),
                "away_conversion":     round((a["gs_l5"] + 1) / (a_sot + 2), 4),
                "away_clean_sheet_l5": round(a_cs, 4),
                "away_pts_momentum":   round(a["pts_l3"] - a["pts_l5"], 4),
                "away_goals_momentum": round(a["gs_l3"] - a["gs_l5"], 4),
                "away_sot_momentum":   round(a["sot_l3"] - a_sot, 4),
                "away_days_rest":      a["days_rest"],
                "elo_diff":   round(elo, 2),
                "ppg_diff":   round(h_pos["ppg"]   - a_pos["ppg"],   4),
                "ppg_mean":   round((h_pos["ppg"]  + a_pos["ppg"])  / 2, 4),
                "gd_pg_diff": round(h_pos["gd_pg"] - a_pos["gd_pg"], 4),
                "gd_pg_mean": round((h_pos["gd_pg"] + a_pos["gd_pg"]) / 2, 4),
                "rank_diff":  round(float(h_pos["rank"] - a_pos["rank"]), 4),
                "rank_mean":  round((h_pos["rank"] + a_pos["rank"]) / 2, 4),
                "matchweek":  matchweek,
            }
        finally:
            conn.close()
