"""
V4 live predictor — 26 features: V3 base with EWM replacing momentum + opp_ppg_l5.

New vs V3:
  - pts/goals/sot EWM span=7 replaces linear l3-l5 momentum
  - home_opp_ppg_l5, away_opp_ppg_l5: rolling avg PPG of last 5 opponents
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

DB_PATH    = PROJECT_ROOT / "data" / "processed" / "asil.db"
MODEL_PATH = PROJECT_ROOT / "models" / "lr_v4_final.pkl"

FEATURE_COLS = [
    "home_sot_l5", "home_sot_conceded_l5", "home_conversion",
    "home_clean_sheet_l5", "home_pts_ewm", "home_goals_ewm",
    "home_sot_ewm", "home_days_rest",
    "away_sot_l5", "away_sot_conceded_l5", "away_conversion",
    "away_clean_sheet_l5", "away_pts_ewm", "away_goals_ewm",
    "away_sot_ewm", "away_days_rest",
    "elo_diff",
    "ppg_diff", "ppg_mean",
    "gd_pg_diff", "gd_pg_mean",
    "rank_diff", "rank_mean",
    "matchweek",
    "home_opp_ppg_l5", "away_opp_ppg_l5",
]

EWM_SPAN          = 7
VENUE_FALLBACK_MIN = 3
LAST_N            = 5
LAST_N3           = 3
CONFIDENCE_HIGH   = 0.65
CONFIDENCE_MEDIUM = 0.50

_DEFAULT_OPP_PPG = 1.2   # league average


def _season_code(date_str: str) -> int:
    d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    if d.month >= 8:
        y1, y2 = d.year % 100, (d.year + 1) % 100
    else:
        y1, y2 = (d.year - 1) % 100, d.year % 100
    return y1 * 100 + y2


def _rolling_avg(vals: list, n: int) -> float:
    return float(np.mean(vals[-n:])) if vals else 0.0


def _ewm_value(vals: list, span: int = EWM_SPAN) -> float:
    """EWMA of chronologically ordered values (most recent = highest weight)."""
    if not vals:
        return 0.0
    alpha = 2 / (span + 1)
    result = float(vals[0])
    for v in vals[1:]:
        result = (1 - alpha) * result + alpha * float(v)
    return result


def _days_rest(all_dates: list, match_date: str) -> int:
    if not all_dates:
        return 30
    last = datetime.strptime(all_dates[-1][:10], "%Y-%m-%d")
    curr = datetime.strptime(match_date[:10], "%Y-%m-%d")
    return min(int((curr - last).days), 30)


def _team_stats(team: str, match_date: str, season: int, conn: sqlite3.Connection) -> dict:
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
        return _default_stats()

    all_sot, all_sotc, all_gs, all_gc, all_pts, all_cs = [], [], [], [], [], []
    all_dates = []
    home_sot, home_sotc, home_gs, home_gc, home_cs = [], [], [], [], []
    away_sot, away_sotc, away_gs, away_gc, away_cs = [], [], [], [], []
    home_season_count = 0
    away_season_count = 0

    for row in rows:
        date, ht, at, hg, ag, hs, as_, result, row_season = row
        is_home = (ht == team)
        gs   = hg if is_home else ag
        gc   = ag if is_home else hg
        sot  = hs if is_home else as_
        sotc = as_ if is_home else hs
        cs   = int(gc == 0)
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
                home_sot.append(sot);  home_sotc.append(sotc)
                home_gs.append(gs);    home_gc.append(gc)
                home_cs.append(cs);    home_season_count += 1
            else:
                away_sot.append(sot);  away_sotc.append(sotc)
                away_gs.append(gs);    away_gc.append(gc)
                away_cs.append(cs);    away_season_count += 1

    def venue_val(venue_list, ovr_vals, count):
        if count >= VENUE_FALLBACK_MIN and venue_list:
            return _rolling_avg(venue_list, LAST_N)
        return _rolling_avg(ovr_vals, LAST_N)

    return {
        # L5 / L3 rolling (for sot_l5, sotc_l5, cs_l5, conversion)
        "sot_l5":   venue_val(home_sot,  all_sot,  home_season_count),
        "sotc_l5":  venue_val(home_sotc, all_sotc, home_season_count),
        "cs_l5":    venue_val(home_cs,   all_cs,   home_season_count),
        "sot_l5_away":  venue_val(away_sot,  all_sot,  away_season_count),
        "sotc_l5_away": venue_val(away_sotc, all_sotc, away_season_count),
        "cs_l5_away":   venue_val(away_cs,   all_cs,   away_season_count),
        "gs_l5": _rolling_avg(all_gs, LAST_N),
        # EWM signals — use full chronological history
        "pts_ewm":   _ewm_value(all_pts),
        "goals_ewm": _ewm_value(all_gs),
        "sot_ewm":   _ewm_value(all_sot),
        # Days rest
        "days_rest": _days_rest(all_dates, match_date),
        "home_season_count": home_season_count,
        "away_season_count": away_season_count,
    }


def _opp_ppg_l5(team: str, match_date: str, conn: sqlite3.Connection) -> float:
    """Rolling avg PPG of last 5 opponents at time of each match."""
    rows = conn.execute(
        """
        SELECT date, home_team, away_team, season
        FROM matches
        WHERE (home_team = ? OR away_team = ?)
          AND date < ?
          AND home_goals IS NOT NULL
        ORDER BY date DESC
        LIMIT 5
        """,
        (team, team, match_date),
    ).fetchall()

    if not rows:
        return _DEFAULT_OPP_PPG

    ppgs = []
    for date, ht, at, season in rows:
        opp = at if ht == team else ht
        res = conn.execute(
            """
            SELECT
                SUM(CASE
                    WHEN home_team=? AND result='H' THEN 3
                    WHEN away_team=? AND result='A' THEN 3
                    WHEN result='D' THEN 1
                    ELSE 0
                END),
                COUNT(*)
            FROM matches
            WHERE (home_team=? OR away_team=?)
              AND season=?
              AND date < ?
              AND home_goals IS NOT NULL
            """,
            (opp, opp, opp, opp, season, date),
        ).fetchone()
        pts, played = res
        ppgs.append(pts / played if (played and played > 0) else _DEFAULT_OPP_PPG)

    return round(float(np.mean(ppgs)), 4)


def _league_position(team: str, season: int, match_date: str, conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT home_team, away_team, home_goals, away_goals, result, date
        FROM matches
        WHERE season = ? AND date < ? AND home_goals IS NOT NULL
        ORDER BY date ASC
        """,
        (season, match_date),
    ).fetchall()

    table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0})
    for ht, at, hg, ag, result, _ in rows:
        h_pts = 3 if result == "H" else (1 if result == "D" else 0)
        a_pts = 3 if result == "A" else (1 if result == "D" else 0)
        for t, pts, gf, ga in [(ht, h_pts, hg, ag), (at, a_pts, ag, hg)]:
            table[t]["pts"]    += pts
            table[t]["gf"]     += gf
            table[t]["ga"]     += ga
            table[t]["gd"]     += gf - ga
            table[t]["played"] += 1

    ranked  = sorted(table.items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]))
    rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}

    info   = table.get(team, {"pts": 0, "gd": 0, "played": 0})
    played = info["played"] or 1
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
        WHERE home_goals IS NOT NULL AND date < ?
        ORDER BY date ASC
        """,
        (match_date,),
    ).fetchall()
    matches  = [{"match_id": r[0], "date": r[1], "home_team": r[2], "away_team": r[3], "result": r[4]} for r in rows]
    ratings  = EloCalculator().get_current_ratings(matches)
    return float(ratings.get(home_team, 1500) - ratings.get(away_team, 1500))


def _default_stats() -> dict:
    return {
        "sot_l5": 4.0,  "sotc_l5": 4.0,  "cs_l5": 0.2,
        "sot_l5_away": 4.0, "sotc_l5_away": 4.0, "cs_l5_away": 0.2,
        "gs_l5": 1.3,
        "pts_ewm": 1.2, "goals_ewm": 1.3, "sot_ewm": 4.0,
        "days_rest": 7,
        "home_season_count": 0, "away_season_count": 0,
    }


class V4Predictor:
    def __init__(self, db_path: str | None = None, model_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        mp = Path(model_path) if model_path else MODEL_PATH
        with open(mp, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, home_team: str, away_team: str, match_date: str) -> dict:
        feats = self._compute_features(home_team, away_team, match_date)
        X = np.array([[feats[f] for f in FEATURE_COLS]])
        p_raw   = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        p = {"A": 0.0, "D": 0.0, "H": 0.0}
        for i, c in enumerate(classes):
            p[c] = float(p_raw[i])

        prediction = max(p, key=p.get)
        conf_val   = p[prediction]
        confidence = "high" if conf_val >= CONFIDENCE_HIGH else \
                     "medium" if conf_val >= CONFIDENCE_MEDIUM else "low"

        return {
            "home_prob":     round(p["H"], 4),
            "draw_prob":     round(p["D"], 4),
            "away_prob":     round(p["A"], 4),
            "prediction":    prediction,
            "confidence":    confidence,
            "features":      feats,
            "model_version": "v4",
        }

    def _compute_features(self, home_team: str, away_team: str, match_date: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        try:
            season = _season_code(match_date)

            h     = _team_stats(home_team, match_date, season, conn)
            a     = _team_stats(away_team, match_date, season, conn)
            h_pos = _league_position(home_team, season, match_date, conn)
            a_pos = _league_position(away_team, season, match_date, conn)
            elo   = _elo_diff(home_team, away_team, match_date, conn)

            h_opp = _opp_ppg_l5(home_team, match_date, conn)
            a_opp = _opp_ppg_l5(away_team, match_date, conn)

            def venue(home_list_val, away_list_val, is_home_team, h_cnt, a_cnt):
                if is_home_team:
                    return home_list_val if h_cnt >= VENUE_FALLBACK_MIN else away_list_val
                return away_list_val if a_cnt >= VENUE_FALLBACK_MIN else home_list_val

            h_sot  = venue(h["sot_l5"],  h["sot_l5"],       True,  h["home_season_count"], h["away_season_count"])
            h_sotc = venue(h["sotc_l5"], h["sotc_l5"],      True,  h["home_season_count"], h["away_season_count"])
            h_cs   = venue(h["cs_l5"],   h["cs_l5"],        True,  h["home_season_count"], h["away_season_count"])
            a_sot  = venue(a["sot_l5"],  a["sot_l5_away"],  False, a["home_season_count"], a["away_season_count"])
            a_sotc = venue(a["sotc_l5"], a["sotc_l5_away"], False, a["home_season_count"], a["away_season_count"])
            a_cs   = venue(a["cs_l5"],   a["cs_l5_away"],   False, a["home_season_count"], a["away_season_count"])

            matchweek = round((h_pos["played"] + a_pos["played"]) / 2) + 1

            return {
                "home_sot_l5":          round(h_sot, 4),
                "home_sot_conceded_l5": round(h_sotc, 4),
                "home_conversion":      round((h["gs_l5"] + 1) / (h_sot + 2), 4),
                "home_clean_sheet_l5":  round(h_cs, 4),
                "home_pts_ewm":         round(h["pts_ewm"],   4),
                "home_goals_ewm":       round(h["goals_ewm"], 4),
                "home_sot_ewm":         round(h["sot_ewm"],   4),
                "home_days_rest":       h["days_rest"],
                "away_sot_l5":          round(a_sot, 4),
                "away_sot_conceded_l5": round(a_sotc, 4),
                "away_conversion":      round((a["gs_l5"] + 1) / (a_sot + 2), 4),
                "away_clean_sheet_l5":  round(a_cs, 4),
                "away_pts_ewm":         round(a["pts_ewm"],   4),
                "away_goals_ewm":       round(a["goals_ewm"], 4),
                "away_sot_ewm":         round(a["sot_ewm"],   4),
                "away_days_rest":       a["days_rest"],
                "elo_diff":    round(elo, 2),
                "ppg_diff":    round(h_pos["ppg"]   - a_pos["ppg"],   4),
                "ppg_mean":    round((h_pos["ppg"]  + a_pos["ppg"])  / 2, 4),
                "gd_pg_diff":  round(h_pos["gd_pg"] - a_pos["gd_pg"], 4),
                "gd_pg_mean":  round((h_pos["gd_pg"] + a_pos["gd_pg"]) / 2, 4),
                "rank_diff":   round(float(h_pos["rank"] - a_pos["rank"]), 4),
                "rank_mean":   round((h_pos["rank"] + a_pos["rank"]) / 2, 4),
                "matchweek":   matchweek,
                "home_opp_ppg_l5": h_opp,
                "away_opp_ppg_l5": a_opp,
            }
        finally:
            conn.close()
