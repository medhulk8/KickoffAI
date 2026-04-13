"""
Dixon-Coles Poisson goal model for KickoffAI.

Estimates per-team attack/defence parameters from historical match data
using time-decayed maximum likelihood. Derives H/D/A probabilities from
the Poisson scoreline distribution with the Dixon-Coles low-score correction.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market"

Design decisions (V1):
  - All prior matches used for fitting, weighted by time decay
  - Exponential decay, half-life = 90 days (default)
  - Global fixed rho = -0.10 (not fitted per season)
  - Unknown/promoted teams default to league-average (attack=0, defence=0)
  - Sum-to-zero normalization on attack parameters after fitting
  - Scoreline grid: 0–6 goals each side
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from datetime import datetime


# ============================================================================
# DC correction helpers
# ============================================================================

def _dc_tau_vec(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Vectorized DC tau correction (returns values, not log)."""
    tau = np.ones(len(home_goals))

    m00 = (home_goals == 0) & (away_goals == 0)
    m10 = (home_goals == 1) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m11 = (home_goals == 1) & (away_goals == 1)

    tau[m00] = 1.0 - lambda_h[m00] * lambda_a[m00] * rho
    tau[m10] = 1.0 + lambda_a[m10] * rho
    tau[m01] = 1.0 + lambda_h[m01] * rho
    tau[m11] = 1.0 - rho

    return np.maximum(tau, 1e-10)


def _dc_tau_scalar(i: int, j: int, lh: float, la: float, rho: float) -> float:
    if i == 0 and j == 0:
        return max(1.0 - lh * la * rho, 1e-10)
    if i == 1 and j == 0:
        return max(1.0 + la * rho, 1e-10)
    if i == 0 and j == 1:
        return max(1.0 + lh * rho, 1e-10)
    if i == 1 and j == 1:
        return max(1.0 - rho, 1e-10)
    return 1.0


# ============================================================================
# Model
# ============================================================================

class DixonColesModel:
    """
    Poisson goal model with Dixon-Coles low-score correction.

    Fit on a list of historical matches, then call predict_proba(home, away)
    to get [p_away, p_draw, p_home] in CLASSES=[A,D,H] order.

    Unknown teams (promoted sides with no prior data) default to
    league-average parameters (attack=0, defence=0).
    """

    def __init__(
        self,
        rho: float = -0.10,
        half_life_days: float = 90.0,
        max_goals: int = 6,
    ):
        self.rho = rho
        self.half_life_days = half_life_days
        self.max_goals = max_goals

        self._home_adv: float = 0.3
        self._attacks: dict[str, float] = {}
        self._defences: dict[str, float] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches: list[dict]) -> "DixonColesModel":
        """
        Fit attack/defence parameters from historical matches.

        Args:
            matches: dicts with date (YYYY-MM-DD str), home_team, away_team,
                     home_goals (int), away_goals (int). Must already be
                     filtered to only matches before the prediction date.
        """
        valid = [
            m for m in matches
            if m.get("home_goals") is not None and m.get("away_goals") is not None
        ]
        if len(valid) < 10:
            raise ValueError(f"Too few valid matches to fit ({len(valid)})")

        # Time weights
        dates = [datetime.strptime(m["date"], "%Y-%m-%d") for m in valid]
        ref_date = max(dates)
        decay = np.log(2.0) / self.half_life_days
        weights = np.array([np.exp(-decay * (ref_date - d).days) for d in dates])

        teams = sorted({t for m in valid for t in (m["home_team"], m["away_team"])})
        n = len(teams)
        tidx = {t: i for i, t in enumerate(teams)}

        h_idx = np.array([tidx[m["home_team"]] for m in valid])
        a_idx = np.array([tidx[m["away_team"]] for m in valid])
        hg    = np.array([m["home_goals"] for m in valid], dtype=float)
        ag    = np.array([m["away_goals"] for m in valid], dtype=float)
        log_fac_hg = gammaln(hg + 1)
        log_fac_ag = gammaln(ag + 1)

        # Params: [home_adv, attack_1..N-1, defence_1..N-1]
        # teams[0] is reference: attack=0, defence=0 (identifiability)
        n_free = 1 + (n - 1) + (n - 1)
        x0 = np.zeros(n_free)
        x0[0] = 0.3

        def neg_ll(params: np.ndarray) -> float:
            ha = params[0]
            att = np.zeros(n); att[1:] = params[1:n]
            dfe = np.zeros(n); dfe[1:] = params[n: 2 * n - 1]

            lh = np.exp(ha + att[h_idx] + dfe[a_idx])
            la = np.exp(att[a_idx] + dfe[h_idx])

            log_p_h = hg * np.log(np.maximum(lh, 1e-10)) - lh - log_fac_hg
            log_p_a = ag * np.log(np.maximum(la, 1e-10)) - la - log_fac_ag
            tau = _dc_tau_vec(hg, ag, lh, la, self.rho)

            return -float(weights @ (np.log(tau) + log_p_h + log_p_a))

        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})

        p = res.x
        att = np.zeros(n); att[1:] = p[1:n]
        dfe = np.zeros(n); dfe[1:] = p[n: 2 * n - 1]

        # Normalize: subtract mean attack so league average = 0
        att -= att.mean()

        self._home_adv = float(p[0])
        self._attacks  = {t: float(att[i]) for i, t in enumerate(teams)}
        self._defences = {t: float(dfe[i]) for i, t in enumerate(teams)}
        self._fitted   = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Returns np.array([p_away, p_draw, p_home]) — CLASSES=[A,D,H] order.
        Unknown teams use league-average parameters (attack=0, defence=0).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        lh = np.exp(
            self._home_adv
            + self._attacks.get(home_team, 0.0)
            + self._defences.get(away_team, 0.0)
        )
        la = np.exp(
            self._attacks.get(away_team, 0.0)
            + self._defences.get(home_team, 0.0)
        )
        return self._scoreline_probs(lh, la)

    def predict(self, home_team: str, away_team: str) -> str:
        return ["A", "D", "H"][int(np.argmax(self.predict_proba(home_team, away_team)))]

    def _scoreline_probs(self, lh: float, la: float) -> np.ndarray:
        g = self.max_goals
        p_home = p_draw = p_away = 0.0
        for i in range(g + 1):
            for j in range(g + 1):
                tau = _dc_tau_scalar(i, j, lh, la, self.rho)
                p = tau * poisson.pmf(i, lh) * poisson.pmf(j, la)
                if i > j:   p_home += p
                elif i == j: p_draw += p
                else:        p_away += p
        total = p_home + p_draw + p_away
        if total <= 0:
            return np.array([1/3, 1/3, 1/3])
        return np.array([p_away / total, p_draw / total, p_home / total])

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def top_teams(self, n: int = 10) -> list[tuple[str, float, float]]:
        """Return top N teams by (attack - defence_conceded) net strength."""
        if not self._fitted:
            return []
        teams = list(self._attacks.keys())
        ranked = sorted(teams, key=lambda t: self._attacks[t] - self._defences[t], reverse=True)
        return [(t, round(self._attacks[t], 3), round(self._defences[t], 3)) for t in ranked[:n]]
