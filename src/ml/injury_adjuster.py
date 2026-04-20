"""
Injury-based probability adjustment layer for KickoffAI.

Takes base H/D/A probabilities from the ML model and applies small,
heuristic adjustments based on team disruption scores extracted by
InjuryExtractor.

Design:
  - Post-hoc layer: does NOT retrain the model
  - Maximum shift: 6% per team (conservative — we don't know player importance)
  - Direction: disruption hurts the disrupted team's win probability
  - The lost probability is redistributed proportionally to the other two outcomes
  - Adjustments below 0.5% are suppressed (not worth showing)

Honest caveat: these are calibrated heuristics, not trained parameters.
The 6% cap is deliberately conservative. Validate on live matches before relying on it.
"""

from __future__ import annotations

import numpy as np


# Maximum probability shift per team (kept conservative)
MAX_DISRUPTION_SHIFT = 0.06


def adjust_probabilities(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_disruption: float,
    away_disruption: float,
) -> dict:
    """
    Apply injury-based adjustments to base H/D/A probabilities.

    Args:
        home_prob: Base home win probability (0-1)
        draw_prob: Base draw probability (0-1)
        away_prob: Base away win probability (0-1)
        home_disruption: Home team disruption score (0-1) from InjuryExtractor
        away_disruption: Away team disruption score (0-1) from InjuryExtractor

    Returns:
        {
          "home_prob": float,
          "draw_prob": float,
          "away_prob": float,
          "home_shift": float,  # raw adjustment applied to home_prob
          "away_shift": float,  # raw adjustment applied to away_prob
          "adjusted": bool,     # True if any shift exceeded 0.5%
          "note": str,
        }
    """
    probs = np.array([home_prob, draw_prob, away_prob], dtype=float)

    # Clamp disruption scores
    h_dis = max(0.0, min(1.0, home_disruption))
    a_dis = max(0.0, min(1.0, away_disruption))

    home_shift = -h_dis * MAX_DISRUPTION_SHIFT
    away_shift = -a_dis * MAX_DISRUPTION_SHIFT

    # Apply home disruption: reduce probs[0], redistribute to probs[1] and probs[2]
    if abs(home_shift) > 0:
        delta = home_shift  # negative
        probs[0] += delta
        # Redistribute the lost probability proportionally to draw and away
        denom = probs[1] + probs[2]
        if denom > 0:
            probs[1] -= delta * (probs[1] / denom)
            probs[2] -= delta * (probs[2] / denom)

    # Apply away disruption: reduce probs[2], redistribute to probs[0] and probs[1]
    if abs(away_shift) > 0:
        delta = away_shift  # negative
        probs[2] += delta
        denom = probs[0] + probs[1]
        if denom > 0:
            probs[0] -= delta * (probs[0] / denom)
            probs[1] -= delta * (probs[1] / denom)

    # Clamp to [0,1] and renormalize
    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    if total > 0:
        probs /= total

    adjusted = abs(home_shift) >= 0.005 or abs(away_shift) >= 0.005

    notes = []
    if abs(home_shift) >= 0.005:
        notes.append(f"home −{abs(home_shift):.1%}")
    if abs(away_shift) >= 0.005:
        notes.append(f"away −{abs(away_shift):.1%}")
    note = "Injury adj: " + ", ".join(notes) if notes else "No injury adjustment."

    return {
        "home_prob": round(float(probs[0]), 4),
        "draw_prob": round(float(probs[1]), 4),
        "away_prob": round(float(probs[2]), 4),
        "home_shift": round(home_shift, 4),
        "away_shift": round(away_shift, 4),
        "adjusted": adjusted,
        "note": note,
    }
