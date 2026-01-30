"""
Draw Detector for ASIL Predictions

Calculates objective draw likelihood based on match factors.
Helps address the draw blindspot issue where the model fails to predict draws.

Draws occur in ~25% of Premier League matches. This module identifies matches
where draws are more likely based on:
- Even form between teams
- Close baseline probabilities
- Low-scoring expected games
- Historical draw patterns in H2H
"""

from typing import Dict, Optional


class DrawDetector:
    """
    Detect matches likely to end in draws.

    Returns a score from 0.0 to 1.0 where:
    - 0.0-0.3: Draw unlikely
    - 0.3-0.5: Normal draw probability
    - 0.5-0.7: Draw more likely than average
    - 0.7-1.0: High draw likelihood
    """

    def detect_draw_likelihood(
        self,
        home_form: Dict,
        away_form: Dict,
        baseline: Dict,
        h2h_stats: Optional[Dict] = None,
        advanced_stats_home: Optional[Dict] = None,
        advanced_stats_away: Optional[Dict] = None
    ) -> float:
        """
        Calculate objective draw likelihood (0.0-1.0).

        High score = match likely to be a draw.

        Args:
            home_form: dict with 'points_per_game', 'goals_scored', etc.
            away_form: dict with 'points_per_game', 'goals_scored', etc.
            baseline: dict with 'home_prob', 'draw_prob', 'away_prob'
            h2h_stats: dict with 'matches_played', 'draws', etc.
            advanced_stats_home: dict with shots, corners, clean_sheets
            advanced_stats_away: dict with shots, corners, clean_sheets

        Returns:
            float between 0.0 and 1.0
        """
        score = 0.0

        # Factor 1: Even form (0.3 points max)
        score += self._score_even_form(home_form, away_form)

        # Factor 2: Close baseline (0.3 points max)
        score += self._score_close_baseline(baseline)

        # Factor 3: Low-scoring expected (0.2 points max)
        score += self._score_low_scoring(home_form, away_form, advanced_stats_home, advanced_stats_away)

        # Factor 4: H2H draw history (0.2 points max)
        if h2h_stats:
            score += self._score_h2h_draws(h2h_stats)

        return min(score, 1.0)

    def _score_even_form(self, home_form: Dict, away_form: Dict) -> float:
        """
        Score based on how evenly matched the teams are.

        Returns: 0.0 to 0.3
        """
        home_ppg = home_form.get('points_per_game', 1.5)
        away_ppg = away_form.get('points_per_game', 1.5)

        form_diff = abs(home_ppg - away_ppg)

        if form_diff < 0.3:
            return 0.3  # Very even
        elif form_diff < 0.6:
            return 0.15  # Somewhat even
        else:
            return 0.0  # Clear favorite

    def _score_close_baseline(self, baseline: Dict) -> float:
        """
        Score based on how close the baseline probabilities are.

        If no clear favorite, draw more likely.

        Returns: 0.0 to 0.3
        """
        home_prob = baseline.get('home_prob', 0.33)
        draw_prob = baseline.get('draw_prob', 0.33)
        away_prob = baseline.get('away_prob', 0.33)

        baseline_max = max(home_prob, draw_prob, away_prob)

        # Also check if draw is already the baseline favorite
        if draw_prob == baseline_max:
            return 0.3  # Baseline already favors draw

        if baseline_max < 0.45:
            return 0.3  # No team above 45% - very close
        elif baseline_max < 0.55:
            return 0.15  # No team above 55% - close
        else:
            return 0.0  # Clear favorite

    def _score_low_scoring(
        self,
        home_form: Dict,
        away_form: Dict,
        advanced_home: Optional[Dict],
        advanced_away: Optional[Dict]
    ) -> float:
        """
        Score based on expected goals.

        Low-scoring games more likely to be draws.

        Returns: 0.0 to 0.2
        """
        # From form data
        home_goals = home_form.get('goals_scored', 7.5)  # Last 5 matches
        away_goals = away_form.get('goals_scored', 7.5)

        # Average per team per 5 matches
        avg_goals_per_5 = (home_goals + away_goals) / 2
        avg_goals_per_match = avg_goals_per_5 / 5

        # Check clean sheets from advanced stats
        clean_sheet_bonus = 0.0
        if advanced_home and advanced_away:
            home_cs = advanced_home.get('clean_sheets', 0)
            away_cs = advanced_away.get('clean_sheets', 0)
            # If both teams have 2+ clean sheets in last 5, defensive game expected
            if home_cs >= 2 and away_cs >= 2:
                clean_sheet_bonus = 0.05

        if avg_goals_per_match < 1.2:
            return 0.2 + clean_sheet_bonus  # Very low scoring
        elif avg_goals_per_match < 1.5:
            return 0.1 + clean_sheet_bonus  # Low scoring
        else:
            return clean_sheet_bonus  # Normal/high scoring

    def _score_h2h_draws(self, h2h_stats: Dict) -> float:
        """
        Score based on historical draws between these teams.

        Returns: 0.0 to 0.2
        """
        matches_played = h2h_stats.get('matches_played', 0)
        draws = h2h_stats.get('draws', 0)

        if matches_played == 0:
            return 0.0  # No H2H data

        draw_rate = draws / matches_played

        if draw_rate > 0.35:
            return 0.2  # High historical draw rate
        elif draw_rate > 0.25:
            return 0.1  # Above average draw rate
        else:
            return 0.0  # Normal/low draw rate

    def get_draw_warning(self, draw_likelihood: float) -> Optional[str]:
        """
        Generate a warning message for the LLM prompt based on draw likelihood.

        Args:
            draw_likelihood: Score from detect_draw_likelihood()

        Returns:
            Warning string or None if no warning needed
        """
        if draw_likelihood >= 0.7:
            return (
                "⚠️ HIGH DRAW LIKELIHOOD DETECTED (score: {:.2f})\n"
                "This match shows strong draw indicators:\n"
                "- Evenly matched teams\n"
                "- Close baseline probabilities\n"
                "- Historical draw pattern\n"
                "Consider: Draw probability should be 35-45%, not the typical 20-25%."
            ).format(draw_likelihood)
        elif draw_likelihood >= 0.5:
            return (
                "⚠️ ELEVATED DRAW LIKELIHOOD (score: {:.2f})\n"
                "This is a close match. Draw probability should be ~30-35%.\n"
                "Avoid extreme predictions (>60% for any outcome)."
            ).format(draw_likelihood)
        elif draw_likelihood >= 0.4:
            return (
                "📊 Close match detected (draw score: {:.2f})\n"
                "Ensure draw probability is at least 25-30%."
            ).format(draw_likelihood)
        else:
            return None


def test_draw_detector():
    """Test the draw detector with sample scenarios."""
    detector = DrawDetector()

    print("=" * 70)
    print(" DRAW DETECTOR TEST")
    print("=" * 70)

    # Test 1: Even match - should be high draw likelihood
    print("\nTest 1: Even Match (Man Utd vs Chelsea)")
    print("-" * 50)
    score1 = detector.detect_draw_likelihood(
        home_form={'points_per_game': 1.8, 'goals_scored': 7},
        away_form={'points_per_game': 1.7, 'goals_scored': 6},
        baseline={'home_prob': 0.38, 'draw_prob': 0.30, 'away_prob': 0.32},
        h2h_stats={'matches_played': 10, 'draws': 4}
    )
    warning1 = detector.get_draw_warning(score1)
    print(f"  Draw Likelihood: {score1:.2f}")
    print(f"  Warning: {warning1[:50] if warning1 else 'None'}...")

    # Test 2: Clear favorite - should be low draw likelihood
    print("\nTest 2: Clear Favorite (Man City vs Burnley)")
    print("-" * 50)
    score2 = detector.detect_draw_likelihood(
        home_form={'points_per_game': 2.6, 'goals_scored': 15},
        away_form={'points_per_game': 0.8, 'goals_scored': 4},
        baseline={'home_prob': 0.82, 'draw_prob': 0.12, 'away_prob': 0.06},
        h2h_stats={'matches_played': 8, 'draws': 1}
    )
    warning2 = detector.get_draw_warning(score2)
    print(f"  Draw Likelihood: {score2:.2f}")
    print(f"  Warning: {warning2[:50] if warning2 else 'None'}")

    # Test 3: Defensive match - should be elevated draw likelihood
    print("\nTest 3: Defensive Match (Burnley vs Wolves)")
    print("-" * 50)
    score3 = detector.detect_draw_likelihood(
        home_form={'points_per_game': 1.2, 'goals_scored': 4},
        away_form={'points_per_game': 1.4, 'goals_scored': 5},
        baseline={'home_prob': 0.35, 'draw_prob': 0.32, 'away_prob': 0.33},
        h2h_stats={'matches_played': 6, 'draws': 2},
        advanced_stats_home={'clean_sheets': 3},
        advanced_stats_away={'clean_sheets': 2}
    )
    warning3 = detector.get_draw_warning(score3)
    print(f"  Draw Likelihood: {score3:.2f}")
    print(f"  Warning: {warning3[:50] if warning3 else 'None'}...")

    # Test 4: High H2H draw rate
    print("\nTest 4: High H2H Draw History")
    print("-" * 50)
    score4 = detector.detect_draw_likelihood(
        home_form={'points_per_game': 1.5, 'goals_scored': 6},
        away_form={'points_per_game': 1.5, 'goals_scored': 6},
        baseline={'home_prob': 0.40, 'draw_prob': 0.28, 'away_prob': 0.32},
        h2h_stats={'matches_played': 10, 'draws': 5}  # 50% draws!
    )
    warning4 = detector.get_draw_warning(score4)
    print(f"  Draw Likelihood: {score4:.2f}")
    print(f"  Warning: {warning4[:50] if warning4 else 'None'}...")

    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (Even match):      {score1:.2f} {'✓' if score1 >= 0.5 else '✗'}")
    print(f"  Test 2 (Clear favorite):  {score2:.2f} {'✓' if score2 < 0.3 else '✗'}")
    print(f"  Test 3 (Defensive):       {score3:.2f} {'✓' if score3 >= 0.5 else '✗'}")
    print(f"  Test 4 (H2H draws):       {score4:.2f} {'✓' if score4 >= 0.6 else '✗'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_draw_detector()
