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
        advanced_stats_away: Optional[Dict] = None,
        home_weighted_form: Optional[Dict] = None,
        away_weighted_form: Optional[Dict] = None,
        form_comparison: Optional[Dict] = None
    ) -> float:
        """
        Calculate objective draw likelihood (0.0-1.0).

        High score = match likely to be a draw.

        Phase 6 enhancement: Uses recency-weighted metrics for better accuracy.
        Incorporates weighted PPG, momentum scores, and form comparison.

        Args:
            home_form: dict with 'points_per_game', 'goals_scored', etc.
            away_form: dict with 'points_per_game', 'goals_scored', etc.
            baseline: dict with 'home_prob', 'draw_prob', 'away_prob'
            h2h_stats: dict with 'matches_played', 'draws', etc.
            advanced_stats_home: dict with shots, corners, clean_sheets
            advanced_stats_away: dict with shots, corners, clean_sheets
            home_weighted_form: dict with weighted_points_per_game, momentum_score
            away_weighted_form: dict with weighted_points_per_game, momentum_score
            form_comparison: dict with ppg_differential, form_advantage

        Returns:
            float between 0.0 and 1.0
        """
        score = 0.0

        # Factor 1: Even form (0.35 points max) - use weighted metrics if available
        score += self._score_even_form(
            home_form, away_form,
            home_weighted_form, away_weighted_form,
            form_comparison
        )

        # Factor 2: Close baseline (0.35 points max, was 0.3)
        score += self._score_close_baseline(baseline)

        # Factor 3: Low-scoring expected (0.2 points max)
        score += self._score_low_scoring(home_form, away_form, advanced_stats_home, advanced_stats_away)

        # Factor 4: H2H draw history (0.2 points max)
        if h2h_stats:
            score += self._score_h2h_draws(h2h_stats)

        # Factor 5: Momentum similarity (0.15 points max) - NEW in Phase 6
        if home_weighted_form and away_weighted_form:
            score += self._score_momentum_similarity(home_weighted_form, away_weighted_form)

        # Note: Max possible is now 1.25 (0.35 + 0.35 + 0.2 + 0.2 + 0.15)
        # This makes it easier to reach high draw likelihood
        return min(score, 1.0)

    def _score_even_form(
        self,
        home_form: Dict,
        away_form: Dict,
        home_weighted_form: Optional[Dict] = None,
        away_weighted_form: Optional[Dict] = None,
        form_comparison: Optional[Dict] = None
    ) -> float:
        """
        Score based on how evenly matched the teams are.

        Phase 6 enhancement: Uses weighted PPG (recency-weighted) when available.
        If form_comparison is provided, uses its classification directly.

        Returns: 0.0 to 0.35
        """
        # Use form comparison classification if available (most accurate)
        if form_comparison:
            advantage = form_comparison.get('form_advantage', 'unknown')
            if advantage == 'even':
                return 0.35  # Very even match
            elif advantage in ['home_slight', 'away_slight']:
                return 0.20  # Slight advantage
            elif advantage in ['home', 'away']:
                return 0.05  # Clear advantage (but not dominant)
            # If 'unknown', fall through to manual calculation

        # Use weighted PPG if available (better than traditional PPG)
        if home_weighted_form and away_weighted_form:
            home_ppg = home_weighted_form.get('weighted_points_per_game', 1.5)
            away_ppg = away_weighted_form.get('weighted_points_per_game', 1.5)
        else:
            # Fallback to traditional PPG
            home_ppg = home_form.get('points_per_game', 1.5)
            away_ppg = away_form.get('points_per_game', 1.5)

        form_diff = abs(home_ppg - away_ppg)

        # More aggressive thresholds to catch more potential draws
        if form_diff < 0.4:
            return 0.35  # Very even
        elif form_diff < 0.8:
            return 0.20  # Somewhat even
        elif form_diff < 1.2:
            return 0.10  # Moderately matched
        else:
            return 0.0  # Clear favorite

    def _score_close_baseline(self, baseline: Dict) -> float:
        """
        Score based on how close the baseline probabilities are.

        If no clear favorite, draw more likely.

        Phase 5 fix: Bookmakers are conservative on draws but good at identifying
        close matches. If baseline is close, that's a strong draw signal.

        Returns: 0.0 to 0.35 (increased from 0.3)
        """
        home_prob = baseline.get('home_prob', 0.33)
        draw_prob = baseline.get('draw_prob', 0.33)
        away_prob = baseline.get('away_prob', 0.33)

        baseline_max = max(home_prob, draw_prob, away_prob)

        # Check if draw is already the baseline favorite
        if draw_prob == baseline_max:
            return 0.35  # Baseline already favors draw (increased)

        # Check if draw is close to the max (within 10%)
        if draw_prob >= baseline_max - 0.10:
            return 0.30  # Draw is competitive

        # More aggressive thresholds
        if baseline_max < 0.48:  # Was 0.45
            return 0.35  # No team above 48% - very close
        elif baseline_max < 0.58:  # Was 0.55
            return 0.20  # No team above 58% - close (increased from 0.15)
        elif baseline_max < 0.68:  # New tier
            return 0.10  # Moderate favorite
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

    def _score_momentum_similarity(
        self,
        home_weighted: Dict,
        away_weighted: Dict
    ) -> float:
        """
        Score based on momentum similarity between teams.

        When both teams have similar momentum (both gaining or both losing),
        draws are more likely as neither has a psychological edge.

        Returns: 0.0 to 0.15
        """
        home_momentum = home_weighted.get('momentum_score', 0)
        away_momentum = away_weighted.get('momentum_score', 0)

        momentum_diff = abs(home_momentum - away_momentum)

        # Similar momentum = more draw likely
        if momentum_diff < 0.5:
            return 0.15  # Very similar momentum
        elif momentum_diff < 1.0:
            return 0.08  # Somewhat similar
        elif momentum_diff < 1.5:
            return 0.03  # Slight difference
        else:
            return 0.0  # Clear momentum advantage

    def enforce_minimum_draw(
        self,
        probabilities: Dict[str, float],
        draw_likelihood: float,
        min_draw: float = 0.15
    ) -> Dict[str, float]:
        """
        Enforce minimum draw probability based on draw likelihood.

        Phase 5 fix: Post-processing to prevent draw suppression.
        Since draws happen in ~25% of matches, they should rarely be below 15%.

        Args:
            probabilities: Dict with 'home', 'draw', 'away' probabilities
            draw_likelihood: Score from detect_draw_likelihood()
            min_draw: Minimum draw probability (default 0.15 = 15%)

        Returns:
            Adjusted probabilities with enforced draw minimum
        """
        home = probabilities.get('home', 0.33)
        draw = probabilities.get('draw', 0.33)
        away = probabilities.get('away', 0.33)

        # Determine minimum based on draw likelihood
        if draw_likelihood >= 0.7:
            min_draw = max(min_draw, 0.35)  # High likelihood: 35% minimum
        elif draw_likelihood >= 0.5:
            min_draw = max(min_draw, 0.28)  # Elevated: 28% minimum
        elif draw_likelihood >= 0.4:
            min_draw = max(min_draw, 0.23)  # Moderate: 23% minimum
        elif draw_likelihood >= 0.3:
            min_draw = max(min_draw, 0.18)  # Some signal: 18% minimum

        # If draw is already above minimum, no adjustment needed
        if draw >= min_draw:
            return {'home': home, 'draw': draw, 'away': away}

        # Boost draw to minimum, reduce home/away proportionally
        boost_needed = min_draw - draw
        non_draw_total = home + away

        if non_draw_total > 0:
            # Reduce home and away proportionally
            reduction_ratio = (1 - min_draw) / non_draw_total
            new_home = home * reduction_ratio
            new_away = away * reduction_ratio
        else:
            # Fallback if somehow home+away = 0
            new_home = (1 - min_draw) / 2
            new_away = (1 - min_draw) / 2

        # Ensure probabilities sum to 1.0
        total = new_home + min_draw + new_away
        new_home /= total
        min_draw /= total
        new_away /= total

        return {
            'home': new_home,
            'draw': min_draw,
            'away': new_away
        }

    def get_draw_warning(self, draw_likelihood: float) -> Optional[str]:
        """
        Generate a warning message for the LLM prompt based on draw likelihood.

        Phase 5 fix: Make warnings more directive and specific about probabilities.
        We're only getting 12.5% draw accuracy - need stronger signals.

        Args:
            draw_likelihood: Score from detect_draw_likelihood()

        Returns:
            Warning string or None if no warning needed
        """
        if draw_likelihood >= 0.7:
            return (
                "🚨 CRITICAL: HIGH DRAW LIKELIHOOD DETECTED (score: {:.2f}) 🚨\n"
                "═══════════════════════════════════════════════════════════\n"
                "STRONG DRAW INDICATORS PRESENT:\n"
                "✓ Teams are evenly matched (similar form)\n"
                "✓ Bookmakers show no clear favorite\n"
                "✓ Historical draw pattern or low-scoring expected\n"
                "\n"
                "🎯 REQUIRED ACTION:\n"
                "→ Draw probability MUST be 35-50% (not 20-25%)\n"
                "→ Neither team should exceed 40% probability\n"
                "→ This is a DRAW-LIKELY match, treat it as such\n"
                "═══════════════════════════════════════════════════════════"
            ).format(draw_likelihood)
        elif draw_likelihood >= 0.5:
            return (
                "⚠️ ELEVATED DRAW LIKELIHOOD DETECTED (score: {:.2f})\n"
                "────────────────────────────────────────────────────────\n"
                "This is a CLOSE, EVEN match.\n"
                "\n"
                "🎯 RECOMMENDED APPROACH:\n"
                "→ Draw probability should be 30-40%\n"
                "→ Avoid extreme predictions (>55% for any single outcome)\n"
                "→ When teams are this evenly matched, draws are common\n"
                "────────────────────────────────────────────────────────"
            ).format(draw_likelihood)
        elif draw_likelihood >= 0.4:
            return (
                "📊 Moderately Close Match (draw score: {:.2f})\n"
                "───────────────────────────────────────────────\n"
                "⚠️ Ensure draw probability is AT LEAST 25-30%\n"
                "   (Real draw rate is ~25% league-wide)"
            ).format(draw_likelihood)
        elif draw_likelihood >= 0.3:
            return (
                "📊 Draw possible (score: {:.2f}) - Keep draw at 20-25% minimum"
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
