"""
Confidence Calculator for KickoffAI Predictions

Calculate prediction confidence based on objective criteria
instead of trusting LLM self-assessment.

Factors considered:
1. Probability Spread - How decisive is the prediction?
2. Baseline Agreement - Do we agree with bookmakers?
3. KG Clarity - Do we have clear tactical insights?
4. Data Availability - Do we have enough information?
5. Form Strength (Phase 6) - Momentum and form dynamics
"""

from typing import Dict, List, Tuple


class ConfidenceCalculator:
    """
    Calculate prediction confidence based on objective criteria
    instead of trusting LLM self-assessment.
    """

    def calculate_confidence(
        self,
        prediction: Dict,
        context: Dict
    ) -> str:
        """
        Calculate confidence level based on multiple factors.

        Phase 6 enhancement: Now considers momentum and form strength.

        Args:
            prediction: dict with home_prob, draw_prob, away_prob
            context: dict with baseline, kg_insights, web_context,
                    home_weighted_form, away_weighted_form, form_comparison

        Returns:
            'high', 'medium', or 'low'
        """
        scores = []

        # Factor 1: Probability Spread (how decisive is prediction?)
        prob_score = self._score_probability_spread(prediction)
        scores.append(('probability_spread', prob_score))

        # Factor 2: Baseline Agreement (do we agree with bookmakers?)
        baseline = context.get('baseline', {})
        if baseline:
            baseline_score = self._score_baseline_agreement(prediction, baseline)
            scores.append(('baseline_agreement', baseline_score))

        # Factor 3: KG Clarity (do we have clear tactical insights?)
        kg_insights = context.get('kg_insights', {})
        kg_score = self._score_kg_clarity(kg_insights)
        scores.append(('kg_clarity', kg_score))

        # Factor 4: Data Availability (do we have enough information?)
        data_score = self._score_data_availability(context)
        scores.append(('data_availability', data_score))

        # Factor 5: Form Strength (NEW - Phase 6: momentum and form dynamics)
        home_weighted = context.get('home_weighted_form', {})
        away_weighted = context.get('away_weighted_form', {})
        form_comp = context.get('form_comparison', {})
        if home_weighted and away_weighted:
            form_score = self._score_form_strength(home_weighted, away_weighted, form_comp)
            scores.append(('form_strength', form_score))

        # Combine scores
        final_confidence = self._combine_scores(scores)

        return final_confidence

    def _score_probability_spread(self, prediction: Dict) -> float:
        """
        Score based on how confident the probabilities are.

        High spread (e.g., 80%, 12%, 8%) → confident
        Low spread (e.g., 40%, 35%, 25%) → uncertain

        PENALTY: If max_prob > 65% but draw_prob < 20%, score capped at 0.5
        This prevents overconfidence when predicting extreme outcomes.

        Returns: 0.0 to 1.0
        """
        probs = [
            prediction.get('home_prob', 0.33),
            prediction.get('draw_prob', 0.33),
            prediction.get('away_prob', 0.33)
        ]

        max_prob = max(probs)
        draw_prob = prediction.get('draw_prob', 0.33)

        # Base score calculation
        if max_prob > 0.70:
            score = 1.0  # Very confident
        elif max_prob > 0.60:
            score = 0.7  # Moderately confident
        elif max_prob > 0.50:
            score = 0.4  # Somewhat confident
        else:
            score = 0.0  # Not confident

        # PENALTY: Extreme predictions with low draw probability
        # Draws happen ~25% of the time - if predicting <20% draw with high
        # confidence in home/away, reduce score (likely overconfident)
        if max_prob > 0.65 and draw_prob < 0.20:
            score = min(score, 0.5)  # Cap at moderate confidence

        return score

    def _score_baseline_agreement(self, prediction: Dict, baseline: Dict) -> float:
        """
        Score based on agreement with bookmaker baseline.

        Close agreement → more confident
        Large deviation → less confident (we disagree with experts)

        Returns: 0.0 to 1.0
        """
        pred_max = max(
            prediction.get('home_prob', 0.33),
            prediction.get('draw_prob', 0.33),
            prediction.get('away_prob', 0.33)
        )

        base_max = max(
            baseline.get('home_prob', 0.33),
            baseline.get('draw_prob', 0.33),
            baseline.get('away_prob', 0.33)
        )

        # Calculate deviation
        deviation = abs(pred_max - base_max)

        if deviation < 0.10:
            return 1.0  # Strong agreement
        elif deviation < 0.20:
            return 0.6  # Moderate agreement
        elif deviation < 0.30:
            return 0.3  # Weak agreement
        else:
            return 0.0  # Strong disagreement

    def _score_kg_clarity(self, kg_insights: Dict) -> float:
        """
        Score based on knowledge graph tactical clarity.

        Clear tactical advantage → confident
        No tactical info → uncertain

        Returns: 0.0 to 1.0
        """
        if not kg_insights:
            return 0.0

        kg_confidence = kg_insights.get('confidence', 'none')

        if kg_confidence == 'high':
            return 1.0
        elif kg_confidence == 'medium':
            return 0.5
        else:  # low or none
            return 0.0

    def _score_data_availability(self, context: Dict) -> float:
        """
        Score based on how much information we have.

        More data sources → more confident
        Missing data → less confident

        Returns: 0.0 to 1.0
        """
        score = 0.0

        # Has baseline?
        if context.get('baseline'):
            score += 0.25

        # Has KG insights?
        kg_insights = context.get('kg_insights', {})
        if kg_insights and kg_insights.get('confidence') not in ['none', None]:
            score += 0.25

        # Has web search results?
        web_context = context.get('web_context', {})
        if web_context and len(web_context.get('results', {})) > 0:
            score += 0.25

        # Has form data?
        if context.get('home_form') and context.get('away_form'):
            score += 0.25

        return score

    def _score_form_strength(
        self,
        home_weighted: Dict,
        away_weighted: Dict,
        form_comp: Dict
    ) -> float:
        """
        Score based on momentum and form dynamics (Phase 6).

        Clear momentum advantage → more confident
        Similar momentum → less confident (draw likely)
        Strong form differential → more confident

        Returns: 0.0 to 1.0
        """
        score = 0.0

        # Factor 1: Momentum differential (0.5 points)
        home_momentum = home_weighted.get('momentum_score', 0)
        away_momentum = away_weighted.get('momentum_score', 0)
        momentum_diff = abs(home_momentum - away_momentum)

        if momentum_diff >= 2.0:
            score += 0.5  # Clear momentum advantage (e.g., 3.0 vs 0.5)
        elif momentum_diff >= 1.5:
            score += 0.35  # Strong momentum advantage
        elif momentum_diff >= 1.0:
            score += 0.20  # Moderate momentum advantage
        elif momentum_diff >= 0.5:
            score += 0.10  # Slight momentum advantage
        else:
            score += 0.0  # Similar momentum (uncertain)

        # Factor 2: Form advantage clarity (0.5 points)
        if form_comp:
            advantage = form_comp.get('form_advantage', 'even')
            ppg_diff = abs(form_comp.get('ppg_differential', 0))

            if advantage in ['home', 'away'] and ppg_diff > 0.8:
                score += 0.5  # Clear form advantage
            elif advantage in ['home', 'away']:
                score += 0.35  # Moderate form advantage
            elif advantage in ['home_slight', 'away_slight']:
                score += 0.15  # Slight form advantage
            else:  # even
                score += 0.0  # Even form (uncertain)

        return score

    def _combine_scores(self, scores: List[Tuple[str, float]]) -> str:
        """
        Combine individual scores into final confidence level.

        Phase 6: Updated weights to include form_strength factor.

        Args:
            scores: List of (name, score) tuples

        Returns:
            'high', 'medium', or 'low'
        """
        # Weighted average (adjusted in Phase 6)
        weights = {
            'probability_spread': 0.30,  # Still most important (reduced from 0.35)
            'baseline_agreement': 0.25,  # Second most (reduced from 0.30)
            'form_strength': 0.20,       # NEW - Phase 6: momentum & form
            'kg_clarity': 0.15,          # Reduced from 0.20
            'data_availability': 0.10    # Reduced from 0.15
        }

        total_score = 0.0
        total_weight = 0.0

        for name, score in scores:
            weight = weights.get(name, 0.20)
            total_score += score * weight
            total_weight += weight

        # Normalize if not all factors present
        if total_weight > 0:
            total_score = total_score / total_weight * sum(weights.values())

        # Convert to confidence level
        # TIGHTENED THRESHOLDS: Previously 0.65/0.35, now 0.75/0.45
        # This reduces overconfidence by requiring higher scores for 'high'
        if total_score >= 0.75:
            return 'high'
        elif total_score >= 0.45:
            return 'medium'
        else:
            return 'low'

    def get_confidence_breakdown(
        self,
        prediction: Dict,
        context: Dict
    ) -> Dict:
        """
        Get detailed breakdown of confidence calculation.
        (for debugging/analysis)

        Phase 6: Includes form_strength factor.

        Returns:
            dict with individual scores and final confidence
        """
        prob_score = self._score_probability_spread(prediction)

        baseline = context.get('baseline', {})
        baseline_score = self._score_baseline_agreement(prediction, baseline) if baseline else 0.0

        kg_insights = context.get('kg_insights', {})
        kg_score = self._score_kg_clarity(kg_insights)

        data_score = self._score_data_availability(context)

        # Phase 6: Add form strength
        home_weighted = context.get('home_weighted_form', {})
        away_weighted = context.get('away_weighted_form', {})
        form_comp = context.get('form_comparison', {})
        form_score = 0.0
        if home_weighted and away_weighted:
            form_score = self._score_form_strength(home_weighted, away_weighted, form_comp)

        scores = [
            ('probability_spread', prob_score),
            ('baseline_agreement', baseline_score),
            ('kg_clarity', kg_score),
            ('data_availability', data_score)
        ]

        # Add form strength if available
        if home_weighted and away_weighted:
            scores.append(('form_strength', form_score))

        final_confidence = self._combine_scores(scores)

        result = {
            'confidence': final_confidence,
            'scores': {
                'probability_spread': prob_score,
                'baseline_agreement': baseline_score,
                'kg_clarity': kg_score,
                'data_availability': data_score
            },
            'total_score': sum(s[1] for s in scores) / len(scores)
        }

        # Include form_strength in scores if available
        if home_weighted and away_weighted:
            result['scores']['form_strength'] = form_score

        return result


def test_confidence_calculator():
    """Test the confidence calculator."""
    calc = ConfidenceCalculator()

    print("=" * 70)
    print(" CONFIDENCE CALCULATOR TEST")
    print("=" * 70)

    # Test case 1: High confidence scenario
    print("\nTest 1: High Confidence (Man City 85% favorite)")
    print("-" * 50)
    result1 = calc.get_confidence_breakdown(
        prediction={'home_prob': 0.85, 'draw_prob': 0.10, 'away_prob': 0.05},
        context={
            'baseline': {'home_prob': 0.82, 'draw_prob': 0.12, 'away_prob': 0.06},
            'kg_insights': {'confidence': 'high'},
            'web_context': {'results': {'q1': [], 'q2': [], 'q3': []}},
            'home_form': {'wins': 5},
            'away_form': {'wins': 2}
        }
    )
    print(f"  Confidence: {result1['confidence']}")
    print(f"  Scores:")
    for k, v in result1['scores'].items():
        print(f"    - {k}: {v:.2f}")
    print(f"  Average: {result1['total_score']:.2f}")

    # Test case 2: Medium confidence scenario
    print("\nTest 2: Medium Confidence (Close match)")
    print("-" * 50)
    result2 = calc.get_confidence_breakdown(
        prediction={'home_prob': 0.45, 'draw_prob': 0.30, 'away_prob': 0.25},
        context={
            'baseline': {'home_prob': 0.48, 'draw_prob': 0.28, 'away_prob': 0.24},
            'kg_insights': {'confidence': 'medium'},
            'web_context': {'results': {'q1': [], 'q2': []}},
            'home_form': {'wins': 3},
            'away_form': {'wins': 3}
        }
    )
    print(f"  Confidence: {result2['confidence']}")
    print(f"  Scores:")
    for k, v in result2['scores'].items():
        print(f"    - {k}: {v:.2f}")
    print(f"  Average: {result2['total_score']:.2f}")

    # Test case 3: Low confidence scenario
    print("\nTest 3: Low Confidence (Disagrees with baseline, no KG)")
    print("-" * 50)
    result3 = calc.get_confidence_breakdown(
        prediction={'home_prob': 0.60, 'draw_prob': 0.25, 'away_prob': 0.15},
        context={
            'baseline': {'home_prob': 0.30, 'draw_prob': 0.35, 'away_prob': 0.35},
            'kg_insights': {'confidence': 'none'},
            'web_context': {},
            'home_form': {},
            'away_form': {}
        }
    )
    print(f"  Confidence: {result3['confidence']}")
    print(f"  Scores:")
    for k, v in result3['scores'].items():
        print(f"    - {k}: {v:.2f}")
    print(f"  Average: {result3['total_score']:.2f}")

    # Test case 4: Edge case - very uncertain prediction
    print("\nTest 4: Very Uncertain (33/33/33 split)")
    print("-" * 50)
    result4 = calc.get_confidence_breakdown(
        prediction={'home_prob': 0.34, 'draw_prob': 0.33, 'away_prob': 0.33},
        context={
            'baseline': {'home_prob': 0.35, 'draw_prob': 0.32, 'away_prob': 0.33},
            'kg_insights': {'confidence': 'low'},
            'web_context': {'results': {}},
            'home_form': {},
            'away_form': {}
        }
    )
    print(f"  Confidence: {result4['confidence']}")
    print(f"  Scores:")
    for k, v in result4['scores'].items():
        print(f"    - {k}: {v:.2f}")
    print(f"  Average: {result4['total_score']:.2f}")

    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (High scenario):   {result1['confidence'].upper()}")
    print(f"  Test 2 (Medium scenario): {result2['confidence'].upper()}")
    print(f"  Test 3 (Low scenario):    {result3['confidence'].upper()}")
    print(f"  Test 4 (Uncertain):       {result4['confidence'].upper()}")

    # Verify expected results
    tests_passed = 0
    if result1['confidence'] == 'high':
        tests_passed += 1
        print("\n  [PASS] Test 1: High confidence correctly identified")
    else:
        print(f"\n  [FAIL] Test 1: Expected 'high', got '{result1['confidence']}'")

    if result2['confidence'] == 'medium':
        tests_passed += 1
        print("  [PASS] Test 2: Medium confidence correctly identified")
    else:
        print(f"  [FAIL] Test 2: Expected 'medium', got '{result2['confidence']}'")

    if result3['confidence'] == 'low':
        tests_passed += 1
        print("  [PASS] Test 3: Low confidence correctly identified")
    else:
        print(f"  [FAIL] Test 3: Expected 'low', got '{result3['confidence']}'")

    if result4['confidence'] == 'low':
        tests_passed += 1
        print("  [PASS] Test 4: Low confidence correctly identified")
    else:
        print(f"  [FAIL] Test 4: Expected 'low', got '{result4['confidence']}'")

    print(f"\n  Total: {tests_passed}/4 tests passed")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_confidence_calculator()
