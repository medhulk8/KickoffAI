"""
Confidence Calculator for ASIL Predictions

Calculate prediction confidence based on objective criteria
instead of trusting LLM self-assessment.

Factors considered:
1. Probability Spread - How decisive is the prediction?
2. Baseline Agreement - Do we agree with bookmakers?
3. KG Clarity - Do we have clear tactical insights?
4. Data Availability - Do we have enough information?
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

        Args:
            prediction: dict with home_prob, draw_prob, away_prob
            context: dict with baseline, kg_insights, web_context

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

        # Combine scores
        final_confidence = self._combine_scores(scores)

        return final_confidence

    def _score_probability_spread(self, prediction: Dict) -> float:
        """
        Score based on how confident the probabilities are.

        High spread (e.g., 80%, 12%, 8%) → confident
        Low spread (e.g., 40%, 35%, 25%) → uncertain

        Returns: 0.0 to 1.0
        """
        probs = [
            prediction.get('home_prob', 0.33),
            prediction.get('draw_prob', 0.33),
            prediction.get('away_prob', 0.33)
        ]

        max_prob = max(probs)

        if max_prob > 0.70:
            return 1.0  # Very confident
        elif max_prob > 0.60:
            return 0.7  # Moderately confident
        elif max_prob > 0.50:
            return 0.4  # Somewhat confident
        else:
            return 0.0  # Not confident

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

    def _combine_scores(self, scores: List[Tuple[str, float]]) -> str:
        """
        Combine individual scores into final confidence level.

        Args:
            scores: List of (name, score) tuples

        Returns:
            'high', 'medium', or 'low'
        """
        # Weighted average
        weights = {
            'probability_spread': 0.35,  # Most important
            'baseline_agreement': 0.30,  # Second most
            'kg_clarity': 0.20,
            'data_availability': 0.15
        }

        total_score = 0.0
        total_weight = 0.0

        for name, score in scores:
            weight = weights.get(name, 0.25)
            total_score += score * weight
            total_weight += weight

        # Normalize if not all factors present
        if total_weight > 0:
            total_score = total_score / total_weight * sum(weights.values())

        # Convert to confidence level
        if total_score >= 0.65:
            return 'high'
        elif total_score >= 0.35:
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

        Returns:
            dict with individual scores and final confidence
        """
        prob_score = self._score_probability_spread(prediction)

        baseline = context.get('baseline', {})
        baseline_score = self._score_baseline_agreement(prediction, baseline) if baseline else 0.0

        kg_insights = context.get('kg_insights', {})
        kg_score = self._score_kg_clarity(kg_insights)

        data_score = self._score_data_availability(context)

        scores = [
            ('probability_spread', prob_score),
            ('baseline_agreement', baseline_score),
            ('kg_clarity', kg_score),
            ('data_availability', data_score)
        ]

        final_confidence = self._combine_scores(scores)

        return {
            'confidence': final_confidence,
            'scores': {
                'probability_spread': prob_score,
                'baseline_agreement': baseline_score,
                'kg_clarity': kg_score,
                'data_availability': data_score
            },
            'total_score': sum(s[1] for s in scores) / len(scores)
        }


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
