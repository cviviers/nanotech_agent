from __future__ import annotations

import unittest

from novelty_app.evaluation.judge import classify_hypothesis_match, judge_candidate_match, score_hypotheses
from novelty_app.evaluation.metrics import aggregate_match_metrics


class JudgeMetricsTests(unittest.TestCase):
    def test_judge_candidate_match_and_metrics(self) -> None:
        fingerprint = {
            "disease": ["breast"],
            "material": ["liposome"],
            "payload": ["sirna"],
            "targeting": ["folate"],
            "mechanism": ["gene silencing"],
            "model": [],
            "route": [],
            "outcome": [],
        }
        candidate = {
            "title": "Folate liposome siRNA for breast cancer",
            "abstract": "A folate liposome enables siRNA gene silencing in breast cancer.",
            "reranker_score": 0.92,
            "embedding_score": 0.81,
        }
        judged = judge_candidate_match(fingerprint, candidate)
        self.assertEqual(judged["label"], "strong_match")

        classification, _meta = classify_hypothesis_match(
            historical_best={"judge": {"label": "no_match"}},
            future_best={"judge": {"label": "strong_match"}},
            support_citations=["p1"],
            grounding_summary={"supported_claim_fraction": 1.0},
        )
        metrics = aggregate_match_metrics(
            [
                {
                    "classification": classification,
                    "method_name": "heuristic",
                    "first_future_year": 2023,
                    "idea_scores": {
                        "importance": {"score": 4},
                        "novelty": {"score": 5},
                        "plausibility": {"score": 4},
                        "feasibility": {"score": 3},
                        "evaluability": {"score": 4},
                        "likely_impact": {"score": 4},
                        "average_score": 4.0,
                    },
                }
            ]
        )
        self.assertEqual(classification, "anticipatory_strong")
        self.assertEqual(metrics["anticipatory_strong_rate"], 1.0)
        self.assertEqual(metrics["n_scored_hypotheses"], 1)
        self.assertEqual(metrics["average_idea_scores"]["novelty"], 5.0)

    def test_score_hypotheses_heuristic_fallback(self) -> None:
        scored = score_hypotheses(
            [
                {
                    "hypothesis_id": "h1",
                    "title": "Folate liposome siRNA for breast cancer",
                    "text": "Use a folate liposome to deliver siRNA in breast cancer with measurable knockdown.",
                    "support_citations": ["p1", "p2"],
                    "grounding_summary": {"supported_claim_fraction": 0.8, "n_evidence_papers": 8},
                    "idea_fingerprint": {
                        "disease": ["breast"],
                        "material": ["liposome"],
                        "payload": ["sirna"],
                        "targeting": ["folate"],
                        "mechanism": ["delivery"],
                        "model": [],
                        "route": [],
                        "outcome": ["knockdown"],
                    },
                }
            ]
        )
        self.assertIn("h1", scored)
        self.assertEqual(scored["h1"]["score_method"], "heuristic_fallback")
        self.assertGreaterEqual(scored["h1"]["importance"]["score"], 1)
        self.assertLessEqual(scored["h1"]["likely_impact"]["score"], 5)


if __name__ == "__main__":
    unittest.main()
