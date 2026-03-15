from __future__ import annotations

import unittest

from novelty_app.evaluation.judge import classify_hypothesis_match, judge_candidate_match
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
            [{"classification": classification, "method_name": "heuristic", "first_future_year": 2023}]
        )
        self.assertEqual(classification, "anticipatory_strong")
        self.assertEqual(metrics["anticipatory_strong_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
