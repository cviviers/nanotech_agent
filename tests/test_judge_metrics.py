from __future__ import annotations

import unittest

from novelty_app.evaluation.judge import classify_recovery_match, judge_candidate_match, score_hypotheses
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

        recovery_label, _meta = classify_recovery_match(
            historical_best={"judge": {"label": "no_match"}},
            gold_rank=1,
            best_future_neighbor={"judge": {"label": "partial_match"}},
        )
        metrics = aggregate_match_metrics(
            [
                {
                    "recovery_label": recovery_label,
                    "method_name": "heuristic",
                    "seed": 0,
                    "gold_future_paper_id": "f1",
                    "gold_rank": 1,
                    "gold_reciprocal_rank": 1.0,
                    "gold_hit_at_1": True,
                    "gold_hit_at_5": True,
                    "gold_hit_at_10": True,
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
        self.assertEqual(recovery_label, "gold_recovered")
        self.assertEqual(metrics["gold_recall_at_1"], 1.0)
        self.assertEqual(metrics["n_scored_hypotheses"], 1)
        self.assertEqual(metrics["average_idea_scores"]["novelty"], 5.0)

    def test_task_aggregation_uses_cue_weighted_selection_when_cue_present(self) -> None:
        metrics = aggregate_match_metrics(
            [
                {
                    "recovery_label": "gold_recovered",
                    "method_name": "orchestrator",
                    "seed": 0,
                    "gold_future_paper_id": "f1",
                    "gold_rank": 1,
                    "gold_reciprocal_rank": 1.0,
                    "gold_hit_at_1": True,
                    "gold_hit_at_5": True,
                    "gold_hit_at_10": True,
                    "cue_score": -1.0,
                    "cue_weighted_rr": 0.0,
                    "discovery_cue": {"text": "folate liposome siRNA"},
                },
                {
                    "recovery_label": "gold_recovered",
                    "method_name": "orchestrator",
                    "seed": 0,
                    "gold_future_paper_id": "f1",
                    "gold_rank": 4,
                    "gold_reciprocal_rank": 0.25,
                    "gold_hit_at_1": False,
                    "gold_hit_at_5": True,
                    "gold_hit_at_10": True,
                    "cue_score": 1.5,
                    "cue_weighted_rr": 0.25,
                    "discovery_cue": {"text": "folate liposome siRNA"},
                },
            ]
        )

        self.assertEqual(metrics["n_hypotheses"], 2)
        self.assertEqual(metrics["n_task_evaluations"], 1)
        self.assertEqual(metrics["gold_recall_at_1"], 0.0)
        self.assertEqual(metrics["gold_recall_at_5"], 1.0)
        self.assertEqual(metrics["gold_mrr"], 0.25)
        self.assertEqual(metrics["cue_weighted_mrr"], 0.25)
        self.assertEqual(metrics["mean_hypothesis_cue_score"], 1.5)

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
