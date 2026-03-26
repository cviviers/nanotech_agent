import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from assement_app.review_logic import build_assessment_record, filter_ideas, model_context_visible, next_incomplete_idea_id
from assement_app.workbook_store import get_assessment, load_or_create_workbook, save_assessment
from novelty_app.evaluation.assessment_bundle import (
    ASSESSMENT_BUNDLE_SCHEMA_VERSION,
    ASSESSMENT_RUBRIC,
    bundle_hash,
    load_assessment_bundle,
    load_assessment_bundle_text,
)


def _sample_bundle() -> dict:
    bundle = {
        "schema_version": ASSESSMENT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": "assessment_test_bundle",
        "created_at": "2026-03-25T00:00:00+00:00",
        "source_kind": "retrospective_run",
        "rubric": ASSESSMENT_RUBRIC,
        "run_manifest": {"run_id": "run_1", "snapshot_id": "snap_1"},
        "ideas": [
            {
                "idea_id": "idea_1",
                "is_review_packet_winner": True,
                "winner_task_count": 1,
                "run_context": {
                    "run_id": "run_1",
                    "snapshot_id": "snap_1",
                    "method_name": "orchestrator",
                    "seed": 0,
                    "target_id": "cluster_pair_0_1",
                    "hypothesis_id": "H1",
                    "queue_sort_key": ["run_1", "orchestrator", 0, "cluster_pair_0_1", "H1"],
                },
                "target": {
                    "target_id": "cluster_pair_0_1",
                    "target_type": "cluster_pair",
                    "effective_target": {"target_type": "cluster_pair", "cluster_a": 0, "cluster_b": 1},
                },
                "discovery_cue": {"text": "focus on liposomes"},
                "ideation_context": {
                    "effective_target": {"target_type": "cluster_pair", "cluster_a": 0, "cluster_b": 1},
                    "evidence_papers": [{"paper_id": "p1", "title": "Paper 1", "abstract": "text"}],
                    "explanation": {"cluster_A_summary": {"one_line": "A"}},
                    "audit": {"supported_claim_fraction": 1.0},
                },
                "hypothesis": {
                    "hypothesis_id": "H1",
                    "title": "Idea One",
                    "text": "First idea text",
                    "support_citations": ["p1"],
                },
                "judge_context": {
                    "idea_scores": {
                        "importance": {"score": 4},
                        "novelty": {"score": 3},
                        "plausibility": {"score": 4},
                        "feasibility": {"score": 4},
                        "evaluability": {"score": 5},
                        "likely_impact": {"score": 4},
                    }
                },
                "benchmark_context": {"evaluations": []},
            },
            {
                "idea_id": "idea_2",
                "is_review_packet_winner": False,
                "winner_task_count": 0,
                "run_context": {
                    "run_id": "run_1",
                    "snapshot_id": "snap_1",
                    "method_name": "heuristic_bridge",
                    "seed": 0,
                    "target_id": "gap_1",
                    "hypothesis_id": "H2",
                    "queue_sort_key": ["run_1", "heuristic_bridge", 0, "gap_1", "H2"],
                },
                "target": {
                    "target_id": "gap_1",
                    "target_type": "gap",
                    "effective_target": {"target_type": "gap", "gap_id": "gap_1"},
                },
                "discovery_cue": {},
                "ideation_context": {
                    "effective_target": {"target_type": "gap", "gap_id": "gap_1"},
                    "evidence_papers": [{"paper_id": "p2", "title": "Paper 2", "abstract": "text"}],
                    "explanation": {},
                    "audit": {},
                },
                "hypothesis": {
                    "hypothesis_id": "H2",
                    "title": "Idea Two",
                    "text": "Second idea text",
                    "support_citations": ["p2"],
                },
                "judge_context": {"idea_scores": {}},
                "benchmark_context": {"evaluations": []},
            },
        ],
    }
    bundle["bundle_sha256"] = bundle_hash(bundle)
    return bundle


class AssessmentAppTests(unittest.TestCase):
    def test_load_assessment_bundle_rejects_legacy_review_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_review_packet.json"
            path.write_text(json.dumps({"run": {}, "rows": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assessment_bundle_v1"):
                load_assessment_bundle(path)

    def test_load_assessment_bundle_text_supports_uploaded_json(self) -> None:
        bundle = _sample_bundle()
        loaded = load_assessment_bundle_text(json.dumps(bundle))
        self.assertEqual(loaded["bundle_id"], bundle["bundle_id"])
        self.assertEqual(len(loaded["ideas"]), len(bundle["ideas"]))
        self.assertEqual(loaded["bundle_sha256"], bundle["bundle_sha256"])

    def test_workbook_roundtrip_supports_resume_and_multi_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle = _sample_bundle()
            bundle_path = tmp / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            workbook_path = tmp / "reviews.xlsx"

            workbook = load_or_create_workbook(workbook_path, bundle, bundle_path=str(bundle_path))
            draft = build_assessment_record(
                bundle_id=bundle["bundle_id"],
                reviewer_id="alice",
                idea_id="idea_1",
                values={
                    "importance": 4,
                    "novelty": 3,
                    "overall_rationale": "Promising idea.",
                    "confidence": "medium",
                },
                existing=None,
                submit=False,
                saved_at="2026-03-25T01:00:00+00:00",
            )
            save_assessment(workbook, draft, bundle=bundle, bundle_path=str(bundle_path))

            reopened = load_or_create_workbook(workbook_path, bundle, bundle_path=str(bundle_path))
            loaded = get_assessment(reopened, bundle_id=bundle["bundle_id"], reviewer_id="alice", idea_id="idea_1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["status"], "draft")
            self.assertEqual(int(loaded["revision"]), 1)

            submitted = build_assessment_record(
                bundle_id=bundle["bundle_id"],
                reviewer_id="bob",
                idea_id="idea_1",
                values={
                    "importance": 5,
                    "novelty": 4,
                    "plausibility": 4,
                    "feasibility": 4,
                    "evaluability": 5,
                    "likely_impact": 5,
                    "overall_rationale": "High quality.",
                },
                existing=None,
                submit=True,
                saved_at="2026-03-25T02:00:00+00:00",
            )
            save_assessment(reopened, submitted, bundle=bundle, bundle_path=str(bundle_path))

            reloaded = load_or_create_workbook(workbook_path, bundle, bundle_path=str(bundle_path))
            alice = get_assessment(reloaded, bundle_id=bundle["bundle_id"], reviewer_id="alice", idea_id="idea_1")
            bob = get_assessment(reloaded, bundle_id=bundle["bundle_id"], reviewer_id="bob", idea_id="idea_1")
            self.assertEqual(alice["status"], "draft")
            self.assertEqual(bob["status"], "submitted")
            self.assertTrue(model_context_visible(bob))
            self.assertFalse(model_context_visible(alice))
            self.assertFalse(reloaded.summary.empty)

    def test_filter_helpers_and_next_incomplete(self) -> None:
        bundle = _sample_bundle()
        assessments = pd.DataFrame(
            [
                {
                    "bundle_id": bundle["bundle_id"],
                    "idea_id": "idea_1",
                    "reviewer_id": "alice",
                    "status": "submitted",
                    "importance": 4,
                    "novelty": 4,
                    "plausibility": 4,
                    "feasibility": 4,
                    "evaluability": 4,
                    "likely_impact": 4,
                    "overall_rationale": "Done",
                    "needs_follow_up": False,
                    "insufficient_context": False,
                },
                {
                    "bundle_id": bundle["bundle_id"],
                    "idea_id": "idea_2",
                    "reviewer_id": "bob",
                    "status": "draft",
                    "importance": 3,
                    "overall_rationale": "Partial",
                    "needs_follow_up": True,
                    "insufficient_context": False,
                },
            ]
        )

        alice_queue = filter_ideas(bundle["ideas"], assessments, "alice", status_filter="all")
        self.assertEqual(len(alice_queue), 2)
        self.assertEqual(next_incomplete_idea_id(alice_queue, assessments, "alice"), "idea_2")

        winner_only = filter_ideas(bundle["ideas"], assessments, "alice", winner_only=True)
        self.assertEqual([idea["idea_id"] for idea in winner_only], ["idea_1"])

        bob_flagged = filter_ideas(bundle["ideas"], assessments, "bob", status_filter="flagged")
        self.assertEqual([idea["idea_id"] for idea in bob_flagged], ["idea_2"])


if __name__ == "__main__":
    unittest.main()
