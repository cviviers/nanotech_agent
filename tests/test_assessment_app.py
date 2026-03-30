import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from assement_app.overlap_analysis import analyze_winner_overlap
from assement_app.review_logic import (
    build_assessment_record,
    filter_ideas,
    model_context_visible,
    next_incomplete_idea_id,
    validate_submission,
)
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


def _criterion_score_card(
    *,
    importance: int = 0,
    novelty: int = 0,
    plausibility: int = 0,
    feasibility: int = 0,
    evaluability: int = 0,
    likely_impact: int = 0,
    average_score: float | None = None,
) -> dict:
    fields = {
        "importance": importance,
        "novelty": novelty,
        "plausibility": plausibility,
        "feasibility": feasibility,
        "evaluability": evaluability,
        "likely_impact": likely_impact,
    }
    payload = {field: {"score": score} for field, score in fields.items() if score}
    if average_score is not None:
        payload["average_score"] = average_score
    return payload


def _make_overlap_idea(
    idea_id: str,
    *,
    winner: bool,
    target_id: str,
    effective_target: dict,
    title: str,
    text: str,
    queue_position: int,
    support_citations: list[str] | None = None,
    evidence_paper_ids: list[str] | None = None,
    idea_fingerprint: dict | None = None,
    average_score: float | None = None,
    criterion_scores: dict | None = None,
    winner_task_count: int = 1,
) -> dict:
    evidence_paper_ids = evidence_paper_ids or []
    judge_scores = dict(criterion_scores or {})
    if average_score is not None:
        judge_scores["average_score"] = average_score
    return {
        "idea_id": idea_id,
        "is_review_packet_winner": winner,
        "winner_task_count": winner_task_count,
        "run_context": {
            "run_id": "run_overlap",
            "snapshot_id": "snap_overlap",
            "method_name": "orchestrator",
            "seed": 0,
            "target_id": target_id,
            "hypothesis_id": idea_id.upper(),
            "queue_sort_key": ["run_overlap", "orchestrator", 0, target_id, queue_position],
        },
        "target": {
            "target_id": target_id,
            "target_type": effective_target.get("target_type"),
            "effective_target": effective_target,
        },
        "discovery_cue": {},
        "ideation_context": {
            "effective_target": effective_target,
            "evidence_papers": [
                {"paper_id": paper_id, "title": f"Paper {paper_id}", "abstract": f"Evidence for {paper_id}"}
                for paper_id in evidence_paper_ids
            ],
            "explanation": {},
            "audit": {},
        },
        "hypothesis": {
            "hypothesis_id": idea_id.upper(),
            "title": title,
            "text": text,
            "support_citations": list(support_citations or []),
            "idea_fingerprint": dict(idea_fingerprint or {}),
        },
        "judge_context": {"idea_scores": judge_scores},
        "benchmark_context": {"evaluations": []},
    }


def _overlap_test_ideas() -> list[dict]:
    shared_target = {"target_type": "cluster_pair", "cluster_a": 0, "cluster_b": 1}
    return [
        _make_overlap_idea(
            "idea_1",
            winner=True,
            target_id="cluster_pair_0_1",
            effective_target=shared_target,
            title="Folate Liposome siRNA",
            text="Liposome folate siRNA delivery for cancer cells",
            queue_position=1,
            support_citations=["p1", "p2"],
            evidence_paper_ids=["p1", "p2", "p3"],
            idea_fingerprint={
                "disease": ["cancer"],
                "material": ["liposome"],
                "payload": ["sirna"],
                "targeting": ["folate"],
                "mechanism": ["delivery"],
                "tokens": ["liposome", "folate", "sirna", "delivery", "cancer"],
            },
            average_score=4.7,
            criterion_scores=_criterion_score_card(importance=5, novelty=4, plausibility=4, feasibility=4, evaluability=5, likely_impact=4),
            winner_task_count=2,
        ),
        _make_overlap_idea(
            "idea_2",
            winner=True,
            target_id="cluster_pair_0_1",
            effective_target=shared_target,
            title="Tumor Folate Liposome",
            text="Folate-targeted liposome siRNA delivery for tumors",
            queue_position=2,
            support_citations=["p1", "p2"],
            evidence_paper_ids=["p1", "p2", "p4"],
            idea_fingerprint={
                "disease": ["cancer"],
                "material": ["liposome"],
                "payload": ["sirna"],
                "targeting": ["folate"],
                "mechanism": ["delivery"],
                "tokens": ["folate", "liposome", "sirna", "delivery", "tumor"],
            },
            average_score=4.1,
            criterion_scores=_criterion_score_card(importance=4, novelty=4, plausibility=4, feasibility=4, evaluability=4, likely_impact=4),
        ),
        _make_overlap_idea(
            "idea_3",
            winner=True,
            target_id="cluster_pair_0_1",
            effective_target=shared_target,
            title="Polymer Fibrosis Therapy",
            text="Polymer nanoparticle payload for fibrosis modulation",
            queue_position=3,
            support_citations=["p9"],
            evidence_paper_ids=["p9", "p10"],
            idea_fingerprint={
                "disease": ["fibrosis"],
                "material": ["polymer"],
                "payload": ["drug"],
                "mechanism": ["modulation"],
                "tokens": ["polymer", "payload", "fibrosis", "modulation"],
            },
            average_score=4.3,
            criterion_scores=_criterion_score_card(importance=4, novelty=4, plausibility=4, feasibility=4, evaluability=5, likely_impact=5),
        ),
        _make_overlap_idea(
            "idea_4",
            winner=False,
            target_id="cluster_pair_0_1",
            effective_target=shared_target,
            title="Background Variant",
            text="Supporting variant that should stay visible when winner overlap filtering is enabled",
            queue_position=4,
            support_citations=["p1"],
            evidence_paper_ids=["p1", "p6"],
            average_score=2.8,
        ),
    ]


class AssessmentAppTests(unittest.TestCase):
    def test_validate_submission_allows_missing_overall_rationale(self) -> None:
        record = build_assessment_record(
            bundle_id="assessment_test_bundle",
            reviewer_id="alice",
            idea_id="idea_1",
            values={
                "importance": 5,
                "novelty": 4,
                "plausibility": 4,
                "feasibility": 4,
                "evaluability": 5,
                "likely_impact": 4,
            },
            existing=None,
            submit=True,
            saved_at="2026-03-25T00:00:00+00:00",
        )

        self.assertEqual(validate_submission(record), [])

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

    def test_overlap_analysis_groups_same_target_winners_and_keeps_top_score(self) -> None:
        analysis = analyze_winner_overlap(_overlap_test_ideas(), threshold=0.68)

        self.assertEqual(analysis["winner_count"], 3)
        self.assertEqual(analysis["overlap_group_count"], 1)
        self.assertEqual(analysis["hidden_winner_ids"], ["idea_2"])
        self.assertEqual(analysis["visible_winner_ids"], ["idea_1", "idea_3"])

        overlap_group = next(group for group in analysis["groups"] if group["group_size"] > 1)
        self.assertEqual(overlap_group["representative_id"], "idea_1")
        self.assertIn("p1", overlap_group["shared_evidence_ids"])

        distant_pair = next(
            pair
            for pair in analysis["pair_scores"]
            if {pair["idea_id_a"], pair["idea_id_b"]} == {"idea_1", "idea_3"}
        )
        self.assertFalse(distant_pair["is_overlap_edge"])

    def test_overlap_filter_only_hides_redundant_winners(self) -> None:
        ideas = _overlap_test_ideas()
        analysis = analyze_winner_overlap(ideas, threshold=0.68)

        queue = filter_ideas(
            ideas,
            pd.DataFrame(),
            "alice",
            status_filter="all",
            keep_idea_ids=analysis["visible_idea_ids"],
        )
        self.assertEqual([idea["idea_id"] for idea in queue], ["idea_1", "idea_3", "idea_4"])

        winner_queue = filter_ideas(
            ideas,
            pd.DataFrame(),
            "alice",
            status_filter="all",
            winner_only=True,
            keep_idea_ids=analysis["visible_idea_ids"],
        )
        self.assertEqual([idea["idea_id"] for idea in winner_queue], ["idea_1", "idea_3"])

    def test_overlap_analysis_falls_back_when_scores_citations_or_fingerprints_are_missing(self) -> None:
        target = {"target_type": "gap", "gap_id": "gap_1"}
        ideas = [
            _make_overlap_idea(
                "idea_a",
                winner=True,
                target_id="gap_1",
                effective_target=target,
                title="Polymer vaccine delivery",
                text="Polymer vaccine delivery for tumor control",
                queue_position=1,
                support_citations=[],
                evidence_paper_ids=["p20", "p21"],
                idea_fingerprint=None,
                average_score=None,
                criterion_scores=_criterion_score_card(importance=5, novelty=4, plausibility=5, feasibility=4, evaluability=4, likely_impact=4),
            ),
            _make_overlap_idea(
                "idea_b",
                winner=True,
                target_id="gap_1",
                effective_target=target,
                title="Polymer vaccine delivery",
                text="Polymer vaccine delivery for tumor control",
                queue_position=2,
                support_citations=[],
                evidence_paper_ids=["p20", "p21"],
                idea_fingerprint=None,
                average_score=3.0,
                criterion_scores=_criterion_score_card(importance=3, novelty=3, plausibility=3, feasibility=3, evaluability=3, likely_impact=3),
            ),
        ]

        analysis = analyze_winner_overlap(ideas, threshold=0.68)
        overlap_group = next(group for group in analysis["groups"] if group["group_size"] > 1)

        self.assertEqual(analysis["hidden_winner_ids"], ["idea_b"])
        self.assertEqual(overlap_group["representative_id"], "idea_a")
        self.assertGreaterEqual(float(overlap_group["mean_overlap"]), 0.68)


if __name__ == "__main__":
    unittest.main()
