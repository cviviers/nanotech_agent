from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import novelty_app.agents.backend_api as backend_api
from novelty_app.agents.knowledge_store import KnowledgeStore


class BackendApiEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        backend_api.STORE = KnowledgeStore(Path(self.tmpdir.name) / "api.sqlite")
        self.client = TestClient(backend_api.app)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_evaluation_endpoints_roundtrip(self) -> None:
        resp = self.client.post(
            "/evaluations/runs",
            json={
                "run_id": "run_api",
                "snapshot_id": "snap_api",
                "created_at": "2026-03-11T00:00:00+00:00",
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
                "future_window_end": "2025-12-31",
                "method_names": ["heuristic"],
                "config": {},
                "summary": {},
                "metrics": {},
                "status": "completed",
                "observability": {
                    "provider": "langfuse",
                    "trace_id": "trace_api_run",
                    "url": "https://langfuse.example/project/p/traces/trace_api_run",
                },
            },
        )
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post(
            "/evaluations/matches/batch",
            json={
                "records": [
                    {
                        "run_id": "run_api",
                        "snapshot_id": "snap_api",
                        "target_id": "t1",
                        "target_type": "cluster_pair",
                        "method_name": "heuristic",
                        "seed": 0,
                        "hypothesis_id": "h1",
                        "recovery_label": "future_neighbor_only",
                        "historical_label": "no_match",
                        "future_neighbor_label": "partial_match",
                        "gold_future_paper_id": "f1",
                        "gold_future_title": "Future paper",
                        "gold_future_year": 2024,
                        "assigned_target_id": "t1",
                        "assigned_target_score": 0.8,
                        "gold_rank": 3,
                        "gold_reciprocal_rank": 0.333333,
                        "gold_hit_at_1": False,
                        "gold_hit_at_5": True,
                        "gold_hit_at_10": True,
                        "best_future_neighbor_paper_id": "f2",
                        "best_historical_confounder_id": None,
                        "support_citations": ["p1"],
                        "hypothesis": {"title": "x"},
                        "idea_scores": {"importance": {"score": 4}, "average_score": 4.0},
                        "fingerprint": {},
                        "historical_match": {},
                        "future_match": {},
                        "trace_ref": {
                            "provider": "langfuse",
                            "trace_id": "trace_api_match",
                            "url": "https://langfuse.example/project/p/traces/trace_api_match",
                        },
                    }
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get("/evaluations/runs").json()["runs"][0]["run_id"], "run_api")
        self.assertEqual(self.client.get("/evaluations/matches").json()["matches"][0]["hypothesis_id"], "h1")
        self.assertEqual(self.client.get("/evaluations/matches").json()["matches"][0]["idea_scores"]["importance"]["score"], 4)
        self.assertEqual(self.client.get("/evaluations/runs").json()["runs"][0]["observability"]["trace_id"], "trace_api_run")
        self.assertEqual(self.client.get("/evaluations/matches").json()["matches"][0]["trace_ref"]["trace_id"], "trace_api_match")

    def test_evidence_pack_request_accepts_discovery_cue(self) -> None:
        req = backend_api.EvidencePackRequest(
            target_type="gap",
            gap_id="gap_0",
            profile="focused_eval",
            required_paper_ids=["paper_1", "paper_2"],
            discovery_cue={
                "text": "Focus on inhaled RNA delivery",
                "soft_constraints": {"route": ["inhalation"], "payload": ["mrna"]},
            },
            cue_source_snapshot_id="snapshot_full_123",
            cue_similarity_top_k=75,
            cue_similarity_sample_n=4,
            cue_similarity_seed="run_seed",
        )
        self.assertEqual(req.discovery_cue.text, "Focus on inhaled RNA delivery")
        self.assertEqual(req.discovery_cue.soft_constraints["route"], ["inhalation"])
        self.assertEqual(req.profile, "focused_eval")
        self.assertEqual(req.required_paper_ids, ["paper_1", "paper_2"])
        self.assertEqual(req.cue_source_snapshot_id, "snapshot_full_123")
        self.assertEqual(req.cue_similarity_top_k, 75)
        self.assertEqual(req.cue_similarity_sample_n, 4)
        self.assertEqual(req.cue_similarity_seed, "run_seed")

    def test_snapshot_and_artifact_lookup_endpoints(self) -> None:
        publish_resp = self.client.post(
            "/admin/snapshots/publish",
            json={
                "snapshot_id": "snap_lookup",
                "created_at": "2026-03-11T00:00:00+00:00",
                "metadata": {"split_role": "historical", "extra": {"bundle_prefix": "retro_lookup"}},
                "papers": [],
                "clusters": [],
                "gaps": [],
                "gap_papers": [],
                "llm_analyses": [],
            },
        )
        self.assertEqual(publish_resp.status_code, 200)

        patch_resp = self.client.patch(
            "/snapshots/snap_lookup/metadata",
            json={"updates": {"extra": {"retrospective_bundle_artifact_id": "artifact_lookup"}}, "replace": False},
        )
        self.assertEqual(patch_resp.status_code, 200)
        self.assertEqual(
            patch_resp.json()["metadata"]["extra"]["retrospective_bundle_artifact_id"],
            "artifact_lookup",
        )

        get_resp = self.client.get("/snapshots/snap_lookup")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["snapshot_id"], "snap_lookup")

        artifact_resp = self.client.post(
            "/artifacts/store",
            json={
                "snapshot_id": "snap_lookup",
                "kind": "retrospective_snapshot_bundle",
                "target": {"target_type": "retrospective_snapshot_bundle", "bundle_prefix": "retro_lookup"},
                "payload": {"historical_snapshot_id": "snap_lookup"},
            },
        )
        self.assertEqual(artifact_resp.status_code, 200)
        artifact_id = artifact_resp.json()["artifact_id"]

        filtered = self.client.get("/artifacts", params={"snapshot_id": "snap_lookup", "kind": "retrospective_snapshot_bundle"})
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(len(filtered.json()["artifacts"]), 1)
        self.assertEqual(filtered.json()["artifacts"][0]["artifact_id"], artifact_id)

        fetched_artifact = self.client.get(f"/artifacts/{artifact_id}")
        self.assertEqual(fetched_artifact.status_code, 200)
        self.assertEqual(fetched_artifact.json()["artifact_id"], artifact_id)


if __name__ == "__main__":
    unittest.main()
