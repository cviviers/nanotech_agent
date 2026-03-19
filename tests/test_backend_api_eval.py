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
                        "classification": "anticipatory_partial",
                        "historical_label": "no_match",
                        "future_label": "partial_match",
                        "support_citations": ["p1"],
                        "hypothesis": {"title": "x"},
                        "idea_scores": {"importance": {"score": 4}, "average_score": 4.0},
                        "fingerprint": {},
                        "historical_match": {},
                        "future_match": {},
                    }
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get("/evaluations/runs").json()["runs"][0]["run_id"], "run_api")
        self.assertEqual(self.client.get("/evaluations/matches").json()["matches"][0]["hypothesis_id"], "h1")
        self.assertEqual(self.client.get("/evaluations/matches").json()["matches"][0]["idea_scores"]["importance"]["score"], 4)

    def test_evidence_pack_request_accepts_discovery_cue(self) -> None:
        req = backend_api.EvidencePackRequest(
            target_type="gap",
            gap_id="gap_0",
            discovery_cue={
                "text": "Focus on inhaled RNA delivery",
                "soft_constraints": {"route": ["inhalation"], "payload": ["mrna"]},
            },
        )
        self.assertEqual(req.discovery_cue.text, "Focus on inhaled RNA delivery")
        self.assertEqual(req.discovery_cue.soft_constraints["route"], ["inhalation"])


if __name__ == "__main__":
    unittest.main()
