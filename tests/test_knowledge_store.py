from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from novelty_app.agents.knowledge_store import KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.sqlite"
        self.store = KnowledgeStore(self.db_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_publish_snapshot_preserves_float_scores(self) -> None:
        payload = {
            "snapshot_id": "snap_1",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "abstract": "Abstract",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": "0.75",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": "0.75", "max_gap_score": "0.75", "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": "0.75"}],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)
        top_gap = self.store.top_gaps(snapshot_id="snap_1", k=1)[0]
        self.assertEqual(top_gap["avg_gap_score"], 0.75)
        paper = self.store.papers_batch(snapshot_id="snap_1", paper_ids=["p1"])[0]
        self.assertEqual(paper["gap_score"], 0.75)

    def test_store_and_list_evaluation_records(self) -> None:
        run = {
            "run_id": "run_1",
            "snapshot_id": "snap_1",
            "created_at": "2026-03-11T00:00:00+00:00",
            "cutoff_date": "2020-12-31",
            "future_window_start": "2022-01-01",
            "future_window_end": "2025-12-31",
            "method_names": ["heuristic"],
            "config": {"x": 1},
            "summary": {"n": 1},
            "metrics": {"anticipatory_strong_rate": 1.0},
            "status": "completed",
        }
        self.store.store_evaluation_run(run)
        self.store.store_evaluation_matches_batch(
            [
                {
                    "run_id": "run_1",
                    "snapshot_id": "snap_1",
                    "target_id": "cluster_pair_1_2",
                    "target_type": "cluster_pair",
                    "method_name": "heuristic",
                    "seed": 0,
                    "hypothesis_id": "h1",
                    "classification": "anticipatory_strong",
                    "historical_label": "no_match",
                    "future_label": "strong_match",
                    "first_future_year": 2023,
                    "historical_best_paper_id": None,
                    "future_best_paper_id": "p9",
                    "support_citations": ["p1"],
                    "hypothesis": {"title": "Hypothesis"},
                    "fingerprint": {"material": ["liposome"]},
                    "historical_match": {},
                    "future_match": {"paper_id": "p9"},
                }
            ]
        )

        runs = self.store.list_evaluation_runs(limit=10)
        matches = self.store.list_evaluation_matches(run_id="run_1", limit=10)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], "run_1")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["classification"], "anticipatory_strong")

    def test_build_evidence_pack_applies_discovery_cue(self) -> None:
        payload = {
            "snapshot_id": "snap_cue",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Folate liposome siRNA for breast cancer",
                    "abstract": "folate liposome sirna gene silencing in breast cancer",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "paper_id": "p2",
                    "title": "Gold imaging in melanoma",
                    "abstract": "gold nanoparticle imaging in melanoma",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.6,
                    "embedding": [0.0, 1.0, 0.0],
                },
            ],
            "clusters": [{"cluster_id": 1, "size": 2, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 2, "avg_gap_score": 0.8, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [
                {"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.9},
                {"gap_id": "gap_0", "paper_id": "p2", "rank": 1, "gap_score": 0.6},
            ],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)

        evidence = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_cue",
                "target_type": "gap",
                "gap_id": "gap_0",
                "exemplars": 2,
                "boundary": 2,
                "diverse": 0,
                "discovery_cue": {
                    "text": "Focus on folate liposome siRNA approaches in breast cancer",
                    "soft_constraints": {
                        "material": ["liposome"],
                        "payload": ["sirna"],
                        "targeting": ["folate"],
                        "disease": ["breast"],
                    },
                },
            }
        )

        self.assertEqual(evidence["papers"][0]["paper_id"], "p1")
        self.assertEqual(evidence["meta"]["discovery_cue"]["text"], "Focus on folate liposome siRNA approaches in breast cancer")
        self.assertGreater(
            float(evidence["papers"][0]["selection_meta"]["cue_alignment"]["score"]),
            float(evidence["papers"][1]["selection_meta"]["cue_alignment"]["score"]),
        )


if __name__ == "__main__":
    unittest.main()
