from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from novelty_app.agents.schemas import AnalysisConfig, GeneratedHypothesis
from novelty_app.evaluation.run_retrospective import run_retrospective


class _FakeBackend:
    def __init__(self) -> None:
        self.snapshot = None
        self.matches = []
        self.run = None

    def publish_snapshot(self, payload):
        self.snapshot = payload
        return {"snapshot_id": payload["snapshot_id"]}

    def top_gaps(self, snapshot_id=None, k=25):
        gaps = [] if self.snapshot is None else self.snapshot.get("gaps", [])
        return {"gaps": gaps[:k]}

    def list_clusters(self, snapshot_id=None, limit=100, sort="size_desc"):
        clusters = [] if self.snapshot is None else self.snapshot.get("clusters", [])
        return {"clusters": clusters[:limit]}

    def store_evaluation_matches_batch(self, records):
        self.matches.extend(records)
        return {"stored": len(records)}

    def store_evaluation_run(self, payload):
        self.run = payload
        return {"run_id": payload["run_id"]}


class _FakeQwen:
    def __init__(self, *_args, **_kwargs):
        pass

    def embed(self, texts, **_kwargs):
        out = []
        for text in texts:
            if "liposome" in text.lower():
                out.append([1.0, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out

    def rank(self, query, documents, **_kwargs):
        rows = []
        for idx, doc in enumerate(documents):
            score = 0.95 if "liposome" in doc.lower() and "folate" in doc.lower() else 0.10
            rows.append({"index": idx, "document": doc, "reranker_score": score, "embedding_score": score})
        rows.sort(key=lambda x: x["reranker_score"], reverse=True)
        return rows


class RetrospectiveMinimalTests(unittest.TestCase):
    def test_run_retrospective_with_mocked_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data = [
                {"id": "h1", "title": "Liposome breast cancer delivery", "abstract": "liposome delivery for breast cancer", "publication_year": 2019, "publication_month": 1, "publication_day": 1},
                {"id": "h2", "title": "Gold melanoma imaging", "abstract": "gold nanoparticle melanoma imaging", "publication_year": 2020, "publication_month": 6, "publication_day": 1},
                {"id": "h3", "title": "PLGA paclitaxel pancreas", "abstract": "plga paclitaxel pancreatic cancer", "publication_year": 2020, "publication_month": 7, "publication_day": 1},
                {"id": "h4", "title": "Silica inflammation oral", "abstract": "silica oral inflammation delivery", "publication_year": 2020, "publication_month": 8, "publication_day": 1},
                {"id": "f1", "title": "Folate liposome siRNA for breast cancer", "abstract": "folate liposome sirna gene silencing in breast cancer", "publication_year": 2023, "publication_month": 3, "publication_day": 1},
                {"id": "f2", "title": "Gold antibody lung imaging", "abstract": "gold antibody lung cancer imaging", "publication_year": 2024, "publication_month": 5, "publication_day": 1},
            ]
            with (tmp / "cleaned_dataset.json").open("w", encoding="utf-8") as f:
                json.dump(data, f)

            qwen = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.8, 0.2],
                    [0.0, 0.6, 0.4],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            bert = qwen.copy()
            np.save(tmp / "qwen_embeddings.npy", qwen)
            np.save(tmp / "bert_embeddings.npy", bert)

            backend = _FakeBackend()

            def fake_generation(_method_name, context):
                hyp = GeneratedHypothesis(
                    hypothesis_id=f"hyp_{context.seed}",
                    target_id="cluster_pair_0_1",
                    target_type="cluster_pair",
                    method_name="dummy",
                    model_name="dummy",
                    seed=context.seed,
                    title="Folate liposome siRNA bridge",
                    text="Use a folate liposome to deliver siRNA in breast cancer.",
                    support_citations=["h1"],
                    grounding_summary={"supported_claim_fraction": 1.0},
                    raw_hypothesis={},
                    normalized_hypothesis={},
                    idea_fingerprint={
                        "query_text": "folate liposome sirna breast cancer",
                        "disease": ["breast"],
                        "material": ["liposome"],
                        "payload": ["sirna"],
                        "targeting": ["folate"],
                        "mechanism": [],
                        "model": [],
                        "route": [],
                        "outcome": [],
                    },
                )
                return [hyp], {"effective_target": context.target}

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(tmp / "cleaned_dataset.json"),
                    data_dir=str(tmp),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="kmeans", pca_components=2),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out"),
                    discovery_cue="Focus on folate liposome siRNA approaches in breast cancer",
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.matches[0]["classification"], "anticipatory_strong")
            self.assertIsNotNone(backend.run)
            self.assertEqual(
                backend.run["config"]["discovery_cue"]["text"],
                "Focus on folate liposome siRNA approaches in breast cancer",
            )
            self.assertEqual(
                result.matches[0]["discovery_cue"]["text"],
                "Focus on folate liposome siRNA approaches in breast cancer",
            )


if __name__ == "__main__":
    unittest.main()
