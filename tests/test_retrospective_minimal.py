from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from novelty_app.agents.corpus_manifest import build_frontend_corpus_manifest, hash_paper_ids, stable_paper_ids
from novelty_app.agents.schemas import AnalysisConfig, GeneratedHypothesis
from novelty_app.agents.snapshot_builder import build_snapshot_payload
from novelty_app.evaluation.analysis_v1 import run_analysis_v1
from novelty_app.evaluation.run_retrospective import (
    _apply_future_prefilter,
    _normalize_future_prefilter,
    run_retrospective,
)
from novelty_app.evaluation.time_split import load_dataset_and_embeddings, split_corpus_by_time


class _FakeBackend:
    def __init__(self) -> None:
        self.snapshot = None
        self.snapshot_record = None
        self.artifacts = []
        self.matches = []
        self.run = None

    def publish_snapshot(self, payload):
        self.snapshot = payload
        self.snapshot_record = {
            "snapshot_id": payload["snapshot_id"],
            "created_at": payload["created_at"],
            "metadata": dict(payload.get("metadata") or {}),
        }
        return {"snapshot_id": payload["snapshot_id"]}

    def get_snapshot(self, snapshot_id):
        if self.snapshot_record and self.snapshot_record["snapshot_id"] == snapshot_id:
            return dict(self.snapshot_record)
        raise ValueError(f"Snapshot not found: {snapshot_id}")

    def top_gaps(self, snapshot_id=None, k=25):
        gaps = [] if self.snapshot is None else self.snapshot.get("gaps", [])
        return {"gaps": gaps[:k]}

    def list_clusters(self, snapshot_id=None, limit=100, sort="size_desc"):
        clusters = [] if self.snapshot is None else self.snapshot.get("clusters", [])
        return {"clusters": clusters[:limit]}

    def evidence_pack(self, payload):
        papers = [] if self.snapshot is None else list(self.snapshot.get("papers", []))[:4]
        return {
            "snapshot_id": payload.get("snapshot_id"),
            "target_type": payload.get("target_type"),
            "papers": papers,
            "stats": {"requested": {"diverse": payload.get("diverse", 0)}},
            "meta": {"profile": payload.get("profile", "default")},
        }

    def store_evaluation_matches_batch(self, records):
        self.matches.extend(records)
        return {"stored": len(records)}

    def store_evaluation_run(self, payload):
        self.run = payload
        return {"run_id": payload["run_id"]}

    def list_artifacts(self, snapshot_id=None, limit=50, kind=None):
        artifacts = list(self.artifacts)
        if snapshot_id:
            artifacts = [artifact for artifact in artifacts if artifact.get("snapshot_id") == snapshot_id]
        if kind:
            artifacts = [artifact for artifact in artifacts if artifact.get("kind") == kind]
        return {"artifacts": artifacts[:limit]}

    def get_artifact(self, artifact_id):
        for artifact in self.artifacts:
            if artifact.get("artifact_id") == artifact_id:
                return dict(artifact)
        raise ValueError(f"Artifact not found: {artifact_id}")


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


class _FakeLangfuseObservation:
    def __init__(self, client, trace_id: str, observation_id: str):
        self._client = client
        self.trace_id = trace_id
        self.observation_id = observation_id
        self.updates = []

    def __enter__(self):
        self._client.stack.append(self)
        self._client.observations.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._client.stack:
            self._client.stack.pop()
        return False

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.counter = 0
        self.stack = []
        self.observations = []
        self.scores = []
        self.flushed = False

    def create_trace_id(self, *, seed=None):
        base = seed or f"trace-{self.counter + 1}"
        return hashlib.sha256(str(base).encode("utf-8")).digest()[:16].hex()

    def start_as_current_observation(self, *, trace_context=None, **_kwargs):
        trace_id = (trace_context or {}).get("trace_id")
        if not trace_id:
            trace_id = self.stack[-1].trace_id if self.stack else self.create_trace_id()
        self.counter += 1
        observation_id = f"{self.counter:016x}"
        return _FakeLangfuseObservation(self, trace_id, observation_id)

    def get_current_trace_id(self):
        return self.stack[-1].trace_id if self.stack else None

    def get_current_observation_id(self):
        return self.stack[-1].observation_id if self.stack else None

    def get_trace_url(self, *, trace_id=None):
        resolved = trace_id or self.get_current_trace_id()
        if not resolved:
            return None
        return f"https://langfuse.local/project/test/traces/{resolved}"

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed = True


def _write_fixture_dataset(tmp: Path) -> tuple[Path, Path]:
    data = [
        {
            "id": "h1",
            "title": "Liposome breast cancer delivery",
            "abstract": "liposome delivery for breast cancer",
            "publication_year": 2019,
            "publication_month": 1,
            "publication_day": 1,
        },
        {
            "id": "h2",
            "title": "Gold melanoma imaging",
            "abstract": "gold nanoparticle melanoma imaging",
            "publication_year": 2020,
            "publication_month": 6,
            "publication_day": 1,
        },
        {
            "id": "h3",
            "title": "PLGA paclitaxel pancreas",
            "abstract": "plga paclitaxel pancreatic cancer",
            "publication_year": 2020,
            "publication_month": 7,
            "publication_day": 1,
        },
        {
            "id": "h4",
            "title": "Silica inflammation oral",
            "abstract": "silica oral inflammation delivery",
            "publication_year": 2020,
            "publication_month": 8,
            "publication_day": 1,
        },
        {
            "id": "f1",
            "title": "Folate liposome siRNA for breast cancer",
            "abstract": "folate liposome sirna gene silencing in breast cancer",
            "publication_year": 2023,
            "publication_month": 3,
            "publication_day": 1,
        },
        {
            "id": "f2",
            "title": "Gold antibody lung imaging",
            "abstract": "gold antibody lung cancer imaging",
            "publication_year": 2024,
            "publication_month": 5,
            "publication_day": 1,
        },
    ]
    data_json = tmp / "cleaned_dataset.json"
    with data_json.open("w", encoding="utf-8") as f:
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
    np.save(tmp / "qwen_embeddings.npy", qwen)
    np.save(tmp / "bert_embeddings.npy", qwen.copy())
    return data_json, tmp


def _fake_generation(_method_name, context):
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
    return [hyp], {
        "effective_target": context.target,
        "evidence_pack": context.backend.evidence_pack(
            {
                "snapshot_id": context.snapshot_id,
                "target_type": context.target["target_type"],
                "cluster_a": context.target.get("cluster_a"),
                "cluster_b": context.target.get("cluster_b"),
                "gap_id": context.target.get("gap_id"),
                "profile": "focused_eval",
                "exemplars": context.exemplars,
                "boundary": context.boundary,
                "diverse": context.diverse,
            }
        ),
    }


def _prepare_existing_snapshot(
    backend: _FakeBackend,
    data_json: Path,
    data_dir: Path,
    *,
    snapshot_paper_id_hash: str | None = None,
) -> str:
    analysis_config = AnalysisConfig(clustering_method="kmeans", pca_components=2)
    df, embeddings = load_dataset_and_embeddings(str(data_json), str(data_dir), embedding_names=["qwen", "bert"])
    split = split_corpus_by_time(
        df,
        embeddings,
        cutoff_date="2020-12-31",
        future_window_start="2022-01-01",
        future_window_end="2025-12-31",
        sensitivity_window_start="2021-01-01",
        sensitivity_window_end="2025-12-31",
    )
    analysis = run_analysis_v1(split.historical.df, split.historical.embeddings["qwen"], config=analysis_config)
    manifest = build_frontend_corpus_manifest(
        df,
        sample_n=None,
        random_seed=42,
        title_exclusion_keywords=[],
        abstract_exclusion_keywords=[],
        embedding_source="qwen",
        available_embeddings=["qwen", "bert"],
        data_json=data_json,
        data_dir=data_dir,
    )
    historical_paper_ids = stable_paper_ids(analysis.df)
    future_paper_ids = stable_paper_ids(split.future.df)
    historical_hash = hash_paper_ids(historical_paper_ids)
    snapshot_id = "existing_hist_snapshot"
    artifact_id = "artifact_existing_bundle"
    payload, _summary = build_snapshot_payload(
        df=analysis.df,
        gap_regions=analysis.gap_regions,
        llm_results=None,
        selected_clustering=analysis.selected_clustering,
        x_primary=analysis.x_primary,
        x_umap_2d=analysis.x_umap_2d,
        include_raw_rows=False,
        include_embeddings=True,
        snapshot_id=snapshot_id,
        source="retrospective_eval_test",
        metadata_overrides={
            "split_role": "historical",
            "cutoff_date": "2020-12-31",
            "future_window_start": "2022-01-01",
            "future_window_end": "2025-12-31",
            "analysis_config": analysis.analysis_config,
            "analysis_config_hash": hashlib.sha256(
                json.dumps(analysis.analysis_config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "embedding_source": "qwen",
            "extra": {
                "bundle_prefix": "existing_bundle",
                "retrospective_bundle_artifact_id": artifact_id,
                "retrospective_bundle_kind": "retrospective_snapshot_bundle",
                "source_corpus_row_count": manifest["row_count"],
                "source_corpus_paper_id_hash": manifest["retained_paper_id_hash"],
                "historical_paper_count": len(historical_paper_ids),
                "historical_paper_id_hash": historical_hash,
                "snapshot_paper_count": len(historical_paper_ids),
                "snapshot_paper_id_hash": snapshot_paper_id_hash or historical_hash,
                "future_paper_count": len(future_paper_ids),
                "future_paper_id_hash": hash_paper_ids(future_paper_ids),
            },
        },
    )
    backend.snapshot = payload
    backend.snapshot_record = {
        "snapshot_id": snapshot_id,
        "created_at": payload["created_at"],
        "metadata": dict(payload["metadata"]),
    }
    backend.artifacts = [
        {
            "artifact_id": artifact_id,
            "snapshot_id": snapshot_id,
            "kind": "retrospective_snapshot_bundle",
            "created_at": payload["created_at"],
            "target": {
                "target_type": "retrospective_snapshot_bundle",
                "bundle_prefix": "existing_bundle",
                "historical_snapshot_id": snapshot_id,
                "future_snapshot_id": None,
            },
            "payload": {
                "schema_version": "retrospective_snapshot_bundle_v1",
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
                "future_window_end": "2025-12-31",
                "historical_snapshot_id": snapshot_id,
                "future_snapshot_id": None,
                "corpus_manifest": manifest,
                "analysis_config": analysis.analysis_config,
                "analysis_config_hash": hashlib.sha256(
                    json.dumps(analysis.analysis_config, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
        }
    ]
    return snapshot_id


class RetrospectiveMinimalTests(unittest.TestCase):
    def test_run_retrospective_with_mocked_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            fake_langfuse = _FakeLangfuseClient()
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ), patch(
                "novelty_app.agents.observability.get_langfuse_client",
                return_value=fake_langfuse,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="kmeans", pca_components=2),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out"),
                    discovery_cue="Focus on folate liposome siRNA approaches in breast cancer",
                    progress_callback=progress_events.append,
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.matches[0]["recovery_label"], "gold_recovered")
            self.assertIsNotNone(backend.run)
            self.assertEqual(
                backend.run["config"]["discovery_cue"]["text"],
                "Focus on folate liposome siRNA approaches in breast cancer",
            )
            self.assertEqual(
                result.matches[0]["discovery_cue"]["text"],
                "Focus on folate liposome siRNA approaches in breast cancer",
            )
            self.assertIn("idea_scores", result.matches[0])
            self.assertIn("importance", result.matches[0]["idea_scores"])
            self.assertEqual(result.run["metrics"]["n_scored_hypotheses"], 1)
            self.assertEqual(result.run["metrics"]["gold_recall_at_1"], 1.0)
            self.assertFalse(result.run["config"]["future_prefilter"]["active"])
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_before"], 2)
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_after"], 2)
            self.assertEqual(result.run["observability"]["provider"], "langfuse")
            self.assertTrue(result.matches[0]["trace_ref"]["trace_id"])
            expected_trace_id = hashlib.sha256(
                (
                    f"{result.run['run_id']}:dummy:0:{result.matches[0]['gold_future_paper_id']}:"
                    f"{result.matches[0]['assigned_target_id']}"
                ).encode("utf-8")
            ).digest()[:16].hex()
            self.assertEqual(result.matches[0]["trace_ref"]["trace_id"], expected_trace_id)
            self.assertTrue(fake_langfuse.scores)
            self.assertTrue(fake_langfuse.flushed)
            self.assertGreater(len(fake_langfuse.observations[0].updates), 1)
            self.assertEqual(progress_events[-1].phase, "completed")
            self.assertEqual(progress_events[-1].status, "completed")
            phases = list(dict.fromkeys(event.phase for event in progress_events))
            self.assertEqual(
                phases,
                [
                    "loading_inputs",
                    "preparing_snapshot",
                    "selecting_targets",
                    "building_target_pool",
                    "building_indices",
                    "selecting_gold_future_papers",
                    "evaluating_tasks",
                    "persisting_matches",
                    "aggregating_results",
                    "exporting_review_packet",
                    "completed",
                ],
            )
            evaluating_events = [event for event in progress_events if event.phase == "evaluating_tasks"]
            self.assertTrue(evaluating_events)
            self.assertEqual(evaluating_events[0].total, 1)
            self.assertEqual(evaluating_events[-1].current, 1)
            self.assertTrue(
                any(
                    update.get("output", {}).get("progress", {}).get("phase") == "completed"
                    for update in fake_langfuse.observations[0].updates
                )
            )
            review_packet = json.loads(Path(result.review_packet_json).read_text(encoding="utf-8"))
            self.assertEqual(review_packet["rows"][0]["trace_id"], expected_trace_id)
            csv_header = Path(result.review_packet_csv).read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("trace_id", csv_header)

    def test_apply_future_prefilter_filters_df_and_embeddings_in_lockstep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            df, embeddings = load_dataset_and_embeddings(str(data_json), str(data_dir), embedding_names=["qwen", "bert"])
            split = split_corpus_by_time(
                df,
                embeddings,
                cutoff_date="2020-12-31",
                future_window_start="2022-01-01",
                future_window_end="2025-12-31",
                sensitivity_window_start="2021-01-01",
                sensitivity_window_end="2025-12-31",
            )
            config = _normalize_future_prefilter(
                future_title_exclude=["gold"],
                future_abstract_exclude=[],
                future_semantic_query="liposome",
                future_semantic_threshold=0.30,
            )

            filtered_df, filtered_embeddings, stats = _apply_future_prefilter(
                future_df=split.future.df,
                future_embeddings=split.future.embeddings,
                qwen_client=_FakeQwen(),
                config=config,
            )

            self.assertEqual(stats["n_future_rows_before"], 2)
            self.assertEqual(stats["n_future_rows_after"], 1)
            self.assertEqual(len(filtered_df), 1)
            self.assertEqual(filtered_df.iloc[0]["id"], "f1")
            self.assertEqual(filtered_embeddings["qwen"].shape, (1, 3))
            self.assertTrue(np.allclose(filtered_embeddings["qwen"][0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32)))
            self.assertEqual(filtered_embeddings["bert"].shape, (1, 3))

    def test_run_retrospective_keyword_future_prefilter_reduces_future_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="kmeans", pca_components=2),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_keyword"),
                    future_title_exclude=["gold"],
                    progress_callback=progress_events.append,
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.run["config"]["future_prefilter"]["title_exclusion_keywords"], ["gold"])
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_before"], 2)
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_after"], 1)
            self.assertEqual(result.run["summary"]["n_future_pool_before_prefilter"], 2)
            self.assertEqual(result.run["summary"]["n_future_pool_after_prefilter"], 1)
            self.assertEqual(result.run["summary"]["n_future_pool"], 1)
            selecting_events = [event for event in progress_events if event.phase == "selecting_gold_future_papers"]
            self.assertTrue(any("Future prefilter applied: 2 -> 1" in event.message for event in selecting_events))
            self.assertEqual(selecting_events[-1].total, 1)

    def test_run_retrospective_semantic_future_prefilter_reduces_future_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="kmeans", pca_components=2),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_semantic"),
                    future_semantic_query="liposome",
                    future_semantic_threshold=0.30,
                    progress_callback=progress_events.append,
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.run["config"]["future_prefilter"]["semantic_query"], "liposome")
            self.assertEqual(result.run["config"]["future_prefilter"]["semantic_threshold"], 0.3)
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_after"], 1)
            selecting_events = [event for event in progress_events if event.phase == "selecting_gold_future_papers"]
            self.assertEqual(selecting_events[-1].total, 1)

    def test_run_retrospective_future_prefilter_can_filter_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="kmeans", pca_components=2),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_empty"),
                    future_title_exclude=["folate", "gold"],
                    progress_callback=progress_events.append,
                )

            self.assertEqual(result.matches, [])
            self.assertEqual(result.run["summary"]["n_gold_benchmark"], 0)
            self.assertEqual(result.run["metrics"]["n_task_evaluations"], 0)
            self.assertEqual(result.run["metrics"]["n_hypotheses"], 0)
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_after"], 0)
            evaluating_events = [event for event in progress_events if event.phase == "evaluating_tasks"]
            self.assertTrue(evaluating_events)
            self.assertEqual(evaluating_events[0].total, 0)
            self.assertIn("No evaluation tasks to run", evaluating_events[0].message)

    def test_run_retrospective_cue_reranks_without_filtering_gold_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="kmeans", pca_components=2),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_off_cue"),
                    discovery_cue="What characteristics should a coating for inorganic nanoparticles have to overcome biofilms?",
                    progress_callback=progress_events.append,
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.run["summary"]["n_gold_benchmark"], 1)
            self.assertEqual(result.run["summary"]["n_cue_filtered"], 0)
            self.assertEqual(
                result.run["summary"]["n_cue_scored"],
                result.run["summary"]["n_frontier_eligible"],
            )
            self.assertEqual(result.run["summary"]["n_cue_positive"], 0)
            selecting_events = [event for event in progress_events if event.phase == "selecting_gold_future_papers"]
            self.assertTrue(selecting_events)
            self.assertIn("cue_scored=2", selecting_events[-1].message)
            self.assertIn("cue_positive=0", selecting_events[-1].message)

    def test_run_retrospective_reuses_existing_snapshot_manifest_and_ignores_cli_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            snapshot_id = _prepare_existing_snapshot(backend, data_json, data_dir)
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    cutoff_date="1900-01-01",
                    future_window_start="1900-01-02",
                    future_window_end="1900-01-03",
                    analysis_config=AnalysisConfig(clustering_method="hdbscan", pca_components=4),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_existing"),
                    existing_snapshot_id=snapshot_id,
                    discovery_cue="Focus on folate liposome siRNA approaches in breast cancer",
                    progress_callback=progress_events.append,
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.run["snapshot_id"], snapshot_id)
            self.assertEqual(result.run["cutoff_date"], "2020-12-31")
            self.assertEqual(result.run["future_window_start"], "2022-01-01")
            self.assertEqual(result.run["future_window_end"], "2025-12-31")
            self.assertTrue(result.run["config"]["resumed_existing_snapshot"])
            self.assertEqual(result.run["config"]["analysis_config"]["clustering_method"], "kmeans")
            preparing_events = [event for event in progress_events if event.phase == "preparing_snapshot"]
            self.assertTrue(preparing_events)
            self.assertIn(snapshot_id, preparing_events[0].message)

    def test_run_retrospective_existing_snapshot_honors_future_prefilter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            snapshot_id = _prepare_existing_snapshot(backend, data_json, data_dir)
            progress_events = []

            with patch("novelty_app.evaluation.run_retrospective.QwenClient", _FakeQwen), patch(
                "novelty_app.evaluation.run_retrospective.run_generation_method",
                _fake_generation,
            ):
                result = run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    analysis_config=AnalysisConfig(clustering_method="hdbscan", pca_components=4),
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_existing_filtered"),
                    existing_snapshot_id=snapshot_id,
                    future_title_exclude=["gold"],
                    progress_callback=progress_events.append,
                )

            self.assertTrue(result.matches)
            self.assertEqual(result.run["snapshot_id"], snapshot_id)
            self.assertTrue(result.run["config"]["resumed_existing_snapshot"])
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_before"], 2)
            self.assertEqual(result.run["config"]["future_prefilter"]["n_future_rows_after"], 1)
            selecting_events = [event for event in progress_events if event.phase == "selecting_gold_future_papers"]
            self.assertEqual(selecting_events[-1].total, 1)

    def test_run_retrospective_existing_snapshot_fails_fast_on_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            snapshot_id = _prepare_existing_snapshot(
                backend,
                data_json,
                data_dir,
                snapshot_paper_id_hash="incorrect_hash",
            )

            with self.assertRaises(ValueError):
                run_retrospective(
                    backend=backend,
                    data_json=str(data_json),
                    data_dir=str(data_dir),
                    qwen_base_url="http://fake",
                    n_gap_targets=0,
                    n_cluster_pair_targets=1,
                    n_gold_future_papers=1,
                    methods=["dummy"],
                    seeds=1,
                    hypotheses_per_target=1,
                    output_dir=str(tmp / "out_invalid"),
                    existing_snapshot_id=snapshot_id,
                )

    def test_run_retrospective_failure_emits_failed_progress_and_updates_langfuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_json, data_dir = _write_fixture_dataset(tmp)
            backend = _FakeBackend()
            snapshot_id = _prepare_existing_snapshot(
                backend,
                data_json,
                data_dir,
                snapshot_paper_id_hash="incorrect_hash",
            )
            fake_langfuse = _FakeLangfuseClient()
            progress_events = []

            with patch("novelty_app.agents.observability.get_langfuse_client", return_value=fake_langfuse):
                with self.assertRaises(ValueError):
                    run_retrospective(
                        backend=backend,
                        data_json=str(data_json),
                        data_dir=str(data_dir),
                        qwen_base_url="http://fake",
                        n_gap_targets=0,
                        n_cluster_pair_targets=1,
                        n_gold_future_papers=1,
                        methods=["dummy"],
                        seeds=1,
                        hypotheses_per_target=1,
                        output_dir=str(tmp / "out_invalid_progress"),
                        existing_snapshot_id=snapshot_id,
                        progress_callback=progress_events.append,
                    )

            self.assertTrue(progress_events)
            self.assertEqual(progress_events[-1].phase, "failed")
            self.assertEqual(progress_events[-1].status, "failed")
            self.assertTrue(
                any(
                    update.get("output", {}).get("progress", {}).get("status") == "failed"
                    for update in fake_langfuse.observations[0].updates
                )
            )


if __name__ == "__main__":
    unittest.main()
