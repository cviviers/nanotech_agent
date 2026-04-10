import unittest
import sys
import types
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from datetime import datetime, timedelta


class _FakeTimestamp:
    def __init__(self, value=None, *, year=None, month=None, day=None):
        if value is not None and year is None:
            if isinstance(value, _FakeTimestamp):
                self._dt = value._dt
            else:
                self._dt = datetime.fromisoformat(str(value))
        else:
            self._dt = datetime(int(year), int(month), int(day))

    def date(self):
        return self._dt.date()

    def isoformat(self):
        return self._dt.isoformat()

    def __lt__(self, other):
        return self._dt < other._dt

    def __le__(self, other):
        return self._dt <= other._dt

    def __gt__(self, other):
        return self._dt > other._dt

    def __ge__(self, other):
        return self._dt >= other._dt

    def __add__(self, other):
        return _FakeTimestamp(self._dt + other)


fake_pandas = types.ModuleType("pandas")
fake_pandas.Timestamp = _FakeTimestamp
fake_pandas.Timedelta = lambda days=0: timedelta(days=days)
fake_pandas.isna = lambda value: value is None
sys.modules.setdefault("pandas", fake_pandas)

fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = {}
sys.modules.setdefault("streamlit", fake_streamlit)


MODULE_PATH = Path(__file__).resolve().parents[1] / "novelty_app" / "pages" / "agent_console.py"
SPEC = spec_from_file_location("agent_console_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_build_prospective_command_preview = MODULE._build_prospective_command_preview
_build_retrospective_command_preview = MODULE._build_retrospective_command_preview
_has_valid_retrospective_dates = MODULE._has_valid_retrospective_dates
_retrospective_metadata_from_state = MODULE._retrospective_metadata_from_state
_snapshot_retrospective_context = MODULE._snapshot_retrospective_context
_default_full_cue_source_snapshot_id = MODULE._default_full_cue_source_snapshot_id
_full_cue_source_snapshot_metadata = MODULE._full_cue_source_snapshot_metadata
_snapshot_metadata_with_defaults = MODULE._snapshot_metadata_with_defaults
_prioritize_cue_source_snapshots = MODULE._prioritize_cue_source_snapshots
_prioritize_required_paper_source_snapshots = MODULE._prioritize_required_paper_source_snapshots
_suggest_cue_source_snapshot_id = MODULE._suggest_cue_source_snapshot_id
_cue_source_scope_error = MODULE._cue_source_scope_error
_required_paper_source_error = MODULE._required_paper_source_error
_qwen_base_url_issue = MODULE._qwen_base_url_issue
_resolve_download_artifact = MODULE._resolve_download_artifact
_should_publish_cutoff_filtered_snapshot = MODULE._should_publish_cutoff_filtered_snapshot
_sync_snapshot_cache_after_publish = MODULE._sync_snapshot_cache_after_publish
_set_active_published_snapshot = MODULE._set_active_published_snapshot
_set_cue_source_published_snapshot = MODULE._set_cue_source_published_snapshot


class AgentConsoleHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_streamlit.session_state.clear()

    def test_has_valid_retrospective_dates_requires_ordered_window(self) -> None:
        self.assertTrue(_has_valid_retrospective_dates("2020-12-31", "2022-01-01", "2025-12-31"))
        self.assertFalse(_has_valid_retrospective_dates("2020-12-31", "2020-12-31", "2025-12-31"))
        self.assertFalse(_has_valid_retrospective_dates("2020-12-31", "2022-01-01", "2021-12-31"))

    def test_retrospective_metadata_empty_for_prospective_intent(self) -> None:
        fake_streamlit.session_state["agent_publish_snapshot_intent"] = "prospective"
        self.assertEqual(_retrospective_metadata_from_state(), {})
        with self.assertRaises(ValueError):
            _retrospective_metadata_from_state(require_enabled=True)

    def test_retrospective_metadata_requires_valid_dates_when_enabled(self) -> None:
        fake_streamlit.session_state.update(
            {
                "agent_publish_snapshot_intent": "retrospective",
                "agent_publish_split_role": "full",
                "agent_publish_cutoff_date": "2020-12-31",
                "agent_publish_future_window_start": "2022-01-01",
                "agent_publish_future_window_end": "2025-12-31",
            }
        )
        metadata = _retrospective_metadata_from_state(require_enabled=True)
        self.assertEqual(metadata["split_role"], "full")
        self.assertEqual(metadata["cutoff_date"], "2020-12-31")

        fake_streamlit.session_state["agent_publish_future_window_start"] = "2020-12-31"
        with self.assertRaises(ValueError):
            _retrospective_metadata_from_state(require_enabled=True)

    def test_historical_split_role_enforces_cutoff_filtering(self) -> None:
        self.assertTrue(_should_publish_cutoff_filtered_snapshot({"split_role": "historical"}))
        self.assertFalse(_should_publish_cutoff_filtered_snapshot({"split_role": "full"}))
        self.assertFalse(_should_publish_cutoff_filtered_snapshot({}))

    def test_snapshot_retrospective_context_detects_reusable_historical_bundle(self) -> None:
        snapshot = {
            "metadata": {
                "split_role": "historical",
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
                "future_window_end": "2025-12-31",
                "extra": {
                    "retrospective_bundle_artifact_id": "artifact_123",
                },
            }
        }
        context = _snapshot_retrospective_context(snapshot)
        self.assertTrue(context["has_dates"])
        self.assertTrue(context["can_reuse_snapshot"])
        self.assertEqual(context["split_role"], "historical")

    def test_snapshot_retrospective_context_rejects_non_historical_reuse(self) -> None:
        snapshot = {
            "metadata": {
                "split_role": "future",
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
                "future_window_end": "2025-12-31",
                "extra": {
                    "retrospective_bundle_kind": "retrospective_snapshot_bundle",
                },
            }
        }
        context = _snapshot_retrospective_context(snapshot)
        self.assertTrue(context["has_dates"])
        self.assertFalse(context["can_reuse_snapshot"])
        self.assertIn("historical snapshot", context["reuse_reason"])

    def test_retrospective_command_preview_prefers_existing_snapshot(self) -> None:
        command = _build_retrospective_command_preview(
            {
                "backend_url": "http://127.0.0.1:8088",
                "qwen_base_url": "http://127.0.0.1:8000",
                "data_json": "data/cleaned_dataset.json",
                "data_dir": "data",
                "existing_snapshot_id": "snapshot_hist_123",
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
                "future_window_end": "2025-12-31",
                "n_gap_targets": 1,
                "n_cluster_pair_targets": 1,
                "n_gold_future_papers": 5,
                "methods": ["orchestrator"],
                "seeds": 1,
                "hypotheses_per_target": 1,
                "output_dir": "data/retrospective_eval",
                "model_name": "gpt-5-mini-2025-08-07",
                "disable_leakage_check": True,
                "discovery_cue_text": "biofilm coating",
                "discovery_cue_goal": None,
                "cue_source_snapshot_id": "snapshot_full_999",
                "cue_similarity_top_k": 77,
                "cue_similarity_sample_n": 5,
                "future_title_exclude": [],
                "future_abstract_exclude": [],
                "future_semantic_query": "biofilm",
                "future_semantic_threshold": 0.45,
            }
        )
        self.assertIn("--existing-snapshot-id snapshot_hist_123", command)
        self.assertNotIn("--cutoff-date", command)
        self.assertIn("--future-semantic-threshold 0.45", command)
        self.assertNotIn("--sensitivity-window-start", command)
        self.assertNotIn("--sensitivity-window-end", command)
        self.assertIn("--cue-source-snapshot-id snapshot_full_999", command)
        self.assertIn("--cue-similarity-top-k 77", command)
        self.assertIn("--cue-similarity-sample-n 5", command)

    def test_retrospective_command_preview_omits_sensitivity_flags_without_existing_snapshot(self) -> None:
        command = _build_retrospective_command_preview(
            {
                "backend_url": "http://127.0.0.1:8088",
                "qwen_base_url": "http://127.0.0.1:8000",
                "data_json": "data/cleaned_dataset.json",
                "data_dir": "data",
                "existing_snapshot_id": None,
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
                "future_window_end": "2025-12-31",
                "n_gap_targets": 1,
                "n_cluster_pair_targets": 1,
                "n_gold_future_papers": 5,
                "methods": ["orchestrator"],
                "seeds": 1,
                "hypotheses_per_target": 1,
                "output_dir": "data/retrospective_eval",
                "model_name": "gpt-5-mini-2025-08-07",
                "disable_leakage_check": False,
                "discovery_cue_text": None,
                "discovery_cue_goal": None,
                "cue_source_snapshot_id": None,
                "cue_similarity_top_k": 77,
                "cue_similarity_sample_n": 5,
                "future_title_exclude": [],
                "future_abstract_exclude": [],
                "future_semantic_query": None,
                "future_semantic_threshold": None,
            }
        )
        self.assertIn("--cutoff-date 2020-12-31", command)
        self.assertIn("--future-window-start 2022-01-01", command)
        self.assertIn("--future-window-end 2025-12-31", command)
        self.assertNotIn("--sensitivity-window-start", command)
        self.assertNotIn("--sensitivity-window-end", command)

    def test_prospective_command_preview_includes_explicit_targets(self) -> None:
        command = _build_prospective_command_preview(
            {
                "backend_url": "http://127.0.0.1:8088",
                "snapshot_id": "snapshot_live_123",
                "methods": ["orchestrator"],
                "seeds": 1,
                "hypotheses_per_target": 1,
                "n_gap_targets": 0,
                "n_cluster_pair_targets": 0,
                "output_dir": "data/prospective_eval",
                "model_name": "gpt-5-mini-2025-08-07",
                "discovery_cue_text": "biofilm coating",
                "discovery_cue_goal": None,
                "cue_source_snapshot_id": "snapshot_full_999",
                "cue_similarity_top_k": 77,
                "cue_similarity_sample_n": 5,
                "exemplars": 8,
                "boundary": 8,
                "diverse": 0,
                "max_iters": 2,
                "gap_ids": ["gap_7"],
                "cluster_pairs": [(1, 4)],
                "required_paper_ids": ["paper_123"],
                "required_paper_source_snapshot_id": "snapshot_full_777",
            }
        )
        self.assertIn("--snapshot-id snapshot_live_123", command)
        self.assertIn("--gap-id gap_7", command)
        self.assertIn("--cluster-pair 1 4", command)
        self.assertIn("--paper-id paper_123", command)
        self.assertIn("--required-paper-source-snapshot-id snapshot_full_777", command)
        self.assertIn("--cue-source-snapshot-id snapshot_full_999", command)
        self.assertIn("--cue-similarity-top-k 77", command)
        self.assertIn("--cue-similarity-sample-n 5", command)

    def test_snapshot_metadata_defaults_include_embedding_source(self) -> None:
        fake_streamlit.session_state["config"] = {"primary_embedding": "qwen"}
        out = _snapshot_metadata_with_defaults({})
        self.assertEqual(out["embedding_source"], "qwen")
        out_existing = _snapshot_metadata_with_defaults({"embedding_source": "bert"})
        self.assertEqual(out_existing["embedding_source"], "bert")

    def test_default_full_cue_source_snapshot_id_normalizes_suffixes(self) -> None:
        fake_streamlit.session_state["agent_snapshot_id"] = "snapshot_live_123"
        self.assertEqual(_default_full_cue_source_snapshot_id(), "snapshot_live_123_full")

        for raw_value, expected in (
            ("snapshot_hist_123_historical", "snapshot_hist_123_full"),
            ("snapshot_hist_123_future", "snapshot_hist_123_full"),
            ("snapshot_hist_123_full", "snapshot_hist_123_full"),
        ):
            with self.subTest(raw_value=raw_value):
                fake_streamlit.session_state["agent_snapshot_publish_id"] = raw_value
                self.assertEqual(_default_full_cue_source_snapshot_id(), expected)

    def test_full_cue_source_snapshot_metadata_marks_snapshot_as_full(self) -> None:
        fake_streamlit.session_state["config"] = {"primary_embedding": "qwen"}
        metadata = _full_cue_source_snapshot_metadata(
            {
                "row_count": 42,
                "retained_paper_id_hash": "hash_123",
            }
        )
        self.assertEqual(metadata["split_role"], "full")
        self.assertEqual(metadata["embedding_source"], "qwen")
        self.assertEqual(metadata["extra"]["publish_mode"], "streamlit_full_cue_source")
        self.assertTrue(metadata["extra"]["cue_source_ready"])
        self.assertEqual(metadata["extra"]["source_corpus_row_count"], 42)
        self.assertEqual(metadata["extra"]["source_corpus_paper_id_hash"], "hash_123")

    def test_cue_source_prioritization_prefers_qwen_full_scope(self) -> None:
        snapshots = [
            {
                "snapshot_id": "snapshot_hist_qwen",
                "created_at": "2026-03-01T12:00:00Z",
                "metadata": {"split_role": "historical", "embedding_source": "qwen"},
            },
            {
                "snapshot_id": "snapshot_full_unknown",
                "created_at": "2026-03-02T12:00:00Z",
                "metadata": {"split_role": "full"},
            },
            {
                "snapshot_id": "snapshot_full_qwen",
                "created_at": "2026-03-03T12:00:00Z",
                "metadata": {"split_role": "full", "embedding_source": "qwen"},
            },
        ]
        prioritized = _prioritize_cue_source_snapshots(snapshots)
        self.assertEqual(prioritized[0]["snapshot_id"], "snapshot_full_qwen")
        self.assertEqual(_suggest_cue_source_snapshot_id(snapshots), "snapshot_full_qwen")

    def test_required_paper_source_prioritization_prefers_full_scope(self) -> None:
        snapshots = [
            {
                "snapshot_id": "snapshot_hist",
                "created_at": "2026-03-01T12:00:00Z",
                "metadata": {"split_role": "historical", "embedding_source": "qwen"},
            },
            {
                "snapshot_id": "snapshot_full",
                "created_at": "2026-03-02T12:00:00Z",
                "metadata": {"split_role": "full"},
            },
            {
                "snapshot_id": "snapshot_unspecified",
                "created_at": "2026-03-03T12:00:00Z",
                "metadata": {},
            },
        ]
        prioritized = _prioritize_required_paper_source_snapshots(snapshots)
        prioritized_ids = [item["snapshot_id"] for item in prioritized]
        self.assertEqual(prioritized_ids[:2], ["snapshot_full", "snapshot_unspecified"])
        self.assertEqual(prioritized_ids[2], "snapshot_hist")

    def test_cue_source_scope_validation(self) -> None:
        self.assertIn("required", str(_cue_source_scope_error("", None)))
        non_qwen_error = _cue_source_scope_error(
            "snapshot_full_bert",
            {
                "snapshot_id": "snapshot_full_bert",
                "metadata": {"split_role": "full", "embedding_source": "bert"},
            },
        )
        self.assertIn("requires `qwen` embeddings", str(non_qwen_error))
        ok_error = _cue_source_scope_error(
            "snapshot_full_qwen",
            {
                "snapshot_id": "snapshot_full_qwen",
                "metadata": {"split_role": "full", "embedding_source": "qwen"},
            },
        )
        self.assertIsNone(ok_error)

    def test_required_paper_source_validation(self) -> None:
        self.assertIn(
            "required",
            str(_required_paper_source_error(["paper_1"], "", None)),
        )
        lookup_error = _required_paper_source_error(
            ["paper_1"],
            "snapshot_full_qwen",
            None,
            "Snapshot not found: snapshot_full_qwen",
        )
        self.assertIn("lookup failed", str(lookup_error))
        self.assertIsNone(
            _required_paper_source_error(
                ["paper_1"],
                "snapshot_full_qwen",
                {"snapshot_id": "snapshot_full_qwen", "metadata": {"split_role": "full"}},
            )
        )

    def test_qwen_base_url_validation_flags_0_0_0_0(self) -> None:
        self.assertIsNotNone(_qwen_base_url_issue("http://0.0.0.0:8000"))
        self.assertIsNone(_qwen_base_url_issue("http://127.0.0.1:8000"))

    def test_resolve_download_artifact_reads_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "result_summary.json"
            expected = b'{"status":"ok"}'
            artifact_path.write_bytes(expected)
            resolved = _resolve_download_artifact(
                {
                    "run": {"run_id": "run_123"},
                    "summary_json": str(artifact_path),
                },
                payload_key="summary_json",
                fallback_suffix="summary.json",
            )
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["filename"], "result_summary.json")
        self.assertEqual(resolved["data"], expected)

    def test_resolve_download_artifact_handles_missing_path(self) -> None:
        resolved = _resolve_download_artifact(
            {"run": {"run_id": "run_123"}},
            payload_key="summary_json",
            fallback_suffix="summary.json",
        )
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["reason"], "missing_path")
        self.assertEqual(resolved["filename"], "run_123_summary.json")

    def test_resolve_download_artifact_uses_fallback_name_without_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_like_path = Path(tmpdir).anchor or str(Path(tmpdir))
            resolved = _resolve_download_artifact(
                {
                    "run": {"run_id": "run_123"},
                    "summary_json": root_like_path,
                },
                payload_key="summary_json",
                fallback_suffix="summary.json",
            )
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["reason"], "not_file")
        self.assertEqual(resolved["filename"], "run_123_summary.json")

    def test_sync_snapshot_cache_refreshes_from_backend_when_available(self) -> None:
        class _BackendOk:
            def list_snapshots(self, limit=200):
                self.limit = limit
                return {"snapshots": [{"snapshot_id": "snapshot_remote", "metadata": {"split_role": "full"}}]}

        backend = _BackendOk()
        err = _sync_snapshot_cache_after_publish(
            backend,
            [{"snapshot_id": "snapshot_local", "metadata": {"split_role": "historical"}}],
        )
        self.assertIsNone(err)
        self.assertTrue(fake_streamlit.session_state["agent_eval_snapshot_options_loaded"])
        snapshots = fake_streamlit.session_state["agent_snapshots_cache"]["snapshots"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["snapshot_id"], "snapshot_remote")
        self.assertEqual(getattr(backend, "limit", None), 200)

    def test_sync_snapshot_cache_upserts_when_refresh_fails(self) -> None:
        class _BackendFail:
            def list_snapshots(self, limit=200):
                raise RuntimeError("backend unavailable")

        fake_streamlit.session_state["agent_snapshots_cache"] = {
            "snapshots": [
                {"snapshot_id": "snapshot_local", "metadata": {"split_role": "future"}},
                {"snapshot_id": "snapshot_existing", "metadata": {}},
            ]
        }
        err = _sync_snapshot_cache_after_publish(
            _BackendFail(),
            [
                {"snapshot_id": "snapshot_local", "metadata": {"split_role": "full"}},
                {"snapshot_id": "snapshot_new", "metadata": {"split_role": "historical"}},
            ],
        )
        self.assertIsNotNone(err)
        snapshots = fake_streamlit.session_state["agent_snapshots_cache"]["snapshots"]
        snapshot_ids = [item.get("snapshot_id") for item in snapshots]
        self.assertEqual(snapshot_ids[0], "snapshot_new")
        self.assertEqual(snapshot_ids[1], "snapshot_local")
        self.assertEqual(snapshot_ids.count("snapshot_local"), 1)
        self.assertEqual(snapshots[1].get("metadata", {}).get("split_role"), "full")

    def test_set_active_published_snapshot_syncs_eval_and_cue_ids(self) -> None:
        _set_active_published_snapshot("snapshot_abc")
        self.assertEqual(fake_streamlit.session_state["agent_snapshot_id"], "snapshot_abc")
        self.assertEqual(fake_streamlit.session_state["agent_eval_snapshot_id"], "snapshot_abc")
        self.assertEqual(fake_streamlit.session_state["agent_eval_cue_source_snapshot_id"], "snapshot_abc")

    def test_set_cue_source_published_snapshot_updates_only_cue_state(self) -> None:
        fake_streamlit.session_state.update(
            {
                "agent_snapshot_id": "snapshot_active",
                "agent_eval_snapshot_id": "snapshot_eval",
                "agent_eval_cue_source_snapshot_id": "snapshot_old_cue",
            }
        )
        _set_cue_source_published_snapshot("snapshot_full")
        self.assertEqual(fake_streamlit.session_state["agent_snapshot_id"], "snapshot_active")
        self.assertEqual(fake_streamlit.session_state["agent_eval_snapshot_id"], "snapshot_eval")
        self.assertEqual(fake_streamlit.session_state["agent_eval_cue_source_snapshot_id"], "snapshot_full")
        self.assertEqual(fake_streamlit.session_state["agent_eval_cue_source_snapshot_picker"], "snapshot_full")

    def test_sync_snapshot_cache_upserts_full_cue_snapshot_record_when_refresh_fails(self) -> None:
        class _BackendFail:
            def list_snapshots(self, limit=200):
                raise RuntimeError("backend unavailable")

        fake_streamlit.session_state["config"] = {"primary_embedding": "qwen"}
        fake_streamlit.session_state["agent_snapshots_cache"] = {
            "snapshots": [
                {"snapshot_id": "snapshot_existing", "metadata": {}},
            ]
        }
        err = _sync_snapshot_cache_after_publish(
            _BackendFail(),
            [
                {
                    "snapshot_id": "snapshot_live_full",
                    "metadata": _full_cue_source_snapshot_metadata(
                        {"row_count": 12, "retained_paper_id_hash": "hash_live"}
                    ),
                }
            ],
        )
        self.assertIsNotNone(err)
        snapshots = fake_streamlit.session_state["agent_snapshots_cache"]["snapshots"]
        self.assertEqual(snapshots[0]["snapshot_id"], "snapshot_live_full")
        self.assertEqual(snapshots[0]["metadata"].get("split_role"), "full")
        self.assertTrue(snapshots[0]["metadata"].get("extra", {}).get("cue_source_ready"))


if __name__ == "__main__":
    unittest.main()
