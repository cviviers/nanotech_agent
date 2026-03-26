import unittest
import sys
import types
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
_snapshot_retrospective_context = MODULE._snapshot_retrospective_context


class AgentConsoleHelperTests(unittest.TestCase):
    def test_has_valid_retrospective_dates_requires_ordered_window(self) -> None:
        self.assertTrue(_has_valid_retrospective_dates("2020-12-31", "2022-01-01", "2025-12-31"))
        self.assertFalse(_has_valid_retrospective_dates("2020-12-31", "2020-12-31", "2025-12-31"))
        self.assertFalse(_has_valid_retrospective_dates("2020-12-31", "2022-01-01", "2021-12-31"))

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
                "sensitivity_window_start": "2021-01-01",
                "sensitivity_window_end": "2025-12-31",
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
                "future_title_exclude": [],
                "future_abstract_exclude": [],
                "future_semantic_query": "biofilm",
                "future_semantic_threshold": 0.45,
            }
        )
        self.assertIn("--existing-snapshot-id snapshot_hist_123", command)
        self.assertNotIn("--cutoff-date", command)
        self.assertIn("--future-semantic-threshold 0.45", command)

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
                "exemplars": 8,
                "boundary": 8,
                "diverse": 0,
                "max_iters": 2,
                "gap_ids": ["gap_7"],
                "cluster_pairs": [(1, 4)],
            }
        )
        self.assertIn("--snapshot-id snapshot_live_123", command)
        self.assertIn("--gap-id gap_7", command)
        self.assertIn("--cluster-pair 1 4", command)


if __name__ == "__main__":
    unittest.main()
