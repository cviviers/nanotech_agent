from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from novelty_app.evaluation.run_prospective import run_prospective


class _FakeBackend:
    def __init__(self) -> None:
        self.papers_batch_calls = []

    def get_snapshot(self, snapshot_id):
        if snapshot_id in {"snap_target", "snap_source"}:
            return {"snapshot_id": snapshot_id, "metadata": {}}
        raise ValueError(f"Snapshot not found: {snapshot_id}")

    def papers_batch(self, snapshot_id, paper_ids, fields=None):
        self.papers_batch_calls.append(
            {
                "snapshot_id": snapshot_id,
                "paper_ids": list(paper_ids),
                "fields": list(fields or []),
            }
        )
        if snapshot_id == "snap_source":
            resolved = []
            aliases = {}
            for paper_id in paper_ids:
                if paper_id == "9165532":
                    aliases[paper_id] = "id:9165532__src2"
                    resolved.append("id:9165532__src2")
                else:
                    aliases[paper_id] = paper_id
                    resolved.append(paper_id)
            papers = [{"paper_id": paper_id} for paper_id in resolved]
        else:
            papers = []
            resolved = []
            aliases = {}
        return {
            "snapshot_id": snapshot_id,
            "papers": papers,
            "resolved_paper_ids": resolved,
            "paper_id_aliases": aliases,
            "unresolved_paper_ids": [],
            "ambiguous_paper_ids": {},
        }


class RunProspectiveMinimalTests(unittest.TestCase):
    def test_required_paper_validation_uses_source_snapshot_and_persists_config(self) -> None:
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "novelty_app.evaluation.run_prospective.BackendClient",
            return_value=backend,
        ):
            result = run_prospective(
                snapshot_id="snap_target",
                backend_url="http://127.0.0.1:8088",
                methods=[],
                explicit_targets=[{"target_type": "gap", "gap_id": "gap_1"}],
                output_dir=tmpdir,
                required_paper_ids=["paper_1"],
                required_paper_source_snapshot_id="snap_source",
            )

        self.assertEqual(backend.papers_batch_calls[0]["snapshot_id"], "snap_source")
        self.assertEqual(result.run["config"]["required_paper_source_snapshot_id"], "snap_source")
        self.assertEqual(result.run["config"]["required_paper_ids"], ["paper_1"])

    def test_required_paper_validation_resolves_bare_aliases_to_canonical_ids(self) -> None:
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "novelty_app.evaluation.run_prospective.BackendClient",
            return_value=backend,
        ):
            result = run_prospective(
                snapshot_id="snap_target",
                backend_url="http://127.0.0.1:8088",
                methods=[],
                explicit_targets=[{"target_type": "gap", "gap_id": "gap_1"}],
                output_dir=tmpdir,
                required_paper_ids=["9165532"],
                required_paper_source_snapshot_id="snap_source",
            )

        self.assertEqual(result.run["config"]["required_paper_ids"], ["id:9165532__src2"])


if __name__ == "__main__":
    unittest.main()
