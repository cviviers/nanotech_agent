from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from novelty_app.agents.snapshot_builder import build_snapshot_payload


class SnapshotBuilderTests(unittest.TestCase):
    def test_build_snapshot_payload_includes_gap_metadata(self) -> None:
        df = pd.DataFrame(
            [
                {"id": "p1", "title": "Paper 1", "abstract": "A", "publication_year": 2020, "cluster_selected": 0, "gap_score": 0.1},
                {"id": "p2", "title": "Paper 2", "abstract": "B", "publication_year": 2020, "cluster_selected": 1, "gap_score": 0.9},
                {"id": "p3", "title": "Paper 3", "abstract": "C", "publication_year": 2021, "cluster_selected": 1, "gap_score": 0.8},
            ]
        )
        x_primary = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]], dtype=float)
        payload, summary = build_snapshot_payload(
            df=df,
            gap_regions=[[1, 2]],
            selected_clustering=None,
            x_primary=x_primary,
            include_raw_rows=False,
            metadata_overrides={"split_role": "historical", "cutoff_date": "2020-12-31"},
        )
        self.assertEqual(summary["n_papers"], 3)
        self.assertEqual(len(payload["gaps"]), 1)
        self.assertEqual(payload["metadata"]["split_role"], "historical")
        self.assertEqual(payload["gap_papers"][0]["gap_id"], "gap_0")


if __name__ == "__main__":
    unittest.main()
