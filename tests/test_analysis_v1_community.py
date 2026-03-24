from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from novelty_app.agents.schemas import AnalysisConfig
from novelty_app.evaluation.analysis_v1 import run_analysis_v1


class AnalysisV1CommunityTests(unittest.TestCase):
    def test_run_analysis_v1_supports_reproducible_louvain_config(self) -> None:
        fake_community = types.SimpleNamespace(
            best_partition=lambda graph, **_kwargs: {idx: idx % 2 for idx in range(len(graph))}
        )
        df = pd.DataFrame(
            [
                {"id": "p1", "title": "Paper 1", "abstract": "A", "publication_year": 2020},
                {"id": "p2", "title": "Paper 2", "abstract": "B", "publication_year": 2020},
                {"id": "p3", "title": "Paper 3", "abstract": "C", "publication_year": 2021},
                {"id": "p4", "title": "Paper 4", "abstract": "D", "publication_year": 2021},
            ]
        )
        x = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        config = AnalysisConfig(
            clustering_method="leiden",
            community_detection_algorithm="louvain",
            community_resolution=1.25,
            community_graph_k=2,
            community_graph_metric="cosine",
            knn_graph_k=2,
            density_k_list=[1, 2],
            use_pca_for_analysis=False,
            random_seed=7,
        )

        with patch.dict(sys.modules, {"community": fake_community}):
            analysis = run_analysis_v1(df, x, config=config)

        self.assertEqual(analysis.selected_clustering, "leiden")
        self.assertIn("cluster_leiden", analysis.df.columns)
        self.assertEqual(analysis.analysis_config["community_detection_algorithm"], "louvain")
        self.assertEqual(analysis.analysis_config["community_resolution"], 1.25)


if __name__ == "__main__":
    unittest.main()
