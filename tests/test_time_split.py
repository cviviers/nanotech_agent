from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from novelty_app.evaluation.time_split import split_corpus_by_time


class TimeSplitTests(unittest.TestCase):
    def test_split_corpus_by_time_respects_cutoff_and_windows(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "a", "publication_year": 2020, "publication_month": 12, "publication_day": 31},
                {"title": "b", "publication_year": 2021, "publication_month": 1, "publication_day": 5},
                {"title": "c", "publication_year": 2022, "publication_month": 6, "publication_day": 1},
                {"title": "d", "publication_year": 2019, "publication_month": 2, "publication_day": 1},
            ]
        )
        embeddings = {"qwen": np.eye(4, dtype=np.float32), "bert": np.eye(4, dtype=np.float32)}
        split = split_corpus_by_time(
            df,
            embeddings,
            cutoff_date="2020-12-31",
            future_window_start="2022-01-01",
            future_window_end="2025-12-31",
            sensitivity_window_start="2021-01-01",
            sensitivity_window_end="2025-12-31",
        )
        self.assertEqual(len(split.historical.df), 2)
        self.assertEqual(len(split.future.df), 1)
        self.assertEqual(len(split.sensitivity_future.df), 2)
        self.assertTrue((split.historical.df["publication_year"] <= 2020).all())


if __name__ == "__main__":
    unittest.main()
