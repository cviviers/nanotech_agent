from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from novelty_app.agents.corpus_manifest import (
    build_frontend_corpus_manifest,
    hash_paper_ids,
    reconstruct_positions_from_manifest,
    stable_paper_ids,
)


class CorpusManifestTests(unittest.TestCase):
    def test_stable_paper_ids_cover_supported_identifier_sources(self) -> None:
        df = pd.DataFrame(
            [
                {"pmid": "1001", "title": "A", "source_row_index": 0},
                {"id": "row_b", "title": "B", "source_row_index": 1},
                {"paper_id": "paper_c", "title": "C", "source_row_index": 2},
                {"doi": "10.1000/example", "title": "D", "source_row_index": 3},
                {"title": "Fallback", "abstract": "Only title/abstract", "publication_year": 2024, "source_row_index": 4},
            ]
        )

        paper_ids = stable_paper_ids(df)
        self.assertEqual(paper_ids[0], "pmid:1001__src0")
        self.assertEqual(paper_ids[1], "id:row_b__src1")
        self.assertEqual(paper_ids[2], "paper_id:paper_c__src2")
        self.assertEqual(paper_ids[3], "doi:10.1000/example__src3")
        self.assertEqual(paper_ids[4], "source_row:4")

    def test_manifest_roundtrip_reconstructs_filtered_frontend_corpus(self) -> None:
        raw_df = pd.DataFrame(
            [
                {"id": "p1", "title": "Paper 1", "publication_year": 2019},
                {"id": "p2", "title": "Paper 2", "publication_year": 2020},
                {"id": "p3", "title": "Paper 3", "publication_year": 2021},
                {"id": "p4", "title": "Paper 4", "publication_year": 2022},
            ]
        )
        raw_df["source_row_index"] = np.arange(len(raw_df), dtype=int)

        frontend_df = raw_df.iloc[[0, 2, 3]].reset_index(drop=True).copy()
        manifest = build_frontend_corpus_manifest(
            frontend_df,
            sample_n=3,
            random_seed=42,
            title_exclusion_keywords=["review"],
            abstract_exclusion_keywords=["overview"],
            embedding_source="qwen",
            available_embeddings=["qwen", "bert"],
            data_json="data/cleaned_dataset.json",
            data_dir="data",
        )

        positions = reconstruct_positions_from_manifest(raw_df, manifest["retained_paper_ids"])
        reconstructed_df = raw_df.iloc[positions].reset_index(drop=True).copy()

        self.assertEqual(positions, [0, 2, 3])
        self.assertEqual(stable_paper_ids(reconstructed_df), manifest["retained_paper_ids"])
        self.assertEqual(hash_paper_ids(stable_paper_ids(reconstructed_df)), manifest["retained_paper_id_hash"])


if __name__ == "__main__":
    unittest.main()
