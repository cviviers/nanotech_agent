from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from embedding_models import eval as embedding_eval
from embedding_models.eval import (
    _fit_predict_torch_probe_scores,
    _multilabel_metrics,
    _resolve_probe_backend,
    _resolve_probe_device,
    _resolve_retrieval_ks,
    _scores_to_predictions,
    evaluate_linear_probe,
    evaluate_retrieval,
)


class EmbeddingEvalTests(unittest.TestCase):
    def test_resolve_retrieval_ks_includes_standard_thresholds(self) -> None:
        self.assertEqual(
            _resolve_retrieval_ks(requested_k=7, n_docs=101),
            [1, 5, 7, 10, 20, 100],
        )

    def test_auto_probe_backend_selects_torch_when_cuda_is_available(self) -> None:
        with patch("embedding_models.eval._torch_cuda_available", return_value=True):
            backend, reason = _resolve_probe_backend(
                "auto",
                n_train=10_000,
                n_classes=2_000,
                probe_device="auto",
            )

        self.assertEqual(backend, "torch")
        self.assertIn("CUDA", reason)

    def test_auto_probe_backend_falls_back_to_sklearn_without_cuda(self) -> None:
        with patch("embedding_models.eval._torch_cuda_available", return_value=False):
            backend, reason = _resolve_probe_backend(
                "auto",
                n_train=10_000,
                n_classes=2_000,
                probe_device="auto",
            )

        self.assertEqual(backend, "sgd")
        self.assertIn("SGD", reason)

    def test_auto_probe_backend_with_cuda_device_requires_visible_cuda(self) -> None:
        with patch("embedding_models.eval._torch_cuda_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "probe_device=cuda"):
                _resolve_probe_backend(
                    "auto",
                    n_train=10_000,
                    n_classes=2_000,
                    probe_device="cuda",
                )

    @unittest.skipIf(embedding_eval.torch is None, "torch is not installed")
    def test_cuda_probe_device_requires_visible_cuda(self) -> None:
        with patch("embedding_models.eval._torch_cuda_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requested CUDA"):
                _resolve_probe_device("cuda")

    @unittest.skipIf(embedding_eval.torch is None, "torch is not installed")
    def test_torch_probe_scores_are_multilabel_and_json_safe(self) -> None:
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(8, 5)).astype(np.float32)
        x_test = rng.normal(size=(4, 5)).astype(np.float32)
        y_train = np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [0, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )
        y_test = np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )

        scores, fit_meta = _fit_predict_torch_probe_scores(
            x_train,
            y_train,
            x_test,
            random_state=0,
            probe_device="cpu",
            epochs=2,
            batch_size=4,
            lr=1e-2,
            weight_decay=0.0,
            pos_weight_clip=10.0,
        )
        predictions = _scores_to_predictions(scores, 0.5)
        metrics = _multilabel_metrics(y_test, predictions, y_score=scores)

        self.assertEqual(scores.shape, (4, 3))
        self.assertEqual(predictions.shape, (4, 3))
        self.assertEqual(fit_meta["probe_device"], "cpu")
        json.dumps({**metrics, **fit_meta})
        for value in {**metrics, **fit_meta}.values():
            if isinstance(value, float):
                self.assertFalse(np.isnan(value))

    @unittest.skipIf(embedding_eval.torch is None, "torch is not installed")
    def test_evaluate_linear_probe_accepts_torch_cpu_backend(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
                [0.7, 0.3, 0.0],
                [0.0, 0.7, 0.3],
            ],
            dtype=np.float32,
        )
        df = pd.DataFrame(
            {
                "clean_keywords": [
                    ["a"],
                    ["a", "b"],
                    ["b"],
                    ["b", "c"],
                    ["c"],
                    ["a", "c"],
                    ["a", "b"],
                    ["b", "c"],
                ]
            }
        )

        metrics = evaluate_linear_probe(
            embeddings,
            df,
            min_keyword_freq=1,
            probe_backend="torch",
            probe_device="cpu",
            threshold_tuning="off",
            n_repeats=1,
            test_size=0.25,
            base_seed=0,
            torch_probe_epochs=2,
            torch_probe_batch_size=4,
            torch_probe_lr=1e-2,
            torch_probe_weight_decay=0.0,
            torch_probe_pos_weight_clip=10.0,
        )
        split = metrics["probe"]["per_split"][0]

        self.assertEqual(split["probe_backend"], "torch")
        self.assertEqual(split["probe_device"], "cpu")
        self.assertEqual(split["torch_probe_epochs"], 2)
        for mean_value in metrics["probe"]["mean"].values():
            self.assertFalse(np.isnan(mean_value))

    def test_evaluate_retrieval_reports_hit_rate_and_query_coverage(self) -> None:
        n_docs = 101
        embeddings = np.eye(n_docs, dtype=np.float32)
        embeddings = embeddings + 0.01
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        keyword_lists = [["nanotech"] for _ in range(n_docs)]
        texts = ["nanoparticle delivery study"] * n_docs

        metrics = evaluate_retrieval(
            embeddings,
            keyword_lists,
            texts,
            k=10,
            min_keyword_freq=1,
            max_queries=25,
            random_state=0,
            batch_size=8,
        )

        self.assertEqual(metrics["ks"], [1, 5, 10, 20, 100])
        self.assertEqual(metrics["query_coverage"], 1.0)
        self.assertEqual(metrics["n_queries_with_relevant_docs"], 25)
        self.assertEqual(metrics["n_queries_without_relevant_docs"], 0)
        self.assertIn("hit_rate_at_10", metrics["embedding"])
        self.assertNotIn("accuracy_at_10", metrics["embedding"])

    def test_evaluate_retrieval_counts_queries_without_relevant_neighbors(self) -> None:
        n_docs = 6
        embeddings = np.eye(n_docs, dtype=np.float32)
        keyword_lists = [[f"keyword_{idx}"] for idx in range(n_docs)]
        texts = [f"doc {idx}" for idx in range(n_docs)]

        metrics = evaluate_retrieval(
            embeddings,
            keyword_lists,
            texts,
            k=3,
            min_keyword_freq=1,
            max_queries=n_docs,
            random_state=0,
            batch_size=2,
        )

        self.assertEqual(metrics["n_queries_with_relevant_docs"], 0)
        self.assertEqual(metrics["n_queries_without_relevant_docs"], n_docs)
        self.assertEqual(metrics["query_coverage"], 0.0)
        self.assertEqual(metrics["embedding"]["mrr"], 0.0)
        self.assertEqual(metrics["embedding"]["hit_rate_at_3"], 0.0)


if __name__ == "__main__":
    unittest.main()
