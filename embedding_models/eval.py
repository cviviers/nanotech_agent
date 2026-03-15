import ast
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


PLACEHOLDER_KEYWORDS = {
    "",
    "not available",
    "not_available",
    "n/a",
    "na",
    "none",
    "nan",
}
DEFAULT_THRESHOLD_GRID = tuple(np.round(np.linspace(0.10, 0.75, 14), 2))


@dataclass
class PreparedEvaluationData:
    X: np.ndarray
    X_normalized: np.ndarray
    Y: np.ndarray
    df: pd.DataFrame
    mlb: MultiLabelBinarizer
    texts: List[str]
    metadata: Dict[str, Any]


def _parse_list_maybe(x: Any) -> List[Any]:
    """Robustly parse list-like cells that may be serialized as strings."""
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
        except Exception:
            val = None
        if isinstance(val, (list, tuple, set)):
            return list(val)
        if val is not None:
            return [val]
        if "," in x:
            return [part.strip() for part in x.split(",") if part.strip()]
        return [x.strip()]
    if pd.isna(x):
        return []
    return [x]


def clean_keywords(raw_kw: Any) -> List[str]:
    """Normalize keyword lists and drop empty or placeholder values."""
    cleaned: List[str] = []
    for keyword in _parse_list_maybe(raw_kw):
        if keyword is None or (isinstance(keyword, float) and math.isnan(keyword)):
            continue
        normalized = str(keyword).strip().lower()
        if normalized in PLACEHOLDER_KEYWORDS:
            continue
        cleaned.append(normalized)
    return sorted(set(cleaned))


def _as_2d_float32(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected embeddings to be a 2D matrix, got shape {arr.shape}."
        )
    return arr


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _first_non_empty(row: pd.Series, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column not in row:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def build_document_texts(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    text_columns_used: List[str] = []
    texts: List[str] = []
    for _, row in df.iterrows():
        title = _first_non_empty(row, ("title",))
        body = _first_non_empty(row, ("abstract", "cleaned_text", "full_text"))
        if title and "title" not in text_columns_used:
            text_columns_used.append("title")
        for body_column in ("abstract", "cleaned_text", "full_text"):
            value = row.get(body_column, None)
            if body and body_column in row and value is not None and not pd.isna(value) and str(value).strip():
                if body_column not in text_columns_used:
                    text_columns_used.append(body_column)
                break
        combined = "\n\n".join(part for part in (title, body) if part).strip()
        texts.append(combined)
    return texts, text_columns_used


def prepare_data(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    keyword_col: str = "keywords",
    min_keyword_freq: int = 20,
) -> PreparedEvaluationData:
    """
    Clean and align dataframe rows with embeddings, then build the keyword label matrix.
    """
    if keyword_col not in df.columns:
        raise KeyError(f"Column '{keyword_col}' not found in the input data.")

    print("Preparing data...")
    df = df.copy().reset_index(drop=True)
    embeddings = _as_2d_float32(embeddings)

    n_input_rows = len(df)
    n_embedding_rows = embeddings.shape[0]
    aligned_rows = min(n_input_rows, n_embedding_rows)
    alignment_trimmed = max(n_input_rows, n_embedding_rows) - aligned_rows
    if alignment_trimmed:
        print(
            f"WARNING: dataframe rows ({n_input_rows}) and embeddings ({n_embedding_rows}) "
            f"do not match. Truncating both to {aligned_rows} aligned rows."
        )
        df = df.iloc[:aligned_rows].reset_index(drop=True)
        embeddings = embeddings[:aligned_rows]

    finite_mask = np.isfinite(embeddings).all(axis=1)
    non_zero_mask = np.linalg.norm(embeddings, axis=1) > 1e-12
    valid_embedding_mask = finite_mask & non_zero_mask
    dropped_invalid_embeddings = int((~valid_embedding_mask).sum())
    if dropped_invalid_embeddings:
        print(f"Dropping {dropped_invalid_embeddings} rows with invalid embedding vectors.")
        df = df.loc[valid_embedding_mask].reset_index(drop=True)
        embeddings = embeddings[valid_embedding_mask]

    df["clean_keywords"] = df[keyword_col].apply(clean_keywords)
    non_empty_keyword_mask = df["clean_keywords"].map(len) > 0
    dropped_empty_keywords = int((~non_empty_keyword_mask).sum())
    df = df.loc[non_empty_keyword_mask].reset_index(drop=True)
    embeddings = embeddings[non_empty_keyword_mask.to_numpy()]

    keyword_counts = Counter(
        keyword
        for keyword_list in df["clean_keywords"]
        for keyword in keyword_list
    )
    allowed_keywords = {
        keyword for keyword, count in keyword_counts.items() if count >= min_keyword_freq
    }
    if not allowed_keywords:
        raise ValueError(
            f"No keywords meet min_keyword_freq={min_keyword_freq}. "
            "Lower the threshold or inspect the keyword quality."
        )

    df["filtered_keywords"] = df["clean_keywords"].apply(
        lambda keywords: [keyword for keyword in keywords if keyword in allowed_keywords]
    )
    retained_keyword_mask = df["filtered_keywords"].map(len) > 0
    dropped_low_frequency_only = int((~retained_keyword_mask).sum())
    df = df.loc[retained_keyword_mask].reset_index(drop=True)
    embeddings = embeddings[retained_keyword_mask.to_numpy()]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["filtered_keywords"])
    x_normalized = _normalize_rows(embeddings)
    texts, text_columns_used = build_document_texts(df)

    label_cardinality = y.sum(axis=1)
    metadata = {
        "n_input_rows": int(n_input_rows),
        "n_embedding_rows": int(n_embedding_rows),
        "n_docs_evaluated": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_keywords": int(y.shape[1]),
        "keyword_col": keyword_col,
        "min_keyword_freq": int(min_keyword_freq),
        "text_columns_used": text_columns_used,
        "label_cardinality_mean": float(label_cardinality.mean()),
        "label_cardinality_std": float(label_cardinality.std()),
        "label_density": float(y.mean()),
        "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
        "dropped_rows": {
            "alignment_trimmed": int(alignment_trimmed),
            "invalid_embeddings": int(dropped_invalid_embeddings),
            "empty_or_placeholder_keywords": int(dropped_empty_keywords),
            "only_low_frequency_keywords": int(dropped_low_frequency_only),
        },
    }

    return PreparedEvaluationData(
        X=embeddings,
        X_normalized=x_normalized,
        Y=y,
        df=df,
        mlb=mlb,
        texts=texts,
        metadata=metadata,
    )


def _build_linear_probe_model() -> OneVsRestClassifier:
    base_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
        ),
    )
    return OneVsRestClassifier(base_estimator, n_jobs=-1)


def _primary_labels_for_stratification(keyword_lists: Sequence[Sequence[str]]) -> List[str]:
    counts = Counter(keyword for keywords in keyword_lists for keyword in keywords)
    return [
        min(keywords, key=lambda keyword: (counts[keyword], keyword))
        for keywords in keyword_lists
    ]


def _can_stratify(labels: Sequence[str], test_size: float) -> bool:
    if not labels:
        return False
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return False
    n_test = int(math.ceil(len(labels) * test_size))
    n_train = len(labels) - n_test
    return n_test >= len(counts) and n_train >= len(counts)


def _safe_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    primary_labels: Sequence[str],
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    if _can_stratify(primary_labels, test_size):
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=primary_labels,
            )
            return x_train, x_test, y_train, y_test, True
        except ValueError:
            pass
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    return x_train, x_test, y_train, y_test, False


def _predict_label_scores(model: OneVsRestClassifier, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X), dtype=np.float32)
    else:
        scores = np.asarray(model.decision_function(X), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores


def _scores_to_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    if scores.ndim == 1:
        scores = scores[:, None]
    predictions = (scores >= threshold).astype(np.int32)
    empty_rows = predictions.sum(axis=1) == 0
    if empty_rows.any():
        best_labels = scores[empty_rows].argmax(axis=1)
        predictions[empty_rows, best_labels] = 1
    return predictions


def _tune_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[float, Dict[str, Any]]:
    if len(X_train) < 40:
        return 0.5, {"tuned": False, "reason": "too_few_training_examples"}

    val_size = 0.15
    pseudo_labels = y_train.argmax(axis=1).astype(str).tolist()
    x_sub_train, x_val, y_sub_train, y_val, used_stratification = _safe_train_test_split(
        X_train,
        y_train,
        pseudo_labels,
        test_size=val_size,
        random_state=seed,
    )
    if len(x_val) == 0:
        return 0.5, {"tuned": False, "reason": "empty_validation_split"}

    model = _build_linear_probe_model()
    model.fit(x_sub_train, y_sub_train)
    val_scores = _predict_label_scores(model, x_val)

    best_threshold = 0.5
    best_micro_f1 = -1.0
    for threshold in DEFAULT_THRESHOLD_GRID:
        y_val_pred = _scores_to_predictions(val_scores, threshold)
        micro_f1 = f1_score(y_val, y_val_pred, average="micro", zero_division=0)
        if micro_f1 > best_micro_f1 + 1e-12 or (
            math.isclose(micro_f1, best_micro_f1, rel_tol=0.0, abs_tol=1e-12)
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_micro_f1 = micro_f1
            best_threshold = threshold

    return best_threshold, {
        "tuned": True,
        "used_stratification": bool(used_stratification),
        "validation_micro_f1": float(best_micro_f1),
        "n_validation": int(len(x_val)),
    }


def _multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    if y_score is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            try:
                metrics["lrap"] = float(
                    label_ranking_average_precision_score(y_true, y_score)
                )
            except ValueError:
                metrics["lrap"] = 0.0
            try:
                metrics["label_ranking_loss"] = float(
                    label_ranking_loss(y_true, y_score)
                )
            except ValueError:
                metrics["label_ranking_loss"] = 1.0
    return metrics


def _predict_label_priors(y_train: np.ndarray, n_test: int) -> Tuple[np.ndarray, np.ndarray, int]:
    label_priors = y_train.mean(axis=0).astype(np.float32)
    predicted_label_count = max(1, int(round(float(y_train.sum(axis=1).mean()))))
    top_labels = np.argsort(-label_priors)[:predicted_label_count]

    baseline_scores = np.tile(label_priors, (n_test, 1))
    baseline_predictions = np.zeros((n_test, y_train.shape[1]), dtype=np.int32)
    baseline_predictions[:, top_labels] = 1
    return baseline_predictions, baseline_scores, predicted_label_count


def _summarize_runs(
    runs: Sequence[Dict[str, Any]],
    exclude_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    exclude = set(exclude_keys or [])
    numeric_keys: List[str] = []
    for key in runs[0]:
        if key in exclude:
            continue
        value = runs[0][key]
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            numeric_keys.append(key)

    mean_metrics = {
        key: float(np.mean([float(run[key]) for run in runs]))
        for key in numeric_keys
    }
    std_metrics = {
        key: float(np.std([float(run[key]) for run in runs]))
        for key in numeric_keys
    }
    return {
        "mean": mean_metrics,
        "std": std_metrics,
        "per_split": list(runs),
    }


def evaluate_linear_probe(
    X: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    *,
    n_repeats: int = 5,
    test_size: float = 0.2,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate embeddings with a repeated-split linear probe and a label-prior baseline.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if len(X) < 2:
        raise ValueError("Linear-probe evaluation requires at least two documents.")

    primary_labels = _primary_labels_for_stratification(df["filtered_keywords"])
    probe_runs: List[Dict[str, Any]] = []
    baseline_runs: List[Dict[str, Any]] = []

    for repeat_idx in range(n_repeats):
        seed = base_seed + repeat_idx
        x_train, x_test, y_train, y_test, used_stratification = _safe_train_test_split(
            X,
            Y,
            primary_labels,
            test_size=test_size,
            random_state=seed,
        )

        threshold, tuning_meta = _tune_threshold(x_train, y_train, seed=seed)
        model = _build_linear_probe_model()
        model.fit(x_train, y_train)

        test_scores = _predict_label_scores(model, x_test)
        test_predictions = _scores_to_predictions(test_scores, threshold)
        run_metrics = _multilabel_metrics(y_test, test_predictions, y_score=test_scores)
        run_metrics.update(
            {
                "seed": int(seed),
                "threshold": float(threshold),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "used_stratification": bool(used_stratification),
                "threshold_tuned": bool(tuning_meta.get("tuned", False)),
                "validation_micro_f1": float(
                    tuning_meta.get("validation_micro_f1", 0.0)
                ),
            }
        )
        probe_runs.append(run_metrics)

        baseline_predictions, baseline_scores, predicted_label_count = _predict_label_priors(
            y_train,
            n_test=len(x_test),
        )
        baseline_metrics = _multilabel_metrics(
            y_test,
            baseline_predictions,
            y_score=baseline_scores,
        )
        baseline_metrics.update(
            {
                "seed": int(seed),
                "predicted_label_count": int(predicted_label_count),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
            }
        )
        baseline_runs.append(baseline_metrics)

    return {
        "n_repeats": int(n_repeats),
        "test_size": float(test_size),
        "probe": _summarize_runs(
            probe_runs,
            exclude_keys={"seed", "used_stratification", "threshold_tuned"},
        ),
        "label_prior_baseline": _summarize_runs(
            baseline_runs,
            exclude_keys={"seed"},
        ),
        "n_stratified_splits": int(sum(run["used_stratification"] for run in probe_runs)),
    }


def _resolve_retrieval_ks(requested_k: int, n_docs: int) -> List[int]:
    return [
        k for k in sorted({1, 5, 10, requested_k})
        if 0 < k < n_docs
    ]


def _average_precision_at_k(relevance_topk: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank_idx, is_relevant in enumerate(relevance_topk[:k], start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / float(rank_idx)
    return precision_sum / float(min(n_relevant, k))


def _ndcg_at_k(topk_relevance: np.ndarray, all_relevance: np.ndarray, k: int) -> float:
    topk = topk_relevance[:k].astype(np.float64)
    if not np.any(topk):
        return 0.0
    discounts = np.log2(np.arange(2, len(topk) + 2))
    dcg = float(np.sum((np.power(2.0, topk) - 1.0) / discounts))

    ideal = np.sort(all_relevance[all_relevance > 0])[::-1][:k].astype(np.float64)
    if ideal.size == 0:
        return 0.0
    ideal_discounts = np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) / ideal_discounts))
    return dcg / idcg if idcg > 0.0 else 0.0


def _init_retrieval_buckets(ks: Sequence[int]) -> Dict[str, List[float]]:
    buckets = {
        "mrr": [],
        "relevant_docs": [],
    }
    for k in ks:
        buckets[f"precision_at_{k}"] = []
        buckets[f"recall_at_{k}"] = []
        buckets[f"hit_rate_at_{k}"] = []
        buckets[f"map_at_{k}"] = []
        buckets[f"ndcg_at_{k}"] = []
        buckets[f"mean_shared_labels_at_{k}"] = []
    return buckets


def _accumulate_retrieval_metrics(
    buckets: Dict[str, List[float]],
    scores: np.ndarray,
    shared_counts: np.ndarray,
    ks: Sequence[int],
    row_idx: int,
) -> bool:
    binary_relevance = shared_counts > 0
    n_relevant = int(binary_relevance.sum())
    if n_relevant == 0:
        return False

    scores = scores.astype(np.float32, copy=True)
    scores[row_idx] = -np.inf
    ranking = np.argsort(-scores, kind="stable")
    ranked_binary = binary_relevance[ranking]
    ranked_shared = shared_counts[ranking]

    first_relevant_idx = np.flatnonzero(ranked_binary)
    buckets["mrr"].append(1.0 / float(first_relevant_idx[0] + 1))
    buckets["relevant_docs"].append(float(n_relevant))

    for k in ks:
        topk_binary = ranked_binary[:k]
        topk_shared = ranked_shared[:k]
        hits = int(topk_binary.sum())
        buckets[f"precision_at_{k}"].append(hits / float(k))
        buckets[f"recall_at_{k}"].append(hits / float(n_relevant))
        buckets[f"hit_rate_at_{k}"].append(float(hits > 0))
        buckets[f"map_at_{k}"].append(_average_precision_at_k(topk_binary, n_relevant, k))
        buckets[f"ndcg_at_{k}"].append(_ndcg_at_k(topk_shared, shared_counts, k))
        buckets[f"mean_shared_labels_at_{k}"].append(float(np.mean(topk_shared)))
    return True


def _finalize_retrieval_buckets(buckets: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in buckets.items()
        if values
    }


def evaluate_retrieval(
    X_normalized: np.ndarray,
    Y: np.ndarray,
    texts: Sequence[str],
    *,
    k: int = 10,
    max_queries: int = 5000,
    random_state: int = 42,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Evaluate semantic retrieval against keyword-overlap relevance.

    Binary relevance: documents share at least one filtered keyword.
    Graded relevance: number of shared filtered keywords.
    """
    n_docs = X_normalized.shape[0]
    if n_docs < 2:
        raise ValueError("Retrieval evaluation requires at least two documents.")

    ks = _resolve_retrieval_ks(k, n_docs)
    if not ks:
        raise ValueError("No valid retrieval cutoff values for the current dataset size.")

    rng = np.random.default_rng(random_state)
    max_queries = min(max_queries, n_docs)
    if n_docs > max_queries:
        query_indices = np.sort(rng.choice(n_docs, size=max_queries, replace=False))
    else:
        query_indices = np.arange(n_docs)

    embedding_buckets = _init_retrieval_buckets(ks)
    tfidf_buckets = _init_retrieval_buckets(ks)
    tfidf_matrix = None
    use_tfidf = any(text.strip() for text in texts)
    if use_tfidf:
        try:
            tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                min_df=2,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        except ValueError:
            tfidf_matrix = None

    queries_used = 0
    for start in range(0, len(query_indices), batch_size):
        batch_indices = query_indices[start:start + batch_size]
        embedding_scores_batch = X_normalized[batch_indices] @ X_normalized.T
        tfidf_scores_batch = None
        if tfidf_matrix is not None:
            tfidf_scores_batch = (tfidf_matrix[batch_indices] @ tfidf_matrix.T).toarray()

        for local_idx, row_idx in enumerate(batch_indices):
            shared_counts = (Y[row_idx] @ Y.T).astype(np.int32, copy=False)
            shared_counts[row_idx] = 0

            used = _accumulate_retrieval_metrics(
                embedding_buckets,
                embedding_scores_batch[local_idx],
                shared_counts,
                ks,
                row_idx=row_idx,
            )
            if not used:
                continue

            queries_used += 1
            if tfidf_scores_batch is not None:
                _accumulate_retrieval_metrics(
                    tfidf_buckets,
                    tfidf_scores_batch[local_idx],
                    shared_counts,
                    ks,
                    row_idx=row_idx,
                )

    if queries_used == 0:
        raise ValueError(
            "No queries had at least one relevant neighbour after filtering. "
            "Inspect the keywords or reduce min_keyword_freq."
        )

    retrieval_results: Dict[str, Any] = {
        "relevance_definition": {
            "binary": "share at least one filtered keyword",
            "graded": "number of shared filtered keywords",
        },
        "ks": ks,
        "n_queries_sampled": int(len(query_indices)),
        "n_queries_used": int(queries_used),
        "embedding": _finalize_retrieval_buckets(embedding_buckets),
    }
    if tfidf_matrix is not None:
        retrieval_results["tfidf_baseline"] = _finalize_retrieval_buckets(tfidf_buckets)
    else:
        retrieval_results["tfidf_baseline"] = {
            "skipped": True,
            "reason": "no non-empty text columns available for lexical retrieval",
        }
    return retrieval_results


def run_full_evaluation(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    keyword_col: str = "keywords",
    min_keyword_freq: int = 20,
    k_retrieval: int = 10,
    n_repeats: int = 5,
    test_size: float = 0.2,
    base_seed: int = 42,
    max_retrieval_queries: int = 5000,
) -> Dict[str, Any]:
    """
    Prepare data and run classification plus retrieval evaluations.
    """
    prepared = prepare_data(
        df,
        embeddings=embeddings,
        keyword_col=keyword_col,
        min_keyword_freq=min_keyword_freq,
    )

    print(
        f"Evaluating {prepared.metadata['n_docs_evaluated']} documents, "
        f"{prepared.metadata['n_keywords']} keyword classes, "
        f"embedding dim {prepared.metadata['embedding_dim']}."
    )

    linear_metrics = evaluate_linear_probe(
        prepared.X,
        prepared.Y,
        prepared.df,
        n_repeats=n_repeats,
        test_size=test_size,
        base_seed=base_seed,
    )
    retrieval_metrics = evaluate_retrieval(
        prepared.X_normalized,
        prepared.Y,
        prepared.texts,
        k=k_retrieval,
        max_queries=max_retrieval_queries,
        random_state=base_seed,
    )

    return {
        "data": prepared.metadata,
        "linear_probe": linear_metrics,
        "retrieval": retrieval_metrics,
        "keyword_classes": prepared.mlb.classes_.tolist(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate embedding quality with repeated linear probes and retrieval metrics."
    )
    parser.add_argument(
        "--data_json",
        type=str,
        required=True,
        help="Path to input JSON file containing documents with keywords.",
    )
    parser.add_argument(
        "--embeddings_npy",
        type=str,
        required=True,
        help="Path to NPY file containing embeddings.",
    )
    parser.add_argument(
        "--keyword_col",
        type=str,
        default="mesh",
        help="Column name containing keyword labels.",
    )
    parser.add_argument(
        "--min_keyword_freq",
        type=int,
        default=20,
        help="Minimum keyword frequency required to keep a label.",
    )
    parser.add_argument(
        "--k_retrieval",
        type=int,
        default=10,
        help="Primary retrieval cutoff. The script also reports standard cutoffs like k=1,5,10 when valid.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of repeated random train/test splits for the linear probe.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data used for the held-out split in each linear-probe repeat.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed used for repeated splits and retrieval sampling.",
    )
    parser.add_argument(
        "--max_retrieval_queries",
        type=int,
        default=5000,
        help="Maximum number of sampled queries used for retrieval evaluation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.data_json}...")
    with open(args.data_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    df_input = pd.DataFrame(data_list)

    print(f"Loading embeddings from {args.embeddings_npy}...")
    embeddings_input = np.load(args.embeddings_npy)

    results = run_full_evaluation(
        df_input,
        embeddings=embeddings_input,
        keyword_col=args.keyword_col,
        min_keyword_freq=args.min_keyword_freq,
        k_retrieval=args.k_retrieval,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        base_seed=args.base_seed,
        max_retrieval_queries=args.max_retrieval_queries,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, indent=2)
    else:
        print(json.dumps(results, indent=2))
