import ast
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def _parse_list_maybe(x):
    """Robustly parse a Python-list-like object that might be stored as a string."""
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple, set)):
                return list(val)
            return [val]
        except Exception:
            # fall back to splitting on commas if it's a simple comma-separated string
            if "," in x:
                return [p.strip() for p in x.split(",") if p.strip()]
            return [x.strip()]
    if pd.isna(x):
        return []
    return [x]


def clean_keywords(raw_kw) -> List[str]:
    """Normalize keyword field -> list of lowercased keywords, removing placeholders."""
    kws = _parse_list_maybe(raw_kw)
    cleaned = []
    for k in kws:
        if k is None or (isinstance(k, float) and math.isnan(k)):
            continue
        s = str(k).strip().lower()
        # filter out junk like 'Not Available'
        if not s or s in {"not available", "not_available", "n/a", "na"}:
            continue
        cleaned.append(s)
    # deduplicate
    return sorted(set(cleaned))


def parse_embedding(raw_emb) -> np.ndarray:
    """Convert embedding cell to a 1D np.ndarray."""
    if isinstance(raw_emb, np.ndarray):
        return raw_emb.astype(np.float32)
    if isinstance(raw_emb, list):
        return np.asarray(raw_emb, dtype=np.float32)
    if isinstance(raw_emb, str):
        val = ast.literal_eval(raw_emb)
        return np.asarray(val, dtype=np.float32)
    raise ValueError(f"Unsupported embedding type: {type(raw_emb)}")


def prepare_data(
    df: pd.DataFrame,
    embedding_col: str = "qwen_content_embedding",
    keyword_col: str = "keywords",
    min_keyword_freq: int = 20,
) -> Tuple[np.ndarray, np.ndarray, MultiLabelBinarizer, pd.DataFrame]:
    """
    Clean dataframe and return:
    X: [N, D] embedding matrix
    Y: [N, K] multi-label indicator matrix
    mlb: fitted MultiLabelBinarizer
    df_used: filtered dataframe aligned with X, Y
    """
    df = df.copy()
    print("Preparing data...")

    # Filter out rows with invalid keywords (NaN, None, empty, etc.) before processing
    df = df[df[keyword_col].notna()].reset_index(drop=True)
    df = df[df[keyword_col].astype(str).str.strip() != ""].reset_index(drop=True)
    # Filter out common placeholder values
    df = df[~df[keyword_col].astype(str).str.lower().isin(["not available", "not_available", "n/a", "na", "none", "nan"])].reset_index(drop=True)
    # Filter out empty lists (represented as "[]" string)
    df = df[df[keyword_col].astype(str).str.strip() != "[]"].reset_index(drop=True)
    # Filter out lists containing only 'Not Available' (e.g., "['Not Available']")
    df = df[df[keyword_col].astype(str).str.strip() != "['Not Available']"].reset_index(drop=True)
    
    print(f"After filtering invalid keywords: {len(df)} rows remain.")
    # Clean keywords
    df["clean_keywords"] = df[keyword_col].apply(clean_keywords)
    df = df[df["clean_keywords"].map(len) > 0].reset_index(drop=True)
    print(f"After cleaning keywords: {len(df)} rows remain.")
    # Build MultiLabelBinarizer with frequency threshold
    all_kws = [kw for kws in df["clean_keywords"] for kw in kws]
    kw_series = pd.Series(all_kws)
    freq = kw_series.value_counts()

    allowed_keywords = set(freq[freq >= min_keyword_freq].index.tolist())
    if not allowed_keywords:
        raise ValueError(
            f"No keywords with frequency >= {min_keyword_freq}. "
            "Reduce min_keyword_freq or inspect your keywords."
        )

    df["filtered_keywords"] = df["clean_keywords"].apply(
        lambda kws: [k for k in kws if k in allowed_keywords]
    )
    df = df[df["filtered_keywords"].map(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["filtered_keywords"])

    # Parse embeddings
    emb_list = [parse_embedding(x) for x in df[embedding_col]]
    X = np.stack(emb_list, axis=0)

    return X, Y, mlb, df


def evaluate_linear_probe(
    X: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a simple One-vs-Rest logistic regression to predict keywords from embeddings.
    Returns standard multi-label metrics.
    """
    # Use first keyword as "primary" for stratification (rough heuristic)
    primary_labels = [kws[0] for kws in df["filtered_keywords"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state,
        stratify=primary_labels,
    )

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            class_weight=None,
        )
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_labels": int(Y.shape[1]),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(y_test, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_test, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
    }
    return metrics


def evaluate_retrieval(
    X: np.ndarray,
    df: pd.DataFrame,
    k: int = 10,
    max_queries: int = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate retrieval quality using keyword overlap as relevance.
    For each query abstract, all others sharing at least 1 keyword are "relevant".

    Returns mean Precision@k, Recall@k, and MRR@k.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    max_queries = min(max_queries, n)

    # Sample queries if dataset is large
    if n > max_queries:
        query_indices = rng.choice(n, size=max_queries, replace=False)
    else:
        query_indices = np.arange(n)

    # Precompute cosine similarity between query set and full set
    sims = cosine_similarity(X[query_indices], X)  # [Q, N]

    recalls = []
    precisions = []
    mrrs = []

    all_kw_sets = [set(kws) for kws in df["filtered_keywords"]]

    for qi, row_idx in enumerate(query_indices):
        q_kws = all_kw_sets[row_idx]

        # Build relevance vector: True if share at least one keyword
        rel = np.array(
            [
                (len(q_kws & all_kw_sets[j]) > 0) if j != row_idx else False
                for j in range(n)
            ],
            dtype=bool,
        )
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue  # skip queries with no relevant neighbors

        sim_row = sims[qi].copy()
        # exclude self
        sim_row[row_idx] = -1.0

        # Top-k indices
        if k >= n:
            topk_idx = np.argsort(-sim_row)
        else:
            topk_part = np.argpartition(-sim_row, k)[:k]
            # sort these top-k
            topk_idx = topk_part[np.argsort(-sim_row[topk_part])]

        hits_at_k = int(rel[topk_idx].sum())
        precisions.append(hits_at_k / float(min(k, n - 1)))
        recalls.append(hits_at_k / float(n_rel))

        # MRR: rank of first relevant
        sorted_idx = np.argsort(-sim_row)
        rel_sorted = rel[sorted_idx]
        if rel_sorted.any():
            first_rel_rank = int(np.where(rel_sorted)[0][0]) + 1
            mrrs.append(1.0 / first_rel_rank)

    if not recalls:
        raise ValueError("No queries with at least one relevant neighbor; check keywords.")

    metrics = {
        f"precision_at_{k}": float(np.mean(precisions)),
        f"recall_at_{k}": float(np.mean(recalls)),
        f"mrr_at_{k}": float(np.mean(mrrs)),
        "n_queries_used": int(len(precisions)),
    }
    return metrics


def run_full_evaluation(
    df: pd.DataFrame,
    embedding_col: str = "qwen_content_embedding",
    keyword_col: str = "keywords",
    min_keyword_freq: int = 20,
    k_retrieval: int = 10,
) -> Dict[str, Any]:
    """
    Convenience wrapper: prepares data and runs both linear-probe and retrieval evaluations.
    """
    X, Y, mlb, df_used = prepare_data(
        df,
        embedding_col=embedding_col,
        keyword_col=keyword_col,
        min_keyword_freq=min_keyword_freq,
    )

    linear_metrics = evaluate_linear_probe(X, Y, df_used)
    retrieval_metrics = evaluate_retrieval(X, df_used, k=k_retrieval)

    return {
        "embedding_col": embedding_col,
        "n_docs": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "n_keywords": int(Y.shape[1]),
        "linear_probe": linear_metrics,
        "retrieval": retrieval_metrics,
        "keyword_classes": mlb.classes_.tolist(),
    }


# main
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate embedding quality using keyword prediction and retrieval."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="Path to input CSV file containing documents with embeddings and keywords.",
    )
    parser.add_argument(
        "--embedding_col",
        type=str,
        default="qwen_content_embedding",
        help="Column name for embeddings.",
    )
    parser.add_argument(
        "--keyword_col",
        type=str,
        default="keywords",
        help="Column name for keywords.",
    )
    parser.add_argument(
        "--min_keyword_freq",
        type=int,
        default=20,
        help="Minimum frequency for keywords to be included.",
    )
    parser.add_argument(
        "--k_retrieval",
        type=int,
        default=10,
        help="Value of k for retrieval evaluation (Precision@k, etc.).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to output JSON file to save results. If not provided, prints to stdout.",
    )

    args = parser.parse_args()

    # Load data
    df_input = pd.read_csv(args.data_csv)

    # Run evaluation
    results = run_full_evaluation(
        df_input,
        embedding_col=args.embedding_col,
        keyword_col=args.keyword_col,
        min_keyword_freq=args.min_keyword_freq,
        k_retrieval=args.k_retrieval,
    )

    # Output results
    if args.output_json:
        with open(args.output_json, "w") as f_out:
            json.dump(results, f_out, indent=4)
    else:
        print(json.dumps(results, indent=4))