import ast
import json
import math
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
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

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


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
DEFAULT_RETRIEVAL_KS = (1, 5, 10, 20, 100)
DEFAULT_PROBE_BACKEND = "auto"
DEFAULT_PROBE_DEVICE = "auto"
DEFAULT_THRESHOLD_TUNING = "auto"
DEFAULT_PROBE_N_JOBS = 1
DEFAULT_TORCH_PROBE_EPOCHS = 20
DEFAULT_TORCH_PROBE_BATCH_SIZE = 512
DEFAULT_TORCH_PROBE_LR = 1e-3
DEFAULT_TORCH_PROBE_WEIGHT_DECAY = 1e-4
DEFAULT_TORCH_PROBE_POS_WEIGHT_CLIP = 100.0
AUTO_PROBE_SGD_CLASS_THRESHOLD = 256
AUTO_PROBE_SGD_TASK_THRESHOLD = 250_000
AUTO_THRESHOLD_TUNING_CLASS_THRESHOLD = 128
AUTO_THRESHOLD_TUNING_TASK_THRESHOLD = 100_000


@dataclass
class PreparedEvaluationData:
    X: np.ndarray
    X_normalized: np.ndarray
    Y: np.ndarray
    df: pd.DataFrame
    mlb: MultiLabelBinarizer
    texts: List[str]
    metadata: Dict[str, Any]


def _progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    **kwargs: Any,
) -> Iterable[Any]:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _log(message: str) -> None:
    print(f"[eval] {message}", flush=True)


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


def _allowed_keywords_by_frequency(
    keyword_lists: Sequence[Sequence[str]],
    min_keyword_freq: int,
) -> List[str]:
    keyword_counts = Counter(
        keyword
        for keyword_list in keyword_lists
        for keyword in keyword_list
    )
    return sorted(
        keyword for keyword, count in keyword_counts.items() if count >= min_keyword_freq
    )


def _filter_keywords_to_allowed(
    keyword_lists: Sequence[Sequence[str]],
    allowed_keywords: Set[str],
) -> List[List[str]]:
    return [
        [keyword for keyword in keyword_list if keyword in allowed_keywords]
        for keyword_list in keyword_lists
    ]


def _binarize_keyword_lists(
    keyword_lists: Sequence[Sequence[str]],
    classes: Sequence[str],
) -> Tuple[np.ndarray, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer(classes=list(classes))
    return mlb.fit_transform(keyword_lists).astype(np.int32, copy=False), mlb


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


def build_document_texts(
    df: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> Tuple[List[str], List[str]]:
    text_columns_used: List[str] = []
    texts: List[str] = []
    rows = _progress(
        df.iterrows(),
        enabled=show_progress,
        total=len(df),
        desc="Building document texts",
        unit="doc",
    )
    for _, row in rows:
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
    strict_alignment: bool = True,
    show_progress: bool = False,
) -> PreparedEvaluationData:
    """
    Clean and align dataframe rows with embeddings, then build retrieval labels.

    For scientific safety, row-count mismatches raise by default instead of silently
    truncating potentially misaligned documents and embeddings.
    """
    if keyword_col not in df.columns:
        raise KeyError(f"Column '{keyword_col}' not found in the input data.")

    _log("Preparing evaluation data.")
    df = df.copy().reset_index(drop=True)
    embeddings = _as_2d_float32(embeddings)

    n_input_rows = len(df)
    n_embedding_rows = embeddings.shape[0]
    aligned_rows = min(n_input_rows, n_embedding_rows)
    alignment_trimmed = max(n_input_rows, n_embedding_rows) - aligned_rows
    if alignment_trimmed:
        if strict_alignment:
            raise ValueError(
                f"Dataframe rows ({n_input_rows}) and embeddings ({n_embedding_rows}) "
                "do not match. Refusing to continue because silent truncation can "
                "invalidate the evaluation. Fix the upstream alignment or rerun with "
                "strict_alignment=False / --allow_alignment_trim only if you have "
                "verified that row order still matches."
            )
        _log(
            "WARNING: dataframe rows "
            f"({n_input_rows}) and embeddings ({n_embedding_rows}) do not match. "
            f"Truncating both to {aligned_rows} aligned rows because "
            "strict_alignment=False."
        )
        df = df.iloc[:aligned_rows].reset_index(drop=True)
        embeddings = embeddings[:aligned_rows]
    else:
        _log(f"Verified aligned row count: {aligned_rows} rows.")

    finite_mask = np.isfinite(embeddings).all(axis=1)
    non_zero_mask = np.linalg.norm(embeddings, axis=1) > 1e-12
    valid_embedding_mask = finite_mask & non_zero_mask
    dropped_invalid_embeddings = int((~valid_embedding_mask).sum())
    if dropped_invalid_embeddings:
        _log(f"Dropping {dropped_invalid_embeddings} rows with invalid embedding vectors.")
        df = df.loc[valid_embedding_mask].reset_index(drop=True)
        embeddings = embeddings[valid_embedding_mask]
    else:
        _log("All embedding vectors are finite and non-zero.")

    df["clean_keywords"] = df[keyword_col].apply(clean_keywords)
    non_empty_keyword_mask = df["clean_keywords"].map(len) > 0
    dropped_empty_keywords = int((~non_empty_keyword_mask).sum())
    df = df.loc[non_empty_keyword_mask].reset_index(drop=True)
    embeddings = embeddings[non_empty_keyword_mask.to_numpy()]
    _log(
        "Retained "
        f"{len(df)} documents after dropping {dropped_empty_keywords} rows with empty "
        "or placeholder keywords."
    )

    allowed_keywords = _allowed_keywords_by_frequency(
        df["clean_keywords"],
        min_keyword_freq=min_keyword_freq,
    )
    if not allowed_keywords:
        raise ValueError(
            f"No keywords meet min_keyword_freq={min_keyword_freq}. "
            "Lower the threshold or inspect the keyword quality."
        )
    allowed_keyword_set = set(allowed_keywords)
    df["filtered_keywords"] = _filter_keywords_to_allowed(
        df["clean_keywords"],
        allowed_keyword_set,
    )
    docs_without_retrieval_keywords = int((df["filtered_keywords"].map(len) == 0).sum())

    y, mlb = _binarize_keyword_lists(df["filtered_keywords"], allowed_keywords)
    x_normalized = _normalize_rows(embeddings)
    _log(
        "Built corpus-level retrieval label matrix with "
        f"{y.shape[1]} keyword classes. "
        f"{docs_without_retrieval_keywords} documents have no labels left after the "
        "corpus-wide frequency filter and will contribute zero-valued retrieval targets."
    )
    _log("Building document texts for the lexical baseline.")
    texts, text_columns_used = build_document_texts(df, show_progress=show_progress)

    label_cardinality = y.sum(axis=1)
    metadata = {
        "n_input_rows": int(n_input_rows),
        "n_embedding_rows": int(n_embedding_rows),
        "n_docs_evaluated": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_keywords": int(y.shape[1]),
        "keyword_col": keyword_col,
        "min_keyword_freq": int(min_keyword_freq),
        "strict_alignment": bool(strict_alignment),
        "text_columns_used": text_columns_used,
        "label_cardinality_mean": float(label_cardinality.mean()),
        "label_cardinality_std": float(label_cardinality.std()),
        "label_density": float(y.mean()),
        "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
        "docs_without_retrieval_keywords": int(docs_without_retrieval_keywords),
        "dropped_rows": {
            "alignment_trimmed": int(alignment_trimmed),
            "invalid_embeddings": int(dropped_invalid_embeddings),
            "empty_or_placeholder_keywords": int(dropped_empty_keywords),
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


def _torch_cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _resolve_probe_device(requested_device: str) -> Tuple[str, str]:
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("probe_device must be one of 'auto', 'cpu', or 'cuda'.")
    if torch is None:
        raise ImportError(
            "The Torch linear probe requires torch to be installed. "
            "Install torch or choose --probe_backend sgd/logistic."
        )
    if requested_device == "cpu":
        return "cpu", "user-requested CPU device"
    if requested_device == "cuda":
        if not _torch_cuda_available():
            raise RuntimeError(
                "Torch probe requested CUDA, but torch.cuda.is_available() is false. "
                "Check your NVIDIA driver, CUDA runtime, or CUDA_VISIBLE_DEVICES."
            )
        return "cuda", "user-requested CUDA device"
    if _torch_cuda_available():
        return "cuda", "auto-selected CUDA device"
    return "cpu", "auto-selected CPU device because CUDA is unavailable"


def _validate_torch_probe_config(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_weight_clip: float,
) -> None:
    if epochs < 1:
        raise ValueError("torch_probe_epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("torch_probe_batch_size must be at least 1.")
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("torch_probe_lr must be a positive finite value.")
    if weight_decay < 0.0 or not math.isfinite(weight_decay):
        raise ValueError("torch_probe_weight_decay must be a finite non-negative value.")
    if pos_weight_clip <= 0.0 or not math.isfinite(pos_weight_clip):
        raise ValueError("torch_probe_pos_weight_clip must be a positive finite value.")


def _resolve_probe_backend(
    requested_backend: str,
    *,
    n_train: int,
    n_classes: int,
    probe_device: str = DEFAULT_PROBE_DEVICE,
) -> Tuple[str, str]:
    if probe_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("probe_device must be one of 'auto', 'cpu', or 'cuda'.")
    if requested_backend not in {"auto", "logistic", "sgd", "torch"}:
        raise ValueError(
            "probe_backend must be one of 'auto', 'logistic', 'sgd', or 'torch'."
        )
    if requested_backend == "torch":
        resolved_device, device_reason = _resolve_probe_device(probe_device)
        return "torch", f"user-requested Torch BCE probe ({device_reason}: {resolved_device})"
    if requested_backend != "auto":
        return requested_backend, "user-requested backend"

    cuda_probe_requested = probe_device in {"auto", "cuda"}
    cuda_available = _torch_cuda_available()
    if cuda_probe_requested and cuda_available:
        return "torch", "auto-selected Torch BCE probe because CUDA is available"
    if probe_device == "cuda":
        raise RuntimeError(
            "probe_backend=auto with probe_device=cuda requested CUDA, but "
            "torch.cuda.is_available() is false. Use --probe_device auto to allow "
            "the sklearn CPU fallback."
        )

    task_size = n_train * max(n_classes, 1)
    if cuda_probe_requested:
        cuda_fallback_reason = "CUDA Torch is unavailable; "
    else:
        cuda_fallback_reason = "probe_device=cpu; "
    if (
        n_classes >= AUTO_PROBE_SGD_CLASS_THRESHOLD
        or task_size >= AUTO_PROBE_SGD_TASK_THRESHOLD
    ):
        return (
            "sgd",
            (
                f"auto-selected SGD because {cuda_fallback_reason}"
                f"n_classes={n_classes} and "
                f"n_train*n_classes={task_size}"
            ),
        )
    return (
        "logistic",
        (
            "auto-selected exact logistic probe because "
            f"{cuda_fallback_reason}the task size is moderate"
        ),
    )


def _resolve_threshold_tuning(
    requested_tuning: str,
    *,
    backend: str,
    n_train: int,
    n_classes: int,
) -> Tuple[bool, str]:
    if requested_tuning not in {"auto", "on", "off"}:
        raise ValueError("threshold_tuning must be one of 'auto', 'on', or 'off'.")
    if requested_tuning == "on":
        return True, "user-requested threshold tuning"
    if requested_tuning == "off":
        return False, "user-disabled threshold tuning"

    task_size = n_train * max(n_classes, 1)
    if backend in {"sgd", "torch"}:
        return False, f"auto-disabled threshold tuning for scalable {backend} probe"
    if (
        n_classes >= AUTO_THRESHOLD_TUNING_CLASS_THRESHOLD
        or task_size >= AUTO_THRESHOLD_TUNING_TASK_THRESHOLD
    ):
        return (
            False,
            (
                f"auto-disabled threshold tuning because n_classes={n_classes} and "
                f"n_train*n_classes={task_size}"
            ),
        )
    return True, "auto-enabled threshold tuning for a moderate task size"


def _build_linear_probe_model(
    *,
    backend: str,
    random_state: int,
    n_jobs: int,
) -> OneVsRestClassifier:
    if backend == "logistic":
        base_estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=300,
                solver="lbfgs",
                class_weight="balanced",
                random_state=random_state,
            ),
        )
    elif backend == "sgd":
        base_estimator = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
    else:
        raise ValueError(f"Unsupported probe backend '{backend}'.")
    return OneVsRestClassifier(base_estimator, n_jobs=n_jobs)


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


def _safe_index_split(
    n_samples: int,
    primary_labels: Sequence[str],
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    indices = np.arange(n_samples)
    if _can_stratify(primary_labels, test_size):
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=primary_labels,
            )
            return np.asarray(train_idx), np.asarray(test_idx), True
        except ValueError:
            pass
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    return np.asarray(train_idx), np.asarray(test_idx), False


def _predict_label_scores(model: OneVsRestClassifier, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X), dtype=np.float32)
    else:
        scores = np.asarray(model.decision_function(X), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores


def _standardize_with_train_stats(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    scale = X_train.std(axis=0, keepdims=True).astype(np.float32, copy=False)
    scale = np.clip(scale, 1e-6, None)
    x_train_scaled = ((X_train - mean) / scale).astype(np.float32, copy=False)
    x_eval_scaled = ((X_eval - mean) / scale).astype(np.float32, copy=False)
    return (
        np.ascontiguousarray(x_train_scaled, dtype=np.float32),
        np.ascontiguousarray(x_eval_scaled, dtype=np.float32),
    )


def _fit_predict_torch_probe_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    random_state: int,
    probe_device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_weight_clip: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if torch is None or nn is None:
        raise ImportError(
            "The Torch linear probe requires torch to be installed. "
            "Install torch or choose --probe_backend sgd/logistic."
        )
    _validate_torch_probe_config(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight_clip=pos_weight_clip,
    )
    resolved_device, device_reason = _resolve_probe_device(probe_device)
    device = torch.device(resolved_device)

    torch.manual_seed(random_state)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(random_state)

    x_train_scaled, x_eval_scaled = _standardize_with_train_stats(X_train, X_eval)
    x_train_cpu = torch.from_numpy(x_train_scaled)
    x_eval_cpu = torch.from_numpy(x_eval_scaled)
    y_train_cpu = torch.from_numpy(np.ascontiguousarray(y_train))

    n_train, n_features = x_train_scaled.shape
    n_classes = y_train.shape[1]
    model = nn.Linear(n_features, n_classes).to(device)

    positive_counts = y_train.sum(axis=0).astype(np.float32)
    negative_counts = float(n_train) - positive_counts
    pos_weight = negative_counts / np.clip(positive_counts, 1.0, None)
    pos_weight = np.clip(pos_weight, 0.0, pos_weight_clip).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    rng = np.random.default_rng(random_state)
    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        last_loss = None
        permutation = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch_indices = torch.from_numpy(permutation[start:start + batch_size])
            batch_x = x_train_cpu.index_select(0, batch_indices).to(device)
            batch_y = y_train_cpu.index_select(0, batch_indices).to(
                device=device,
                dtype=torch.float32,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            last_loss = loss.detach()
        if last_loss is not None:
            final_loss = float(last_loss.cpu())

    model.eval()
    score_batches: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x_eval_cpu), batch_size):
            batch_x = x_eval_cpu[start:start + batch_size].to(device)
            batch_scores = torch.sigmoid(model(batch_x)).cpu().numpy()
            score_batches.append(batch_scores.astype(np.float32, copy=False))

    if resolved_device == "cuda":
        torch.cuda.empty_cache()

    scores = np.vstack(score_batches) if score_batches else np.empty(
        (0, n_classes),
        dtype=np.float32,
    )
    return scores, {
        "probe_device": resolved_device,
        "probe_device_reason": device_reason,
        "torch_probe_epochs": int(epochs),
        "torch_probe_batch_size": int(batch_size),
        "torch_probe_lr": float(lr),
        "torch_probe_weight_decay": float(weight_decay),
        "torch_probe_pos_weight_clip": float(pos_weight_clip),
        "torch_probe_final_loss": float(final_loss),
    }


def _fit_predict_label_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    backend: str,
    random_state: int,
    probe_n_jobs: int,
    probe_device: str,
    torch_probe_epochs: int,
    torch_probe_batch_size: int,
    torch_probe_lr: float,
    torch_probe_weight_decay: float,
    torch_probe_pos_weight_clip: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if backend in {"logistic", "sgd"}:
        model = _build_linear_probe_model(
            backend=backend,
            random_state=random_state,
            n_jobs=probe_n_jobs,
        )
        model.fit(X_train, y_train)
        return _predict_label_scores(model, X_eval), {}
    if backend == "torch":
        return _fit_predict_torch_probe_scores(
            X_train,
            y_train,
            X_eval,
            random_state=random_state,
            probe_device=probe_device,
            epochs=torch_probe_epochs,
            batch_size=torch_probe_batch_size,
            lr=torch_probe_lr,
            weight_decay=torch_probe_weight_decay,
            pos_weight_clip=torch_probe_pos_weight_clip,
        )
    raise ValueError(f"Unsupported probe backend '{backend}'.")


def _scores_to_predictions(
    scores: np.ndarray,
    threshold: float,
    *,
    ensure_non_empty: bool = False,
) -> np.ndarray:
    if scores.ndim == 1:
        scores = scores[:, None]
    predictions = (scores >= threshold).astype(np.int32)
    empty_rows = ensure_non_empty and (predictions.sum(axis=1) == 0)
    if np.any(empty_rows):
        best_labels = scores[empty_rows].argmax(axis=1)
        predictions[empty_rows, best_labels] = 1
    return predictions


def _prepare_probe_targets(
    train_keyword_lists: Sequence[Sequence[str]],
    test_keyword_lists: Sequence[Sequence[str]],
    min_keyword_freq: int,
) -> Tuple[np.ndarray, np.ndarray, MultiLabelBinarizer, Dict[str, int]]:
    allowed_keywords = _allowed_keywords_by_frequency(
        train_keyword_lists,
        min_keyword_freq=min_keyword_freq,
    )
    if not allowed_keywords:
        raise ValueError(
            "No training keywords meet the minimum frequency threshold for this split. "
            "Lower min_keyword_freq or increase the amount of training data."
        )

    allowed_keyword_set = set(allowed_keywords)
    filtered_train_keywords = _filter_keywords_to_allowed(
        train_keyword_lists,
        allowed_keyword_set,
    )
    filtered_test_keywords = _filter_keywords_to_allowed(
        test_keyword_lists,
        allowed_keyword_set,
    )
    y_train, mlb = _binarize_keyword_lists(filtered_train_keywords, allowed_keywords)
    y_test = mlb.transform(filtered_test_keywords).astype(np.int32, copy=False)
    split_meta = {
        "n_classes": int(len(allowed_keywords)),
        "n_train_without_supported_labels": int(
            sum(len(keywords) == 0 for keywords in filtered_train_keywords)
        ),
        "n_test_without_supported_labels": int(
            sum(len(keywords) == 0 for keywords in filtered_test_keywords)
        ),
        "n_train_with_supported_labels": int(
            sum(len(keywords) > 0 for keywords in filtered_train_keywords)
        ),
        "n_test_with_supported_labels": int(
            sum(len(keywords) > 0 for keywords in filtered_test_keywords)
        ),
    }
    return y_train, y_test, mlb, split_meta


def _tune_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    primary_labels: Sequence[str],
    seed: int,
    backend: str,
    probe_n_jobs: int,
    probe_device: str,
    torch_probe_epochs: int,
    torch_probe_batch_size: int,
    torch_probe_lr: float,
    torch_probe_weight_decay: float,
    torch_probe_pos_weight_clip: float,
) -> Tuple[float, Dict[str, Any]]:
    if len(X_train) < 40:
        return 0.5, {"tuned": False, "reason": "too_few_training_examples"}

    val_size = 0.15
    subtrain_idx, val_idx, used_stratification = _safe_index_split(
        len(X_train),
        primary_labels,
        test_size=val_size,
        random_state=seed,
    )
    if len(val_idx) == 0:
        return 0.5, {"tuned": False, "reason": "empty_validation_split"}

    x_sub_train = X_train[subtrain_idx]
    x_val = X_train[val_idx]
    y_sub_train = y_train[subtrain_idx]
    y_val = y_train[val_idx]
    val_scores, fit_meta = _fit_predict_label_scores(
        x_sub_train,
        y_sub_train,
        x_val,
        backend=backend,
        random_state=seed,
        probe_n_jobs=probe_n_jobs,
        probe_device=probe_device,
        torch_probe_epochs=torch_probe_epochs,
        torch_probe_batch_size=torch_probe_batch_size,
        torch_probe_lr=torch_probe_lr,
        torch_probe_weight_decay=torch_probe_weight_decay,
        torch_probe_pos_weight_clip=torch_probe_pos_weight_clip,
    )

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
        **fit_meta,
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
    df: pd.DataFrame,
    *,
    min_keyword_freq: int = 20,
    probe_backend: str = DEFAULT_PROBE_BACKEND,
    probe_device: str = DEFAULT_PROBE_DEVICE,
    threshold_tuning: str = DEFAULT_THRESHOLD_TUNING,
    probe_n_jobs: int = DEFAULT_PROBE_N_JOBS,
    torch_probe_epochs: int = DEFAULT_TORCH_PROBE_EPOCHS,
    torch_probe_batch_size: int = DEFAULT_TORCH_PROBE_BATCH_SIZE,
    torch_probe_lr: float = DEFAULT_TORCH_PROBE_LR,
    torch_probe_weight_decay: float = DEFAULT_TORCH_PROBE_WEIGHT_DECAY,
    torch_probe_pos_weight_clip: float = DEFAULT_TORCH_PROBE_POS_WEIGHT_CLIP,
    n_repeats: int = 5,
    test_size: float = 0.2,
    base_seed: int = 42,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate embeddings with a repeated-split linear probe and a label-prior baseline.

    The multilabel class space is defined separately inside each training split using
    only labels that meet min_keyword_freq in that split. This avoids leaking held-out
    label frequencies into the supervised task definition.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if len(X) < 2:
        raise ValueError("Linear-probe evaluation requires at least two documents.")

    keyword_lists = [list(keywords) for keywords in df["clean_keywords"]]
    primary_labels = _primary_labels_for_stratification(keyword_lists)
    probe_runs: List[Dict[str, Any]] = []
    baseline_runs: List[Dict[str, Any]] = []
    _log(
        "Starting linear-probe evaluation with "
        f"{n_repeats} repeated splits, test_size={test_size:.2f}, "
        f"and train-only min_keyword_freq={min_keyword_freq}. "
        f"requested backend={probe_backend}, threshold_tuning={threshold_tuning}, "
        f"probe_n_jobs={probe_n_jobs}, probe_device={probe_device}."
    )

    repeat_indices = _progress(
        range(n_repeats),
        enabled=show_progress,
        total=n_repeats,
        desc="Linear probe evaluation",
        unit="split",
    )
    for repeat_idx in repeat_indices:
        seed = base_seed + repeat_idx
        train_idx, test_idx, used_stratification = _safe_index_split(
            len(X),
            primary_labels,
            test_size=test_size,
            random_state=seed,
        )
        x_train = X[train_idx]
        x_test = X[test_idx]
        train_keyword_lists = [keyword_lists[idx] for idx in train_idx]
        test_keyword_lists = [keyword_lists[idx] for idx in test_idx]
        y_train, y_test, _, split_meta = _prepare_probe_targets(
            train_keyword_lists,
            test_keyword_lists,
            min_keyword_freq=min_keyword_freq,
        )
        resolved_backend, backend_reason = _resolve_probe_backend(
            probe_backend,
            n_train=len(x_train),
            n_classes=split_meta["n_classes"],
            probe_device=probe_device,
        )
        tune_threshold, tuning_reason = _resolve_threshold_tuning(
            threshold_tuning,
            backend=resolved_backend,
            n_train=len(x_train),
            n_classes=split_meta["n_classes"],
        )
        _log(
            f"Linear probe split {repeat_idx + 1}/{n_repeats}: prepared "
            f"n_train={len(x_train)}, n_test={len(x_test)}, "
            f"classes={split_meta['n_classes']}, backend={resolved_backend} "
            f"({backend_reason}), threshold_tuning={'on' if tune_threshold else 'off'} "
            f"({tuning_reason})."
        )

        if tune_threshold:
            _log(
                f"Linear probe split {repeat_idx + 1}/{n_repeats}: "
                "starting threshold-tuning fit."
            )
            threshold_fit_start = time.perf_counter()
            threshold, tuning_meta = _tune_threshold(
                x_train,
                y_train,
                primary_labels=[primary_labels[idx] for idx in train_idx],
                seed=seed,
                backend=resolved_backend,
                probe_n_jobs=probe_n_jobs,
                probe_device=probe_device,
                torch_probe_epochs=torch_probe_epochs,
                torch_probe_batch_size=torch_probe_batch_size,
                torch_probe_lr=torch_probe_lr,
                torch_probe_weight_decay=torch_probe_weight_decay,
                torch_probe_pos_weight_clip=torch_probe_pos_weight_clip,
            )
            threshold_fit_seconds = time.perf_counter() - threshold_fit_start
            _log(
                f"Linear probe split {repeat_idx + 1}/{n_repeats}: "
                f"threshold tuning finished in {threshold_fit_seconds:.1f}s with "
                f"threshold={threshold:.2f}."
            )
        else:
            threshold = 0.5
            tuning_meta = {
                "tuned": False,
                "reason": tuning_reason,
                "validation_micro_f1": 0.0,
                "n_validation": 0,
            }

        _log(
            f"Linear probe split {repeat_idx + 1}/{n_repeats}: "
            f"starting final {resolved_backend} fit."
        )
        fit_start = time.perf_counter()
        test_scores, fit_meta = _fit_predict_label_scores(
            x_train,
            y_train,
            x_test,
            backend=resolved_backend,
            random_state=seed,
            probe_n_jobs=probe_n_jobs,
            probe_device=probe_device,
            torch_probe_epochs=torch_probe_epochs,
            torch_probe_batch_size=torch_probe_batch_size,
            torch_probe_lr=torch_probe_lr,
            torch_probe_weight_decay=torch_probe_weight_decay,
            torch_probe_pos_weight_clip=torch_probe_pos_weight_clip,
        )
        fit_seconds = time.perf_counter() - fit_start
        _log(
            f"Linear probe split {repeat_idx + 1}/{n_repeats}: "
            f"final fit finished in {fit_seconds:.1f}s."
        )

        test_predictions = _scores_to_predictions(test_scores, threshold)
        run_metrics = _multilabel_metrics(y_test, test_predictions, y_score=test_scores)
        run_metrics.update(
            {
                "seed": int(seed),
                "threshold": float(threshold),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "n_classes": int(split_meta["n_classes"]),
                "n_train_with_supported_labels": int(
                    split_meta["n_train_with_supported_labels"]
                ),
                "n_train_without_supported_labels": int(
                    split_meta["n_train_without_supported_labels"]
                ),
                "n_test_with_supported_labels": int(
                    split_meta["n_test_with_supported_labels"]
                ),
                "n_test_without_supported_labels": int(
                    split_meta["n_test_without_supported_labels"]
                ),
                "probe_backend": resolved_backend,
                "used_stratification": bool(used_stratification),
                "threshold_tuned": bool(tuning_meta.get("tuned", False)),
                "threshold_tuning_reason": str(tuning_meta.get("reason", "")),
                "validation_micro_f1": float(
                    tuning_meta.get("validation_micro_f1", 0.0)
                ),
                "fit_seconds": float(fit_seconds),
                **fit_meta,
            }
        )
        probe_runs.append(run_metrics)
        _log(
            f"Linear probe split {repeat_idx + 1}/{n_repeats}: "
            f"seed={seed}, classes={split_meta['n_classes']}, "
            f"micro_f1={run_metrics['micro_f1']:.4f}, "
            f"lrap={run_metrics.get('lrap', 0.0):.4f}, "
            f"threshold={threshold:.2f}."
        )

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
                "n_classes": int(split_meta["n_classes"]),
                "n_train_with_supported_labels": int(
                    split_meta["n_train_with_supported_labels"]
                ),
                "n_train_without_supported_labels": int(
                    split_meta["n_train_without_supported_labels"]
                ),
                "n_test_with_supported_labels": int(
                    split_meta["n_test_with_supported_labels"]
                ),
                "n_test_without_supported_labels": int(
                    split_meta["n_test_without_supported_labels"]
                ),
                "probe_backend": resolved_backend,
                "fit_seconds": float(fit_seconds),
                **fit_meta,
            }
        )
        baseline_runs.append(baseline_metrics)

    _log("Finished linear-probe evaluation.")
    return {
        "n_repeats": int(n_repeats),
        "test_size": float(test_size),
        "min_keyword_freq_train_only": int(min_keyword_freq),
        "probe_backend_requested": probe_backend,
        "probe_device_requested": probe_device,
        "threshold_tuning_requested": threshold_tuning,
        "probe_n_jobs": int(probe_n_jobs),
        "torch_probe_epochs": int(torch_probe_epochs),
        "torch_probe_batch_size": int(torch_probe_batch_size),
        "torch_probe_lr": float(torch_probe_lr),
        "torch_probe_weight_decay": float(torch_probe_weight_decay),
        "torch_probe_pos_weight_clip": float(torch_probe_pos_weight_clip),
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
        k for k in sorted({*DEFAULT_RETRIEVAL_KS, requested_k})
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
) -> int:
    binary_relevance = shared_counts > 0
    n_relevant = int(binary_relevance.sum())

    scores = scores.astype(np.float32, copy=True)
    scores[row_idx] = -np.inf
    ranking = np.argsort(-scores, kind="stable")
    ranked_binary = binary_relevance[ranking]
    ranked_shared = shared_counts[ranking]

    if n_relevant > 0:
        first_relevant_idx = np.flatnonzero(ranked_binary)
        mrr = 1.0 / float(first_relevant_idx[0] + 1)
    else:
        mrr = 0.0
    buckets["mrr"].append(mrr)
    buckets["relevant_docs"].append(float(n_relevant))

    for k in ks:
        topk_binary = ranked_binary[:k]
        topk_shared = ranked_shared[:k]
        hits = int(topk_binary.sum())
        buckets[f"precision_at_{k}"].append(hits / float(k))
        buckets[f"recall_at_{k}"].append(hits / float(n_relevant) if n_relevant else 0.0)
        buckets[f"hit_rate_at_{k}"].append(float(hits > 0))
        buckets[f"map_at_{k}"].append(_average_precision_at_k(topk_binary, n_relevant, k))
        buckets[f"ndcg_at_{k}"].append(_ndcg_at_k(topk_shared, shared_counts, k))
        buckets[f"mean_shared_labels_at_{k}"].append(float(np.mean(topk_shared)))
    return n_relevant


def _finalize_retrieval_buckets(buckets: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in buckets.items()
        if values
    }


def evaluate_retrieval(
    X_normalized: np.ndarray,
    keyword_lists: Sequence[Sequence[str]],
    texts: Sequence[str],
    *,
    min_keyword_freq: int = 20,
    k: int = 10,
    max_queries: int = 5000,
    random_state: int = 42,
    batch_size: int = 1024,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate semantic retrieval against keyword-overlap relevance.

    Binary relevance: documents share at least one filtered keyword.
    Graded relevance: number of shared filtered keywords.

    Metrics are averaged over all sampled queries. Queries with no relevant neighbours
    under the keyword-overlap definition contribute zeros instead of being dropped.
    """
    n_docs = X_normalized.shape[0]
    if n_docs < 2:
        raise ValueError("Retrieval evaluation requires at least two documents.")
    if len(keyword_lists) != n_docs:
        raise ValueError(
            f"Expected {n_docs} keyword rows for retrieval, got {len(keyword_lists)}."
        )

    ks = _resolve_retrieval_ks(k, n_docs)
    if not ks:
        raise ValueError("No valid retrieval cutoff values for the current dataset size.")

    allowed_keywords = _allowed_keywords_by_frequency(
        keyword_lists,
        min_keyword_freq=min_keyword_freq,
    )
    if not allowed_keywords:
        raise ValueError(
            f"No keywords meet min_keyword_freq={min_keyword_freq} for retrieval."
        )
    filtered_keyword_lists = _filter_keywords_to_allowed(
        keyword_lists,
        set(allowed_keywords),
    )
    Y, _ = _binarize_keyword_lists(filtered_keyword_lists, allowed_keywords)
    docs_without_retrieval_keywords = int((Y.sum(axis=1) == 0).sum())
    max_queries = min(max_queries, n_docs)
    _log(
        "Starting retrieval evaluation with "
        f"{n_docs} documents, "
        f"{len(allowed_keywords)} corpus-level keyword classes, "
        f"and {max_queries} sampled queries."
    )

    rng = np.random.default_rng(random_state)
    if n_docs > max_queries:
        query_indices = np.sort(rng.choice(n_docs, size=max_queries, replace=False))
    else:
        query_indices = np.arange(n_docs)

    embedding_buckets = _init_retrieval_buckets(ks)
    tfidf_buckets = _init_retrieval_buckets(ks)
    tfidf_matrix = None
    tfidf_reason = ""
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
            tfidf_reason = "TF-IDF vocabulary could not be built from the provided texts."
    else:
        tfidf_reason = "no non-empty text columns available for lexical retrieval"
    if tfidf_matrix is not None:
        _log("Built TF-IDF lexical baseline.")
    else:
        _log(f"Skipping TF-IDF lexical baseline: {tfidf_reason}")

    queries_with_relevant_docs = 0
    batch_starts = range(0, len(query_indices), batch_size)
    batch_starts = _progress(
        batch_starts,
        enabled=show_progress,
        total=math.ceil(len(query_indices) / batch_size) if len(query_indices) else 0,
        desc="Retrieval evaluation",
        unit="batch",
    )
    total_batches = math.ceil(len(query_indices) / batch_size) if len(query_indices) else 0
    for batch_number, start in enumerate(batch_starts, start=1):
        _log(
            f"Retrieval batch {batch_number}/{max(total_batches, 1)} "
            f"covering queries {start} to {min(start + batch_size, len(query_indices)) - 1}."
        )
        batch_indices = query_indices[start:start + batch_size]
        embedding_scores_batch = X_normalized[batch_indices] @ X_normalized.T
        tfidf_scores_batch = None
        if tfidf_matrix is not None:
            tfidf_scores_batch = (tfidf_matrix[batch_indices] @ tfidf_matrix.T).toarray()

        for local_idx, row_idx in enumerate(batch_indices):
            shared_counts = (Y[row_idx] @ Y.T).astype(np.int32, copy=False)
            shared_counts[row_idx] = 0

            n_relevant = _accumulate_retrieval_metrics(
                embedding_buckets,
                embedding_scores_batch[local_idx],
                shared_counts,
                ks,
                row_idx=row_idx,
            )
            if n_relevant > 0:
                queries_with_relevant_docs += 1
            if tfidf_scores_batch is not None:
                _accumulate_retrieval_metrics(
                    tfidf_buckets,
                    tfidf_scores_batch[local_idx],
                    shared_counts,
                    ks,
                    row_idx=row_idx,
                )

    if queries_with_relevant_docs == 0:
        _log(
            "WARNING: no sampled queries had a relevant neighbour after keyword filtering. "
            "Retrieval metrics will be zero."
        )

    _log("Finished retrieval evaluation.")

    retrieval_results: Dict[str, Any] = {
        "relevance_definition": {
            "binary": "share at least one filtered keyword",
            "graded": "number of shared filtered keywords",
        },
        "aggregation": (
            "Averages are computed over all sampled queries. Queries without any "
            "relevant neighbours contribute zeros."
        ),
        "min_keyword_freq_corpus": int(min_keyword_freq),
        "n_keyword_classes": int(len(allowed_keywords)),
        "n_docs_without_retrieval_keywords": int(docs_without_retrieval_keywords),
        "ks": ks,
        "n_queries_sampled": int(len(query_indices)),
        "n_queries_with_relevant_docs": int(queries_with_relevant_docs),
        "n_queries_without_relevant_docs": int(len(query_indices) - queries_with_relevant_docs),
        "query_coverage": float(
            queries_with_relevant_docs / float(len(query_indices))
        ) if len(query_indices) else 0.0,
        "embedding": _finalize_retrieval_buckets(embedding_buckets),
    }
    if tfidf_matrix is not None:
        retrieval_results["tfidf_baseline"] = _finalize_retrieval_buckets(tfidf_buckets)
    else:
        retrieval_results["tfidf_baseline"] = {
            "skipped": True,
            "reason": tfidf_reason,
        }
    return retrieval_results


def run_full_evaluation(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    keyword_col: str = "keywords",
    min_keyword_freq: int = 20,
    probe_backend: str = DEFAULT_PROBE_BACKEND,
    probe_device: str = DEFAULT_PROBE_DEVICE,
    threshold_tuning: str = DEFAULT_THRESHOLD_TUNING,
    probe_n_jobs: int = DEFAULT_PROBE_N_JOBS,
    torch_probe_epochs: int = DEFAULT_TORCH_PROBE_EPOCHS,
    torch_probe_batch_size: int = DEFAULT_TORCH_PROBE_BATCH_SIZE,
    torch_probe_lr: float = DEFAULT_TORCH_PROBE_LR,
    torch_probe_weight_decay: float = DEFAULT_TORCH_PROBE_WEIGHT_DECAY,
    torch_probe_pos_weight_clip: float = DEFAULT_TORCH_PROBE_POS_WEIGHT_CLIP,
    k_retrieval: int = 10,
    n_repeats: int = 5,
    test_size: float = 0.2,
    base_seed: int = 42,
    max_retrieval_queries: int = 5000,
    strict_alignment: bool = True,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Prepare data and run classification plus retrieval evaluations.
    """
    prepared = prepare_data(
        df,
        embeddings=embeddings,
        keyword_col=keyword_col,
        min_keyword_freq=min_keyword_freq,
        strict_alignment=strict_alignment,
        show_progress=show_progress,
    )

    _log(
        f"Evaluating {prepared.metadata['n_docs_evaluated']} documents, "
        f"{prepared.metadata['n_keywords']} keyword classes, "
        f"embedding dim {prepared.metadata['embedding_dim']}."
    )

    linear_metrics = evaluate_linear_probe(
        prepared.X,
        prepared.df,
        min_keyword_freq=min_keyword_freq,
        probe_backend=probe_backend,
        probe_device=probe_device,
        threshold_tuning=threshold_tuning,
        probe_n_jobs=probe_n_jobs,
        torch_probe_epochs=torch_probe_epochs,
        torch_probe_batch_size=torch_probe_batch_size,
        torch_probe_lr=torch_probe_lr,
        torch_probe_weight_decay=torch_probe_weight_decay,
        torch_probe_pos_weight_clip=torch_probe_pos_weight_clip,
        n_repeats=n_repeats,
        test_size=test_size,
        base_seed=base_seed,
        show_progress=show_progress,
    )
    retrieval_metrics = evaluate_retrieval(
        prepared.X_normalized,
        prepared.df["clean_keywords"],
        prepared.texts,
        min_keyword_freq=min_keyword_freq,
        k=k_retrieval,
        max_queries=max_retrieval_queries,
        random_state=base_seed,
        show_progress=show_progress,
    )

    _log("Evaluation run completed.")
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
        help=(
            "Minimum keyword frequency threshold used for the train-only linear-probe "
            "label space and the corpus-level retrieval relevance filter."
        ),
    )
    parser.add_argument(
        "--k_retrieval",
        type=int,
        default=10,
        help=(
            "Primary retrieval cutoff. The script also reports standard cutoffs "
            "like k=1,5,10,20,100 when valid."
        ),
    )
    parser.add_argument(
        "--probe_backend",
        type=str,
        choices=("auto", "logistic", "sgd", "torch"),
        default=DEFAULT_PROBE_BACKEND,
        help=(
            "Linear-probe backend. 'auto' uses the Torch BCE probe when CUDA is "
            "available, otherwise exact logistic regression for moderate problems "
            "and SGD-based logistic probing for large ones."
        ),
    )
    parser.add_argument(
        "--probe_device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_PROBE_DEVICE,
        help=(
            "Device used by the Torch probe backend. Ignored by sklearn backends. "
            "'auto' uses CUDA when torch can see it, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--threshold_tuning",
        type=str,
        choices=("auto", "on", "off"),
        default=DEFAULT_THRESHOLD_TUNING,
        help=(
            "Whether to tune the multilabel prediction threshold on a validation split. "
            "'auto' disables tuning for large probe problems to keep runtime practical."
        ),
    )
    parser.add_argument(
        "--probe_n_jobs",
        type=int,
        default=DEFAULT_PROBE_N_JOBS,
        help=(
            "Number of parallel jobs used by the one-vs-rest probe wrapper. "
            "Use 1 to avoid CPU and RAM oversubscription on large label spaces."
        ),
    )
    parser.add_argument(
        "--torch_probe_epochs",
        type=int,
        default=DEFAULT_TORCH_PROBE_EPOCHS,
        help="Number of epochs for the Torch BCE linear probe.",
    )
    parser.add_argument(
        "--torch_probe_batch_size",
        type=int,
        default=DEFAULT_TORCH_PROBE_BATCH_SIZE,
        help="Mini-batch size for the Torch BCE linear probe.",
    )
    parser.add_argument(
        "--torch_probe_lr",
        type=float,
        default=DEFAULT_TORCH_PROBE_LR,
        help="Learning rate for the Torch BCE linear probe.",
    )
    parser.add_argument(
        "--torch_probe_weight_decay",
        type=float,
        default=DEFAULT_TORCH_PROBE_WEIGHT_DECAY,
        help="AdamW weight decay for the Torch BCE linear probe.",
    )
    parser.add_argument(
        "--torch_probe_pos_weight_clip",
        type=float,
        default=DEFAULT_TORCH_PROBE_POS_WEIGHT_CLIP,
        help=(
            "Upper clip for BCE positive-class weights computed as "
            "negative_count / positive_count."
        ),
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
    parser.add_argument(
        "--allow_alignment_trim",
        action="store_true",
        help=(
            "Allow truncating the dataframe and embeddings to their shared row count. "
            "Use only if you have independently verified that row order still matches."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars for long-running evaluation steps.",
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
        probe_backend=args.probe_backend,
        probe_device=args.probe_device,
        threshold_tuning=args.threshold_tuning,
        probe_n_jobs=args.probe_n_jobs,
        torch_probe_epochs=args.torch_probe_epochs,
        torch_probe_batch_size=args.torch_probe_batch_size,
        torch_probe_lr=args.torch_probe_lr,
        torch_probe_weight_decay=args.torch_probe_weight_decay,
        torch_probe_pos_weight_clip=args.torch_probe_pos_weight_clip,
        k_retrieval=args.k_retrieval,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        base_seed=args.base_seed,
        max_retrieval_queries=args.max_retrieval_queries,
        strict_alignment=not args.allow_alignment_trim,
        show_progress=args.progress,
    )

    if args.output_json:
        _log(f"Writing evaluation report to {args.output_json}.")
        with open(args.output_json, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, indent=2)
    else:
        print(json.dumps(results, indent=2))

    # example usage:
    # python embedding_models/eval.py --data_json ./data/cleaned_dataset.json --embeddings_npy ./data/qwen_embeddings.npy --keyword_col mesh --min_keyword_freq 5 --n_repeats 3 --test_size 0.2 --base_seed 42 --max_retrieval_queries 5000 --output_json ./qwen_evaluation_report.json
