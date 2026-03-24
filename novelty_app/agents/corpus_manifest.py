from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


IDENTIFIER_PRIORITY = ("pmid", "id", "paper_id", "doi")


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    shape = getattr(value, "shape", None)
    if shape is not None and shape != ():
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    if hasattr(result, "shape") and getattr(result, "shape", None) not in (None, ()):
        return False
    try:
        return bool(result)
    except Exception:
        return False


def _normalize_text(value: Any) -> Optional[str]:
    if is_null(value):
        return None
    text = str(value).strip()
    return text or None


def _source_row_suffix(row: pd.Series) -> Optional[str]:
    value = row.get("source_row_index")
    if is_null(value):
        return None
    try:
        return str(int(value))
    except Exception:
        text = _normalize_text(value)
        return text


def stable_paper_id_from_row(row: pd.Series) -> str:
    source_suffix = _source_row_suffix(row)
    for key in IDENTIFIER_PRIORITY:
        if key not in row.index:
            continue
        value = _normalize_text(row.get(key))
        if not value:
            continue
        if source_suffix is not None:
            return f"{key}:{value}__src{source_suffix}"
        return f"{key}:{value}"

    if source_suffix is not None:
        return f"source_row:{source_suffix}"

    fallback_payload = {
        "title": _normalize_text(row.get("title")),
        "abstract": _normalize_text(row.get("abstract") or row.get("processed_content") or row.get("cleaned_text")),
        "publication_year": _normalize_text(row.get("publication_year") or row.get("year")),
        "publication_month": _normalize_text(row.get("publication_month")),
        "publication_day": _normalize_text(row.get("publication_day")),
        "journal": _normalize_text(row.get("journal")),
    }
    digest = hashlib.sha256(
        json.dumps(fallback_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"fallback:{digest[:24]}"


def stable_paper_ids(df: pd.DataFrame) -> List[str]:
    return [stable_paper_id_from_row(row) for _, row in df.iterrows()]


def hash_paper_ids(paper_ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(paper_ids), ensure_ascii=False).encode("utf-8")).hexdigest()


def build_frontend_corpus_manifest(
    df: pd.DataFrame,
    *,
    sample_n: Optional[int],
    random_seed: int,
    title_exclusion_keywords: Sequence[str],
    abstract_exclusion_keywords: Sequence[str],
    embedding_source: str,
    available_embeddings: Sequence[str],
    data_json: str | Path,
    data_dir: str | Path,
) -> Dict[str, Any]:
    retained_paper_ids = stable_paper_ids(df)
    return {
        "schema_version": "frontend_corpus_manifest_v1",
        "source": "streamlit_frontend",
        "retained_paper_ids": retained_paper_ids,
        "retained_paper_id_hash": hash_paper_ids(retained_paper_ids),
        "row_count": len(retained_paper_ids),
        "sample_n": int(sample_n) if sample_n is not None else None,
        "random_seed": int(random_seed),
        "title_exclusion_keywords": sorted({str(item).strip() for item in title_exclusion_keywords if str(item).strip()}),
        "abstract_exclusion_keywords": sorted(
            {str(item).strip() for item in abstract_exclusion_keywords if str(item).strip()}
        ),
        "embedding_source": str(embedding_source),
        "available_embeddings": [str(item) for item in available_embeddings],
        "data_json": str(Path(data_json)),
        "data_dir": str(Path(data_dir)),
    }


def reconstruct_positions_from_manifest(df: pd.DataFrame, retained_paper_ids: Sequence[str]) -> List[int]:
    observed_ids = stable_paper_ids(df)
    pos_by_id: Dict[str, int] = {}
    duplicates: List[str] = []
    for pos, paper_id in enumerate(observed_ids):
        if paper_id in pos_by_id:
            duplicates.append(paper_id)
            continue
        pos_by_id[paper_id] = pos
    if duplicates:
        sample = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate stable paper ids encountered while reconstructing corpus: {sample}")

    missing = [paper_id for paper_id in retained_paper_ids if paper_id not in pos_by_id]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"Manifest paper ids were not found in the raw corpus: {sample}")

    return [pos_by_id[paper_id] for paper_id in retained_paper_ids]


def subset_embeddings_by_positions(
    embeddings: Mapping[str, np.ndarray],
    positions: Sequence[int],
) -> Dict[str, np.ndarray]:
    indices = np.asarray(list(positions), dtype=int)
    return {name: arr[indices].copy() for name, arr in embeddings.items()}
