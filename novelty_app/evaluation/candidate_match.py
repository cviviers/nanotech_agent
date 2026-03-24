from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

try:
    from agents.corpus_manifest import stable_paper_id_from_row
except Exception:  # pragma: no cover
    from novelty_app.agents.corpus_manifest import stable_paper_id_from_row

from .judge import judge_candidate_match
from .qwen_client import QwenClient


LABEL_PRIORITY = {
    "strong_match": 3,
    "partial_match": 2,
    "background_only": 1,
    "no_match": 0,
}


@dataclass
class CorpusIndex:
    df: pd.DataFrame
    texts: List[str]
    lower_texts: List[str]
    paper_ids: List[str]
    paper_id_to_idx: Dict[str, int]
    embeddings: np.ndarray
    normalized_embeddings: np.ndarray


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def build_corpus_index(df: pd.DataFrame, embeddings: np.ndarray) -> CorpusIndex:
    if len(df) != len(embeddings):
        raise ValueError("df and embeddings must align")
    paper_ids = [stable_paper_id_from_row(row) for _, row in df.iterrows()]
    texts = [
        f"{row.get('title', '')}\n{row.get('abstract', row.get('cleaned_text', ''))}".strip()
        for _, row in df.iterrows()
    ]
    lower_texts = [t.lower() for t in texts]
    return CorpusIndex(
        df=df.reset_index(drop=True).copy(),
        texts=texts,
        lower_texts=lower_texts,
        paper_ids=paper_ids,
        paper_id_to_idx={paper_id: idx for idx, paper_id in enumerate(paper_ids)},
        embeddings=embeddings.astype(np.float32, copy=False),
        normalized_embeddings=_normalize_embeddings(embeddings.astype(np.float32, copy=False)),
    )


def _keyword_score(lower_text: str, query_terms: Sequence[str]) -> float:
    if not query_terms:
        return 0.0
    hits = 0
    for term in query_terms:
        if term and term in lower_text:
            hits += 1
    return hits / float(max(1, len(query_terms)))


def _top_indices(scores: np.ndarray, top_k: int, *, positive_only: bool = False) -> List[int]:
    if scores.size == 0:
        return []
    if positive_only:
        eligible = np.where(scores > 0)[0]
        if eligible.size == 0:
            return []
        subset_scores = scores[eligible]
        top_k = min(top_k, len(eligible))
        order = np.argpartition(-subset_scores, top_k - 1)[:top_k]
        return eligible[order[np.argsort(-subset_scores[order])]].tolist()
    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return []
    order = np.argpartition(-scores, top_k - 1)[:top_k]
    return order[np.argsort(-scores[order])].tolist()


def retrieve_candidates_for_hypothesis(
    *,
    query_text: str,
    fingerprint: Dict[str, Any],
    corpus: CorpusIndex,
    qwen_client: Optional[QwenClient],
    required_paper_ids: Optional[Sequence[str]] = None,
    top_k_keyword: int = 25,
    top_k_semantic: int = 40,
    top_k_final: Optional[int] = 10,
    rerank_max_docs: int = 16,
    doc_char_limit: int = 2500,
) -> List[Dict[str, Any]]:
    query_terms: List[str] = []
    for field in ("disease", "material", "payload", "targeting", "mechanism", "model", "route", "outcome"):
        query_terms.extend([str(x).lower() for x in (fingerprint.get(field) or [])])
    query_terms = sorted({term for term in query_terms if term})

    keyword_scores = np.asarray([_keyword_score(text, query_terms) for text in corpus.lower_texts], dtype=float)
    keyword_idx = _top_indices(keyword_scores, top_k_keyword, positive_only=True)

    semantic_scores = np.zeros(len(corpus.df), dtype=float)
    if qwen_client is not None and query_text.strip():
        try:
            query_emb = np.asarray(
                qwen_client.embed(
                    [query_text],
                    instruction="Retrieve scientific papers that describe the same concrete research idea.",
                    normalize=True,
                )[0],
                dtype=np.float32,
            )
            semantic_scores = corpus.normalized_embeddings @ query_emb
        except (requests.RequestException, RuntimeError, IndexError, ValueError):
            semantic_scores = np.zeros(len(corpus.df), dtype=float)
    semantic_idx = _top_indices(semantic_scores, top_k_semantic, positive_only=False)

    required_idx = [
        corpus.paper_id_to_idx[paper_id]
        for paper_id in (required_paper_ids or [])
        if paper_id in corpus.paper_id_to_idx
    ]
    candidate_idx = list(dict.fromkeys(keyword_idx + semantic_idx + required_idx))
    if not candidate_idx:
        fallback_limit = min(top_k_final or 10, len(corpus.df))
        candidate_idx = list(range(fallback_limit))

    candidate_idx = sorted(
        candidate_idx,
        key=lambda idx: (keyword_scores[idx], semantic_scores[idx]),
        reverse=True,
    )
    rerank_cap = max(top_k_final or 10, min(rerank_max_docs, len(candidate_idx)))
    rerank_candidate_idx = candidate_idx[:rerank_cap]
    for idx in required_idx:
        if idx not in rerank_candidate_idx:
            rerank_candidate_idx.append(idx)

    docs = [corpus.texts[i][:doc_char_limit] for i in rerank_candidate_idx]
    rerank_results: Dict[int, Dict[str, Any]] = {}
    if qwen_client is not None and docs:
        try:
            ranked = qwen_client.rank(
                query=query_text,
                documents=docs,
                instruction="Rank scientific abstracts by how strongly they describe the same research idea.",
                top_k=None if top_k_final is None else min(top_k_final, len(docs)),
                return_embedding_similarity=True,
                normalize_embeddings=True,
            )
            for item in ranked:
                rerank_results[int(item["index"])] = item
        except (requests.RequestException, RuntimeError):
            rerank_results = {}

    out: List[Dict[str, Any]] = []
    rerank_lookup = {corpus_idx: local_idx for local_idx, corpus_idx in enumerate(rerank_candidate_idx)}
    for corpus_idx in candidate_idx:
        row = corpus.df.iloc[corpus_idx]
        rerank = rerank_results.get(rerank_lookup.get(corpus_idx, -1), {})
        candidate = {
            "paper_id": stable_paper_id_from_row(row),
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", row.get("cleaned_text", "")))[:doc_char_limit],
            "publication_year": int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
            "keyword_score": float(keyword_scores[corpus_idx]),
            "embedding_score": float(rerank.get("embedding_score", semantic_scores[corpus_idx])),
            "reranker_score": float(rerank.get("reranker_score", 0.0)),
        }
        candidate["judge"] = judge_candidate_match(fingerprint, candidate)
        out.append(candidate)

    out.sort(
        key=lambda item: (
            LABEL_PRIORITY[item["judge"]["label"]],
            item["judge"]["combined_score"],
            item.get("reranker_score", 0.0),
            item.get("embedding_score", 0.0),
        ),
        reverse=True,
    )
    if top_k_final is None:
        return out
    return out[:top_k_final]


def best_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"judge": {"label": "no_match", "combined_score": 0.0}}
    return dict(candidates[0])


def candidate_by_paper_id(candidates: Sequence[Dict[str, Any]], paper_id: str) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        if str(candidate.get("paper_id") or "") == str(paper_id):
            return dict(candidate)
    return None


def candidate_rank(candidates: Sequence[Dict[str, Any]], paper_id: str) -> Optional[int]:
    for idx, candidate in enumerate(candidates, start=1):
        if str(candidate.get("paper_id") or "") == str(paper_id):
            return idx
    return None


def best_non_excluded_candidate(
    candidates: Sequence[Dict[str, Any]],
    excluded_paper_ids: Sequence[str],
) -> Dict[str, Any]:
    excluded = {str(paper_id) for paper_id in excluded_paper_ids}
    for candidate in candidates:
        if str(candidate.get("paper_id") or "") not in excluded:
            return dict(candidate)
    return {"judge": {"label": "no_match", "combined_score": 0.0}}


def first_matching_year(candidates: Sequence[Dict[str, Any]]) -> Optional[int]:
    years = [
        int(c["publication_year"])
        for c in candidates
        if c.get("publication_year") is not None and c.get("judge", {}).get("label") in {"strong_match", "partial_match"}
    ]
    return min(years) if years else None
