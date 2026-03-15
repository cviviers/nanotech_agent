from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .idea_fingerprint import fingerprint_text


FIELD_WEIGHTS = {
    "disease": 0.22,
    "material": 0.20,
    "payload": 0.14,
    "targeting": 0.14,
    "mechanism": 0.12,
    "model": 0.08,
    "route": 0.05,
    "outcome": 0.05,
}


def _field_overlap(query_terms: List[str], candidate_terms: List[str]) -> float:
    if not query_terms:
        return 0.0
    q = set(query_terms)
    c = set(candidate_terms)
    return len(q & c) / float(max(1, len(q)))


def judge_candidate_match(
    fingerprint: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_fp = fingerprint_text(
        f"{candidate.get('title', '')} {candidate.get('abstract', candidate.get('text', ''))}"
    )
    field_scores: Dict[str, float] = {}
    overlap = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        score = _field_overlap(fingerprint.get(field, []), candidate_fp.get(field, []))
        field_scores[field] = score
        overlap += weight * score

    rerank = float(candidate.get("reranker_score", 0.0) or 0.0)
    emb = float(candidate.get("embedding_score", 0.0) or 0.0)
    emb_norm = (emb + 1.0) / 2.0
    combined = 0.55 * rerank + 0.25 * overlap + 0.20 * emb_norm

    if rerank >= 0.80 and overlap >= 0.45 and combined >= 0.70:
        label = "strong_match"
    elif rerank >= 0.58 and overlap >= 0.22 and combined >= 0.50:
        label = "partial_match"
    elif rerank >= 0.38 or overlap >= 0.15:
        label = "background_only"
    else:
        label = "no_match"

    return {
        "label": label,
        "combined_score": combined,
        "field_overlap": overlap,
        "field_scores": field_scores,
        "candidate_fingerprint": candidate_fp,
    }


def classify_hypothesis_match(
    *,
    historical_best: Dict[str, Any],
    future_best: Dict[str, Any],
    support_citations: List[str],
    grounding_summary: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    historical_label = historical_best.get("judge", {}).get("label", "no_match")
    future_label = future_best.get("judge", {}).get("label", "no_match")
    supported_fraction = float(grounding_summary.get("supported_claim_fraction", 1.0) or 0.0)

    if historical_label == "strong_match":
        classification = "already_present"
    elif supported_fraction < 0.25 or not support_citations:
        classification = "unsupported"
    elif future_label == "strong_match":
        classification = "anticipatory_strong"
    elif future_label == "partial_match":
        classification = "anticipatory_partial"
    else:
        classification = "unrealized"

    return classification, {
        "historical_label": historical_label,
        "future_label": future_label,
        "supported_claim_fraction": supported_fraction,
    }
