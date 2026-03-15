from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from agents.schemas import DiscoveryCue
except Exception:  # pragma: no cover
    from novelty_app.agents.schemas import DiscoveryCue

try:
    from config import DELIVERY_HINTS, DISEASE_HINTS, LIGAND_HINTS, MATERIAL_HINTS, MODEL_HINTS
except Exception:  # pragma: no cover
    from novelty_app.config import DELIVERY_HINTS, DISEASE_HINTS, LIGAND_HINTS, MATERIAL_HINTS, MODEL_HINTS


FIELD_NAMES = ("disease", "material", "payload", "targeting", "mechanism", "model", "route", "outcome")
FIELD_ALIASES = {
    "diseases": "disease",
    "materials": "material",
    "payloads": "payload",
    "ligand": "targeting",
    "ligands": "targeting",
    "delivery": "route",
    "deliveries": "route",
    "routes": "route",
    "mechanisms": "mechanism",
    "models": "model",
    "outcomes": "outcome",
}
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

PAYLOAD_HINTS = [
    "mrna",
    "sirna",
    "dna",
    "oligonucleotide",
    "protein",
    "peptide",
    "drug",
    "doxorubicin",
    "paclitaxel",
    "cisplatin",
    "antigen",
    "adjuvant",
    "small molecule",
]

MECHANISM_HINTS = [
    "targeting",
    "penetration",
    "endosomal escape",
    "gene silencing",
    "photothermal",
    "photodynamic",
    "immune",
    "immunomodulation",
    "release",
    "biodistribution",
    "uptake",
]

OUTCOME_HINTS = [
    "efficacy",
    "toxicity",
    "diagnosis",
    "imaging",
    "theranostic",
    "delivery",
    "accumulation",
    "survival",
    "response",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-']+", _normalize_text(text))


def _find_terms(text: str, terms: Iterable[str]) -> List[str]:
    normalized = _normalize_text(text)
    found: List[str] = []
    for term in terms:
        normalized_term = _normalize_text(term)
        if not normalized_term:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(normalized_term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(term)
    return sorted(set(found))


def _dedupe_texts(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _normalize_field_mapping(mapping: Any) -> Dict[str, List[str]]:
    if not isinstance(mapping, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for raw_key, raw_value in mapping.items():
        key = FIELD_ALIASES.get(_normalize_text(raw_key), _normalize_text(raw_key))
        if key not in FIELD_NAMES:
            continue
        if isinstance(raw_value, str):
            values = [raw_value]
        elif isinstance(raw_value, Sequence):
            values = [str(v) for v in raw_value]
        else:
            values = [str(raw_value)]
        normalized_values = [_normalize_text(v) for v in _dedupe_texts(values)]
        if normalized_values:
            out[key] = normalized_values
    return out


def fingerprint_text(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    return {
        "normalized_text": normalized,
        "tokens": _tokens(normalized),
        "disease": _find_terms(normalized, DISEASE_HINTS),
        "material": _find_terms(normalized, MATERIAL_HINTS),
        "payload": _find_terms(normalized, PAYLOAD_HINTS),
        "targeting": _find_terms(normalized, LIGAND_HINTS),
        "mechanism": _find_terms(normalized, MECHANISM_HINTS),
        "model": _find_terms(normalized, MODEL_HINTS),
        "route": _find_terms(normalized, DELIVERY_HINTS),
        "outcome": _find_terms(normalized, OUTCOME_HINTS),
    }


def fingerprint_hypothesis(hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    parts = [
        str(hypothesis.get("title") or ""),
        str(hypothesis.get("text") or ""),
        str(hypothesis.get("bridge_type") or ""),
        str(hypothesis.get("mechanistic_rationale") or ""),
        " ".join(str(x) for x in (hypothesis.get("novel_elements") or [])),
    ]
    text = " ".join(part for part in parts if part).strip()
    fp = fingerprint_text(text)
    fp["query_text"] = " ".join(
        [
            str(hypothesis.get("title") or ""),
            " ".join(fp["material"]),
            " ".join(fp["targeting"]),
            " ".join(fp["disease"]),
            " ".join(fp["payload"]),
            " ".join(fp["mechanism"]),
        ]
    ).strip()
    return fp


def normalize_discovery_cue(value: Any) -> Optional[DiscoveryCue]:
    if value is None:
        return None
    if isinstance(value, DiscoveryCue):
        cue = value.model_copy(deep=True)
    elif isinstance(value, str):
        cue = DiscoveryCue(text=value)
    elif isinstance(value, dict):
        cue = DiscoveryCue(**value)
    else:
        raise TypeError(f"Unsupported discovery cue type: {type(value)!r}")

    cue.text = str(cue.text or "").strip()
    cue.goal = str(cue.goal or "").strip() or None
    cue.include_terms = [_normalize_text(v) for v in _dedupe_texts(cue.include_terms)]
    cue.avoid_terms = [_normalize_text(v) for v in _dedupe_texts(cue.avoid_terms)]
    cue.counter_queries = _dedupe_texts(cue.counter_queries)
    cue.preferred_fields = _normalize_field_mapping(cue.preferred_fields)
    cue.hard_constraints = _normalize_field_mapping(cue.hard_constraints)
    cue.soft_constraints = _normalize_field_mapping(cue.soft_constraints)

    if (
        not cue.text
        and not cue.goal
        and not cue.include_terms
        and not cue.preferred_fields
        and not cue.hard_constraints
        and not cue.soft_constraints
        and not cue.counter_queries
    ):
        return None

    parts: List[str] = []
    if cue.goal:
        parts.append(cue.goal)
    if cue.text:
        parts.append(cue.text)
    parts.extend(cue.include_terms)
    fp = fingerprint_text(" ".join(parts))
    for mapping in (cue.preferred_fields, cue.hard_constraints, cue.soft_constraints):
        for field, values in mapping.items():
            fp[field] = sorted(set(fp.get(field, []) + list(values)))
    cue.fingerprint = fp
    return cue


def discovery_cue_to_dict(value: Any) -> Optional[Dict[str, Any]]:
    cue = normalize_discovery_cue(value)
    return cue.model_dump() if cue is not None else None


def discovery_cue_query_terms(value: Any, *, max_queries: int = 8) -> List[str]:
    cue = normalize_discovery_cue(value)
    if cue is None:
        return []
    queries: List[str] = []
    queries.extend(cue.counter_queries)
    if cue.goal:
        queries.append(cue.goal)
    if cue.text and cue.text != cue.goal:
        queries.append(cue.text)
    queries.extend(cue.include_terms)
    for mapping in (cue.hard_constraints, cue.soft_constraints, cue.preferred_fields):
        for values in mapping.values():
            if values:
                queries.append(" ".join(values[:3]))
    fp = cue.fingerprint or {}
    merged_terms: List[str] = []
    for field in FIELD_NAMES:
        merged_terms.extend(list(fp.get(field) or [])[:2])
    if merged_terms:
        queries.append(" ".join(_dedupe_texts(merged_terms)[:6]))
    return _dedupe_texts(queries)[:max_queries]


def _field_overlap(query_terms: Sequence[str], candidate_terms: Sequence[str]) -> float:
    if not query_terms:
        return 0.0
    q = set(_normalize_text(term) for term in query_terms if term)
    c = set(_normalize_text(term) for term in candidate_terms if term)
    return len(q & c) / float(max(1, len(q)))


def score_fingerprint_against_cue(candidate_fingerprint: Dict[str, Any], value: Any) -> Dict[str, Any]:
    cue = normalize_discovery_cue(value)
    if cue is None:
        return {
            "score": 0.0,
            "field_overlap": 0.0,
            "field_scores": {},
            "include_matches": [],
            "avoid_matches": [],
            "hard_constraint_matches": {},
            "hard_constraint_misses": [],
            "soft_constraint_matches": {},
            "preferred_field_matches": {},
        }

    cue_fp = cue.fingerprint or {}
    field_scores: Dict[str, float] = {}
    field_overlap = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        score = _field_overlap(cue_fp.get(field, []), candidate_fingerprint.get(field, []))
        field_scores[field] = score
        field_overlap += weight * score

    normalized_text = str(candidate_fingerprint.get("normalized_text") or "")
    include_matches = [term for term in cue.include_terms if term in normalized_text]
    avoid_matches = [term for term in cue.avoid_terms if term in normalized_text]

    hard_matches: Dict[str, List[str]] = {}
    hard_misses: List[str] = []
    for field, values in cue.hard_constraints.items():
        hits = sorted(set(values) & set(candidate_fingerprint.get(field, [])))
        if hits:
            hard_matches[field] = hits
        else:
            hard_misses.append(field)

    soft_matches: Dict[str, List[str]] = {}
    for field, values in cue.soft_constraints.items():
        hits = sorted(set(values) & set(candidate_fingerprint.get(field, [])))
        if hits:
            soft_matches[field] = hits

    preferred_matches: Dict[str, List[str]] = {}
    for field, values in cue.preferred_fields.items():
        hits = sorted(set(values) & set(candidate_fingerprint.get(field, [])))
        if hits:
            preferred_matches[field] = hits

    include_bonus = 0.15 * (len(include_matches) / max(1, len(cue.include_terms))) if cue.include_terms else 0.0
    avoid_penalty = 0.25 * (len(avoid_matches) / max(1, len(cue.avoid_terms))) if cue.avoid_terms else 0.0
    hard_bonus = 0.15 * (len(hard_matches) / max(1, len(cue.hard_constraints))) if cue.hard_constraints else 0.0
    hard_penalty = 0.30 * (len(hard_misses) / max(1, len(cue.hard_constraints))) if cue.hard_constraints else 0.0
    soft_bonus = 0.10 * (len(soft_matches) / max(1, len(cue.soft_constraints))) if cue.soft_constraints else 0.0
    preferred_bonus = 0.07 * (len(preferred_matches) / max(1, len(cue.preferred_fields))) if cue.preferred_fields else 0.0
    score = field_overlap + include_bonus + hard_bonus + soft_bonus + preferred_bonus - hard_penalty - avoid_penalty
    score = max(-1.0, min(1.5, float(score)))

    return {
        "score": score,
        "field_overlap": field_overlap,
        "field_scores": field_scores,
        "include_matches": include_matches,
        "avoid_matches": avoid_matches,
        "hard_constraint_matches": hard_matches,
        "hard_constraint_misses": hard_misses,
        "soft_constraint_matches": soft_matches,
        "preferred_field_matches": preferred_matches,
        "candidate_fingerprint": candidate_fingerprint,
        "cue_fingerprint": cue_fp,
    }


def score_text_against_cue(text: str, value: Any) -> Dict[str, Any]:
    return score_fingerprint_against_cue(fingerprint_text(text), value)


def score_record_against_cue(record: Dict[str, Any], value: Any) -> Dict[str, Any]:
    text = f"{record.get('title', '')} {record.get('abstract', record.get('processed_content', ''))}".strip()
    return score_text_against_cue(text, value)


def cue_prompt_block(value: Any) -> str:
    cue = normalize_discovery_cue(value)
    if cue is None:
        return ""
    return (
        "RESEARCH DIRECTION CUE (STEERING ONLY, NOT EVIDENCE):\n"
        f"{cue.model_dump_json(indent=2)}\n"
        "Follow hard_constraints where feasible, treat soft_constraints and preferred_fields as preferences, "
        "and do not cite the cue as evidence.\n"
    )


__all__ = [
    "DiscoveryCue",
    "FIELD_NAMES",
    "cue_prompt_block",
    "discovery_cue_query_terms",
    "discovery_cue_to_dict",
    "fingerprint_hypothesis",
    "fingerprint_text",
    "normalize_discovery_cue",
    "score_fingerprint_against_cue",
    "score_record_against_cue",
    "score_text_against_cue",
]
