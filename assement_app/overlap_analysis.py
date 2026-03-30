from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence

from novelty_app.evaluation.assessment_bundle import ASSESSMENT_RUBRIC_FIELDS
from novelty_app.evaluation.idea_fingerprint import fingerprint_text


FINGERPRINT_FIELD_WEIGHTS: Dict[str, float] = {
    "disease": 0.22,
    "material": 0.20,
    "payload": 0.14,
    "targeting": 0.14,
    "mechanism": 0.12,
    "model": 0.08,
    "route": 0.05,
    "outcome": 0.05,
}
IDEA_SCORE_FIELDS = tuple(ASSESSMENT_RUBRIC_FIELDS)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_score(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _normalized_set(values: Iterable[Any]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set = _normalized_set(left)
    right_set = _normalized_set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / float(len(union))


def _idea_id(idea: Dict[str, Any]) -> str:
    return _clean_text(idea.get("idea_id"))


def _queue_order(idea: Dict[str, Any], order_lookup: Dict[str, int]) -> int:
    return int(order_lookup.get(_idea_id(idea), 10**9))


def _target_key(idea: Dict[str, Any]) -> str:
    target = dict(idea.get("target") or {})
    effective_target = dict(target.get("effective_target") or {})
    if effective_target:
        return json.dumps(effective_target, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    fallback = _clean_text(target.get("target_id") or (idea.get("run_context") or {}).get("target_id"))
    return fallback or "unscoped_target"


def _support_citations(idea: Dict[str, Any]) -> set[str]:
    hypothesis = dict(idea.get("hypothesis") or {})
    return _normalized_set(hypothesis.get("support_citations") or [])


def _evidence_paper_ids(idea: Dict[str, Any]) -> set[str]:
    ideation_context = dict(idea.get("ideation_context") or {})
    papers = list(ideation_context.get("evidence_papers") or [])
    return _normalized_set(paper.get("paper_id") for paper in papers if isinstance(paper, dict))


def _evidence_signal_ids(idea: Dict[str, Any]) -> set[str]:
    support = _support_citations(idea)
    if support:
        return support
    return _evidence_paper_ids(idea)


def _fallback_fingerprint(idea: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis = dict(idea.get("hypothesis") or {})
    title = _clean_text(hypothesis.get("title"))
    text = _clean_text(hypothesis.get("text"))
    return fingerprint_text(" ".join(part for part in (title, text) if part).strip())


def _idea_fingerprint(idea: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis = dict(idea.get("hypothesis") or {})
    fingerprint = dict(hypothesis.get("idea_fingerprint") or {})
    if fingerprint:
        return fingerprint
    return _fallback_fingerprint(idea)


def _text_token_set(idea: Dict[str, Any]) -> set[str]:
    fingerprint = _idea_fingerprint(idea)
    tokens = fingerprint.get("tokens") or []
    return _normalized_set(tokens)


def _fingerprint_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_fp = _idea_fingerprint(left)
    right_fp = _idea_fingerprint(right)
    total_weight = sum(FINGERPRINT_FIELD_WEIGHTS.values()) or 1.0
    weighted_sum = 0.0
    for field, weight in FINGERPRINT_FIELD_WEIGHTS.items():
        weighted_sum += weight * _jaccard(left_fp.get(field) or [], right_fp.get(field) or [])
    return weighted_sum / float(total_weight)


def _evidence_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> tuple[float, List[str]]:
    left_support = _support_citations(left)
    right_support = _support_citations(right)
    if left_support and right_support:
        shared = sorted(left_support & right_support)
        return _jaccard(left_support, right_support), shared
    left_evidence = _evidence_paper_ids(left)
    right_evidence = _evidence_paper_ids(right)
    shared = sorted(left_evidence & right_evidence)
    return _jaccard(left_evidence, right_evidence), shared


def _text_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    return _jaccard(_text_token_set(left), _text_token_set(right))


def _model_score(idea: Dict[str, Any]) -> float | None:
    judge_context = dict(idea.get("judge_context") or {})
    idea_scores = dict(judge_context.get("idea_scores") or {})
    average_score = _safe_float(idea_scores.get("average_score"))
    if average_score is not None:
        return average_score
    criterion_scores = [
        _safe_float((idea_scores.get(field) or {}).get("score"))
        for field in IDEA_SCORE_FIELDS
    ]
    numeric_scores = [score for score in criterion_scores if score is not None]
    if not numeric_scores:
        return None
    return _mean(numeric_scores)


def _representative_sort_key(idea: Dict[str, Any], order_lookup: Dict[str, int]) -> tuple[Any, ...]:
    score = _model_score(idea)
    score_value = score if score is not None else -1.0
    return (
        -score_value,
        -int(idea.get("winner_task_count") or 0),
        -len(_support_citations(idea)),
        _queue_order(idea, order_lookup),
        _idea_id(idea),
    )


def _pair_payload(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    threshold: float,
) -> Dict[str, Any]:
    evidence_overlap, shared_evidence_ids = _evidence_overlap(left, right)
    fingerprint_overlap = _fingerprint_overlap(left, right)
    text_overlap = _text_overlap(left, right)
    combined_overlap = 0.50 * evidence_overlap + 0.30 * fingerprint_overlap + 0.20 * text_overlap
    is_overlap_edge = combined_overlap >= threshold and (evidence_overlap >= 0.50 or fingerprint_overlap >= 0.60)
    return {
        "idea_id_a": _idea_id(left),
        "idea_id_b": _idea_id(right),
        "target_key": _target_key(left),
        "combined_overlap": _round_score(combined_overlap),
        "evidence_overlap": _round_score(evidence_overlap),
        "fingerprint_overlap": _round_score(fingerprint_overlap),
        "text_overlap": _round_score(text_overlap),
        "is_overlap_edge": bool(is_overlap_edge),
        "shared_evidence_ids": shared_evidence_ids,
    }


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((str(left_id), str(right_id))))


def _component_pairs(member_ids: Sequence[str], pair_lookup: Dict[tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for left_id, right_id in combinations(sorted(member_ids), 2):
        pair = pair_lookup.get(_pair_key(left_id, right_id))
        if pair:
            out.append(pair)
    return out


def analyze_winner_overlap(ideas: Sequence[Dict[str, Any]], *, threshold: float = 0.68) -> Dict[str, Any]:
    normalized_threshold = max(0.0, min(1.0, float(threshold)))
    ordered_ideas = [dict(idea or {}) for idea in ideas]
    idea_lookup = {_idea_id(idea): idea for idea in ordered_ideas if _idea_id(idea)}
    order_lookup = {_idea_id(idea): idx for idx, idea in enumerate(ordered_ideas) if _idea_id(idea)}
    winner_ideas = [idea for idea in ordered_ideas if bool(idea.get("is_review_packet_winner")) and _idea_id(idea)]

    winners_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idea in winner_ideas:
        winners_by_target[_target_key(idea)].append(idea)

    pair_scores: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    pair_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    all_pair_overlaps: List[float] = []
    group_counter = 1

    for target_key in sorted(winners_by_target):
        target_winners = sorted(winners_by_target[target_key], key=lambda idea: _queue_order(idea, order_lookup))
        adjacency: Dict[str, set[str]] = {_idea_id(idea): set() for idea in target_winners}
        for left, right in combinations(target_winners, 2):
            pair = _pair_payload(left, right, threshold=normalized_threshold)
            pair_scores.append(pair)
            pair_lookup[_pair_key(pair["idea_id_a"], pair["idea_id_b"])] = pair
            all_pair_overlaps.append(float(pair["combined_overlap"]))
            if pair["is_overlap_edge"]:
                adjacency[pair["idea_id_a"]].add(pair["idea_id_b"])
                adjacency[pair["idea_id_b"]].add(pair["idea_id_a"])

        visited: set[str] = set()
        for idea in target_winners:
            root_id = _idea_id(idea)
            if root_id in visited:
                continue
            stack = [root_id]
            component_ids: List[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_ids.append(current)
                stack.extend(sorted(adjacency.get(current) or []))
            component_ids = sorted(component_ids, key=lambda idea_id: order_lookup.get(idea_id, 10**9))
            component_ideas = [idea_lookup[idea_id] for idea_id in component_ids if idea_id in idea_lookup]
            component_pairs = _component_pairs(component_ids, pair_lookup)
            mean_overlap = _mean(float(pair["combined_overlap"]) for pair in component_pairs) if component_pairs else 0.0
            diversity = max(0.0, 1.0 - mean_overlap)
            representative = sorted(component_ideas, key=lambda entry: _representative_sort_key(entry, order_lookup))[0]
            representative_id = _idea_id(representative)
            evidence_counter = Counter()
            for component_idea in component_ideas:
                for evidence_id in _evidence_signal_ids(component_idea):
                    evidence_counter[evidence_id] += 1
            shared_evidence_ids = sorted(evidence_id for evidence_id, count in evidence_counter.items() if count >= 2)
            hidden_ids = [idea_id for idea_id in component_ids if idea_id != representative_id]
            groups.append(
                {
                    "group_id": f"overlap_group_{group_counter}",
                    "target_key": target_key,
                    "member_ids": component_ids,
                    "representative_id": representative_id,
                    "hidden_ids": hidden_ids,
                    "group_size": len(component_ids),
                    "mean_overlap": _round_score(mean_overlap),
                    "diversity": _round_score(diversity),
                    "representative_model_score": _round_score(_model_score(representative) or 0.0)
                    if _model_score(representative) is not None
                    else None,
                    "shared_evidence_ids": shared_evidence_ids,
                }
            )
            group_counter += 1

    hidden_winner_ids = {
        idea_id
        for group in groups
        if int(group.get("group_size") or 0) > 1
        for idea_id in (group.get("hidden_ids") or [])
    }
    visible_winner_ids = [
        _idea_id(idea)
        for idea in ordered_ideas
        if bool(idea.get("is_review_packet_winner")) and _idea_id(idea) and _idea_id(idea) not in hidden_winner_ids
    ]
    visible_idea_ids = [_idea_id(idea) for idea in ordered_ideas if _idea_id(idea) and _idea_id(idea) not in hidden_winner_ids]
    overall_diversity = max(0.0, 1.0 - _mean(all_pair_overlaps)) if all_pair_overlaps else 1.0

    groups.sort(
        key=lambda group: (
            -int(group.get("group_size") or 0),
            -float(group.get("mean_overlap") or 0.0),
            str(group.get("target_key") or ""),
            str(group.get("representative_id") or ""),
        )
    )
    pair_scores.sort(
        key=lambda pair: (
            -float(pair.get("combined_overlap") or 0.0),
            str(pair.get("target_key") or ""),
            str(pair.get("idea_id_a") or ""),
            str(pair.get("idea_id_b") or ""),
        )
    )

    return {
        "threshold": _round_score(normalized_threshold),
        "winner_count": len(winner_ideas),
        "target_group_count": len(winners_by_target),
        "overlap_group_count": sum(1 for group in groups if int(group.get("group_size") or 0) > 1),
        "hidden_winner_count": len(hidden_winner_ids),
        "overall_diversity": _round_score(overall_diversity),
        "pair_scores": pair_scores,
        "groups": groups,
        "visible_winner_ids": visible_winner_ids,
        "hidden_winner_ids": sorted(hidden_winner_ids, key=lambda idea_id: order_lookup.get(idea_id, 10**9)),
        "visible_idea_ids": visible_idea_ids,
    }
