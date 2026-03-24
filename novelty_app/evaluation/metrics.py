from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


IDEA_SCORE_FIELDS = (
    "importance",
    "novelty",
    "plausibility",
    "feasibility",
    "evaluability",
    "likely_impact",
)


def _safe_rate(num: float, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _extract_idea_scores(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(row.get("idea_scores") or (row.get("hypothesis") or {}).get("idea_scores") or {})


def _has_cue(row: Dict[str, Any]) -> bool:
    cue = dict(row.get("discovery_cue") or {})
    return any(
        bool(cue.get(field))
        for field in (
            "text",
            "goal",
            "include_terms",
            "avoid_terms",
            "preferred_fields",
            "hard_constraints",
            "soft_constraints",
            "counter_queries",
        )
    )


def _task_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row.get("method_name") or "unknown"),
        int(row.get("seed") or 0),
        str(row.get("gold_future_paper_id") or ""),
    )


def _select_best_task_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_task_key(row), []).append(row)

    best_rows: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        cue_active = any(_has_cue(row) for row in group_rows)

        def sort_key(row: Dict[str, Any]) -> Tuple[float, float, int, int, int, float]:
            primary_value = row.get("cue_weighted_rr") if cue_active else row.get("gold_reciprocal_rank")
            primary = float(primary_value or 0.0)
            return (
                primary,
                float(row.get("gold_reciprocal_rank") or 0.0),
                int(bool(row.get("gold_hit_at_1"))),
                int(bool(row.get("gold_hit_at_5"))),
                int(bool(row.get("gold_hit_at_10"))),
                float(_extract_idea_scores(row).get("average_score") or 0.0),
            )

        best_rows.append(max(group_rows, key=sort_key))
    return best_rows


def select_best_task_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _select_best_task_rows(rows)


def _aggregate_idea_scores(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    score_sums = {field: 0.0 for field in IDEA_SCORE_FIELDS}
    score_counts = {field: 0 for field in IDEA_SCORE_FIELDS}
    average_sum = 0.0
    average_count = 0

    for row in rows:
        idea_scores = _extract_idea_scores(row)
        average_score = idea_scores.get("average_score")
        if average_score is not None:
            average_sum += float(average_score)
            average_count += 1
        for field in IDEA_SCORE_FIELDS:
            criterion = idea_scores.get(field) or {}
            score = criterion.get("score")
            if score is None:
                continue
            score_sums[field] += float(score)
            score_counts[field] += 1

    return {
        "n_scored_tasks": average_count,
        "mean_average_idea_score": round(average_sum / average_count, 3) if average_count else None,
        "average_idea_scores": {
            field: round(score_sums[field] / score_counts[field], 3) if score_counts[field] else None
            for field in IDEA_SCORE_FIELDS
        },
    }


def _aggregate_task_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    counts: Dict[str, int] = {}
    gold_ranks: List[int] = []
    gold_rr_sum = 0.0
    cue_weighted_rr_sum = 0.0
    cue_scores: List[float] = []
    cue_weighted_hit_1 = 0.0
    cue_weighted_hit_5 = 0.0
    cue_weighted_hit_10 = 0.0

    for row in rows:
        label = str(row.get("recovery_label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
        gold_rank = row.get("gold_rank")
        if gold_rank is not None:
            gold_ranks.append(int(gold_rank))
        gold_rr_sum += float(row.get("gold_reciprocal_rank") or 0.0)
        cue_weighted_rr_sum += float(row.get("cue_weighted_rr") or 0.0)

        cue_score = row.get("cue_score")
        if cue_score is not None:
            cue_scores.append(float(cue_score))
            cue_norm = max(0.0, min(1.0, (float(cue_score) + 1.0) / 2.5))
            cue_weighted_hit_1 += cue_norm * float(bool(row.get("gold_hit_at_1")))
            cue_weighted_hit_5 += cue_norm * float(bool(row.get("gold_hit_at_5")))
            cue_weighted_hit_10 += cue_norm * float(bool(row.get("gold_hit_at_10")))

    idea_metrics = _aggregate_idea_scores(rows)
    metrics = {
        "n_task_evaluations": total,
        "gold_recall_at_1": _safe_rate(sum(1 for row in rows if row.get("gold_hit_at_1")), total),
        "gold_recall_at_5": _safe_rate(sum(1 for row in rows if row.get("gold_hit_at_5")), total),
        "gold_recall_at_10": _safe_rate(sum(1 for row in rows if row.get("gold_hit_at_10")), total),
        "gold_mrr": round(_safe_rate(gold_rr_sum, total), 6),
        "future_neighbor_only_rate": _safe_rate(counts.get("future_neighbor_only", 0), total),
        "historical_confound_rate": _safe_rate(counts.get("historical_confound", 0), total),
        "gold_recovered_rate": _safe_rate(counts.get("gold_recovered", 0), total),
        "not_recovered_rate": _safe_rate(counts.get("not_recovered", 0), total),
        "median_gold_rank": median(gold_ranks) if gold_ranks else None,
        "cue_weighted_recall_at_1": round(_safe_rate(cue_weighted_hit_1, total), 6) if cue_scores else None,
        "cue_weighted_recall_at_5": round(_safe_rate(cue_weighted_hit_5, total), 6) if cue_scores else None,
        "cue_weighted_recall_at_10": round(_safe_rate(cue_weighted_hit_10, total), 6) if cue_scores else None,
        "cue_weighted_mrr": round(_safe_rate(cue_weighted_rr_sum, total), 6) if cue_scores else None,
        "mean_hypothesis_cue_score": round(sum(cue_scores) / len(cue_scores), 6) if cue_scores else None,
        "counts": counts,
    }
    metrics.update(idea_metrics)
    return metrics


def aggregate_match_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    best_rows = _select_best_task_rows(rows)

    by_method: Dict[str, Dict[str, Any]] = {}
    for method_name in sorted({str(row.get("method_name") or "unknown") for row in best_rows}):
        method_rows = [row for row in best_rows if str(row.get("method_name") or "unknown") == method_name]
        by_method[method_name] = _aggregate_task_rows(method_rows)

    overall = _aggregate_task_rows(best_rows)
    overall.update(
        {
            "n_hypotheses": len(rows),
            "n_scored_hypotheses": sum(1 for row in rows if _extract_idea_scores(row)),
            "by_method": by_method,
        }
    )
    return overall
