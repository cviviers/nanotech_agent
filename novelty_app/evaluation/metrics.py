from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List


IDEA_SCORE_FIELDS = (
    "importance",
    "novelty",
    "plausibility",
    "feasibility",
    "evaluability",
    "likely_impact",
)


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _extract_idea_scores(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(row.get("idea_scores") or (row.get("hypothesis") or {}).get("idea_scores") or {})


def aggregate_match_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    total = len(rows)
    counts: Dict[str, int] = {}
    lead_times: List[int] = []
    by_method: Dict[str, Dict[str, Any]] = {}
    overall_score_sums = {field: 0.0 for field in IDEA_SCORE_FIELDS}
    overall_score_counts = {field: 0 for field in IDEA_SCORE_FIELDS}
    overall_average_sum = 0.0
    overall_average_count = 0

    for row in rows:
        classification = str(row.get("classification") or "unknown")
        counts[classification] = counts.get(classification, 0) + 1
        if row.get("first_future_year") is not None:
            lead_times.append(int(row["first_future_year"]))
        method = str(row.get("method_name") or "unknown")
        method_counts = by_method.setdefault(
            method,
            {
                "count": 0,
                "classifications": {},
                "n_scored_hypotheses": 0,
                "average_idea_scores": {field: None for field in IDEA_SCORE_FIELDS},
                "mean_average_idea_score": None,
                "_score_sums": {field: 0.0 for field in IDEA_SCORE_FIELDS},
                "_score_counts": {field: 0 for field in IDEA_SCORE_FIELDS},
                "_average_sum": 0.0,
                "_average_count": 0,
            },
        )
        method_counts["count"] += 1
        method_counts["classifications"][classification] = method_counts["classifications"].get(classification, 0) + 1

        idea_scores = _extract_idea_scores(row)
        if idea_scores:
            method_counts["n_scored_hypotheses"] += 1
            average_score = idea_scores.get("average_score")
            if average_score is not None:
                method_counts["_average_sum"] += float(average_score)
                method_counts["_average_count"] += 1
                overall_average_sum += float(average_score)
                overall_average_count += 1
            for field in IDEA_SCORE_FIELDS:
                criterion = idea_scores.get(field) or {}
                score = criterion.get("score")
                if score is None:
                    continue
                numeric = float(score)
                method_counts["_score_sums"][field] += numeric
                method_counts["_score_counts"][field] += 1
                overall_score_sums[field] += numeric
                overall_score_counts[field] += 1

    already_present = counts.get("already_present", 0)
    anticipatory_strong = counts.get("anticipatory_strong", 0)
    anticipatory_partial = counts.get("anticipatory_partial", 0)
    unsupported = counts.get("unsupported", 0)
    unrealized = counts.get("unrealized", 0)

    for method, data in by_method.items():
        method_total = int(data["count"])
        method_class = data["classifications"]
        data["anticipatory_strong_rate"] = _safe_rate(method_class.get("anticipatory_strong", 0), method_total)
        data["novelty_adjusted_hit_rate"] = _safe_rate(
            method_class.get("anticipatory_strong", 0),
            max(0, method_total - method_class.get("already_present", 0)),
        )
        if data["_average_count"]:
            data["mean_average_idea_score"] = round(data["_average_sum"] / data["_average_count"], 3)
        for field in IDEA_SCORE_FIELDS:
            if data["_score_counts"][field]:
                data["average_idea_scores"][field] = round(
                    data["_score_sums"][field] / data["_score_counts"][field],
                    3,
                )
        data.pop("_score_sums", None)
        data.pop("_score_counts", None)
        data.pop("_average_sum", None)
        data.pop("_average_count", None)

    return {
        "n_hypotheses": total,
        "n_scored_hypotheses": overall_average_count,
        "historical_leakage_rate": _safe_rate(already_present, total),
        "anticipatory_strong_rate": _safe_rate(anticipatory_strong, total),
        "anticipatory_partial_rate": _safe_rate(anticipatory_partial, total),
        "unsupported_rate": _safe_rate(unsupported, total),
        "unrealized_rate": _safe_rate(unrealized, total),
        "novelty_adjusted_hit_rate": _safe_rate(anticipatory_strong, max(0, total - already_present)),
        "median_time_to_first_future_match_year": median(lead_times) if lead_times else None,
        "mean_average_idea_score": round(overall_average_sum / overall_average_count, 3) if overall_average_count else None,
        "average_idea_scores": {
            field: round(overall_score_sums[field] / overall_score_counts[field], 3)
            if overall_score_counts[field]
            else None
            for field in IDEA_SCORE_FIELDS
        },
        "counts": counts,
        "by_method": by_method,
    }
