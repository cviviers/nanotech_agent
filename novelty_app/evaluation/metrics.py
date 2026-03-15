from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def aggregate_match_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    total = len(rows)
    counts: Dict[str, int] = {}
    lead_times: List[int] = []
    by_method: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        classification = str(row.get("classification") or "unknown")
        counts[classification] = counts.get(classification, 0) + 1
        if row.get("first_future_year") is not None:
            lead_times.append(int(row["first_future_year"]))
        method = str(row.get("method_name") or "unknown")
        method_counts = by_method.setdefault(method, {"count": 0, "classifications": {}})
        method_counts["count"] += 1
        method_counts["classifications"][classification] = method_counts["classifications"].get(classification, 0) + 1

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

    return {
        "n_hypotheses": total,
        "historical_leakage_rate": _safe_rate(already_present, total),
        "anticipatory_strong_rate": _safe_rate(anticipatory_strong, total),
        "anticipatory_partial_rate": _safe_rate(anticipatory_partial, total),
        "unsupported_rate": _safe_rate(unsupported, total),
        "unrealized_rate": _safe_rate(unrealized, total),
        "novelty_adjusted_hit_rate": _safe_rate(anticipatory_strong, max(0, total - already_present)),
        "median_time_to_first_future_match_year": median(lead_times) if lead_times else None,
        "counts": counts,
        "by_method": by_method,
    }
