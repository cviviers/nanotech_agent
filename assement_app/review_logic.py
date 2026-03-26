from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from novelty_app.evaluation.assessment_bundle import ASSESSMENT_RUBRIC_FIELDS


CRITERION_FIELDS = ASSESSMENT_RUBRIC_FIELDS
CRITERION_NOTE_FIELDS = tuple(f"{field}_note" for field in CRITERION_FIELDS)

ASSESSMENT_SHEET_COLUMNS = (
    "bundle_id",
    "idea_id",
    "reviewer_id",
    "status",
    "revision",
    "started_at",
    "updated_at",
    "submitted_at",
    *CRITERION_FIELDS,
    "overall_rationale",
    *CRITERION_NOTE_FIELDS,
    "confidence",
    "insufficient_context",
    "needs_follow_up",
    "reviewer_notes",
)

CONTENT_FIELDS = (
    *CRITERION_FIELDS,
    "overall_rationale",
    *CRITERION_NOTE_FIELDS,
    "confidence",
    "insufficient_context",
    "needs_follow_up",
    "reviewer_notes",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_score(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return score
    return None


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0, "0", "false", "False", "no", "No"):
        return False
    return bool(value)


def _normalize_confidence(value: Any) -> str:
    text = _clean_text(value).lower()
    return text if text in {"low", "medium", "high"} else ""


def empty_assessment(bundle_id: str, reviewer_id: str, idea_id: str) -> Dict[str, Any]:
    row = {column: "" for column in ASSESSMENT_SHEET_COLUMNS}
    row.update(
        {
            "bundle_id": bundle_id,
            "idea_id": idea_id,
            "reviewer_id": reviewer_id,
            "status": "",
            "revision": 0,
            "started_at": "",
            "updated_at": "",
            "submitted_at": "",
            "insufficient_context": False,
            "needs_follow_up": False,
        }
    )
    for field in CRITERION_FIELDS:
        row[field] = None
    return row


def normalize_assessment_row(record: Dict[str, Any] | None, *, bundle_id: str, reviewer_id: str, idea_id: str) -> Dict[str, Any]:
    row = empty_assessment(bundle_id, reviewer_id, idea_id)
    row.update(dict(record or {}))
    row["bundle_id"] = bundle_id
    row["idea_id"] = idea_id
    row["reviewer_id"] = reviewer_id
    row["status"] = _clean_text(row.get("status")).lower()
    row["revision"] = int(row.get("revision") or 0)
    row["started_at"] = _clean_text(row.get("started_at"))
    row["updated_at"] = _clean_text(row.get("updated_at"))
    row["submitted_at"] = _clean_text(row.get("submitted_at"))
    row["overall_rationale"] = _clean_text(row.get("overall_rationale"))
    row["confidence"] = _normalize_confidence(row.get("confidence"))
    row["reviewer_notes"] = _clean_text(row.get("reviewer_notes"))
    row["insufficient_context"] = _normalize_bool(row.get("insufficient_context"))
    row["needs_follow_up"] = _normalize_bool(row.get("needs_follow_up"))
    for field in CRITERION_FIELDS:
        row[field] = _normalize_score(row.get(field))
    for field in CRITERION_NOTE_FIELDS:
        row[field] = _clean_text(row.get(field))
    return row


def assessment_content(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_assessment_row(
        record,
        bundle_id=str(record.get("bundle_id") or ""),
        reviewer_id=str(record.get("reviewer_id") or ""),
        idea_id=str(record.get("idea_id") or ""),
    )
    return {field: normalized.get(field) for field in CONTENT_FIELDS}


def assessment_has_content(record: Dict[str, Any]) -> bool:
    content = assessment_content(record)
    for field, value in content.items():
        if field in {"insufficient_context", "needs_follow_up"}:
            if bool(value):
                return True
        elif value not in ("", None):
            return True
    return False


def validate_submission(record: Dict[str, Any]) -> List[str]:
    normalized = normalize_assessment_row(
        record,
        bundle_id=str(record.get("bundle_id") or ""),
        reviewer_id=str(record.get("reviewer_id") or ""),
        idea_id=str(record.get("idea_id") or ""),
    )
    missing_scores = [field for field in CRITERION_FIELDS if normalized.get(field) is None]
    errors: List[str] = []
    if missing_scores:
        errors.append("All six judge criteria must be scored before submission.")
    if not normalized.get("overall_rationale"):
        errors.append("Overall rationale is required before submission.")
    return errors


def build_assessment_record(
    *,
    bundle_id: str,
    reviewer_id: str,
    idea_id: str,
    values: Dict[str, Any],
    existing: Dict[str, Any] | None = None,
    submit: bool = False,
    saved_at: str | None = None,
) -> Dict[str, Any]:
    current = normalize_assessment_row(existing, bundle_id=bundle_id, reviewer_id=reviewer_id, idea_id=idea_id)
    merged = dict(current)
    merged.update(values)
    merged = normalize_assessment_row(merged, bundle_id=bundle_id, reviewer_id=reviewer_id, idea_id=idea_id)

    timestamp = saved_at or _utc_now_iso()
    content_changed = assessment_content(current) != assessment_content(merged)
    if not current.get("started_at") and assessment_has_content(merged):
        merged["started_at"] = timestamp
    merged["updated_at"] = timestamp if content_changed or submit else current.get("updated_at") or ""
    if submit:
        merged["status"] = "submitted"
        merged["submitted_at"] = timestamp
    else:
        merged["status"] = "submitted" if current.get("status") == "submitted" else ("draft" if assessment_has_content(merged) else "")
        merged["submitted_at"] = current.get("submitted_at") or ""

    if content_changed or submit:
        merged["revision"] = int(current.get("revision") or 0) + 1
    else:
        merged["revision"] = int(current.get("revision") or 0)
    return merged


def reviewer_assessment_lookup(assessments_df: pd.DataFrame, reviewer_id: str) -> Dict[str, Dict[str, Any]]:
    if assessments_df.empty:
        return {}
    subset = assessments_df.loc[assessments_df["reviewer_id"].astype(str) == str(reviewer_id)]
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in subset.to_dict(orient="records"):
        idea_id = str(row.get("idea_id") or "")
        if idea_id:
            lookup[idea_id] = row
    return lookup


def idea_review_status(assessment: Dict[str, Any] | None) -> str:
    if not assessment:
        return "incomplete"
    status = _clean_text((assessment or {}).get("status")).lower()
    if status == "submitted":
        return "submitted"
    if assessment_has_content(assessment):
        return "draft"
    return "incomplete"


def filter_ideas(
    ideas: Sequence[Dict[str, Any]],
    assessments_df: pd.DataFrame,
    reviewer_id: str,
    *,
    method_names: Sequence[str] | None = None,
    target_types: Sequence[str] | None = None,
    winner_only: bool = False,
    status_filter: str = "all",
    flagged_only: bool = False,
    search_text: str = "",
) -> List[Dict[str, Any]]:
    method_allow = {str(item) for item in (method_names or [])}
    target_allow = {str(item) for item in (target_types or [])}
    query = _clean_text(search_text).lower()
    reviewer_lookup = reviewer_assessment_lookup(assessments_df, reviewer_id)
    out: List[Dict[str, Any]] = []

    for idea in ideas:
        run_context = dict(idea.get("run_context") or {})
        target = dict(idea.get("target") or {})
        hypothesis = dict(idea.get("hypothesis") or {})
        assessment = reviewer_lookup.get(str(idea.get("idea_id") or ""))
        status = idea_review_status(assessment)
        flagged = bool((assessment or {}).get("insufficient_context")) or bool((assessment or {}).get("needs_follow_up"))

        if method_allow and str(run_context.get("method_name") or "") not in method_allow:
            continue
        if target_allow and str(target.get("effective_target", {}).get("target_type") or target.get("target_type") or "") not in target_allow:
            continue
        if winner_only and not bool(idea.get("is_review_packet_winner")):
            continue
        if status_filter != "all":
            if status_filter == "flagged" and not flagged:
                continue
            if status_filter != "flagged" and status != status_filter:
                continue
        if flagged_only and not flagged:
            continue
        if query:
            haystack = " ".join(
                [
                    str(idea.get("idea_id") or ""),
                    str(hypothesis.get("title") or ""),
                    str(hypothesis.get("text") or ""),
                    str(run_context.get("target_id") or ""),
                    str((idea.get("discovery_cue") or {}).get("text") or ""),
                ]
            ).lower()
            if query not in haystack:
                continue
        out.append(idea)

    out.sort(key=lambda idea: tuple((idea.get("run_context") or {}).get("queue_sort_key") or []))
    return out


def next_incomplete_idea_id(ideas: Sequence[Dict[str, Any]], assessments_df: pd.DataFrame, reviewer_id: str) -> str | None:
    reviewer_lookup = reviewer_assessment_lookup(assessments_df, reviewer_id)
    for idea in ideas:
        if idea_review_status(reviewer_lookup.get(str(idea.get("idea_id") or ""))) == "incomplete":
            return str(idea.get("idea_id") or "")
    return None


def model_context_visible(assessment: Dict[str, Any] | None) -> bool:
    return idea_review_status(assessment) == "submitted"


def progress_rows(ideas: Sequence[Dict[str, Any]], assessments_df: pd.DataFrame) -> List[Dict[str, Any]]:
    reviewers = sorted({str(value) for value in assessments_df.get("reviewer_id", pd.Series(dtype="object")).fillna("").tolist() if str(value)})
    if not reviewers:
        reviewers = []
    rows: List[Dict[str, Any]] = []
    total_ideas = len(list(ideas))
    for reviewer_id in reviewers:
        lookup = reviewer_assessment_lookup(assessments_df, reviewer_id)
        draft = submitted = flagged = 0
        for idea in ideas:
            assessment = lookup.get(str(idea.get("idea_id") or ""))
            status = idea_review_status(assessment)
            if status == "draft":
                draft += 1
            elif status == "submitted":
                submitted += 1
            if bool((assessment or {}).get("insufficient_context")) or bool((assessment or {}).get("needs_follow_up")):
                flagged += 1
        rows.append(
            {
                "reviewer_id": reviewer_id,
                "total_ideas": total_ideas,
                "submitted": submitted,
                "draft": draft,
                "incomplete": max(0, total_ideas - submitted - draft),
                "flagged": flagged,
                "completion_rate": round((submitted / total_ideas), 3) if total_ideas else 0.0,
            }
        )
    return rows
