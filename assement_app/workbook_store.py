from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from novelty_app.evaluation.assessment_bundle import bundle_hash

from . import APP_VERSION
from .review_logic import ASSESSMENT_SHEET_COLUMNS, CRITERION_FIELDS, progress_rows


META_SHEET = "meta"
IDEAS_SHEET = "ideas"
ASSESSMENTS_SHEET = "assessments"
SUMMARY_SHEET = "summary"


@dataclass
class WorkbookState:
    path: Path
    meta: pd.DataFrame
    ideas: pd.DataFrame
    assessments: pd.DataFrame
    summary: pd.DataFrame


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _bundle_sha(bundle: Dict[str, Any]) -> str:
    if bundle.get("bundle_sha256"):
        return str(bundle["bundle_sha256"])
    return bundle_hash({key: value for key, value in bundle.items() if key != "bundle_sha256"})


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    ordered = columns + [column for column in out.columns if column not in columns]
    return out[ordered]


def _meta_frame(bundle: Dict[str, Any], bundle_path: str) -> pd.DataFrame:
    rows = [
        {"key": "bundle_id", "value": bundle.get("bundle_id")},
        {"key": "bundle_sha256", "value": _bundle_sha(bundle)},
        {"key": "schema_version", "value": bundle.get("schema_version")},
        {"key": "source_kind", "value": bundle.get("source_kind")},
        {"key": "app_version", "value": APP_VERSION},
        {"key": "bundle_path", "value": bundle_path},
        {"key": "created_at", "value": _utc_now_iso()},
        {"key": "opened_at", "value": _utc_now_iso()},
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def _idea_manifest_frame(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idea in bundle.get("ideas") or []:
        run_context = dict(idea.get("run_context") or {})
        target = dict(idea.get("target") or {})
        effective_target = dict(target.get("effective_target") or {})
        hypothesis = dict(idea.get("hypothesis") or {})
        rows.append(
            {
                "idea_id": idea.get("idea_id"),
                "run_id": run_context.get("run_id"),
                "snapshot_id": run_context.get("snapshot_id"),
                "method_name": run_context.get("method_name"),
                "seed": run_context.get("seed"),
                "target_id": run_context.get("target_id"),
                "target_type": effective_target.get("target_type") or target.get("target_type"),
                "hypothesis_id": run_context.get("hypothesis_id"),
                "is_review_packet_winner": bool(idea.get("is_review_packet_winner")),
                "winner_task_count": int(idea.get("winner_task_count") or 0),
                "title": hypothesis.get("title"),
                "cue_text": ((idea.get("discovery_cue") or {}).get("text") or ""),
            }
        )
    return pd.DataFrame(rows)


def _summary_frame(bundle: Dict[str, Any], assessments_df: pd.DataFrame) -> pd.DataFrame:
    ideas = list(bundle.get("ideas") or [])
    rows: List[Dict[str, Any]] = []
    for progress in progress_rows(ideas, assessments_df):
        rows.append({"section": "reviewer_progress", **progress})

    if not assessments_df.empty:
        submitted = assessments_df.loc[assessments_df["status"].astype(str).str.lower() == "submitted"].copy()
        if not submitted.empty:
            for reviewer_id, reviewer_df in submitted.groupby(submitted["reviewer_id"].astype(str)):
                for criterion in CRITERION_FIELDS:
                    numeric = pd.to_numeric(reviewer_df[criterion], errors="coerce").dropna()
                    rows.append(
                        {
                            "section": "criterion_average",
                            "reviewer_id": reviewer_id,
                            "criterion": criterion,
                            "n_submitted": int(len(numeric)),
                            "mean_score": round(float(numeric.mean()), 3) if len(numeric) else None,
                        }
                    )
    return pd.DataFrame(rows)


def _meta_lookup(meta_df: pd.DataFrame) -> Dict[str, str]:
    if meta_df.empty:
        return {}
    return {str(row["key"]): str(row["value"]) for row in meta_df.to_dict(orient="records") if row.get("key")}


def _coerce_assessments_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ASSESSMENT_SHEET_COLUMNS)
    out = _ensure_columns(df, list(ASSESSMENT_SHEET_COLUMNS))
    return out.where(pd.notna(out), "")


def _write_workbook(state: WorkbookState, bundle: Dict[str, Any], bundle_path: str) -> None:
    state.summary = _summary_frame(bundle, state.assessments)
    meta_lookup = _meta_lookup(state.meta)
    meta_lookup["opened_at"] = _utc_now_iso()
    meta_lookup["app_version"] = APP_VERSION
    meta_lookup["bundle_path"] = bundle_path
    state.meta = pd.DataFrame(
        [{"key": key, "value": value} for key, value in meta_lookup.items()],
        columns=["key", "value"],
    )
    with pd.ExcelWriter(state.path, engine="openpyxl") as writer:
        state.meta.to_excel(writer, sheet_name=META_SHEET, index=False)
        state.ideas.to_excel(writer, sheet_name=IDEAS_SHEET, index=False)
        state.assessments.to_excel(writer, sheet_name=ASSESSMENTS_SHEET, index=False)
        state.summary.to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)


def load_or_create_workbook(path: str | Path, bundle: Dict[str, Any], *, bundle_path: str) -> WorkbookState:
    workbook_path = Path(path)
    if workbook_path.exists():
        meta = pd.read_excel(workbook_path, sheet_name=META_SHEET, engine="openpyxl")
        ideas = pd.read_excel(workbook_path, sheet_name=IDEAS_SHEET, engine="openpyxl")
        try:
            assessments = pd.read_excel(workbook_path, sheet_name=ASSESSMENTS_SHEET, engine="openpyxl")
        except ValueError:
            assessments = pd.DataFrame(columns=ASSESSMENT_SHEET_COLUMNS)
        try:
            summary = pd.read_excel(workbook_path, sheet_name=SUMMARY_SHEET, engine="openpyxl")
        except ValueError:
            summary = pd.DataFrame()
        meta_lookup = _meta_lookup(meta)
        if meta_lookup.get("bundle_id") and str(meta_lookup["bundle_id"]) != str(bundle.get("bundle_id") or ""):
            raise ValueError("Workbook bundle id does not match the selected assessment bundle.")
        if meta_lookup.get("bundle_sha256") and str(meta_lookup["bundle_sha256"]) != _bundle_sha(bundle):
            raise ValueError("Workbook bundle hash does not match the selected assessment bundle.")
        state = WorkbookState(
            path=workbook_path,
            meta=meta.fillna(""),
            ideas=_idea_manifest_frame(bundle),
            assessments=_coerce_assessments_df(assessments),
            summary=summary.fillna("") if not summary.empty else pd.DataFrame(),
        )
        _write_workbook(state, bundle, bundle_path)
        return state

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    state = WorkbookState(
        path=workbook_path,
        meta=_meta_frame(bundle, bundle_path),
        ideas=_idea_manifest_frame(bundle),
        assessments=pd.DataFrame(columns=ASSESSMENT_SHEET_COLUMNS),
        summary=pd.DataFrame(),
    )
    _write_workbook(state, bundle, bundle_path)
    return state


def save_assessment(state: WorkbookState, record: Dict[str, Any], *, bundle: Dict[str, Any], bundle_path: str) -> None:
    row = {column: record.get(column, "") for column in ASSESSMENT_SHEET_COLUMNS}
    assessments = _coerce_assessments_df(state.assessments)
    mask = (
        assessments["bundle_id"].astype(str).eq(str(row["bundle_id"]))
        & assessments["reviewer_id"].astype(str).eq(str(row["reviewer_id"]))
        & assessments["idea_id"].astype(str).eq(str(row["idea_id"]))
    )
    if mask.any():
        assessments.loc[mask, list(ASSESSMENT_SHEET_COLUMNS)] = pd.DataFrame([row]).values
    else:
        assessments = pd.concat([assessments, pd.DataFrame([row])], ignore_index=True)
    state.assessments = _coerce_assessments_df(assessments)
    _write_workbook(state, bundle, bundle_path)


def get_assessment(state: WorkbookState, *, bundle_id: str, reviewer_id: str, idea_id: str) -> Dict[str, Any] | None:
    assessments = _coerce_assessments_df(state.assessments)
    mask = (
        assessments["bundle_id"].astype(str).eq(str(bundle_id))
        & assessments["reviewer_id"].astype(str).eq(str(reviewer_id))
        & assessments["idea_id"].astype(str).eq(str(idea_id))
    )
    if not mask.any():
        return None
    return assessments.loc[mask].iloc[0].to_dict()


def reviewer_ids(state: WorkbookState) -> List[str]:
    if state.assessments.empty or "reviewer_id" not in state.assessments.columns:
        return []
    reviewers = {
        str(value).strip()
        for value in state.assessments["reviewer_id"].fillna("").tolist()
        if str(value).strip()
    }
    return sorted(reviewers)
