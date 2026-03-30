from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Allow `streamlit run app.py` from inside `assement_app/` as well as
# `streamlit run assement_app/app.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from novelty_app.evaluation.assessment_bundle import (
    ASSESSMENT_BUNDLE_SCHEMA_VERSION,
    ASSESSMENT_RUBRIC,
    load_assessment_bundle,
    load_assessment_bundle_bytes,
)

from assement_app.overlap_analysis import analyze_winner_overlap
from assement_app.review_logic import (
    CRITERION_FIELDS,
    CRITERION_NOTE_FIELDS,
    assessment_content,
    assessment_has_content,
    build_assessment_record,
    empty_assessment,
    filter_ideas,
    idea_review_status,
    model_context_visible,
    next_incomplete_idea_id,
    progress_rows,
    reviewer_assessment_lookup,
    validate_submission,
)
from assement_app.workbook_store import get_assessment, load_or_create_workbook, reviewer_ids, save_assessment


DEFAULT_REVIEWS_DIR = APP_DIR / "reviews"
DISCOVERED_BUNDLE_LIMIT = 50
WORKBOOK_DOWNLOAD_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAIN_VIEWS = ("Load", "Review", "Progress")
OVERLAP_HANDLING_SHOW_ALL = "Show all ideas"
OVERLAP_HANDLING_HIDE = "Hide overlapping winners, keep top LLM-scored representative"
DEFAULT_OVERLAP_THRESHOLD = 0.4


def _set_flash(level: str, message: str) -> None:
    st.session_state["flash_message"] = {"level": level, "message": message}


def _render_flash() -> None:
    flash = st.session_state.pop("flash_message", None)
    if not isinstance(flash, dict):
        return
    level = str(flash.get("level") or "info").lower()
    message = str(flash.get("message") or "").strip()
    if not message:
        return
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --assessment-ink: #16313a;
            --assessment-accent: #1f6f78;
            --assessment-accent-soft: #dcefee;
            --assessment-highlight: #b86f52;
            --assessment-paper: #fbfaf5;
            --assessment-line: #d6ddd7;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(220, 239, 238, 0.65), transparent 26rem),
                linear-gradient(180deg, #f4f1e8 0%, #fbfaf5 100%);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        h1, h2, h3 {
            color: var(--assessment-ink);
            font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
            letter-spacing: 0.01em;
        }

        .assessment-hero {
            padding: 1.15rem 1.25rem;
            border: 1px solid rgba(31, 111, 120, 0.15);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(220, 239, 238, 0.78));
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(22, 49, 58, 0.08);
            margin-bottom: 0.9rem;
        }

        .assessment-hero-title {
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 700;
            color: var(--assessment-ink);
            margin-bottom: 0.25rem;
        }

        .assessment-hero-subtitle {
            color: rgba(22, 49, 58, 0.76);
            font-size: 1rem;
        }

        .assessment-chip-row {
            margin-top: 0.85rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .assessment-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            background: rgba(31, 111, 120, 0.10);
            color: var(--assessment-accent);
            border: 1px solid rgba(31, 111, 120, 0.14);
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .assessment-card {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(22, 49, 58, 0.08);
            box-shadow: 0 14px 32px rgba(22, 49, 58, 0.05);
            margin-bottom: 0.75rem;
        }

        .assessment-card-title {
            color: var(--assessment-ink);
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }

        .assessment-card-meta {
            color: rgba(22, 49, 58, 0.74);
            font-size: 0.9rem;
        }

        .assessment-section-label {
            color: var(--assessment-highlight);
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .assessment-score-hint {
            margin-top: -0.45rem;
            margin-bottom: 0.4rem;
            color: rgba(22, 49, 58, 0.72);
            font-size: 0.86rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(22, 49, 58, 0.08);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.90);
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(22, 49, 58, 0.06);
            padding: 0.55rem 0.7rem;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="assessment-hero">
            <div class="assessment-hero-title">Agent Idea Assessment</div>
            <div class="assessment-hero-subtitle">
                Blind-first review workspace for assessment bundles with resumable Excel-backed scoring.
            </div>
            <div class="assessment-chip-row">
                <span class="assessment-chip">Blind-first scoring</span>
                <span class="assessment-chip">Named raters</span>
                <span class="assessment-chip">Excel resume support</span>
                <span class="assessment-chip">Context-rich evidence review</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_card(title: str, body: str = "", *, meta: str = "") -> None:
    html = [
        '<div class="assessment-card">',
        f'<div class="assessment-card-title">{title}</div>',
    ]
    if meta:
        html.append(f'<div class="assessment-card-meta">{meta}</div>')
    if body:
        html.append(f'<div class="assessment-card-meta" style="margin-top:0.35rem">{body}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _discover_bundle_files() -> List[str]:
    candidates = sorted(
        (str(path) for path in REPO_ROOT.rglob(f"*{ASSESSMENT_BUNDLE_SCHEMA_VERSION}.json")),
        key=lambda item: item.lower(),
    )
    return candidates[:DISCOVERED_BUNDLE_LIMIT]


def _discover_workbooks() -> List[str]:
    if not DEFAULT_REVIEWS_DIR.exists():
        return []
    candidates = sorted((str(path) for path in DEFAULT_REVIEWS_DIR.rglob("*.xlsx")), key=lambda item: item.lower())
    return candidates[:DISCOVERED_BUNDLE_LIMIT]


def _bundle() -> Dict[str, Any] | None:
    return st.session_state.get("assessment_bundle")


def _workbook():
    return st.session_state.get("assessment_workbook")


def _reviewer_id() -> str:
    return str(st.session_state.get("active_reviewer_id") or st.session_state.get("reviewer_id") or "").strip()


def _entered_reviewer_id() -> str:
    return str(st.session_state.get("reviewer_id") or "").strip()


def _bundle_path() -> str:
    return str(st.session_state.get("bundle_path") or "").strip()


def _bundle_source() -> str:
    return str(st.session_state.get("bundle_source") or _bundle_path()).strip()


def _uploaded_bundle_file():
    return st.session_state.get("bundle_upload")


def _uploaded_workbook_file():
    return st.session_state.get("workbook_upload")


def _workbook_path() -> str:
    return str(st.session_state.get("workbook_path") or "").strip()


def _workbook_source() -> str:
    return str(st.session_state.get("workbook_source") or _workbook_path()).strip()


def _workbook_download_name() -> str:
    return str(st.session_state.get("workbook_download_name") or "").strip()


def _bundle_ideas() -> List[Dict[str, Any]]:
    bundle = _bundle() or {}
    return list(bundle.get("ideas") or [])


def _basename(value: str) -> str:
    return str(value or "").replace("\\", "/").split("/")[-1].strip()


def _xlsx_filename(name: str, fallback: str) -> str:
    candidate = _basename(name) or fallback
    if not candidate.lower().endswith(".xlsx"):
        candidate = f"{candidate}.xlsx"
    return candidate


def _default_workbook_path(bundle: Dict[str, Any]) -> str:
    DEFAULT_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    return str(DEFAULT_REVIEWS_DIR / f"{bundle.get('bundle_id')}_assessments.xlsx")


def _default_workbook_filename(bundle: Dict[str, Any]) -> str:
    bundle_id = str(bundle.get("bundle_id") or "assessment_bundle").strip() or "assessment_bundle"
    return _xlsx_filename(f"{bundle_id}_assessments.xlsx", "assessment_bundle_assessments.xlsx")


def _session_runtime_dir() -> Path:
    session_id = str(st.session_state.get("runtime_session_id") or "")
    if not session_id:
        session_id = uuid.uuid4().hex
        st.session_state["runtime_session_id"] = session_id
    runtime_dir = Path(tempfile.gettempdir()) / "agent_idea_assessment" / session_id
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _load_selected_bundle() -> tuple[Dict[str, Any], str]:
    uploaded_bundle = _uploaded_bundle_file()
    if uploaded_bundle is not None:
        upload_name = str(getattr(uploaded_bundle, "name", "") or "assessment_bundle.json")
        return load_assessment_bundle_bytes(uploaded_bundle.getvalue()), f"uploaded://{upload_name}"

    bundle_path = _bundle_path()
    if not bundle_path:
        raise ValueError("Provide an assessment bundle JSON path or upload a local bundle file.")
    return load_assessment_bundle(bundle_path), bundle_path


def _resolve_workbook_target(bundle: Dict[str, Any]) -> tuple[str, str, str]:
    default_name = _default_workbook_filename(bundle)
    uploaded_workbook = _uploaded_workbook_file()
    if uploaded_workbook is not None:
        workbook_bytes = uploaded_workbook.getvalue()
        download_name = _xlsx_filename(str(getattr(uploaded_workbook, "name", "") or ""), default_name)
        workbook_source = f"uploaded://{download_name}"
        storage_path = _session_runtime_dir() / "uploaded_workbook.xlsx"
        fingerprint = hashlib.sha256(workbook_bytes).hexdigest()
        existing_fingerprint = str(st.session_state.get("workbook_upload_fingerprint") or "")
        if existing_fingerprint != fingerprint or not storage_path.exists():
            storage_path.write_bytes(workbook_bytes)
            st.session_state["workbook_upload_fingerprint"] = fingerprint
        return str(storage_path), workbook_source, download_name

    workbook_path = _workbook_path()
    if workbook_path:
        st.session_state.pop("workbook_upload_fingerprint", None)
        download_name = _xlsx_filename(workbook_path, default_name)
        return workbook_path, workbook_path, download_name

    st.session_state.pop("workbook_upload_fingerprint", None)
    session_path = _session_runtime_dir() / default_name
    return str(session_path), f"session://{default_name}", default_name


def _render_workbook_download_button(*, key: str, label: str = "Download workbook") -> None:
    workbook = _workbook()
    if not workbook or not workbook.path.exists():
        st.caption("The workbook becomes downloadable after it is loaded.")
        return
    st.download_button(
        label,
        data=workbook.path.read_bytes(),
        file_name=_workbook_download_name() or _default_workbook_filename(_bundle() or {}),
        mime=WORKBOOK_DOWNLOAD_MIME,
        key=key,
    )


def _current_assessment() -> Dict[str, Any] | None:
    bundle = _bundle()
    workbook = _workbook()
    reviewer_id = _reviewer_id()
    idea_id = str(st.session_state.get("current_idea_id") or "")
    if not bundle or not workbook or not reviewer_id or not idea_id:
        return None
    return get_assessment(
        workbook,
        bundle_id=str(bundle.get("bundle_id") or ""),
        reviewer_id=reviewer_id,
        idea_id=idea_id,
    )


def _normalize_main_view(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in MAIN_VIEWS else MAIN_VIEWS[0]


def _active_main_view() -> str:
    return _normalize_main_view(st.session_state.get("active_main_view"))


def _set_main_view(view: str) -> None:
    st.session_state["active_main_view"] = _normalize_main_view(view)


def _render_main_view_nav() -> str:
    active_view = _active_main_view()
    nav_cols = st.columns(len(MAIN_VIEWS))
    for idx, view in enumerate(MAIN_VIEWS):
        if nav_cols[idx].button(
            view,
            key=f"main_view_{view.lower()}",
            use_container_width=True,
            type="primary" if active_view == view else "secondary",
        ):
            if view != active_view:
                _set_main_view(view)
                st.rerun()
    return _active_main_view()


@st.cache_data(show_spinner=False)
def _cached_overlap_analysis(bundle_sha: str, threshold: float, ideas_json: str) -> Dict[str, Any]:
    return analyze_winner_overlap(json.loads(ideas_json), threshold=threshold)


def _overlap_handling() -> str:
    value = str(st.session_state.get("filter_overlap_handling") or OVERLAP_HANDLING_HIDE)
    if value in {OVERLAP_HANDLING_SHOW_ALL, OVERLAP_HANDLING_HIDE}:
        return value
    return OVERLAP_HANDLING_HIDE


def _overlap_threshold() -> float:
    raw_value = st.session_state.get("filter_overlap_threshold", DEFAULT_OVERLAP_THRESHOLD)
    try:
        threshold = float(raw_value)
    except (TypeError, ValueError):
        threshold = DEFAULT_OVERLAP_THRESHOLD
    return max(0.30, min(0.90, threshold))


def _overlap_analysis() -> Dict[str, Any]:
    bundle = _bundle() or {}
    threshold = round(_overlap_threshold(), 2)
    ideas_json = json.dumps(bundle.get("ideas") or [], ensure_ascii=False, sort_keys=True)
    bundle_sha = str(bundle.get("bundle_sha256") or "")
    return _cached_overlap_analysis(bundle_sha, threshold, ideas_json)


def _overlap_keep_idea_ids(overlap_analysis: Dict[str, Any] | None = None) -> set[str] | None:
    if _overlap_handling() != OVERLAP_HANDLING_HIDE:
        return None
    analysis = overlap_analysis or _overlap_analysis()
    keep_ids = {str(idea_id) for idea_id in (analysis.get("visible_idea_ids") or []) if str(idea_id)}
    return keep_ids or None


def _idea_label(idea: Dict[str, Any] | None, *, fallback_id: str = "") -> str:
    payload = dict(idea or {})
    idea_id = str(payload.get("idea_id") or fallback_id or "").strip()
    title = str((payload.get("hypothesis") or {}).get("title") or "").strip()
    if title and idea_id:
        return f"{title} [{idea_id}]"
    return title or idea_id or "Untitled idea"


def _render_overlap_diagnostics(overlap_analysis: Dict[str, Any]) -> None:
    with st.expander("Winner Overlap Diagnostics", expanded=False):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Winner ideas", int(overlap_analysis.get("winner_count") or 0))
        metric_cols[1].metric("Overlap groups", int(overlap_analysis.get("overlap_group_count") or 0))
        metric_cols[2].metric("Hidden winners", int(overlap_analysis.get("hidden_winner_count") or 0))
        metric_cols[3].metric("Diversity score", f"{float(overlap_analysis.get('overall_diversity') or 0.0):.2f}")

        if _overlap_handling() == OVERLAP_HANDLING_HIDE:
            st.caption(
                f"Overlap filtering is active at threshold {_overlap_threshold():.2f}. "
                "One representative winner is kept per overlap group using the model's average idea score as an internal tie-breaker."
            )
        else:
            st.caption(
                "Overlap diagnostics still identify a representative winner internally using the model's average idea score, "
                "but all ideas remain visible until you enable the overlap filter."
            )

        groups = [group for group in (overlap_analysis.get("groups") or []) if int(group.get("group_size") or 0) > 1]
        if not groups:
            st.caption("No overlapping winner groups were detected at the current threshold.")
            return

        idea_lookup = {str(idea.get("idea_id") or ""): idea for idea in _bundle_ideas() if str(idea.get("idea_id") or "")}
        rows: List[Dict[str, Any]] = []
        for group in groups:
            representative_id = str(group.get("representative_id") or "")
            hidden_ids = [str(idea_id) for idea_id in (group.get("hidden_ids") or []) if str(idea_id)]
            representative_idea = idea_lookup.get(representative_id)
            rows.append(
                {
                    "target_key": str(group.get("target_key") or ""),
                    "representative_idea": _idea_label(representative_idea, fallback_id=representative_id),
                    "group_size": int(group.get("group_size") or 0),
                    "mean_overlap": float(group.get("mean_overlap") or 0.0),
                    "hidden_ideas": ", ".join(
                        _idea_label(idea_lookup.get(idea_id), fallback_id=idea_id) for idea_id in hidden_ids
                    ),
                    "shared_evidence_ids": ", ".join(str(value) for value in (group.get("shared_evidence_ids") or [])),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _current_form_values() -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "overall_rationale": st.session_state.get("form_overall_rationale", ""),
        "confidence": st.session_state.get("form_confidence", ""),
        "insufficient_context": bool(st.session_state.get("form_insufficient_context", False)),
        "needs_follow_up": bool(st.session_state.get("form_needs_follow_up", False)),
        "reviewer_notes": st.session_state.get("form_reviewer_notes", ""),
    }
    for field in CRITERION_FIELDS:
        values[field] = st.session_state.get(f"form_{field}", "")
    for field in CRITERION_NOTE_FIELDS:
        values[field] = st.session_state.get(f"form_{field}", "")
    return values


def _load_form_state(idea_id: str) -> None:
    bundle = _bundle()
    reviewer_id = _reviewer_id()
    workbook = _workbook()
    if not bundle or not reviewer_id or not workbook:
        return
    record = get_assessment(
        workbook,
        bundle_id=str(bundle.get("bundle_id") or ""),
        reviewer_id=reviewer_id,
        idea_id=idea_id,
    ) or empty_assessment(str(bundle.get("bundle_id") or ""), reviewer_id, idea_id)
    for field in CRITERION_FIELDS:
        score = record.get(field)
        st.session_state[f"form_{field}"] = "" if score in ("", None) else int(score)
    for field in CRITERION_NOTE_FIELDS:
        st.session_state[f"form_{field}"] = str(record.get(field) or "")
    st.session_state["form_overall_rationale"] = str(record.get("overall_rationale") or "")
    st.session_state["form_confidence"] = str(record.get("confidence") or "")
    st.session_state["form_insufficient_context"] = bool(record.get("insufficient_context") or False)
    st.session_state["form_needs_follow_up"] = bool(record.get("needs_follow_up") or False)
    st.session_state["form_reviewer_notes"] = str(record.get("reviewer_notes") or "")
    st.session_state["loaded_form_idea_id"] = idea_id
    st.session_state["loaded_form_reviewer_id"] = reviewer_id


def _invalidate_loaded_form_state() -> None:
    st.session_state["loaded_form_idea_id"] = None
    st.session_state["loaded_form_reviewer_id"] = None


def _ensure_current_form_loaded() -> None:
    idea_id = str(st.session_state.get("current_idea_id") or "")
    if not idea_id:
        return
    if (
        st.session_state.get("loaded_form_idea_id") != idea_id
        or st.session_state.get("loaded_form_reviewer_id") != _reviewer_id()
    ):
        _load_form_state(idea_id)


def _is_form_dirty(existing: Dict[str, Any] | None) -> bool:
    bundle = _bundle()
    reviewer_id = _reviewer_id()
    idea_id = str(st.session_state.get("current_idea_id") or "")
    if not bundle or not reviewer_id or not idea_id:
        return False
    candidate = build_assessment_record(
        bundle_id=str(bundle.get("bundle_id") or ""),
        reviewer_id=reviewer_id,
        idea_id=idea_id,
        values=_current_form_values(),
        existing=existing,
        submit=False,
    )
    existing_payload = existing or empty_assessment(str(bundle.get("bundle_id") or ""), reviewer_id, idea_id)
    return assessment_content(candidate) != assessment_content(existing_payload)


def _persist_current_form(*, submit: bool = False, reason: str = "") -> bool:
    bundle = _bundle()
    workbook = _workbook()
    reviewer_id = _reviewer_id()
    idea_id = str(st.session_state.get("current_idea_id") or "")
    if not bundle or not workbook or not reviewer_id or not idea_id:
        return False

    existing = _current_assessment()
    values = _current_form_values()
    candidate = build_assessment_record(
        bundle_id=str(bundle.get("bundle_id") or ""),
        reviewer_id=reviewer_id,
        idea_id=idea_id,
        values=values,
        existing=existing,
        submit=submit,
    )

    if submit:
        errors = validate_submission(candidate)
        if errors:
            for error in errors:
                st.error(error)
            return False

    if not assessment_has_content(candidate) and existing is None:
        return False
    if not submit and not _is_form_dirty(existing):
        return False

    save_assessment(workbook, candidate, bundle=bundle, bundle_path=_bundle_source())
    if submit:
        _set_flash("success", "Assessment submitted.")
    elif reason:
        _set_flash("info", f"Draft saved before {reason}.")
    else:
        _set_flash("success", "Draft saved.")
    return True


def _review_queue(*, overlap_analysis: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    workbook = _workbook()
    reviewer_id = _reviewer_id()
    if not workbook or not reviewer_id:
        return []
    keep_idea_ids = _overlap_keep_idea_ids(overlap_analysis)
    return filter_ideas(
        _bundle_ideas(),
        workbook.assessments,
        reviewer_id,
        method_names=st.session_state.get("filter_methods") or [],
        target_types=st.session_state.get("filter_target_types") or [],
        winner_only=bool(st.session_state.get("filter_winner_only", True)),
        status_filter=str(st.session_state.get("filter_status", "all") or "all"),
        flagged_only=bool(st.session_state.get("filter_flagged_only", False)),
        search_text=str(st.session_state.get("filter_search_text", "") or ""),
        keep_idea_ids=sorted(keep_idea_ids) if keep_idea_ids is not None else None,
    )


def _set_current_idea(idea_id: str) -> None:
    st.session_state["current_idea_id"] = idea_id
    st.session_state["jump_idea_id"] = idea_id
    st.session_state["pending_jump_idea_id"] = idea_id
    _invalidate_loaded_form_state()
    _load_form_state(idea_id)


def _schedule_current_idea(idea_id: str) -> None:
    st.session_state["current_idea_id"] = idea_id
    st.session_state["pending_jump_idea_id"] = idea_id
    _invalidate_loaded_form_state()


def _prepare_review_state() -> None:
    pending_jump = st.session_state.pop("pending_jump_idea_id", None)
    current_idea_id = str(st.session_state.get("current_idea_id") or "")
    target_idea_id = str(pending_jump or current_idea_id or "")
    if target_idea_id:
        st.session_state["jump_idea_id"] = target_idea_id
    _ensure_current_form_loaded()


def _navigate_to(idea_id: str, *, reason: str) -> None:
    existing = _current_assessment()
    if _is_form_dirty(existing):
        _persist_current_form(reason=reason)
    _schedule_current_idea(idea_id)
    _set_main_view("Review")
    st.rerun()


def _criterion_score_table(idea: Dict[str, Any]) -> pd.DataFrame:
    idea_scores = dict(((idea.get("judge_context") or {}).get("idea_scores") or {}))
    rows = []
    for field in CRITERION_FIELDS:
        criterion = dict(idea_scores.get(field) or {})
        rows.append(
            {
                "criterion": ASSESSMENT_RUBRIC[field]["label"],
                "score": criterion.get("score"),
                "rationale": criterion.get("rationale") or "",
            }
        )
    return pd.DataFrame(rows)


def _render_scoring_readiness() -> None:
    values = _current_form_values()
    scored_count = sum(1 for field in CRITERION_FIELDS if str(values.get(field) or "").strip())
    has_rationale = bool(str(values.get("overall_rationale") or "").strip())
    progress_cols = st.columns(3)
    progress_cols[0].metric("Criteria scored", f"{scored_count} / {len(CRITERION_FIELDS)}")
    progress_cols[1].metric("Overall rationale", "Added" if has_rationale else "Optional")
    progress_cols[2].metric(
        "Flags",
        int(bool(values.get("insufficient_context"))) + int(bool(values.get("needs_follow_up"))),
    )
    st.progress(scored_count / float(len(CRITERION_FIELDS)))
    if scored_count < len(CRITERION_FIELDS):
        st.caption("Submission becomes available once all six criteria are scored. The overall rationale is optional.")
    else:
        st.caption("Submission requirements are satisfied. The overall rationale is optional.")


def _render_idea_context(idea: Dict[str, Any]) -> None:
    hypothesis = dict(idea.get("hypothesis") or {})
    target = dict(idea.get("target") or {})
    ideation_context = dict(idea.get("ideation_context") or {})
    discovery_cue = dict(idea.get("discovery_cue") or {})
    support_citations = {str(item) for item in (hypothesis.get("support_citations") or []) if str(item)}

    st.subheader(hypothesis.get("title") or "Untitled idea")
    st.write(hypothesis.get("text") or "")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Idea ID", str(idea.get("idea_id") or ""))
    summary_cols[1].metric("Target", str((idea.get("run_context") or {}).get("target_id") or ""))
    summary_cols[2].metric("Method", str((idea.get("run_context") or {}).get("method_name") or ""))
    summary_cols[3].metric("Winner", "Yes" if idea.get("is_review_packet_winner") else "No")

    insight_cols = st.columns(2)
    with insight_cols[0]:
        _render_card(
            "Hypothesis Focus",
            body=str(hypothesis.get("text") or "")[:280],
            meta=f"Support citations: {len(support_citations)}",
        )
    with insight_cols[1]:
        cue_text = str(discovery_cue.get("text") or "").strip()
        _render_card(
            "Discovery Cue",
            body=cue_text or "No discovery cue was provided for this idea.",
            meta=f"Target type: {str((target.get('effective_target') or {}).get('target_type') or target.get('target_type') or '')}",
        )

    context_tabs = st.tabs(["Summary", "Reasoning", "Evidence"])

    with context_tabs[0]:
        with st.expander("Hypothesis Metadata", expanded=False):
            st.json(
                {
                    "support_citations": hypothesis.get("support_citations"),
                    "target": target,
                    "queue_sort_key": (idea.get("run_context") or {}).get("queue_sort_key"),
                }
            )
        if discovery_cue:
            st.markdown('<div class="assessment-section-label">Cue Details</div>', unsafe_allow_html=True)
            st.json(discovery_cue)
        else:
            st.caption("No discovery cue was provided for this idea.")

    with context_tabs[1]:
        reasoning_cols = st.columns(2)
        with reasoning_cols[0]:
            st.markdown('<div class="assessment-section-label">Contrastive Explanation</div>', unsafe_allow_html=True)
            explanation = dict(ideation_context.get("explanation") or {})
            if explanation:
                st.json(explanation)
            else:
                st.caption("No explanation payload was stored.")
        with reasoning_cols[1]:
            st.markdown('<div class="assessment-section-label">Audit</div>', unsafe_allow_html=True)
            audit = dict(ideation_context.get("audit") or {})
            if audit:
                st.json(audit)
            else:
                st.caption("No audit payload was stored.")

    with context_tabs[2]:
        evidence_papers = list(ideation_context.get("evidence_papers") or [])
        if not evidence_papers:
            st.caption("No evidence papers were stored in the bundle.")
            return

        ordered_papers = sorted(
            evidence_papers,
            key=lambda paper: (
                0 if str(paper.get("paper_id") or "") in support_citations else 1,
                str(paper.get("title") or "").lower(),
            ),
        )
        cited_count = sum(1 for paper in ordered_papers if str(paper.get("paper_id") or "") in support_citations)
        with st.expander(
            f"Evidence Pack ({len(ordered_papers)} papers, {cited_count} cited directly)",
            expanded=False,
        ):
            st.caption("Cited evidence is listed first. Use the browser below to inspect one paper at a time without leaving the scoring flow.")
            evidence_stats = st.columns(3)
            evidence_stats[0].metric("Evidence papers", len(ordered_papers))
            evidence_stats[1].metric("Cited directly", cited_count)
            evidence_stats[2].metric("Pack profile", str((ideation_context.get("evidence_pack_meta") or {}).get("profile") or ""))
            evidence_rows = []
            option_labels: List[str] = []
            option_map: Dict[str, Dict[str, Any]] = {}
            for idx, paper in enumerate(ordered_papers, start=1):
                title = str(paper.get("title") or f"Evidence paper {idx}")
                paper_id = str(paper.get("paper_id") or "")
                year = paper.get("publication_year", paper.get("year"))
                cited = paper_id in support_citations
                label = f"{idx}. {title}"
                if paper_id:
                    label += f" [{paper_id}]"
                if year:
                    label += f" ({year})"
                if cited:
                    label += " • cited"
                option_labels.append(label)
                option_map[label] = paper
                evidence_rows.append(
                    {
                        "#": idx,
                        "cited": "yes" if cited else "",
                        "paper_id": paper_id,
                        "year": year,
                        "title": title,
                    }
                )

            st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

            selector_key = f"evidence_focus_{str(idea.get('idea_id') or '')}"
            selected_label = st.selectbox(
                "Inspect evidence paper",
                options=option_labels,
                key=selector_key,
            )
            selected_paper = option_map[selected_label]
            selected_title = str(selected_paper.get("title") or "Evidence paper")
            selected_id = str(selected_paper.get("paper_id") or "")
            selected_year = selected_paper.get("publication_year", selected_paper.get("year"))
            selected_meta = {
                "paper_id": selected_paper.get("paper_id"),
                "year": selected_year,
                "cluster_id": selected_paper.get("cluster_id"),
                "selection_sources": selected_paper.get("selection_sources"),
                "doi": selected_paper.get("doi"),
            }
            inspect_cols = st.columns([1.35, 0.9])
            with inspect_cols[0]:
                _render_card(
                    selected_title,
                    body=str(
                        selected_paper.get("abstract")
                        or selected_paper.get("processed_content")
                        or selected_paper.get("text")
                        or "No abstract/text excerpt stored."
                    ),
                    meta=f"{selected_id} | {selected_year}" if selected_id or selected_year else "",
                )
            with inspect_cols[1]:
                st.markdown('<div class="assessment-section-label">Paper Metadata</div>', unsafe_allow_html=True)
                st.json({key: value for key, value in selected_meta.items() if value not in (None, "", [])})


def _render_assessment_form(idea: Dict[str, Any], assessment: Dict[str, Any] | None, *, queue: List[Dict[str, Any]]) -> None:
    st.markdown("### Human Assessment")
    _render_scoring_readiness()

    score_cols = st.columns(2)
    for idx, field in enumerate(CRITERION_FIELDS):
        rubric = ASSESSMENT_RUBRIC[field]
        with score_cols[idx % 2]:
            st.markdown(f'<div class="assessment-section-label">{rubric["label"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="assessment-score-hint">{rubric["guidance"]}</div>',
                unsafe_allow_html=True,
            )
            st.radio(
                rubric["label"],
                options=["", 1, 2, 3, 4, 5],
                key=f"form_{field}",
                horizontal=True,
                label_visibility="collapsed",
            )

    with st.expander("Optional criterion notes", expanded=False):
        note_cols = st.columns(2)
        for idx, field in enumerate(CRITERION_FIELDS):
            rubric = ASSESSMENT_RUBRIC[field]
            with note_cols[idx % 2]:
                st.text_area(
                    f"{rubric['label']} note",
                    key=f"form_{field}_note",
                    height=90,
                )

    st.text_area(
        "Overall rationale (optional)",
        key="form_overall_rationale",
        height=140,
        help="Optional summary of your overall judgment.",
    )
    flags_col_1, flags_col_2, flags_col_3 = st.columns(3)
    flags_col_1.selectbox("Confidence", options=["", "low", "medium", "high"], key="form_confidence")
    flags_col_2.checkbox("Insufficient context/evidence", key="form_insufficient_context")
    flags_col_3.checkbox("Needs expert follow-up", key="form_needs_follow_up")
    st.text_area("Reviewer notes", key="form_reviewer_notes", height=100)

    button_cols = st.columns(4)
    if button_cols[0].button("Save Draft", use_container_width=True):
        if _persist_current_form(submit=False):
            st.rerun()
    if button_cols[1].button("Submit", use_container_width=True, type="primary"):
        if _persist_current_form(submit=True):
            st.rerun()

    current_id = str(st.session_state.get("current_idea_id") or "")
    current_index = next((idx for idx, item in enumerate(queue) if str(item.get("idea_id")) == current_id), 0)
    prev_id = str(queue[current_index - 1].get("idea_id") or "") if queue and current_index > 0 else ""
    next_id = str(queue[current_index + 1].get("idea_id") or "") if queue and current_index + 1 < len(queue) else ""
    next_incomplete_id = next_incomplete_idea_id(queue, _workbook().assessments, _reviewer_id()) if _workbook() else None

    if button_cols[2].button("Previous", use_container_width=True, disabled=not prev_id):
        _navigate_to(prev_id, reason="moving to the previous idea")
    if button_cols[3].button("Next Incomplete", use_container_width=True, disabled=not next_incomplete_id):
        _navigate_to(str(next_incomplete_id), reason="moving to the next incomplete idea")

    nav_cols = st.columns(2)
    if nav_cols[0].button("Next", use_container_width=True, disabled=not next_id):
        _navigate_to(next_id, reason="moving to the next idea")

    jump_options = {str(item.get("idea_id") or ""): str((item.get("hypothesis") or {}).get("title") or item.get("idea_id")) for item in queue}
    if jump_options:
        st.selectbox(
            "Jump to idea",
            options=list(jump_options.keys()),
            format_func=lambda idea_id: f"{idea_id} | {jump_options.get(idea_id, idea_id)}",
            key="jump_idea_id",
        )
        selected_jump = str(st.session_state.get("jump_idea_id") or current_id)
        if selected_jump and selected_jump != current_id:
            _navigate_to(selected_jump, reason="jumping to another idea")

    current_status = idea_review_status(assessment)
    if model_context_visible(assessment):
        st.markdown("### Post-Submission Comparison")
        st.dataframe(_criterion_score_table(idea), use_container_width=True, hide_index=True)
        judge_context = dict(idea.get("judge_context") or {})
        if judge_context.get("score_summary"):
            st.info(str(judge_context.get("score_summary")))
        st.markdown("#### Retrospective Retrieval Outcomes")
        evaluations = list(((idea.get("benchmark_context") or {}).get("evaluations") or []))
        if evaluations:
            st.dataframe(pd.DataFrame(evaluations), use_container_width=True, hide_index=True)
        else:
            st.caption("No retrospective benchmark rows were stored.")
    else:
        st.info(
            "Blind mode is active for per-criterion judge detail and retrieval outcomes until submission. "
            f"Overlap diagnostics may still use the model's average idea score. Current status: {current_status}."
        )


def _render_progress_tab() -> None:
    bundle = _bundle()
    workbook = _workbook()
    reviewer_id = _reviewer_id()
    if not bundle or not workbook or not reviewer_id:
        st.info("Load an assessment bundle and workbook first.")
        return

    ideas = _bundle_ideas()
    reviewer_progress = progress_rows(ideas, workbook.assessments)
    my_progress = next((row for row in reviewer_progress if row["reviewer_id"] == reviewer_id), None)
    if my_progress:
        cols = st.columns(5)
        cols[0].metric("Submitted", my_progress["submitted"])
        cols[1].metric("Draft", my_progress["draft"])
        cols[2].metric("Incomplete", my_progress["incomplete"])
        cols[3].metric("Flagged", my_progress["flagged"])
        cols[4].metric("Completion", f"{my_progress['completion_rate'] * 100:.1f}%")

    if reviewer_progress:
        st.markdown("### Reviewer Progress")
        st.dataframe(pd.DataFrame(reviewer_progress), use_container_width=True, hide_index=True)

    lookup = reviewer_assessment_lookup(workbook.assessments, reviewer_id)
    disagreement_rows: List[Dict[str, Any]] = []
    for idea in ideas:
        idea_id = str(idea.get("idea_id") or "")
        assessment = lookup.get(idea_id)
        if idea_review_status(assessment) != "submitted":
            continue
        judge_scores = dict(((idea.get("judge_context") or {}).get("idea_scores") or {}))
        for criterion in CRITERION_FIELDS:
            human_score = assessment.get(criterion)
            model_score = (judge_scores.get(criterion) or {}).get("score")
            if human_score in ("", None) or model_score in ("", None):
                continue
            disagreement_rows.append(
                {
                    "idea_id": idea_id,
                    "title": ((idea.get("hypothesis") or {}).get("title") or ""),
                    "criterion": criterion,
                    "human_score": human_score,
                    "model_score": model_score,
                    "difference": int(human_score) - int(model_score),
                }
            )
    st.markdown("### Human vs Model Disagreement")
    if disagreement_rows:
        disagreement_df = pd.DataFrame(disagreement_rows)
        disagreement_df["abs_difference"] = disagreement_df["difference"].abs()
        disagreement_df = disagreement_df.sort_values(by=["abs_difference", "idea_id"], ascending=[False, True])
        st.dataframe(disagreement_df.drop(columns=["abs_difference"]), use_container_width=True, hide_index=True)
    else:
        st.caption("No submitted assessments are available for disagreement analysis yet.")

    st.markdown("### Workbook Status")
    st.write(f"Workbook source: `{_workbook_source()}`")
    st.write(f"Download name: `{_workbook_download_name() or _default_workbook_filename(bundle)}`")
    st.write(f"Bundle: `{_bundle_source()}`")
    st.write(f"Schema: `{bundle.get('schema_version')}`")
    _render_workbook_download_button(key="download_progress_tab")


def _load_bundle_and_workbook() -> None:
    reviewer_id = _entered_reviewer_id()
    if not _bundle_path() and _uploaded_bundle_file() is None:
        st.error("Provide an assessment bundle JSON path or upload a local bundle file.")
        return
    if not reviewer_id:
        st.error("Provide a reviewer id.")
        return
    try:
        bundle, bundle_source = _load_selected_bundle()
        workbook_storage_path, workbook_source, workbook_download_name = _resolve_workbook_target(bundle)
        workbook = load_or_create_workbook(workbook_storage_path, bundle, bundle_path=bundle_source)
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state["assessment_bundle"] = bundle
    st.session_state["assessment_workbook"] = workbook
    st.session_state["bundle_source"] = bundle_source
    st.session_state["workbook_source"] = workbook_source
    st.session_state["workbook_download_name"] = workbook_download_name
    st.session_state["active_reviewer_id"] = reviewer_id
    queue = filter_ideas(bundle.get("ideas") or [], workbook.assessments, reviewer_id, status_filter="all")
    current_idea_id = next_incomplete_idea_id(queue, workbook.assessments, reviewer_id) or (
        str(queue[0].get("idea_id") or "") if queue else ""
    )
    st.session_state["current_idea_id"] = current_idea_id
    st.session_state["jump_idea_id"] = current_idea_id
    _load_form_state(current_idea_id)
    _set_flash("success", "Assessment bundle loaded.")
    _set_main_view("Review")
    st.rerun()


def _render_load_tab() -> None:
    st.markdown("### Bundle Input")
    st.caption("Workbook upload is optional. If you do not upload or select one, the app creates a session workbook automatically and you can download it later.")
    discovered_bundles = _discover_bundle_files()
    discovered_workbooks = _discover_workbooks()
    if "reviewer_id" not in st.session_state and st.session_state.get("active_reviewer_id"):
        st.session_state["reviewer_id"] = str(st.session_state.get("active_reviewer_id") or "")

    if discovered_bundles:
        selected_bundle = st.selectbox(
            "Discovered bundles",
            options=[""] + discovered_bundles,
            index=0,
            help="Select a discovered assessment bundle or enter a path manually below.",
        )
        if selected_bundle:
            st.session_state["bundle_path"] = selected_bundle

    uploaded_bundle = st.file_uploader(
        "Upload local assessment bundle",
        type=["json"],
        key="bundle_upload",
        help="Use this when the app is deployed remotely and the bundle JSON lives on your machine.",
    )
    if uploaded_bundle is not None:
        st.caption(f"Using uploaded bundle: `{uploaded_bundle.name}`")

    st.text_input(
        "Assessment bundle path",
        key="bundle_path",
        help="Optional server-side path. If an uploaded JSON is present, the upload takes precedence.",
    )

    if discovered_workbooks:
        selected_workbook = st.selectbox(
            "Existing workbooks",
            options=[""] + discovered_workbooks,
            index=0,
            help="Select an existing workbook to resume, or enter a new path below.",
        )
        if selected_workbook:
            st.session_state["workbook_path"] = selected_workbook

    uploaded_workbook = st.file_uploader(
        "Upload local workbook",
        type=["xlsx"],
        key="workbook_upload",
        help="Upload an existing review workbook when the app is deployed remotely. If present, the upload takes precedence.",
    )
    if uploaded_workbook is not None:
        st.caption(f"Using uploaded workbook: `{uploaded_workbook.name}`")

    st.text_input(
        "Workbook path",
        key="workbook_path",
        placeholder=str(DEFAULT_REVIEWS_DIR / "my_review.xlsx"),
        help="Optional server-side workbook path. If left empty, the app creates a session workbook that you can download.",
    )
    reviewer_candidates = reviewer_ids(_workbook()) if _workbook() else []
    if reviewer_candidates:
        st.caption(f"Existing reviewers in this workbook: {', '.join(reviewer_candidates)}")
    st.text_input("Reviewer id", key="reviewer_id", placeholder="e.g. reviewer_alice")

    if st.button("Load / Resume Review", type="primary"):
        _load_bundle_and_workbook()

    bundle = _bundle()
    workbook = _workbook()
    if bundle and workbook:
        st.markdown("### Active Session")
        st.write(f"Bundle id: `{bundle.get('bundle_id')}`")
        st.write(f"Ideas in bundle: `{len(bundle.get('ideas') or [])}`")
        st.write(f"Workbook source: `{_workbook_source()}`")
        st.write(f"Download name: `{_workbook_download_name() or _default_workbook_filename(bundle)}`")
        st.write(f"Reviewer: `{_reviewer_id()}`")
        st.write("Blind mode: per-criterion model scores and retrieval labels remain hidden until submission.")
        st.caption("Download the current workbook before closing the session if you want to keep your progress locally.")
        _render_workbook_download_button(key="download_load_tab")


def _render_review_tab() -> None:
    bundle = _bundle()
    workbook = _workbook()
    reviewer_id = _reviewer_id()
    if not bundle or not workbook or not reviewer_id:
        st.info("Load an assessment bundle and workbook first.")
        return

    st.session_state.setdefault("filter_winner_only", True)
    st.session_state.setdefault("filter_overlap_handling", OVERLAP_HANDLING_HIDE)

    with st.expander("Filters", expanded=True):
        available_methods = sorted(
            {str((idea.get("run_context") or {}).get("method_name") or "") for idea in _bundle_ideas() if (idea.get("run_context") or {}).get("method_name")}
        )
        available_targets = sorted(
            {
                str(((idea.get("target") or {}).get("effective_target") or {}).get("target_type") or (idea.get("target") or {}).get("target_type") or "")
                for idea in _bundle_ideas()
                if ((idea.get("target") or {}).get("effective_target") or {}).get("target_type") or (idea.get("target") or {}).get("target_type")
            }
        )
        filter_cols = st.columns(3)
        filter_cols[0].multiselect("Method", options=available_methods, key="filter_methods")
        filter_cols[1].multiselect("Target type", options=available_targets, key="filter_target_types")
        filter_cols[2].selectbox(
            "Reviewer status",
            options=["all", "incomplete", "draft", "submitted", "flagged"],
            key="filter_status",
        )
        filter_cols_2 = st.columns(3)
        filter_cols_2[0].checkbox("Winner ideas only", key="filter_winner_only")
        filter_cols_2[1].checkbox("Flagged only", key="filter_flagged_only")
        filter_cols_2[2].text_input("Search", key="filter_search_text")
        overlap_cols = st.columns([1.7, 1.0])
        overlap_cols[0].selectbox(
            "Overlap handling",
            options=[OVERLAP_HANDLING_SHOW_ALL, OVERLAP_HANDLING_HIDE],
            key="filter_overlap_handling",
        )
        overlap_cols[1].slider(
            "Overlap threshold",
            min_value=0.30,
            max_value=0.90,
            value=DEFAULT_OVERLAP_THRESHOLD,
            step=0.01,
            format="%.2f",
            key="filter_overlap_threshold",
        )

    overlap_analysis = _overlap_analysis()
    _render_overlap_diagnostics(overlap_analysis)

    queue = _review_queue(overlap_analysis=overlap_analysis)
    if not queue:
        st.warning("No ideas match the current filters.")
        return

    current_idea_id = str(st.session_state.get("current_idea_id") or "")
    if current_idea_id not in {str(item.get("idea_id") or "") for item in queue}:
        _set_current_idea(str(queue[0].get("idea_id") or ""))
        current_idea_id = str(st.session_state.get("current_idea_id") or "")

    _prepare_review_state()

    current_index = next((idx for idx, idea in enumerate(queue) if str(idea.get("idea_id") or "") == current_idea_id), 0)
    current_idea = queue[current_index]
    assessment = _current_assessment()
    status = idea_review_status(assessment)

    status_cols = st.columns(4)
    status_cols[0].metric("Queue position", f"{current_index + 1} / {len(queue)}")
    status_cols[1].metric("Review status", status)
    status_cols[2].metric("Reviewer", reviewer_id)
    status_cols[3].metric("Dirty form", "Yes" if _is_form_dirty(assessment) else "No")
    review_action_cols = st.columns([1, 1.4])
    with review_action_cols[0]:
        _render_workbook_download_button(key="download_review_tab", label="Download workbook now")
    with review_action_cols[1]:
        st.caption("Use this at any point to save the current Excel workbook, including in-session workbooks created without an upload.")

    context_col, scoring_col = st.columns([1.35, 1.0], gap="large")
    with context_col:
        _render_idea_context(current_idea)
    with scoring_col:
        _render_assessment_form(current_idea, assessment, queue=queue)


def main() -> None:
    st.set_page_config(page_title="Agent Idea Assessment", page_icon="📋", layout="wide")
    _inject_styles()
    _render_header()
    _render_flash()

    active_view = _render_main_view_nav()

    if active_view == "Load":
        _render_load_tab()
    elif active_view == "Review":
        _render_review_tab()
    else:
        _render_progress_tab()


if __name__ == "__main__":
    main()

# example usage:
# 1. Run this Streamlit app: `streamlit run app.py`
# 2. In the "Load" tab, upload or select an `assessment_bundle_v1.json`, optionally upload a workbook, set a reviewer id, then click "Load / Resume Review".
# 3. In the "Review" tab, assess the ideas using the provided form, and navigate through the queue using the buttons or jump selectbox.
# 4. Download the workbook from the "Load" or "Progress" tab when you want to save the completed review locally.
