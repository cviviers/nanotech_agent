"""Agent Console page for publishing snapshots and interacting with the agent backend."""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# Keep both the repo root and `novelty_app/` importable when Streamlit loads
# this page from different working directories.
_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from agents.backend_client import BackendClient
    _BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    try:
        from novelty_app.agents.backend_client import BackendClient
        _BACKEND_IMPORT_ERROR = None
    except Exception:
        BackendClient = None  # type: ignore
        _BACKEND_IMPORT_ERROR = exc

try:
    from agents.corpus_manifest import (
        build_frontend_corpus_manifest,
        hash_paper_ids,
        stable_paper_id_from_row,
        stable_paper_ids,
    )
    from agents.schemas import AnalysisConfig
    from agents.snapshot_builder import build_snapshot_payload
    from evaluation.analysis_v1 import run_analysis_v1
except Exception:  # pragma: no cover
    try:
        from novelty_app.agents.corpus_manifest import (
            build_frontend_corpus_manifest,
            hash_paper_ids,
            stable_paper_id_from_row,
            stable_paper_ids,
        )
        from novelty_app.agents.schemas import AnalysisConfig
        from novelty_app.agents.snapshot_builder import build_snapshot_payload
        from novelty_app.evaluation.analysis_v1 import run_analysis_v1
    except Exception:
        AnalysisConfig = None  # type: ignore
        build_snapshot_payload = None  # type: ignore
        build_frontend_corpus_manifest = None  # type: ignore
        hash_paper_ids = None  # type: ignore
        stable_paper_id_from_row = None  # type: ignore
        stable_paper_ids = None  # type: ignore
        run_analysis_v1 = None  # type: ignore

try:
    from agents.orchestrator_langgraph import build_orchestrator
except Exception:  # pragma: no cover
    build_orchestrator = None  # type: ignore

try:
    from evaluation.run_prospective import run_prospective
    from evaluation.run_retrospective import run_retrospective
    _EVALUATION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    try:
        from novelty_app.evaluation.run_prospective import run_prospective
        from novelty_app.evaluation.run_retrospective import run_retrospective
        _EVALUATION_IMPORT_ERROR = None
    except Exception:
        run_prospective = None  # type: ignore
        run_retrospective = None  # type: ignore
        _EVALUATION_IMPORT_ERROR = exc


DEFAULT_BACKEND_URL = "http://localhost:8088"
DEFAULT_EVALUATION_METHODS = [
    "orchestrator",
    "single_shot_llm",
    "retrieval_summary_direct",
    "heuristic_bridge",
    "pack_query_baseline",
    "random_target_control",
]
LLM_REQUIRED_EVALUATION_METHODS = {"orchestrator", "single_shot_llm", "retrieval_summary_direct"}
SNAPSHOT_PUBLISH_INTENTS = ("prospective", "retrospective")
_USE_SESSION = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    # Treat array-like objects (e.g., numpy vectors) as non-null containers here.
    shape = getattr(value, "shape", None)
    if shape is not None and shape != ():
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    # pandas/numpy may return array-like masks for array inputs; do not collapse them here.
    if hasattr(result, "shape") and getattr(result, "shape", None) not in (None, ()):
        return False
    try:
        return bool(result)
    except Exception:
        return False


def _to_int(value: Any) -> Optional[int]:
    if _is_null(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _to_float(value: Any) -> Optional[float]:
    if _is_null(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if _is_null(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cluster_column() -> Optional[str]:
    selected = st.session_state.get("selected_clustering")
    df = st.session_state.get("df_valid")
    if df is None:
        return None

    candidates: List[str] = []
    if selected:
        candidates.append(f"cluster_{selected}")
    candidates.extend(["cluster_selected", "cluster_kmeans", "cluster_hdbscan", "cluster_leiden"])
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _paper_id_from_row(row: pd.Series, row_label: Any, row_pos: int) -> str:
    del row_label, row_pos
    return stable_paper_id_from_row(row)


def _safe_row_lookup(df: pd.DataFrame, idx_like: Any) -> Tuple[int, Any, pd.Series]:
    if idx_like in df.index:
        pos = int(df.index.get_loc(idx_like))
        return pos, idx_like, df.loc[idx_like]
    pos = int(idx_like)
    row = df.iloc[pos]
    return pos, df.index[pos], row


def _build_snapshot_payload(
    include_raw_rows: bool = True,
    include_embeddings: bool = True,
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    df = st.session_state.get("df_valid")
    if df is None:
        raise ValueError("No df_valid in session state. Load data first.")
    return _build_snapshot_payload_for_df(
        df=df,
        include_raw_rows=include_raw_rows,
        include_embeddings=include_embeddings,
        snapshot_id=st.session_state.get("agent_snapshot_id") or f"snapshot_{uuid.uuid4().hex[:10]}",
        metadata_overrides=_snapshot_metadata_with_defaults(metadata_overrides),
    )


def _get_backend() -> BackendClient:
    if BackendClient is None:
        raise RuntimeError(f"Backend client unavailable: {_BACKEND_IMPORT_ERROR}")
    base_url = (st.session_state.get("agent_backend_url") or DEFAULT_BACKEND_URL).strip()
    return BackendClient(base_url=base_url)


def _normalize_optional_date(value: Any, field_label: str) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return pd.Timestamp(text).date().isoformat()
    except Exception as exc:
        raise ValueError(f"{field_label} must be a valid date in YYYY-MM-DD format.") from exc


def _default_data_json_path() -> str:
    config = st.session_state.get("config") or {}
    return str(config.get("data_path") or "data/cleaned_dataset.json")


def _default_data_dir() -> str:
    return os.fspath(os.path.dirname(_default_data_json_path()) or ".")


def _has_valid_retrospective_dates(
    cutoff_date: Any,
    future_window_start: Any,
    future_window_end: Any,
) -> bool:
    try:
        cutoff = _normalize_optional_date(cutoff_date, "Cutoff date")
        future_start = _normalize_optional_date(future_window_start, "Future window start")
        future_end = _normalize_optional_date(future_window_end, "Future window end")
    except ValueError:
        return False
    if not cutoff or not future_start or not future_end:
        return False
    cutoff_ts = pd.Timestamp(cutoff)
    future_start_ts = pd.Timestamp(future_start)
    future_end_ts = pd.Timestamp(future_end)
    return bool(future_start_ts > cutoff_ts and future_end_ts >= future_start_ts)


def _snapshot_retrospective_context(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = dict((snapshot or {}).get("metadata") or {})
    extra = dict(metadata.get("extra") or {})
    split_role = str(metadata.get("split_role") or "").strip()
    has_dates = _has_valid_retrospective_dates(
        metadata.get("cutoff_date"),
        metadata.get("future_window_start"),
        metadata.get("future_window_end"),
    )
    has_bundle_metadata = bool(
        extra.get("retrospective_bundle_artifact_id")
        or extra.get("bundle_prefix")
        or extra.get("retrospective_bundle_kind") == "retrospective_snapshot_bundle"
    )
    can_reuse_snapshot = bool(has_dates and split_role in {"", "historical"} and has_bundle_metadata)
    if not has_dates:
        reuse_reason = "Active snapshot is missing valid cutoff and future-window metadata."
    elif split_role not in {"", "historical"}:
        reuse_reason = (
            f"Active snapshot split_role is `{split_role}`; retrospective reuse requires a historical snapshot."
        )
    elif not has_bundle_metadata:
        reuse_reason = (
            "Active snapshot has retrospective dates but no retrospective bundle metadata; "
            "the runner will rebuild the historical snapshot from the dataset instead."
        )
    else:
        reuse_reason = ""
    return {
        "has_dates": has_dates,
        "split_role": split_role or None,
        "cutoff_date": str(metadata.get("cutoff_date") or "").strip(),
        "future_window_start": str(metadata.get("future_window_start") or "").strip(),
        "future_window_end": str(metadata.get("future_window_end") or "").strip(),
        "has_bundle_metadata": has_bundle_metadata,
        "can_reuse_snapshot": can_reuse_snapshot,
        "reuse_reason": reuse_reason,
    }


def _retrospective_requested_in_ui_state() -> bool:
    return _retrospective_publish_intent_enabled()


def _resolved_evaluation_mode(snapshot: Optional[Dict[str, Any]]) -> str:
    if _snapshot_retrospective_context(snapshot).get("has_dates"):
        return "retrospective"
    if _retrospective_requested_in_ui_state():
        return "retrospective"
    return "prospective"


def _snapshot_embedding_scope(snapshot: Optional[Dict[str, Any]]) -> str:
    metadata = dict((snapshot or {}).get("metadata") or {})
    candidates: List[str] = []
    if metadata.get("embedding_source") is not None:
        candidates.append(str(metadata.get("embedding_source") or ""))
    if metadata.get("embedding_name") is not None:
        candidates.append(str(metadata.get("embedding_name") or ""))
    analysis_config = metadata.get("analysis_config")
    if isinstance(analysis_config, dict) and analysis_config.get("embedding_name") is not None:
        candidates.append(str(analysis_config.get("embedding_name") or ""))
    for value in candidates:
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return "unknown"


def _snapshot_catalog_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        snapshots = payload.get("snapshots")
        if isinstance(snapshots, list):
            return [dict(s) for s in snapshots if isinstance(s, dict)]
    if isinstance(payload, list):
        return [dict(s) for s in payload if isinstance(s, dict)]
    return []


def _snapshot_option_label(snapshot: Dict[str, Any]) -> str:
    snapshot_id = str(snapshot.get("snapshot_id") or "").strip() or "<missing-id>"
    metadata = dict(snapshot.get("metadata") or {})
    split_role = str(metadata.get("split_role") or "").strip() or "unspecified"
    scope = _snapshot_embedding_scope(snapshot)
    created_at = str(snapshot.get("created_at") or "").strip()
    if created_at:
        return f"{snapshot_id} | role={split_role} | embedding={scope} | {created_at}"
    return f"{snapshot_id} | role={split_role} | embedding={scope}"


def _prioritize_cue_source_snapshots(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, snapshot in enumerate(snapshots):
        snapshot_id = str(snapshot.get("snapshot_id") or "").strip()
        if not snapshot_id:
            continue
        metadata = dict(snapshot.get("metadata") or {})
        split_role = str(metadata.get("split_role") or "").strip().lower()
        scope = _snapshot_embedding_scope(snapshot)
        is_full_or_unspecified = split_role in {"", "full"}
        if scope == "qwen" and is_full_or_unspecified:
            rank = 0
        elif scope == "qwen":
            rank = 1
        elif is_full_or_unspecified:
            rank = 2
        else:
            rank = 3
        ranked.append((rank, idx, snapshot))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [snapshot for _, _, snapshot in ranked]


def _suggest_cue_source_snapshot_id(snapshots: List[Dict[str, Any]]) -> Optional[str]:
    prioritized = _prioritize_cue_source_snapshots(snapshots)
    for snapshot in prioritized:
        metadata = dict(snapshot.get("metadata") or {})
        split_role = str(metadata.get("split_role") or "").strip().lower()
        scope = _snapshot_embedding_scope(snapshot)
        if scope == "qwen" and split_role in {"", "full"}:
            return str(snapshot.get("snapshot_id") or "").strip() or None
    for snapshot in prioritized:
        if _snapshot_embedding_scope(snapshot) == "qwen":
            return str(snapshot.get("snapshot_id") or "").strip() or None
    return None


def _resolve_snapshot_by_id(
    snapshot_id: str,
    snapshot_by_id: Dict[str, Dict[str, Any]],
    backend: BackendClient,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    sid = str(snapshot_id or "").strip()
    if not sid:
        return None, None
    if sid in snapshot_by_id:
        return snapshot_by_id[sid], None
    try:
        return backend.get_snapshot(sid), None
    except Exception as exc:
        return None, str(exc)


def _cue_source_scope_error(
    cue_source_snapshot_id: str,
    cue_source_snapshot: Optional[Dict[str, Any]],
    cue_source_lookup_error: Optional[str] = None,
) -> Optional[str]:
    sid = str(cue_source_snapshot_id or "").strip()
    if not sid:
        return "Cue source snapshot ID is required when discovery cue is active."
    if cue_source_snapshot is None:
        if cue_source_lookup_error:
            return f"Cue source snapshot `{sid}` lookup failed: {cue_source_lookup_error}"
        return f"Cue source snapshot `{sid}` was not found."
    scope = _snapshot_embedding_scope(cue_source_snapshot)
    if scope != "qwen":
        return (
            f"Cue source snapshot `{sid}` uses embedding scope `{scope}`; "
            "cue similarity requires `qwen` embeddings."
        )
    return None


def _qwen_base_url_issue(base_url: str) -> Optional[str]:
    text = str(base_url or "").strip()
    if not text:
        return "Qwen base URL is required for retrospective evaluation."
    candidate = text if "://" in text else f"http://{text}"
    parsed = urlparse(candidate)
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return "Qwen base URL must include a valid host."
    if host == "0.0.0.0":
        return (
            "Qwen base URL host `0.0.0.0` is not reachable from the evaluator. "
            "Use `127.0.0.1` or a routable host."
        )
    return None


def _render_preflight_status(hard_failures: List[str], warnings: List[str]) -> None:
    st.caption("Preflight")
    if not hard_failures and not warnings:
        st.success("Ready to run.")
        return
    if hard_failures:
        st.error("Hard failures must be resolved before running.")
        for message in hard_failures:
            st.caption(f"- {message}")
    if warnings:
        st.warning("Warnings")
        for message in warnings:
            st.caption(f"- {message}")


def _publication_year_span(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or "publication_year" not in df.columns:
        return None
    years = pd.to_numeric(df["publication_year"], errors="coerce").dropna()
    if years.empty:
        return None
    min_year = int(years.min())
    max_year = int(years.max())
    if min_year == max_year:
        return str(min_year)
    return f"{min_year}-{max_year}"


def _sanitize_snapshot_id_fragment(value: str, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _cli_quote(value: Any) -> str:
    text = str(value or "")
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in {'"', "'", ","}):
        return json.dumps(text)
    return text


def _command_preview(parts: List[str]) -> str:
    return " ".join(_cli_quote(part) for part in parts if str(part or "").strip())


def _publication_dates(df: pd.DataFrame) -> pd.Series:
    def _numeric_column(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series([None] * len(df), index=df.index, dtype="float64")

    years = _numeric_column("publication_year")
    months = _numeric_column("publication_month")
    days = _numeric_column("publication_day")

    values: List[pd.Timestamp] = []
    for idx in range(len(df)):
        year = years.iloc[idx]
        if pd.isna(year):
            values.append(pd.NaT)
            continue
        y = int(year)
        month = int(months.iloc[idx]) if not pd.isna(months.iloc[idx]) else 12
        month = min(max(month, 1), 12)
        if not pd.isna(days.iloc[idx]):
            day = int(days.iloc[idx])
        else:
            day = calendar.monthrange(y, month)[1]
        day = min(max(day, 1), calendar.monthrange(y, month)[1])
        values.append(pd.Timestamp(year=y, month=month, day=day))
    return pd.Series(values, index=df.index, name="publication_date")


def _subset_array(values: Any, keep_mask: List[bool]) -> Any:
    if values is None:
        return None
    try:
        return values[keep_mask]
    except Exception:
        pass
    try:
        return [value for value, keep in zip(values, keep_mask) if keep]
    except Exception:
        return None


def _subset_gap_regions(
    df: pd.DataFrame,
    gap_regions: Optional[List[List[Any]]],
    keep_mask: List[bool],
) -> List[List[int]]:
    if not gap_regions:
        return []

    label_to_pos: Dict[Any, int] = {}
    for pos, label in enumerate(df.index):
        label_to_pos.setdefault(label, pos)

    pos_map: Dict[int, int] = {}
    next_pos = 0
    for orig_pos, keep in enumerate(keep_mask):
        if keep:
            pos_map[orig_pos] = next_pos
            next_pos += 1

    subset_regions: List[List[int]] = []
    for region in gap_regions:
        subset_region: List[int] = []
        seen: set[int] = set()
        for idx_like in region:
            orig_pos: Optional[int] = None
            if idx_like in label_to_pos:
                orig_pos = label_to_pos[idx_like]
            else:
                try:
                    orig_pos = int(idx_like)
                except Exception:
                    orig_pos = None
            if orig_pos is None:
                continue
            new_pos = pos_map.get(orig_pos)
            if new_pos is None or new_pos in seen:
                continue
            seen.add(new_pos)
            subset_region.append(new_pos)
        if subset_region:
            subset_regions.append(subset_region)
    return subset_regions


def _build_snapshot_payload_for_df(
    *,
    df: pd.DataFrame,
    include_raw_rows: bool,
    include_embeddings: bool,
    snapshot_id: str,
    metadata_overrides: Optional[Dict[str, Any]] = None,
    gap_regions: Optional[List[List[Any]]] = None,
    llm_results: Any = _USE_SESSION,
    x_primary: Any = _USE_SESSION,
    x_umap_2d: Any = _USE_SESSION,
    selected_clustering: Any = _USE_SESSION,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return build_snapshot_payload(
        df=df,
        gap_regions=gap_regions if gap_regions is not None else st.session_state.get("gap_regions") or [],
        llm_results=st.session_state.get("llm_results") if llm_results is _USE_SESSION else llm_results,
        selected_clustering=(
            st.session_state.get("selected_clustering") if selected_clustering is _USE_SESSION else selected_clustering
        ),
        x_primary=st.session_state.get("X_primary") if x_primary is _USE_SESSION else x_primary,
        x_umap_2d=st.session_state.get("X_umap_2d") if x_umap_2d is _USE_SESSION else x_umap_2d,
        include_raw_rows=include_raw_rows,
        include_embeddings=include_embeddings,
        snapshot_id=snapshot_id,
        source="streamlit_agent_console",
        metadata_overrides=_snapshot_metadata_with_defaults(metadata_overrides),
    )


def _snapshot_metadata_with_defaults(metadata_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensure agent-console snapshots carry stable embedding scope metadata."""
    merged = dict(metadata_overrides or {})
    if merged.get("embedding_source"):
        return merged
    config = st.session_state.get("config") or {}
    primary_embedding = str(config.get("primary_embedding") or "").strip()
    if primary_embedding:
        merged["embedding_source"] = primary_embedding
    return merged


def _snapshot_publish_intent_from_state() -> str:
    raw = str(st.session_state.get("agent_publish_snapshot_intent") or "prospective").strip().lower()
    if raw in SNAPSHOT_PUBLISH_INTENTS:
        return raw
    return "prospective"


def _retrospective_publish_intent_enabled() -> bool:
    return _snapshot_publish_intent_from_state() == "retrospective"


def _retrospective_metadata_from_state(*, require_enabled: bool = False) -> Dict[str, Any]:
    enabled = _retrospective_publish_intent_enabled()
    if not enabled:
        if require_enabled:
            raise ValueError("Set Snapshot intent to `Retrospective` before publishing retrospective metadata.")
        return {}

    split_role = str(st.session_state.get("agent_publish_split_role") or "historical")
    cutoff_date = _normalize_optional_date(st.session_state.get("agent_publish_cutoff_date"), "Cutoff date")
    future_window_start = _normalize_optional_date(
        st.session_state.get("agent_publish_future_window_start"),
        "Future window start",
    )
    future_window_end = _normalize_optional_date(
        st.session_state.get("agent_publish_future_window_end"),
        "Future window end",
    )

    missing_fields = [
        label
        for label, value in (
            ("cutoff date", cutoff_date),
            ("future window start", future_window_start),
            ("future window end", future_window_end),
        )
        if not value
    ]
    if missing_fields:
        raise ValueError("Retrospective snapshot metadata requires " + ", ".join(missing_fields) + ".")

    cutoff_ts = pd.Timestamp(cutoff_date)
    future_start_ts = pd.Timestamp(future_window_start)
    future_end_ts = pd.Timestamp(future_window_end)
    if future_start_ts <= cutoff_ts:
        raise ValueError("Future window start must be after the cutoff date.")
    if future_end_ts < future_start_ts:
        raise ValueError("Future window end must be on or after future window start.")

    return {
        "split_role": split_role,
        "cutoff_date": cutoff_date,
        "future_window_start": future_window_start,
        "future_window_end": future_window_end,
    }


def _retrospective_split_plan() -> Dict[str, Any]:
    return _retrospective_split_plan_for_df(st.session_state.get("df_valid"))


def _retrospective_split_plan_for_df(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None:
        raise ValueError("No df_valid in session state. Load data first.")

    metadata = _retrospective_metadata_from_state(require_enabled=True)
    publication_dates = _publication_dates(df)
    cutoff_ts = pd.Timestamp(metadata["cutoff_date"])
    future_start_ts = pd.Timestamp(metadata["future_window_start"])
    future_end_ts = pd.Timestamp(metadata["future_window_end"])

    historical_mask = (publication_dates.notna()) & (publication_dates <= cutoff_ts)
    future_mask = (publication_dates.notna()) & (publication_dates >= future_start_ts) & (publication_dates <= future_end_ts)
    excluded_mask = publication_dates.notna() & ~(historical_mask | future_mask)
    undated_mask = publication_dates.isna()

    return {
        "metadata": metadata,
        "publication_dates": publication_dates,
        "historical_mask": historical_mask,
        "future_mask": future_mask,
        "excluded_mask": excluded_mask,
        "undated_mask": undated_mask,
        "counts": {
            "historical": int(historical_mask.sum()),
            "future": int(future_mask.sum()),
            "excluded": int(excluded_mask.sum()),
            "undated": int(undated_mask.sum()),
        },
    }


def _frontend_authoritative_df() -> pd.DataFrame:
    df = st.session_state.get("df_valid_full")
    if df is not None:
        return df.copy()
    df = st.session_state.get("df_valid")
    if df is None:
        raise ValueError("No frontend corpus is loaded.")
    return df.copy()


def _frontend_authoritative_embeddings() -> Dict[str, Any]:
    embeddings = st.session_state.get("embeddings_dict_full") or {}
    if embeddings:
        return {key: value.copy() if value is not None else None for key, value in embeddings.items()}
    embeddings = st.session_state.get("embeddings_dict") or {}
    if not embeddings:
        raise ValueError("No embeddings are loaded for the frontend corpus.")
    return {key: value.copy() if value is not None else None for key, value in embeddings.items()}


def _normalized_frontend_analysis_config() -> Dict[str, Any]:
    if AnalysisConfig is None:
        raise RuntimeError("AnalysisConfig import is unavailable.")

    selected_clustering = st.session_state.get("selected_clustering")
    if selected_clustering not in {"hdbscan", "kmeans", "leiden"}:
        raise ValueError(
            "Retrospective export currently requires `kmeans`, `hdbscan`, or community detection (`leiden`)."
        )

    if st.session_state.get("gap_config") is None:
        raise ValueError("Gap analysis configuration is missing. Run the Gap Analysis page before retrospective export.")

    config = st.session_state.get("config") or {}
    clustering_config = st.session_state.get("clustering_config") or {}
    gap_config = st.session_state.get("gap_config") or {}
    embedding_processing_config = st.session_state.get("embedding_processing_config") or {}
    df = _frontend_authoritative_df()
    working_df = st.session_state.get("df_valid")

    x_pca = st.session_state.get("X_pca")
    x_umap_2d = st.session_state.get("X_umap_2d")
    primary_embedding = str(config.get("primary_embedding") or "qwen")
    use_pca_for_analysis = x_pca is not None
    pca_components = int(x_pca.shape[1]) if x_pca is not None else int(embedding_processing_config.get("pca_components") or 102)
    kmeans_n_clusters: Optional[int] = (
        int(clustering_config["kmeans_n_clusters"]) if clustering_config.get("kmeans_n_clusters") is not None else None
    )
    if selected_clustering == "kmeans" and working_df is not None and "cluster_kmeans" in working_df.columns:
        kmeans_values = pd.to_numeric(working_df["cluster_kmeans"], errors="coerce").dropna()
        unique_clusters = {int(value) for value in kmeans_values.tolist()}
        if unique_clusters and kmeans_n_clusters is None:
            kmeans_n_clusters = len(unique_clusters)

    analysis_config = AnalysisConfig(
        embedding_name=primary_embedding,
        use_pca_for_analysis=use_pca_for_analysis,
        pca_components=pca_components,
        clustering_method=selected_clustering,
        kmeans_n_clusters=kmeans_n_clusters if selected_clustering == "kmeans" else None,
        hdbscan_min_cluster_size=int(clustering_config.get("hdbscan_min_cluster_size") or 5),
        hdbscan_min_samples=int(clustering_config.get("hdbscan_min_samples") or 10),
        community_detection_algorithm=str(
            clustering_config.get("community_detection_algorithm") or "leiden"
        ),
        community_resolution=float(clustering_config.get("leiden_resolution") or 1.0),
        community_graph_k=int(clustering_config.get("knn_graph_k") or 21),
        community_graph_metric=str(clustering_config.get("community_graph_metric") or "cosine"),
        knn_graph_k=int(gap_config.get("knn_graph_k") or clustering_config.get("knn_graph_k") or 21),
        density_metric=str(gap_config.get("density_metric") or "cosine"),
        density_k_list=[int(item) for item in (gap_config.get("k_neighbors") or [10, 20, 30, 50])],
        gap_quantile=float(gap_config.get("gap_quantile") or 0.95),
        min_gap_region_size=int(gap_config.get("min_gap_region_size") or 3),
        random_seed=int(st.session_state.get("random_seed") or config.get("random_seed") or 42),
        compute_umap=bool(x_umap_2d is not None),
        umap_neighbors=int(embedding_processing_config.get("umap_neighbors") or 50),
        umap_min_dist=float(embedding_processing_config.get("umap_min_dist") or 0.1),
        notes="normalized_from_streamlit_frontend",
    ).model_dump()
    st.session_state.frontend_analysis_config = analysis_config
    return analysis_config


def _frontend_corpus_manifest_from_state() -> Dict[str, Any]:
    manifest = st.session_state.get("frontend_corpus_manifest")
    if manifest:
        return dict(manifest)
    if build_frontend_corpus_manifest is None:
        raise RuntimeError("Frontend corpus manifest helpers are unavailable.")
    config = st.session_state.get("config") or {}
    keyword_filters = st.session_state.get("frontend_keyword_filters") or {}
    data_path = str(config.get("data_path") or "data/cleaned_dataset.json")
    data_dir = os.fspath(os.path.dirname(data_path) or ".")
    manifest = build_frontend_corpus_manifest(
        _frontend_authoritative_df(),
        sample_n=config.get("sample_n"),
        random_seed=int(st.session_state.get("random_seed") or config.get("random_seed") or 42),
        title_exclusion_keywords=keyword_filters.get("title_exclusion_keywords") or [],
        abstract_exclusion_keywords=keyword_filters.get("abstract_exclusion_keywords") or [],
        embedding_source=str(config.get("primary_embedding") or "qwen"),
        available_embeddings=config.get("embedding_cols") or [str(config.get("primary_embedding") or "qwen")],
        data_json=data_path,
        data_dir=data_dir,
    )
    st.session_state.frontend_corpus_manifest = manifest
    return dict(manifest)


def _snapshot_metadata_overrides() -> Dict[str, Any]:
    intent = st.radio(
        "Snapshot intent",
        options=list(SNAPSHOT_PUBLISH_INTENTS),
        key="agent_publish_snapshot_intent",
        horizontal=True,
        format_func=lambda value: str(value).title(),
        help=(
            "Use `Prospective` for full working-corpus snapshots. Use `Retrospective` when publishing "
            "dated split metadata for benchmark workflows."
        ),
    )
    if intent != "retrospective":
        st.caption("Prospective intent publishes the current working corpus without retrospective split metadata.")
        return {}

    year_span = _publication_year_span(st.session_state.get("df_valid"))
    if year_span:
        st.caption(f"Current dataframe publication_year span: {year_span}")

    split_role = st.selectbox(
        "Snapshot split role",
        options=["historical", "full", "future"],
        key="agent_publish_split_role",
        help=(
            "Use `historical` for retrospective historical snapshots. Use `full` for cue-source "
            "or inspection snapshots that still carry retrospective cutoff metadata."
        ),
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        cutoff_raw = st.text_input(
            "Cutoff date",
            key="agent_publish_cutoff_date",
            help="Historical papers must fall on or before this date.",
            placeholder="YYYY-MM-DD",
        )
    with col2:
        future_start_raw = st.text_input(
            "Future window start",
            key="agent_publish_future_window_start",
            help="First future-paper date used by the retrospective benchmark.",
            placeholder="YYYY-MM-DD",
        )
    with col3:
        future_end_raw = st.text_input(
            "Future window end",
            key="agent_publish_future_window_end",
            help="Last future-paper date used by the retrospective benchmark.",
            placeholder="YYYY-MM-DD",
        )

    return _retrospective_metadata_from_state(require_enabled=True)


def _parse_multivalue_text(value: str) -> List[str]:
    parts = str(value or "").replace(",", "\n").splitlines()
    out: List[str] = []
    seen = set()
    for part in parts:
        text = part.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _parse_cluster_pair_text(value: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for line in str(value or "").splitlines():
        text = line.strip()
        if not text:
            continue
        normalized = text.replace(":", ",").replace(" ", ",")
        tokens = [token for token in normalized.split(",") if token]
        if len(tokens) != 2:
            raise ValueError(f"Invalid cluster pair `{text}`. Use `cluster_a,cluster_b` per line.")
        pair = (int(tokens[0]), int(tokens[1]))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _render_evaluation_progress(placeholder: Any, progress_payload: Optional[Dict[str, Any]]) -> None:
    if progress_payload is None:
        return
    phase = str(progress_payload.get("phase") or "running")
    status = str(progress_payload.get("status") or "running")
    message = str(progress_payload.get("message") or status)
    current = progress_payload.get("current")
    total = progress_payload.get("total")
    progress_text = f"[{phase}] {message}"
    if current is not None and total is not None:
        progress_text = f"{progress_text} ({current}/{total})"

    with placeholder.container():
        if total is not None and current is not None and int(total or 0) > 0:
            ratio = min(max(float(current) / float(total), 0.0), 1.0)
            st.progress(ratio, text=progress_text)
        elif status == "failed":
            st.error(progress_text)
        elif status == "completed":
            st.success(progress_text)
        else:
            st.info(progress_text)

        cols = st.columns(4)
        cols[0].metric("Phase", phase)
        cols[1].metric("Completed", int(current) if current is not None else 0)
        cols[2].metric("Total", int(total) if total is not None else 0)
        cols[3].metric("Failures", int(progress_payload.get("n_failures") or 0))


def _should_publish_cutoff_filtered_snapshot(metadata_overrides: Optional[Dict[str, Any]]) -> bool:
    split_role = str((metadata_overrides or {}).get("split_role") or "").strip().lower()
    return split_role == "historical"


def _set_active_published_snapshot(snapshot_id: str) -> None:
    sid = str(snapshot_id or "").strip()
    if not sid:
        return
    st.session_state["agent_snapshot_id"] = sid
    st.session_state["agent_eval_snapshot_id"] = sid
    st.session_state["agent_eval_cue_source_snapshot_id"] = sid


def _upsert_snapshot_record_in_cache(cache_payload: Any, snapshot_record: Dict[str, Any]) -> Dict[str, Any]:
    sid = str(snapshot_record.get("snapshot_id") or "").strip()
    if not sid:
        if isinstance(cache_payload, dict):
            return dict(cache_payload)
        return {"snapshots": _snapshot_catalog_records(cache_payload)}

    normalized_record = dict(snapshot_record)
    normalized_record["snapshot_id"] = sid
    existing_records = _snapshot_catalog_records(cache_payload)
    merged_records = [normalized_record] + [
        dict(item)
        for item in existing_records
        if str(item.get("snapshot_id") or "").strip() != sid
    ]

    if isinstance(cache_payload, dict):
        out = dict(cache_payload)
        out.pop("error", None)
        out["snapshots"] = merged_records
        return out
    return {"snapshots": merged_records}


def _sync_snapshot_cache_after_publish(
    backend: Any,
    snapshot_records: List[Dict[str, Any]],
) -> Optional[str]:
    candidates: List[Dict[str, Any]] = []
    for record in snapshot_records:
        sid = str((record or {}).get("snapshot_id") or "").strip()
        if not sid:
            continue
        normalized = dict(record)
        normalized["snapshot_id"] = sid
        if "created_at" not in normalized:
            normalized["created_at"] = _now_iso()
        candidates.append(normalized)

    if not candidates:
        return None

    st.session_state["agent_eval_snapshot_options_loaded"] = True
    try:
        st.session_state["agent_snapshots_cache"] = backend.list_snapshots(limit=200)
        return None
    except Exception as exc:
        cache_payload = st.session_state.get("agent_snapshots_cache")
        for record in candidates:
            cache_payload = _upsert_snapshot_record_in_cache(cache_payload, record)
        st.session_state["agent_snapshots_cache"] = cache_payload
        return str(exc)


def _fallback_artifact_filename(result_payload: Dict[str, Any], fallback_suffix: str) -> str:
    run_id_raw = str(((result_payload.get("run") or {}).get("run_id") or "")).strip()
    run_id = run_id_raw.replace("/", "_").replace("\\", "_") or "evaluation"
    suffix = str(fallback_suffix or "").strip().lstrip("_") or "artifact.bin"
    return f"{run_id}_{suffix}"


def _resolve_download_artifact(
    result_payload: Dict[str, Any],
    *,
    payload_key: str,
    fallback_suffix: str,
) -> Dict[str, Any]:
    fallback_name = _fallback_artifact_filename(result_payload, fallback_suffix)
    raw_path = str(result_payload.get(payload_key) or "").strip()
    filename = Path(raw_path).name if raw_path else ""
    filename = filename or fallback_name

    if not raw_path:
        return {
            "ok": False,
            "reason": "missing_path",
            "error": "No artifact path was provided.",
            "path": "",
            "filename": filename,
            "data": b"",
        }

    path = Path(raw_path)
    if not path.exists():
        return {
            "ok": False,
            "reason": "missing_file",
            "error": f"File not found: {path}",
            "path": str(path),
            "filename": filename,
            "data": b"",
        }
    if not path.is_file():
        return {
            "ok": False,
            "reason": "not_file",
            "error": f"Path is not a file: {path}",
            "path": str(path),
            "filename": filename,
            "data": b"",
        }
    try:
        data = path.read_bytes()
    except Exception as exc:
        return {
            "ok": False,
            "reason": "unreadable",
            "error": f"Failed to read file: {exc}",
            "path": str(path),
            "filename": filename,
            "data": b"",
        }
    return {
        "ok": True,
        "reason": "ok",
        "error": "",
        "path": str(path),
        "filename": filename,
        "data": data,
    }


def _render_evaluation_artifact_downloads(result_payload: Dict[str, Any]) -> None:
    mode = str(result_payload.get("mode") or "").strip().lower()
    if mode == "prospective":
        artifact_specs = [
            ("Download Summary JSON", "summary_json", "application/json", "summary.json"),
            ("Download Hypotheses JSON", "hypotheses_json", "application/json", "hypotheses.json"),
            ("Download Hypotheses CSV", "hypotheses_csv", "text/csv", "hypotheses.csv"),
        ]
    elif mode == "retrospective":
        artifact_specs = [
            ("Download Review Packet CSV", "review_packet_csv", "text/csv", "review_packet.csv"),
            ("Download Review Packet JSON", "review_packet_json", "application/json", "review_packet.json"),
            ("Download Assessment Bundle", "assessment_bundle_json", "application/json", "assessment_bundle_v1.json"),
        ]
    else:
        artifact_specs = []

    if not artifact_specs:
        return

    st.markdown("**Download Artifacts**")
    cols = st.columns(len(artifact_specs))
    key_run_id = str(((result_payload.get("run") or {}).get("run_id") or "evaluation")).strip() or "evaluation"
    key_run_id = key_run_id.replace(" ", "_")
    for idx, (label, payload_key, mime, fallback_suffix) in enumerate(artifact_specs):
        artifact = _resolve_download_artifact(
            result_payload,
            payload_key=payload_key,
            fallback_suffix=fallback_suffix,
        )
        with cols[idx]:
            if artifact.get("ok"):
                st.download_button(
                    label=label,
                    data=artifact.get("data") or b"",
                    file_name=str(artifact.get("filename") or _fallback_artifact_filename(result_payload, fallback_suffix)),
                    mime=mime,
                    key=f"agent_eval_download_{mode}_{payload_key}_{key_run_id}",
                    use_container_width=True,
                )
            else:
                reason = str(artifact.get("reason") or "")
                if reason == "missing_path":
                    st.caption(f"{label}: unavailable for this run.")
                else:
                    st.caption(f"{label}: {artifact.get('error')}")


def _build_retrospective_command_preview(config: Dict[str, Any]) -> str:
    parts = [
        "python",
        "-m",
        "novelty_app.evaluation.run_retrospective",
        "--backend-url",
        config["backend_url"],
        "--qwen-base-url",
        config["qwen_base_url"],
        "--data-json",
        config["data_json"],
        "--data-dir",
        config["data_dir"],
    ]
    if config.get("existing_snapshot_id"):
        parts.extend(["--existing-snapshot-id", str(config["existing_snapshot_id"])])
    else:
        parts.extend(
            [
                "--cutoff-date",
                config["cutoff_date"],
                "--future-window-start",
                config["future_window_start"],
                "--future-window-end",
                config["future_window_end"],
            ]
        )
    parts.extend(
        [
            "--n-gap-targets",
            str(config["n_gap_targets"]),
            "--n-cluster-pair-targets",
            str(config["n_cluster_pair_targets"]),
            "--n-gold-future-papers",
            str(config["n_gold_future_papers"]),
            "--methods",
            *list(config["methods"]),
            "--seeds",
            str(config["seeds"]),
            "--hypotheses-per-target",
            str(config["hypotheses_per_target"]),
            "--output-dir",
            config["output_dir"],
            "--openai-model",
            config["model_name"],
        ]
    )
    if config.get("disable_leakage_check"):
        parts.append("--disable-leakage-check")
    if config.get("discovery_cue_text"):
        parts.extend(["--discovery-cue-text", config["discovery_cue_text"]])
    if config.get("discovery_cue_goal"):
        parts.extend(["--discovery-cue-goal", config["discovery_cue_goal"]])
    if config.get("cue_source_snapshot_id"):
        parts.extend(
            [
                "--cue-source-snapshot-id",
                str(config["cue_source_snapshot_id"]),
                "--cue-similarity-top-k",
                str(config.get("cue_similarity_top_k", 50)),
                "--cue-similarity-sample-n",
                str(config.get("cue_similarity_sample_n", 6)),
            ]
        )
    title_terms = list(config.get("future_title_exclude") or [])
    if title_terms:
        parts.extend(["--future-title-exclude", *title_terms])
    abstract_terms = list(config.get("future_abstract_exclude") or [])
    if abstract_terms:
        parts.extend(["--future-abstract-exclude", *abstract_terms])
    if config.get("future_semantic_query"):
        parts.extend(["--future-semantic-query", config["future_semantic_query"]])
        if config.get("future_semantic_threshold") is not None:
            parts.extend(["--future-semantic-threshold", str(config["future_semantic_threshold"])])
    return _command_preview(parts)


def _build_prospective_command_preview(config: Dict[str, Any]) -> str:
    parts = [
        "python",
        "-m",
        "novelty_app.evaluation.run_prospective",
        "--backend-url",
        config["backend_url"],
        "--snapshot-id",
        config["snapshot_id"],
        "--n-gap-targets",
        str(config["n_gap_targets"]),
        "--n-cluster-pair-targets",
        str(config["n_cluster_pair_targets"]),
        "--methods",
        *list(config["methods"]),
        "--seeds",
        str(config["seeds"]),
        "--hypotheses-per-target",
        str(config["hypotheses_per_target"]),
        "--exemplars",
        str(config["exemplars"]),
        "--boundary",
        str(config["boundary"]),
        "--diverse",
        str(config["diverse"]),
        "--max-iters",
        str(config["max_iters"]),
        "--output-dir",
        config["output_dir"],
        "--openai-model",
        config["model_name"],
    ]
    for gap_id in config.get("gap_ids") or []:
        parts.extend(["--gap-id", gap_id])
    for cluster_a, cluster_b in config.get("cluster_pairs") or []:
        parts.extend(["--cluster-pair", str(cluster_a), str(cluster_b)])
    if config.get("discovery_cue_text"):
        parts.extend(["--discovery-cue-text", config["discovery_cue_text"]])
    if config.get("discovery_cue_goal"):
        parts.extend(["--discovery-cue-goal", config["discovery_cue_goal"]])
    if config.get("cue_source_snapshot_id"):
        parts.extend(
            [
                "--cue-source-snapshot-id",
                str(config["cue_source_snapshot_id"]),
                "--cue-similarity-top-k",
                str(config.get("cue_similarity_top_k", 50)),
                "--cue-similarity-sample-n",
                str(config.get("cue_similarity_sample_n", 6)),
            ]
        )
    return _command_preview(parts)


def page_agent_console() -> None:
    st.title("Agent Console")
    st.caption(
        "Publish current analysis state as a snapshot, query the agent backend endpoints, and run the LangGraph orchestrator."
    )
    st.code("uvicorn agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088", language="bash")
    if _BACKEND_IMPORT_ERROR is not None:
        st.error(
            "Agent backend client could not be imported. Install the missing dependency (likely `requests`) "
            "and reload the app."
        )
        st.caption(str(_BACKEND_IMPORT_ERROR))
        return

    if "agent_backend_url" not in st.session_state:
        st.session_state.agent_backend_url = DEFAULT_BACKEND_URL
    if "agent_snapshot_id" not in st.session_state:
        st.session_state.agent_snapshot_id = ""

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.session_state.agent_backend_url = st.text_input(
            "Backend URL",
            value=st.session_state.agent_backend_url,
            key="agent_backend_url_input",
        )
    with col2:
        if st.button("Health Check", use_container_width=True):
            try:
                st.session_state.agent_last_health = _get_backend().health()
            except Exception as exc:
                st.session_state.agent_last_health = {"error": str(exc)}
    with col3:
        if st.button("List Snapshots", use_container_width=True):
            try:
                st.session_state.agent_snapshots_cache = _get_backend().list_snapshots(limit=50)
            except Exception as exc:
                st.session_state.agent_snapshots_cache = {"error": str(exc)}

    if st.session_state.get("agent_snapshot_id"):
        st.info(f"Active snapshot: {st.session_state.agent_snapshot_id}")

    if st.session_state.get("agent_last_health"):
        st.json(st.session_state.agent_last_health)

    tabs = st.tabs(["Snapshot Publish", "Skills Playground", "Orchestrator", "Artifacts"])

    with tabs[0]:
        _tab_snapshot_publish()
    with tabs[1]:
        _tab_skills_playground()
    with tabs[2]:
        _tab_orchestrator()
    with tabs[3]:
        _tab_artifacts()


def _tab_snapshot_publish() -> None:
    st.subheader("Publish Snapshot")
    if st.session_state.get("df_valid") is None:
        st.warning("Load data and run clustering/gap analysis first.")
        return

    include_raw_rows = st.checkbox("Include raw row payload for each paper", value=True, key="agent_include_raw_rows")
    include_embeddings = st.checkbox(
        "Include primary embeddings for centroid/boundary sampling",
        value=True,
        key="agent_include_embeddings",
        help="Uses st.session_state.X_primary. Increases snapshot size, but enables richer evidence-pack selection.",
    )

    try:
        metadata_overrides = _snapshot_metadata_overrides()
        if _should_publish_cutoff_filtered_snapshot(metadata_overrides):
            split_plan = _retrospective_split_plan()
            historical_mask_list = [bool(v) for v in split_plan["historical_mask"].tolist()]
            if not any(historical_mask_list):
                raise ValueError("No papers remain on or before the cutoff date.")
            filtered_metadata = dict(metadata_overrides or {})
            filtered_metadata["split_role"] = "historical"
            extra = dict(filtered_metadata.get("extra") or {})
            extra["cutoff_filter_applied"] = True
            extra["excluded_post_cutoff_papers"] = split_plan["counts"]["excluded"]
            extra["excluded_undated_papers"] = split_plan["counts"]["undated"]
            filtered_metadata["extra"] = extra
            payload, summary = _build_snapshot_payload_for_df(
                df=st.session_state.df_valid.loc[split_plan["historical_mask"]].copy(),
                include_raw_rows=include_raw_rows,
                include_embeddings=include_embeddings,
                snapshot_id=st.session_state.get("agent_snapshot_id") or f"snapshot_{uuid.uuid4().hex[:10]}",
                metadata_overrides=filtered_metadata,
                gap_regions=_subset_gap_regions(
                    st.session_state.df_valid,
                    st.session_state.get("gap_regions") or [],
                    historical_mask_list,
                ),
                llm_results=None,
                x_primary=_subset_array(st.session_state.get("X_primary"), historical_mask_list),
                x_umap_2d=_subset_array(st.session_state.get("X_umap_2d"), historical_mask_list),
            )
            st.info(
                "Publishing a cutoff-filtered historical snapshot. "
                f"Included {summary['n_papers']} papers on or before {split_plan['metadata']['cutoff_date']}."
            )
        else:
            payload, summary = _build_snapshot_payload(
                include_raw_rows=include_raw_rows,
                include_embeddings=include_embeddings,
                metadata_overrides=metadata_overrides,
            )
            split_role = str((metadata_overrides or {}).get("split_role") or "").strip().lower()
            if split_role in {"full", "future"}:
                st.info(
                    "Publishing a retrospective metadata snapshot without cutoff filtering "
                    f"(split_role={split_role})."
                )
    except Exception as exc:
        st.error(f"Failed to build snapshot payload: {exc}")
        return

    payload["snapshot_id"] = st.text_input("Snapshot ID", value=payload["snapshot_id"], key="agent_snapshot_publish_id")
    st.session_state.agent_snapshot_id = payload["snapshot_id"]

    cols = st.columns(8)
    cols[0].metric("Papers", summary["n_papers"])
    cols[1].metric("Clusters", summary["n_clusters"])
    cols[2].metric("Gaps", summary["n_gaps"])
    cols[3].metric("Gap Papers", summary["n_gap_papers"])
    cols[4].metric("LLM Analyses", summary["n_llm_analyses"])
    cols[5].metric("Embeddings", summary["n_embeddings"])
    cols[6].metric("Emb Dim", summary["embedding_dim"] or "None")
    cols[7].metric("Cluster Col", summary["cluster_column"] or "None")

    with st.expander("Snapshot Metadata", expanded=False):
        st.json(payload["metadata"])

    if st.button("Publish to Backend", type="primary"):
        try:
            backend = _get_backend()
            resp = backend.publish_snapshot(payload)
            st.session_state.agent_publish_result = resp
            published_snapshot_id = str(resp.get("snapshot_id") or payload.get("snapshot_id") or "").strip()
            if published_snapshot_id:
                _set_active_published_snapshot(published_snapshot_id)
            sync_error = _sync_snapshot_cache_after_publish(
                backend,
                [
                    {
                        "snapshot_id": published_snapshot_id or str(payload.get("snapshot_id") or "").strip(),
                        "metadata": dict(payload.get("metadata") or {}),
                    }
                ],
            )
            st.success("Snapshot published")
            if sync_error:
                st.caption(
                    "Snapshot options refresh failed after publish; the new snapshot was added locally to picker options."
                )
        except Exception as exc:
            st.session_state.agent_publish_result = {
                "error": str(exc),
                "repr": repr(exc),
                "traceback": traceback.format_exc(),
            }
            st.error(f"Publish failed: {exc}")
            with st.expander("Publish Debug Details", expanded=True):
                st.json(
                    {
                        "error": str(exc),
                        "repr": repr(exc),
                    }
                )
                st.code(traceback.format_exc())

    if st.session_state.get("agent_publish_result"):
        st.json(st.session_state.agent_publish_result)

    if st.checkbox("Preview first 3 paper records", value=False, key="agent_preview_records"):
        st.json(payload["papers"][:3])

    st.divider()
    st.subheader("Retrospective Split Export")
    st.caption(
        "Rerun historical-only analysis on the preserved frontend corpus, publish a historical snapshot, "
        "optionally publish the future window, and store a manifest artifact for later evaluation."
    )
    publish_future_snapshot = st.checkbox(
        "Also publish future snapshot for inspection",
        value=True,
        key="agent_publish_future_snapshot",
        help="The historical snapshot plus manifest artifact are sufficient for evaluation. The future snapshot is optional.",
    )

    split_prefix_raw = st.text_input(
        "Split Snapshot Prefix (optional)",
        key="agent_split_bundle_prefix",
        help="If blank, the current snapshot id or cutoff date is used to derive snapshot ids.",
        placeholder="retro_eval_20201231",
    )

    split_plan: Optional[Dict[str, Any]] = None
    source_df: Optional[pd.DataFrame] = None
    frontend_manifest: Optional[Dict[str, Any]] = None
    analysis_config_payload: Optional[Dict[str, Any]] = None
    split_error: Optional[str] = None
    try:
        source_df = _frontend_authoritative_df()
        split_plan = _retrospective_split_plan_for_df(source_df)
        frontend_manifest = _frontend_corpus_manifest_from_state()
        expected_paper_ids = stable_paper_ids(source_df)
        expected_hash = hash_paper_ids(expected_paper_ids)
        if frontend_manifest.get("retained_paper_id_hash") != expected_hash or int(frontend_manifest.get("row_count") or 0) != len(source_df):
            raise ValueError(
                "The preserved frontend corpus manifest no longer matches the authoritative frontend corpus in memory."
            )
        analysis_config_payload = _normalized_frontend_analysis_config()
    except Exception as exc:
        split_error = str(exc)
        st.info("Set Snapshot intent to `Retrospective` and provide valid dates to prepare a split export.")
        st.caption(split_error)

    if split_plan is not None and source_df is not None and frontend_manifest is not None and analysis_config_payload is not None:
        counts = split_plan["counts"]
        metadata = split_plan["metadata"]
        default_prefix = st.session_state.get("agent_snapshot_publish_id") or st.session_state.get("agent_snapshot_id") or (
            f"retro_{metadata['cutoff_date'].replace('-', '')}"
        )
        bundle_prefix = _sanitize_snapshot_id_fragment(split_prefix_raw or default_prefix, "retrospective_bundle")
        historical_snapshot_id = f"{bundle_prefix}_historical"
        future_snapshot_id = f"{bundle_prefix}_future" if publish_future_snapshot else None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical Papers", counts["historical"])
        c2.metric("Future Papers", counts["future"])
        c3.metric("Excluded Dated Papers", counts["excluded"])
        c4.metric("Undated Papers", counts["undated"])
        if st.session_state.get("load_cutoff_applied"):
            st.info(
                "The active working dataframe is cutoff-filtered, but retrospective export will use the preserved "
                "full frontend corpus captured at load time."
            )
        st.code(
            json.dumps(
                {
                    "historical_snapshot_id": historical_snapshot_id,
                    "future_snapshot_id": future_snapshot_id,
                    "authoritative_frontend_corpus_rows": len(source_df),
                    "analysis_config_hash": _hash_payload(analysis_config_payload),
                    **metadata,
                },
                indent=2,
            ),
            language="json",
        )
        st.success(
            "Historical export will rerun analysis on pre-cutoff papers only, using the preserved frontend corpus "
            "definition and the normalized frontend analysis configuration."
        )

        if st.button("Publish Retrospective Evaluation Bundle", key="agent_publish_split_snapshots"):
            try:
                backend = _get_backend()
                historical_mask_list = [bool(v) for v in split_plan["historical_mask"].tolist()]
                future_mask_list = [bool(v) for v in split_plan["future_mask"].tolist()]
                if not any(historical_mask_list):
                    raise ValueError("Historical split is empty for the selected cutoff date.")
                if not any(future_mask_list):
                    raise ValueError("Future split is empty for the selected future window.")

                embeddings_full = _frontend_authoritative_embeddings()
                embedding_source = str(analysis_config_payload["embedding_name"])
                if embedding_source not in embeddings_full:
                    raise ValueError(f"Primary embedding `{embedding_source}` is not available in the preserved corpus.")

                analysis = run_analysis_v1(
                    source_df.loc[split_plan["historical_mask"]].reset_index(drop=True).copy(),
                    embeddings_full[embedding_source][split_plan["historical_mask"].to_numpy()].copy(),
                    config=AnalysisConfig(**analysis_config_payload),
                )
                future_df = source_df.loc[split_plan["future_mask"]].reset_index(drop=True).copy()
                future_embeddings = embeddings_full[embedding_source][split_plan["future_mask"].to_numpy()].copy()
                analysis_config_hash = _hash_payload(analysis.analysis_config)
                historical_paper_ids = stable_paper_ids(analysis.df)
                future_paper_ids = stable_paper_ids(future_df)

                shared_metadata = {
                    "cutoff_date": metadata["cutoff_date"],
                    "future_window_start": metadata["future_window_start"],
                    "future_window_end": metadata["future_window_end"],
                    "analysis_config": analysis.analysis_config,
                    "analysis_config_hash": analysis_config_hash,
                    "embedding_source": embedding_source,
                }
                historical_metadata = {
                    **shared_metadata,
                    "split_role": "historical",
                    "extra": {
                        "bundle_prefix": bundle_prefix,
                        "paired_snapshot_id": future_snapshot_id,
                        "export_mode": "streamlit_retrospective_rerun",
                        "retrospective_bundle_kind": "retrospective_snapshot_bundle",
                        "source_corpus_row_count": frontend_manifest["row_count"],
                        "source_corpus_paper_id_hash": frontend_manifest["retained_paper_id_hash"],
                        "historical_paper_count": len(historical_paper_ids),
                        "historical_paper_id_hash": hash_paper_ids(historical_paper_ids),
                        "future_paper_count": len(future_paper_ids),
                        "future_paper_id_hash": hash_paper_ids(future_paper_ids),
                        "excluded_dated_papers": counts["excluded"],
                        "undated_papers": counts["undated"],
                    },
                }

                historical_payload, historical_summary = _build_snapshot_payload_for_df(
                    df=analysis.df,
                    include_raw_rows=include_raw_rows,
                    include_embeddings=include_embeddings,
                    snapshot_id=historical_snapshot_id,
                    metadata_overrides=historical_metadata,
                    gap_regions=analysis.gap_regions,
                    llm_results=None,
                    x_primary=analysis.x_primary,
                    x_umap_2d=analysis.x_umap_2d,
                    selected_clustering=analysis.selected_clustering,
                )

                historical_resp = backend.publish_snapshot(historical_payload)
                future_resp: Optional[Dict[str, Any]] = None
                future_summary: Dict[str, Any] = {
                    "n_papers": len(future_df),
                    "paper_id_hash": hash_paper_ids(future_paper_ids),
                }
                if publish_future_snapshot and future_snapshot_id is not None:
                    future_metadata = {
                        **shared_metadata,
                        "split_role": "future",
                        "extra": {
                            "bundle_prefix": bundle_prefix,
                            "paired_snapshot_id": historical_snapshot_id,
                            "export_mode": "streamlit_retrospective_rerun",
                            "retrospective_bundle_kind": "retrospective_snapshot_bundle",
                            "source_corpus_row_count": frontend_manifest["row_count"],
                            "source_corpus_paper_id_hash": frontend_manifest["retained_paper_id_hash"],
                            "future_paper_count": len(future_paper_ids),
                            "future_paper_id_hash": hash_paper_ids(future_paper_ids),
                        },
                    }
                    future_payload, future_summary = _build_snapshot_payload_for_df(
                        df=future_df,
                        include_raw_rows=include_raw_rows,
                        include_embeddings=include_embeddings,
                        snapshot_id=future_snapshot_id,
                        metadata_overrides=future_metadata,
                        gap_regions=[],
                        llm_results=None,
                        x_primary=future_embeddings,
                        x_umap_2d=None,
                        selected_clustering=None,
                    )
                    future_resp = backend.publish_snapshot(future_payload)

                manifest_resp = backend.store_artifact(
                    kind="retrospective_snapshot_bundle",
                    snapshot_id=historical_snapshot_id,
                    target={
                        "target_type": "retrospective_snapshot_bundle",
                        "bundle_prefix": bundle_prefix,
                        "historical_snapshot_id": historical_snapshot_id,
                        "future_snapshot_id": future_snapshot_id,
                    },
                    payload={
                        "schema_version": "retrospective_snapshot_bundle_v1",
                        "source": "streamlit_agent_console",
                        **shared_metadata,
                        "historical_snapshot_id": historical_snapshot_id,
                        "future_snapshot_id": future_snapshot_id,
                        "publish_future_snapshot": publish_future_snapshot,
                        "corpus_manifest": frontend_manifest,
                        "analysis_config": analysis.analysis_config,
                        "analysis_config_hash": analysis_config_hash,
                        "historical_summary": historical_summary,
                        "future_summary": future_summary,
                        "historical_paper_count": len(historical_paper_ids),
                        "historical_paper_id_hash": hash_paper_ids(historical_paper_ids),
                        "future_paper_count": len(future_paper_ids),
                        "future_paper_id_hash": hash_paper_ids(future_paper_ids),
                        "excluded_dated_papers": counts["excluded"],
                        "undated_papers": counts["undated"],
                    },
                )
                backend.update_snapshot_metadata(
                    historical_snapshot_id,
                    {
                        "extra": {
                            **dict(historical_payload.get("metadata", {}).get("extra") or {}),
                            "retrospective_bundle_artifact_id": manifest_resp.get("artifact_id"),
                            "retrospective_bundle_kind": "retrospective_snapshot_bundle",
                        }
                    },
                )
                if publish_future_snapshot and future_snapshot_id is not None:
                    backend.update_snapshot_metadata(
                        future_snapshot_id,
                        {
                            "extra": {
                                **dict(future_payload.get("metadata", {}).get("extra") or {}),
                                "retrospective_bundle_artifact_id": manifest_resp.get("artifact_id"),
                                "retrospective_bundle_kind": "retrospective_snapshot_bundle",
                            }
                        },
                    )
                _set_active_published_snapshot(historical_snapshot_id)
                sync_records = [
                    {
                        "snapshot_id": historical_snapshot_id,
                        "metadata": dict(historical_payload.get("metadata") or {}),
                    }
                ]
                if publish_future_snapshot and future_snapshot_id is not None:
                    sync_records.append(
                        {
                            "snapshot_id": future_snapshot_id,
                            "metadata": dict(future_payload.get("metadata") or {}),
                        }
                    )
                sync_error = _sync_snapshot_cache_after_publish(backend, sync_records)
                st.session_state.agent_split_publish_result = {
                    "ok": True,
                    "bundle_prefix": bundle_prefix,
                    "analysis_config": analysis.analysis_config,
                    "historical": historical_resp,
                    "future": future_resp,
                    "manifest_artifact": manifest_resp,
                }
                st.success("Retrospective evaluation bundle published")
                if sync_error:
                    st.caption(
                        "Snapshot options refresh failed after split publish; published snapshots were added locally "
                        "to picker options."
                    )
            except Exception as exc:
                st.session_state.agent_split_publish_result = {
                    "error": str(exc),
                    "repr": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                st.error(f"Split publish failed: {exc}")
                with st.expander("Split Publish Debug Details", expanded=True):
                    st.json(
                        {
                            "error": str(exc),
                            "repr": repr(exc),
                        }
                    )
                    st.code(traceback.format_exc())

    if st.session_state.get("agent_split_publish_result"):
        st.json(st.session_state.agent_split_publish_result)

    _tab_evaluation_runner()


def _tab_evaluation_runner() -> None:
    st.divider()
    st.subheader("Evaluation Runner")
    st.caption("Run retrospective or prospective evaluation directly from the agent console.")

    if run_prospective is None or run_retrospective is None:
        st.warning("Evaluation runners could not be imported in this environment.")
        if _EVALUATION_IMPORT_ERROR is not None:
            st.caption(str(_EVALUATION_IMPORT_ERROR))
        return

    backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.get("agent_backend_url") or DEFAULT_BACKEND_URL,
        key="agent_eval_backend_url",
    )
    backend = BackendClient(backend_url)
    model_name = st.text_input(
        "OpenAI model",
        value=os.environ.get("OPENAI_MODEL", "gpt-5-mini-2025-08-07"),
        key="agent_eval_model_name",
    )
    methods = st.multiselect(
        "Methods",
        options=DEFAULT_EVALUATION_METHODS,
        default=["orchestrator"],
        key="agent_eval_methods",
        help="Choose one or more generation methods to score against the selected evaluation protocol.",
    )
    openai_api_key = st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")

    if "agent_eval_snapshot_id" not in st.session_state:
        st.session_state.agent_eval_snapshot_id = st.session_state.get("agent_snapshot_id") or ""
    if "agent_eval_cue_source_snapshot_id" not in st.session_state:
        st.session_state.agent_eval_cue_source_snapshot_id = ""
    if "agent_eval_snapshot_options_loaded" not in st.session_state:
        st.session_state.agent_eval_snapshot_options_loaded = False

    refresh_cols = st.columns([1, 3])
    with refresh_cols[0]:
        if st.button("Refresh Snapshot Options", key="agent_eval_refresh_snapshot_options", use_container_width=True):
            try:
                st.session_state.agent_snapshots_cache = backend.list_snapshots(limit=200)
                st.session_state.agent_eval_snapshot_options_loaded = True
            except Exception as exc:
                st.session_state.agent_snapshots_cache = {"error": str(exc)}
                st.session_state.agent_eval_snapshot_options_loaded = True
    if (
        not st.session_state.get("agent_eval_snapshot_options_loaded")
        and st.session_state.get("agent_snapshots_cache") is None
    ):
        try:
            st.session_state.agent_snapshots_cache = backend.list_snapshots(limit=200)
        except Exception as exc:
            st.session_state.agent_snapshots_cache = {"error": str(exc)}
        st.session_state.agent_eval_snapshot_options_loaded = True

    snapshot_payload = st.session_state.get("agent_snapshots_cache")
    snapshot_lookup_error = str((snapshot_payload or {}).get("error") or "").strip() if isinstance(snapshot_payload, dict) else ""
    snapshot_options = _snapshot_catalog_records(snapshot_payload)
    snapshot_by_id = {
        str(snapshot.get("snapshot_id") or "").strip(): snapshot
        for snapshot in snapshot_options
        if str(snapshot.get("snapshot_id") or "").strip()
    }
    snapshot_label_by_id = {sid: _snapshot_option_label(snapshot) for sid, snapshot in snapshot_by_id.items()}

    snapshot_picker_ids = [""] + list(snapshot_by_id.keys())
    current_snapshot_id = str(st.session_state.get("agent_eval_snapshot_id") or "").strip()
    snapshot_picker_index = snapshot_picker_ids.index(current_snapshot_id) if current_snapshot_id in snapshot_picker_ids else 0
    selected_snapshot_id = st.selectbox(
        "Snapshot Picker",
        options=snapshot_picker_ids,
        index=snapshot_picker_index,
        key="agent_eval_snapshot_picker",
        format_func=lambda sid: (
            "Manual entry"
            if not sid
            else snapshot_label_by_id.get(str(sid), str(sid))
        ),
        help="Pick a published snapshot and optionally refine it with the manual override field below.",
    )
    if selected_snapshot_id:
        st.session_state.agent_eval_snapshot_id = str(selected_snapshot_id)

    snapshot_id = st.text_input(
        "Snapshot ID (manual override)",
        key="agent_eval_snapshot_id",
        help="Published snapshot to evaluate. Retrospective runs may also reuse this historical snapshot.",
    ).strip()

    snapshot: Optional[Dict[str, Any]] = None
    snapshot_error: Optional[str] = None
    if snapshot_id:
        snapshot, snapshot_error = _resolve_snapshot_by_id(snapshot_id, snapshot_by_id, backend)

    snapshot_context = _snapshot_retrospective_context(snapshot)
    mode = _resolved_evaluation_mode(snapshot)
    if mode == "retrospective":
        if snapshot_context.get("has_dates"):
            st.info("Inferred mode: retrospective. The selected snapshot has cutoff and future-window metadata.")
        elif _retrospective_publish_intent_enabled():
            st.info("Inferred mode: retrospective. Snapshot intent in the publish panel is set to Retrospective.")
        else:
            st.info("Inferred mode: retrospective.")
    else:
        st.info("Inferred mode: prospective. No retrospective cutoff metadata is active.")
    if snapshot_lookup_error:
        st.caption(f"Snapshot option refresh error: {snapshot_lookup_error}")
    if snapshot_error:
        st.caption(f"Snapshot lookup failed: {snapshot_error}")

    common_cols = st.columns(4)
    seeds = int(common_cols[0].number_input("Seeds", min_value=1, value=1, step=1, key="agent_eval_seeds"))
    hypotheses_per_target = int(
        common_cols[1].number_input(
            "Hypotheses / target",
            min_value=1,
            value=1,
            step=1,
            key="agent_eval_hypotheses_per_target",
        )
    )
    n_gap_targets = int(
        common_cols[2].number_input("Gap targets", min_value=0, value=25, step=1, key="agent_eval_n_gap_targets")
    )
    n_cluster_pair_targets = int(
        common_cols[3].number_input(
            "Cluster-pair targets",
            min_value=0,
            value=10,
            step=1,
            key="agent_eval_n_cluster_pair_targets",
        )
    )

    discovery_cue_text = st.text_area(
        "Discovery cue text",
        value="",
        height=100,
        key="agent_eval_discovery_cue_text",
        placeholder=(
            "Example: Focus on antimicrobial nanocarriers that combine membrane-disruptive peptides "
            "with pH-responsive polymer coatings to improve biofilm penetration while reducing "
            "mammalian cytotoxicity."
        ),
    ).strip()
    discovery_cue_goal = st.text_input(
        "Discovery cue goal (optional)",
        value="",
        key="agent_eval_discovery_cue_goal",
        placeholder=(
            "Example: Prioritize hypotheses for multidrug-resistant Gram-negative biofilm infections."
        ),
    ).strip()

    cue_is_active = bool(discovery_cue_text or discovery_cue_goal)
    if cue_is_active and not str(st.session_state.get("agent_eval_cue_source_snapshot_id") or "").strip():
        suggested_cue_snapshot_id = _suggest_cue_source_snapshot_id(snapshot_options)
        if suggested_cue_snapshot_id:
            st.session_state.agent_eval_cue_source_snapshot_id = suggested_cue_snapshot_id

    prioritized_cue_options = _prioritize_cue_source_snapshots(snapshot_options)
    cue_picker_ids = [
        str(snapshot.get("snapshot_id") or "").strip()
        for snapshot in prioritized_cue_options
        if str(snapshot.get("snapshot_id") or "").strip()
    ]
    cue_picker_values = [""] + cue_picker_ids
    current_cue_id = str(st.session_state.get("agent_eval_cue_source_snapshot_id") or "").strip()
    cue_picker_index = cue_picker_values.index(current_cue_id) if current_cue_id in cue_picker_values else 0

    cue_cols = st.columns(3)
    with cue_cols[0]:
        selected_cue_source_snapshot_id = st.selectbox(
            "Cue Source Snapshot Picker",
            options=cue_picker_values,
            index=cue_picker_index,
            key="agent_eval_cue_source_snapshot_picker",
            format_func=lambda sid: (
                "Manual entry"
                if not sid
                else snapshot_label_by_id.get(str(sid), str(sid))
            ),
            help="Prioritizes full-corpus qwen snapshots for cue-semantic evidence retrieval.",
        )
        if selected_cue_source_snapshot_id:
            st.session_state.agent_eval_cue_source_snapshot_id = str(selected_cue_source_snapshot_id)
        cue_source_snapshot_id = st.text_input(
            "Cue source snapshot ID (manual override)",
            value="",
            key="agent_eval_cue_source_snapshot_id",
            help="Full-corpus snapshot used for cue-semantic retrieval. Required when discovery cue is active.",
        ).strip()
    cue_similarity_top_k = int(
        cue_cols[1].number_input(
            "Cue top-K",
            min_value=1,
            value=50,
            step=1,
            key="agent_eval_cue_similarity_top_k",
        )
    )
    cue_similarity_sample_n = int(
        cue_cols[2].number_input(
            "Cue sample N",
            min_value=0,
            value=6,
            step=1,
            key="agent_eval_cue_similarity_sample_n",
        )
    )

    cue_source_snapshot: Optional[Dict[str, Any]] = None
    cue_source_lookup_error: Optional[str] = None
    if cue_source_snapshot_id:
        if snapshot_id and cue_source_snapshot_id == snapshot_id and snapshot is not None:
            cue_source_snapshot = snapshot
        else:
            cue_source_snapshot, cue_source_lookup_error = _resolve_snapshot_by_id(
                cue_source_snapshot_id,
                snapshot_by_id,
                backend,
            )

    if cue_is_active and not cue_source_snapshot_id:
        st.warning("Cue source snapshot ID is required when discovery cue is active.")

    if not methods:
        st.warning("Select at least one method before running an evaluation.")
    elif any(method in LLM_REQUIRED_EVALUATION_METHODS for method in methods) and not openai_api_key:
        st.warning("OPENAI_API_KEY is required for the selected LLM-backed methods.")

    progress_placeholder = st.empty()
    if st.session_state.get("agent_evaluation_progress"):
        _render_evaluation_progress(progress_placeholder, st.session_state.agent_evaluation_progress)

    with st.expander("Retrospective Evaluation", expanded=False):
        default_cutoff = (
            snapshot_context.get("cutoff_date")
            or str(st.session_state.get("agent_publish_cutoff_date") or "").strip()
            or "2020-12-31"
        )
        default_future_start = (
            snapshot_context.get("future_window_start")
            or str(st.session_state.get("agent_publish_future_window_start") or "").strip()
            or "2022-01-01"
        )
        default_future_end = (
            snapshot_context.get("future_window_end")
            or str(st.session_state.get("agent_publish_future_window_end") or "").strip()
            or "2025-12-31"
        )

        reuse_snapshot = st.checkbox(
            "Reuse an existing published historical snapshot",
            value=bool(snapshot_context.get("can_reuse_snapshot") and snapshot_id),
            key="agent_eval_retro_reuse_snapshot",
            help="If enabled, the runner uses the published historical snapshot and its stored retrospective bundle.",
        )
        existing_snapshot_id = st.text_input(
            "Existing snapshot ID",
            value=snapshot_id,
            key="agent_eval_retro_existing_snapshot_id",
            disabled=not reuse_snapshot,
        ).strip()
        if snapshot_context.get("reuse_reason"):
            st.caption(snapshot_context["reuse_reason"])
        if reuse_snapshot:
            st.caption("When reuse is enabled, the stored snapshot metadata and bundle manifest take precedence.")

        retro_cols = st.columns(3)
        cutoff_date = retro_cols[0].text_input(
            "Cutoff date",
            value=default_cutoff,
            key="agent_eval_retro_cutoff_date",
        ).strip()
        future_window_start = retro_cols[1].text_input(
            "Future window start",
            value=default_future_start,
            key="agent_eval_retro_future_window_start",
        ).strip()
        future_window_end = retro_cols[2].text_input(
            "Future window end",
            value=default_future_end,
            key="agent_eval_retro_future_window_end",
        ).strip()

        retro_more_cols = st.columns(2)
        n_gold_future_papers = int(
            retro_more_cols[0].number_input(
                "Gold future papers",
                min_value=1,
                value=5,
                step=1,
                key="agent_eval_retro_n_gold_future_papers",
            )
        )
        disable_leakage_check = retro_more_cols[1].checkbox(
            "Disable leakage check",
            value=False,
            key="agent_eval_retro_disable_leakage_check",
        )

        io_cols = st.columns(3)
        data_json = io_cols[0].text_input(
            "Data JSON",
            value=_default_data_json_path(),
            key="agent_eval_retro_data_json",
        ).strip()
        data_dir = io_cols[1].text_input(
            "Data dir",
            value=_default_data_dir(),
            key="agent_eval_retro_data_dir",
        ).strip()
        qwen_base_url = io_cols[2].text_input(
            "Qwen base URL",
            value=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000"),
            key="agent_eval_retro_qwen_base_url",
        ).strip()
        output_dir = st.text_input(
            "Output dir",
            value="data/retrospective_eval",
            key="agent_eval_retro_output_dir",
        ).strip()

        st.markdown("**Future Prefilter**")
        future_title_exclude_text = st.text_area(
            "Title exclusions (comma or newline separated)",
            value="",
            height=80,
            key="agent_eval_retro_future_title_exclude",
        )
        future_abstract_exclude_text = st.text_area(
            "Abstract exclusions (comma or newline separated)",
            value="",
            height=80,
            key="agent_eval_retro_future_abstract_exclude",
        )
        future_semantic_query = st.text_area(
            "Future semantic query",
            value="",
            height=80,
            key="agent_eval_retro_future_semantic_query",
        ).strip()
        future_semantic_threshold = float(
            st.number_input(
                "Future semantic threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.01,
                key="agent_eval_retro_future_semantic_threshold",
            )
        )

        future_title_exclude = _parse_multivalue_text(future_title_exclude_text)
        future_abstract_exclude = _parse_multivalue_text(future_abstract_exclude_text)

        retro_hard_failures: List[str] = []
        retro_warnings: List[str] = []
        if not methods:
            retro_hard_failures.append("Select at least one method.")
        if not data_json:
            retro_hard_failures.append("Data JSON is required.")
        if not data_dir:
            retro_hard_failures.append("Data dir is required.")
        if not output_dir:
            retro_hard_failures.append("Output dir is required.")
        if reuse_snapshot and not existing_snapshot_id:
            retro_hard_failures.append("Existing snapshot ID is required when snapshot reuse is enabled.")
        if not reuse_snapshot and not _has_valid_retrospective_dates(
            cutoff_date,
            future_window_start,
            future_window_end,
        ):
            retro_hard_failures.append("Cutoff date and future window dates must be valid before running retrospective evaluation.")

        existing_snapshot_error: Optional[str] = None
        existing_snapshot: Optional[Dict[str, Any]] = None
        existing_snapshot_context = {}
        if reuse_snapshot and existing_snapshot_id:
            if snapshot_id and existing_snapshot_id == snapshot_id and snapshot is not None:
                existing_snapshot = snapshot
            else:
                existing_snapshot, existing_snapshot_error = _resolve_snapshot_by_id(
                    existing_snapshot_id,
                    snapshot_by_id,
                    backend,
                )
            if existing_snapshot is not None:
                existing_snapshot_context = _snapshot_retrospective_context(existing_snapshot)
            if existing_snapshot_error:
                retro_hard_failures.append(f"Existing snapshot lookup failed: {existing_snapshot_error}")
            elif existing_snapshot is None:
                retro_hard_failures.append(f"Existing snapshot `{existing_snapshot_id}` was not found.")
            elif not existing_snapshot_context.get("can_reuse_snapshot"):
                retro_hard_failures.append(
                    str(existing_snapshot_context.get("reuse_reason") or "Existing snapshot is not reusable for retrospective mode.")
                )

        if cue_is_active:
            cue_error = _cue_source_scope_error(
                cue_source_snapshot_id,
                cue_source_snapshot,
                cue_source_lookup_error,
            )
            if cue_error:
                retro_hard_failures.append(cue_error)

        qwen_issue = _qwen_base_url_issue(qwen_base_url)
        if qwen_issue:
            retro_hard_failures.append(qwen_issue)

        if any(method in LLM_REQUIRED_EVALUATION_METHODS for method in methods) and not openai_api_key:
            retro_warnings.append("OPENAI_API_KEY is required for selected LLM-backed methods.")
        if snapshot_error:
            retro_warnings.append(f"Evaluation snapshot lookup warning: {snapshot_error}")
        if cue_source_lookup_error and cue_source_snapshot_id:
            retro_warnings.append(f"Cue source snapshot lookup warning: {cue_source_lookup_error}")

        _render_preflight_status(retro_hard_failures, retro_warnings)

        retro_command = _build_retrospective_command_preview(
            {
                "backend_url": backend_url,
                "qwen_base_url": qwen_base_url,
                "data_json": data_json,
                "data_dir": data_dir,
                "existing_snapshot_id": existing_snapshot_id if reuse_snapshot and existing_snapshot_id else None,
                "cutoff_date": cutoff_date,
                "future_window_start": future_window_start,
                "future_window_end": future_window_end,
                "n_gap_targets": n_gap_targets,
                "n_cluster_pair_targets": n_cluster_pair_targets,
                "n_gold_future_papers": n_gold_future_papers,
                "methods": methods,
                "seeds": seeds,
                "hypotheses_per_target": hypotheses_per_target,
                "output_dir": output_dir,
                "model_name": model_name,
                "disable_leakage_check": disable_leakage_check,
                "discovery_cue_text": discovery_cue_text or None,
                "discovery_cue_goal": discovery_cue_goal or None,
                "cue_source_snapshot_id": cue_source_snapshot_id or None,
                "cue_similarity_top_k": cue_similarity_top_k,
                "cue_similarity_sample_n": cue_similarity_sample_n,
                "future_title_exclude": future_title_exclude,
                "future_abstract_exclude": future_abstract_exclude,
                "future_semantic_query": future_semantic_query or None,
                "future_semantic_threshold": future_semantic_threshold if future_semantic_query else None,
            }
        )
        st.code(retro_command, language="bash")

        if st.button(
            "Run Retrospective Evaluation",
            type="primary",
            key="agent_eval_run_retrospective",
            disabled=bool(retro_hard_failures),
        ):
            try:
                if retro_hard_failures:
                    raise ValueError(retro_hard_failures[0])

                analysis_config_obj = None
                if AnalysisConfig is not None:
                    try:
                        analysis_config_obj = AnalysisConfig(**_normalized_frontend_analysis_config())
                    except Exception:
                        analysis_config_obj = AnalysisConfig()

                st.session_state.agent_evaluation_result = None
                st.session_state.agent_evaluation_progress = None

                def _progress_callback(progress: Any) -> None:
                    payload = progress.to_payload() if hasattr(progress, "to_payload") else dict(progress)
                    st.session_state.agent_evaluation_progress = payload
                    _render_evaluation_progress(progress_placeholder, payload)

                with st.spinner("Running retrospective evaluation..."):
                    result = run_retrospective(
                        backend=BackendClient(backend_url),
                        data_json=data_json,
                        data_dir=data_dir,
                        qwen_base_url=qwen_base_url,
                        cutoff_date=cutoff_date,
                        future_window_start=future_window_start,
                        future_window_end=future_window_end,
                        analysis_config=analysis_config_obj,
                        n_gap_targets=n_gap_targets,
                        n_cluster_pair_targets=n_cluster_pair_targets,
                        n_gold_future_papers=n_gold_future_papers,
                        methods=methods,
                        seeds=seeds,
                        hypotheses_per_target=hypotheses_per_target,
                        output_dir=output_dir,
                        openai_api_key=openai_api_key,
                        model_name=model_name or None,
                        existing_snapshot_id=existing_snapshot_id if reuse_snapshot and existing_snapshot_id else None,
                        discovery_cue={
                            "text": discovery_cue_text or "",
                            "goal": discovery_cue_goal or None,
                        }
                        if discovery_cue_text or discovery_cue_goal
                        else None,
                        cue_source_snapshot_id=cue_source_snapshot_id or None,
                        cue_similarity_top_k=cue_similarity_top_k,
                        cue_similarity_sample_n=cue_similarity_sample_n,
                        disable_leakage_check=disable_leakage_check,
                        future_title_exclude=future_title_exclude or None,
                        future_abstract_exclude=future_abstract_exclude or None,
                        future_semantic_query=future_semantic_query or None,
                        future_semantic_threshold=future_semantic_threshold if future_semantic_query else None,
                        progress_callback=_progress_callback,
                    )
                st.session_state.agent_evaluation_result = {
                    "mode": "retrospective",
                    "run": result.run,
                    "review_packet_csv": result.review_packet_csv,
                    "review_packet_json": result.review_packet_json,
                    "assessment_bundle_json": result.assessment_bundle_json,
                }
            except Exception as exc:
                st.session_state.agent_evaluation_result = {
                    "mode": "retrospective",
                    "error": str(exc),
                    "repr": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                st.session_state.agent_evaluation_progress = {
                    "phase": "failed",
                    "status": "failed",
                    "message": str(exc),
                }
                _render_evaluation_progress(progress_placeholder, st.session_state.agent_evaluation_progress)

    with st.expander("Prospective Evaluation", expanded=False):
        pack_cols = st.columns(4)
        exemplars = int(pack_cols[0].number_input("Exemplars", min_value=0, value=8, step=1, key="agent_eval_pro_exemplars"))
        boundary = int(pack_cols[1].number_input("Boundary", min_value=0, value=8, step=1, key="agent_eval_pro_boundary"))
        diverse = int(pack_cols[2].number_input("Diverse", min_value=0, value=0, step=1, key="agent_eval_pro_diverse"))
        max_iters = int(pack_cols[3].number_input("Max iters", min_value=0, value=2, step=1, key="agent_eval_pro_max_iters"))
        output_dir = st.text_input(
            "Output dir",
            value="data/prospective_eval",
            key="agent_eval_pro_output_dir",
        ).strip()

        cluster_pair_parse_error: Optional[str] = None
        gap_ids_text = ""
        cluster_pairs_text = ""
        st.markdown("**Explicit Targets**")
        gap_ids_text = st.text_area(
            "Gap IDs (one per line)",
            value="",
            height=80,
            key="agent_eval_pro_gap_ids",
        )
        cluster_pairs_text = st.text_area(
            "Cluster pairs (`cluster_a,cluster_b` per line)",
            value="",
            height=80,
            key="agent_eval_pro_cluster_pairs",
        )

        gap_ids = _parse_multivalue_text(gap_ids_text)
        try:
            cluster_pairs = _parse_cluster_pair_text(cluster_pairs_text)
        except ValueError as exc:
            cluster_pair_parse_error = str(exc)
            cluster_pairs = []
        if cluster_pair_parse_error:
            st.caption(cluster_pair_parse_error)
        if gap_ids or cluster_pairs:
            st.caption("Explicit targets override the gap-target and cluster-pair counts.")

        prospective_hard_failures: List[str] = []
        prospective_warnings: List[str] = []
        if not snapshot_id:
            prospective_hard_failures.append("Snapshot ID is required for prospective evaluation.")
        if not methods:
            prospective_hard_failures.append("Select at least one method.")
        if cluster_pair_parse_error:
            prospective_hard_failures.append(cluster_pair_parse_error)
        if not output_dir:
            prospective_hard_failures.append("Output dir is required.")
        if cue_is_active:
            cue_error = _cue_source_scope_error(
                cue_source_snapshot_id,
                cue_source_snapshot,
                cue_source_lookup_error,
            )
            if cue_error:
                prospective_hard_failures.append(cue_error)
        if any(method in LLM_REQUIRED_EVALUATION_METHODS for method in methods) and not openai_api_key:
            prospective_warnings.append("OPENAI_API_KEY is required for selected LLM-backed methods.")
        if snapshot_error and snapshot_id:
            prospective_warnings.append(f"Evaluation snapshot lookup warning: {snapshot_error}")
        if cue_source_lookup_error and cue_source_snapshot_id:
            prospective_warnings.append(f"Cue source snapshot lookup warning: {cue_source_lookup_error}")

        _render_preflight_status(prospective_hard_failures, prospective_warnings)

        prospective_command = _build_prospective_command_preview(
            {
                "backend_url": backend_url,
                "snapshot_id": snapshot_id,
                "methods": methods,
                "seeds": seeds,
                "hypotheses_per_target": hypotheses_per_target,
                "n_gap_targets": 0 if (gap_ids or cluster_pairs) else n_gap_targets,
                "n_cluster_pair_targets": 0 if (gap_ids or cluster_pairs) else n_cluster_pair_targets,
                "output_dir": output_dir,
                "model_name": model_name,
                "discovery_cue_text": discovery_cue_text or None,
                "discovery_cue_goal": discovery_cue_goal or None,
                "cue_source_snapshot_id": cue_source_snapshot_id or None,
                "cue_similarity_top_k": cue_similarity_top_k,
                "cue_similarity_sample_n": cue_similarity_sample_n,
                "exemplars": exemplars,
                "boundary": boundary,
                "diverse": diverse,
                "max_iters": max_iters,
                "gap_ids": gap_ids,
                "cluster_pairs": cluster_pairs,
            }
        )
        st.code(prospective_command, language="bash")

        if st.button(
            "Run Prospective Evaluation",
            type="primary",
            key="agent_eval_run_prospective",
            disabled=bool(prospective_hard_failures),
        ):
            try:
                if prospective_hard_failures:
                    raise ValueError(prospective_hard_failures[0])

                explicit_targets = [
                    {"target_type": "gap", "gap_id": gap_id}
                    for gap_id in gap_ids
                ] + [
                    {"target_type": "cluster_pair", "cluster_a": cluster_a, "cluster_b": cluster_b}
                    for cluster_a, cluster_b in cluster_pairs
                ]

                st.session_state.agent_evaluation_result = None
                st.session_state.agent_evaluation_progress = None

                def _progress_callback(progress: Any) -> None:
                    payload = progress.to_payload() if hasattr(progress, "to_payload") else dict(progress)
                    st.session_state.agent_evaluation_progress = payload
                    _render_evaluation_progress(progress_placeholder, payload)

                with st.spinner("Running prospective evaluation..."):
                    result = run_prospective(
                        snapshot_id=snapshot_id,
                        backend_url=backend_url,
                        methods=methods,
                        seeds=seeds,
                        hypotheses_per_target=hypotheses_per_target,
                        n_gap_targets=0 if explicit_targets else n_gap_targets,
                        n_cluster_pair_targets=0 if explicit_targets else n_cluster_pair_targets,
                        explicit_targets=explicit_targets or None,
                        output_dir=output_dir,
                        openai_api_key=openai_api_key,
                        model_name=model_name or None,
                        discovery_cue={
                            "text": discovery_cue_text or "",
                            "goal": discovery_cue_goal or None,
                        }
                        if discovery_cue_text or discovery_cue_goal
                        else None,
                        cue_source_snapshot_id=cue_source_snapshot_id or None,
                        cue_similarity_top_k=cue_similarity_top_k,
                        cue_similarity_sample_n=cue_similarity_sample_n,
                        exemplars=exemplars,
                        boundary=boundary,
                        diverse=diverse,
                        max_iters=max_iters,
                        progress_callback=_progress_callback,
                    )
                st.session_state.agent_evaluation_result = {
                    "mode": "prospective",
                    "run": result.run,
                    "summary_json": result.summary_json,
                    "hypotheses_json": result.hypotheses_json,
                    "hypotheses_csv": result.hypotheses_csv,
                }
            except Exception as exc:
                st.session_state.agent_evaluation_result = {
                    "mode": "prospective",
                    "error": str(exc),
                    "repr": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                st.session_state.agent_evaluation_progress = {
                    "phase": "failed",
                    "status": "failed",
                    "message": str(exc),
                }
                _render_evaluation_progress(progress_placeholder, st.session_state.agent_evaluation_progress)

    result_payload = st.session_state.get("agent_evaluation_result")
    if result_payload:
        if result_payload.get("error"):
            st.error(f"Evaluation failed: {result_payload['error']}")
            with st.expander("Evaluation Debug Details", expanded=False):
                st.json(
                    {
                        "error": result_payload.get("error"),
                        "repr": result_payload.get("repr"),
                    }
                )
                st.code(str(result_payload.get("traceback") or ""))
        else:
            st.success(
                f"{str(result_payload.get('mode') or 'evaluation').title()} evaluation completed: "
                f"{((result_payload.get('run') or {}).get('run_id') or '')}"
            )
            _render_evaluation_artifact_downloads(result_payload)
            st.json(result_payload)


def _tab_skills_playground() -> None:
    st.subheader("Skills Playground")
    backend = _get_backend()
    snapshot_id_default = st.session_state.get("agent_snapshot_id") or None

    subtabs = st.tabs(["Top Gaps", "Clusters", "Evidence Pack", "Papers Batch", "Store Artifact"])

    with subtabs[0]:
        k = st.number_input("k", min_value=1, max_value=200, value=25, key="agent_gaps_k")
        snapshot_id = st.text_input("Snapshot ID (optional)", value=snapshot_id_default or "", key="agent_gaps_snapshot")
        if st.button("GET /gaps/top", key="agent_btn_gaps"):
            try:
                resp = backend.top_gaps(k=int(k), snapshot_id=snapshot_id or None)
                st.session_state.agent_last_tool_response = resp
            except Exception as exc:
                st.session_state.agent_last_tool_response = {"error": str(exc)}
        if st.session_state.get("agent_last_tool_response"):
            st.json(st.session_state.agent_last_tool_response)

    with subtabs[1]:
        limit = st.number_input("Limit", min_value=1, max_value=1000, value=100, key="agent_clusters_limit")
        sort = st.selectbox(
            "Sort",
            ["size_desc", "size_asc", "cluster_id_asc", "cluster_id_desc"],
            index=0,
            key="agent_clusters_sort",
        )
        snapshot_id = st.text_input("Snapshot ID (optional)", value=snapshot_id_default or "", key="agent_clusters_snapshot")
        if st.button("GET /clusters", key="agent_btn_clusters"):
            try:
                resp = backend.list_clusters(snapshot_id=snapshot_id or None, limit=int(limit), sort=sort)
                st.session_state.agent_last_tool_response = resp
            except Exception as exc:
                st.session_state.agent_last_tool_response = {"error": str(exc)}
        if st.session_state.get("agent_last_tool_response"):
            st.json(st.session_state.agent_last_tool_response)

    with subtabs[2]:
        target_type = st.selectbox("Target Type", ["gap", "cluster_pair"], key="agent_pack_target_type")
        snapshot_id = st.text_input("Snapshot ID (optional)", value=snapshot_id_default or "", key="agent_pack_snapshot")
        col1, col2, col3 = st.columns(3)
        exemplars = col1.number_input("Exemplars", min_value=0, max_value=200, value=25, key="agent_pack_exemplars")
        boundary = col2.number_input("Boundary", min_value=0, max_value=200, value=25, key="agent_pack_boundary")
        diverse = col3.number_input("Diverse", min_value=0, max_value=200, value=25, key="agent_pack_diverse")

        gap_id = None
        cluster_a = None
        cluster_b = None
        if target_type == "gap":
            gap_id = st.text_input("gap_id", value="", key="agent_pack_gap_id")
        else:
            c1, c2 = st.columns(2)
            cluster_a = c1.number_input("cluster_a", step=1, value=0, key="agent_pack_cluster_a")
            cluster_b = c2.number_input("cluster_b", step=1, value=1, key="agent_pack_cluster_b")

        counter_queries_text = st.text_area(
            "Counter Queries (one per line)", value="", height=100, key="agent_pack_counter_queries"
        )
        counter_queries = [q.strip() for q in counter_queries_text.splitlines() if q.strip()]

        if st.button("POST /evidence/pack", key="agent_btn_pack"):
            payload: Dict[str, Any] = {
                "snapshot_id": snapshot_id or None,
                "target_type": target_type,
                "exemplars": int(exemplars),
                "boundary": int(boundary),
                "diverse": int(diverse),
                "counter_queries": counter_queries,
            }
            if target_type == "gap":
                payload["gap_id"] = gap_id
            else:
                payload["cluster_a"] = int(cluster_a)
                payload["cluster_b"] = int(cluster_b)
            try:
                resp = backend.evidence_pack(payload)
                st.session_state.agent_last_pack = resp
                st.session_state.agent_last_tool_response = resp
            except Exception as exc:
                st.session_state.agent_last_tool_response = {"error": str(exc)}
        if st.session_state.get("agent_last_tool_response"):
            st.json(st.session_state.agent_last_tool_response)

    with subtabs[3]:
        snapshot_id = st.text_input("Snapshot ID", value=snapshot_id_default or "", key="agent_batch_snapshot")
        paper_ids_text = st.text_area(
            "Paper IDs (comma or newline separated)", value="", height=80, key="agent_batch_ids"
        )
        fields_text = st.text_input("Fields (comma-separated, optional)", value="", key="agent_batch_fields")
        if st.button("POST /papers/batch", key="agent_btn_batch"):
            paper_ids = [x.strip() for x in paper_ids_text.replace("\n", ",").split(",") if x.strip()]
            fields = [x.strip() for x in fields_text.split(",") if x.strip()]
            try:
                resp = backend.papers_batch(snapshot_id=snapshot_id, paper_ids=paper_ids, fields=fields or None)
                st.session_state.agent_last_tool_response = resp
            except Exception as exc:
                st.session_state.agent_last_tool_response = {"error": str(exc)}
        if st.session_state.get("agent_last_tool_response"):
            st.json(st.session_state.agent_last_tool_response)

    with subtabs[4]:
        snapshot_id = st.text_input("Snapshot ID (optional)", value=snapshot_id_default or "", key="agent_artifact_snapshot")
        kind = st.text_input("Kind", value="research_brief", key="agent_artifact_kind")
        target_text = st.text_area(
            "Target JSON",
            value=json.dumps({"target_type": "gap", "gap_id": "gap_0"}, indent=2),
            height=120,
            key="agent_artifact_target",
        )
        default_payload = st.session_state.get("agent_last_run") or st.session_state.get("agent_last_pack") or {}
        payload_text = st.text_area(
            "Payload JSON",
            value=json.dumps(default_payload, indent=2) if default_payload else "{}",
            height=180,
            key="agent_artifact_payload",
        )
        if st.button("POST /artifacts/store", key="agent_btn_store_artifact"):
            try:
                target = json.loads(target_text or "{}")
                payload = json.loads(payload_text or "{}")
                resp = backend.store_artifact(
                    kind=kind,
                    target=target,
                    payload=payload,
                    snapshot_id=snapshot_id or None,
                )
                st.session_state.agent_last_tool_response = resp
                st.success("Artifact stored")
            except Exception as exc:
                st.session_state.agent_last_tool_response = {"error": str(exc)}
        if st.session_state.get("agent_last_tool_response"):
            st.json(st.session_state.agent_last_tool_response)


def _tab_orchestrator() -> None:
    st.subheader("Run LangGraph Orchestrator")
    if build_orchestrator is None:
        st.warning("LangGraph/LangChain orchestrator imports are unavailable in this environment.")
        return

    snapshot_id = st.text_input(
        "Snapshot ID",
        value=st.session_state.get("agent_snapshot_id") or "",
        key="agent_orch_snapshot",
    )
    target_type = st.radio("Target Type", ["gap", "cluster_pair"], horizontal=True, key="agent_orch_target")
    if target_type == "gap":
        gap_id = st.text_input("gap_id", value="gap_0", key="agent_orch_gap_id")
        cluster_a = cluster_b = None
    else:
        c1, c2 = st.columns(2)
        cluster_a = int(c1.number_input("cluster_a", step=1, value=0, key="agent_orch_cluster_a"))
        cluster_b = int(c2.number_input("cluster_b", step=1, value=1, key="agent_orch_cluster_b"))
        gap_id = None

    c1, c2, c3, c4 = st.columns(4)
    exemplars = int(c1.number_input("Exemplars", min_value=0, max_value=200, value=25, key="agent_orch_exemplars"))
    boundary = int(c2.number_input("Boundary", min_value=0, max_value=200, value=25, key="agent_orch_boundary"))
    diverse = int(c3.number_input("Diverse", min_value=0, max_value=200, value=25, key="agent_orch_diverse"))
    max_iters = int(c4.number_input("Max Iters", min_value=0, max_value=5, value=2, key="agent_orch_max_iters"))

    if st.button("Run Orchestrator", type="primary", key="agent_orch_run"):
        if not snapshot_id:
            st.error("Snapshot ID is required")
            return
        backend = _get_backend()
        try:
            openai_api_key = st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("OpenAI API key not found. Set it in Data & Config or OPENAI_API_KEY environment variable.")
                return
            model_name = os.environ.get("OPENAI_MODEL", "gpt-5")
            app = build_orchestrator(backend, openai_api_key=openai_api_key, model_name=model_name)
            state: Dict[str, Any] = {
                "snapshot_id": snapshot_id,
                "target_type": target_type,
                "max_iters": max_iters,
                "iter": 0,
                "exemplars": exemplars,
                "boundary": boundary,
                "diverse": diverse,
            }
            if target_type == "gap":
                state["gap_id"] = gap_id
            else:
                state["cluster_a"] = cluster_a
                state["cluster_b"] = cluster_b

            with st.spinner("Running orchestrator..."):
                out = app.invoke(state)
            st.session_state.agent_last_run = out
            st.success("Orchestrator run completed")
        except Exception as exc:
            st.error(f"Orchestrator failed: {exc}")

    if st.session_state.get("agent_last_run"):
        observability = dict((st.session_state.get("agent_last_run") or {}).get("observability") or {})
        if observability:
            cols = st.columns([2, 3])
            cols[0].text_input(
                "Latest Trace ID",
                value=str(observability.get("trace_id") or ""),
                disabled=True,
                key="agent_last_run_trace_id",
            )
            trace_url = str(observability.get("url") or "").strip()
            if trace_url:
                cols[1].markdown(f"[Open Langfuse Trace]({trace_url})")
        st.json(st.session_state.agent_last_run)


def _tab_artifacts() -> None:
    st.subheader("Artifacts")
    backend = _get_backend()
    snapshot_id = st.text_input(
        "Snapshot ID filter (optional)",
        value=st.session_state.get("agent_snapshot_id") or "",
        key="agent_artifacts_filter_snapshot",
    )
    limit = st.number_input("Limit", min_value=1, max_value=500, value=50, key="agent_artifacts_limit")
    if st.button("Refresh Artifacts", key="agent_refresh_artifacts"):
        try:
            st.session_state.agent_artifacts_cache = backend.list_artifacts(
                snapshot_id=snapshot_id or None,
                limit=int(limit),
            )
        except Exception as exc:
            st.session_state.agent_artifacts_cache = {"error": str(exc)}

    cache = st.session_state.get("agent_artifacts_cache")
    if cache:
        if isinstance(cache, dict) and "artifacts" in cache:
            artifacts = cache["artifacts"]
            if artifacts:
                table_rows = [
                    {
                        "artifact_id": a.get("artifact_id"),
                        "snapshot_id": a.get("snapshot_id"),
                        "kind": a.get("kind"),
                        "created_at": a.get("created_at"),
                    }
                    for a in artifacts
                ]
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
                selected_idx = st.number_input(
                    "Artifact index to inspect",
                    min_value=0,
                    max_value=max(len(artifacts) - 1, 0),
                    value=0,
                    step=1,
                    key="agent_artifact_inspect_idx",
                )
                if artifacts:
                    st.json(artifacts[int(selected_idx)])
            else:
                st.info("No artifacts found")
        else:
            st.json(cache)
