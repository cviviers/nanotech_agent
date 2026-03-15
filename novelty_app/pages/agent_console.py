"""Agent Console page for publishing snapshots and interacting with the agent backend."""

from __future__ import annotations

import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from agents.backend_client import BackendClient
    _BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    BackendClient = None  # type: ignore
    _BACKEND_IMPORT_ERROR = exc

try:
    from agents.orchestrator_langgraph import build_orchestrator
except Exception:  # pragma: no cover
    build_orchestrator = None  # type: ignore

try:
    from agents.snapshot_builder import build_snapshot_payload
except Exception:  # pragma: no cover
    from novelty_app.agents.snapshot_builder import build_snapshot_payload


DEFAULT_BACKEND_URL = "http://localhost:8088"


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
    for key in ("pmid", "id", "paper_id", "doi"):
        if key in row.index:
            value = row.get(key)
            if not _is_null(value):
                text = str(value).strip()
                if text:
                    return text
    return f"row_{row_pos}_{row_label}"


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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    df = st.session_state.get("df_valid")
    if df is None:
        raise ValueError("No df_valid in session state. Load data first.")
    return build_snapshot_payload(
        df=df,
        gap_regions=st.session_state.get("gap_regions") or [],
        llm_results=st.session_state.get("llm_results"),
        selected_clustering=st.session_state.get("selected_clustering"),
        x_primary=st.session_state.get("X_primary"),
        x_umap_2d=st.session_state.get("X_umap_2d"),
        include_raw_rows=include_raw_rows,
        include_embeddings=include_embeddings,
        snapshot_id=st.session_state.get("agent_snapshot_id") or f"snapshot_{uuid.uuid4().hex[:10]}",
        source="streamlit_agent_console",
    )


def _get_backend() -> BackendClient:
    if BackendClient is None:
        raise RuntimeError(f"Backend client unavailable: {_BACKEND_IMPORT_ERROR}")
    base_url = (st.session_state.get("agent_backend_url") or DEFAULT_BACKEND_URL).strip()
    return BackendClient(base_url=base_url)


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
        payload, summary = _build_snapshot_payload(
            include_raw_rows=include_raw_rows,
            include_embeddings=include_embeddings,
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
            resp = _get_backend().publish_snapshot(payload)
            st.session_state.agent_publish_result = resp
            if resp.get("snapshot_id"):
                st.session_state.agent_snapshot_id = resp["snapshot_id"]
            st.success("Snapshot published")
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
                resp = backend.store_artifact(kind=kind, target=target, payload=payload)
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
