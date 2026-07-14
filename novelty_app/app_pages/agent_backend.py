"""Agent Backend page for publishing snapshots and managing backend artifacts."""

from __future__ import annotations

import streamlit as st

from .agent_console import (
    _BACKEND_IMPORT_ERROR,
    _ensure_agent_page_state,
    _get_backend,
    _tab_artifacts,
    _tab_orchestrator,
    _tab_skills_playground,
    _tab_snapshot_publish,
)


def page_agent_backend() -> None:
    st.title("Agent Backend")
    st.caption(
        "Publish analysis snapshots, inspect backend state, run backend tools, and manage stored artifacts."
    )
    st.code("uvicorn agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088", language="bash")
    if _BACKEND_IMPORT_ERROR is not None:
        st.error(
            "Agent backend client could not be imported. Install the missing dependency (likely `requests`) "
            "and reload the app."
        )
        st.caption(str(_BACKEND_IMPORT_ERROR))
        return

    _ensure_agent_page_state()

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