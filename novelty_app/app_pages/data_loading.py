"""
Data Loading and Configuration Page
"""

import calendar
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from agents.corpus_manifest import build_frontend_corpus_manifest
except Exception:  # pragma: no cover
    from novelty_app.agents.corpus_manifest import build_frontend_corpus_manifest

from core.data_utils import extract_embeddings


def _normalize_required_date(value: str, field_label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_label} is required.")
    try:
        return pd.Timestamp(text).date().isoformat()
    except Exception as exc:
        raise ValueError(f"{field_label} must be a valid date in YYYY-MM-DD format.") from exc


def _build_publication_dates(df: pd.DataFrame) -> pd.Series:
    def _numeric_column(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series([None] * len(df), index=df.index, dtype="float64")

    years = _numeric_column("publication_year")
    months = _numeric_column("publication_month")
    days = _numeric_column("publication_day")

    values = []
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


def page_data_loading():
    """Data loading and initial configuration page."""
    st.title("Data Loading & Configuration")
    st.markdown("Configure the basic data parameters and load your dataset.")

    config_col, embedding_col = st.columns(2)

    with config_col:
        st.subheader("Data Configuration")
        data_path = st.text_input(
            "Data File Path",
            value=r"./data/cleaned_dataset.json",
            help="Path to your JSON dataset file",
        )
        sample_n = st.number_input(
            "Sample Size (0 = all data)",
            min_value=0,
            max_value=100000,
            value=0,
            help="Limit dataset size for faster processing.",
        )
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="Seed for reproducible results across all random operations.",
        )

    with embedding_col:
        st.subheader("Embedding Configuration")
        available_embeddings = st.multiselect(
            "Available Embedding Files",
            ["qwen", "bert"],
            default=["qwen", "bert"],
            help="Select which embedding files to load from the data directory.",
        )
        primary_embedding = st.selectbox(
            "Primary Embedding",
            available_embeddings,
            index=0 if available_embeddings else None,
        )
        default_qwen_api_url = (
            (st.session_state.get("config") or {}).get("qwen_api_url")
            or os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000")
        )
        qwen_api_url = st.text_input(
            "Qwen API Endpoint",
            value=str(default_qwen_api_url),
            key="config_qwen_api_url",
            help="Base URL for the Qwen embedding service (e.g. http://127.0.0.1:8000).",
        ).strip()

    st.subheader("OpenAI API Key (Optional)")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("openai_api_key", os.environ.get("OPENAI_API_KEY", "")),
        key="config_openai_api_key",
        help=(
            "Enter your OpenAI API key for LLM-based analysis features. "
            "Can also be set via OPENAI_API_KEY."
        ),
    )

    if st.button("Save Configuration", type="primary"):
        normalized_qwen_api_url = qwen_api_url.rstrip("/") if qwen_api_url else "http://127.0.0.1:8000"
        st.session_state.config = {
            "data_path": data_path,
            "sample_n": sample_n if sample_n > 0 else None,
            "embedding_cols": available_embeddings,
            "primary_embedding": primary_embedding,
            "random_seed": random_seed,
            "qwen_api_url": normalized_qwen_api_url,
        }
        st.session_state.random_seed = random_seed
        st.session_state.openai_api_key = openai_api_key
        st.session_state.qwen_api_url = normalized_qwen_api_url
        st.success("Configuration saved.")

    st.divider()

    if "config" not in st.session_state:
        st.info("Save configuration first.")
        return

    st.subheader("Load Dataset")
    load_col, button_col = st.columns([3, 1])
    with load_col:
        keywords_title_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["review", "not available", "overview"],
        )
        keywords_abstract_exclusion = st.multiselect(
            "Exclusion Keywords (abstract)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["not available", "retraction", "overview"],
        )
        drop_post_cutoff = st.checkbox(
            "Drop papers after cutoff date during initial load",
            key="load_drop_post_cutoff",
            help="Makes the active working dataset historical-only immediately after load.",
        )
        cutoff_date = ""
        if drop_post_cutoff:
            cutoff_date = st.text_input(
                "Initial load cutoff date",
                key="load_cutoff_date",
                placeholder="YYYY-MM-DD",
                help="Papers after this date are removed from the working dataset during load.",
            )
            st.info(
                "The active working dataset becomes historical-only, but the full frontend corpus is still preserved "
                "for leakage-safe retrospective export and later evaluation."
            )
    with button_col:
        st.write("")
        st.write("")
        if st.button("Load Data", type="primary"):
            load_data(
                st.session_state.config,
                keywords_title_exclusion,
                keywords_abstract_exclusion,
                drop_post_cutoff=drop_post_cutoff,
                cutoff_date=cutoff_date,
            )

    if st.session_state.df_original is not None:
        st.success(f"Data loaded: {len(st.session_state.df_original)} papers")
        if st.session_state.df_filtered is not None:
            st.info(f"After sampling: {len(st.session_state.df_filtered)} papers")
        if st.session_state.get("load_cutoff_applied"):
            st.info(f"Initial cutoff filter active: {st.session_state.get('load_cutoff_date')}")
        with st.expander("Data Preview"):
            preview_df = st.session_state.df_valid if st.session_state.df_valid is not None else st.session_state.df_original
            st.dataframe(preview_df.head(10))


def load_data(
    config,
    keywords_title_exclusion,
    keywords_abstract_exclusion,
    *,
    drop_post_cutoff: bool = False,
    cutoff_date: str = "",
):
    """Load and filter dataset from JSON file, and load embeddings."""
    data_path = Path(config["data_path"])

    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return

    with st.spinner("Loading dataset from JSON..."):
        try:
            with data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["source_row_index"] = np.arange(len(df), dtype=int)
            st.session_state.df_original = df.copy()

            if config["sample_n"] is not None and len(df) > config["sample_n"]:
                df = df.sample(config["sample_n"], random_state=config.get("random_seed", 42))

            df = df.reset_index(drop=True)
            st.session_state.df_filtered = df
        except Exception as exc:
            st.error(f"Error loading JSON file: {exc}")
            st.code(traceback.format_exc())
            return

    with st.spinner("Loading embeddings from .npy files..."):
        try:
            data_dir = Path(config["data_path"]).parent
            embeddings_dict, valid_idx = extract_embeddings(
                st.session_state.df_filtered,
                config["embedding_cols"],
                data_dir,
            )
            st.session_state.embeddings_dict = embeddings_dict
            st.session_state.df_valid = st.session_state.df_filtered.iloc[valid_idx].reset_index(drop=True)
            st.session_state.X_primary = embeddings_dict[config["primary_embedding"]]
            st.session_state.embeddings_extracted = True
            st.success(f"Loaded embeddings: {len(valid_idx)} valid rows")
        except Exception as exc:
            st.error(f"Error loading embeddings: {exc}")
            st.code(traceback.format_exc())
            return

    with st.spinner("Applying keyword filters..."):
        n_before = len(st.session_state.df_valid)
        mask = pd.Series([True] * len(st.session_state.df_valid), index=st.session_state.df_valid.index)

        for keyword in keywords_title_exclusion:
            if "title" in st.session_state.df_valid.columns:
                mask &= ~st.session_state.df_valid["title"].str.lower().str.contains(keyword, na=False)

        for keyword in keywords_abstract_exclusion:
            if "abstract" in st.session_state.df_valid.columns:
                mask &= ~st.session_state.df_valid["abstract"].str.lower().str.contains(keyword, na=False)

        st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
        for key in st.session_state.embeddings_dict:
            st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][mask.values]
        st.session_state.X_primary = st.session_state.embeddings_dict[config["primary_embedding"]]

        n_after = len(st.session_state.df_valid)
        if n_before != n_after:
            st.info(f"Keyword filtering: {n_before} -> {n_after} papers")

    st.session_state.frontend_keyword_filters = {
        "title_exclusion_keywords": list(keywords_title_exclusion),
        "abstract_exclusion_keywords": list(keywords_abstract_exclusion),
    }
    st.session_state.df_valid_full = st.session_state.df_valid.copy()
    st.session_state.embeddings_dict_full = {
        key: value.copy() if value is not None else None
        for key, value in st.session_state.embeddings_dict.items()
    }
    st.session_state.X_primary_full = (
        st.session_state.X_primary.copy() if st.session_state.X_primary is not None else None
    )
    st.session_state.frontend_corpus_manifest = build_frontend_corpus_manifest(
        st.session_state.df_valid_full,
        sample_n=config.get("sample_n"),
        random_seed=int(config.get("random_seed", 42)),
        title_exclusion_keywords=keywords_title_exclusion,
        abstract_exclusion_keywords=keywords_abstract_exclusion,
        embedding_source=str(config["primary_embedding"]),
        available_embeddings=config.get("embedding_cols") or [str(config["primary_embedding"])],
        data_json=config["data_path"],
        data_dir=data_path.parent,
    )

    if drop_post_cutoff:
        try:
            cutoff_iso = _normalize_required_date(cutoff_date, "Initial load cutoff date")
            publication_dates = _build_publication_dates(st.session_state.df_valid)
            historical_mask = (publication_dates.notna()) & (publication_dates <= pd.Timestamp(cutoff_iso))
            excluded_post_cutoff = int((publication_dates.notna() & ~historical_mask).sum())
            excluded_undated = int(publication_dates.isna().sum())
            if not historical_mask.any():
                raise ValueError("No papers remain on or before the selected cutoff date.")

            st.session_state.df_valid = st.session_state.df_valid.loc[historical_mask].reset_index(drop=True)
            for key in st.session_state.embeddings_dict:
                st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][historical_mask.to_numpy()]
            st.session_state.X_primary = st.session_state.embeddings_dict[config["primary_embedding"]]
            st.session_state.load_cutoff_applied = True
            st.session_state.load_cutoff_date = cutoff_iso
            st.info(
                "Initial load cutoff applied: "
                f"{len(st.session_state.df_valid_full)} -> {len(st.session_state.df_valid)} papers "
                f"(excluded post-cutoff: {excluded_post_cutoff}, undated: {excluded_undated})."
            )
        except Exception as exc:
            st.error(f"Error applying initial cutoff filter: {exc}")
            st.code(traceback.format_exc())
            return
    else:
        st.session_state.load_cutoff_applied = False
        st.session_state.load_cutoff_date = ""
