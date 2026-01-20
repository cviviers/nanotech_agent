"""
Data Loading and Configuration Page
"""
import os
import json
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd

from core.data_utils import extract_embeddings


def page_data_loading():
    """Data loading and initial configuration page"""
    st.title("📊 Data Loading & Configuration")
    
    st.markdown("""
    Configure the basic data parameters and load your dataset.
    """)
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Configuration")
        
        data_path = st.text_input(
            "Data File Path",
            value=r"./data/cleaned_dataset.json",
            help="Path to your JSON dataset file"
        )
        
        sample_n = st.number_input(
            "Sample Size (0 = all data)",
            min_value=0,
            max_value=100000,
            value=0,
            help="Limit dataset size for faster processing"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="Seed for reproducible results across all random operations"
        )
    
    with col2:
        st.subheader("🎯 Embedding Configuration")
        
        available_embeddings = st.multiselect(
            "Available Embedding Files",
            ["qwen", "bert"],
            default=["qwen", "bert"],
            help="Select which embedding files to load from /data/ folder"
        )
        
        primary_embedding = st.selectbox(
            "Primary Embedding",
            available_embeddings,
            index=0 if available_embeddings else None
        )
    
    # OpenAI API Key Configuration
    st.subheader("🔑 OpenAI API Key (Optional)")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get('openai_api_key', os.environ.get('OPENAI_API_KEY', '')),
        key="config_openai_api_key",
        help="Enter your OpenAI API key for LLM-based analysis features. Can also be set via OPENAI_API_KEY environment variable."
    )
    
    # Store in session state
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.config = {
            'data_path': data_path,
            'sample_n': sample_n if sample_n > 0 else None,
            'embedding_cols': available_embeddings,
            'primary_embedding': primary_embedding,
            'random_seed': random_seed
        }
        st.session_state.random_seed = random_seed
        st.session_state.openai_api_key = openai_api_key
        st.success("✅ Configuration saved!")
    
    st.divider()
    
    # Data loading
    if 'config' not in st.session_state:
        st.info("👆 Please save configuration first")
        return
    
    st.subheader("📥 Load Dataset")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        keywords_title_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["review", "not available", "overview"]
        )
        keywords_abstract_exclusion = st.multiselect(
            "Exclusion Keywords (abstract)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["not available", "retraction", "overview"]
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Load Data", type="primary"):
            load_data(st.session_state.config, keywords_title_exclusion, keywords_abstract_exclusion)
    
    # Show data status
    if st.session_state.df_original is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.df_original)} papers")
        
        if st.session_state.df_filtered is not None:
            st.info(f"After filtering: {len(st.session_state.df_filtered)} papers")
        
        with st.expander("📊 Data Preview"):
            st.dataframe(st.session_state.df_filtered.head(10) if st.session_state.df_filtered is not None 
                        else st.session_state.df_original.head(10))


def load_data(config, keywords_title_exclusion, keywords_abstract_exclusion):
    """Load and filter dataset from JSON file, and load embeddings"""
    data_path = Path(config['data_path'])
    
    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return
    
    with st.spinner("Loading dataset from JSON..."):
        # Load JSON file
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            st.session_state.df_original = df.copy()
            
            # Sample if needed (before filtering to maintain reproducibility)
            if config['sample_n'] is not None and len(df) > config['sample_n']:
                df = df.sample(config['sample_n'], random_state=config.get('random_seed', 42))
            
            df = df.reset_index(drop=True)
            st.session_state.df_filtered = df
            
        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")
            st.code(traceback.format_exc())
            return
    
    # Load embeddings immediately after loading data
    with st.spinner("Loading embeddings from .npy files..."):
        try:
            data_dir = Path(config['data_path']).parent
            
            embeddings_dict, valid_idx = extract_embeddings(
                st.session_state.df_filtered,
                config['embedding_cols'],
                data_dir
            )
            
            st.session_state.embeddings_dict = embeddings_dict
            st.session_state.df_valid = st.session_state.df_filtered.iloc[valid_idx].reset_index(drop=True)
            st.session_state.X_primary = embeddings_dict[config['primary_embedding']]
            st.session_state.embeddings_extracted = True
            
            st.success(f"✅ Loaded embeddings: {len(valid_idx)} valid rows")
            
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            st.code(traceback.format_exc())
            return
    
    # Apply keyword filters AFTER embeddings are loaded
    with st.spinner("Applying keyword filters..."):
        n_before = len(st.session_state.df_valid)
        
        # Create mask for filtering
        mask = pd.Series([True] * len(st.session_state.df_valid), index=st.session_state.df_valid.index)
        
        # Apply title filters
        for keyword in keywords_title_exclusion:
            if 'title' in st.session_state.df_valid.columns:
                mask &= ~st.session_state.df_valid['title'].str.lower().str.contains(keyword, na=False)
        
        # Apply abstract filters
        for keyword in keywords_abstract_exclusion:
            if 'abstract' in st.session_state.df_valid.columns:
                mask &= ~st.session_state.df_valid['abstract'].str.lower().str.contains(keyword, na=False)
        
        # Apply mask to dataframe
        st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
        
        # Apply mask to embeddings
        for key in st.session_state.embeddings_dict:
            st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][mask.values]
        
        # Update primary embedding
        st.session_state.X_primary = st.session_state.embeddings_dict[config['primary_embedding']]
        
        n_after = len(st.session_state.df_valid)
        if n_before != n_after:
            st.info(f"Keyword filtering: {n_before} → {n_after} papers")
