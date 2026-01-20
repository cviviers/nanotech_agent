"""
Data loading and embedding extraction functions
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st


def load_data(config, keywords_title_exclusion, keywords_abstract_exclusion):
    """Load and filter dataset from JSON file, and load embeddings"""
    data_path = Path(config['data_path'])
    
    if not data_path.exists():
        st.error(f"❌ Data file not found: {data_path}")
        return None
    
    with st.spinner("Loading dataset from JSON..."):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Apply title exclusions
        if keywords_title_exclusion:
            for kw in keywords_title_exclusion:
                kw_lower = kw.strip().lower()
                df = df[~df['title'].fillna('').str.lower().str.contains(kw_lower, regex=False)]
        
        # Apply abstract exclusions
        if keywords_abstract_exclusion:
            for kw in keywords_abstract_exclusion:
                kw_lower = kw.strip().lower()
                df = df[~df.get('abstract', df.get('processed_content', pd.Series([''] * len(df)))).fillna('').str.lower().str.contains(kw_lower, regex=False)]
        
        df = df.reset_index(drop=True)
        st.session_state.df_filtered = df
        st.success(f"✅ Loaded {len(df)} papers after filtering")
    
    # Load embeddings immediately after loading data
    with st.spinner("Loading embeddings from .npy files..."):
        try:
            data_dir = Path(config['data_dir'])
            embeddings_dict, valid_idx = extract_embeddings(
                df,
                config['embedding_cols'],
                data_dir
            )
            
            st.session_state.embeddings_dict = embeddings_dict
            st.session_state.df_valid = df.iloc[valid_idx].reset_index(drop=True)
            st.session_state.embeddings_extracted = True
            
            # Set primary embedding
            primary_key = config['primary_embedding']
            if primary_key in embeddings_dict:
                st.session_state.X_primary = embeddings_dict[primary_key]
            else:
                st.error(f"❌ Primary embedding '{primary_key}' not found in loaded embeddings")
                
            st.success(f"✅ Loaded embeddings for {len(st.session_state.df_valid)} papers")
            
        except Exception as e:
            st.error(f"❌ Error loading embeddings: {str(e)}")
    
    # Apply keyword filters AFTER embeddings are loaded
    with st.spinner("Applying keyword filters..."):
        st.session_state.df_original = st.session_state.df_filtered.copy()
        st.success("✅ Ready for analysis")


def extract_embeddings(df: pd.DataFrame, embed_names: List[str], data_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load embeddings from .npy files in data directory."""
    if not embed_names:
        raise ValueError(f"No embedding names provided")
    
    result = {}
    valid_mask = None
    
    for embed_name in embed_names:
        # Construct paths
        npy_path = data_dir / f"{embed_name}_embeddings.npy"
        metadata_path = data_dir / f"{embed_name}_embeddings_metadata.json"
        
        if not npy_path.exists():
            st.warning(f"Embedding file not found: {npy_path}")
            continue
        
        try:
            # Load embeddings
            embeddings = np.load(npy_path)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                st.info(f"Loaded {embed_name}: {embeddings.shape}, model: {metadata.get('model', 'unknown')}")
            else:
                st.info(f"Loaded {embed_name}: {embeddings.shape}")
            
            # Check if number of embeddings matches dataframe
            if len(embeddings) != len(df):
                st.warning(f"Embedding count mismatch for {embed_name}: {len(embeddings)} embeddings vs {len(df)} papers")
                # Use minimum length
                min_len = min(len(embeddings), len(df))
                embeddings = embeddings[:min_len]
                if valid_mask is None:
                    valid_mask = np.ones(min_len, dtype=bool)
                else:
                    valid_mask = valid_mask[:min_len]
            else:
                if valid_mask is None:
                    valid_mask = np.ones(len(embeddings), dtype=bool)
            
            result[embed_name] = embeddings.astype(np.float32)
            
        except Exception as e:
            st.error(f"Error loading {embed_name} embeddings: {str(e)}")
            continue
    
    if not result:
        raise ValueError(f"No embeddings could be loaded from: {embed_names}")
    
    valid_idx = np.where(valid_mask)[0]
    
    # Trim all embeddings to valid indices
    for key in result:
        result[key] = result[key][valid_idx]
    
    return result, valid_idx
