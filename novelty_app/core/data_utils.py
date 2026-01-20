"""
Data utilities for loading and processing datasets and embeddings
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st


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
