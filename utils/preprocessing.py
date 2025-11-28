"""
Preprocessing utilities for computing dimensionality reduction on embeddings
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import umap
from sklearn.preprocessing import StandardScaler


@dataclass
class EmbeddingConfig:
    """Configuration for UMAP dimensionality reduction"""
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "cosine"
    random_state: int = 42


def preprocess_embeddings(
    df: pd.DataFrame,
    embedding_col: str,
    config: EmbeddingConfig,
    cache_dir: Path
) -> Dict[str, Any]:
    """
    Compute UMAP projection for a given embedding column
    
    Args:
        df: DataFrame containing embeddings
        embedding_col: Name of the embedding column to process
        config: UMAP configuration
        cache_dir: Directory to save preprocessed results
    
    Returns:
        Dictionary containing:
            - df: DataFrame with added 'low_x' and 'low_y' columns
            - umap_2d: UMAP projection (N, 2)
            - config: Configuration used
            - cache_path: Path to cached file
    """
    # Extract embeddings
    embeddings = np.stack(df[embedding_col].values)

    # remove other embeddings columns that are not needed
    for col in df.columns:
        if col != embedding_col and col.endswith('_embedding'):
            df = df.drop(columns=[col])
    
    # Compute UMAP
    reducer = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        n_components=config.n_components,
        metric=config.metric,
        random_state=config.random_state,
        verbose=True
    )
    
    umap_2d = reducer.fit_transform(embeddings)
    
    # Add to dataframe
    df_processed = df.copy()
    df_processed['low_x'] = umap_2d[:, 0]
    df_processed['low_y'] = umap_2d[:, 1]
    df_processed['size'] = 20
    df_processed['cluster_label'] = 'unlabeled'
    
    # Cache results
    cache_path = cache_dir / f"{embedding_col}.pkl"
    result = {
        'df': df_processed,
        'umap_2d': umap_2d,
        'embedding_col': embedding_col,
        'config': config,
        'cache_path': str(cache_path)
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def load_preprocessed_data(cache_path: Path) -> Dict[str, Any]:
    """Load preprocessed data from cache"""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def preprocess_all_embeddings(
    df: pd.DataFrame,
    embedding_cols: list,
    config: EmbeddingConfig,
    cache_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Preprocess multiple embedding columns
    
    Returns:
        Dictionary mapping embedding_col -> preprocessed results
    """
    results = {}
    
    for emb_col in embedding_cols:
        print(f"Processing {emb_col}...")
        try:
            result = preprocess_embeddings(df, emb_col, config, cache_dir)
            results[emb_col] = result
            print(f"  ✓ Success: {result['cache_path']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return results


def compute_combined_embedding(
    df: pd.DataFrame,
    embedding_cols: list,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Combine multiple embeddings with optional weighting
    
    Args:
        df: DataFrame with embedding columns
        embedding_cols: List of embedding column names
        weights: Optional dictionary of weights per column
    
    Returns:
        Combined embedding matrix (N, D)
    """
    if weights is None:
        weights = {col: 1.0 for col in embedding_cols}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Stack and weight embeddings
    embeddings = []
    for col in embedding_cols:
        emb = np.stack(df[col].values)
        w = weights.get(col, 1.0)
        embeddings.append(emb * w)
    
    # Average
    combined = np.mean(embeddings, axis=0)
    
    # L2 normalize
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / (norms + 1e-8)
    
    return combined
