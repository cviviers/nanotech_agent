"""
Density computation functions
"""
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors


def compute_knn_density(X: np.ndarray, k: int, metric: str = 'cosine') -> np.ndarray:
    """Compute average distance to k nearest neighbors."""
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    return dists[:, 1:].mean(axis=1)


def compute_density_features(X: np.ndarray, k_list: List[int], metric: str = 'cosine') -> pd.DataFrame:
    """Compute density features for multiple k values."""
    features = {}
    
    for k in k_list:
        avg_dist = compute_knn_density(X, k, metric)
        features[f'density_k{k}'] = avg_dist
        features[f'density_k{k}_z'] = zscore(avg_dist, nan_policy='omit')
    
    df = pd.DataFrame(features)
    z_cols = [c for c in df.columns if c.endswith('_z')]
    df['gap_score'] = df[z_cols].mean(axis=1)
    
    return df
