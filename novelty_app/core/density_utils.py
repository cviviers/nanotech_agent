"""
Density computation utilities for gap analysis
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore
from typing import List


def compute_knn_density(X: np.ndarray, k: int, metric: str = 'cosine') -> np.ndarray:
    """Compute k-NN density (average distance to k nearest neighbors)."""
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X, return_distance=True)
    avg_distance = distances[:, 1:].mean(axis=1)
    return avg_distance


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
