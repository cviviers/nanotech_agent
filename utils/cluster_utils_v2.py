"""
Clustering and filtering utilities (v2)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import List


def kmeans_cluster(df: pd.DataFrame, num_clusters: int = 3) -> pd.DataFrame:
    """
    Apply K-means clustering to the UMAP coordinates
    
    Args:
        df: DataFrame with 'low_x' and 'low_y' columns
        num_clusters: Number of clusters
    
    Returns:
        DataFrame with added 'cluster_label' column
    """
    df = df.copy()
    
    # Use UMAP coordinates for clustering
    X = df[['low_x', 'low_y']].values
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Convert to strings for categorical coloring
    df['cluster_label'] = [str(label) for label in labels]
    
    return df


def assign_class_to_embeddings(
    df: pd.DataFrame,
    search_terms: str,
    assigned_class: str
) -> pd.DataFrame:
    """
    Assign class labels based on text search in abstract/title
    
    Args:
        df: DataFrame
        search_terms: Comma-separated search terms
        assigned_class: Class label(s) to assign
    
    Returns:
        DataFrame with updated 'cluster_label' column
    """
    df = df.copy()
    
    # Parse inputs
    terms = [t.strip() for t in search_terms.split(',')]
    classes = [c.strip() for c in assigned_class.split(',')]
    
    # Initialize all to 'Not Assigned' if not already clustered
    if 'cluster_label' not in df.columns:
        df['cluster_label'] = 'Not Assigned'
    
    # If single class for multiple terms, replicate
    if len(classes) == 1 and len(terms) > 1:
        classes = classes * len(terms)
    
    # Search in abstract and title
    search_cols = []
    if 'abstract' in df.columns:
        search_cols.append('abstract')
    if 'title' in df.columns:
        search_cols.append('title')
    
    for term, cls in zip(terms, classes):
        mask = pd.Series([False] * len(df), index=df.index)
        
        for col in search_cols:
            mask = mask | df[col].fillna('').str.contains(term, case=False, na=False)
        
        num_matches = mask.sum()
        print(f"Assigned {num_matches} records to class '{cls}' for term '{term}'")
        
        df.loc[mask, 'cluster_label'] = cls
    
    return df


def filter_by_bounding_box(
    df: pd.DataFrame,
    x1: float,
    x2: float,
    y1: float,
    y2: float
) -> pd.DataFrame:
    """
    Filter dataframe by bounding box in UMAP space
    
    Args:
        df: DataFrame with 'low_x' and 'low_y'
        x1, x2: X-axis bounds
        y1, y2: Y-axis bounds
    
    Returns:
        Filtered DataFrame
    """
    mask = (
        (df['low_x'] >= x1) &
        (df['low_x'] <= x2) &
        (df['low_y'] >= y1) &
        (df['low_y'] <= y2)
    )
    
    df_filtered = df[mask].reset_index(drop=True)
    print(f"Filtered to {len(df_filtered)} records (from {len(df)})")
    
    return df_filtered


def filter_by_clusters(df: pd.DataFrame, cluster_labels: List[str]) -> pd.DataFrame:
    """
    Filter dataframe to keep only specified clusters
    
    Args:
        df: DataFrame with 'cluster_label' column
        cluster_labels: List of cluster labels to keep
    
    Returns:
        Filtered DataFrame
    """
    mask = df['cluster_label'].isin(cluster_labels)
    df_filtered = df[mask].reset_index(drop=True)
    
    print(f"Filtered to {len(df_filtered)} records in clusters {cluster_labels}")
    
    return df_filtered
