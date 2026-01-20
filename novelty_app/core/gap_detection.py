"""
Gap detection and region identification
"""
from typing import List
import numpy as np
import networkx as nx
import pandas as pd


def identify_gap_regions(df: pd.DataFrame, G: nx.Graph, gap_quantile: float, min_gap_region_size: int) -> List[List[int]]:
    """Identify gap regions from gap candidates"""
    gap_threshold = df['gap_score'].quantile(gap_quantile)
    gap_candidates_idx = df[df['gap_score'] >= gap_threshold].index.tolist()
    
    # Create subgraph
    gap_subgraph = G.subgraph(gap_candidates_idx).copy()
    
    # Find connected components
    gap_regions = [list(component) for component in nx.connected_components(gap_subgraph)]
    gap_regions = [r for r in gap_regions if len(r) >= min_gap_region_size]
    gap_regions.sort(key=len, reverse=True)
    
    return gap_regions
