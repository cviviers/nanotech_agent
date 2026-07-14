"""
Clustering algorithms and utilities
"""
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(X: np.ndarray, k: int, metric: str = 'cosine') -> nx.Graph:
    """Build k-NN graph from embeddings."""
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    dists, indices = nn.kneighbors(X, return_distance=True)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    
    for i in range(len(X)):
        for j, d in zip(indices[i, 1:], dists[i, 1:]):
            weight = 1.0 - float(d)
            if weight <= 0.0:
                continue
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=weight)
            elif G[i][j]['weight'] < weight:
                G[i][j]['weight'] = weight
    
    return G
