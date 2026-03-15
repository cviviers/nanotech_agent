from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore

try:
    from agents.schemas import AnalysisConfig
except Exception:  # pragma: no cover
    from novelty_app.agents.schemas import AnalysisConfig


@dataclass
class AnalysisArtifacts:
    df: pd.DataFrame
    x_primary: np.ndarray
    x_analysis: np.ndarray
    x_pca: Optional[np.ndarray]
    x_umap_2d: Optional[np.ndarray]
    graph: Any
    gap_regions: list[list[int]]
    selected_clustering: str
    cluster_column: str
    analysis_config: Dict[str, Any]


def _kneighbors_batched(
    nn: NearestNeighbors,
    x: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    distances: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        stop = min(len(x), start + batch_size)
        batch_distances, batch_indices = nn.kneighbors(x[start:stop], return_distance=True)
        distances.append(batch_distances)
        indices.append(batch_indices)
    return np.vstack(distances), np.vstack(indices)


def _build_knn_graph(x: np.ndarray, k: int, metric: str = "cosine") -> nx.Graph:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(x)
    dists, indices = _kneighbors_batched(nn, x)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(x)))
    for i in range(len(x)):
        for j, d in zip(indices[i, 1:], dists[i, 1:]):
            weight = 1.0 - float(d)
            if not graph.has_edge(i, j):
                graph.add_edge(i, j, weight=weight)
            elif graph[i][j]["weight"] < weight:
                graph[i][j]["weight"] = weight
    return graph


def _compute_density_features(x: np.ndarray, k_list: list[int], metric: str = "cosine") -> pd.DataFrame:
    features: Dict[str, Any] = {}
    for k in k_list:
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        nn.fit(x)
        distances, _ = _kneighbors_batched(nn, x)
        avg_distance = distances[:, 1:].mean(axis=1)
        features[f"density_k{k}"] = avg_distance
        features[f"density_k{k}_z"] = zscore(avg_distance, nan_policy="omit")
    df = pd.DataFrame(features)
    z_cols = [c for c in df.columns if c.endswith("_z")]
    df["gap_score"] = df[z_cols].mean(axis=1)
    return df


def _identify_gap_regions(df: pd.DataFrame, graph: nx.Graph, gap_quantile: float, min_gap_region_size: int) -> list[list[int]]:
    gap_threshold = df["gap_score"].quantile(gap_quantile)
    gap_candidates_idx = df[df["gap_score"] >= gap_threshold].index.tolist()
    gap_subgraph = graph.subgraph(gap_candidates_idx).copy()
    gap_regions = [list(component) for component in nx.connected_components(gap_subgraph)]
    gap_regions = [region for region in gap_regions if len(region) >= min_gap_region_size]
    gap_regions.sort(key=len, reverse=True)
    return gap_regions


def _safe_neighbors(k: int, n_samples: int) -> int:
    return max(1, min(int(k), max(1, n_samples - 1)))


def _cluster_embeddings(x: np.ndarray, config: AnalysisConfig) -> tuple[np.ndarray, str]:
    hdbscan_mod = None
    if config.clustering_method == "hdbscan":
        try:
            import hdbscan as hdbscan_mod  # type: ignore
        except Exception:  # pragma: no cover
            hdbscan_mod = None
    if config.clustering_method == "hdbscan" and hdbscan_mod is not None and len(x) >= 5:
        clusterer = hdbscan_mod.HDBSCAN(
            min_cluster_size=max(2, min(config.hdbscan_min_cluster_size, len(x))),
            min_samples=max(1, min(config.hdbscan_min_samples, len(x) - 1)),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(x)
        non_noise = {int(v) for v in labels if int(v) != -1}
        if non_noise:
            return labels.astype(int), "hdbscan"
    n_clusters = config.kmeans_n_clusters or min(20, max(2, int(np.sqrt(len(x)))))
    n_clusters = max(2, min(n_clusters, len(x)))
    model = KMeans(n_clusters=n_clusters, random_state=config.random_seed, n_init=10)
    return model.fit_predict(x).astype(int), "kmeans"


def run_analysis_v1(
    df: pd.DataFrame,
    x_primary: np.ndarray,
    *,
    config: Optional[AnalysisConfig] = None,
) -> AnalysisArtifacts:
    if df is None or len(df) == 0:
        raise ValueError("df must be non-empty")
    if len(x_primary) != len(df):
        raise ValueError("x_primary must align with df")

    config = config or AnalysisConfig()
    df_work = df.reset_index(drop=True).copy()

    x_pca: Optional[np.ndarray] = None
    if config.use_pca_for_analysis:
        n_components = min(config.pca_components, x_primary.shape[1], max(2, len(df_work) - 1))
        reducer = PCA(n_components=n_components, random_state=config.random_seed)
        x_pca = reducer.fit_transform(x_primary)
        x_analysis = x_pca
    else:
        x_analysis = x_primary

    labels, selected = _cluster_embeddings(x_analysis, config)
    cluster_column = f"cluster_{selected}"
    df_work[cluster_column] = labels
    df_work["cluster_selected"] = labels

    graph_k = _safe_neighbors(config.knn_graph_k, len(df_work))
    graph = _build_knn_graph(x_analysis, graph_k, config.density_metric)

    density_k_list = sorted({_safe_neighbors(k, len(df_work)) for k in config.density_k_list})
    density_df = _compute_density_features(x_analysis, density_k_list, config.density_metric)
    for col in density_df.columns:
        df_work[col] = density_df[col].values

    gap_regions = _identify_gap_regions(df_work, graph, config.gap_quantile, config.min_gap_region_size)
    for region_id, region in enumerate(gap_regions):
        for idx in region:
            df_work.loc[idx, "gap_region"] = region_id

    x_umap_2d: Optional[np.ndarray] = None
    umap_mod = None
    if config.compute_umap:
        try:
            import umap as umap_mod  # type: ignore
        except Exception:  # pragma: no cover
            umap_mod = None
    if config.compute_umap and umap_mod is not None and len(df_work) >= 3:
        reducer = umap_mod.UMAP(
            n_neighbors=_safe_neighbors(config.umap_neighbors, len(df_work)),
            min_dist=config.umap_min_dist,
            n_components=2,
            random_state=config.random_seed,
        )
        x_umap_2d = reducer.fit_transform(x_analysis)
        df_work["umap_x"] = x_umap_2d[:, 0]
        df_work["umap_y"] = x_umap_2d[:, 1]

    return AnalysisArtifacts(
        df=df_work,
        x_primary=x_primary,
        x_analysis=x_analysis,
        x_pca=x_pca,
        x_umap_2d=x_umap_2d,
        graph=graph,
        gap_regions=[list(map(int, region)) for region in gap_regions],
        selected_clustering=selected,
        cluster_column=cluster_column,
        analysis_config=config.model_dump(),
    )
