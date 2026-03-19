from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiscoveryCue(BaseModel):
    text: str = ""
    goal: Optional[str] = None
    include_terms: List[str] = Field(default_factory=list)
    avoid_terms: List[str] = Field(default_factory=list)
    preferred_fields: Dict[str, List[str]] = Field(default_factory=dict)
    hard_constraints: Dict[str, List[str]] = Field(default_factory=dict)
    soft_constraints: Dict[str, List[str]] = Field(default_factory=dict)
    counter_queries: List[str] = Field(default_factory=list)
    fingerprint: Dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    embedding_name: str = "qwen"
    use_pca_for_analysis: bool = True
    pca_components: int = 102
    clustering_method: Literal["hdbscan", "kmeans"] = "hdbscan"
    kmeans_n_clusters: Optional[int] = None
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 10
    knn_graph_k: int = 21
    density_metric: str = "cosine"
    density_k_list: List[int] = Field(default_factory=lambda: [10, 20, 30, 50])
    gap_quantile: float = 0.95
    min_gap_region_size: int = 3
    random_seed: int = 42
    compute_umap: bool = False
    umap_neighbors: int = 50
    umap_min_dist: float = 0.1
    notes: Optional[str] = None


class SnapshotMetadata(BaseModel):
    source: str = "streamlit_agent_console"
    selected_clustering: Optional[str] = None
    cluster_column: Optional[str] = None
    n_rows: int = 0
    has_gap_regions: bool = False
    has_llm_results: bool = False
    has_embeddings: bool = False
    embedding_dim: Optional[int] = None
    split_role: Optional[Literal["historical", "future", "full"]] = None
    cutoff_date: Optional[str] = None
    future_window_start: Optional[str] = None
    future_window_end: Optional[str] = None
    analysis_config: Dict[str, Any] = Field(default_factory=dict)
    embedding_source: Optional[str] = None
    data_hash: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SnapshotPayload(BaseModel):
    snapshot_id: str
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    gap_papers: List[Dict[str, Any]] = Field(default_factory=list)
    llm_analyses: List[Dict[str, Any]] = Field(default_factory=list)


class EvidencePack(BaseModel):
    snapshot_id: str
    target_type: Literal["gap", "cluster_pair"]
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    discovery_cue: Optional[Dict[str, Any]] = None


class GeneratedHypothesis(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_id: Optional[str] = None
    hypothesis_id: str
    target_id: str
    target_type: str
    method_name: str
    model_name: Optional[str] = None
    seed: int = 0
    title: str
    text: str
    support_citations: List[str] = Field(default_factory=list)
    grounding_summary: Dict[str, Any] = Field(default_factory=dict)
    raw_hypothesis: Dict[str, Any] = Field(default_factory=dict)
    normalized_hypothesis: Dict[str, Any] = Field(default_factory=dict)
    idea_fingerprint: Dict[str, Any] = Field(default_factory=dict)
    idea_scores: Dict[str, Any] = Field(default_factory=dict)
    discovery_cue: Dict[str, Any] = Field(default_factory=dict)


class EvaluationRun(BaseModel):
    run_id: str
    snapshot_id: Optional[str] = None
    created_at: str
    cutoff_date: Optional[str] = None
    future_window_start: Optional[str] = None
    future_window_end: Optional[str] = None
    method_names: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    discovery_cue: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMatch(BaseModel):
    match_id: Optional[str] = None
    run_id: str
    snapshot_id: Optional[str] = None
    created_at: Optional[str] = None
    target_id: str
    target_type: str
    method_name: str
    seed: int = 0
    hypothesis_id: str
    classification: Literal[
        "already_present",
        "anticipatory_strong",
        "anticipatory_partial",
        "unsupported",
        "unrealized",
    ]
    historical_label: Literal["strong_match", "partial_match", "background_only", "no_match"] = "no_match"
    future_label: Literal["strong_match", "partial_match", "background_only", "no_match"] = "no_match"
    first_future_year: Optional[int] = None
    historical_best_paper_id: Optional[str] = None
    future_best_paper_id: Optional[str] = None
    support_citations: List[str] = Field(default_factory=list)
    hypothesis: Dict[str, Any] = Field(default_factory=dict)
    idea_scores: Dict[str, Any] = Field(default_factory=dict)
    fingerprint: Dict[str, Any] = Field(default_factory=dict)
    historical_match: Dict[str, Any] = Field(default_factory=dict)
    future_match: Dict[str, Any] = Field(default_factory=dict)
    discovery_cue: Dict[str, Any] = Field(default_factory=dict)


class ReviewPacket(BaseModel):
    run_id: str
    created_at: str
    cutoff_date: Optional[str] = None
    future_window_start: Optional[str] = None
    future_window_end: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
