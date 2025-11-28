import os
import json
import pandas as pd
import time
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Step1Config:
    k_list: Tuple[int, ...] = (15, 30, 50)
    metric: str = "cosine"  # 'cosine' recommended for sentence embeddings
    n_bootstrap: int = 100
    bootstrap_frac: float = 0.8
    random_state: int = 42
    trust_k: int = 15  # for trustworthiness
    compute_tsne_umap: bool = True  # set True if you want 2D viz + trustworthiness checks


@dataclass
class Step2Config:
    # Order reflects preference. First one is the primary.
    embedding_cols: Tuple[str, ...] = (
        'bert_processed_content_embedding',
        'bert_content_embedding',
        'qwen_processed_content_embedding',
        'qwen_content_embedding',
    )
    weights: Optional[Dict[str, float]] = None  # e.g., {'bert_processed_content_embedding': 2.0}


@dataclass
class Step3Config:
    knn_for_graph: int = 30
    graph_metric: str = "cosine"
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: Optional[int] = 15
    random_state: int = 42


@dataclass
class Step4Config:
    evidence_docs_per_cluster: int = 30
    max_features_tfidf: int = 30000
    min_df: int = 3
    lr_C: float = 0.5
    lr_max_iter: int = 2000
    openai_model: str = "gpt-5"  # adjust as needed
    temperature: float = 0.2


@dataclass
class Step5Config:
    use_scispacy: bool = False  # set True if you installed scispacy and a model
    scispacy_model: str = "en_core_sci_sm"


@dataclass
class Step6Config:
    window_years: int = 5
    min_year: int = 1995
    max_year: Optional[int] = None  # infer from data if None


@dataclass
class Step7Config:
    # thresholds and weights used in scoring (the actual formula is below)
    pass


@dataclass
class PipelineConfig:
    step1: Step1Config = Step1Config()
    step2: Step2Config = Step2Config()
    step3: Step3Config = Step3Config()
    step4: Step4Config = Step4Config()
    step5: Step5Config = Step5Config()
    step6: Step6Config = Step6Config()
    step7: Step7Config = Step7Config()

