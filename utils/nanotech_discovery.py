"""
Nanomedicine Novelty Discovery Pipeline (Steps 1–7)
--------------------------------------------------
This module implements the core steps for discovering low-density "gaps" in embedding space,
clustering, explaining separations with an LLM (OpenAI GPT‑5 family), building a lightweight
knowledge graph, adding temporal signals, and scoring/ranking candidates.

Input expectation: a pandas.DataFrame with columns (as provided by user):
['id','title','abstract','authors','journal','publication_year','publication_month','publication_day',
 'doi','keywords','language_list','embedding','processed_abstract','content','processed_content',
 'qwen_content_embedding','qwen_processed_content_embedding','bert_content_embedding','bert_processed_content_embedding']

Preferred embedding: 'bert_processed_content_embedding' (BioClinical‑ModernBERT‑base).

Dependencies (install as needed):
    pip install numpy pandas scikit-learn scipy networkx tqdm openai
Optional:
    pip install umap-learn igraph leidenalg python-louvain hdbscan spacy scispacy en-core-sci-sm

Set your OpenAI key:
    export OPENAI_API_KEY=...  (or set in code via environment variable)

Author: (you)
"""
from __future__ import annotations
import os
import ast
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
try:
    # trustworthiness available in sklearn.manifold (>=0.22)
    from sklearn.manifold import trustworthiness as sk_trustworthiness
except Exception:
    sk_trustworthiness = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Optional imports
try:
    import umap  # type: ignore
except Exception:
    umap = None

# Graph clustering backends (best-effort)
_leiden_ready = False
try:
    import igraph as ig  # type: ignore
    import leidenalg as la  # type: ignore
    _leiden_ready = True
except Exception:
    pass

try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

from utils.data_utils import Step1Config, Step2Config, Step3Config, Step4Config, Step5Config, PipelineConfig
# -----------------------------
# Utilities & Config
# -----------------------------

def _ensure_array(x: Any) -> Optional[np.ndarray]:
    """Coerce a cell value to a 1D float numpy array. Handles python lists or stringified lists.
    Returns None if cannot parse.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            arr = np.asarray(parsed, dtype=float)
            return arr
        except Exception:
            return None
    return None


def extract_embeddings(df: pd.DataFrame,
                       embed_cols: List[str]) -> Dict[str, np.ndarray]:
    """Extracts and stacks embeddings from the given columns. Skips columns not present.
    Returns dict: {col_name: (n, d)}. Rows with missing embeddings are dropped consistently across cols.
    """
    present = [c for c in embed_cols if c in df.columns]
    if not present:
        raise ValueError(f"None of the requested embedding columns exist: {embed_cols}")

    # Build mask of rows having vectors in the preferred column
    row_ok = None
    arrays_by_col: Dict[str, List[Optional[np.ndarray]]] = {}
    for col in present:
        col_arrays = [_ensure_array(v) for v in df[col].tolist()]
        arrays_by_col[col] = col_arrays
        col_mask = np.array([a is not None for a in col_arrays])
        row_ok = col_mask if row_ok is None else (row_ok & col_mask)

    idx = np.where(row_ok)[0]
    out: Dict[str, np.ndarray] = {}
    for col in present:
        mats = [arrays_by_col[col][i] for i in idx]
        out[col] = np.vstack(mats).astype(np.float32)
    return out

# -----------------------------
# STEP 1 — kNN graph & density in ORIGINAL space (+stability)
# -----------------------------

def knn_avg_distance(X: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    dists, idx = nn.kneighbors(X, n_neighbors=k+1, return_distance=True)
    # dists[:,0] is zero/self; exclude it
    return dists[:, 1:].mean(axis=1)


def compute_density_panel(X: np.ndarray, k_list: Tuple[int, ...], metric: str) -> pd.DataFrame:
    rows = {}
    for k in k_list:
        avgd = knn_avg_distance(X, k=k, metric=metric)
        rows[f"avgd_k{k}"] = avgd
        rows[f"avgd_k{k}_z"] = zscore(avgd, nan_policy='omit')
    return pd.DataFrame(rows)


def bootstrap_low_density_stability(X: np.ndarray,
                                    k: int = 30,
                                    n_bootstrap: int = 10,
                                    frac: float = 0.8,
                                    metric: str = "cosine",
                                    random_state: int = 42,
                                    low_density_top_p: float = 0.10) -> np.ndarray:
    """Return per-point frequency of being in the lowest-density p-quantile across bootstraps."""
    rng = check_random_state(random_state)
    n = X.shape[0]
    counts = np.zeros(n, dtype=float)
    for b in range(n_bootstrap):
        mask = rng.rand(n) < frac
        idx = np.where(mask)[0]
        if len(idx) < k + 5:
            continue
        Xb = X[idx]
        avgd = knn_avg_distance(Xb, k=k, metric=metric)
        thresh = np.quantile(avgd, 1 - low_density_top_p)  # top distances = low density
        low_mask = avgd >= thresh
        counts[idx[low_mask]] += 1
    return counts / max(1, n_bootstrap)


def compute_trustworthiness_metrics(X: np.ndarray,
                                    compute_tsne_umap: bool = False,
                                    trust_k: int = 15,
                                    random_state: int = 42) -> Dict[str, float]:
    metrics = {}
    if sk_trustworthiness is None or not compute_tsne_umap:
        return metrics
    rng = check_random_state(random_state)
    # UMAP
    if umap is not None:
        um = umap.UMAP(n_neighbors=trust_k, min_dist=0.1, random_state=random_state)
        X_um = um.fit_transform(X)
        try:
            tw_um = sk_trustworthiness(X, X_um, n_neighbors=trust_k, metric='euclidean')
            metrics['trustworthiness_umap'] = float(tw_um)
        except Exception:
            pass
    # t-SNE
    try:
        ts = TSNE(n_components=2, perplexity=max(5, trust_k//2), random_state=random_state, init='pca')
        X_ts = ts.fit_transform(X)
        tw_ts = sk_trustworthiness(X, X_ts, n_neighbors=trust_k, metric='euclidean')
        metrics['trustworthiness_tsne'] = float(tw_ts)
    except Exception:
        pass
    return metrics


# -----------------------------
# STEP 2 — Ensemble across embeddings (if available)
# -----------------------------

def ensemble_density(density_per_encoder: Dict[str, pd.DataFrame],
                     weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Combine z-scored average distances across encoders via weighted mean of z-scores.
    Returns DataFrame with columns: ['gap_z_mean', 'gap_z_max', 'enc_*'] per k.
    """
    # Align by index
    ks = sorted({int(c.split('avgd_k')[1].split('_')[0])
                 for df in density_per_encoder.values() for c in df.columns if c.startswith('avgd_k') and not c.endswith('_z')})
    zcols = [f"avgd_k{k}_z" for k in ks]

    W = {k: 1.0 for k in density_per_encoder.keys()}
    if weights:
        for enc, w in weights.items():
            if enc in W:
                W[enc] = float(w)

    # Weighted average of z-scores per k
    zs = []
    for k in ks:
        num = 0.0
        den = 0.0
        for enc, dfz in density_per_encoder.items():
            if f"avgd_k{k}_z" in dfz.columns:
                w = W.get(enc, 1.0)
                num += w * dfz[f"avgd_k{k}_z"].values
                den += w
        zs.append(num / max(1e-9, den))
    zs = np.vstack(zs).T  # shape (n, len(ks))

    out = pd.DataFrame({f"gap_z_k{k}": zs[:, i] for i, k in enumerate(ks)})
    out['gap_z_mean'] = out[[f"gap_z_k{k}" for k in ks]].mean(axis=1)
    out['gap_z_max'] = out[[f"gap_z_k{k}" for k in ks]].max(axis=1)
    return out


# -----------------------------
# STEP 3 — Graph clustering (Leiden/Louvain) + HDBSCAN + stability
# -----------------------------

def build_knn_graph(X: np.ndarray, k: int, metric: str = 'cosine') -> nx.Graph:
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, n_neighbors=k+1, return_distance=True)
    n = X.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Add mutual edges with similarity weights
    # Convert distance to similarity (cosine): sim = 1 - dist
    for i in range(n):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if i == j:
                continue
            w = 1.0 - float(d)
            if G.has_edge(i, j):
                # keep max weight
                if G[i][j]['weight'] < w:
                    G[i][j]['weight'] = w
            else:
                G.add_edge(i, j, weight=w)
    return G


def cluster_graph_leiden(G: nx.Graph, resolution: float = 1.0, random_state: int = 42) -> np.ndarray:
    if not _leiden_ready:
        raise RuntimeError("leidenalg/igraph not available")
    # Convert to igraph
    mapping = {n: i for i, n in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    igG = ig.Graph(n=len(mapping), edges=edges)
    igG.es['weight'] = weights
    part = la.find_partition(igG, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution, seed=random_state)
    labels = np.zeros(len(mapping), dtype=int)
    for cid, comm in enumerate(part):
        labels[np.array(list(comm))] = cid
    # Map back to original node order
    inv = {i: n for n, i in mapping.items()}
    labels_ordered = np.zeros(len(mapping), dtype=int)
    for i in range(len(mapping)):
        labels_ordered[inv[i]] = labels[i]
    return labels_ordered


def cluster_graph_louvain(G: nx.Graph, random_state: int = 42) -> np.ndarray:
    if community_louvain is None:
        raise RuntimeError("python-louvain not available")
    part = community_louvain.best_partition(G, weight='weight', random_state=random_state)
    # part is dict node -> community id
    n = G.number_of_nodes()
    labels = np.array([part[i] for i in range(n)], dtype=int)
    return labels


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 50, min_samples: Optional[int] = None) -> np.ndarray:
    if hdbscan is None:
        warnings.warn("hdbscan not installed; falling back to DBSCAN-like behavior with simple threshold.")
        # crude fallback: assign all to one cluster (0)
        return np.zeros(X.shape[0], dtype=int)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    labels = cl.fit_predict(X)
    # relabel noise (-1) sequentially after max label to keep ints non-negative
    if np.any(labels == -1):
        maxlab = labels[labels >= 0].max() if np.any(labels >= 0) else -1
        noise_ids = np.where(labels == -1)[0]
        labels[noise_ids] = np.arange(maxlab + 1, maxlab + 1 + len(noise_ids))
    return labels


def jaccard_overlap(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    # Cluster-level Jaccard based on pairwise co-assignment agreements
    n = len(labels_a)
    same_a = (labels_a[:, None] == labels_a[None, :])
    same_b = (labels_b[:, None] == labels_b[None, :])
    iu = np.triu_indices(n, k=1)
    inter = np.logical_and(same_a, same_b)[iu].sum()
    union = np.logical_or(same_a, same_b)[iu].sum()
    return float(inter) / max(1, float(union))


# -----------------------------
# STEP 4 — Contrastive, evidence-grounded summaries via OpenAI
# -----------------------------
class OpenAIClient:
    def __init__(self, model: str = "gpt-5", temperature: float = 0.2):
        # Uses the new OpenAI Python SDK v1 interface if available, fallback to legacy
        self.model = model
        self.temperature = temperature
        self._mode = None
        try:
            from openai import OpenAI as _OpenAI  # type: ignore
            self.client = _OpenAI()
            self._mode = 'sdkv1'
        except Exception:
            import openai  # type: ignore
            self.client = openai
            self._mode = 'legacy'

    def chat_json(self, system: str, user: str, seed: Optional[int] = None) -> Dict[str, Any]:
        if self._mode == 'sdkv1':
            resp = self.client.chat.completions.create(
                model=self.model,
                # temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **({"seed": seed} if seed is not None else {})
            )
            txt = resp.choices[0].message.content
            return json.loads(txt)
        else:
            # legacy
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            resp = self.client.ChatCompletion.create(
                model=self.model,
                # temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            txt = resp["choices"][0]["message"]["content"]
            return json.loads(txt)


def build_evidence_pack(df: pd.DataFrame,
                        X: np.ndarray,
                        labels: np.ndarray,
                        cluster_id: int,
                        top_n: int = 30) -> List[Dict[str, Any]]:
    # find centroid
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    Xc = X[idx]
    centroid = Xc.mean(axis=0, keepdims=True)
    d = pairwise_distances(Xc, centroid, metric='cosine').ravel()
    order = idx[np.argsort(d)]
    take = order[:min(top_n, len(order))]
    pack = []
    for i in take:
        row = df.iloc[i]
        pack.append({
            "doc_id": str(row.get('id', i)),
            "title": row.get('title', ''),
            "year": int(row.get('publication_year', -1)) if not pd.isna(row.get('publication_year', np.nan)) else -1,
            "mesh": row.get('keywords', []) if isinstance(row.get('keywords', None), list) else [],
            "entities": {},
            "text": row.get('abstract') or row.get('processed_content') or row.get('content') or '',
            "url": f"https://doi.org/{row['doi']}" if isinstance(row.get('doi', None), str) else ''
        })
    return pack


def discriminative_terms(df: pd.DataFrame, labels: np.ndarray, A: int, B: int,
                         text_col: str = 'processed_content',
                         max_features: int = 30000, min_df: int = 3,
                         C: float = 0.5, max_iter: int = 2000) -> Dict[str, List[Dict[str, Any]]]:
    idxA = np.where(labels == A)[0]
    idxB = np.where(labels == B)[0]
    idx = np.concatenate([idxA, idxB])
    y = np.array([0]*len(idxA) + [1]*len(idxB))
    corpus = [str(df.iloc[i].get(text_col) or df.iloc[i].get('abstract') or '') for i in idx]
    tfidf = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=(1,2))
    Xtf = tfidf.fit_transform(corpus)
    lr = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=max_iter, n_jobs=1)
    lr.fit(Xtf, y)
    vocab = np.array(tfidf.get_feature_names_out())
    coefs = lr.coef_[0]
    a_minus_b = (
        [
            {"term": t, "weight": float(w)}
            for t, w in zip(vocab, coefs)
            if w < 0
        ]
    )
    b_minus_a = (
        [
            {"term": t, "weight": float(w)}
            for t, w in zip(vocab, coefs)
            if w > 0
        ]
    )
    a_minus_b = sorted(a_minus_b, key=lambda d: d['weight'])[:50]
    b_minus_a = sorted(b_minus_a, key=lambda d: -d['weight'])[:50]
    return {"coeffs_A_minus_B": a_minus_b, "coeffs_B_minus_A": b_minus_a}


def llm_contrastive_explain(client: OpenAIClient,
                            cluster_A_meta: Dict[str, Any],
                            cluster_B_meta: Dict[str, Any],
                            discrim_terms: Dict[str, Any],
                            evidence_pack: List[Dict[str, Any]],
                            temperature: float = 0.2) -> Dict[str, Any]:
    system = (
        "You are a nanomedicine domain expert. Only use the EVIDENCE PACK provided. "
        "Never invent facts or cite outside sources. If evidence is insufficient for any claim, "
        "state 'unknown'. Cite by doc_id for every claim. Output exactly the JSON schema."
    )

    user = f"""
TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
Focus on: materials, surface chemistry/coatings, size/shape, targeting ligands, disease areas,
models (in vitro/in vivo/clinical), delivery routes, pharmacokinetics/biodistribution,
toxicity/regulatory language, endpoints/outcomes.

CONTEXT
- cluster_A_meta: {json.dumps(cluster_A_meta)}
- cluster_B_meta: {json.dumps(cluster_B_meta)}

DISCRIMINATIVE_TERMS:
{json.dumps(discrim_terms, ensure_ascii=False)}

EVIDENCE PACK (JSONL; each line is one doc)
```jsonl
{chr(10).join(json.dumps(d, ensure_ascii=False) for d in evidence_pack)}
```

OUTPUT JSON SCHEMA
{{
  "cluster_A_summary": {{
    "one_line": "string",
    "bullets": ["string"],
    "salient_entities": {{"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]}},
    "citations": ["doc_id"]
  }},
  "cluster_B_summary": {{
    "one_line": "string",
    "bullets": ["string"],
    "salient_entities": {{"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]}},
    "citations": ["doc_id"]
  }},
  "axes_of_separation": [{{
      "axis": "materials|ligands|disease|model|delivery|toxicity|methods|other",
      "what_differs": "short explanation (evidence-grounded)",
      "evidence_A": ["doc_id"],
      "evidence_B": ["doc_id"],
      "confidence": 0.0
  }}],
  "bridge_seeds": [{{
      "idea": "short description of a possible bridge",
      "why_plausible": "mechanistic rationale, grounded in docs",
      "support": ["doc_id"],
      "risks": ["toxicity","aggregation","RES","immunogenicity","scaleup","IP","assay_limitations"]
  }}],
  "insufficient_evidence": false
}}
"""
    return client.chat_json(system=system, user=user)


# -----------------------------
# STEP 5 — Lightweight Knowledge Graph + Link Prediction (heuristic)
# -----------------------------
MATERIAL_HINTS = [
    'liposome','plga','gold','agnp','au','iron oxide','magnetite','silica',
    'mesoporous','graphene','go','peg','chitosan','albumin','micelle',
    'dendrimer','hydrogel','quantum dot','nanotube','nanoemulsion'
]

LIGAND_HINTS = [
    'rgd','folate','transferrin','aptamer','peptide','antibody','egf',
    'her2','mannose','galactose','hyaluronic'
]

DISEASE_HINTS = [
    'cancer','glioblastoma','breast','lung','pancreatic','pancreatic cancer',
    'prostate','melanoma','liver','ovarian','colorectal','colorectal cancer',
    'infection','inflammation','chronic inflammation','chronic inflammatory disease',
    'alzheimer','alzheimer\'s disease','neurodegenerative','neurodegenerative disease',
    'inflammatory bowel disease','ibd',
    'rheumatoid arthritis','autoimmune','autoimmunity'
]

DELIVERY_HINTS = [
    'intravenous','iv','oral','oral delivery','intratumoral','inhalation','topical','intranasal',
    'systemic','systemic delivery',
    'local','local delivery','local effects',
    'sustained release','local sustained release',
    'brain delivery',
    'blood brain barrier','barrier passage',
    'barrier penetration','barrier disruption'
]

MODEL_HINTS = [
    'in vitro','in vivo','mouse','murine','rat','xenograft','clinical','phase'
]

def simple_entity_extract(text: str) -> Dict[str, List[str]]:
    t = (text or '').lower()
    ents = {
        'materials': sorted({w for w in MATERIAL_HINTS if w in t}),
        'ligands': sorted({w for w in LIGAND_HINTS if w in t}),
        'diseases': sorted({w for w in DISEASE_HINTS if w in t}),
        'delivery': sorted({w for w in DELIVERY_HINTS if w in t}),
        'models': sorted({w for w in MODEL_HINTS if w in t}),
    }
    return ents


def build_light_kg(df: pd.DataFrame,
                   text_col: str = 'processed_content') -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for i, row in df.iterrows():
        doc_id = str(row.get('id', i))
        text = str(row.get(text_col) or row.get('abstract') or '')
        ents = simple_entity_extract(text)
        # add nodes
        for cat, values in ents.items():
            for v in values:
                G.add_node(f"{cat}:{v}", kind=cat, name=v)
        # add edges (co-occur)
        # material - disease
        for m in ents['materials']:
            for d in ents['diseases']:
                G.add_edge(f"materials:{m}", f"diseases:{d}", key=f"doc:{doc_id}", rel='co_occurs', doc_id=doc_id)
        # material - ligand
        for m in ents['materials']:
            for l in ents['ligands']:
                G.add_edge(f"materials:{m}", f"ligands:{l}", key=f"doc:{doc_id}", rel='co_occurs', doc_id=doc_id)
    return G


def adamic_adar_candidates(G: nx.MultiDiGraph,
                           head_prefix: str = 'materials:',
                           tail_prefix: str = 'diseases:',
                           top_k: int = 50) -> List[Tuple[str, str, float]]:
    # Convert to simple undirected for AA index
    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        UG.add_edge(u, v)
    heads = [n for n in UG.nodes() if isinstance(n, str) and n.startswith(head_prefix)]
    tails = [n for n in UG.nodes() if isinstance(n, str) and n.startswith(tail_prefix)]
    existing = set(UG.edges()) | set((b, a) for a, b in UG.edges())
    scores = []
    for h in heads:
        for t in tails:
            if (h, t) in existing:
                continue
            try:
                aa = list(nx.adamic_adar_index(UG, [(h, t)]))[0][2]
            except Exception:
                aa = 0.0
            scores.append((h, t, float(aa)))
    scores.sort(key=lambda x: -x[2])
    return scores[:top_k]


# -----------------------------
# STEP 6 — Time as a first-class signal
# -----------------------------

def year_bins(df: pd.DataFrame, win: int = 5, min_year: int = 1995, max_year: Optional[int] = None) -> List[Tuple[int,int]]:
    if max_year is None:
        max_year = int(pd.to_numeric(df['publication_year'], errors='coerce').max())
    bins = []
    y = max(min_year, int(pd.to_numeric(df['publication_year'], errors='coerce').min()))
    while y <= max_year:
        bins.append((y, min(y + win - 1, max_year)))
        y += win
    return bins


def density_over_time(X: np.ndarray, years: np.ndarray, k: int = 30, metric: str = 'cosine', bins: Optional[List[Tuple[int,int]]] = None) -> List[Tuple[Tuple[int,int], float]]:
    if bins is None:
        bins = year_bins(pd.DataFrame({'publication_year': years}))
    out = []
    for lo, hi in bins:
        mask = (years >= lo) & (years <= hi)
        if mask.sum() < max(10, k+1):
            out.append(((lo, hi), np.nan))
            continue
        avgd = knn_avg_distance(X[mask], k=min(k, mask.sum()-1), metric=metric)
        out.append(((lo, hi), float(np.mean(avgd))))
    return out


def slope_of_series(series: List[Tuple[Tuple[int,int], float]]) -> float:
    xs = []
    ys = []
    for (lo, hi), val in series:
        if not np.isnan(val):
            xs.append((lo+hi)/2)
            ys.append(val)
    if len(xs) < 2:
        return float('nan')
    x = np.array(xs)
    y = np.array(ys)
    # simple linear slope normalized by time span
    coef = np.polyfit(x, y, 1)[0]
    return float(coef)


# -----------------------------
# STEP 7 — Scoring rubric (novelty/feasibility/impact/confidence)
# -----------------------------
@dataclass
class GapFeatures:
    density_z: float
    nn_distance_z: float
    kg_rarity_z: float
    method_enablers: float
    toxicity_rate_z: float
    model_availability: float
    translational_language: float
    burden_score: float
    adjacency_citation_velocity: float
    stability: float
    evidence_coverage: float


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip01(x):
    return float(np.minimum(1.0, np.maximum(0.0, x)))


def score_gap(f: GapFeatures) -> Dict[str, float]:
    novelty = sigmoid(0.45*f.density_z + 0.35*f.nn_distance_z + 0.20*f.kg_rarity_z)
    feasibility = clip01(0.35*f.method_enablers + 0.25*(1 - sigmoid(f.toxicity_rate_z)) + 0.20*f.model_availability + 0.20*f.translational_language)
    impact = clip01(0.60*f.burden_score + 0.40*f.adjacency_citation_velocity)
    confidence = clip01(0.7*f.stability + 0.3*f.evidence_coverage)
    pgs = 0.45*novelty + 0.30*feasibility + 0.25*impact
    priority = pgs * math.sqrt(confidence)
    return {"novelty": novelty, "feasibility": feasibility, "impact": impact, "confidence": confidence, "PGS": pgs, "priority": priority}


# -----------------------------
# High-level orchestrator helpers
# -----------------------------
class NoveltyPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig(), save_plots: bool = False):
        self.cfg = config
        self.save_plots = save_plots

    def step1_density(self, X_by_enc: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
        s1 = self.cfg.step1
        out = {}
        for enc, X in X_by_enc.items():
            print(f"[Step1] Computing densities for encoder: {enc}  shape={X.shape}")
            dens = compute_density_panel(X, s1.k_list, s1.metric)
            print("  Computing low-density stability via bootstrap...")
            stab = bootstrap_low_density_stability(X, k=max(s1.k_list), n_bootstrap=s1.n_bootstrap,
                                                   frac=s1.bootstrap_frac, metric=s1.metric,
                                                   random_state=s1.random_state)
            print("  Computing trustworthiness metrics...")
            dens['low_density_stability'] = stab
            trust = compute_trustworthiness_metrics(X, compute_tsne_umap=s1.compute_tsne_umap,
                                                    trust_k=s1.trust_k, random_state=s1.random_state)
            
            for k,v in trust.items():
                dens[k] = v
            out[enc] = dens

        return out

    def step2_ensemble(self, density_by_enc: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        print("[Step2] Ensembling density signals across embeddings (if multiple)")
        ens = ensemble_density(density_by_enc, weights=self.cfg.step2.weights)
        return ens

    def step3_clustering(self, X: np.ndarray, save_plots: bool) -> Dict[str, Any]:
        s3 = self.cfg.step3
        print("[Step3] Building kNN graph and clustering...")
        G = build_knn_graph(X, k=s3.knn_for_graph, metric=s3.graph_metric)
        # Try Leiden, fallback to Louvain
        try:
            comm_labels = cluster_graph_leiden(G, resolution=s3.leiden_resolution, random_state=s3.random_state)
            algo = 'leiden'
        except Exception:
            if community_louvain is None:
                warnings.warn("No Leiden or Louvain available; community clustering skipped. Using HDBSCAN only.")
                comm_labels = None
                algo = 'none'
            else:
                comm_labels = cluster_graph_louvain(G, random_state=s3.random_state)
                algo = 'louvain'
        # HDBSCAN on X (euclidean by default)
        hdb_labels = cluster_hdbscan(X, min_cluster_size=s3.hdbscan_min_cluster_size,
                                     min_samples=s3.hdbscan_min_samples)
        # Stability measure between the two labelings
        jac = None
        if comm_labels is not None:
            jac = jaccard_overlap(np.array(comm_labels), np.array(hdb_labels))
        return {"graph": G, "community_algo": algo, "community_labels": comm_labels, "hdbscan_labels": hdb_labels, "jaccard_overlap": jac}

    def step4_contrastive(self, df: pd.DataFrame, X: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray,
                          cluster_A: int, cluster_B: int, openai_client: Optional[OpenAIClient] = None,
                          evidence_docs_per_cluster: Optional[int] = None) -> Dict[str, Any]:
        s4 = self.cfg.step4
        if openai_client is None:
            openai_client = OpenAIClient(model=s4.openai_model, temperature=s4.temperature)
        if evidence_docs_per_cluster is None:
            evidence_docs_per_cluster = s4.evidence_docs_per_cluster

        packA = build_evidence_pack(df, X, labels_a, cluster_A, top_n=evidence_docs_per_cluster)
        packB = build_evidence_pack(df, X, labels_b, cluster_B, top_n=evidence_docs_per_cluster)
        pack = packA + packB
        discr = discriminative_terms(df, labels_a, cluster_A, cluster_B,
                                     text_col='processed_content',
                                     max_features=s4.max_features_tfidf,
                                     min_df=s4.min_df, C=s4.lr_C, max_iter=s4.lr_max_iter)
        metaA = {"id": str(cluster_A), "n_docs": int((labels_a==cluster_A).sum())}
        metaB = {"id": str(cluster_B), "n_docs": int((labels_b==cluster_B).sum())}
        return llm_contrastive_explain(openai_client, metaA, metaB, discr, pack, temperature=s4.temperature)

    def step5_kg(self, df: pd.DataFrame, text_col: str = 'processed_content') -> Tuple[nx.MultiDiGraph, List[Tuple[str,str,float]]]:
        print("[Step5] Building lightweight knowledge graph and proposing link predictions...")
        KG = build_light_kg(df, text_col=text_col)
        preds = adamic_adar_candidates(KG, head_prefix='materials:', tail_prefix='diseases:', top_k=50)
        return KG, preds

    def step6_temporal(self, X: np.ndarray, years: np.ndarray, k: int = 30, metric: str = 'cosine') -> Dict[str, Any]:
        print("[Step6] Computing density trends over time...")
        bins = year_bins(pd.DataFrame({'publication_year': years}), win=self.cfg.step6.window_years,
                         min_year=self.cfg.step6.min_year, max_year=self.cfg.step6.max_year)
        series = density_over_time(X, years, k=k, metric=metric, bins=bins)
        slope = slope_of_series(series)
        return {"bins": bins, "density_series": series, "slope": slope}

    def step7_score(self,
                    density_z: float,
                    nn_distance_z: float,
                    kg_rarity_z: float,
                    method_enablers: float,
                    toxicity_rate_z: float,
                    model_availability: float,
                    translational_language: float,
                    burden_score: float,
                    adjacency_citation_velocity: float,
                    stability: float,
                    evidence_coverage: float) -> Dict[str, float]:
        f = GapFeatures(density_z, nn_distance_z, kg_rarity_z, method_enablers, toxicity_rate_z,
                        model_availability, translational_language, burden_score,
                        adjacency_citation_velocity, stability, evidence_coverage)
        return score_gap(f)


# -----------------------------
# Example driver (pseudo-usage)
# -----------------------------
if __name__ == "__main__":
    # Example sketch (commented). Replace with your dataframe loading.
    # df = pd.read_parquet("your_dataframe.parquet")
    # cfg = PipelineConfig()
    # pipe = NoveltyPipeline(cfg)

    # Step 2: get embeddings (ensemble-ready)
    # X_by = extract_embeddings(df, list(cfg.step2.embedding_cols))

    # Step 1: densities per encoder
    # dens_by = pipe.step1_density(X_by)

    # Step 2: ensemble gap signals
    # ens = pipe.step2_ensemble(dens_by)

    # Choose the primary embedding (bert_processed_content_embedding) for downstream clustering
    # X = X_by[cfg.step2.embedding_cols[0]]

    # Step 3: clustering
    # clus = pipe.step3_clustering(X)
    # comm_labels = clus['community_labels'] if clus['community_labels'] is not None else clus['hdbscan_labels']

    # Step 4: contrastive explanation between two clusters (example: 0 vs 1)
    # client = OpenAIClient(model=cfg.step4.openai_model, temperature=cfg.step4.temperature)
    # explanation = pipe.step4_contrastive(df, X, comm_labels, comm_labels, 0, 1, openai_client=client)
    # print(json.dumps(explanation, indent=2))

    # Step 5: knowledge graph + link prediction
    # KG, link_preds = pipe.step5_kg(df)

    # Step 6: temporal trends
    # years = pd.to_numeric(df['publication_year'], errors='coerce').fillna(-1).astype(int).values
    # temporal = pipe.step6_temporal(X, years, k=cfg.step3.knn_for_graph, metric=cfg.step3.graph_metric)

    # Step 7: scoring (example with placeholder features)
    # scores = pipe.step7_score(density_z=1.2, nn_distance_z=0.8, kg_rarity_z=0.4,
    #                           method_enablers=0.6, toxicity_rate_z=0.3, model_availability=0.7,
    #                           translational_language=0.5, burden_score=0.8, adjacency_citation_velocity=0.5,
    #                           stability=0.65, evidence_coverage=0.7)
    # print(scores)

    print("Module loaded. Import and use NoveltyPipeline with your DataFrame.")
