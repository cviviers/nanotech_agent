"""
Streamlit App for Interactive Novelty Analysis and Gap Discovery
Based on novelty_analysis_direct.ipynb workflow
"""
import os
import sys
import json
import ast
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

# ML libraries
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
import umap
import hdbscan
import networkx as nx

# Graph clustering
try:
    import igraph as ig
    import leidenalg as la
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

# OpenAI for LLM analysis
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add utils path
sys.path.append(str(Path(__file__).parent))


# ============================================================================
# DOMAIN-SPECIFIC ENTITY HINTS
# ============================================================================

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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_embedding(value: Any) -> Optional[np.ndarray]:
    """Parse embedding from various formats (list, array, string)"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=float)
        except Exception:
            return None
    return None


def extract_embeddings(df: pd.DataFrame, embed_names: List[str], data_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load embeddings from .npy files in data directory."""
    if not embed_names:
        raise ValueError(f"No embedding names provided")
    
    result = {}
    valid_mask = None
    
    for embed_name in embed_names:
        # Construct paths
        npy_path = data_dir / f"{embed_name}_embeddings.npy"
        metadata_path = data_dir / f"{embed_name}_embeddings_metadata.json"
        
        if not npy_path.exists():
            st.warning(f"Embedding file not found: {npy_path}")
            continue
        
        try:
            # Load embeddings
            embeddings = np.load(npy_path)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                st.info(f"Loaded {embed_name}: {embeddings.shape}, model: {metadata.get('model', 'unknown')}")
            else:
                st.info(f"Loaded {embed_name}: {embeddings.shape}")
            
            # Check if number of embeddings matches dataframe
            if len(embeddings) != len(df):
                st.warning(f"Embedding count mismatch for {embed_name}: {len(embeddings)} embeddings vs {len(df)} papers")
                # Use minimum length
                min_len = min(len(embeddings), len(df))
                embeddings = embeddings[:min_len]
                if valid_mask is None:
                    valid_mask = np.ones(min_len, dtype=bool)
                else:
                    valid_mask = valid_mask[:min_len]
            else:
                if valid_mask is None:
                    valid_mask = np.ones(len(embeddings), dtype=bool)
            
            result[embed_name] = embeddings.astype(np.float32)
            
        except Exception as e:
            st.error(f"Error loading {embed_name} embeddings: {str(e)}")
            continue
    
    if not result:
        raise ValueError(f"No embeddings could be loaded from: {embed_names}")
    
    valid_idx = np.where(valid_mask)[0]
    
    # Trim all embeddings to valid indices
    for key in result:
        result[key] = result[key][valid_idx]
    
    return result, valid_idx


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


def simple_entity_extract(text: str) -> Dict[str, List[str]]:
    """Extract domain-specific entities from text using keyword matching."""
    text_lower = (text or '').lower()
    
    entities = {
        'materials': sorted({w for w in MATERIAL_HINTS if w in text_lower}),
        'ligands': sorted({w for w in LIGAND_HINTS if w in text_lower}),
        'diseases': sorted({w for w in DISEASE_HINTS if w in text_lower}),
        'delivery': sorted({w for w in DELIVERY_HINTS if w in text_lower}),
        'models': sorted({w for w in MODEL_HINTS if w in text_lower}),
    }
    
    return entities


def extract_entities_from_dataframe(df: pd.DataFrame, text_col: str = 'processed_content') -> pd.DataFrame:
    """Extract entities for all papers in dataframe."""
    entity_lists = {
        'materials': [],
        'ligands': [],
        'diseases': [],
        'delivery': [],
        'models': []
    }
    
    for _, row in df.iterrows():
        text = str(row.get(text_col) or row.get('abstract') or row.get('content') or '')
        entities = simple_entity_extract(text)
        
        for key in entity_lists.keys():
            entity_lists[key].append(entities.get(key, []))
    
    # Add as columns to dataframe copy
    df_with_entities = df.copy()
    for key, values in entity_lists.items():
        df_with_entities[f'entities_{key}'] = values
    
    return df_with_entities


def summarize_gap_region_entities(df: pd.DataFrame, region_indices: List[int]) -> Dict[str, Any]:
    """Summarize entity distribution in a gap region."""
    region_df = df.loc[region_indices]
    
    summary = {}
    for entity_type in ['materials', 'ligands', 'diseases', 'delivery', 'models']:
        col_name = f'entities_{entity_type}'
        if col_name in region_df.columns:
            # Flatten list of lists and count
            all_entities = []
            for entity_list in region_df[col_name]:
                if isinstance(entity_list, list):
                    all_entities.extend(entity_list)
            
            entity_counts = Counter(all_entities)
            summary[entity_type] = {
                'total_unique': len(entity_counts),
                'total_mentions': sum(entity_counts.values()),
                'top_5': entity_counts.most_common(5)
            }
        else:
            summary[entity_type] = {
                'total_unique': 0,
                'total_mentions': 0,
                'top_5': []
            }
    
    return summary


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
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=weight)
            elif G[i][j]['weight'] < weight:
                G[i][j]['weight'] = weight
    
    return G


# ============================================================================
# UNDO FUNCTIONALITY
# ============================================================================

def save_state_for_undo(action_name: str):
    """Save current state to undo history"""
    state_snapshot = {
        'action_name': action_name,
        'df_valid': st.session_state.df_valid.copy() if st.session_state.df_valid is not None else None,
        'X_pca': st.session_state.X_pca.copy() if st.session_state.X_pca is not None else None,
        'X_primary': st.session_state.X_primary.copy() if st.session_state.X_primary is not None else None,
        'X_umap_2d': st.session_state.X_umap_2d.copy() if st.session_state.X_umap_2d is not None else None,
        'kmeans_applied': st.session_state.kmeans_applied,
        'similarity_applied': st.session_state.similarity_applied,
        'G': st.session_state.G,  # Graph is immutable, can reference directly
        'clustering_done': st.session_state.clustering_done
    }
    st.session_state.undo_history.append(state_snapshot)
    
    # Limit history to last 10 actions
    if len(st.session_state.undo_history) > 10:
        st.session_state.undo_history.pop(0)


def undo_last_action():
    """Restore previous state from undo history"""
    if not st.session_state.undo_history:
        return False
    
    snapshot = st.session_state.undo_history.pop()
    
    # Restore state
    st.session_state.df_valid = snapshot['df_valid']
    st.session_state.X_pca = snapshot['X_pca']
    st.session_state.X_primary = snapshot['X_primary']
    st.session_state.X_umap_2d = snapshot['X_umap_2d']
    st.session_state.kmeans_applied = snapshot['kmeans_applied']
    st.session_state.similarity_applied = snapshot['similarity_applied']
    st.session_state.G = snapshot.get('G', None)
    st.session_state.clustering_done = snapshot.get('clustering_done', False)
    
    # Update UMAP coordinates in dataframe if available
    if st.session_state.df_valid is not None and st.session_state.X_umap_2d is not None:
        st.session_state.df_valid['umap_x'] = st.session_state.X_umap_2d[:, 0]
        st.session_state.df_valid['umap_y'] = st.session_state.X_umap_2d[:, 1]
    
    return True


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    
    # Data states
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_filtered' not in st.session_state:
        st.session_state.df_filtered = None
    if 'df_valid' not in st.session_state:
        st.session_state.df_valid = None
    
    # Processing states
    if 'embeddings_extracted' not in st.session_state:
        st.session_state.embeddings_extracted = False
    if 'embeddings_dict' not in st.session_state:
        st.session_state.embeddings_dict = {}
    if 'X_primary' not in st.session_state:
        st.session_state.X_primary = None
    if 'X_pca' not in st.session_state:
        st.session_state.X_pca = None
    if 'X_umap_2d' not in st.session_state:
        st.session_state.X_umap_2d = None
    
    # Analysis states
    if 'density_computed' not in st.session_state:
        st.session_state.density_computed = False
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'gaps_identified' not in st.session_state:
        st.session_state.gaps_identified = False
    if 'G' not in st.session_state:
        st.session_state.G = None
    if 'gap_regions' not in st.session_state:
        st.session_state.gap_regions = []
    
    # Filter states
    if 'kmeans_applied' not in st.session_state:
        st.session_state.kmeans_applied = False
    if 'similarity_applied' not in st.session_state:
        st.session_state.similarity_applied = False
    if 'qa_retrieval_applied' not in st.session_state:
        st.session_state.qa_retrieval_applied = False
    
    # Clustering selection
    if 'selected_clustering' not in st.session_state:
        st.session_state.selected_clustering = None
    
    # Undo history
    if 'undo_history' not in st.session_state:
        st.session_state.undo_history = []


# ============================================================================
# PAGE: CONFIGURATION & DATA LOADING
# ============================================================================

def page_data_loading():
    """Data loading and initial configuration page"""
    st.title("📊 Data Loading & Configuration")
    
    st.markdown("""
    Configure the basic data parameters and load your dataset.
    """)
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Configuration")
        
        data_path = st.text_input(
            "Data File Path",
            value=r"./data/cleaned_dataset.json",
            help="Path to your JSON dataset file"
        )
        
        sample_n = st.number_input(
            "Sample Size (0 = all data)",
            min_value=0,
            max_value=100000,
            value=0,
            help="Limit dataset size for faster processing"
        )
    
    with col2:
        st.subheader("🎯 Embedding Configuration")
        
        available_embeddings = st.multiselect(
            "Available Embedding Files",
            ["qwen", "bert"],
            default=["qwen", "bert"],
            help="Select which embedding files to load from /data/ folder"
        )
        
        primary_embedding = st.selectbox(
            "Primary Embedding",
            available_embeddings,
            index=0 if available_embeddings else None
        )
    
    # Store in session state
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.config = {
            'data_path': data_path,
            'sample_n': sample_n if sample_n > 0 else None,
            'embedding_cols': available_embeddings,
            'primary_embedding': primary_embedding
        }
        st.success("✅ Configuration saved!")
    
    st.divider()
    
    # Data loading
    if 'config' not in st.session_state:
        st.info("👆 Please save configuration first")
        return
    
    st.subheader("📥 Load Dataset")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        keywords_title_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["review", "not available", "overview"]
        )
        keywords_abstract_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["not available", "retraction", "overview"]
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Load Data", type="primary"):
            load_data(st.session_state.config, keywords_title_exclusion, keywords_abstract_exclusion)
    
    # Show data status
    if st.session_state.df_original is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.df_original)} papers")
        
        if st.session_state.df_filtered is not None:
            st.info(f"After filtering: {len(st.session_state.df_filtered)} papers")
        
        with st.expander("📊 Data Preview"):
            st.dataframe(st.session_state.df_filtered.head(10) if st.session_state.df_filtered is not None 
                        else st.session_state.df_original.head(10))


def load_data(config, keywords_title_exclusion, keywords_abstract_exclusion):
    """Load and filter dataset from JSON file"""
    data_path = Path(config['data_path'])
    
    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return
    
    with st.spinner("Loading dataset from JSON..."):
        # Load JSON file
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            st.session_state.df_original = df.copy()
            
            # Apply filters
            for keyword in keywords_title_exclusion:
                if 'title' in df.columns:
                    df = df[~df['title'].str.lower().str.contains(keyword, na=False)]
            for keyword in keywords_abstract_exclusion:
                if 'abstract' in df.columns:
                    df = df[~df['abstract'].str.lower().str.contains(keyword, na=False)]
            
            # Sample if needed
            if config['sample_n'] is not None and len(df) > config['sample_n']:
                df = df.sample(config['sample_n'], random_state=42)
            
            df = df.reset_index(drop=True)
            st.session_state.df_filtered = df
            
        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================================
# PAGE: EMBEDDING EXTRACTION & DIMENSIONALITY REDUCTION
# ============================================================================

def page_embedding_processing():
    """Extract embeddings and apply dimensionality reduction"""
    st.title("🧬 Embedding Processing")
    
    if st.session_state.df_filtered is None:
        st.warning("⚠️ Please load data first")
        return
    
    config = st.session_state.config
    
    st.markdown(f"""
    **Dataset**: {len(st.session_state.df_filtered)} papers  
    **Primary Embedding**: {config['primary_embedding']}  
    **Available Embeddings**: {', '.join(config['embedding_cols'])}
    """)
    
    # Extract embeddings
    if not st.session_state.embeddings_extracted:
        if st.button("🔍 Load Embeddings", type="primary"):
            with st.spinner("Loading embeddings from .npy files..."):
                try:
                    # Get data directory from config path
                    data_dir = Path(config['data_path']).parent
                    
                    embeddings_dict, valid_idx = extract_embeddings(
                        st.session_state.df_filtered,
                        config['embedding_cols'],
                        data_dir
                    )
                    
                    st.session_state.embeddings_dict = embeddings_dict
                    st.session_state.df_valid = st.session_state.df_filtered.iloc[valid_idx].reset_index(drop=True)
                    st.session_state.X_primary = embeddings_dict[config['primary_embedding']]
                    st.session_state.embeddings_extracted = True
                    
                    st.success(f"✅ Loaded embeddings: {len(valid_idx)} valid rows")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        return
    
    st.success(f"✅ Embeddings extracted: {st.session_state.X_primary.shape}")
    
    st.divider()
    
    # PCA reduction
    st.subheader("📉 PCA Dimensionality Reduction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pca_components = st.number_input("PCA Components", min_value=10, max_value=1024, value=102)
    with col2:
        st.write("")
        st.write("")
        if st.button("▶️ Run PCA"):
            with st.spinner("Running PCA..."):
                pca = PCA(n_components=pca_components, random_state=42)
                X_pca = pca.fit_transform(st.session_state.X_primary)
                st.session_state.X_pca = X_pca
                explained_var = pca.explained_variance_ratio_.sum()
                st.success(f"✅ PCA: {X_pca.shape}, explained variance: {explained_var:.2%}")
    with col3:
        if st.session_state.X_pca is not None:
            st.metric("PCA Shape", f"{st.session_state.X_pca.shape}")
    
    if st.session_state.X_pca is None:
        return
    
    st.divider()
    
    # UMAP projection
    st.subheader("🗺️ UMAP 2D Projection")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        umap_neighbors = st.number_input("n_neighbors", min_value=5, max_value=200, value=50)
    with col2:
        umap_min_dist = st.number_input("min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col3:
        st.write("")
        st.write("")
        if st.button("▶️ Run UMAP"):
            with st.spinner("Computing UMAP projection..."):
                reducer_2d = umap.UMAP(
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    n_components=2,
                    random_state=42
                )
                X_umap_2d = reducer_2d.fit_transform(st.session_state.X_pca)
                st.session_state.X_umap_2d = X_umap_2d
                st.session_state.df_valid['umap_x'] = X_umap_2d[:, 0]
                st.session_state.df_valid['umap_y'] = X_umap_2d[:, 1]
                st.success(f"✅ UMAP 2D: {X_umap_2d.shape}")
    with col4:
        if st.session_state.X_umap_2d is not None:
            st.metric("UMAP Shape", f"{st.session_state.X_umap_2d.shape}")
    
    # Visualize UMAP
    if st.session_state.X_umap_2d is not None:
        st.subheader("📊 UMAP Visualization")
        
        # Prepare hover data
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            title="UMAP 2D Projection",
            opacity=0.6,
            height=1000,
            hover_data={'umap_x': False, 'umap_y': False, 'hover_title': True, 'hover_abstract': True}
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: OPTIONAL FILTERS
# ============================================================================

def page_filters():
    """Optional filtering page"""
    st.title("🎯 Optional Filters")
    
    if st.session_state.X_pca is None:
        st.warning("⚠️ Please extract embeddings and run PCA first")
        return
    
    st.markdown("""
    Apply optional filters to focus your analysis on specific research areas.
    """)
    
    # K-means filter
    st.subheader("1️⃣ K-means Clustering Filter")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        kmeans_n_clusters = st.slider("Number of Clusters", min_value=5, max_value=50, value=20)
    with col2:
        st.write("")
        if st.button("🎯 Run K-means"):
            run_kmeans_filter(kmeans_n_clusters)
    
    if st.session_state.kmeans_applied:
        st.success("✅ K-means clustering complete")
        
        # Cluster selection
        cluster_labels = st.session_state.df_valid['kmeans_cluster'].values
        unique_clusters = sorted(np.unique(cluster_labels))
        
        # Show cluster distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        col1, col2 = st.columns([3, 2])
        with col1:
            selected_clusters = st.multiselect(
                "Select Clusters to Keep",
                unique_clusters,
                help="Leave empty to keep all clusters"
            )
        with col2:
            st.write("")
            if selected_clusters and st.button("✂️ Apply Cluster Filter"):
                apply_cluster_filter(selected_clusters)
        
        # Visualize clusters
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            color='kmeans_cluster',
            title=f"K-means Clusters (n={kmeans_n_clusters})",
            opacity=0.7,
            height=1000,
            color_continuous_scale='rainbow',
            hover_data={'umap_x': False, 'umap_y': False, 'kmeans_cluster': True, 'hover_title': True, 'hover_abstract': True}
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Cluster Distribution"):
            st.bar_chart(cluster_counts)
    
    st.divider()
    
    # Semantic similarity filter
    st.subheader("2️⃣ Semantic Retrieval")
    
    retrieval_mode = st.radio(
        "Retrieval Mode",
        ["Semantic Similarity", "Question Answering (Reranker)"],
        help="Semantic Similarity: Find papers similar to a topic. Q&A: Find papers that answer a question."
    )
    
    if retrieval_mode == "Semantic Similarity":
        col1, col2 = st.columns([3, 1])
        with col1:
            query_text = st.text_area(
                "Search Query (Topic/Description)",
                value="Brain delivery for treatment of neurodegenerative diseases",
                height=100
            )
            similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Compute Similarities"):
                compute_semantic_similarity(query_text, similarity_threshold)
        
        if st.session_state.similarity_applied:
            similarities = st.session_state.df_valid['similarity_score'].values
            
            st.success(f"✅ Similarity computed. {(similarities >= similarity_threshold).sum()} papers above threshold")
            
            if st.button("✂️ Apply Similarity Filter"):
                apply_similarity_filter(similarity_threshold)
            
            # Visualize similarities
            df_plot = st.session_state.df_valid.copy()
            df_plot['hover_title'] = df_plot['title'].fillna('N/A')
            df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
            
            fig = px.scatter(
                df_plot,
                x='umap_x',
                y='umap_y',
                color='similarity_score',
                title="Semantic Similarity to Query",
                opacity=0.7,
                height=1000,
                color_continuous_scale='Viridis',
                hover_data={'umap_x': False, 'umap_y': False, 'similarity_score': ':.3f', 'hover_title': True, 'hover_abstract': True}
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top matches
            with st.expander("📄 Top 10 Most Similar Papers"):
                top_papers = st.session_state.df_valid.nlargest(10, 'similarity_score')
                for idx, row in top_papers.iterrows():
                    st.write(f"**[{row['similarity_score']:.3f}]** {row.get('title', 'N/A')}")
    
    else:  # Question Answering mode
        col1, col2 = st.columns([3, 1])
        with col1:
            question_text = st.text_area(
                "Research Question",
                value="What are the most effective nanoparticle delivery systems for crossing the blood-brain barrier?",
                height=100,
                help="Ask a specific research question. The system will find papers that best answer it."
            )
            qa_threshold = st.slider("Relevance Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                                    help="Papers with embedding similarity above this threshold will be reranked")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Retrieve Answers"):
                compute_question_answering_retrieval(question_text, qa_threshold)
        
        if st.session_state.qa_retrieval_applied:
            st.success("✅ Q&A retrieval complete")
            
            # Filters based on Q&A scores
            col1, col2 = st.columns(2)
            with col1:
                qa_filter_threshold = st.slider("Combined Score Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                                               help="Filter to keep only papers above this combined score")
            with col2:
                if st.button("✂️ Apply Q&A Filter"):
                    apply_qa_filter(qa_filter_threshold)
            
            # Visualize Q&A scores
            df_plot = st.session_state.df_valid.copy()
            df_plot['hover_title'] = df_plot['title'].fillna('N/A')
            df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
            
            tab1, tab2, tab3 = st.tabs(["Combined Score", "Reranker Score", "Embedding Score"])
            
            with tab1:
                fig = px.scatter(
                    df_plot,
                    x='umap_x',
                    y='umap_y',
                    color='qa_combined_score',
                    title="Q&A Combined Score",
                    opacity=0.7,
                    height=800,
                    color_continuous_scale='Viridis',
                    hover_data={'umap_x': False, 'umap_y': False, 'qa_combined_score': ':.3f', 
                               'qa_reranker_score': ':.3f', 'qa_embedding_score': ':.3f',
                               'hover_title': True, 'hover_abstract': True}
                )
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = px.scatter(
                    df_plot,
                    x='umap_x',
                    y='umap_y',
                    color='qa_reranker_score',
                    title="Reranker Score (Qwen3-Reranker)",
                    opacity=0.7,
                    height=800,
                    color_continuous_scale='Reds',
                    hover_data={'umap_x': False, 'umap_y': False, 'qa_reranker_score': ':.3f', 
                               'hover_title': True, 'hover_abstract': True}
                )
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = px.scatter(
                    df_plot,
                    x='umap_x',
                    y='umap_y',
                    color='qa_embedding_score',
                    title="Embedding Score (Qwen3-Embedding)",
                    opacity=0.7,
                    height=800,
                    color_continuous_scale='Blues',
                    hover_data={'umap_x': False, 'umap_y': False, 'qa_embedding_score': ':.3f', 
                               'hover_title': True, 'hover_abstract': True}
                )
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Entity-based filter
    st.subheader("3️⃣ Entity-Based Filter")
    
    st.markdown("Filter papers by domain-specific entities (materials, diseases, delivery methods, etc.)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_entity_type = st.selectbox(
            "Entity Type",
            ['materials', 'ligands', 'diseases', 'delivery', 'models']
        )
        
        # Show available entities for this type
        if filter_entity_type == 'materials':
            entity_options = MATERIAL_HINTS
        elif filter_entity_type == 'ligands':
            entity_options = LIGAND_HINTS
        elif filter_entity_type == 'diseases':
            entity_options = DISEASE_HINTS
        elif filter_entity_type == 'delivery':
            entity_options = DELIVERY_HINTS
        else:
            entity_options = MODEL_HINTS
        
        selected_entities = st.multiselect(
            f"Select {filter_entity_type.title()}",
            entity_options,
            help=f"Papers must contain at least one of the selected {filter_entity_type}"
        )
    
    with col2:
        st.write("")
        st.write("")
        if selected_entities and st.button("🔬 Apply Entity Filter"):
            apply_entity_filter(filter_entity_type, selected_entities)
    
    # Show current entity counts if any filters applied
    if selected_entities:
        st.info(f"Filter will keep papers containing: {', '.join(selected_entities)}")


def run_kmeans_filter(n_clusters):
    """Run K-means clustering"""
    with st.spinner(f"Running K-means with {n_clusters} clusters..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(st.session_state.X_pca)
        st.session_state.df_valid['kmeans_cluster'] = labels
        st.session_state.kmeans_applied = True


def apply_cluster_filter(selected_clusters):
    """Filter to selected clusters"""
    save_state_for_undo(f"K-means filter to {len(selected_clusters)} clusters")
    
    mask = st.session_state.df_valid['kmeans_cluster'].isin(selected_clusters)
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask]
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask]
    
    st.success(f"✅ Filtered: {n_before} → {len(st.session_state.df_valid)} papers")
    st.rerun()


def compute_semantic_similarity(query_text, threshold):
    """Compute semantic similarity to query using preloaded embeddings"""
    try:
        import requests
        
        # API endpoint for Qwen embedding service
        QWEN_API_URL = "http://localhost:8000"
        
        with st.spinner("Generating query embedding..."):
            # Generate query embedding with instruction for better retrieval
            query_payload = {
                "texts": [query_text],
                "instruction": "Represent this query for retrieving relevant biomedical research papers",
                "normalize": True
            }
            
            query_response = requests.post(
                f"{QWEN_API_URL}/embed",
                json=query_payload,
                timeout=60
            )
            
            if query_response.status_code != 200:
                st.error(f"❌ Query embedding failed: {query_response.text}")
                return
            
            query_embedding = np.array(query_response.json()['embeddings'][0])
        
        with st.spinner("Computing similarities with preloaded paper embeddings..."):
            # Use the already-loaded embeddings from session state
            paper_embeddings = st.session_state.X_primary
            
            # Normalize embeddings for cosine similarity
            paper_embeddings_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Compute cosine similarity: normalized dot product
            similarities = paper_embeddings_norm @ query_embedding_norm
            
            st.session_state.df_valid['similarity_score'] = similarities
            st.session_state.similarity_applied = True
            
            st.success(f"✅ Computed similarities: {(similarities >= threshold).sum()} papers above threshold {threshold:.2f}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Qwen API. Make sure the service is running at http://localhost:8000")
        st.info("Start the service with: `uvicorn qwen:app --host 0.0.0.0 --port 8000` in the embedding_models directory")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def compute_question_answering_retrieval(question, threshold=0.3):
    """
    Compute question-answering style retrieval using Qwen reranker.
    Uses both embedding similarity and reranker scores for better results.
    """
    try:
        import requests
        
        # API endpoint for Qwen service
        QWEN_API_URL = "http://localhost:8000"
        
        # Get paper texts
        text_col = 'processed_content' if 'processed_content' in st.session_state.df_valid.columns else 'abstract'
        paper_texts = st.session_state.df_valid[text_col].fillna('').astype(str).tolist()
        
        # First pass: Get candidates above threshold using embedding similarity with preloaded embeddings
        with st.spinner("Phase 1/2: Computing embedding similarities with preloaded embeddings..."):
            # Get question embedding from API
            query_payload = {
                "texts": [question],
                "instruction": "Represent this question for retrieving relevant research papers that answer it",
                "normalize": True
            }
            
            query_response = requests.post(
                f"{QWEN_API_URL}/embed",
                json=query_payload,
                timeout=60
            )
            
            if query_response.status_code != 200:
                st.error(f"❌ Query embedding failed: {query_response.text}")
                return
            
            question_embedding = np.array(query_response.json()['embeddings'][0])
            
            # Use preloaded paper embeddings
            paper_embeddings = st.session_state.X_primary
            
            # Normalize and compute similarities
            paper_embeddings_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            question_embedding_norm = question_embedding / np.linalg.norm(question_embedding)
            embedding_scores = paper_embeddings_norm @ question_embedding_norm
            
            # Get candidates above threshold
            above_threshold = embedding_scores >= threshold
            top_candidate_indices = np.where(above_threshold)[0]
            
            # Sort by score descending
            sorted_order = np.argsort(embedding_scores[top_candidate_indices])[::-1]
            top_candidate_indices = top_candidate_indices[sorted_order]
            top_candidate_texts = [paper_texts[i] for i in top_candidate_indices]
            
            st.info(f"Found {len(top_candidate_indices)} papers above threshold {threshold:.2f}")
        
        if len(top_candidate_indices) == 0:
            st.warning(f"❌ No papers found above threshold {threshold:.2f}. Try lowering the threshold.")
            return
        
        # Second pass: Rerank candidates
        with st.spinner(f"Phase 2/2: Reranking {len(top_candidate_texts)} candidates..."):
            rank_payload = {
                "query": question,
                "documents": top_candidate_texts,
                "instruction": "Given a research question, determine if this research paper provides relevant information to answer it",
                "top_k": None,  # Return all
                "return_embedding_similarity": True,
                "normalize_embeddings": True
            }
            
            rank_response = requests.post(
                f"{QWEN_API_URL}/rank",
                json=rank_payload,
                timeout=180
            )
            
            if rank_response.status_code != 200:
                st.error(f"❌ Reranking failed: {rank_response.text}")
                return
            
            rank_results = rank_response.json()['results']
        
        # Map results back to original indices and store scores
        reranker_scores = np.zeros(len(paper_texts))
        combined_scores = np.zeros(len(paper_texts))
        
        for result in rank_results:
            original_idx = top_candidate_indices[result['index']]
            reranker_scores[original_idx] = result['reranker_score']
            # Combine reranker and embedding scores (weighted average)
            if result['embedding_score'] is not None:
                combined_scores[original_idx] = 0.7 * result['reranker_score'] + 0.3 * result['embedding_score']
            else:
                combined_scores[original_idx] = result['reranker_score']
        
        # Store all scores
        st.session_state.df_valid['qa_reranker_score'] = reranker_scores
        st.session_state.df_valid['qa_embedding_score'] = embedding_scores
        st.session_state.df_valid['qa_combined_score'] = combined_scores
        st.session_state.qa_retrieval_applied = True
        
        n_reranked = len(top_candidate_indices)
        n_with_scores = (combined_scores > 0).sum()
        st.success(f"✅ Q&A retrieval complete. Reranked {n_reranked} papers above threshold.")
        
        # Show top results
        with st.expander("📄 Top 10 Results", expanded=True):
            top_papers = st.session_state.df_valid.nlargest(10, 'qa_combined_score')
            for idx, (_, row) in enumerate(top_papers.iterrows(), 1):
                st.write(f"**{idx}. [{row['qa_combined_score']:.3f}]** {row.get('title', 'N/A')}")
                col1, col2, col3 = st.columns(3)
                col1.caption(f"Reranker: {row['qa_reranker_score']:.3f}")
                col2.caption(f"Embedding: {row['qa_embedding_score']:.3f}")
                col3.caption(f"Year: {row.get('publication_year', 'N/A')}")
                if idx < 5:  # Show abstract for top 5
                    st.caption(str(row.get('abstract', row.get('processed_content', '')))[:300] + '...')
                st.divider()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Qwen API. Make sure the service is running at http://localhost:8000")
        st.info("Start the service with: `uvicorn qwen:app --host 0.0.0.0 --port 8000` in the embedding_models directory")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def apply_similarity_filter(threshold):
    """Filter by similarity threshold"""
    save_state_for_undo(f"Similarity filter (threshold={threshold:.2f})")
    
    mask = st.session_state.df_valid['similarity_score'] >= threshold
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask]
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask]
    
    st.success(f"✅ Filtered: {n_before} → {len(st.session_state.df_valid)} papers")
    st.rerun()


def apply_qa_filter(threshold):
    """Filter by Q&A combined score threshold"""
    save_state_for_undo(f"Q&A filter (threshold={threshold:.2f})")
    
    mask = st.session_state.df_valid['qa_combined_score'] >= threshold
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask]
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask]
    
    st.success(f"✅ Filtered: {n_before} → {len(st.session_state.df_valid)} papers")
    st.rerun()


def apply_entity_filter(entity_type, selected_entities):
    """Filter papers by entity presence"""
    with st.spinner("Filtering by entities..."):
        # Extract entities for all papers
        text_col = 'processed_content' if 'processed_content' in st.session_state.df_valid.columns else 'abstract'
        
        # Check which papers contain at least one of the selected entities
        mask = []
        for idx, row in st.session_state.df_valid.iterrows():
            text = str(row.get(text_col) or row.get('content') or '').lower()
            has_entity = any(entity.lower() in text for entity in selected_entities)
            mask.append(has_entity)
        
        mask = np.array(mask)
        n_before = len(st.session_state.df_valid)
        
        if mask.sum() == 0:
            st.error("❌ No papers contain the selected entities")
            return
        
        save_state_for_undo(f"Entity filter: {entity_type}")
        
        st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
        st.session_state.X_pca = st.session_state.X_pca[mask]
        st.session_state.X_primary = st.session_state.X_primary[mask]
        st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask]
        
        st.success(f"✅ Filtered by {entity_type}: {n_before} → {len(st.session_state.df_valid)} papers")
        st.rerun()


# ============================================================================
# PAGE: DENSITY & GAP ANALYSIS
# ============================================================================

def page_gap_analysis():
    """Compute density features and identify gaps"""
    st.title("🔍 Gap Analysis")
    
    if not st.session_state.clustering_done:
        st.warning("⚠️ Please complete clustering first")
        return
    
    if st.session_state.selected_clustering is None:
        st.warning("⚠️ Please select a clustering method to continue (HDBSCAN or Community Detection)")
        return
    
    # Display selected clustering method
    clustering_method_map = {
        'kmeans': 'K-means',
        'hdbscan': 'HDBSCAN',
        'leiden': 'Community Detection (Leiden/Louvain)'
    }
    clustering_method_display = clustering_method_map.get(st.session_state.selected_clustering, 'Unknown')
    st.info(f"📌 Using clustering method: **{clustering_method_display}**")
    
    st.markdown(f"""
    **Working Dataset**: {len(st.session_state.df_valid)} papers  
    Configure density and gap detection parameters.
    """)
    
    # Analysis Parameters Configuration
    st.subheader("🔧 Analysis Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        knn_graph_k = st.number_input(
            "k-NN Graph K", 
            min_value=5, 
            max_value=50, 
            value=21,
            help="Number of neighbors for k-NN graph construction"
        )
        
        k_neighbors = st.multiselect(
            "K-Neighbors for Density",
            [5, 10, 15, 20, 30, 40, 50],
            default=[10, 20, 30, 50],
            help="Multiple k values for robust density estimation"
        )
    
    with col2:
        density_metric = st.selectbox(
            "Density Metric",
            ["cosine", "euclidean", "manhattan"],
            index=0,
            help="Distance metric for density computation"
        )
        
        gap_quantile = st.slider(
            "Gap Quantile (top %)", 
            min_value=0.90, 
            max_value=0.999, 
            value=0.95, 
            step=0.001,
            help="Percentile threshold for gap candidates"
        )
    
    with col3:
        min_gap_region_size = st.number_input(
            "Min Gap Region Size", 
            min_value=2, 
            max_value=20, 
            value=3,
            help="Minimum papers needed to form a gap region"
        )
    
    # Store gap analysis config
    gap_config = {
        'knn_graph_k': knn_graph_k,
        'k_neighbors': sorted(k_neighbors),
        'density_metric': density_metric,
        'gap_quantile': gap_quantile,
        'min_gap_region_size': min_gap_region_size
    }
    
    # Store in session state for use in later pages
    st.session_state.gap_config = gap_config
    
    st.divider()
    
    # Build k-NN graph
    if st.session_state.G is None:
        if st.button("🕸️ Build k-NN Graph", type="primary"):
            with st.spinner("Building k-NN graph..."):
                G = build_knn_graph(st.session_state.X_pca, gap_config['knn_graph_k'], 'cosine')
                st.session_state.G = G
                st.success(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                st.rerun()
        return
    
    st.success(f"✅ k-NN graph built: {st.session_state.G.number_of_nodes()} nodes, {st.session_state.G.number_of_edges()} edges")
    
    # Visualize k-NN graph
    with st.expander("📊 k-NN Graph Visualization", expanded=False):
        st.markdown("""
        Visualizing the k-NN graph structure overlaid on the UMAP projection.
        Each point represents a paper, and edges connect k-nearest neighbors.
        """)
        
        # Prepare plot data
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        # Create figure with nodes
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            title=f"k-NN Graph (k={gap_config['knn_graph_k']})",
            opacity=0.6,
            height=1000,
            hover_data={'umap_x': False, 'umap_y': False, 'hover_title': True, 'hover_abstract': True}
        )
        fig.update_traces(marker=dict(size=5, color='lightblue'))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        
        # Add edges (sample for performance if too many)
        G = st.session_state.G
        edges = list(G.edges())
        max_edges_to_plot = 5000
        
        if len(edges) > max_edges_to_plot:
            st.info(f"Sampling {max_edges_to_plot} edges out of {len(edges)} for visualization performance")
            edge_indices = np.random.choice(len(edges), max_edges_to_plot, replace=False)
            edges_to_plot = [edges[i] for i in edge_indices]
        else:
            edges_to_plot = edges
        
        # Draw edges
        edge_x = []
        edge_y = []
        for u, v in edges_to_plot:
            edge_x.extend([df_plot.iloc[u]['umap_x'], df_plot.iloc[v]['umap_x'], None])
            edge_y.extend([df_plot.iloc[u]['umap_y'], df_plot.iloc[v]['umap_y'], None])
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='rgba(125, 125, 125, 0.2)'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Re-add nodes on top
        fig.add_trace(go.Scatter(
            x=df_plot['umap_x'],
            y=df_plot['umap_y'],
            mode='markers',
            marker=dict(size=5, color='lightblue'),
            text=[f"<b>{row['hover_title']}</b><br>{row['hover_abstract']}" for _, row in df_plot.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", G.number_of_nodes())
        col2.metric("Edges", G.number_of_edges())
        avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        col3.metric("Avg Degree", f"{avg_degree:.1f}")
        col4.metric("Components", nx.number_connected_components(G))
    
    st.divider()
    
    # Compute density
    if not st.session_state.density_computed:
        if st.button("📊 Compute Density Features", type="primary"):
            with st.spinner("Computing density features..."):
                density_df = compute_density_features(
                    st.session_state.X_pca,
                    gap_config['k_neighbors'],
                    gap_config['density_metric']
                )
                
                for col in density_df.columns:
                    st.session_state.df_valid[col] = density_df[col].values
                
                st.session_state.density_computed = True
                st.success("✅ Density features computed")
                st.rerun()
        return
    
    st.success("✅ Density features computed")
    
    # Gap statistics
    gap_scores = st.session_state.df_valid['gap_score']
    gap_threshold = gap_scores.quantile(gap_config['gap_quantile'])
    is_gap = gap_scores >= gap_threshold
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Gap Score", f"{gap_scores.mean():.3f}")
    col2.metric("Std Gap Score", f"{gap_scores.std():.3f}")
    col3.metric("Gap Threshold", f"{gap_threshold:.3f}")
    col4.metric("Gap Candidates", f"{is_gap.sum()}")
    
    # Visualizations
    st.subheader("📊 Gap Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=gap_scores, nbinsx=50, name='Gap Scores'))
        fig.add_vline(x=gap_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold ({gap_config['gap_quantile']:.0%})")
        fig.update_layout(title="Distribution of Gap Scores", xaxis_title="Gap Score", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            color='gap_score',
            title="Gap Scores in Embedding Space",
            color_continuous_scale='Viridis',
            opacity=0.7,
            hover_data={'umap_x': False, 'umap_y': False, 'gap_score': ':.3f', 'hover_title': True, 'hover_abstract': True}
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
    
    # Binary gap view
    st.subheader("🎯 Gap Candidates")
    
    df_plot = st.session_state.df_valid.copy()
    df_plot['is_gap'] = is_gap
    df_plot['hover_title'] = df_plot['title'].fillna('N/A')
    df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
    
    fig = px.scatter(
        df_plot,
        x='umap_x',
        y='umap_y',
        color='is_gap',
        title=f"Gap Candidates (top {int((1-gap_config['gap_quantile'])*100)}%)",
        color_discrete_map={True: 'red', False: 'lightgray'},
        opacity=0.7,
        height=1000,
        hover_data={'umap_x': False, 'umap_y': False, 'is_gap': True, 'gap_score': ':.3f', 'hover_title': True, 'hover_abstract': True}
    )
    fig.update_traces(marker=dict(size=8), selector=dict(name='True'))
    fig.update_traces(marker=dict(size=4), selector=dict(name='False'))
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: CLUSTERING
# ============================================================================

def page_clustering():
    """Run clustering algorithms"""
    st.title("🎯 Clustering Analysis")
    
    if st.session_state.X_pca is None:
        st.warning("⚠️ Please complete embedding processing first")
        return
    
    if st.session_state.G is None:
        st.warning("⚠️ Please complete gap analysis (k-NN graph construction) first")
        return
    
    st.markdown(f"""
    **Working Dataset**: {len(st.session_state.df_valid)} papers  
    **Graph**: {st.session_state.G.number_of_nodes()} nodes, {st.session_state.G.number_of_edges()} edges
    
    Configure and run clustering algorithms to identify research communities.
    """)
    
    # Clustering configuration
    st.subheader("🔧 Clustering Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        hdbscan_min_cluster = st.number_input("HDBSCAN Min Cluster Size", min_value=5, max_value=100, value=5,
                                             help="Minimum number of papers per cluster")
        hdbscan_min_samples = st.number_input("HDBSCAN Min Samples", min_value=1, max_value=50, value=10,
                                             help="Minimum samples in neighborhood")
    
    with col2:
        leiden_resolution = st.slider("Leiden Resolution", min_value=0.1, max_value=5.0, value=1.0, step=0.01,
                                     help="Higher values create more communities")
    
    # Store clustering config in session state
    st.session_state.clustering_config = {
        'hdbscan_min_cluster_size': hdbscan_min_cluster,
        'hdbscan_min_samples': hdbscan_min_samples,
        'leiden_resolution': leiden_resolution
    }
    
    clustering_config = st.session_state.clustering_config
    
    st.divider()
    
    # K-means clustering
    st.subheader("🔹 K-means Clustering")
    
    if 'cluster_kmeans' not in st.session_state.df_valid.columns:
        col1, col2 = st.columns([2, 1])
        with col1:
            kmeans_n_clusters_main = st.number_input("Number of Clusters", min_value=5, max_value=100, value=20, key="kmeans_main")
        with col2:
            st.write("")
            st.write("")
            if st.button("▶️ Run K-means"):
                save_state_for_undo("K-means Clustering")
                with st.spinner(f"Running K-means with {kmeans_n_clusters_main} clusters..."):
                    kmeans = KMeans(n_clusters=kmeans_n_clusters_main, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(st.session_state.X_pca)
                    st.session_state.df_valid['cluster_kmeans'] = labels
                    st.success(f"✅ K-means: {kmeans_n_clusters_main} clusters")
                    st.rerun()
    else:
        labels = st.session_state.df_valid['cluster_kmeans'].values
        n_clusters = len(set(labels))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters", n_clusters)
        with col2:
            if st.button("↩️ Undo K-means"):
                if undo_last_action():
                    st.success("✅ Undone!")
                    st.rerun()
                else:
                    st.warning("⚠️ No actions to undo")
        
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='cluster_kmeans',
            title=f"K-means Clusters (n={n_clusters})",
            color_continuous_scale='rainbow',
            opacity=0.7,
            height=1000,
            hover_data={'title': True, 'abstract': True, 'cluster_kmeans': True}
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
        
        # Selection button
        if st.session_state.selected_clustering == 'kmeans':
            st.success("✅ K-means selected for gap analysis")
        else:
            if st.button("✔️ Use K-means for Gap Analysis", type="primary"):
                st.session_state.selected_clustering = 'kmeans'
                st.session_state.df_valid['cluster_selected'] = st.session_state.df_valid['cluster_kmeans']
                st.success("✅ K-means clustering selected!")
                st.rerun()
    
    st.divider()
    
    # HDBSCAN
    st.subheader("🔹 HDBSCAN Clustering")
    
    if 'cluster_hdbscan' not in st.session_state.df_valid.columns:
        if st.button("▶️ Run HDBSCAN"):
            save_state_for_undo("HDBSCAN Clustering")
            with st.spinner("Running HDBSCAN..."):
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=clustering_config['hdbscan_min_cluster_size'],
                    min_samples=clustering_config['hdbscan_min_samples'],
                    metric='euclidean'
                )
                labels = clusterer.fit_predict(st.session_state.X_pca)
                st.session_state.df_valid['cluster_hdbscan'] = labels
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = np.sum(labels == -1)
                
                st.success(f"✅ HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
                st.rerun()
    else:
        labels = st.session_state.df_valid['cluster_hdbscan'].values
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clusters", n_clusters)
        col2.metric("Noise Points", n_noise)
        col3.metric("Noise %", f"{100*n_noise/len(labels):.1f}%")
        
        # Undo button
        with col4:
            if st.button("↩️ Undo HDBSCAN"):
                if undo_last_action():
                    st.success("✅ Undone!")
                    st.rerun()
                else:
                    st.warning("⚠️ No actions to undo")
        
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='cluster_hdbscan',
            title=f"HDBSCAN Clusters (n={n_clusters})",
            color_continuous_scale='rainbow',
            opacity=0.7,
            height=1000,
            hover_data={'title': True, 'abstract': True, 'cluster_hdbscan': True}
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
        
        # Selection button
        if st.session_state.selected_clustering == 'hdbscan':
            st.success("✅ HDBSCAN selected for gap analysis")
        else:
            if st.button("✔️ Use HDBSCAN for Gap Analysis", type="primary"):
                st.session_state.selected_clustering = 'hdbscan'
                st.session_state.df_valid['cluster_selected'] = st.session_state.df_valid['cluster_hdbscan']
                st.success("✅ HDBSCAN clustering selected!")
                st.rerun()
    
    st.divider()
    
    # Leiden/Louvain
    st.subheader("🔹 Community Detection")
    
    st.markdown("""
    Community detection algorithms require a k-NN graph to identify clusters based on network structure.
    The graph will be built first, then communities will be detected.
    """)
    
    # Build k-NN graph for community detection
    if st.session_state.G is None:
        if st.button("🕸️ Build k-NN Graph for Community Detection", type="primary"):
            with st.spinner("Building k-NN graph..."):
                G = build_knn_graph(st.session_state.X_pca, clustering_config['knn_graph_k'], 'cosine')
                st.session_state.G = G
                st.success(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                st.rerun()
        return
    
    st.success(f"✅ k-NN graph built: {st.session_state.G.number_of_nodes()} nodes, {st.session_state.G.number_of_edges()} edges")
    
    # Run community detection
    if 'cluster_leiden' not in st.session_state.df_valid.columns:
        if LEIDEN_AVAILABLE:
            if st.button("▶️ Run Leiden Community Detection"):
                run_leiden_clustering(clustering_config)
        elif LOUVAIN_AVAILABLE:
            if st.button("▶️ Run Louvain Community Detection"):
                run_louvain_clustering()
        else:
            st.warning("⚠️ No community detection algorithm available")
    else:
        labels = st.session_state.df_valid['cluster_leiden'].values
        n_communities = len(set(labels))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Communities", n_communities)
        with col2:
            if st.button("↩️ Undo Community Detection"):
                if undo_last_action():
                    st.success("✅ Undone!")
                    st.rerun()
                else:
                    st.warning("⚠️ No actions to undo")
        
        # Two-column layout: k-NN graph + communities
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**k-NN Graph Structure**")
            
            # Prepare plot data
            df_plot = st.session_state.df_valid.copy()
            df_plot['hover_title'] = df_plot['title'].fillna('N/A')
            df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
            
            # Create figure with nodes
            fig_graph = px.scatter(
                df_plot,
                x='umap_x',
                y='umap_y',
                title=f"k-NN Graph (k={clustering_config['knn_graph_k']})",
                opacity=0.6,
                height=500,
                hover_data={'umap_x': False, 'umap_y': False, 'hover_title': True, 'hover_abstract': True}
            )
            fig_graph.update_traces(marker=dict(size=5, color='lightblue'))
            fig_graph.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
            
            # Add edges (sample for performance if too many)
            G = st.session_state.G
            edges = list(G.edges())
            max_edges_to_plot = 3000
            
            if len(edges) > max_edges_to_plot:
                edge_indices = np.random.choice(len(edges), max_edges_to_plot, replace=False)
                edges_to_plot = [edges[i] for i in edge_indices]
            else:
                edges_to_plot = edges
            
            # Draw edges
            edge_x = []
            edge_y = []
            for u, v in edges_to_plot:
                edge_x.extend([df_plot.iloc[u]['umap_x'], df_plot.iloc[v]['umap_x'], None])
                edge_y.extend([df_plot.iloc[u]['umap_y'], df_plot.iloc[v]['umap_y'], None])
            
            fig_graph.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='rgba(125, 125, 125, 0.2)'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Re-add nodes on top
            fig_graph.add_trace(go.Scatter(
                x=df_plot['umap_x'],
                y=df_plot['umap_y'],
                mode='markers',
                marker=dict(size=5, color='lightblue'),
                text=[f"<b>{row['hover_title']}</b><br>{row['hover_abstract']}" for _, row in df_plot.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ))
            
            fig_graph.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
            st.plotly_chart(fig_graph, use_container_width=True)
            
            # Graph statistics
            subcol1, subcol2 = st.columns(2)
            subcol1.metric("Nodes", G.number_of_nodes())
            subcol2.metric("Edges", G.number_of_edges())
            avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
            subcol1.metric("Avg Degree", f"{avg_degree:.1f}")
            subcol2.metric("Components", nx.number_connected_components(G))
        
        with col2:
            st.markdown("**Detected Communities**")
            
            fig = px.scatter(
                st.session_state.df_valid,
                x='umap_x',
                y='umap_y',
                color='cluster_leiden',
                title=f"Community Detection (n={n_communities})",
                color_continuous_scale='rainbow',
                opacity=0.7,
                height=500
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Community size distribution
            unique, counts = np.unique(labels, return_counts=True)
            st.bar_chart({f"C{c}": cnt for c, cnt in zip(unique[:10], counts[:10])})
        
        # Selection button
        if st.session_state.selected_clustering == 'leiden':
            st.success("✅ Community Detection selected for gap analysis")
        else:
            if st.button("✔️ Use Community Detection for Gap Analysis", type="primary"):
                st.session_state.selected_clustering = 'leiden'
                st.session_state.df_valid['cluster_selected'] = st.session_state.df_valid['cluster_leiden']
                st.success("✅ Community Detection clustering selected!")
                st.rerun()
    
    if 'cluster_kmeans' in st.session_state.df_valid.columns or 'cluster_hdbscan' in st.session_state.df_valid.columns or 'cluster_leiden' in st.session_state.df_valid.columns:
        st.session_state.clustering_done = True


def run_leiden_clustering(clustering_config):
    """Run Leiden algorithm"""
    save_state_for_undo("Leiden Clustering")
    with st.spinner("Running Leiden algorithm..."):
        G = st.session_state.G
        mapping = {n: i for i, n in enumerate(G.nodes())}
        edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        
        ig_graph = ig.Graph(n=len(mapping), edges=edges)
        ig_graph.es['weight'] = weights
        
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=clustering_config['leiden_resolution'],
            seed=42
        )
        
        labels = np.zeros(len(G), dtype=int)
        for cluster_id, community in enumerate(partition):
            labels[list(community)] = cluster_id
        
        st.session_state.df_valid['cluster_leiden'] = labels
        st.success(f"✅ Leiden: {len(set(labels))} communities")
        st.rerun()


def run_louvain_clustering():
    """Run Louvain algorithm"""
    save_state_for_undo("Louvain Clustering")
    with st.spinner("Running Louvain algorithm..."):
        G = st.session_state.G
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        labels = np.array([partition[i] for i in range(len(G))], dtype=int)
        
        st.session_state.df_valid['cluster_leiden'] = labels
        st.success(f"✅ Louvain: {len(set(labels))} communities")
        st.rerun()


# ============================================================================
# PAGE: GAP REGIONS
# ============================================================================

def page_gap_regions():
    """Identify and explore gap regions"""
    st.title("🌉 Gap Regions")
    
    if not st.session_state.density_computed:
        st.warning("⚠️ Please complete gap analysis first")
        return
    
    gap_config = st.session_state.get('gap_config', {
        'gap_quantile': 0.95,
        'min_gap_region_size': 3
    })
    
    if not st.session_state.gaps_identified:
        if st.button("🔍 Identify Gap Regions", type="primary"):
            identify_gap_regions(gap_config)
        return
    
    gap_regions = st.session_state.gap_regions
    
    st.success(f"✅ Identified {len(gap_regions)} gap regions")
    
    # Summary metrics
    if gap_regions:
        region_sizes = [len(r) for r in gap_regions]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Gap Regions", len(gap_regions))
        col2.metric("Avg Region Size", f"{np.mean(region_sizes):.1f}")
        col3.metric("Largest Region", max(region_sizes))
    
    # Visualizations
    st.subheader("📊 Gap Regions Visualization")
    
    # Create region labels
    region_labels = np.full(len(st.session_state.df_valid), -1, dtype=int)
    for region_id, region_indices in enumerate(gap_regions):
        region_labels[region_indices] = region_id
    
    st.session_state.df_valid['gap_region'] = region_labels
    
    # Multi-panel visualization
    tab1, tab2, tab3 = st.tabs(["📍 All Regions", "📊 By Score", "🔄 Over Clusters"])
    
    with tab1:
        df_plot = st.session_state.df_valid.copy()
        df_plot['gap_region_str'] = df_plot['gap_region'].astype(str)
        df_plot['is_gap_region'] = df_plot['gap_region'] >= 0
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            color='is_gap_region',
            title=f"Gap Regions (n={len(gap_regions)})",
            color_discrete_map={True: 'red', False: 'lightgray'},
            opacity=0.7,
            height=1000,
            hover_data={'umap_x': False, 'umap_y': False, 'is_gap_region': True, 'gap_region': True, 'hover_title': True, 'hover_abstract': True}
        )
        fig.update_traces(marker=dict(size=10), selector=dict(name='True'))
        fig.update_traces(marker=dict(size=4), selector=dict(name='False'))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df_gap = st.session_state.df_valid[st.session_state.df_valid['gap_region'] >= 0]
        
        if len(df_gap) > 0:
            df_plot = df_gap.copy()
            df_plot['hover_title'] = df_plot['title'].fillna('N/A')
            df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
            
            fig = px.scatter(
                df_plot,
                x='umap_x',
                y='umap_y',
                color='gap_score',
                title="Gap Regions by Score",
                color_continuous_scale='Reds',
                opacity=0.8,
                height=1000,
                hover_data={'umap_x': False, 'umap_y': False, 'gap_score': ':.3f', 'gap_region': True, 'hover_title': True, 'hover_abstract': True}
            )
            fig.update_traces(marker=dict(size=10))
            fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        # Determine which clustering to display based on selected method
        selected_method = st.session_state.get('selected_clustering', 'hdbscan')
        clustering_method_map = {
            'kmeans': ('cluster_kmeans', 'K-means'),
            'hdbscan': ('cluster_hdbscan', 'HDBSCAN'),
            'leiden': ('cluster_leiden', 'Community Detection')
        }
        
        cluster_col, method_name = clustering_method_map.get(selected_method, ('cluster_hdbscan', 'HDBSCAN'))
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            color=cluster_col,
            title=f"Gap Regions over {method_name} Clusters",
            color_continuous_scale='rainbow',
            opacity=0.3,
            height=1000,
            hover_data={'umap_x': False, 'umap_y': False, cluster_col: True, 'gap_region': True, 'hover_title': True, 'hover_abstract': True}
        )
        
        df_gap = st.session_state.df_valid[st.session_state.df_valid['gap_region'] >= 0]
        if len(df_gap) > 0:
            hover_text = [
                f"<b>{row['title']}</b><br>" +
                f"Gap Region: {row['gap_region']}<br>" +
                f"Gap Score: {row.get('gap_score', 0):.3f}<br>" +
                f"{str(row.get('abstract', row.get('processed_content', '')))[:200]}..."
                for _, row in df_gap.iterrows()
            ]
            
            fig.add_trace(go.Scatter(
                x=df_gap['umap_x'],
                y=df_gap['umap_y'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star', line=dict(color='darkred', width=1)),
                name='Gap Regions',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Region explorer
    st.subheader("🔎 Explore Gap Regions")
    
    if gap_regions:
        region_id = st.selectbox("Select Region", range(len(gap_regions)))
        
        if region_id is not None:
            display_gap_region_details(region_id, gap_regions)


def identify_gap_regions(config):
    """Identify gap regions from gap candidates"""
    with st.spinner("Identifying gap regions..."):
        gap_threshold = st.session_state.df_valid['gap_score'].quantile(config['gap_quantile'])
        gap_candidates_idx = st.session_state.df_valid[
            st.session_state.df_valid['gap_score'] >= gap_threshold
        ].index.tolist()
        
        # Create subgraph
        G = st.session_state.G
        gap_subgraph = G.subgraph(gap_candidates_idx).copy()
        
        # Find connected components
        gap_regions = [list(component) for component in nx.connected_components(gap_subgraph)]
        gap_regions = [r for r in gap_regions if len(r) >= config['min_gap_region_size']]
        gap_regions.sort(key=len, reverse=True)
        
        st.session_state.gap_regions = gap_regions
        st.session_state.gaps_identified = True
        
        st.rerun()


def display_gap_region_details(region_id, gap_regions):
    """Display detailed information about a gap region"""
    region_indices = gap_regions[region_id]
    region_df = st.session_state.df_valid.loc[region_indices]
    
    st.markdown(f"### Region {region_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers", len(region_indices))
    col2.metric("Avg Gap Score", f"{region_df['gap_score'].mean():.3f}")
    col3.metric("Max Gap Score", f"{region_df['gap_score'].max():.3f}")
    
    if 'cluster_hdbscan' in region_df.columns:
        col4.metric("Clusters Spanned", region_df['cluster_hdbscan'].nunique())
    
    # Entity Analysis
    st.markdown("#### 🧬 Domain Entity Analysis")
    
    # Extract entities if not already done
    if not any(col.startswith('entities_') for col in region_df.columns):
        with st.spinner("Extracting domain entities..."):
            region_df = extract_entities_from_dataframe(region_df, text_col='processed_content')
    
    entity_summary = summarize_gap_region_entities(region_df, region_indices)
    
    # Display entity distributions
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**Materials**")
        materials = entity_summary.get('materials', {})
        st.metric("Unique", materials.get('total_unique', 0))
        if materials.get('top_5'):
            for mat, count in materials['top_5'][:3]:
                st.caption(f"{mat}: {count}")
    
    with col2:
        st.markdown("**Ligands**")
        ligands = entity_summary.get('ligands', {})
        st.metric("Unique", ligands.get('total_unique', 0))
        if ligands.get('top_5'):
            for lig, count in ligands['top_5'][:3]:
                st.caption(f"{lig}: {count}")
    
    with col3:
        st.markdown("**Diseases**")
        diseases = entity_summary.get('diseases', {})
        st.metric("Unique", diseases.get('total_unique', 0))
        if diseases.get('top_5'):
            for dis, count in diseases['top_5'][:3]:
                st.caption(f"{dis}: {count}")
    
    with col4:
        st.markdown("**Delivery**")
        delivery = entity_summary.get('delivery', {})
        st.metric("Unique", delivery.get('total_unique', 0))
        if delivery.get('top_5'):
            for del_method, count in delivery['top_5'][:3]:
                st.caption(f"{del_method}: {count}")
    
    with col5:
        st.markdown("**Models**")
        models = entity_summary.get('models', {})
        st.metric("Unique", models.get('total_unique', 0))
        if models.get('top_5'):
            for mod, count in models['top_5'][:3]:
                st.caption(f"{mod}: {count}")
    
    # Detailed entity view
    with st.expander("📋 View All Entities", expanded=False):
        for entity_type, data in entity_summary.items():
            if data.get('top_5'):
                st.markdown(f"**{entity_type.title()}**")
                entity_df = pd.DataFrame(data['top_5'], columns=['Entity', 'Count'])
                st.dataframe(entity_df, use_container_width=True)
    
    # Temporal distribution
    if 'publication_year' in region_df.columns:
        st.markdown("#### 📅 Temporal Distribution")
        years = pd.to_numeric(region_df['publication_year'], errors='coerce').dropna()
        if len(years) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Year Range**: {int(years.min())} - {int(years.max())}")
                st.write(f"**Median Year**: {int(years.median())}")
            with col2:
                year_counts = years.value_counts().sort_index()
                st.bar_chart(year_counts)
    
    # Top papers
    st.markdown("#### 📄 Top Papers by Gap Score")
    top_papers = region_df.nlargest(5, 'gap_score')
    
    for idx, (_, row) in enumerate(top_papers.iterrows(), 1):
        with st.expander(f"{idx}. [{row['gap_score']:.3f}] {row.get('title', 'N/A')}"):
            st.write(f"**Year**: {row.get('publication_year', 'N/A')}")
            st.write(f"**Journal**: {row.get('journal', 'N/A')}")
            if 'abstract' in row and pd.notna(row['abstract']):
                st.write(f"**Abstract**: {row['abstract'][:500]}...")


# ============================================================================
# PAGE: LLM ANALYSIS
# ============================================================================

def page_llm_analysis():
    """LLM-based gap explanation with contrastive cluster analysis"""
    st.title("🤖 LLM Gap Analysis")
    
    if not st.session_state.gaps_identified:
        st.warning("⚠️ Please identify gap regions first")
        return
    
    if not OPENAI_AVAILABLE:
        st.error("❌ OpenAI package not available")
        return
    
    gap_regions = st.session_state.gap_regions
    
    if not gap_regions:
        st.warning("No gap regions found")
        return
    
    st.markdown("""
    Generate evidence-grounded contrastive explanations for gap regions by comparing neighboring clusters.
    The LLM identifies key differences, salient entities, and potential bridge opportunities.
    """)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                       value=os.environ.get('OPENAI_API_KEY', ''))
        openai_model = st.selectbox("Model", ["gpt-5-mini", "gpt-5", "gpt-5-nano"], index=0)
    with col2:
        region_id = st.selectbox("Select Gap Region", range(len(gap_regions)))
        n_papers_per_cluster = st.number_input("Papers per Cluster", min_value=5, max_value=30, value=15)
    with col3:
        show_prompt_editor = st.checkbox("Show/Edit Prompt", value=False, 
                                         help="Display and edit the full prompt before sending")
    
    # Custom question and keywords
    st.markdown("---")
    st.subheader("💡 Additional Guidance (Optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        custom_question = st.text_area(
            "Specific Question",
            value="",
            height=100,
            help="Optional: Ask a specific question that the LLM should try to answer based on the evidence"
        )
    with col2:
        guidance_keywords = st.text_area(
            "Keywords for Bridge Opportunities",
            value="",
            height=100,
            help="Optional: Enter keywords (comma-separated) to guide the LLM when identifying bridge opportunities"
        )
    
    # Show region preview
    region_indices = gap_regions[region_id]
    region_df = st.session_state.df_valid.loc[region_indices]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Region Papers", len(region_indices))
    if 'gap_score' in region_df.columns:
        col2.metric("Avg Gap Score", f"{region_df['gap_score'].mean():.3f}")
    if 'cluster_hdbscan' in region_df.columns:
        col3.metric("Clusters Spanned", region_df['cluster_hdbscan'].nunique())
    
    # Cluster selection if region spans multiple clusters
    cluster_A_selected = None
    cluster_B_selected = None
    
    if 'cluster_hdbscan' in region_df.columns:
        cluster_counts = Counter(region_df['cluster_hdbscan'].tolist())
        unique_clusters = [c for c, _ in cluster_counts.most_common()]
        
        if len(unique_clusters) >= 2:
            st.markdown("---")
            st.subheader("🎯 Select Clusters to Contrast")
            
            col1, col2 = st.columns(2)
            with col1:
                cluster_A_selected = st.selectbox(
                    "Cluster A",
                    unique_clusters,
                    index=0,
                    format_func=lambda x: f"Cluster {x} (n={cluster_counts[x]} papers)",
                    key="cluster_a_select"
                )
            with col2:
                # Filter out Cluster A from options for Cluster B
                cluster_b_options = [c for c in unique_clusters if c != cluster_A_selected]
                cluster_B_selected = st.selectbox(
                    "Cluster B",
                    cluster_b_options,
                    index=0,
                    format_func=lambda x: f"Cluster {x} (n={cluster_counts[x]} papers)",
                    key="cluster_b_select"
                )
            
            st.info(f"Will compare Cluster {cluster_A_selected} ({cluster_counts[cluster_A_selected]} papers) vs Cluster {cluster_B_selected} ({cluster_counts[cluster_B_selected]} papers)")
        elif len(unique_clusters) == 1:
            st.warning("⚠️ This region only spans one cluster - contrastive analysis requires at least 2 clusters")
    
    if st.button("🚀 Generate Contrastive Explanation", type="primary"):
        if not openai_api_key:
            st.error("Please provide OpenAI API key")
            return
        
        generate_llm_explanation(
            region_id, 
            openai_api_key, 
            openai_model, 
            n_papers_per_cluster, 
            True, 
            show_prompt_editor,
            custom_question.strip() if custom_question.strip() else None,
            guidance_keywords.strip() if guidance_keywords.strip() else None,
            cluster_A_selected,
            cluster_B_selected
        )


def generate_llm_explanation(region_id, api_key, model, n_papers, show_viz, show_prompt_editor, custom_question=None, guidance_keywords=None, cluster_A_override=None, cluster_B_override=None):
    """Generate evidence-grounded LLM explanation for gap region using contrastive analysis"""
    gap_regions = st.session_state.gap_regions
    region_indices = gap_regions[region_id]
    region_df = st.session_state.df_valid.loc[region_indices]
    
    # Determine clusters to compare
    cluster_counts = Counter(region_df['cluster_hdbscan'].tolist())
    if len(cluster_counts) < 2:
        st.warning("⚠️ Region doesn't span multiple clusters - need at least 2 clusters for contrastive analysis")
        return
    
    # Use override clusters if provided, otherwise use two most common
    if cluster_A_override is not None and cluster_B_override is not None:
        cluster_A, cluster_B = cluster_A_override, cluster_B_override
    else:
        cluster_A, cluster_B = [c for c, _ in cluster_counts.most_common(2)]
    
    st.markdown(f"### 🔍 Analyzing Region {region_id}")
    st.markdown(f"**Cluster A**: {cluster_A} (n={cluster_counts[cluster_A]}) | **Cluster B**: {cluster_B} (n={cluster_counts[cluster_B]})")
    
    # Visualization of clusters and gap region
    if show_viz and st.session_state.X_umap_2d is not None:
        st.markdown("#### 📊 Cluster Visualization with Gap Region")
        
        fig = go.Figure()
        
        # All points colored by cluster (background)
        df_plot = st.session_state.df_valid.copy()
        df_plot['umap_x'] = st.session_state.X_umap_2d[:, 0]
        df_plot['umap_y'] = st.session_state.X_umap_2d[:, 1]
        
        # Plot all clusters in background
        for cluster_id in df_plot['cluster_hdbscan'].unique():
            if cluster_id == -1:
                continue
            cluster_mask = df_plot['cluster_hdbscan'] == cluster_id
            cluster_data = df_plot[cluster_mask]
            
            # Highlight the two clusters being compared
            if cluster_id == cluster_A:
                color = 'blue'
                size = 10
                opacity = 0.6
                name = f'Cluster {cluster_id} (A)'
            elif cluster_id == cluster_B:
                color = 'green'
                size = 10
                opacity = 0.6
                name = f'Cluster {cluster_id} (B)'
            else:
                color = 'lightgray'
                size = 5
                opacity = 0.2
                name = f'Cluster {cluster_id}'
            
            # Create hover text
            hover_text = [
                f"<b>{row.get('title', 'N/A')}</b><br>" +
                f"Cluster: {cluster_id}<br>" +
                f"{str(row.get('abstract', row.get('processed_content', '')))[:200]}..."
                for _, row in cluster_data.iterrows()
            ]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['umap_x'],
                y=cluster_data['umap_y'],
                mode='markers',
                marker=dict(size=size, color=color, opacity=opacity),
                name=name,
                showlegend=True,
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Overlay gap region as red stars
        gap_data = df_plot.loc[region_indices]
        gap_hover_text = [
            f"<b>{row.get('title', 'N/A')}</b><br>" +
            f"Gap Region: {region_id}<br>" +
            f"Gap Score: {row.get('gap_score', 0):.3f}<br>" +
            f"{str(row.get('abstract', row.get('processed_content', '')))[:200]}..."
            for _, row in gap_data.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=gap_data['umap_x'],
            y=gap_data['umap_y'],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star', 
                       line=dict(color='darkred', width=1)),
            name=f'Gap Region {region_id}',
            text=gap_hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Gap Region {region_id} between Cluster {cluster_A} and {cluster_B}',
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            height=500,
            hovermode='closest',
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with st.spinner(f"🤖 Generating contrastive explanation using {model}..."):
        try:
            # Get representative papers from each cluster (closest to centroid)
            X_primary = st.session_state.X_primary
            
            # Cluster A papers
            idx_A = np.where(st.session_state.df_valid['cluster_hdbscan'] == cluster_A)[0]
            X_A = X_primary[idx_A]
            centroid_A = X_A.mean(axis=0, keepdims=True)
            dists_A = pairwise.cosine_distances(X_A, centroid_A).ravel()
            top_A_local = np.argsort(dists_A)[:n_papers]
            top_A_idx = idx_A[top_A_local]
            
            # Cluster B papers
            idx_B = np.where(st.session_state.df_valid['cluster_hdbscan'] == cluster_B)[0]
            X_B = X_primary[idx_B]
            centroid_B = X_B.mean(axis=0, keepdims=True)
            dists_B = pairwise.cosine_distances(X_B, centroid_B).ravel()
            top_B_local = np.argsort(dists_B)[:n_papers]
            top_B_idx = idx_B[top_B_local]
            
            # Build evidence pack
            evidence_pack = []
            for idx in top_A_idx:
                row = st.session_state.df_valid.iloc[idx]
                # Try to get paper ID from common column names
                paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
                evidence_pack.append({
                    "doc_id": f"A_{idx}",
                    "paper_id": str(paper_id),
                    "title": str(row.get('title', '')),
                    "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                    "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                    "cluster": "A"
                })
            
            for idx in top_B_idx:
                row = st.session_state.df_valid.iloc[idx]
                # Try to get paper ID from common column names
                paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
                evidence_pack.append({
                    "doc_id": f"B_{idx}",
                    "paper_id": str(paper_id),
                    "title": str(row.get('title', '')),
                    "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                    "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                    "cluster": "B"
                })
            
            # Enhanced prompt with domain-specific axes
            system_prompt = """You are a nanomedicine domain expert. Only use the EVIDENCE PACK provided.
            Never invent facts or cite outside sources. If evidence is insufficient for any claim,
            state 'unknown'. Cite by doc_id for every claim. Output exactly the JSON schema."""
                        
            # Build additional guidance sections
            custom_question_section = ""
            if custom_question:
                custom_question_section = f"""

            SPECIFIC QUESTION TO ADDRESS:
            {custom_question}

            Please include your answer to this question in a dedicated field called "custom_question_answer" in the JSON output.
            Base your answer strictly on the evidence provided. If the evidence is insufficient, state this clearly.
            """
            
            keywords_guidance_section = ""
            if guidance_keywords:
                keywords_list = [kw.strip() for kw in guidance_keywords.split(',') if kw.strip()]
                if keywords_list:
                    keywords_guidance_section = f"""

            KEYWORDS FOR BRIDGE OPPORTUNITIES:
            When identifying bridge opportunities, pay special attention to these keywords and concepts: {', '.join(keywords_list)}
            Consider how these keywords might relate to potential connections between the two clusters.
            """
            
            # Build output schema with optional custom question field
            output_schema = """{
            "cluster_A_summary": {
                "one_line": "string",
                "bullets": ["string"],
                "salient_entities": {"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]},
                "citations": ["doc_id"]
            },
            "cluster_B_summary": {
                "one_line": "string",
                "bullets": ["string"],
                "salient_entities": {"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]},
                "citations": ["doc_id"]
            },
            "axes_of_separation": [{
                "axis": "materials|ligands|disease|model|delivery|toxicity|methods|other",
                "what_differs": "short explanation (evidence-grounded)",
                "evidence_A": ["doc_id"],
                "evidence_B": ["doc_id"],
                "confidence": 0.0-1.0
            }],
            "bridge_seeds": [{
                "idea": "short description of a possible bridge",
                "why_plausible": "mechanistic rationale, grounded in docs",
                "support": ["doc_id"],
                "risks": ["toxicity","aggregation","RES","immunogenicity","scaleup","IP","assay_limitations"]
            }],"""
                        
            if custom_question:
                output_schema += """
            "custom_question_answer": {
                "answer": "string",
                "supporting_evidence": ["doc_id"],
                "confidence": 0.0-1.0,
                "limitations": "string"
            },"""
                        
                output_schema += """
            "insufficient_evidence": false
            }"""
                        
                user_prompt = f"""TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
            Focus on: materials, surface chemistry/coatings, size/shape, targeting ligands, disease areas,
            models (in vitro/in vivo/clinical), delivery routes, pharmacokinetics/biodistribution,
            toxicity/regulatory language, endpoints/outcomes.

            CONTEXT:
            - cluster_A_meta: {{"id": {cluster_A}, "n_docs": {cluster_counts[cluster_A]}}}
            - cluster_B_meta: {{"id": {cluster_B}, "n_docs": {cluster_counts[cluster_B]}}}
            - Gap region: {len(region_indices)} papers spanning both clusters{custom_question_section}{keywords_guidance_section}

            EVIDENCE PACK (JSONL; each line is one doc):
            ```jsonl
            {chr(10).join(json.dumps(d, ensure_ascii=False) for d in evidence_pack)}
            ```

            OUTPUT JSON SCHEMA:
            {output_schema}
            """
            
            # Show prompt editor if requested
            if show_prompt_editor:
                st.markdown("---")
                st.markdown("### 📝 Prompt Editor")
                st.markdown("Edit the system and user prompts below before sending to the LLM:")
                
                with st.expander("🔧 System Prompt", expanded=True):
                    edited_system_prompt = st.text_area(
                        "System Prompt",
                        value=system_prompt,
                        height=150,
                        key="system_prompt_editor",
                        label_visibility="collapsed"
                    )
                
                with st.expander("📋 User Prompt", expanded=True):
                    edited_user_prompt = st.text_area(
                        "User Prompt",
                        value=user_prompt,
                        height=400,
                        key="user_prompt_editor",
                        label_visibility="collapsed"
                    )
                
                st.info("💡 You can modify the prompts above. Click the button below to send the edited prompts to the LLM.")
                
                if not st.button("✅ Send Edited Prompts to LLM", type="primary"):
                    st.warning("⚠️ Click 'Send Edited Prompts to LLM' above to proceed with the analysis.")
                    return
                
                # Use edited prompts
                system_prompt = edited_system_prompt
                user_prompt = edited_user_prompt
                st.markdown("---")
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Display comprehensive results
            display_llm_results(result, region_id, cluster_A, cluster_B, len(region_indices))
            
        except Exception as e:
            st.error(f"❌ Error generating explanation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_llm_results(result, region_id, cluster_A, cluster_B, region_size):
    """Display LLM analysis results in a structured format"""
    st.success("✅ Analysis complete!")
    
    st.markdown("---")
    st.markdown(f"### 📋 Contrastive Analysis Results")
    st.markdown(f"**Region {region_id}** | Size: {region_size} papers | Comparing Cluster {cluster_A} vs {cluster_B}")
    
    # Cluster summaries side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 🔵 Cluster {cluster_A} Summary")
        if 'cluster_A_summary' in result:
            summary_A = result['cluster_A_summary']
            st.markdown(f"**{summary_A.get('one_line', 'N/A')}**")
            
            if 'bullets' in summary_A and summary_A['bullets']:
                st.markdown("**Key characteristics:**")
                for bullet in summary_A['bullets'][:5]:
                    st.write(f"• {bullet}")
            
            if 'salient_entities' in summary_A:
                entities = summary_A['salient_entities']
                with st.expander("📌 Salient Entities", expanded=False):
                    for entity_type, items in entities.items():
                        if items:
                            st.write(f"**{entity_type.title()}:** {', '.join(items[:7])}")
            
            if 'citations' in summary_A and summary_A['citations']:
                st.caption(f"Based on: {', '.join(summary_A['citations'][:5])}")
    
    with col2:
        st.markdown(f"#### 🟢 Cluster {cluster_B} Summary")
        if 'cluster_B_summary' in result:
            summary_B = result['cluster_B_summary']
            st.markdown(f"**{summary_B.get('one_line', 'N/A')}**")
            
            if 'bullets' in summary_B and summary_B['bullets']:
                st.markdown("**Key characteristics:**")
                for bullet in summary_B['bullets'][:5]:
                    st.write(f"• {bullet}")
            
            if 'salient_entities' in summary_B:
                entities = summary_B['salient_entities']
                with st.expander("📌 Salient Entities", expanded=False):
                    for entity_type, items in entities.items():
                        if items:
                            st.write(f"**{entity_type.title()}:** {', '.join(items[:7])}")
            
            if 'citations' in summary_B and summary_B['citations']:
                st.caption(f"Based on: {', '.join(summary_B['citations'][:5])}")
    
    # Custom question answer (if provided)
    if 'custom_question_answer' in result:
        st.markdown("---")
        st.markdown("### ❓ Custom Question Analysis")
        
        qa = result['custom_question_answer']
        st.markdown(f"**Answer:** {qa.get('answer', 'N/A')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'confidence' in qa:
                st.metric("Confidence", f"{qa['confidence']:.2f}")
        with col2:
            if 'supporting_evidence' in qa and qa['supporting_evidence']:
                st.caption(f"Evidence: {', '.join(qa['supporting_evidence'][:5])}")
        
        if 'limitations' in qa and qa['limitations']:
            st.info(f"**Limitations:** {qa['limitations']}")
    
    # Axes of separation
    st.markdown("---")
    st.markdown("### 🎯 Axes of Separation (Key Differences)")
    
    if 'axes_of_separation' in result and result['axes_of_separation']:
        for i, axis in enumerate(result['axes_of_separation'], 1):
            with st.expander(f"{i}. {axis.get('axis', 'unknown').upper()} (confidence: {axis.get('confidence', 0):.2f})", expanded=i<=3):
                st.write(axis.get('what_differs', 'N/A'))
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'evidence_A' in axis and axis['evidence_A']:
                        st.caption(f"Evidence A: {', '.join(axis['evidence_A'][:3])}")
                with col2:
                    if 'evidence_B' in axis and axis['evidence_B']:
                        st.caption(f"Evidence B: {', '.join(axis['evidence_B'][:3])}")
    else:
        st.info("No specific axes of separation identified")
    
    # Bridge opportunities
    st.markdown("---")
    st.markdown("### 🌉 Bridge Opportunities (Research Gaps)")
    
    if 'bridge_seeds' in result and result['bridge_seeds']:
        for i, bridge in enumerate(result['bridge_seeds'], 1):
            st.markdown(f"**{i}. {bridge.get('idea', 'N/A')}**")
            st.write(f"**Rationale:** {bridge.get('why_plausible', 'N/A')}")
            
            if bridge.get('risks'):
                st.write(f"⚠️ **Potential Risks:** {', '.join(bridge['risks'][:5])}")
            
            if bridge.get('support'):
                st.caption(f"Supporting evidence: {', '.join(bridge['support'][:3])}")
            
            st.markdown("")
    else:
        st.info("No bridge opportunities identified")
    
    # Warnings
    if result.get('insufficient_evidence', False):
        st.warning("⚠️ Note: LLM flagged insufficient evidence for some conclusions")
    
    # Download option
    st.markdown("---")
    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Full Analysis (JSON)",
        data=result_json,
        file_name=f"gap_region_{region_id}_analysis.json",
        mime="application/json"
    )


# ============================================================================
# PAGE: EXPORT
# ============================================================================

def page_export():
    """Export results"""
    st.title("💾 Export Results")
    
    if st.session_state.df_valid is None:
        st.warning("⚠️ No data to export")
        return
    
    st.markdown("Export your analysis results and gap regions.")
    
    # Data summary
    st.subheader("📊 Data Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Papers", len(st.session_state.df_valid))
    
    if st.session_state.density_computed:
        gap_threshold = st.session_state.df_valid['gap_score'].quantile(st.session_state.config['gap_quantile'])
        n_gaps = (st.session_state.df_valid['gap_score'] >= gap_threshold).sum()
        col2.metric("Gap Candidates", n_gaps)
    
    if st.session_state.gaps_identified:
        col3.metric("Gap Regions", len(st.session_state.gap_regions))
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Full Dataset with Features**")
        if st.button("Download CSV", key="download_full"):
            csv = st.session_state.df_valid.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"novelty_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.session_state.gaps_identified and st.session_state.gap_regions:
            st.markdown("**Gap Regions Summary**")
            gap_summary = create_gap_summary()
            
            csv = gap_summary.to_csv(index=False)
            st.download_button(
                label="📄 Download Gap Summary",
                data=csv,
                file_name=f"gap_regions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Preview
    st.subheader("👀 Data Preview")
    st.dataframe(st.session_state.df_valid.head(20))


def create_gap_summary():
    """Create gap regions summary dataframe"""
    gap_summary = []
    
    for region_id, region_indices in enumerate(st.session_state.gap_regions):
        region_df = st.session_state.df_valid.loc[region_indices]
        
        summary = {
            'region_id': region_id,
            'n_papers': len(region_indices),
            'avg_gap_score': region_df['gap_score'].mean(),
            'max_gap_score': region_df['gap_score'].max(),
        }
        
        if 'cluster_hdbscan' in region_df.columns:
            summary['dominant_cluster'] = region_df['cluster_hdbscan'].mode()[0] if len(region_df) > 0 else -1
            summary['n_clusters_spanned'] = region_df['cluster_hdbscan'].nunique()
        
        if 'publication_year' in region_df.columns:
            years = pd.to_numeric(region_df['publication_year'], errors='coerce').dropna()
            if len(years) > 0:
                summary['median_year'] = int(years.median())
                summary['year_range'] = f"{int(years.min())}-{int(years.max())}"
        
        gap_summary.append(summary)
    
    return pd.DataFrame(gap_summary).sort_values('avg_gap_score', ascending=False)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Novelty Analysis App",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🔬 Novelty Analysis")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            [
                "📊 Data & Config",
                "🧬 Embeddings",
                "🎯 Filters",
                "🔬 Clustering",
                "🔍 Gap Analysis",
                "🌉 Gap Regions",
                "🤖 LLM Analysis",
                "💾 Export"
            ]
        )
        
        st.markdown("---")
        
        # Status indicators
        st.markdown("### 📌 Status")
        st.write("✅" if st.session_state.df_filtered is not None else "⬜", "Data Loaded")
        st.write("✅" if st.session_state.embeddings_extracted else "⬜", "Embeddings Extracted")
        st.write("✅" if st.session_state.X_pca is not None else "⬜", "PCA Reduction")
        st.write("✅" if st.session_state.X_umap_2d is not None else "⬜", "UMAP Projection")
        st.write("✅" if st.session_state.clustering_done else "⬜", "Clustering Done")
        st.write("✅" if st.session_state.density_computed else "⬜", "Density Computed")
        st.write("✅" if st.session_state.gaps_identified else "⬜", "Gaps Identified")
        
        st.markdown("---")
        
        # Undo button
        if st.session_state.undo_history:
            last_action = st.session_state.undo_history[-1]['action_name']
            if st.button(f"↩️ Undo: {last_action}", use_container_width=True):
                if undo_last_action():
                    st.success(f"Undone: {last_action}")
                    st.rerun()
        else:
            st.button("↩️ Undo (no actions)", disabled=True, use_container_width=True)
        
        st.markdown("---")
        
        if st.session_state.df_valid is not None:
            st.metric("Current Papers", len(st.session_state.df_valid))
    
    # Route to pages
    if page == "📊 Data & Config":
        page_data_loading()
    elif page == "🧬 Embeddings":
        page_embedding_processing()
    elif page == "🎯 Filters":
        page_filters()
    elif page == "🔬 Clustering":
        page_clustering()
    elif page == "🔍 Gap Analysis":
        page_gap_analysis()
    elif page == "🌉 Gap Regions":
        page_gap_regions()
    elif page == "🤖 LLM Analysis":
        page_llm_analysis()
    elif page == "💾 Export":
        page_export()


if __name__ == "__main__":
    main()
