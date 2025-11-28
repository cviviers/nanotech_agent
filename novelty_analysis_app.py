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


def extract_embeddings(df: pd.DataFrame, embed_cols: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Extract embeddings from dataframe columns."""
    present_cols = [c for c in embed_cols if c in df.columns]
    if not present_cols:
        raise ValueError(f"None of the embedding columns found: {embed_cols}")
    
    arrays_by_col = {}
    valid_mask = None
    
    for col in present_cols:
        col_arrays = [parse_embedding(v) for v in df[col].tolist()]
        arrays_by_col[col] = col_arrays
        col_mask = np.array([a is not None for a in col_arrays])
        valid_mask = col_mask if valid_mask is None else (valid_mask & col_mask)
    
    valid_idx = np.where(valid_mask)[0]
    result = {}
    for col in present_cols:
        embeddings = [arrays_by_col[col][i] for i in valid_idx]
        result[col] = np.vstack(embeddings).astype(np.float32)
    
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


# ============================================================================
# PAGE: CONFIGURATION & DATA LOADING
# ============================================================================

def page_data_loading():
    """Data loading and initial configuration page"""
    st.title("📊 Data Loading & Configuration")
    
    st.markdown("""
    Configure the analysis parameters and load your dataset.
    """)
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Configuration")
        
        data_path = st.text_input(
            "Data File Path",
            value=r"C:\Users\20195435\OneDrive - TU Eindhoven\TUe\Playground\Nanotechnology\papers_dataframe_full_processed_with_processed_embeddings.csv",
            help="Path to your CSV or Parquet file"
        )
        
        sample_n = st.number_input(
            "Sample Size (0 = all data)",
            min_value=0,
            max_value=100000,
            value=0,
            help="Limit dataset size for faster processing"
        )
        
        st.subheader("🎯 Embedding Configuration")
        
        available_embeddings = st.multiselect(
            "Available Embedding Columns",
            ["qwen_content_embedding", "bert_content_embedding", 
             "qwen_processed_content_embedding", "bert_processed_content_embedding"],
            default=["qwen_content_embedding", "bert_content_embedding"]
        )
        
        primary_embedding = st.selectbox(
            "Primary Embedding",
            available_embeddings,
            index=0 if available_embeddings else None
        )
    
    with col2:
        st.subheader("🔧 Analysis Parameters")
        
        k_neighbors = st.multiselect(
            "K-Neighbors for Density",
            [5, 10, 15, 20, 30, 40, 50],
            default=[10, 20, 30, 50]
        )
        
        density_metric = st.selectbox(
            "Density Metric",
            ["cosine", "euclidean", "manhattan"],
            index=0
        )
        
        st.subheader("📊 Clustering Parameters")
        
        col_a, col_b = st.columns(2)
        with col_a:
            knn_graph_k = st.number_input("k-NN Graph K", min_value=5, max_value=50, value=21)
            hdbscan_min_cluster = st.number_input("HDBSCAN Min Cluster Size", min_value=5, max_value=100, value=10)
        with col_b:
            leiden_resolution = st.number_input("Leiden Resolution", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            hdbscan_min_samples = st.number_input("HDBSCAN Min Samples", min_value=1, max_value=50, value=5)
        
        st.subheader("🎯 Gap Detection Parameters")
        
        gap_quantile = st.slider("Gap Quantile (top %)", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        min_gap_region_size = st.number_input("Min Gap Region Size", min_value=2, max_value=20, value=3)
    
    # Store in session state
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.config = {
            'data_path': data_path,
            'sample_n': sample_n if sample_n > 0 else None,
            'embedding_cols': available_embeddings,
            'primary_embedding': primary_embedding,
            'k_neighbors': sorted(k_neighbors),
            'density_metric': density_metric,
            'knn_graph_k': knn_graph_k,
            'hdbscan_min_cluster_size': hdbscan_min_cluster,
            'hdbscan_min_samples': hdbscan_min_samples,
            'leiden_resolution': leiden_resolution,
            'gap_quantile': gap_quantile,
            'min_gap_region_size': min_gap_region_size
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
        keywords_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction"],
            default=["review", "not available"]
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Load Data", type="primary"):
            load_data(st.session_state.config, keywords_exclusion)
    
    # Show data status
    if st.session_state.df_original is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.df_original)} papers")
        
        if st.session_state.df_filtered is not None:
            st.info(f"After filtering: {len(st.session_state.df_filtered)} papers")
        
        with st.expander("📊 Data Preview"):
            st.dataframe(st.session_state.df_filtered.head(10) if st.session_state.df_filtered is not None 
                        else st.session_state.df_original.head(10))


def load_data(config, keywords_exclusion):
    """Load and filter dataset"""
    data_path = Path(config['data_path'])
    
    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return
    
    with st.spinner("Loading dataset..."):
        # Load file
        if data_path.suffix.lower() in {'.parquet', '.pq'}:
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        st.session_state.df_original = df.copy()
        
        # Apply filters
        for keyword in keywords_exclusion:
            if 'title' in df.columns:
                df = df[~df['title'].str.lower().str.contains(keyword, na=False)]
            if 'abstract' in df.columns:
                df = df[~df['abstract'].str.lower().str.contains(keyword, na=False)]
        
        # Sample if needed
        if config['sample_n'] is not None and len(df) > config['sample_n']:
            df = df.sample(config['sample_n'], random_state=42)
        
        df = df.reset_index(drop=True)
        st.session_state.df_filtered = df


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
        if st.button("🔍 Extract Embeddings", type="primary"):
            with st.spinner("Extracting embeddings..."):
                try:
                    embeddings_dict, valid_idx = extract_embeddings(
                        st.session_state.df_filtered,
                        config['embedding_cols']
                    )
                    
                    st.session_state.embeddings_dict = embeddings_dict
                    st.session_state.df_valid = st.session_state.df_filtered.iloc[valid_idx].reset_index(drop=True)
                    st.session_state.X_primary = embeddings_dict[config['primary_embedding']]
                    st.session_state.embeddings_extracted = True
                    
                    st.success(f"✅ Extracted embeddings: {len(valid_idx)} valid rows")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error extracting embeddings: {str(e)}")
        return
    
    st.success(f"✅ Embeddings extracted: {st.session_state.X_primary.shape}")
    
    st.divider()
    
    # PCA reduction
    st.subheader("📉 PCA Dimensionality Reduction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pca_components = st.number_input("PCA Components", min_value=10, max_value=1024, value=50)
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
        
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            title="UMAP 2D Projection",
            opacity=0.6,
            height=600
        )
        fig.update_traces(marker=dict(size=5))
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
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='kmeans_cluster',
            title=f"K-means Clusters (n={kmeans_n_clusters})",
            opacity=0.7,
            height=600,
            color_continuous_scale='rainbow'
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Cluster Distribution"):
            st.bar_chart(cluster_counts)
    
    st.divider()
    
    # Semantic similarity filter
    st.subheader("2️⃣ Semantic Similarity Filter")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_area(
            "Search Query",
            value="Brain delivery for treatment of neurodegenerative diseases",
            height=100
        )
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
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
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='similarity_score',
            title="Semantic Similarity to Query",
            opacity=0.7,
            height=600,
            color_continuous_scale='Viridis'
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top matches
        with st.expander("📄 Top 10 Most Similar Papers"):
            top_papers = st.session_state.df_valid.nlargest(10, 'similarity_score')
            for idx, row in top_papers.iterrows():
                st.write(f"**[{row['similarity_score']:.3f}]** {row.get('title', 'N/A')}")


def run_kmeans_filter(n_clusters):
    """Run K-means clustering"""
    with st.spinner(f"Running K-means with {n_clusters} clusters..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(st.session_state.X_pca)
        st.session_state.df_valid['kmeans_cluster'] = labels
        st.session_state.kmeans_applied = True


def apply_cluster_filter(selected_clusters):
    """Filter to selected clusters"""
    mask = st.session_state.df_valid['kmeans_cluster'].isin(selected_clusters)
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask]
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask]
    
    st.success(f"✅ Filtered: {n_before} → {len(st.session_state.df_valid)} papers")
    st.rerun()


def compute_semantic_similarity(query_text, threshold):
    """Compute semantic similarity to query"""
    try:
        # Try to use embed_api
        from embed_api import embed_single, TextInput
        
        with st.spinner("Generating query embedding..."):
            query_input = TextInput(text=query_text, task='s2p')
            query_embedding = np.array(embed_single(query_input)['embedding'])
        
        with st.spinner("Computing similarities..."):
            X_norm = st.session_state.X_primary / np.linalg.norm(st.session_state.X_primary, axis=1, keepdims=True)
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities = X_norm @ query_norm
            
            st.session_state.df_valid['similarity_score'] = similarities
            st.session_state.similarity_applied = True
    
    except ImportError:
        st.error("❌ embed_api not available. Cannot compute query embedding.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def apply_similarity_filter(threshold):
    """Filter by similarity threshold"""
    mask = st.session_state.df_valid['similarity_score'] >= threshold
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask]
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask]
    
    st.success(f"✅ Filtered: {n_before} → {len(st.session_state.df_valid)} papers")
    st.rerun()


# ============================================================================
# PAGE: DENSITY & GAP ANALYSIS
# ============================================================================

def page_gap_analysis():
    """Compute density features and identify gaps"""
    st.title("🔍 Gap Analysis")
    
    if st.session_state.X_pca is None:
        st.warning("⚠️ Please complete embedding processing first")
        return
    
    config = st.session_state.config
    
    st.markdown(f"""
    **Working Dataset**: {len(st.session_state.df_valid)} papers  
    **K-Neighbors**: {config['k_neighbors']}  
    **Gap Quantile**: {config['gap_quantile']}
    """)
    
    # Compute density
    if not st.session_state.density_computed:
        if st.button("📊 Compute Density Features", type="primary"):
            with st.spinner("Computing density features..."):
                density_df = compute_density_features(
                    st.session_state.X_pca,
                    config['k_neighbors'],
                    config['density_metric']
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
    gap_threshold = gap_scores.quantile(config['gap_quantile'])
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
                     annotation_text=f"Threshold ({config['gap_quantile']:.0%})")
        fig.update_layout(title="Distribution of Gap Scores", xaxis_title="Gap Score", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='gap_score',
            title="Gap Scores in Embedding Space",
            color_continuous_scale='Viridis',
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
    
    # Binary gap view
    st.subheader("🎯 Gap Candidates")
    
    df_plot = st.session_state.df_valid.copy()
    df_plot['is_gap'] = is_gap
    
    fig = px.scatter(
        df_plot,
        x='umap_x',
        y='umap_y',
        color='is_gap',
        title=f"Gap Candidates (top {int((1-config['gap_quantile'])*100)}%)",
        color_discrete_map={True: 'red', False: 'lightgray'},
        opacity=0.7,
        height=600
    )
    fig.update_traces(marker=dict(size=8), selector=dict(name='True'))
    fig.update_traces(marker=dict(size=4), selector=dict(name='False'))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: CLUSTERING
# ============================================================================

def page_clustering():
    """Run clustering algorithms"""
    st.title("🎯 Clustering Analysis")
    
    if not st.session_state.density_computed:
        st.warning("⚠️ Please compute density features first")
        return
    
    config = st.session_state.config
    
    st.markdown(f"""
    **k-NN Graph K**: {config['knn_graph_k']}  
    **HDBSCAN**: min_cluster_size={config['hdbscan_min_cluster_size']}, min_samples={config['hdbscan_min_samples']}  
    **Leiden**: resolution={config['leiden_resolution']}
    """)
    
    # Build k-NN graph
    if st.session_state.G is None:
        if st.button("🕸️ Build k-NN Graph", type="primary"):
            with st.spinner("Building k-NN graph..."):
                G = build_knn_graph(st.session_state.X_pca, config['knn_graph_k'], 'cosine')
                st.session_state.G = G
                st.success(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                st.rerun()
        return
    
    st.success(f"✅ k-NN graph built: {st.session_state.G.number_of_nodes()} nodes, {st.session_state.G.number_of_edges()} edges")
    
    st.divider()
    
    # HDBSCAN
    st.subheader("🔹 HDBSCAN Clustering")
    
    if 'cluster_hdbscan' not in st.session_state.df_valid.columns:
        if st.button("▶️ Run HDBSCAN"):
            with st.spinner("Running HDBSCAN..."):
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=config['hdbscan_min_cluster_size'],
                    min_samples=config['hdbscan_min_samples'],
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
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", n_clusters)
        col2.metric("Noise Points", n_noise)
        col3.metric("Noise %", f"{100*n_noise/len(labels):.1f}%")
        
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='cluster_hdbscan',
            title=f"HDBSCAN Clusters (n={n_clusters})",
            color_continuous_scale='rainbow',
            opacity=0.7,
            height=600
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Leiden/Louvain
    st.subheader("🔹 Community Detection")
    
    if 'cluster_leiden' not in st.session_state.df_valid.columns:
        if LEIDEN_AVAILABLE:
            if st.button("▶️ Run Leiden"):
                run_leiden_clustering(config)
        elif LOUVAIN_AVAILABLE:
            if st.button("▶️ Run Louvain"):
                run_louvain_clustering()
        else:
            st.warning("⚠️ No community detection algorithm available")
    else:
        labels = st.session_state.df_valid['cluster_leiden'].values
        n_communities = len(set(labels))
        
        st.metric("Communities", n_communities)
        
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='cluster_leiden',
            title=f"Community Detection (n={n_communities})",
            color_continuous_scale='rainbow',
            opacity=0.7,
            height=600
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
    
    if 'cluster_hdbscan' in st.session_state.df_valid.columns:
        st.session_state.clustering_done = True


def run_leiden_clustering(config):
    """Run Leiden algorithm"""
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
            resolution_parameter=config['leiden_resolution'],
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
    
    if not st.session_state.clustering_done:
        st.warning("⚠️ Please complete clustering first")
        return
    
    config = st.session_state.config
    
    if not st.session_state.gaps_identified:
        if st.button("🔍 Identify Gap Regions", type="primary"):
            identify_gap_regions(config)
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
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            color='is_gap_region',
            title=f"Gap Regions (n={len(gap_regions)})",
            color_discrete_map={True: 'red', False: 'lightgray'},
            opacity=0.7,
            height=600
        )
        fig.update_traces(marker=dict(size=10), selector=dict(name='True'))
        fig.update_traces(marker=dict(size=4), selector=dict(name='False'))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df_gap = st.session_state.df_valid[st.session_state.df_valid['gap_region'] >= 0]
        
        if len(df_gap) > 0:
            fig = px.scatter(
                df_gap,
                x='umap_x',
                y='umap_y',
                color='gap_score',
                title="Gap Regions by Score",
                color_continuous_scale='Reds',
                opacity=0.8,
                height=600
            )
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.scatter(
            st.session_state.df_valid,
            x='umap_x',
            y='umap_y',
            color='cluster_hdbscan',
            title="Gap Regions over HDBSCAN Clusters",
            color_continuous_scale='rainbow',
            opacity=0.3,
            height=600
        )
        
        df_gap = st.session_state.df_valid[st.session_state.df_valid['gap_region'] >= 0]
        if len(df_gap) > 0:
            fig.add_trace(go.Scatter(
                x=df_gap['umap_x'],
                y=df_gap['umap_y'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star', line=dict(color='darkred', width=1)),
                name='Gap Regions'
            ))
        
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
    
    # Temporal distribution
    if 'publication_year' in region_df.columns:
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
                st.write(f"**Abstract**: {row['abstract']}")


# ============================================================================
# PAGE: LLM ANALYSIS
# ============================================================================

def page_llm_analysis():
    """LLM-based gap explanation"""
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
    Use GPT to generate contrastive explanations for gap regions by comparing neighboring clusters.
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                       value=os.environ.get('OPENAI_API_KEY', ''))
        openai_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)
    with col2:
        region_id = st.selectbox("Select Gap Region", range(len(gap_regions)))
        n_papers_per_cluster = st.number_input("Papers per Cluster", min_value=5, max_value=20, value=10)
    
    if st.button("🚀 Generate Explanation", type="primary"):
        if not openai_api_key:
            st.error("Please provide OpenAI API key")
            return
        
        generate_llm_explanation(region_id, openai_api_key, openai_model, n_papers_per_cluster)


def generate_llm_explanation(region_id, api_key, model, n_papers):
    """Generate LLM explanation for gap region"""
    gap_regions = st.session_state.gap_regions
    region_indices = gap_regions[region_id]
    region_df = st.session_state.df_valid.loc[region_indices]
    
    # Find two most common clusters
    cluster_counts = Counter(region_df['cluster_hdbscan'].tolist())
    if len(cluster_counts) < 2:
        st.warning("Region doesn't span multiple clusters")
        return
    
    cluster_A, cluster_B = [c for c, _ in cluster_counts.most_common(2)]
    
    with st.spinner(f"Analyzing Region {region_id}: Cluster {cluster_A} vs {cluster_B}..."):
        # Sample papers
        papers_A = st.session_state.df_valid[
            st.session_state.df_valid['cluster_hdbscan'] == cluster_A
        ].sample(min(n_papers, (st.session_state.df_valid['cluster_hdbscan'] == cluster_A).sum()), random_state=42)
        
        papers_B = st.session_state.df_valid[
            st.session_state.df_valid['cluster_hdbscan'] == cluster_B
        ].sample(min(n_papers, (st.session_state.df_valid['cluster_hdbscan'] == cluster_B).sum()), random_state=42)
        
        # Format papers
        def format_papers(df):
            result = []
            for _, row in df.iterrows():
                title = row.get('title', 'N/A')
                abstract = row.get('abstract', row.get('processed_abstract', 'N/A'))
                if pd.notna(abstract):
                    abstract = str(abstract)[:300]
                result.append(f"Title: {title}\nAbstract: {abstract}")
            return "\n\n".join(result)
        
        prompt = f"""You are analyzing a gap region in nanomedicine research that spans two clusters.

CLUSTER A papers:
{format_papers(papers_A)}

CLUSTER B papers:
{format_papers(papers_B)}

The gap region contains {len(region_indices)} papers that lie between these clusters.

Please provide:
1. A brief summary of what each cluster focuses on
2. Key differences between the clusters (axes of separation)
3. What research opportunities exist in the gap between them (bridge ideas)
4. Why these bridge ideas might be promising

Format your response as JSON with keys: cluster_A_summary, cluster_B_summary, differences, bridge_opportunities
"""
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in nanomedicine research analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Display results
            st.success("✅ Analysis complete!")
            
            st.markdown(f"### Region {region_id} Analysis")
            st.markdown(f"**Comparing Cluster {cluster_A} vs Cluster {cluster_B}**")
            st.markdown(f"**Region size**: {len(region_indices)} papers")
            
            if 'cluster_A_summary' in result:
                st.markdown(f"#### 🔵 Cluster {cluster_A}")
                st.write(result['cluster_A_summary'])
            
            if 'cluster_B_summary' in result:
                st.markdown(f"#### 🟢 Cluster {cluster_B}")
                st.write(result['cluster_B_summary'])
            
            if 'differences' in result:
                st.markdown("#### 🎯 Key Differences")
                diffs = result['differences']
                if isinstance(diffs, list):
                    for diff in diffs:
                        st.write(f"• {diff}")
                else:
                    st.write(diffs)
            
            if 'bridge_opportunities' in result:
                st.markdown("#### 🌉 Bridge Opportunities")
                bridges = result['bridge_opportunities']
                if isinstance(bridges, list):
                    for i, bridge in enumerate(bridges, 1):
                        st.write(f"{i}. {bridge}")
                else:
                    st.write(bridges)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


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
                "🔍 Gap Analysis",
                "🎯 Clustering",
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
        st.write("✅" if st.session_state.density_computed else "⬜", "Density Computed")
        st.write("✅" if st.session_state.clustering_done else "⬜", "Clustering Done")
        st.write("✅" if st.session_state.gaps_identified else "⬜", "Gaps Identified")
        
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
    elif page == "🔍 Gap Analysis":
        page_gap_analysis()
    elif page == "🎯 Clustering":
        page_clustering()
    elif page == "🌉 Gap Regions":
        page_gap_regions()
    elif page == "🤖 LLM Analysis":
        page_llm_analysis()
    elif page == "💾 Export":
        page_export()


if __name__ == "__main__":
    main()
