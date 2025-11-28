"""
Streamlit App for Interactive Embedding Exploration (v2)
Supports multiple embedding types with preprocessing phase for dimensionality reduction
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from typing import Literal, Optional, List
import ast
import requests
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import (
    preprocess_embeddings,
    load_preprocessed_data,
    EmbeddingConfig
)
from utils.cluster_utils_v2 import (
    kmeans_cluster,
    assign_class_to_embeddings,
    filter_by_bounding_box,
    filter_by_clusters
)
from utils.data_utils_v2 import (
    load_dataframe,
    write_df_to_excel,
    parse_embedding_column
)
from utils.utils import preprocess_text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
from collections import Counter
import networkx as nx


# Configuration
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Embedding API configuration (same as embed_api.py)
EMBED_API_HOST = os.getenv("HOST", "127.0.0.1")
EMBED_API_PORT = int(os.getenv("PORT", "54288"))
EMBED_API_URL = f"http://{EMBED_API_HOST}:{EMBED_API_PORT}"


def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'df_history' not in st.session_state:
        st.session_state['df_history'] = []
    if 'current_embedding' not in st.session_state:
        st.session_state['current_embedding'] = 'qwen_content_embedding'
    if 'preprocessed_data' not in st.session_state:
        st.session_state['preprocessed_data'] = {}
    if 'kmeans_clusters' not in st.session_state:
        st.session_state['kmeans_clusters'] = 3
    if 'plot' not in st.session_state:
        st.session_state['plot'] = None
    if 'similarity_scores' not in st.session_state:
        st.session_state['similarity_scores'] = None
    if 'similarity_threshold' not in st.session_state:
        st.session_state['similarity_threshold'] = 0.5


def get_available_embeddings(df: pd.DataFrame) -> List[str]:
    """Extract available embedding columns from dataframe"""
    embedding_cols = [
        col for col in df.columns 
        if 'embedding' in col.lower() and df[col].dtype == 'object'
    ]
    return embedding_cols


@st.cache_data(show_spinner="Loading dataframe...")
def load_data_cached(file_path: str) -> pd.DataFrame:
    """Load and parse dataframe with embedding columns"""
    df = load_dataframe(file_path)
    
    # Parse embedding columns (convert string representations to arrays)
    embedding_cols = get_available_embeddings(df)
    for col in embedding_cols:
        df = parse_embedding_column(df, col)
    
    return df


def preprocessing_page():
    """Preprocessing page - compute UMAP projections for all embeddings"""
    st.title("🔧 Preprocessing: Compute Dimensionality Reduction")
    
    st.markdown("""
    This page computes low-dimensional (2D) projections of your high-dimensional embeddings using UMAP.
    These projections are cached for fast loading in the main exploration interface.
    """)
    
    # File selection
    data_file = st.text_input(
        "Path to CSV file",
        value="papers_dataframe_full_processed_with_processed_embeddings.csv"
    )
    
    if not os.path.exists(data_file):
        st.error(f"File not found: {data_file}")
        return
    
    # Load data
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            df = load_data_cached(data_file)
            # Exclusion criteria
            keywords_exclusion = ["review", "not available"]
            for keyword in keywords_exclusion:
                df = df[~df['title'].str.lower().str.contains(keyword)]
            st.session_state['df'] = df
            st.success(f"Loaded {len(df)} records")
    
    if st.session_state['df'] is None:
        st.info("👆 Click 'Load Data' to begin")
        return
    
    

    
    df = st.session_state['df']

    


    
    # Show available embeddings
    embedding_cols = get_available_embeddings(df)
    st.subheader("Available Embedding Columns")
    st.write(embedding_cols)
    
    # UMAP Configuration
    st.subheader("UMAP Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_neighbors = st.slider("n_neighbors", 5, 200, 15)
    with col2:
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1)
    with col3:
        metric = st.selectbox("metric", ["cosine", "euclidean", "manhattan"])
    
    # Select which embeddings to process
    selected_embeddings = st.multiselect(
        "Select embeddings to preprocess",
        embedding_cols,
        default=embedding_cols[:2] if len(embedding_cols) >= 2 else embedding_cols
    )
    
    # Run preprocessing
    if st.button("🚀 Run Preprocessing", type="primary"):
        if not selected_embeddings:
            st.warning("Please select at least one embedding column")
            return
        
        config = EmbeddingConfig(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        for i, emb_col in enumerate(selected_embeddings):
            status_text.text(f"Processing {emb_col}...")
            
            try:
                result = preprocess_embeddings(
                    df=df,
                    embedding_col=emb_col,
                    config=config,
                    cache_dir=CACHE_DIR
                )
                results[emb_col] = result
                st.success(f"✓ {emb_col} processed successfully")
            except Exception as e:
                st.error(f"✗ Error processing {emb_col}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(selected_embeddings))
        
        st.session_state['preprocessed_data'] = results
        status_text.text("✅ All preprocessing complete!")
        
        # Show preview
        st.subheader("Preview of Projections")
        for emb_col, result in results.items():
            st.write(f"**{emb_col}**: {result['umap_2d'].shape}")
            st.write(f"  - Cache: {result['cache_path']}")


def exploration_page():
    """Main exploration interface"""
    st.title("🔍 Interactive Embedding Exploration")
    
    # Load preprocessed data
    available_preprocessed = list(CACHE_DIR.glob("*.pkl"))
    
    if not available_preprocessed:
        st.warning("⚠️ No preprocessed data found. Please run preprocessing first.")
        if st.button("Go to Preprocessing"):
            st.session_state['page'] = 'preprocessing'
            st.rerun()
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Select embedding
        embedding_options = [p.stem for p in available_preprocessed]
        selected_embedding = st.selectbox(
            "Select Embedding",
            embedding_options,
            index=0
        )
        
        # Load preprocessed data
        if selected_embedding != st.session_state.get('current_embedding'):
            with st.spinner("Loading preprocessed data..."):
                data = load_preprocessed_data(CACHE_DIR / f"{selected_embedding}.pkl")
                # Ensure native pandas DataFrame (not Narwhals)
                df_loaded = data['df']
                if hasattr(df_loaded, 'to_native'):
                    df_loaded = df_loaded.to_native()
                elif not isinstance(df_loaded, pd.DataFrame):
                    df_loaded = pd.DataFrame(df_loaded)
                st.session_state['df'] = df_loaded
                st.session_state['df_history'] = [df_loaded.copy()]
                st.session_state['current_embedding'] = selected_embedding
        
        st.divider()
        
        # Clustering controls
        st.header("🎯 Clustering")
        k_means = st.number_input(
            "K-means clusters",
            min_value=2,
            max_value=100,
            value=st.session_state.get('kmeans_clusters', 21),
            key='kmeans_clusters'
        )
        run_kmeans = st.button("Run K-means", use_container_width=True)
        
        st.divider()
        
        # Bounding box selection
        st.header("📦 Bounding Box")
        col1, col2 = st.columns(2)
        x1 = col1.number_input("x_min", value=0.0, format="%.2f")
        x2 = col2.number_input("x_max", value=1.0, format="%.2f")
        y1 = col1.number_input("y_min", value=0.0, format="%.2f")
        y2 = col2.number_input("y_max", value=1.0, format="%.2f")
        run_bbox = st.button("Apply Filter", use_container_width=True)
        
        st.divider()
        
        # Cluster filter
        st.header("🏷️ Filter by Cluster")
        if 'cluster_label' in st.session_state['df'].columns:
            unique_clusters = sorted(st.session_state['df']['cluster_label'].unique())
            selected_clusters = st.multiselect(
                "Select clusters",
                unique_clusters
            )
            run_cluster_filter = st.button("Filter", key="cluster_filter", use_container_width=True)
        else:
            st.info("Run K-means first to create clusters")
            run_cluster_filter = False
        
        st.divider()
        
        # Text-based assignment
        st.header("🔤 Text-based Assignment")
        search_text = st.text_input("Search term", value="cancer")
        assigned_class = st.text_input("Assign to class", value="1")
        col1, col2 = st.columns(2)
        run_assign = col1.button("Assign", use_container_width=True)
        run_text_filter = col2.button("Filter", key="text_filter", use_container_width=True)

        st.divider()
        
        # Semantic search
        st.header("🔍 Semantic Search")
        query_text = st.text_area(
            "Search query",
            value="Use gold as material for drug delivery.",
            height=100,
            help="Enter a query to find semantically similar papers"
        )
        
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('similarity_threshold', 0.5),
            step=0.05,
            help="Papers above this threshold will be highlighted"
        )
        st.session_state['similarity_threshold'] = similarity_threshold
        
        col1, col2 = st.columns(2)
        run_similarity = col1.button("🔍 Search", key="run_similarity", use_container_width=True)
        clear_similarity = col2.button("Clear", key="clear_similarity", use_container_width=True)
        
        if st.session_state.get('similarity_scores') is not None:
            n_above = (st.session_state['similarity_scores'] >= similarity_threshold).sum()
            n_total = len(st.session_state['similarity_scores'])
            st.info(f"Found {n_above}/{n_total} papers above threshold ({100*n_above/n_total:.1f}%)")

        # End of sidebar
    
    # Main content area
    df = st.session_state['df']
    
    # Top controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if st.button("↩️ Undo"):
            if len(st.session_state['df_history']) > 1:
                st.session_state['df_history'].pop()
                st.session_state['df'] = st.session_state['df_history'][-1].copy()
                st.rerun()
    with col3:
        if st.button("💾 Export"):
            write_df_to_excel(df)
            st.success("Exported!")
    with col4:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Plot
    similarity_scores = st.session_state.get('similarity_scores')
    similarity_threshold = st.session_state.get('similarity_threshold', 0.5)
    plot = draw_plot(df, similarity_scores=similarity_scores, similarity_threshold=similarity_threshold)
    st.altair_chart(plot, use_container_width=True)
    
    # Handle actions
    if run_kmeans:
        df_new = kmeans_cluster(df, num_clusters=k_means)
        update_dataframe(df_new)
        st.rerun()
    
    if run_bbox:
        df_new = filter_by_bounding_box(df, x1, x2, y1, y2)
        update_dataframe(df_new)
        st.rerun()
    
    if 'run_cluster_filter' in locals() and run_cluster_filter:
        if selected_clusters:
            df_new = filter_by_clusters(df, selected_clusters)
            update_dataframe(df_new)
            st.rerun()
    
    if run_assign:
        df_new = assign_class_to_embeddings(df, search_text, assigned_class)
        update_dataframe(df_new)
        st.rerun()
    
    if run_text_filter:
        df_new = filter_by_clusters(df, assigned_class.split(','))
        update_dataframe(df_new)
        st.rerun()
    
    if run_similarity:
        with st.spinner("Computing similarities..."):
            try:
                # Get the current embedding column from session state
                current_emb = st.session_state.get('current_embedding', 'qwen_content_embedding')
                
                # Preprocess query text
                # processed_query = preprocess_text(query_text)
                
                # Get query embedding from API
                response = requests.post(
                    f"{EMBED_API_URL}/embed",
                    json={
                        "text": query_text,
                        "task": "s2p",
                        "normalize": True
                    }
                )
                
                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                else:
                    query_result = response.json()
                    query_embedding = np.array(query_result['embedding'])
                    
                    # Get embeddings from current dataframe
                    # Find the embedding column that matches the current selection
                    embedding_col = None
                    for col in df.columns:
                        if current_emb in col or col.endswith('_embedding'):
                            if df[col].dtype == 'object' and len(df) > 0:
                                first_val = df[col].iloc[0]
                                if isinstance(first_val, (list, np.ndarray)):
                                    embedding_col = col
                                    break
                    
                    if embedding_col is None:
                        st.error("Could not find embedding column in dataframe")
                    else:
                        # Stack embeddings
                        embeddings = np.stack(df[embedding_col].values)
                        
                        # Compute cosine similarities
                        similarities = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            embeddings
                        )[0]
                        
                        # Store in session state
                        st.session_state['similarity_scores'] = similarities
                        st.success(f"Similarity computed! Max: {similarities.max():.4f}, Mean: {similarities.mean():.4f}")
                        st.rerun()
            except Exception as e:
                st.error(f"Error computing similarity: {str(e)}")
    
    if clear_similarity:
        st.session_state['similarity_scores'] = None
        st.rerun()
    
    # Show data table
    with st.expander("📊 View Data Table"):
        st.dataframe(
            df[['title', 'abstract', 'cluster_label', 'low_x', 'low_y']].head(100),
            use_container_width=True
        )


def update_dataframe(df_new: pd.DataFrame):
    """Update dataframe and history"""
    st.session_state['df_history'].append(df_new.copy())
    st.session_state['df'] = df_new


@st.cache_data(show_spinner=False)
def draw_plot(df: pd.DataFrame, similarity_scores: Optional[np.ndarray] = None, 
              similarity_threshold: float = 0.5) -> alt.Chart:
    """Create Altair scatter plot with optional similarity-based coloring"""
    # Ensure native pandas DataFrame (not Narwhals)
    if hasattr(df, 'to_native'):
        df = df.to_native()
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Fix column types for Arrow serialization
    if 'publication_year' in df.columns:
        df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce').fillna(0).astype(int)
    
    # Ensure we have the required columns
    if 'cluster_label' not in df.columns:
        df['cluster_label'] = 'unlabeled'
    
    if 'size' not in df.columns:
        df['size'] = 20
    
    # Add similarity scores if provided
    if similarity_scores is not None:
        df['similarity'] = similarity_scores
        df['above_threshold'] = similarity_scores >= similarity_threshold
        df['size'] = np.where(df['above_threshold'], 30, 20)
    
    # Create tooltip columns (handle missing columns gracefully)
    tooltip_cols = []
    if 'title' in df.columns:
        tooltip_cols.append('title')
    if 'abstract' in df.columns:
        tooltip_cols.append('abstract')
    if 'cluster_label' in df.columns:
        tooltip_cols.append('cluster_label')
    if 'similarity' in df.columns:
        tooltip_cols.append(alt.Tooltip('similarity:Q', format='.4f'))
    
    # Choose color encoding based on whether we have similarity scores
    if similarity_scores is not None:
        # Similarity-based coloring
        # Points below threshold in grey, above threshold colored by similarity
        chart = alt.Chart(df).mark_circle().encode(
            x=alt.X('low_x:Q', title='UMAP 1'),
            y=alt.Y('low_y:Q', title='UMAP 2'),
            color=alt.condition(
                alt.datum.above_threshold,
                alt.Color('similarity:Q',
                         scale=alt.Scale(scheme='yelloworangered', domain=[similarity_threshold, 1.0]),
                         title='Similarity'),
                alt.value('lightgrey')
            ),
            size=alt.Size('size:Q', legend=None),
            opacity=alt.condition(
                alt.datum.above_threshold,
                alt.value(0.8),
                alt.value(0.3)
            ),
            tooltip=tooltip_cols
        ).properties(
            height=1000,
        ).interactive()
    else:
        # Regular cluster-based coloring
        chart = alt.Chart(df).mark_circle().encode(
            x=alt.X('low_x:Q', title='UMAP 1'),
            y=alt.Y('low_y:Q', title='UMAP 2'),
            color=alt.Color('cluster_label:N', 
                           scale=alt.Scale(scheme='tableau20'),
                           title='Cluster'),
            size=alt.Size('size:Q', legend=None),
            opacity=alt.value(0.6),
            tooltip=tooltip_cols
        ).properties(
            height=1000,
        ).interactive()
    
    return chart

def ideation():
    pass


def gap_analysis_page():
    """Gap Analysis page - comprehensive pipeline for identifying research gaps"""
    st.title("🎯 Gap Analysis & Novelty Discovery")
    
    st.markdown("""
    This page performs comprehensive gap analysis on your filtered dataset to identify:
    - Low-density regions in embedding space (research gaps)
    - Cluster-based organization of the literature
    - Bridge opportunities between different research areas
    """)
    
    # Check if we have preprocessed data
    if st.session_state.get('df') is None:
        st.warning("⚠️ No data loaded. Please load data from the Exploration page first.")
        if st.button("Go to Exploration"):
            st.session_state['page'] = 'exploration'
            st.rerun()
        return
    
    df = st.session_state['df']
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Gap Analysis Settings")
        
        # PCA settings
        st.subheader("1️⃣ PCA Dimensionality Reduction")
        pca_components = st.slider("PCA components", 10, 100, 50, step=10)
        
        # Clustering settings
        st.subheader("2️⃣ Clustering")
        cluster_method = st.selectbox("Clustering method", ["HDBSCAN", "K-means"])
        
        if cluster_method == "K-means":
            n_clusters = st.slider("Number of clusters", 5, 50, 20, step=5)
        else:
            min_cluster_size = st.slider("Min cluster size", 10, 100, 20, step=10)
            min_samples = st.slider("Min samples", 5, 50, 8, step=1)
        
        # Gap detection settings
        st.subheader("3️⃣ Gap Detection")
        gap_threshold_q = st.slider("Gap threshold (quantile)", 0.90, 0.99, 0.95, step=0.01)
        min_region_size = st.slider("Min gap region size", 2, 10, 3, step=1)
        
        # LLM settings
        st.subheader("4️⃣ LLM Analysis (Optional)")
        run_llm = st.checkbox("Generate LLM explanations", value=False)
        if run_llm:
            api_key = st.text_input("OpenAI API Key", type="password", 
                                   value=os.getenv('OPENAI_API_KEY', ''))
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
            llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
            max_regions_llm = st.slider("Max regions to analyze", 1, 10, 3)
        
        st.divider()
        run_analysis = st.button("🚀 Run Gap Analysis", type="primary", use_container_width=True)
    
    # Main content
    if not run_analysis and 'gap_analysis_results' not in st.session_state:
        st.info("👈 Configure settings in the sidebar and click 'Run Gap Analysis' to begin")
        
        # Show current dataset info
        st.subheader("📊 Current Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", len(df))
        with col2:
            if 'publication_year' in df.columns:
                years = pd.to_numeric(df['publication_year'], errors='coerce').dropna()
                if len(years) > 0:
                    st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
        with col3:
            if 'cluster_label' in df.columns:
                st.metric("Existing Clusters", df['cluster_label'].nunique())
        
        return
    
    if run_analysis:
        # Run the gap analysis pipeline
        with st.spinner("Running gap analysis pipeline..."):
            try:
                results = run_gap_analysis_pipeline(
                    df=df,
                    pca_components=pca_components,
                    cluster_method=cluster_method,
                    n_clusters=n_clusters if cluster_method == "K-means" else None,
                    min_cluster_size=min_cluster_size if cluster_method == "HDBSCAN" else None,
                    min_samples=min_samples if cluster_method == "HDBSCAN" else None,
                    gap_threshold_q=gap_threshold_q,
                    min_region_size=min_region_size,
                    run_llm=run_llm,
                    llm_model=llm_model if run_llm else None,
                    max_regions_llm=max_regions_llm if run_llm else 0
                )
                
                st.session_state['gap_analysis_results'] = results
                st.success("✅ Gap analysis complete!")
            except Exception as e:
                st.error(f"Error during gap analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results
    if 'gap_analysis_results' in st.session_state:
        results = st.session_state['gap_analysis_results']
        display_gap_analysis_results(results)


def run_gap_analysis_pipeline(df, pca_components, cluster_method, n_clusters=None, 
                              min_cluster_size=None, min_samples=None,
                              gap_threshold_q=0.95, min_region_size=3,
                              run_llm=False, llm_model=None, max_regions_llm=0):
    """Run the complete gap analysis pipeline"""
    
    results = {}
    progress = st.progress(0)
    status = st.empty()
    
    # Step 1: Extract embeddings
    status.text("Step 1/7: Extracting embeddings...")
    progress.progress(1/7)
    
    # Find primary embedding column
    embedding_col = None
    for col in df.columns:
        if col.endswith('_embedding') and df[col].dtype == 'object':
            first_val = df[col].iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                embedding_col = col
                break
    
    if embedding_col is None:
        raise ValueError("No valid embedding column found in dataframe")
    
    X = np.stack(df[embedding_col].values)
    results['embedding_col'] = embedding_col
    results['X_original'] = X
    
    # Step 2: PCA
    status.text("Step 2/7: Running PCA...")
    progress.progress(2/7)
    
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X)
    results['X_pca'] = X_pca
    results['pca_explained_variance'] = pca.explained_variance_ratio_.sum()
    
    # Step 3: Compute density (gap_z scores)
    status.text("Step 3/7: Computing density scores...")
    progress.progress(3/7)
    
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=30, metric='euclidean')
    knn.fit(X_pca)
    distances, _ = knn.kneighbors(X_pca)
    avg_distances = distances[:, 1:].mean(axis=1)  # Exclude self
    
    # Z-score normalization (higher = lower density = gap)
    gap_z = (avg_distances - avg_distances.mean()) / (avg_distances.std() + 1e-9)
    results['gap_z'] = gap_z
    
    # Step 4: Clustering
    status.text(f"Step 4/7: Clustering ({cluster_method})...")
    progress.progress(4/7)
    
    if cluster_method == "K-means":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_pca)
    else:  # HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(X_pca)
    
    results['labels'] = labels
    results['cluster_method'] = cluster_method
    results['n_clusters'] = len(np.unique(labels[labels >= 0]))
    results['n_noise'] = np.sum(labels == -1)
    
    # Step 5: Identify gap regions
    status.text("Step 5/7: Identifying gap regions...")
    progress.progress(5/7)
    
    threshold = np.quantile(gap_z, gap_threshold_q)
    gap_candidates = set(np.where(gap_z >= threshold)[0].tolist())
    
    # Build k-NN graph for gap candidates
    if len(gap_candidates) > 0:
        knn_graph = NearestNeighbors(n_neighbors=min(10, len(gap_candidates)), metric='euclidean')
        gap_indices = list(gap_candidates)
        knn_graph.fit(X_pca[gap_indices])
        distances, indices = knn_graph.kneighbors(X_pca[gap_indices])
        
        # Build networkx graph
        G = nx.Graph()
        G.add_nodes_from(gap_indices)
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                G.add_edge(gap_indices[i], gap_indices[j])
        
        # Find connected components (gap regions)
        regions = [list(comp) for comp in nx.connected_components(G)]
        regions = [r for r in regions if len(r) >= min_region_size]
    else:
        regions = []
    
    results['gap_threshold'] = threshold
    results['gap_candidates'] = gap_candidates
    results['gap_regions'] = regions
    
    # Step 6: Create 2D visualization
    status.text("Step 6/7: Creating 2D UMAP visualization...")
    progress.progress(6/7)
    
    import umap
    reducer_2d = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
    X_plot = reducer_2d.fit_transform(X_pca)
    results['X_plot'] = X_plot
    
    # Step 7: LLM analysis (optional)
    if run_llm and len(regions) > 0:
        status.text("Step 7/7: Running LLM analysis...")
        progress.progress(6.5/7)
        
        llm_results = run_llm_analysis(
            df, X_pca, labels, regions, gap_z, 
            llm_model, max_regions_llm
        )
        results['llm_explanations'] = llm_results
    else:
        status.text("Step 7/7: Skipping LLM analysis...")
    
    progress.progress(1.0)
    status.text("✅ Analysis complete!")
    
    return results


def run_llm_analysis(df, X, labels, regions, gap_z, model, max_regions):
    """Run LLM analysis on top gap regions"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Sort regions by average gap_z
        region_scores = []
        for i, region in enumerate(regions):
            avg_score = gap_z[region].mean()
            region_scores.append((i, avg_score, region))
        region_scores.sort(key=lambda x: -x[1])
        
        llm_results = []
        for i, avg_score, region in region_scores[:max_regions]:
            # Find dominant clusters
            region_labels = labels[region]
            cluster_counts = Counter(region_labels)
            
            if len(cluster_counts) < 2:
                continue
            
            cluster_A, cluster_B = [c for c, _ in cluster_counts.most_common(2)]
            
            # Get sample papers from each cluster
            mask_A = labels == cluster_A
            mask_B = labels == cluster_B
            
            sample_A = df[mask_A].head(5)
            sample_B = df[mask_B].head(5)
            
            # Construct prompt
            prompt = f"""Analyze these two clusters of nanomedicine research papers and identify the research gap between them.

**Cluster A papers:**
{chr(10).join([f"- {row['title']}" for _, row in sample_A.iterrows()])}

**Cluster B papers:**
{chr(10).join([f"- {row['title']}" for _, row in sample_B.iterrows()])}

Provide:
1. One-sentence summary of each cluster
2. Key differences between the clusters
3. Potential bridge research opportunities (research that could connect these areas)
4. Risks or challenges

Format as JSON with keys: cluster_A_summary, cluster_B_summary, differences, bridge_opportunities, risks"""
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                explanation_text = response.choices[0].message.content
                
                # Try to parse as JSON
                import json
                try:
                    if explanation_text and "```json" in explanation_text:
                        explanation_text = explanation_text.split("```json")[1].split("```")[0].strip()
                    elif explanation_text and "```" in explanation_text:
                        explanation_text = explanation_text.split("```")[1].split("```")[0].strip()
                    explanation = json.loads(explanation_text) if explanation_text else {"raw": "No response"}
                except:
                    explanation = {"raw": explanation_text or "No response"}
                
                llm_results.append({
                    'region_id': i,
                    'cluster_A': int(cluster_A),
                    'cluster_B': int(cluster_B),
                    'avg_gap_z': float(avg_score),
                    'region_size': len(region),
                    'explanation': explanation
                })
            except Exception as e:
                st.warning(f"LLM error for region {i}: {str(e)}")
                continue
        
        return llm_results
    except Exception as e:
        st.error(f"LLM initialization error: {str(e)}")
        return []


def display_gap_analysis_results(results):
    """Display comprehensive gap analysis results"""
    
    # Summary statistics
    st.header("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PCA Variance Explained", f"{100*results['pca_explained_variance']:.1f}%")
    with col2:
        st.metric("Clusters Found", results['n_clusters'])
    with col3:
        st.metric("Gap Regions", len(results['gap_regions']))
    with col4:
        st.metric("Noise Points", results['n_noise'])
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎨 Gap Visualization", 
        "🔍 Cluster Explorer", 
        "📍 Gap Regions",
        "🤖 LLM Insights",
        "📥 Export"
    ])
    
    with tab1:
        st.subheader("Gap Score Distribution")
        display_gap_visualization(results)
    
    with tab2:
        st.subheader("Cluster Analysis")
        display_cluster_explorer(results)
    
    with tab3:
        st.subheader("Identified Gap Regions")
        display_gap_regions(results)
    
    with tab4:
        st.subheader("LLM-Generated Insights")
        display_llm_insights(results)
    
    with tab5:
        st.subheader("Export Results")
        display_export_options(results)


def display_gap_visualization(results):
    """Create comprehensive gap visualizations"""
    import matplotlib.pyplot as plt
    
    X_plot = results['X_plot']
    gap_z = results['gap_z']
    labels = results['labels']
    threshold = results['gap_threshold']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Panel 1: Continuous gap_z coloring
    sc1 = axes[0, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=gap_z, 
                            s=10, alpha=0.6, cmap='viridis')
    axes[0, 0].set_title('Gap Score (gap_z): Higher = Lower Density', fontsize=12)
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    plt.colorbar(sc1, ax=axes[0, 0], label='gap_z')
    
    # Panel 2: Binary view - top candidates
    is_gap = gap_z >= threshold
    colors = ['red' if g else 'lightgray' for g in is_gap]
    axes[0, 1].scatter(X_plot[:, 0], X_plot[:, 1], c=colors, s=10, alpha=0.6)
    axes[0, 1].set_title(f'Gap Candidates (gap_z ≥ {threshold:.2f})', fontsize=12)
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    
    # Panel 3: Clusters
    sc3 = axes[1, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=labels, 
                            s=10, alpha=0.6, cmap='tab20')
    axes[1, 0].set_title('Clusters', fontsize=12)
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    plt.colorbar(sc3, ax=axes[1, 0], label='Cluster ID')
    
    # Panel 4: Combined view
    axes[1, 1].scatter(X_plot[:, 0], X_plot[:, 1], c=gap_z, 
                      s=10, alpha=0.4, cmap='viridis')
    gap_points = X_plot[is_gap]
    axes[1, 1].scatter(gap_points[:, 0], gap_points[:, 1], 
                      facecolors='none', edgecolors='red', s=50, linewidths=1.5)
    axes[1, 1].set_title('Gap Scores with Candidates Highlighted', fontsize=12)
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics
    st.write(f"""
    **Gap Analysis Statistics:**
    - Total papers: {len(gap_z)}
    - Gap candidates (top {100*(1-results['gap_threshold'])}%): {is_gap.sum()}
    - Gap_z range: [{gap_z.min():.2f}, {gap_z.max():.2f}]
    - Gap_z mean: {gap_z.mean():.2f}, std: {gap_z.std():.2f}
    """)


def display_cluster_explorer(results):
    """Interactive cluster exploration"""
    labels = results['labels']
    unique_labels = sorted(set(labels[labels >= 0]))
    
    if len(unique_labels) == 0:
        st.warning("No clusters found")
        return
    
    selected_cluster = st.selectbox("Select cluster to explore", unique_labels)
    
    mask = labels == selected_cluster
    cluster_df = st.session_state['df'][mask]
    
    st.write(f"**Cluster {selected_cluster}:** {len(cluster_df)} papers")
    
    # Show sample papers
    st.dataframe(cluster_df[['title', 'publication_year', 'journal']].head(10))


def display_gap_regions(results):
    """Display detailed information about gap regions"""
    regions = results['gap_regions']
    
    if len(regions) == 0:
        st.warning("No gap regions found. Try lowering the threshold.")
        return
    
    df = st.session_state['df']
    gap_z = results['gap_z']
    labels = results['labels']
    
    # Summary table
    region_data = []
    for i, region in enumerate(regions):
        region_data.append({
            'Region ID': i,
            'Size': len(region),
            'Avg gap_z': gap_z[region].mean(),
            'Dominant Cluster': Counter(labels[region]).most_common(1)[0][0] if len(region) > 0 else -1
        })
    
    region_df = pd.DataFrame(region_data)
    st.dataframe(region_df, use_container_width=True)
    
    # Detail view
    selected_region = st.selectbox("Select region to explore", range(len(regions)))
    region_indices = regions[selected_region]
    region_papers = df.iloc[region_indices]
    
    st.write(f"**Papers in Region {selected_region}:**")
    st.dataframe(region_papers[['title', 'publication_year', 'journal']].head(10))


def display_llm_insights(results):
    """Display LLM-generated insights"""
    if 'llm_explanations' not in results or len(results['llm_explanations']) == 0:
        st.info("No LLM explanations available. Enable LLM analysis in settings.")
        return
    
    llm_results = results['llm_explanations']
    
    for result in llm_results:
        with st.expander(f"Region {result['region_id']} — Cluster {result['cluster_A']} vs {result['cluster_B']}"):
            explanation = result['explanation']
            
            if 'raw' in explanation:
                st.write(explanation['raw'])
            else:
                if 'cluster_A_summary' in explanation:
                    st.write(f"**Cluster {result['cluster_A']}:** {explanation['cluster_A_summary']}")
                if 'cluster_B_summary' in explanation:
                    st.write(f"**Cluster {result['cluster_B']}:** {explanation['cluster_B_summary']}")
                if 'differences' in explanation:
                    st.write(f"**Key Differences:** {explanation['differences']}")
                if 'bridge_opportunities' in explanation:
                    st.write(f"**Bridge Opportunities:** {explanation['bridge_opportunities']}")
                if 'risks' in explanation:
                    st.write(f"**Risks:** {explanation['risks']}")


def display_export_options(results):
    """Provide export options for results"""
    df = st.session_state['df'].copy()
    df['gap_z'] = results['gap_z']
    df['cluster_label'] = results['labels']
    
    # Mark gap candidates
    threshold = results['gap_threshold']
    df['is_gap_candidate'] = results['gap_z'] >= threshold
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Gap Analysis Results"):
            csv = df[['title', 'publication_year', 'gap_z', 'cluster_label', 'is_gap_candidate']].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="gap_analysis_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Full Dataset with Scores"):
            csv_full = df.to_csv(index=False)
            st.download_button(
                label="Download Full CSV",
                data=csv_full,
                file_name="full_dataset_with_scores.csv",
                mime="text/csv"
            )


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Embedding Explorer v2",
        page_icon="🧬",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Create config directory
    if not os.path.exists('.streamlit'):
        os.makedirs('.streamlit')
        with open(".streamlit/config.toml", "w") as f:
            f.write("[server]\n")
            f.write("maxMessageSize = 1000\n")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Page navigation
    if 'page' not in st.session_state:
        st.session_state['page'] = 'preprocessing'
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧬 Embedding Explorer")
        page = st.radio(
            "Navigation",
            ['preprocessing', 'exploration', 'gap_analysis'],
            format_func=lambda x: {
                'preprocessing': "🔧 Preprocessing",
                'exploration': "🔍 Exploration",
                'gap_analysis': "🎯 Gap Analysis"
            }[x],
            key='page'
        )
    
    # Route to appropriate page
    if page == 'preprocessing':
        preprocessing_page()
    elif page == 'exploration':
        exploration_page()
    else:
        gap_analysis_page()


if __name__ == "__main__":
    main()
