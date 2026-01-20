"""
Session state management for the Novelty Analysis App
"""
import streamlit as st


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
    
    # Random seed for reproducibility
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 42
    
    # LLM analysis results
    if 'llm_results' not in st.session_state:
        st.session_state.llm_results = None
    
    # LLM prompts (for editing before sending)
    if 'llm_prompts' not in st.session_state:
        st.session_state.llm_prompts = None
    
    # Config dict
    if 'config' not in st.session_state:
        from config import DEFAULT_CONFIG
        st.session_state.config = DEFAULT_CONFIG.copy()
