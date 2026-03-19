"""
State management utilities for undo/redo functionality
"""
import streamlit as st


def save_state_for_undo(action_name: str):
    """Save current state to undo history"""
    # Deep copy embeddings_dict
    embeddings_dict_copy = {}
    if st.session_state.embeddings_dict:
        for key, value in st.session_state.embeddings_dict.items():
            embeddings_dict_copy[key] = value.copy() if value is not None else None
    
    state_snapshot = {
        'action_name': action_name,
        'df_valid': st.session_state.df_valid.copy() if st.session_state.df_valid is not None else None,
        'X_pca': st.session_state.X_pca.copy() if st.session_state.X_pca is not None else None,
        'X_primary': st.session_state.X_primary.copy() if st.session_state.X_primary is not None else None,
        'X_umap_2d': st.session_state.X_umap_2d.copy() if st.session_state.X_umap_2d is not None else None,
        'embeddings_dict': embeddings_dict_copy,
        'kmeans_applied': st.session_state.kmeans_applied,
        'similarity_applied': st.session_state.similarity_applied,
        'G': st.session_state.G,  # Graph is immutable, can reference directly
        'clustering_done': st.session_state.clustering_done,
        'selected_clustering': st.session_state.selected_clustering
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
    st.session_state.embeddings_dict = snapshot.get('embeddings_dict', {})
    st.session_state.kmeans_applied = snapshot['kmeans_applied']
    st.session_state.similarity_applied = snapshot['similarity_applied']
    st.session_state.G = snapshot.get('G', None)
    st.session_state.clustering_done = snapshot.get('clustering_done', False)
    st.session_state.selected_clustering = snapshot.get('selected_clustering', None)
    
    # Update UMAP coordinates in dataframe if available
    if st.session_state.df_valid is not None and st.session_state.X_umap_2d is not None:
        st.session_state.df_valid['umap_x'] = st.session_state.X_umap_2d[:, 0]
        st.session_state.df_valid['umap_y'] = st.session_state.X_umap_2d[:, 1]
    
    return True


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

    # Agent backend / console state
    if 'agent_backend_url' not in st.session_state:
        st.session_state.agent_backend_url = "http://localhost:8088"
    if 'agent_snapshot_id' not in st.session_state:
        st.session_state.agent_snapshot_id = ""
    if 'agent_last_health' not in st.session_state:
        st.session_state.agent_last_health = None
    if 'agent_snapshots_cache' not in st.session_state:
        st.session_state.agent_snapshots_cache = None
    if 'agent_publish_result' not in st.session_state:
        st.session_state.agent_publish_result = None
    if 'agent_last_tool_response' not in st.session_state:
        st.session_state.agent_last_tool_response = None
    if 'agent_last_pack' not in st.session_state:
        st.session_state.agent_last_pack = None
    if 'agent_last_run' not in st.session_state:
        st.session_state.agent_last_run = None
    if 'agent_artifacts_cache' not in st.session_state:
        st.session_state.agent_artifacts_cache = None
