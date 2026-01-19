"""
Undo/redo functionality for the Novelty Analysis App
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
