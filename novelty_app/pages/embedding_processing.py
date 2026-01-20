"""
Embedding Processing Page - PCA and UMAP dimensionality reduction
"""
import streamlit as st
import plotly.express as px
import umap
from sklearn.decomposition import PCA
from ui.export_utils import display_figure_with_export


def page_embedding_processing():
    """Extract embeddings and apply dimensionality reduction"""
    st.title("🧬 Embedding Processing")
    
    if st.session_state.df_filtered is None:
        st.warning("⚠️ Please load data first")
        return
    
    config = st.session_state.config
    
    st.markdown(f"""
    **Dataset**: {len(st.session_state.df_valid) if st.session_state.df_valid is not None else len(st.session_state.df_filtered)} papers  
    **Primary Embedding**: {config['primary_embedding']}  
    **Available Embeddings**: {', '.join(config['embedding_cols'])}
    """)
    
    # Check if embeddings are already loaded
    if not st.session_state.embeddings_extracted:
        st.warning("⚠️ Embeddings should have been loaded with the data. Please reload the dataset.")
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
        display_figure_with_export(fig, "umap_2d_projection", key="export_umap_embedding")
