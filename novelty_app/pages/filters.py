"""
Filters Page - Optional filtering tools (K-means, semantic similarity, entity-based)
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans

from core.state_management import save_state_for_undo
from core.constants import MATERIAL_HINTS, LIGAND_HINTS, DISEASE_HINTS, DELIVERY_HINTS, MODEL_HINTS
from ui.export_utils import display_figure_with_export


def to_native(obj):
    """Convert narwhals-wrapped objects to native pandas."""
    if hasattr(obj, 'to_native'):
        return obj.to_native()
    return obj


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
        cluster_counts = to_native(pd.Series(cluster_labels).value_counts().sort_index())
        
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
        display_figure_with_export(fig, "kmeans_clusters", key="export_kmeans")
        
        with st.expander("📊 Cluster Distribution"):
            st.bar_chart(cluster_counts.to_dict())
    
    st.divider()
    
    # Semantic similarity filter
    st.subheader("2️⃣ Semantic Retrieval")
    
    retrieval_mode = st.radio(
        "Retrieval Mode",
        ["Semantic Similarity", ], # "Question Answering (Reranker)"
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
            instruction_text = st.text_input(
                "Instruction (optional)",
                value="Given a web search query, retrieve relevant passages that answer the query",
                help="Leave empty to pass None as instruction"
            )
            similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Compute Similarities"):
                compute_semantic_similarity(query_text, similarity_threshold, instruction_text)
        
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
            display_figure_with_export(fig, "semantic_similarity", key="export_similarity")
            
            # Show top matches
            with st.expander("📄 Top 10 Most Similar Papers"):
                top_papers = st.session_state.df_valid.nlargest(10, 'similarity_score')
                for idx, row in top_papers.iterrows():
                    st.write(f"**[{row['similarity_score']:.3f}]** {row.get('title', 'N/A')}")
    
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=st.session_state.random_seed, n_init=10)
        labels = kmeans.fit_predict(st.session_state.X_pca)
        st.session_state.df_valid['kmeans_cluster'] = labels
        st.session_state.kmeans_applied = True


def apply_cluster_filter(selected_clusters):
    """Filter to selected clusters"""
    save_state_for_undo(f"K-means filter to {len(selected_clusters)} clusters")
    
    mask = st.session_state.df_valid['kmeans_cluster'].isin(selected_clusters)
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask] if st.session_state.X_pca is not None else None
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask] if st.session_state.X_umap_2d is not None else None
    
    # Update embeddings_dict
    for key in st.session_state.embeddings_dict:
        st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][mask]
    
    st.success(f"✅ Filtered: {n_before} → {len(st.session_state.df_valid)} papers")
    st.rerun()


def compute_semantic_similarity(query_text, threshold, instruction=None):
    """Compute semantic similarity to query using preloaded embeddings"""
    try:
        import requests
        
        # API endpoint for Qwen embedding service
        QWEN_API_URL = "http://localhost:8000"
        
        with st.spinner("Generating query embedding..."):
            # Generate query embedding with instruction for better retrieval
            # Use None if instruction is empty string
            instruction_param = instruction if instruction and instruction.strip() else None
            
            query_payload = {
                "texts": [query_text],
                "instruction": instruction_param,
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
            
            query_embedding_norm = query_embedding
            paper_embeddings_norm = paper_embeddings
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


def apply_similarity_filter(threshold):
    """Filter by similarity threshold"""
    save_state_for_undo(f"Similarity filter (threshold={threshold:.2f})")
    
    mask = st.session_state.df_valid['similarity_score'] >= threshold
    n_before = len(st.session_state.df_valid)
    
    st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
    st.session_state.X_pca = st.session_state.X_pca[mask] if st.session_state.X_pca is not None else None
    st.session_state.X_primary = st.session_state.X_primary[mask]
    st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask] if st.session_state.X_umap_2d is not None else None
    
    # Update embeddings_dict
    for key in st.session_state.embeddings_dict:
        st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][mask]
    
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
        st.session_state.X_pca = st.session_state.X_pca[mask] if st.session_state.X_pca is not None else None
        st.session_state.X_primary = st.session_state.X_primary[mask]
        st.session_state.X_umap_2d = st.session_state.X_umap_2d[mask] if st.session_state.X_umap_2d is not None else None
        
        # Update embeddings_dict
        for key in st.session_state.embeddings_dict:
            st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][mask]
        
        st.success(f"✅ Filtered by {entity_type}: {n_before} → {len(st.session_state.df_valid)} papers")
        st.rerun()
