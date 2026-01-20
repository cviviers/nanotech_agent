"""
Gap Analysis Page - Density computation and gap detection
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from core.graph_utils import build_knn_graph
from core.density_utils import compute_density_features


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
            np.random.seed(st.session_state.random_seed)
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
    if not st.session_state.density_computed or 'gap_score' not in st.session_state.df_valid.columns:
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
