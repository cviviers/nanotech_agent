"""
Clustering Page - K-means, HDBSCAN, and Community Detection (Leiden/Louvain)
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import hdbscan
from sklearn.cluster import KMeans

try:
    from agents.observability import observe_current
except Exception:  # pragma: no cover
    from novelty_app.agents.observability import observe_current

from core.state_management import save_state_for_undo, undo_last_action
from core.graph_utils import build_knn_graph, explore_cluster
from ui.export_utils import display_figure_with_export

# Check for optional community detection libraries
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

# Check for OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def page_clustering():
    """Run clustering algorithms"""
    st.title("🎯 Clustering Analysis")
    
    if st.session_state.X_pca is None:
        st.warning("⚠️ Please complete embedding processing first")
        return
    
    st.markdown(f"""
    **Working Dataset**: {len(st.session_state.df_valid)} papers  
    
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
        knn_graph_k = st.number_input("k-NN Graph k", min_value=5, max_value=100, value=21,
                                     help="Number of neighbors for k-NN graph (used in Community Detection)")
        leiden_resolution = st.slider("Leiden Resolution", min_value=0.1, max_value=5.0, value=1.0, step=0.01,
                                     help="Higher values create more communities")
    
    # Store clustering config in session state
    st.session_state.clustering_config = {
        'hdbscan_min_cluster_size': hdbscan_min_cluster,
        'knn_graph_k': knn_graph_k,
        'hdbscan_min_samples': hdbscan_min_samples,
        'leiden_resolution': leiden_resolution,
        'kmeans_n_clusters': int(st.session_state.get('kmeans_main', 20)),
        'community_detection_algorithm': (
            'leiden' if LEIDEN_AVAILABLE else 'louvain' if LOUVAIN_AVAILABLE else None
        ),
        'community_graph_metric': 'cosine',
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
                    kmeans = KMeans(n_clusters=kmeans_n_clusters_main, random_state=st.session_state.random_seed, n_init=10)
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
        display_figure_with_export(fig, "kmeans_clustering", key="export_kmeans_clustering")
        
        # Cluster exploration
        with st.expander("🔍 Explore Clusters", expanded=False):
            unique_clusters = sorted(st.session_state.df_valid['cluster_kmeans'].unique())
            cluster_sizes = st.session_state.df_valid['cluster_kmeans'].value_counts().sort_index()
            
            selected_cluster = st.selectbox(
                "Select Cluster to Explore",
                unique_clusters,
                format_func=lambda x: f"Cluster {x} ({cluster_sizes[x]} papers)"
            )
            
            if selected_cluster is not None:
                explore_cluster(st.session_state.df_valid, 'cluster_kmeans', selected_cluster)
        
        # Selection button
        if st.session_state.selected_clustering == 'kmeans':
            st.success("✅ K-means selected for gap analysis")
        else:
            if st.button("✔️ Use K-means for Gap Analysis", type="primary"):
                save_state_for_undo("K-means Clustering Selection")
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
        display_figure_with_export(fig, "hdbscan_clustering", key="export_hdbscan")
        
        # Cluster exploration
        with st.expander("🔍 Explore Clusters", expanded=False):
            unique_clusters = sorted([c for c in st.session_state.df_valid['cluster_hdbscan'].unique() if c != -1])
            cluster_sizes = st.session_state.df_valid[st.session_state.df_valid['cluster_hdbscan'] != -1]['cluster_hdbscan'].value_counts().sort_index()
            
            if len(unique_clusters) > 0:
                selected_cluster = st.selectbox(
                    "Select Cluster to Explore",
                    unique_clusters,
                    format_func=lambda x: f"Cluster {x} ({cluster_sizes[x]} papers)",
                    key="hdbscan_explore"
                )
                
                if selected_cluster is not None:
                    explore_cluster(st.session_state.df_valid, 'cluster_hdbscan', selected_cluster)
            else:
                st.info("No clusters found (all noise points)")
        
        # Selection button
        if st.session_state.selected_clustering == 'hdbscan':
            st.success("✅ HDBSCAN selected for gap analysis")
        else:
            if st.button("✔️ Use HDBSCAN for Gap Analysis", type="primary"):
                save_state_for_undo("HDBSCAN Clustering Selection")
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
                run_louvain_clustering(clustering_config)
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
            display_figure_with_export(fig_graph, "knn_graph", key="export_knn_graph")
            
            # Graph statistics
            subcol1, subcol2 = st.columns(2)
            subcol1.metric("Nodes", G.number_of_nodes())
            subcol2.metric("Edges", G.number_of_edges())
            avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
            subcol1.metric("Avg Degree", f"{avg_degree:.1f}")
            subcol2.metric("Components", nx.number_connected_components(G))
        
        with col2:
            st.markdown("**Detected Communities**")
            
            # Prepare plot data with hover columns
            df_plot_communities = st.session_state.df_valid.copy()
            df_plot_communities['hover_title'] = df_plot_communities['title'].fillna('N/A')
            df_plot_communities['hover_abstract'] = df_plot_communities.get('abstract', df_plot_communities.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
            
            fig = px.scatter(
                df_plot_communities,
                x='umap_x',
                y='umap_y',
                color='cluster_leiden',
                title=f"Community Detection (n={n_communities})",
                color_continuous_scale='rainbow',
                hover_data={'umap_x': False, 'umap_y': False, 'hover_title': True, 'hover_abstract': True},
                opacity=0.7,
                height=500
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
            display_figure_with_export(fig, "community_detection", key="export_community")
            
            # Community size distribution
            unique, counts = np.unique(labels, return_counts=True)
            st.bar_chart({f"C{c}": cnt for c, cnt in zip(unique[:10], counts[:10])})
        
        # Cluster exploration
        with st.expander("🔍 Explore Communities", expanded=False):
            unique_clusters = sorted(st.session_state.df_valid['cluster_leiden'].unique())
            cluster_sizes = st.session_state.df_valid['cluster_leiden'].value_counts().sort_index()
            
            selected_cluster = st.selectbox(
                "Select Community to Explore",
                unique_clusters,
                format_func=lambda x: f"Community {x} ({cluster_sizes[x]} papers)",
                key="leiden_explore"
            )
            
            if selected_cluster is not None:
                explore_cluster(st.session_state.df_valid, 'cluster_leiden', selected_cluster)
        
        # Create summary of all communities
        with st.expander("🔍 Summarize Communities", expanded=False):
            st.markdown("""
            Generate high-level summaries for all communities using an LLM.
            The LLM will analyze paper titles from each community to create a brief overview.
            """)
            
            summary_model = st.selectbox(
                "Model",
                ["gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-5.2-2025-12-11"],
                index=0,
                key="community_summary_model"
            )
            
            if st.button("🚀 Generate All Community Summaries", type="primary"):
                summary_api_key = st.session_state.get('openai_api_key', os.environ.get('OPENAI_API_KEY', ''))
                if not summary_api_key:
                    st.error("❌ Please provide OpenAI API key in Data & Config page")
                elif not OPENAI_AVAILABLE:
                    st.error("❌ OpenAI library not available. Install with: pip install openai")
                else:
                    # Get unique communities
                    unique_communities = sorted(st.session_state.df_valid['cluster_leiden'].unique())
                    
                    # Initialize results storage
                    if 'community_summaries' not in st.session_state:
                        st.session_state.community_summaries = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, community_id in enumerate(unique_communities):
                        status_text.text(f"Summarizing Community {community_id}... ({idx+1}/{len(unique_communities)})")
                        
                        # Get papers in this community
                        community_df = st.session_state.df_valid[
                            st.session_state.df_valid['cluster_leiden'] == community_id
                        ]
                        
                        # Get paper titles
                        titles = community_df['title'].fillna('Untitled').tolist()
                        
                        # Create prompt
                        prompt = f"""You are analyzing a research community in nanomedicine.
                        
                        Below are the titles of {len(titles)} papers in Community {community_id}:

                        {chr(10).join(f"{i+1}. {title}" for i, title in enumerate(titles))}

                        Based on these paper titles, provide a brief high-level summary (2-3 sentences) describing:
                        1. The main research focus or theme of this community
                        2. Key topics or approaches that appear frequently

                        Keep your response concise and focused on the overarching themes."""
                        
                        try:
                            # Call OpenAI API
                            client = OpenAI(api_key=summary_api_key)
                            messages = [
                               {"role": "system", "content": "You are a research analyst specializing in nanomedicine literature analysis."},
                               {"role": "user", "content": prompt}
                            ]
                            with observe_current(
                                name="community_summary_llm",
                                as_type="generation",
                                input_payload={
                                    "community_id": int(community_id),
                                    "n_titles": len(titles),
                                },
                                metadata={
                                    "community_id": int(community_id),
                                    "n_titles": len(titles),
                                },
                                model=summary_model,
                            ) as observation:
                                response = client.chat.completions.create(
                                    model=summary_model,
                                    messages=messages
                                )
                                summary = response.choices[0].message.content.strip()
                                observation.update(
                                    output={
                                        "community_id": int(community_id),
                                        "summary_preview": summary[:200],
                                    }
                                )
                            st.session_state.community_summaries[community_id] = {
                                'summary': summary,
                                'n_papers': len(titles),
                                'model': summary_model
                            }
                            
                        except Exception as e:
                            st.error(f"❌ Error summarizing Community {community_id}: {str(e)}")
                            st.session_state.community_summaries[community_id] = {
                                'summary': f"Error: {str(e)}",
                                'n_papers': len(titles),
                                'model': summary_model
                            }
                        
                        progress_bar.progress((idx + 1) / len(unique_communities))
                    
                    status_text.text("✅ All communities summarized!")
                    st.success(f"✅ Generated summaries for {len(unique_communities)} communities")
            
            # Display summaries if they exist
            if 'community_summaries' in st.session_state and st.session_state.community_summaries:
                st.markdown("---")
                st.markdown("### 📋 Community Summaries")
                
                for community_id in sorted(st.session_state.community_summaries.keys()):
                    summary_data = st.session_state.community_summaries[community_id]
                    
                    with st.container():
                        st.markdown(f"#### Community {community_id}")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(summary_data['summary'])
                        with col2:
                            st.metric("Papers", summary_data['n_papers'])
                        st.markdown("---")
        
        # Selection button
        if st.session_state.selected_clustering == 'leiden':
            st.success("✅ Community Detection selected for gap analysis")
        else:
            if st.button("✔️ Use Community Detection for Gap Analysis", type="primary"):
                save_state_for_undo("Community Detection Clustering Selection")
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
        st.session_state.clustering_config = {
            **dict(st.session_state.get("clustering_config") or {}),
            "community_detection_algorithm": "leiden",
            "community_graph_metric": "cosine",
            "knn_graph_k": int(clustering_config["knn_graph_k"]),
            "leiden_resolution": float(clustering_config["leiden_resolution"]),
        }
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
            seed=st.session_state.random_seed
        )
        
        labels = np.zeros(len(G), dtype=int)
        for cluster_id, community in enumerate(partition):
            labels[list(community)] = cluster_id
        
        st.session_state.df_valid['cluster_leiden'] = labels
        st.success(f"✅ Leiden: {len(set(labels))} communities")
        st.rerun()


def run_louvain_clustering(clustering_config):
    """Run Louvain algorithm"""
    save_state_for_undo("Louvain Clustering")
    with st.spinner("Running Louvain algorithm..."):
        st.session_state.clustering_config = {
            **dict(st.session_state.get("clustering_config") or {}),
            "community_detection_algorithm": "louvain",
            "community_graph_metric": "cosine",
            "knn_graph_k": int(clustering_config["knn_graph_k"]),
            "leiden_resolution": float(clustering_config["leiden_resolution"]),
        }
        G = st.session_state.G
        partition = community_louvain.best_partition(
            G,
            weight='weight',
            resolution=float(clustering_config['leiden_resolution']),
            random_state=st.session_state.random_seed,
        )
        labels = np.array([partition[i] for i in range(len(G))], dtype=int)
        
        st.session_state.df_valid['cluster_leiden'] = labels
        st.success(f"✅ Louvain: {len(set(labels))} communities")
        st.rerun()
