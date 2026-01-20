"""
Gap Regions Page - Identify and explore gap regions
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors
import networkx as nx

from core.entity_utils import extract_entities_from_dataframe, summarize_gap_region_entities


def page_gap_regions():
    """Identify and explore gap regions"""
    st.title("🌉 Gap Regions")
    
    if not st.session_state.density_computed or 'gap_score' not in st.session_state.df_valid.columns:
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
    tab1, tab2, tab3, tab4 = st.tabs(["📍 All Regions", "🎨 By Region ID", "📊 By Score", "🔄 Over Clusters"])
    
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
        # Color by region ID with background showing all papers
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        # Create base figure with all papers in light gray
        fig = go.Figure()
        
        # Add all papers as background
        background = df_plot[df_plot['gap_region'] == -1]
        if len(background) > 0:
            fig.add_trace(go.Scatter(
                x=background['umap_x'],
                y=background['umap_y'],
                mode='markers',
                marker=dict(size=4, color='lightgray', opacity=0.3),
                name='Other Papers',
                text=[f"<b>{row['hover_title']}</b><br>{row['hover_abstract']}" for _, row in background.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))
        
        # Add each gap region with unique color
        df_gap = df_plot[df_plot['gap_region'] >= 0]
        if len(df_gap) > 0:
            # Use a color palette for different regions
            color_palette = colors.qualitative.Plotly + colors.qualitative.Set3
            
            for region_id in sorted(df_gap['gap_region'].unique()):
                region_data = df_gap[df_gap['gap_region'] == region_id]
                color_idx = region_id % len(color_palette)
                
                hover_text = [
                    f"<b>{row['hover_title']}</b><br>" +
                    f"Gap Region: {region_id}<br>" +
                    f"Gap Score: {row.get('gap_score', 0):.3f}<br>" +
                    f"{row['hover_abstract']}"
                    for _, row in region_data.iterrows()
                ]
                
                fig.add_trace(go.Scatter(
                    x=region_data['umap_x'],
                    y=region_data['umap_y'],
                    mode='markers',
                    marker=dict(size=10, color=color_palette[color_idx], opacity=0.8,
                               line=dict(color='white', width=1)),
                    name=f'Region {region_id} (n={len(region_data)})',
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ))
        
        fig.update_layout(
            title=f"Gap Regions Colored by ID (n={len(gap_regions)})",
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            height=1000,
            hovermode='closest',
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
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
    
    with tab4:
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
    
    # Use selected clustering method for "Clusters Spanned" metric
    if st.session_state.selected_clustering:
        cluster_col = f'cluster_{st.session_state.selected_clustering}'
        if cluster_col in region_df.columns:
            col4.metric("Clusters Spanned", region_df[cluster_col].nunique())
    elif 'cluster_hdbscan' in region_df.columns:
        # Fallback to HDBSCAN if no clustering method selected
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
                st.bar_chart(year_counts.to_dict())
    
    # Top papers
    st.markdown("#### 📄 Top Papers by Gap Score")
    top_papers = region_df.nlargest(5, 'gap_score')
    
    for idx, (_, row) in enumerate(top_papers.iterrows(), 1):
        with st.expander(f"{idx}. [{row['gap_score']:.3f}] {row.get('title', 'N/A')}"):
            st.write(f"**Year**: {row.get('publication_year', 'N/A')}")
            st.write(f"**Journal**: {row.get('journal', 'N/A')}")
            if 'abstract' in row and pd.notna(row['abstract']):
                st.write(f"**Abstract**: {row['abstract'][:500]}...")
