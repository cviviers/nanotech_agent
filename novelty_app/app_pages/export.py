"""
Export Page - Export analysis results and gap regions
"""
import pandas as pd
import streamlit as st


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
        gap_config = st.session_state.get('gap_config', {'gap_quantile': 0.95})
        gap_threshold = st.session_state.df_valid['gap_score'].quantile(gap_config['gap_quantile'])
        n_gaps = (st.session_state.df_valid['gap_score'] >= gap_threshold).sum()
        col2.metric("Gap Candidates", n_gaps)
    
    if st.session_state.gaps_identified:
        col3.metric("Gap Regions", len(st.session_state.gap_regions))
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Full Dataset with Features**")
        csv = st.session_state.df_valid.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"novelty_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_export_full"
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
                mime="text/csv",
                key="download_gap_summary"
            )
    
    # Preview
    st.subheader("👀 Data Preview")
    st.dataframe(st.session_state.df_valid.head(20), width="stretch")


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
        
        # Add cluster information based on selected clustering method
        if st.session_state.selected_clustering:
            cluster_col = f'cluster_{st.session_state.selected_clustering}'
            if cluster_col in region_df.columns:
                summary['dominant_cluster'] = region_df[cluster_col].mode()[0] if len(region_df) > 0 else -1
                summary['n_clusters_spanned'] = region_df[cluster_col].nunique()
        elif 'cluster_hdbscan' in region_df.columns:
            # Fallback to HDBSCAN if no clustering method selected
            summary['dominant_cluster'] = region_df['cluster_hdbscan'].mode()[0] if len(region_df) > 0 else -1
            summary['n_clusters_spanned'] = region_df['cluster_hdbscan'].nunique()
        
        if 'publication_year' in region_df.columns:
            years = pd.to_numeric(region_df['publication_year'], errors='coerce').dropna()
            if len(years) > 0:
                summary['median_year'] = int(years.median())
                summary['year_range'] = f"{int(years.min())}-{int(years.max())}"
        
        gap_summary.append(summary)
    
    return pd.DataFrame(gap_summary).sort_values('avg_gap_score', ascending=False)
