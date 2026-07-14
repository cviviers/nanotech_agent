"""
Database Explorer Page - Browse, search, and filter papers
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from core.state_management import save_state_for_undo
from ui.export_utils import display_figure_with_export


def to_native(obj):
    """Convert narwhals-wrapped objects to native pandas."""
    if hasattr(obj, 'to_native'):
        return obj.to_native()
    return obj


def page_database_explorer():
    """Database explorer - browse, search, and filter papers"""
    st.title("📚 Database Explorer")
    
    if st.session_state.df_valid is None:
        st.warning("⚠️ Please load data first")
        return
    
    # Use df_valid as the current dataset
    df = st.session_state.df_valid.copy()
    
    st.markdown(f"""
    **Current Dataset**: {len(df):,} papers
    
    Browse and search the paper database with advanced filtering options.
    """)
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📋 Data View", "🔍 Search & Filter", "📊 Column Stats"])
    
    with tab1:
        st.subheader("Data Table")
        
        # Column selector
        all_columns = df.columns.tolist()
        
        # Suggest important columns at the top
        suggested_cols = []
        for col in ['pmid', 'title', 'abstract', 'publication_year', 'journal', 
                   'gap_score', 'gap_region', 'cluster_kmeans', 'cluster_hdbscan', 'cluster_leiden',
                   'umap_x', 'umap_y']:
            if col in all_columns:
                suggested_cols.append(col)
        
        # Add remaining columns
        other_cols = [c for c in all_columns if c not in suggested_cols]
        ordered_cols = suggested_cols + other_cols
        
        with st.expander("🎯 Select Columns to Display", expanded=False):
            select_all = st.checkbox("Select all columns", value=False)
            
            if select_all:
                selected_columns = all_columns
            else:
                selected_columns = st.multiselect(
                    "Choose columns",
                    options=ordered_cols,
                    default=suggested_cols[:8] if len(suggested_cols) >= 8 else suggested_cols
                )
        
        if not selected_columns:
            selected_columns = suggested_cols[:8] if len(suggested_cols) >= 8 else suggested_cols
        
        # Display settings
        col1, col2, col3 = st.columns(3)
        with col1:
            rows_to_show = st.number_input(
                "Rows to display", 
                min_value=1, 
                max_value=len(df), 
                value=min(50, len(df)),
                step=10
            )
        with col2:
            show_index = st.checkbox("Show row index", value=False)
        with col3:
            sort_column = st.selectbox(
                "Sort by column",
                options=['None'] + selected_columns,
                index=0
            )
        
        # Sort if requested
        df_display = df.copy()
        if sort_column != 'None' and sort_column in df_display.columns:
            ascending = st.checkbox("Ascending order", value=False)
            df_display = df_display.sort_values(by=sort_column, ascending=ascending)
        
        # Display dataframe
        st.dataframe(
            df_display[selected_columns].head(rows_to_show),
            width="stretch",
            hide_index=not show_index,
            height=600
        )
        
        # Display summary stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Displayed Rows", rows_to_show)
        col4.metric("Displayed Columns", len(selected_columns))
        
        # Export options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Full Dataset (CSV)",
                data=csv,
                file_name=f"database_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_db_full"
            )
        
        with col2:
            if selected_columns:
                csv = df[selected_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Export Selected Columns (CSV)",
                    data=csv,
                    file_name=f"database_export_selected_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_db_selected"
                )
    
    with tab2:
        st.subheader("🔍 Search & Filter")
        
        # Keyword search
        st.markdown("### 🔎 Keyword Search")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_keywords = st.text_area(
                "Enter keywords (one per line)",
                height=100,
                help="Enter one keyword per line. Search will find rows containing these keywords."
            )
        
        with col2:
            search_columns = st.multiselect(
                "Search in columns",
                options=df.columns.tolist(),
                default=['title', 'abstract'] if 'title' in df.columns else []
            )
            
            search_mode = st.radio(
                "Match mode",
                options=['any', 'all'],
                index=0,
                help="'any': match ANY keyword (OR), 'all': match ALL keywords (AND)"
            )
        
        if st.button("🔍 Search"):
            if search_keywords and search_columns:
                keywords = [k.strip() for k in search_keywords.split('\n') if k.strip()]
                
                # Create search mask
                masks = []
                for keyword in keywords:
                    column_masks = []
                    for col in search_columns:
                        if col in df.columns:
                            column_masks.append(
                                df[col].astype(str).str.contains(keyword, case=False, na=False)
                            )
                    
                    if column_masks:
                        keyword_mask = pd.concat(column_masks, axis=1).any(axis=1)
                        masks.append(keyword_mask)
                
                if masks:
                    if search_mode == 'all':
                        final_mask = pd.concat(masks, axis=1).all(axis=1)
                    else:
                        final_mask = pd.concat(masks, axis=1).any(axis=1)
                    
                    filtered_df = df[final_mask]
                    st.success(f"✅ Found {len(filtered_df):,} papers")
                    
                    # Update session state with filtered data
                    save_state_for_undo("Database keyword search")
                    st.session_state.df_valid = filtered_df
                    st.rerun()
        
        st.markdown("---")
        
        # Filter by numeric range
        st.markdown("### 📊 Numeric Range Filter")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2, col3 = st.columns(3)
            with col1:
                numeric_col = st.selectbox("Select numeric column", options=numeric_cols)
            
            if numeric_col:
                col_min = float(df[numeric_col].min())
                col_max = float(df[numeric_col].max())
                
                with col2:
                    min_val = st.number_input("Min value", value=col_min, min_value=col_min, max_value=col_max)
                with col3:
                    max_val = st.number_input("Max value", value=col_max, min_value=col_min, max_value=col_max)
                
                if st.button("Apply Numeric Filter"):
                    filtered_df = df[(df[numeric_col] >= min_val) & (df[numeric_col] <= max_val)]
                    st.success(f"✅ Filtered to {len(filtered_df):,} papers")
                    
                    save_state_for_undo("Database numeric filter")
                    st.session_state.df_valid = filtered_df
                    st.rerun()
        
        st.markdown("---")
        
        # Filter by categorical values
        st.markdown("### 🏷️ Categorical Filter")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            cat_col = st.selectbox("Select categorical column", options=categorical_cols)
            
            if cat_col:
                unique_vals = df[cat_col].dropna().unique()
                if len(unique_vals) <= 100:  # Only show if not too many values
                    selected_vals = st.multiselect(
                        f"Select values from {cat_col}",
                        options=sorted(unique_vals.astype(str)),
                        help="Select one or more values to filter by"
                    )
                    
                    if st.button("Apply Categorical Filter"):
                        if selected_vals:
                            filtered_df = df[df[cat_col].isin(selected_vals)]
                            st.success(f"✅ Filtered to {len(filtered_df):,} papers")
                            
                            save_state_for_undo("Database categorical filter")
                            st.session_state.df_valid = filtered_df
                            st.rerun()
                else:
                    st.info(f"Too many unique values ({len(unique_vals)}) to display. Use keyword search instead.")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset to Original Dataset", type="secondary"):
            if st.session_state.df_filtered is not None:
                save_state_for_undo("Reset to filtered dataset")
                st.session_state.df_valid = st.session_state.df_filtered.copy()
                st.success("✅ Reset to filtered dataset")
                st.rerun()
    
    with tab3:
        st.subheader("📊 Column Statistics")
        
        col = st.selectbox("Select column for statistics", options=df.columns.tolist())
        
        if col:
            col_data = df[col]
            
            # Basic info
            st.write(f"**Data Type:** {col_data.dtype}")
            st.write(f"**Non-null Count:** {col_data.notna().sum():,} ({col_data.notna().sum()/len(col_data)*100:.1f}%)")
            st.write(f"**Null Count:** {col_data.isna().sum():,}")
            st.write(f"**Unique Values:** {col_data.nunique():,}")
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(col_data):
                st.markdown("**Numeric Statistics:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean", f"{col_data.mean():.4f}")
                col2.metric("Median", f"{col_data.median():.4f}")
                col3.metric("Std Dev", f"{col_data.std():.4f}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Min", f"{col_data.min():.4f}")
                col2.metric("Max", f"{col_data.max():.4f}")
                col3.metric("Range", f"{col_data.max() - col_data.min():.4f}")
                
                # Distribution plot
                fig = px.histogram(
                    df, 
                    x=col, 
                    nbins=50,
                    title=f"Distribution of {col}"
                )
                display_figure_with_export(fig, f"distribution_{col}", key=f"export_dist_{col}")
            
            # Categorical statistics
            else:
                st.markdown("**Top 10 Values:**")
                value_counts = to_native(col_data.value_counts().head(10))
                
                # Bar chart
                fig = px.bar(
                    x=value_counts.values,
                    y=[str(idx) for idx in value_counts.index],
                    orientation='h',
                    labels={'x': 'Count', 'y': col},
                    title=f"Top 10 values in {col}"
                )
                display_figure_with_export(fig, f"top_values_{col}", key=f"export_top_{col}")
                
                # Table
                st.dataframe(
                    pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(df) * 100).round(2)
                    }),
                    width="stretch"
                )
