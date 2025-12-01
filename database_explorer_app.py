"""
Streamlit Database Explorer App
View, filter, and search papers database with cross-reference to novelty analysis
"""
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Add utils path
sys.path.append(str(Path(__file__).parent))


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_filtered' not in st.session_state:
        st.session_state.df_filtered = None
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""
    if 'config' not in st.session_state:
        st.session_state.config = {}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV or Parquet file"""
    path = Path(data_path)
    
    if not path.exists():
        st.error(f"❌ File not found: {data_path}")
        return None
    
    try:
        with st.spinner("Loading dataset..."):
            if path.suffix.lower() in {'.parquet', '.pq'}:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            
            st.success(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
            return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


# ============================================================================
# FILTERING & SEARCH FUNCTIONS
# ============================================================================

def filter_by_keywords(df: pd.DataFrame, keywords: List[str], columns: List[str], 
                       mode: str = 'any') -> pd.DataFrame:
    """Filter dataframe by keywords in specified columns
    
    Args:
        df: Input dataframe
        keywords: List of keywords to search for
        columns: Columns to search in
        mode: 'any' (OR logic) or 'all' (AND logic)
    """
    if not keywords or not columns:
        return df
    
    # Create mask for each keyword across all specified columns
    masks = []
    for keyword in keywords:
        column_masks = []
        for col in columns:
            if col in df.columns:
                column_masks.append(
                    df[col].astype(str).str.contains(keyword, case=False, na=False)
                )
        
        if column_masks:
            # Combine column masks with OR
            keyword_mask = pd.concat(column_masks, axis=1).any(axis=1)
            masks.append(keyword_mask)
    
    if not masks:
        return df
    
    # Combine keyword masks based on mode
    if mode == 'all':
        final_mask = pd.concat(masks, axis=1).all(axis=1)
    else:  # mode == 'any'
        final_mask = pd.concat(masks, axis=1).any(axis=1)
    
    return df[final_mask]


def filter_by_column_values(df: pd.DataFrame, column: str, values: List[Any]) -> pd.DataFrame:
    """Filter dataframe by specific values in a column"""
    if not values or column not in df.columns:
        return df
    
    return df[df[column].isin(values)]


def filter_by_numeric_range(df: pd.DataFrame, column: str, 
                            min_val: Optional[float] = None, 
                            max_val: Optional[float] = None) -> pd.DataFrame:
    """Filter dataframe by numeric range"""
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    if min_val is not None:
        result = result[result[column] >= min_val]
    if max_val is not None:
        result = result[result[column] <= max_val]
    
    return result


def filter_by_paper_ids(df: pd.DataFrame, paper_ids: List[str], id_column: str = 'pmid') -> pd.DataFrame:
    """Filter dataframe by specific paper IDs"""
    if not paper_ids or id_column not in df.columns:
        return df
    
    # Convert to string for comparison
    paper_ids_str = [str(pid).strip() for pid in paper_ids]
    return df[df[id_column].astype(str).isin(paper_ids_str)]


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_dataframe_with_options(df: pd.DataFrame, selected_columns: List[str]):
    """Display dataframe with customization options"""
    if df is None or df.empty:
        st.warning("⚠️ No data to display")
        return
    
    # Column selection
    display_cols = selected_columns if selected_columns else df.columns.tolist()
    
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
        sortable = st.checkbox("Enable sorting", value=True)
    
    # Display dataframe
    st.dataframe(
        df[display_cols].head(rows_to_show),
        use_container_width=True,
        hide_index=not show_index,
        height=600
    )
    
    # Display summary stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Displayed Rows", rows_to_show)
    col4.metric("Displayed Columns", len(display_cols))


def display_column_statistics(df: pd.DataFrame, column: str):
    """Display statistics for a specific column"""
    if column not in df.columns:
        st.warning(f"Column '{column}' not found")
        return
    
    st.subheader(f"📊 Statistics: {column}")
    
    col_data = df[column]
    
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
            x=column, 
            nbins=50,
            title=f"Distribution of {column}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical statistics
    else:
        st.markdown("**Top 10 Values:**")
        value_counts = col_data.value_counts().head(10)
        
        # Bar chart
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index.astype(str),
            orientation='h',
            labels={'x': 'Count', 'y': column},
            title=f"Top 10 values in {column}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(
            pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            }),
            use_container_width=True
        )


# ============================================================================
# MAIN PAGE: DATABASE EXPLORER
# ============================================================================

def page_explorer():
    """Main database explorer page"""
    st.title("🔍 Database Explorer")
    
    if st.session_state.df_original is None:
        st.warning("⚠️ Please load data first from the sidebar")
        return
    
    df_display = st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df_original
    
    # Main content area
    st.markdown(f"""
    **Dataset**: {len(df_display):,} papers (filtered from {len(st.session_state.df_original):,} total)
    """)
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📋 Data View", "🔍 Search & Filter", "📊 Column Stats"])
    
    with tab1:
        st.subheader("Data Table")
        
        # Column selector
        all_columns = df_display.columns.tolist()
        
        # Suggest important columns at the top
        suggested_cols = []
        for col in ['pmid', 'title', 'abstract', 'year', 'authors', 'journal', 
                   'cluster_label', 'gap_score', 'density_k20', 'umap_x', 'umap_y']:
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
        
        st.session_state.selected_columns = selected_columns
        
        # Display dataframe
        display_dataframe_with_options(df_display, selected_columns)
        
        # Export options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export to CSV"):
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"filtered_papers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Export Selected Columns"):
                if selected_columns:
                    csv = df_display[selected_columns].to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"filtered_papers_selected_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    with tab2:
        st.subheader("🔍 Search & Filter")
        
        # Reset filters button
        if st.button("🔄 Reset All Filters"):
            st.session_state.df_filtered = st.session_state.df_original.copy()
            st.rerun()
        
        st.markdown("---")
        
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
            # Detect ID column
            id_columns = [c for c in df_display.columns if c in ['pmid', 'id', 'doi', 'paper_id']]
            default_id = id_columns[0] if id_columns else 'pmid'
            
            search_columns = st.multiselect(
                "Search in columns",
                options=df_display.columns.tolist(),
                default=['title', 'abstract'] if 'title' in df_display.columns else []
            )
            
            search_mode = st.radio(
                "Match mode",
                options=['any', 'all'],
                index=0,
                help="'any': match ANY keyword (OR), 'all': match ALL keywords (AND)"
            )
        
        if st.button("🔍 Apply Keyword Search"):
            if search_keywords and search_columns:
                keywords = [k.strip() for k in search_keywords.split('\n') if k.strip()]
                df_current = st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df_original
                st.session_state.df_filtered = filter_by_keywords(
                    df_current, 
                    keywords, 
                    search_columns, 
                    search_mode
                )
                st.success(f"✅ Found {len(st.session_state.df_filtered):,} papers")
                st.rerun()
        
        st.markdown("---")
        
        # Filter by paper IDs
        st.markdown("### 🆔 Filter by Paper IDs")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            paper_ids_input = st.text_area(
                "Enter paper IDs (one per line)",
                height=100,
                help="Enter PMIDs or other paper IDs, one per line"
            )
        
        with col2:
            id_column = st.selectbox(
                "ID Column",
                options=id_columns if id_columns else df_display.columns.tolist(),
                index=0
            )
        
        if st.button("🔍 Filter by IDs"):
            if paper_ids_input:
                paper_ids = [pid.strip() for pid in paper_ids_input.split('\n') if pid.strip()]
                df_current = st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df_original
                st.session_state.df_filtered = filter_by_paper_ids(
                    df_current,
                    paper_ids,
                    id_column
                )
                st.success(f"✅ Found {len(st.session_state.df_filtered):,} papers")
                st.rerun()
        
        st.markdown("---")
        
        # Filter by categorical column
        st.markdown("### 🏷️ Filter by Category")
        
        col1, col2 = st.columns(2)
        with col1:
            # Find categorical columns
            categorical_cols = []
            for col in df_display.columns:
                if df_display[col].dtype == 'object' or df_display[col].nunique() < 100:
                    categorical_cols.append(col)
            
            if categorical_cols:
                filter_column = st.selectbox(
                    "Select column",
                    options=categorical_cols
                )
                
                unique_values = sorted(df_display[filter_column].dropna().unique())
                selected_values = st.multiselect(
                    f"Select {filter_column} values",
                    options=unique_values,
                    default=[]
                )
                
                if st.button("🔍 Apply Category Filter"):
                    if selected_values:
                        df_current = st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df_original
                        st.session_state.df_filtered = filter_by_column_values(
                            df_current,
                            filter_column,
                            selected_values
                        )
                        st.success(f"✅ Filtered to {len(st.session_state.df_filtered):,} papers")
                        st.rerun()
        
        st.markdown("---")
        
        # Filter by numeric range
        st.markdown("### 📊 Filter by Numeric Range")
        
        numeric_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                numeric_column = st.selectbox(
                    "Select numeric column",
                    options=numeric_cols
                )
            
            if numeric_column:
                col_min = float(df_display[numeric_column].min())
                col_max = float(df_display[numeric_column].max())
                
                with col2:
                    range_min = st.number_input(
                        "Minimum value",
                        value=col_min,
                        min_value=col_min,
                        max_value=col_max
                    )
                
                with col3:
                    range_max = st.number_input(
                        "Maximum value",
                        value=col_max,
                        min_value=col_min,
                        max_value=col_max
                    )
                
                if st.button("🔍 Apply Range Filter"):
                    df_current = st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df_original
                    st.session_state.df_filtered = filter_by_numeric_range(
                        df_current,
                        numeric_column,
                        range_min,
                        range_max
                    )
                    st.success(f"✅ Filtered to {len(st.session_state.df_filtered):,} papers")
                    st.rerun()
    
    with tab3:
        st.subheader("📊 Column Statistics")
        
        stat_column = st.selectbox(
            "Select column to analyze",
            options=df_display.columns.tolist()
        )
        
        if stat_column:
            display_column_statistics(df_display, stat_column)


# ============================================================================
# PAGE: DATA LOADING
# ============================================================================

def page_data_loading():
    """Data loading configuration page"""
    st.title("📁 Data Loading")
    
    st.markdown("""
    Load your papers database to explore and cross-reference with novelty analysis results.
    """)
    
    # File path input
    data_path = st.text_input(
        "Data File Path",
        value=r"C:\Users\20195435\OneDrive - TU Eindhoven\TUe\Playground\Nanotechnology\papers_dataframe_full_processed_with_processed_embeddings_parsed.csv",
        help="Path to your CSV or Parquet file"
    )
    
    # Load button
    if st.button("🚀 Load Data", type="primary"):
        df = load_data(data_path)
        if df is not None:
            st.session_state.df_original = df
            st.session_state.df_filtered = df.copy()
            st.session_state.config['data_path'] = data_path
            st.rerun()
    
    # Show data preview if loaded
    if st.session_state.df_original is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.df_original):,} papers")
        
        with st.expander("📊 Data Preview & Information"):
            st.markdown("### First 10 rows")
            st.dataframe(st.session_state.df_original.head(10), use_container_width=True)
            
            st.markdown("### Column Information")
            col_info = pd.DataFrame({
                'Column': st.session_state.df_original.columns,
                'Type': st.session_state.df_original.dtypes.values,
                'Non-Null': st.session_state.df_original.notna().sum().values,
                'Null': st.session_state.df_original.isna().sum().values,
                'Unique': [st.session_state.df_original[col].nunique() for col in st.session_state.df_original.columns]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    st.set_page_config(
        page_title="Database Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Database Explorer")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate",
            [
                "📁 Load Data",
                "🔍 Explore Database"
            ]
        )
        
        st.markdown("---")
        
        # Status indicators
        st.markdown("### 📌 Status")
        st.write("✅" if st.session_state.df_original is not None else "⬜", "Data Loaded")
        st.write("✅" if st.session_state.df_filtered is not None else "⬜", "Filters Applied")
        
        st.markdown("---")
        
        # Dataset info
        if st.session_state.df_original is not None:
            st.metric("Total Papers", f"{len(st.session_state.df_original):,}")
            if st.session_state.df_filtered is not None:
                st.metric("Filtered Papers", f"{len(st.session_state.df_filtered):,}")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Database Explorer**
            
            This app allows you to:
            - Load and view your papers database
            - Filter by keywords, IDs, categories, or numeric ranges
            - Cross-reference with novelty analysis results
            - Export filtered results
            - View column statistics
            
            **Tips:**
            - Use Paper ID filter to cross-reference with LLM analysis
            - Combine multiple filters for precise results
            - Export filtered results for further analysis
            """)
    
    # Route to pages
    if page == "📁 Load Data":
        page_data_loading()
    elif page == "🔍 Explore Database":
        page_explorer()


if __name__ == "__main__":
    main()
