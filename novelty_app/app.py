"""
Main entry point for the refactored Novelty Analysis App

This is a transitional version that imports from the original file
while allowing gradual migration to the new modular structure.

Usage:
    streamlit run novelty_app/app.py
"""
import sys
from pathlib import Path

# Add parent directory to path to import original functions
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from original file (temporary during migration)
from novelty_analysis_app import *

# Import our new modular components
try:
    from core import init_session_state, save_state_for_undo, undo_last_action
    print("✅ Using modular core functions")
except ImportError:
    print("⚠️  Using legacy core functions")
    pass

# Main app function
def main():
    st.set_page_config(
        page_title="Novelty Analysis App",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🔬 Novelty Analysis")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            [
                "📊 Data & Config",
                "🧬 Embeddings",
                "🎯 Filters",
                "🔬 Clustering",
                "🔍 Gap Analysis",
                "🌉 Gap Regions",
                "🤖 LLM Analysis",
                "📚 Database Explorer",
                "💾 Export"
            ]
        )
        
        st.markdown("---")
        
        # Status indicators
        st.markdown("### 📌 Status")
        st.write("✅" if st.session_state.df_filtered is not None else "⬜", "Data Loaded")
        st.write("✅" if st.session_state.embeddings_extracted else "⬜", "Embeddings Extracted")
        st.write("✅" if st.session_state.X_pca is not None else "⬜", "PCA Reduction")
        st.write("✅" if st.session_state.X_umap_2d is not None else "⬜", "UMAP Projection")
        st.write("✅" if st.session_state.clustering_done else "⬜", "Clustering Done")
        st.write("✅" if st.session_state.density_computed else "⬜", "Density Computed")
        st.write("✅" if st.session_state.gaps_identified else "⬜", "Gaps Identified")
        
        st.markdown("---")
        
        # Undo button
        if st.session_state.undo_history:
            last_action = st.session_state.undo_history[-1]['action_name']
            if st.button(f"↩️ Undo: {last_action}", use_container_width=True):
                if undo_last_action():
                    st.success(f"Undone: {last_action}")
                    st.rerun()
        else:
            st.button("↩️ Undo (no actions)", disabled=True, use_container_width=True)
        
        st.markdown("---")
        
        if st.session_state.df_valid is not None:
            st.metric("Current Papers", len(st.session_state.df_valid))
    
    # Route to pages
    if page == "📊 Data & Config":
        page_data_loading()
    elif page == "🧬 Embeddings":
        page_embedding_processing()
    elif page == "🎯 Filters":
        page_filters()
    elif page == "🔬 Clustering":
        page_clustering()
    elif page == "🔍 Gap Analysis":
        page_gap_analysis()
    elif page == "🌉 Gap Regions":
        page_gap_regions()
    elif page == "🤖 LLM Analysis":
        page_llm_analysis()
    elif page == "📚 Database Explorer":
        page_database_explorer()
    elif page == "💾 Export":
        page_export()


if __name__ == "__main__":
    main()
