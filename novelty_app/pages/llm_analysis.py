"""
LLM Analysis Page - AI-powered contrastive cluster analysis for gap explanation
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise

try:
    from agents.observability import observe_current
except Exception:  # pragma: no cover
    from novelty_app.agents.observability import observe_current

from ui.export_utils import display_figure_with_export

# Check for OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def page_llm_analysis():
    """LLM-based gap explanation with contrastive cluster analysis"""
    st.title("🤖 LLM Gap Analysis")
    
    if not st.session_state.gaps_identified:
        st.warning("⚠️ Please identify gap regions first")
        return
    
    if not OPENAI_AVAILABLE:
        st.error("❌ OpenAI package not available")
        return
    
    gap_regions = st.session_state.gap_regions
    
    if not gap_regions:
        st.warning("No gap regions found")
        return
    
    st.markdown("""
    Generate evidence-grounded contrastive explanations for gap regions by comparing neighboring clusters.
    The LLM identifies key differences, salient entities, and potential bridge opportunities.
    """)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        openai_model = st.selectbox("Model", ["gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-5.2-2025-12-11"], index=0)
    with col2:
        region_id = st.selectbox("Select Gap Region", range(len(gap_regions)))
        use_all_gap_regions = st.checkbox("Use ALL Gap Regions", value=False, 
                                          help="Include papers from all gap regions instead of just the selected one.")
        n_papers_per_cluster = st.number_input("Papers per Cluster", min_value=5, max_value=100, value=15)
    with col3:
        use_all_gap_papers = st.checkbox("Use ALL Gap Papers (Use with care)", value=False,
                                         help="Include all gap papers (ignores number limit below)")
        n_gap_papers = st.number_input("Gap Papers to Include", min_value=5, max_value=100, value=5,
                                       disabled=use_all_gap_papers,
                                       help="Number of gap region papers to include in evidence pack (sorted by gap score)")
    
    # Custom question and keywords
    st.markdown("---")
    st.subheader("💡 Additional Guidance (Optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        custom_question = st.text_area(
            "Specific Question",
            value="",
            height=100,
            help="Optional: Ask a specific question that the LLM should try to answer based on the evidence"
        )
    with col2:
        guidance_keywords = st.text_area(
            "Keywords for Bridge Opportunities",
            value="",
            height=100,
            help="Optional: Enter keywords (comma-separated) to guide the LLM when identifying bridge opportunities"
        )
    
    # Show region preview
    if use_all_gap_regions:
        # Calculate stats across all gap regions
        all_gap_indices = [idx for region in gap_regions for idx in region]
        region_df = st.session_state.df_valid.loc[all_gap_indices]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Gap Papers (All Regions)", len(all_gap_indices))
        col1.caption(f"From {len(gap_regions)} regions")
        if 'gap_score' in region_df.columns:
            col2.metric("Avg Gap Score", f"{region_df['gap_score'].mean():.3f}")
        if st.session_state.selected_clustering and f'cluster_{st.session_state.selected_clustering}' in st.session_state.df_valid.columns:
            cluster_col = f'cluster_{st.session_state.selected_clustering}'
            col3.metric("Clusters Touched", region_df[cluster_col].nunique())
    else:
        # Single region stats
        region_indices = gap_regions[region_id]
        region_df = st.session_state.df_valid.loc[region_indices]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Gap Region Papers", len(region_indices))
        if 'gap_score' in region_df.columns:
            col2.metric("Avg Gap Score", f"{region_df['gap_score'].mean():.3f}")
        if st.session_state.selected_clustering and f'cluster_{st.session_state.selected_clustering}' in st.session_state.df_valid.columns:
            cluster_col = f'cluster_{st.session_state.selected_clustering}'
            col3.metric("Clusters Touched", region_df[cluster_col].nunique())
    
    # Cluster selection - allow ANY clusters to be compared, not just those in gap region
    st.markdown("---")
    st.subheader("🎯 Select Clusters to Contrast")
    
    st.info("Select any two clusters from your dataset. The gap region above will be used as contextual evidence.")
    
    # Visualization to help choose clusters
    if st.session_state.selected_clustering and st.session_state.X_umap_2d is not None:
        cluster_col = f'cluster_{st.session_state.selected_clustering}'
        if cluster_col in st.session_state.df_valid.columns:
            st.markdown("#### 📊 Cluster Overview with Gap Region")
            
            fig = go.Figure()
            
            # Plot all clusters
            df_plot = st.session_state.df_valid.copy()
            df_plot['umap_x'] = st.session_state.X_umap_2d[:, 0]
            df_plot['umap_y'] = st.session_state.X_umap_2d[:, 1]
            
            # Color by cluster
            unique_clusters = sorted(df_plot[cluster_col].unique())
            colors_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise cluster
                    continue
                    
                cluster_data = df_plot[df_plot[cluster_col] == cluster_id]
                color_idx = cluster_id % len(colors_palette)
                
                fig.add_trace(go.Scatter(
                    x=cluster_data['umap_x'],
                    y=cluster_data['umap_y'],
                    mode='markers',
                    marker=dict(size=6, color=colors_palette[color_idx], opacity=0.6),
                    name=f'Cluster {cluster_id} (n={len(cluster_data)})',
                    showlegend=True,
                    hovertemplate=f'Cluster {cluster_id}<extra></extra>'
                ))
            
            # Overlay gap region as red stars
            if use_all_gap_regions:
                # Show all gap regions
                all_gap_indices = [idx for region in gap_regions for idx in region]
                gap_data = df_plot.loc[all_gap_indices]
                region_label = f'All Gap Regions'
            else:
                # Show single selected region
                gap_data = df_plot.loc[region_indices]
                region_label = f'Gap Region {region_id}'
            
            # Create hover text with title and abstract
            gap_hover_text = [
                f"<b>{row.get('title', 'N/A')}</b><br>" +
                f"Gap Region: {row.get('gap_region', 'N/A')}<br>" +
                f"Gap Score: {row.get('gap_score', 0):.3f}<br>" +
                f"{str(row.get('abstract', row.get('processed_content', '')))[:200]}..."
                for _, row in gap_data.iterrows()
            ]
            
            fig.add_trace(go.Scatter(
                x=gap_data['umap_x'],
                y=gap_data['umap_y'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star', 
                           line=dict(color='darkred', width=1)),
                name=f'{region_label} (n={len(gap_data)})',
                showlegend=True,
                text=gap_hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Clusters with {region_label} Highlighted',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                height=600,
                hovermode='closest',
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            )
            
            display_figure_with_export(fig, "llm_gap_clusters", key="export_llm_clusters")
            st.caption("💡 Select two clusters from the legend above to contrast in the analysis")
    
    # Get all available clusters from the selected clustering method
    if st.session_state.selected_clustering:
        cluster_col = f'cluster_{st.session_state.selected_clustering}'
        if cluster_col in st.session_state.df_valid.columns:
            all_cluster_ids = sorted(st.session_state.df_valid[cluster_col].unique())
            all_cluster_ids = [c for c in all_cluster_ids if c != -1]  # Remove noise cluster if present
            
            # Count papers in each cluster
            cluster_sizes = st.session_state.df_valid[cluster_col].value_counts().to_dict()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                cluster_A_selected = st.selectbox(
                    "Cluster A",
                    all_cluster_ids,
                    index=0 if all_cluster_ids else None,
                    format_func=lambda x: f"Cluster {x} (n={cluster_sizes.get(x, 0)} papers)",
                    key="cluster_a_select"
                )
            with col2:
                # Filter out Cluster A from options for Cluster B
                cluster_b_options = [c for c in all_cluster_ids if c != cluster_A_selected]
                cluster_B_selected = st.selectbox(
                    "Cluster B",
                    cluster_b_options,
                    index=0 if cluster_b_options else None,
                    format_func=lambda x: f"Cluster {x} (n={cluster_sizes.get(x, 0)} papers)",
                    key="cluster_b_select"
                )
            with col3:
                # Filter out Cluster A and B from options for Cluster C (optional)
                cluster_c_options = [c for c in all_cluster_ids if c not in [cluster_A_selected, cluster_B_selected]]
                # Add None as first option (default)
                cluster_c_display_options = [None] + cluster_c_options
                cluster_C_selected = st.selectbox(
                    "Cluster C (Optional)",
                    cluster_c_display_options,
                    index=0,  # Default to None
                    format_func=lambda x: "None" if x is None else f"Cluster {x} (n={cluster_sizes.get(x, 0)} papers)",
                    key="cluster_c_select",
                    help="Optional third cluster to include in comparison"
                )
            
            if cluster_A_selected is not None and cluster_B_selected is not None:
                # Build comparison message
                if cluster_C_selected is not None:
                    comparison_msg = f"✅ Ready to compare Cluster {cluster_A_selected} ({cluster_sizes.get(cluster_A_selected, 0)} papers) vs Cluster {cluster_B_selected} ({cluster_sizes.get(cluster_B_selected, 0)} papers) vs Cluster {cluster_C_selected} ({cluster_sizes.get(cluster_C_selected, 0)} papers)"
                else:
                    comparison_msg = f"✅ Ready to compare Cluster {cluster_A_selected} ({cluster_sizes.get(cluster_A_selected, 0)} papers) vs Cluster {cluster_B_selected} ({cluster_sizes.get(cluster_B_selected, 0)} papers)"
                
                if use_all_gap_regions:
                    all_gap_indices = [idx for region in gap_regions for idx in region]
                    st.success(comparison_msg)
                    st.caption(f"Using ALL Gap Regions ({len(all_gap_indices)} papers from {len(gap_regions)} regions) as contextual evidence")
                else:
                    st.success(comparison_msg)
                    st.caption(f"Using Gap Region {region_id} ({len(region_indices)} papers) as contextual evidence")
        else:
            st.error(f"❌ Clustering column '{cluster_col}' not found")
            cluster_A_selected = None
            cluster_B_selected = None
    else:
        st.error("❌ No clustering method selected. Please complete clustering first.")
        cluster_A_selected = None
        cluster_B_selected = None
    
    
    if st.button("🚀 Prepare and Review Prompt", type="primary"):
        openai_api_key = st.session_state.get('openai_api_key', os.environ.get('OPENAI_API_KEY', ''))
        if not openai_api_key:
            st.error("Please provide OpenAI API key in Data & Config page")
            return
        
        if cluster_A_selected is None or cluster_B_selected is None:
            st.error("Please select two clusters to compare")
            return
        
        # Generate the prompt and store in session state
        prepare_llm_prompt(
            region_id, 
            n_papers_per_cluster,
            n_gap_papers,
            custom_question.strip() if custom_question.strip() else None,
            guidance_keywords.strip() if guidance_keywords.strip() else None,
            cluster_A_selected,
            cluster_B_selected,
            cluster_C_selected,
            use_all_gap_regions,
            use_all_gap_papers
        )
    
    # Display prompt editor if prompts have been generated
    st.markdown("---")
    if 'llm_prompts' in st.session_state and st.session_state.llm_prompts is not None:
        prompt_data = st.session_state.llm_prompts
        
        st.markdown("### 📝 Generated Prompts - Review and Edit")
        
        # Initialize prompt storage in session state if needed
        if 'edited_system_prompt' not in st.session_state:
            st.session_state.edited_system_prompt = prompt_data['system_prompt']
        if 'edited_user_prompt' not in st.session_state:
            st.session_state.edited_user_prompt = prompt_data['user_prompt']
        
        with st.expander("🔧 System Prompt", expanded=True):
            st.session_state.edited_system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.edited_system_prompt,
                height=150,
                key="system_prompt_editor",
                label_visibility="collapsed"
            )
        
        with st.expander("📋 User Prompt", expanded=True):
            st.session_state.edited_user_prompt = st.text_area(
                "User Prompt",
                value=st.session_state.edited_user_prompt,
                height=400,
                key="user_prompt_editor",
                label_visibility="collapsed"
            )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Reset Prompts", help="Reset to original prompts"):
                st.session_state.edited_system_prompt = prompt_data['system_prompt']
                st.session_state.edited_user_prompt = prompt_data['user_prompt']
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", help="Clear prompts and start over"):
                st.session_state.llm_prompts = None
                if 'edited_system_prompt' in st.session_state:
                    del st.session_state['edited_system_prompt']
                if 'edited_user_prompt' in st.session_state:
                    del st.session_state['edited_user_prompt']
                st.rerun()
        
        with col3:
            if st.button("✅ Send Edited Prompts to LLM", type="primary"):
                openai_api_key = st.session_state.get('openai_api_key', os.environ.get('OPENAI_API_KEY', ''))
                if not openai_api_key:
                    st.error("Please provide OpenAI API key in Data & Config page")
                    return
                
                # Call the LLM with edited prompts
                send_llm_prompt(
                    openai_api_key,
                    openai_model,
                    st.session_state.edited_system_prompt,
                    st.session_state.edited_user_prompt,
                    prompt_data['region_id'],
                    prompt_data['cluster_A'],
                    prompt_data['cluster_B'],
                    prompt_data.get('cluster_C'),
                    prompt_data['region_size']
                )
    
    # Display analysis results if available
    elif st.session_state.llm_results is not None:
        llm_data = st.session_state.llm_results
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 Analysis Results")
            st.caption(f"Generated: {llm_data.get('timestamp', 'Unknown')} | Model: {llm_data.get('model', 'Unknown')}")
        with col2:
            if st.button("🗑️ Clear", key="clear_llm_results"):
                st.session_state.llm_results = None
                st.rerun()
        
        display_llm_results(
            llm_data['result'],
            llm_data['region_id'],
            llm_data['cluster_A'],
            llm_data['cluster_B'],
            llm_data.get('cluster_C'),
            llm_data['region_size']
        )
    else:
        st.info("💡 Click 'Prepare and Review Prompt' above to generate analysis prompts.")


def prepare_llm_prompt(region_id, n_papers, n_gap_papers, custom_question=None, guidance_keywords=None, cluster_A=None, cluster_B=None, cluster_C=None, use_all_gap_regions=False, use_all_gap_papers=False):
    """Generate and store prompts without sending to LLM yet"""
    gap_regions = st.session_state.gap_regions
    
    # Get gap region papers based on selection
    if use_all_gap_regions:
        # Collect all papers from all gap regions
        all_gap_indices = [idx for region in gap_regions for idx in region]
        region_indices = all_gap_indices
        region_df = st.session_state.df_valid.loc[region_indices]
        region_label = f"All Gap Regions ({len(gap_regions)} regions)"
    else:
        # Single region
        region_indices = gap_regions[region_id]
        region_df = st.session_state.df_valid.loc[region_indices]
        region_label = f"Gap Region {region_id}"
    
    # Validate cluster selection
    if cluster_A is None or cluster_B is None:
        st.error("❌ At least two clusters (A and B) must be selected")
        return
    
    # Get cluster column based on selected clustering method
    cluster_col = f'cluster_{st.session_state.selected_clustering}'
    if cluster_col not in st.session_state.df_valid.columns:
        st.error(f"❌ Clustering column '{cluster_col}' not found")
        return
    
    try:
        # Get representative papers from each cluster (closest to centroid)
        X_primary = st.session_state.X_primary
        
        # Get cluster sizes for the two selected clusters
        cluster_counts = st.session_state.df_valid[cluster_col].value_counts().to_dict()
        
        # Cluster A papers
        idx_A = np.where(st.session_state.df_valid[cluster_col] == cluster_A)[0]
        X_A = X_primary[idx_A]
        centroid_A = X_A.mean(axis=0, keepdims=True)
        dists_A = pairwise.cosine_distances(X_A, centroid_A).ravel()
        top_A_local = np.argsort(dists_A)[:n_papers]
        top_A_idx = idx_A[top_A_local]
        
        # Cluster B papers
        idx_B = np.where(st.session_state.df_valid[cluster_col] == cluster_B)[0]
        X_B = X_primary[idx_B]
        centroid_B = X_B.mean(axis=0, keepdims=True)
        dists_B = pairwise.cosine_distances(X_B, centroid_B).ravel()
        top_B_local = np.argsort(dists_B)[:n_papers]
        top_B_idx = idx_B[top_B_local]
        
        # Cluster C papers (optional)
        top_C_idx = None
        if cluster_C is not None:
            idx_C = np.where(st.session_state.df_valid[cluster_col] == cluster_C)[0]
            if len(idx_C) > 0:
                X_C = X_primary[idx_C]
                centroid_C = X_C.mean(axis=0, keepdims=True)
                dists_C = pairwise.cosine_distances(X_C, centroid_C).ravel()
                top_C_local = np.argsort(dists_C)[:n_papers]
                top_C_idx = idx_C[top_C_local]
        
        # Build evidence pack
        evidence_pack = []
        for idx in top_A_idx:
            row = st.session_state.df_valid.iloc[idx]
            paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
            evidence_pack.append({
                "doc_id": f"A_{paper_id}",
                "paper_id": str(paper_id),
                "title": str(row.get('title', '')),
                "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                "cluster": "A"
            })
        
        for idx in top_B_idx:
            row = st.session_state.df_valid.iloc[idx]
            paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
            evidence_pack.append({
                "doc_id": f"B_{paper_id}",
                "paper_id": str(paper_id),
                "title": str(row.get('title', '')),
                "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                "cluster": "B"
            })
        
        # Cluster C papers (if provided)
        if top_C_idx is not None:
            for idx in top_C_idx:
                row = st.session_state.df_valid.iloc[idx]
                paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
                evidence_pack.append({
                    "doc_id": f"C_{paper_id}",
                    "paper_id": str(paper_id),
                    "title": str(row.get('title', '')),
                    "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                    "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                    "cluster": "C"
                })
        
        # Gap region papers - top N papers sorted by gap_score (highest first)
        region_df_sorted = region_df.sort_values('gap_score', ascending=False) if 'gap_score' in region_df.columns else region_df
        n_gap_to_include = len(region_df_sorted) if use_all_gap_papers else min(n_gap_papers, len(region_df_sorted))
        
        for i, (idx, row) in enumerate(region_df_sorted.iterrows()):
            if i >= n_gap_to_include:
                break
            paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
            gap_cluster = row.get(cluster_col, -1)
            evidence_pack.append({
                "doc_id": f"GAP_{paper_id}",
                "paper_id": str(paper_id),
                "title": str(row.get('title', '')),
                "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                "cluster": "GAP",
                "gap_score": float(row.get('gap_score', 0)),
                "assigned_cluster": int(gap_cluster) if gap_cluster != -1 else None
            })
        
        # Build system prompt
        system_prompt = """You are a nanomedicine domain expert. Only use the EVIDENCE PACK provided.
Never invent facts or cite outside sources. If evidence is insufficient for any claim,
state 'unknown'. Cite by doc_id for every claim. Output exactly the JSON schema."""
        
        # Build additional guidance sections
        custom_question_section = ""
        if custom_question:
            custom_question_section = f"""

SPECIFIC QUESTION TO ADDRESS:
{custom_question}

Please include your answer to this question in a dedicated field called "custom_question_answer" in the JSON output.
Base your answer strictly on the evidence provided. If the evidence is insufficient, state this clearly.
"""
        
        keywords_guidance_section = ""
        if guidance_keywords:
            keywords_list = [kw.strip() for kw in guidance_keywords.split(',') if kw.strip()]
            if keywords_list:
                keywords_guidance_section = f"""

KEYWORDS FOR BRIDGE OPPORTUNITIES:
When identifying bridge opportunities, pay special attention to these keywords and concepts: {', '.join(keywords_list)}
Consider how these keywords might relate to potential connections between the two clusters.
"""
        
        # Build output schema with optional cluster C and custom question fields
        output_schema = """{
"cluster_A_summary": {
    "one_line": "string",
    "bullets": ["string"],
    "salient_entities": {"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]},
    "citations": ["doc_id"]
},
"cluster_B_summary": {
    "one_line": "string",
    "bullets": ["string"],
    "salient_entities": {"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]},
    "citations": ["doc_id"]
},"""
        
        if cluster_C is not None:
            output_schema += """
"cluster_C_summary": {
    "one_line": "string",
    "bullets": ["string"],
    "salient_entities": {"materials":[], "ligands":[], "diseases":[], "delivery":[], "models":[]},
    "citations": ["doc_id"]
},"""
        
        output_schema += """
"axes_of_separation": [{
    "axis": "materials|ligands|disease|model|delivery|toxicity|methods|other",
    "what_differs": "short explanation (evidence-grounded)",
    "evidence_A": ["doc_id"],
    "evidence_B": ["doc_id"],"""
        
        if cluster_C is not None:
            output_schema += """
    "evidence_C": ["doc_id"],"""
        
        output_schema += """
    "confidence": 0.0-1.0
}],
"bridge_seeds": [{
    "idea": "short description of a possible bridge",
    "why_plausible": "mechanistic rationale, grounded in docs",
    "support": ["doc_id"],
    "risks": ["toxicity","aggregation","RES","immunogenicity","scaleup","IP","assay_limitations"]
}],"""
        
        if custom_question:
            output_schema += """
"custom_question_answer": {
    "answer": "string",
    "supporting_evidence": ["doc_id"],
    "confidence": 0.0-1.0,
    "limitations": "string"
},"""
        
        output_schema += """
"insufficient_evidence": false
}"""
        
        # Build task description based on number of clusters
        if cluster_C is not None:
            task_desc = f"""TASK: Contrast Cluster A vs Cluster B vs Cluster C to explain why they are separated in embedding space.
Focus on: materials, surface chemistry/coatings, size/shape, targeting ligands, disease areas,
models (in vitro/in vivo/clinical), delivery routes, pharmacokinetics/biodistribution,
toxicity/regulatory language, endpoints/outcomes.

CONTEXT:
- cluster_A_meta: {{"id": {cluster_A}, "n_docs": {cluster_counts[cluster_A]}}}
- cluster_B_meta: {{"id": {cluster_B}, "n_docs": {cluster_counts[cluster_B]}}}
- cluster_C_meta: {{"id": {cluster_C}, "n_docs": {cluster_counts[cluster_C]}}}
- {region_label}: {len(region_indices)} total papers with low density (potential research opportunities)

The gap papers are included in the evidence pack below. Use them to understand what research lies
between the clusters and identify bridge opportunities."""
            evidence_desc = f"""EVIDENCE PACK (JSONL; each line is one doc):
- Papers with cluster="A": Top {n_papers} representative papers from Cluster A (closest to centroid)
- Papers with cluster="B": Top {n_papers} representative papers from Cluster B (closest to centroid)
- Papers with cluster="C": Top {n_papers} representative papers from Cluster C (closest to centroid)
- Papers with cluster="GAP": Top {n_gap_to_include} gap papers from {region_label} (sorted by gap_score, highest first)"""
        else:
            task_desc = f"""TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
Focus on: materials, surface chemistry/coatings, size/shape, targeting ligands, disease areas,
models (in vitro/in vivo/clinical), delivery routes, pharmacokinetics/biodistribution,
toxicity/regulatory language, endpoints/outcomes.

CONTEXT:
- cluster_A_meta: {{"id": {cluster_A}, "n_docs": {cluster_counts[cluster_A]}}}
- cluster_B_meta: {{"id": {cluster_B}, "n_docs": {cluster_counts[cluster_B]}}}
- {region_label}: {len(region_indices)} total papers with low density (potential research opportunities)

The gap papers are included in the evidence pack below. Use them to understand what research lies
between the two clusters and identify bridge opportunities."""
            evidence_desc = f"""EVIDENCE PACK (JSONL; each line is one doc):
- Papers with cluster="A": Top {n_papers} representative papers from Cluster A (closest to centroid)
- Papers with cluster="B": Top {n_papers} representative papers from Cluster B (closest to centroid)
- Papers with cluster="GAP": Top {n_gap_to_include} gap papers from {region_label} (sorted by gap_score, highest first)"""
        
        user_prompt = f"""{task_desc}

{custom_question_section}{keywords_guidance_section}

{evidence_desc}

```jsonl
{chr(10).join(json.dumps(d, ensure_ascii=False) for d in evidence_pack)}
```

OUTPUT JSON SCHEMA:
{output_schema}
"""
        
        # Store prompts in session state
        st.session_state.llm_prompts = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'region_id': region_id,
            'cluster_A': cluster_A,
            'cluster_B': cluster_B,
            'cluster_C': cluster_C,
            'region_size': len(region_indices)
        }
        
        # Initialize edited prompts
        st.session_state.edited_system_prompt = system_prompt
        st.session_state.edited_user_prompt = user_prompt
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error preparing prompts: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def send_llm_prompt(api_key, model, system_prompt, user_prompt, region_id, cluster_A, cluster_B, cluster_C, region_size):
    """Send the edited prompts to LLM and get results"""
    try:
        with st.spinner(f"🤖 Sending prompt to {model}..."):
            client = OpenAI(api_key=api_key)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            with observe_current(
                name="llm_gap_analysis_prompt",
                as_type="generation",
                input_payload={
                    "region_id": region_id,
                    "cluster_A": cluster_A,
                    "cluster_B": cluster_B,
                    "cluster_C": cluster_C,
                    "region_size": region_size,
                },
                metadata={
                    "region_id": region_id,
                    "cluster_A": cluster_A,
                    "cluster_B": cluster_B,
                    "cluster_C": cluster_C,
                    "region_size": region_size,
                },
                model=model,
            ) as observation:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                observation.update(output={"region_id": region_id, "result_keys": sorted(result.keys())})
            
            # Store results in session state for persistence
            st.session_state.llm_results = {
                'result': result,
                'region_id': region_id,
                'cluster_A': cluster_A,
                'cluster_B': cluster_B,
                'cluster_C': cluster_C,
                'region_size': region_size,
                'model': model,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Clear prompt editing state
            st.session_state.llm_prompts = None
            if 'edited_system_prompt' in st.session_state:
                del st.session_state['edited_system_prompt']
            if 'edited_user_prompt' in st.session_state:
                del st.session_state['edited_user_prompt']
            
            st.success("✅ Analysis complete! Results are displayed below.")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error sending prompt to LLM: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def display_llm_results(result, region_id, cluster_A, cluster_B, cluster_C, region_size):
    """Display LLM analysis results in a structured format"""
    st.success("✅ Analysis complete!")
    
    st.markdown("---")
    st.markdown(f"### 📋 Contrastive Analysis Results")
    cluster_comparison = f"Cluster {cluster_A} vs {cluster_B}"
    if cluster_C is not None:
        cluster_comparison += f" vs {cluster_C}"
    st.markdown(f"**Region {region_id}** | Size: {region_size} papers | Comparing {cluster_comparison}")
    
    # Cluster summaries side-by-side
    num_cols = 3 if cluster_C is not None else 2
    cols = st.columns(num_cols)
    col1, col2 = cols[0], cols[1]
    
    with col1:
        st.markdown(f"#### 🔵 Cluster {cluster_A} Summary")
        if 'cluster_A_summary' in result:
            summary_A = result['cluster_A_summary']
            st.markdown(f"**{summary_A.get('one_line', 'N/A')}**")
            
            if 'bullets' in summary_A and summary_A['bullets']:
                st.markdown("**Key characteristics:**")
                for bullet in summary_A['bullets'][:5]:
                    st.write(f"• {bullet}")
            
            if 'salient_entities' in summary_A:
                entities = summary_A['salient_entities']
                with st.expander("📌 Salient Entities", expanded=False):
                    for entity_type, items in entities.items():
                        if items:
                            st.write(f"**{entity_type.title()}:** {', '.join(items[:7])}")
            
            if 'citations' in summary_A and summary_A['citations']:
                st.caption(f"Based on: {', '.join(summary_A['citations'][:5])}")
    
    with col2:
        st.markdown(f"#### 🟢 Cluster {cluster_B} Summary")
        if 'cluster_B_summary' in result:
            summary_B = result['cluster_B_summary']
            st.markdown(f"**{summary_B.get('one_line', 'N/A')}**")
            
            if 'bullets' in summary_B and summary_B['bullets']:
                st.markdown("**Key characteristics:**")
                for bullet in summary_B['bullets'][:5]:
                    st.write(f"• {bullet}")
            
            if 'salient_entities' in summary_B:
                entities = summary_B['salient_entities']
                with st.expander("📌 Salient Entities", expanded=False):
                    for entity_type, items in entities.items():
                        if items:
                            st.write(f"**{entity_type.title()}:** {', '.join(items[:7])}")
            
            if 'citations' in summary_B and summary_B['citations']:
                st.caption(f"Based on: {', '.join(summary_B['citations'][:5])}")
    
    # Cluster C summary (if present)
    if cluster_C is not None and 'cluster_C_summary' in result:
        col3 = cols[2]
        with col3:
            st.markdown(f"#### 🟣 Cluster {cluster_C} Summary")
            summary_C = result['cluster_C_summary']
            st.markdown(f"**{summary_C.get('one_line', 'N/A')}**")
            
            if 'bullets' in summary_C and summary_C['bullets']:
                st.markdown("**Key characteristics:**")
                for bullet in summary_C['bullets'][:5]:
                    st.write(f"• {bullet}")
            
            if 'salient_entities' in summary_C:
                entities = summary_C['salient_entities']
                with st.expander("📌 Salient Entities", expanded=False):
                    for entity_type, items in entities.items():
                        if items:
                            st.write(f"**{entity_type.title()}:** {', '.join(items[:7])}")
            
            if 'citations' in summary_C and summary_C['citations']:
                st.caption(f"Based on: {', '.join(summary_C['citations'][:5])}")
    
    # Custom question answer (if provided)
    if 'custom_question_answer' in result:
        st.markdown("---")
        st.markdown("### ❓ Custom Question Analysis")
        
        qa = result['custom_question_answer']
        st.markdown(f"**Answer:** {qa.get('answer', 'N/A')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'confidence' in qa:
                st.metric("Confidence", f"{qa['confidence']:.2f}")
        with col2:
            if 'supporting_evidence' in qa and qa['supporting_evidence']:
                st.caption(f"Evidence: {', '.join(qa['supporting_evidence'][:5])}")
        
        if 'limitations' in qa and qa['limitations']:
            st.info(f"**Limitations:** {qa['limitations']}")
    
    # Axes of separation
    st.markdown("---")
    st.markdown("### 🎯 Axes of Separation (Key Differences)")
    
    if 'axes_of_separation' in result and result['axes_of_separation']:
        for i, axis in enumerate(result['axes_of_separation'], 1):
            with st.expander(f"{i}. {axis.get('axis', 'unknown').upper()} (confidence: {axis.get('confidence', 0):.2f})", expanded=i<=3):
                st.write(axis.get('what_differs', 'N/A'))
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'evidence_A' in axis and axis['evidence_A']:
                        st.caption(f"Evidence A: {', '.join(axis['evidence_A'][:3])}")
                with col2:
                    if 'evidence_B' in axis and axis['evidence_B']:
                        st.caption(f"Evidence B: {', '.join(axis['evidence_B'][:3])}")
    else:
        st.info("No specific axes of separation identified")
    
    # Bridge opportunities
    st.markdown("---")
    st.markdown("### 🌉 Bridge Opportunities (Research Gaps)")
    
    if 'bridge_seeds' in result and result['bridge_seeds']:
        for i, bridge in enumerate(result['bridge_seeds'], 1):
            st.markdown(f"**{i}. {bridge.get('idea', 'N/A')}**")
            st.write(f"**Rationale:** {bridge.get('why_plausible', 'N/A')}")
            
            if bridge.get('risks'):
                st.write(f"⚠️ **Potential Risks:** {', '.join(bridge['risks'][:5])}")
            
            if bridge.get('support'):
                st.caption(f"Supporting evidence: {', '.join(bridge['support'][:3])}")
            
            st.markdown("")
    else:
        st.info("No bridge opportunities identified")
    
    # Warnings
    if result.get('insufficient_evidence', False):
        st.warning("⚠️ Note: LLM flagged insufficient evidence for some conclusions")
    
    # Download option
    st.markdown("---")
    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Full Analysis (JSON)",
        data=result_json,
        file_name=f"gap_region_{region_id}_analysis.json",
        mime="application/json",
        key=f"download_llm_analysis_{region_id}"
    )
