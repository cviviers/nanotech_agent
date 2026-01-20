"""
Graph utilities for k-NN graph construction and cluster exploration
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from sklearn.neighbors import NearestNeighbors

# Check for OpenAI availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def build_knn_graph(X: np.ndarray, k: int, metric: str = 'cosine') -> nx.Graph:
    """Build k-NN graph from embeddings."""
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    dists, indices = nn.kneighbors(X, return_distance=True)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    
    for i in range(len(X)):
        for j, d in zip(indices[i, 1:], dists[i, 1:]):
            weight = 1.0 - float(d)
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=weight)
            elif G[i][j]['weight'] < weight:
                G[i][j]['weight'] = weight
    
    return G


def explore_cluster(df: pd.DataFrame, cluster_column: str, cluster_id: int):
    """Display detailed information about a cluster"""
    cluster_df = df[df[cluster_column] == cluster_id]
    
    st.markdown(f"**Cluster {cluster_id}** — {len(cluster_df)} papers")
    
    # Provide cluster statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'publication_year' in cluster_df.columns:
            years = pd.to_numeric(cluster_df['publication_year'], errors='coerce')
            valid_years = years.dropna()
            if len(valid_years) > 0:
                st.metric("Year Range", f"{int(valid_years.min())}–{int(valid_years.max())}")
            else:
                st.metric("Year Range", "N/A")
        else:
            st.metric("Year Range", "N/A")
    
    with col2:
        avg_title_len = cluster_df['title'].fillna('').str.len().mean()
        st.metric("Avg Title Length", f"{int(avg_title_len)} chars")
    
    with col3:
        if 'abstract' in cluster_df.columns:
            avg_abstract_len = cluster_df['abstract'].fillna('').str.len().mean()
            st.metric("Avg Abstract Length", f"{int(avg_abstract_len)} chars")
        elif 'processed_content' in cluster_df.columns:
            avg_content_len = cluster_df['processed_content'].fillna('').str.len().mean()
            st.metric("Avg Content Length", f"{int(avg_content_len)} chars")
        else:
            st.metric("Avg Abstract Length", "N/A")
    
    # Display sample papers
    st.markdown("**Sample Papers**")
    n_display = min(5, len(cluster_df))
    
    for idx, row in cluster_df.head(n_display).iterrows():
        with st.container():
            st.markdown(f"**{row.get('title', 'N/A')}**")
            
            # Metadata
            meta_items = []
            if 'publication_year' in row and pd.notna(row['publication_year']):
                meta_items.append(f"Year: {row['publication_year']}")
            if 'pmid' in row and pd.notna(row['pmid']):
                meta_items.append(f"PMID: {row['pmid']}")
            
            if meta_items:
                st.caption(" | ".join(meta_items))
            
            # Abstract/content
            if 'abstract' in row and pd.notna(row['abstract']):
                abstract_text = str(row['abstract'])[:500]
                st.markdown(f"_{abstract_text}..._" if len(str(row['abstract'])) > 500 else f"_{abstract_text}_")
            elif 'processed_content' in row and pd.notna(row['processed_content']):
                content_text = str(row['processed_content'])[:500]
                st.markdown(f"_{content_text}..._" if len(str(row['processed_content'])) > 500 else f"_{content_text}_")
            
            st.markdown("---")
    
    # CSV download
    csv = cluster_df.to_csv(index=False)
    st.download_button(
        label=f"💾 Download Cluster {cluster_id} as CSV",
        data=csv,
        file_name=f"cluster_{cluster_id}.csv",
        mime="text/csv"
    )
    
    # LLM analysis option
    if OPENAI_AVAILABLE:
        st.markdown("**🤖 AI Cluster Analysis**")
        st.info("Analyze this cluster using OpenAI's language model to extract key themes, materials, and trends.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            n_papers_to_analyze = st.number_input(
                "Papers to analyze",
                min_value=5,
                max_value=len(cluster_df),
                value=min(20, len(cluster_df)),
                key=f"n_papers_llm_{cluster_column}_{cluster_id}"
            )
        with col2:
            llm_model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4o-nano"],
                index=0,
                key=f"llm_model_{cluster_column}_{cluster_id}"
            )
        with col3:
            st.write("")
            st.write("")
            analyze_button = st.button(
                "🔬 Analyze with LLM",
                key=f"analyze_{cluster_column}_{cluster_id}"
            )
        
        if analyze_button:
            llm_api_key = st.session_state.get('openai_api_key', os.environ.get('OPENAI_API_KEY', ''))
            if not llm_api_key:
                st.error("❌ Please provide OpenAI API key in Data & Config page")
            else:
                with st.spinner(f"Analyzing {n_papers_to_analyze} papers from cluster {cluster_id}..."):
                    try:
                        result = summarize_cluster_with_llm(
                            cluster_df,
                            cluster_id,
                            n_papers_to_analyze,
                            llm_api_key,
                            llm_model
                        )
                        
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        st.markdown("##### 📊 Summary")
                        st.info(result.get('main_focus', 'N/A'))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Key Themes**")
                            for theme in result.get('key_themes', []):
                                st.write(f"• {theme}")
                            
                            st.markdown("**Materials**")
                            for material in result.get('materials', [])[:5]:
                                st.write(f"• {material}")
                            
                            st.markdown("**Delivery Methods**")
                            for method in result.get('delivery_methods', [])[:5]:
                                st.write(f"• {method}")
                        
                        with col2:
                            st.markdown("**Diseases/Applications**")
                            for disease in result.get('diseases_applications', [])[:5]:
                                st.write(f"• {disease}")
                            
                            st.markdown("**Experimental Models**")
                            for model in result.get('experimental_models', [])[:5]:
                                st.write(f"• {model}")
                        
                        st.markdown("**Trends & Patterns**")
                        st.write(result.get('trends', 'N/A'))
                        
                        st.markdown("---")
                        st.markdown("**📝 Detailed Summary**")
                        st.markdown(result.get('detailed_summary', 'N/A'))
                        
                        # Download option
                        st.download_button(
                            label="💾 Download Summary (JSON)",
                            data=json.dumps(result, indent=2),
                            file_name=f"cluster_{cluster_id}_summary.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.info("💡 Install OpenAI package to enable AI cluster summarization")


def summarize_cluster_with_llm(cluster_df: pd.DataFrame, cluster_id: int, n_papers: int, api_key: str, model: str) -> dict:
    """Use LLM to generate a comprehensive summary of a cluster."""
    try:
        # Sample papers from cluster
        if len(cluster_df) > n_papers:
            # Take a diverse sample - some recent, some random
            if 'publication_year' in cluster_df.columns:
                # Convert to numeric for sorting
                cluster_df_copy = cluster_df.copy()
                cluster_df_copy['year_numeric'] = pd.to_numeric(cluster_df_copy['publication_year'], errors='coerce')
                recent = cluster_df_copy.nlargest(n_papers // 2, 'year_numeric')
            else:
                recent = cluster_df.head(n_papers // 2)
            random_sample = cluster_df.sample(n=n_papers - len(recent), random_state=42)
            sample_df = pd.concat([recent, random_sample])
        else:
            sample_df = cluster_df
        
        # Build evidence pack
        evidence_pack = []
        for idx, row in sample_df.iterrows():
            doc = {
                'title': row.get('title', 'N/A'),
                'abstract': str(row.get('abstract', row.get('processed_content', '')))[:1000],
                'year': str(row.get('publication_year', 'N/A'))
            }
            evidence_pack.append(doc)
        
        # Create prompts
        system_prompt = """You are an expert scientific research analyst specializing in nanotechnology and nanomedicine.
        Your task is to analyze a cluster of research papers and provide a comprehensive summary of the research focus,
        key themes, methodologies, and trends within this cluster."""
        
        user_prompt = f"""TASK: Analyze the following cluster of research papers and provide a comprehensive summary.

CLUSTER INFORMATION:
- Cluster ID: {cluster_id}
- Total papers in cluster: {len(cluster_df)}
- Papers analyzed: {len(sample_df)}

Please provide a structured analysis covering:
1. Main research focus and themes
2. Common materials and nanoparticle types
3. Target diseases or applications
4. Delivery methods and routes
5. Experimental models used (in vitro, in vivo, clinical)
6. Key trends or temporal patterns (if applicable)

EVIDENCE PACK (JSONL format; each line is one paper):
```jsonl
{chr(10).join(json.dumps(d, ensure_ascii=False) for d in evidence_pack)}
```

OUTPUT: Provide your analysis as a JSON object with the following structure:
{{
    "cluster_id": {cluster_id},
    "main_focus": "Brief description of the main research focus",
    "key_themes": ["theme1", "theme2", "theme3"],
    "materials": ["material1", "material2"],
    "diseases_applications": ["disease1", "application1"],
    "delivery_methods": ["method1", "method2"],
    "experimental_models": ["model1", "model2"],
    "trends": "Description of key trends or patterns",
    "detailed_summary": "Comprehensive narrative summary (2-3 paragraphs)"
}}
"""
        
        # Call LLM
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        raise Exception(f"Error in LLM summarization: {str(e)}")
