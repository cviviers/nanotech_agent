"""
Streamlit App for Interactive Novelty Analysis and Gap Discovery
Based on novelty_analysis_direct.ipynb workflow
"""
import os
import sys
import json
import ast
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

# ML libraries
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
import umap
import hdbscan
import networkx as nx

# Graph clustering
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

# OpenAI for LLM analysis
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add utils path
sys.path.append(str(Path(__file__).parent))


# ============================================================================
# DOMAIN-SPECIFIC ENTITY HINTS
# ============================================================================

MATERIAL_HINTS = [
    'liposome','plga','gold','agnp','au','iron oxide','magnetite','silica',
    'mesoporous','graphene','go','peg','chitosan','albumin','micelle',
    'dendrimer','hydrogel','quantum dot','nanotube','nanoemulsion'
]

LIGAND_HINTS = [
    'rgd','folate','transferrin','aptamer','peptide','antibody','egf',
    'her2','mannose','galactose','hyaluronic'
]

DISEASE_HINTS = [
    'cancer','glioblastoma','breast','lung','pancreatic','pancreatic cancer',
    'prostate','melanoma','liver','ovarian','colorectal','colorectal cancer',
    'infection','inflammation','chronic inflammation','chronic inflammatory disease',
    'alzheimer','alzheimer\'s disease','neurodegenerative','neurodegenerative disease',
    'inflammatory bowel disease','ibd',
    'rheumatoid arthritis','autoimmune','autoimmunity'
]

DELIVERY_HINTS = [
    'intravenous','iv','oral','oral delivery','intratumoral','inhalation','topical','intranasal',
    'systemic','systemic delivery',
    'local','local delivery','local effects',
    'sustained release','local sustained release',
    'brain delivery',
    'blood brain barrier','barrier passage',
    'barrier penetration','barrier disruption'
]

MODEL_HINTS = [
    'in vitro','in vivo','mouse','murine','rat','xenograft','clinical','phase'
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_embedding(value: Any) -> Optional[np.ndarray]:
    """Parse embedding from various formats (list, array, string)"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=float)
        except Exception:
            return None
    return None


def extract_embeddings(df: pd.DataFrame, embed_names: List[str], data_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load embeddings from .npy files in data directory."""
    if not embed_names:
        raise ValueError(f"No embedding names provided")
    
    result = {}
    valid_mask = None
    
    for embed_name in embed_names:
        # Construct paths
        npy_path = data_dir / f"{embed_name}_embeddings.npy"
        metadata_path = data_dir / f"{embed_name}_embeddings_metadata.json"
        
        if not npy_path.exists():
            st.warning(f"Embedding file not found: {npy_path}")
            continue
        
        try:
            # Load embeddings
            embeddings = np.load(npy_path)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                st.info(f"Loaded {embed_name}: {embeddings.shape}, model: {metadata.get('model', 'unknown')}")
            else:
                st.info(f"Loaded {embed_name}: {embeddings.shape}")
            
            # Check if number of embeddings matches dataframe
            if len(embeddings) != len(df):
                st.warning(f"Embedding count mismatch for {embed_name}: {len(embeddings)} embeddings vs {len(df)} papers")
                # Use minimum length
                min_len = min(len(embeddings), len(df))
                embeddings = embeddings[:min_len]
                if valid_mask is None:
                    valid_mask = np.ones(min_len, dtype=bool)
                else:
                    valid_mask = valid_mask[:min_len]
            else:
                if valid_mask is None:
                    valid_mask = np.ones(len(embeddings), dtype=bool)
            
            result[embed_name] = embeddings.astype(np.float32)
            
        except Exception as e:
            st.error(f"Error loading {embed_name} embeddings: {str(e)}")
            continue
    
    if not result:
        raise ValueError(f"No embeddings could be loaded from: {embed_names}")
    
    valid_idx = np.where(valid_mask)[0]
    
    # Trim all embeddings to valid indices
    for key in result:
        result[key] = result[key][valid_idx]
    
    return result, valid_idx


def compute_knn_density(X: np.ndarray, k: int, metric: str = 'cosine') -> np.ndarray:
    """Compute average distance to k nearest neighbors."""
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    return dists[:, 1:].mean(axis=1)


def compute_density_features(X: np.ndarray, k_list: List[int], metric: str = 'cosine') -> pd.DataFrame:
    """Compute density features for multiple k values."""
    features = {}
    
    for k in k_list:
        avg_dist = compute_knn_density(X, k, metric)
        features[f'density_k{k}'] = avg_dist
        features[f'density_k{k}_z'] = zscore(avg_dist, nan_policy='omit')
    
    df = pd.DataFrame(features)
    z_cols = [c for c in df.columns if c.endswith('_z')]
    df['gap_score'] = df[z_cols].mean(axis=1)
    
    return df


def simple_entity_extract(text: str) -> Dict[str, List[str]]:
    """Extract domain-specific entities from text using keyword matching."""
    text_lower = (text or '').lower()
    
    entities = {
        'materials': sorted({w for w in MATERIAL_HINTS if w in text_lower}),
        'ligands': sorted({w for w in LIGAND_HINTS if w in text_lower}),
        'diseases': sorted({w for w in DISEASE_HINTS if w in text_lower}),
        'delivery': sorted({w for w in DELIVERY_HINTS if w in text_lower}),
        'models': sorted({w for w in MODEL_HINTS if w in text_lower}),
    }
    
    return entities


def extract_entities_from_dataframe(df: pd.DataFrame, text_col: str = 'processed_content') -> pd.DataFrame:
    """Extract entities for all papers in dataframe."""
    entity_lists = {
        'materials': [],
        'ligands': [],
        'diseases': [],
        'delivery': [],
        'models': []
    }
    
    for _, row in df.iterrows():
        text = str(row.get(text_col) or row.get('abstract') or row.get('content') or '')
        entities = simple_entity_extract(text)
        
        for key in entity_lists.keys():
            entity_lists[key].append(entities.get(key, []))
    
    # Add as columns to dataframe copy
    df_with_entities = df.copy()
    for key, values in entity_lists.items():
        df_with_entities[f'entities_{key}'] = values
    
    return df_with_entities


def summarize_gap_region_entities(df: pd.DataFrame, region_indices: List[int]) -> Dict[str, Any]:
    """Summarize entity distribution in a gap region."""
    region_df = df.loc[region_indices]
    
    summary = {}
    for entity_type in ['materials', 'ligands', 'diseases', 'delivery', 'models']:
        col_name = f'entities_{entity_type}'
        if col_name in region_df.columns:
            # Flatten list of lists and count
            all_entities = []
            for entity_list in region_df[col_name]:
                if isinstance(entity_list, list):
                    all_entities.extend(entity_list)
            
            entity_counts = Counter(all_entities)
            summary[entity_type] = {
                'total_unique': len(entity_counts),
                'total_mentions': sum(entity_counts.values()),
                'top_5': entity_counts.most_common(5)
            }
        else:
            summary[entity_type] = {
                'total_unique': 0,
                'total_mentions': 0,
                'top_5': []
            }
    
    return summary


def explore_cluster(df: pd.DataFrame, cluster_column: str, cluster_id: int) -> None:
    """Display detailed information about a specific cluster."""
    cluster_df = df[df[cluster_column] == cluster_id]
    n_papers = len(cluster_df)
    
    st.markdown(f"### 📊 Cluster {cluster_id} Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers in Cluster", n_papers)
    
    # Year distribution
    if 'publication_year' in cluster_df.columns:
        years = cluster_df['publication_year'].dropna()
        if len(years) > 0:
             # Convert to numeric, handling string values
            years_numeric = pd.to_numeric(years, errors='coerce').dropna()
            if len(years_numeric) > 0:
                col2.metric("Year Range", f"{int(years_numeric.min())}-{int(years_numeric.max())}")
                col3.metric("Median Year", f"{int(years_numeric.median())}")
    
    # Citation statistics if available
    if 'citation_count' in cluster_df.columns:
        citations = pd.to_numeric(cluster_df['citation_count'], errors='coerce').dropna()
        if len(citations) > 0:
            col4.metric("Avg Citations", f"{citations.mean():.0f}")
    
    # Entity analysis
    st.markdown("#### 🔬 Domain Entity Analysis")
    
    # Extract entities if not already done
    text_col = 'processed_content' if 'processed_content' in cluster_df.columns else 'abstract'
    entity_summary = {}
    
    for entity_type in ['materials', 'ligands', 'diseases', 'delivery', 'models']:
        all_entities = []
        for _, row in cluster_df.iterrows():
            text = str(row.get(text_col) or row.get('content') or '').lower()
            
            if entity_type == 'materials':
                hints = MATERIAL_HINTS
            elif entity_type == 'ligands':
                hints = LIGAND_HINTS
            elif entity_type == 'diseases':
                hints = DISEASE_HINTS
            elif entity_type == 'delivery':
                hints = DELIVERY_HINTS
            else:
                hints = MODEL_HINTS
            
            found = [h for h in hints if h in text]
            all_entities.extend(found)
        
        entity_counts = Counter(all_entities)
        entity_summary[entity_type] = entity_counts.most_common(10)
    
    # Display top entities in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**Materials**")
        for entity, count in entity_summary['materials'][:5]:
            st.write(f"• {entity} ({count})")
    
    with col2:
        st.markdown("**Ligands**")
        for entity, count in entity_summary['ligands'][:5]:
            st.write(f"• {entity} ({count})")
    
    with col3:
        st.markdown("**Diseases**")
        for entity, count in entity_summary['diseases'][:5]:
            st.write(f"• {entity} ({count})")
    
    with col4:
        st.markdown("**Delivery**")
        for entity, count in entity_summary['delivery'][:5]:
            st.write(f"• {entity} ({count})")
    
    with col5:
        st.markdown("**Models**")
        for entity, count in entity_summary['models'][:5]:
            st.write(f"• {entity} ({count})")
    
    # Top papers by various metrics
    st.markdown("#### 📄 Representative Papers")
    
    # Most recent papers
    if 'publication_year' in cluster_df.columns:
        st.markdown("**Most Recent (Top 5)**")
        # Convert to numeric for sorting
        cluster_df_copy = cluster_df.copy()
        cluster_df_copy['year_numeric'] = pd.to_numeric(cluster_df_copy['publication_year'], errors='coerce')
        recent = cluster_df_copy.nlargest(5, 'year_numeric')[['title', 'publication_year']]
        for idx, (_, row) in enumerate(recent.iterrows(), 1):
            st.write(f"{idx}. [{row.get('publication_year', 'N/A')}] {row.get('title', 'N/A')}")
    
    st.markdown("**Sample Papers from Cluster**")
    sample_papers = cluster_df.head(10)
    for idx, (_, row) in enumerate(sample_papers.iterrows(), 1):
        year = f"[{row.get('publication_year', 'N/A')}]" if 'publication_year' in row else ""
        title = row.get('title', 'N/A')
        st.write(f"{idx}. {year} {title}")
        
        # Show abstract for first 3
        if idx <= 3:
            abstract = str(row.get('abstract', row.get('processed_content', '')))[:300]
            if abstract:
                st.caption(abstract + '...')
    
    # LLM Summarization section
    if OPENAI_AVAILABLE:
        st.markdown("---")
        st.markdown("#### 🤖 AI Cluster Summary")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            n_papers_to_analyze = st.number_input(
                "Number of papers to send to LLM",
                min_value=5,
                max_value=min(100, len(cluster_df)),
                value=min(20, len(cluster_df)),
                key=f"n_papers_llm_{cluster_column}_{cluster_id}"
            )
        with col2:
            llm_model = st.selectbox(
                "Model",
                ["gpt-5-mini", "gpt-5", "gpt-5-nano"],
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
        
        # API key input
        llm_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get('OPENAI_API_KEY', ''),
            key=f"api_key_{cluster_column}_{cluster_id}",
            help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
        )
        
        if analyze_button:
            if not llm_api_key:
                st.error("❌ Please provide OpenAI API key")
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


def summarize_cluster_with_llm(cluster_df: pd.DataFrame, cluster_id: int, n_papers: int, api_key: str, model: str) -> str:
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


# ============================================================================
# UNDO FUNCTIONALITY
# ============================================================================

def save_state_for_undo(action_name: str):
    """Save current state to undo history"""
    # Deep copy embeddings_dict
    embeddings_dict_copy = {}
    if st.session_state.embeddings_dict:
        for key, value in st.session_state.embeddings_dict.items():
            embeddings_dict_copy[key] = value.copy() if value is not None else None
    
    state_snapshot = {
        'action_name': action_name,
        'df_valid': st.session_state.df_valid.copy() if st.session_state.df_valid is not None else None,
        'X_pca': st.session_state.X_pca.copy() if st.session_state.X_pca is not None else None,
        'X_primary': st.session_state.X_primary.copy() if st.session_state.X_primary is not None else None,
        'X_umap_2d': st.session_state.X_umap_2d.copy() if st.session_state.X_umap_2d is not None else None,
        'embeddings_dict': embeddings_dict_copy,
        'kmeans_applied': st.session_state.kmeans_applied,
        'similarity_applied': st.session_state.similarity_applied,
        'G': st.session_state.G,  # Graph is immutable, can reference directly
        'clustering_done': st.session_state.clustering_done
    }
    st.session_state.undo_history.append(state_snapshot)
    
    # Limit history to last 10 actions
    if len(st.session_state.undo_history) > 10:
        st.session_state.undo_history.pop(0)


def undo_last_action():
    """Restore previous state from undo history"""
    if not st.session_state.undo_history:
        return False
    
    snapshot = st.session_state.undo_history.pop()
    
    # Restore state
    st.session_state.df_valid = snapshot['df_valid']
    st.session_state.X_pca = snapshot['X_pca']
    st.session_state.X_primary = snapshot['X_primary']
    st.session_state.X_umap_2d = snapshot['X_umap_2d']
    st.session_state.embeddings_dict = snapshot.get('embeddings_dict', {})
    st.session_state.kmeans_applied = snapshot['kmeans_applied']
    st.session_state.similarity_applied = snapshot['similarity_applied']
    st.session_state.G = snapshot.get('G', None)
    st.session_state.clustering_done = snapshot.get('clustering_done', False)
    
    # Update UMAP coordinates in dataframe if available
    if st.session_state.df_valid is not None and st.session_state.X_umap_2d is not None:
        st.session_state.df_valid['umap_x'] = st.session_state.X_umap_2d[:, 0]
        st.session_state.df_valid['umap_y'] = st.session_state.X_umap_2d[:, 1]
    
    return True


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    
    # Data states
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_filtered' not in st.session_state:
        st.session_state.df_filtered = None
    if 'df_valid' not in st.session_state:
        st.session_state.df_valid = None
    
    # Processing states
    if 'embeddings_extracted' not in st.session_state:
        st.session_state.embeddings_extracted = False
    if 'embeddings_dict' not in st.session_state:
        st.session_state.embeddings_dict = {}
    if 'X_primary' not in st.session_state:
        st.session_state.X_primary = None
    if 'X_pca' not in st.session_state:
        st.session_state.X_pca = None
    if 'X_umap_2d' not in st.session_state:
        st.session_state.X_umap_2d = None
    
    # Analysis states
    if 'density_computed' not in st.session_state:
        st.session_state.density_computed = False
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'gaps_identified' not in st.session_state:
        st.session_state.gaps_identified = False
    if 'G' not in st.session_state:
        st.session_state.G = None
    if 'gap_regions' not in st.session_state:
        st.session_state.gap_regions = []
    
    # Filter states
    if 'kmeans_applied' not in st.session_state:
        st.session_state.kmeans_applied = False
    if 'similarity_applied' not in st.session_state:
        st.session_state.similarity_applied = False
    if 'qa_retrieval_applied' not in st.session_state:
        st.session_state.qa_retrieval_applied = False
    
    # Clustering selection
    if 'selected_clustering' not in st.session_state:
        st.session_state.selected_clustering = None
    
    # Undo history
    if 'undo_history' not in st.session_state:
        st.session_state.undo_history = []
    
    # Random seed for reproducibility
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 42
    
    # LLM analysis results
    if 'llm_results' not in st.session_state:
        st.session_state.llm_results = None
    
    # LLM prompts (for editing before sending)
    if 'llm_prompts' not in st.session_state:
        st.session_state.llm_prompts = None


# ============================================================================
# PAGE: CONFIGURATION & DATA LOADING
# ============================================================================

def page_data_loading():
    """Data loading and initial configuration page"""
    st.title("📊 Data Loading & Configuration")
    
    st.markdown("""
    Configure the basic data parameters and load your dataset.
    """)
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Configuration")
        
        data_path = st.text_input(
            "Data File Path",
            value=r"./data/cleaned_dataset.json",
            help="Path to your JSON dataset file"
        )
        
        sample_n = st.number_input(
            "Sample Size (0 = all data)",
            min_value=0,
            max_value=100000,
            value=0,
            help="Limit dataset size for faster processing"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="Seed for reproducible results across all random operations"
        )
    
    with col2:
        st.subheader("🎯 Embedding Configuration")
        
        available_embeddings = st.multiselect(
            "Available Embedding Files",
            ["qwen", "bert"],
            default=["qwen", "bert"],
            help="Select which embedding files to load from /data/ folder"
        )
        
        primary_embedding = st.selectbox(
            "Primary Embedding",
            available_embeddings,
            index=0 if available_embeddings else None
        )
    
    # Store in session state
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.config = {
            'data_path': data_path,
            'sample_n': sample_n if sample_n > 0 else None,
            'embedding_cols': available_embeddings,
            'primary_embedding': primary_embedding,
            'random_seed': random_seed
        }
        st.session_state.random_seed = random_seed
        st.success("✅ Configuration saved!")
    
    st.divider()
    
    # Data loading
    if 'config' not in st.session_state:
        st.info("👆 Please save configuration first")
        return
    
    st.subheader("📥 Load Dataset")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        keywords_title_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["review", "not available", "overview"]
        )
        keywords_abstract_exclusion = st.multiselect(
            "Exclusion Keywords (title)",
            ["review", "survey", "not available", "retraction", "overview"],
            default=["not available", "retraction", "overview"]
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Load Data", type="primary"):
            load_data(st.session_state.config, keywords_title_exclusion, keywords_abstract_exclusion)
    
    # Show data status
    if st.session_state.df_original is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.df_original)} papers")
        
        if st.session_state.df_filtered is not None:
            st.info(f"After filtering: {len(st.session_state.df_filtered)} papers")
        
        with st.expander("📊 Data Preview"):
            st.dataframe(st.session_state.df_filtered.head(10) if st.session_state.df_filtered is not None 
                        else st.session_state.df_original.head(10))


def load_data(config, keywords_title_exclusion, keywords_abstract_exclusion):
    """Load and filter dataset from JSON file, and load embeddings"""
    data_path = Path(config['data_path'])
    
    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return
    
    with st.spinner("Loading dataset from JSON..."):
        # Load JSON file
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            st.session_state.df_original = df.copy()
            
            # Sample if needed (before filtering to maintain reproducibility)
            if config['sample_n'] is not None and len(df) > config['sample_n']:
                df = df.sample(config['sample_n'], random_state=config.get('random_seed', 42))
            
            df = df.reset_index(drop=True)
            st.session_state.df_filtered = df
            
        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Load embeddings immediately after loading data
    with st.spinner("Loading embeddings from .npy files..."):
        try:
            data_dir = Path(config['data_path']).parent
            
            embeddings_dict, valid_idx = extract_embeddings(
                st.session_state.df_filtered,
                config['embedding_cols'],
                data_dir
            )
            
            st.session_state.embeddings_dict = embeddings_dict
            st.session_state.df_valid = st.session_state.df_filtered.iloc[valid_idx].reset_index(drop=True)
            st.session_state.X_primary = embeddings_dict[config['primary_embedding']]
            st.session_state.embeddings_extracted = True
            
            st.success(f"✅ Loaded embeddings: {len(valid_idx)} valid rows")
            
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Apply keyword filters AFTER embeddings are loaded
    with st.spinner("Applying keyword filters..."):
        n_before = len(st.session_state.df_valid)
        
        # Create mask for filtering
        mask = pd.Series([True] * len(st.session_state.df_valid), index=st.session_state.df_valid.index)
        
        # Apply title filters
        for keyword in keywords_title_exclusion:
            if 'title' in st.session_state.df_valid.columns:
                mask &= ~st.session_state.df_valid['title'].str.lower().str.contains(keyword, na=False)
        
        # Apply abstract filters
        for keyword in keywords_abstract_exclusion:
            if 'abstract' in st.session_state.df_valid.columns:
                mask &= ~st.session_state.df_valid['abstract'].str.lower().str.contains(keyword, na=False)
        
        # Apply mask to dataframe
        st.session_state.df_valid = st.session_state.df_valid[mask].reset_index(drop=True)
        
        # Apply mask to embeddings
        for key in st.session_state.embeddings_dict:
            st.session_state.embeddings_dict[key] = st.session_state.embeddings_dict[key][mask.values]
        
        # Update primary embedding
        st.session_state.X_primary = st.session_state.embeddings_dict[config['primary_embedding']]
        
        n_after = len(st.session_state.df_valid)
        if n_before != n_after:
            st.info(f"Keyword filtering: {n_before} → {n_after} papers")


# ============================================================================
# PAGE: EMBEDDING EXTRACTION & DIMENSIONALITY REDUCTION
# ============================================================================

def page_embedding_processing():
    """Extract embeddings and apply dimensionality reduction"""
    st.title("🧬 Embedding Processing")
    
    if st.session_state.df_filtered is None:
        st.warning("⚠️ Please load data first")
        return
    
    config = st.session_state.config
    
    st.markdown(f"""
    **Dataset**: {len(st.session_state.df_valid) if st.session_state.df_valid is not None else len(st.session_state.df_filtered)} papers  
    **Primary Embedding**: {config['primary_embedding']}  
    **Available Embeddings**: {', '.join(config['embedding_cols'])}
    """)
    
    # Check if embeddings are already loaded
    if not st.session_state.embeddings_extracted:
        st.warning("⚠️ Embeddings should have been loaded with the data. Please reload the dataset.")
        return
    
    st.success(f"✅ Embeddings extracted: {st.session_state.X_primary.shape}")
    
    st.divider()
    
    # PCA reduction
    st.subheader("📉 PCA Dimensionality Reduction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pca_components = st.number_input("PCA Components", min_value=10, max_value=1024, value=102)
    with col2:
        st.write("")
        st.write("")
        if st.button("▶️ Run PCA"):
            with st.spinner("Running PCA..."):
                pca = PCA(n_components=pca_components, random_state=42)
                X_pca = pca.fit_transform(st.session_state.X_primary)
                st.session_state.X_pca = X_pca
                explained_var = pca.explained_variance_ratio_.sum()
                st.success(f"✅ PCA: {X_pca.shape}, explained variance: {explained_var:.2%}")
    with col3:
        if st.session_state.X_pca is not None:
            st.metric("PCA Shape", f"{st.session_state.X_pca.shape}")
    
    if st.session_state.X_pca is None:
        return
    
    st.divider()
    
    # UMAP projection
    st.subheader("🗺️ UMAP 2D Projection")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        umap_neighbors = st.number_input("n_neighbors", min_value=5, max_value=200, value=50)
    with col2:
        umap_min_dist = st.number_input("min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col3:
        st.write("")
        st.write("")
        if st.button("▶️ Run UMAP"):
            with st.spinner("Computing UMAP projection..."):
                reducer_2d = umap.UMAP(
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    n_components=2,
                    random_state=42
                )
                X_umap_2d = reducer_2d.fit_transform(st.session_state.X_pca)
                st.session_state.X_umap_2d = X_umap_2d
                st.session_state.df_valid['umap_x'] = X_umap_2d[:, 0]
                st.session_state.df_valid['umap_y'] = X_umap_2d[:, 1]
                st.success(f"✅ UMAP 2D: {X_umap_2d.shape}")
    with col4:
        if st.session_state.X_umap_2d is not None:
            st.metric("UMAP Shape", f"{st.session_state.X_umap_2d.shape}")
    
    # Visualize UMAP
    if st.session_state.X_umap_2d is not None:
        st.subheader("📊 UMAP Visualization")
        
        # Prepare hover data
        df_plot = st.session_state.df_valid.copy()
        df_plot['hover_title'] = df_plot['title'].fillna('N/A')
        df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
        
        fig = px.scatter(
            df_plot,
            x='umap_x',
            y='umap_y',
            title="UMAP 2D Projection",
            opacity=0.6,
            height=1000,
            hover_data={'umap_x': False, 'umap_y': False, 'hover_title': True, 'hover_abstract': True}
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: OPTIONAL FILTERS
# ============================================================================

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
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
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
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Cluster Distribution"):
            st.bar_chart(cluster_counts)
    
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
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top matches
            with st.expander("📄 Top 10 Most Similar Papers"):
                top_papers = st.session_state.df_valid.nlargest(10, 'similarity_score')
                for idx, row in top_papers.iterrows():
                    st.write(f"**[{row['similarity_score']:.3f}]** {row.get('title', 'N/A')}")
    
    else:  # Question Answering mode
        col1, col2 = st.columns([3, 1])
        with col1:
            question_text = st.text_area(
                "Research Question",
                value="What are the most effective nanoparticle delivery systems for crossing the blood-brain barrier?",
                height=100,
                help="Ask a specific research question. The system will find papers that best answer it."
            )
            qa_threshold = st.slider("Relevance Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                                    help="Papers with embedding similarity above this threshold will be reranked")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Retrieve Answers"):
                compute_question_answering_retrieval(question_text, qa_threshold)
        
        if st.session_state.qa_retrieval_applied:
            st.success("✅ Q&A retrieval complete")
            
            # Filters based on Q&A scores
            col1, col2 = st.columns(2)
            with col1:
                qa_filter_threshold = st.slider("Combined Score Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                                               help="Filter to keep only papers above this combined score")
            with col2:
                if st.button("✂️ Apply Q&A Filter"):
                    apply_qa_filter(qa_filter_threshold)
            
            # Visualize Q&A scores
            df_plot = st.session_state.df_valid.copy()
            df_plot['hover_title'] = df_plot['title'].fillna('N/A')
            df_plot['hover_abstract'] = df_plot.get('abstract', df_plot.get('processed_content', '')).fillna('').astype(str).str[:200] + '...'
            
            tab1, tab2, tab3 = st.tabs(["Combined Score", "Reranker Score", "Embedding Score"])
            
            with tab1:
                fig = px.scatter(
                    df_plot,
                    x='umap_x',
                    y='umap_y',
                    color='qa_combined_score',
                    title="Q&A Combined Score",
                    opacity=0.7,
                    height=800,
                    color_continuous_scale='Viridis',
                    hover_data={'umap_x': False, 'umap_y': False, 'qa_combined_score': ':.3f', 
                               'qa_reranker_score': ':.3f', 'qa_embedding_score': ':.3f',
                               'hover_title': True, 'hover_abstract': True}
                )
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = px.scatter(
                    df_plot,
                    x='umap_x',
                    y='umap_y',
                    color='qa_reranker_score',
                    title="Reranker Score (Qwen3-Reranker)",
                    opacity=0.7,
                    height=800,
                    color_continuous_scale='Reds',
                    hover_data={'umap_x': False, 'umap_y': False, 'qa_reranker_score': ':.3f', 
                               'hover_title': True, 'hover_abstract': True}
                )
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = px.scatter(
                    df_plot,
                    x='umap_x',
                    y='umap_y',
                    color='qa_embedding_score',
                    title="Embedding Score (Qwen3-Embedding)",
                    opacity=0.7,
                    height=800,
                    color_continuous_scale='Blues',
                    hover_data={'umap_x': False, 'umap_y': False, 'qa_embedding_score': ':.3f', 
                               'hover_title': True, 'hover_abstract': True}
                )
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1))
                st.plotly_chart(fig, use_container_width=True)
    
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
            print(query_embedding)
        
        with st.spinner("Computing similarities with preloaded paper embeddings..."):
            # Use the already-loaded embeddings from session state
            paper_embeddings = st.session_state.X_primary
            
            # Normalize embeddings for cosine similarity
            # paper_embeddings_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            # query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            query_embedding_norm = query_embedding
            paper_embeddings_norm = paper_embeddings
            # print(paper_embeddings_norm.shape, query_embedding_norm.shape)
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


def compute_question_answering_retrieval(question, threshold=0.3):
    """
    Compute question-answering style retrieval using Qwen reranker.
    Uses both embedding similarity and reranker scores for better results.
    """
    try:
        import requests
        
        # API endpoint for Qwen service
        QWEN_API_URL = "http://localhost:8000"
        
        # Get paper texts
        text_col = 'processed_content' if 'processed_content' in st.session_state.df_valid.columns else 'abstract'
        paper_texts = st.session_state.df_valid[text_col].fillna('').astype(str).tolist()
        
        # First pass: Get candidates above threshold using embedding similarity with preloaded embeddings
        with st.spinner("Phase 1/2: Computing embedding similarities with preloaded embeddings..."):
            # Get question embedding from API
            query_payload = {
                "texts": [question],
                "instruction": "Represent this question for retrieving relevant research papers that answer it",
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
            
            question_embedding = np.array(query_response.json()['embeddings'][0])
            
            # Use preloaded paper embeddings
            paper_embeddings = st.session_state.X_primary
            
            # Normalize and compute similarities
            paper_embeddings_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            question_embedding_norm = question_embedding / np.linalg.norm(question_embedding)
            embedding_scores = paper_embeddings_norm @ question_embedding_norm
            
            # Get candidates above threshold
            above_threshold = embedding_scores >= threshold
            top_candidate_indices = np.where(above_threshold)[0]
            
            # Sort by score descending
            sorted_order = np.argsort(embedding_scores[top_candidate_indices])[::-1]
            top_candidate_indices = top_candidate_indices[sorted_order]
            top_candidate_texts = [paper_texts[i] for i in top_candidate_indices]
            
            st.info(f"Found {len(top_candidate_indices)} papers above threshold {threshold:.2f}")
        
        if len(top_candidate_indices) == 0:
            st.warning(f"❌ No papers found above threshold {threshold:.2f}. Try lowering the threshold.")
            return
        
        # Second pass: Rerank candidates
        with st.spinner(f"Phase 2/2: Reranking {len(top_candidate_texts)} candidates..."):
            rank_payload = {
                "query": question,
                "documents": top_candidate_texts,
                "instruction": "Given a research question, determine if this research paper provides relevant information to answer it",
                "top_k": None,  # Return all
                "return_embedding_similarity": True,
                "normalize_embeddings": True
            }
            
            rank_response = requests.post(
                f"{QWEN_API_URL}/rank",
                json=rank_payload,
                timeout=180
            )
            
            if rank_response.status_code != 200:
                st.error(f"❌ Reranking failed: {rank_response.text}")
                return
            
            rank_results = rank_response.json()['results']
        
        # Map results back to original indices and store scores
        reranker_scores = np.zeros(len(paper_texts))
        combined_scores = np.zeros(len(paper_texts))
        
        for result in rank_results:
            original_idx = top_candidate_indices[result['index']]
            reranker_scores[original_idx] = result['reranker_score']
            # Combine reranker and embedding scores (weighted average)
            if result['embedding_score'] is not None:
                combined_scores[original_idx] = 0.7 * result['reranker_score'] + 0.3 * result['embedding_score']
            else:
                combined_scores[original_idx] = result['reranker_score']
        
        # Store all scores
        st.session_state.df_valid['qa_reranker_score'] = reranker_scores
        st.session_state.df_valid['qa_embedding_score'] = embedding_scores
        st.session_state.df_valid['qa_combined_score'] = combined_scores
        st.session_state.qa_retrieval_applied = True
        
        n_reranked = len(top_candidate_indices)
        n_with_scores = (combined_scores > 0).sum()
        st.success(f"✅ Q&A retrieval complete. Reranked {n_reranked} papers above threshold.")
        
        # Show top results
        with st.expander("📄 Top 10 Results", expanded=True):
            top_papers = st.session_state.df_valid.nlargest(10, 'qa_combined_score')
            for idx, (_, row) in enumerate(top_papers.iterrows(), 1):
                st.write(f"**{idx}. [{row['qa_combined_score']:.3f}]** {row.get('title', 'N/A')}")
                col1, col2, col3 = st.columns(3)
                col1.caption(f"Reranker: {row['qa_reranker_score']:.3f}")
                col2.caption(f"Embedding: {row['qa_embedding_score']:.3f}")
                col3.caption(f"Year: {row.get('publication_year', 'N/A')}")
                if idx < 5:  # Show abstract for top 5
                    st.caption(str(row.get('abstract', row.get('processed_content', '')))[:300] + '...')
                st.divider()
    
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


def apply_qa_filter(threshold):
    """Filter by Q&A combined score threshold"""
    save_state_for_undo(f"Q&A filter (threshold={threshold:.2f})")
    
    mask = st.session_state.df_valid['qa_combined_score'] >= threshold
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


# ============================================================================
# PAGE: DENSITY & GAP ANALYSIS
# ============================================================================

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


# ============================================================================
# PAGE: CLUSTERING
# ============================================================================

def page_clustering():
    """Run clustering algorithms"""
    st.title("🎯 Clustering Analysis")
    
    if st.session_state.X_pca is None:
        st.warning("⚠️ Please complete embedding processing first")
        return
    
    # if st.session_state.G is None:
    #     st.warning("⚠️ Please complete gap analysis (k-NN graph construction) first")
    #     return
    # **Graph**: {st.session_state.G.number_of_nodes()} nodes, {st.session_state.G.number_of_edges()} edges
    
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
        'leiden_resolution': leiden_resolution
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
        st.plotly_chart(fig, use_container_width=True)
        
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
        st.plotly_chart(fig, use_container_width=True)
        
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
                run_louvain_clustering()
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
            st.plotly_chart(fig_graph, use_container_width=True)
            
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
            st.plotly_chart(fig, use_container_width=True)
            
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
        
        # Selection button
        if st.session_state.selected_clustering == 'leiden':
            st.success("✅ Community Detection selected for gap analysis")
        else:
            if st.button("✔️ Use Community Detection for Gap Analysis", type="primary"):
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


def run_louvain_clustering():
    """Run Louvain algorithm"""
    save_state_for_undo("Louvain Clustering")
    with st.spinner("Running Louvain algorithm..."):
        G = st.session_state.G
        partition = community_louvain.best_partition(G, weight='weight', random_state=st.session_state.random_seed)
        labels = np.array([partition[i] for i in range(len(G))], dtype=int)
        
        st.session_state.df_valid['cluster_leiden'] = labels
        st.success(f"✅ Louvain: {len(set(labels))} communities")
        st.rerun()


# ============================================================================
# PAGE: GAP REGIONS
# ============================================================================

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
            import plotly.colors as colors
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
                st.bar_chart(year_counts)
    
    # Top papers
    st.markdown("#### 📄 Top Papers by Gap Score")
    top_papers = region_df.nlargest(5, 'gap_score')
    
    for idx, (_, row) in enumerate(top_papers.iterrows(), 1):
        with st.expander(f"{idx}. [{row['gap_score']:.3f}] {row.get('title', 'N/A')}"):
            st.write(f"**Year**: {row.get('publication_year', 'N/A')}")
            st.write(f"**Journal**: {row.get('journal', 'N/A')}")
            if 'abstract' in row and pd.notna(row['abstract']):
                st.write(f"**Abstract**: {row['abstract'][:500]}...")


# ============================================================================
# PAGE: LLM ANALYSIS
# ============================================================================

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
        openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                       value=os.environ.get('OPENAI_API_KEY', ''))
        openai_model = st.selectbox("Model", ["gpt-5-mini", "gpt-5", "gpt-5-nano"], index=0)
    with col2:
        region_id = st.selectbox("Select Gap Region", range(len(gap_regions)))
        n_papers_per_cluster = st.number_input("Papers per Cluster", min_value=5, max_value=100, value=15)
    with col3:
        n_gap_papers = st.number_input("Gap Papers to Include", min_value=5, max_value=100, value=5,
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
    
    # Remove the show_prompt_editor checkbox - it will always be shown
    
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
            gap_data = df_plot.loc[region_indices]
            fig.add_trace(go.Scatter(
                x=gap_data['umap_x'],
                y=gap_data['umap_y'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star', 
                           line=dict(color='darkred', width=1)),
                name=f'Gap Region {region_id} (n={len(region_indices)})',
                showlegend=True,
                hovertemplate=f'Gap Region {region_id}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Clusters with Gap Region {region_id} Highlighted',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                height=600,
                hovermode='closest',
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Select two clusters from the legend above to contrast in the analysis")
    
    # Get all available clusters from the selected clustering method
    if st.session_state.selected_clustering:
        cluster_col = f'cluster_{st.session_state.selected_clustering}'
        if cluster_col in st.session_state.df_valid.columns:
            all_cluster_ids = sorted(st.session_state.df_valid[cluster_col].unique())
            all_cluster_ids = [c for c in all_cluster_ids if c != -1]  # Remove noise cluster if present
            
            # Count papers in each cluster
            cluster_sizes = st.session_state.df_valid[cluster_col].value_counts().to_dict()
            
            col1, col2 = st.columns(2)
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
            
            if cluster_A_selected is not None and cluster_B_selected is not None:
                st.success(f"✅ Ready to compare Cluster {cluster_A_selected} ({cluster_sizes.get(cluster_A_selected, 0)} papers) vs Cluster {cluster_B_selected} ({cluster_sizes.get(cluster_B_selected, 0)} papers)")
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
        if not openai_api_key:
            st.error("Please provide OpenAI API key")
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
            cluster_B_selected
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
                if not openai_api_key:
                    st.error("Please provide OpenAI API key")
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
            llm_data['region_size']
        )
    else:
        st.info("💡 Click 'Prepare and Review Prompt' above to generate analysis prompts.")


def prepare_llm_prompt(region_id, n_papers, n_gap_papers, custom_question=None, guidance_keywords=None, cluster_A=None, cluster_B=None):
    """Generate and store prompts without sending to LLM yet"""
    gap_regions = st.session_state.gap_regions
    region_indices = gap_regions[region_id]
    region_df = st.session_state.df_valid.loc[region_indices]
    
    # Validate cluster selection
    if cluster_A is None or cluster_B is None:
        st.error("❌ Both clusters must be selected")
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
        
        # Gap region papers - top N papers sorted by gap_score (highest first)
        region_df_sorted = region_df.sort_values('gap_score', ascending=False) if 'gap_score' in region_df.columns else region_df
        n_gap_to_include = min(n_gap_papers, len(region_df_sorted))
        
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
        
        # Build output schema with optional custom question field
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
},
"axes_of_separation": [{
    "axis": "materials|ligands|disease|model|delivery|toxicity|methods|other",
    "what_differs": "short explanation (evidence-grounded)",
    "evidence_A": ["doc_id"],
    "evidence_B": ["doc_id"],
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
        
        user_prompt = f"""TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
Focus on: materials, surface chemistry/coatings, size/shape, targeting ligands, disease areas,
models (in vitro/in vivo/clinical), delivery routes, pharmacokinetics/biodistribution,
toxicity/regulatory language, endpoints/outcomes.

CONTEXT:
- cluster_A_meta: {{"id": {cluster_A}, "n_docs": {cluster_counts[cluster_A]}}}
- cluster_B_meta: {{"id": {cluster_B}, "n_docs": {cluster_counts[cluster_B]}}}
- Gap region {region_id}: {len(region_indices)} total papers with low density (potential research opportunities)

The gap papers are included in the evidence pack below. Use them to understand what research lies
between the two clusters and identify bridge opportunities.

{custom_question_section}{keywords_guidance_section}

EVIDENCE PACK (JSONL; each line is one doc):
- Papers with cluster="A": Top {n_papers} representative papers from Cluster A (closest to centroid)
- Papers with cluster="B": Top {n_papers} representative papers from Cluster B (closest to centroid)
- Papers with cluster="GAP": Top {n_gap_to_include} gap papers from Region {region_id} (sorted by gap_score, highest first)

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


def send_llm_prompt(api_key, model, system_prompt, user_prompt, region_id, cluster_A, cluster_B, region_size):
    """Send the edited prompts to LLM and get results"""
    try:
        with st.spinner(f"🤖 Sending prompt to {model}..."):
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
            
            # Store results in session state for persistence
            st.session_state.llm_results = {
                'result': result,
                'region_id': region_id,
                'cluster_A': cluster_A,
                'cluster_B': cluster_B,
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


def generate_llm_explanation(region_id, api_key, model, n_papers, n_gap_papers, show_viz, show_prompt_editor, custom_question=None, guidance_keywords=None, cluster_A=None, cluster_B=None):
    """Generate evidence-grounded LLM explanation using gap region as context and contrasting any two clusters"""
    gap_regions = st.session_state.gap_regions
    region_indices = gap_regions[region_id]
    region_df = st.session_state.df_valid.loc[region_indices]
    
    # Validate cluster selection
    if cluster_A is None or cluster_B is None:
        st.error("❌ Both clusters must be selected")
        return
    
    # Get cluster column based on selected clustering method
    cluster_col = f'cluster_{st.session_state.selected_clustering}'
    if cluster_col not in st.session_state.df_valid.columns:
        st.error(f"❌ Clustering column '{cluster_col}' not found")
        return
    
    st.markdown(f"### 🔍 Analyzing Gap Region {region_id}")
    st.markdown(f"**Contrasting**: Cluster {cluster_A} vs Cluster {cluster_B}")
    st.markdown(f"**Evidence Source**: Gap Region {region_id} ({len(region_indices)} papers)")
    
    # Visualization of clusters and gap region
    if show_viz and st.session_state.X_umap_2d is not None:
        st.markdown("#### 📊 Cluster Visualization with Gap Region")
        
        fig = go.Figure()
        
        # All points colored by cluster (background)
        df_plot = st.session_state.df_valid.copy()
        df_plot['umap_x'] = st.session_state.X_umap_2d[:, 0]
        df_plot['umap_y'] = st.session_state.X_umap_2d[:, 1]
        
        # Plot all clusters in background using the selected clustering method
        for cluster_id in df_plot[cluster_col].unique():
            if cluster_id == -1:
                continue
            cluster_mask = df_plot[cluster_col] == cluster_id
            cluster_data = df_plot[cluster_mask]
            
            # Highlight the two clusters being compared
            if cluster_id == cluster_A:
                color = 'blue'
                size = 10
                opacity = 0.6
                name = f'Cluster {cluster_id} (A)'
            elif cluster_id == cluster_B:
                color = 'green'
                size = 10
                opacity = 0.6
                name = f'Cluster {cluster_id} (B)'
            else:
                color = 'lightgray'
                size = 5
                opacity = 0.2
                name = f'Cluster {cluster_id}'
            
            # Create hover text
            hover_text = [
                f"<b>{row.get('title', 'N/A')}</b><br>" +
                f"Cluster: {cluster_id}<br>" +
                f"{str(row.get('abstract', row.get('processed_content', '')))[:200]}..."
                for _, row in cluster_data.iterrows()
            ]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['umap_x'],
                y=cluster_data['umap_y'],
                mode='markers',
                marker=dict(size=size, color=color, opacity=opacity),
                name=name,
                showlegend=True,
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Overlay gap region as red stars
        gap_data = df_plot.loc[region_indices]
        gap_hover_text = [
            f"<b>{row.get('title', 'N/A')}</b><br>" +
            f"Gap Region: {region_id}<br>" +
            f"Gap Score: {row.get('gap_score', 0):.3f}<br>" +
            f"{str(row.get('abstract', row.get('processed_content', '')))[:200]}..."
            for _, row in gap_data.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=gap_data['umap_x'],
            y=gap_data['umap_y'],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star', 
                       line=dict(color='darkred', width=1)),
            name=f'Gap Region {region_id}',
            text=gap_hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Gap Region {region_id} between Cluster {cluster_A} and {cluster_B}',
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            height=500,
            hovermode='closest',
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", namelength=-1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with st.spinner(f"🤖 Generating contrastive explanation using {model}..."):
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
            
            # Build evidence pack
            evidence_pack = []
            for idx in top_A_idx:
                row = st.session_state.df_valid.iloc[idx]
                # Try to get paper ID from common column names
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
                # Try to get paper ID from common column names
                paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
                evidence_pack.append({
                    "doc_id": f"B_{paper_id}",
                    "paper_id": str(paper_id),
                    "title": str(row.get('title', '')),
                    "year": int(row.get('publication_year', -1)) if pd.notna(row.get('publication_year')) else -1,
                    "abstract": str(row.get('abstract', row.get('processed_content', '')))[:500],
                    "cluster": "B"
                })
            
            # Gap region papers - top N papers sorted by gap_score (highest first)
            region_df = st.session_state.df_valid.loc[region_indices].copy()
            region_df_sorted = region_df.sort_values('gap_score', ascending=False) if 'gap_score' in region_df.columns else region_df
            
            # Limit to n_gap_papers
            n_gap_to_include = min(n_gap_papers, len(region_df_sorted))
            
            for i, (idx, row) in enumerate(region_df_sorted.iterrows()):
                if i >= n_gap_to_include:
                    break
                # Try to get paper ID from common column names
                paper_id = row.get('pmid', row.get('id', row.get('paper_id', row.get('doi', idx))))
                # Determine which cluster this gap paper belongs to (if any)
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
            
            # Enhanced prompt with domain-specific axes
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
            
            # Build output schema with optional custom question field
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
            },
            "axes_of_separation": [{
                "axis": "materials|ligands|disease|model|delivery|toxicity|methods|other",
                "what_differs": "short explanation (evidence-grounded)",
                "evidence_A": ["doc_id"],
                "evidence_B": ["doc_id"],
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
            
            user_prompt = f"""TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
            Focus on: materials, surface chemistry/coatings, size/shape, targeting ligands, disease areas,
            models (in vitro/in vivo/clinical), delivery routes, pharmacokinetics/biodistribution,
            toxicity/regulatory language, endpoints/outcomes.

            CONTEXT:
            - cluster_A_meta: {{"id": {cluster_A}, "n_docs": {cluster_counts[cluster_A]}}}
            - cluster_B_meta: {{"id": {cluster_B}, "n_docs": {cluster_counts[cluster_B]}}}
            - Gap region {region_id}: {len(region_indices)} total papers with low density (potential research opportunities)
            
            The gap papers are included in the evidence pack below. Use them to understand what research lies
            between the two clusters and identify bridge opportunities.
            
            {custom_question_section}{keywords_guidance_section}

            EVIDENCE PACK (JSONL; each line is one doc):
            - Papers with cluster="A": Top {n_papers} representative papers from Cluster A (closest to centroid)
            - Papers with cluster="B": Top {n_papers} representative papers from Cluster B (closest to centroid)
            - Papers with cluster="GAP": Top {n_gap_to_include} gap papers from Region {region_id} (sorted by gap_score, highest first)
            
            ```jsonl
            {chr(10).join(json.dumps(d, ensure_ascii=False) for d in evidence_pack)}
            ```

            OUTPUT JSON SCHEMA:
            {output_schema}
            """
            
            # Show prompt editor if requested
            if show_prompt_editor:
                st.markdown("---")
                st.markdown("### 📝 Prompt Editor")
                st.markdown("Edit the system and user prompts below before sending to the LLM:")
                
                # Initialize prompt storage in session state if needed
                if 'edited_system_prompt' not in st.session_state:
                    st.session_state.edited_system_prompt = system_prompt
                if 'edited_user_prompt' not in st.session_state:
                    st.session_state.edited_user_prompt = user_prompt
                
                with st.expander("🔧 System Prompt", expanded=True):
                    edited_system_prompt = st.text_area(
                        "System Prompt",
                        value=st.session_state.edited_system_prompt,
                        height=150,
                        key="system_prompt_editor",
                        label_visibility="collapsed",
                        on_change=lambda: setattr(st.session_state, 'edited_system_prompt', st.session_state.system_prompt_editor)
                    )
                
                with st.expander("📋 User Prompt", expanded=True):
                    edited_user_prompt = st.text_area(
                        "User Prompt",
                        value=st.session_state.edited_user_prompt,
                        height=400,
                        key="user_prompt_editor",
                        label_visibility="collapsed",
                        on_change=lambda: setattr(st.session_state, 'edited_user_prompt', st.session_state.user_prompt_editor)
                    )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("💡 You can modify the prompts above. Click the button to send the edited prompts to the LLM.")
                with col2:
                    if st.button("🔄 Reset Prompts", help="Reset to original prompts"):
                        st.session_state.edited_system_prompt = system_prompt
                        st.session_state.edited_user_prompt = user_prompt
                        st.rerun()
                
                send_edited = st.button("✅ Send Edited Prompts to LLM", type="primary", key="send_edited_prompts")
                
                if not send_edited:
                    st.warning("⚠️ Click 'Send Edited Prompts to LLM' above to proceed with the analysis.")
                    return
                
                # Use edited prompts from session state
                system_prompt = st.session_state.edited_system_prompt
                user_prompt = st.session_state.edited_user_prompt
                
                # Clear the stored prompts after sending
                del st.session_state['edited_system_prompt']
                del st.session_state['edited_user_prompt']
                
                st.markdown("---")
            
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
            
            # Store results in session state for persistence
            st.session_state.llm_results = {
                'result': result,
                'region_id': region_id,
                'cluster_A': cluster_A,
                'cluster_B': cluster_B,
                'region_size': len(region_indices),
                'model': model,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.success("✅ Analysis complete! Results are displayed below.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating explanation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_llm_results(result, region_id, cluster_A, cluster_B, region_size):
    """Display LLM analysis results in a structured format"""
    st.success("✅ Analysis complete!")
    
    st.markdown("---")
    st.markdown(f"### 📋 Contrastive Analysis Results")
    st.markdown(f"**Region {region_id}** | Size: {region_size} papers | Comparing Cluster {cluster_A} vs {cluster_B}")
    
    # Cluster summaries side-by-side
    col1, col2 = st.columns(2)
    
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


# ============================================================================
# PAGE: DATABASE EXPLORER
# ============================================================================

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
                    labels={'x': 'Count', 'y': col},
                    title=f"Top 10 values in {col}"
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
# PAGE: EXPORT
# ============================================================================

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
        gap_threshold = st.session_state.df_valid['gap_score'].quantile(st.session_state.config['gap_quantile'])
        n_gaps = (st.session_state.df_valid['gap_score'] >= gap_threshold).sum()
        col2.metric("Gap Candidates", n_gaps)
    
    if st.session_state.gaps_identified:
        col3.metric("Gap Regions", len(st.session_state.gap_regions))
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Full Dataset with Features**")
        if st.button("Download CSV", key="download_full"):
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
    st.dataframe(st.session_state.df_valid.head(20))


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
        
        if 'cluster_hdbscan' in region_df.columns:
            summary['dominant_cluster'] = region_df['cluster_hdbscan'].mode()[0] if len(region_df) > 0 else -1
            summary['n_clusters_spanned'] = region_df['cluster_hdbscan'].nunique()
        
        if 'publication_year' in region_df.columns:
            years = pd.to_numeric(region_df['publication_year'], errors='coerce').dropna()
            if len(years) > 0:
                summary['median_year'] = int(years.median())
                summary['year_range'] = f"{int(years.min())}-{int(years.max())}"
        
        gap_summary.append(summary)
    
    return pd.DataFrame(gap_summary).sort_values('avg_gap_score', ascending=False)


# ============================================================================
# MAIN APP
# ============================================================================

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
