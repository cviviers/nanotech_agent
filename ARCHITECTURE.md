# Embedding Explorer v2 - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING EXPLORER V2                                │
│                         Two-Phase Architecture                               │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
PHASE 1: PREPROCESSING (One-time, or when data changes)
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────┐
│   Your CSV File  │
│  ┌─────────────┐ │
│  │ id          │ │
│  │ title       │ │
│  │ abstract    │ │
│  │ qwen_emb    │ │  <- String representation: "[-0.027, -0.067, ...]"
│  │ bert_emb    │ │
│  └─────────────┘ │
└────────┬─────────┘
         │
         │ python preprocess_embeddings.py --data_file data.csv
         │
         ▼
┌─────────────────────┐
│  Parse Embeddings   │
│  (utils/            │
│   data_utils_v2.py) │
└─────────┬───────────┘
          │ Convert strings → numpy arrays
          ▼
┌─────────────────────┐      ┌──────────────────────┐
│  Compute UMAP       │      │  Configuration       │
│  (utils/            │◀─────│  • n_neighbors: 15   │
│   preprocessing.py) │      │  • min_dist: 0.1     │
└─────────┬───────────┘      │  • metric: cosine    │
          │                  └──────────────────────┘
          │ Project to 2D
          │
          ▼
┌────────────────────────────────────────┐
│  Cached Results (cache/*.pkl)          │
│  ┌───────────────────────────────────┐ │
│  │ • Original dataframe              │ │
│  │ • Added columns:                  │ │
│  │   - low_x (UMAP dimension 1)      │ │
│  │   - low_y (UMAP dimension 2)      │ │
│  │   - size (plot marker size)       │ │
│  │   - cluster_label (initial state) │ │
│  │ • UMAP projection array           │ │
│  │ • Config used                     │ │
│  └───────────────────────────────────┘ │
└────────────────────────────────────────┘
          │
          │ One file per embedding type:
          │ • cache/qwen_processed_content_embedding.pkl
          │ • cache/bert_processed_content_embedding.pkl
          │ • etc.
          ▼
    ✅ Ready for exploration!


═══════════════════════════════════════════════════════════════════════════════
PHASE 2: INTERACTIVE EXPLORATION (Instant loading, real-time interaction)
═══════════════════════════════════════════════════════════════════════════════

        streamlit run streamlit_app_v2.py
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT APP                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         PAGE: PREPROCESSING                           │  │
│  │  • Load CSV file                                                      │  │
│  │  • Configure UMAP parameters                                          │  │
│  │  • Run preprocessing (calls utils/preprocessing.py)                   │  │
│  │  • View progress                                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         PAGE: EXPLORATION                             │  │
│  │                                                                        │  │
│  │  ┌──────────────────┐         ┌──────────────────────────────────┐   │  │
│  │  │   SIDEBAR        │         │      MAIN VIEW                   │   │  │
│  │  │                  │         │                                  │   │  │
│  │  │ Select Embedding │         │  ┌─────────────────────────────┐ │   │  │
│  │  │  ▢ qwen_proc     │         │  │                             │ │   │  │
│  │  │  ▣ bert_proc     │         │  │    Interactive Scatter Plot │ │   │  │
│  │  │  ▢ qwen          │         │  │         (Altair)            │ │   │  │
│  │  │  ▢ bert          │         │  │                             │ │   │  │
│  │  │                  │         │  │  • Hover for tooltips       │ │   │  │
│  │  ├──────────────────┤         │  │  • Zoom/pan controls        │ │   │  │
│  │  │ K-means          │         │  │  • Color by cluster         │ │   │  │
│  │  │  Clusters: [3 ]  │         │  │                             │ │   │  │
│  │  │  [Run K-means]   │────────▶│  └─────────────────────────────┘ │   │  │
│  │  │                  │         │                                  │   │  │
│  │  ├──────────────────┤         │  [↩️ Undo] [💾 Export] [🔄 Refresh] │   │  │
│  │  │ Bounding Box     │         │                                  │   │  │
│  │  │  x: [0.0][1.0]   │         │  ┌─────────────────────────────┐ │   │  │
│  │  │  y: [0.0][1.0]   │         │  │     Data Table (Optional)   │ │   │  │
│  │  │  [Apply Filter]  │────────▶│  │  title | abstract | cluster │ │   │  │
│  │  │                  │         │  └─────────────────────────────┘ │   │  │
│  │  ├──────────────────┤         │                                  │   │  │
│  │  │ Filter Clusters  │         │                                  │   │  │
│  │  │  ☑ Cluster 0     │         │                                  │   │  │
│  │  │  ☑ Cluster 1     │         │                                  │   │  │
│  │  │  ☐ Cluster 2     │────────▶│  (Updates plot in real-time)    │   │  │
│  │  │  [Filter]        │         │                                  │   │  │
│  │  │                  │         │                                  │   │  │
│  │  ├──────────────────┤         │                                  │   │  │
│  │  │ Text Search      │         │                                  │   │  │
│  │  │  Term: [cancer]  │         │                                  │   │  │
│  │  │  Class: [1]      │         │                                  │   │  │
│  │  │  [Assign][Filter]│────────▶│  (Searches title + abstract)    │   │  │
│  │  └──────────────────┘         └──────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    │ Export action
                    ▼
┌─────────────────────────────────────────┐
│  output/export_20250118_143022.xlsx     │
│  (Filtered subset, without embeddings)  │
└─────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
DATA FLOW & TRANSFORMATIONS
═══════════════════════════════════════════════════════════════════════════════

Original Data (CSV)
│
├─ Columns: id, title, abstract, qwen_emb (string), bert_emb (string), ...
│
└─► Parse (utils/data_utils_v2.py::parse_embedding_column)
    │
    ├─ Embeddings: qwen_emb (numpy array), bert_emb (numpy array)
    │
    └─► UMAP (utils/preprocessing.py::preprocess_embeddings)
        │
        ├─ New columns: low_x, low_y, size, cluster_label
        │
        └─► Cache (pickle format)
            │
            └─► Load in Streamlit (instant)
                │
                ├─► K-means (utils/cluster_utils_v2.py::kmeans_cluster)
                │   └─ Updates: cluster_label
                │
                ├─► Filter BBox (utils/cluster_utils_v2.py::filter_by_bounding_box)
                │   └─ Returns: Filtered dataframe
                │
                ├─► Text Assign (utils/cluster_utils_v2.py::assign_class_to_embeddings)
                │   └─ Updates: cluster_label (based on text search)
                │
                └─► Export (utils/data_utils_v2.py::write_df_to_excel)
                    └─ Saves: Filtered data to Excel (excludes embeddings)


═══════════════════════════════════════════════════════════════════════════════
COMPONENT DEPENDENCIES
═══════════════════════════════════════════════════════════════════════════════

streamlit_app_v2.py
│
├─► utils/preprocessing.py
│   ├─ EmbeddingConfig (dataclass)
│   ├─ preprocess_embeddings()
│   ├─ load_preprocessed_data()
│   └─ Dependencies: umap-learn, numpy, pandas
│
├─► utils/data_utils_v2.py
│   ├─ load_dataframe()
│   ├─ parse_embedding_column()
│   ├─ write_df_to_excel()
│   └─ Dependencies: pandas, openpyxl
│
└─► utils/cluster_utils_v2.py
    ├─ kmeans_cluster()
    ├─ assign_class_to_embeddings()
    ├─ filter_by_bounding_box()
    ├─ filter_by_clusters()
    └─ Dependencies: scikit-learn, numpy, pandas


preprocess_embeddings.py (CLI)
│
├─► utils/preprocessing.py
└─► utils/data_utils_v2.py


═══════════════════════════════════════════════════════════════════════════════
SESSION STATE MANAGEMENT (Streamlit)
═══════════════════════════════════════════════════════════════════════════════

st.session_state
│
├─ 'df': Current working dataframe
├─ 'df_history': List[DataFrame] (for undo)
├─ 'current_embedding': str (which embedding is loaded)
├─ 'preprocessed_data': Dict (in-memory cache)
├─ 'kmeans_clusters': int (k value)
└─ 'plot': Altair chart object

Actions that modify state:
│
├─ Load new embedding → Updates 'df', resets 'df_history'
├─ K-means → Appends to 'df_history', updates 'df'
├─ Filter → Appends to 'df_history', updates 'df'
├─ Text assign → Appends to 'df_history', updates 'df'
└─ Undo → Pops from 'df_history', updates 'df'


═══════════════════════════════════════════════════════════════════════════════
FILE FORMATS
═══════════════════════════════════════════════════════════════════════════════

Input (CSV):
├─ Embeddings as strings: "[-0.027, -0.067, 0.031, ...]"
├─ Metadata: title, abstract, authors, year, etc.
└─ Size: Can be 100MB+ for large datasets

Cache (Pickle):
├─ Full dataframe with UMAP coordinates
├─ UMAP projection array (N x 2)
├─ Configuration metadata
└─ Size: ~10% smaller due to numpy arrays

Output (Excel):
├─ Filtered dataframe
├─ Excludes embedding columns (too large)
├─ Includes: title, abstract, cluster_label, low_x, low_y
└─ Size: Small, human-readable


═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE CHARACTERISTICS
═══════════════════════════════════════════════════════════════════════════════

Operation               Original App        V2 App              Speedup
────────────────────────────────────────────────────────────────────────────
Initial load            Compute UMAP        Load cache          10-100x
                        (~2 min for 10K)    (~2 sec for 10K)    

Switch embedding        Recompute UMAP      Load different      Instant
                        (~2 min)            cache (~2 sec)      

K-means clustering      On full embeddings  On UMAP coords      5-10x
                        (~5 sec)            (<1 sec)            

Filtering               DataFrame ops       DataFrame ops       Same
                        (instant)           (instant)           

Export                  Basic              Smart (exclude      Better
                                           embeddings)         

Memory usage            High (all data)    Medium (cached)     Lower


═══════════════════════════════════════════════════════════════════════════════
KEY DESIGN PATTERNS
═══════════════════════════════════════════════════════════════════════════════

1. SEPARATION OF CONCERNS
   • Preprocessing: utils/preprocessing.py
   • Data I/O: utils/data_utils_v2.py
   • Clustering: utils/cluster_utils_v2.py
   • UI: streamlit_app_v2.py

2. CACHING STRATEGY
   • Expensive operations (UMAP) → Disk cache (pickle)
   • UI state → Session state
   • Visualization → @st.cache_data

3. IMMUTABILITY
   • Always copy dataframe before modifications
   • Maintain history stack for undo
   • Cache is read-only

4. CONFIGURATION
   • Dataclass for type safety (EmbeddingConfig)
   • CLI arguments for flexibility
   • Sensible defaults

5. ERROR HANDLING
   • Validate file paths
   • Check for required columns
   • Graceful fallbacks (missing columns)
   • User-friendly error messages
```

## Summary

**Two-Phase Architecture:**
1. **Preprocessing** (slow, one-time): CSV → UMAP → Cache
2. **Exploration** (fast, interactive): Cache → Visualize → Analyze

**Key Benefits:**
- ⚡ 10-100x faster loading
- 🔄 Switch between embeddings instantly
- 🎯 Configurable UMAP parameters
- 📊 Scales to 100K+ documents

**Workflow:**
```
First time: Preprocess → Cache → Explore
Next time:  ──────────────────┘ Explore (instant!)
```
