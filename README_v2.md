# Embedding Explorer v2

A modern Streamlit application for interactive exploration of high-dimensional embeddings with preprocessing pipeline.

## Features

### 🔧 Preprocessing Phase
- **Dimensionality Reduction**: Compute UMAP projections for multiple embedding types
- **Caching**: Preprocessed results are cached for fast loading
- **Multiple Embeddings**: Support for different embedding models (BERT, Qwen, etc.)
- **Configurable**: Adjust UMAP parameters (n_neighbors, min_dist, metric)

### 🔍 Exploration Interface
- **Interactive Visualization**: Explore embeddings in 2D UMAP space
- **K-means Clustering**: Dynamically cluster data points
- **Bounding Box Selection**: Filter data by spatial region
- **Text-based Assignment**: Assign labels based on keyword search
- **Cluster Filtering**: Filter by cluster labels
- **Export**: Save filtered results to Excel
- **Undo/Redo**: Track and revert changes

## Installation

### Requirements
```bash
pip install streamlit pandas numpy umap-learn scikit-learn altair openpyxl
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Two-Phase Workflow (Recommended)

#### Step 1: Preprocessing
Run the standalone preprocessing script:

```bash
python preprocess_embeddings.py \
    --data_file papers_dataframe_full_processed_with_processed_embeddings_parsed.csv \
    --embedding_cols qwen_processed_content_embedding,bert_processed_content_embedding \
    --n_neighbors 15 \
    --min_dist 0.1 \
    --metric cosine
```

**Parameters:**
- `--data_file`: Path to your CSV file with embeddings
- `--embedding_cols`: Comma-separated list of embedding columns to process (optional, defaults to all)
- `--n_neighbors`: UMAP n_neighbors (default: 15)
- `--min_dist`: UMAP min_dist (default: 0.1)
- `--metric`: Distance metric (cosine, euclidean, manhattan) (default: cosine)
- `--cache_dir`: Directory to save cached results (default: cache)
- `--subset`: Use only N records for testing (optional)

This creates cached UMAP projections in the `cache/` directory.

#### Step 2: Launch Streamlit App
```bash
streamlit run streamlit_app_v2.py
```

Navigate to the **Exploration** page to interactively explore your preprocessed embeddings.

### Option 2: All-in-One Streamlit Interface

```bash
streamlit run streamlit_app_v2.py
```

1. Go to **Preprocessing** page
2. Load your CSV file
3. Select embedding columns to process
4. Configure UMAP parameters
5. Click "Run Preprocessing"
6. Switch to **Exploration** page

## Data Format

Your CSV should contain:

### Required Columns
- Embedding columns (e.g., `qwen_processed_content_embedding`, `bert_content_embedding`)
  - Format: String representation of lists or actual numpy arrays
  - Example: `"[0.1, -0.2, 0.3, ...]"`

### Optional Columns (for better exploration)
- `title`: Paper/document title
- `abstract`: Paper/document abstract
- `authors`: Author list
- `publication_year`: Year of publication
- Any other metadata you want to explore

### Example Data Structure
```csv
id,title,abstract,qwen_processed_content_embedding,bert_content_embedding
10025624,"MR imaging...","The objective...","[-0.027, -0.067, ...]","[0.007, -0.062, ...]"
```

## Workflow

### 1. Preprocessing
```
CSV File → Parse Embeddings → UMAP Projection → Cache Results
```

### 2. Exploration
```
Load Cached Data → Visualize → Cluster/Filter → Export
```

### Actions Available

#### Clustering
- **K-means**: Automatically group similar embeddings
- Configurable number of clusters (2-100)

#### Filtering
- **Bounding Box**: Select rectangular region in UMAP space
- **Cluster Filter**: Show only specific clusters
- **Text Search**: Find documents by keyword and assign labels

#### Analysis
- **Interactive Plot**: Zoom, pan, hover for details
- **Tooltips**: View title, abstract, cluster on hover
- **Data Table**: Examine filtered results in tabular format

#### Export
- **Excel Export**: Save current filtered dataset
- Automatically excludes large embedding columns
- Timestamped filenames

## File Structure

```
├── streamlit_app_v2.py           # Main Streamlit application
├── preprocess_embeddings.py      # Standalone preprocessing script
├── utils/
│   ├── preprocessing.py          # UMAP preprocessing functions
│   ├── data_utils_v2.py         # Data loading/parsing utilities
│   └── cluster_utils_v2.py      # Clustering and filtering functions
├── cache/                        # Preprocessed UMAP projections (auto-created)
├── output/                       # Exported Excel files (auto-created)
└── README_v2.md                 # This file
```

## Advanced Usage

### Custom UMAP Parameters

For different dataset characteristics:

**Dense clusters:**
```bash
--n_neighbors 30 --min_dist 0.0
```

**Preserve global structure:**
```bash
--n_neighbors 50 --min_dist 0.3
```

**Fast preview (small subset):**
```bash
--subset 1000 --n_neighbors 10
```

### Multiple Embedding Comparison

Preprocess multiple embeddings:
```bash
python preprocess_embeddings.py \
    --data_file data.csv \
    --embedding_cols "qwen_processed_content_embedding,bert_processed_content_embedding,qwen_content_embedding,bert_content_embedding"
```

Then switch between them in the Streamlit app sidebar.

### Programmatic Usage

```python
from utils.preprocessing import preprocess_embeddings, EmbeddingConfig
from utils.data_utils_v2 import load_dataframe, parse_embedding_column
from pathlib import Path

# Load data
df = load_dataframe("data.csv")
df = parse_embedding_column(df, "qwen_processed_content_embedding")

# Configure UMAP
config = EmbeddingConfig(n_neighbors=15, min_dist=0.1, metric="cosine")

# Preprocess
result = preprocess_embeddings(
    df=df,
    embedding_col="qwen_processed_content_embedding",
    config=config,
    cache_dir=Path("cache")
)

# Access results
df_with_umap = result['df']
umap_coords = result['umap_2d']
```

## Troubleshooting

### "No preprocessed data found"
- Run preprocessing first (either via script or Streamlit preprocessing page)
- Check that `cache/` directory contains `.pkl` files

### "File too large"
- Use `--subset` parameter to test with smaller dataset first
- Consider preprocessing on a machine with more RAM

### Embeddings not parsing
- Ensure embedding columns are string representations of lists
- Check for NaN values in embedding columns
- Verify embedding format: `"[0.1, 0.2, ...]"`

### UMAP runs slowly
- Reduce `n_neighbors` for faster computation
- Use `--subset` for initial testing
- Consider using `metric='euclidean'` instead of `'cosine'`

## Tips

1. **Start Small**: Test with `--subset 1000` first
2. **Cache Everything**: Preprocessing can take time, but cached results load instantly
3. **Compare Embeddings**: Preprocess multiple embedding types to find the best representation
4. **Iterate on UMAP**: Try different parameters to find optimal 2D projections
5. **Use Text Search**: Quickly identify and label interesting document clusters

## Comparison with Original App

| Feature | Original | v2 |
|---------|----------|-----|
| Preprocessing | On-the-fly | Cached, separate phase |
| Multiple Embeddings | Single | Multiple, switchable |
| UMAP Parameters | Fixed | Configurable |
| Performance | Slower startup | Instant load from cache |
| Data Format | Specific JSON | Flexible CSV/JSON |
| Export | Excel | Excel with better formatting |

## License

MIT

## Contributing

Feel free to open issues or submit pull requests for improvements!
