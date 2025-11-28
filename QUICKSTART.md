# Quick Start Guide - Embedding Explorer v2

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```powershell
pip install -r requirements_v2.txt
```

### Step 2: Preprocess Your Embeddings
```powershell
python preprocess_embeddings.py --data_file papers_dataframe_full_processed_with_processed_embeddings_parsed.csv
```

This will:
- ✅ Load your CSV file
- ✅ Parse all embedding columns
- ✅ Compute UMAP 2D projections
- ✅ Cache results in `cache/` directory

**Expected output:**
```
Loading data...
  Loaded 5000 records

Available embedding columns:
  - qwen_processed_content_embedding
  - bert_processed_content_embedding
  - qwen_content_embedding
  - bert_content_embedding

Processing 4 embedding column(s)...
  Processing qwen_processed_content_embedding...
    ✓ Success: cache/qwen_processed_content_embedding.pkl
  Processing bert_processed_content_embedding...
    ✓ Success: cache/bert_processed_content_embedding.pkl
  ...

PREPROCESSING COMPLETE
```

### Step 3: Launch the App
```powershell
streamlit run streamlit_app_v2.py
```

Your browser will open to `http://localhost:8501`

## 📊 Using the App

### Exploration Page

1. **Select Embedding** (sidebar)
   - Choose which embedding type to visualize
   - Switch between different models (Qwen, BERT, etc.)

2. **Interactive Visualization**
   - Hover over points to see title/abstract
   - Zoom and pan to explore regions
   - Each point is a document in 2D UMAP space

3. **Cluster & Filter** (sidebar)
   - **K-means**: Auto-cluster similar documents
   - **Bounding Box**: Select spatial region
   - **Text Search**: Find by keywords
   - **Cluster Filter**: Show specific clusters

4. **Export Results**
   - Click "💾 Export" to save to Excel
   - Files saved to `output/` directory

### Tips for Best Results

**For Dense Clusters:**
```powershell
python preprocess_embeddings.py --data_file <file> --n_neighbors 30 --min_dist 0.0
```

**For Global Structure:**
```powershell
python preprocess_embeddings.py --data_file <file> --n_neighbors 50 --min_dist 0.3
```

**Quick Test (subset of data):**
```powershell
python preprocess_embeddings.py --data_file <file> --subset 1000
```

## 🧪 Test the System

Run the test script to verify everything works:
```powershell
python test_embedding_explorer.py
```

This creates a synthetic dataset and tests the full pipeline.

## 📁 File Structure After Setup

```
your_project/
├── cache/                                           # Preprocessed UMAP projections
│   ├── qwen_processed_content_embedding.pkl
│   ├── bert_processed_content_embedding.pkl
│   └── ...
├── output/                                          # Exported Excel files
│   └── export_20250118_143022.xlsx
├── streamlit_app_v2.py                             # Main app
├── preprocess_embeddings.py                        # Preprocessing script
└── utils/
    ├── preprocessing.py
    ├── data_utils_v2.py
    └── cluster_utils_v2.py
```

## 🔧 Troubleshooting

### "No preprocessed data found"
→ Run preprocessing first: `python preprocess_embeddings.py --data_file <file>`

### "File not found"
→ Check the path to your CSV file (use absolute path if needed)

### UMAP is slow
→ Use `--subset 1000` for testing, or reduce `--n_neighbors`

### Can't find embedding columns
→ Your CSV needs columns with "embedding" in the name containing arrays

## 📞 Need Help?

Check the detailed README: `README_v2.md`

## 🎯 Example Workflow

```powershell
# 1. Preprocess all embeddings
python preprocess_embeddings.py `
    --data_file papers_dataframe_full_processed_with_processed_embeddings_parsed.csv `
    --embedding_cols "qwen_processed_content_embedding,bert_processed_content_embedding"

# 2. Launch app
streamlit run streamlit_app_v2.py

# In the app:
# - Select "qwen_processed_content_embedding"
# - Run K-means with 10 clusters
# - Search for "cancer" and assign to class "oncology"
# - Filter to show only "oncology" cluster
# - Export to Excel
```

That's it! 🎉
