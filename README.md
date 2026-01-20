# Nanotechnology Research Gap Discovery System

A comprehensive platform for analyzing scientific literature in nanotechnology and nanomedicine to identify research gaps, novelty opportunities, and unexplored areas using advanced NLP embeddings, clustering, and LLM-powered analysis.

## 🎯 Overview

This repository implements an end-to-end pipeline for discovering research gaps in nanotechnology literature by:

1. **Data Collection**: Extracting scientific papers from PubMed
2. **Preprocessing**: Cleaning and standardizing paper metadata and content
3. **Embedding Generation**: Creating semantic embeddings using BERT and Qwen models
4. **Novelty Analysis**: Identifying low-density regions in embedding space that represent potential research gaps
5. **Interactive Exploration**: Streamlit-based application for visualizing and analyzing results
6. **LLM-Powered Insights**: Using GPT models to explain and characterize research gaps

## 📁 Repository Structure

```
├── 1.pubmed_data_extraction.ipynb       # PubMed data extraction pipeline
├── 2.data_preprocessing.ipynb           # Data cleaning and preprocessing
├── 3.create_bert_embeddings.ipynb       # BERT embedding generation
├── 3.create_qwen_embeddings.ipynb       # Qwen embedding generation
├── novelty_analysis_app.py              # Main Streamlit application
├── requirements.txt                      # Python dependencies
│
├── data/                                 # Processed datasets
│   ├── all_papers.json                  # Complete paper database
│   ├── cleaned_dataset.json             # Preprocessed papers
│   ├── bert_embeddings.npy              # BERT embeddings
│   ├── bert_embeddings_metadata.json    # BERT metadata
│   ├── qwen_embeddings.npy              # Qwen embeddings
│   └── qwen_embeddings_metadata.json    # Qwen metadata
│
├── papers/                               # Individual paper JSON files (by PMID)
│   ├── 10025624.json
│   ├── 10080268.json
│   └── ...
│
├── embedding_models/                     # Embedding model services
│   ├── bert.py                          # BERT embedding service
│   ├── qwen.py                          # Qwen embedding FastAPI service
│   ├── eval.py                          # Model evaluation utilities
│   ├── bert_eval_results.json           # BERT evaluation metrics
│   ├── qwen_eval_results.json           # Qwen evaluation metrics
│   └── README.md                        # Embedding models documentation
│
├── novelty_app/                         # Refactored modular app (in progress)
│   ├── app.py                           # App entry point
│   ├── config.py                        # Configuration settings
│   ├── core/                            # Core business logic
│   │   ├── state.py                     # Session state management
│   │   ├── data_loader.py               # Data loading utilities
│   │   ├── entities.py                  # Entity extraction
│   │   ├── density.py                   # Density computation
│   │   ├── clustering.py                # Clustering algorithms
│   │   ├── gap_detection.py             # Gap identification
│   │   └── undo.py                      # Undo functionality
│   ├── ui/                              # UI components (planned)
│   └── pages/                           # Page modules (planned)
│
└── utils/                                # Utility modules
    ├── utils.py                         # General utilities & Paper class
    ├── nanotech_discovery.py            # Core novelty discovery pipeline
    ├── preprocessing.py                 # Preprocessing utilities
    ├── data_utils.py                    # Data loading/export utilities
    ├── cluster_utils.py                 # Clustering utilities
    └── lda_utils.py                     # LDA topic modeling
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- OpenAI API key (for LLM analysis features)
- GPU recommended for embedding generation

### Installation

1. **Clone the repository**
   ```bash
   cd "c:\Users\20195435\OneDrive - TU Eindhoven\TUe\Playground\Nanotechnology"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key** (optional, for LLM features & also available in the app) 
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Quick Start

**Download Sample Data:**

We provide a preprocessed sample dataset for quick testing:

1. **Download the sample dataset** (recommended for first-time users):
   - [Sample Dataset (all_papers.json + embeddings)](https://drive.google.com/file/d/17s2DqYdRrUP3nuxPViFs5KFChtYc-xMJ/view?usp=sharing)
   - Extract the ZIP file to your project root directory
   - The ZIP contains:
     - `data/all_papers.json` - Preprocessed papers with metadata
     - `data/bert_embeddings.npy` - BERT embeddings
     - `data/bert_embeddings_metadata.json` - BERT metadata
     - `data/qwen_embeddings.npy` - Qwen embeddings
     - `data/qwen_embeddings_metadata.json` - Qwen metadata

2. **Optional: Download raw papers** (for full pipeline exploration):
   - [Raw Papers (individual JSON files)](https://drive.google.com/file/d/1MP64I2MEPuNb5agwrJADemYer-ibAcfy/view?usp=sharing)
   - Extract to `papers/` directory in project root

**Run the Novelty Analysis App:**
```bash
streamlit run novelty_app/app.py
```
**Run the Embeddings Models:**
In two separate terminals

```bash
cd embedding_models

uvicorn qwen:app --host 0.0.0.0 --port 8000
```

```bash
cd embedding_models

uvicorn bert:app --host 0.0.0.0 --port 8001
```


## 📊 Data Pipeline

### 1. PubMed Data Extraction (`1.pubmed_data_extraction.ipynb`)

Extracts scientific papers from PubMed database using Biopython:

- **Input**: PubMed query or list of PMIDs
- **Output**: Individual JSON files per paper in `papers/` directory
- **Features**:
  - Batch processing with chunking (10,000 papers per chunk)
  - Duplicate detection
  - Incremental updates based on revision dates
  - Extracts: title, abstract, authors, journal, publication date, DOI, keywords, MeSH terms

**Key Functions**:
- `fetch_pubmed_data_given_ids_in_chunks()`: Batch fetching with error handling
- `Paper` dataclass: Structured paper representation

### 2. Data Preprocessing (`2.data_preprocessing.ipynb`)

Cleans and standardizes the extracted data:

- **Input**: JSON files from `papers/` directory
- **Output**: `data/cleaned_dataset.json` and `data/all_papers.json`
- **Processing Steps**:
  - Text cleaning and normalization
  - Abstract processing
  - Metadata standardization
  - Language detection
  - Duplicate removal
  - Missing value handling

### 3. Embedding Generation

#### BERT Embeddings (`3.create_bert_embeddings.ipynb`)

Generates semantic embeddings using BioClinical-ModernBERT:

- **Model**: `Lihuchen/BioClinical-ModernBERT-base` (768-dim)
- **Output**: 
  - `data/bert_embeddings.npy`: Embedding vectors
  - `data/bert_embeddings_metadata.json`: Paper metadata
- **Features**:
  - Batch processing for efficiency
  - GPU acceleration
  - Content + processed content embeddings
  - Dimensionality reduction evaluation (UMAP, PCA, t-SNE)
  - Clustering quality metrics

#### Qwen Embeddings (`3.create_qwen_embeddings.ipynb`)

Alternative embeddings using Qwen3-Embedding:

- **Model**: `Qwen/Qwen3-Embedding-0.6B` (1024-dim)
- **Output**: 
  - `data/qwen_embeddings.npy`: Embedding vectors
  - `data/qwen_embeddings_metadata.json`: Paper metadata
- **Features**:
  - FastAPI service for embeddings
  - Reranking capabilities
  - Instruction-aware embeddings

## 🔬 Novelty Discovery Pipeline

The core novelty discovery is implemented in `utils/nanotech_discovery.py` following a 7-step process:

### Step 1: Load Data
- Load embeddings and metadata
- Parse string representations of arrays
- Prepare DataFrame with all features

### Step 2: Dimensionality Reduction
- **Methods**: UMAP, PCA, t-SNE
- **Purpose**: Reduce high-dimensional embeddings (768/1024-D) to 2D/3D for visualization
- **Evaluation**: Trustworthiness scores to validate reduction quality

### Step 3: Nearest Neighbor Graph Construction
- Build k-NN graph in high-dimensional space
- Connect papers to their semantic neighbors
- Basis for density estimation and gap detection

### Step 4: Density Estimation
- **Local Density**: Average distance to k-nearest neighbors
- **Bootstrap Stability**: Repeated sampling to identify consistently low-density regions
- **Gap Score**: Frequency of appearing in lowest-density quantile across bootstraps
- **Output**: Papers ranked by "gap score" (higher = more novel/unexplored)

### Step 5: Gap Region Identification
- Connected components in gap subgraph
- Identifies coherent regions of low density
- Minimum region size filtering
- Characterization of each gap region

### Step 6: Temporal Analysis
- Publication date patterns in gap regions
- Identifies emerging vs. persistent gaps
- Time-based evolution of research areas

### Step 7: LLM-Powered Gap Explanation
- Uses OpenAI GPT models to:
  - Summarize papers in gap regions
  - Explain why the gap exists
  - Suggest potential research directions
  - Compare gap regions to dense areas

## 🎨 Novelty Analysis App Features

The Streamlit application (`novelty_analysis_app.py`) provides an interactive interface with 9 main sections:

### 📊 1. Data & Config
- Load embeddings (BERT or Qwen)
- Configure analysis parameters:
  - Number of neighbors (k)
  - Density quantile
  - Bootstrap iterations
  - Random seed
- Data quality metrics and statistics

### 🧬 2. Embeddings
- Dimensionality reduction (UMAP, PCA, t-SNE)
- Interactive 2D/3D visualizations
- Parameter tuning and trustworthiness evaluation
- Embedding quality metrics

### 🎯 3. Filters
- Filter papers by:
  - Publication year range
  - Keywords (AND/OR logic)
  - Journals
  - Authors
  - Custom metadata fields
- Apply filters with undo functionality
- Visual feedback on filtering impact

### 🔬 4. Clustering
- **Algorithms**: 
  - K-Means
  - HDBSCAN
  - Leiden (community detection)
  - Louvain (community detection)
- **Features**:
  - Silhouette score evaluation
  - Cluster visualization in reduced space
  - TF-IDF analysis for cluster characterization
  - Entity extraction (diseases, chemicals, genes)

### 🔍 5. Gap Analysis
- Compute local density scores
- Bootstrap stability analysis
- Gap score calculation
- Identify top gap candidates
- Interactive scatter plots with gap highlighting

### 📍 6. Gap Regions
- Connected component analysis
- Region-based gap identification
- Minimum size filtering
- Region characterization:
  - Size and density statistics
  - Keyword analysis
  - Temporal patterns
  - Representative papers

### 🤖 7. LLM Analysis
- OpenAI GPT integration for:
  - Gap region summarization
  - Research direction suggestions
  - Novelty assessment
  - Comparative analysis
- Configurable prompts and parameters
- Batch processing multiple regions

### 🗄️ 8. Database Explorer
- Search and filter papers
- View detailed paper information
- Explore paper metadata
- Export selected papers

### 📤 9. Export
- Export analysis results to Excel
- Save filtered datasets
- Export gap regions and candidates
- Configuration export for reproducibility

## 🛠️ Embedding Models

### BERT Service (`embedding_models/bert.py`)

Local BERT embedding service:
- Model: `Lihuchen/BioClinical-ModernBERT-base`
- 768-dimensional embeddings
- Optimized for biomedical text

### Qwen FastAPI Service (`embedding_models/qwen.py`)

RESTful API for Qwen embeddings:

**Start the service:**
```bash
cd embedding_models
uvicorn qwen:app --port 8000 --reload
```

**Endpoints:**
- `GET /`: Health check and model info
- `POST /embed`: Generate embeddings
- `POST /similarity`: Compute cosine similarities
- `POST /rerank`: Rerank documents by relevance

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Model Evaluation

Evaluation results available in:
- `embedding_models/bert_eval_results.json`
- `embedding_models/qwen_eval_results.json`

Metrics include:
- Embedding quality scores
- Dimensionality reduction performance
- Clustering quality (silhouette scores)
- Trustworthiness metrics

## 📊 Key Dependencies

```
streamlit>=1.28.0           # Interactive web application
pandas>=2.0.0               # Data manipulation
numpy>=1.24.0               # Numerical computing
scikit-learn>=1.3.0         # Machine learning algorithms
umap-learn>=0.5.4           # Dimensionality reduction
plotly>=6.5.0               # Interactive visualizations
networkx                    # Graph analysis
hdbscan>=0.8.33             # Density-based clustering
leidenalg                   # Community detection
openai                      # LLM integration
transformers>=4.51.0        # Hugging Face models
torch                       # PyTorch for models
biopython                   # PubMed data extraction
```

See `requirements.txt` for complete list.

## 🎓 Methodology

### Gap Discovery Algorithm

The novelty discovery algorithm is based on identifying **low-density regions** in embedding space:

1. **Semantic Embeddings**: Papers represented as high-dimensional vectors capturing semantic meaning
2. **k-NN Graph**: Connect papers to their k most similar neighbors
3. **Local Density**: Papers with few nearby neighbors = potential gaps
4. **Bootstrap Validation**: Repeated sampling identifies stable low-density regions
5. **Gap Score**: Papers consistently in low-density areas across samples
6. **Region Identification**: Connected components of gap candidates form coherent unexplored areas

### Why This Works

- **Dense regions** = well-studied topics with many similar papers
- **Sparse regions** = understudied combinations of concepts
- **Stable gaps** = persistent research opportunities (not just noise)
- **Temporal analysis** = distinguish emerging from persistent gaps

## 🔧 Configuration

### Analysis Parameters

Key parameters in the novelty analysis:

- **n_neighbors** (k): Number of nearest neighbors (default: 20)
  - Lower k = more sensitive to local gaps
  - Higher k = more robust to noise
  
- **density_quantile**: Threshold for low-density (default: 0.10)
  - Lower = stricter gap definition
  - Higher = more gap candidates
  
- **n_bootstrap**: Bootstrap iterations (default: 100)
  - More iterations = more stable results
  - Computational cost increases linearly
  
- **gap_quantile**: Percentile for gap region identification (default: 0.90)
  - Only papers in top 90% of gap scores
  
- **min_gap_region_size**: Minimum papers per gap region (default: 3)
  - Filters out singleton gaps

### UMAP Parameters

- **n_neighbors**: Local neighborhood size (default: 15)
- **min_dist**: Minimum distance in low-D space (default: 0.1)
- **metric**: Distance metric (default: 'cosine')

## 📈 Use Cases

1. **Literature Review**: Identify unexplored research areas
2. **Grant Writing**: Find novel research directions with evidence
3. **Research Planning**: Discover interdisciplinary opportunities
4. **Technology Transfer**: Identify gaps between basic research and applications
5. **Trend Analysis**: Track evolution of research areas over time

## 🚧 Development Status

### Current Status
- ✅ Complete data pipeline (extraction, preprocessing, embeddings)
- ✅ Fully functional Streamlit application
- ✅ Core novelty discovery algorithm
- ✅ LLM integration for gap explanation
- ✅ Multiple embedding models (BERT, Qwen)
- 🚧 Modular app refactoring in progress (`novelty_app/`)

### Future Enhancements
- [ ] Complete modular app migration
- [ ] Additional embedding models
- [ ] Enhanced entity recognition
- [ ] Citation network analysis
- [ ] Collaborative filtering features
- [ ] API for programmatic access
- [ ] Docker containerization

## 📝 Citation

If you use this repository in your research, please cite:

```bibtex
@software{nanotech_gap_discovery,
  title = {Nanotechnology Research Gap Discovery System},
  author = {[Your Name]},
  year = {2026},
  url = {https://github.com/yourusername/nanotechnology}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional embedding models
- New clustering algorithms
- Improved gap detection methods
- UI/UX enhancements
- Documentation improvements

## 📄 License

This project is for academic and research purposes. Please check with your institution regarding data usage policies for PubMed data.

## 🙏 Acknowledgments

- **PubMed/NCBI**: For providing access to biomedical literature
- **Hugging Face**: For pre-trained transformer models
- **BioClinical-ModernBERT**: Biomedical text embeddings
- **Qwen3-Embedding**: Advanced embedding and reranking models
- **Streamlit**: For the interactive web application framework

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on the repository.

---

**Built with ❤️ for advancing nanotechnology research**
