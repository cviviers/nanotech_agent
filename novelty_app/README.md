# Refactored Novelty Analysis App

## 📁 Project Structure

```
novelty_app/
├── app.py                      # Main entry point
├── config.py                   # Configuration and constants
│
├── core/                       # Core business logic
│   ├── __init__.py
│   ├── state.py               # Session state management
│   ├── undo.py                # Undo functionality
│   ├── utils.py               # General utilities
│   ├── data_loader.py         # Data loading functions
│   ├── entities.py            # Entity extraction
│   ├── density.py             # Density computation
│   ├── clustering.py          # Clustering algorithms
│   └── gap_detection.py       # Gap identification
│
├── ui/                        # UI components (to be created)
│   ├── __init__.py
│   ├── visualizations.py      # Plotly charts
│   └── components.py          # Reusable UI elements
│
└── pages/                     # Page modules (to be migrated)
    ├── __init__.py
    ├── data_loading.py        # Data & Config page
    ├── embedding_processing.py # Embeddings page
    ├── filters.py             # Filters page
    ├── clustering.py          # Clustering page
    ├── gap_analysis.py        # Gap Analysis page
    ├── gap_regions.py         # Gap Regions page
    ├── llm_analysis.py        # LLM Analysis page
    ├── database_explorer.py   # Database Explorer page
    └── export.py              # Export page
```

## 🚀 Usage

### Running the Refactored App

```bash
streamlit run novelty_app/app.py
```

