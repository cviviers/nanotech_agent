# Nanotechnology Literature Mapping and Grounded Hypothesis Generation

This repository is research code for mapping nanotechnology and nanomedicine literature, identifying sparse or weakly explored regions in embedding space, and testing whether retrieval-grounded LLM workflows can generate useful research hypotheses from those frontiers.

It currently contains:

- a notebook pipeline for PubMed extraction, preprocessing, and embedding generation
- a Streamlit novelty-analysis workbench
- a FastAPI backend plus SQLite knowledge store for published analysis snapshots
- agentic and baseline hypothesis-generation workflows
- retrospective and prospective evaluation runners
- a separate human assessment app for blind-first review of generated ideas
- local embedding and reranking services for Qwen and BioClinical ModernBERT

## Main Components

```text
1.pubmed_data_extraction.ipynb   PubMed collection workflow
2.data_preprocessing.ipynb       Corpus cleaning / normalization
3.create_bert_embeddings.ipynb   BioClinical ModernBERT embedding creation
3.create_qwen_embeddings.ipynb   Qwen embedding creation

embedding_models/                FastAPI services for local embedding/reranking
novelty_app/                     Main analysis app, agent backend, and evaluation code
assement_app/                    Human review app for assessment bundles
tests/                           Automated tests
paper/                           Manuscript notes, figures, and writeup assets
data/                            Local datasets, embeddings, SQLite DB, evaluation outputs
```

Useful subsystem docs:

- `embedding_models/README.md`
- `novelty_app/agents/README.md`
- `novelty_app/evaluation/README.md`
- `assement_app/README.md`

## What The Code Actually Does

The current novelty-analysis path is centered on the `novelty_app` package and `novelty_app.evaluation.analysis_v1`.

At a high level it:

1. loads a paper corpus plus precomputed embeddings
2. optionally applies PCA for downstream analysis
3. clusters papers with K-means, HDBSCAN, or graph community detection
4. builds a k-nearest-neighbor graph in embedding space
5. computes density features as average k-NN distance across multiple `k` values
6. averages z-scored density features into a `gap_score`
7. identifies gap regions as connected components among papers above a chosen `gap_quantile`
8. publishes those results as reusable snapshots for agent and evaluation workflows

That means the current codebase is more about:

- frontier mapping over embedding space
- snapshot publishing and retrieval
- evidence-pack construction
- grounded ideation and blueprint generation
- benchmarking generated ideas against held-out future papers

It is less accurately described as the older monolithic `utils/` pipeline referenced by earlier README text.

## Setup

### Requirements

- Python 3.10+
- local corpus files and embeddings under `data/`
- optional GPU for local embedding services
- `OPENAI_API_KEY` for LLM-backed pages and generation methods

### Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Optional Environment Variables

- `OPENAI_API_KEY`: required for LLM analysis, orchestrator, and LLM-backed evaluation methods
- `OPENAI_MODEL`: optional override for generation/judging
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENABLED`: optional tracing
- `NOVELTY_AGENT_DB`: override the default SQLite path; otherwise the backend uses `data/novelty_agent_knowledge.sqlite`
- `QWEN_TORCH_DTYPE`, `QWEN_EMBED_TORCH_DTYPE`, `QWEN_RERANK_TORCH_DTYPE`: optional dtype controls for the local Qwen service

### Expected Local Data

The repo code assumes data artifacts like these are available locally:

- `data/cleaned_dataset.json`
- `data/all_papers.json`
- `data/bert_embeddings.npy`
- `data/bert_embeddings_metadata.json`
- `data/qwen_embeddings.npy`
- `data/qwen_embeddings_metadata.json`

Large runtime artifacts are intentionally not tracked in git. The current `.gitignore` excludes `data/*`, `paper/*`, generated review workbooks, and SQLite databases, so a fresh clone will not be self-contained.

## Run The Main Pieces

### 1. Streamlit Novelty Workbench

```powershell
streamlit run novelty_app/app.py
```

The app currently exposes pages for:

- Data & Config
- Embeddings
- Filters
- Clustering
- Gap Analysis
- Gap Regions
- LLM Analysis
- Agent Console
- Database Explorer
- Export

The Agent Console is the bridge into the backend and evaluation stack. It can publish analyzed snapshots, inspect backend state, run the orchestrator, and launch retrospective or prospective evaluations from the UI.

### 2. Local Embedding Services

From `embedding_models/`:

```powershell
cd embedding_models
uvicorn qwen:app --host 0.0.0.0 --port 8000
```

```powershell
cd embedding_models
uvicorn bert:app --host 0.0.0.0 --port 8001
```

Notes:

- the Qwen service provides both embeddings and reranking
- the evaluation stack uses the Qwen service as its retrieval backbone
- the BERT service is available for embedding experiments and simple ranking/similarity

### 3. Agent Backend

```powershell
uvicorn agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088
```

The backend stores and serves:

- published snapshots
- papers, clusters, and gap membership
- evidence packs
- generated artifacts
- retrospective evaluation runs and match records

By default the persistence layer is SQLite-backed at `data/novelty_agent_knowledge.sqlite`.

### 4. Interactive Agent Smoke Test

```powershell
python -m novelty_app.agents.run_interactive
```

This is a CLI path for selecting a snapshot and target manually, then running the LangGraph orchestrator outside Streamlit.

### 5. Retrospective Evaluation

The retrospective runner benchmarks whether historical frontier evidence can recover held-out future papers.

Prerequisites:

- backend running on `http://127.0.0.1:8088`
- Qwen service running on `http://127.0.0.1:8000`
- local corpus files under `data/`

Smoke example:

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --n-gap-targets 2 `
  --n-cluster-pair-targets 2 `
  --n-gold-future-papers 10 `
  --methods orchestrator single_shot_llm heuristic_bridge pack_query_baseline random_target_control `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --analysis-clustering-method kmeans `
  --analysis-pca-components 16 `
  --output-dir data/retrospective_eval_smoke
```

Outputs include:

- an evaluation run record
- raw hypothesis-level match records
- a review packet CSV/JSON
- an `assessment_bundle_v1` JSON export for manual review

### 6. Prospective Generation

The prospective runner reuses the same generator registry and snapshot-backed target selection, but does not do time splitting or future-paper recovery evaluation.

```powershell
python -m novelty_app.evaluation.run_prospective `
  --backend-url http://127.0.0.1:8088 `
  --snapshot-id <snapshot_id> `
  --n-gap-targets 2 `
  --n-cluster-pair-targets 2 `
  --methods orchestrator single_shot_llm heuristic_bridge pack_query_baseline `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --output-dir data/prospective_eval_smoke
```

Outputs include:

- `<run_id>_summary.json`
- `<run_id>_hypotheses.json`
- `<run_id>_hypotheses.csv`

### 7. Human Assessment App

```powershell
streamlit run assement_app/app.py
```

This app is for blind-first human review of generated ideas. It expects an `assessment_bundle_v1` JSON file produced by the retrospective runner and writes reviewer state to an Excel workbook.

Workbook sheets:

- `meta`
- `ideas`
- `assessments`
- `summary`

## Notebook Pipeline

The top-level notebooks still matter. They are the corpus-building side of the project:

- `1.pubmed_data_extraction.ipynb`: PubMed retrieval and raw paper collection
- `2.data_preprocessing.ipynb`: corpus cleanup and normalization
- `3.create_bert_embeddings.ipynb`: BioClinical ModernBERT embeddings
- `3.create_qwen_embeddings.ipynb`: Qwen embeddings

Those notebooks feed the local `data/` artifacts consumed by the app and evaluation code.

## Testing

Run the test suite with:

```powershell
pytest
```

The current tests cover core backend, evaluation, assessment, and agent-support utilities.

## Notes

- The directory name is `assement_app/` in the current repo. The spelling is preserved here because that is the actual import/run path.
- `paper/` contains manuscript and figure assets, not runtime application code.
- If you want subsystem-specific details, the most accurate docs are the package READMEs inside `embedding_models/`, `novelty_app/agents/`, `novelty_app/evaluation/`, and `assement_app/`.
