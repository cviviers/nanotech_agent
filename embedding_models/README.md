# Qwen3-0.6B Embedding & Reranker FastAPI Service

This repository provides a small FastAPI web service that exposes:

- **Qwen/Qwen3-Embedding-0.6B** for text embeddings
- **Qwen/Qwen3-Reranker-0.6B** for reranking (cross-encoder)

The implementation follows the official usage patterns from the Qwen3-Embedding project:

- Qwen3-Embedding GitHub: https://github.com/QwenLM/Qwen3-Embedding/blob/main/README.md

The service gives you simple HTTP endpoints to:

- Generate embedding vectors for arbitrary texts
- Compute cosine similarities between two sets of texts
- Rerank a list of documents for a given query, optionally also returning embedding-based similarity scores

## 1. Models

We use the 0.6B variants from the Qwen3 Embedding series:

- **Embedding model**: `Qwen/Qwen3-Embedding-0.6B`
  - Embedding dimension: **1024** :contentReference[oaicite:3]{index=3}
- **Reranker model**: `Qwen/Qwen3-Reranker-0.6B`

Qwen3-Embedding is specifically designed for text embedding and ranking tasks (retrieval, classification, clustering, bitext mining, etc.), and supports instruction-aware usage that can give 1–5% improvements in many tasks. :contentReference[oaicite:4]{index=4}  

For more details on training, benchmarks, and design, see the official README and technical report:

- Qwen3-Embedding GitHub: https://github.com/QwenLM/Qwen3-Embedding
- Technical report PDF in the same repo: `qwen3_embedding_technical_report.pdf` :contentReference[oaicite:5]{index=5}  

## 2. Prerequisites

- Python 3.10+ (recommended)
- A machine with:
  - CPU only: works, but slower
  - GPU (CUDA) strongly recommended for non-trivial workloads

You also need access to the models:

- Hugging Face model IDs:
  - `Qwen/Qwen3-Embedding-0.6B`
  - `Qwen/Qwen3-Reranker-0.6B` :contentReference[oaicite:6]{index=6}  

If these models are gated, ensure you have accepted their terms on Hugging Face and that you are authenticated (e.g. via `huggingface-cli login` or `HF_TOKEN` environment variable).

## 3. Installation

Clone your project and install dependencies:

```bash
git clone <this-repo-url> qwen3-embed-api
cd qwen3-embed-api

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

fastapi
uvicorn[standard]
torch
transformers>=4.51.0


## 4. Configuration

You can customize the models and max sequence length via environment variables:

QWEN_EMBEDDING_MODEL (default: Qwen/Qwen3-Embedding-0.6B)

QWEN_RERANKER_MODEL (default: Qwen/Qwen3-Reranker-0.6B)

QWEN_EMBED_MAX_LENGTH (default: 8192)

QWEN_RERANK_MAX_LENGTH (default: 8192)

Example:
```bash
export QWEN_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
export QWEN_RERANKER_MODEL="Qwen/Qwen3-Reranker-0.6B"
```

## 5. Running the server
From the project root:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You can then browse the interactive docs at:

* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

## 6. API Endpoints
### 6.1 Health / Info

GET /

Returns basic metadata:
```bash
{
  "service": "qwen3-embedding-reranker-api",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
  "device": "cuda",
  "embedding_dim": 1024
}
```