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

QWEN_TORCH_DTYPE (default: float16, fallback for both models on CUDA)

QWEN_EMBED_TORCH_DTYPE (optional override for the embedding model on CUDA)

QWEN_RERANK_TORCH_DTYPE (optional override for the reranker model on CUDA)

Example:
```bash
export QWEN_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
export QWEN_RERANKER_MODEL="Qwen/Qwen3-Reranker-0.6B"
export QWEN_RERANK_TORCH_DTYPE="float16"
```

## 5. Running the server
From the project root:
```bash
uvicorn qwen:app  --port 8000 --reload
```

You can then browse the interactive docs at:

* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

## 6. API Endpoints
### 6.1 Health / Info

GET /

Returns basic metadata:
```json
{
  "service": "qwen3-embedding-reranker-api",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
  "device": "cuda",
  "embed_torch_dtype": "torch.float16",
  "rerank_torch_dtype": "torch.float16",
  "embedding_dim": 1024
}
```

### 6.1 Embeddings

POST /embed

Compute embeddings for a list of texts, optionally with an instruction (instruction-aware usage, as recommended by Qwen)
```json
{
  "texts": [
    "What is the capital of China?",
    "Explain gravity"
  ],
  "instruction": "Given a web search query, retrieve relevant passages that answer the query",
  "normalize": true
}
```
* texts (List[str]): Texts to embed.

* instruction (str, optional): A one-sentence instruction describing the task.

      * If provided, the service will wrap each text as:

* Instruct: <instruction>\n Query:<text>

* mirroring the official Qwen3-Embedding pattern.

* normalize (bool, default true): If true, embeddings are L2-normalized.

Response
```json
{
  "embeddings": [
    [0.0123, -0.0456, ...],
    [-0.0222, 0.0011, ...]
  ],
  "dimension": 1024,
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "normalize": true
}
```
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "What is the capital of China?",
      "Explain gravity"
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "normalize": true
  }'
  ```
Example Python client
```python
import requests

payload = {
    "texts": [
        "What is the capital of China?",
        "Explain gravity",
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "normalize": True,
}

resp = requests.post("http://localhost:8000/embed", json=payload)
resp.raise_for_status()
data = resp.json()

print("Embedding dimension:", data["dimension"])
print("First vector length:", len(data["embeddings"][0]))
```

### 6.2 Reranking

POST /rank

Use Qwen3-Reranker-0.6B to score each document for a given query, following the official reranker usage. You can optionally also get embedding-based similarity scores from the embedding model for comparison.

Request body

```json
{
  "query": "What is the capital of China?",
  "documents": [
    "The capital of China is Beijing.",
    "The capital of France is Paris."
  ],
  "instruction": "Given a web search query, retrieve relevant passages that answer the query",
  "top_k": 2,
  "return_embedding_similarity": true,
  "normalize_embeddings": true
}

```

* query: User query string.

* documents: List of candidate documents to rank.

* instruction (optional): Task description; if omitted, a default web-search-style instruction is used.

* top_k (optional): If set, only the top-k highest-scoring documents are returned.

* return_embedding_similarity: If true, the service also returns cosine similarities between the query embedding and each document embedding.

* normalize_embeddings: Whether to normalize embeddings before similarity.

```python
import requests

payload = {
    "query": "What is the capital of China?",
    "documents": [
        "The capital of China is Beijing.",
        "The capital of France is Paris."
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "top_k": 2,
    "return_embedding_similarity": True,
    "normalize_embeddings": True
}

resp = requests.post("http://localhost:8000/rank", json=payload)
resp.raise_for_status()
data = resp.json()

for r in data["results"]:
    print(f"Doc: {r['document']}")
    print(f"  Reranker score:  {r['reranker_score']:.4f}")
    if r.get("embedding_score") is not None:
        print(f"  Embedding score: {r['embedding_score']:.4f}")

```

### 6.1 Similarity (Embedding-based)
