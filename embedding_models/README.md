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

QWEN_EMBED_BATCH_SIZE (default: 16)

QWEN_RERANK_BATCH_SIZE (default: 2)

QWEN_RERANK_LOGITS_TO_KEEP (default: 1; keeps only last-token logits when supported)

QWEN_CUDA_EMPTY_CACHE_EACH_BATCH (default: 0; set to 1 if fragmentation keeps causing OOMs)

QWEN_TORCH_DTYPE (default: float16, fallback for both models on CUDA)

QWEN_EMBED_TORCH_DTYPE (optional override for the embedding model on CUDA)

QWEN_RERANK_TORCH_DTYPE (optional override for the reranker model on CUDA)

Example:
```bash
export QWEN_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
export QWEN_RERANKER_MODEL="Qwen/Qwen3-Reranker-0.6B"
export QWEN_RERANK_TORCH_DTYPE="float16"
export QWEN_RERANK_BATCH_SIZE="2"
export QWEN_EMBED_BATCH_SIZE="16"
export QWEN_RERANK_LOGITS_TO_KEEP="1"
```

For memory-constrained reranking, lower `QWEN_RERANK_BATCH_SIZE` first. If OOMs continue on very long inputs, lower `QWEN_RERANK_MAX_LENGTH` to `4096` or `2048`.

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

## 7. Evaluation Script

`embedding_models/eval.py` evaluates precomputed document embeddings against document-level keywords. It is designed for offline embedding quality checks rather than serving-time inference.

### What the script does

- Loads a JSON dataset of documents and a matching `.npy` embedding matrix.
- Cleans keyword fields and removes placeholder values such as `""`, `n/a`, and `none`.
- Runs a repeated-split multilabel linear probe with a label-prior baseline.
- Runs a retrieval benchmark where relevance is defined by keyword overlap.
- Optionally builds a TF-IDF lexical baseline from `title`, `abstract`, `cleaned_text`, or `full_text`.
- Writes a JSON report with dataset statistics, probe metrics, retrieval metrics, and the retained corpus-level keyword classes.

### Scientific guardrails

The current version of the script is intentionally stricter than a quick exploratory evaluation:

- Row-count mismatches between the dataframe and embeddings raise an error by default instead of silently truncating.
- The linear probe defines its label space from the training fold only, which avoids leaking held-out label frequencies into the supervised task definition.
- Threshold tuning uses a validation split inside the training fold rather than the final test fold.
- Retrieval metrics are averaged over all sampled queries. Queries with no relevant neighbours contribute zeros instead of being dropped from the averages.
- The linear probe now uses an adaptive backend: `auto` keeps exact one-vs-rest logistic regression for moderate problems and switches to an SGD-based logistic probe for large label spaces so the run stays practical.

If you have independently verified that row order still matches and want the old truncation behavior, pass `--allow_alignment_trim`.

### Inputs

The script expects:

- `--data_json`: a JSON file that loads into a list of records.
- `--embeddings_npy`: a 2D NumPy array with one embedding row per document.
- `--keyword_col`: the column containing keyword lists or keyword-like strings. The default is `mesh`.

Useful text columns for the lexical baseline are:

- `title`
- `abstract`
- `cleaned_text`
- `full_text`

### Metrics

Linear probe outputs include:

- `micro_f1`, `macro_f1`, `samples_f1`
- micro and macro precision/recall
- `hamming_loss`
- `lrap`
- `label_ranking_loss`
- per-split metadata such as selected threshold, class count, and how many documents had no supported labels inside that split

Retrieval outputs include:

- `mrr`
- `precision_at_k`
- `recall_at_k`
- `hit_rate_at_k`
- `map_at_k`
- `ndcg_at_k`
- `mean_shared_labels_at_k`
- query coverage metadata such as `n_queries_with_relevant_docs` and `query_coverage`

Retrieval relevance is defined as:

- Binary relevance: two documents share at least one retained keyword.
- Graded relevance: the number of retained keywords they share.

### Logging and progress

The script always prints stage-level log messages with an `[eval]` prefix so long runs are easier to monitor.

For richer progress bars, add:

```bash
--progress
```

For large datasets, the most relevant probe controls are:

- `--probe_backend auto|logistic|sgd`
- `--threshold_tuning auto|on|off`
- `--probe_n_jobs 1`

If the probe had previously appeared stuck, `--probe_backend sgd --threshold_tuning off --probe_n_jobs 1` is the safest fast configuration.

### Example

Run from the repository root:

```bash
python3 embedding_models/eval.py \
  --data_json ./data/cleaned_dataset.json \
  --embeddings_npy ./data/qwen_embeddings.npy \
  --keyword_col mesh \
  --min_keyword_freq 5 \
  --probe_backend auto \
  --threshold_tuning auto \
  --probe_n_jobs 1 \
  --n_repeats 3 \
  --test_size 0.2 \
  --base_seed 42 \
  --k_retrieval 10 \
  --max_retrieval_queries 5000 \
  --progress \
  --output_json ./qwen_evaluation_report.json
```

If your embeddings file and dataframe length differ and you have already verified row-order alignment:

```bash
python3 embedding_models/eval.py \
  --data_json ./data/cleaned_dataset.json \
  --embeddings_npy ./data/qwen_embeddings.npy \
  --probe_backend sgd \
  --threshold_tuning off \
  --probe_n_jobs 1 \
  --allow_alignment_trim
```

### Output structure

The output JSON contains four top-level sections:

- `data`: cleaned dataset statistics and filtering metadata
- `linear_probe`: repeated-split linear probe results and label-prior baseline
- `retrieval`: embedding retrieval metrics and optional TF-IDF baseline
- `keyword_classes`: the corpus-level keyword classes retained for retrieval evaluation
