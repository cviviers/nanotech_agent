# app/main.py
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

EMBED_MODEL_NAME = os.getenv("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL_NAME = os.getenv("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

EMBED_MAX_LENGTH = int(os.getenv("QWEN_EMBED_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.getenv("QWEN_RERANK_MAX_LENGTH", "8192"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Load models (Embedding + Reranker)
# Implementation follows:
# https://github.com/QwenLM/Qwen3-Embedding/blob/main/README.md
# -------------------------------------------------------------------

# Embedding model
embed_tokenizer = AutoTokenizer.from_pretrained(
    EMBED_MODEL_NAME,
    padding_side="left",
)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()

# Reranker model
rerank_tokenizer = AutoTokenizer.from_pretrained(
    RERANK_MODEL_NAME,
    padding_side="left",
)
rerank_model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL_NAME).to(DEVICE).eval()

# Tokens and prompts for reranking (adapted from official README)
token_false_id = rerank_tokenizer.convert_tokens_to_ids("no")
token_true_id = rerank_tokenizer.convert_tokens_to_ids("yes")

prefix = (
    "<|im_start|>system\n"
    " Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = rerank_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = rerank_tokenizer.encode(suffix, add_special_tokens=False)

# -------------------------------------------------------------------
# Helper functions (Embedding)
# -------------------------------------------------------------------

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pooling strategy from Qwen3-Embedding README:
    use the last token corresponding to the actual text (considering padding side).
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Instruction-aware input formatting, as recommended in Qwen3-Embedding README.
    """
    return f"Instruct: {task_description}\n Query:{query}"


def embed_texts(
    texts: List[str],
    instruction: Optional[str] = None,
    normalize: bool = True,
) -> Tensor:
    """
    Compute embeddings for a list of texts using Qwen3-Embedding-0.6B.
    If `instruction` is provided, apply instruction-aware formatting.
    """
    if len(texts) == 0:
        raise ValueError("texts must be a non-empty list")

    if instruction:
        processed_texts = [get_detailed_instruct(instruction, t) for t in texts]
    else:
        processed_texts = texts

    batch_dict = embed_tokenizer(
        processed_texts,
        padding=True,
        truncation=True,
        max_length=EMBED_MAX_LENGTH,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = embed_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    """
    Assuming a and b are already L2-normalized, cosine sim = dot product.
    a: [N, D], b: [M, D] -> [N, M]
    """
    return a @ b.T


# -------------------------------------------------------------------
# Helper functions (Reranker)
# -------------------------------------------------------------------

def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    output = (
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        .format(instruction=instruction, query=query, doc=doc)
    )
    return output


def process_rerank_inputs(pairs: List[str]):
    inputs = rerank_tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=RERANK_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = rerank_tokenizer.pad(
        inputs,
        padding=True,
        return_tensors="pt",
        max_length=RERANK_MAX_LENGTH,
    )
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)
    return inputs


@torch.no_grad()
def compute_rerank_scores(inputs) -> List[float]:
    """
    Compute reranking scores following the official Qwen3-Reranker usage:
    probability that answer is "yes".
    """
    batch_scores = rerank_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def rerank_query_documents(
    query: str,
    documents: List[str],
    instruction: Optional[str] = None,
) -> List[float]:
    if len(documents) == 0:
        raise ValueError("documents must be a non-empty list")
    pairs = [format_instruction(instruction, query, doc) for doc in documents]
    inputs = process_rerank_inputs(pairs)
    return compute_rerank_scores(inputs)


# -------------------------------------------------------------------
# Pydantic models (API schemas)
# -------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    texts: List[str]
    instruction: Optional[str] = None
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    normalize: bool


class SimilarityRequest(BaseModel):
    texts_a: List[str]
    texts_b: List[str]
    instruction_a: Optional[str] = None
    instruction_b: Optional[str] = None
    normalize: bool = True


class SimilarityResponse(BaseModel):
    similarity: List[List[float]]
    model: str
    normalized: bool


class RankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: Optional[str] = None
    top_k: Optional[int] = None
    return_embedding_similarity: bool = True
    normalize_embeddings: bool = True


class RankedDocument(BaseModel):
    index: int
    document: str
    reranker_score: float
    embedding_score: Optional[float] = None


class RankResponse(BaseModel):
    query: str
    instruction: Optional[str]
    model_reranker: str
    model_embedding: Optional[str] = None
    results: List[RankedDocument]


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="Qwen3-0.6B Embedding & Reranker API",
    description=(
        "FastAPI wrapper around Qwen/Qwen3-Embedding-0.6B and "
        "Qwen/Qwen3-Reranker-0.6B.\n"
        "Model usage is based on the official Qwen3-Embedding README."
    ),
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {
        "service": "qwen3-embedding-reranker-api",
        "embedding_model": EMBED_MODEL_NAME,
        "reranker_model": RERANK_MODEL_NAME,
        "device": DEVICE,
        "embedding_dim": 1024,  # per Qwen3-Embedding-0.6B spec
    }


@app.post("/embed", response_model=EmbeddingResponse)
def create_embeddings(payload: EmbeddingRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    try:
        embeddings = embed_texts(
            texts=payload.texts,
            instruction=payload.instruction,
            normalize=payload.normalize,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EmbeddingResponse(
        embeddings=embeddings.cpu().tolist(),
        dimension=embeddings.shape[1],
        model=EMBED_MODEL_NAME,
        normalize=payload.normalize,
    )


@app.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(payload: SimilarityRequest):
    if not payload.texts_a:
        raise HTTPException(status_code=400, detail="texts_a must be a non-empty list")
    if not payload.texts_b:
        raise HTTPException(status_code=400, detail="texts_b must be a non-empty list")

    try:
        emb_a = embed_texts(
            texts=payload.texts_a,
            instruction=payload.instruction_a,
            normalize=payload.normalize,
        )
        emb_b = embed_texts(
            texts=payload.texts_b,
            instruction=payload.instruction_b,
            normalize=payload.normalize,
        )
        sims = cosine_similarity_matrix(emb_a, emb_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SimilarityResponse(
        similarity=sims.cpu().tolist(),
        model=EMBED_MODEL_NAME,
        normalized=payload.normalize,
    )


@app.post("/rank", response_model=RankResponse)
def rank_documents(payload: RankRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents must be a non-empty list")

    # Reranker scores
    try:
        rerank_scores = rerank_query_documents(
            query=payload.query,
            documents=payload.documents,
            instruction=payload.instruction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reranker error: {e}")

    # Optional embedding-based similarity scores
    embedding_scores: Optional[List[float]] = None
    if payload.return_embedding_similarity:
        try:
            # One query vs many documents
            query_emb = embed_texts(
                texts=[payload.query],
                instruction=payload.instruction,
                normalize=payload.normalize_embeddings,
            )
            doc_embs = embed_texts(
                texts=payload.documents,
                instruction=None,  # typically documents are not instructed
                normalize=payload.normalize_embeddings,
            )
            sims = cosine_similarity_matrix(query_emb, doc_embs)[0]
            embedding_scores = sims.cpu().tolist()
        except Exception as e:
            # Do not fail the whole request if embedding similarity fails
            embedding_scores = None

    # Build result objects
    results = []
    for idx, (doc, r_score) in enumerate(zip(payload.documents, rerank_scores)):
        emb_score = None
        if embedding_scores is not None:
            emb_score = float(embedding_scores[idx])
        results.append(
            RankedDocument(
                index=idx,
                document=doc,
                reranker_score=float(r_score),
                embedding_score=emb_score,
            )
        )

    # Sort by reranker_score desc, then embedding_score desc (if available)
    results_sorted = sorted(
        results,
        key=lambda r: (r.reranker_score, r.embedding_score if r.embedding_score is not None else -1.0),
        reverse=True,
    )

    # Apply top_k trimming if requested
    if payload.top_k is not None:
        results_sorted = results_sorted[: payload.top_k]

    return RankResponse(
        query=payload.query,
        instruction=payload.instruction,
        model_reranker=RERANK_MODEL_NAME,
        model_embedding=EMBED_MODEL_NAME if payload.return_embedding_similarity else None,
        results=results_sorted,
    )

# example usage:

# start the server:
# uvicorn qwen:app --host 0.0.0.0 --port 8000