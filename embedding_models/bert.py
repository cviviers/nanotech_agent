# app/main.py
import os
import sys
import traceback
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = os.getenv(
    "BIOCLINICAL_MODEL_ID",
    "thomas-sounack/BioClinical-ModernBERT-large",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0 if torch.cuda.is_available() else -1  # for HF pipeline


# ---------------------------------------------------------
# Load tokenizer + model
# As per model card: use transformers >= 4.48.0 and
# AutoModelForMaskedLM (MLM / fill-mask). :contentReference[oaicite:1]{index=1}
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Masked LM head (for fill-mask) – this wraps the ModernBERT encoder
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
mlm_model.to(DEVICE).eval()

# Use the MLM model directly for embeddings
EMBED_DIM = mlm_model.config.hidden_size

# Max sequence length (long-context encoder, default 8192). :contentReference[oaicite:2]{index=2}
DEFAULT_MAX_LENGTH = int(
    os.getenv("BIOCLINICAL_MAX_LENGTH", str(getattr(mlm_model.config, "max_position_embeddings", 8192)))
)

# Optional: fill-mask pipeline using the same model
fill_mask_pipeline = pipeline(
    "fill-mask",
    model=mlm_model,
    tokenizer=tokenizer,
    device=DEVICE_INDEX,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Standard mean pooling over the sequence, masking out padding tokens.
    This provides a simple sentence/document embedding for downstream use.
    """
    # last_hidden_states: [batch, seq, hidden]
    # attention_mask:    [batch, seq]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_states)  # [batch, seq, 1]
    masked_states = last_hidden_states * mask
    summed = masked_states.sum(dim=1)  # [batch, hidden]
    counts = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
    return summed / counts


def embed_texts(
    texts: List[str],
    max_length: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create embeddings by passing texts through the encoder and mean-pooling.
    BioClinical ModernBERT is primarily an encoder for MLM and downstream tasks;
    this is a generic way to obtain embeddings for similarity/retrieval.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list")

    max_len = max_length or DEFAULT_MAX_LENGTH
    
    # print(f"[DEBUG] embed_texts called with {len(texts)} texts, max_length={max_len}, normalize={normalize}", file=sys.stderr, flush=True)

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    
    # print(f"[DEBUG] Tokenized batch: input_ids shape={batch['input_ids'].shape}", file=sys.stderr, flush=True)

    with torch.no_grad():
        outputs = mlm_model(**batch, output_hidden_states=True)
        
        # print(f"[DEBUG] Model output shape: {outputs.hidden_states[-1].shape}", file=sys.stderr, flush=True)
        
        # Debug: Check model outputs
        last_hidden_state = outputs.hidden_states[-1]
        if torch.isnan(last_hidden_state).any():
            error_msg = f"Model output contains NaN values! Text lengths: {[len(t) for t in texts]}"
            # print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if torch.isinf(last_hidden_state).any():
            error_msg = f"Model output contains Inf values! Text lengths: {[len(t) for t in texts]}"
            # print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
        # print("[DEBUG] Model output validation passed", file=sys.stderr, flush=True)
        
        embeddings = mean_pool(last_hidden_state, batch["attention_mask"])
        
        # print(f"[DEBUG] After pooling shape: {embeddings.shape}", file=sys.stderr, flush=True)
        
        # Debug: Check after pooling
        if torch.isnan(embeddings).any():
            error_msg = f"Embeddings contain NaN after pooling! Text lengths: {[len(t) for t in texts]}"
            # print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if torch.isinf(embeddings).any():
            error_msg = f"Embeddings contain Inf after pooling! Text lengths: {[len(t) for t in texts]}"
            # print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
        # print("[DEBUG] Pooling validation passed", file=sys.stderr, flush=True)
        
        if normalize:
            # print("[DEBUG] Normalizing embeddings...", file=sys.stderr, flush=True)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Debug: Check after normalization
            if torch.isnan(embeddings).any():
                error_msg = f"Embeddings contain NaN after normalization! Text lengths: {[len(t) for t in texts]}"
                # print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
                raise ValueError(error_msg)
            if torch.isinf(embeddings).any():
                error_msg = f"Embeddings contain Inf after normalization! Text lengths: {[len(t) for t in texts]}"
                # print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
                raise ValueError(error_msg)
            
            # print("[DEBUG] Normalization validation passed", file=sys.stderr, flush=True)
    
    # print(f"[DEBUG] Returning embeddings with shape: {embeddings.shape}", file=sys.stderr, flush=True)
    return embeddings


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Assuming a and b are (optionally) normalized, compute cosine similarity as dot product.
    a: [N, D], b: [M, D] -> [N, M]
    """
    return a @ b.T


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------

class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True
    max_length: Optional[int] = None


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    normalize: bool
    max_length: int


class SimilarityRequest(BaseModel):
    texts_a: List[str]
    texts_b: List[str]
    normalize: bool = True
    max_length: Optional[int] = None


class SimilarityResponse(BaseModel):
    similarity: List[List[float]]
    model: str
    normalized: bool
    max_length: int


class FillMaskRequest(BaseModel):
    text: str
    top_k: int = 5


class FillMaskPrediction(BaseModel):
    token: int
    token_str: str
    score: float
    sequence: str


class FillMaskResponse(BaseModel):
    predictions: List[FillMaskPrediction]
    model: str


class RankRequest(BaseModel):
    query: str
    documents: List[str]
    normalize: bool = True
    max_length: Optional[int] = None
    top_k: Optional[int] = None


class RankedDocument(BaseModel):
    index: int
    document: str
    score: float


class RankResponse(BaseModel):
    query: str
    model: str
    normalize: bool
    max_length: int
    results: List[RankedDocument]


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="BioClinical-ModernBERT-large API",
    description=(
        "FastAPI wrapper around thomas-sounack/BioClinical-ModernBERT-large.\n\n"
        "Capabilities:\n"
        "- Masked Language Modeling (fill-mask)\n"
        "- Generic encoder embeddings via mean pooling\n"
        "- Cosine similarity & simple embedding-based ranking\n\n"
        "Model setup and capabilities follow the official Hugging Face model card "
        "and the BioClinical ModernBERT research repository."
    ),
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "service": "bioclinical-modernbert-api",
        "model": MODEL_ID,
        "device": DEVICE,
        "embedding_dim": EMBED_DIM,
        "default_max_length": DEFAULT_MAX_LENGTH,
        "note": (
            "BioClinical ModernBERT is a domain-adapted, long-context encoder for "
            "biomedical and clinical NLP. For downstream tasks (classification, "
            "retrieval, QA), fine-tune following standard BERT recipes."
        ),
    }


# ---------------- Embeddings ---------------- #

@app.post("/embed", response_model=EmbedResponse)
def create_embeddings(payload: EmbedRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    # print(f"[/embed] Received request for {len(payload.texts)} texts, normalize={payload.normalize}", file=sys.stderr, flush=True)
    
    try:
        embeddings = embed_texts(
            texts=payload.texts,
            max_length=payload.max_length,
            normalize=payload.normalize,
        )
        
        # print(f"[/embed] Got embeddings with shape: {embeddings.shape}", file=sys.stderr, flush=True)
        
        # Final validation before JSON serialization
        embeddings_cpu = embeddings.cpu()
        if torch.isnan(embeddings_cpu).any():
            nan_indices = torch.where(torch.isnan(embeddings_cpu))[0].tolist()
            error_msg = f"Final embeddings contain NaN at indices: {nan_indices[:10]}"
            # print(f"[/embed ERROR] {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if torch.isinf(embeddings_cpu).any():
            inf_indices = torch.where(torch.isinf(embeddings_cpu))[0].tolist()
            error_msg = f"Final embeddings contain Inf at indices: {inf_indices[:10]}"
            # print(f"[/embed ERROR] {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
        # print(f"[/embed] Final validation passed, converting to list", file=sys.stderr, flush=True)
        embeddings_list = embeddings_cpu.tolist()
        # print(f"[/embed] Successfully converted to list with {len(embeddings_list)} embeddings", file=sys.stderr, flush=True)
        
    except ValueError as e:
        # Provide detailed error message for debugging
        # print(f"[/embed EXCEPTION] ValueError: {str(e)}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Embedding validation error: {str(e)}")
    except Exception as e:
        # print(f"[/embed EXCEPTION] Unexpected error: {str(e)}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    max_len = payload.max_length or DEFAULT_MAX_LENGTH

    return EmbedResponse(
        embeddings=embeddings_list,
        dimension=embeddings.shape[1],
        model=MODEL_ID,
        normalize=payload.normalize,
        max_length=max_len,
    )


# -------------- Similarity (embeddings) --------------- #

@app.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(payload: SimilarityRequest):
    if not payload.texts_a:
        raise HTTPException(status_code=400, detail="texts_a must be a non-empty list")
    if not payload.texts_b:
        raise HTTPException(status_code=400, detail="texts_b must be a non-empty list")

    try:
        emb_a = embed_texts(
            texts=payload.texts_a,
            max_length=payload.max_length,
            normalize=payload.normalize,
        )
        emb_b = embed_texts(
            texts=payload.texts_b,
            max_length=payload.max_length,
            normalize=payload.normalize,
        )
        sims = cosine_similarity_matrix(emb_a, emb_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    max_len = payload.max_length or DEFAULT_MAX_LENGTH

    return SimilarityResponse(
        similarity=sims.cpu().tolist(),
        model=MODEL_ID,
        normalized=payload.normalize,
        max_length=max_len,
    )


# ----------------- Fill-mask (MLM) ----------------- #

@app.post("/fill-mask", response_model=FillMaskResponse)
def fill_mask(payload: FillMaskRequest):
    if "[MASK]" not in payload.text:
        raise HTTPException(
            status_code=400,
            detail="Input text must contain at least one [MASK] token.",
        )

    if payload.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")

    try:
        results = fill_mask_pipeline(payload.text, top_k=payload.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # HF pipeline returns dict or list[dict]; we normalize to list
    if isinstance(results, dict):
        results = [results]

    predictions = [
        FillMaskPrediction(
            token=int(r["token"]),
            token_str=r["token_str"],
            score=float(r["score"]),
            sequence=r["sequence"],
        )
        for r in results
    ]

    return FillMaskResponse(
        predictions=predictions,
        model=MODEL_ID,
    )


# -------------- Simple embedding-based ranking -------------- #

@app.post("/rank", response_model=RankResponse)
def rank_documents(payload: RankRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents must be a non-empty list")

    max_len = payload.max_length or DEFAULT_MAX_LENGTH

    try:
        # Query embedding
        q_emb = embed_texts(
            texts=[payload.query],
            max_length=max_len,
            normalize=payload.normalize,
        )[0]  # [hidden]

        # Document embeddings
        doc_embs = embed_texts(
            texts=payload.documents,
            max_length=max_len,
            normalize=payload.normalize,
        )  # [num_docs, hidden]

        # Cosine similarity between query and each document
        scores = (doc_embs @ q_emb.unsqueeze(-1)).squeeze(-1)  # [num_docs]
        scores_list = scores.cpu().tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    ranked = [
        RankedDocument(index=i, document=doc, score=float(s))
        for i, (doc, s) in enumerate(zip(payload.documents, scores_list))
    ]
    ranked_sorted = sorted(ranked, key=lambda x: x.score, reverse=True)

    if payload.top_k is not None:
        ranked_sorted = ranked_sorted[: payload.top_k]

    return RankResponse(
        query=payload.query,
        model=MODEL_ID,
        normalize=payload.normalize,
        max_length=max_len,
        results=ranked_sorted,
    )

# example usage:

# start the server:
# uvicorn bert:app --host 0.0.0.0 --port 8001