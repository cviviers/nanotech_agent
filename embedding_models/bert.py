# app/main.py
import os
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

# Use the base encoder inside the MLM model for embeddings
encoder = mlm_model.base_model
EMBED_DIM = encoder.config.hidden_size

# Max sequence length (long-context encoder, default 8192). :contentReference[oaicite:2]{index=2}
DEFAULT_MAX_LENGTH = int(
    os.getenv("BIOCLINICAL_MAX_LENGTH", str(getattr(encoder.config, "max_position_embeddings", 8192)))
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

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        outputs = encoder(**batch)
        embeddings = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

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

    try:
        embeddings = embed_texts(
            texts=payload.texts,
            max_length=payload.max_length,
            normalize=payload.normalize,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    max_len = payload.max_length or DEFAULT_MAX_LENGTH

    return EmbedResponse(
        embeddings=embeddings.cpu().tolist(),
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
