# server.py
import os
import math
import json
from typing import List, Literal, Optional, Tuple

# ---- Env defaults (tweak as you like) ---------------------------------------
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))

import torch
import torch.nn.functional as F
from torch import nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
# ---------------------------- Config -----------------------------------------
# Choose at startup via env vars:
#   MODEL_CHOICE ∈ {"stella", "bioclinical-modernbert", "qwen3-embedding"}
#   MODEL_ID     override with an exact HF repo if you want
#   VECTOR_DIM   desired output dim (used for Stella's projection detection)
#   MAX_LENGTH   model max tokens per chunk
#   CHUNK_STRATEGY ∈ {"truncate", "mean_over_chunks"}
#   DTYPE ∈ {"auto","float16","bfloat16","float32"}
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "qwen3-embedding").lower().strip()
MODEL_ID_OVERRIDE = os.getenv("MODEL_ID", "").strip() or None
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))  # long context models can use big values
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "mean_over_chunks").lower().strip()
DTYPE_STR = os.getenv("DTYPE", "float16").lower().strip()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "54288"))

# Reasonable, conservative defaults for each preset
PRESETS = {
    "stella": {
        "repo": "dunzhang/stella_en_1.5B_v5",
        "pooling": "mean",
        "use_projection_if_available": True,
        "max_length": 8192,
    },
    "bioclinical-modernbert": {
        # Replace with the exact repo id you use for BioClinical ModernBERT (examples vary).
        # If you have a local/private repo, export MODEL_ID=your/repo
        "repo": "thomas-sounack/BioClinical-ModernBERT-base",
        "pooling": "mean",
        "use_projection_if_available": False,
        "max_length": 8192,
    },
    "qwen3-embedding": {
        # Replace with the exact embedding repo id you use, e.g.:
        # "repo": "Qwen/Qwen3-Embedding-0.6B"
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "pooling": "mean",
        "use_projection_if_available": False,
        "max_length": 8192,
    },
}

if MODEL_CHOICE not in PRESETS:
    raise RuntimeError(f"Unknown MODEL_CHOICE='{MODEL_CHOICE}'. Choose one of: {list(PRESETS)}")

PRESET = PRESETS[MODEL_CHOICE]
MODEL_ID = MODEL_ID_OVERRIDE or PRESET["repo"]
POOLING = PRESET["pooling"]
USE_PROJECTION = PRESET["use_projection_if_available"]
if MAX_LENGTH <= 0:
    MAX_LENGTH = PRESET["max_length"]


def choose_dtype() -> torch.dtype:
    if DTYPE_STR == "float16":
        return torch.float16
    if DTYPE_STR == "bfloat16":
        return torch.bfloat16
    if DTYPE_STR == "float32":
        return torch.float32
    # auto
    if torch.cuda.is_available():
        # BF16 is usually safer than FP16; fall back to FP16 if not available
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


DTYPE = choose_dtype()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Utilities -----------------------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)                   # [B, H]
    lengths = mask.sum(dim=1).clamp(min=1e-6)                  # [B, 1]
    return summed / lengths

def _cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    # Use first token as [CLS]
    return last_hidden[:, 0, :]

def detect_stella_projection_path(hf_home: str, repo_id: str, vector_dim: int) -> Optional[str]:
    """
    Try to find the Stella projection weights like:
    {HF_HOME}/modules/transformers_modules/{repo_id}/2_Dense_{vector_dim}/pytorch_model.bin
    """
    # Sanitize repo path for local cache
    repo_sanitized = repo_id.replace("/", os.sep)
    candidates = [
        os.path.join(hf_home, "modules", "transformers_modules", repo_sanitized, f"2_Dense_{vector_dim}", "pytorch_model.bin"),
        os.path.join(hf_home, "modules", "transformers_modules", repo_sanitized, f"2_Dense_{vector_dim}", "model.safetensors"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'



# ------------------------- Backend -------------------------------------------
class EncoderBackend(nn.Module):
    def __init__(self, model_id: str, pooling: Literal["mean","cls"]="mean",
                 use_projection_if_available: bool=False, vector_dim: int=1024):
        super().__init__()
        self.model_id = model_id
        self.pooling = pooling
        self.vector_dim = vector_dim
        self.is_qwen = "qwen" in MODEL_CHOICE.lower()

        # For Qwen models, use left padding
        if self.is_qwen:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(DEVICE)
        self.model.eval()

        self.hidden_size = getattr(self.model.config, "hidden_size", None)
        if self.hidden_size is None:
            # Fallback for some custom models
            self.hidden_size = getattr(self.model.config, "d_model", None)
        if self.hidden_size is None:
            raise RuntimeError("Could not determine model hidden size from config.")

        # Optional projection (Stella style)
        self.projector: nn.Module = nn.Identity()

        if use_projection_if_available:
            proj_path = detect_stella_projection_path(os.environ["HF_HOME"], model_id, vector_dim)
            if proj_path and os.path.basename(proj_path).endswith(".bin"):
                # pytorch bin
                linear = nn.Linear(in_features=self.hidden_size, out_features=vector_dim, bias=True)
                state = torch.load(proj_path, map_location="cpu")
                # Some dumps use a "linear." prefix:
                cleaned = {k.replace("linear.", ""): v for k, v in state.items()}
                linear.load_state_dict(cleaned)
                self.projector = linear.to(DEVICE)
                self.output_dim = vector_dim
            elif proj_path and os.path.basename(proj_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                linear = nn.Linear(in_features=self.hidden_size, out_features=vector_dim, bias=True)
                cleaned = {k.replace("linear.", ""): v for k, v in load_file(proj_path).items()}
                linear.load_state_dict(cleaned)
                self.projector = linear.to(DEVICE)
                self.output_dim = vector_dim
            else:
                # No projector found; fall back to hidden size
                self.output_dim = self.hidden_size
        else:
            self.output_dim = self.hidden_size

        # sensible tokenizer max length fallback
        self.model_max_length = min(
            getattr(self.tokenizer, "model_max_length", MAX_LENGTH) or MAX_LENGTH,
            MAX_LENGTH
        )

        # task-specific prompts (use only when asked)
        self.task_prompts = {
            "s2s": "Instruct: Retrieve semantically similar text.\nQuery:",
            "s2p": "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:",
            "default": "",
        }
        

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: List[str],
        task: Literal["default", "s2s", "s2p"]="default",
        normalize: bool=True,
        chunk_strategy: Literal["truncate","mean_over_chunks"]="mean_over_chunks",
        max_length: Optional[int]=None,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Returns (embeddings, token_counts_per_text).
        """
        if max_length is None:
            max_length = self.model_max_length

        # Apply task prompts if requested
        if task in self.task_prompts:
            if self.is_qwen and task != "default":
                # Qwen uses get_detailed_instruct format
                task_description = self.task_prompts[task]
                texts = [get_detailed_instruct(task_description, t) for t in texts]
            elif task in self.task_prompts:
                prefix = self.task_prompts[task]
                texts = [prefix + t for t in texts]

        all_vecs: List[torch.Tensor] = []
        all_token_counts: List[int] = []

        # Mixed precision for speed on GPU
        autocast_enabled = (DEVICE.type == "cuda" and DTYPE in (torch.float16, torch.bfloat16))

        for text in texts:
            # Tokenize once to know length
            base_inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
            num_tokens = base_inputs["input_ids"].shape[1]
            all_token_counts.append(int(num_tokens))

            if num_tokens <= max_length or chunk_strategy == "truncate":
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length,
                    padding="longest"
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.autocast(device_type="cuda" if autocast_enabled else "cpu", dtype=DTYPE) if autocast_enabled else torch.no_grad():
                    outputs = self.model(**inputs)
                    last_hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state

                    if self.is_qwen:
                        # Qwen uses last token pooling
                        vec = last_token_pool(last_hidden, inputs["attention_mask"])
                    elif self.pooling == "mean":
                        vec = _mean_pool(last_hidden, inputs["attention_mask"])
                    else:
                        vec = _cls_pool(last_hidden)

                    vec = self.projector(vec)
                    if normalize:
                        if self.is_qwen:
                            vec = F.normalize(vec, p=2, dim=1)
                        else:
                            vec = l2_normalize(vec)
                all_vecs.append(vec.squeeze(0).to("cpu"))
            else:
                # mean over chunks
                # Sliding window chunks respecting token boundaries
                input_ids = base_inputs["input_ids"].squeeze(0)
                attn_mask = base_inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)

                stride = max_length  # non-overlapping chunks by default
                chunk_vecs = []
                with torch.autocast(device_type="cuda" if autocast_enabled else "cpu", dtype=DTYPE) if autocast_enabled else torch.no_grad():
                    for start in range(0, num_tokens, stride):
                        end = min(start + max_length, num_tokens)
                        ids = input_ids[start:end].unsqueeze(0).to(DEVICE)
                        mask = attn_mask[start:end].unsqueeze(0).to(DEVICE)
                        chunk_outputs = self.model(input_ids=ids, attention_mask=mask)
                        chunk_last = chunk_outputs[0] if isinstance(chunk_outputs, (tuple, list)) else chunk_outputs.last_hidden_state
                        if self.is_qwen:
                            chunk_vec = last_token_pool(chunk_last, mask)
                        elif self.pooling == "mean":
                            chunk_vec = _mean_pool(chunk_last, mask)
                        else:
                            chunk_vec = _cls_pool(chunk_last)
                        chunk_vec = self.projector(chunk_vec)  # [1, D]
                        chunk_vecs.append(chunk_vec)

                    chunk_stack = torch.cat(chunk_vecs, dim=0)  # [C, D]
                    vec = chunk_stack.mean(dim=0, keepdim=True)  # [1, D]
                    if normalize:
                        if self.is_qwen:
                            vec = F.normalize(vec, p=2, dim=1)
                        else:
                            vec = l2_normalize(vec)
                all_vecs.append(vec.squeeze(0).to("cpu"))

        # Return as plain lists for JSON
        emb_lists = [v.tolist() for v in all_vecs]
        return emb_lists, all_token_counts


# --------------------------- FastAPI -----------------------------------------
app = FastAPI(title="Clinical Embedding Service", version="2.0.0")

# Instantiate backend once at startup
BACKEND = EncoderBackend(
    model_id=MODEL_ID,
    pooling=POOLING,
    use_projection_if_available=USE_PROJECTION,
    vector_dim=VECTOR_DIM
).to(DEVICE)

# --------------------------- Schemas -----------------------------------------
class TextInput(BaseModel):
    text: str = Field(..., description="Single text to embed")
    task: Literal["default","s2s","s2p"] = "default"
    normalize: bool = True
    chunk_strategy: Literal["truncate","mean_over_chunks"] = CHUNK_STRATEGY
    max_length: Optional[int] = None

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="Batch of texts to embed")
    task: Literal["default","s2s","s2p"] = "default"
    normalize: bool = True
    chunk_strategy: Literal["truncate","mean_over_chunks"] = CHUNK_STRATEGY
    max_length: Optional[int] = None

class SimilarityInput(BaseModel):
    embedding_docs: List[List[float]]
    embedding_query: List[float]
    metric: Literal["cosine", "dot"] = "cosine"

# --------------------------- Routes ------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "dtype": str(DTYPE).replace("torch.", ""),
        "model_choice": MODEL_CHOICE,
        "model_id": MODEL_ID,
        "output_dim": BACKEND.output_dim,
        "pooling": BACKEND.pooling,
        "max_length": BACKEND.model_max_length,
        "projection": BACKEND.projector.__class__.__name__,
        "version": "2.0.0",
    }

@app.get("/model")
def model_info():
    cfg = getattr(BACKEND.model, "config", None)
    return {
        "hidden_size": BACKEND.hidden_size,
        "output_dim": BACKEND.output_dim,
        "model_max_length": BACKEND.model_max_length,
        "config": json.loads(cfg.to_json_string()) if cfg is not None else None,
    }

@app.post("/embed")
def embed_single(inp: TextInput):
    try:
        vecs, token_counts = BACKEND.encode_texts(
            [inp.text],
            task=inp.task,
            normalize=inp.normalize,
            chunk_strategy=inp.chunk_strategy,
            max_length=inp.max_length,
        )
        return {"embedding": vecs[0], "num_tokens": token_counts[0]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_batch")
def embed_batch(inp: BatchTextInput):
    if not inp.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    try:
        vecs, token_counts = BACKEND.encode_texts(
            inp.texts,
            task=inp.task,
            normalize=inp.normalize,
            chunk_strategy=inp.chunk_strategy,
            max_length=inp.max_length,
        )
        return {"embeddings": vecs, "num_tokens": token_counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Backwards-compatible convenience endpoints
@app.post("/embed_queries_s2s")
def embed_s2s(inp: TextInput):
    return embed_single(TextInput(text=inp.text, task="s2s", normalize=inp.normalize,
                                  chunk_strategy=inp.chunk_strategy, max_length=inp.max_length))

@app.post("/embed_queries_s2p")
def embed_s2p(inp: TextInput):
    return embed_single(TextInput(text=inp.text, task="s2p", normalize=inp.normalize,
                                  chunk_strategy=inp.chunk_strategy, max_length=inp.max_length))

@app.post("/compute_similarity")
def compute_similarity(inp: SimilarityInput):
    try:
        docs = torch.tensor(inp.embedding_docs, dtype=torch.float32, device=DEVICE)  # [N, D]
        q = torch.tensor(inp.embedding_query, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, D]

        if inp.metric == "cosine":
            docs = l2_normalize(docs)
            q = l2_normalize(q)

        sim = q @ docs.T  # [1, N]
        return {"similarity": sim.squeeze(0).to("cpu").tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------- Entrypoint ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    print(
        f"Starting server with MODEL_CHOICE='{MODEL_CHOICE}' "
        f"MODEL_ID='{MODEL_ID}' on {HOST}:{PORT} | device={DEVICE} dtype={DTYPE}"
    )
    uvicorn.run(app, host=HOST, port=PORT)
