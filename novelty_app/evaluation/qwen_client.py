from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class QwenClient:
    """HTTP client for the local Qwen embedding + reranker service."""

    def __init__(self, base_url: str = "http://0.0.0.0:8000", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout_s,
        )
        if not resp.ok:
            detail = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text.strip() or None
            raise RuntimeError(f"POST {path} failed with HTTP {resp.status_code}: {detail}")
        return resp.json()

    def embed(
        self,
        texts: List[str],
        *,
        instruction: Optional[str] = None,
        normalize: bool = True,
    ) -> List[List[float]]:
        data = self._post(
            "/embed",
            {"texts": texts, "instruction": instruction, "normalize": normalize},
        )
        return data.get("embeddings", [])

    def rank(
        self,
        *,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        return_embedding_similarity: bool = True,
        normalize_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        data = self._post(
            "/rank",
            {
                "query": query,
                "documents": documents,
                "instruction": instruction,
                "top_k": top_k,
                "return_embedding_similarity": return_embedding_similarity,
                "normalize_embeddings": normalize_embeddings,
            },
        )
        return data.get("results", [])
