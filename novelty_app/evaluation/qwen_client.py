from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import requests


def _print_local_qwen_error(message: str) -> None:
    print(f"[QwenClient] {message}", file=sys.stderr, flush=True)


class QwenClient:
    """HTTP client for the local Qwen embedding + reranker service."""

    def __init__(self, base_url: str = "http://0.0.0.0:8000", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            _print_local_qwen_error(f"POST {path} request error: {exc}")
            raise
        if not resp.ok:
            detail = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text.strip() or None
            message = f"POST {path} failed with HTTP {resp.status_code}: {detail}"
            _print_local_qwen_error(message)
            raise RuntimeError(message)
        try:
            return resp.json()
        except ValueError as exc:
            body = resp.text.strip()
            body_preview = body[:500] + ("..." if len(body) > 500 else "")
            message = f"POST {path} returned invalid JSON: {body_preview or '<empty response>'}"
            _print_local_qwen_error(message)
            raise RuntimeError(message) from exc

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
