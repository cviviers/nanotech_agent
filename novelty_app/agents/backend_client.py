from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class BackendClient:
    """Thin HTTP client for the agent backend API."""

    def __init__(self, base_url: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        resp = requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=json_body,
            timeout=timeout_s or self.timeout_s,
        )
        if not resp.ok:
            detail = None
            try:
                body = resp.json()
                if isinstance(body, dict):
                    detail = body.get("detail") or body
                else:
                    detail = body
            except Exception:
                detail = resp.text.strip() or None
            msg = f"{method} {path} failed with HTTP {resp.status_code}"
            if detail:
                msg = f"{msg}: {detail}"
            raise RuntimeError(msg)
        if resp.content:
            return resp.json()
        return {}

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def list_snapshots(self, limit: int = 20) -> Dict[str, Any]:
        return self._request("GET", "/snapshots", params={"limit": limit})

    def publish_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/admin/snapshots/publish",
            json_body=payload,
            timeout_s=max(self.timeout_s, 300.0),
        )

    def list_clusters(
        self,
        snapshot_id: Optional[str] = None,
        limit: int = 100,
        sort: str = "size_desc",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit, "sort": sort}
        if snapshot_id:
            params["snapshot_id"] = snapshot_id
        return self._request("GET", "/clusters", params=params)

    def top_gaps(self, k: int = 25, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"k": k}
        if snapshot_id:
            params["snapshot_id"] = snapshot_id
        return self._request("GET", "/gaps/top", params=params)

    def papers_batch(
        self,
        snapshot_id: str,
        paper_ids: List[str],
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"snapshot_id": snapshot_id, "paper_ids": paper_ids}
        if fields:
            payload["fields"] = fields
        return self._request("POST", "/papers/batch", json_body=payload)

    def evidence_pack(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/evidence/pack", json_body=payload)

    def store_artifact(self, kind: str, target: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/artifacts/store",
            json_body={"kind": kind, "target": target, "payload": payload},
        )

    def list_artifacts(self, snapshot_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if snapshot_id:
            params["snapshot_id"] = snapshot_id
        return self._request("GET", "/artifacts", params=params)

    def store_evaluation_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/evaluations/runs", json_body=payload)

    def list_evaluation_runs(self, snapshot_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if snapshot_id:
            params["snapshot_id"] = snapshot_id
        return self._request("GET", "/evaluations/runs", params=params)

    def store_evaluation_matches_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", "/evaluations/matches/batch", json_body={"records": records})

    def list_evaluation_matches(
        self,
        *,
        run_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        limit: int = 250,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if run_id:
            params["run_id"] = run_id
        if snapshot_id:
            params["snapshot_id"] = snapshot_id
        return self._request("GET", "/evaluations/matches", params=params)
