from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from agents.knowledge_store import KnowledgeStore
    from agents.schemas import DiscoveryCue, EvaluationMatch, EvaluationRun, SnapshotPayload
except Exception:  # pragma: no cover
    from novelty_app.agents.knowledge_store import KnowledgeStore
    from novelty_app.agents.schemas import DiscoveryCue, EvaluationMatch, EvaluationRun, SnapshotPayload


def _build_store() -> KnowledgeStore:
    db_path = os.getenv("NOVELTY_AGENT_DB", os.path.join("data", "novelty_agent_knowledge.sqlite"))
    return KnowledgeStore(db_path)


STORE = _build_store()

app = FastAPI(
    title="Novelty Agent Backend",
    version="0.1.0",
    description="Agent-facing query API for novelty analysis snapshots and artifacts.",
)


class SnapshotPublishRequest(SnapshotPayload):
    pass


class PaperBatchRequest(BaseModel):
    snapshot_id: Optional[str] = None
    paper_ids: List[str] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)


class EvidencePackRequest(BaseModel):
    snapshot_id: Optional[str] = None
    target_type: str
    gap_id: Optional[str] = None
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    exemplars: int = 25
    boundary: int = 25
    diverse: int = 25
    counter_queries: List[str] = Field(default_factory=list)
    discovery_cue: Optional[DiscoveryCue] = None


class ArtifactStoreRequest(BaseModel):
    snapshot_id: Optional[str] = None
    kind: str
    target: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMatchesBatchRequest(BaseModel):
    records: List[EvaluationMatch] = Field(default_factory=list)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "novelty-agent-backend",
        "endpoints": [
            "GET /health",
            "GET /snapshots",
            "POST /admin/snapshots/publish",
            "GET /gaps/top",
            "GET /clusters",
            "POST /papers/batch",
            "POST /evidence/pack",
            "POST /artifacts/store",
            "GET /artifacts",
            "POST /evaluations/runs",
            "GET /evaluations/runs",
            "POST /evaluations/matches/batch",
            "GET /evaluations/matches",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "db_path": str(STORE.db_path)}


@app.get("/snapshots")
def list_snapshots(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    return {"snapshots": STORE.list_snapshots(limit=limit)}


@app.post("/admin/snapshots/publish")
def publish_snapshot(req: SnapshotPublishRequest) -> Dict[str, Any]:
    try:
        result = STORE.publish_snapshot(req.model_dump())
        return {"ok": True, **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/gaps/top")
def top_gaps(k: int = Query(25, ge=1, le=500), snapshot_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        resolved = STORE.resolve_snapshot_id(snapshot_id)
        return {"snapshot_id": resolved, "gaps": STORE.top_gaps(snapshot_id=resolved, k=k)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/clusters")
def list_clusters(
    snapshot_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=2000),
    sort: str = Query("size_desc"),
) -> Dict[str, Any]:
    try:
        resolved = STORE.resolve_snapshot_id(snapshot_id)
        return {
            "snapshot_id": resolved,
            "clusters": STORE.list_clusters(snapshot_id=resolved, limit=limit, sort=sort),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/papers/batch")
def papers_batch(req: PaperBatchRequest) -> Dict[str, Any]:
    try:
        resolved = STORE.resolve_snapshot_id(req.snapshot_id)
        papers = STORE.papers_batch(snapshot_id=resolved, paper_ids=req.paper_ids, fields=req.fields or None)
        return {"snapshot_id": resolved, "papers": papers}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/evidence/pack")
def evidence_pack(req: EvidencePackRequest) -> Dict[str, Any]:
    try:
        return STORE.build_evidence_pack(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/artifacts/store")
def artifacts_store(req: ArtifactStoreRequest) -> Dict[str, Any]:
    try:
        out = STORE.store_artifact(
            kind=req.kind,
            target=req.target,
            payload=req.payload,
            snapshot_id=req.snapshot_id,
        )
        return {"ok": True, **out}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/artifacts")
def artifacts_list(snapshot_id: Optional[str] = None, limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    try:
        return {"artifacts": STORE.list_artifacts(snapshot_id=snapshot_id, limit=limit)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/evaluations/runs")
def evaluations_runs_store(req: EvaluationRun) -> Dict[str, Any]:
    try:
        out = STORE.store_evaluation_run(req.model_dump())
        return {"ok": True, **out}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/evaluations/runs")
def evaluations_runs_list(snapshot_id: Optional[str] = None, limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    try:
        return {"runs": STORE.list_evaluation_runs(snapshot_id=snapshot_id, limit=limit)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/evaluations/matches/batch")
def evaluations_matches_store(req: EvaluationMatchesBatchRequest) -> Dict[str, Any]:
    try:
        out = STORE.store_evaluation_matches_batch([r.model_dump() for r in req.records])
        return {"ok": True, **out}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/evaluations/matches")
def evaluations_matches_list(
    run_id: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    limit: int = Query(250, ge=1, le=2000),
) -> Dict[str, Any]:
    try:
        return {
            "matches": STORE.list_evaluation_matches(run_id=run_id, snapshot_id=snapshot_id, limit=limit)
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# uvicorn agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088
