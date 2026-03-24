from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from agents.corpus_manifest import hash_paper_ids, stable_paper_id_from_row
    from agents.schemas import SnapshotMetadata, SnapshotPayload
except Exception:  # pragma: no cover
    from novelty_app.agents.corpus_manifest import hash_paper_ids, stable_paper_id_from_row
    from novelty_app.agents.schemas import SnapshotMetadata, SnapshotPayload


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    shape = getattr(value, "shape", None)
    if shape is not None and shape != ():
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    if hasattr(result, "shape") and getattr(result, "shape", None) not in (None, ()):
        return False
    try:
        return bool(result)
    except Exception:
        return False


def to_int(value: Any) -> Optional[int]:
    if is_null(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def to_float(value: Any) -> Optional[float]:
    if is_null(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def jsonable(value: Any) -> Any:
    if is_null(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def resolve_cluster_column(df: pd.DataFrame, selected_clustering: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if selected_clustering:
        candidates.append(f"cluster_{selected_clustering}")
    candidates.extend(["cluster_selected", "cluster_kmeans", "cluster_hdbscan", "cluster_leiden"])
    for col in candidates:
        if col in df.columns:
            return col
    return None


def paper_id_from_row(row: pd.Series, row_label: Any, row_pos: int) -> str:
    del row_label, row_pos
    return stable_paper_id_from_row(row)


def safe_row_lookup(df: pd.DataFrame, idx_like: Any) -> Tuple[int, Any, pd.Series]:
    if idx_like in df.index:
        pos = int(df.index.get_loc(idx_like))
        return pos, idx_like, df.loc[idx_like]
    pos = int(idx_like)
    row = df.iloc[pos]
    return pos, df.index[pos], row


def compute_data_hash(df: pd.DataFrame, max_rows: int = 5000) -> str:
    h = hashlib.sha256()
    limited = df.head(max_rows)
    cols = [c for c in ("id", "pmid", "paper_id", "doi", "title", "publication_year") if c in limited.columns]
    if not cols:
        cols = list(limited.columns[:5])
    for row in limited[cols].itertuples(index=False, name=None):
        h.update(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8"))
    h.update(str(len(df)).encode("utf-8"))
    return h.hexdigest()


def build_snapshot_payload(
    *,
    df: pd.DataFrame,
    gap_regions: Optional[Sequence[Sequence[Any]]] = None,
    llm_results: Optional[Dict[str, Any]] = None,
    selected_clustering: Optional[str] = None,
    x_primary: Any = None,
    x_umap_2d: Any = None,
    include_raw_rows: bool = True,
    include_embeddings: bool = True,
    snapshot_id: Optional[str] = None,
    source: str = "streamlit_agent_console",
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if df is None:
        raise ValueError("df is required")

    gap_regions = list(gap_regions or [])
    cluster_col = resolve_cluster_column(df, selected_clustering)
    has_primary_embeddings = include_embeddings and x_primary is not None
    embedding_dim: Optional[int] = None
    n_embeddings = 0

    row_to_gap_region: Dict[int, int] = {}
    for region_id, region in enumerate(gap_regions):
        for idx_like in region:
            try:
                pos, _, _ = safe_row_lookup(df, idx_like)
                row_to_gap_region[pos] = region_id
            except Exception:
                continue

    papers: List[Dict[str, Any]] = []
    paper_id_by_pos: Dict[int, str] = {}
    seen_ids: Dict[str, int] = {}

    for row_pos, (row_label, row) in enumerate(df.iterrows()):
        paper_id = paper_id_from_row(row, row_label, row_pos)
        if paper_id in seen_ids:
            seen_ids[paper_id] += 1
            paper_id = f"{paper_id}__{seen_ids[paper_id]}"
        else:
            seen_ids[paper_id] = 0

        paper_id_by_pos[row_pos] = paper_id

        publication_year = None
        for y_key in ("publication_year", "year"):
            if y_key in row.index:
                publication_year = to_int(row.get(y_key))
                if publication_year is not None:
                    break

        cluster_id = to_int(row.get(cluster_col)) if cluster_col else None
        gap_score = to_float(row.get("gap_score")) if "gap_score" in row.index else None
        gap_region = row_to_gap_region.get(row_pos)
        if gap_region is None and "gap_region" in row.index:
            gap_region = to_int(row.get("gap_region"))

        record = {
            "paper_id": paper_id,
            "row_index": row_pos,
            "title": None if "title" not in row.index else jsonable(row.get("title")),
            "abstract": (
                jsonable(row.get("abstract"))
                if "abstract" in row.index
                else jsonable(row.get("processed_content")) if "processed_content" in row.index else None
            ),
            "publication_year": publication_year,
            "doi": jsonable(row.get("doi")) if "doi" in row.index else None,
            "journal": jsonable(row.get("journal")) if "journal" in row.index else None,
            "cluster_id": cluster_id,
            "gap_score": gap_score,
            "gap_region": gap_region,
            "embedding": None,
            "umap_2d": None,
            "raw": {},
        }
        if has_primary_embeddings:
            try:
                emb = jsonable(x_primary[row_pos])
                if isinstance(emb, list) and emb:
                    record["embedding"] = emb
                    n_embeddings += 1
                    if embedding_dim is None:
                        embedding_dim = len(emb)
            except Exception:
                pass
        if x_umap_2d is not None:
            try:
                umap = jsonable(x_umap_2d[row_pos])
                if isinstance(umap, list) and len(umap) >= 2:
                    record["umap_2d"] = umap[:2]
            except Exception:
                pass
        if include_raw_rows:
            record["raw"] = {str(col): jsonable(row.get(col)) for col in df.columns}
        papers.append(record)

    clusters: List[Dict[str, Any]] = []
    if cluster_col and cluster_col in df.columns:
        value_counts = df[cluster_col].value_counts(dropna=True)
        for cid, size in value_counts.items():
            cid_int = to_int(cid)
            if cid_int is None:
                continue
            clusters.append(
                {
                    "cluster_id": cid_int,
                    "size": int(size),
                    "metadata": {
                        "cluster_column": cluster_col,
                        "selected_clustering": selected_clustering,
                    },
                }
            )
        clusters.sort(key=lambda x: (-x["size"], x["cluster_id"]))

    gaps: List[Dict[str, Any]] = []
    gap_papers: List[Dict[str, Any]] = []
    for region_id, region in enumerate(gap_regions):
        rows_for_region: List[Tuple[int, Any, pd.Series]] = []
        for idx_like in region:
            try:
                rows_for_region.append(safe_row_lookup(df, idx_like))
            except Exception:
                continue
        if not rows_for_region:
            continue

        scored: List[Tuple[int, str, Optional[float], Optional[int]]] = []
        cluster_ids: set[int] = set()
        gap_scores: List[float] = []
        for row_pos, _label, row in rows_for_region:
            pid = paper_id_by_pos.get(row_pos)
            if not pid:
                continue
            score = to_float(row.get("gap_score")) if "gap_score" in row.index else None
            cid = to_int(row.get(cluster_col)) if cluster_col and cluster_col in row.index else None
            if cid is not None:
                cluster_ids.add(cid)
            if score is not None:
                gap_scores.append(score)
            scored.append((row_pos, pid, score, cid))

        scored.sort(key=lambda t: (t[2] is None, -(t[2] or -1e9), t[0]))
        gap_id = f"gap_{region_id}"
        for rank_idx, (_pos, pid, score, _cid) in enumerate(scored):
            gap_papers.append({"gap_id": gap_id, "paper_id": pid, "rank": rank_idx, "gap_score": score})

        gaps.append(
            {
                "gap_id": gap_id,
                "region_index": region_id,
                "size": len(scored),
                "avg_gap_score": (sum(gap_scores) / len(gap_scores)) if gap_scores else None,
                "max_gap_score": max(gap_scores) if gap_scores else None,
                "density_z": None,
                "cluster_ids": sorted(cluster_ids),
                "metadata": {
                    "cluster_column": cluster_col,
                    "selected_clustering": selected_clustering,
                },
            }
        )

    llm_analyses: List[Dict[str, Any]] = []
    if llm_results:
        target: Dict[str, Any] = {"target_type": "llm_analysis"}
        for key in ("region_id", "cluster_A", "cluster_B", "cluster_C", "region_size"):
            if key in llm_results:
                target[key] = jsonable(llm_results.get(key))
        llm_analyses.append(
            {
                "analysis_id": str(uuid.uuid4()),
                "created_at": now_iso(),
                "target": target,
                "result": llm_results.get("result", {}),
                "metadata": {
                    "timestamp": llm_results.get("timestamp"),
                    "model": llm_results.get("model"),
                },
            }
        )

    metadata_dict = SnapshotMetadata(
        source=source,
        selected_clustering=selected_clustering,
        cluster_column=cluster_col,
        n_rows=len(df),
        has_gap_regions=bool(gap_regions),
        has_llm_results=bool(llm_results),
        has_embeddings=n_embeddings > 0,
        embedding_dim=embedding_dim,
        data_hash=compute_data_hash(df),
    ).model_dump()
    overrides = dict(metadata_overrides or {})
    extra = overrides.pop("extra", None)
    metadata_dict.update({k: jsonable(v) for k, v in overrides.items()})
    extra_payload = dict(metadata_dict.get("extra") or {})
    if extra is not None:
        extra_payload.update(jsonable(extra) or {})
    extra_payload.setdefault("snapshot_paper_id_hash", hash_paper_ids([paper["paper_id"] for paper in papers]))
    extra_payload.setdefault("snapshot_paper_count", len(papers))
    metadata_dict["extra"] = extra_payload

    payload = SnapshotPayload(
        snapshot_id=snapshot_id or f"snapshot_{uuid.uuid4().hex[:10]}",
        created_at=now_iso(),
        metadata=metadata_dict,
        papers=papers,
        clusters=clusters,
        gaps=gaps,
        gap_papers=gap_papers,
        llm_analyses=llm_analyses,
    ).model_dump()
    summary = {
        "n_papers": len(papers),
        "n_clusters": len(clusters),
        "n_gaps": len(gaps),
        "n_gap_papers": len(gap_papers),
        "n_llm_analyses": len(llm_analyses),
        "n_embeddings": n_embeddings,
        "embedding_dim": embedding_dim,
        "cluster_column": cluster_col,
    }
    return payload, summary
