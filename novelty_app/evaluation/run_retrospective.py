from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import pandas as pd

try:
    from agents.backend_client import BackendClient
    from agents.corpus_manifest import (
        hash_paper_ids,
        reconstruct_positions_from_manifest,
        stable_paper_id_from_row,
        stable_paper_ids,
        subset_embeddings_by_positions,
    )
    from agents.schemas import AnalysisConfig, EvaluationMatch, EvaluationRun, TraceRef
    from agents.snapshot_builder import build_snapshot_payload
except Exception:  # pragma: no cover
    from novelty_app.agents.backend_client import BackendClient
    from novelty_app.agents.corpus_manifest import (
        hash_paper_ids,
        reconstruct_positions_from_manifest,
        stable_paper_id_from_row,
        stable_paper_ids,
        subset_embeddings_by_positions,
    )
    from novelty_app.agents.schemas import AnalysisConfig, EvaluationMatch, EvaluationRun, TraceRef
    from novelty_app.agents.snapshot_builder import build_snapshot_payload

from novelty_app.discovery_cue import (
    discovery_cue_to_dict,
    normalize_discovery_cue,
    score_fingerprint_against_cue,
    score_record_against_cue,
)
from novelty_app.agents.observability import (
    create_trace_score,
    current_trace_ref,
    deterministic_trace_id,
    flush_langfuse,
    observe_current,
    trace_attributes,
)

from .analysis_v1 import run_analysis_v1
from .candidate_match import (
    best_candidate,
    best_non_excluded_candidate,
    build_corpus_index,
    candidate_rank,
    retrieve_candidates_for_hypothesis,
)
from .generators import GenerationContext, run_generation_method, target_id
from .idea_fingerprint import fingerprint_hypothesis, fingerprint_text
from .judge import classify_recovery_match, normalize_cue_score, score_hypotheses
from .metrics import aggregate_match_metrics, select_best_task_rows
from .qwen_client import QwenClient
from .time_split import load_dataset_and_embeddings, split_corpus_by_time


DEFAULT_METHODS = [
    "orchestrator",
    "single_shot_llm",
    "retrieval_summary_direct",
    "heuristic_bridge",
    "pack_query_baseline",
    "random_target_control",
]


@dataclass
class RetrospectiveResult:
    run: Dict[str, Any]
    matches: List[Dict[str, Any]]
    review_packet_csv: str
    review_packet_json: str


@dataclass
class GoldFutureAssignment:
    paper_id: str
    title: str
    abstract: str
    publication_year: Optional[int]
    paper_fingerprint: Dict[str, Any]
    best_target_score: float
    cue_alignment_score: Optional[float]
    ranking_score: float
    assigned_target: Dict[str, Any]
    assigned_target_id: str
    assigned_target_score: float


@dataclass
class RetrospectiveProgress:
    phase: str
    status: str = "running"
    current: Optional[int] = None
    total: Optional[int] = None
    message: str = ""
    run_id: str = ""
    method_name: Optional[str] = None
    seed: Optional[int] = None
    gold_future_paper_id: Optional[str] = None
    n_matches: int = 0
    n_failures: int = 0

    def to_payload(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _coerce_trace_ref(trace_ref: Any) -> TraceRef:
    if isinstance(trace_ref, TraceRef):
        return trace_ref
    if isinstance(trace_ref, dict):
        return TraceRef.model_validate(trace_ref)
    return TraceRef()


def _trace_ref_payload(trace_ref: Any) -> Dict[str, Any]:
    return _coerce_trace_ref(trace_ref).model_dump(exclude_none=True)


def _normalize_string_list(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_future_prefilter(
    *,
    future_title_exclude: Optional[Sequence[str]],
    future_abstract_exclude: Optional[Sequence[str]],
    future_semantic_query: Optional[str],
    future_semantic_threshold: Optional[float],
) -> Dict[str, Any]:
    title_terms = _normalize_string_list(future_title_exclude)
    abstract_terms = _normalize_string_list(future_abstract_exclude)
    semantic_query = str(future_semantic_query or "").strip() or None
    semantic_threshold = float(future_semantic_threshold) if semantic_query else None
    if semantic_query and semantic_threshold is None:
        semantic_threshold = 0.30
    return {
        "active": bool(title_terms or abstract_terms or semantic_query),
        "title_exclusion_keywords": title_terms,
        "abstract_exclusion_keywords": abstract_terms,
        "semantic_query": semantic_query,
        "semantic_threshold": semantic_threshold,
    }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _text_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field in df.columns:
        return df[field].fillna("").astype(str)
    if field == "abstract":
        for fallback in ("cleaned_text", "processed_content", "content"):
            if fallback in df.columns:
                return df[fallback].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _apply_future_prefilter(
    *,
    future_df: pd.DataFrame,
    future_embeddings: Dict[str, np.ndarray],
    qwen_client: QwenClient,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    before_count = int(len(future_df))
    if not before_count:
        stats = {
            **config,
            "n_future_rows_before": before_count,
            "n_future_rows_after": before_count,
        }
        return future_df, future_embeddings, stats

    mask = pd.Series(True, index=future_df.index, dtype=bool)
    title_series = _text_series(future_df, "title").str.lower()
    abstract_series = _text_series(future_df, "abstract").str.lower()

    for keyword in config.get("title_exclusion_keywords") or []:
        mask &= ~title_series.str.contains(str(keyword).lower(), na=False, regex=False)
    for keyword in config.get("abstract_exclusion_keywords") or []:
        mask &= ~abstract_series.str.contains(str(keyword).lower(), na=False, regex=False)

    semantic_query = str(config.get("semantic_query") or "").strip()
    if semantic_query:
        if "qwen" not in future_embeddings:
            raise ValueError("Future semantic prefilter requires `qwen` embeddings in the future split.")
        query_embedding = np.asarray(
            qwen_client.embed(
                [semantic_query],
                instruction="Given a web search query, retrieve relevant passages that answer the query",
                normalize=True,
            )[0],
            dtype=np.float32,
        )
        future_qwen = np.asarray(future_embeddings["qwen"], dtype=np.float32)
        similarities = _normalize_rows(future_qwen) @ query_embedding
        mask &= similarities >= float(config["semantic_threshold"])

    mask_values = mask.to_numpy(dtype=bool, copy=False)
    filtered_df = future_df.loc[mask].reset_index(drop=True).copy()
    filtered_embeddings = {name: arr[mask_values].copy() for name, arr in future_embeddings.items()}
    stats = {
        **config,
        "n_future_rows_before": before_count,
        "n_future_rows_after": int(len(filtered_df)),
    }
    return filtered_df, filtered_embeddings, stats


class _RetrospectiveCliProgressReporter:
    _ITERATIVE_PHASES = {"selecting_gold_future_papers", "evaluating_tasks", "persisting_matches"}

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream or sys.stdout
        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._last_phase: Optional[str] = None
        self._live_line = False
        self._line_width = 0

    def _render_line(self, progress: RetrospectiveProgress) -> str:
        parts = [f"[{progress.phase}]"]
        if progress.message:
            parts.append(progress.message)
        if progress.current is not None and progress.total is not None:
            parts.append(f"({progress.current}/{progress.total})")
        if progress.phase in {"evaluating_tasks", "persisting_matches", "completed", "failed"}:
            parts.append(f"matches={progress.n_matches}")
        if progress.n_failures or progress.phase in {"evaluating_tasks", "completed", "failed"}:
            parts.append(f"failures={progress.n_failures}")
        return " ".join(parts)

    def _end_live_line(self) -> None:
        if not self._live_line:
            return
        self._stream.write("\n")
        self._stream.flush()
        self._live_line = False

    def __call__(self, progress: RetrospectiveProgress) -> None:
        line = self._render_line(progress)
        iterative = (
            self._isatty
            and progress.status == "running"
            and progress.phase in self._ITERATIVE_PHASES
            and progress.current is not None
            and progress.total is not None
        )
        if not self._isatty:
            self._stream.write(f"{line}\n")
            self._stream.flush()
            self._last_phase = progress.phase
            return

        if iterative:
            if progress.phase != self._last_phase:
                self._end_live_line()
                milestone = f"[{progress.phase}] {progress.message or progress.status}"
                self._stream.write(f"{milestone}\n")
                self._stream.flush()
                self._last_phase = progress.phase
            self._line_width = max(self._line_width, len(line))
            self._stream.write("\r" + line.ljust(self._line_width))
            self._stream.flush()
            self._live_line = True
            return

        self._end_live_line()
        self._stream.write(f"{line}\n")
        self._stream.flush()
        self._last_phase = progress.phase

    def close(self) -> None:
        self._end_live_line()


def _gold_selection_progress_message(stats: Dict[str, int]) -> str:
    parts = [
        f"leakage={int(stats.get('n_leakage_filtered', 0))}",
        f"eligible={int(stats.get('n_frontier_eligible', 0))}",
    ]
    if "n_cue_scored" in stats or "n_cue_positive" in stats:
        parts.append(f"cue_scored={int(stats.get('n_cue_scored', 0))}")
        parts.append(f"cue_positive={int(stats.get('n_cue_positive', 0))}")
    return "Scanning future papers " + "(" + ", ".join(parts) + ")"


def _hash_config(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _metadata_date(snapshot: Dict[str, Any], field: str) -> str:
    metadata = dict(snapshot.get("metadata") or {})
    value = metadata.get(field)
    if not value:
        raise ValueError(f"Snapshot metadata is missing `{field}`.")
    return str(value)


def _resolve_retrospective_bundle_artifact(
    backend: BackendClient,
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = dict(snapshot.get("metadata") or {})
    extra = dict(metadata.get("extra") or {})
    artifact_id = extra.get("retrospective_bundle_artifact_id")
    if artifact_id:
        artifact = backend.get_artifact(str(artifact_id))
        if artifact.get("kind") != "retrospective_snapshot_bundle":
            raise ValueError(
                f"Artifact `{artifact_id}` linked from snapshot `{snapshot['snapshot_id']}` is not a retrospective bundle."
            )
        return artifact

    bundle_prefix = extra.get("bundle_prefix")
    artifacts = backend.list_artifacts(
        snapshot_id=str(snapshot["snapshot_id"]),
        kind="retrospective_snapshot_bundle",
        limit=20,
    ).get("artifacts", [])
    if bundle_prefix:
        matches = [
            artifact
            for artifact in artifacts
            if str((artifact.get("target") or {}).get("bundle_prefix") or "") == str(bundle_prefix)
        ]
        if matches:
            return matches[0]
    if artifacts:
        return artifacts[0]
    raise ValueError(f"No retrospective bundle artifact was found for snapshot `{snapshot['snapshot_id']}`.")


def _reconstruct_frontend_corpus(
    df: pd.DataFrame,
    embeddings: Dict[str, Any],
    manifest: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    retained_paper_ids = list(manifest.get("retained_paper_ids") or [])
    if not retained_paper_ids:
        raise ValueError("Retrospective corpus manifest is missing `retained_paper_ids`.")

    positions = reconstruct_positions_from_manifest(df, retained_paper_ids)
    reconstructed_df = df.iloc[positions].reset_index(drop=True).copy()
    reconstructed_embeddings = subset_embeddings_by_positions(embeddings, positions)
    reconstructed_paper_ids = stable_paper_ids(reconstructed_df)
    reconstructed_hash = hash_paper_ids(reconstructed_paper_ids)

    expected_hash = manifest.get("retained_paper_id_hash")
    if expected_hash and str(expected_hash) != reconstructed_hash:
        raise ValueError("Retrospective corpus manifest hash does not match the reconstructed frontend corpus.")
    expected_count = manifest.get("row_count")
    if expected_count is not None and int(expected_count) != len(reconstructed_df):
        raise ValueError("Retrospective corpus manifest row count does not match the reconstructed frontend corpus.")
    return reconstructed_df, reconstructed_embeddings


def _validate_reconstructed_historical_split(
    snapshot: Dict[str, Any],
    manifest: Dict[str, Any],
    historical_df: pd.DataFrame,
) -> None:
    metadata = dict(snapshot.get("metadata") or {})
    extra = dict(metadata.get("extra") or {})

    historical_paper_ids = stable_paper_ids(historical_df)
    historical_hash = hash_paper_ids(historical_paper_ids)
    historical_count = len(historical_paper_ids)

    source_hash = extra.get("source_corpus_paper_id_hash")
    if source_hash and str(source_hash) != str(manifest.get("retained_paper_id_hash")):
        raise ValueError("Historical snapshot metadata points to a different source corpus than the bundle manifest.")
    source_count = extra.get("source_corpus_row_count")
    if source_count is not None and int(source_count) != int(manifest.get("row_count") or 0):
        raise ValueError("Historical snapshot metadata source corpus row count does not match the bundle manifest.")

    expected_snapshot_hash = extra.get("snapshot_paper_id_hash") or extra.get("historical_paper_id_hash")
    if expected_snapshot_hash and str(expected_snapshot_hash) != historical_hash:
        raise ValueError("Reconstructed historical split does not match the published historical snapshot paper ids.")
    expected_snapshot_count = extra.get("snapshot_paper_count") or extra.get("historical_paper_count")
    if expected_snapshot_count is not None and int(expected_snapshot_count) != historical_count:
        raise ValueError("Reconstructed historical split does not match the published historical snapshot paper count.")

def _chunked(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _select_cluster_pair_targets(
    backend: BackendClient,
    snapshot_id: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    pairs: List[Dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    top_gap_resp = backend.top_gaps(snapshot_id=snapshot_id, k=max(50, limit * 4))
    for gap in top_gap_resp.get("gaps", []):
        cluster_ids = sorted(
            {
                int(c)
                for c in (gap.get("cluster_ids") or [])
                if c is not None and str(c) not in {"", "-1"}
            }
        )
        for cluster_a, cluster_b in combinations(cluster_ids, 2):
            pair = (cluster_a, cluster_b)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(
                {
                    "target_type": "cluster_pair",
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "source_gap_id": gap.get("gap_id"),
                }
            )
            if len(pairs) >= limit:
                return pairs

    clusters = backend.list_clusters(snapshot_id=snapshot_id, limit=max(20, limit * 3)).get("clusters", [])
    ordered = [int(c["cluster_id"]) for c in clusters]
    for cluster_a, cluster_b in combinations(ordered, 2):
        pair = tuple(sorted((cluster_a, cluster_b)))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append({"target_type": "cluster_pair", "cluster_a": pair[0], "cluster_b": pair[1]})
        if len(pairs) >= limit:
            break
    return pairs


def _target_request(snapshot_id: str, target: Dict[str, Any], *, discovery_cue: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "target_type": str(target.get("target_type") or ""),
        "profile": "focused_eval",
        "exemplars": 8,
        "boundary": 8,
        "diverse": 0,
    }
    if target.get("target_type") == "gap":
        payload["gap_id"] = target.get("gap_id")
    else:
        payload["cluster_a"] = target.get("cluster_a")
        payload["cluster_b"] = target.get("cluster_b")
    if discovery_cue:
        payload["discovery_cue"] = discovery_cue
    return payload


def _target_score_from_pack(pack: Dict[str, Any], discovery_cue: Optional[Dict[str, Any]]) -> float:
    if not discovery_cue:
        return 0.0
    papers = list(pack.get("papers") or [])
    if not papers:
        return 0.0
    scores = [
        float(score_record_against_cue(paper, discovery_cue).get("score", 0.0) or 0.0)
        for paper in papers[:6]
    ]
    return (sum(scores) / len(scores)) if scores else 0.0


def _paper_id_from_row(row: pd.Series, row_idx: int) -> str:
    del row_idx
    return stable_paper_id_from_row(row)


def _paper_record_from_row(row: pd.Series, row_idx: int) -> Dict[str, Any]:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", row.get("cleaned_text", "")) or "").strip()
    return {
        "paper_id": _paper_id_from_row(row, row_idx),
        "title": title,
        "abstract": abstract,
        "publication_year": int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
    }


def _paper_to_pseudo_cue(paper: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
    fingerprint = fingerprint_text(text)
    soft_constraints = {
        field: list(fingerprint.get(field) or [])
        for field in ("disease", "material", "payload", "targeting", "mechanism", "model", "route", "outcome")
        if fingerprint.get(field)
    }
    return {"text": text, "soft_constraints": soft_constraints}


def _summarize_evidence_pack(pack: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pack = dict(pack or {})
    papers = list(pack.get("papers") or [])
    compact_papers = []
    for paper in papers[:5]:
        compact_papers.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title", ""),
                "year": paper.get("publication_year", paper.get("year")),
                "selection_sources": list(paper.get("selection_sources") or []),
            }
        )
    return {
        "stats": dict(pack.get("stats") or {}),
        "meta": dict(pack.get("meta") or {}),
        "top_papers": compact_papers,
        "n_papers": len(papers),
    }


def _compact_candidates(candidates: Sequence[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for candidate in list(candidates)[:limit]:
        judge = dict(candidate.get("judge") or {})
        compact.append(
            {
                "paper_id": candidate.get("paper_id"),
                "title": candidate.get("title", ""),
                "publication_year": candidate.get("publication_year"),
                "judge_label": candidate.get("judge_label") or judge.get("label") or "no_match",
                "combined_score": candidate.get("combined_score", judge.get("combined_score")),
                "reranker_score": candidate.get("reranker_score"),
                "embedding_score": candidate.get("embedding_score"),
            }
        )
    return compact


def _generation_task_metadata(
    *,
    run_id: str,
    method_name: str,
    seed: int,
    gold: GoldFutureAssignment,
    snapshot_id: str,
) -> Dict[str, Any]:
    assigned_target = dict(gold.assigned_target or {})
    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "method_name": method_name,
        "seed": seed,
        "gold_future_paper_id": gold.paper_id,
        "assigned_target_id": gold.assigned_target_id,
        "target_type": assigned_target.get("target_type"),
    }
    if assigned_target.get("target_type") == "gap":
        metadata["gap_id"] = assigned_target.get("gap_id")
    else:
        metadata["cluster_a"] = assigned_target.get("cluster_a")
        metadata["cluster_b"] = assigned_target.get("cluster_b")
    return metadata


def _generation_task_tags(method_name: str, seed: int, gold: GoldFutureAssignment) -> List[str]:
    tags = ["retrospective_eval", "generation_task", method_name, f"seed_{seed}"]
    target_type = str((gold.assigned_target or {}).get("target_type") or "")
    if target_type:
        tags.append(target_type)
    return tags


def _prepare_target_pool(
    backend: BackendClient,
    snapshot_id: str,
    targets: Sequence[Dict[str, Any]],
    discovery_cue: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target_pool: List[Dict[str, Any]] = []
    for target in targets:
        pack = backend.evidence_pack(_target_request(snapshot_id, target, discovery_cue=discovery_cue))
        target_pool.append(
            {
                "target": dict(target),
                "target_id": target_id(target),
                "pack": pack,
                "cue_target_score": _target_score_from_pack(pack, discovery_cue),
            }
        )
    return target_pool


def _select_gold_future_papers(
    *,
    future_df: pd.DataFrame,
    historical_index: Any,
    qwen_client: QwenClient,
    target_pool: Sequence[Dict[str, Any]],
    discovery_cue: Optional[Dict[str, Any]],
    n_gold_future_papers: int,
    progress_callback: Optional[Callable[[int, int, Dict[str, int]], None]] = None,
) -> tuple[List[GoldFutureAssignment], Dict[str, int]]:
    selected: List[GoldFutureAssignment] = []
    leakage_filtered = 0
    frontier_eligible = 0
    cue_scored = 0
    cue_positive = 0
    total_rows = int(len(future_df))

    for row_number, (row_idx, row) in enumerate(future_df.iterrows(), start=1):
        try:
            paper = _paper_record_from_row(row, row_idx)
            if not paper["title"] or not paper["abstract"]:
                continue

            query_text = f"{paper['title']}\n{paper['abstract']}".strip()
            fingerprint = fingerprint_text(query_text)
            historical_candidates = retrieve_candidates_for_hypothesis(
                query_text=query_text,
                fingerprint=fingerprint,
                corpus=historical_index,
                qwen_client=qwen_client,
                top_k_keyword=40,
                top_k_semantic=80,
                top_k_final=10,
                rerank_max_docs=24,
            )
            historical_best = best_candidate(historical_candidates)
            if (historical_best.get("judge") or {}).get("label") == "strong_match":
                leakage_filtered += 1
                continue

            paper_cue = _paper_to_pseudo_cue(paper)
            best_target_score = float("-inf")
            best_assignment_score = float("-inf")
            best_target: Optional[Dict[str, Any]] = None
            for target_info in target_pool:
                paper_target_score = _target_score_from_pack(target_info["pack"], paper_cue)
                if paper_target_score > best_target_score:
                    best_target_score = paper_target_score
                assignment_score = paper_target_score
                if discovery_cue:
                    assignment_score = 0.7 * paper_target_score + 0.3 * float(target_info["cue_target_score"] or 0.0)
                if best_target is None or assignment_score > best_assignment_score or (
                    assignment_score == best_assignment_score
                    and str(target_info["target_id"]) < str(best_target["target_id"])
                ):
                    best_assignment_score = assignment_score
                    best_target = target_info

            if best_target_score <= 0 or best_target is None:
                continue
            frontier_eligible += 1

            cue_alignment_score: Optional[float] = None
            ranking_score = best_target_score
            if discovery_cue:
                cue_alignment_score = float(score_record_against_cue(paper, discovery_cue).get("score", 0.0) or 0.0)
                cue_scored += 1
                if cue_alignment_score > 0:
                    cue_positive += 1
                # Discovery cues steer ranking; they do not hard-filter future papers.
                ranking_score = 0.7 * best_target_score + 0.3 * max(0.0, cue_alignment_score)

            selected.append(
                GoldFutureAssignment(
                    paper_id=str(paper["paper_id"]),
                    title=str(paper["title"]),
                    abstract=str(paper["abstract"]),
                    publication_year=paper.get("publication_year"),
                    paper_fingerprint=fingerprint,
                    best_target_score=float(best_target_score),
                    cue_alignment_score=cue_alignment_score,
                    ranking_score=float(ranking_score),
                    assigned_target=dict(best_target["target"]),
                    assigned_target_id=str(best_target["target_id"]),
                    assigned_target_score=float(best_assignment_score),
                )
            )
        finally:
            if progress_callback and (row_number % 10 == 0 or row_number == total_rows):
                progress_callback(
                    row_number,
                    total_rows,
                    {
                        "n_leakage_filtered": leakage_filtered,
                        "n_frontier_eligible": frontier_eligible,
                        "n_cue_filtered": 0,
                        "n_cue_scored": cue_scored,
                        "n_cue_positive": cue_positive,
                    },
                )

    selected.sort(
        key=lambda item: (
            item.ranking_score,
            item.assigned_target_score,
            item.best_target_score,
            item.publication_year or -1,
            item.paper_id,
        ),
        reverse=True,
    )
    return selected[:n_gold_future_papers], {
        "n_future_pool": int(len(future_df)),
        "n_leakage_filtered": leakage_filtered,
        "n_frontier_eligible": frontier_eligible,
        "n_cue_filtered": 0,
        "n_cue_scored": cue_scored,
        "n_cue_positive": cue_positive,
    }


def _review_rows(matches: Sequence[Dict[str, Any]], *, cue_active: bool) -> List[Dict[str, Any]]:
    best_rows = select_best_task_rows(matches)

    def order_key(row: Dict[str, Any]) -> Tuple[float, float, str, int]:
        primary = row.get("cue_weighted_rr") if cue_active else row.get("gold_reciprocal_rank")
        return (
            float(primary or 0.0),
            float(row.get("gold_reciprocal_rank") or 0.0),
            str(row.get("method_name") or ""),
            int(row.get("seed") or 0),
        )

    recovered = sorted(
        [row for row in best_rows if row.get("recovery_label") == "gold_recovered"],
        key=order_key,
        reverse=True,
    )
    near_miss = sorted(
        [row for row in best_rows if row.get("recovery_label") == "future_neighbor_only"],
        key=order_key,
        reverse=True,
    )
    confound_or_control = sorted(
        [row for row in best_rows if row.get("recovery_label") in {"historical_confound", "not_recovered"}],
        key=order_key,
        reverse=True,
    )
    off_cue: List[Dict[str, Any]] = []
    if cue_active:
        off_cue = sorted(
            [
                row
                for row in best_rows
                if row.get("recovery_label") == "gold_recovered" and normalize_cue_score(row.get("cue_score")) < 0.4
            ],
            key=order_key,
            reverse=True,
        )

    sampled: List[Dict[str, Any]] = []
    if cue_active:
        sampled.extend(recovered[:20])
        sampled.extend(near_miss[:20])
        sampled.extend(confound_or_control[:20])
        sampled.extend(off_cue[:20])
    else:
        sampled.extend(recovered[:20])
        sampled.extend(near_miss[:20])
        sampled.extend(confound_or_control[:20])

    seen: set[Tuple[str, int, str]] = set()
    rows: List[Dict[str, Any]] = []
    for row in sampled:
        key = (str(row.get("method_name") or ""), int(row.get("seed") or 0), str(row.get("gold_future_paper_id") or ""))
        if key in seen:
            continue
        seen.add(key)

        hypothesis = dict(row.get("hypothesis") or {})
        idea_scores = dict(row.get("idea_scores") or {})
        historical_match = dict(row.get("historical_match") or {})
        future_match = dict(row.get("future_match") or {})
        rows.append(
            {
                "run_id": row.get("run_id"),
                "method_name": row.get("method_name"),
                "seed": row.get("seed"),
                "target_id": row.get("target_id"),
                "target_type": row.get("target_type"),
                "assigned_target_id": row.get("assigned_target_id"),
                "gold_future_paper_id": row.get("gold_future_paper_id"),
                "gold_future_title": row.get("gold_future_title"),
                "gold_future_year": row.get("gold_future_year"),
                "recovery_label": row.get("recovery_label"),
                "gold_rank": row.get("gold_rank"),
                "gold_reciprocal_rank": row.get("gold_reciprocal_rank"),
                "gold_hit_at_1": row.get("gold_hit_at_1"),
                "gold_hit_at_5": row.get("gold_hit_at_5"),
                "gold_hit_at_10": row.get("gold_hit_at_10"),
                "cue_text": ((row.get("discovery_cue") or {}).get("text") or ""),
                "cue_score": row.get("cue_score"),
                "cue_weighted_rr": row.get("cue_weighted_rr"),
                "trace_id": ((row.get("trace_ref") or {}).get("trace_id") or ""),
                "trace_url": ((row.get("trace_ref") or {}).get("url") or ""),
                "trace_session_id": ((row.get("trace_ref") or {}).get("session_id") or ""),
                "trace_observation_id": ((row.get("trace_ref") or {}).get("observation_id") or ""),
                "hypothesis_id": row.get("hypothesis_id"),
                "title": hypothesis.get("title"),
                "text": hypothesis.get("text"),
                "support_citations": list(row.get("support_citations") or []),
                "importance_score": (idea_scores.get("importance") or {}).get("score"),
                "novelty_score": (idea_scores.get("novelty") or {}).get("score"),
                "plausibility_score": (idea_scores.get("plausibility") or {}).get("score"),
                "feasibility_score": (idea_scores.get("feasibility") or {}).get("score"),
                "evaluability_score": (idea_scores.get("evaluability") or {}).get("score"),
                "likely_impact_score": (idea_scores.get("likely_impact") or {}).get("score"),
                "average_idea_score": idea_scores.get("average_score"),
                "idea_score_summary": idea_scores.get("summary"),
                "idea_score_method": idea_scores.get("score_method"),
                "historical_label": row.get("historical_label"),
                "best_historical_confounder_id": row.get("best_historical_confounder_id"),
                "historical_confound_title": historical_match.get("title"),
                "future_neighbor_label": row.get("future_neighbor_label"),
                "best_future_neighbor_paper_id": row.get("best_future_neighbor_paper_id"),
                "future_neighbor_title": future_match.get("title"),
                "evidence_pack_summary": dict(row.get("evidence_pack_summary") or {}),
                "top_future_retrievals": _compact_candidates(row.get("future_candidates") or []),
                "top_historical_retrievals": _compact_candidates(row.get("historical_candidates") or []),
                "manual_review_label": "",
                "manual_review_notes": "",
                "manual_reviewer": "",
            }
        )
    return rows


def _export_review_packets(output_dir: Path, run_payload: Dict[str, Any], matches: Sequence[Dict[str, Any]]) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cue_active = bool((run_payload.get("discovery_cue") or {}).get("text"))
    rows = _review_rows(matches, cue_active=cue_active)

    csv_rows: List[Dict[str, Any]] = []
    for row in rows:
        csv_row: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                csv_row[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                csv_row[key] = ""
            else:
                csv_row[key] = value
        csv_rows.append(csv_row)

    df = pd.DataFrame(csv_rows)
    csv_path = output_dir / f"{run_payload['run_id']}_review_packet.csv"
    json_path = output_dir / f"{run_payload['run_id']}_review_packet.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"run": run_payload, "rows": rows}, f, indent=2, ensure_ascii=False)
    return str(csv_path), str(json_path)


def run_retrospective(
    *,
    backend: BackendClient,
    data_json: str,
    data_dir: str,
    qwen_base_url: str,
    cutoff_date: str = "2020-12-31",
    future_window_start: str = "2022-01-01",
    future_window_end: str = "2025-12-31",
    sensitivity_window_start: Optional[str] = "2021-01-01",
    sensitivity_window_end: Optional[str] = "2025-12-31",
    analysis_config: Optional[AnalysisConfig] = None,
    n_gap_targets: int = 20,
    n_cluster_pair_targets: int = 10,
    n_gold_future_papers: int = 50,
    methods: Optional[Sequence[str]] = None,
    seeds: int = 3,
    hypotheses_per_target: int = 3,
    output_dir: str = "data/retrospective_eval",
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    existing_snapshot_id: Optional[str] = None,
    discovery_cue: Optional[Dict[str, Any] | str] = None,
    future_title_exclude: Optional[Sequence[str]] = None,
    future_abstract_exclude: Optional[Sequence[str]] = None,
    future_semantic_query: Optional[str] = None,
    future_semantic_threshold: Optional[float] = None,
    progress_callback: Optional[Callable[[RetrospectiveProgress], None]] = None,
) -> RetrospectiveResult:
    return _run_retrospective_progress_core_v2(
        backend=backend,
        data_json=data_json,
        data_dir=data_dir,
        qwen_base_url=qwen_base_url,
        cutoff_date=cutoff_date,
        future_window_start=future_window_start,
        future_window_end=future_window_end,
        sensitivity_window_start=sensitivity_window_start,
        sensitivity_window_end=sensitivity_window_end,
        analysis_config=analysis_config,
        n_gap_targets=n_gap_targets,
        n_cluster_pair_targets=n_cluster_pair_targets,
        n_gold_future_papers=n_gold_future_papers,
        methods=methods,
        seeds=seeds,
        hypotheses_per_target=hypotheses_per_target,
        output_dir=output_dir,
        openai_api_key=openai_api_key,
        model_name=model_name,
        existing_snapshot_id=existing_snapshot_id,
        discovery_cue=discovery_cue,
        future_title_exclude=future_title_exclude,
        future_abstract_exclude=future_abstract_exclude,
        future_semantic_query=future_semantic_query,
        future_semantic_threshold=future_semantic_threshold,
        progress_callback=progress_callback,
    )


def _run_retrospective_progress_core_v2(
    *,
    backend: BackendClient,
    data_json: str,
    data_dir: str,
    qwen_base_url: str,
    cutoff_date: str = "2020-12-31",
    future_window_start: str = "2022-01-01",
    future_window_end: str = "2025-12-31",
    sensitivity_window_start: Optional[str] = "2021-01-01",
    sensitivity_window_end: Optional[str] = "2025-12-31",
    analysis_config: Optional[AnalysisConfig] = None,
    n_gap_targets: int = 20,
    n_cluster_pair_targets: int = 10,
    n_gold_future_papers: int = 50,
    methods: Optional[Sequence[str]] = None,
    seeds: int = 3,
    hypotheses_per_target: int = 3,
    output_dir: str = "data/retrospective_eval",
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    existing_snapshot_id: Optional[str] = None,
    discovery_cue: Optional[Dict[str, Any] | str] = None,
    future_title_exclude: Optional[Sequence[str]] = None,
    future_abstract_exclude: Optional[Sequence[str]] = None,
    future_semantic_query: Optional[str] = None,
    future_semantic_threshold: Optional[float] = None,
    progress_callback: Optional[Callable[[RetrospectiveProgress], None]] = None,
) -> RetrospectiveResult:
    run_id = f"retro_eval_{uuid.uuid4().hex[:12]}"
    methods = list(methods or DEFAULT_METHODS)
    analysis_config = analysis_config or AnalysisConfig()
    normalized_cue = discovery_cue_to_dict(normalize_discovery_cue(discovery_cue))
    future_prefilter = _normalize_future_prefilter(
        future_title_exclude=future_title_exclude,
        future_abstract_exclude=future_abstract_exclude,
        future_semantic_query=future_semantic_query,
        future_semantic_threshold=future_semantic_threshold,
    )
    all_match_records: List[Dict[str, Any]] = []
    generation_failures: List[Dict[str, Any]] = []
    run_trace_ref: Dict[str, Any] = {}
    run_output_state: Dict[str, Any] = {}

    with observe_current(
        name="retrospective_run",
        as_type="evaluator",
        input_payload={
            "run_id": run_id,
            "requested_snapshot_id": existing_snapshot_id,
            "method_names": list(methods),
            "seeds": seeds,
            "n_gold_future_papers": n_gold_future_papers,
            "future_prefilter": dict(future_prefilter),
        },
        metadata={
            "run_id": run_id,
            "requested_snapshot_id": existing_snapshot_id,
            "protocol": "future_paper_recovery",
        },
        trace_id=deterministic_trace_id(run_id),
    ) as run_observation:
        with trace_attributes(
            session_id=run_id,
            tags=["retrospective_eval", "run_summary"],
            trace_name="retrospective_run",
            metadata={"run_id": run_id, "requested_snapshot_id": existing_snapshot_id or ""},
        ):
            run_trace_ref = current_trace_ref(
                session_id=run_id,
                tags=["retrospective_eval", "run_summary"],
                metadata={"run_id": run_id, "requested_snapshot_id": existing_snapshot_id or ""},
            )

            def _update_run_output(**kwargs: Any) -> None:
                run_output_state.update({key: value for key, value in kwargs.items() if value is not None})
                run_observation.update(output=dict(run_output_state))

            def _emit_progress(
                phase: str,
                *,
                status: str = "running",
                current: Optional[int] = None,
                total: Optional[int] = None,
                message: str = "",
                method_name: Optional[str] = None,
                seed: Optional[int] = None,
                gold_future_paper_id: Optional[str] = None,
            ) -> RetrospectiveProgress:
                progress = RetrospectiveProgress(
                    phase=phase,
                    status=status,
                    current=current,
                    total=total,
                    message=message,
                    run_id=run_id,
                    method_name=method_name,
                    seed=seed,
                    gold_future_paper_id=gold_future_paper_id,
                    n_matches=len(all_match_records),
                    n_failures=len(generation_failures),
                )
                if progress_callback is not None:
                    progress_callback(progress)
                _update_run_output(progress=progress.to_payload())
                return progress

            try:
                _emit_progress("loading_inputs", message="Loading dataset and embeddings")
                raw_df, raw_embeddings = load_dataset_and_embeddings(data_json, data_dir, embedding_names=["qwen", "bert"])

                analysis_config_payload = analysis_config.model_dump()
                if existing_snapshot_id:
                    _emit_progress("preparing_snapshot", message=f"Reusing historical snapshot `{existing_snapshot_id}`")
                    snapshot = backend.get_snapshot(existing_snapshot_id)
                    snapshot_metadata = dict(snapshot.get("metadata") or {})
                    if snapshot_metadata.get("split_role") not in {None, "historical"}:
                        raise ValueError(
                            f"Existing snapshot `{existing_snapshot_id}` has split_role={snapshot_metadata.get('split_role')!r}; "
                            "retrospective evaluation requires a historical snapshot."
                        )
                    cutoff_date = _metadata_date(snapshot, "cutoff_date")
                    future_window_start = _metadata_date(snapshot, "future_window_start")
                    future_window_end = _metadata_date(snapshot, "future_window_end")
                    bundle_artifact = _resolve_retrospective_bundle_artifact(backend, snapshot)
                    bundle_payload = dict(bundle_artifact.get("payload") or {})
                    if str(bundle_payload.get("historical_snapshot_id") or existing_snapshot_id) != str(existing_snapshot_id):
                        raise ValueError("Retrospective bundle artifact does not point back to the requested historical snapshot.")
                    for field_name, field_value in (
                        ("cutoff_date", cutoff_date),
                        ("future_window_start", future_window_start),
                        ("future_window_end", future_window_end),
                    ):
                        bundle_value = bundle_payload.get(field_name)
                        if bundle_value and str(bundle_value) != str(field_value):
                            raise ValueError(f"Retrospective bundle artifact `{field_name}` does not match snapshot metadata.")

                    manifest = dict(bundle_payload.get("corpus_manifest") or {})
                    df, embeddings = _reconstruct_frontend_corpus(raw_df, raw_embeddings, manifest)
                    split = split_corpus_by_time(
                        df,
                        embeddings,
                        cutoff_date=cutoff_date,
                        future_window_start=future_window_start,
                        future_window_end=future_window_end,
                        sensitivity_window_start=sensitivity_window_start,
                        sensitivity_window_end=sensitivity_window_end,
                    )
                    _validate_reconstructed_historical_split(snapshot, manifest, split.historical.df)

                    stored_analysis_config = dict(
                        snapshot_metadata.get("analysis_config") or bundle_payload.get("analysis_config") or {}
                    )
                    if stored_analysis_config:
                        stored_analysis_config_hash = str(
                            snapshot_metadata.get("analysis_config_hash") or bundle_payload.get("analysis_config_hash") or ""
                        )
                        if stored_analysis_config_hash and _hash_config(stored_analysis_config) != stored_analysis_config_hash:
                            raise ValueError("Stored analysis_config_hash does not match the persisted retrospective analysis_config.")
                        analysis_config_payload = stored_analysis_config
                    snapshot_id = existing_snapshot_id
                else:
                    _emit_progress("preparing_snapshot", message="Building and publishing historical snapshot")
                    df = raw_df
                    embeddings = raw_embeddings
                    split = split_corpus_by_time(
                        df,
                        embeddings,
                        cutoff_date=cutoff_date,
                        future_window_start=future_window_start,
                        future_window_end=future_window_end,
                        sensitivity_window_start=sensitivity_window_start,
                        sensitivity_window_end=sensitivity_window_end,
                    )
                    analysis_embedding = split.historical.embeddings[analysis_config.embedding_name]
                    analysis = run_analysis_v1(split.historical.df, analysis_embedding, config=analysis_config)
                    analysis_config_payload = analysis.analysis_config
                    metadata_overrides = {
                        "split_role": "historical",
                        "cutoff_date": cutoff_date,
                        "future_window_start": future_window_start,
                        "future_window_end": future_window_end,
                        "analysis_config": analysis.analysis_config,
                        "analysis_config_hash": _hash_config(analysis.analysis_config),
                        "embedding_source": analysis_config.embedding_name,
                    }
                    snapshot_id = f"retro_hist_{cutoff_date.replace('-', '')}_{uuid.uuid4().hex[:8]}"
                    payload, _summary = build_snapshot_payload(
                        df=analysis.df,
                        gap_regions=analysis.gap_regions,
                        llm_results=None,
                        selected_clustering=analysis.selected_clustering,
                        x_primary=split.historical.embeddings[analysis_config.embedding_name],
                        x_umap_2d=analysis.x_umap_2d,
                        include_raw_rows=False,
                        include_embeddings=True,
                        snapshot_id=snapshot_id,
                        source="retrospective_eval",
                        metadata_overrides=metadata_overrides,
                    )
                    backend.publish_snapshot(payload)

                _update_run_output(snapshot_id=snapshot_id)
                if run_trace_ref:
                    run_trace_ref["metadata"] = dict(run_trace_ref.get("metadata") or {})
                    run_trace_ref["metadata"]["snapshot_id"] = snapshot_id

                _emit_progress("selecting_targets", message="Selecting retrospective frontier targets")
                gap_targets = (
                    backend.top_gaps(snapshot_id=snapshot_id, k=n_gap_targets).get("gaps", [])
                    if n_gap_targets > 0
                    else []
                )
                gap_target_rows = [
                    {"target_type": "gap", "gap_id": g["gap_id"], "source_rank": idx}
                    for idx, g in enumerate(gap_targets)
                ]
                cluster_targets = _select_cluster_pair_targets(backend, snapshot_id, limit=n_cluster_pair_targets)
                targets: List[Dict[str, Any]] = [*gap_target_rows, *cluster_targets]

                _emit_progress("building_target_pool", message=f"Building focused evidence packs for {len(targets)} targets")
                target_pool = _prepare_target_pool(backend, snapshot_id, targets, normalized_cue)
                cluster_ids = [
                    int(c["cluster_id"])
                    for c in backend.list_clusters(snapshot_id=snapshot_id, limit=200).get("clusters", [])
                ]

                _emit_progress("building_indices", message="Building historical and future retrieval indices")
                qwen_client = QwenClient(qwen_base_url)
                split.future.df, split.future.embeddings, future_prefilter_stats = _apply_future_prefilter(
                    future_df=split.future.df,
                    future_embeddings=split.future.embeddings,
                    qwen_client=qwen_client,
                    config=future_prefilter,
                )
                if future_prefilter_stats["active"]:
                    _emit_progress(
                        "selecting_gold_future_papers",
                        current=0,
                        total=int(future_prefilter_stats["n_future_rows_after"]),
                        message=(
                            "Future prefilter applied: "
                            f"{int(future_prefilter_stats['n_future_rows_before'])} -> "
                            f"{int(future_prefilter_stats['n_future_rows_after'])}"
                        ),
                    )
                historical_index = build_corpus_index(split.historical.df, split.historical.embeddings["qwen"])
                future_index = build_corpus_index(split.future.df, split.future.embeddings["qwen"])

                future_total = int(len(split.future.df))
                _emit_progress(
                    "selecting_gold_future_papers",
                    current=0,
                    total=future_total,
                    message=_gold_selection_progress_message({}),
                )

                def _on_gold_selection_progress(current: int, total: int, stats: Dict[str, int]) -> None:
                    _emit_progress(
                        "selecting_gold_future_papers",
                        current=current,
                        total=total,
                        message=_gold_selection_progress_message(stats),
                    )

                gold_assignments, benchmark_stats = _select_gold_future_papers(
                    future_df=split.future.df,
                    historical_index=historical_index,
                    qwen_client=qwen_client,
                    target_pool=target_pool,
                    discovery_cue=normalized_cue,
                    n_gold_future_papers=n_gold_future_papers,
                    progress_callback=_on_gold_selection_progress if future_total > 0 else None,
                )
                if future_total == 0:
                    _emit_progress(
                        "selecting_gold_future_papers",
                        current=0,
                        total=0,
                        message=_gold_selection_progress_message(benchmark_stats),
                    )

                all_targets = [dict(item["target"]) for item in target_pool]
                total_tasks = len(methods) * seeds * len(gold_assignments)
                task_counter = 0
                _emit_progress(
                    "evaluating_tasks",
                    current=0,
                    total=total_tasks,
                    message="Evaluating retrospective tasks" if total_tasks > 0 else "No evaluation tasks to run",
                )

                for method_name in methods:
                    for seed in range(seeds):
                        for gold in gold_assignments:
                            context = GenerationContext(
                                backend=backend,
                                snapshot_id=snapshot_id,
                                target=dict(gold.assigned_target),
                                seed=seed,
                                openai_api_key=openai_api_key,
                                model_name=model_name,
                                discovery_cue=normalized_cue,
                                hypotheses_per_target=hypotheses_per_target,
                                all_clusters=cluster_ids,
                                all_targets=all_targets,
                            )
                            task_trace_ref: Dict[str, Any] = {}
                            gen_meta: Dict[str, Any] = {}
                            task_metadata = _generation_task_metadata(
                                run_id=run_id,
                                method_name=method_name,
                                seed=seed,
                                gold=gold,
                                snapshot_id=snapshot_id,
                            )
                            task_tags = _generation_task_tags(method_name, seed, gold)
                            task_trace_id = deterministic_trace_id(
                                f"{run_id}:{method_name}:{seed}:{gold.paper_id}:{gold.assigned_target_id}"
                            )
                            task_message = f"{method_name} seed={seed} gold={gold.paper_id}"
                            try:
                                with observe_current(
                                    name="retrospective_generation_task",
                                    as_type="agent",
                                    input_payload=task_metadata,
                                    metadata=task_metadata,
                                    trace_id=task_trace_id,
                                ) as task_observation:
                                    with trace_attributes(
                                        session_id=run_id,
                                        tags=task_tags,
                                        trace_name="retrospective_generation_task",
                                        metadata={
                                            "run_id": run_id,
                                            "snapshot_id": snapshot_id,
                                            "method_name": method_name,
                                            "seed": seed,
                                            "gold_future_paper_id": gold.paper_id,
                                        },
                                    ):
                                        task_trace_ref = current_trace_ref(
                                            session_id=run_id,
                                            tags=task_tags,
                                            metadata=task_metadata,
                                        )
                                        hypotheses, gen_meta = run_generation_method(method_name, context)
                                        task_observation.update(
                                            output={
                                                "n_hypotheses": len(hypotheses),
                                                "effective_target": gen_meta.get("effective_target") or gold.assigned_target,
                                                "evidence_pack_summary": _summarize_evidence_pack(
                                                    gen_meta.get("evidence_pack") or {}
                                                ),
                                            }
                                        )
                            except Exception as exc:
                                generation_failures.append(
                                    {
                                        "method_name": method_name,
                                        "seed": seed,
                                        "gold_future_paper_id": gold.paper_id,
                                        "assigned_target_id": gold.assigned_target_id,
                                        "error": str(exc),
                                        "trace_ref": task_trace_ref,
                                    }
                                )
                                task_counter += 1
                                _emit_progress(
                                    "evaluating_tasks",
                                    current=task_counter,
                                    total=total_tasks,
                                    message=f"Task failed: {task_message}",
                                    method_name=method_name,
                                    seed=seed,
                                    gold_future_paper_id=gold.paper_id,
                                )
                                continue

                            for hypothesis in hypotheses:
                                hypothesis.trace_ref = _coerce_trace_ref(task_trace_ref)

                            effective_target = dict(gen_meta.get("effective_target") or gold.assigned_target)
                            evidence_pack = dict(gen_meta.get("evidence_pack") or {})
                            scored_hypotheses = score_hypotheses(
                                [hypothesis.model_dump() for hypothesis in hypotheses],
                                evidence_pack=evidence_pack,
                                audit=gen_meta.get("audit"),
                                explanation=gen_meta.get("explanation"),
                                target=effective_target,
                                discovery_cue=normalized_cue,
                                openai_api_key=openai_api_key,
                                model_name=model_name,
                            )

                            for hypothesis in hypotheses:
                                hypothesis.idea_scores = dict(scored_hypotheses.get(hypothesis.hypothesis_id) or {})
                                hypothesis_trace_ref = _trace_ref_payload(hypothesis.trace_ref)
                                hyp_payload = hypothesis.model_dump()
                                fingerprint = dict(hypothesis.idea_fingerprint or fingerprint_hypothesis(hyp_payload))
                                query_text = str(fingerprint.get("query_text") or hypothesis.text or hypothesis.title)
                                historical_candidates = retrieve_candidates_for_hypothesis(
                                    query_text=query_text,
                                    fingerprint=fingerprint,
                                    corpus=historical_index,
                                    qwen_client=qwen_client,
                                    top_k_keyword=50,
                                    top_k_semantic=100,
                                    top_k_final=None,
                                    rerank_max_docs=48,
                                )
                                future_candidates = retrieve_candidates_for_hypothesis(
                                    query_text=query_text,
                                    fingerprint=fingerprint,
                                    corpus=future_index,
                                    qwen_client=qwen_client,
                                    required_paper_ids=[gold.paper_id],
                                    top_k_keyword=60,
                                    top_k_semantic=120,
                                    top_k_final=None,
                                    rerank_max_docs=64,
                                )
                                historical_best = best_candidate(historical_candidates)
                                gold_rank = candidate_rank(future_candidates, gold.paper_id)
                                gold_reciprocal_rank = (1.0 / float(gold_rank)) if gold_rank else 0.0
                                best_future_neighbor = best_non_excluded_candidate(future_candidates, [gold.paper_id])
                                recovery_label, judge_meta = classify_recovery_match(
                                    historical_best=historical_best,
                                    gold_rank=gold_rank,
                                    best_future_neighbor=best_future_neighbor,
                                )

                                cue_score: Optional[float] = None
                                cue_weighted_rr = gold_reciprocal_rank
                                if normalized_cue:
                                    cue_score = float(
                                        score_fingerprint_against_cue(fingerprint, normalized_cue).get("score", 0.0) or 0.0
                                    )
                                    cue_weighted_rr = gold_reciprocal_rank * normalize_cue_score(cue_score)

                                match = EvaluationMatch(
                                    run_id=run_id,
                                    snapshot_id=snapshot_id,
                                    target_id=hypothesis.target_id,
                                    target_type=hypothesis.target_type,
                                    method_name=method_name,
                                    seed=seed,
                                    hypothesis_id=hypothesis.hypothesis_id,
                                    recovery_label=recovery_label,
                                    historical_label=judge_meta["historical_label"],
                                    future_neighbor_label=judge_meta["future_neighbor_label"],
                                    gold_future_paper_id=gold.paper_id,
                                    gold_future_title=gold.title,
                                    gold_future_year=gold.publication_year,
                                    assigned_target_id=gold.assigned_target_id,
                                    assigned_target_score=gold.assigned_target_score,
                                    gold_rank=gold_rank,
                                    gold_reciprocal_rank=round(gold_reciprocal_rank, 6),
                                    gold_hit_at_1=bool(gold_rank == 1),
                                    gold_hit_at_5=bool(gold_rank is not None and gold_rank <= 5),
                                    gold_hit_at_10=bool(gold_rank is not None and gold_rank <= 10),
                                    cue_score=cue_score,
                                    cue_weighted_rr=round(cue_weighted_rr, 6),
                                    best_future_neighbor_paper_id=best_future_neighbor.get("paper_id"),
                                    best_historical_confounder_id=historical_best.get("paper_id"),
                                    support_citations=hypothesis.support_citations,
                                    hypothesis=hyp_payload,
                                    idea_scores=hypothesis.idea_scores,
                                    fingerprint=fingerprint,
                                    evidence_pack_summary=_summarize_evidence_pack(evidence_pack),
                                    historical_match=historical_best,
                                    future_match=best_future_neighbor,
                                    historical_candidates=_compact_candidates(historical_candidates, limit=5),
                                    future_candidates=_compact_candidates(future_candidates, limit=5),
                                    discovery_cue=normalized_cue or {},
                                    trace_ref=hypothesis_trace_ref,
                                ).model_dump(exclude_none=True)
                                all_match_records.append(match)
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="recovery_label",
                                    value=recovery_label,
                                    data_type="CATEGORICAL",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="gold_reciprocal_rank",
                                    value=round(gold_reciprocal_rank, 6),
                                    data_type="NUMERIC",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="cue_weighted_rr",
                                    value=round(cue_weighted_rr, 6),
                                    data_type="NUMERIC",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="assigned_target_score",
                                    value=gold.assigned_target_score,
                                    data_type="NUMERIC",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="gold_hit_at_1",
                                    value=float(bool(gold_rank == 1)),
                                    data_type="BOOLEAN",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="gold_hit_at_5",
                                    value=float(bool(gold_rank is not None and gold_rank <= 5)),
                                    data_type="BOOLEAN",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )
                                create_trace_score(
                                    trace_ref=hypothesis_trace_ref,
                                    name="gold_hit_at_10",
                                    value=float(bool(gold_rank is not None and gold_rank <= 10)),
                                    data_type="BOOLEAN",
                                    metadata={"hypothesis_id": hypothesis.hypothesis_id},
                                )

                            task_counter += 1
                            _emit_progress(
                                "evaluating_tasks",
                                current=task_counter,
                                total=total_tasks,
                                message=f"Completed {task_message}",
                                method_name=method_name,
                                seed=seed,
                                gold_future_paper_id=gold.paper_id,
                            )

                batch_count = (len(all_match_records) + 199) // 200 if all_match_records else 0
                _emit_progress(
                    "persisting_matches",
                    current=0,
                    total=batch_count,
                    message=f"Persisting {len(all_match_records)} evaluation matches",
                )
                for batch_idx, batch in enumerate(_chunked(all_match_records, 200), start=1):
                    backend.store_evaluation_matches_batch(list(batch))
                    _emit_progress(
                        "persisting_matches",
                        current=batch_idx,
                        total=batch_count,
                        message=f"Persisting {len(all_match_records)} evaluation matches",
                    )

                _emit_progress("aggregating_results", message="Aggregating retrospective metrics")
                metrics = aggregate_match_metrics(all_match_records)
                summary = {
                    "protocol": "future_paper_recovery",
                    "n_targets": len(targets),
                    "n_gap_targets": len(gap_target_rows),
                    "n_cluster_pair_targets": len(cluster_targets),
                    "n_gold_benchmark": len(gold_assignments),
                    "n_matches": len(all_match_records),
                    "generation_failures": generation_failures,
                    "n_future_pool_before_prefilter": int(future_prefilter_stats["n_future_rows_before"]),
                    "n_future_pool_after_prefilter": int(future_prefilter_stats["n_future_rows_after"]),
                    **benchmark_stats,
                }
                _update_run_output(summary=summary, metrics=metrics)

                run_payload = EvaluationRun(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    created_at=pd.Timestamp.utcnow().isoformat(),
                    cutoff_date=cutoff_date,
                    future_window_start=future_window_start,
                    future_window_end=future_window_end,
                    method_names=list(methods),
                    config={
                        "protocol": "future_paper_recovery",
                        "analysis_config": analysis_config_payload,
                        "analysis_config_hash": _hash_config(analysis_config_payload),
                        "hypotheses_per_target": hypotheses_per_target,
                        "seeds": seeds,
                        "n_gold_future_papers": n_gold_future_papers,
                        "qwen_base_url": qwen_base_url,
                        "resumed_existing_snapshot": bool(existing_snapshot_id),
                        "discovery_cue": normalized_cue or {},
                        "future_prefilter": future_prefilter_stats,
                        "target_pool_profile": "focused_eval",
                    },
                    summary=summary,
                    metrics=metrics,
                    status="completed",
                    discovery_cue=normalized_cue or {},
                    observability=run_trace_ref or {},
                ).model_dump(exclude_none=True)
                backend.store_evaluation_run(run_payload)

                _emit_progress("exporting_review_packet", message="Writing review packet files")
                csv_path, json_path = _export_review_packets(Path(output_dir), run_payload, all_match_records)
                _update_run_output(review_packet_csv=csv_path, review_packet_json=json_path)
                _emit_progress("completed", status="completed", message="Retrospective evaluation completed")
                return RetrospectiveResult(
                    run=run_payload,
                    matches=all_match_records,
                    review_packet_csv=csv_path,
                    review_packet_json=json_path,
                )
            except Exception as exc:
                _emit_progress("failed", status="failed", message=f"Retrospective evaluation failed: {exc}")
                raise
            finally:
                flush_langfuse()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrospective future-paper recovery evaluation.")
    parser.add_argument("--backend-url", default="http://localhost:8088")
    parser.add_argument("--data-json", default="data/cleaned_dataset.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--qwen-base-url", default="http://0.0.0.0:8000")
    parser.add_argument("--cutoff-date", default="2020-12-31")
    parser.add_argument("--future-window-start", default="2022-01-01")
    parser.add_argument("--future-window-end", default="2025-12-31")
    parser.add_argument("--sensitivity-window-start", default="2021-01-01")
    parser.add_argument("--sensitivity-window-end", default="2025-12-31")
    parser.add_argument("--n-gap-targets", type=int, default=20)
    parser.add_argument("--n-cluster-pair-targets", type=int, default=10)
    parser.add_argument("--n-gold-future-papers", type=int, default=50)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hypotheses-per-target", type=int, default=3)
    parser.add_argument("--output-dir", default="data/retrospective_eval")
    parser.add_argument("--analysis-embedding", default="qwen")
    parser.add_argument("--analysis-clustering-method", default="hdbscan", choices=["hdbscan", "kmeans", "leiden"])
    parser.add_argument("--analysis-community-algorithm", default="leiden", choices=["leiden", "louvain"])
    parser.add_argument("--analysis-community-resolution", type=float, default=1.0)
    parser.add_argument("--analysis-community-graph-k", type=int, default=21)
    parser.add_argument("--analysis-community-graph-metric", default="cosine")
    parser.add_argument("--analysis-pca-components", type=int, default=102)
    parser.add_argument("--analysis-random-seed", type=int, default=42)
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"))
    parser.add_argument("--existing-snapshot-id", default=None)
    parser.add_argument("--discovery-cue-text", default=None)
    parser.add_argument("--discovery-cue-goal", default=None)
    parser.add_argument("--future-title-exclude", nargs="+", default=None)
    parser.add_argument("--future-abstract-exclude", nargs="+", default=None)
    parser.add_argument("--future-semantic-query", default=None)
    parser.add_argument("--future-semantic-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis_config = AnalysisConfig(
        embedding_name=args.analysis_embedding,
        clustering_method=args.analysis_clustering_method,
        community_detection_algorithm=args.analysis_community_algorithm,
        community_resolution=args.analysis_community_resolution,
        community_graph_k=args.analysis_community_graph_k,
        community_graph_metric=args.analysis_community_graph_metric,
        pca_components=args.analysis_pca_components,
        random_seed=args.analysis_random_seed,
    )
    backend = BackendClient(args.backend_url)
    progress_reporter = _RetrospectiveCliProgressReporter()
    try:
        result = run_retrospective(
            backend=backend,
            data_json=args.data_json,
            data_dir=args.data_dir,
            qwen_base_url=args.qwen_base_url,
            cutoff_date=args.cutoff_date,
            future_window_start=args.future_window_start,
            future_window_end=args.future_window_end,
            sensitivity_window_start=args.sensitivity_window_start,
            sensitivity_window_end=args.sensitivity_window_end,
            analysis_config=analysis_config,
            n_gap_targets=args.n_gap_targets,
            n_cluster_pair_targets=args.n_cluster_pair_targets,
            n_gold_future_papers=args.n_gold_future_papers,
            methods=args.methods,
            seeds=args.seeds,
            hypotheses_per_target=args.hypotheses_per_target,
            output_dir=args.output_dir,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=args.openai_model,
            existing_snapshot_id=args.existing_snapshot_id,
            discovery_cue={
                "text": args.discovery_cue_text or "",
                "goal": args.discovery_cue_goal,
            }
            if args.discovery_cue_text or args.discovery_cue_goal
            else None,
            future_title_exclude=args.future_title_exclude,
            future_abstract_exclude=args.future_abstract_exclude,
            future_semantic_query=args.future_semantic_query,
            future_semantic_threshold=args.future_semantic_threshold,
            progress_callback=progress_reporter,
        )
    finally:
        progress_reporter.close()
    print(
        json.dumps(
            {
                "run": result.run,
                "review_packet_csv": result.review_packet_csv,
                "review_packet_json": result.review_packet_json,
                "observability": result.run.get("observability", {}),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
