from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:
    from agents.backend_client import BackendClient
    from agents.schemas import AnalysisConfig, EvaluationMatch, EvaluationRun
    from agents.snapshot_builder import build_snapshot_payload
except Exception:  # pragma: no cover
    from novelty_app.agents.backend_client import BackendClient
    from novelty_app.agents.schemas import AnalysisConfig, EvaluationMatch, EvaluationRun
    from novelty_app.agents.snapshot_builder import build_snapshot_payload

from .analysis_v1 import run_analysis_v1
from .candidate_match import best_candidate, build_corpus_index, first_matching_year, retrieve_candidates_for_hypothesis
from .generators import GenerationContext, run_generation_method, target_id
from .idea_fingerprint import fingerprint_hypothesis
from .judge import classify_hypothesis_match
from .metrics import aggregate_match_metrics
from .qwen_client import QwenClient
from .time_split import load_dataset_and_embeddings, split_corpus_by_time
from novelty_app.discovery_cue import discovery_cue_to_dict, normalize_discovery_cue


@dataclass
class RetrospectiveResult:
    run: Dict[str, Any]
    matches: List[Dict[str, Any]]
    review_packet_csv: str
    review_packet_json: str


def _hash_config(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _chunked(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _select_cluster_pair_targets(
    backend: BackendClient,
    snapshot_id: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
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
        "exemplars": 4,
        "boundary": 4,
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


def _cue_target_score(
    backend: BackendClient,
    snapshot_id: str,
    target: Dict[str, Any],
    discovery_cue: Optional[Dict[str, Any]],
) -> float:
    if not discovery_cue or not hasattr(backend, "evidence_pack"):
        return 0.0
    try:
        pack = backend.evidence_pack(_target_request(snapshot_id, target, discovery_cue=discovery_cue))
    except Exception:
        return 0.0
    scores: List[float] = []
    for paper in pack.get("papers", [])[:6]:
        scores.append(float(paper.get("selection_meta", {}).get("cue_score", 0.0) or 0.0))
    return (sum(scores) / len(scores)) if scores else 0.0


def _rerank_targets_by_discovery_cue(
    backend: BackendClient,
    snapshot_id: str,
    targets: Sequence[Dict[str, Any]],
    discovery_cue: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for target in targets:
        enriched = dict(target)
        enriched["cue_target_score"] = _cue_target_score(backend, snapshot_id, target, discovery_cue)
        ranked.append(enriched)
    ranked.sort(key=lambda item: float(item.get("cue_target_score", 0.0) or 0.0), reverse=True)
    return ranked


def _review_rows(matches: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in matches:
        hypothesis = match.get("hypothesis", {})
        historical = match.get("historical_match", {})
        future = match.get("future_match", {})
        rows.append(
            {
                "run_id": match.get("run_id"),
                "method_name": match.get("method_name"),
                "seed": match.get("seed"),
                "target_id": match.get("target_id"),
                "target_type": match.get("target_type"),
                "hypothesis_id": match.get("hypothesis_id"),
                "classification": match.get("classification"),
                "title": hypothesis.get("title"),
                "text": hypothesis.get("text"),
                "discovery_cue_text": ((match.get("discovery_cue") or {}).get("text") or ""),
                "support_citations": "; ".join(match.get("support_citations") or []),
                "historical_label": match.get("historical_label"),
                "historical_best_paper_id": match.get("historical_best_paper_id"),
                "historical_best_title": historical.get("title"),
                "future_label": match.get("future_label"),
                "future_best_paper_id": match.get("future_best_paper_id"),
                "future_best_title": future.get("title"),
                "first_future_year": match.get("first_future_year"),
            }
        )
    return rows


def _export_review_packets(output_dir: Path, run_payload: Dict[str, Any], matches: Sequence[Dict[str, Any]]) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _review_rows(matches)
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{run_payload['run_id']}_review_packet.csv"
    json_path = output_dir / f"{run_payload['run_id']}_review_packet.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run": run_payload,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
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
    methods: Optional[Sequence[str]] = None,
    seeds: int = 3,
    hypotheses_per_target: int = 3,
    output_dir: str = "data/retrospective_eval",
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    existing_snapshot_id: Optional[str] = None,
    discovery_cue: Optional[Dict[str, Any] | str] = None,
) -> RetrospectiveResult:
    methods = list(methods or ["orchestrator", "single_shot_llm", "retrieval_summary_direct", "cluster_only", "random_cluster_pair_control"])
    analysis_config = analysis_config or AnalysisConfig()
    normalized_cue = discovery_cue_to_dict(normalize_discovery_cue(discovery_cue))
    df, embeddings = load_dataset_and_embeddings(data_json, data_dir, embedding_names=["qwen", "bert"])
    split = split_corpus_by_time(
        df,
        embeddings,
        cutoff_date=cutoff_date,
        future_window_start=future_window_start,
        future_window_end=future_window_end,
        sensitivity_window_start=sensitivity_window_start,
        sensitivity_window_end=sensitivity_window_end,
    )

    analysis_config_payload = analysis_config.model_dump()
    if existing_snapshot_id:
        snapshot_id = existing_snapshot_id
    else:
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

    gap_targets = backend.top_gaps(snapshot_id=snapshot_id, k=n_gap_targets).get("gaps", [])
    cluster_targets = _select_cluster_pair_targets(backend, snapshot_id, limit=n_cluster_pair_targets)
    if normalized_cue:
        gap_target_rows = [
            {"target_type": "gap", "gap_id": g["gap_id"], "source_rank": idx}
            for idx, g in enumerate(gap_targets)
        ]
        cluster_targets = _rerank_targets_by_discovery_cue(backend, snapshot_id, cluster_targets, normalized_cue)
        gap_target_rows = _rerank_targets_by_discovery_cue(backend, snapshot_id, gap_target_rows, normalized_cue)
    else:
        gap_target_rows = [
            {"target_type": "gap", "gap_id": g["gap_id"], "source_rank": idx}
            for idx, g in enumerate(gap_targets)
        ]
    targets: List[Dict[str, Any]] = [
        *gap_target_rows,
        *cluster_targets,
    ]

    cluster_ids = [int(c["cluster_id"]) for c in backend.list_clusters(snapshot_id=snapshot_id, limit=200).get("clusters", [])]
    qwen_client = QwenClient(qwen_base_url)
    historical_index = build_corpus_index(split.historical.df, split.historical.embeddings["qwen"])
    future_index = build_corpus_index(split.future.df, split.future.embeddings["qwen"])
    sensitivity_index = None
    if split.sensitivity_future is not None and len(split.sensitivity_future.df) > 0:
        sensitivity_index = build_corpus_index(split.sensitivity_future.df, split.sensitivity_future.embeddings["qwen"])

    run_id = f"retro_eval_{uuid.uuid4().hex[:12]}"
    all_match_records: List[Dict[str, Any]] = []
    generation_failures: List[Dict[str, Any]] = []

    for method_name in methods:
        for seed in range(seeds):
            for target in targets:
                context = GenerationContext(
                    backend=backend,
                    snapshot_id=snapshot_id,
                    target=target,
                    seed=seed,
                    openai_api_key=openai_api_key,
                    model_name=model_name,
                    discovery_cue=normalized_cue,
                    hypotheses_per_target=hypotheses_per_target,
                    all_clusters=cluster_ids,
                )
                try:
                    hypotheses, gen_meta = run_generation_method(method_name, context)
                except Exception as exc:
                    generation_failures.append(
                        {
                            "method_name": method_name,
                            "seed": seed,
                            "target_id": target_id(target),
                            "error": str(exc),
                        }
                    )
                    continue

                effective_target = dict(gen_meta.get("effective_target") or target)
                for hypothesis in hypotheses:
                    hyp_payload = hypothesis.model_dump()
                    fingerprint = dict(hypothesis.idea_fingerprint or fingerprint_hypothesis(hyp_payload))
                    query_text = str(fingerprint.get("query_text") or hypothesis.text or hypothesis.title)
                    historical_candidates = retrieve_candidates_for_hypothesis(
                        query_text=query_text,
                        fingerprint=fingerprint,
                        corpus=historical_index,
                        qwen_client=qwen_client,
                    )
                    future_candidates = retrieve_candidates_for_hypothesis(
                        query_text=query_text,
                        fingerprint=fingerprint,
                        corpus=future_index,
                        qwen_client=qwen_client,
                    )
                    sensitivity_best: Optional[Dict[str, Any]] = None
                    if sensitivity_index is not None:
                        sensitivity_candidates = retrieve_candidates_for_hypothesis(
                            query_text=query_text,
                            fingerprint=fingerprint,
                            corpus=sensitivity_index,
                            qwen_client=qwen_client,
                        )
                        sensitivity_best = best_candidate(sensitivity_candidates)
                    historical_best = best_candidate(historical_candidates)
                    future_best = best_candidate(future_candidates)
                    classification, judge_meta = classify_hypothesis_match(
                        historical_best=historical_best,
                        future_best=future_best,
                        support_citations=hypothesis.support_citations,
                        grounding_summary=hypothesis.grounding_summary,
                    )
                    future_payload = dict(future_best)
                    if sensitivity_best is not None:
                        future_payload["sensitivity_best"] = sensitivity_best
                    match = EvaluationMatch(
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        target_id=hypothesis.target_id,
                        target_type=hypothesis.target_type,
                        method_name=method_name,
                        seed=seed,
                        hypothesis_id=hypothesis.hypothesis_id,
                        classification=classification,
                        historical_label=judge_meta["historical_label"],
                        future_label=judge_meta["future_label"],
                        first_future_year=first_matching_year(future_candidates),
                        historical_best_paper_id=historical_best.get("paper_id"),
                        future_best_paper_id=future_best.get("paper_id"),
                        support_citations=hypothesis.support_citations,
                        hypothesis=hyp_payload,
                        fingerprint=fingerprint,
                        historical_match=historical_best,
                        future_match=future_payload,
                        discovery_cue=normalized_cue or {},
                    ).model_dump()
                    all_match_records.append(match)

    for batch in _chunked(all_match_records, 200):
        backend.store_evaluation_matches_batch(list(batch))

    metrics = aggregate_match_metrics(all_match_records)
    summary = {
        "n_targets": len(targets),
        "n_gap_targets": len(gap_targets),
        "n_cluster_pair_targets": len(cluster_targets),
        "n_matches": len(all_match_records),
        "generation_failures": generation_failures,
    }
    run_payload = EvaluationRun(
        run_id=run_id,
        snapshot_id=snapshot_id,
        created_at=pd.Timestamp.utcnow().isoformat(),
        cutoff_date=cutoff_date,
        future_window_start=future_window_start,
        future_window_end=future_window_end,
        method_names=list(methods),
        config={
            "analysis_config": analysis_config_payload,
            "analysis_config_hash": _hash_config(analysis_config_payload),
            "hypotheses_per_target": hypotheses_per_target,
            "seeds": seeds,
            "qwen_base_url": qwen_base_url,
            "resumed_existing_snapshot": bool(existing_snapshot_id),
            "discovery_cue": normalized_cue or {},
            "cue_target_reranking": bool(normalized_cue),
        },
        summary=summary,
        metrics=metrics,
        status="completed",
        discovery_cue=normalized_cue or {},
    ).model_dump()
    backend.store_evaluation_run(run_payload)

    csv_path, json_path = _export_review_packets(Path(output_dir), run_payload, all_match_records)
    return RetrospectiveResult(
        run=run_payload,
        matches=all_match_records,
        review_packet_csv=csv_path,
        review_packet_json=json_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrospective novelty evaluation.")
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
    parser.add_argument("--methods", nargs="+", default=["orchestrator", "single_shot_llm", "retrieval_summary_direct", "cluster_only", "random_cluster_pair_control"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hypotheses-per-target", type=int, default=3)
    parser.add_argument("--output-dir", default="data/retrospective_eval")
    parser.add_argument("--analysis-embedding", default="qwen")
    parser.add_argument("--analysis-clustering-method", default="hdbscan", choices=["hdbscan", "kmeans"])
    parser.add_argument("--analysis-pca-components", type=int, default=102)
    parser.add_argument("--analysis-random-seed", type=int, default=42)
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"))
    parser.add_argument("--existing-snapshot-id", default=None)
    parser.add_argument("--discovery-cue-text", default=None)
    parser.add_argument("--discovery-cue-goal", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis_config = AnalysisConfig(
        embedding_name=args.analysis_embedding,
        clustering_method=args.analysis_clustering_method,
        pca_components=args.analysis_pca_components,
        random_seed=args.analysis_random_seed,
    )
    backend = BackendClient(args.backend_url)
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
    )
    print(json.dumps(
        {
            "run": result.run,
            "review_packet_csv": result.review_packet_csv,
            "review_packet_json": result.review_packet_json,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
