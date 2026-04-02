from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .metrics import select_best_task_rows


ASSESSMENT_BUNDLE_SCHEMA_VERSION = "assessment_bundle_v1"

ASSESSMENT_RUBRIC: Dict[str, Dict[str, str]] = {
    "importance": {
        "label": "Importance",
        "guidance": "How important is the addressed problem if the idea works?",
    },
    "novelty": {
        "label": "Novelty",
        "guidance": "Judge novelty relative to the provided evidence context, not all science ever published.",
    },
    "plausibility": {
        "label": "Plausibility",
        "guidance": "How well grounded is the idea in the provided evidence and mechanism?",
    },
    "feasibility": {
        "label": "Feasibility",
        "guidance": "Assume a capable academic lab and a 6-12 month time horizon.",
    },
    "evaluability": {
        "label": "Evaluability",
        "guidance": "Can the idea be tested with clear experiments and measurable readouts?",
    },
    "likely_impact": {
        "label": "Likely Impact",
        "guidance": "If successful, how much would the idea matter scientifically or clinically?",
    },
}

ASSESSMENT_RUBRIC_FIELDS: Tuple[str, ...] = tuple(ASSESSMENT_RUBRIC.keys())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bundle_hash(bundle: Dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(bundle))


def _text_hash(row: Dict[str, Any]) -> str:
    hypothesis = dict(row.get("hypothesis") or {})
    payload = {
        "hypothesis_id": str(row.get("hypothesis_id") or hypothesis.get("hypothesis_id") or ""),
        "title": str(hypothesis.get("title") or ""),
        "text": str(hypothesis.get("text") or ""),
    }
    return _sha256_text(_canonical_json(payload))[:16]


def _queue_sort_key(row: Dict[str, Any]) -> List[Any]:
    return [
        str(row.get("run_id") or ""),
        str(row.get("method_name") or ""),
        int(row.get("seed") or 0),
        str(row.get("target_id") or row.get("assigned_target_id") or ""),
        str(row.get("hypothesis_id") or ""),
    ]


def _idea_group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("snapshot_id") or ""),
        str(row.get("method_name") or ""),
        int(row.get("seed") or 0),
        str(row.get("target_id") or row.get("assigned_target_id") or ""),
        str(row.get("hypothesis_id") or ""),
        _text_hash(row),
    )


def _task_row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("method_name") or ""),
        int(row.get("seed") or 0),
        str(row.get("gold_future_paper_id") or ""),
        str(row.get("target_id") or row.get("assigned_target_id") or ""),
        str(row.get("hypothesis_id") or ""),
        _text_hash(row),
    )


def build_idea_id(row: Dict[str, Any]) -> str:
    payload = {
        "run_id": str(row.get("run_id") or ""),
        "snapshot_id": str(row.get("snapshot_id") or ""),
        "method_name": str(row.get("method_name") or ""),
        "seed": int(row.get("seed") or 0),
        "target_id": str(row.get("target_id") or row.get("assigned_target_id") or ""),
        "hypothesis_id": str(row.get("hypothesis_id") or ""),
        "text_hash": _text_hash(row),
    }
    return f"idea_{_sha256_text(_canonical_json(payload))[:20]}"


def _compact_benchmark_row(row: Dict[str, Any], *, is_review_packet_winner: bool) -> Dict[str, Any]:
    return {
        "task_key": {
            "method_name": row.get("method_name"),
            "seed": row.get("seed"),
            "gold_future_paper_id": row.get("gold_future_paper_id"),
        },
        "assigned_target_id": row.get("assigned_target_id"),
        "assigned_target_score": row.get("assigned_target_score"),
        "gold_future_paper_id": row.get("gold_future_paper_id"),
        "gold_future_title": row.get("gold_future_title"),
        "gold_future_year": row.get("gold_future_year"),
        "recovery_label": row.get("recovery_label"),
        "historical_label": row.get("historical_label"),
        "future_neighbor_label": row.get("future_neighbor_label"),
        "gold_rank": row.get("gold_rank"),
        "gold_reciprocal_rank": row.get("gold_reciprocal_rank"),
        "gold_hit_at_1": row.get("gold_hit_at_1"),
        "gold_hit_at_5": row.get("gold_hit_at_5"),
        "gold_hit_at_10": row.get("gold_hit_at_10"),
        "cue_score": row.get("cue_score"),
        "cue_weighted_rr": row.get("cue_weighted_rr"),
        "is_review_packet_winner": is_review_packet_winner,
        "historical_match": dict(row.get("historical_match") or {}),
        "future_match": dict(row.get("future_match") or {}),
        "top_historical_retrievals": list(row.get("historical_candidates") or []),
        "top_future_retrievals": list(row.get("future_candidates") or []),
    }


def _first_non_empty(rows: Iterable[Dict[str, Any]], field_name: str) -> Dict[str, Any]:
    for row in rows:
        payload = dict(row.get(field_name) or {})
        if payload:
            return payload
    return {}


def _evidence_pack_papers(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    pack = dict(row.get("evidence_pack") or {})
    return list(pack.get("papers") or [])


def _prospective_queue_sort_key(row: Dict[str, Any]) -> List[Any]:
    return [
        str(row.get("run_id") or ""),
        str(row.get("method_name") or ""),
        int(row.get("seed") or 0),
        str(row.get("target_id") or ""),
        str(row.get("hypothesis_id") or ""),
    ]


def _prospective_idea_record(run_payload: Dict[str, Any], task: Dict[str, Any], hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(task.get("run_id") or run_payload.get("run_id") or "")
    snapshot_id = str(task.get("snapshot_id") or run_payload.get("snapshot_id") or "")
    method_name = str(task.get("method_name") or "")
    seed = int(task.get("seed") or 0)
    target_id = str(task.get("target_id") or "")
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
    title = str(hypothesis.get("title") or "")
    text = str(hypothesis.get("text") or "")
    target = dict(task.get("target") or {})
    effective_target = dict(task.get("effective_target") or target)
    evidence_pack = dict(task.get("evidence_pack") or {})
    discovery_cue = dict(hypothesis.get("discovery_cue") or run_payload.get("discovery_cue") or {})
    idea_scores = dict(hypothesis.get("idea_scores") or {})
    row_for_id = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "method_name": method_name,
        "seed": seed,
        "target_id": target_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "title": title,
            "text": text,
        },
    }
    queue_sort_key = _prospective_queue_sort_key(row_for_id)
    trace_ref = dict(hypothesis.get("trace_ref") or task.get("trace_ref") or {})
    return {
        "idea_id": build_idea_id(row_for_id),
        "is_review_packet_winner": True,
        "winner_task_count": 1,
        "run_context": {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "method_name": method_name,
            "seed": seed,
            "target_id": target_id,
            "hypothesis_id": hypothesis_id,
            "queue_sort_key": queue_sort_key,
            "trace_ref": trace_ref,
            "source_match_count": 1,
        },
        "target": {
            "target_id": target_id,
            "target_type": task.get("target_type") or target.get("target_type"),
            "effective_target": effective_target,
        },
        "discovery_cue": discovery_cue,
        "ideation_context": {
            "effective_target": effective_target,
            "evidence_pack_summary": dict(task.get("evidence_pack_summary") or {}),
            "evidence_pack_meta": dict(evidence_pack.get("meta") or {}),
            "evidence_papers": list(evidence_pack.get("papers") or []),
            "explanation": dict(task.get("explanation") or {}),
            "audit": dict(task.get("audit") or {}),
        },
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "title": title,
            "text": text,
            "support_citations": list(hypothesis.get("support_citations") or []),
            "raw_hypothesis": dict(hypothesis.get("raw_hypothesis") or {}),
            "normalized_hypothesis": dict(hypothesis.get("normalized_hypothesis") or {}),
            "grounding_summary": dict(hypothesis.get("grounding_summary") or {}),
            "idea_fingerprint": dict(hypothesis.get("idea_fingerprint") or {}),
            "trace_ref": trace_ref,
        },
        "judge_context": {
            "idea_scores": idea_scores,
            "score_summary": idea_scores.get("summary"),
            "judge_model": idea_scores.get("judge_model"),
            "score_method": idea_scores.get("score_method"),
        },
        "benchmark_context": {
            "evaluations": [],
            "historical_match": {},
            "future_match": {},
        },
    }


def _build_prospective_assessment_bundle(run_payload: Dict[str, Any], tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    run_id = str(run_payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Prospective hypotheses JSON is missing `run.run_id`.")

    ideas: List[Dict[str, Any]] = []
    for task in tasks:
        hypotheses = list(task.get("hypotheses") or [])
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                raise ValueError("Prospective hypotheses JSON contains an invalid hypothesis row.")
            ideas.append(_prospective_idea_record(run_payload, task, hypothesis))
    ideas.sort(key=lambda idea: (tuple((idea.get("run_context") or {}).get("queue_sort_key") or []), str(idea.get("idea_id") or "")))

    bundle = {
        "schema_version": ASSESSMENT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": f"assessment_{run_id}",
        "created_at": run_payload.get("created_at"),
        "source_kind": "prospective_run",
        "rubric": dict(ASSESSMENT_RUBRIC),
        "run_manifest": {
            "run_id": run_payload.get("run_id"),
            "snapshot_id": run_payload.get("snapshot_id"),
            "created_at": run_payload.get("created_at"),
            "method_names": list(run_payload.get("method_names") or []),
            "config": dict(run_payload.get("config") or {}),
            "summary": dict(run_payload.get("summary") or {}),
            "metrics": dict(run_payload.get("metrics") or {}),
            "discovery_cue": dict(run_payload.get("discovery_cue") or {}),
            "status": run_payload.get("status"),
        },
        "ideas": ideas,
    }
    bundle["bundle_sha256"] = bundle_hash(bundle)
    return bundle


def _idea_record(group_rows: Sequence[Dict[str, Any]], winner_task_keys: set[Tuple[Any, ...]]) -> Dict[str, Any]:
    rows = sorted(group_rows, key=_queue_sort_key)
    primary = rows[0]
    hypothesis = dict(primary.get("hypothesis") or {})
    idea_id = build_idea_id(primary)
    winners = [_task_row_key(row) for row in rows if _task_row_key(row) in winner_task_keys]
    queue_sort_key = _queue_sort_key(primary)
    benchmark_evaluations = [
        _compact_benchmark_row(row, is_review_packet_winner=_task_row_key(row) in winner_task_keys) for row in rows
    ]

    evidence_pack = dict(primary.get("evidence_pack") or {})
    explanation = dict(primary.get("explanation") or {})
    audit = dict(primary.get("audit") or {})
    effective_target = dict(primary.get("effective_target") or {})
    discovery_cue = dict(primary.get("discovery_cue") or hypothesis.get("discovery_cue") or {})

    return {
        "idea_id": idea_id,
        "is_review_packet_winner": bool(winners),
        "winner_task_count": len(winners),
        "run_context": {
            "run_id": primary.get("run_id"),
            "snapshot_id": primary.get("snapshot_id"),
            "method_name": primary.get("method_name"),
            "seed": primary.get("seed"),
            "target_id": primary.get("target_id"),
            "hypothesis_id": primary.get("hypothesis_id"),
            "queue_sort_key": queue_sort_key,
            "trace_ref": dict(primary.get("trace_ref") or hypothesis.get("trace_ref") or {}),
            "source_match_count": len(rows),
        },
        "target": {
            "target_id": primary.get("target_id"),
            "target_type": primary.get("target_type"),
            "assigned_target_id": primary.get("assigned_target_id"),
            "effective_target": effective_target,
        },
        "discovery_cue": discovery_cue,
        "ideation_context": {
            "effective_target": effective_target,
            "evidence_pack_summary": dict(primary.get("evidence_pack_summary") or {}),
            "evidence_pack_meta": dict(evidence_pack.get("meta") or {}),
            "evidence_papers": _evidence_pack_papers(primary),
            "explanation": explanation,
            "audit": audit,
        },
        "hypothesis": {
            "hypothesis_id": primary.get("hypothesis_id"),
            "title": hypothesis.get("title"),
            "text": hypothesis.get("text"),
            "support_citations": list(primary.get("support_citations") or hypothesis.get("support_citations") or []),
            "raw_hypothesis": dict(hypothesis.get("raw_hypothesis") or {}),
            "normalized_hypothesis": dict(hypothesis.get("normalized_hypothesis") or {}),
            "grounding_summary": dict(hypothesis.get("grounding_summary") or {}),
            "idea_fingerprint": dict(primary.get("fingerprint") or hypothesis.get("idea_fingerprint") or {}),
            "trace_ref": dict(primary.get("trace_ref") or hypothesis.get("trace_ref") or {}),
        },
        "judge_context": {
            "idea_scores": dict(primary.get("idea_scores") or hypothesis.get("idea_scores") or {}),
            "score_summary": (primary.get("idea_scores") or {}).get("summary"),
            "judge_model": (primary.get("idea_scores") or {}).get("judge_model"),
            "score_method": (primary.get("idea_scores") or {}).get("score_method"),
        },
        "benchmark_context": {
            "evaluations": benchmark_evaluations,
            "historical_match": _first_non_empty(rows, "historical_match"),
            "future_match": _first_non_empty(rows, "future_match"),
        },
    }


def build_assessment_bundle(run_payload: Dict[str, Any], matches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    winner_rows = select_best_task_rows(matches)
    winner_task_keys = {_task_row_key(row) for row in winner_rows}

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in matches:
        grouped.setdefault(_idea_group_key(row), []).append(row)

    ideas = [_idea_record(rows, winner_task_keys) for rows in grouped.values()]
    ideas.sort(key=lambda idea: tuple(idea.get("run_context", {}).get("queue_sort_key") or []))

    bundle = {
        "schema_version": ASSESSMENT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": f"assessment_{str(run_payload.get('run_id') or '')}",
        "created_at": run_payload.get("created_at"),
        "source_kind": "retrospective_run",
        "rubric": dict(ASSESSMENT_RUBRIC),
        "run_manifest": {
            "run_id": run_payload.get("run_id"),
            "snapshot_id": run_payload.get("snapshot_id"),
            "created_at": run_payload.get("created_at"),
            "cutoff_date": run_payload.get("cutoff_date"),
            "future_window_start": run_payload.get("future_window_start"),
            "future_window_end": run_payload.get("future_window_end"),
            "method_names": list(run_payload.get("method_names") or []),
            "config": dict(run_payload.get("config") or {}),
            "summary": dict(run_payload.get("summary") or {}),
            "metrics": dict(run_payload.get("metrics") or {}),
            "discovery_cue": dict(run_payload.get("discovery_cue") or {}),
        },
        "ideas": ideas,
    }
    bundle["bundle_sha256"] = bundle_hash(bundle)
    return bundle


def write_assessment_bundle(output_dir: Path, run_payload: Dict[str, Any], matches: Sequence[Dict[str, Any]]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_assessment_bundle(run_payload, matches)
    json_path = output_dir / f"{run_payload['run_id']}_{ASSESSMENT_BUNDLE_SCHEMA_VERSION}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    return str(json_path)


def _validate_assessment_bundle_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Assessment bundle JSON must contain an object at the top level.")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != ASSESSMENT_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "This file is not a full assessment bundle. Review requires an `assessment_bundle_v1` JSON export."
        )
    if not isinstance(payload.get("ideas"), list):
        raise ValueError("Assessment bundle is missing `ideas`.")
    normalized = dict(payload)
    computed_hash = bundle_hash({key: value for key, value in normalized.items() if key != "bundle_sha256"})
    embedded_hash = str(normalized.get("bundle_sha256") or "")
    if embedded_hash and embedded_hash != computed_hash:
        raise ValueError("Assessment bundle hash does not match the file contents.")
    normalized["bundle_sha256"] = computed_hash
    return normalized


def _validate_prospective_hypotheses_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Prospective hypotheses JSON must contain an object at the top level.")
    run_payload = payload.get("run")
    tasks = payload.get("tasks")
    if not isinstance(run_payload, dict) or not isinstance(tasks, list):
        raise ValueError(
            "This file is not a prospective hypotheses export. Prospective mode requires a `<run_id>_hypotheses.json` file containing `run` and `tasks`."
        )
    normalized_tasks: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Prospective hypotheses JSON contains an invalid task row.")
        hypothesis_rows = task.get("hypotheses")
        if hypothesis_rows is None:
            normalized_hypotheses: List[Dict[str, Any]] = []
        elif isinstance(hypothesis_rows, list):
            normalized_hypotheses = []
            for row in hypothesis_rows:
                if not isinstance(row, dict):
                    raise ValueError("Prospective hypotheses JSON contains an invalid hypothesis row.")
                normalized_hypotheses.append(dict(row))
        else:
            raise ValueError("Prospective hypotheses JSON contains an invalid `task.hypotheses` payload.")
        normalized_task = dict(task)
        normalized_task["hypotheses"] = normalized_hypotheses
        normalized_tasks.append(normalized_task)
    return {
        "run": dict(run_payload),
        "tasks": normalized_tasks,
        "failures": list(payload.get("failures") or []),
    }


def load_assessment_bundle_text(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Assessment bundle is not valid JSON.") from exc
    return _validate_assessment_bundle_payload(payload)


def load_assessment_bundle_bytes(data: bytes) -> Dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Assessment bundle must be UTF-8 encoded JSON.") from exc
    return load_assessment_bundle_text(text)


def load_assessment_bundle(path: str | Path) -> Dict[str, Any]:
    bundle_path = Path(path)
    return load_assessment_bundle_text(bundle_path.read_text(encoding="utf-8"))


def load_prospective_hypotheses_text(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Prospective hypotheses file is not valid JSON.") from exc
    normalized = _validate_prospective_hypotheses_payload(payload)
    return _build_prospective_assessment_bundle(dict(normalized.get("run") or {}), list(normalized.get("tasks") or []))


def load_prospective_hypotheses_bytes(data: bytes) -> Dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Prospective hypotheses file must be UTF-8 encoded JSON.") from exc
    return load_prospective_hypotheses_text(text)


def load_prospective_hypotheses(path: str | Path) -> Dict[str, Any]:
    hypotheses_path = Path(path)
    return load_prospective_hypotheses_text(hypotheses_path.read_text(encoding="utf-8"))
