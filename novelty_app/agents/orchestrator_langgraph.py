from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field

try:
    from agents.backend_client import BackendClient
except Exception:  # pragma: no cover
    try:
        from novelty_app.agents.backend_client import BackendClient  # type: ignore
    except Exception:
        from backend_client import BackendClient  # type: ignore

try:
    from discovery_cue import cue_prompt_block, discovery_cue_to_dict
except Exception:  # pragma: no cover
    from novelty_app.discovery_cue import cue_prompt_block, discovery_cue_to_dict

try:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    END = None  # type: ignore
    StateGraph = None  # type: ignore


class Axis(BaseModel):
    axis: str
    what_differs: str
    evidence_A: List[str] = Field(default_factory=list)
    evidence_B: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class BridgeSeed(BaseModel):
    idea: str
    why_plausible: str
    support: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class ClusterSummary(BaseModel):
    one_line: str
    bullets: List[str] = Field(default_factory=list)
    salient_entities: Dict[str, List[str]] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)


class ContrastiveExplanation(BaseModel):
    cluster_A_summary: ClusterSummary
    cluster_B_summary: ClusterSummary
    axes_of_separation: List[Axis] = Field(default_factory=list)
    bridge_seeds: List[BridgeSeed] = Field(default_factory=list)
    insufficient_evidence: bool = False


class AuditClaim(BaseModel):
    claim: str
    supported: bool
    missing_evidence_queries: List[str] = Field(default_factory=list)
    notes: str = ""


class AuditReport(BaseModel):
    supported_claim_fraction: float = Field(ge=0.0, le=1.0)
    needs_patch: bool
    missing_facets: List[str] = Field(default_factory=list)
    patch_queries: List[str] = Field(default_factory=list)
    claims: List[AuditClaim] = Field(default_factory=list)
    cue_alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    cue_violations: List[str] = Field(default_factory=list)
    missing_cue_facets: List[str] = Field(default_factory=list)
    respects_hard_constraints: bool = True


class Hypothesis(BaseModel):
    id: str
    title: str
    bridge_type: str
    mechanistic_rationale: str
    novel_elements: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class HypothesesOut(BaseModel):
    hypotheses: List[Hypothesis]


class BlueprintOut(BaseModel):
    bill_of_materials: List[str] = Field(default_factory=list)
    synthesis_and_characterization: Dict[str, Any] = Field(default_factory=dict)
    in_vitro_plan: Dict[str, Any] = Field(default_factory=dict)
    in_vivo_plan: Dict[str, Any] = Field(default_factory=dict)
    risks_and_mitigations: List[Dict[str, Any]] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class OrchestratorState(TypedDict, total=False):
    target_type: Literal["gap", "cluster_pair"]
    snapshot_id: str
    gap_id: str
    cluster_a: int
    cluster_b: int
    max_iters: int
    iter: int
    exemplars: int
    boundary: int
    diverse: int
    discovery_cue: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    evidence_meta: Dict[str, Any]
    explanation: Dict[str, Any]
    audit: Dict[str, Any]
    hypotheses: Dict[str, Any]
    blueprint: Dict[str, Any]
    published: bool


SYSTEM_EXPLAIN = (
    "You are a nanomedicine domain expert. Only use the EVIDENCE PACK provided. "
    "A RESEARCH DIRECTION CUE may be provided as steering context, but it is not evidence. "
    "Never invent facts or cite outside sources. If evidence is insufficient for any claim, say 'unknown'. "
    "Cite by paper_id for every claim/axis. Output strictly valid JSON matching the schema."
)

SYSTEM_AUDIT = (
    "You are a strict scientific auditor. You receive an explanation and an evidence pack. "
    "A RESEARCH DIRECTION CUE may also be provided. The cue is not evidence, but the output should be checked for alignment with it. "
    "Your job is to identify unsupported claims, missing facets, and propose retrieval queries to patch gaps. "
    "Be conservative. Output strictly valid JSON matching the schema."
)

SYSTEM_IDEATE = (
    "You propose testable nanomedicine hypotheses. Use only the evidence pack. "
    "A RESEARCH DIRECTION CUE may be provided to steer direction, but it is not evidence and must not be cited. "
    "Every hypothesis must be testable in 6-12 months by an academic lab. "
    "Cite by paper_id. Output strictly valid JSON matching the schema."
)

SYSTEM_BLUEPRINT = (
    "You produce a concise but complete preclinical blueprint for one hypothesis. "
    "A RESEARCH DIRECTION CUE may be provided to constrain direction, but it is not evidence. "
    "Use only the evidence pack and cite by paper_id. Mark anything not supported as 'assumption'. "
    "Output strictly valid JSON matching the schema."
)


def format_pack_jsonl(papers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for p in papers:
        lines.append(
            json.dumps(
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title", ""),
                    "year": p.get("year", p.get("publication_year", -1)),
                    "doi": p.get("doi", ""),
                    "cluster_id": p.get("cluster_id", None),
                    "text": (p.get("abstract", "") or p.get("processed_content", ""))[:2000],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _target_payload(state: OrchestratorState) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "target_type": state["target_type"],
        "exemplars": state.get("exemplars", 25),
        "boundary": state.get("boundary", 25),
        "diverse": state.get("diverse", 25),
        "counter_queries": [],
    }
    if state.get("snapshot_id"):
        payload["snapshot_id"] = state["snapshot_id"]
    if state["target_type"] == "gap":
        payload["gap_id"] = state["gap_id"]
    else:
        payload["cluster_a"] = state["cluster_a"]
        payload["cluster_b"] = state["cluster_b"]
    discovery_cue = discovery_cue_to_dict(state.get("discovery_cue"))
    if discovery_cue is not None:
        payload["discovery_cue"] = discovery_cue
    return payload


def node_build_pack(state: OrchestratorState, backend: BackendClient) -> OrchestratorState:
    payload = _target_payload(state)
    pack = backend.evidence_pack(payload)
    state["evidence"] = pack.get("papers", [])
    state["evidence_meta"] = pack.get("meta", {})
    return state


def node_explain(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    if state["target_type"] == "gap":
        ctx = {"gap_id": state["gap_id"], "snapshot_id": state.get("snapshot_id")}
        user = f"""
TASK: Explain what lies on either side of this gap region and propose bridge seeds.
CONTEXT: {json.dumps(ctx)}
{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    else:
        ctx = {
            "cluster_a": state["cluster_a"],
            "cluster_b": state["cluster_b"],
            "snapshot_id": state.get("snapshot_id"),
        }
        user = f"""
TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
CONTEXT: {json.dumps(ctx)}
{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    structured = llm.with_structured_output(ContrastiveExplanation, method="function_calling")
    out = structured.invoke(
        [
            {"role": "system", "content": SYSTEM_EXPLAIN},
            {"role": "user", "content": user},
        ]
    )
    state["explanation"] = out.model_dump()
    return state


def node_audit(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    expl = json.dumps(state.get("explanation", {}), ensure_ascii=False)
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    user = f"""
INPUT EXPLANATION JSON:
{expl}

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Audit the explanation: identify unsupported claims, missing facets, cue violations, and propose patch retrieval queries.
Return JSON per schema.
"""
    structured = llm.with_structured_output(AuditReport, method="function_calling")
    out = structured.invoke(
        [
            {"role": "system", "content": SYSTEM_AUDIT},
            {"role": "user", "content": user},
        ]
    )
    state["audit"] = out.model_dump()
    return state


def node_patch_retrieve(state: OrchestratorState, backend: BackendClient) -> OrchestratorState:
    audit = state.get("audit", {})
    queries = (audit.get("patch_queries") or [])[:8]
    payload = _target_payload(state)
    payload["exemplars"] = max(10, state.get("exemplars", 25) // 2)
    payload["boundary"] = max(10, state.get("boundary", 25) // 2)
    payload["diverse"] = max(10, state.get("diverse", 25) // 2)
    payload["counter_queries"] = queries

    papers = backend.evidence_pack(payload).get("papers", [])
    seen = {p.get("paper_id") for p in state.get("evidence", [])}
    merged = list(state.get("evidence", []))
    for p in papers:
        if p.get("paper_id") not in seen:
            merged.append(p)
            seen.add(p.get("paper_id"))
    state["evidence"] = merged
    state["iter"] = state.get("iter", 0) + 1
    return state


def node_ideate(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    expl = json.dumps(state.get("explanation", {}), ensure_ascii=False)
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    user = f"""
GOAL: Propose 5 bridge hypotheses grounded in the evidence pack and the contrastive explanation.
EXPLANATION JSON:
{expl}

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    structured = llm.with_structured_output(HypothesesOut, method="function_calling")
    out = structured.invoke(
        [
            {"role": "system", "content": SYSTEM_IDEATE},
            {"role": "user", "content": user},
        ]
    )
    state["hypotheses"] = out.model_dump()
    return state


def node_blueprint(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    hyps = state.get("hypotheses", {}).get("hypotheses", [])
    if not hyps:
        state["blueprint"] = {}
        return state
    h1 = hyps[0]
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    user = f"""
HYPOTHESIS JSON:
{json.dumps(h1, ensure_ascii=False)}

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    structured = llm.with_structured_output(BlueprintOut, method="function_calling")
    out = structured.invoke(
        [
            {"role": "system", "content": SYSTEM_BLUEPRINT},
            {"role": "user", "content": user},
        ]
    )
    state["blueprint"] = out.model_dump()
    return state


def node_publish(state: OrchestratorState, backend: BackendClient) -> OrchestratorState:
    target: Dict[str, Any] = {"target_type": state["target_type"]}
    if state.get("snapshot_id"):
        target["snapshot_id"] = state["snapshot_id"]
    if state["target_type"] == "gap":
        target["gap_id"] = state["gap_id"]
    else:
        target["cluster_a"] = state["cluster_a"]
        target["cluster_b"] = state["cluster_b"]

    payload = {
        "evidence_size": len(state.get("evidence", [])),
        "discovery_cue": discovery_cue_to_dict(state.get("discovery_cue")),
        "evidence_meta": state.get("evidence_meta", {}),
        "explanation": state.get("explanation", {}),
        "audit": state.get("audit", {}),
        "hypotheses": state.get("hypotheses", {}),
        "blueprint": state.get("blueprint", {}),
        "iterations": state.get("iter", 0),
    }
    backend.store_artifact(kind="research_brief", target=target, payload=payload)
    state["published"] = True
    return state


def route_after_audit(state: OrchestratorState) -> str:
    audit = state.get("audit", {})
    needs_patch = bool(audit.get("needs_patch", False))
    cue_alignment = audit.get("cue_alignment_score")
    it = state.get("iter", 0)
    max_iters = state.get("max_iters", 2)
    if (needs_patch or (cue_alignment is not None and float(cue_alignment) < 0.35)) and it < max_iters:
        return "patch"
    return "ideate"


def build_orchestrator(
    backend: BackendClient,
    *,
    openai_api_key: str | None = None,
    model_name: str | None = None,
) -> Any:
    if ChatOpenAI is None or StateGraph is None or END is None:
        raise ImportError(
            "langchain-openai and langgraph are required to build the orchestrator. "
            "Install them in the active environment first."
        )

    model = model_name or os.getenv("OPENAI_MODEL", "gpt-5")
    llm_kwargs: Dict[str, Any] = {"model": model, "temperature": 0.2}
    if openai_api_key:
        llm_kwargs["api_key"] = openai_api_key
    llm = ChatOpenAI(**llm_kwargs)
    # type: ignore[arg-type]
    g = StateGraph(OrchestratorState)

    g.add_node("build_pack", lambda s: node_build_pack(s, backend))
    g.add_node("explain", lambda s: node_explain(s, llm))
    g.add_node("audit", lambda s: node_audit(s, llm))
    g.add_node("patch_retrieve", lambda s: node_patch_retrieve(s, backend))
    g.add_node("ideate", lambda s: node_ideate(s, llm))
    g.add_node("blueprint", lambda s: node_blueprint(s, llm))
    g.add_node("publish", lambda s: node_publish(s, backend))

    g.set_entry_point("build_pack")
    g.add_edge("build_pack", "explain")
    g.add_edge("explain", "audit")
    g.add_conditional_edges("audit", route_after_audit, {"patch": "patch_retrieve", "ideate": "ideate"})
    g.add_edge("patch_retrieve", "explain")
    g.add_edge("ideate", "blueprint")
    g.add_edge("blueprint", "publish")
    g.add_edge("publish", END)

    return g.compile()


__all__ = [
    "BackendClient",
    "OrchestratorState",
    "build_orchestrator",
    "route_after_audit",
]
