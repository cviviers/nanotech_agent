from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "agentic_orchestration_framework.drawio"


class PageBuilder:
    def __init__(self, diagram: ET.Element, width: int, height: int) -> None:
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1600",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(width),
                "pageHeight": str(height),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.next_id = 2

    def _id(self) -> str:
        value = str(self.next_id)
        self.next_id += 1
        return value

    def vertex(self, value: str, x: float, y: float, w: float, h: float, style: str, parent: str = "1") -> str:
        cell_id = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "width": f"{w:.1f}",
                "height": f"{h:.1f}",
                "as": "geometry",
            },
        )
        return cell_id

    def edge(
        self,
        style: str,
        source: str | None = None,
        target: str | None = None,
        value: str = "",
        points: list[tuple[float, float]] | None = None,
        source_point: tuple[float, float] | None = None,
        target_point: tuple[float, float] | None = None,
        parent: str = "1",
    ) -> str:
        cell_id = self._id()
        attrs = {
            "id": cell_id,
            "value": value,
            "style": style,
            "edge": "1",
            "parent": parent,
        }
        if source:
            attrs["source"] = source
        if target:
            attrs["target"] = target
        cell = ET.SubElement(self.root, "mxCell", attrs)
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if source_point:
            ET.SubElement(geom, "mxPoint", {"x": f"{source_point[0]:.1f}", "y": f"{source_point[1]:.1f}", "as": "sourcePoint"})
        if target_point:
            ET.SubElement(geom, "mxPoint", {"x": f"{target_point[0]:.1f}", "y": f"{target_point[1]:.1f}", "as": "targetPoint"})
        if points:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(arr, "mxPoint", {"x": f"{x:.1f}", "y": f"{y:.1f}"})
        return cell_id


class DrawioDocument:
    def __init__(self) -> None:
        modified = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": modified,
                "agent": "Codex",
                "version": "24.7.17",
                "type": "device",
                "compressed": "false",
            },
        )

    def add_page(self, name: str, width: int, height: int) -> PageBuilder:
        diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {
                "id": uuid.uuid4().hex[:12],
                "name": name,
            },
        )
        return PageBuilder(diagram, width, height)

    def save(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)


def title_style(size: int = 28, align: str = "center") -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=hidden;"
        f"fontFamily=Helvetica;fontSize={size};fontStyle=1;align={align};verticalAlign=middle;"
    )


def body_style(fill: str, stroke: str, font_size: int = 13, align: str = "left", rounded: bool = True, extra: str = "") -> str:
    rounded_flag = "1" if rounded else "0"
    return (
        f"rounded={rounded_flag};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontFamily=Helvetica;fontSize={font_size};align={align};verticalAlign=top;spacing=8;spacingTop=8;arcSize=14;"
        + extra
    )


def lane_style(fill: str, stroke: str) -> str:
    return f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;arcSize=20;"


def text_style(font_size: int = 13, bold: bool = False, align: str = "left", color: str = "#1D1D1D") -> str:
    style = (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=hidden;"
        f"fontFamily=Helvetica;fontSize={font_size};fontColor={color};align={align};verticalAlign=middle;"
    )
    if bold:
        style += "fontStyle=1;"
    return style


def edge_style(color: str = "#2E3440", dashed: bool = False, width: int = 2, end_arrow: str = "block", extra: str = "") -> str:
    dashed_flag = "1" if dashed else "0"
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth={width};dashed={dashed_flag};endArrow={end_arrow};endFill=1;"
        "fontFamily=Helvetica;fontSize=12;labelBackgroundColor=#FFFFFF;"
        + extra
    )


def diamond_style(fill: str, stroke: str) -> str:
    return f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontFamily=Helvetica;fontSize=13;fontStyle=1;"


def header_band(page: PageBuilder, width: int, title: str, subtitle: str) -> None:
    page.vertex("", 0, 0, width, 78, body_style("#E8F0F7", "none", rounded=False))
    page.vertex(title, 28, 18, width - 56, 30, title_style(26))
    page.vertex(subtitle, 36, 48, width - 72, 18, text_style(12, align="center", color="#4A5568"))


def lane(page: PageBuilder, x: float, y: float, w: float, h: float, fill: str, stroke: str, title: str) -> None:
    page.vertex("", x, y, w, h, lane_style(fill, stroke))
    page.vertex(title, x + 20, y + 12, w - 40, 24, title_style(18, align="left"))


def card(title: str, lines: list[str]) -> str:
    content = [f"<b>{title}</b>"]
    content.extend(lines)
    return "<br>".join(content)


def add_card(page: PageBuilder, x: float, y: float, w: float, h: float, title: str, lines: list[str], fill: str, stroke: str, font_size: int = 13) -> str:
    return page.vertex(card(title, lines), x, y, w, h, body_style(fill, stroke, font_size=font_size))


def build_architecture_page(doc: DrawioDocument) -> None:
    page = doc.add_page("System Architecture", 2250, 1450)
    header_band(
        page,
        2250,
        "Agentic Novelty Framework: System Architecture and Capability Map",
        "Shared backend + LangGraph orchestration + retrospective evaluation reuse",
    )

    lane(page, 30, 105, 430, 1285, "#E7F1FA", "#6E9EC4", "Entry Points and Operator Surfaces")
    lane(page, 485, 105, 785, 1285, "#EAF6EE", "#73A67E", "Core Agent Services")
    lane(page, 1295, 105, 925, 1285, "#F7F0E7", "#C49A63", "Persistence, Data Products, and Evaluation Reuse")

    streamlit = add_card(
        page,
        55,
        165,
        380,
        150,
        "Streamlit app / agent console",
        [
            "- Publishes analyzed snapshots to the backend",
            "- Lets users inspect gaps, clusters, and paper batches",
            "- Can launch agent generation against a chosen target",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    cli = add_card(
        page,
        55,
        340,
        380,
        175,
        "Interactive CLI (`run_interactive.py`)",
        [
            "- Lists recent snapshots from the backend",
            "- Offers target selection: top gap or manual cluster pair",
            "- Accepts optional discovery cue and invokes the orchestrator",
            "- Prints summary fields from the published brief",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    retro = add_card(
        page,
        55,
        540,
        380,
        210,
        "Retrospective CLI (`run_retrospective.py`)",
        [
            "- Builds or reuses a historical snapshot",
            "- Sweeps gap + cluster-pair targets across multiple methods and seeds",
            "- Reuses the same backend/orchestrator stack headlessly",
            "- Exports review packets for expert adjudication",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    tools = add_card(
        page,
        55,
        775,
        380,
        200,
        "Structured backend tools (`tools_backend.py`)",
        [
            "- `get_top_gap_candidates`",
            "- `list_clusters`",
            "- `build_evidence_pack`",
            "- `fetch_papers_batch`",
            "- `store_artifact`",
            "- Provides a LangChain tool surface over the backend client",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    client = add_card(
        page,
        55,
        1000,
        380,
        165,
        "Backend client",
        [
            "- Shared HTTP boundary used by app, CLI, generators, and evaluation",
            "- Keeps orchestration logic decoupled from persistence details",
        ],
        "#F7FBFF",
        "#6E9EC4",
    )

    cue = add_card(
        page,
        525,
        165,
        310,
        210,
        "DiscoveryCue steering contract",
        [
            "- Text, goal, include / avoid terms",
            "- Preferred fields, hard constraints, soft constraints",
            "- Counter-queries and parsed fingerprint",
            "- Important rule: cue steers retrieval / prompting but is not evidence",
        ],
        "#F4E9FA",
        "#9D70B1",
    )
    backend_api = add_card(
        page,
        865,
        165,
        370,
        250,
        "FastAPI backend (`backend_api.py`)",
        [
            "- `GET /snapshots`, `POST /admin/snapshots/publish`",
            "- `GET /gaps/top`, `GET /clusters`, `POST /papers/batch`",
            "- `POST /evidence/pack`, `POST /artifacts/store`, `GET /artifacts`",
            "- `POST /evaluations/runs`, `POST /evaluations/matches/batch`",
            "- Treats the knowledge store as the source of truth",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    orchestrator = add_card(
        page,
        525,
        425,
        710,
        240,
        "LangGraph orchestrator (`orchestrator_langgraph.py`)",
        [
            "- Canonical flow: build_pack -> explain -> audit -> patch_retrieve? -> ideate -> blueprint -> publish",
            "- Uses structured outputs for contrastive explanation, audit report, hypotheses, and blueprint",
            "- Maintains state for target descriptor, evidence, iterations, cue, and publish status",
            "- Only evidence-pack papers may be cited; unsupported claims must be surfaced as unknown / missing",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    evidence = add_card(
        page,
        525,
        695,
        345,
        285,
        "Evidence-pack retrieval primitive",
        [
            "- Gap target: gap boundary papers + touched-cluster exemplars + diverse filler",
            "- Cluster pair: A/B exemplars + cluster-boundary papers + gap bridge papers + diverse filler",
            "- Patch queries and cue-derived queries can add more papers",
            "- Each paper carries selection provenance (`selection_sources`) and optional cue alignment metadata",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    llm = add_card(
        page,
        890,
        695,
        345,
        285,
        "LLM call surfaces",
        [
            "- `SYSTEM_EXPLAIN`: explain separation / bridge seeds",
            "- `SYSTEM_AUDIT`: identify unsupported claims and patch queries",
            "- `SYSTEM_IDEATE`: generate 5 grounded hypotheses",
            "- `SYSTEM_BLUEPRINT`: produce a preclinical blueprint for the top hypothesis",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    services = add_card(
        page,
        525,
        1010,
        710,
        200,
        "External model services",
        [
            "- OpenAI chat model via `ChatOpenAI` for explain / audit / ideate / blueprint",
            "- Local Qwen embedding + reranker service used by retrospective candidate matching",
            "- Model choice and API key are injected via environment or CLI parameters",
        ],
        "#F9FCF7",
        "#73A67E",
    )

    snapshot = add_card(
        page,
        1335,
        165,
        365,
        235,
        "Snapshot ingestion / publication",
        [
            "- `snapshot_builder.py` converts analyzed dataframe + embeddings into a reusable snapshot payload",
            "- Published snapshots contain papers, clusters, gaps, gap-paper membership, and metadata",
            "- The same publish path is used by the app and retrospective runs",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    store = add_card(
        page,
        1720,
        165,
        470,
        350,
        "SQLite knowledge store (`knowledge_store.py`)",
        [
            "- Tables: snapshots, papers, clusters, gaps, gap_papers, llm_analyses",
            "- Tables: artifacts, evaluation_runs, evaluation_matches",
            "- `build_evidence_pack()` is the main retrieval workhorse",
            "- Cue scoring / reranking is applied server-side and stored as auditable selection metadata",
            "- Stores generated research briefs and evaluation records",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    artifact = add_card(
        page,
        1335,
        435,
        365,
        230,
        "Research brief artifact",
        [
            "- Stored via `store_artifact(kind='research_brief')`",
            "- Payload includes evidence size / meta, cue, explanation, audit, hypotheses, blueprint, iterations",
            "- Persists the full grounded reasoning trace, not just final text",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    evaluation = add_card(
        page,
        1335,
        690,
        855,
        280,
        "Retrospective evaluation reuse",
        [
            "- Historical snapshot -> target selection -> generation methods -> matching against future literature",
            "- Orchestrator is one method among several baselines: single-shot LLM, retrieval-summary direct, cluster-only, random control, heuristic fallback",
            "- Matches and runs are persisted through the same backend / store",
            "- Review packets (`csv` / `json`) are exported for manual inspection",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    outputs = add_card(
        page,
        1335,
        995,
        405,
        220,
        "Evaluation outputs",
        [
            "- `EvaluationRun` summary + metrics",
            "- `EvaluationMatch` records per hypothesis",
            "- Review packets with titles, citations, best historical/future matches, and first future year",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    metrics = add_card(
        page,
        1760,
        995,
        430,
        220,
        "Metrics / manual review",
        [
            "- Leakage, anticipatory strong / partial, unsupported, unrealized",
            "- Novelty-adjusted hit rate and time-to-match summaries",
            "- Manual adjudication remains necessary for ambiguous cases",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    legend = add_card(
        page,
        1335,
        1240,
        855,
        115,
        "Legend",
        [
            "- Solid arrows: runtime control / request flow",
            "- Dashed arrows: persisted artifacts or cross-run reuse",
            "- Purple edges: discovery cue steering path",
        ],
        "#FFFDF8",
        "#C49A63",
        font_size=12,
    )

    for source in [streamlit, cli, retro, tools, client]:
        page.edge(edge_style(), source=source, target=backend_api)
    page.edge(edge_style("#9D70B1", dashed=True), source=cue, target=evidence, value="steers retrieval")
    page.edge(edge_style("#9D70B1", dashed=True), source=cue, target=orchestrator, value="steers prompts")
    page.edge(edge_style(), source=backend_api, target=orchestrator, value="launch / invoke")
    page.edge(edge_style(), source=backend_api, target=evidence, value="retrieve pack")
    page.edge(edge_style(), source=evidence, target=orchestrator, value="grounding")
    page.edge(edge_style(), source=llm, target=orchestrator, value="structured model calls")
    page.edge(edge_style(dashed=True), source=orchestrator, target=artifact, value="persist brief")
    page.edge(edge_style(dashed=True), source=backend_api, target=store, value="CRUD + retrieval")
    page.edge(edge_style(dashed=True), source=snapshot, target=store, value="publish snapshot")
    page.edge(edge_style(dashed=True), source=artifact, target=store, value="artifacts table")
    page.edge(edge_style(dashed=True), source=evaluation, target=store, value="evaluation runs / matches")
    page.edge(edge_style(), source=retro, target=evaluation, value="benchmark driver")
    page.edge(edge_style(), source=services, target=evaluation, value="matching dependencies")
    page.edge(edge_style(dashed=True), source=outputs, target=metrics, value="aggregate + inspect")
    page.edge(edge_style(dashed=True), source=store, target=outputs, value="records")


def build_orchestrator_page(doc: DrawioDocument) -> None:
    page = doc.add_page("Orchestrator Flow", 3120, 1380)
    header_band(
        page,
        3120,
        "LangGraph Orchestrator: Detailed Agent Flow",
        "Grounded novelty generation with audit loop, cue steering, and artifact publication",
    )

    backend_box = add_card(
        page,
        60,
        110,
        260,
        115,
        "Backend dependency",
        [
            "- `BackendClient.evidence_pack()`",
            "- `BackendClient.store_artifact()`",
        ],
        "#F7FBFF",
        "#6E9EC4",
        font_size=12,
    )
    llm_box = add_card(
        page,
        910,
        110,
        520,
        115,
        "OpenAI chat model dependency",
        [
            "- `ChatOpenAI(model, temperature=0.2)`",
            "- Used with function-calling structured outputs for four generation nodes",
        ],
        "#F9FCF7",
        "#73A67E",
        font_size=12,
    )

    input_state = add_card(
        page,
        60,
        275,
        245,
        250,
        "Input state / target descriptor",
        [
            "- `target_type`: `gap` or `cluster_pair`",
            "- `snapshot_id`",
            "- `gap_id` or `cluster_a` / `cluster_b`",
            "- `exemplars`, `boundary`, `diverse`",
            "- `iter`, `max_iters`",
            "- optional `discovery_cue`",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    build_pack = add_card(
        page,
        350,
        275,
        265,
        250,
        "1. `node_build_pack`",
        [
            "- Calls `_target_payload(state)`",
            "- Requests an evidence pack from the backend",
            "- Stores `evidence` and `evidence_meta` in state",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    explain = add_card(
        page,
        670,
        275,
        265,
        250,
        "2. `node_explain`",
        [
            "- Formats evidence as JSONL (`format_pack_jsonl`)",
            "- Builds cue prompt block and target-specific user prompt",
            "- Returns `ContrastiveExplanation`",
            "- Stores `explanation` in state",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    audit = add_card(
        page,
        990,
        275,
        265,
        250,
        "3. `node_audit`",
        [
            "- Audits the explanation against the same evidence pack",
            "- Flags unsupported claims and missing facets",
            "- Emits patch retrieval queries and cue alignment score",
            "- Stores `audit` in state",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    decision = page.vertex(
        "Patch evidence?<br><font style='font-size:11px'>`needs_patch` or cue alignment &lt; 0.35<br>and `iter &lt; max_iters`</font>",
        1325,
        315,
        180,
        170,
        diamond_style("#FFF3D8", "#C49A63"),
    )
    patch = add_card(
        page,
        1570,
        275,
        265,
        250,
        "4. `node_patch_retrieve`",
        [
            "- Uses up to 8 audit-proposed patch queries",
            "- Re-requests an evidence pack with smaller budgets",
            "- Merges unseen papers into `evidence`",
            "- Increments `iter` and loops back to explain",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    ideate = add_card(
        page,
        1900,
        275,
        265,
        250,
        "5. `node_ideate`",
        [
            "- Consumes evidence + explanation + cue block",
            "- Generates `HypothesesOut`",
            "- Goal in prompt: 5 bridge hypotheses grounded in evidence",
            "- Stores `hypotheses` in state",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    blueprint = add_card(
        page,
        2230,
        275,
        265,
        250,
        "6. `node_blueprint`",
        [
            "- Selects the top hypothesis if any exist",
            "- Produces `BlueprintOut` for one hypothesis",
            "- Marks unsupported items as assumptions",
            "- Stores `blueprint` in state",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    publish = add_card(
        page,
        2560,
        275,
        250,
        250,
        "7. `node_publish`",
        [
            "- Builds artifact target payload",
            "- Persists a `research_brief` artifact",
            "- Sets `published = True`",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    artifact = add_card(
        page,
        2865,
        300,
        195,
        200,
        "Stored artifact",
        [
            "- research_brief",
            "- full reasoning trace",
            "- reusable for inspection",
        ],
        "#FFFDF8",
        "#C49A63",
        font_size=12,
    )

    state_box = add_card(
        page,
        60,
        610,
        255,
        520,
        "State mutations across the graph",
        [
            "- After build_pack: `evidence`, `evidence_meta`",
            "- After explain: `explanation`",
            "- After audit: `audit`",
            "- After patch: merged `evidence`, incremented `iter`",
            "- After ideate: `hypotheses`",
            "- After blueprint: `blueprint`",
            "- After publish: `published = True`",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    pack_box = add_card(
        page,
        350,
        610,
        560,
        520,
        "Evidence-pack composition and retrieval behavior",
        [
            "- Target payload always includes target descriptor + budgets + optional snapshot id + optional cue",
            "- Gap target: boundary papers from the gap, plus exemplars from touched clusters",
            "- Cluster pair: exemplars from A and B, cluster-boundary papers, then gap bridge papers if available",
            "- `diverse` budget adds background / filler papers not already selected",
            "- Discovery cue may add cue-derived retrieval queries and rerank selected papers by cue score",
            "- Patch retrieval reuses the same target descriptor but injects `counter_queries` from the audit",
            "- Retrieved papers are auditable via `selection_sources` and `selection_meta.cue_alignment`",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    prompts = add_card(
        page,
        950,
        610,
        400,
        300,
        "Prompt guardrails",
        [
            "- Cue can constrain direction, but it is never valid evidence and must not be cited",
            "- Explanations must cite `paper_id` for every claim / axis",
            "- Auditor is instructed to be conservative and propose retrieval patches",
            "- Ideation must stay testable in a 6-12 month academic-lab horizon",
            "- Blueprint must mark unsupported specifics as assumptions",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    route_box = add_card(
        page,
        1385,
        610,
        470,
        300,
        "Routing logic (`route_after_audit`)",
        [
            "- `needs_patch = audit.needs_patch`",
            "- `cue_alignment = audit.cue_alignment_score`",
            "- If `needs_patch` OR cue alignment exists and is `< 0.35`, and `iter < max_iters`, route to `patch_retrieve`",
            "- Otherwise route directly to `ideate`",
            "- This makes cue misalignment a first-class reason to continue retrieval",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    schemas = add_card(
        page,
        1900,
        610,
        580,
        520,
        "Structured outputs / schemas",
        [
            "- `ContrastiveExplanation`",
            "  cluster summaries, axes of separation, bridge seeds, insufficient-evidence flag",
            "- `AuditReport`",
            "  supported claim fraction, patch queries, cue violations, missing facets, hard-constraint status",
            "- `HypothesesOut`",
            "  hypothesis id, title, bridge type, rationale, novel elements, risks, unknowns, citations",
            "- `BlueprintOut`",
            "  bill of materials, synthesis / characterization, in vitro, in vivo, risks, success criteria, citations",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    publish_box = add_card(
        page,
        2510,
        610,
        550,
        520,
        "Published research brief payload",
        [
            "- `evidence_size` and `evidence_meta`",
            "- normalized `discovery_cue` payload",
            "- `explanation`, `audit`, `hypotheses`, `blueprint`",
            "- `iterations` count from the retrieval / audit loop",
            "- target metadata includes gap id or cluster pair plus snapshot id",
            "- downstream consumers can inspect the artifact without rerunning the graph",
        ],
        "#FFFFFF",
        "#C49A63",
    )

    page.edge(edge_style(), source=backend_box, target=build_pack, value="backend client")
    page.edge(edge_style(), source=input_state, target=build_pack, value="entry state")
    page.edge(edge_style(), source=build_pack, target=explain, value="evidence pack")
    page.edge(edge_style(), source=llm_box, target=explain, value="structured output")
    page.edge(edge_style(), source=explain, target=audit, value="contrastive explanation")
    page.edge(edge_style(), source=llm_box, target=audit, value="structured output")
    page.edge(edge_style(), source=audit, target=decision, value="audit report")
    page.edge(edge_style(), source=decision, target=patch, value="yes")
    page.edge(edge_style(), source=decision, target=ideate, value="no")
    page.edge(
        edge_style("#C49A63"),
        source=patch,
        target=explain,
        value="loop with merged evidence",
        points=[(1702, 230), (1702, 220), (802, 220)],
    )
    page.edge(edge_style(), source=llm_box, target=ideate, value="structured output")
    page.edge(edge_style(), source=ideate, target=blueprint, value="hypotheses")
    page.edge(edge_style(), source=llm_box, target=blueprint, value="structured output")
    page.edge(edge_style(), source=blueprint, target=publish, value="top hypothesis + blueprint")
    page.edge(edge_style(dashed=True), source=publish, target=artifact, value="store_artifact")
    page.edge(edge_style(dashed=True), source=publish_box, target=artifact, value="payload")


def build_evaluation_page(doc: DrawioDocument) -> None:
    page = doc.add_page("Retrospective Evaluation", 3000, 1620)
    header_band(
        page,
        3000,
        "Retrospective Evaluation Harness",
        "Historical-snapshot generation, multi-method hypothesis sweep, and future-literature matching",
    )

    inputs = add_card(
        page,
        50,
        150,
        300,
        230,
        "Inputs and required services",
        [
            "- `cleaned_dataset.json`",
            "- precomputed Qwen and BERT embeddings",
            "- running backend API (`127.0.0.1:8088`)",
            "- local Qwen embedding + reranker service (`127.0.0.1:8000`)",
            "- optional discovery cue + OpenAI model settings",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    split = add_card(
        page,
        395,
        150,
        310,
        230,
        "1. Time split",
        [
            "- Load full corpus + embeddings",
            "- Split into `historical`, `future`, and optional `sensitivity_future`",
            "- Cutoff and window dates are explicit CLI parameters",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    snapshot = add_card(
        page,
        750,
        120,
        420,
        290,
        "2. Historical snapshot construction",
        [
            "- If `existing_snapshot_id` is supplied, reuse it",
            "- Else run `analysis_v1` on historical data",
            "- Build snapshot payload with gaps, clusters, embeddings, metadata overrides",
            "- Publish snapshot to the same backend used by the live agent stack",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    targets = add_card(
        page,
        1215,
        120,
        425,
        290,
        "3. Target selection and cue reranking",
        [
            "- Gap targets from `backend.top_gaps()`",
            "- Cluster-pair targets from shared gaps first, then cluster combinations",
            "- If a discovery cue exists, rank targets by cue-target score using small evidence packs",
            "- Result: unified list of gap + cluster-pair targets",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    generation = add_card(
        page,
        1685,
        120,
        430,
        290,
        "4. Generation sweep",
        [
            "- Nested loop over `methods x seeds x targets`",
            "- Builds a `GenerationContext` for each run",
            "- Orchestrator is one method among several baselines / controls",
            "- Failures are captured and logged without aborting the full benchmark",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    matching = add_card(
        page,
        2160,
        120,
        465,
        290,
        "5. Candidate retrieval + judging",
        [
            "- Normalize each generated idea into a fingerprint",
            "- Retrieve historical and future candidates",
            "- Choose best matches and classify the hypothesis outcome",
            "- Also compute `first_future_year` and optional sensitivity-window best match",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    persist = add_card(
        page,
        2670,
        120,
        280,
        290,
        "6. Persist + export",
        [
            "- Store evaluation matches in batches",
            "- Aggregate run metrics",
            "- Store evaluation run record",
            "- Export CSV / JSON review packets",
        ],
        "#FFFFFF",
        "#C49A63",
    )

    methods = add_card(
        page,
        1685,
        450,
        430,
        310,
        "Generation methods (`generators.py`)",
        [
            "- `orchestrator`: full evidence -> explain -> audit -> ideate -> blueprint path",
            "- `single_shot_llm`: one structured ideation call over the evidence pack",
            "- `retrieval_summary_direct`: summary first, then ideation",
            "- `cluster_only`: disables boundary evidence; falls back to heuristic on failure",
            "- `random_cluster_pair_control`: random cluster pair; also has heuristic fallback",
            "- `heuristic_bridge`: deterministic debug / fallback generator",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    cue_box = add_card(
        page,
        1215,
        450,
        425,
        195,
        "Discovery cue behavior inside evaluation",
        [
            "- Can rerank targets before generation",
            "- Flows into evidence-pack retrieval and generator prompts",
            "- Is stored in hypotheses, runs, matches, and review packets for reproducibility",
        ],
        "#F4E9FA",
        "#9D70B1",
    )
    indexes = add_card(
        page,
        750,
        450,
        420,
        220,
        "Corpus indexes",
        [
            "- Build Qwen-backed indexes for historical and future corpora",
            "- Optional sensitivity index for a broader future window",
            "- These indexes are used downstream by the candidate matcher",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    retrieval = add_card(
        page,
        2160,
        450,
        465,
        330,
        "Candidate retrieval signals (`candidate_match.py`)",
        [
            "- Keyword retrieval from fingerprint terms",
            "- Query embedding against precomputed paper embeddings",
            "- Qwen reranking over retrieved candidates",
            "- If Qwen errors or times out, the matcher degrades gracefully instead of aborting the benchmark",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    loops = add_card(
        page,
        1685,
        800,
        430,
        170,
        "Loop cardinality",
        [
            "- `n_gap_targets`, `n_cluster_pair_targets`",
            "- `seeds`",
            "- `hypotheses_per_target`",
            "- Each method / seed / target can emit multiple hypotheses",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    classes = add_card(
        page,
        2160,
        820,
        465,
        300,
        "Judging, labels, and metrics",
        [
            "- Match labels: `strong_match`, `partial_match`, `background_only`, `no_match`",
            "- Final classifications: `already_present`, `anticipatory_strong`, `anticipatory_partial`, `unsupported`, `unrealized`",
            "- Metrics include leakage rate, anticipatory rates, unsupported rate, novelty-adjusted hit rate, and median time to future match",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    packets = add_card(
        page,
        2670,
        450,
        280,
        310,
        "Review packets",
        [
            "- `<run_id>_review_packet.csv`",
            "- `<run_id>_review_packet.json`",
            "- Include title, text, citations, best historical / future matches, and first future year",
            "- Intended for expert manual review",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    outputs = add_card(
        page,
        1215,
        690,
        425,
        265,
        "Run payload / outputs",
        [
            "- `EvaluationRun` with config, summary, metrics, status, and discovery cue",
            "- `EvaluationMatch` per hypothesis with fingerprint and best-match payloads",
            "- Summary JSON printed by the CLI includes run metadata plus review packet paths",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    notes = add_card(
        page,
        50,
        450,
        655,
        220,
        "Practical constraints surfaced by the code",
        [
            "- Benchmark assumes publication dates can be derived from year / month / day fields",
            "- Full runs are expensive in both runtime and API cost, so smoke runs are recommended first",
            "- Manual review is still required before publication-grade claims",
        ],
        "#FFFDF8",
        "#6E9EC4",
    )

    page.edge(edge_style(), source=inputs, target=split)
    page.edge(edge_style(), source=split, target=snapshot)
    page.edge(edge_style(), source=snapshot, target=targets)
    page.edge(edge_style("#9D70B1", dashed=True), source=cue_box, target=targets, value="rerank")
    page.edge(edge_style(), source=targets, target=generation)
    page.edge(edge_style("#9D70B1", dashed=True), source=cue_box, target=methods, value="steer prompts / retrieval")
    page.edge(edge_style(), source=methods, target=generation, value="registry")
    page.edge(edge_style(), source=generation, target=matching, value="generated hypotheses")
    page.edge(edge_style(), source=indexes, target=matching, value="historical / future indexes")
    page.edge(edge_style(), source=retrieval, target=matching, value="retrieval signals")
    page.edge(edge_style(), source=matching, target=persist, value="matches + classifications")
    page.edge(edge_style(dashed=True), source=persist, target=packets, value="export")
    page.edge(edge_style(dashed=True), source=persist, target=outputs, value="store run")
    page.edge(edge_style(dashed=True), source=classes, target=outputs, value="metrics")
    page.edge(edge_style(), source=loops, target=generation, value="sweep size")


def main() -> None:
    doc = DrawioDocument()
    build_architecture_page(doc)
    build_orchestrator_page(doc)
    build_evaluation_page(doc)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
