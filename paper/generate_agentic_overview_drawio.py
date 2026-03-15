from __future__ import annotations

from pathlib import Path

from generate_agentic_framework_drawio import (
    DrawioDocument,
    add_card,
    body_style,
    edge_style,
    header_band,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "agentic_overview.drawio"


def main() -> None:
    doc = DrawioDocument()
    page = doc.add_page("Overview", 1900, 1020)

    header_band(
        page,
        1900,
        "Agentic Novelty Framework: High-Level Overview",
        "Simple overview of the same backend, orchestrator, and evaluation agents",
    )

    page.vertex("", 30, 100, 1840, 885, body_style("#F7FAFC", "#D7E2EC"))

    cue = add_card(
        page,
        120,
        130,
        320,
        120,
        "Discovery cue",
        [
            "- Optional steering input",
            "- Shapes retrieval and prompting",
            "- Never treated as evidence",
        ],
        "#F4E9FA",
        "#9D70B1",
    )
    principle = add_card(
        page,
        1470,
        130,
        300,
        120,
        "Core principle",
        [
            "- Evidence pack is the grounding mechanism",
            "- Audit loop checks support before ideation",
        ],
        "#FFF6E8",
        "#C49A63",
    )

    entry = add_card(
        page,
        70,
        320,
        270,
        280,
        "1. Entry points",
        [
            "- Streamlit app / agent console",
            "- Interactive CLI",
            "- Retrospective evaluation runner",
            "- Optional LangChain tool surface",
        ],
        "#FFFFFF",
        "#6E9EC4",
    )
    backend = add_card(
        page,
        400,
        320,
        330,
        280,
        "2. Backend and retrieval layer",
        [
            "- FastAPI backend over a SQLite knowledge store",
            "- Stores snapshots, papers, clusters, gaps, artifacts, and evaluation records",
            "- Builds evidence packs for gaps or cluster pairs",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    orchestrator = add_card(
        page,
        790,
        290,
        420,
        340,
        "3. Agent orchestrator",
        [
            "- Build evidence pack",
            "- Explain the target region / cluster separation",
            "- Audit unsupported claims",
            "- Patch retrieval if evidence is weak",
            "- Generate grounded hypotheses",
            "- Create a blueprint for the top idea",
        ],
        "#FFFFFF",
        "#73A67E",
    )
    artifact = add_card(
        page,
        1270,
        320,
        240,
        280,
        "4. Research brief output",
        [
            "- Persisted artifact",
            "- Includes evidence, explanation, audit, hypotheses, and blueprint",
            "- Reusable for inspection and follow-up",
        ],
        "#FFFFFF",
        "#C49A63",
    )
    evaluation = add_card(
        page,
        1570,
        320,
        260,
        280,
        "5. Retrospective evaluation",
        [
            "- Reuses the same backend and agents on historical snapshots",
            "- Matches generated ideas against later literature",
            "- Exports review packets and metrics",
        ],
        "#FFFFFF",
        "#C49A63",
    )

    highlights = add_card(
        page,
        120,
        700,
        1650,
        180,
        "What makes it agentic",
        [
            "- The system does not generate from raw prompts alone: it first retrieves a target-specific evidence pack from the backend.",
            "- An audit node can trigger another focused retrieval pass before hypothesis generation.",
            "- The same grounded workflow is reused interactively and in retrospective benchmarking.",
        ],
        "#FFFFFF",
        "#4A6F8F",
    )

    page.edge(edge_style("#9D70B1", dashed=True), source=cue, target=backend, value="steers retrieval")
    page.edge(edge_style("#9D70B1", dashed=True), source=cue, target=orchestrator, value="steers prompts")
    page.edge(edge_style(), source=entry, target=backend)
    page.edge(edge_style(), source=backend, target=orchestrator, value="evidence pack")
    page.edge(edge_style(), source=orchestrator, target=artifact, value="publish")
    page.edge(edge_style(), source=artifact, target=evaluation, value="review / compare")
    page.edge(edge_style(dashed=True), source=principle, target=orchestrator, value="guardrails")
    page.edge(edge_style(dashed=True), source=orchestrator, target=highlights, value="workflow properties")
    page.edge(edge_style(dashed=True), source=evaluation, target=highlights, value="shared reuse")

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
