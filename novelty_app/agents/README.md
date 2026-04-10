# Agents README

## Overview

The `novelty_app/agents` package implements the non-UI agent stack for grounded novelty discovery.

At a high level, the agents do four things:

1. expose the analyzed literature snapshot through a backend API
2. retrieve an evidence pack for a novelty target
3. run an audited multi-step generation workflow over that evidence
4. persist generated artifacts and retrospective evaluation records

This package is designed so the same backend and orchestration logic can be used from:

- the Streamlit app
- the interactive CLI runner
- the retrospective evaluation pipeline

## Main Components

### `backend_api.py`

FastAPI service exposing the agent-facing backend.

It provides endpoints for:

- listing and publishing snapshots
- listing top gap candidates
- listing clusters
- fetching paper batches
- building evidence packs
- storing generated artifacts
- storing and listing retrospective evaluation runs
- storing and listing retrospective recovery-match records

This is the main boundary between the agents and the stored novelty-analysis state.

### `knowledge_store.py`

SQLite-backed persistence and retrieval layer.

This is the backend workhorse. It stores:

- snapshots
- paper records
- clusters
- gaps and gap-paper membership
- generated artifacts
- evaluation runs
- evaluation matches

It also implements `build_evidence_pack()`, which is the main retrieval primitive used by the generators.

### `snapshot_builder.py`

Builds a reusable snapshot payload from an analyzed dataframe and embedding outputs.

This extracts the snapshot construction logic out of Streamlit-specific code so snapshots can be created from:

- the app
- scripts
- retrospective evaluation

### `schemas.py`

Shared typed contracts used across backend, orchestration, and evaluation.

Important models include:

- `SnapshotPayload`
- `AnalysisConfig`
- `EvidencePack`
- `GeneratedHypothesis`
- `EvaluationRun`
- `EvaluationMatch`
- `DiscoveryCue`

### `tools_backend.py`

Wraps the backend client as structured LangChain tools.

This is useful when the agent runtime wants tool-style access rather than raw HTTP calls. The exposed tools are:

- `get_top_gap_candidates`
- `list_clusters`
- `build_evidence_pack`
- `fetch_papers_batch`
- `store_artifact`

### `orchestrator_langgraph.py`

Implements the main multi-step novelty-discovery agent workflow using LangGraph and OpenAI chat models.

This is the canonical agent flow for grounded hypothesis generation.

### `run_interactive.py`

Simple CLI entrypoint for manually running the orchestrator against a selected snapshot and target.

This is useful for smoke testing and debugging the agent path outside Streamlit.

## What The Agents Actually Do

The implemented orchestration flow is:

`target -> evidence pack -> explanation -> audit -> patch retrieval -> ideation -> score -> blueprint -> publish`

### 1. Target Selection

The current agent stack assumes the target is already defined before generation starts.

A target is one of:

- a `gap`
- a `cluster_pair`

The target can come from:

- the app
- the interactive CLI
- the retrospective evaluation runner

### 2. Evidence Pack Construction

The backend builds an evidence pack around the target.

For a `gap` target, the evidence pack typically includes:

- papers inside the gap region boundary
- exemplar papers from clusters touched by the gap
- diverse filler/background papers

For a `cluster_pair` target, the evidence pack typically includes:

- exemplar papers from cluster A
- exemplar papers from cluster B
- boundary papers near the interface between A and B
- gap papers that connect both clusters
- diverse filler/background papers

Optional `counter_queries` can expand the pack with papers matching missing-evidence queries.

Optional `required_paper_ids` can force specific papers into the pack before the usual retrieval expansion.

Optional `discovery_cue` can further steer retrieval by:

- generating cue-derived retrieval queries
- scoring each selected paper for cue alignment
- reordering the pack toward cue-relevant papers

The cue is treated as steering context, not evidence.

### 3. Contrastive Explanation

The first LLM step explains the structure around the target.

For a gap target, the model is asked to explain what lies on either side of the gap and propose bridge seeds.

For a cluster-pair target, the model is asked to explain why the clusters are separated in embedding space.

The output is structured as:

- summaries of each side
- axes of separation
- bridge seeds
- explicit insufficient-evidence signaling

The model is instructed to use only the evidence pack and cite `paper_id`s.

### 4. Audit

The second LLM step audits the explanation.

It checks:

- unsupported claims
- missing evidence facets
- patch-retrieval queries
- cue alignment
- cue violations
- whether hard cue constraints were respected

This step decides whether another retrieval pass is needed.

### 5. Patch Retrieval

If the audit indicates unsupported claims or poor cue alignment, the backend retrieves additional papers using:

- patch queries proposed by the auditor
- the same target descriptor
- smaller evidence budgets for focused retrieval

The new papers are merged into the evidence pack and the explanation/audit loop runs again.

### 6. Hypothesis Generation

Once the evidence is considered sufficient, the agent generates bridge hypotheses.

Each hypothesis is intended to be:

- grounded in the evidence pack
- testable in a typical academic lab
- structured
- citation-backed

The output stores:

- title
- mechanism/rationale
- novel elements
- risks
- unknowns
- citations

### 7. Experimental Blueprint

Before blueprinting, the agent scores the generated ideas.

Each hypothesis is scored on a 1-5 scale for:

- importance
- novelty
- plausibility
- feasibility
- evaluability
- likely impact

When OpenAI-backed judging is available, the scorer uses a structured LLM judge. If it is unavailable, the framework falls back to a deterministic heuristic scorer so the pipeline still runs.

The blueprint stage then selects the top-scored hypothesis.

### 8. Experimental Blueprint

The agent then generates a preclinical blueprint for the selected top hypothesis.

The blueprint includes:

- bill of materials
- synthesis/characterization
- in vitro plan
- in vivo plan
- risks and mitigations
- success criteria

Unsupported items are supposed to be marked as assumptions.

### 9. Publish

The final step stores a `research_brief` artifact in the backend.

The persisted payload includes:

- evidence size
- evidence metadata
- discovery cue
- explanation
- audit output
- hypotheses
- idea scores
- blueprint
- number of retrieval/audit iterations

## Discovery Cue Behavior

The `DiscoveryCue` is a structured steering input that lets the user push the novelty search in a specific scientific direction.

It contains:

- free-text cue text
- optional goal
- include/avoid terms
- preferred fields
- hard constraints
- soft constraints
- counter-queries
- a parsed fingerprint

The cue is currently used in three places:

1. evidence-pack retrieval and reranking
2. orchestrator prompts
3. retrospective evaluation metadata and target reranking

Important rule: the cue is not evidence and should never be cited as if it were literature support.

## How Retrieval Works

The retrieval logic is implemented server-side in `KnowledgeStore.build_evidence_pack()`.

Selected papers are annotated with selection metadata such as:

- `selection_sources`
- `cue_alignment`
- `cue_score`

This makes the evidence pack auditable. The generator can inspect not only which papers were retrieved, but also why they were included.

## How The Agents Are Used In Evaluation

The retrospective evaluation pipeline in `novelty_app/evaluation` reuses this agent stack.

In that setting:

1. a historical snapshot is built from pre-cutoff papers
2. targets are selected from the historical snapshot
3. generation methods call the same backend/orchestrator logic
4. generated hypotheses are ranked against held-out future papers
5. runs and matches are stored through the same backend

This is how the project measures whether grounded ideas generated from historical evidence recover held-out future papers anchored to historical frontiers.

## Typical Execution Paths

### Backend

Run the backend:

```powershell
uvicorn agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088
```

### Interactive agent run

Run the CLI orchestrator:

```powershell
python -m novelty_app.agents.run_interactive
```

This will:

- connect to the backend
- let you choose a snapshot
- let you choose a gap or cluster pair
- optionally accept a discovery cue
- run the orchestrator

### Retrospective evaluation

The evaluation runner uses the same backend/orchestrator stack headlessly:

```powershell
python -m novelty_app.evaluation.run_retrospective --backend-url http://127.0.0.1:8088
```

## Practical Notes

- The backend is the source of truth for snapshots, evidence packs, artifacts, and evaluation records.
- The orchestrator is the main agentic novelty-discovery path.
- The evidence pack is the central grounding mechanism; the rest of the agent stack depends on its quality.
- The discovery cue changes directionality, but it does not substitute for evidence.
- Retrospective evaluation is the main route for measuring whether the agents are doing something scientifically meaningful rather than just producing plausible text.

## File Map

- `novelty_app/agents/backend_api.py`: FastAPI backend
- `novelty_app/agents/backend_client.py`: HTTP client for the backend
- `novelty_app/agents/knowledge_store.py`: SQLite store + evidence-pack retrieval
- `novelty_app/agents/orchestrator_langgraph.py`: multi-step generation workflow
- `novelty_app/agents/schemas.py`: shared contracts
- `novelty_app/agents/snapshot_builder.py`: reusable snapshot creation
- `novelty_app/agents/tools_backend.py`: structured tool wrappers
- `novelty_app/agents/run_interactive.py`: manual CLI runner
