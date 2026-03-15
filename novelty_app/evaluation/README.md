# Retrospective Evaluation

This package evaluates whether grounded ideas generated from a historical literature snapshot later appear in the literature.

The core question is:

`Given evidence only up to a cutoff date, does the system generate ideas that are absent historically and later show up in future papers?`

## What This Pipeline Does

The retrospective benchmark in [run_retrospective.py](run_retrospective.py) runs this sequence:

1. Load the master corpus and precomputed embeddings.
2. Split the corpus into:
   - `historical`: papers on or before the cutoff
   - `future`: papers in the evaluation window
   - `sensitivity_future`: optional broader future window
3. Build a historical novelty snapshot.
4. Select gap targets and cluster-pair targets from that snapshot.
5. Generate hypotheses with one or more methods.
6. Normalize each generated idea into a fingerprint.
7. Retrieve historical and future candidate papers for that idea.
8. Judge the best historical and future matches.
9. Assign a retrospective classification.
10. Persist the run and export review packets for human inspection.

## Module Layout

- [run_retrospective.py](run_retrospective.py): CLI entrypoint and full benchmark orchestration.
- [time_split.py](time_split.py): corpus loading and time-based splitting.
- [analysis_v1.py](analysis_v1.py): headless analysis used to build historical snapshots.
- [generators.py](generators.py): generation methods and baselines.
- [idea_fingerprint.py](idea_fingerprint.py): converts hypotheses into structured fields.
- [candidate_match.py](candidate_match.py): keyword, embedding, and rerank retrieval.
- [judge.py](judge.py): match labeling and hypothesis classification.
- [metrics.py](metrics.py): aggregate metrics.
- [qwen_client.py](qwen_client.py): local embedding and reranker HTTP client.

## Inputs

The benchmark expects:

- `data/cleaned_dataset.json`
- `data/qwen_embeddings.npy`
- `data/bert_embeddings.npy`
- a running backend API
- a running local Qwen embedding and reranking service

The dataset is expected to contain `publication_year`. If available, `publication_month` and `publication_day` are also used to build a more precise publication date.

## Required Services

### Backend

Run the novelty backend, typically on:

```powershell
uvicorn novelty_app.agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088
```

When calling it from the CLI, use `127.0.0.1`, not `0.0.0.0`:

```text
http://127.0.0.1:8088
```

### Qwen Embedding + Reranker Service

Run the local Qwen service:

```powershell
cd embedding_models
uvicorn qwen:app --host 0.0.0.0 --port 8000
```

The retrospective CLI should point to:

```text
http://127.0.0.1:8000
```

## Environment

For OpenAI-backed generation methods, set:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-5-mini-2025-08-07"
```

`run_retrospective.py` reads `OPENAI_API_KEY` and `OPENAI_MODEL`.

If your local `.env` uses `openai_key` instead of `OPENAI_API_KEY`, map it before running:

```powershell
$env:OPENAI_API_KEY=$env:openai_key
```

## Default Evaluation Setup

The current defaults in [run_retrospective.py](run_retrospective.py) are:

- cutoff date: `2020-12-31`
- future window: `2022-01-01` to `2025-12-31`
- sensitivity window: `2021-01-01` to `2025-12-31`
- gap targets: `20`
- cluster-pair targets: `10`
- methods:
  - `orchestrator`
  - `single_shot_llm`
  - `retrieval_summary_direct`
  - `cluster_only`
  - `random_cluster_pair_control`
- seeds: `3`
- hypotheses per target: `3`

An optional discovery cue can also be supplied to steer retrieval and generation without changing the historical evidence cutoff.

## Generation Methods

Defined in [generators.py](generators.py):

- `orchestrator`: full LangGraph evidence -> explain -> audit -> ideate -> blueprint path.
- `single_shot_llm`: one LLM call over the evidence pack.
- `retrieval_summary_direct`: retrieval summary followed by direct ideation.
- `cluster_only`: removes boundary evidence; falls back to a heuristic generator if LLM generation fails.
- `random_cluster_pair_control`: random cluster-pair baseline; also has a heuristic fallback.
- `heuristic_bridge`: deterministic heuristic generator used as a fallback/debug baseline.

## Quick Start

### Smoke Run

Use this first. It exercises the full path without the full publication-scale budget.

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --n-gap-targets 1 `
  --n-cluster-pair-targets 1 `
  --methods orchestrator single_shot_llm cluster_only random_cluster_pair_control `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --analysis-clustering-method kmeans `
  --analysis-pca-components 16 `
  --discovery-cue-text "Focus on inhaled RNA delivery for inflammatory lung disease" `
  --output-dir data/retrospective_eval_smoke
```

### Full Benchmark Run

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --cutoff-date 2020-12-31 `
  --future-window-start 2022-01-01 `
  --future-window-end 2025-12-31 `
  --n-gap-targets 20 `
  --n-cluster-pair-targets 10 `
  --methods orchestrator single_shot_llm retrieval_summary_direct cluster_only random_cluster_pair_control `
  --seeds 3 `
  --hypotheses-per-target 3 `
  --discovery-cue-text "Focus on folate-targeted RNA delivery for breast cancer" `
  --output-dir data/retrospective_eval
```

### Resume From an Existing Historical Snapshot

If a historical snapshot has already been published to the backend, reuse it:

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --existing-snapshot-id retro_hist_20201231_abcd1234 `
  --n-gap-targets 20 `
  --n-cluster-pair-targets 10 `
  --methods orchestrator single_shot_llm retrieval_summary_direct cluster_only random_cluster_pair_control `
  --seeds 3 `
  --hypotheses-per-target 3 `
  --output-dir data/retrospective_eval_resume
```

This avoids rebuilding and re-publishing the historical snapshot.

### Cue Semantics

The retrospective CLI accepts:

- `--discovery-cue-text`: free-text steering cue
- `--discovery-cue-goal`: optional shorter statement of the intended direction

The cue is treated as steering context, not evidence. It is used to:

- rerank automatically selected gap and cluster-pair targets inside retrospective runs
- expand and rerank evidence-pack retrieval
- steer orchestrator and baseline prompts
- annotate generated hypotheses and evaluation runs for reproducibility

## What Gets Written

### Backend

The run writes:

- a historical snapshot if `--existing-snapshot-id` is not used
- evaluation run records
- evaluation match records

### Files

Each run exports:

- `<run_id>_review_packet.csv`
- `<run_id>_review_packet.json`

Both are written under the configured `--output-dir`.

The CLI also prints a JSON summary containing:

- `run`
- `review_packet_csv`
- `review_packet_json`

## Review Packet Columns

The exported CSV includes:

- `run_id`
- `method_name`
- `seed`
- `target_id`
- `target_type`
- `hypothesis_id`
- `classification`
- `title`
- `text`
- `support_citations`
- `historical_label`
- `historical_best_paper_id`
- `historical_best_title`
- `future_label`
- `future_best_paper_id`
- `future_best_title`
- `first_future_year`

This file is intended for expert review and manual adjudication.

## Matching and Classification

### Idea Fingerprint

Each hypothesis is normalized into fields in [idea_fingerprint.py](idea_fingerprint.py):

- `disease`
- `material`
- `payload`
- `targeting`
- `mechanism`
- `model`
- `route`
- `outcome`

### Candidate Retrieval

Implemented in [candidate_match.py](candidate_match.py):

- keyword retrieval from fingerprint terms
- embedding retrieval using Qwen query embeddings against precomputed paper embeddings
- reranking with the local Qwen reranker

If the Qwen service times out or throws an error, the matcher degrades gracefully and continues with the available signals instead of aborting the whole benchmark.

### Match Labels

Candidate papers are assigned one of:

- `strong_match`
- `partial_match`
- `background_only`
- `no_match`

### Hypothesis-Level Classifications

Implemented in [judge.py](judge.py):

- `already_present`: strong historical match before the cutoff
- `anticipatory_strong`: not historically present and strongly matched in the future
- `anticipatory_partial`: not historically present and partially matched in the future
- `unsupported`: poor grounding or no support citations
- `unrealized`: no convincing future match in the evaluation window

## Reported Metrics

Aggregated in [metrics.py](metrics.py):

- `historical_leakage_rate`
- `anticipatory_strong_rate`
- `anticipatory_partial_rate`
- `unsupported_rate`
- `unrealized_rate`
- `novelty_adjusted_hit_rate`
- `median_time_to_first_future_match_year`
- per-method counts and hit rates

## Practical Notes

- Use `127.0.0.1` for local service URLs when calling from the CLI.
- The Qwen service can run out of GPU memory on large runs. The matcher now falls back rather than hard-failing, but runtime can still increase.
- Publishing a full historical snapshot can take time. If you already have a valid historical snapshot in the backend, prefer `--existing-snapshot-id`.
- The full benchmark is expensive in both runtime and API usage. Start with a smoke run.
- OpenAI-backed methods require `OPENAI_API_KEY`. The heuristic fallback methods do not.

## Recommended Workflow

1. Run a smoke benchmark and inspect the review packet.
2. Fix obvious matching or prompt failures.
3. Run the full `2020 -> 2022-2025` benchmark.
4. Add rolling historical cutoffs such as `2016`, `2018`, and `2020`.
5. Manually review a stratified subset of outputs.
6. Freeze the protocol before writing paper results.

## Current Limitations

- Matching is heuristic and still needs expert calibration.
- A clean future hit rate does not by itself prove scientific value.
- The sensitivity window is stored in the future match payload, but the main reported classification still uses the primary future window.
- Publication-grade evaluation still requires manual review of ambiguous cases.
