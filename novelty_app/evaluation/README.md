# Retrospective Future-Paper Recovery

This package evaluates whether historical frontier evidence can recover held-out future papers.

The core question is:

`Given evidence only up to a cutoff date, can the system recover later papers that are plausibly anchored to a historical frontier target?`

## What The Benchmark Does

The retrospective runner in [run_retrospective.py](run_retrospective.py) now works as a future-paper recovery benchmark:

1. Load the corpus and embeddings.
2. Split the corpus into `historical` and `future`.
3. Build a historical novelty snapshot.
4. Select historical gap and cluster-pair frontier targets.
5. Build focused evidence packs for those targets.
6. Filter future papers:
   - remove future papers with a strong historical near-duplicate
   - keep only frontier-eligible future papers with positive target affinity
   - if a discovery cue is provided, use it to rerank future papers without hard-filtering them
7. Assign each benchmark future paper to a single historical target.
8. Generate hypotheses from the assigned historical target using the configured method(s).
9. Retrieve against the future corpus and record the rank of the gold future paper.
10. Retrieve against the historical corpus and record the strongest confounder.
11. Aggregate task-level recovery metrics and export hybrid review packets.

## Cue Semantics

Discovery cues are part of the core protocol:

- without a cue, the benchmark uses the broader frontier-eligible future-paper pool
- with a cue, the benchmark reranks that frontier-eligible pool toward cue-relevant papers
- cues steer evidence-pack construction and generation
- cue mismatch is penalized in cue-aware metrics but does not hard-reject a hypothesis

The cue is steering context, not evidence.

## Generation Methods

The default method set is:

- `orchestrator`
- `single_shot_llm`
- `retrieval_summary_direct`
- `heuristic_bridge`
- `pack_query_baseline`
- `random_target_control`

`pack_query_baseline` is a deterministic retrieval-oriented baseline built from the focused evidence pack.

## Evidence Packs

Retrospective evaluation uses the `focused_eval` evidence-pack profile:

- `diverse=0`
- small local frontier neighborhoods only
- cue-aware reranking when a discovery cue is provided

This keeps the pack closer to the actual frontier target and avoids broad global filler papers.

## Outputs

Each run writes:

- one evaluation run record
- raw hypothesis-level match records
- `<run_id>_review_packet.csv`
- `<run_id>_review_packet.json`
- `<run_id>_assessment_bundle_v1.json`

The review packet is built from the best hypothesis per `(method, seed, gold_future_paper_id)` task and includes:

- the gold future paper
- the assigned historical target
- the best hypothesis
- focused evidence-pack summary
- top future retrievals
- top historical retrievals
- cue text and cue score
- blank manual review fields

The assessment bundle is built from all generated hypotheses and includes the full ideation context needed for blind human review:

- discovery cue
- effective target
- evidence papers used for ideation
- explanation and audit payloads
- raw and normalized hypothesis payloads
- model judge output for post-submit reveal
- retrospective benchmark rows per idea

## Primary Outcome Labels

The benchmark uses recovery-oriented labels:

- `gold_recovered`
- `future_neighbor_only`
- `historical_confound`
- `not_recovered`

## Reported Metrics

Aggregated in [metrics.py](metrics.py):

- `gold_recall_at_1`
- `gold_recall_at_5`
- `gold_recall_at_10`
- `gold_mrr`
- `future_neighbor_only_rate`
- `historical_confound_rate`
- `median_gold_rank`
- `gold_recovered_rate`
- `not_recovered_rate`
- `mean_average_idea_score`
- average per-criterion idea scores
- per-method versions of the same metrics

When a cue is provided, the benchmark also reports:

- `cue_weighted_recall_at_1`
- `cue_weighted_recall_at_5`
- `cue_weighted_recall_at_10`
- `cue_weighted_mrr`
- `mean_hypothesis_cue_score`

## Quick Start

Smoke run:

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --n-gap-targets 2 `
  --n-cluster-pair-targets 2 `
  --n-gold-future-papers 10 `
  --methods orchestrator single_shot_llm heuristic_bridge pack_query_baseline random_target_control `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --analysis-clustering-method kmeans `
  --analysis-pca-components 16 `
  --output-dir data/retrospective_eval_smoke
```

For faster smoke runs, you can add `--disable-leakage-check` to skip only the initial historical near-duplicate screen used when selecting gold future papers. This does not disable later historical-confound scoring for generated hypotheses.

Cue-aware smoke run:

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --n-gap-targets 5 `
  --n-cluster-pair-targets 5 `
  --n-gold-future-papers 20 `
  --methods orchestrator single_shot_llm retrieval_summary_direct heuristic_bridge pack_query_baseline random_target_control `
  --seeds 1 `
  --hypotheses-per-target 2 `
  --discovery-cue-text "Focus on folate-targeted RNA delivery for breast cancer" `
  --output-dir data/retrospective_eval_cued
```

## Prospective Generation

`run_prospective.py` reuses the same generation registry and snapshot-backed target selection, but does not do time-splitting, future-paper assignment, or recovery evaluation.

It is intended for:

- running the orchestrator or baselines against an already published snapshot
- generating across the top gap and cluster-pair targets for that snapshot
- exporting a local bundle of generated hypotheses for review

Explicit target smoke run:

```powershell
python -m novelty_app.evaluation.run_prospective `
  --backend-url http://127.0.0.1:8088 `
  --snapshot-id your_snapshot_id `
  --gap-id gap_0 `
  --methods orchestrator `
  --seeds 1 `
  --hypotheses-per-target 2 `
  --output-dir data/prospective_eval_smoke
```

Auto-target smoke run:

```powershell
python -m novelty_app.evaluation.run_prospective `
  --backend-url http://127.0.0.1:8088 `
  --snapshot-id your_snapshot_id `
  --n-gap-targets 2 `
  --n-cluster-pair-targets 2 `
  --methods orchestrator single_shot_llm heuristic_bridge pack_query_baseline `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --output-dir data/prospective_eval_smoke
```

Each run writes:

- `<run_id>_summary.json`
- `<run_id>_hypotheses.json`
- `<run_id>_hypotheses.csv`

## Practical Notes

- Use `127.0.0.1` for local service URLs when calling from the CLI.
- OpenAI-backed methods require `OPENAI_API_KEY`.
- The Qwen service remains the retrieval backbone.
- The benchmark still depends on heuristic matching and should be reviewed manually for paper-grade claims.
- Start with a smoke run and inspect the review packet before scaling up.
