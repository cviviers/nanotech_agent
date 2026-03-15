from __future__ import annotations

import json

try:
    from agents.backend_client import BackendClient
    from agents.orchestrator_langgraph import build_orchestrator
    from discovery_cue import discovery_cue_to_dict
except Exception:  # pragma: no cover
    from backend_client import BackendClient  # type: ignore
    from orchestrator_langgraph import build_orchestrator  # type: ignore
    from novelty_app.discovery_cue import discovery_cue_to_dict  # type: ignore


def pick_target(backend: BackendClient, snapshot_id: str):
    print("\nChoose target type:")
    print("  1) gap (from top gap candidates)")
    print("  2) cluster_pair (A,B)")
    choice = input("> ").strip()
    if choice == "1":
        top = backend.top_gaps(snapshot_id=snapshot_id, k=15).get("gaps", [])
        for i, g in enumerate(top):
            score = g.get("density_z")
            if score is None:
                score = g.get("avg_gap_score")
            print(f"[{i}] {g['gap_id']}  score={float(score or 0):.3f}  size={g.get('size')}")
        idx = int(input("Select gap index: ").strip())
        return {"target_type": "gap", "gap_id": top[idx]["gap_id"]}

    clusters = backend.list_clusters(snapshot_id=snapshot_id, limit=20).get("clusters", [])
    print("\nTop clusters by size:")
    for c in clusters:
        print(f"  cluster_id={c['cluster_id']}  size={c['size']}")
    a = int(input("Cluster A id: ").strip())
    b = int(input("Cluster B id: ").strip())
    return {"target_type": "cluster_pair", "cluster_a": a, "cluster_b": b}


def main() -> None:
    base_url = input("Backend URL (default http://localhost:8088): ").strip() or "http://localhost:8088"
    backend = BackendClient(base_url)

    snapshots = backend.list_snapshots(limit=10).get("snapshots", [])
    if not snapshots:
        print("No snapshots found. Publish a snapshot from the Streamlit Agent Console first.")
        return
    print("\nAvailable snapshots:")
    for i, s in enumerate(snapshots):
        print(f"[{i}] {s['snapshot_id']}  {s['created_at']}")
    sidx = int(input("Select snapshot index [0]: ").strip() or "0")
    snapshot_id = snapshots[sidx]["snapshot_id"]

    app = build_orchestrator(backend)
    target = pick_target(backend, snapshot_id)

    state = {
        **target,
        "snapshot_id": snapshot_id,
        "max_iters": 2,
        "iter": 0,
        "exemplars": 25,
        "boundary": 25,
        "diverse": 25,
    }
    cue_text = input("Discovery cue (optional): ").strip()
    if cue_text:
        state["discovery_cue"] = discovery_cue_to_dict(cue_text)

    print("\nRunning orchestrator...\n")
    out = app.invoke(state)

    print("\n=== PUBLISHED BRIEF (summary) ===")
    print("evidence_size:", len(out.get("evidence", [])))
    print("iterations:", out.get("iter", 0))
    print("published:", out.get("published", False))
    print("audit supported_claim_fraction:", out.get("audit", {}).get("supported_claim_fraction"))
    axes = out.get("explanation", {}).get("axes_of_separation", [])[:3]
    print("\nAxes of separation (first 3):")
    print(json.dumps(axes, indent=2, ensure_ascii=False))
    hyps = out.get("hypotheses", {}).get("hypotheses", [])
    print("\nTop hypothesis:")
    print(json.dumps(hyps[0] if hyps else {}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
