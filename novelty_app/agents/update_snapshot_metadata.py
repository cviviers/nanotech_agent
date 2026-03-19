from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    from agents.knowledge_store import KnowledgeStore, default_db_path
except Exception:  # pragma: no cover
    from novelty_app.agents.knowledge_store import KnowledgeStore, default_db_path


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--metadata-json must decode to a JSON object")
    return value


def _build_updates(args: argparse.Namespace) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if args.metadata_json:
        updates.update(_parse_json_object(args.metadata_json))
    if args.split_role is not None:
        updates["split_role"] = args.split_role
    if args.cutoff_date is not None:
        updates["cutoff_date"] = args.cutoff_date
    if args.future_window_start is not None:
        updates["future_window_start"] = args.future_window_start
    if args.future_window_end is not None:
        updates["future_window_end"] = args.future_window_end
    if args.source is not None:
        updates["source"] = args.source
    if args.embedding_source is not None:
        updates["embedding_source"] = args.embedding_source
    if args.analysis_config_hash is not None:
        updates["analysis_config_hash"] = args.analysis_config_hash
    return updates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch metadata for an existing snapshot.")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--db-path", default=None, help="SQLite file to update. Defaults to NOVELTY_AGENT_DB or the project default.")
    parser.add_argument("--split-role", choices=["historical", "future", "full"], default=None)
    parser.add_argument("--cutoff-date", default=None)
    parser.add_argument("--future-window-start", default=None)
    parser.add_argument("--future-window-end", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--embedding-source", default=None)
    parser.add_argument("--analysis-config-hash", default=None)
    parser.add_argument("--metadata-json", default=None, help="Additional metadata fields as a JSON object.")
    parser.add_argument("--replace", action="store_true", help="Replace the full metadata object instead of merging.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resulting metadata without writing it.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    updates = _build_updates(args)
    if not updates and not args.replace:
        raise SystemExit("No metadata updates provided.")

    db_path = Path(args.db_path).expanduser().resolve() if args.db_path else default_db_path()
    store = KnowledgeStore(db_path)
    current = store.get_snapshot(args.snapshot_id)
    new_metadata = {} if args.replace else dict(current.get("metadata") or {})
    new_metadata.update(updates)

    payload = {
        "db_path": str(db_path),
        "snapshot_id": current["snapshot_id"],
        "created_at": current["created_at"],
        "metadata": new_metadata,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    updated = store.update_snapshot_metadata(args.snapshot_id, updates, replace=args.replace)
    print(
        json.dumps(
            {
                "db_path": str(db_path),
                "snapshot_id": updated["snapshot_id"],
                "created_at": updated["created_at"],
                "metadata": updated["metadata"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
