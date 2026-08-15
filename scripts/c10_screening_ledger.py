#!/usr/bin/env python3
"""Atomic bounded-run ledger for seed-42 C10/C12 screening."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="results/diagnostics/c10_screening_ledger.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=4)
    parser.add_argument("--status", choices=["reserve", "complete", "failed"], default="reserve")
    args = parser.parse_args()
    path = Path(args.ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "protocol": "bounded_validation_screening", "max_new_seed42_training_runs": args.max_runs,
        "runs": [], "test_used_for_selection": False,
    }
    matches = [run for run in payload["runs"] if run["run_id"] == args.run_id]
    if args.status == "reserve":
        if matches:
            raise SystemExit(f"Run id already registered: {args.run_id}")
        counted = [run for run in payload["runs"] if run.get("seed") == 42]
        if args.seed == 42 and len(counted) >= args.max_runs:
            raise SystemExit(
                f"Bounded screening exhausted: {len(counted)}/{args.max_runs} seed-42 runs"
            )
        payload["runs"].append({
            "run_id": args.run_id, "branch": args.branch, "seed": args.seed,
            "status": "reserved", "reserved_at": datetime.now().isoformat(),
        })
    else:
        if not matches:
            raise SystemExit(f"Cannot update unregistered run: {args.run_id}")
        matches[0]["status"] = args.status
        matches[0][f"{args.status}_at"] = datetime.now().isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps({"ledger": str(path), "run_id": args.run_id,
                      "status": args.status, "count": len(payload["runs"])}, indent=2))


if __name__ == "__main__":
    main()
