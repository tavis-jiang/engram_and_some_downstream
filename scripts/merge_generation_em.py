#!/usr/bin/env python3
"""Merge sharded generation EM JSONL files into one summary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["gsm8k_em", "triviaqa_em"], required=True)
    parser.add_argument("--records", nargs="+", required=True, help="Shard JSONL files")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--run-label", default="merged")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate IDs instead of failing",
    )
    return parser.parse_args()


def load_records(paths: List[Path], *, allow_duplicates: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen = set()
    for path in paths:
        with path.open("r") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record.get("task"), record.get("model"), record.get("id"))
                if key in seen and not allow_duplicates:
                    raise ValueError(f"Duplicate record {key} in {path}:{line_no}")
                seen.add(key)
                records.append(record)
    return records


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.records]
    records = load_records(paths, allow_duplicates=args.allow_duplicates)
    filtered = [
        r
        for r in records
        if r.get("model") == args.model and r.get("task") == args.task
    ]
    if len(filtered) != len(records):
        raise ValueError("Some records do not match --model/--task")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"generation_em_{args.model}_{args.task}_{args.run_label}_{timestamp}"
    records_path = output_dir / f"{prefix}.jsonl"
    summary_path = output_dir / f"{prefix}.summary.json"

    filtered.sort(key=lambda r: str(r.get("id", "")))
    with records_path.open("w") as f:
        for record in filtered:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct = sum(int(r.get("exact_match", 0)) for r in filtered)
    total = len(filtered)
    exact_match = correct / total if total else 0.0
    summary = {
        "model": args.model,
        "task": args.task,
        "run_label": args.run_label,
        "num_examples": total,
        "correct": correct,
        "exact_match": exact_match,
        "records": str(records_path),
        "source_files": [str(p) for p in paths],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.task} exact_match={exact_match:.4f}")
    print(f"Records: {records_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
