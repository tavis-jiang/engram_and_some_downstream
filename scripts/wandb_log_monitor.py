#!/usr/bin/env python3
"""Mirror OLMo console metrics into W&B and optional local plots.

This is intentionally separate from the training process so an already-running
Slurm job does not need to be restarted just to see curves.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


STEP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+.*\[step=(?P<step>\d+)/(?P<total>\d+)(?:,[^\]]*)?\]"
)
METRIC_RE = re.compile(r"^\s+(?P<name>[A-Za-z0-9_./ ()%-]+)=(?P<value>[-+0-9.,Ee]+|nan|inf|-inf)\s*$")


def _coerce_metric_value(raw: str) -> Optional[float]:
    value = raw.replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def parse_log(path: Path, *, start_offset: int = 0) -> Iterator[tuple[int, int, Dict[str, float]]]:
    """Yield (offset, step, metrics) parsed from a console log."""
    with path.open("r", errors="ignore") as f:
        if start_offset:
            f.seek(start_offset)
        current_step: Optional[int] = None
        current_metrics: Dict[str, float] = {}
        block_end = f.tell()

        while True:
            line = f.readline()
            if not line:
                if current_step is not None and current_metrics:
                    yield f.tell(), current_step, current_metrics
                break

            step_match = STEP_RE.search(line)
            if step_match:
                if current_step is not None and current_metrics:
                    yield block_end, current_step, current_metrics
                current_step = int(step_match.group("step"))
                current_metrics = {}
                block_end = f.tell()
                continue

            metric_match = METRIC_RE.match(line)
            if metric_match and current_step is not None:
                value = _coerce_metric_value(metric_match.group("value"))
                if value is not None:
                    current_metrics[metric_match.group("name").strip()] = value
                block_end = f.tell()


def follow_log(path: Path, *, poll_seconds: float, start_at_end: bool) -> Iterator[tuple[int, Dict[str, float]]]:
    offset = path.stat().st_size if start_at_end and path.exists() else 0
    seen_steps: set[int] = set()
    while True:
        latest_offset = offset
        for latest_offset, step, metrics in parse_log(path, start_offset=offset):
            if step in seen_steps:
                continue
            seen_steps.add(step)
            yield step, metrics
        offset = latest_offset
        time.sleep(poll_seconds)


def merge_log_rows(paths: Iterable[Path]) -> list[tuple[int, Dict[str, float]]]:
    rows: Dict[int, Dict[str, float]] = {}
    for path in paths:
        for _, step, metrics in parse_log(path):
            rows[step] = metrics
    return sorted(rows.items())


def metric_subset(metrics: Dict[str, float]) -> Dict[str, float]:
    aliases = {
        "train/CE loss": "train/ce_loss",
        "train/PPL": "train/ppl",
        "train/Z loss": "train/z_loss",
        "throughput/device/TPS": "throughput/device_tps",
        "throughput/device/TPS (actual avg)": "throughput/device_tps_actual_avg",
        "throughput/device/MFU": "throughput/device_mfu",
        "throughput/device/MFU (actual avg)": "throughput/device_mfu_actual_avg",
        "throughput/total tokens": "throughput/total_tokens",
        "system/GPU active mem (GiB)": "system/gpu_active_mem_gib",
        "system/GPU reserved mem (GiB)": "system/gpu_reserved_mem_gib",
        "optim/total grad norm": "optim/total_grad_norm",
        "optim/step skipped": "optim/step_skipped",
    }
    return {dst: metrics[src] for src, dst in aliases.items() if src in metrics}


def write_csv(path: Path, rows: Iterable[tuple[int, Dict[str, float]]]) -> None:
    import csv

    rows = list(rows)
    keys = sorted({k for _, metrics in rows for k in metric_subset(metrics)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *keys])
        writer.writeheader()
        for step, metrics in rows:
            selected = metric_subset(metrics)
            writer.writerow({"step": step, **selected})


def write_plot(path: Path, rows: Iterable[tuple[int, Dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parsed = [(step, metric_subset(metrics)) for step, metrics in rows]
    steps = [step for step, _ in parsed]
    ce = [metrics.get("train/ce_loss") for _, metrics in parsed]
    tps = [metrics.get("throughput/device_tps_actual_avg", metrics.get("throughput/device_tps")) for _, metrics in parsed]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if any(v is not None for v in ce):
        axes[0].plot([s for s, v in zip(steps, ce) if v is not None], [v for v in ce if v is not None], color="#1f77b4")
    axes[0].set_ylabel("CE loss")
    axes[0].grid(True, alpha=0.25)
    if any(v is not None for v in tps):
        axes[1].plot([s for s, v in zip(steps, tps) if v is not None], [v for v in tps if v is not None], color="#2ca02c")
    axes[1].set_ylabel("device TPS")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--history-log", action="append", type=Path, default=[])
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT") or "xgram-reproduction")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--start-at-end", action="store_true")
    parser.add_argument("--csv-out", type=Path, default=Path("reports/xgram_metrics.csv"))
    parser.add_argument("--plot-out", type=Path, default=Path("reports/xgram_metrics.png"))
    args = parser.parse_args()

    source_logs = [*args.history_log, args.log]
    rows = merge_log_rows(source_logs)
    write_csv(args.csv_out, rows)
    write_plot(args.plot_out, rows)

    import wandb

    mode = args.mode or ("online if key" if os.environ.get("WANDB_API_KEY") else "offline")
    if mode == "online if key":
        mode = "online"

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name or args.log.stem,
        mode=mode,
        resume="allow",
        config={"source_logs": [str(path) for path in source_logs]},
    )
    for step, metrics in rows:
        selected = metric_subset(metrics)
        if selected:
            wandb.log(selected, step=step)
    wandb.save(str(args.csv_out))
    wandb.save(str(args.plot_out))
    print(f"W&B run: {run.get_url() if hasattr(run, 'get_url') else '<offline>'}")
    print(f"CSV: {args.csv_out}")
    print(f"Plot: {args.plot_out}")

    if args.follow:
        for step, metrics in follow_log(args.log, poll_seconds=args.poll_seconds, start_at_end=True):
            selected = metric_subset(metrics)
            if selected:
                wandb.log(selected, step=step)


if __name__ == "__main__":
    main()
