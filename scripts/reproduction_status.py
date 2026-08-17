#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_RE = re.compile(r"\[step=(\d+)/(\d+),epoch=(\d+)(?:,eta=([^\]]+))?\]")
LOSS_RE = re.compile(r"train/CE loss=([0-9.]+)")
TOKENS_RE = re.compile(r"throughput/total tokens=([0-9,]+)")


def parse_log(path: Path) -> Dict[str, Any]:
    latest_step: Optional[Tuple[int, int, int, Optional[str]]] = None
    latest_loss: Optional[float] = None
    latest_tokens: Optional[int] = None
    errors: List[str] = []

    if not path.exists():
        return {"log": str(path), "exists": False}

    for line in path.read_text(errors="replace").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            latest_step = (
                int(step_match.group(1)),
                int(step_match.group(2)),
                int(step_match.group(3)),
                step_match.group(4),
            )
        loss_match = LOSS_RE.search(line)
        if loss_match:
            latest_loss = float(loss_match.group(1))
        tokens_match = TOKENS_RE.search(line)
        if tokens_match:
            latest_tokens = int(tokens_match.group(1).replace(",", ""))
        if any(token in line for token in ("Traceback", "ERROR", "RuntimeError", "Exception", "CUDA out of memory", "Killed")):
            errors.append(line[-500:])

    result: Dict[str, Any] = {"log": str(path), "exists": True}
    if latest_step is not None:
        step, total, epoch, eta = latest_step
        result.update({"step": step, "total_steps": total, "epoch": epoch, "eta": eta})
    result["ce_loss"] = latest_loss
    result["total_tokens"] = latest_tokens
    result["recent_errors"] = errors[-5:]
    return result


def latest_checkpoint(run_dir: Path) -> Optional[str]:
    if not run_dir.exists():
        return None
    checkpoints = []
    for path in run_dir.glob("step*"):
        if path.is_dir():
            try:
                step = int(path.name.removeprefix("step"))
            except ValueError:
                continue
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(max(checkpoints)[1])


def slurm_state(job_id: str) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["squeue", "-j", job_id, "-h", "-o", "%i %T %M %R"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize X-gram reproduction jobs.")
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "jobs": {job_id: slurm_state(job_id) for job_id in args.job_id},
        "logs": [parse_log(path) for path in args.log],
        "runs": [
            {
                "run_dir": str(path),
                "latest_checkpoint": latest_checkpoint(path),
            }
            for path in args.run_dir
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for job_id, state in report["jobs"].items():
        print(f"job {job_id}: {state or 'not in squeue'}")
    for log in report["logs"]:
        if not log.get("exists"):
            print(f"log {log['log']}: missing")
            continue
        step = log.get("step")
        total = log.get("total_steps")
        loss = log.get("ce_loss")
        tokens = log.get("total_tokens")
        eta = log.get("eta")
        print(f"log {log['log']}: step={step}/{total} ce_loss={loss} tokens={tokens} eta={eta}")
        for error in log.get("recent_errors", []):
            print(f"  error: {error}")
    for run in report["runs"]:
        print(f"run {run['run_dir']}: latest_checkpoint={run['latest_checkpoint']}")


if __name__ == "__main__":
    main()
