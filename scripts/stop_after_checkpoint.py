#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional


def latest_checkpoint_step(run_dir: Path) -> Optional[int]:
    latest = None
    if not run_dir.exists():
        return None
    for path in run_dir.glob("step*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("step"))
        except ValueError:
            continue
        if (path / ".metadata.json").is_file():
            latest = step if latest is None else max(latest, step)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait until a run has checkpointed at/after a target step, then cancel the Slurm job."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target-step", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(
        f"waiting for {args.run_dir} to reach checkpoint step >= {args.target_step}; "
        f"will cancel job {args.job_id}"
    )
    while True:
        step = latest_checkpoint_step(args.run_dir)
        print(f"latest checkpoint step: {step}")
        if step is not None and step >= args.target_step:
            if args.dry_run:
                print(f"dry run: would run scancel {args.job_id}")
            else:
                subprocess.check_call(["scancel", args.job_id])
                print(f"cancelled job {args.job_id} after checkpoint step {step}")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
