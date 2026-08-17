#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  stage_and_stop_after_checkpoint.sh JOB_ID RUN_DIR TARGET_STEP REMOTE_DATALOADER_CKPT

Wait until RUN_DIR has a finalized checkpoint step >= TARGET_STEP, copy the
running job's dataloader DuckDB files from the compute node into RUN_DIR, then
cancel the Slurm job.
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

JOB_ID=$1
RUN_DIR=$2
TARGET_STEP=$3
REMOTE_DATALOADER_CKPT=$4
POLL_SECONDS=${POLL_SECONDS:-30}
STAGED_DIR="${RUN_DIR}/dataloader_ckpt_for_resume"

latest_checkpoint_step() {
  python3 - "$RUN_DIR" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
latest = None
for path in run_dir.glob("step*"):
    if not path.is_dir():
        continue
    try:
        step = int(path.name.removeprefix("step"))
    except ValueError:
        continue
    if (path / ".metadata.json").is_file():
        latest = step if latest is None else max(latest, step)
print("" if latest is None else latest)
PY
}

echo "Waiting for ${RUN_DIR} checkpoint >= ${TARGET_STEP} before stopping job ${JOB_ID}"
while true; do
  STEP=$(latest_checkpoint_step)
  echo "$(date) latest checkpoint step: ${STEP:-none}"
  if [[ -n "${STEP}" && "${STEP}" -ge "${TARGET_STEP}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

mkdir -p "${STAGED_DIR}"
echo "Copying dataloader checkpoint from job ${JOB_ID}:${REMOTE_DATALOADER_CKPT} to ${STAGED_DIR}"
srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 \
  bash -lc "python3 - <<'PY'
from pathlib import Path
import shutil

src = Path('${REMOTE_DATALOADER_CKPT}')
dst = Path('${STAGED_DIR}')
dst.mkdir(parents=True, exist_ok=True)
for path in sorted(src.parent.glob(src.name + '*')):
    shutil.copy2(path, dst / path.name)
for path in sorted(dst.glob(src.name + '*')):
    print(path, path.stat().st_size)
PY"

echo "Cancelling job ${JOB_ID} after checkpoint ${STEP} and dataloader checkpoint staging"
scancel "${JOB_ID}"
