#!/bin/bash
#SBATCH --job-name=baseline-handoff-4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=cpu
#SBATCH --mail-user=913669023@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

cd /home/bcjiang/X-gram

BASELINE_JOB_ID=${BASELINE_JOB_ID:-65168}
BASELINE_RUN_DIR=${BASELINE_RUN_DIR:-runs/baseline-smollm2-360m-fineweb10b-64984}
BASELINE_LOG=${BASELINE_LOG:-logs/20260531-143259_our_baseline_config_rank0.log}
REMOTE_DATALOADER_CKPT=${REMOTE_DATALOADER_CKPT:-/tmp/olmo_stream_ckpts/run-20260531-143259-25850-2827306/dataloader.ckpt}
RESUME_SCRIPT=${RESUME_SCRIPT:-python_slurm/resume_baseline_8x4090.slurm}
WAIT_FOR_FRESH_CHECKPOINT=${WAIT_FOR_FRESH_CHECKPOINT:-1}
POLL_SECONDS=${POLL_SECONDS:-60}

latest_checkpoint_step() {
  python3 - "$BASELINE_RUN_DIR" <<'PY'
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

latest_logged_step() {
  python3 - "$BASELINE_LOG" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
step = None
if path.exists():
    pattern = re.compile(r"\[step=(\d+)/")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
print("" if step is None else step)
PY
}

echo "=========================================="
echo "Baseline handoff to RTX4090"
echo "=========================================="
echo "Handoff job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Baseline job: ${BASELINE_JOB_ID}"
echo "Baseline run dir: ${BASELINE_RUN_DIR}"
echo "Baseline log: ${BASELINE_LOG}"
echo "Remote dataloader ckpt: ${REMOTE_DATALOADER_CKPT}"
echo "Resume script: ${RESUME_SCRIPT}"
echo "WAIT_FOR_FRESH_CHECKPOINT: ${WAIT_FOR_FRESH_CHECKPOINT}"
echo "=========================================="

if ! squeue -j "${BASELINE_JOB_ID}" -h >/dev/null 2>&1 || [[ -z "$(squeue -j "${BASELINE_JOB_ID}" -h 2>/dev/null)" ]]; then
  echo "Baseline job ${BASELINE_JOB_ID} is not running; refusing to submit a 4090 resume with an unstaged dataloader checkpoint."
  exit 1
fi

LATEST_CKPT=$(latest_checkpoint_step)
LATEST_LOGGED=$(latest_logged_step)
echo "Latest baseline checkpoint before handoff: ${LATEST_CKPT:-none}"
echo "Latest baseline logged step before handoff: ${LATEST_LOGGED:-none}"

if [[ -z "${LATEST_CKPT}" ]]; then
  TARGET_STEP=1
elif [[ "${WAIT_FOR_FRESH_CHECKPOINT}" == "1" ]]; then
  TARGET_STEP=$((LATEST_CKPT + 1))
else
  TARGET_STEP=${LATEST_CKPT}
fi

echo "Target checkpoint step for safe cutover: ${TARGET_STEP}"
echo "Staging dataloader checkpoint and stopping baseline job ${BASELINE_JOB_ID}..."

POLL_SECONDS="${POLL_SECONDS}" scripts/stage_and_stop_after_checkpoint.sh \
  "${BASELINE_JOB_ID}" \
  "${BASELINE_RUN_DIR}" \
  "${TARGET_STEP}" \
  "${REMOTE_DATALOADER_CKPT}"

echo "Submitting baseline 8xRTX4090 resume..."
SBATCH_OUTPUT=$(sbatch "${RESUME_SCRIPT}")
echo "${SBATCH_OUTPUT}"
NEW_JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $NF}')
echo "${NEW_JOB_ID}" > logs/latest_baseline_4090_job_id.txt

echo "New baseline 8xRTX4090 job ID: ${NEW_JOB_ID}"
echo "Handoff completed at: $(date)"
