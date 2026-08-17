#!/usr/bin/env bash
set -euo pipefail

JOB_ID=${1:-65042}
LOG_FILE=${2:-logs/20260530-193938_our_xgram_config_rank0.log}
RUN_DIR=${3:-runs/xgram-smollm2-360m-fineweb10b-64971}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
MAX_ITERATIONS=${MAX_ITERATIONS:-192}

cd /home/bcjiang/X-gram

for ((i = 1; i <= MAX_ITERATIONS; i++)); do
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') iteration=${i} ====="

  echo "[slurm]"
  squeue -j "${JOB_ID}" -o '%.18i %.9P %.35j %.8u %.2t %.12M %.12l %.6D %R' || true

  echo "[status]"
  python3 scripts/reproduction_status.py \
    --job-id "${JOB_ID}" \
    --log "${LOG_FILE}" \
    --run-dir "${RUN_DIR}" || true

  echo "[latest metrics]"
  rg -n "\[step=|throughput/device/TPS|throughput/device/TPS \(actual avg\)|throughput/total tokens|train/CE loss|train/PPL" \
    "${LOG_FILE}" | tail -80 || true

  if (( i == 1 || i % 4 == 0 )); then
    echo "[gpu]"
    timeout 30s srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 \
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits || true
  fi

  if ! squeue -j "${JOB_ID}" -h >/dev/null 2>&1 || [[ -z "$(squeue -j "${JOB_ID}" -h 2>/dev/null)" ]]; then
    echo "Job ${JOB_ID} is no longer in squeue; stopping monitor."
    break
  fi

  echo
  sleep "${INTERVAL_SECONDS}"
done
