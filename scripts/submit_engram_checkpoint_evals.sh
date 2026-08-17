#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-runs/faircompare-engram-360m-88169}"
MODEL="${2:-engram}"
STATE_DIR="${3:-reports/checkpoint_eval_state}"
ENGRAM_CONFIG_PATH="${ENGRAM_CONFIG_PATH:-${4:-}}"
CHECKPOINT_RUN_LABEL="${CHECKPOINT_RUN_LABEL:-${5:-}}"

mkdir -p "$STATE_DIR"

safe_label() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

submit_for_checkpoint() {
  local ckpt_dir="$1"
  local step_name
  step_name="$(basename "$ckpt_dir")"
  local checkpoint_label="$step_name"
  local state_prefix="${STATE_DIR}/${MODEL}_${step_name}"
  if [[ -n "$CHECKPOINT_RUN_LABEL" ]]; then
    local run_label
    run_label="$(safe_label "$CHECKPOINT_RUN_LABEL")"
    checkpoint_label="${run_label}_${step_name}"
    state_prefix="${STATE_DIR}/${MODEL}_${checkpoint_label}"
  fi

  if [[ ! -f "${state_prefix}.sciq.submitted" ]]; then
    local sciq_job
    if [[ "$MODEL" == "engram" && -n "$ENGRAM_CONFIG_PATH" ]]; then
      sciq_job="$(
        ENGRAM_CONFIG="$ENGRAM_CONFIG_PATH" \
        ENGRAM_CHECKPOINT_ROOT="$ckpt_dir" \
        CHECKPOINT_LABEL="$checkpoint_label" \
        sbatch --parsable python_slurm/eval_downstream.slurm "$MODEL" sciq
      )"
    else
      sciq_job="$(
        ENGRAM_CHECKPOINT_ROOT="$ckpt_dir" \
        CHECKPOINT_LABEL="$checkpoint_label" \
        sbatch --parsable python_slurm/eval_downstream.slurm "$MODEL" sciq
      )"
    fi
    printf '%s\n' "$sciq_job" > "${state_prefix}.sciq.submitted"
    echo "[submit] ${MODEL} ${checkpoint_label} sciq job=${sciq_job}"
  fi

  if [[ ! -f "${state_prefix}.full.submitted" ]]; then
    local full_job
    if [[ "$MODEL" == "engram" && -n "$ENGRAM_CONFIG_PATH" ]]; then
      full_job="$(
        ENGRAM_CONFIG="$ENGRAM_CONFIG_PATH" \
        ENGRAM_CHECKPOINT_ROOT="$ckpt_dir" \
        CHECKPOINT_LABEL="$checkpoint_label" \
        sbatch --parsable python_slurm/eval_downstream.slurm "$MODEL"
      )"
    else
      full_job="$(
        ENGRAM_CHECKPOINT_ROOT="$ckpt_dir" \
        CHECKPOINT_LABEL="$checkpoint_label" \
        sbatch --parsable python_slurm/eval_downstream.slurm "$MODEL"
      )"
    fi
    printf '%s\n' "$full_job" > "${state_prefix}.full.submitted"
    echo "[submit] ${MODEL} ${checkpoint_label} full job=${full_job}"
  fi
}

shopt -s nullglob
for ckpt_dir in "${ROOT}"/step*; do
  [[ -d "$ckpt_dir/model_and_optim" ]] || continue
  submit_for_checkpoint "$ckpt_dir"
done
