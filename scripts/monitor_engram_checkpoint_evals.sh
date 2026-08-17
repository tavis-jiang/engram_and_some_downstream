#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-runs/faircompare-engram-360m-88169}"
ENGRAM_CONFIG_PATH="${ENGRAM_CONFIG_PATH:-${2:-}}"
MODEL="${MODEL:-${3:-engram}}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
STATE_DIR="${STATE_DIR:-${4:-reports/checkpoint_eval_state}}"
CHECKPOINT_RUN_LABEL="${CHECKPOINT_RUN_LABEL:-${5:-}}"

mkdir -p "$STATE_DIR" logs

while true; do
  date '+[%Y-%m-%d %H:%M:%S %Z] monitor tick'
  bash scripts/submit_engram_checkpoint_evals.sh \
    "$ROOT" \
    "$MODEL" \
    "$STATE_DIR" \
    "$ENGRAM_CONFIG_PATH" \
    "$CHECKPOINT_RUN_LABEL"
  sleep "$INTERVAL_SECONDS"
done
