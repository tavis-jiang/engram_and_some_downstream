#!/bin/bash
# =============================================================================
# Generation-based EM eval for GSM8K and TriviaQA.
#
# Use sbatch from the login/storage node:
#   sbatch python_slurm/eval_generation_em.slurm baseline gsm8k_em 5
#   sbatch python_slurm/eval_generation_em.slurm xgram triviaqa_em 5
#   sbatch python_slurm/eval_generation_em.slurm baseline all all
#
# Args:
#   $1 model: baseline | xgram | engram
#   $2 task : gsm8k_em | triviaqa_em | all
#   $3 limit: number of examples, or all. Default is 5 for smoke testing.
#
# Optional env:
#   BATCH_SIZE=1
#   MAX_NEW_TOKENS=512
#   MAX_PROMPT_TOKENS=2048
#   NO_CACHE=1
#   ALLOW_ONLINE_DATASETS=1   # needed if TriviaQA is not already cached locally
#   OFFSET=0
#   NUM_SHARDS=8 SHARD_INDEX=0
#     For Slurm arrays, pass --array=0-7 and set NUM_SHARDS=8; SHARD_INDEX
#     defaults to SLURM_ARRAY_TASK_ID.
# =============================================================================
#SBATCH --job-name=xgram-gen-em
#SBATCH --partition=ADA6000
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

MODEL="${1:?need model: baseline|xgram|engram}"
TASK="${2:?need task: gsm8k_em|triviaqa_em|all}"
LIMIT="${3:-5}"

cd /home/bcjiang/X-gram
source /home/bcjiang/miniconda3/etc/profile.d/conda.sh
conda activate xgram_env

export PYTHONPATH="/home/bcjiang/X-gram/OLMo-core/src:/home/bcjiang/X-gram/packages/ubdataloader/src:/home/bcjiang/X-gram/packages/olmo_in_loop_evals/src:${PYTHONPATH:-}"
export OLMO_SHARED_FS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [[ "${ALLOW_ONLINE_DATASETS:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
else
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

RUN_TAG=$(date +%Y%m%d-%H%M%S)
mkdir -p logs reports out

SHARD_ARGS=()
SHARD_LOG_SUFFIX=""
if [[ -n "${NUM_SHARDS:-}" ]]; then
  SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
  if [[ -z "$SHARD_INDEX" ]]; then
    echo "NUM_SHARDS is set, but SHARD_INDEX and SLURM_ARRAY_TASK_ID are empty" >&2
    exit 1
  fi
  SHARD_ARGS+=(--num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --run-label "shard${SHARD_INDEX}-of-${NUM_SHARDS}")
  SHARD_LOG_SUFFIX="_shard${SHARD_INDEX}-of-${NUM_SHARDS}"
elif [[ -n "${OFFSET:-}" ]]; then
  SHARD_ARGS+=(--offset "$OFFSET" --run-label "offset${OFFSET}")
  SHARD_LOG_SUFFIX="_offset${OFFSET}"
fi

LOG="logs/generation_em_${MODEL}_${TASK}_${LIMIT}${SHARD_LOG_SUFFIX}_${RUN_TAG}.log"

ARGS=(
  --model "$MODEL"
  --task "$TASK"
  --limit "$LIMIT"
  --batch-size "${BATCH_SIZE:-1}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-2048}"
  --progress-interval "${PROGRESS_INTERVAL:-50}"
  --output-dir reports
  "${SHARD_ARGS[@]}"
)

MODEL_UPPER=$(printf '%s' "$MODEL" | tr '[:lower:]' '[:upper:]')
MODEL_CHECKPOINT_VAR="${MODEL_UPPER}_CHECKPOINT_ROOT"
MODEL_CONFIG_VAR="${MODEL_UPPER}_CONFIG"
MODEL_TOKENIZER_VAR="${MODEL_UPPER}_TOKENIZER"

if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  ARGS+=(--checkpoint-root "$CHECKPOINT_ROOT")
fi
if [[ -n "${!MODEL_CHECKPOINT_VAR:-}" ]]; then
  ARGS+=(--checkpoint-root "${!MODEL_CHECKPOINT_VAR}")
fi
if [[ -n "${CONFIG:-}" ]]; then
  ARGS+=(--config "$CONFIG")
fi
if [[ -n "${!MODEL_CONFIG_VAR:-}" ]]; then
  ARGS+=(--config "${!MODEL_CONFIG_VAR}")
fi
if [[ -n "${TOKENIZER:-}" ]]; then
  ARGS+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "${!MODEL_TOKENIZER_VAR:-}" ]]; then
  ARGS+=(--tokenizer "${!MODEL_TOKENIZER_VAR}")
fi

if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
if [[ "${NO_CACHE:-0}" == "1" ]]; then
  ARGS+=(--no-cache)
fi
if [[ -n "${TRIVIAQA_DATASET_DIR:-}" ]]; then
  ARGS+=(--triviaqa-dataset-dir "$TRIVIAQA_DATASET_DIR")
fi
if [[ -n "${DATASET_CACHE_DIR:-}" ]]; then
  ARGS+=(--dataset-cache-dir "$DATASET_CACHE_DIR")
fi

echo "== generation EM model=$MODEL task=$TASK limit=$LIMIT job=$SLURM_JOB_ID node=$SLURMD_NODENAME =="
echo "== log=$LOG =="

srun --label python -u scripts/eval_generation_em.py "${ARGS[@]}" 2>&1 | tee "$LOG"

echo "== generation EM done, log=$LOG =="
