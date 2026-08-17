#!/bin/bash
#SBATCH --job-name=engram-xgrammatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=L40S,ADA6000
#SBATCH --qos=normal
#SBATCH --mail-user=913669023@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

CONFIG_PATH="${1:-configs/faircompare_engram_vpath_xgrammatch_360m.yaml}"
RUN_NAME_ARG="${2:-faircompare-engram-vpath-xgrammatch}"

echo "=========================================="
echo "Engram X-grammatch Training Job"
echo "Config: ${CONFIG_PATH}"
echo "Run name: ${RUN_NAME_ARG}"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"
echo "=========================================="

cd /home/bcjiang/X-gram
source /home/bcjiang/miniconda3/etc/profile.d/conda.sh
conda activate xgram_env

export CODE_ROOT="/home/bcjiang/X-gram"
export OLMo_CORE_SRC="${CODE_ROOT}/OLMo-core/src"
export UBDATALOADER_SRC="${CODE_ROOT}/packages/ubdataloader/src"
export PYTHONPATH="${OLMo_CORE_SRC}:${UBDATALOADER_SRC}:${PYTHONPATH:-}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OLMO_SHARED_FS=1
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
export NCCL_ALGO=^NVLS
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((8700 + SLURM_JOB_ID % 1000))

export STREAMING_DATA_PATH="1 /tmp/faircompare-engram-vpath-xgrammatch-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE="${PREFETCH_QUEUE_SIZE:-4}"
export STREAMING_PACK_METHOD="native:truncate"

export SAVE_ROOT="./runs"
if [[ -n "${RUN_NAME_EXACT:-}" ]]; then
  export RUN_NAME="${RUN_NAME_EXACT}"
else
  export RUN_NAME="${RUN_NAME_ARG}-${SLURM_JOB_ID}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OLMO_PRE_TRAIN_CHECKPOINT="${OLMO_PRE_TRAIN_CHECKPOINT:-0}"
export OLMO_COMPILE="${OLMO_COMPILE:-0}"
export OLMO_EPHEMERAL_SAVE_INTERVAL="${OLMO_EPHEMERAL_SAVE_INTERVAL:-none}"
export OLMO_SAVE_ASYNC="${OLMO_SAVE_ASYNC:-0}"
export OLMO_SKIP_MODEL_DIAGNOSTICS="${OLMO_SKIP_MODEL_DIAGNOSTICS:-1}"
export OLMO_FREEZE_BACKBONE="${OLMO_FREEZE_BACKBONE:-0}"

echo "Environment:"
echo "  Python: $(which python3)"
echo "  STREAMING_DATA_PATH: ${STREAMING_DATA_PATH}"
echo "  SAVE_ROOT: ${SAVE_ROOT}"
echo "  RUN_NAME: ${RUN_NAME}"
echo "  LOAD_PATH: ${LOAD_PATH:-<none>}"
echo "  LOAD_STRATEGY: ${LOAD_STRATEGY:-never}"
echo "  OLMO_COMPILE: ${OLMO_COMPILE}"
echo "  PREFETCH_QUEUE_SIZE: ${PREFETCH_QUEUE_SIZE}"
echo "  VALIDATE_ENGRAM_XGRAMMATCH: ${VALIDATE_ENGRAM_XGRAMMATCH:-strict}"
echo ""

case "${VALIDATE_ENGRAM_XGRAMMATCH:-strict}" in
  1|strict)
    python scripts/validate_engram_xgrammatch_config.py --config "${CONFIG_PATH}" --strict-xgrammatch
    ;;
  variant)
    python scripts/validate_engram_xgrammatch_config.py --config "${CONFIG_PATH}" --allow-engram-variant
    ;;
  0|skip)
    echo "Skipping Engram xgrammatch validation"
    ;;
  *)
    echo "Unsupported VALIDATE_ENGRAM_XGRAMMATCH=${VALIDATE_ENGRAM_XGRAMMATCH}" >&2
    exit 2
    ;;
esac

echo "Starting training with config: ${CONFIG_PATH}"
bash ./scripts/train/olmo_train.sh "${CONFIG_PATH}"

echo "Engram X-grammatch training completed at: $(date)"
