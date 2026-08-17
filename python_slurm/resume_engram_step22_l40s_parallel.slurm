#!/bin/bash
# =============================================================================
# Parallel L40S fallback for complete Engram reproduction.
#
# Purpose:
# - start earlier on node10 while the 16-GPU ADA6000 job waits for node8-node9;
# - keep the same model/data/training-token/optimizer-resume semantics;
# - write to an isolated run directory so it cannot corrupt the source step22 run.
#
# This branch is slower than the 16-GPU job, but useful as a progress fallback.
# =============================================================================
#SBATCH --job-name=engram-resume-l40s-parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=700G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --mail-user=913669023@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

echo "=========================================="
echo "Engram L40S Parallel Resume Job"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
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
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 1000))

SOURCE_RUN_DIR="./runs/engram-smollm2-360m-fineweb10b-rerun-normal-20260606-142032"
if [[ ! -f "${SOURCE_RUN_DIR}/step22/config.json" || ! -d "${SOURCE_RUN_DIR}/step22/model_and_optim" ]]; then
  echo "ERROR: expected complete step22 checkpoint under ${SOURCE_RUN_DIR}/step22" >&2
  exit 1
fi
export CKPT_SRC="${SOURCE_RUN_DIR}/dataloader_ckpt_for_resume/dataloader.ckpt"
if [[ ! -f "${CKPT_SRC}" ]]; then
  echo "ERROR: expected dataloader checkpoint at ${CKPT_SRC}" >&2
  exit 1
fi

export STREAMING_CACHE_BASE="/tmp/engram-smollm2-fineweb-cache"
export STREAMING_DATA_PATH="1 /tmp/engram-smollm2-fineweb-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=${PREFETCH_QUEUE_SIZE:-4}
export STREAMING_PACK_METHOD="native:truncate"

export SAVE_ROOT="./runs"
export RUN_NAME="engram-smollm2-360m-fineweb10b-rerun-l40s-parallel-${SLURM_JOB_ID}"
export LOAD_PATH="${SOURCE_RUN_DIR}"
export LOAD_STRATEGY="if_available"

export OLMO_PRE_TRAIN_CHECKPOINT=0
export OLMO_EPHEMERAL_SAVE_INTERVAL=none
export OLMO_SAVE_ASYNC=0
export OLMO_SKIP_MODEL_DIAGNOSTICS=1
export OLMO_FREEZE_BACKBONE=1
export OLMO_COMPILE=${OLMO_COMPILE:-0}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

echo "Starting L40S parallel Engram resume with config: configs/our_engram_config.yaml"
echo "Source run dir: ${SOURCE_RUN_DIR}"
echo "Target run name: ${RUN_NAME}"
echo "Dataloader checkpoint: ${CKPT_SRC}"
echo "Load strategy: ${LOAD_STRATEGY}"
echo "Checkpoint policy: ephemeral=${OLMO_EPHEMERAL_SAVE_INTERVAL}, save_async=${OLMO_SAVE_ASYNC}"

bash ./scripts/train/olmo_train.sh ./configs/our_engram_config.yaml

echo "Engram L40S parallel resume completed at: $(date)"
