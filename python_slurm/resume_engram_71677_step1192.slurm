#!/bin/bash
# =============================================================================
# Resume formal Engram training (job 71677) from its step1192 checkpoint.
#
# Why this script exists:
# - 71677 has valid model+optim checkpoints at step596 and step1192.
# - The dataloader DuckDB state (/tmp .../olmo_stream_ckpts) was NOT persisted
#   when the previous launch ended, so a normal resume aborts with:
#     "Cannot open database ... dataloader.ckpt in read-only mode: does not exist"
# - Fix applied BEFORE submitting this job: step1192/train/rank*.pt had
#   data_loader.consumed_samples reset 305152 -> 0 (backups saved as
#   rank*.pt.bak.before_dataloader_reset). Same treatment step596 already had.
#   With consumed_samples==0, load_text_chunk_state() returns None early and
#   never touches DuckDB, so the dataloader cold-starts while model/optim/
#   global_step(=1192)/global_train_tokens_seen(~5.0B) are preserved.
# - Net effect: training continues from step1192 to step2385; some FineWeb
#   shards may be re-seen (streaming reshuffle), but the token budget and
#   optimizer trajectory stay aligned with baseline / X-gram (total 10B tokens).
#
# DO NOT set OLMO_FREEZE_BACKBONE=1 here. Engram must train the backbone
# from scratch, identical to baseline / X-gram. Freezing is what made 71624 fail.
#
# Submit: sbatch python_slurm/resume_engram_71677_step1192.slurm
# =============================================================================
#SBATCH --job-name=engram-resume-71677
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=L40S,ADA6000
#SBATCH --qos=normal
#SBATCH --mail-user=913669023@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

echo "=========================================="
echo "Engram Resume Job (71677 step1192 cold-dataloader)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"
echo "=========================================="

cd /home/bcjiang/X-gram
source /home/bcjiang/miniconda3/etc/profile.d/conda.sh
conda activate xgram_env

RUN_DIR="./runs/engram-smollm2-360m-fineweb10b-71677"
if [[ ! -f "${RUN_DIR}/step1192/.metadata.json" || ! -d "${RUN_DIR}/step1192/model_and_optim" ]]; then
  echo "ERROR: expected complete step1192 checkpoint under ${RUN_DIR}/step1192" >&2
  exit 1
fi

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

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((8000 + SLURM_JOB_ID % 1000))

export STREAMING_CACHE_BASE="/tmp/engram-smollm2-fineweb-cache"
export STREAMING_DATA_PATH="1 /tmp/engram-smollm2-fineweb-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=${PREFETCH_QUEUE_SIZE:-4}
export STREAMING_PACK_METHOD="native:truncate"

# Reuse the same save_folder so the trainer auto-loads step1192 from it.
export SAVE_ROOT="./runs"
export RUN_NAME="engram-smollm2-360m-fineweb10b-71677"
export LOAD_STRATEGY="if_available"

# Memory-safe checkpoint policy for the 9.9B-param Engram run (same as 71677).
export OLMO_PRE_TRAIN_CHECKPOINT=0
export OLMO_EPHEMERAL_SAVE_INTERVAL=none
export OLMO_SAVE_ASYNC=0
export OLMO_SKIP_MODEL_DIAGNOSTICS=1
export OLMO_COMPILE=${OLMO_COMPILE:-0}
# Engram from-scratch: never freeze the backbone.
export OLMO_FREEZE_BACKBONE=0
# 9.9B params: mb must stay at 1 (mb>=2 OOMs before step 1).
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

echo "Resuming Engram from ${RUN_DIR}/step1192 (dataloader cold-start, consumed_samples=0)"
echo "Load strategy: ${LOAD_STRATEGY}"
echo "Checkpoint policy: ephemeral=${OLMO_EPHEMERAL_SAVE_INTERVAL}, save_async=${OLMO_SAVE_ASYNC}, freeze=${OLMO_FREEZE_BACKBONE}"

bash ./scripts/train/olmo_train.sh ./configs/our_engram_config.yaml

echo "Engram resume completed at: $(date)"
