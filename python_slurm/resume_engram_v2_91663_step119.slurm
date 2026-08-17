#!/bin/bash
# Resume the interrupted Engram v2 run 91663 from its last complete checkpoint.
# The original 91663 job was requeued after step160; step119 is the latest
# checkpoint present on disk.
#SBATCH --job-name=engram-v2-resume119
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

CONFIG_PATH="./configs/engram_v2_360m.yaml"
RUN_DIR="./runs/engram-v2-91663"
RESUME_STEP="${RUN_DIR}/step119"

if [[ ! -d "${RESUME_STEP}/model_and_optim" ]]; then
  echo "ERROR: missing checkpoint ${RESUME_STEP}/model_and_optim" >&2
  exit 1
fi

echo "=========================================="
echo "Engram V2 Resume Job"
echo "Config: ${CONFIG_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "Resume checkpoint: ${RESUME_STEP}"
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

export STREAMING_DATA_PATH="1 /tmp/engram-v2-smollm2-fineweb-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE="${PREFETCH_QUEUE_SIZE:-16}"
export STREAMING_PACK_METHOD="native:truncate"

export SAVE_ROOT="./runs"
export RUN_NAME="engram-v2-91663"
export LOAD_PATH="${RUN_DIR}"
export LOAD_STRATEGY="if_available"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OLMO_PRE_TRAIN_CHECKPOINT="${OLMO_PRE_TRAIN_CHECKPOINT:-0}"
export OLMO_COMPILE="${OLMO_COMPILE:-0}"
export OLMO_EPHEMERAL_SAVE_INTERVAL="${OLMO_EPHEMERAL_SAVE_INTERVAL:-none}"
export OLMO_SAVE_ASYNC="${OLMO_SAVE_ASYNC:-0}"
export OLMO_SKIP_MODEL_DIAGNOSTICS="${OLMO_SKIP_MODEL_DIAGNOSTICS:-1}"
export OLMO_FREEZE_BACKBONE="${OLMO_FREEZE_BACKBONE:-0}"
export OLMO_ALLOW_MISSING_DATALOADER_CKPT="${OLMO_ALLOW_MISSING_DATALOADER_CKPT:-1}"

echo "Load path: ${LOAD_PATH}"
echo "Load strategy: ${LOAD_STRATEGY}"
echo "Run name (preserved): ${RUN_NAME}"
echo "Latest checkpoints:"
find "${RUN_DIR}" -maxdepth 1 -type d -name 'step*' -printf '%f\n' | sort -V | tail -5
echo "=========================================="

bash ./scripts/train/olmo_train.sh "${CONFIG_PATH}"

echo "Engram v2 resume completed at: $(date)"
