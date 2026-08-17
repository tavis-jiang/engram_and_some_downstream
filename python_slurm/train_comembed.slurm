#!/bin/bash
#SBATCH --job-name=comembed-training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=RTX3090

set -euo pipefail

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
export WANDB_MODE=offline

export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((6200 + SLURM_JOB_ID % 1000))
export UBDATALOADER_MASTER_PORT=$((24100 + SLURM_JOB_ID % 1000))

export STREAMING_CACHE_BASE="/tmp/xgram-smollm2-fineweb-cache"
export STREAMING_DATA_PATH="1 /tmp/xgram-smollm2-fineweb-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=16
export STREAMING_PACK_METHOD="native:truncate"

CONFIG_PATH="${1:-./configs/our_comembed_faqr_2v.yaml}"
CONFIG_STEM="$(basename "${CONFIG_PATH}" .yaml)"
export SAVE_ROOT="./runs"
export RUN_NAME="${CONFIG_STEM}-${SLURM_JOB_ID}"

echo "ComEmbed training job ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_PATH}"
echo "Run: ${RUN_NAME}"

bash ./scripts/train/olmo_train.sh "${CONFIG_PATH}"
