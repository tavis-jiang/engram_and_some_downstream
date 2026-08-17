#!/bin/bash
# =============================================================================
# E-ngram 基线训练（LUMIA 集群，sbatch 提交）
# 与 train_baseline.slurm / train_xgram.slurm 结构一致，只换 config 与 RUN_NAME。
# 提交：sbatch python_slurm/train_engram.slurm
# 注意：跑前需先建好 configs/our_engram_config.yaml
# =============================================================================
#SBATCH --job-name=engram-smollm2-training
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
echo "Engram Training Job"
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OLMO_SHARED_FS=1
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
export NCCL_ALGO=^NVLS
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 若改用 RTX3090/RTX4090 队列，取消下面两行注释：
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

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
export OLMO_PRE_TRAIN_CHECKPOINT=${OLMO_PRE_TRAIN_CHECKPOINT:-0}
export OLMO_COMPILE=${OLMO_COMPILE:-0}
export OLMO_EPHEMERAL_SAVE_INTERVAL=${OLMO_EPHEMERAL_SAVE_INTERVAL:-none}
export OLMO_SAVE_ASYNC=${OLMO_SAVE_ASYNC:-0}
export OLMO_SKIP_MODEL_DIAGNOSTICS=${OLMO_SKIP_MODEL_DIAGNOSTICS:-1}
# Engram 必须 from-scratch 训练 backbone, 与 baseline / X-gram 对齐。
# 不要在这里强制 freeze backbone, 否则随机初始化主干 + 冻结会让 CE loss 卡在 ~7。
export OLMO_FREEZE_BACKBONE=${OLMO_FREEZE_BACKBONE:-0}
# 9.9B 参数下 mb 只能压到 1, mb>=2 会在 step 1 之前就 OOM。
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

export SAVE_ROOT="./runs"
export RUN_NAME="${RUN_NAME:-engram-smollm2-360m-fineweb10b-${SLURM_JOB_ID}}"
CONFIG_PATH="${1:-./configs/our_engram_config.yaml}"

echo "Starting training with config: ${CONFIG_PATH}"
bash ./scripts/train/olmo_train.sh "${CONFIG_PATH}"

echo "Engram training completed at: $(date)"
