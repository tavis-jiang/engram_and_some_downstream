#!/bin/bash
# =============================================================================
# E-ngram 续跑脚本 (从 71677 run 目录的最后一个完整 checkpoint 接着训)
#
# 用法:
#   sbatch python_slurm/resume_engram_71677.slurm
#
# 行为:
#   - 复用原 run 目录: runs/engram-smollm2-360m-fineweb10b-71677
#     (Slurm RUN_NAME override 保证不会因为新 JobID 又开一个新 run 文件夹)
#   - load_strategy=if_available 会让 OLMo 自动找到最后一个完整 checkpoint
#     (会跳过半截写入的目录)
#   - 若 /tmp 下的 dataloader DuckDB ckpt 已消失，则自动退回 fresh dataloader
#     cold-start，同时保留 model/optim/global_step
#
# 何时用:
#   - 71677 因 72h 时间到被 Slurm killed
#   - 或 71677 OOM 后, 修复内存问题, 想从最后一个完整 checkpoint 继续
#   - 或本作业自身被 killed, 再次提交本脚本即可继续 (RUN_NAME 不变)
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
echo "Engram Resume Job (from 71677)"
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
export OLMO_FREEZE_BACKBONE=${OLMO_FREEZE_BACKBONE:-0}
export OLMO_ALLOW_MISSING_DATALOADER_CKPT=${OLMO_ALLOW_MISSING_DATALOADER_CKPT:-1}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

# === 续跑核心: 复用 71677 run 目录 + if_available 策略 ===
export SAVE_ROOT="./runs"
export RUN_NAME="engram-smollm2-360m-fineweb10b-71677"
export LOAD_PATH="./runs/engram-smollm2-360m-fineweb10b-71677"
export LOAD_STRATEGY="if_available"

CONFIG_PATH="${1:-./configs/our_engram_config.yaml}"

echo "Resuming from: ${LOAD_PATH}"
echo "Run name (preserved): ${RUN_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Latest checkpoints in run dir:"
ls -1 "${LOAD_PATH}/" | grep -E "^step[0-9]+" | sort -V | tail -5 || true
echo "=========================================="

bash ./scripts/train/olmo_train.sh "${CONFIG_PATH}"

echo "Engram resume completed at: $(date)"
