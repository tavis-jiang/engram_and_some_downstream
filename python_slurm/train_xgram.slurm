#!/bin/bash
#SBATCH --job-name=xgram-smollm2-training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=RTX3090
#SBATCH --mail-user=913669023@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

echo "=========================================="
echo "X-gram Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Change to project directory
cd /home/bcjiang/X-gram

# Activate the training environment explicitly so sbatch does not depend on
# the submitting shell's PATH.
source /home/bcjiang/miniconda3/etc/profile.d/conda.sh
conda activate xgram_env

# Set Python path to include OLMo-core and ubdataloader
export CODE_ROOT="/home/bcjiang/X-gram"
export OLMo_CORE_SRC="${CODE_ROOT}/OLMo-core/src"
export UBDATALOADER_SRC="${CODE_ROOT}/packages/ubdataloader/src"
export PYTHONPATH="${OLMo_CORE_SRC}:${UBDATALOADER_SRC}:${PYTHONPATH:-}"

# Training environment settings
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OLMO_SHARED_FS=1
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
export NCCL_ALGO=^NVLS
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Distributed training settings
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((6000 + SLURM_JOB_ID % 1000))

# Data paths
export STREAMING_CACHE_BASE="/tmp/xgram-smollm2-fineweb-cache"
export STREAMING_DATA_PATH="1 /tmp/xgram-smollm2-fineweb-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=16
export STREAMING_PACK_METHOD="native:truncate"

# Checkpoint and logging
export SAVE_ROOT="./runs"
export RUN_NAME="xgram-smollm2-360m-fineweb10b-${SLURM_JOB_ID}"

# Optional: Weights & Biases logging
# export WANDB_API_KEY="your-api-key"
# export WANDB_PROJECT="xgram"
# export WANDB_ENTITY="your-username"
# export WANDB_MODE="online"  # or "offline" / "disabled"

echo ""
echo "Environment:"
echo "  Python: $(which python3)"
python3 - <<'PY'
import torch

print(f"  PyTorch: {torch.__version__}")
print(f"  PyTorch CUDA: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA device count: {torch.cuda.device_count()}")
PY
echo "  PYTHONPATH: $PYTHONPATH"
echo "  STREAMING_DATA_PATH: $STREAMING_DATA_PATH"
echo "  SAVE_ROOT: $SAVE_ROOT"
echo "  RUN_NAME: $RUN_NAME"
echo ""

# Launch training
echo "Starting training..."
bash ./scripts/train/olmo_train.sh ./configs/our_xgram_config.yaml

echo ""
echo "=========================================="
echo "Training job completed at: $(date)"
echo "=========================================="
