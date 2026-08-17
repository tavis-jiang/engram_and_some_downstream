#!/bin/bash
#SBATCH --job-name=baseline-smollm2-resume
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
echo "Baseline Resume Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
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
export MASTER_PORT=$((7100 + SLURM_JOB_ID % 1000))

export STREAMING_CACHE_BASE="/tmp/baseline-smollm2-fineweb-cache"
export STREAMING_DATA_PATH="1 /tmp/baseline-smollm2-fineweb-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=16
export STREAMING_PACK_METHOD="native:truncate"

export SAVE_ROOT="./runs"
export RUN_NAME="baseline-smollm2-360m-fineweb10b-64984"
export LOAD_PATH="./runs/baseline-smollm2-360m-fineweb10b-64984"
export LOAD_STRATEGY="if_available"
export CKPT_SRC="./runs/baseline-smollm2-360m-fineweb10b-64984/dataloader_ckpt_for_resume/dataloader.ckpt"

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
echo "  LOAD_PATH: $LOAD_PATH"
echo "  LOAD_STRATEGY: $LOAD_STRATEGY"
echo "  CKPT_SRC: $CKPT_SRC"
echo ""

echo "Starting baseline resume..."
bash ./scripts/train/olmo_train.sh ./configs/our_baseline_config.yaml

echo ""
echo "=========================================="
echo "Baseline resume job completed at: $(date)"
echo "=========================================="
