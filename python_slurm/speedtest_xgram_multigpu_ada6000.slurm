#!/bin/bash
#SBATCH --job-name=xgram-ada-mgpu-speed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --partition=ADA6000
#SBATCH --mail-user=913669023@qq.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

SPEEDTEST_CONFIG="${SPEEDTEST_CONFIG:-./configs/speedtest_xgram_multigpu_ada6000_8gpu.yaml}"
SPEEDTEST_TAG="${SPEEDTEST_TAG:-$(basename "${SPEEDTEST_CONFIG}" .yaml)}"

echo "=========================================="
echo "X-gram Multi-GPU ADA6000 Speedtest"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Config: $SPEEDTEST_CONFIG"
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
export WANDB_MODE=disabled

export GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((7400 + SLURM_JOB_ID % 1000))

export STREAMING_DATA_PATH="1 /tmp/${SPEEDTEST_TAG}-cache-${SLURM_JOB_ID}-${SLURMD_NODENAME} ./data/fineweb_10b_streaming"
export STREAMING_TOKENIZER_MODEL="/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
export STREAMING_USE_TOKEN_COLUMN="tokens"
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=16
export STREAMING_PACK_METHOD="native:truncate"

export SAVE_ROOT="./runs_speedtest"
export RUN_NAME="${SPEEDTEST_TAG}-${SLURM_JOB_ID}"
unset LOAD_PATH
unset LOAD_STRATEGY
unset PROJ_ROOT
unset CKPT_SRC

echo ""
echo "Environment:"
echo "  Python: $(which python3)"
python3 - <<'PY'
import torch

print(f"  PyTorch: {torch.__version__}")
print(f"  PyTorch CUDA: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"  CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
PY
echo "  PYTHONPATH: $PYTHONPATH"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  STREAMING_DATA_PATH: $STREAMING_DATA_PATH"
echo "  SAVE_ROOT: $SAVE_ROOT"
echo "  RUN_NAME: $RUN_NAME"
echo ""

echo "Starting multi-GPU ADA6000 speedtest..."
bash ./scripts/train/olmo_train.sh "${SPEEDTEST_CONFIG}"

echo ""
echo "=========================================="
echo "Multi-GPU ADA6000 speedtest completed at: $(date)"
echo "=========================================="
