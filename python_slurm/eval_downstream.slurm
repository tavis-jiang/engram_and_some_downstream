#!/bin/bash
# =============================================================================
# LUMIA 集群下游评测脚本（必须用 sbatch 提交，禁止在登录节点直接跑）
#
# 用法（在登录节点）：
#   sbatch python_slurm/eval_downstream.slurm <model> [tasks]
#     <model> : baseline | xgram | engram
#     [tasks] : 留空 = 默认全套；写 sciq = 只跑一门冒烟测试
#
# 例：
#   sbatch python_slurm/eval_downstream.slurm baseline sciq
#   sbatch python_slurm/eval_downstream.slurm xgram
#   sbatch python_slurm/eval_downstream.slurm engram
# =============================================================================
#SBATCH --job-name=xgram-eval
#SBATCH --partition=ADA6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

MODEL="${1:?need model: baseline|xgram|engram}"
TASKS="${2:-__FULL__}"

checkpoint_model_and_optim_path() {
  local root="$1"
  if [[ -d "${root}/model_and_optim" ]]; then
    printf '%s/model_and_optim' "$root"
  else
    printf '%s' "$root"
  fi
}

cd /home/bcjiang/X-gram
source /home/bcjiang/miniconda3/etc/profile.d/conda.sh
# ⚠️ 必须用 xgram_env，不要用 engram_x（那个环境 pip 装了 olmo_core，会报 ImportError）
conda activate xgram_env

case "$MODEL" in
  baseline)
    CONFIG="${BASELINE_CONFIG:-configs/faircompare_baseline_360m.yaml}"
    LOAD_PATH="${BASELINE_LOAD_PATH:-$(checkpoint_model_and_optim_path "${BASELINE_CHECKPOINT_ROOT:-runs/faircompare-baseline-360m-88161/step2385}")}"
    ;;
  xgram)
    CONFIG="${XGRAM_CONFIG:-configs/faircompare_xgram_360m.yaml}"
    LOAD_PATH="${XGRAM_LOAD_PATH:-$(checkpoint_model_and_optim_path "${XGRAM_CHECKPOINT_ROOT:-runs/faircompare-xgram-360m-88162/step2385}")}"
    ;;
  engram)
    CONFIG="${ENGRAM_CONFIG:-configs/faircompare_engram_360m.yaml}"
    LOAD_PATH="${ENGRAM_LOAD_PATH:-$(checkpoint_model_and_optim_path "${ENGRAM_CHECKPOINT_ROOT:-runs/faircompare-engram-360m-88169/step2385}")}"
    ;;
  *) echo "unknown model: $MODEL"; exit 1 ;;
esac

CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-$(basename "$(dirname "$LOAD_PATH")")}"
if [[ "$CHECKPOINT_LABEL" == "model_and_optim" ]]; then
  CHECKPOINT_LABEL="$(basename "$(dirname "$(dirname "$LOAD_PATH")")")"
fi

# 代码路径（源码版 olmo_core / ubdataloader / olmo_eval）
export PYTHONPATH="/home/bcjiang/X-gram/OLMo-core/src:/home/bcjiang/X-gram/packages/ubdataloader/src:/home/bcjiang/X-gram/packages/olmo_in_loop_evals/src:${PYTHONPATH:-}"

export OLMO_SHARED_FS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OLMO_EVAL_LOAD_MODEL_ONLY="${OLMO_EVAL_LOAD_MODEL_ONLY:-1}"
# 3090/4090 队列必须开；ADA6000/A100 留着无妨
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 让 trainer 以为还没训完，否则它会判定"训练已完成"直接退出、不评测
export OLMO_LR_DECAY_TOKENS=20000000000

RUN_TAG=$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}
mkdir -p logs runs_eval reports out
LOG="logs/downstream_eval_${MODEL}_${CHECKPOINT_LABEL}_${TASKS//,/_}_${RUN_TAG}.log"
[[ "$TASKS" == "__FULL__" ]] && LOG="logs/downstream_eval_${MODEL}_${CHECKPOINT_LABEL}_full_${RUN_TAG}.log"
RUN_CKPT="/tmp/xgram_eval_${MODEL}_${RUN_TAG}.ckpt"
rm -f "$RUN_CKPT" "$RUN_CKPT".*

MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1) + 30000 ))

# 评测专用 override（dotlist 直接传给 olmo_train.py；它用 parse_known_args 收为 overrides）
EVAL_OVERRIDES=(
  'trainer.callbacks.checkpointer.enabled=false'
  'trainer.callbacks.downstream_evaluator.enabled=true'
  'trainer.callbacks.downstream_evaluator.cancel_after_first_eval=true'
)
[[ "$TASKS" != "__FULL__" ]] && EVAL_OVERRIDES+=("trainer.callbacks.downstream_evaluator.tasks=[${TASKS}]")
if [[ "$MODEL" == "xgram" && "${XGRAM_EVAL_DISABLE_LAMBDA_WARMUP:-1}" == "1" ]]; then
  # We intentionally load model_and_optim for eval to avoid restoring the training
  # data loader. For post-warmup X-gram checkpoints such as step2385, disable
  # lambda warmup here so injection runs at the same full strength as the saved step.
  EVAL_OVERRIDES+=('model.embedding_injection.lambda_warmup_enabled=false')
fi
if [[ "$MODEL" == "engram" && "${ENGRAM_EVAL_DISABLE_LAMBDA_WARMUP:-1}" == "1" ]]; then
  # Same reason as X-gram: model-only startup eval begins at step 0.
  # Post-warmup Engram checkpoints must be evaluated at full injection strength.
  EVAL_OVERRIDES+=('model.embedding_injection.lambda_warmup_enabled=false')
fi
if [[ "$MODEL" == "engram" && "${ENGRAM_LEGACY_H_PATH:-0}" == "1" ]]; then
  # Compatibility for older H-path Engram checkpoints saved before external
  # lambda/depth/warmup gates were added to the checkpointed module graph.
  EVAL_OVERRIDES+=('model.embedding_injection.engram_legacy_h_path=true')
fi
if [[ -n "${EXTRA_EVAL_OVERRIDES:-}" ]]; then
  read -r -a _extra_eval_overrides <<< "$EXTRA_EVAL_OVERRIDES"
  EVAL_OVERRIDES+=("${_extra_eval_overrides[@]}")
fi

echo "== eval model=$MODEL tasks=$TASKS job=$SLURM_JOB_ID node=$SLURMD_NODENAME =="
echo "== loading model weights from: $LOAD_PATH =="
echo "== checkpoint label: $CHECKPOINT_LABEL =="
echo "== model-only checkpoint load: $OLMO_EVAL_LOAD_MODEL_ONLY =="
[[ "$MODEL" == "xgram" && "${XGRAM_EVAL_DISABLE_LAMBDA_WARMUP:-1}" == "1" ]] && echo "== xgram eval override: model.embedding_injection.lambda_warmup_enabled=false =="
[[ "$MODEL" == "engram" && "${ENGRAM_EVAL_DISABLE_LAMBDA_WARMUP:-1}" == "1" ]] && echo "== engram eval override: model.embedding_injection.lambda_warmup_enabled=false =="
[[ "$MODEL" == "engram" && "${ENGRAM_LEGACY_H_PATH:-0}" == "1" ]] && echo "== engram eval override: model.embedding_injection.engram_legacy_h_path=true =="
[[ -n "${EXTRA_EVAL_OVERRIDES:-}" ]] && echo "== extra eval overrides: $EXTRA_EVAL_OVERRIDES =="

# 用 srun + torchrun 在计算节点起 1 个进程（单卡评测足够），直接调 olmo_train.py
srun --label torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
  scripts/train/olmo_train.py \
  --config "$CONFIG" \
  --launcher-mode shell_compatible \
  --run-name "eval-${MODEL}-${CHECKPOINT_LABEL}-${RUN_TAG}" \
  --save-root ./runs_eval \
  --load-path "$LOAD_PATH" \
  --load-strategy always \
  --streaming-ckpt-path "$RUN_CKPT" \
  "${EVAL_OVERRIDES[@]}" \
  2>&1 | tee "$LOG"

echo "== eval done, log=$LOG =="
