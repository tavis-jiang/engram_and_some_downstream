#!/bin/bash
#SBATCH --job-name=engram-eval-monitor
#SBATCH --partition=ADA6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/home/bcjiang/X-gram/out/%j.out
#SBATCH --error=/home/bcjiang/X-gram/out/%j.err

set -euo pipefail

cd /home/bcjiang/X-gram
source /home/bcjiang/miniconda3/etc/profile.d/conda.sh
conda activate xgram_env

export PYTHONHASHSEED=0

ROOT="${1:-runs/faircompare-engram-360m-88169}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

echo "== monitor engram checkpoints root=${ROOT} interval=${INTERVAL_SECONDS}s job=${SLURM_JOB_ID} =="
bash scripts/monitor_engram_checkpoint_evals.sh "$ROOT"
