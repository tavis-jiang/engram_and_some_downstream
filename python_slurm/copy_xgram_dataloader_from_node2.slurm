#!/bin/bash
#SBATCH --job-name=copy-xgram-loader
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --partition=RTX3090
#SBATCH --nodelist=node2
#SBATCH --output=/home/bcjiang/X-gram/logs/copy_xgram_dataloader_%j.out
#SBATCH --error=/home/bcjiang/X-gram/logs/copy_xgram_dataloader_%j.err

set -euo pipefail

SRC_DIR="/tmp/olmo_stream_ckpts/run-20260530-115632-28728-2623406"
DST_DIR="/home/bcjiang/X-gram/runs/xgram-smollm2-360m-fineweb10b-64971/dataloader_ckpt_for_resume"

echo "Node: $(hostname)"
echo "Start: $(date)"
mkdir -p "${DST_DIR}"
ls -lh "${SRC_DIR}"/dataloader.ckpt*
cp -p "${SRC_DIR}"/dataloader.ckpt* "${DST_DIR}/"
ls -lh "${DST_DIR}"
echo "Done: $(date)"
