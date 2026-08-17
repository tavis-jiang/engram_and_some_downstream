#!/bin/bash
#SBATCH --job-name=download_sample10bt
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

unset http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /data3/bcjiang/hf_datasets/fineweb-edu-sample10bt
cd /data3/bcjiang/hf_datasets/fineweb-edu-sample10bt

echo "Downloading Fineweb-edu sample-10BT data files..."

# 使用 huggingface-cli 直接下载 sample-10BT 目录下的文件
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --include "sample-10BT/**" \
    --local-dir . \
    --local-dir-use-symlinks False

echo "Download complete!"
ls -lh
