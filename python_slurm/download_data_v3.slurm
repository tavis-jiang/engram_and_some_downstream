#!/bin/bash

#SBATCH --job-name=download_data
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 下载 Fineweb-edu 数据集到用户目录
mkdir -p /data3/bcjiang/hf_datasets
cd /data3/bcjiang/hf_datasets

unset http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading Fineweb-edu dataset..."
# 下载整个数据集（不指定 include）
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --local-dir HuggingFaceFW/fineweb-edu \
    --local-dir-use-symlinks False \
    --resume-download

echo "Dataset download complete!"
ls -la HuggingFaceFW/fineweb-edu/
