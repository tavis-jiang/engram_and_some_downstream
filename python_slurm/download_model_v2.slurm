#!/bin/bash

#SBATCH --job-name=download_model
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 下载 SmolLM2-360M 模型到用户目录
mkdir -p /data3/bcjiang/hf_models
cd /data3/bcjiang/hf_models

unset http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading SmolLM2-360M model to /data3/bcjiang/hf_models/..."
huggingface-cli download HuggingFaceTB/SmolLM2-360M \
    --local-dir HuggingFaceTB/SmolLM2-360M \
    --local-dir-use-symlinks False \
    --resume-download

echo "Model download complete!"
ls -la HuggingFaceTB/SmolLM2-360M/
