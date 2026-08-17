#!/bin/bash

#SBATCH --job-name=download_model
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 下载 SmolLM2-360M 模型
cd /data1/public/hf/
unset http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading SmolLM2-360M model..."
huggingface-cli download HuggingFaceTB/SmolLM2-360M \
    --local-dir HuggingFaceTB/SmolLM2-360M \
    --local-dir-use-symlinks False \
    --resume-download

echo "Model download complete!"
