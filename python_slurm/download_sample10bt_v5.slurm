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

mkdir -p /data3/bcjiang/hf_datasets/fineweb-edu
cd /data3/bcjiang/hf_datasets

echo "Downloading Fineweb-edu (all files, will filter later)..."

# 下载整个仓库，不使用 include
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --local-dir fineweb-edu \
    --local-dir-use-symlinks False

echo "Download complete!"
echo "Checking what was downloaded..."
ls -lh fineweb-edu/ | head -20
find fineweb-edu/ -name "*sample*" -type d
