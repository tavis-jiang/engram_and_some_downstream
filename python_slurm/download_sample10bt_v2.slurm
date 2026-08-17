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
export HF_DATASETS_CACHE=/data3/bcjiang/hf_cache

cd /home/bcjiang/X-gram
python3 download_sample10bt_v2.py

echo "Done!"
ls -lh /data3/bcjiang/hf_datasets/
