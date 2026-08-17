#!/bin/bash
#SBATCH --job-name=download_10bt
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 清空代理
unset http_proxy
unset https_proxy
unset all_proxy

echo "Starting download..."
cd /home/bcjiang/X-gram
python3 download_10bt_final.py

echo "Done!"
