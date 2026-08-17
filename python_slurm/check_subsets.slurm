#!/bin/bash
#SBATCH --job-name=check_subsets
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

unset http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com

cd /home/bcjiang/X-gram
python3 check_subsets.py
