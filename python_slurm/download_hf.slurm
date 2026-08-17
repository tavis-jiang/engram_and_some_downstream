#!/bin/bash
#SBATCH --job-name=download_hf
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 清空代理 (遵循 LUMIA 指南)
unset http_proxy
unset https_proxy
unset all_proxy

# 验证代理已清空
echo "Proxy settings:"
echo "  http_proxy: '$http_proxy'"
echo "  https_proxy: '$https_proxy'"
echo "  all_proxy: '$all_proxy'"

cd /home/bcjiang/X-gram
python3 download_hf_data.py

echo "Done!"
