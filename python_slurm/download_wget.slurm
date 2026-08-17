#!/bin/bash
#SBATCH --job-name=download_wget
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

unset http_proxy
unset https_proxy
unset all_proxy

mkdir -p /data3/bcjiang/hf_datasets/fineweb-edu-sample10bt
cd /data3/bcjiang/hf_datasets/fineweb-edu-sample10bt

echo "Downloading Fineweb-edu sample-10BT via wget..."

# 基础URL
BASE_URL="https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample-10BT"

# 下载 README
echo "Downloading README.md..."
wget -q --show-progress "${BASE_URL}/README.md" -O README.md || echo "README not found"

# 下载 parquet 文件（sample-10BT 有多个 parquet 文件）
# 先尝试下载前几个文件
echo "Downloading parquet files..."
for i in $(seq -w 000 099); do
    FILE="train-${i}-of-00144.parquet"
    echo "Downloading ${FILE}..."
    wget -q --show-progress "${BASE_URL}/${FILE}" -O "${FILE}" || echo "Failed: ${FILE}"

    # 如果下载失败，可能是文件不存在，停止循环
    if [ ! -f "${FILE}" ]; then
        echo "File not found, stopping..."
        break
    fi
done

echo "Download complete!"
ls -lh
