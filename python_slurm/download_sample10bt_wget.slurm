#!/bin/bash
#SBATCH --job-name=download_10bt
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

echo "Downloading Fineweb-edu sample/10BT via wget..."
echo "Files will be saved to: $(pwd)"
echo ""

# 基础URL
BASE_URL="https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"

# 下载所有 parquet 文件 (000_00000.parquet 到 004_00000.parquet = 约 10GB)
# 总共5个文件，每个约2GB
for i in 000 001 002 003 004; do
    FILE="${i}_00000.parquet"
    URL="${BASE_URL}/${FILE}"

    if [ -f "${FILE}" ]; then
        echo "✓ ${FILE} already exists, skipping..."
        continue
    fi

    echo "Downloading ${FILE}..."
    echo "  URL: ${URL}"
    wget --progress=bar:force -O "${FILE}.tmp" "${URL}"

    if [ $? -eq 0 ]; then
        mv "${FILE}.tmp" "${FILE}"
        echo "  ✓ ${FILE} downloaded successfully!"
    else
        echo "  ✗ Failed to download ${FILE}"
        rm -f "${FILE}.tmp"
    fi
    echo ""
done

echo ""
echo "Download complete!"
echo "Downloaded files:"
ls -lh *.parquet 2>/dev/null || echo "No .parquet files found"
