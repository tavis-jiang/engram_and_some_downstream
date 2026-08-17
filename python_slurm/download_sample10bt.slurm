#!/bin/bash
  #SBATCH --job-name=download_sample10bt
  #SBATCH --partition=RTX3090
  #SBATCH --nodes=1
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=4
  #SBATCH --gres=gpu:0
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  cd /data3/bcjiang/hf_datasets
  unset http_proxy https_proxy all_proxy
  export HF_ENDPOINT=https://hf-mirror.com

  echo "Downloading Fineweb-edu sample-10BT..."
  python3 /home/bcjiang/X-gram/download_sample10bt.py
  echo "Done!"
