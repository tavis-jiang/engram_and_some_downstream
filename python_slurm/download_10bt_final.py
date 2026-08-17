#!/usr/bin/env python3
"""
Download Fineweb-edu sample-10BT using datasets library with mirror
"""
import os
import sys

# 严格遵循 LUMIA 指南 - 清空代理并设置镜像
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('all_proxy', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_DATASETS_CACHE'] = '/data3/bcjiang/hf_cache'

print("="*60)
print("Downloading Fineweb-edu sample-10BT")
print("="*60)
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
print()

# 方法：直接用 datasets 库加载 sample-10BT
from datasets import load_dataset

print("Loading dataset... (this may take a few minutes)")
try:
    # 加载 sample-10BT subset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=False,
        download_mode="reuse_cache_if_exists"
    )

    print(f"✅ Dataset loaded!")
    print(f"   Number of examples: {len(dataset)}")
    print(f"   Features: {dataset.features}")

    # 保存到磁盘
    output_dir = "/data3/bcjiang/hf_datasets/fineweb-edu-sample10bt"
    print(f"\nSaving to {output_dir}...")
    dataset.save_to_disk(output_dir)

    print(f"✅ Saved successfully!")
    print(f"   Output: {output_dir}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Download complete!")
print("="*60)
