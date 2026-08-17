#!/usr/bin/env python3
"""
Download HuggingFace dataset using mirror
"""
import os
import sys

# 严格遵循 LUMIA 指南
# Strictly follow LUMIA guide
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('all_proxy', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 验证代理已清空
print("Checking proxy settings:")
print(f"  http_proxy: {os.environ.get('http_proxy', 'None')}")
print(f"  https_proxy: {os.environ.get('https_proxy', 'None')}")
print(f"  all_proxy: {os.environ.get('all_proxy', 'None')}")
print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")

from huggingface_hub import snapshot_download

print("\nDownloading Fineweb-edu sample-10BT...")
print("This may take a while...\n")

try:
    snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        local_dir="/data3/bcjiang/hf_datasets/fineweb-edu",
        local_dir_use_symlinks=False,
        allow_patterns=["sample-10BT/**/*"],  # 使用 ** 递归匹配
        resume_download=True,
    )
    print("\n✅ Download complete!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
