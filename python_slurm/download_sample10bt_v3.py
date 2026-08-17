from datasets import load_dataset
import os

# 使用 hf-mirror.com 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Loading Fineweb-edu sample-10BT from hf-mirror.com...")

# 直接加载 sample-10BT 子集
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    split="train",
    streaming=False
)

print(f"Dataset loaded! Size: {len(dataset)}")

# 保存到本地
output_dir = "/data3/bcjiang/hf_datasets/fineweb-edu-sample10bt"
dataset.save_to_disk(output_dir)
print(f"Saved to {output_dir}")
print("Download complete!")
