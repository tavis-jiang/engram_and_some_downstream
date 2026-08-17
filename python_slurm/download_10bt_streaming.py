from datasets import load_dataset
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Loading Fineweb-edu sample-10BT in streaming mode...")

# 使用 streaming 模式加载
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    split="train",
    streaming=True
)

print("Dataset stream opened!")

# 只取前 10000 条作为测试
output_dir = "/data3/bcjiang/hf_datasets/fineweb-edu-10bt-sample"
os.makedirs(output_dir, exist_ok=True)

print("Saving first 10000 samples...")
samples = []
for i, sample in enumerate(dataset):
    if i >= 10000:
        break
    samples.append(sample)
    if (i + 1) % 1000 == 0:
        print(f"  Downloaded {i + 1} samples...")

# 保存为 JSONL
import json
with open(f"{output_dir}/data.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print(f"Saved {len(samples)} samples to {output_dir}/data.jsonl")
print("Done!")
