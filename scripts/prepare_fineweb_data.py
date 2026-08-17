#!/usr/bin/env python3
"""
Prepare Fineweb-edu 10B data for X-gram training.
准备 Fineweb-edu 10B 数据用于 X-gram 训练。

This script converts HuggingFace datasets to MosaicML Streaming format (.mds).
这个脚本将 HuggingFace 数据集转换为 MosaicML Streaming 格式 (.mds)。
"""

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from streaming import MDSWriter
from tqdm import tqdm


def prepare_fineweb_for_xgram(
    output_dir: str,
    model_name: str = "HuggingFaceTB/SmolLM2-360M",
    max_tokens: int = 10_000_000_000,  # 10B tokens
    shard_size: int = 512_000_000,  # 512MB per shard
):
    """
    Download Fineweb-edu sampled 10B, tokenize it, and convert to streaming format.
    下载 Fineweb-edu 采样 10B 数据，进行 tokenize，并转换为 streaming 格式。
    """
    print(f"Loading tokenizer from {model_name}...")
    print(f"从 {model_name} 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use EOS token as PAD if not present
    # 如果没有 PAD token，使用 EOS token 作为 PAD
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Fineweb-edu sampled 10B dataset...")
    print("加载 Fineweb-edu 采样 10B 数据集...")

    # Load the sampled 10B subset
    # 加载采样的 10B 子集
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    os.makedirs(output_dir, exist_ok=True)

    # Define the schema for streaming format
    # 定义 streaming 格式的 schema
    columns = {
        'tokens': 'ndarray:int32',  # Token IDs as int32 array
    }

    print(f"Converting to streaming format and saving to {output_dir}...")
    print(f"正在转换为 streaming 格式并保存到 {output_dir}...")
    print(f"Target: {max_tokens:,} tokens")
    print(f"目标: {max_tokens:,} tokens")

    total_tokens = 0
    num_samples = 0

    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression='zstd:12',  # Good compression ratio
        hashes=['sha1', 'xxh3_64'],  # For data integrity
        size_limit=shard_size,  # Shard size limit
    ) as out:
        # Tokenize the dataset
        # 对数据集进行 tokenize
        for example in tqdm(dataset, desc="Processing"):
            text = example["text"]

            # Tokenize with EOS token at the end
            # 在末尾添加 EOS token 进行 tokenize
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            # Add EOS token
            # 添加 EOS token
            tokens.append(tokenizer.eos_token_id)

            token_count = len(tokens)

            # Check if adding this sample exceeds our target
            # 检查添加这个样本是否会超过我们的目标
            if total_tokens + token_count > max_tokens:
                remaining = max_tokens - total_tokens
                if remaining > 0:
                    # Truncate the last sample to fit exactly
                    # 截断最后一个样本以精确匹配
                    tokens = tokens[:remaining]
                    out.write({'tokens': tokens})
                    total_tokens += remaining
                    num_samples += 1
                break

            # Write sample to streaming format
            # 将样本写入 streaming 格式
            out.write({'tokens': tokens})
            total_tokens += token_count
            num_samples += 1

    print(f"\n✅ Conversion complete! / 转换完成！")
    print(f"Total samples: {num_samples:,}")
    print(f"总样本数: {num_samples:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"总 token 数: {total_tokens:,}")
    print(f"Output directory: {output_dir}")
    print(f"输出目录: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Fineweb-edu data for X-gram training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/fineweb_10b_streaming",
        help="Output directory for streaming format data"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10_000_000_000,
        help="Maximum number of tokens to process (default: 10B)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("X-gram Data Preparation - Fineweb-edu 10B to Streaming Format")
    print("X-gram 数据准备 - Fineweb-edu 10B 转 Streaming 格式")
    print("=" * 70)

    prepare_fineweb_for_xgram(
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    print("\n" + "=" * 70)
    print("Next steps / 下一步:")
    print("1. Update your config file with the new data path")
    print("   更新配置文件中的新数据路径")
    print(f"   data.streaming_data_path: {args.output_dir}")
    print("2. Update the tokenizer path in config")
    print("   更新配置文件中的 tokenizer 路径")
    print(f"   data.streaming_tokenizer_model: {args.model_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
