#!/usr/bin/env python3
"""
Convert Fineweb-edu parquet to Streaming format for X-gram training.
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from streaming import MDSWriter

# Configuration
DATA_DIR = "/data3/bcjiang/hf_datasets/fineweb-edu-sample10bt"
OUTPUT_DIR = "./data/fineweb_10b_streaming"
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_TOKENS = 10_000_000_000
SHARD_SIZE = 512_000_000


def prepare_data():
    """
    Main function to convert parquet to streaming format.
    """
    print("=" * 60)
    print("X-gram Data Preparation")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded!")

    # Find parquet files
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  - {os.path.basename(f)}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Define schema
    columns = {
        'tokens': 'ndarray:int32',
    }
    print(f"Schema: {columns}")

    # Convert
    print("\n" + "=" * 60)
    print("Converting to streaming format...")
    print("=" * 60)

    total_tokens = 0
    num_samples = 0

    with MDSWriter(
        out=OUTPUT_DIR,
        columns=columns,
        compression='zstd:12',
        hashes=['sha1', 'xxh3_64'],
        size_limit=SHARD_SIZE,
    ) as out:
        print("Writer initialized!")

        for parquet_file in parquet_files:
            print(f"\nProcessing: {os.path.basename(parquet_file)}")

            df = pd.read_parquet(parquet_file)
            print(f"  Rows: {len(df)}")

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
                text = row['text']
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens.append(tokenizer.eos_token_id)

                if total_tokens + len(tokens) > MAX_TOKENS:
                    break

                out.write({'tokens': np.array(tokens, dtype=np.int32)})
                total_tokens += len(tokens)
                num_samples += 1

            if total_tokens >= MAX_TOKENS:
                print("Reached MAX_TOKENS, stopping...")
                break

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Total samples: {num_samples:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()
