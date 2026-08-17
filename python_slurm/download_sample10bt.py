from huggingface_hub import snapshot_download
import os

print("Downloading Fineweb-edu sample-10BT...")

snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    local_dir="/data3/bcjiang/hf_datasets/HuggingFaceFW/fineweb-edu",
    local_dir_use_symlinks=False,
    allow_patterns=["sample-10BT/*"],
)

print("Download complete!")
