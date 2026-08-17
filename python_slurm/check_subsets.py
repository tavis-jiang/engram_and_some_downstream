from datasets import get_dataset_config_names
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Checking available configs for HuggingFaceFW/fineweb-edu...")
configs = get_dataset_config_names("HuggingFaceFW/fineweb-edu")
print(f"Available configs: {configs}")
