# SmolLM2 FineWeb Reproduction

This workspace is running a matched X-gram vs. no-injection control on the
local FineWeb 10B streaming shards.

## Data And Assets

- Streaming data: `./data/fineweb_10b_streaming`
- Tokenizer: `/home/bcjiang/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49`
- X-gram token map: `./assets/token_maps/injection_token_map_smollm2_fineweb_alpha0_5_M32_Cap75968_mc3_mwd0_8.npz`
- FineWeb token frequencies: `./assets/frequency_stats/smollm2_fineweb_10b_token_freqs.npy`

## Matched Runs

| Variant | Config | Slurm script | Job | Run folder |
| --- | --- | --- | --- | --- |
| X-gram value injection | `configs/our_xgram_config.yaml` | `python_slurm/train_xgram.slurm` | `64971` | `runs/xgram-smollm2-360m-fineweb10b-64971` |
| No-injection baseline | `configs/our_baseline_config.yaml` | `python_slurm/train_baseline.slurm` | `64984` | `runs/baseline-smollm2-360m-fineweb10b-64984` |

Both runs use:

- SmolLM2-360M-style 32-layer backbone: hidden size 960, FFN size 2560, 15 attention heads, 3 query groups.
- Sequence length 8192.
- Micro batch size 4.
- Global batch size 512 samples.
- Training budget 10,000,000,000 tokens, resolved by the trainer to 2385 steps.
- Learning rate 0.0003.
- Streaming token column `tokens` with `native:truncate` packing.

The X-gram run differs only by enabling value-stream X-gram injection with 20
value views over layers 0 through 9, ShortConv kernels `[3, 5, 7]`, and the
SmolLM2/FineWeb hash token map.

## Monitoring

```bash
python3 scripts/reproduction_status.py \
  --job-id 64971 \
  --job-id 64984 \
  --log logs/20260530-115632_our_xgram_config_rank0.log \
  --log logs/20260530-150549_our_baseline_config_rank0.log \
  --run-dir runs/xgram-smollm2-360m-fineweb10b-64971 \
  --run-dir runs/baseline-smollm2-360m-fineweb10b-64984
```

Useful raw checks:

```bash
squeue -u bcjiang
tail -f logs/20260530-115632_our_xgram_config_rank0.log
tail -f logs/20260530-150549_our_baseline_config_rank0.log
```

## Relaunch Commands

```bash
sbatch python_slurm/train_xgram.slurm
sbatch python_slurm/train_baseline.slurm
```

Each Slurm script gives the job a unique run folder and a unique local streaming
cache path derived from the Slurm job ID and node name.
