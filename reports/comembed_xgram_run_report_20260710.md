# ComEmbed on X-GRAM Run Report

Date: 2026-07-10

## Goal

Reproduce the ComEmbed-on-X-GRAM setting from `METHOD.md` and get a training run that finishes cleanly without NaN or optimizer failure.

## Bottom line

- The original `2v` ComEmbed setting is still unstable in this codepath.
- A stable fallback was found and verified:
  - `mode: ComEmbed`
  - `comembed_variant: fa_add_qr`
  - `targets: [v]`
  - `v_layers: [0]`
  - `OLMO_USE_PLAIN_ADAMW=1`
  - `OLMO_COMPILE=0`
  - `OLMO_COMEMBED_LR=1e-6`
  - `OLMO_COMEMBED_WEIGHT_DECAY=0`
- This stable `1v` setup finished cleanly for 2, 10, 50, and 200 steps.

## Main configs

- Stable 200-step config: `configs/our_comembed_faaddqr_1v_smoke200.yaml`
- Stable 50-step config: `configs/our_comembed_faaddqr_1v_smoke50.yaml`
- Original unstable target config: `configs/our_comembed_faaddqr_2v.yaml`

## Launch recipe

```bash
sbatch -p RTX4090 --gres=gpu:1 \
  --job-name=comembed-1v200 \
  --export=ALL,OLMO_USE_PLAIN_ADAMW=1,OLMO_COMPILE=0,OLMO_COMEMBED_LR=1e-6,OLMO_COMEMBED_WEIGHT_DECAY=0 \
  python_slurm/train_comembed_smoke.slurm \
  configs/our_comembed_faaddqr_1v_smoke200.yaml
```

## Results

### Stable 2-step smoke

- Job: `83031`
- Log: `logs/20260710-182028_our_comembed_faaddqr_1v_smoke2_rank0.log`
- Final:
  - `step=2/2`
  - `train/CE loss=10.96`
  - `optim/total grad norm=0.0044`
  - no NaN

### Stable 10-step smoke

- Job: `83033`
- Log: `logs/20260710-182453_our_comembed_faaddqr_1v_smoke10_rank0.log`
- Final:
  - `step=10/10`
  - `train/CE loss=8.871`
  - `train/PPL=7,123`
  - no NaN

### Stable 50-step smoke

- Job: `83034`
- Log: `logs/20260710-182845_our_comembed_faaddqr_1v_smoke50_rank0.log`
- Final:
  - `step=50/50`
  - `train/CE loss=7.709`
  - `train/PPL=2,229`
  - `optim/total grad norm=0.7164`
  - `throughput/total tokens=204,800`
  - checkpoint saved successfully
  - training complete

### Stable 200-step run

- Job: `83040`
- Log: `logs/20260710-183759_our_comembed_faaddqr_1v_smoke200_rank0.log`
- Run dir: `runs/our_comembed_faaddqr_1v_smoke200-83040`
- Final:
  - `step=200/200`
  - `train/CE loss=6.999`
  - `train/PPL=1,095`
  - `optim/total grad norm=1.007`
  - `throughput/total tokens=819,200`
  - checkpoint saved successfully
  - training complete

## Failure summary for original 2v target

- Representative failing logs:
  - `logs/20260710-142036_our_comembed_faaddqr_2v_smoke2_rank0.log`
  - `logs/20260710-181604_our_comembed_faaddqr_2v_smoke2_rank0.log`
- Repeated pattern:
  - `2v` runs fail very early, typically around step 2.
  - The instability is tied to duplicated value-stream injection modules (`v_layers: [0, 0]` in smoke, repeated `2v` pattern in full config), not the basic training harness.

## Code/config state used for the stable run

- `OLMo-core/src/olmo_core/nn/embedding_injection/comembed.py`
- `OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py`
- `OLMo-core/src/olmo_core/nn/transformer/config.py`
- `scripts/train/olmo_train.py`

Stability-related behavior in the current tree:

- ComEmbed modules are built in `float32`
- ComEmbed shortconv is disabled by default internally for this path
- gate init is reduced
- row gate can be disabled
- output RMS norm is enabled
- optimizer override exists for ComEmbed parameter groups
- plain AdamW path is available

## Practical conclusion

If the immediate requirement is "get a working ComEmbed-on-X-GRAM run and collect results", use the `1v` recipe above.

If the requirement is "match the exact original 2v method setting from `METHOD.md`", that part is not solved yet. The current blocker is the duplicated `2v` injection path, not Slurm, data loading, or the general training pipeline.
