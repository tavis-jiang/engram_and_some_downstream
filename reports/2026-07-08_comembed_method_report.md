# ComEmbed on X-GRAM Report

Date: 2026-07-08

## Summary

I implemented the METHOD.md ComEmbed integration on top of the current X-GRAM codebase and ran a set of smoke experiments.

Current status:

- ComEmbed code integration is complete enough to build, initialize, and train.
- `fa_add_qr`, `fa_qr`, and `fa_norm_qr` all pass build/init and 1-step execution, but all fail 2-step smoke with divergence at step 2.
- The formal 10B-token `fa_qr` training job was submitted once and then cancelled before start because the smoke stability gate failed.

Conclusion:

- Engineering integration: success.
- Main METHOD.md row-memory variant (`fa_qr` / QR add-product reverse row memory): not stable enough yet for full training.
- The adjacent ComEmbed controls (`fa_add_qr`, `fa_norm_qr`) are also not stable enough yet for full training in the current X-GRAM training stack.
- Full formal experiment result: not available, because it would have been invalid to let the long run start after the smoke failures.

## Code Changes

Implemented:

- `OLMo-core/src/olmo_core/nn/embedding_injection/comembed.py`
  - `QRAddProductResidualLookup`
  - `QRRowMemory`
  - `QRAddNormProductRowMemory`
  - `QRAddResidualRowMemory`
  - `FrequencyAwareQRLookup`
  - `ContextMaskNgramPQLookup`
- `OLMo-core/src/olmo_core/nn/embedding_injection/ops/hash_injection.py`
  - added `HashTokenRouter`
- `OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py`
  - added `build_comembed_modules()`
- `OLMo-core/src/olmo_core/nn/transformer/config.py`
  - added `ComEmbed` config fields and parameter counting
- `OLMo-core/src/olmo_core/nn/transformer/model.py`
  - added `ComEmbed` mode dispatch
  - reused existing X-GRAM runtime path
- `scripts/train/olmo_train.py`
  - added `ComEmbed` mode parsing
  - added optimizer group override for ComEmbed injection params

Configs added:

- `configs/our_comembed_faaddqr_2v_smoke.yaml`
- `configs/our_comembed_faaddqr_2v.yaml`
- `configs/our_comembed_faqr_2v_smoke.yaml`
- `configs/our_comembed_faqr_2v_smoke2.yaml`
- `configs/our_comembed_faqr_2v.yaml`
- `configs/our_comembed_fanormqr_2v.yaml`
- `configs/our_comembed_qr_rev_1v.yaml`

Slurm scripts added:

- `python_slurm/train_comembed_smoke.slurm`
- `python_slurm/train_comembed.slurm`

## Experiment Log

### 1. `fa_add_qr` smoke

Job:

- `80904`

Config:

- `configs/our_comembed_faaddqr_2v_smoke.yaml`

Artifacts:

- `runs/our_comembed_faaddqr_2v_smoke-80904/config.json`
- `runs/our_comembed_faaddqr_2v_smoke-80904/step0/config.json`
- `runs/our_comembed_faaddqr_2v_smoke-80904/step1/config.json`
- `logs/20260707-170019_our_comembed_faaddqr_2v_smoke_rank0.log`

Observed:

- training completed
- `optim/step skipped=0.0`
- `train/CE loss=11.01`
- step 0 and step 1 checkpoints saved

Note:

- `optim/total grad norm=nan` was logged, but the run still completed and did not trigger the trainer's `nan loss` guard in 1-step smoke.

### 2. `fa_qr` smoke: first failure

Job:

- `80906`

Config:

- `configs/our_comembed_faqr_2v_smoke.yaml`

Observed:

- failed during FSDP setup
- root cause: scalar `beta_logit` not supported by `fully_shard`

Fix applied:

- changed `beta_logit` from scalar tensor to shape `[1]`

### 3. `fa_qr` smoke: 1-step after FSDP fix

Job:

- `80908`

Config:

- `configs/our_comembed_faqr_2v_smoke.yaml`

Artifacts:

- `runs/our_comembed_faqr_2v_smoke-80908/config.json`
- `logs/20260707-171933_our_comembed_faqr_2v_smoke_rank0.log`

Observed:

- training completed
- `optim/step skipped=0.0`
- `train/CE loss=10.96`

Interpretation:

- the builder/runtime path is functional
- 1-step smoke is not enough to establish stability

### 4. `fa_qr` smoke: 2-step instability

Jobs:

- `80912`
- `80934`

Config:

- `configs/our_comembed_faqr_2v_smoke2.yaml`

Artifacts:

- `runs/our_comembed_faqr_2v_smoke2-80912/config.json`
- `runs/our_comembed_faqr_2v_smoke2-80934/config.json`
- `logs/20260707-172631_our_comembed_faqr_2v_smoke2_rank0.log`
- `logs/20260708-140037_our_comembed_faqr_2v_smoke2_rank0.log`

Observed in both runs:

- step 1 finished
- `optim/step skipped=0.0`
- `train/CE loss=10.96`
- step 2 triggered `nan loss encountered at step 2`

Stabilization attempts tried:

- added explicit `lambda_warmup_steps: 100` to smoke configs
- aligned ComEmbed row-memory init scale to transformer `init_std=0.02`

Result:

- instability remained

### 5. `fa_qr` deeper stabilization attempts

Additional jobs:

- `80954`
- `80959`
- `81498`
- `82313`

Observed:

- reducing ComEmbed optimizer override LR to base LR did not fix the step-2 failure
- explicitly instrumenting finite checks showed the first hard failure at:
  - `QRRowMemory.c1`
- further attempts with:
  - smaller `beta_logit` init
  - strongly negative `row_gate` init
  - clipped `float32` product path
  did not stabilize the run

Interpretation:

- `fa_qr` is not just "too aggressive"
- its row-memory parameter update becomes non-finite after the first optimization step in the current training stack

### 6. `fa_add_qr` smoke: 2-step instability

Jobs:

- `82314`
- `82316`

Observed:

- both the original X-GRAM-style elevated LR and a reduced base-LR optimizer override were tested
- both runs still hit `nan loss encountered at step 2`

Interpretation:

- instability is not confined to the multiplicative `fa_qr` path
- the current ComEmbed-on-X-GRAM training recipe is unstable more broadly

### 7. `fa_norm_qr` smoke: 2-step instability

Jobs:

- `82315`
- `82317`

Observed:

- `fa_norm_qr` also hit `nan loss encountered at step 2`
- disabling `torch.compile` did not remove the failure

Interpretation:

- the issue is not explained solely by the raw product interaction or by `torch.compile`
- the broader ComEmbed injection stack under this recipe still diverges immediately after the first update

## Formal Run Status

Job:

- `80905`

Config:

- `configs/our_comembed_faqr_2v.yaml`

Status:

- submitted once
- cancelled before running

Reason:

- the smoke gate for the main `fa_qr` variant failed
- allowing the 10B-token run to proceed would have consumed cluster time on a known-unstable configuration

## Interpretation

The current evidence supports these statements:

- ComEmbed is successfully integrated into X-GRAM's routing, ShortConv, gate, warmup, and injection stack.
- all tested ComEmbed variants can execute a 1-step smoke, so construction/runtime wiring is valid
- all tested ComEmbed variants currently diverge by step 2 under the present training recipe

The current evidence does not support these statements:

- that `fa_qr` improves over X-GRAM on FineWeb 10B
- that the main METHOD.md setting is ready for a 10B-token full run
- that `fa_add_qr` or `fa_norm_qr` are currently safe fallbacks for a full run without further stabilization

## Likely Root Causes

Most likely:

- a broader incompatibility between the current ComEmbed injection formulation and the X-GRAM training recipe after the first optimizer step
- immediate post-step parameter corruption inside the injected row-memory modules
- interaction among X-GRAM injection warmup/gates, shared shortconv path, and dense optimizer updates

Less likely but still possible:

- gradient metric path under FSDP is misreporting `nan` before the actual loss failure
- some instability is amplified by the tiny smoke regime, though the repeated step-2 `nan loss` suggests a real issue
- hidden optimizer-state interaction that only appears on GPU/FSDP and not in small local CPU checks

## Recommended Next Work

1. Do not launch a full 10B ComEmbed run from the current codepath.
2. Isolate the first-update divergence with more aggressive ablations:
   - no ShortConv
   - no X-GRAM outer gate
   - no row gate
   - frozen backbone / injection-only update
3. Compare against a pure non-X-GRAM standalone ComEmbed training stub on the same GPU stack to separate method instability from integration instability.
4. Only after a 10-50 step stable run exists should any 10B formal training be scheduled.
