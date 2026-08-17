# Engram Variant Sweep - 2026-08-04

本轮只提交 Engram 变体，不重跑 Baseline 或 X-gram。Baseline/X-gram 继续使用既有结果作为对照。

## 对照固定项

- data/model/evaluation 与 `configs/faircompare_baseline_360m.yaml`、`configs/faircompare_xgram_360m.yaml` 对齐。
- training 的 `seq_len/global_batch_size/train_tokens/warmup_fraction/save_tokens/lr/min_lr/ac_mode/log_interval` 与最新 v-path Engram 对齐。
- Engram `micro_batch_size=1` 保持不变，用于避免 v-path OOM；有效 `global_batch_size=512` 不变。
- eval 阶段沿用 `python_slurm/eval_downstream.slurm`，Engram eval 默认关闭 lambda warmup，评测 checkpoint 的 full-strength 注入。

## 新增训练任务

| Train job | Run name | Config | 主要变动 | 目的 |
| --- | --- | --- | --- | --- |
| 93378 | `faircompare-engram-2g3g-vpath-d512-lam05-warm240` | `configs/faircompare_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml` | `2gram+3gram`, dim 512, lambda 0.5, warmup 240 | 在不增大整体注入压力的前提下恢复 3-gram 信息 |
| 93379 | `faircompare-engram-vpath-single-lam05-warm240` | `configs/faircompare_engram_vpath_single_lam05_warm240_360m.yaml` | v_layers 改成 `[0..9]`, lambda 0.5, warmup 240 | 测试重复 v 注入是否导致 CE/downstream 受损 |
| 93380 | `faircompare-engram-vpath-lam05-warm240` | `configs/faircompare_engram_vpath_lam05_warm240_360m.yaml` | 保持 v_layers，与 91976 相比 lambda 1.0 -> 0.5, warmup 120 -> 240 | 隔离注入强度/warmup 对 CE 的影响 |
| 93381 | `faircompare-engram-vpath-lam025-warm480` | `configs/faircompare_engram_vpath_lam025_warm480_360m.yaml` | 保持 v_layers，lambda 0.25, warmup 480 | 更保守的注入，优先保护 backbone CE |

## 自动评测依赖

| Train job | SciQ eval job | Full eval job | Checkpoint |
| --- | --- | --- | --- |
| 93378 | 93385 | 93386 | `runs/faircompare-engram-2g3g-vpath-d512-lam05-warm240-93378/step2385` |
| 93379 | 93387 | 93388 | `runs/faircompare-engram-vpath-single-lam05-warm240-93379/step2385` |
| 93380 | 93389 | 93390 | `runs/faircompare-engram-vpath-lam05-warm240-93380/step2385` |
| 93381 | 93391 | 93392 | `runs/faircompare-engram-vpath-lam025-warm480-93381/step2385` |

当前状态：训练任务均为 `PENDING (Priority)`；评测任务均为 `PENDING (Dependency)`，依赖对应训练 job `afterok`。
