# Downstream Comparison Update - 2026-07-29

## Summary

- Engram e1 (job 91069, step2385) full downstream eval completed as Slurm job 91673.
- Engram 2gram-xgrammatch rerun (job 91067, step2385) full downstream eval completed as Slurm job 91694.
- Engram improved-v1 (job 91347, step596) full downstream eval completed as Slurm job 91695.
- Engram improved-v1 (job 91347, step1192) full downstream eval completed as Slurm job 91696.
- e1 does not beat the normal X-Gram step2385 eval on downstream metrics.
- 91067 rerun matches the earlier 2gram-xgrammatch eval numerically: Global Avg 0.3980, MMLU Avg 0.2563.
- 91347 step596 is only an intermediate checkpoint: Global Avg 0.3431, MMLU Avg 0.2470, SciQ 0.4430.
- 91347 step1192 improves to Global Avg 0.3890, MMLU Avg 0.2484, SciQ 0.6080; it is the best observed step1192 among the compared Engram runs, but still intermediate.
- The best completed Engram-style downstream result in this set remains 2g3g-xgrammatch, Global Avg 0.4142, still slightly below X-Gram 0.4167.
- `out/89545` / `logs/downstream_eval_engram_step2385_full_20260720-202223.log` loads `runs/faircompare-engram-360m-88169`, not job 91069.
- `logs/downstream_eval_engram_step2385_full_20260729-121605-91651.log` loads e2 (`runs/faircompare-engram-e2-20l-2g-d512-b49152-91070`), not e1.
- Use `logs/downstream_eval_xgram_step2385_full_20260606-162307.log` for the normal X-Gram full eval. Earlier X-Gram full logs with SciQ 0.2280 are anomalous.

## Main Table

| metric | Baseline 64984 | X-Gram 64971 good rerun | Engram old 88169 | Engram 2gram 91067 | Engram 2g3g | Engram e1 91069 | Engram e2 91070 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mmlu_average | 0.2504 | 0.2634 | 0.2447 | 0.2563 | 0.2451 | 0.2333 | 0.2371 |
| arc_challenge_test_rc_5shot | 0.2321 | 0.2466 | 0.2457 | 0.2381 | 0.2483 | 0.2278 | 0.2346 |
| arc_easy | 0.4211 | 0.4509 | 0.4421 | 0.4175 | 0.5158 | 0.4211 | 0.4684 |
| boolq | 0.6095 | 0.6190 | 0.5804 | 0.6031 | 0.5985 | 0.5725 | 0.5361 |
| commonsense_qa | 0.2539 | 0.2752 | 0.2686 | 0.2621 | 0.2973 | 0.2654 | 0.2899 |
| csqa_val_rc_5shot | 0.2981 | 0.3260 | 0.3071 | 0.3170 | 0.3219 | 0.3342 | 0.3022 |
| hellaswag | 0.2766 | 0.3020 | 0.2860 | 0.2859 | 0.2923 | 0.2907 | 0.2877 |
| openbook_qa | 0.2740 | 0.2960 | 0.2680 | 0.2700 | 0.2740 | 0.2640 | 0.2800 |
| piqa | 0.5647 | 0.6045 | 0.5658 | 0.5789 | 0.5892 | 0.5696 | 0.5979 |
| sciq | 0.6480 | 0.7010 | 0.6590 | 0.6410 | 0.6840 | 0.6410 | 0.6510 |
| social_iqa | 0.3910 | 0.4058 | 0.3966 | 0.4012 | 0.3905 | 0.3966 | 0.3915 |
| winogrande | 0.5138 | 0.5099 | 0.4972 | 0.5051 | 0.5138 | 0.5107 | 0.5051 |
| global_average | 0.3944 | 0.4167 | 0.3968 | 0.3980 | 0.4142 | 0.3939 | 0.3985 |

## Intermediate Checkpoints

These are not directly comparable to the final step2385 table above, but they are useful for monitoring training trajectory.

| metric | Old Engram step596 Jul 19 | Old Engram step596 Jul 23 | Engram improved-v1 91347 step596 |
|---|---:|---:|---:|
| mmlu_average | 0.2352 | 0.2356 | 0.2470 |
| arc_challenge_test_rc_5shot | 0.2167 | 0.2193 | 0.2218 |
| arc_easy | 0.3737 | 0.3544 | 0.3544 |
| boolq | 0.3991 | 0.3856 | 0.3780 |
| commonsense_qa | 0.2490 | 0.2523 | 0.2391 |
| csqa_val_rc_5shot | 0.2563 | 0.2776 | 0.2563 |
| hellaswag | 0.2555 | 0.2573 | 0.2590 |
| openbook_qa | 0.2780 | 0.2660 | 0.2820 |
| piqa | 0.5408 | 0.5397 | 0.5473 |
| sciq | 0.4630 | 0.4590 | 0.4430 |
| social_iqa | 0.3946 | 0.3956 | 0.3869 |
| winogrande | 0.5036 | 0.5012 | 0.5028 |
| global_average | 0.3471 | 0.3453 | 0.3431 |

### Step1192

| metric | Old Engram step1192 Jul 19 | Old Engram step1192 Jul 23 | Engram improved-v1 91347 step1192 |
|---|---:|---:|---:|
| mmlu_average | 0.2466 | 0.2461 | 0.2484 |
| arc_challenge_test_rc_5shot | 0.2218 | 0.2287 | 0.2338 |
| arc_easy | 0.4491 | 0.4193 | 0.4193 |
| boolq | 0.5746 | 0.5122 | 0.5691 |
| commonsense_qa | 0.2703 | 0.2555 | 0.2695 |
| csqa_val_rc_5shot | 0.2850 | 0.2826 | 0.2907 |
| hellaswag | 0.2688 | 0.2705 | 0.2723 |
| openbook_qa | 0.2600 | 0.2700 | 0.2720 |
| piqa | 0.5702 | 0.5707 | 0.5756 |
| sciq | 0.5960 | 0.5800 | 0.6080 |
| social_iqa | 0.3966 | 0.3925 | 0.4074 |
| winogrande | 0.4980 | 0.4933 | 0.5020 |
| global_average | 0.3864 | 0.3768 | 0.3890 |

## Source Logs

- Baseline: `logs/downstream_eval_baseline_step2385_full_20260606-104406.log`
- X-Gram: `logs/downstream_eval_xgram_step2385_full_20260606-162307.log`
- Engram old: `logs/downstream_eval_engram_step2385_full_20260720-202223.log`
- Engram 2gram rerun 91067: `logs/downstream_eval_engram_engram_2gramxgrammatch_rerun91067_step2385_full_20260729-202755-91694.log`
- Engram 2g3g: `logs/downstream_eval_engram_2g3gxgrammatch_step2385_full_20260727-105716-91063.log`
- Engram e1: `logs/downstream_eval_engram_e1_step2385_full_20260729-173505-91673.log`
- Engram e2: `logs/downstream_eval_engram_step2385_full_20260729-121605-91651.log`
- Engram improved-v1 step596: `logs/downstream_eval_engram_engram_improved_v1_91347_step596_full_20260729-204705-91695.log`
- Engram improved-v1 step1192: `logs/downstream_eval_engram_engram_improved_v1_91347_step1192_full_20260730-113122-91696.log`
- Old Engram step596 Jul 23: `logs/downstream_eval_engram_step596_full_20260723-165022.log`
- Old Engram step596 Jul 19: `logs/downstream_eval_engram_step596_full_20260719-231217.log`
- Old Engram step1192 Jul 23: `logs/downstream_eval_engram_step1192_full_20260723-164610.log`
- Old Engram step1192 Jul 19: `logs/downstream_eval_engram_step1192_full_20260719-231217.log`
