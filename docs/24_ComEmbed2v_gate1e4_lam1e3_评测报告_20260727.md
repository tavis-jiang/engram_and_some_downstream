tmux # ComEmbed 2v gate1e4/lam1e3 最新训练与下游评测报告

日期：2026-07-27

## 1. 结论

新的 ComEmbed 2v `gate1e4/lam1e3` 已完成 5000-step 训练，并完成 full downstream eval。

核心结论：

- 训练终点优于 `docs/21` 里的旧 2v stabilized：CE loss `4.852` vs `4.874`，PPL `128.0` vs `130.9`。
- 下游 Global Average 继续提升：新 2v `0.3310`，旧 2v `0.3272`，1v `0.3218`。
- MMLU average 略低：新 2v `0.2504`，旧 2v `0.2519`，1v `0.2520`。
- 新 2v 训练日志中没有 `Sanitized/non-finite/Traceback/ERROR`，比旧 stabilized 2v 更干净。

因此，这次新 2v 的推荐表述是：

> 在同样 20.48M tokens、5000 step 设置下，`fa_norm_qr + same-layer 2v + gate=1e-4 + lambda=1e-3` 进一步提升了整体 downstream Global Average，并且训练中没有出现旧 stabilized 2v 依赖的 non-finite gradient sanitize；但收益主要由 BoolQ 拉动，MMLU average 略低，不能表述为所有任务全面优胜。

## 2. 实验配置

### 2.1 新 2v 配置

- 配置：`configs/our_comembed_fanormqr_2v_gate1e4_lam1e3_overnight5000.yaml`
- 训练 job：`90854`
- 训练 run：`runs/our_comembed_fanormqr_2v_gate1e4_lam1e3_overnight5000-90854`
- 训练 checkpoint：`runs/our_comembed_fanormqr_2v_gate1e4_lam1e3_overnight5000-90854/step5000`
- 训练日志：`logs/20260724-233805_our_comembed_fanormqr_2v_gate1e4_lam1e3_overnight5000_rank0.log`

关键结构：

| 项目 | 设置 |
|---|---|
| mode | `ComEmbed` |
| target | `v` |
| `v_layers` | `[0, 0]` |
| variant | `fa_norm_qr` |
| `comembed_gate_init` | `1.0e-4` |
| `comembed_disable_row_gate` | `true` |
| `comembed_output_rmsnorm` | `true` |
| `lambda_init` | `1.0e-3` |
| `lambda_warmup_steps` | `5000` |
| train tokens | `20,480,000` |

说明：这版是基于 `docs/21` 中已证明能完整跑完的 `2v stabilized` 配方，只提高 gate 和 lambda 强度，保持 same-layer 2v，即 `v_layers: [0, 0]`。

### 2.2 下游评测

- Eval job：`91064`
- Eval log：`logs/downstream_eval_xgram_comembed_2v_gate1e4_lam1e3_step5000_full_20260727-105754-91064.log`
- Slurm 状态：`COMPLETED`
- Exit code：`0:0`
- stderr：`out/91064.err` 为空
- 评测耗时：`01:04:11`

评测使用：

- `MODEL=xgram`
- `XGRAM_CONFIG=configs/our_comembed_fanormqr_2v_gate1e4_lam1e3_overnight5000.yaml`
- `XGRAM_CHECKPOINT_ROOT=runs/our_comembed_fanormqr_2v_gate1e4_lam1e3_overnight5000-90854/step5000`
- `CHECKPOINT_LABEL=comembed_2v_gate1e4_lam1e3_step5000`

## 3. 训练结果

| 模型 | 状态 | 最后 step | tokens | Train CE loss | Train PPL | total grad norm | sanitize/non-finite |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1v `fa_add_qr` | 完成 | 5000/5000 | 20,480,000 | 4.922 | 137.3 | 0.8596 | 0 |
| 旧 2v `fa_norm_qr stabilized` | 完成 | 5000/5000 | 20,480,000 | 4.874 | 130.9 | 0.7835 | 5000 |
| 2v `layer01 gate1e4/lam1e3` | 完成 | 5000/5000 | 20,480,000 | 4.846 | 127.2 | 0.8478 | 0 |
| 新 2v `same-layer gate1e4/lam1e3` | 完成 | 5000/5000 | 20,480,000 | 4.852 | 128.0 | 0.8383 | 0 |

相对旧 2v stabilized，新 2v same-layer：

- CE loss 降低 `0.022`。
- PPL 降低 `2.9`。
- 非有限梯度清洗从 `5000` 次降为 `0` 次。

相对 1v：

- CE loss 降低 `0.070`。
- PPL 降低 `9.3`。

注意：训练 job `90854` 在训练完成、step5000 checkpoint 成功保存、cleanup 完成后，Slurm wrapper 没有自动退出；之后手动 `scancel` 释放资源。因此 Slurm 最终状态里出现 `CANCELLED by 1105` 不代表训练失败。

## 4. 下游汇总

| 模型 | MMLU avg | Global avg | raw 15-task avg |
|---|---:|---:|---:|
| 1v `fa_add_qr` | 0.2520 | 0.3218 | 0.3079 |
| 旧 2v `fa_norm_qr stabilized` | 0.2519 | 0.3272 | 0.3121 |
| 新 2v `same-layer gate1e4/lam1e3` | 0.2504 | 0.3310 | 0.3148 |

新 2v 相对 1v：

- Global avg：`+0.0092`
- MMLU avg：`-0.0016`
- raw 15-task avg：`+0.0070`
- 单项：新 2v 赢 5 项，1v 赢 6 项，平 4 项

新 2v 相对旧 2v stabilized：

- Global avg：`+0.0038`
- MMLU avg：`-0.0015`
- raw 15-task avg：`+0.0027`
- 单项：新 2v 赢 5 项，旧 2v 赢 8 项，平 2 项

## 5. 下游单项结果

| Task | 1v | 旧 2v stabilized | 新 2v gate1e4/lam1e3 | 新 2v - 1v | 新 2v - 旧 2v |
|---|---:|---:|---:|---:|---:|
| MMLU stem | 0.2866 | 0.2835 | 0.2804 | -0.0062 | -0.0031 |
| MMLU hum | 0.2413 | 0.2413 | 0.2413 | +0.0000 | +0.0000 |
| MMLU social | 0.2433 | 0.2433 | 0.2433 | +0.0000 | +0.0000 |
| MMLU other | 0.2366 | 0.2394 | 0.2366 | +0.0000 | -0.0028 |
| ARC-E | 0.2965 | 0.3123 | 0.3158 | +0.0193 | +0.0035 |
| BoolQ | 0.4177 | 0.4498 | 0.5446 | +0.1269 | +0.0948 |
| CSQA | 0.2482 | 0.2457 | 0.2400 | -0.0082 | -0.0057 |
| HellaSwag | 0.2517 | 0.2532 | 0.2558 | +0.0041 | +0.0026 |
| OpenBookQA | 0.2420 | 0.2580 | 0.2420 | +0.0000 | -0.0160 |
| PIQA | 0.5125 | 0.5033 | 0.5103 | -0.0022 | +0.0070 |
| SciQ | 0.2920 | 0.2880 | 0.2700 | -0.0220 | -0.0180 |
| SocialIQA | 0.3936 | 0.3971 | 0.3951 | +0.0015 | -0.0020 |
| WinoGrande | 0.4909 | 0.4949 | 0.4807 | -0.0102 | -0.0142 |
| ARC-C | 0.2193 | 0.2329 | 0.2235 | +0.0042 | -0.0094 |
| CSQA val rc | 0.2457 | 0.2391 | 0.2432 | -0.0025 | +0.0041 |

## 6. 解读

### 6.1 明确变好的部分

新 2v 的整体 Global Average 是目前三者最高：

- 高于 1v `+0.0092`
- 高于旧 2v stabilized `+0.0038`

最主要的单项增益来自 BoolQ：

- 相对 1v：`+0.1269`
- 相对旧 2v：`+0.0948`

此外，新 2v 相对旧 2v 在 ARC-E、HellaSwag、PIQA、CSQA val rc 上也有小幅提升。

### 6.2 需要保守表述的部分

新 2v 不是所有任务都优于旧 2v。它相对旧 2v 在 OpenBookQA、SciQ、WinoGrande、ARC-C、CSQA、MMLU stem、MMLU other 等任务上下降。

MMLU avg 也不是最优：

- 1v：`0.2520`
- 旧 2v：`0.2519`
- 新 2v：`0.2504`

因此，报告中不应说“新 2v 全面优于旧 2v”或“2v 在所有下游能力上更强”。更准确的表述是：新 2v 在当前 full downstream 的 Global Average 上最好，并且训练稳定性明显改善。

## 7. 推荐汇报口径

可以直接使用下面这段：

> 根据最新完成的 `gate1e4/lam1e3` same-layer 2v 实验，ComEmbed 2v 在 20.48M tokens、5000 step 下训练完成，终点 CE loss 为 `4.852`，PPL 为 `128.0`，优于 1v 和旧 2v stabilized。更重要的是，这次训练日志中没有出现 non-finite gradient sanitize，而旧 stabilized 2v 有 5000 次 sanitize。下游 full eval 中，新 2v 的 Global Average 达到 `0.3310`，高于旧 2v 的 `0.3272` 和 1v 的 `0.3218`；但 MMLU average 略低，且 Global 提升主要由 BoolQ 拉动，所以结论应表述为“整体下游均值和训练稳定性改善”，而不是“全任务全面胜出”。

## 8. 后续建议

如果继续推进 2v，建议下一步做两件事：

1. 对新 2v 再跑至少一个 seed 或相邻数据顺序，确认 BoolQ 增益不是偶然波动。
2. 对 `layer01 gate1e4/lam1e3` 的 checkpoint 也跑 full downstream，因为它训练 CE/PPL 比新 same-layer 2v 略好，目前缺少下游结果。
