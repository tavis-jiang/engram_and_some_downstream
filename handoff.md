# X-Gram / Engram 项目 Handoff 文档

> 生成时间: 2026-07-29
> 当前工作目录: `/home/bcjiang/X-gram`

---

## 2026-07-30 11:40 CST 续作更新

### 已处理

1. **检查当前训练状态**
   - 当前时间: 2026-07-30 11:39 CST。
   - job **91347** `engram-improved-v1` 仍在 node9 ADA6000 运行，已到 `step1669+`。
   - job **91674** `engram-v2-resume119` 仍在 node10 L40S 运行，已到 `step938+`。
   - 当前没有 residual eval job 在跑。

2. **91347 已到 step1192，已补 full downstream**
   - 新 checkpoint: `runs/engram-improved-v1-91347/step1192/model_and_optim`
   - 已提交并完成 Slurm job **91696**，ExitCode `0:0`
   - config: `configs/engram_improved_v1_360m.yaml`
   - log: `logs/downstream_eval_engram_engram_improved_v1_91347_step1192_full_20260730-113122-91696.log`
   - 结果：
     - Global Average **0.3890**
     - MMLU Average **0.2484**
     - SciQ **0.6080**
   - 主要单项：
     - ARC-Challenge **0.2338**
     - ARC-Easy **0.4193**
     - BoolQ **0.5691**
     - CommonsenseQA **0.2695**
     - CSQA val **0.2907**
     - HellaSwag **0.2723**
     - OpenBookQA **0.2720**
     - PIQA **0.5756**
     - SocialIQA **0.4074**
     - Winogrande **0.5020**

3. **91347 step1192 对比**
   - 与 91347 step596 相比明显提升：
     - step596 Global **0.3431** -> step1192 Global **0.3890**
     - step596 SciQ **0.4430** -> step1192 SciQ **0.6080**
     - MMLU 基本持平略升：**0.2470** -> **0.2484**
   - 与旧 Engram step1192 对比：
     - 旧 Engram step1192 Jul 23: Global **0.3768**, MMLU **0.2461**, SciQ **0.5800**
     - 旧 Engram step1192 Jul 19: Global **0.3864**, MMLU **0.2466**, SciQ **0.5960**
     - 91347 step1192 是三者里当前最好的 step1192：Global **0.3890**、MMLU **0.2484**、SciQ **0.6080**
   - 但它仍是中期 checkpoint，尚不能和 final step2385 的 X-Gram **0.4167** 或 2g3g-xgrammatch **0.4142** 直接等价。

### 当前状态

| Job ID | 名称 | 状态 | 节点 | 最新进度 | 备注 |
|--------|------|------|------|----------|------|
| **91696** | 91347 step1192 downstream eval | COMPLETED | node8 ADA6000 | Global 0.3890 | 已完成 |
| **91347** | engram-improved-v1 | R | node9 ADA6000 | step1669+；step1660 CE 3.194 | 训练中；已有 step596/1192 eval |
| **91674** | engram-v2-resume119 | R | node10 L40S | step938+；step930 CE 3.541 | 训练中；已有 step833 checkpoint |

### 下一步

1. 继续等 91347 到 `step1788`，到达后建议再补 full downstream。
2. 91674 当前已有 `step833` checkpoint，但还没做 downstream；短期可以继续等 `step952/step1071/step1190`，优先在接近 old step1192 的点评测。
3. 最关键结论仍要等 91347 和 91674 的 final `step2385`。

## 2026-07-29 20:55 CST 续作更新

### 已处理

1. **确认 91067 已补跑 downstream**
   - job **91694** 已完成，ExitCode `0:0`。
   - 结果仍是 Global **0.3980**、MMLU **0.2563**、SciQ **0.6410**。
   - 结论不变：91067 训练 CE 改善，但 downstream 没有提升；当前最好完成结果仍是 **2g3g-xgrammatch Global 0.4142**，略低于正常 X-Gram **0.4167**。

2. **确认 91674 v2 resume 的 step238 checkpoint 已落盘**
   - checkpoint: `runs/engram-v2-91663/step238/model_and_optim`
   - 保存日志: `logs/20260729-173857_engram_v2_360m_rank0.log`
   - `step260` 最新完整指标：
     - CE **4.830**
     - PPL **125.3**
     - TPS/device **6,705**
     - MFU **29.76%**
   - 当前继续训练中；下一常规 checkpoint 预计是 `step357`。

3. **提交并完成 91347 step596 full downstream**
   - Slurm job **91695**，ExitCode `0:0`
   - checkpoint: `runs/engram-improved-v1-91347/step596/model_and_optim`
   - config: `configs/engram_improved_v1_360m.yaml`
   - log: `logs/downstream_eval_engram_engram_improved_v1_91347_step596_full_20260729-204705-91695.log`
   - 结果：
     - Global Average **0.3431**
     - MMLU Average **0.2470**
     - SciQ **0.4430**
   - 与旧 Engram step596 对比：
     - 旧 Engram step596 Jul 23: Global **0.3453**, MMLU **0.2356**, SciQ **0.4590**
     - 旧 Engram step596 Jul 19: Global **0.3471**, MMLU **0.2352**, SciQ **0.4630**
   - 解读：91347 中期 MMLU 更高，但 Global/SciQ 还低于旧 Engram step596；这只是 25% 训练进度附近的中期信号，最终仍要看 step2385。

### 当前状态

| Job ID | 名称 | 状态 | 节点 | 最新进度 | 备注 |
|--------|------|------|------|----------|------|
| **91695** | 91347 step596 downstream eval | COMPLETED | node8 ADA6000 | Global 0.3431 | 已完成 |
| **91347** | engram-improved-v1 | R | node9 ADA6000 | step625+；step620 CE 3.900 | 训练中，已有 step596 checkpoint |
| **91674** | engram-v2-resume119 | R | node10 L40S | step261+；step260 CE 4.830 | 训练中，step238 checkpoint 已确认 |

### 下一步

1. 继续监控 91347；下一常规 checkpoint 预计 `step1192`，可在到达后再做 full downstream。
2. 继续监控 91674；下一常规 checkpoint 预计 `step357`，短期重点看 CE 是否继续快速下降。
3. 不建议评测 91674 的 `step238`，太早且信号价值低；先等至少 `step596` 或更后。

## 2026-07-29 20:45 CST 续作更新

### 已处理

1. **补跑 91067 downstream**
   - 用户指出 91067 之前还没有 downstream；已确认属实。
   - 已用 `normal` QOS 提交并完成 Slurm job **91694**：
     - checkpoint: `runs/faircompare-engram-2gram-xgrammatch-rerun-360m-91067/step2385/model_and_optim`
     - config: `configs/faircompare_engram_2gram_xgrammatch_rerun_360m.yaml`
     - log: `logs/downstream_eval_engram_engram_2gramxgrammatch_rerun91067_step2385_full_20260729-202755-91694.log`
     - ExitCode: `0:0`
   - 为匹配 91067 保存时的 config，eval 显式设置：
     - `model.embedding_injection.lambda_warmup_enabled=false`
     - `model.embedding_injection.lambda_warmup_steps=0`

2. **91067 downstream 结果**
   - Global Average **0.3980**
   - MMLU Average **0.2563**
   - SciQ **0.6410**
   - 主要单项：
     - ARC-Challenge **0.2381**
     - ARC-Easy **0.4175**
     - BoolQ **0.6031**
     - CommonsenseQA **0.2621**
     - CSQA val **0.3170**
     - HellaSwag **0.2859**
     - OpenBookQA **0.2700**
     - PIQA **0.5789**
     - SocialIQA **0.4012**
     - Winogrande **0.5051**
   - 结论：91067 rerun 的 downstream 与早先 `89613` 2gram-xgrammatch eval 数字一致；虽然训练 CE 较好，但 downstream 没有提升。
   - 当前已完成 Engram 结果里，最好的是 **2g3g-xgrammatch Global 0.4142**，仍略低于正常 X-Gram **0.4167**。

3. **报告更新**
   - 已更新 `reports/downstream_comparison_20260729.md`，加入：
     - Engram 2gram rerun 91067
     - Engram 2g3g-xgrammatch
   - 不再将 91067 标记为 downstream 缺失。

### 当前状态

| Job ID | 名称 | 状态 | 节点 | 最新进度 | 备注 |
|--------|------|------|------|----------|------|
| **91694** | 91067 downstream eval | COMPLETED | node8 ADA6000 | Global 0.3980 | 已补跑完成 |
| **91674** | engram-v2-resume119 | R | node10 L40S | 训练中 | 等 step238 checkpoint |
| **91347** | engram-improved-v1 | R | node9 ADA6000 | 训练中 | CE 已到 3.9 左右 |

### 下一步

1. 继续监控 91674，重点看 step238 checkpoint 是否正常保存。
2. 继续监控 91347，后续到关键 checkpoint 后再安排 downstream。
3. 当前已完成 downstream 结论：91067 不是突破点；2g3g-xgrammatch 最接近 X-Gram，但还没超过。

## 2026-07-29 18:05 CST 续作更新

### 已处理

1. **继续监控 91674 v2 resume**
   - job **91674** 仍在 `normal` QOS 运行，节点 `node10`，分区 `L40S`。
   - 已确认不是卡住：从 step120 继续推进到 step130。
   - 最新完整指标（`logs/20260729-173857_engram_v2_360m_rank0.log`）：
     - step130/2385，CE **5.879**，PPL **357.6**
     - TPS/device **6,679**，MFU **29.65%**
     - data loading **0.0006s / 0.0007%**
     - GPU active mem **15.39 GiB / 34.66%**
   - 与 step120 CE **6.043** 相比，CE 已下降到 **5.879**；恢复训练有效。
   - 单步耗时约 **78s/step**，明显慢于 ADA6000 上的 91347；当前瓶颈不是 dataloader。
   - 运行时 `config.json` 显示 `checkpointer.save_interval` 为 **119**；当前只有 `step119` checkpoint，下一次常规 checkpoint 预计是 **step238**。

2. **继续监控 91347 improved-v1**
   - job **91347** 仍在 `normal` QOS 运行，节点 `node9`，分区 `ADA6000`。
   - 最新完整指标（`logs/20260729-121510_engram_improved_v1_360m_rank0.log`）：
     - step420/2385，CE **4.261**，PPL **70.91**
     - TPS/device **10,505**，MFU **35.28%**
     - data loading **0.0004s / 0.0007%**
     - GPU active mem **21.89 GiB / 46.19%**
   - 近期低点是 step410 CE **4.221**；step420 回到 **4.261**，属于正常波动，但仍低于 step400 CE **4.304**。
   - 单步耗时约 **49s/step**，明显快于 91674 的 L40S。

3. **日志/指标开销观察**
   - `scripts/train/olmo_train.py` 当前硬编码开启 `TransformerMetricsConfig` 的 layer/module grad norm、param mean 等重指标。
   - 这会每步写很长的 `Transformer grad norms at step ...` 日志，并触发多处 CUDA synchronize warning。
   - 目前没有导致训练失败，但会增加日志噪声和潜在同步开销；后续若需要提速，可给这些重指标加 env/YAML 开关，或在新提交的 Slurm 脚本里默认关闭。

### 当前状态

| Job ID | 名称 | 状态 | 节点 | 最新进度 | 备注 |
|--------|------|------|------|----------|------|
| **91674** | engram-v2-resume119 | R | node10 L40S | step130 CE 5.879 | resume 有效，L40S 上约 78s/step |
| **91347** | engram-improved-v1 | R | node9 ADA6000 | step420 CE 4.261 | 约 49s/step，近期低点 step410 CE 4.221 |

### 下一步

1. 继续监控 91674 到 step160+，确认是否追平被抢占前的进度；重点再看 **step238 checkpoint** 是否正常保存。
2. 继续监控 91347；短期可看 step500 附近 CE 是否明显优于 v2。
3. 暂不提交 downstream eval；等候训练完成或至少到关键 checkpoint 后再评估。
4. 后续新训练/恢复脚本建议考虑关闭重 grad/param 指标，避免日志和 CUDA sync 开销。

## 2026-07-29 17:50 CST 续作更新

### 已处理

1. **纠正 downstream 归因**
   - `out/89545` / `logs/downstream_eval_engram_step2385_full_20260720-202223.log` 实际加载的是 `runs/faircompare-engram-360m-88169/step2385`，不是 91069/e1。
   - `logs/downstream_eval_engram_step2385_full_20260729-121605-91651.log` 实际加载的是 e2: `runs/faircompare-engram-e2-20l-2g-d512-b49152-91070/step2385`。
   - 91069/e1 的 full downstream 已补交并完成：job **91673**，日志 `logs/downstream_eval_engram_e1_step2385_full_20260729-173505-91673.log`。

2. **e1 downstream 结论**
   - Engram e1 step2385: Global Avg **0.3939**, MMLU Avg **0.2333**, SciQ **0.6410**。
   - 正常 X-Gram full eval 应使用 `logs/downstream_eval_xgram_step2385_full_20260606-162307.log`，Global Avg **0.4167**。
   - 结论：e1 的 downstream 没有超过 X-Gram；也略低于 Baseline Global Avg 0.3944。
   - 已新增汇总报告：`reports/downstream_comparison_20260729.md`。

3. **清理已完成但仍占资源的训练 job**
   - 91067 和 91069 rank0 日志均显示 step2385 checkpoint saved、Training complete、Cleanup process completed successfully，但 Slurm 仍显示 R。
   - 已 `scancel 91067 91069` 释放 L40S 资源；checkpoint 均已落盘。

4. **v2 被 eval 抢占后的恢复**
   - e1 eval 91673 使用 high QOS，抢占并 requeue 了原 v2 job 91663。
   - 原 91663 最后到 step160，但最新完整 checkpoint 只有 step119。
   - 已 hold 后取消原 91663，新增 resume 脚本 `python_slurm/resume_engram_v2_91663_step119.slurm`，并提交新 job **91674**。
   - 91674 已成功加载 `runs/engram-v2-91663/step119`，日志显示 `Will resume training from step 119, epoch 1`。
   - 91674 dry-run 已完成，并在 2026-07-29 17:51:07 CST 进入 **step120/2385**，CE **6.043**。
   - dataloader DuckDB 未恢复，按 `OLMO_ALLOW_MISSING_DATALOADER_CKPT=1` fallback fresh data loader；模型/优化器/trainer step 已恢复。
   - 已将 `python_slurm/eval_downstream.slurm` 默认 QOS 从 `high` 改为 `normal`，避免后续 eval 默认抢占训练。

### 当前状态

| Job ID | 名称 | 状态 | 节点 | 备注 |
|--------|------|------|------|------|
| **91674** | engram-v2-resume119 | R | node10 L40S | 从 91663 step119 resume；已进入正式训练，step120 CE 6.043 |
| **91347** | engram-improved-v1 | R | node9 ADA6000 | 继续训练中；约 step402，最近 CE 4.304 |
| **91673** | e1 full downstream eval | 已完成 | node8 ADA6000 | e1 Global Avg 0.3939 |
| **91067** | faircompare-engram-2gram-xgrammatch-rerun | CANCELLED after complete | node10 L40S | step2385 checkpoint 已保存 |
| **91069** | faircompare-engram-e1 | CANCELLED after complete | node11 L40S | step2385 checkpoint 已保存 |

### 下一步

1. 继续监控 91674 step120+ 后的 CE/throughput；注意它现在在 L40S 上，速度可能低于原 node8 ADA6000。
2. 继续监控 91347 CE 曲线。
3. 不再使用 `out/89545` 作为 91069/e1 downstream 结果。

## 一、项目目标

在 Fineweb-10B 上训练 360M 参数的语言模型，对比 **Baseline**、**X-Gram** 和 **Engram** 三种架构的 CE (Cross-Entropy) 和下游任务表现。

核心问题：**Engram 是否能在相同参数量下超越 X-Gram？**

---

## 二、当前 Running Jobs

| Job ID | 分区 | 名称 | 状态 | 运行时长 | 节点 | 当前进度 | 备注 |
|--------|------|------|------|---------|------|---------|------|
| **91663** | ADA6000 | engram-v2 | R | ~3h | node8 | **step 144/2385**, CE ~5.78 | 最新提交，v-path only |
| **91347** | ADA6000 | engram-improved-v1 | R | ~5h | node9 | **step 367/2385**, CE ~4.38 | h+v 改进版 |
| **91069** | L40S | faircompare-smollm2 (engram e1) | R | 2d+ | node11 | **已完成 step 2385**, CE ~3.109 | 等待下游评测结果 |
| **91067** | L40S | faircompare-smollm2 (xgram 2g) | R | 2d+ | node10 | **已完成 step 2385**, CE ~3.126 | 已结束训练 |

> 注：91067 和 91069 的日志显示 "Training complete"，但 squeue 仍显示 R，可能是评测进程或状态未更新。

---

## 三、已完成实验结果汇总

### 3.1 Training CE / PPL (Final Step 附近)

| 实验 | Job ID | Final CE | Final PPL | 路径 | 状态 |
|------|--------|---------|-----------|------|------|
| **Baseline** | 91068? | ~3.074 (step 2380) | ~21.63 | - | 已完成 |
| **X-Gram (2gram)** | 91067 | ~3.126 | ~22.79 | xgram | 已完成 |
| **Engram e1 (28l, 2g, d384)** | 91069 | ~3.109* | ~22.40 | h+v | 已完成 |
| **Engram improved v1** | 91347 | ~4.38 (step 367) | ~79.8 | h+v | **训练中** |
| **Engram v2** | 91663 | ~5.78 (step 144) | - | v-only | **训练中** |

> *91069 末期 CE 波动大 (2.968 ~ 3.126)，用户建议使用 **2.968** 作为最终 CE 参考值。

### 3.2 下游任务评测 (Downstream Eval)

#### Engram e1 (91069, job 89545) - 已完成
评测日期: 2026-07-20

| 任务 | 得分 |
|------|------|
| mmlu_stem_mc_5shot | 0.2181 |
| mmlu_humanities_mc_5shot | 0.2490 |
| mmlu_social_sciences_mc_5shot | 0.2552 |
| mmlu_other_mc_5shot | 0.2563 |
| arc_challenge_test_rc_5shot | 0.2457 |
| arc_easy | 0.4421 |
| boolq | 0.5804 |
| commonsense_qa | 0.2686 |
| csqa_val_rc_5shot | 0.3071 |
| hellaswag | 0.2860 |
| openbook_qa | 0.2680 |
| piqa | 0.5658 |
| sciq | **0.6590** |
| social_iqa | 0.3966 |
| winogrande | 0.4972 |

> 与 Baseline 和 X-Gram 的 downstream 对比需要进一步整理。

### 3.3 关键发现

1. **Engram e1 (91069) 的 CE 略差于 Baseline，略好于 X-Gram**
   - Baseline: ~3.074
   - Engram e1: ~3.109
   - X-Gram: ~3.126
   - 但 Engram 的 downstream 表现是否更好仍需完整对比。

2. **末期 CE/PPL 波动问题**
   - 91069 在最后几十步 CE 波动很大 (2.968 → 3.126)
   - **建议**: 以后训练一定要在末期多保存 checkpoint (如 step 2370, 2380, 2384, 2385)，用于选取最优模型。

---

## 四、代码修改清单

### 4.1 核心模块

| 文件 | 修改内容 |
|------|---------|
| `OLMo-core/src/olmo_core/nn/embedding_injection/engram.py` | **Engram 注入模块** (h-path + v-path)，含 `EngramInjectionEmbedding`, `EngramNgramHash`, `reset_buffers` 等 |
| `OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py` | X-Gram 注入模块 |
| `OLMo-core/src/olmo_core/nn/embedding_injection/__init__.py` | 导出 |
| `OLMo-core/src/olmo_core/nn/embedding_injection/ops/hash_injection.py` | hash injection ops |
| `OLMo-core/src/olmo_core/nn/transformer/model.py` | Transformer 支持 Engram targets (`_injection_h_embeddings`, `_injection_v_embeddings`) |
| `OLMo-core/src/olmo_core/nn/transformer/config.py` | TransformerConfig 添加 injection 配置 |
| `OLMo-core/src/olmo_core/train/train_module/transformer/train_module.py` | Train module 适配 |
| `OLMo-core/src/olmo_core/train/trainer.py` | Trainer 适配 |
| `scripts/train/olmo_train.py` | **主训练脚本**，解析新配置、调整优化器 (group override)、warmup |
| `scripts/train/olmo_train.sh` | 启动脚本 |
| `packages/ubdataloader/src/ubdataloader/text_dataset.py` | 数据加载器修改 |
| `packages/ubdataloader/src/ubdataloader/tokenizer_worker.py` | tokenizer worker |
| `OLMo-core/src/olmo_core/data/data_loader.py` | data loader 日志 |

### 4.2 关键 Bug 修复 (v2 开发过程中)

1. **FSDP meta-tensor 初始化问题**
   - `EngramInjectionEmbedding` 添加 `reset_buffers()` 方法
   - `Transformer.build()` 中遍历 `_injection_v_embeddings` 等字典并调用 `reset_buffers`

2. **`half_bound` 溢出**
   - `EngramNgramHash` 的 `half_bound` 从 `max_long // vocab_size // 2` 降到 `// 4`
   - 避免 `shifted * multipliers` 乘法溢出 `torch.long` 触发 CUDA assert

3. **v-only 优化器 group override**
   - v-only 配置下 optimizer group override 只包含 v pattern，不再包含 h pattern

4. **`use_compressed_lookup=false`**
   - 返回 identity mapping，不走压缩 lookup

---

## 五、配置文件

| 配置 | 路径 | 说明 |
|------|------|------|
| engram_v2_360m.yaml | `configs/` | **最新 v2 配置**，v-path only |
| engram_improved_v1_360m.yaml | `configs/` | improved v1 配置，h+v |
| faircompare 系列 | `configs/` | baseline, xgram, engram e1/e2/e3 等 |

### 5.1 v2 关键配置参数

```yaml
# engram_v2_360m.yaml (推测)
engram:
  injection_targets: ["v"]  # 只注入 value
  use_compressed_lookup: false
  # ... 其他参数
```

---

## 六、失败任务与问题记录

### 6.1 v2 启动历程 (91663 之前多次失败)

| 时间 | 问题 | 原因 | 解决 |
|------|------|------|------|
| 2026-07-29 12:58 | v2 noshortconv 启动失败 | 未知 | 切换配置 |
| 2026-07-29 13:11 | v2 OOM | micro_batch_size=4, 48GB 不够 | 改为 micro_batch_size=1 |
| 2026-07-29 13:15 | v2 启动失败 | optimizer group override 含 h pattern | 修复为 v-only pattern |
| 2026-07-29 13:57 | v2 noshortconv 启动失败 | 未知 | 放弃 noshortconv |
| 2026-07-29 14:11 | v2 启动失败 | FSDP meta-tensor buffer 未分配 | 添加 reset_buffers |
| 2026-07-29 14:18 | v2 启动失败 | half_bound 溢出 CUDA assert | half_bound // 4 |
| 2026-07-29 14:21 | **91663 成功启动** | - | 运行中 |

### 6.2 待解决问题

1. **Engram 为什么比 Baseline 差？**
   - 假设: h-path 注入破坏了注意力机制 (query/key 被 ngram embedding 干扰)
   - v-path (v2) 可能更温和，只影响 value，不破坏 attention score

2. **v2 数据加载慢**
   - node8 上初期 fetch item time 长达 12 秒
   - 目前恢复正常 (~0.0014s)

3. **末期 checkpoint 保存策略**
   - 需要在最后 100 步内密集保存 (每 10 步或每 5 步)

---

## 七、关键概念解释

### 7.1 H-path vs V-path

| 路径 | 注入位置 | 影响 |
|------|---------|------|
| **H-path** | 注入到 attention 的 **query 和 key** | 直接影响 attention score 计算，改变 token 之间的相似度 |
| **V-path** | 注入到 attention 的 **value** | 只影响 attention 输出内容，不改变 attention pattern |

> **核心区别**: H-path 改变 "关注什么"，V-path 改变 "提取什么信息"。
> H-path 更激进，可能破坏原始 attention 机制；V-path 更温和，保留了原始 attention pattern。

### 7.2 Engram vs X-Gram 机制区别

- **X-Gram**: 在 embedding layer 直接替换/加权 token embedding，是 "input-level" 的 ngram 增强
- **Engram**: 在 Transformer block 内部通过 injection module 将 ngram 信息注入到 attention，是 "layer-level" 的增强
- **关键风险**: 如果 injection 方式不当 (如 h-path)，Engram 可能更像 "干扰" 而非 "增强"

---

## 八、下一步建议操作

### 8.1 短期 (接下来几小时)

1. **监控 91663 (v2)** 和 **91347 (improved v1)** 的训练曲线
   - 检查 CE 是否正常下降
   - 检查 GPU 内存是否稳定
   - 检查 throughput 是否正常

2. **获取 91069 的 downstream 完整评测结果**
   - 对比 Baseline 和 X-Gram 的 downstream
   - 确认 Engram 是否在下游任务上更好 (即使 CE 略差)

### 8.2 中期 (接下来几天)

3. **如果 v2 CE 优于 xgram**: 立即安排 downstream 评测
4. **如果 v2 CE 仍差于 baseline**: 分析原因，可能需要:
   - 调整 injection 强度 (lambda/gate)
   - 尝试不同的 ngram hash 策略
   - 调整学习率 (当前 v2 的 Z loss 学习率可能太低)

5. **优化 checkpoint 保存策略**
   - 在 config 中设置 `save_interval=5` 或 `save_interval_unsharded=5` 用于最后 100 步

### 8.3 长期研究方向

6. **Engram 的 scaling law**: Engram 在小模型上可能不占优，但在更大模型上可能显现优势
7. **组合策略**: h-path + v-path 的加权组合，而非二选一
8. **更精细的 ngram 选择**: 不是均匀 hash，而是基于频率或信息量的选择

---

## 九、重要文件路径速查

```
# 日志
logs/20260729-142210_engram_v2_360m_rank0.log          # 91663 v2
logs/20260729-121510_engram_improved_v1_360m_rank0.log  # 91347 improved v1
logs/20260727-124252_engram_faircompare_e1_28l_2g_d384_b49152_rank0.log  # 91069
logs/20260727-115348_faircompare_engram_2gram_xgrammatch_rerun_360m_rank0.log  # 91067

# 运行目录
runs/engram-v2-91663/
runs/engram-improved-v1-91347/
runs/faircompare-engram-e1-28l-2g-d384-b49152-91069/
runs/faircompare-engram-2gram-xgrammatch-rerun-360m-91067/

# 评测结果
out/89545.out  # 91069 downstream eval (engram step2385)
runs_eval/eval-engram-step2385-20260729-121605-91651/  # 最新 engram eval

# 报告
reports/baseline_full_metrics.csv
reports/xgram_full_metrics.csv
reports/checkpoint_eval_state/          # engram eval state
reports/checkpoint_eval_state_xgrammatch/  # xgram eval state

# 代码
OLMo-core/src/olmo_core/nn/embedding_injection/engram.py
OLMo-core/src/olmo_core/nn/transformer/model.py
scripts/train/olmo_train.py
configs/engram_v2_360m.yaml
python_slurm/train_engram_v2.slurm
```

---

## 十、联系人 / 上下文

- 项目基于 **OLMo-core** 框架
- 训练数据: **Fineweb-10B**
- 模型规模: **360M 参数**
- 主要对比维度: **CE Loss** 和 **Downstream Tasks** (mmlu, arc, boolq, piqa, sciq 等)
- 用户核心诉求: **Engram 必须在相同参数量下效果超过 X-Gram**
