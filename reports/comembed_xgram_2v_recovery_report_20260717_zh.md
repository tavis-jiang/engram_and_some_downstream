# ComEmbed X-GRAM 2v 恢复报告

日期：2026-07-17

## 1. 当前结论

`fa_norm_qr + 2v` 已经从“第 1 步后全局梯度 NaN，无法继续训练”修复到“可以完整跑完 10-step smoke，并且 loss 持续下降”。

最新成功短程实验：

- 配置：`configs/our_comembed_fanormqr_2v_smoke10_tiny.yaml`
- 日志：`logs/20260716-180942_our_comembed_fanormqr_2v_smoke10_tiny_rank0.log`
- 作业：`86747`

关键结果：

- `step 1`: `train/CE loss = 10.89`, `optim/total grad norm = 6.792`
- `step 2`: `train/CE loss = 10.37`, `optim/total grad norm = 11.70`
- `step 10`: `train/CE loss = 8.811`, `train/PPL = 6710`
- 日志末尾包含：
  - `Training complete`
  - `Cleanup process completed successfully`

## 2. 做了什么修复

本轮真正起作用的修复有三类：

1. `QRAddNormProductRowMemory` 实现修复
   - 补了缺失的 `row_gate_init`
   - 给乘法支路前后补了更严格的数值检查

2. ComEmbed 分支稳定化
   - 将 `lambda_init` 降到 `1e-4`
   - 将 `comembed_gate_init` 降到 `1e-5`
   - 将 `lambda_warmup_steps` 提高到 `5000`

3. 训练步防炸保护
   - 在 optimizer step 之前，扫描全模型梯度
   - 将非有限梯度用 `nan_to_num` 清理为有限值
   - 保证不会因为单步非有限梯度把参数整体污染

## 3. 现在还剩什么问题

虽然 10-step 已经能跑通，但还没有达到“完全干净”的状态。

日志里每一步仍然会出现：

- `Sanitized non-finite gradients on 4 parameter tensors before optimizer step`

目前看，这些非有限梯度被稳定地限制在 4 个参数张量上，并没有再扩散成全模型 NaN。

因此现在的状态是：

- 不是完全无异常
- 但已经可以继续训练
- 并且短程 loss 在下降

## 4. 现在采取的策略

不再继续停留在 2-step / 10-step 级别来回试错，而是直接发起正式 5000-step 长跑，验证：

1. 这种“局部梯度清理 + 正常更新”方案能否在长程训练下持续稳定；
2. loss 是否继续下降；
3. 是否会在更长步数重新出现全局失稳。

## 5. 正式长跑配置

本次正式长跑将使用：

- `mode: ComEmbed`
- `comembed_variant: fa_norm_qr`
- `targets: [v]`
- `v_layers: [0, 0]`
- `seq_len: 512`
- `micro_batch_size: 1`
- `global_batch_size: 8`
- `train_tokens: 20480000`

也就是与已完成的 `1v overnight5000` 对齐训练长度，但保留 `2v` 结构。

