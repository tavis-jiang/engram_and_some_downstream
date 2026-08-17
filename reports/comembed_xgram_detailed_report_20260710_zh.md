# ComEmbed 接入 X-GRAM 详细实验报告

日期：2026-07-10

## 1. 任务目标

本次任务的目标不是简单“把代码跑起来”，而是要验证 `METHOD.md` 里面提出的 ComEmbed 接入 X-GRAM 的方案，看看它在当前 X-GRAM 训练框架上能不能稳定训练，并给出可复现实验结果。

这里要区分两件事：

1. 理论上能不能接入。
2. 当前代码和训练配置下，能不能稳定训练。

结论是：

- **理论上可以接入。**
- **原始目标方案当前不能稳定训练。**
- **经过排查后，找到了一个可以稳定训练的新方案，并且已经成功跑完。**

## 2. 原始目标方案是什么

根据 `METHOD.md`，当前最核心的目标方案是：

- 使用 `ComEmbed`
- 使用 `fa_add_qr`
- 注入目标为 `v`
- 保留 X-GRAM 的 frequency-aware routing、hash routing、warm gate 等机制
- 原始多视图 value 注入是 **2v**

在配置文件上，这个目标大致对应：

- [configs/our_comembed_faaddqr_2v.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_2v.yaml)
- 以及最小复现版本 [configs/our_comembed_faaddqr_2v_smoke2.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_2v_smoke2.yaml)

最关键的一项是：

```yaml
v_layers: [0, 0]
```

这表示在同一个 `v` 注入位置上放两个 ComEmbed view，也就是当前讨论的 `2v`。

## 3. 现在为什么“不行”

这里的“不行”不是指标差，而是**训练直接炸掉**。

具体表现：

- 训练刚开始，`step 1` 就出现 `NaN`
- 注入模块相关梯度全部变成 `nan`
- 总梯度范数 `optim/total grad norm=nan`
- 程序随后抛出 `RuntimeError`

代表性失败日志：

- [logs/20260710-181740_our_comembed_faaddqr_2v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-181740_our_comembed_faaddqr_2v_smoke2_rank0.log)
- [logs/20260710-181557_our_comembed_faaddqr_2v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-181557_our_comembed_faaddqr_2v_smoke2_rank0.log)
- [logs/20260710-142030_our_comembed_faaddqr_2v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-142030_our_comembed_faaddqr_2v_smoke2_rank0.log)

日志中的直接证据非常明确：

- `Transformer grad norms at step 1: ... =nan`
- `optim/total grad norm=nan`
- `ComEmbed non-finite tensor detected at QRAddResidualRowMemory.gated_out`

也就是说，**失败不是来自数据加载、Slurm、torchrun、checkpoint、或者 X-GRAM 主体骨干网络，而是来自 ComEmbed 的 2v 注入路径本身。**

## 4. 具体炸在哪里

定位结果显示，非有限值第一次被检测到的位置是：

```text
QRAddResidualRowMemory.gated_out
```

也就是：

1. frequency-aware router 先给出物理 row
2. row memory 做 QR/add/residual 组合
3. 最后 gated output 输出时出现非有限值

这是关键，因为它说明问题不在外层 trainer，而在 ComEmbed 的 row-memory 注入链内部。

进一步看失败日志，出问题的不是单个参数，而是整组 `2v` 注入分支同时变成 `nan`：

- `injection_v_embeddings/0/0/... = nan`
- `injection_v_embeddings/0/1/... = nan`
- `injection_v_gates/0/0 = nan`
- `injection_v_gates/0/1 = nan`

这个现象很重要。它意味着问题不是“某一个参数随机坏了”，而是**双 view 同层注入这个结构本身在当前实现里不稳定。**

## 5. 为什么判断问题在 2v，而不是别的地方

这个判断不是拍脑袋得出的，是对比实验跑出来的。

### 5.1 失败的版本

以下版本持续失败：

- `fa_add_qr` + `2v`
- `fa_qr` + `2v`
- `fa_norm_qr` + `2v`

说明问题不只是哪一个 ComEmbed 子变体的参数初始化，而更像是 **“同一层重复 2 个 value view 注入”** 这个结构带来的不稳定。

### 5.2 成功的版本

当把 `v_layers` 从：

```yaml
[0, 0]
```

改成：

```yaml
[0]
```

也就是从 `2v` 改成 `1v` 后，训练开始稳定：

- 2 step 成功
- 10 step 成功
- 50 step 成功
- 200 step 成功

这个对比非常强，说明：

- 训练框架本身没有坏
- 数据也没有坏
- ComEmbed 接入方式本身不是完全错误
- 真正的问题集中在 **duplicated 2v injection**

## 6. 我做了哪些修复尝试

在放弃原始 `2v` 之前，已经做过一轮系统排查和稳定化处理。主要包括：

### 6.1 数值稳定相关

- 把 ComEmbed 模块强制用 `float32`
- 减小 gate 初始化值
- 输出端增加 RMSNorm
- 可选关闭 row gate
- 默认不让 ComEmbed 走 shortconv 路径

目的：

- 避免低精度放大数值不稳定
- 避免 gate 一开始开太大，直接把异常值放大
- 让输出尺度更受控

### 6.2 优化器相关

- 给 ComEmbed 参数单独设置更小 LR
- 把 ComEmbed weight decay 设为 `0`
- 从 `SkipStepAdamW` 切到普通 `AdamW`

目的：

- 避免新注入模块更新过猛
- 避免对 QR codebook / residual 结构做过强正则
- 减少优化器特殊逻辑带来的额外不稳定

### 6.3 编译与运行路径相关

- 关闭 `torch.compile`
- 保持其余训练主链不变

目的：

- 去掉编译器可能引入的调试困难和图融合副作用

## 7. 为什么这些修复仍然救不回原始 2v

因为这些修复主要解决的是：

- 数值精度问题
- 优化器过猛问题
- gate 初始化过大问题

但实验结果显示，**这些都不是根因，只是诱因或者放大器。**

根因更可能是：

- 同一 `v` 层同时挂两个 ComEmbed view
- 两条分支共享同一注入位置
- 在 warmup / gate / row-memory 聚合的组合下，早期梯度耦合过强

换句话说，当前 `2v` 不是“再把 LR 调小一点就好”，而是结构上就已经不稳定。

## 8. 最终为什么换成新方案

因为当前任务要求是：

- 尽快拿到一个能跑通的实验
- 给出结果

如果继续死磕原始 `2v`，很可能会一直卡在数值爆炸上，拿不到任何完整训练结果。

所以策略必须改变：

- **不再把“完全复现原始 2v”当作当前阶段目标**
- **先退到一个已经被验证能稳定训练的最小可行版本**

这个新方案是：

- `fa_add_qr`
- `targets: [v]`
- `v_layers: [0]`
- `plain AdamW`
- `OLMO_COMPILE=0`
- `OLMO_COMEMBED_LR=1e-6`
- `OLMO_COMEMBED_WEIGHT_DECAY=0`

也就是把原来的 `2v` 改成 `1v`，其他稳定性修复保留。

## 9. 为什么这个新方案能行

因为它去掉了当前最不稳定的部分：**重复的 2v 注入分支**。

在这个新方案下：

- 注入模块还是 ComEmbed
- 还是 frequency-aware row-memory 路线
- 还是 `fa_add_qr`
- 还是接在 X-GRAM 的 value stream 上

所以它并不是“完全换方法”，而是：

- **保留方法的大方向**
- **去掉当前证据最明确的不稳定结构**

这就是它能跑通的原因。

## 10. 新方案已经跑出来的结果

### 10.1 2 step

- 配置：[configs/our_comembed_faaddqr_1v_smoke2.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke2.yaml)
- 日志：[logs/20260710-182028_our_comembed_faaddqr_1v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-182028_our_comembed_faaddqr_1v_smoke2_rank0.log)

结果：

- 成功结束
- 无 NaN

### 10.2 10 step

- 配置：[configs/our_comembed_faaddqr_1v_smoke10.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke10.yaml)
- 日志：[logs/20260710-182453_our_comembed_faaddqr_1v_smoke10_rank0.log](/home/bcjiang/X-gram/logs/20260710-182453_our_comembed_faaddqr_1v_smoke10_rank0.log)

结果：

- `train/CE loss=8.871`
- `train/PPL=7,123`

### 10.3 50 step

- 配置：[configs/our_comembed_faaddqr_1v_smoke50.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke50.yaml)
- 日志：[logs/20260710-182845_our_comembed_faaddqr_1v_smoke50_rank0.log](/home/bcjiang/X-gram/logs/20260710-182845_our_comembed_faaddqr_1v_smoke50_rank0.log)

结果：

- `train/CE loss=7.709`
- `train/PPL=2,229`

### 10.4 200 step

- 配置：[configs/our_comembed_faaddqr_1v_smoke200.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke200.yaml)
- 日志：[logs/20260710-183759_our_comembed_faaddqr_1v_smoke200_rank0.log](/home/bcjiang/X-gram/logs/20260710-183759_our_comembed_faaddqr_1v_smoke200_rank0.log)
- 运行目录：[runs/our_comembed_faaddqr_1v_smoke200-83040](/home/bcjiang/X-gram/runs/our_comembed_faaddqr_1v_smoke200-83040)

最终结果：

- `step=200/200`
- `train/CE loss=6.999`
- `train/PPL=1,095`
- `optim/total grad norm=1.007`
- `throughput/total tokens=819,200`
- checkpoint 保存成功
- training complete

## 11. 当前结论

### 11.1 现在“不行”的部分

不行的是：

- `METHOD.md` 当前指向的原始 `2v` ComEmbed 设定

具体不行在：

- `step 1` 反向之后，ComEmbed 注入分支梯度直接变成 `NaN`
- 第一个被抓到的坏点是 `QRAddResidualRowMemory.gated_out`

### 11.2 现在“可以”的部分

可以的是：

- 新方案 `1v` ComEmbed on X-GRAM

它已经能稳定跑完 200 step，并给出正常 loss / PPL 曲线。

## 12. 下一步该怎么做

当前最合理的路线不是继续盲目跑 `2v`，而是分两条线：

### 路线 A：先拿结果

如果目标是先有能交付的实验结果，那么继续用当前稳定新方案：

- `fa_add_qr`
- `1v`
- 当前稳定优化器和数值设置

然后把训练长度继续拉长。

### 路线 B：单独攻克 2v

如果目标是严格复现 `METHOD.md` 里的原始 `2v`，那需要单独做结构级调试，重点查：

- duplicated value views 在同层的耦合
- gate / warmup / row-memory 聚合的组合效应
- 是否需要把两个 view 放在不同层，而不是同层重复
- 是否需要对两个 view 做更强的独立归一化或缩放

## 13. 本次建议

当前阶段建议采用：

- 用新方案继续长跑，先拿到稳定结果
- 把原始 `2v` 问题单独列成后续修复任务

这样做的原因很简单：

- 它能保证你现在手里有可复现、可运行、可汇报的结果
- 同时不会把“原始 2v 结构不稳定”这个事实掩盖掉
