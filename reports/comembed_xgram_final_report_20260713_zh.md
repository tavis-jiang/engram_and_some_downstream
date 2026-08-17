# ComEmbed on X-GRAM 最终实验报告

日期：2026-07-13

## 1. 结论摘要

这次实验有两个结论，必须分开说：

1. `METHOD.md` 里原始目标的 `FAAddQR-2v` 设定，在当前 X-GRAM 训练代码路径下 **不能稳定训练**。
2. 在同一路方法上退成稳定化版本 `FAAddQR-1v` 后，实验已经 **完整跑通到 5000 step**，并拿到了有效结果。

因此，这次最终拿到的结果是：

- **不是原始 exact `2v` setting 的结果**
- **而是原始方法线上的稳定替代版本 `1v` 的结果**

## 2. 原始目标到底是什么

根据 `METHOD.md`，本次最接近主目标的方案是：

- `ComEmbed`
- `fa_add_qr`
- 注入目标 `targets: [v]`
- 保留 X-GRAM 的 frequency-aware routing / hash routing / warm gate / shortconv 主体路径
- 使用多视图 value injection，也就是 `2v`

对应配置文件是：

- [configs/our_comembed_faaddqr_2v.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_2v.yaml)

关键特征是：

```yaml
v_layers: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
```

也就是说，同一层 value path 上重复挂两个 ComEmbed view，这就是这里说的 `2v`。

## 3. 最终真正跑完的 setting 是什么

最终跑通并完成 5000 step 的不是上面的原始 `2v`，而是稳定化后的 `1v`：

- 配置文件：[configs/our_comembed_faaddqr_1v_overnight5000.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_overnight5000.yaml)
- 最终运行目录：[runs/our_comembed_faaddqr_1v_overnight5000-83745](/home/bcjiang/X-gram/runs/our_comembed_faaddqr_1v_overnight5000-83745)
- 最终日志：[logs/20260711-171733_our_comembed_faaddqr_1v_overnight5000_rank0.log](/home/bcjiang/X-gram/logs/20260711-171733_our_comembed_faaddqr_1v_overnight5000_rank0.log)

它的核心设定是：

- `mode: ComEmbed`
- `comembed_variant: fa_add_qr`
- `targets: [v]`
- `v_layers: [0]`

另外，实际提交训练时还叠加了稳定化环境变量：

- `OLMO_USE_PLAIN_ADAMW=1`
- `OLMO_COMPILE=0`
- `OLMO_COMEMBED_LR=1e-6`
- `OLMO_COMEMBED_WEIGHT_DECAY=0`

所以准确说法应该是：

- **方法路线没变**
- **结构从 `2v` 改成了 `1v`**
- **训练超参数也做了稳定化收缩**

## 4. 原始 2v 为什么失败

### 4.1 失败现象

原始 `2v` 不是“指标不好”，而是训练会非常早地数值爆炸。

最小复现配置：

- [configs/our_comembed_faaddqr_2v_smoke2.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_2v_smoke2.yaml)

代表性失败日志：

- [logs/20260710-181740_our_comembed_faaddqr_2v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-181740_our_comembed_faaddqr_2v_smoke2_rank0.log)
- [logs/20260710-181557_our_comembed_faaddqr_2v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-181557_our_comembed_faaddqr_2v_smoke2_rank0.log)
- [logs/20260710-142030_our_comembed_faaddqr_2v_smoke2_rank0.log](/home/bcjiang/X-gram/logs/20260710-142030_our_comembed_faaddqr_2v_smoke2_rank0.log)

反复出现的模式是：

- 第一次前向/反向可以走完
- 在第一次优化更新之后，ComEmbed 注入分支梯度变成 `NaN`
- 很快触发非有限值检查并报错

### 4.2 首个明确出问题的位置

最关键的定位信息是：

```text
ComEmbed non-finite tensor detected at QRAddResidualRowMemory.gated_out
```

这说明：

- 问题不在 Slurm
- 不在 torchrun
- 不在数据集读取主链
- 不在 checkpoint
- 也不在普通 transformer 骨干本身

而是在：

- X-GRAM router 已经给出 row 之后
- ComEmbed row-memory 做 QR/add/residual 聚合
- 最终 gated output 输出时

数值已经坏掉了。

### 4.3 为什么判断根因更偏向 `2v` 结构

这个判断来自对比实验，不是猜的。

失败的版本包括：

- `fa_add_qr + 2v`
- `fa_qr + 2v`
- `fa_norm_qr + 2v`

也就是说，坏掉的不只是某一个 ComEmbed 子变体，而是 **“同层重复双 view 注入”** 这件事本身就不稳定。

同时，当把：

```yaml
v_layers: [0, 0]
```

改成：

```yaml
v_layers: [0]
```

以后，训练就稳定了，并且可以从 2 step 一路扩大到 5000 step。

所以当前最合理的判断是：

- 问题的主因不是“ComEmbed 完全不能接进 X-GRAM”
- 而是“当前实现下 duplicated `2v` injection 在训练初期梯度耦合过强，导致数值不稳定”

## 5. 为什么 2v 会在这里炸

这部分是基于现象和代码路径的工程判断。

当前最可能的原因有四个：

1. 同一层 value stream 同时挂两个 ComEmbed view，早期梯度耦合过强。
2. 两条 view 都经过 row-memory、gate、warmup、hash/frequency-aware routing 聚合，导致注入尺度在第一次更新后迅速放大。
3. `fa_add_qr` 虽然比乘法更稳，但仍然带有 QR codebook、residual、row gate 的组合结构；双分支叠加时更容易把局部异常值同步放大。
4. 当前实现虽然已经做了数值保护，但保护主要是“减轻”，不能根治结构级不稳定。

所以这里更像是：

- **结构问题**
- 而不是单纯学习率问题
- 也不是简单精度问题

## 6. 为了救 2v，已经做过哪些尝试

### 6.1 数值稳定处理

已经做过：

- ComEmbed 模块强制走 `float32`
- 降低 gate 初始化强度
- 增加输出侧 norm / 有限值检查
- 默认限制短卷积路径影响

目的都是控制注入输出尺度，避免刚开始就炸。

### 6.2 优化器稳定处理

已经做过：

- ComEmbed 参数单独更小学习率
- ComEmbed weight decay 设为 `0`
- 改用普通 `AdamW`

目的都是让新注入模块更新更保守。

### 6.3 运行路径稳定处理

已经做过：

- 关闭 `torch.compile`

这样可以减少图编译带来的不确定性，也便于定位问题。

### 6.4 结果

这些修复 **没有救回原始 `2v`**。

这恰好说明：

- 它不是单一超参数太激进
- 也不是某个小 bug 一修就好
- 当前阶段最主要的问题确实在 `2v` 结构本身

## 7. 为什么最终换成 1v

因为任务目标不是无限期调 bug，而是尽快拿到一个可交付结果。

如果继续硬跑 `2v`，大概率只会重复遇到：

- step 1 之后梯度坏掉
- 注入分支 `NaN`
- 长实验无法开始

所以策略改成：

1. 先承认原始 `2v` 当前不稳定。
2. 在同一方法线上退到最小稳定结构 `1v`。
3. 先把可运行、可复现、可汇报的结果拿出来。

这是工程上更合理的路线。

## 8. 实验全过程

### 第一步：实现 ComEmbed 接入 X-GRAM

已完成的主要代码接入包括：

- [OLMo-core/src/olmo_core/nn/embedding_injection/comembed.py](/home/bcjiang/X-gram/OLMo-core/src/olmo_core/nn/embedding_injection/comembed.py)
- [OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py](/home/bcjiang/X-gram/OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py)
- [OLMo-core/src/olmo_core/nn/transformer/config.py](/home/bcjiang/X-gram/OLMo-core/src/olmo_core/nn/transformer/config.py)
- [scripts/train/olmo_train.py](/home/bcjiang/X-gram/scripts/train/olmo_train.py)

这一步说明方法已经能接进去，代码构建和训练主流程是通的。

### 第二步：先跑原始 2v smoke

结果：

- 1 step 可以过
- 2 step 就开始数值炸

这一步确认：代码接线不是完全错的，但训练稳定性不过关。

### 第三步：排查并做稳定化修改

做过：

- float32
- gate 调弱
- optimizer 调保守
- compile 关闭
- 其他数值保护

结果：

- 仍然救不回 `2v`

### 第四步：退到 1v 做逐级验证

新增并验证了这些配置：

- [configs/our_comembed_faaddqr_1v_smoke2.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke2.yaml)
- [configs/our_comembed_faaddqr_1v_smoke10.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke10.yaml)
- [configs/our_comembed_faaddqr_1v_smoke50.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke50.yaml)
- [configs/our_comembed_faaddqr_1v_smoke200.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke200.yaml)
- [configs/our_comembed_faaddqr_1v_smoke500.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_smoke500.yaml)

结果是：

- 2 step 成功
- 10 step 成功
- 50 step 成功
- 200 step 成功
- 500 step 成功

这一步确认：`1v` 不是偶然跑通，而是确实稳定。

### 第五步：启动正式 5000-step 训练

使用配置：

- [configs/our_comembed_faaddqr_1v_overnight5000.yaml](/home/bcjiang/X-gram/configs/our_comembed_faaddqr_1v_overnight5000.yaml)

第一次长跑作业：

- `83080`

结果：

- 不是训练崩溃
- 是运行时限到了，停在大约 `step1600`

这说明训练本身是稳定的，只是单次作业时长不够。

### 第六步：从 checkpoint 恢复继续跑

从 `step1600` 恢复时，又遇到一个独立问题：

- 原始 dataloader 的临时 DuckDB checkpoint 文件在 `/tmp` 下
- 作业结束后文件消失
- 恢复训练时 dataloader state 无法打开

代表性失败日志：

- [logs/20260711-104351_our_comembed_faaddqr_1v_overnight5000_rank0.log](/home/bcjiang/X-gram/logs/20260711-104351_our_comembed_faaddqr_1v_overnight5000_rank0.log)

### 第七步：修 dataloader 恢复逻辑

为了解决上面的恢复问题，补了一个保守回退逻辑：

- 文件：[OLMo-core/src/olmo_core/data/data_loader.py](/home/bcjiang/X-gram/OLMo-core/src/olmo_core/data/data_loader.py)

逻辑是：

- 如果 dataloader ckpt 缺失
- 且设置 `OLMO_ALLOW_MISSING_DATALOADER_CKPT=1`
- 则放弃恢复 dataloader 内部游标
- 直接重建一个新的 dataloader

这里中间还修过一次小错误：

- 回退分支最初漏了 `import os`
- 导致第一次 resume 补丁触发时报 `NameError`
- 补上之后恢复逻辑正常生效

### 第八步：重新 resume 并完成训练

最终恢复作业：

- `83745`

它成功从 `step1600` 接着跑，并最终到达：

- `step5000/5000`

日志明确写到：

- `Saving checkpoint for step 5000 ...`
- `Checkpoint for step 5000 saved successfully`
- `Training complete`
- `Cleanup process completed successfully`

虽然 `sacct` 把这个作业记成了 `TIMEOUT`，但日志已经证明训练本身完成了，最终 checkpoint 也成功写出，所以结果有效。

## 9. 最终结果

最终完成的正式 run：

- 运行目录：[runs/our_comembed_faaddqr_1v_overnight5000-83745](/home/bcjiang/X-gram/runs/our_comembed_faaddqr_1v_overnight5000-83745)
- 最终 checkpoint：[runs/our_comembed_faaddqr_1v_overnight5000-83745/step5000](/home/bcjiang/X-gram/runs/our_comembed_faaddqr_1v_overnight5000-83745/step5000)
- 最终日志：[logs/20260711-171733_our_comembed_faaddqr_1v_overnight5000_rank0.log](/home/bcjiang/X-gram/logs/20260711-171733_our_comembed_faaddqr_1v_overnight5000_rank0.log)

最终指标：

- `step=5000/5000`
- `train/CE loss = 4.922`
- `train/PPL = 137.3`
- `optim/total grad norm = 0.8596`
- `throughput/total tokens = 20,480,000`

## 10. 这次结果应该怎么表述

最准确的表述方式是：

> 原始 `METHOD.md` 中的 `FAAddQR-2v` 设定在当前 X-GRAM 代码路径下不稳定，无法直接完成正式训练；在同一路方法上退成稳定化 `FAAddQR-1v` 后，已成功完成 5000-step 正式实验并得到有效结果。

不要把这次结果说成：

- “原始 2v 已验证成功”

因为这不准确。

正确说法应该是：

- “原始 2v 未解决”
- “稳定化 1v 已跑通并给出结果”

## 11. 后续怎么改进 2v

如果后面要单独攻克原始 `2v`，建议按下面顺序做。

### 11.1 先做结构隔离实验

建议先做最小化对照：

1. `1v` 基线
2. 同层 `2v`
3. 不同层分开 `2v`
4. 去掉 row gate 的 `2v`
5. 去掉 shortconv 的 `2v`

目标不是一开始跑大实验，而是先查清楚到底是哪一段叠加让它炸。

### 11.2 加强分支尺度控制

建议优先试：

- 每个 view 独立更小的 `lambda_init`
- 更长的 warmup
- 更强的 early-stage gate clamp
- 对双分支输出做额外 norm 或 rescale

核心思路是：先压住双分支初期的联合注入强度。

### 11.3 拆开 optimizer 策略

建议试：

- codebook / residual / gate 分不同 param group
- gate 用更小 LR
- residual_proj 继续零初始化并延长 warmup

因为现在很可能不是所有 ComEmbed 参数都应该用同一更新强度。

### 11.4 先做短程稳定性门槛

建议要求：

- 先通过 2 step
- 再通过 50 step
- 再通过 200 step
- 最后再启动正式长跑

不要直接拿还没过短程稳定门槛的 `2v` 去跑长实验。

## 12. 最终判断

这次实验从工程角度是成功的，因为它回答了三个关键问题：

1. `ComEmbed` 能不能接进 X-GRAM？  
   能。

2. `METHOD.md` 原始 `FAAddQR-2v` 能不能直接稳定训练？  
   不能。

3. 在当前代码基础上，能不能拿到一个完整、可复现、可汇报的 ComEmbed-on-X-GRAM 结果？  
   能，使用稳定化 `FAAddQR-1v` 已经完成。

所以当前阶段最合理的结论不是“方法失败”，而是：

- **原始 `2v` 结构还需要继续调**
- **稳定化 `1v` 已经可以作为当前阶段的正式结果hi *
