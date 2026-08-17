# Engram 复现错误复盘

日期：2026-07-20  
目的：解释早期 Engram 为什么“能跑但无效”。  
说明：本文件记录的是错误实验的根因，不代表当前正式结果。

---

## 1. bg

早期 Engram 的核心问题不是单纯 loss 高，而是实验对象已经被改坏。

## 2. Objective

回答三个问题：

1. 旧版 Engram 错在什么地方。
2. 这个错误为什么会直接毁掉研究结论。
3. 之后是怎么修正回来的。

## 3. Experimental Setup

### 3.1 已确认错误

- `embeddings.*`、`blocks.*`、`lm_head.*` 被冻结。
- 只有注入模块在更新。
- 主干随机初始化但不学习。

## 4. Results

- 这种设置不等价于 `Engram + LM backbone` 联合训练。
- 旧结果不能拿来与 `Baseline / X-gram` 做公平对照。
- 修正方向是恢复 from-scratch 全量训练主干。

## 5. Conclusion

只要 backbone 被冻结，Engram 结果就不具备正式研究意义。
