# ComEmbed 2v 失败实验报告

日期：2026-07-11  
目的：记录 `ComEmbed fa_add_qr 2v` 的失败现象与根因。  
说明：本文件记录的是失败实验，不是当前可交付正式结果。

---

## 1. bg

`2v` 的问题在于真实训练一开始就发生数值不稳定，而不是训练后期退化。

## 2. Objective

需要回答：

1. 它是在哪里炸的。
2. 为什么可以排除数据、硬件和 dry-run。
3. 失败点更像是结构问题还是数值问题。

## 3. Experimental Setup

### 3.1 已确认事实

- dry-run 可以通过。
- 第一个真实 batch backward 后出现 `NaN/Inf`。
- `optim/total grad norm` 非有限。

## 4. Results

结论是数值稳定性问题，而不是数据或调度问题。

## 5. Conclusion

原始 2v 设定不能直接作为正式实验线继续推进。
