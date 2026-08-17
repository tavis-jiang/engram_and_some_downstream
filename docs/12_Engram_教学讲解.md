# Engram 实验教学讲解文档

日期：2026-07-20  
目的：从研究和工程两个视角讲清楚 Engram 实验发生了什么。  
说明：本文件是教学文档，不等于最终组会结果口径。

---

## 1. bg

Engram 这条线既有研究变量，也有大量工程修复，因此需要单独讲解。

## 2. Objective

本文件帮助读者理解：

1. Engram 与 Baseline/X-gram 的关系。
2. 为什么错误版 Engram 不成立。
3. 为什么当前 faircompare 版可以作为正式对照起点。

## 3. Experimental Setup

### 3.1 关键问题

- 注入层放在哪。
- 主干是否冻结。
- 为什么 OOM 会改变实验策略。

## 4. Results

当前可汇报的正式结果以 `docs/15_Engram_faircompare_三模型汇报报告.md` 为准。

## 5. Conclusion

教学层的核心是把“方法差异”和“工程差异”严格分开。
