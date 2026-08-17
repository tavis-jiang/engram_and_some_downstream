# ComEmbed 零基础教学

日期：2026-07-20  
目的：帮助第一次接触 ComEmbed/X-GRAM 的读者建立正确概念。  
说明：本文件是教学入口，不作为结果引用口径。

---

## 1. bg

ComEmbed 的难点通常不在命令，而在不知道它到底替换了哪一层逻辑。

## 2. Objective

读完后应能回答：

1. X-GRAM 的 lookup、ShortConv、injection 分别是什么。
2. ComEmbed 替换了哪一段。
3. 为什么这会影响参数效率和稳定性。

## 3. Experimental Setup

### 3.1 简化理解

- X-GRAM：原始注入主线。
- ComEmbed：lookup 替代方案。

## 4. Results

只要这层边界想清楚，后续实现和调试会简单很多。

## 5. Conclusion

教学层最重要的是先建立模块边界，而不是先背配置名。
