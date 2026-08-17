# ComEmbed 实现教程

日期：2026-07-20  
目的：说明 ComEmbed 接入 X-GRAM 时真正需要改动的模块边界。  
说明：本文件面向实现者，不是最终结果汇报。

---

## 1. bg

ComEmbed 的思想是替换 lookup memory，而不是重写整个 X-GRAM 注入框架。

## 2. Objective

实现时只回答两件事：

1. 应该改哪些文件。
2. 哪些模块不应该碰。

## 3. Experimental Setup

### 3.1 建议修改范围

- `embedding_injection/comembed.py`
- `xgram.py`
- 配置类与 dispatch 逻辑

### 3.2 不建议改动

- attention 主干
- ShortConv 主逻辑
- 基础训练器

## 4. Results

ComEmbed 已证明可以作为一条独立方法线接入 X-GRAM，但稳定性依赖具体 setting。

## 5. Conclusion

实现 ComEmbed 时应始终把目标限制在“替换 lookup，而不是重写 X-GRAM”。
