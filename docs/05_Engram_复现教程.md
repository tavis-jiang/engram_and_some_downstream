# Engram 复现教程

日期：2026-07-20  
目的：给出 Engram 的统一复现路径。  
说明：本文件保留操作逻辑，不再保留历史临时分支。

---

## 1. bg

Engram 复现的重点不是“脚本能启动”，而是“实验口径必须与对照组公平一致”。

## 2. Objective

复现时需要依次确认：

1. 配置是否正确。
2. 主干是否参与训练。
3. checkpoint 是否可恢复。
4. 后续评测是否对齐历史 12 任务口径。

## 3. Experimental Setup

### 3.1 当前推荐配置

- `configs/faircompare_engram_360m.yaml`

### 3.2 关键检查项

- `embedding_injection.mode=Engram`
- `h_layers` 是否覆盖目标层
- 是否没有冻结 backbone
- 保存路径是否连续

## 4. Results

当前正式 Engram 组会结果应以：

- `docs/15_Engram_faircompare_三模型汇报报告.md`

为准。

## 5. Conclusion

Engram 教程的核心不是命令，而是“配置口径正确 + 训练状态可验收 + 评测口径一致”。
