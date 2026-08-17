# Engram 正式训练监控与验收

日期：2026-07-20  
目的：记录 Engram 正式训练过程中应监控与验收的事项。  
说明：本文件保留监控标准，不再保留过时的实时状态。

---

## 1. bg

Engram 训练历史上既遇到过口径错误，也遇到过恢复链路问题，因此监控标准必须明确。

## 2. Objective

训练期间主要看三件事：

1. loss 是否正常下降。
2. checkpoint 是否按计划写出。
3. 恢复后 run 是否仍然连续有效。

## 3. Experimental Setup

### 3.1 验收标准

- 能训练到目标终点。
- 最终 checkpoint 目录完整。
- 日志中出现 `Training complete`。
- 下游评测可成功读取最终 checkpoint。

## 4. Results

当前 Engram faircompare 已满足正式验收条件。

## 5. Conclusion

后续若再开新 Engram run，仍应沿用同样的验收清单。
