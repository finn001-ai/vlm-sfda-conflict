# VisDA 冲突样本特征稳定性离线审计

本阶段只判断一个问题：task 模型的 weak/strong 特征余弦，能否比 task
confidence 更可靠地识别 DUET 冲突样本中的错误自一致性梯度。

该审计不会训练模型：不构造 optimizer、不执行 optimizer step、不修改模型参数，
也不保存 checkpoint。

## 运行

在代码目录执行：

```bash
bash tools/run_visda_conflict_feature_gravity_audit.sh
```

输入仍是原始 DUET seed 2020 的三个 source checkpoint、完整 VisDA validation
列表以及原始 CLIP ViT-B/32。脚本拒绝使用 25% adaptation list，避免把代理集选择
混入本次机制判断。

预计只需要一次完整目标域 weak/strong 前向和输出层反事实梯度计算，目标运行时间
不超过约 5--10 分钟；它不会调用 `image_target_of_oh_vs.py`。

## 标签隔离契约

脚本分为两个阶段：

1. 忽略 DataLoader 返回的标签，只计算 task/CLIP 预测、weak/strong 特征余弦、
   原始一致性与 CLIP KL 的 logit 梯度分量。
2. 写出 label-free CSV/NPZ，并用 SHA256 清单锁定。
3. 锁定后才从 `validation_list.txt` 解析真实标签，生成明确标记为
   `oracle_diagnostic` 的报告。

无标签权重固定为：

```text
raw_weight = max(cos(task_weak_feature, task_strong_feature), 0)
weight = raw_weight / mean(raw_weight over task/CLIP conflicts)
```

均值归一化用于保持冲突样本上一致性项的总体尺度，避免把效果混同为降低
`CON_PAR`。权重只作为离线反事实，不进入模型。

## 预声明否决门槛

全部满足才输出 `PASS_OFFLINE_GATE`：

1. 原始 DUET/CLIP/算术融合输出通过复现检查。
2. 特征余弦预测 `task_correct` 的 AUROC 至少比 task confidence 高 `0.02`，
   且配对 bootstrap 95% CI 下界大于 0。
3. 特征余弦最高与最低五分位的 task 正确率差至少为 `5 pp`。
4. 相对原始 DUET，oracle-harmful logit-gradient mass 至少下降 `10%`，同时
   oracle-helpful mass 至少保留 `95%`。
5. car、person、truck 的 harmful gradient mass 分别均不增加。

任意一项不满足即输出 `REJECT`，结论是不得运行 proxy 或完整 VisDA。
即使通过，本脚本也只建议请求一次单独批准的 matched proxy，不自动授权训练。

## 输出

结果位于：

```text
output/uda/VISDA-C/TV/plmatch_visda_feature_gravity_audit_seed2020/
  feature_gravity_audit/
```

关键文件：

- `visda_conflict_feature_gravity_signals.csv`：无标签逐样本信号；
- `visda_conflict_feature_gravity_signals.npz`：无标签概率和梯度张量；
- `visda_conflict_feature_gravity_signal_lock.json`：标签揭示前的 SHA256 锁；
- `visda_conflict_feature_gravity_oracle_diagnostic.csv`：oracle 逐样本诊断；
- `visda_conflict_feature_gravity_classwise.csv`：逐类梯度安全性；
- `visda_conflict_feature_gravity_summary.json`：最终 gate；
- `visda_conflict_feature_gravity_summary.md`：简明报告。
