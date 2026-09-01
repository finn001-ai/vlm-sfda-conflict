# DCR Rank-Adaptive Alignment 记录（2026-09-01）

## 结论

DCR 的三个正式数据集统一使用 `rank_adaptive`，不再为不同数据集手工指定 `batch_iic` 或 `samplewise_kl`。

训练开始时计算类别联合矩阵的最大秩覆盖率：

```text
rank_coverage = min(configured_batch_size, class_count) / class_count
```

使用一个全局固定门槛 `0.75`：

```text
rank_coverage >= 0.75  -> batch_iic + diversity
rank_coverage <  0.75  -> samplewise_kl，不额外使用 diversity
```

分支在训练开始时只选择一次，并在整次训练中固定。判断不读取数据集名称和目标域 GT。

正式公共配置：

```yaml
ALIGNMENT_MODE: rank_adaptive
MIN_IIC_RANK_COVERAGE: 0.75
DIVERSITY_DELTA: 0.1
```

## 三个数据集的自动选择

正式训练 batch size 均为 64：

| 数据集 | 类别数 | 最大秩覆盖率 | 自动分支 | 有效 diversity |
|---|---:|---:|---|---:|
| VisDA-C | 12 | 100.00% | batch_iic | 0.1 |
| Office-Home | 65 | 98.46% | batch_iic | 0.1 |
| DomainNet-126 | 126 | 50.79% | samplewise_kl | 0.0 |

因此它是同一条结构规则，不是按数据集名称写死的三套方法。

## 为什么放弃“全部统一为 samplewise_kl”

提交 `31c49f9` 曾把三个数据集全部改为：

```text
samplewise_kl + DIVERSITY_DELTA=0.0
```

该修改一次改变了两个变量，不能把下降只归因于 samplewise KL。

VisDA-C 的失败运行在第 8/15 epoch 已表现出过快共识：

```text
initial_conflicts = 28577
epoch8 conflicts  = 5440
feedback_mean     = 0.9556
task_preferred    = 12.18%
overall accuracy  = 87.54%
last class acc    = 40.01%
```

整体共识迅速增加，但少数类继续下降，说明逐样本追随噪声记忆并关闭类别多样性约束不适合低类别数据集。Office-Home 的明显下降由用户报告，当前没有在本地归档到完整数值日志。

## DomainNet-126 已确认结果

DomainNet-126 CP 的有效设置为：

```text
samplewise_kl
effective diversity = 0.0
final fixed ACC = 80.69%
trajectory peak = 80.75%
```

Rank-adaptive 在 DomainNet-126 上自动选择完全相同的有效损失，因此该 CP 结果仍然有效，不需要为方法一致性重跑。

## 旧结果能否复用

- 早期完整 Office-Home/VisDA-C 的 `batch_iic + diversity=0.1` 结果，与 rank-adaptive 在这两个数据集上的有效损失完全相同，可以保留。
- 2026-09-01 之后产生的“全数据集 samplewise”Office-Home/VisDA-C 结果不能作为正式结果，应标记为失败路线。
- DomainNet-126 已完成的 samplewise CP 结果与 rank-adaptive 的自动分支完全相同，可以保留。
- 若论文需要新版日志名称或精确 `stage_timing.csv`，每个数据集选择一个代表任务重跑即可；不需要仅为改名重跑所有任务。

## 建议的验证顺序

1. 停止尚未完成的 VisDA-C 全 samplewise 运行。
2. 用 rank-adaptive 重跑一次 VisDA-C，确认恢复原稳定轨迹。
3. Office-Home 先重跑一个代表任务；与旧 IIC 结果一致后，直接复用其余已有完整任务。
4. 尚未做过的 seed 或迁移任务仍需正常补跑。

## 对应提交

```text
31c49f9 统一 samplewise 与训练计时（失败的统一尝试）
dc204ff 引入 rank-adaptive alignment
```

后续固定分支补丁应位于 `dc204ff` 之后。

