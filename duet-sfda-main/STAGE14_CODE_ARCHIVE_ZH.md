# Stage14 与 Dual-tier Pending 代码存档

## 存档目的

本文件固定记录两个可回退版本、当前代码入口和已知边界，避免后续开发覆盖
Office-Home 表现最好的 Stage14，或误把 Dual-tier Pending 消融当成已经利用
Conflict 的最终方法。

## Git 存档点

| 存档 | Git 引用 | 内容 |
| --- | --- | --- |
| Stage14 stable | `archive/stage14-stable-20260727` | Dual-tier 合入前的 Stage14 |
| Stage14 + Pending | `archive/stage14-dual-tier-20260727` | 提交 `bd93e29`，包含 Dual-tier、测试和完整运行脚本 |

查看某个存档而不修改当前分支：

```bash
git show archive/stage14-stable-20260727:duet-sfda-main/src/methods/oh/dccl.py
git show archive/stage14-dual-tier-20260727:duet-sfda-main/src/utils/pseudo_label_memory.py
```

如需实际回退，应先创建新分支，不要覆盖当前实验分支。

## 实验结果存档

- `../archive/sfda_conflict_visda_stage14_prior_memory_2026-07-27/`：
  prior × memory 因果汇总、逐类别表和四份原始日志。
- `../archive/sfda_conflict_visda_stage14_dual_tier_2026-07-27/`：
  Dual-tier 汇总与逐类别表。

两个目录均包含 `SHA256SUMS`，用于核对云端回传文件是否发生变化。

## Stage14 stable 的固定入口

配置文件：`cfgs/visda/temporal_precision_head.yaml`

核心设置：

```text
CALIB_MODE=both_prior
PL_MEMORY=stable
PL_STABLE_CYCLES=2
PL_STABLE_MEMORY=reversible
TARGET_HEAD_ADAPT=True
GTR_PAR=0.05
```

Stable 的定义是“当前 source/CLIP top-1 一致，且同一一致标签连续出现至少
2 cycles”。未达到稳定条件的样本不进入伪标签 hard CE。

## Dual-tier Pending 的固定入口

运行脚本：`tools/run_visda_stage14_dual_tier_full4.sh`

相对 Stage14 只改变伪标签记忆监督：

| 状态 | 条件 | hard CE 权重 |
| --- | --- | --- |
| Stable | 当前一致，且同一标签连续至少 2 cycles | `1.0` |
| Pending | 当前一致，但尚未连续 2 cycles | `0.5 * mix_conf` |
| Conflict | 当前 source/CLIP top-1 不一致 | `0` |

Cycle 1 是 memory warmup，所有当前一致样本权重为 1。

## “进入 CE”的准确含义

训练时整个 batch 都会完成网络前向传播并计算 `weak_logits` 和
`strong_logits`。`hard_mask` 只筛选参加伪标签分类交叉熵的样本：

```python
filtered_idx = tar_idx[hard_mask[tar_idx]]
```

Dual-tier 随后计算逐样本 CE：

```python
per_sample = cross_entropy(logits, labels, reduction="none")
loss = (per_sample * weights).sum() / weights.sum()
```

因此：

- 权重 1：完整参与该项 CE；
- 权重 0.4：相对权重 1 的样本，损失和梯度贡献被缩小；
- 权重 0：不贡献该项 CE 的损失和梯度。

权重 0 不代表样本从 dataloader 删除，也不代表完全不训练。它仍可能参与
consistency、KL 等其他无监督目标。

## 当前尚未完成的目标

本项目最初目标是利用 source/CLIP Conflict。Dual-tier 目前只恢复了 Pending，
并没有完成该目标。代码在 Dual-tier 模式下明确禁止旧 promotion/ACCD 路径将
Conflict 送回 hard CE，以保证这次消融只测量 Pending 的因果效应。

因此，当前结果只能支持：

1. Stable 直接删除 Pending 会损失有效监督；
2. `0.5 * mix_conf` 能恢复一部分性能；
3. 不能据此声称 Conflict 已被利用或冲突标签已被纠正。

后续 Conflict 方法必须单独实现和消融：只有当 source、CLIP 和历史证据能够
判断一方更可靠时，才恢复为弱监督；无法消解的冲突继续不进入 hard CE。

## 代码阅读顺序

1. `cfgs/visda/temporal_precision_head.yaml`
2. `src/methods/oh/dccl.py::apply_pseudo_label_memory`
3. `src/utils/pseudo_label_memory.py::dual_tier_supervision`
4. `src/utils/pseudo_label_memory.py::weighted_cross_entropy`
5. `src/methods/oh/dccl.py` 中 `hard_mask`、`filtered_idx` 和
   `stable_ce_loss` 所在训练段
6. `tests/test_dual_tier_memory.py`
7. `tools/run_visda_stage14_dual_tier_full4.sh`

## 配置清理原则

- Stage14 基线 YAML 不再混入 Dual-tier 专用的 `PL_PENDING_WEIGHT`。
- Dual-tier 参数由专用脚本显式传入，日志中必须验证。
- 已被历史脚本引用的失败/诊断配置不删除，以保证实验可复现；通过
  `cfgs/visda/README_ZH.md` 标记主线与历史入口。
- 后续新增伪标签方法必须使用独立 mode、测试和运行脚本，不直接覆盖
  `stable` 或 `dual_tier`。
