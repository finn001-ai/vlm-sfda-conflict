# CT-DUET v1 归档

日期：2026-08-01  
状态：**完整 VisDA-C 实验未超过原始 DUET；从当前主线移除。**

## 归档目的

本目录保存 CT-DUET v1 的实现、完整运行日志和结果。该实验验证的是一个
窄假设：对于尚未通过 DUET 双视角一致性门控的冲突样本，如果真实类别
大概率位于 task/CLIP 两个 top-1 候选之一，是否可以把候选对之外的
`C-2` 个类别作为互补负标签，从而提高目标域适应准确率。

实验结果不支持把这一静态互补损失继续作为当前候选，但不否定“识别并
利用冲突样本”这一研究方向。

## 实现快照

- 原实现提交：`27b63bee2c9e21d6cfacfe0db4f431a697583d8f`
- 本地归档标签：`archive/ct-duet-v1-20260801`
- 完整补丁：`ct_duet_implementation.patch`

原实现的核心路径为：

```text
duet-sfda-main/src/methods/oh/ct_duet.py
duet-sfda-main/src/methods/oh/plmatch.py
duet-sfda-main/src/utils/complementary_learning.py
duet-sfda-main/cfgs/visda/ct_duet.yaml
duet-sfda-main/tools/run_visda_ct_duet.sh
duet-sfda-main/tests/test_ct_duet.py
```

CT-DUET v1 没有单独的时间模块。它只在 DUET 每轮 refresh 后，对
`未准入且 task_top1 != clip_top1` 的样本增加互补负标签损失；样本一旦
进入 DUET 单调准入 mask，便永久退出 CT 路径。当前运行还同时开启了
首轮 prior (`POWER=0.5`)，因此不是纯 CT 消融。

## VisDA-C 完整实验

设置：完整 55,388 个目标样本、8 cycles、每轮 4 epochs、seed 2020。

| 指标 | 原始 DUET | CT-DUET v1 | 差值 |
| --- | ---: | ---: | ---: |
| final mean per-class accuracy | 91.50 | 91.49 | -0.01 |
| oracle peak | 91.52 | 91.51 | -0.01 |

CT-DUET 每轮最终 checkpoint 相对原始 DUET 的差值为：

```text
+0.71, +0.14, +0.10, +0.06, 0.00, +0.06, -0.01, -0.01
```

相关结果文件：

```text
ct_duet_visda_seed2020_raw.txt
ct_duet_visda_seed2020_summary.json
ct_duet_visda_seed2020_per_class.csv
```

## 主要因果结论

1. Cycle 1 训练开始前，CT 尚未产生任何梯度，校准后的 `all_mix` 已经比
   DUET 高 0.82 个百分点、准入样本多 774 个。因此早期优势首先由首轮
   prior 触发，不能归因给 CT。
2. 从 cycle 2 到 cycle 8，CT-DUET 的 mixed 准入伪标签精度持续比原始
   DUET 低 0.10--0.20 个百分点；all-mix 与 CLIP 刷新准确率也持续更低。
3. CT 使用的 task weak top-1 和 CLIP top-1 已经分别通过原始 DUET 的
   weak/strong consistency 与 CLIP KL 进入训练，因此互补损失没有引入
   独立信息，只是对同一信息增加类别外概率抑制。
4. CT 不判断两个候选中谁正确。其负学习梯度会更强地强化当前概率较大的
   候选，存在重复自训练偏好的风险。
5. CT 有效样本从 cycle 1 的 27,449 个下降到 cycle 8 的 608 个，并按完整
   batch 归一化；后期整体作用约为首轮的 1/43。
6. 最终逐类变化主要是收益与损失抵消，例如 bicycle `+0.66`、truck
   `+0.85`，但 car `-1.21`。由于本实验同时改变 prior 与 CT，不能把这一
   类别重分配单独归因给 CT。

## 结论与使用限制

该实验只否定以下具体机制：

> 仅对尚未准入的双视角冲突样本施加静态候选集合外互补负学习，可以超过
> 原始 DUET。

它不否定冲突样本本身的价值。若后续继续该研究方向，必须先离线证明某个
无标签信号能够在 task/CLIP 两个候选之间产生净纠正，再进入完整训练；
不应继续调节 CT v1 权重或运行更多 seed。

## 恢复方式

查看原始实现：

```bash
git show archive/ct-duet-v1-20260801:duet-sfda-main/src/utils/complementary_learning.py
```

需要恢复完整补丁时，在保存本归档后的基线提交上执行：

```bash
git apply archive/sfda_conflict_ct_duet_v1_2026-08-01/ct_duet_implementation.patch
```
