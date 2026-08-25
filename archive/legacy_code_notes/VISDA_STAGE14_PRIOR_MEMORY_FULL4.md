# Stage14 VisDA 完整数据因果消融

> 历史报告：相关运行代码已从当前 `main` 删除，请从 Git 标签
> `archive/dccl-full-pre-prune-20260728` 查阅。

本实验保留 Stage14，不加入 DINO、CLIP 之外的新视觉模块，也不固定
`car/truck` 等目标类别。它只回答两个问题：

1. `both_prior` 类别先验校正是否过度改变了 VisDA 的类别质量；
2. `stable/reversible` 伪标签记忆是否以过多 coverage 换取 precision。

除下面两个因素外，Stage14 的 target head、GTR 和三个 loss 权重全部固定：

| 组别 | `CALIB_MODE` | `PL_MEMORY` |
| --- | --- | --- |
| 当前 Stage14 | `both_prior` | `stable` |
| 去先验校正 | `none` | `stable` |
| 去稳定筛选 | `both_prior` | `monotonic` |
| 同时去除 | `none` | `monotonic` |

另有一组完全匹配的官方 DUET control。所有组使用 VisDA 完整 55,388 个
目标样本、seed 2020 和 4 个 cycle。

## 云端运行

在仓库根目录执行：

```bash
git pull
bash tools/run_visda_stage14_prior_memory_full4.sh
```

脚本分成两阶段。第一阶段先运行 DUET 与当前 Stage14；只有当
`Stage14 - DUET <= -0.15` 个百分点、即完整数据上的退化被复现时，才继续
剩余三组。没有复现则以退出码 2 停止，避免继续消耗算力。

按照已有日志估算，第一阶段约需要 2.5 小时；通过后，第二阶段约需要
4 小时。实际时间取决于 GPU 和数据读取速度。中断后可直接重跑，已经完整的
实验会自动复用；脚本不会混用不完整日志。

## 输出

结果写入：

```text
output/uda/VISDA-C/stage14_prior_memory_full4_seed2020/
```

主要文件：

- `reproduction_gate.json`：Stage14 退化是否在完整数据上复现；
- `factorial_summary.json`：先验、记忆及二者交互的效应和下一步决策；
- `per_class_factorial.csv`：四个 Stage14 组相对 DUET/当前 Stage14 的
  per-class 变化；
- `source_sha256.txt` 与 `validation_list_sha256.txt`：输入一致性记录。

这是因果诊断，不会自动启动 8-cycle、seed sweep 或 Office-Home。只有当某个
结构在 cycle 4 超过 DUET 至少 0.10 个百分点时，汇总文件才建议冻结该结构并
进入 8-cycle seed-2020 确认。
