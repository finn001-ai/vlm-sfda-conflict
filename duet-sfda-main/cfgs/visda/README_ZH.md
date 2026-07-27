# VisDA 配置索引

本目录同时包含上游 DUET 基线配置和本项目研究配置。为避免把历史消融误当成
当前主方法，按下面的入口查阅。

## 当前保留的 Stage14

- `temporal_precision_head.yaml`：Stage14 原始稳定记忆配置。其
  `PL_MEMORY=stable`，不包含 Dual-tier 专用的 Pending 权重。
- `tools/run_visda_temporal_precision_head.sh`：Stage14 常规运行入口。

## 最新 Pending 改动

- `src/utils/pseudo_label_memory.py`：Stable/Pending/Conflict 权重构造及加权 CE。
- `src/methods/oh/dccl.py::apply_pseudo_label_memory`：逐 cycle 状态更新和主训练
  路径。
- `tools/run_visda_stage14_dual_tier_full4.sh`：完整 VisDA、4 cycles 的固定实验
  入口；脚本显式覆盖：

```text
DCCL.PL_MEMORY=dual_tier
DCCL.PL_PENDING_WEIGHT=0.5
DCCL.PL_STABLE_CYCLES=2
DCCL.PL_STABLE_MEMORY=reversible
```

- `tests/test_dual_tier_memory.py`：三态掩码、权重和加权 CE 的单元测试。

## 历史/诊断配置

- `reciprocal_boundary.yaml`：已完成的困难边界诊断实验，不是当前主线。
- `plmatch*.yaml`：DUET/PLMatch 对照。
- 其他算法名 YAML：仓库原有对照方法。

历史配置仍被对应运行脚本引用，因此保留文件以保证旧实验可复现；清理时不移动
或删除这些配置。当前研究入口以本文件上方列出的 Stage14 和 Dual-tier 为准。
