# Stage14 / Boundary-Flip 代码索引

当前 `main` 只保留两条可运行研究路径：

1. `temporal_precision_head`：Stage14 control；
2. `boundary_flip_duet`：相同 Stage14 宿主加 Boundary-Flip。

清理前包含 ACCD、reciprocal boundary、residual/pair-flow target head、
pair-feature、covariance、three-view、trajectory 等实验的完整仓库，已固定在：

```text
archive/dccl-full-pre-prune-20260728
```

只查看旧文件而不改变工作区：

```bash
git show archive/dccl-full-pre-prune-20260728:duet-sfda-main/src/methods/oh/dccl.py
git show archive/dccl-full-pre-prune-20260728:duet-sfda-main/conf.py
```

当前代码建议按以下顺序阅读：

1. `cfgs/office-home/temporal_precision_head.yaml`
2. `src/methods/oh/dccl.py::train_target`
3. `src/methods/oh/dccl.py::obtain_label`
4. `src/utils/boundary_flip.py::update_boundary_flip_state`
5. `src/methods/oh/boundary_flip_duet.py`

Stage14 的 hard CE 只使用跨 cycle 连续一致的 source/CLIP 伪标签；
Boundary-Flip 只对通过视角、语义、时序和类别对预算门控的稳定翻转增加
方向性监督。真实目标标签只用于实验日志与评估。
