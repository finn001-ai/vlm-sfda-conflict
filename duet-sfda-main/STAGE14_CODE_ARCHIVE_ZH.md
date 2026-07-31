# Stage14 历史代码索引

Stage14、Boundary-Flip、stable memory、target head 和 GTR 已从当前
`main` 的运行路径删除。当前候选是原始 DUET 加且仅加首轮 prior。

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

1. `src/methods/oh/plmatch.py::train_target`
2. `src/utils/first_cycle_prior.py`
3. `src/methods/oh/duet_first_cycle_prior.py`
4. `cfgs/office-home/duet_first_cycle_prior.yaml`

旧 Stage14 代码只从上面的 Git 标签查阅，不再从当前分支运行。
