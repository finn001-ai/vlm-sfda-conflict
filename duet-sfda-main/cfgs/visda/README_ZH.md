# VisDA 当前研究入口

当前保留两个严格匹配的 YAML：

- `plmatch.yaml`：原始 DUET；
- `duet_first_cycle_prior.yaml`：原始 DUET 加且仅加首轮 prior。

先运行一次 25% DUET control，再运行候选：

```bash
bash tools/run_visda_plmatch_proxy25_control.sh
bash tools/run_visda_duet_first_cycle_prior_proxy25.sh
```

Stage14、Boundary-Flip 和更早实验统一保存在 Git 标签：

```text
archive/dccl-full-pre-prune-20260728
```
