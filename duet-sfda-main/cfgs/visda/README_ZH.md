# VisDA 当前研究入口

当前保留两个严格匹配的 YAML：

- `plmatch.yaml`：原始 DUET；
- `duet_first_cycle_prior.yaml`：原始 DUET 加且仅加首轮 prior。

先运行一次 25% DUET control，再运行候选：

```bash
bash tools/run_visda_plmatch_proxy25_control.sh
bash tools/run_visda_duet_first_cycle_prior_proxy25.sh
```

直接运行完整 8-cycle DUET-FCP：

```bash
bash tools/run_visda_duet_first_cycle_prior_full8.sh
```

Office-Home 12 任务纯 DUET 完成后，统一表格写入
`output/uda/benchmark_tables/`：

```bash
bash tools/run_office_home_plmatch_all.sh
python tools/build_duet_benchmark_tables.py
```

Stage14、Boundary-Flip 和更早实验统一保存在 Git 标签：

```text
archive/dccl-full-pre-prune-20260728
```
