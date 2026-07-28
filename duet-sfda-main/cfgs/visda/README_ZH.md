# VisDA 当前研究入口

本目录的 DCCL/Boundary-Flip 主线只保留两个 YAML：

- `temporal_precision_head.yaml`：Stage14 matched control；
- `boundary_flip_duet.yaml`：相同宿主，仅额外启用 Boundary-Flip。

直接覆盖并运行 matched control 与 candidate：

```bash
bash tools/run_visda_boundary_flip_duet.sh
```

脚本只会删除它自己命名的两个实验输出目录，不会触碰其他结果。若已有可用
control，可使用：

```bash
RUN_CONTROL=0 bash tools/run_visda_boundary_flip_duet.sh
```

清理前的实验 YAML、脚本和完整代码统一保存在 Git 标签：

```text
archive/dccl-full-pre-prune-20260728
```
