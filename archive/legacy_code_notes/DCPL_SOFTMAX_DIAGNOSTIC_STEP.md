# DCPL 噪声转移矩阵：离线诊断数据获取与评估

## 为什么需要这份数据

候选方案 DCPL（ECCV 2024, De-Confusing Pseudo-Labels in SFDA）通过估计
12×12 噪声转移矩阵 T 来校正伪标签噪声，核心估计是：

    T[L][j] = 伪标签为 L 的样本的 task softmax 均值（按 L 逐行归一化）

这需要**全部目标样本**的**完整 12 类 softmax**（task + CLIP），而不是只存了
top-2 的冲突子集。现有 `conflict_samples.csv` 只有冲突样本（cycle 0 时 28,223 /
55,388），非冲突样本恰好是 task 与 CLIP 一致的部分，正是 T 对角线的主要来源，
缺失会导致矩阵不可信。

## 数据获取（在 GPU 服务器上执行）

导出脚本：`tools/run_visda_duet_softmax_dump.sh`。它不改训练逻辑，只是在 probe
原本就在内存里算好的全量概率上多存一份 npz；默认不导出，行为与旧版完全一致。

先做最省的验证（推荐）：

    PROBE_MODE=cycle0 bash tools/run_visda_duet_softmax_dump.sh

只跑 cycle 0：源模型 + CLIP 对目标池一次前向，**不训练**，约半小时。产出：

    output/uda/VISDA-C/TV/duet_first_cycle_prior_topk_probe_softmax_dump_cycle0_preflight_seed2020/softmax_dump/cycle_000.npz

npz 内容：`task_probability` [55388,12]、`clip_probability` [55388,12]、
`labels`（仅评估用）、`sample_indices`、`class_names`。

若 cycle 0 诊断通过，再跑完整 8 cycle（需要一次完整训练）：

    PROBE_MODE=full bash tools/run_visda_duet_softmax_dump.sh

## 离线诊断（本地，无需 GPU）

    python tools/analyze_dcpl_confusion.py --dump-dir <softmax_dump 目录>

输出指标：

- 基线：task argmax 精度、CLIP 伪标签精度（全量，首次能算）
- 校正后：`argmax(task_softmax @ T)` 的精度（mean 与 rank 两种 T 估计）
- 关键判据：`corrected - max(task, CLIP)` 是否显著为正；car/truck 两类的
  混淆行是否被 T 正确吸收
- 每类精度明细 + T 矩阵行

## 已知限制

- `cycle_000` 冲突数 28,223 与仓库契约 28,255 不一致：以实际 CSV/npz 为准，
  不改 `VISDA_CYCLE0_REGRESSION` 常量（已知偏差，与诊断无关）。
- GT 只用于离线评估，不参与 T 估计与训练，维持 oracle-diagnostic 原则。
- 诊断是近似：DUET 训练用的伪标签是 task/CLIP 协商后的 `mem_label`，这里先用
  原始 CLIP top1 作为伪标签源验证概念；若通过，再接入真实训练伪标签重估。
