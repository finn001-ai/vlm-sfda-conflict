# VisDA DAC → DUET Handoff 存档

更新时间：2026-08-24  
数据：VisDA-C 全量，55,388 个目标样本  
任务：T→V  
seed：2020  
指标：12 类 macro accuracy  
原则：目标域 GT 只用于日志评估，不参与训练、阈值或 checkpoint 选择。

## 最终结论

当前正式候选是 **DAC 15 passes + DUET 17 passes，完整继承 DAC 的 F/B/C**：

```text
固定最终 ACC = 91.70%
原始 DUET 固定最终 ACC = 91.50%
同为 32 target passes
差值 = +0.20 pp
```

91.70 是最后一步 `Iter:4330/4330; Cycle:4/4` 的固定最终结果，不是从中途 checkpoint 按 GT 挑出的 oracle peak。额外一轮仅用于将总预算从31补齐到32，与原始 DUET 的 `8 cycles × 4 passes` 严格对齐；它不是单独的方法创新。

## 三个 handoff 结果

| 版本 | 初始化 | DAC passes | DUET passes | 总 passes | 固定最终 ACC | 相对 DUET |
|---|---|---:|---:|---:|---:|---:|
| 原始 DUET | source F/B/C | 0 | 32 | 32 | 91.50 | 0.00 |
| all-F/B/C handoff | DAC target F/B/C | 15 | 16 | 31 | 91.54 | +0.04 |
| **all-F/B/C exact-budget** | **DAC target F/B/C** | **15** | **17** | **32** | **91.70** | **+0.20** |
| F/B + source-C 消融 | DAC target F/B + source C | 15 | 16 | 31 | 91.51 | +0.01 |

同为31 passes 时，保留 source C 得到91.51，低于完整继承 DAC F/B/C 的91.54。因此“用 source C 保存类别几何”的修改没有带来收益，保留为消融，不作为后续主线。

## 91.70 的最后一个 Cycle

| DUET Cycle 4 进度 | ACC |
|---:|---:|
| 1/5 | 91.38 |
| 2/5 | 91.47 |
| 3/5 | 91.58 |
| 4/5 | 91.56 |
| 5/5，固定最终 | **91.70** |

对应固定最终分类别结果：

```text
airplane 98.82, bicycle 92.66, bus 89.96, car 78.92,
horse 97.91, knife 98.22, motorcycle 94.19, person 85.50,
plant 95.54, skateboard 97.02, train 95.02, truck 76.62
```

相对 DUET 的增益主要来自 bicycle `+3.28`、bus `+0.98` 和 truck `+1.89`；car `-1.69`、motorcycle `-1.74` 有明显下降。因此目前是有希望的单 seed 结果，还不能宣称稳定提升。

## 代码与运行入口

主版本：

```bash
bash tools/run_visda_dac_duet_handoff_full.sh 2020
```

source-C 消融：

```bash
bash tools/run_visda_dac_duet_handoff_fb_sourcec_full.sh 2020
```

成功运行时的代码节点是 Git commit `d1002f0`。当前 main 已重新同时保留两个入口：主脚本对应91.70方案，带 `fb_sourcec` 的脚本只对应91.51消融。

云端91.70 checkpoint 目录：

```text
output/uda/VISDA-C/TV/plmatch_dac_handoff_full32_seed2020/
  target_F.pt
  target_B.pt
  target_C.pt
```

日志不能代替模型权重；该云端目录应继续保留。

## 下一步

只验证完全相同的 exact-budget 主版本在其他 seed 上是否稳定。论文应同时报告：

- 31-pass 结果91.54：说明少于基线预算时没有退化；
- 32-pass 结果91.70：公平预算下的正式结果；
- source-C 结果91.51：分类头回退消融失败；
- 多 seed 均值和标准差。

不要用91.70反向选择新的 epoch 数，也不要把中途91.58等 checkpoint 当作正式模型选择结果。

## 文件

- `raw/dac15_duet16_all_fbc_seed2020_final_91.54.txt`
- `raw/dac15_duet17_all_fbc_seed2020_final_91.70.txt`
- `raw/dac15_duet16_fb_sourcec_seed2020_final_91.51.txt`
- `code/run_visda_dac_duet_handoff_full32.sh`
- `code/run_visda_dac_duet_handoff_fb_sourcec_full.sh`
- `summary.json`
- `per_class_comparison.csv`
- `cycle4_trajectory.csv`
- `SHA256SUMS`
